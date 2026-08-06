/-
Live Weights & Biases tracking — the Lean side of `puffer train <env> --wandb`.

PufferLib logs to wandb inline every ~0.6 s from its Python loop (`pufferl.py::_train`, ~10 lines
around `wandb.init` / `wandb.log`). We reproduce that faithfully, but our trainer is Lean and wandb is
a Python SDK with no in-process binding — so the ONLY thing that has to be Python is the handful of
`wandb.*` calls. Rather than ship a separate helper script, that glue is embedded inline below
(`wandbDaemonPy`) and run as `python3 -c …`: a short-lived subprocess that reads one native-key metric
row per line on stdin and makes the real `wandb.init(config=…)` / `wandb.log(step=agent_steps)` /
checkpoint→`Artifact(run_id,'model')` / `run.finish()` calls. From wandb's side the run is
indistinguishable from a native PufferLib run; from the repo's side there is no extra tool — just the
inline glue, exactly the shape PufferLib has (its wandb lines live inside its trainer too).

The session lives in a module global so the SHARED dashboard `redraw` (Puffer.RL.Dashboard) can stream
rows without threading a handle through all six trainer loops: the trainer dispatch calls `start` before
training and `finish` after; `redraw` calls `emitRow` on every render tick (a no-op when no session is
active). Every call swallows its own errors — a missing/dead tracker must never take down a run.
-/

namespace Puffer.RL.Wandb

/-- The inline wandb bridge, run as `python3 -c wandbDaemonPy`. Protocol on stdin: line 1 is the JSON
    config dict (→ `wandb.init(config=…)`, as PufferLib passes `config=args`); each later line is a native
    `_C`-key metric row; a `{"__finish__":{"checkpoint":…}}` line ends the run (uploading the model as an
    Artifact). project/group/tag arrive as env vars (WB_*). Gates on `env/score` like PufferLib's sweep
    target; `step=agent_steps`. Uses only single quotes + `print(...,file=sys.stderr)` so the Lean string
    literal needs no escaping. `WB_DRYRUN=1` prints the calls instead of importing the SDK (test/preview);
    if wandb is absent it drains stdin so training is never blocked. -/
def wandbDaemonPy : String :=
"import sys, json, os
def emit(m):
    print(m, file=sys.stderr)
first = sys.stdin.readline()
try:
    cfg = json.loads(first) if first.strip() else {}
except Exception:
    cfg = {}
dry = bool(os.environ.get('WB_DRYRUN'))
proj = os.environ.get('WB_PROJECT', 'puffer4')
grp = os.environ.get('WB_GROUP', 'debug')
tags = [t for t in [os.environ.get('WB_TAG')] if t]
wb = None
rid = 'dryrun'
if not dry:
    try:
        import wandb
    except ImportError:
        emit('puffer: wandb not installed (pip install wandb + wandb login, or PUFFER_WANDB_DRYRUN=1); --wandb is a no-op')
        for line in sys.stdin:
            if '__finish__' in line:
                break
        sys.exit(0)
    rid = wandb.util.generate_id()
    wb = wandb.init(id=rid, config=cfg, project=proj, group=grp, tags=tags, settings=wandb.Settings(console='off'))
else:
    emit('[dry-run] init project=' + proj + ' group=' + grp + ' tags=' + str(tags) + ' config_keys=' + str(len(cfg)))
seen = False
for line in sys.stdin:
    s = line.strip()
    if not s:
        continue
    try:
        row = json.loads(s)
    except Exception:
        continue
    if '__finish__' in row:
        ck = (row['__finish__'] or {}).get('checkpoint')
        if ck and not dry:
            import wandb
            art = wandb.Artifact(rid, type='model')
            art.add_file(ck)
            wb.log_artifact(art)
        if dry:
            emit('[dry-run] finish checkpoint=' + str(ck))
        break
    if not seen:
        if 'env/score' not in row:
            continue
        seen = True
    keys = {k: v for k, v in row.items() if '/' in k or k in ('SPS', 'agent_steps', 'uptime', 'epoch')}
    step = int(row.get('agent_steps', 0))
    if dry:
        emit('[dry-run] log step=' + str(step) + ' nkeys=' + str(len(keys)))
    else:
        wb.log(keys, step=step)
if wb is not None:
    wb.finish()
"

/-- A live tracker subprocess: the pipe we stream metric rows into, and a waiter that blocks until it
    exits (so the final checkpoint-artifact upload completes before the process tree tears down). The
    waiter is stored as a closure to keep the `IO.Process.Child`'s stdio-dependent type out of the ref. -/
structure Session where
  stdin : IO.FS.Handle
  wait  : IO Unit

/-- The one live session (or none). Set by `start`, read by `emitRow`, cleared by `finish`. -/
initialize sessionRef : IO.Ref (Option Session) ← IO.mkRef none

/-- Minimal JSON string escaping (backslash + double-quote) for config values / checkpoint paths. -/
def escJson (s : String) : String :=
  s.foldl (fun acc c =>
    acc ++ (if c == '\\' then "\\\\" else if c == '"' then "\\\"" else String.singleton c)) ""

/-- Spawn the inline wandb bridge (`python3 -c wandbDaemonPy`) and hand it the run config on stdin (as
    PufferLib passes `config=args`); project/group/tag + a dry-run flag go via WB_* env vars. No-op-safe:
    if the process can't spawn we warn and continue with no session, so training is never blocked on
    wandb. `PUFFER_WANDB_DRYRUN=1` → bridge prints the `wandb.*` calls instead of importing the SDK;
    `PUFFER_PYTHON` overrides the interpreter (default `python3`). -/
def start (project group : String) (tag : Option String) (configJson : String) : IO Unit := do
  try
    let py := (← IO.getEnv "PUFFER_PYTHON").getD "python3"
    let dry := (← IO.getEnv "PUFFER_WANDB_DRYRUN").isSome
    let env : Array (String × Option String) :=
      #[("WB_PROJECT", some project), ("WB_GROUP", some group), ("WB_TAG", tag),
        ("WB_DRYRUN", if dry then some "1" else none)]
    -- stdout → null so the bridge can never corrupt our in-place dashboard redraw (same stdout);
    -- stderr inherited so wandb's run URL / errors reach the user, exactly like PufferLib.
    let child ← IO.Process.spawn {
      cmd := py, args := #["-c", wandbDaemonPy], env := env,
      stdin := .piped, stdout := .null, stderr := .inherit }
    let h := child.stdin
    h.putStr (configJson ++ "\n")
    h.flush
    sessionRef.set (some { stdin := h, wait := do let _ ← child.wait; pure () })
  catch e =>
    IO.eprintln s!"wandb: could not start tracker ({e}); continuing without wandb"

/-- Stream one native-key metric row (a JSON object, no trailing newline) to the bridge. No-op when no
    session is live; a broken pipe (dead bridge) is swallowed so it can't crash the training loop. -/
def emitRow (json : String) : IO Unit := do
  match ← sessionRef.get with
  | none => pure ()
  | some s => try s.stdin.putStr (json ++ "\n"); s.stdin.flush catch _ => pure ()

/-- End the run: tell the bridge to upload `checkpoint` as `Artifact(run_id, type='model')` (if any) and
    `run.finish()`, then wait for it to exit. Matches `pufferl.py`'s end-of-`_train` artifact upload. -/
def finish (checkpoint : Option String) : IO Unit := do
  match ← sessionRef.get with
  | none => pure ()
  | some s =>
    (try
      let ck := match checkpoint with | some p => "\"" ++ escJson p ++ "\"" | none => "null"
      s.stdin.putStr ("{\"__finish__\": {\"checkpoint\": " ++ ck ++ "}}\n")
      s.stdin.flush
      s.wait
     catch _ => pure ())
    sessionRef.set none

end Puffer.RL.Wandb
