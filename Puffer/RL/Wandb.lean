/-
Live Weights & Biases tracking — the Lean side of `puffer train <env> --wandb`.

PufferLib logs to wandb inline every ~0.6 s from its Python loop (`pufferl.py::_train`). Lean has
no HTTP client and wandb is a Python SDK, so we reproduce that faithfully by spawning
`tools/puffer_track.py --daemon` once at train start and streaming it one native-key metric row
per dashboard tick; it does the real `wandb.init` / `wandb.log(step=agent_steps)` / checkpoint→
Artifact / `run.finish()`. From wandb's side the run matches a native PufferLib run.

The session lives in a module global so the SHARED dashboard `redraw` (Puffer.RL.Dashboard) can emit
rows without threading a handle through all six trainer loops: the trainer dispatch calls `start`
before training and `finish` after; `redraw` calls `emitRow` on every render tick (a no-op when no
session is active). Every call swallows its own errors — a missing/dead tracker must never take down
a training run.
-/

namespace Puffer.RL.Wandb

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

/-- Spawn `tools/puffer_track.py --daemon` and hand it the run config (the resolved flag dict, as
    PufferLib passes `config=args`). No-op-safe: if the process can't spawn we warn and continue with
    no session, so training is never blocked on wandb. `PUFFER_WANDB_DRYRUN=1` makes the daemon print
    the `wandb.*` calls it WOULD make instead of importing the SDK (credential-free preview/testing);
    `PUFFER_PYTHON` overrides the interpreter (default `python3`). -/
def start (project group : String) (tag : Option String) (configJson : String) : IO Unit := do
  try
    let py := (← IO.getEnv "PUFFER_PYTHON").getD "python3"
    let dry := (← IO.getEnv "PUFFER_WANDB_DRYRUN").isSome
    let mut a : Array String := #["tools/puffer_track.py", "--daemon", "--project", project, "--group", group]
    if let some t := tag then a := a ++ #["--tag", t]
    if dry then a := a ++ #["--dry-run"]
    -- stdout → null so the daemon can never corrupt our in-place dashboard redraw (same stdout);
    -- stderr inherited so wandb's run URL / errors reach the user, exactly like PufferLib.
    let child ← IO.Process.spawn { cmd := py, args := a, stdin := .piped, stdout := .null, stderr := .inherit }
    let h := child.stdin
    h.putStr (configJson ++ "\n")
    h.flush
    sessionRef.set (some { stdin := h, wait := do let _ ← child.wait; pure () })
  catch e =>
    IO.eprintln s!"wandb: could not start tracker ({e}); continuing without wandb"

/-- Stream one native-key metric row (a JSON object, no trailing newline) to the daemon. No-op when no
    session is live; a broken pipe (dead daemon) is swallowed so it can't crash the training loop. -/
def emitRow (json : String) : IO Unit := do
  match ← sessionRef.get with
  | none => pure ()
  | some s => try s.stdin.putStr (json ++ "\n"); s.stdin.flush catch _ => pure ()

/-- End the run: tell the daemon to upload `checkpoint` as `Artifact(run_id, type='model')` (if any) and
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
