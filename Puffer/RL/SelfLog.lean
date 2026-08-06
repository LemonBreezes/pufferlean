/-
Per-run self-log — PufferLib's `logs/<env>/<run_id>.json` (`pufferl.py::_train`, ~L357-361).

Every run PufferLib dumps `{**args, 'metrics': metrics}` where `metrics` is the per-tick metric history
DOWNSAMPLED into `[sweep] downsample` (=5) bins by agent_steps (each bin the mean; the last bin the final
raw value). This file is what feeds the `constellation` galaxy (its `cache_data.py` reads `agent_steps`,
`uptime`, `env/score`, `env/perf` out of these and drops `loss/*`).

We accumulate one row per dashboard render tick (the same native-key dict that drives the on-screen
monitor + wandb — see `Puffer.RL.Dashboard.nativePairs`) into a module global, then Exe's train dispatch
writes the file after the trainer returns. `reset` clears it between sweep trials (each trial is a fresh
run). No-op safe: no ticks recorded (e.g. `PUFFER_PLAIN_LOG=1`, dashboard off) ⇒ no file written.
-/

namespace Puffer.RL.SelfLog

/-- The per-tick metric-dict history for the current run (each entry is one render tick's native-key
    pairs, in a stable key order). Appended by `record`, drained by `write`, cleared by `reset`. -/
initialize historyRef : IO.Ref (Array (Array (String × Float))) ← IO.mkRef #[]

/-- Clear the history — call at the start of each run (matters for sweeps: many trials, one process). -/
def reset : IO Unit := historyRef.set #[]

/-- Record one render tick's metric dict (native `_C` keys → values). Gated on `env/score` being present,
    exactly like PufferLib's log path (`if target_key not in flat_logs: continue`) — so the history (and
    thus the metric key set, taken from its first row) always carries the `env/*` fields. Cheap. -/
def record (row : Array (String × Float)) : IO Unit :=
  if row.any (·.1 == "env/score") then historyRef.modify (·.push row) else pure ()

/-- JSON-safe float: non-finite (NaN/±inf) collapses to `0`. -/
@[inline] def jsonF (x : Float) : String := if x == x && x.abs < 1.0e308 then toString x else "0"

/-- JSON string literal with backslash/quote escaping. -/
def jstr (s : String) : String :=
  "\"" ++ s.foldl (fun a c => a ++ (if c == '\\' then "\\\\" else if c == '"' then "\\\"" else String.singleton c)) "" ++ "\""

@[inline] private def meanOf (xs : Array Float) : Float :=
  if xs.isEmpty then 0.0 else (xs.foldl (· + ·) 0.0) / Float.ofNat xs.size

/-- Downsample the tick history into `n` bins per key, EXACTLY as `pufferl.py`: walk ticks accumulating
    into the current bin, close a bin (→ its mean) each time `agent_steps` crosses `finalSteps/(n-1)`,
    and overwrite the last bin with the final raw value. Keys/order come from the first tick. -/
def downsample (hist : Array (Array (String × Float))) (n : Nat) : Array (String × Array Float) := Id.run do
  if hist.isEmpty then return #[]
  let keys := hist[0]!.map (·.1)
  let getV := fun (row : Array (String × Float)) (k : String) => (row.find? (·.1 == k)).elim 0.0 (·.2)
  let last := hist.back!
  let finalSteps := getV last "agent_steps"
  let binW := if n > 1 then finalSteps / Float.ofNat (n - 1) else finalSteps + 1.0   -- n≤1 ⇒ one bin
  let mut closed : Array (Array Float) := keys.map (fun _ => #[])   -- closed-bin means per key
  let mut acc    : Array (Array Float) := keys.map (fun _ => #[])   -- current-bin accumulator per key
  let mut nextBin := binW
  for row in hist do
    for ki in [0:keys.size] do
      acc := acc.set! ki (acc[ki]!.push (getV row keys[ki]!))
    if getV row "agent_steps" ≥ nextBin then
      nextBin := nextBin + binW
      for ki in [0:keys.size] do
        closed := closed.set! ki (closed[ki]!.push (meanOf acc[ki]!))
        acc := acc.set! ki #[]
  -- last bin = the final raw value (replaces the still-open accumulator), per PufferLib.
  return keys.mapIdx (fun ki k => (k, (closed[ki]!).push (getV last k)))

/-- Write `logs/<env>/<run_id>.json = {**config, "metrics": {key: [downsampled…]}}` if any ticks were
    recorded. `config` is the resolved flag map (PufferLib's `{**args}`); `run_id` a unique-per-run id. -/
def write (env : String) (config : List (String × String)) (n : Nat) (runId : String) : IO Unit := do
  let hist ← historyRef.get
  if hist.isEmpty then return
  let metrics := downsample hist n
  let dir := s!"logs/{env}"
  IO.FS.createDirAll dir
  let cfgPart := String.intercalate "," (config.map (fun kv => jstr kv.1 ++ ":" ++ jstr kv.2))
  let metricsPart := String.intercalate "," (metrics.toList.map (fun kv =>
    jstr kv.1 ++ ":[" ++ String.intercalate "," (kv.2.toList.map jsonF) ++ "]"))
  let json := "{" ++ cfgPart ++ (if cfgPart.isEmpty then "" else ",")
                  ++ jstr "metrics" ++ ":{" ++ metricsPart ++ "}}"
  IO.FS.writeFile s!"{dir}/{runId}.json" json

end Puffer.RL.SelfLog
