/-!
# Runtime env plugins (`ffi/puffer_loader.c`)

Bindings for the PufferLib-style env plugin loader. `envOpen name N seed config` dlopen's
`libenv_<name>.so` at runtime (the env is NOT visible at `puffer`'s compile time) and returns an
opaque handle (`0` on failure). The env holds hidden mutable state, so `open/reset/step/close` are
`IO`; the spec getters are pure (fixed after open). `reset`/`step` operate on all `N` copies in one
native C call, so the Lean rollout loop stays `O(T)` per update — the `N`-parallelism is inside the
plugin. This module is completely env-agnostic: it names no specific env, exactly like PufferLib's
vecenv, and is the ONLY env-facing Lean code in `puffer`.
-/
namespace Puffer.Plugin

/-- dlopen `libenv_<name>.so`, make `N` copies seeded by `seed` with the `k=v,…` `config`.
    Returns an opaque handle, or `0` if the env can't be found/loaded. -/
@[extern "lean_puffer_env_open"]
opaque envOpen (name : String) (N : USize) (seed : UInt64) (config : String) : IO USize

@[extern "lean_puffer_env_obsdim"]     opaque envObsDim     (h : USize) : USize
@[extern "lean_puffer_env_numactions"] opaque envNumActions (h : USize) : USize
@[extern "lean_puffer_env_maxsteps"]   opaque envMaxSteps   (h : USize) : USize
/-- Agents per env instance. The batch is `numEnvs · numAgents` rows (each agent a training row). -/
@[extern "lean_puffer_env_numagents"]  opaque envNumAgents  (h : USize) : USize
/-- Number of categorical action heads (1 = single discrete, >1 = multi-discrete). -/
@[extern "lean_puffer_env_nheads"]     opaque envNHeads     (h : USize) : USize
/-- Size (action count) of head `i`. -/
@[extern "lean_puffer_env_headsize"]   opaque envHeadSize   (h i : USize) : USize
/-- `1` if the action space is CONTINUOUS (diagonal-Gaussian): `nHeads` is the real action dim `d`, the
    policy head is `2·d+1`, and the trainer runs the Gaussian PPO path. `0` = discrete/multi-discrete. -/
@[extern "lean_puffer_env_iscont"]     opaque envIsCont     (h : USize) : USize

/-- The action-head sizes `[s₀,…,s_{K−1}]` as an `Array Nat`. -/
def envHeadSizes (h : USize) : Array Nat :=
  let K := (envNHeads h).toNat
  (Array.range K).map (fun i => (envHeadSize h (USize.ofNat i)).toNat)

/-- Reset all copies; returns obs `N·obsDim`. -/
@[extern "lean_puffer_env_reset"]
opaque envReset (h : USize) : IO FloatArray

/-- Step all copies with `actions` (`N`, discrete idx as f64); returns `[obs(N·obsDim); rewards(N);
    terminals(N)]`. Terminated copies auto-reset in place. -/
@[extern "lean_puffer_env_step"]
opaque envStep (h : USize) (actions : FloatArray) : IO FloatArray

@[extern "lean_puffer_env_close"]
opaque envClose (h : USize) : IO Unit

/-! ## The env's own `Log` (optional metrics channel)

PufferLib's ocean envs each keep a `Log` struct on the env that `add_log()` fills at every EPISODE end
(`episode_return`, `score`, `perf`, `episode_length`, `n`, plus per-env extras), and PufferLib reports
episode statistics from THAT (`static_vec_aggregate_logs`/`static_vec_log` in `src/vecenv.h`) rather
than by summing rewards between terminal flags. The two are different units: 14 of the 39 built ocean
envs never raise a terminal at all (their `c_reset` runs inside `c_step`), so reward-summing reports a
permanent `0.0` for them, and even where terminals exist the log's episode boundary can differ
(`trash_pickup` 11.06 vs 1.38, `tripletriad` −3.50 vs −22.5, `nmmo3` −1.00 vs −3.84 under a random
policy). This channel carries the real numbers, so our reported episode statistics are comparable with
upstream PufferLib's. `puffer env-log <env>` prints both side by side, on CPU, for any env.

The fields differ per env in name, count AND order, so they are looked up BY NAME (`envLogFields`).
The channel is optional: `envLogNFields h = 0` means this plugin doesn't export it. -/

/-- Number of named fields in this env's `Log`. `0` ⇒ the plugin exports no log channel. -/
@[extern "lean_puffer_env_log_nfields"] opaque envLogNFields (h : USize) : USize
/-- Name of log field `i` (this env's own `Log` field name; `""` out of range). -/
@[extern "lean_puffer_env_log_name"]    opaque envLogName    (h i : USize) : String

/-- The log field names `[f₀,…,f_{k−1}]`, in the env's own declaration order. Empty ⇒ unsupported. -/
def envLogFields (h : USize) : Array String :=
  let k := (envLogNFields h).toNat
  (Array.range k).map (fun i => envLogName h (USize.ofNat i))

/-- Read + ZERO the env's log, exactly as PufferLib's `static_vec_log` does: sum each field over the
    copies with `n > 0`, divide by the summed `n`, then clear every copy's log — so each call reports
    the window since the previous call. The result pairs positionally with `envLogFields`; the field
    named `"n"` carries the episode COUNT (undivided). Returns an EMPTY array when the channel is
    unsupported or when no episode completed since the last call (logs then left untouched).

    Metrics only — this never touches obs/rewards/terminals. Not safe to call concurrently with a
    rollout: call it between updates. -/
@[extern "lean_puffer_env_log"]
opaque envLog (h : USize) : IO FloatArray

/-- The env's log as `(name, value)` pairs — `#[]` when unsupported / no episode finished. -/
def envLogPairs (h : USize) : IO (Array (String × Float)) := do
  let v ← envLog h
  let names := envLogFields h
  return (Array.range (min v.size names.size)).map (fun i => (names[i]!, v.get! i))

/-- Look up one log field by NAME (`none` if this env has no such field, or nothing to report).
    `episode_return` is present on 37 of the 39 built ocean envs; `cartpole` and `minimal` have no
    such field (cartpole's comparable quantity is `score`, minimal's is `perf`/`score`). -/
def envLogField? (kvs : Array (String × Float)) (name : String) : Option Float :=
  (kvs.find? (·.1 == name)).map (·.2)

end Puffer.Plugin
