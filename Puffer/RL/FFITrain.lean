/-
# FFI-accelerated PPO training (M7 hot path)

The training bottleneck is the per-transition AD-tape gradient (`mlpGradPPO`), which
allocates a Wengert tape of thousands of nodes per transition. This module routes the
minibatch gradient through the native C kernel `mlpPPOGradBatchFFI` (`ffi/pufferffi.c`):
the MLP forward + PPO objective + backward for a whole minibatch in one FFI call,
accumulated in C, no tape. The Lean `mlpGradPPO` stays as the ORACLE — the C gradient
is FD-validated and cross-checked against it (`puffer verify-grad-ffi`).

The MLP is flattened to one `FloatArray` (`W1[H·D], b1[H], W2[O·H], b2[O]`, `O=A+1`)
per minibatch; the returned flat gradient is applied with the same grad-norm-clip +
mean-scale as `updatePPOIdx`. These native-C gradient helpers back the feed-forward
plugin trainers (`trainPluginEnv`/`Cont`/`MD`) dispatched from `Exe/Puffer.lean`.
-/
import Puffer.RL.VecTrain
import Puffer.RL.ContVecTrain
import Puffer.RL.CnnVecTrain
import Puffer.RL.MultiVecTrain
import Puffer.RL.RecVecTrain
import Puffer.Float.FFI
import Puffer.Float.BLAS
import Puffer.Float.CUDA
import Puffer.Plugin
import Puffer.RL.Dashboard

namespace Puffer.RL.NNTrain

open Puffer.RL (Env ContEnv MultiEnv)
open Puffer.FloatR
open Puffer.RL.Train (rngNext uniform01 softmax sampleCat)

/-! ### STREAM 3 — policy-weight checkpointing (OUR binary format).

`puffer train` persists the flat policy weights the trainers keep resident (`wFlat`) so a run can be
resumed and `puffer eval` / `--load` can score a saved policy. Written straight from Lean via `IO.FS`
(no device readback). Layout is little-endian: a `UInt64` param count `P`, then `P` IEEE-754 doubles as
their 8-byte bit patterns (`Float.toBits`). Interop with PufferLib's own `.bin` is OUT OF SCOPE — this
is our format; it only has to round-trip our `wFlat`. -/

/-- Append `x` as 8 little-endian bytes. -/
@[inline] def pushU64LE (b : ByteArray) (x : UInt64) : ByteArray := Id.run do
  let mut b := b; let mut x := x
  for _ in [0:8] do
    b := b.push (x &&& 0xff).toUInt8
    x := x >>> 8
  return b

/-- Read 8 little-endian bytes at `off` as a `UInt64` (bytes past the end read as 0). -/
@[inline] def readU64LE (b : ByteArray) (off : Nat) : UInt64 := Id.run do
  let mut x : UInt64 := 0
  for k in [0:8] do
    if off + k < b.size then
      x := x ||| ((b.get! (off+k)).toUInt64 <<< (UInt64.ofNat (8*k)))
  return x

/-- Serialize flat policy weights to OUR checkpoint byte layout `[P : u64][P doubles-as-bits]`. -/
def ckptToBytes (wFlat : FloatArray) : ByteArray := Id.run do
  let P := wFlat.size
  let mut b := pushU64LE (ByteArray.emptyWithCapacity ((P+1)*8)) (UInt64.ofNat P)
  for i in [0:P] do b := pushU64LE b (Float.toBits wFlat[i]!)
  return b

/-- Parse OUR checkpoint byte layout back to a flat weight array (bounded by the file size, so a
    truncated/short file yields at most the doubles it actually contains rather than reading OOB). -/
def bytesToCkpt (b : ByteArray) : FloatArray :=
  let avail := if b.size ≥ 8 then (b.size - 8) / 8 else 0
  let P := min (readU64LE b 0).toNat avail
  FloatArray.mk ((Array.range P).map (fun i => Float.ofBits (readU64LE b (8 + i*8))))

/-- Checkpoint directory PufferLib writes under: `checkpoints/<env>/<runId>/`. -/
def ckptDir (env runId : String) : String := s!"checkpoints/{env}/{runId}"

/-- Persist the resident policy weights to `checkpoints/<env>/<runId>/<step>.bin` (mkdir -p first).
    Returns the path written. `step` is the global env-step count, mirroring PufferLib's `<step>.bin`. -/
def savePolicyCheckpoint (env runId : String) (step : Nat) (wFlat : FloatArray) : IO String := do
  let dir := ckptDir env runId
  IO.FS.createDirAll dir
  let path := s!"{dir}/{step}.bin"
  IO.FS.writeBinFile path (ckptToBytes wFlat)
  return path

/-- Load a checkpoint file into a flat weight array, or `none` if it does not exist. -/
def loadPolicyCheckpoint (path : String) : IO (Option FloatArray) := do
  if ← System.FilePath.pathExists path then
    return some (bytesToCkpt (← IO.FS.readBinFile path))
  else
    return none

/-- Find the highest-step checkpoint under `checkpoints/<env>/` (scanning every run subdir), or `none`.
    Used by `puffer eval <env>` when no explicit `--load <path>` is given. -/
def findLatestCheckpoint (env : String) : IO (Option String) := do
  let base : System.FilePath := s!"checkpoints/{env}"
  if !(← System.FilePath.pathExists base) then return none
  let mut best : Option (Nat × String) := none
  for runDir in (← base.readDir) do
    if !(← System.FilePath.isDir runDir.path) then continue
    for f in (← runDir.path.readDir) do
      if f.fileName.endsWith ".bin" then
        let step := (f.fileName.dropRight 4).toNat?.getD 0
        match best with
        | some (bs, _) => if step ≥ bs then best := some (step, f.path.toString)
        | none         => best := some (step, f.path.toString)
  return best.map (·.2)

/-- Env-step worker count reported by the `Num workers:` startup line — mirrors the native rollout's
    `rp_threads()` (env `PUFFER_ROLL_THREADS`, default 8, clamped to [1,63]). -/
def rollWorkers : IO Nat := do
  let n := ((← IO.getEnv "PUFFER_ROLL_THREADS").bind String.toNat?).getD 0
  return if n < 1 then 8 else if n > 63 then 63 else n

/-! ### PufferLib-parity logging shared across the vectorized trainers.

`PufferMetrics` = PufferLib's per-epoch metric dict; `metricsFromArrays` computes it from
per-transition `(newLogp, oldLogp, adv, return, value, entropy)` (the CleanRL formulas), so
each head just builds those arrays with its own forward (`mlpMetrics`/`cnnMetrics`/
`gaussMetrics`/`recMetrics`). `pufferDashboard` renders the grouped terminal panel and
`metricsJson` serializes the dict under PufferLib's exact wandb keys (for `tools/puffer_track.py`). -/

structure PufferMetrics where
  policyLoss : Float := 0.0
  valueLoss : Float := 0.0
  valueStd : Float := 0.0
  entropy : Float := 0.0
  entropyStd : Float := 0.0
  approxKl : Float := 0.0
  oldApproxKl : Float := 0.0
  clipfrac : Float := 0.0
  explainedVar : Float := 0.0
  deriving Inhabited

/-- PufferLib/CleanRL loss diagnostics from aligned per-transition arrays:
    `policy_loss = −E[min(ρA, clip(ρ)A)]`, `approx_kl = E[(ρ−1)−log ρ]`,
    `clipfrac = P(|ρ−1|>ε)`, `explained_variance = 1 − Var(R−V)/Var(R)`. -/
def metricsFromArrays (newLogps oldLogps advs returns values entropies : Array Float) (clip : Float) : PufferMetrics := Id.run do
  let n := newLogps.size
  if n == 0 then return {}
  let mut plSum := 0.0; let mut vlSum := 0.0; let mut entSum := 0.0
  let mut klSum := 0.0; let mut oklSum := 0.0; let mut clipCnt := 0; let mut retSum := 0.0; let mut valSum := 0.0
  for t in [0:n] do
    let logratio := newLogps[t]! - oldLogps[t]!
    let ratio := Float.exp logratio
    let A := advs[t]!
    let ratioC := if ratio < 1.0 - clip then 1.0 - clip else if ratio > 1.0 + clip then 1.0 + clip else ratio
    plSum := plSum - min (ratio * A) (ratioC * A)
    let vd := values[t]! - returns[t]!
    vlSum := vlSum + vd * vd
    entSum := entSum + entropies[t]!
    klSum := klSum + (ratio - 1.0 - logratio)
    oklSum := oklSum - logratio
    if Float.abs (ratio - 1.0) > clip then clipCnt := clipCnt + 1
    retSum := retSum + returns[t]!; valSum := valSum + values[t]!
  let nf := Float.ofNat n
  let retMean := retSum / nf
  let mut resSum := 0.0
  for t in [0:n] do resSum := resSum + (returns[t]! - values[t]!)
  let resMean := resSum / nf
  let entMean := entSum / nf
  let valMean := valSum / nf
  let mut retVar := 0.0; let mut resVar := 0.0; let mut entVar := 0.0; let mut valVar := 0.0
  for t in [0:n] do
    let rd := returns[t]! - retMean
    retVar := retVar + rd * rd
    let sd := (returns[t]! - values[t]!) - resMean
    resVar := resVar + sd * sd
    let ed := entropies[t]! - entMean
    entVar := entVar + ed * ed
    let vdv := values[t]! - valMean
    valVar := valVar + vdv * vdv
  let ev := if retVar > 0.0 then 1.0 - resVar / retVar else 0.0
  return { policyLoss := plSum / nf, valueLoss := vlSum / nf, valueStd := Float.sqrt (valVar / nf),
           entropy := entMean, entropyStd := Float.sqrt (entVar / nf),
           approxKl := klSum / nf, oldApproxKl := oklSum / nf,
           clipfrac := Float.ofNat clipCnt / nf, explainedVar := ev }

/-- Discrete-MLP head metrics over a `Transition` minibatch. -/
def mlpMetrics (p : MLP) (buf : Array Transition) (advs returns : Array Float) (clip : Float) : PufferMetrics := Id.run do
  let mut nl : Array Float := #[]; let mut ol : Array Float := #[]
  let mut vs : Array Float := #[]; let mut ents : Array Float := #[]
  for tr in buf do
    let (probs, v) := policyAndValue p tr.obs
    nl := nl.push (Float.log probs[tr.action]!); ol := ol.push tr.oldLogp; vs := vs.push v
    let mut e := 0.0
    for pk in probs do e := e - (if pk > 0.0 then pk * Float.log pk else 0.0)
    ents := ents.push e
  return metricsFromArrays nl ol advs returns vs ents clip

/-- CNN head metrics over a `Transition` minibatch. -/
def cnnMetrics (p : CnnPolicy) (buf : Array Transition) (advs returns : Array Float) (A : Nat) (clip : Float) : PufferMetrics := Id.run do
  let mut nl : Array Float := #[]; let mut ol : Array Float := #[]
  let mut vs : Array Float := #[]; let mut ents : Array Float := #[]
  for tr in buf do
    let (probs, v) := cnnProbsValue p A tr.obs
    nl := nl.push (Float.log probs[tr.action]!); ol := ol.push tr.oldLogp; vs := vs.push v
    let mut e := 0.0
    for pk in probs do e := e - (if pk > 0.0 then pk * Float.log pk else 0.0)
    ents := ents.push e
  return metricsFromArrays nl ol advs returns vs ents clip

/-- Gaussian (continuous) head metrics over a `ContTransition` minibatch: Gaussian
    log-prob `Σ(−½z² − logstd − ½log2π)` and differential entropy `Σ(logstd + ½(1+log2π))`. -/
def gaussMetrics (p : MLP) (buf : Array ContTransition) (advs returns : Array Float) (d : Nat) (clip : Float) : PufferMetrics := Id.run do
  let mut nl : Array Float := #[]; let mut ol : Array Float := #[]
  let mut vs : Array Float := #[]; let mut ents : Array Float := #[]
  for tr in buf do
    let (mean, logstd, v) := contPolicy p d tr.obs
    let mut lp := 0.0; let mut ent := 0.0
    for i in [0:d] do
      let z := (tr.action[i]! - mean[i]!) * Float.exp (- logstd[i]!)
      lp := lp + (-0.5 * z * z - logstd[i]! - halfLog2piC)
      ent := ent + (logstd[i]! + halfLog2pieEC)
    nl := nl.push lp; ol := ol.push tr.oldLogp; vs := vs.push v; ents := ents.push ent
  return metricsFromArrays nl ol advs returns vs ents clip

/-- LSTM head metrics: re-run the recurrent forward over each sequence (threading `(h,c)`
    from the detached initial state, resetting at terminals) with batch-normalized advantages. -/
def recMetrics (p : RecPolicy) (trajs : Array (Array Transition)) (initHs initCs : Array (Array Float))
    (segAdvRet : Array (Array Float × Array Float)) (mean std : Float) (numActions : Nat) (clip : Float) : PufferMetrics := Id.run do
  let mut nl : Array Float := #[]; let mut ol : Array Float := #[]; let mut advs : Array Float := #[]
  let mut rets : Array Float := #[]; let mut vs : Array Float := #[]; let mut ents : Array Float := #[]
  for e in [0:trajs.size] do
    let traj := trajs[e]!
    let (adv, ret) := segAdvRet[e]!
    let mut h := initHs[e]!; let mut c := initCs[e]!
    for t in [0:traj.size] do
      let tr := traj[t]!
      if t > 0 && traj[t-1]!.terminal then h := zeros p.hSize; c := zeros p.hSize
      let (h', c', out) := lstmCellF p tr.obs h c
      h := h'; c := c'
      let (probs, v) := recProbsValue out numActions
      nl := nl.push (Float.log probs[tr.action]!); ol := ol.push tr.oldLogp
      advs := advs.push ((adv[t]! - mean) / (std + 1.0e-8)); rets := rets.push ret[t]!; vs := vs.push v
      let mut en := 0.0
      for pk in probs do en := en - (if pk > 0.0 then pk * Float.log pk else 0.0)
      ents := ents.push en
  return metricsFromArrays nl ol advs rets vs ents clip

/-- A unicode block sparkline of a series, normalized to its own [min,max] over 8 levels. -/
def sparkline (xs : Array Float) : String := Id.run do
  if xs.size == 0 then return ""
  let mut lo := xs[0]!; let mut hi := xs[0]!
  for x in xs do
    if x < lo then lo := x
    if x > hi then hi := x
  let range := hi - lo
  let blocks : Array Char := #['▁','▂','▃','▄','▅','▆','▇','█']
  let mut s := ""
  for x in xs do
    let t := if range ≤ 0.0 then 0.0 else (x - lo) / range
    let mut idx := 0
    for b in [1:8] do
      if t * 8.0 ≥ Float.ofNat b then idx := b
    s := s.push blocks[idx]!
  return s

@[inline] def padR2 (s : String) (w : Nat) : String :=
  if s.length ≥ w then s else s ++ String.ofList (List.replicate (w - s.length) ' ')

/-- One dashboard history point: the loss metrics plus the performance series that get
    trend sparklines (episode_return, SPS, uptime). -/
structure DashPoint where
  m : PufferMetrics
  epReturn : Float
  epLen : Float
  sps : Float
  uptime : Float
  steps : Float
  epoch : Float
  gradNorm : Float
  lr : Float
  weightNorm : Float
  rewardStd : Float
  envMetrics : Array (String × Float) := #[]
  deriving Inhabited

/-- L2 norm of a flat parameter vector — the policy `weight_norm` curve. -/
def l2norm (a : FloatArray) : Float := Id.run do
  let mut s := 0.0
  for i in [0:a.size] do s := s + a[i]! * a[i]!
  return Float.sqrt s

/-- Population standard deviation of a sample — the `reward_std` curve over a rollout's
    episode returns. -/
def stdOf (xs : Array Float) : Float := Id.run do
  let n := xs.size
  if n == 0 then return 0.0
  let mean := xs.foldl (· + ·) 0.0 / Float.ofNat n
  let var := xs.foldl (fun a x => a + (x - mean) * (x - mean)) 0.0 / Float.ofNat n
  return Float.sqrt var

/-- PufferLib's COSINE learning-rate anneal (`torch_pufferl.py train()`): the full `lr` on the
    first update (`u=0`), then `lr_min + ½·(lr − lr_min)·(1 + cos(π·u/updates))` with
    `lr_min = lr·minLrRatio`. Replaces the earlier linear `lr·(1 − u/updates)` schedule. -/
def cosineLr (lr minLrRatio : Float) (u updates : Nat) : Float :=
  if u == 0 then lr
  else
    let lrMin := lr * minLrRatio
    let ratio := Float.ofNat u / Float.ofNat (max updates 1)
    lrMin + 0.5 * (lr - lrMin) * (1.0 + Float.cos (3.14159265358979323846 * ratio))

/-- Mean each env-specific metric across the parallel env instances (all share the same
    keys in the same order, since it's one env), for the `environment/<name>` metrics. -/
def aggregateEnvMetrics (per : Array (Array (String × Float))) : Array (String × Float) := Id.run do
  if per.size == 0 then return #[]
  let keys := per[0]!
  let mut out : Array (String × Float) := #[]
  for k in [0:keys.size] do
    let mut sum := 0.0
    for e in per do sum := sum + (e[k]!).2
    out := out.push ((keys[k]!).1, sum / Float.ofNat per.size)
  return out

/-- Look up a named env metric (0 if absent). -/
@[inline] def envVal (ms : Array (String × Float)) (key : String) : Float :=
  (ms.find? (fun kv => kv.1 == key)).elim 0.0 (·.2)

/-- Render the plugin env's OWN PufferLib `Log` (`Puffer.Plugin.envLogPairs`) for the progress line.

    This is the number that is comparable with upstream PufferLib: their `static_vec_log` reports
    episode statistics from each env's `Log` (filled by the env's `add_log()` at every episode end),
    NOT by summing rewards between terminal flags the way the `batch reward Σ / terminals` figure
    beside it does. The two genuinely differ — 14 of the 39 built ocean envs never raise a terminal at
    all (reward-summing gives them a permanent `0.0`), and even where terminals exist the log's episode
    boundary can be coarser (`trash_pickup` 11.06 vs 1.38, `tripletriad` −3.50 vs −22.5 under a random
    policy). `""` when the env exports no log channel, or when no episode completed in this window. -/
def fmtEnvLog (kvs : Array (String × Float)) : String :=
  if kvs.isEmpty then ""
  else "\n      env log: " ++ String.intercalate "  " (kvs.toList.map (fun kv => s!"{kv.1}={kv.2}"))

/-- The PufferLib grouped terminal dashboard (Summary / Losses / Environment) with recent
    trend sparklines on performance (SPS, uptime), the losses, and the reward — `hist` is the
    series accumulated across log intervals (last ~40 shown). -/
def pufferDashboard (name netStr : String) (seed : Nat) (track : Option String)
    (uptime steps epoch sps : Nat) (m : PufferMetrics) (epReturn epLen : Float)
    (hist : Array DashPoint) : IO Unit := do
  let trk := match track with | some b => s!" · track:{b}" | none => ""
  let recent := if hist.size > 40 then hist.extract (hist.size - 40) hist.size else hist
  let spark := fun (sel : PufferMetrics → Float) => sparkline (recent.map (fun hr => sel hr.m))
  let line := fun (label : String) (v : Float) (sp : String) => s!"     {padR2 label 12} {padR2 (toString v) 12} {sp}"
  IO.println ""
  IO.println   "╔══════════════════════════════ puffer ══════════════════════════════╗"
  IO.println s!"  env {name}   net {netStr}   seed {seed}{trk}"
  IO.println   "  ── performance (recent trend) ───────────────────────────────────"
  IO.println (line "SPS" (Float.ofNat sps) (sparkline (recent.map (·.sps))))
  IO.println (line "uptime_s" (Float.ofNat uptime) (sparkline (recent.map (·.uptime))))
  IO.println (line "agent_steps" (Float.ofNat steps) (sparkline (recent.map (·.steps))))
  IO.println (line "epoch" (Float.ofNat epoch) (sparkline (recent.map (·.epoch))))
  IO.println   "  ── losses (recent trend) ────────────────────────────────────────"
  IO.println (line "policy_loss" m.policyLoss (spark (·.policyLoss)))
  IO.println (line "value_loss" m.valueLoss (spark (·.valueLoss)))
  IO.println (line "value_std" m.valueStd (spark (·.valueStd)))
  IO.println (line "entropy" m.entropy (spark (·.entropy)))
  IO.println (line "entropy_std" m.entropyStd (spark (·.entropyStd)))
  IO.println (line "expl_var" m.explainedVar (spark (·.explainedVar)))
  IO.println (line "approx_kl" m.approxKl (spark (·.approxKl)))
  IO.println (line "old_apx_kl" m.oldApproxKl (spark (·.oldApproxKl)))
  IO.println (line "clipfrac" m.clipfrac (spark (·.clipfrac)))
  IO.println   "  ── optimizer (recent trend) ─────────────────────────────────────"
  IO.println (line "grad_norm" (if hist.size == 0 then 0.0 else hist.back!.gradNorm) (sparkline (recent.map (·.gradNorm))))
  IO.println (line "weight_norm" (if hist.size == 0 then 0.0 else hist.back!.weightNorm) (sparkline (recent.map (·.weightNorm))))
  IO.println (line "learn_rate" (if hist.size == 0 then 0.0 else hist.back!.lr) (sparkline (recent.map (·.lr))))
  IO.println   "  ── environment (recent trend) ───────────────────────────────────"
  IO.println (line "ep_return" epReturn (sparkline (recent.map (·.epReturn))))
  IO.println (line "reward_std" (if hist.size == 0 then 0.0 else hist.back!.rewardStd) (sparkline (recent.map (·.rewardStd))))
  IO.println (line "ep_length" epLen (sparkline (recent.map (·.epLen))))
  for kv in (if hist.size == 0 then #[] else hist.back!.envMetrics) do
    IO.println (line (String.ofList (kv.1.toList.take 12)) kv.2 (sparkline (recent.map (fun dp => envVal dp.envMetrics kv.1))))
  IO.println   "╚═════════════════════════════════════════════════════════════════════╝"

/-- One JSONL record with the full metric dict — every value the dashboard curves, under
    PufferLib's `losses/`·`performance/`·`environment/` namespaces (+ an `optimizer/` group
    for grad/weight-norm and LR). Concatenated to dodge `s!` escaping. -/
def metricsJson (uptime steps epoch sps : Nat) (m : PufferMetrics)
    (epReturn epLen gradNorm weightNorm lr rewardStd : Float) (envMetrics : Array (String × Float)) : String :=
  let q := fun (k v : String) => "\"" ++ k ++ "\":" ++ v
  let envPart := envMetrics.foldl (fun acc kv => acc ++ "," ++ q ("environment/" ++ kv.1) (toString kv.2)) ""
  "{" ++ q "performance/uptime" (toString uptime)
    ++ "," ++ q "performance/agent_steps" (toString steps)
    ++ "," ++ q "performance/epoch" (toString epoch)
    ++ "," ++ q "performance/sps" (toString sps)
    ++ "," ++ q "losses/policy_loss" (toString m.policyLoss)
    ++ "," ++ q "losses/value_loss" (toString m.valueLoss)
    ++ "," ++ q "losses/value_std" (toString m.valueStd)
    ++ "," ++ q "losses/entropy" (toString m.entropy)
    ++ "," ++ q "losses/entropy_std" (toString m.entropyStd)
    ++ "," ++ q "losses/approx_kl" (toString m.approxKl)
    ++ "," ++ q "losses/old_approx_kl" (toString m.oldApproxKl)
    ++ "," ++ q "losses/clipfrac" (toString m.clipfrac)
    ++ "," ++ q "losses/explained_variance" (toString m.explainedVar)
    ++ "," ++ q "optimizer/grad_norm" (toString gradNorm)
    ++ "," ++ q "optimizer/weight_norm" (toString weightNorm)
    ++ "," ++ q "optimizer/learning_rate" (toString lr)
    ++ "," ++ q "environment/episode_return" (toString epReturn)
    ++ "," ++ q "environment/reward_std" (toString rewardStd)
    ++ "," ++ q "environment/episode_length" (toString epLen)
    ++ envPart
    ++ "}\n"

/-- Write the accumulated PufferLib metric JSONL to `dest` (if set) + print the upload hint. -/
def writeMetricsLog (dest : Option String) (track : Option String) (jsonl : String) : IO Unit := do
  match dest with
  | some path => do
      IO.FS.writeFile path jsonl
      let nrows := (jsonl.splitOn "\n").length - 1
      IO.println s!"wrote {nrows} PufferLib metric rows → {path}"
      match track with
      | some b => IO.println s!"  upload:  python tools/puffer_track.py --backend {b} {path}   (or add --dry-run)"
      | none => pure ()
  | none => pure ()

/-- Flatten an MLP to the params layout the C kernel expects. -/
def flattenMLP (p : MLP) : FloatArray := Id.run do
  let mut a : FloatArray := FloatArray.emptyWithCapacity 0
  for row in p.W1 do for x in row do a := a.push x
  for x in p.b1 do a := a.push x
  for row in p.W2 do for x in row do a := a.push x
  for x in p.b2 do a := a.push x
  return a

/-- Apply a flat gradient (C-kernel layout) as `p := p + scale·grad`, reconstructing
    the MLP tensors. -/
def applyFlatGrad (p : MLP) (g : FloatArray) (scale : Float) : MLP := Id.run do
  let H := p.b1.size
  let D := (p.W1[0]!).size
  let O := p.b2.size
  let mut W1 := p.W1
  for j in [0:H] do
    let mut row := W1[j]!
    for d in [0:D] do row := row.set! d (row[d]! + scale * g[j * D + d]!)
    W1 := W1.set! j row
  let off1 := H * D
  let mut b1 := p.b1
  for j in [0:H] do b1 := b1.set! j (b1[j]! + scale * g[off1 + j]!)
  let off2 := off1 + H
  let mut W2 := p.W2
  for k in [0:O] do
    let mut row := W2[k]!
    for j in [0:H] do row := row.set! j (row[j]! + scale * g[off2 + k * H + j]!)
    W2 := W2.set! k row
  let off3 := off2 + O * H
  let mut b2 := p.b2
  for k in [0:O] do b2 := b2.set! k (b2[k]! + scale * g[off3 + k]!)
  return { W1 := W1, b1 := b1, W2 := W2, b2 := b2 }

/-- Pack a minibatch (by buffer index) into the C kernel's flat input arrays:
    `(obsB [N·D], acts [N], advs [N], rets [N], oldlps [N])`. -/
def mkBatchArrays (buf : Array Transition) (advN returns : Array Float) (idxs : Array Nat) :
    FloatArray × FloatArray × FloatArray × FloatArray × FloatArray := Id.run do
  let mut obsB : FloatArray := FloatArray.emptyWithCapacity 0
  let mut acts : FloatArray := FloatArray.emptyWithCapacity 0
  let mut advs : FloatArray := FloatArray.emptyWithCapacity 0
  let mut rets : FloatArray := FloatArray.emptyWithCapacity 0
  let mut olps : FloatArray := FloatArray.emptyWithCapacity 0
  for t in idxs do
    let tr : Transition := buf[t]!
    for x in tr.obs do obsB := obsB.push x
    acts := acts.push (Float.ofNat tr.action)
    advs := advs.push advN[t]!
    rets := rets.push returns[t]!
    olps := olps.push tr.oldLogp
  return (obsB, acts, advs, rets, olps)

/-- One minibatch ascent step via the native C gradient kernel (same mean-scale +
    global grad-norm clip as `updatePPOIdx`). -/
def updatePPOIdxFFI (p : MLP) (buf : Array Transition) (advN returns : Array Float)
    (idxs : Array Nat) (numActions : Nat) (lr maxGradNorm vfCoef entCoef clipEps : Float) : MLP × Float := Id.run do
  let params := flattenMLP p
  let H := p.b1.size
  let D := (p.W1[0]!).size
  let (obsB, acts, advs, rets, olps) := mkBatchArrays buf advN returns idxs
  let g := Puffer.Float.FFI.mlpPPOGradBatchFFI params obsB acts advs rets olps
             (USize.ofNat idxs.size) (USize.ofNat H) (USize.ofNat D) (USize.ofNat numActions)
             vfCoef entCoef clipEps
  let mut sq := 0.0
  for i in [0:g.size] do sq := sq + g[i]! * g[i]!
  let mb := Float.ofNat (max idxs.size 1)
  let meanNorm := Float.sqrt sq / mb
  let cc := if maxGradNorm > 0.0 && meanNorm > maxGradNorm then maxGradNorm / meanNorm else 1.0
  return (applyFlatGrad p g (lr * cc / mb), meanNorm)

/-- Unflatten the C kernel's SUMMED flat gradient (the `flattenMLP`/`applyFlatGrad` layout:
    `W1` row-major `H×D`, then `b1`, then `W2` row-major `O×H`, then `b2`) into the structured
    `(gW1, gb1, gW2, gb2)` that `applyMuon` consumes, scaling every entry by `s`. Passing
    `s = clip / mb` converts the summed gradient into PufferLib's clipped MEAN gradient (its
    loss is `.mean()`-reduced and `clip_grad_norm_` runs before `optimizer.step()`). -/
def unflattenMLPGrad (g : FloatArray) (s : Float) (H D O : Nat) :
    Array (Array Float) × Array Float × Array (Array Float) × Array Float := Id.run do
  let mut gW1 : Array (Array Float) := #[]
  for j in [0:H] do
    let mut row : Array Float := #[]
    for d in [0:D] do row := row.push (s * g[j * D + d]!)
    gW1 := gW1.push row
  let off1 := H * D
  let gb1 := (Array.range H).map (fun j => s * g[off1 + j]!)
  let off2 := off1 + H
  let mut gW2 : Array (Array Float) := #[]
  for k in [0:O] do
    let mut row : Array Float := #[]
    for j in [0:H] do row := row.push (s * g[off2 + k * H + j]!)
    gW2 := gW2.push row
  let off3 := off2 + O * H
  let gb2 := (Array.range O).map (fun k => s * g[off3 + k]!)
  return (gW1, gb1, gW2, gb2)

/-- One minibatch **Muon** step via the native C gradient kernel — the PufferLib
    (`torch_pufferl.py`) optimizer in place of SGD. Same PPO gradient as `updatePPOIdxFFI`,
    but the update is `applyMuon` (Nesterov momentum → Newton–Schulz orthogonalization of the
    2D weight updates → decoupled weight decay), with the momentum state `st` PERSISTING across
    every minibatch/epoch/update (as `torch.optim`'s buffer does). `clip_grad_norm_` (global L2
    norm of the mean gradient) is applied BEFORE the step, exactly as PufferLib clips before
    `optimizer.step()`. Muon hypers match PufferLib's default: `mu = beta1 = 0.95`, `wd = 0`,
    NS `eps = 1e-7` (muon.py's hardcoded normalization eps). Returns `(policy', state', gradNorm)`. -/
def updatePPOIdxFFIMuon (p : MLP) (st : MuonState) (buf : Array Transition) (advN returns : Array Float)
    (idxs : Array Nat) (numActions : Nat) (lr maxGradNorm vfCoef entCoef clipEps : Float) :
    MLP × MuonState × Float := Id.run do
  let params := flattenMLP p
  let H := p.b1.size
  let D := (p.W1[0]!).size
  let O := p.b2.size
  let (obsB, acts, advs, rets, olps) := mkBatchArrays buf advN returns idxs
  let g := Puffer.Float.FFI.mlpPPOGradBatchFFI params obsB acts advs rets olps
             (USize.ofNat idxs.size) (USize.ofNat H) (USize.ofNat D) (USize.ofNat numActions)
             vfCoef entCoef clipEps
  let mut sq := 0.0
  for i in [0:g.size] do sq := sq + g[i]! * g[i]!
  let mb := Float.ofNat (max idxs.size 1)
  let meanNorm := Float.sqrt sq / mb                                    -- ‖mean gradient‖₂
  let cc := if maxGradNorm > 0.0 && meanNorm > maxGradNorm then maxGradNorm / meanNorm else 1.0
  let structured := unflattenMLPGrad g (cc / mb) H D O                  -- clipped mean gradient
  let (p', st') := applyMuon p st structured lr 0.0 0.95 1.0e-7         -- lr, wd=0, mu=0.95, nsEps
  return (p', st', meanNorm)

/-- The kernel returns `gradient[P] ++ new_logp[n] ++ new_value[n]`; this extracts the last `2n`
    into `(new_logp, new_value)` so the PER loop can iterate the ratio/value buffers WITHOUT a
    separate Lean forward (the kernel's own forward already computed them). -/
@[inline] def splitLV (g : FloatArray) (P n : Nat) : FloatArray × FloatArray := Id.run do
  let mut lp : FloatArray := FloatArray.emptyWithCapacity n
  let mut vv : FloatArray := FloatArray.emptyWithCapacity n
  for i in [0:n] do
    lp := lp.push g[P + i]!
    vv := vv.push g[P + n + i]!
  return (lp, vv)

/-- Muon step from PRE-BUILT flat minibatch arrays; also returns the kernel's per-sample
    `(new_logp, new_value)` (its forward computes them) so the PER loop skips a separate forward. -/
def muonStepFromArrays (p : MLP) (st : MuonState) (obsB acts advs rets olps oldvals : FloatArray)
    (n numActions : Nat) (lr maxGradNorm vfCoef entCoef clipEps vfClip mu : Float) :
    MLP × MuonState × Float × FloatArray × FloatArray := Id.run do
  let params := flattenMLP p
  let H := p.b1.size
  let D := (p.W1[0]!).size
  let O := p.b2.size
  let P := params.size
  let g := Puffer.Float.FFI.mlpPPOGradBatchVclipFFI params obsB acts advs rets olps oldvals
             (USize.ofNat n) (USize.ofNat H) (USize.ofNat D) (USize.ofNat numActions) vfCoef entCoef clipEps vfClip
  let mut sq := 0.0
  for i in [0:P] do sq := sq + g[i]! * g[i]!                 -- gradient is g[0:P]; g[P:P+2n] is logp/value
  let mb := Float.ofNat (max n 1)
  let meanNorm := Float.sqrt sq / mb
  let cc := if maxGradNorm > 0.0 && meanNorm > maxGradNorm then maxGradNorm / meanNorm else 1.0
  let structured := unflattenMLPGrad g (cc / mb) H D O
  let (p', st') := applyMuon p st structured lr 0.0 mu 1.0e-7
  let (lp, vv) := splitLV g P n
  return (p', st', meanNorm, lp, vv)

/-- One minibatch ascent step via the BLAS gradient kernel — identical to
    `updatePPOIdxFFI` (same mean-scale + global grad-norm clip + `applyFlatGrad`), but the
    minibatch forward/backward matmuls run through OpenBLAS (`mlpPPOGradBatchBlasFFI`). The
    gradient matches the scalar kernel to tolerance (~1e-11), not bit-exactly. -/
def updatePPOIdxBlasFFI (p : MLP) (buf : Array Transition) (advN returns : Array Float)
    (idxs : Array Nat) (numActions : Nat) (lr maxGradNorm vfCoef entCoef clipEps : Float) : MLP := Id.run do
  let params := flattenMLP p
  let H := p.b1.size
  let D := (p.W1[0]!).size
  let (obsB, acts, advs, rets, olps) := mkBatchArrays buf advN returns idxs
  let g := Puffer.Float.BLAS.mlpPPOGradBatchBlasFFI params obsB acts advs rets olps
             (USize.ofNat idxs.size) (USize.ofNat H) (USize.ofNat D) (USize.ofNat numActions)
             vfCoef entCoef clipEps
  let mut sq := 0.0
  for i in [0:g.size] do sq := sq + g[i]! * g[i]!
  let mb := Float.ofNat (max idxs.size 1)
  let meanNorm := Float.sqrt sq / mb
  let cc := if maxGradNorm > 0.0 && meanNorm > maxGradNorm then maxGradNorm / meanNorm else 1.0
  return applyFlatGrad p g (lr * cc / mb)

/-! ### Rollout-forward FFI (native MLP forward in the rollout) -/

/-- Action probabilities + value from the native forward kernel (flat `params`,
    `O = numActions+1`). Bit-identical to `policyAndValue` (both use the right-folded
    forward). -/
def policyAndValueFFI (params : FloatArray) (H D A : Nat) (obs : Array Float) : Array Float × Float :=
  let out := Puffer.Float.FFI.mlpForwardFFI params (FloatArray.mk obs)
               (USize.ofNat H) (USize.ofNat D) (USize.ofNat (A + 1))
  (softmax ((Array.range A).map (fun k => out[k]!)), out[A]!)

/-- Discrete rollout using the native forward kernel (params flattened once). -/
def segmentRolloutFFI {S : Type} (env : Env S) (params : FloatArray) (H : Nat) (horizon : Nat)
    (s0 : S) (rng0 : UInt64) : Array Transition × Float × S × UInt64 × Array Float := Id.run do
  let A := env.numActions
  let D := env.obsDim
  let mut st := s0
  let mut rng := rng0
  let mut traj : Array Transition := #[]
  let mut epReturns : Array Float := #[]
  let mut epRet := 0.0
  for _ in [0:horizon] do
    let obs := env.observe st
    let (probs, v) := policyAndValueFFI params H D A obs
    let (word, rng') := rngNext rng
    rng := rng'
    let a := sampleCat probs (uniform01 word)
    let (st', r, term) := env.step st a
    traj := traj.push { obs := obs, action := a, reward := r, value := v,
                        oldLogp := Float.log probs[a]!, terminal := term }
    epRet := epRet + r
    if term then
      epReturns := epReturns.push epRet; epRet := 0.0
      let (sReset, rng'') := env.reset rng; rng := rng''; st := sReset
    else st := st'
  let (_, bootV) := policyAndValueFFI params H D A (env.observe st)
  return (traj, bootV, st, rng, epReturns)

def vecRolloutFFI {S : Type} (env : Env S) (p : MLP) (horizon : Nat)
    (states : Array S) (rng0 : UInt64) :
    Array (Array Transition) × Array Float × Array S × UInt64 × Array Float := Id.run do
  let params := flattenMLP p
  let H := p.b1.size
  let mut rng := rng0
  let mut trajs : Array (Array Transition) := #[]
  let mut bootVals : Array Float := #[]
  let mut newStates : Array S := #[]
  let mut epReturns : Array Float := #[]
  for st in states do
    let (traj, bootV, st', rng', epRets) := segmentRolloutFFI env params H horizon st rng
    rng := rng'
    trajs := trajs.push traj; bootVals := bootVals.push bootV
    newStates := newStates.push st'; epReturns := epReturns ++ epRets
  return (trajs, bootVals, newStates, rng, epReturns)

/-! ### BLAS batched-forward rollout (the GPU/BLAS dense-forward, wired in)

The vectorized rollout steps `numEnvs` env instances in lockstep, so at each timestep the
policy forward over all `N` observations is one batched dense-forward — a GEMM. This
timestep-major rollout gathers the `N` obs into `Xb[N×D]`, runs ONE batched MLP forward
(`Puffer.Float.BLAS.mlpForwardBatch{Ref,Blas}FFI`), then samples + steps each env. It
produces the same per-env `trajs`/`bootVals` as `vecRollout`, so `buildBatch`/GAE/the PPO
update are unchanged. `useBlas` selects the backend; the scalar twin is the bit-exact
reference, and both share this rollout so a benchmark isolates the forward kernel. -/
def vecRolloutBatched {S : Type} (env : Env S) (p : MLP) (horizon : Nat)
    (states : Array S) (rng0 : UInt64) (useBlas : Bool) :
    Array (Array Transition) × Array Float × Array S × UInt64 × Array Float := Id.run do
  let params := flattenMLP p
  let N := states.size
  let D := env.obsDim
  let H := p.b1.size
  let A := env.numActions
  let O := A + 1
  let u := USize.ofNat
  let fwd := fun (Xb : FloatArray) =>
    if useBlas then Puffer.Float.BLAS.mlpForwardBatchBlasFFI params Xb (u N) (u D) (u H) (u O)
    else Puffer.Float.BLAS.mlpForwardBatchRefFFI params Xb (u N) (u D) (u H) (u O)
  let mut sts := states
  let mut rng := rng0
  let mut trajs : Array (Array Transition) := Array.replicate N #[]
  let mut epReturns : Array Float := #[]
  let mut epRet : Array Float := Array.replicate N 0.0
  for _ in [0:horizon] do
    let obsArr := sts.map env.observe
    let mut xb : Array Float := #[]
    for o in obsArr do for x in o do xb := xb.push x
    let Yb := fwd (FloatArray.mk xb)
    -- Batched categorical sample in native C: `[actions(N); logps(N); values(N)]`. This replaces the
    -- per-env `softmax → sampleCat → log → value` glue (4 small Array allocs/env) with one call, and is
    -- BIT-EXACT for deterministic-reset envs: env `n` draws `rngNext` word `hash(rng+(n+1)·G)`, so
    -- advancing `rng` by `N·G` reproduces the per-env stream (`G` = splitmix64 golden ratio).
    let avl := Puffer.Float.FFI.sampleActionsBatchFFI Yb (u N) (u A) (u O) rng
    rng := rng + (UInt64.ofNat N) * 0x9E3779B97F4A7C15
    -- iterate env states element-wise (avoids needing `Inhabited S` for `sts[i]!`)
    let mut newSts : Array S := Array.mkEmpty N
    let mut i := 0
    for st in sts do
      let a := (avl[i]!).toUInt64.toNat
      let (st', r, term) := env.step st a
      trajs := trajs.set! i (trajs[i]!.push
        { obs := obsArr[i]!, action := a, reward := r, value := avl[2*N+i]!,
          oldLogp := avl[N+i]!, terminal := term })
      epRet := epRet.set! i (epRet[i]! + r)
      if term then
        epReturns := epReturns.push epRet[i]!
        epRet := epRet.set! i 0.0
        let (sReset, rng'') := env.reset rng; rng := rng''
        newSts := newSts.push sReset
      else
        newSts := newSts.push st'
      i := i + 1
    sts := newSts
  -- bootstrap values from the final observations (one more batched forward)
  let obsArr := sts.map env.observe
  let mut xb : Array Float := #[]
  for o in obsArr do for x in o do xb := xb.push x
  let Yb := fwd (FloatArray.mk xb)
  let mut bootVals : Array Float := #[]
  for i in [0:N] do bootVals := bootVals.push Yb[i*O + A]!
  return (trajs, bootVals, sts, rng, epReturns)

/-- SoA trajectory columns (env-major: env `e`'s timesteps occupy rows `e·T … e·T+T-1`) — the
    per-transition `Transition` record and `Array (Array Transition)` nesting are gone; every field is a
    flat `FloatArray`. `obs` is `(N·T)·D` row-major, kept UNBOXED (filled per timestep by `scatterObsFFI`
    from the forward's `xb`) so the minibatch gather (`gatherMinibatchFFI`) is a C row-copy rather than a
    boxed `Array Float` walk. -/
structure TrajCols where
  obs : FloatArray       -- (N·T)·D
  actions : FloatArray   -- N·T  (f64-encoded action index)
  values : FloatArray    -- N·T
  oldLogps : FloatArray  -- N·T
  rewards : FloatArray   -- N·T
  terms : FloatArray     -- N·T  (0.0 / 1.0)
  n : Nat
  t : Nat
  d : Nat

/-- SoA rollout: same policy as `vecRolloutBatched` (BLAS forward + native-C batched sampler) but writes
    the trajectory into flat `TrajCols` columns — NO per-transition `Transition` record and no
    `Array (Array …)`. Row `e·T+s` holds env `e`'s step `s` (env-major, matching `buildBatch`'s
    concatenation order, so the downstream shuffle/minibatches stay bit-identical). Skips the unused
    bootstrap forward (`buildBatch` ignores `bootVals`, keeping the last step's advantage at 0). -/
def vecRolloutBatchedSoA {S : Type} (env : Env S) (p : MLP) (horizon : Nat)
    (states : Array S) (rng0 : UInt64) (useBlas : Bool) :
    TrajCols × Array S × UInt64 × Array Float := Id.run do
  let params := flattenMLP p
  let N := states.size; let D := env.obsDim; let H := p.b1.size; let A := env.numActions; let O := A + 1
  let T := horizon
  let u := USize.ofNat
  let fwd := fun (Xb : FloatArray) =>
    if useBlas then Puffer.Float.BLAS.mlpForwardBatchBlasFFI params Xb (u N) (u D) (u H) (u O)
    else Puffer.Float.BLAS.mlpForwardBatchRefFFI params Xb (u N) (u D) (u H) (u O)
  let mut sts := states
  let mut rng := rng0
  let mut obs      := FloatArray.mk (Array.replicate (N*T*D) 0.0)
  let mut actions  := FloatArray.mk (Array.replicate (N*T) 0.0)
  let mut values   := FloatArray.mk (Array.replicate (N*T) 0.0)
  let mut oldLogps := FloatArray.mk (Array.replicate (N*T) 0.0)
  let mut rewards  := FloatArray.mk (Array.replicate (N*T) 0.0)
  let mut terms    := FloatArray.mk (Array.replicate (N*T) 0.0)
  let mut epReturns : Array Float := #[]
  let mut epRet : Array Float := Array.replicate N 0.0
  for s in [0:T] do
    let obsArr := sts.map env.observe
    let mut xb : Array Float := #[]
    for o in obsArr do for x in o do xb := xb.push x
    let xbF := FloatArray.mk xb
    let Yb := fwd xbF
    obs := Puffer.Float.FFI.scatterObsFFI obs xbF (u N) (u D) (u T) (u s)   -- fill obs column block (C)
    let avl := Puffer.Float.FFI.sampleActionsBatchFFI Yb (u N) (u A) (u O) rng
    rng := rng + (UInt64.ofNat N) * 0x9E3779B97F4A7C15
    let mut newSts : Array S := Array.mkEmpty N
    let mut e := 0
    for st in sts do
      let idx := e*T + s
      let a := (avl[e]!).toUInt64.toNat
      actions  := actions.set!  idx avl[e]!
      values   := values.set!   idx avl[2*N+e]!
      oldLogps := oldLogps.set! idx avl[N+e]!
      let (st', r, term) := env.step st a
      rewards := rewards.set! idx r
      terms   := terms.set!   idx (if term then 1.0 else 0.0)
      epRet := epRet.set! e (epRet[e]! + r)
      if term then
        epReturns := epReturns.push epRet[e]!
        epRet := epRet.set! e 0.0
        let (sReset, rng'') := env.reset rng; rng := rng''
        newSts := newSts.push sReset
      else
        newSts := newSts.push st'
      e := e + 1
    sts := newSts
  return ({ obs, actions, values, oldLogps, rewards, terms, n := N, t := T, d := D }, sts, rng, epReturns)

/-- GAE (`computePuffAdvantage` with `importance = 1`, so `ρ = c = 1`) over the SoA columns: per env `e`,
    backward over rows `e·T+T-1 … e·T` (last step keeps `adv = 0`). Returns `(advRaw, returns)` as flat
    `FloatArray`s aligned to the `TrajCols` rows. Bit-exact with `buildBatch`. -/
def buildBatchSoA (tc : TrajCols) (gamma lam : Float) : FloatArray × FloatArray := Id.run do
  let N := tc.n; let T := tc.t; let sz := N*T
  let mut adv := FloatArray.mk (Array.replicate sz 0.0)
  for e in [0:N] do
    let base := e*T
    let mut lastA := 0.0
    for i in [0:T-1] do
      let s := T - 2 - i
      let t := base + s
      let nnt := if tc.terms[t]! != 0.0 then 0.0 else 1.0
      let rr := tc.rewards[t]!
      let r := if rr > 1.0 then 1.0 else if rr < -1.0 then -1.0 else rr
      let delta := r + gamma * tc.values[t+1]! * nnt - tc.values[t]!
      lastA := delta + gamma * lam * lastA * nnt
      adv := adv.set! t lastA
  let mut returns := FloatArray.mk (Array.replicate sz 0.0)
  for t in [0:sz] do returns := returns.set! t (adv[t]! + tc.values[t]!)
  return (adv, returns)

/-! ### Persistent resident-policy rollout (the GPU policy wired into the actual rollout)

`vecRolloutResident` is `vecRolloutBatched` with the per-timestep batched policy forward run through
a PERSISTENT GPU handle (`Puffer.Float.BLAS.mlpPolicy*`): the weights and the hidden activation stay
resident, so each timestep pays only an obs upload + logits/value download instead of re-uploading
the policy. The handle is stateful — its device weights are set by `mlpPolicyUpdateFFI` before each
rollout and released at the end — so the rollout is an IO action, keeping load → update → forward →
free properly ordered. Produces the same per-env `trajs`/`bootVals` as `vecRolloutBatched`. -/
def vecRolloutResident {S : Type} (env : Env S) (handle : USize) (D H A horizon : Nat)
    (states : Array S) (rng0 : UInt64) :
    IO (Array (Array Transition) × Array Float × Array S × UInt64 × Array Float) := do
  let N := states.size
  let O := A + 1
  let fwd := fun (Xb : FloatArray) => Puffer.Float.BLAS.mlpPolicyForwardFFI handle Xb (USize.ofNat N)
  let mut sts := states
  let mut rng := rng0
  let mut trajs : Array (Array Transition) := Array.replicate N #[]
  let mut epReturns : Array Float := #[]
  let mut epRet : Array Float := Array.replicate N 0.0
  for _ in [0:horizon] do
    let obsArr := sts.map env.observe
    let mut xb : Array Float := #[]
    for o in obsArr do for x in o do xb := xb.push x
    let Yb := fwd (FloatArray.mk xb)
    let mut newSts : Array S := Array.mkEmpty N
    let mut i := 0
    for st in sts do
      let out := (Array.range O).map (fun k => Yb[i*O + k]!)
      let probs := softmax ((Array.range A).map (fun k => out[k]!))
      let v := out[A]!
      let (word, rng') := rngNext rng; rng := rng'
      let a := sampleCat probs (uniform01 word)
      let (st', r, term) := env.step st a
      trajs := trajs.set! i (trajs[i]!.push
        { obs := obsArr[i]!, action := a, reward := r, value := v,
          oldLogp := Float.log probs[a]!, terminal := term })
      epRet := epRet.set! i (epRet[i]! + r)
      if term then
        epReturns := epReturns.push epRet[i]!
        epRet := epRet.set! i 0.0
        let (sReset, rng'') := env.reset rng; rng := rng''
        newSts := newSts.push sReset
      else
        newSts := newSts.push st'
      i := i + 1
    sts := newSts
  let obsArrF := sts.map env.observe
  let mut xbF : Array Float := #[]
  for o in obsArrF do for x in o do xbF := xbF.push x
  let YbF := fwd (FloatArray.mk xbF)
  let mut bootVals : Array Float := #[]
  for j in [0:N] do bootVals := bootVals.push YbF[j*O + A]!
  return (trajs, bootVals, sts, rng, epReturns)

/-- **The generic `puffer train <env>` trainer.** Env-AGNOSTIC: it drives a runtime-loaded env plugin
    (`Puffer.Plugin`, dlopen'd `libenv_<name>.so`) purely through the C ABI — `puffer` never sees the
    env at compile time. Per timestep it does one batched GPU forward (`cudaMlpForwardFFI`) + one batched
    device sample (`cudaSampleActionsFFI`) + one native `envStep` over all N copies, scatters the SoA
    columns, then GAE (`buildBatchSoA`) → resident PPO+Muon (`cudaTrainUpdateFFI`). The per-timestep loop
    is O(T) (all N-parallelism is inside the batched calls), so this matches PufferLib's model: native C
    envs + a GPU policy, obs/actions crossing as bulk batched buffers. `config` is the env's `k=v,…` string. -/
def trainPluginEnv (name config : String)
    (hidden numEnvs horizon totalTimesteps epochs numMB : Nat)
    (lr wd mu eps gamma lam vfCoef entCoef clipEps vfClip maxGradNorm minLrRatio : Float)
    (bf16 : UInt8) (logDash : Bool := false) (seed : UInt64) : IO Unit := do
  let u := USize.ofNat; let mk := FloatArray.mk
  let h ← Puffer.Plugin.envOpen name (u numEnvs) seed config
  if h == 0 then
    IO.println s!"puffer train: env '{name}' not found — build ocean/{name}/libenv_{name}.so (e.g. `ocean/build.sh {name}`)"
    return
  let D := (Puffer.Plugin.envObsDim h).toNat
  let A := (Puffer.Plugin.envNumActions h).toNat
  let nAgents := (Puffer.Plugin.envNumAgents h).toNat
  let N := numEnvs * nAgents            -- batch rows: each of the env's agents is a training row
  let O := A + 1; let H := hidden; let T := horizon; let NT := N*T; let P := H*D + H + O*H + O
  -- PufferLib: train_epochs = total_timesteps / (agents · horizon). N is the full agent batch.
  let updates := max 1 (totalTimesteps / (N * T))
  let G : UInt64 := 0x9E3779B97F4A7C15
  let (p0, _) := initMLP D H O seed
  let flat0 := flattenMLP p0
  let pm0 : FloatArray := mk ((Array.range P).map (fun i => flat0[i]!) ++ Array.replicate P 0.0)
  -- Policy weights live device-RESIDENT for the whole run: the rollout reads them and the optimizer
  -- updates them in place, so [params;mom] never round-trips the PCIe bus per update (PufferLib-style).
  let ph ← Puffer.Float.CUDA.policyLoadFFI pm0 (u P)
  if ph == 0 then
    IO.println "puffer train: policy weights device alloc failed (out of VRAM?)"
    Puffer.Plugin.envClose h; return
  -- Enable the buffered + graph-replayed rollout path (N-way concurrent-stream buffer split, per-buffer
  -- CUDA graph replay of the per-step forward): already built, already measured (breakout MLP@4096
  -- 4.73M→7.25M SPS), but dormant since launch because nothing in the CLI ever set its env-var gate.
  -- Size-aware count (not a blanket max) avoids the too-small-per-buffer regime an earlier sweep found.
  Puffer.Float.CUDA.cudaSetRollBuffersFFI (u (min 8 numEnvs)) 1
  let mut rng := seed
  let agentStr := if nAgents == 1 then "" else s!" × {nAgents} agents"
  IO.println s!"puffer train [{name}] — {numEnvs} envs{agentStr} × {T} horizon (batch {N}), {D}→{H}→{O} MLP, PPO+Muon on GPU (ocean env plugin via C ABI), {updates} updates"
  (← IO.getStdout).flush
  let mut obs ← Puffer.Plugin.envReset h            -- N·D
  let mut rollNs : Nat := 0; let mut trainNs : Nat := 0; let mut updNs : Nat := 0
  -- The env's OWN PufferLib `Log` (optional plugin channel; `#[]` if this .so doesn't export it), read
  -- + zeroed once per update exactly as `static_vec_log` does, and shown at the print cadence. Metrics
  -- only: nothing below it feeds the rollout, GAE or the optimizer.
  let mut lastLog : Array (String × Float) := #[]
  -- `--log` live dashboard state (opt-in; without it the ad-hoc per-update lines stay byte-identical).
  -- The 7 losses are surfaced from the whole-update resident kernel ONLY on render frames (0.6s cadence,
  -- last minibatch of the last epoch), toggled by cudaMgLossEnableFFI — a read-only D2H reduction that
  -- cannot perturb determinism. rollNs/trainNs already accumulate every update (the phase timers below).
  let tStart ← IO.monoNanosNow
  let mut idx : Nat := 0; let mut lastRenderNs : Nat := 0
  let mut lrRoll : Nat := 0; let mut lrTrain : Nat := 0; let mut lrUpd : Nat := 0
  for upd in [0:updates] do
    let ub0 ← IO.monoNanosNow
    -- Decide the 0.6s render frame at update START so the loss readback lines up with this update.
    let willRender := logDash && (ub0 ≥ lastRenderNs + 600000000 || upd + 1 == updates)
    let lrNow := cosineLr lr minLrRatio upd updates
    let rolloutRng := rng
    rng := rng + (UInt64.ofNat NT) * G
    let r0 ← IO.monoNanosNow
    -- Native per-update rollout in ONE FFI call: resident weights, device forward+sample, CPU env-step,
    -- C column scatter — no per-step Lean interpreter loop / weight re-upload / logits round-trip.
    let roll ← Puffer.Float.CUDA.cudaPluginRolloutFFI h ph obs (u N) (u D) (u H) (u A) (u T) bf16 rolloutRng
    let obsCol  := roll   -- obsCol is roll's prefix [0, NT·D); readers use only that, so skip the 62MB slice-copy
    let actCol  := Puffer.Float.FFI.sliceFFI roll (u (NT*D)) (u NT)
    let logpCol := Puffer.Float.FFI.sliceFFI roll (u (NT*D + NT)) (u NT)
    let valCol  := Puffer.Float.FFI.sliceFFI roll (u (NT*D + 2*NT)) (u NT)
    let rewCol  := Puffer.Float.FFI.sliceFFI roll (u (NT*D + 3*NT)) (u NT)
    let termCol := Puffer.Float.FFI.sliceFFI roll (u (NT*D + 4*NT)) (u NT)
    obs := Puffer.Float.FFI.sliceFFI roll (u (NT*D + 5*NT)) (u (N*D))   -- persistent obs → next update
    -- GAE in native C (was ~35ms/update boxed-Lean buildBatchSoA — the biggest slice of rollout+GAE)
    let gaeOut := Puffer.Float.FFI.gaeSoaFFI valCol rewCol termCol (u N) (u T) gamma lam
    let advRaw  := Puffer.Float.FFI.sliceFFI gaeOut (u 0) (u (N*T))
    let returns := Puffer.Float.FFI.sliceFFI gaeOut (u (N*T)) (u (N*T))
    if advRaw.get! 0 > 1.0e300 then IO.println ""
    let r1 ← IO.monoNanosNow; rollNs := rollNs + (r1 - r0)
    -- Flat per-epoch shuffle built in C (was epochs·NT Array.push in interpreted Lean — the top host cost).
    let permFlat := Puffer.Float.FFI.shufflePermFFI (u NT) (u epochs) rng
    rng := rng + (UInt64.ofNat (epochs * NT)) * G
    -- Enable the loss readback ONLY on render frames; reset to 0 otherwise so a prior frame's enable
    -- doesn't leak into the next (non-render) update. The resident kernel then reduces the 7 losses on
    -- its last minibatch of the last epoch (read-only), which cudaMgReadLossesFFI reads below.
    if logDash then Puffer.Float.CUDA.cudaMgLossEnableFFI (if willRender then 1 else 0)
    let t0 ← IO.monoNanosNow
    -- Resident PPO+Muon: updates the device-resident [params;mom] in place; returns only params[0] (guard).
    -- `valCol` is PufferLib's `mb_values`: the value-loss clip and the global grad-norm clip
    -- (train.vf_clip_coef / train.max_grad_norm) both run inside the resident step now.
    let g0 ← Puffer.Float.CUDA.cudaTrainUpdateResidentFFI ph obsCol actCol advRaw returns logpCol valCol permFlat
      (u NT) (u D) (u H) (u A) (u epochs) (u numMB) lrNow wd mu eps vfCoef entCoef clipEps vfClip maxGradNorm bf16
    if g0.get! 0 > 1.0e300 then IO.println ""
    let t1 ← IO.monoNanosNow; trainNs := trainNs + (t1 - t0)
    let ub1 ← IO.monoNanosNow; updNs := updNs + (ub1 - ub0)
    let lg ← Puffer.Plugin.envLogPairs h
    if !lg.isEmpty then lastLog := lg
    -- Live dashboard (PufferLib's rich monitor): redraw in place on the render frame decided at update start.
    if willRender then
      let now ← IO.monoNanosNow
      let lossArr ← Puffer.Float.CUDA.cudaMgReadLossesFFI
      let du := max 1 (upd + 1 - lrUpd)                 -- updates since last render (per-update avg)
      let per := fun (cur last : Nat) => Float.ofNat ((cur - last) / du) / 1.0e9
      Puffer.RL.Dashboard.redrawFrom name P ((upd+1)*NT) (upd+1) tStart now totalTimesteps
        (per rollNs lrRoll) (per trainNs lrTrain) 0.0 0.0
        lossArr (lastLog.filter (fun kv => kv.1 != "n")) idx
      lrRoll := rollNs; lrTrain := trainNs; lrUpd := upd + 1
      idx := (idx + 9) % 10; lastRenderNs := now
    if !logDash && ((upd+1) % 20 == 0 || upd == 0) then
      let mut rsum := 0.0; let mut nterm := 0
      for i in [0:NT] do rsum := rsum + rewCol[i]!; if termCol[i]! > 0.5 then nterm := nterm + 1
      -- Per-step mean is the ONLY universally-defined signal reconstructible from the step buffers:
      -- many ocean envs (target, minimal, convert, drive, snake, rware, whackamole, …) never set a
      -- terminal flag, so a terminal-normalised mean shows a flat zero for them. `env log` beside it
      -- is upstream PufferLib's own per-env `Log` — the comparable episode statistics.
      let meanStep := rsum / Float.ofNat NT
      let epPart := if nterm == 0 then "" else s!", mean ep return {rsum / Float.ofNat nterm}"
      IO.println s!"  update {upd+1}: batch reward Σ={rsum} (mean/step {meanStep}) over {nterm} terminals{epPart}{fmtEnvLog lastLog}"
      (← IO.getStdout).flush
  Puffer.Float.CUDA.policyFreeFFI ph          -- release the device-resident policy buffer
  Puffer.Plugin.envClose h
  IO.println "done."
  let envSteps := updates * N * T
  let ms := fun (ns : Nat) => Float.ofNat (ns / 1000) / 1000.0
  let sps := if updNs == 0 then 0 else envSteps * 1000000000 / updNs
  IO.println s!"perf: {envSteps} env-steps in {ms updNs}ms ⇒ {sps} SPS  |  rollout+GAE {ms rollNs}ms   GPU-train {ms trainNs}ms"

/-- **Multi-discrete plugin trainer** — the `trainPluginEnv` twin for envs with `K>1` categorical action
    heads. Rollout: GPU forward → `cudaSampleActionsMDFFI` (samples each head) → native `envStep` with the
    `K` per-agent actions. GAE (joint log-prob as `oldLogp`) → per-minibatch `cudaMlpPpoGradMDFFI` (the
    verified md gradient) → native-C Muon. Action column is `NT·K` (row-major). -/
def trainPluginEnvMD (name config : String)
    (hidden numEnvs horizon totalTimesteps epochs numMB : Nat)
    (lr wd mu eps gamma lam vfCoef entCoef clipEps vfClip maxGradNorm minLrRatio : Float)
    (bf16 : UInt8) (logDash : Bool := false) (seed : UInt64) : IO Unit := do
  let u := USize.ofNat; let mk := FloatArray.mk
  let h ← Puffer.Plugin.envOpen name (u numEnvs) seed config
  if h == 0 then IO.println s!"puffer train: env '{name}' not found — run ocean/build.sh {name}"; return
  let D := (Puffer.Plugin.envObsDim h).toNat
  let nAgents := (Puffer.Plugin.envNumAgents h).toNat
  let N := numEnvs * nAgents
  let K := (Puffer.Plugin.envNHeads h).toNat
  let headSizes := Puffer.Plugin.envHeadSizes h
  let A := headSizes.foldl (·+·) 0
  let O := A + 1; let H := hidden; let T := horizon; let NT := N*T; let P := H*D + H + O*H + O
  let hsF := mk (headSizes.map Float.ofNat)
  let updates := max 1 (totalTimesteps / (N * T))
  let G : UInt64 := 0x9E3779B97F4A7C15
  let (p0, _) := initMLP D H O seed; let flat0 := flattenMLP p0
  let mut pm : FloatArray := mk ((Array.range P).map (fun i => flat0[i]!) ++ Array.replicate P 0.0)
  -- Policy weights + momentum live device-RESIDENT: the rollout reads them and the resident Muon updates
  -- them in place, so [params;mom] no longer round-trips PCIe per minibatch (only the raw gradient uploads).
  let ph ← Puffer.Float.CUDA.policyLoadFFI pm (u P)
  if ph == 0 then
    IO.println "puffer train: policy weights device alloc failed (out of VRAM?)"
    Puffer.Plugin.envClose h; return
  -- Enable the buffered + graph-replayed rollout path (ported from the MLP plugin trainer). 10, not
  -- MLP's own 8: a direct sweep (6/8/10/12/14, both target and drone, real config scale) found MD/Cont's
  -- own shape peaks at 10 (~4%/~3% further over 8) and MLP's breakout is a wash at 10 vs its established
  -- 8 -- different trainers, different optimal buffer count, measured rather than copied.
  Puffer.Float.CUDA.cudaSetRollBuffersFFI (u (min 10 numEnvs)) 1
  let mut rng := seed
  IO.println s!"puffer train [{name}] MULTI-DISCRETE ({K} heads {headSizes}) — {numEnvs} envs × {nAgents} agents × {T} (batch {N}), {D}→{H}→{O} MLP, {updates} updates"
  (← IO.getStdout).flush
  let mut obs ← Puffer.Plugin.envReset h
  let mut updNs : Nat := 0
  let mut lastLog : Array (String × Float) := #[]   -- env's own PufferLib `Log`, latest non-empty window
  -- `--log` live dashboard state (opt-in; the ad-hoc lines below stay byte-identical without it). Phase
  -- timers + the 7-loss readback (last minibatch of the last epoch inside the resident kernel) are taken
  -- ONLY under logDash, so a non-dashboard run is unchanged.
  let tStart ← IO.monoNanosNow
  let mut idx : Nat := 0; let mut lastRenderNs : Nat := 0
  let mut rollNs : Nat := 0; let mut trainNs : Nat := 0
  let mut lrRoll : Nat := 0; let mut lrTrain : Nat := 0; let mut lrUpd : Nat := 0
  for upd in [0:updates] do
    let ub0 ← IO.monoNanosNow
    let willRender := logDash && (ub0 ≥ lastRenderNs + 600000000 || upd + 1 == updates)
    let lrNow := cosineLr lr minLrRatio upd updates
    let rolloutRng := rng; rng := rng + (UInt64.ofNat NT) * G
    let r0 ← if logDash then IO.monoNanosNow else pure (0 : Nat)
    -- Native per-update rollout in ONE FFI call (resident weights, device forward + md-sample, CPU
    -- env-step, C column scatter) — the multi-discrete twin of trainPluginEnv's native driver.
    let roll ← Puffer.Float.CUDA.cudaPluginRolloutMultiFFI h ph obs hsF (u N) (u D) (u H) (u O) (u K) (u T) 1 bf16 rolloutRng
    let obsCol  := roll   -- obsCol is roll's prefix [0, NT·D); readers use only that, so skip the 62MB slice-copy
    let actCol  := Puffer.Float.FFI.sliceFFI roll (u (NT*D)) (u (NT*K))
    let logpCol := Puffer.Float.FFI.sliceFFI roll (u (NT*D + NT*K)) (u NT)
    let valCol  := Puffer.Float.FFI.sliceFFI roll (u (NT*D + NT*K + NT)) (u NT)
    let rewCol  := Puffer.Float.FFI.sliceFFI roll (u (NT*D + NT*K + 2*NT)) (u NT)
    let termCol := Puffer.Float.FFI.sliceFFI roll (u (NT*D + NT*K + 3*NT)) (u NT)
    obs := Puffer.Float.FFI.sliceFFI roll (u (NT*D + NT*K + 4*NT)) (u (N*D))   -- persistent obs → next update
    -- GAE in native C (was ~35ms/update boxed-Lean buildBatchSoA — the biggest slice of rollout+GAE)
    let gaeOut := Puffer.Float.FFI.gaeSoaFFI valCol rewCol termCol (u N) (u T) gamma lam
    let advRaw  := Puffer.Float.FFI.sliceFFI gaeOut (u 0) (u (N*T))
    let returns := Puffer.Float.FFI.sliceFFI gaeOut (u (N*T)) (u (N*T))
    if advRaw.get! 0 > 1.0e300 then IO.println ""
    -- Per-epoch shuffle built in C (was interpreted-Lean permFlat) → one flat perm; the whole update
    -- (gather → adv-norm → md-grad → Muon) runs entirely on the GPU in ONE resident call.
    let permFlat := Puffer.Float.FFI.shufflePermFFI (u NT) (u epochs) rng
    rng := rng + (UInt64.ofNat (epochs * NT)) * G
    if logDash then rollNs := rollNs + ((← IO.monoNanosNow) - r0)   -- rollout+GAE+shuffle phase
    -- Enable the loss readback ONLY on render frames (reset to 0 otherwise); the resident kernel reduces
    -- the 7 losses on its last minibatch of the last epoch (read-only), read by cudaMgReadLossesFFI below.
    if logDash then Puffer.Float.CUDA.cudaMgLossEnableFFI (if willRender then 1 else 0)
    let tt0 ← if logDash then IO.monoNanosNow else pure (0 : Nat)
    let g0 ← Puffer.Float.CUDA.cudaTrainUpdateWideResidentFFI ph obsCol actCol advRaw returns logpCol valCol permFlat hsF
               (u NT) (u D) (u H) (u O) (u K) 1 (u epochs) (u numMB) lrNow wd mu eps vfCoef entCoef clipEps
               vfClip maxGradNorm bf16
    if logDash then trainNs := trainNs + ((← IO.monoNanosNow) - tt0)
    if g0.get! 0 > 1.0e300 then IO.println ""
    let ub1 ← IO.monoNanosNow; updNs := updNs + (ub1 - ub0)
    let lg ← Puffer.Plugin.envLogPairs h        -- env's own PufferLib `Log` (read + zero); metrics only
    if !lg.isEmpty then lastLog := lg
    -- Live dashboard (PufferLib's rich monitor): redraw in place on the render frame decided at update start.
    if willRender then
      let now ← IO.monoNanosNow
      let lossArr ← Puffer.Float.CUDA.cudaMgReadLossesFFI
      let du := max 1 (upd + 1 - lrUpd)
      let per := fun (cur last : Nat) => Float.ofNat ((cur - last) / du) / 1.0e9
      Puffer.RL.Dashboard.redrawFrom name P ((upd+1)*NT) (upd+1) tStart now totalTimesteps
        (per rollNs lrRoll) (per trainNs lrTrain) 0.0 0.0
        lossArr (lastLog.filter (fun kv => kv.1 != "n")) idx
      lrRoll := rollNs; lrTrain := trainNs; lrUpd := upd + 1
      idx := (idx + 9) % 10; lastRenderNs := now
    if !logDash && ((upd+1) % 20 == 0 || upd == 0) then
      let mut rsum := 0.0; let mut nterm := 0
      for i in [0:NT] do rsum := rsum + rewCol[i]!; if termCol[i]! > 0.5 then nterm := nterm + 1
      let meanStep := rsum / Float.ofNat NT
      let epPart := if nterm == 0 then "" else s!", mean ep return {rsum / Float.ofNat nterm}"
      IO.println s!"  update {upd+1}: batch reward Σ={rsum} (mean/step {meanStep}) over {nterm} terminals{epPart}{fmtEnvLog lastLog}"
      (← IO.getStdout).flush
  Puffer.Float.CUDA.policyFreeFFI ph          -- release the device-resident policy buffer
  Puffer.Plugin.envClose h
  let envSteps := updates * N * T
  let sps := if updNs == 0 then 0 else envSteps * 1000000000 / updNs
  IO.println s!"done. perf: {envSteps} env-steps ⇒ {sps} SPS"

/-- **Continuous (diagonal-Gaussian) plugin trainer** — the `trainPluginEnv` twin for envs with a Box
    action space (`envIsCont h = 1`). The net head is `2·d+1` (means, raw logstds, value). Rollout: GPU
    forward → `cudaSampleActionsContFFI` (`aᵢ = μᵢ+σᵢ·zᵢ`) → native `envStep` with the `d` real actions.
    GAE (Gaussian log-prob as `oldLogp`) → per-minibatch `cudaMlpPpoGradContFFI` (the verified Gaussian
    gradient) → native-C Muon. Action column is `NT·d` (row-major real values). -/
def trainPluginEnvCont (name config : String)
    (hidden numEnvs horizon totalTimesteps epochs numMB : Nat)
    (lr wd mu eps gamma lam vfCoef entCoef clipEps vfClip maxGradNorm minLrRatio : Float)
    (bf16 : UInt8) (logDash : Bool := false) (seed : UInt64) : IO Unit := do
  let u := USize.ofNat; let mk := FloatArray.mk
  let h ← Puffer.Plugin.envOpen name (u numEnvs) seed config
  if h == 0 then IO.println s!"puffer train: env '{name}' not found — run ocean/build.sh {name}"; return
  let D := (Puffer.Plugin.envObsDim h).toNat
  let nAgents := (Puffer.Plugin.envNumAgents h).toNat
  let N := numEnvs * nAgents
  let d := (Puffer.Plugin.envNHeads h).toNat          -- continuous: nHeads = action dim
  let O := 2*d + 1; let H := hidden; let T := horizon; let NT := N*T; let P := H*D + H + O*H + O
  let updates := max 1 (totalTimesteps / (N * T))
  let G : UInt64 := 0x9E3779B97F4A7C15
  let (p0, _) := initMLP D H O seed; let flat0 := flattenMLP p0
  -- NOTE (measured 2026-08-04): PufferLib parameterises the Gaussian with a STATE-INDEPENDENT logstd
  -- (a bare {1,output_dim} parameter, src/models.cu:510); ours is state-DEPENDENT (rows d..2d of the
  -- output head). We built their version — zero the logstd rows of W2 at init and drop their gradient
  -- each minibatch, so logstd reduces to the learned bias — and MEASURED IT WORSE on our stack:
  -- drone 4621 -> 3825 and squared_continuous's peak 0.73 -> 0.31 on 2 of 3 seeds. Kept ours.
  -- Reference point for the whole path: PufferLib's own squared_continuous reaches -0.76 @6M steps,
  -- -0.49 @50M and +0.29 @200M, while ours reaches +0.27..+0.77 @6M — their 200M level at 3% of the
  -- budget. Gradient math is verified exact against finite differences (verify-cont-grad).
  let mut pm : FloatArray := mk ((Array.range P).map (fun i => flat0[i]!) ++ Array.replicate P 0.0)
  -- Policy weights + momentum device-RESIDENT (rollout reads them; resident Muon updates in place).
  let ph ← Puffer.Float.CUDA.policyLoadFFI pm (u P)
  if ph == 0 then
    IO.println "puffer train: policy weights device alloc failed (out of VRAM?)"
    Puffer.Plugin.envClose h; return
  -- Enable the buffered + graph-replayed rollout path (ported from the MLP plugin trainer). 10, not
  -- MLP's own 8: a direct sweep (6/8/10/12/14, both target and drone, real config scale) found MD/Cont's
  -- own shape peaks at 10 (~4%/~3% further over 8) and MLP's breakout is a wash at 10 vs its established
  -- 8 -- different trainers, different optimal buffer count, measured rather than copied.
  Puffer.Float.CUDA.cudaSetRollBuffersFFI (u (min 10 numEnvs)) 1
  let mut rng := seed
  IO.println s!"puffer train [{name}] CONTINUOUS ({d} Gaussian dims) — {numEnvs} envs × {nAgents} agents × {T} (batch {N}), {D}→{H}→2·{d}+1 MLP, {updates} updates"
  (← IO.getStdout).flush
  let mut obs ← Puffer.Plugin.envReset h
  let mut updNs : Nat := 0
  let mut lastLog : Array (String × Float) := #[]   -- env's own PufferLib `Log`, latest non-empty window
  -- `--log` live dashboard state (opt-in; the ad-hoc lines below stay byte-identical without it). Phase
  -- timers + the 7-loss readback (last minibatch of the last epoch inside the resident kernel) are taken
  -- ONLY under logDash, so a non-dashboard run is unchanged.
  let tStart ← IO.monoNanosNow
  let mut idx : Nat := 0; let mut lastRenderNs : Nat := 0
  let mut rollNs : Nat := 0; let mut trainNs : Nat := 0
  let mut lrRoll : Nat := 0; let mut lrTrain : Nat := 0; let mut lrUpd : Nat := 0
  for upd in [0:updates] do
    let ub0 ← IO.monoNanosNow
    let willRender := logDash && (ub0 ≥ lastRenderNs + 600000000 || upd + 1 == updates)
    let lrNow := cosineLr lr minLrRatio upd updates
    let rolloutRng := rng; rng := rng + (UInt64.ofNat NT) * G
    let r0 ← if logDash then IO.monoNanosNow else pure (0 : Nat)
    -- Native per-update rollout in ONE FFI call (resident weights, device forward + Gaussian-sample, CPU
    -- env-step, C column scatter) — the continuous twin of trainPluginEnv's native driver. hsF unused (mode 2).
    let roll ← Puffer.Float.CUDA.cudaPluginRolloutMultiFFI h ph obs (mk #[]) (u N) (u D) (u H) (u O) (u d) (u T) 2 bf16 rolloutRng
    let obsCol  := roll   -- obsCol is roll's prefix [0, NT·D); readers use only that, so skip the 62MB slice-copy
    let actCol  := Puffer.Float.FFI.sliceFFI roll (u (NT*D)) (u (NT*d))
    let logpCol := Puffer.Float.FFI.sliceFFI roll (u (NT*D + NT*d)) (u NT)
    let valCol  := Puffer.Float.FFI.sliceFFI roll (u (NT*D + NT*d + NT)) (u NT)
    let rewCol  := Puffer.Float.FFI.sliceFFI roll (u (NT*D + NT*d + 2*NT)) (u NT)
    let termCol := Puffer.Float.FFI.sliceFFI roll (u (NT*D + NT*d + 3*NT)) (u NT)
    obs := Puffer.Float.FFI.sliceFFI roll (u (NT*D + NT*d + 4*NT)) (u (N*D))   -- persistent obs → next update
    -- GAE in native C (was ~35ms/update boxed-Lean buildBatchSoA — the biggest slice of rollout+GAE)
    let gaeOut := Puffer.Float.FFI.gaeSoaFFI valCol rewCol termCol (u N) (u T) gamma lam
    let advRaw  := Puffer.Float.FFI.sliceFFI gaeOut (u 0) (u (N*T))
    let returns := Puffer.Float.FFI.sliceFFI gaeOut (u (N*T)) (u (N*T))
    if advRaw.get! 0 > 1.0e300 then IO.println ""
    -- Per-epoch shuffle built in C (was interpreted-Lean permFlat) → one flat perm; the whole update
    -- (gather → adv-norm → Gaussian-grad → Muon) runs entirely on the GPU in ONE resident call. hsF unused.
    let permFlat := Puffer.Float.FFI.shufflePermFFI (u NT) (u epochs) rng
    rng := rng + (UInt64.ofNat (epochs * NT)) * G
    if logDash then rollNs := rollNs + ((← IO.monoNanosNow) - r0)   -- rollout+GAE+shuffle phase
    -- Enable the loss readback ONLY on render frames (reset to 0 otherwise); the resident kernel reduces
    -- the 7 losses on its last minibatch of the last epoch (read-only), read by cudaMgReadLossesFFI below.
    if logDash then Puffer.Float.CUDA.cudaMgLossEnableFFI (if willRender then 1 else 0)
    let tt0 ← if logDash then IO.monoNanosNow else pure (0 : Nat)
    let g0 ← Puffer.Float.CUDA.cudaTrainUpdateWideResidentFFI ph obsCol actCol advRaw returns logpCol valCol permFlat (mk #[])
               (u NT) (u D) (u H) (u O) (u d) 2 (u epochs) (u numMB) lrNow wd mu eps vfCoef entCoef clipEps
               vfClip maxGradNorm bf16
    if logDash then trainNs := trainNs + ((← IO.monoNanosNow) - tt0)
    if g0.get! 0 > 1.0e300 then IO.println ""
    let ub1 ← IO.monoNanosNow; updNs := updNs + (ub1 - ub0)
    let lg ← Puffer.Plugin.envLogPairs h        -- env's own PufferLib `Log` (read + zero); metrics only
    if !lg.isEmpty then lastLog := lg
    -- Live dashboard (PufferLib's rich monitor): redraw in place on the render frame decided at update start.
    if willRender then
      let now ← IO.monoNanosNow
      let lossArr ← Puffer.Float.CUDA.cudaMgReadLossesFFI
      let du := max 1 (upd + 1 - lrUpd)
      let per := fun (cur last : Nat) => Float.ofNat ((cur - last) / du) / 1.0e9
      Puffer.RL.Dashboard.redrawFrom name P ((upd+1)*NT) (upd+1) tStart now totalTimesteps
        (per rollNs lrRoll) (per trainNs lrTrain) 0.0 0.0
        lossArr (lastLog.filter (fun kv => kv.1 != "n")) idx
      lrRoll := rollNs; lrTrain := trainNs; lrUpd := upd + 1
      idx := (idx + 9) % 10; lastRenderNs := now
    if !logDash && ((upd+1) % 20 == 0 || upd == 0) then
      let mut rsum := 0.0; let mut nterm := 0
      for i in [0:NT] do rsum := rsum + rewCol[i]!; if termCol[i]! > 0.5 then nterm := nterm + 1
      let meanStep := rsum / Float.ofNat NT
      let epPart := if nterm == 0 then "" else s!", mean ep return {rsum / Float.ofNat nterm}"
      IO.println s!"  update {upd+1}: batch reward Σ={rsum} (mean/step {meanStep}) over {nterm} terminals{epPart}{fmtEnvLog lastLog}"
      (← IO.getStdout).flush
  Puffer.Float.CUDA.policyFreeFFI ph          -- release the device-resident policy buffer
  Puffer.Plugin.envClose h
  let envSteps := updates * N * T
  let sps := if updNs == 0 then 0 else envSteps * 1000000000 / updNs
  IO.println s!"done. perf: {envSteps} env-steps ⇒ {sps} SPS"

/-! ### LSTM head — native truncated-BPTT gradient trainer

The recurrent gradient needs backprop through time: the whole sequence's forward unroll
on one AD tape, reverse-mode over it. That tape is the largest per-*sequence* structure
in the trainer, so the native BPTT kernel (recurrence in C, one call per env-sequence) is
the accelerator. The recurrent trainer is per-sequence (not per-transition minibatched),
so `updateRecSeqFFI` folds the C gradient + the same mean-scale/grad-clip as `applyRecGrad`
into one step. The Lean rollout forward is kept (not the bottleneck). -/

/-- Flatten a `RecPolicy` to the params layout the LSTM C kernel expects
    (`Wx`, `Wh`, `bih`, `Wo`, `bo`). -/
def flattenRec (p : RecPolicy) : FloatArray := Id.run do
  let mut a : FloatArray := FloatArray.emptyWithCapacity 0
  for row in p.Wx do for x in row do a := a.push x
  for row in p.Wh do for x in row do a := a.push x
  for x in p.bih do a := a.push x
  for row in p.Wo do for x in row do a := a.push x
  for x in p.bo do a := a.push x
  return a

/-- Apply a flat LSTM gradient (C-kernel layout) as `p := p + scale·grad`, reconstructing
    the `Wx, Wh, bih, Wo, bo` tensors in order. -/
def applyFlatGradRec (p : RecPolicy) (g : FloatArray) (scale : Float) : RecPolicy := Id.run do
  let mut off := 0
  let mut Wx := p.Wx
  for k in [0:Wx.size] do
    let mut row := Wx[k]!; let n := row.size
    for i in [0:n] do row := row.set! i (row[i]! + scale * g[off + i]!)
    off := off + n; Wx := Wx.set! k row
  let mut Wh := p.Wh
  for k in [0:Wh.size] do
    let mut row := Wh[k]!; let n := row.size
    for i in [0:n] do row := row.set! i (row[i]! + scale * g[off + i]!)
    off := off + n; Wh := Wh.set! k row
  let mut bih := p.bih
  for i in [0:bih.size] do bih := bih.set! i (bih[i]! + scale * g[off + i]!)
  off := off + bih.size
  let mut Wo := p.Wo
  for m in [0:Wo.size] do
    let mut row := Wo[m]!; let n := row.size
    for i in [0:n] do row := row.set! i (row[i]! + scale * g[off + i]!)
    off := off + n; Wo := Wo.set! m row
  let mut bo := p.bo
  for i in [0:bo.size] do bo := bo.set! i (bo[i]! + scale * g[off + i]!)
  return { p with Wx := Wx, Wh := Wh, bih := bih, Wo := Wo, bo := bo }

/-- Pack one env-sequence into the LSTM kernel's flat arrays `(obsSeq [T·D], acts [T],
    oldlps [T], terms [T])` (advantages/returns come from the caller's GAE). -/
def mkSeqArrays (traj : Array Transition) : FloatArray × FloatArray × FloatArray × FloatArray := Id.run do
  let mut obsSeq : FloatArray := FloatArray.emptyWithCapacity 0
  let mut acts : FloatArray := FloatArray.emptyWithCapacity 0
  let mut olps : FloatArray := FloatArray.emptyWithCapacity 0
  let mut terms : FloatArray := FloatArray.emptyWithCapacity 0
  for tr in traj do
    for x in tr.obs do obsSeq := obsSeq.push x
    acts := acts.push (Float.ofNat tr.action)
    olps := olps.push tr.oldLogp
    terms := terms.push (if tr.terminal then 1.0 else 0.0)
  return (obsSeq, acts, olps, terms)

/-- One BPTT ascent step for a sequence via the native C kernel (same mean-scale over
    `seqLen` + global grad-norm clip as `applyRecGrad`). -/
def updateRecSeqFFI (p : RecPolicy) (traj : Array Transition) (h0 c0 advN returns : Array Float)
    (numActions : Nat) (lr maxGradNorm vfCoef entCoef clipEps : Float) : RecPolicy × Float := Id.run do
  let params := flattenRec p
  let H := p.hSize
  let D := (p.Wx[0]!).size
  let (obsSeq, acts, olps, terms) := mkSeqArrays traj
  let g := Puffer.Float.FFI.lstmPPOGradSeqFFI params obsSeq acts (FloatArray.mk advN)
             (FloatArray.mk returns) olps terms (FloatArray.mk h0) (FloatArray.mk c0)
             (USize.ofNat traj.size) (USize.ofNat H) (USize.ofNat D) (USize.ofNat numActions) vfCoef entCoef clipEps
  let mut sq := 0.0
  for i in [0:g.size] do sq := sq + g[i]! * g[i]!
  let sl := Float.ofNat (max traj.size 1)
  let meanNorm := Float.sqrt sq / sl
  let cc := if maxGradNorm > 0.0 && meanNorm > maxGradNorm then maxGradNorm / meanNorm else 1.0
  return (applyFlatGradRec p g (lr * cc / sl), meanNorm)

/-- **Recurrent (LSTM) plugin trainer** — the `trainPluginEnv` twin for POMDP envs where a
    feed-forward policy can't solve the task because the cue is only observed once. Policy is a single LSTM
    layer + linear head (`RecVecTrain.RecPolicy`); the hidden state `(h,c)` is threaded per env (flat
    `FloatArray`, `N·H`) across the rollout and zeroed at episode boundaries (truncated BPTT, window =
    horizon). Per update: drive the runtime plugin (`envReset`/`envStep`) for `T` steps, each step ONE
    batched forward (`lstmFwdStepBatchBlasFFI` — BLAS `cblas_dgemm` gate/head matmuls, tolerance-close
    to `lstmCellF`; verified against the bit-exact scalar twin `lstmFwdStepBatchFFI` by
    `verify-lstm-fwd-blas`) over all `N` rows feeding ONE batched native-C categorical sampler
    (`sampleActionsBatchFFI`, bit-exact vs `softmax`+`sampleCat`). Then per-segment GAE
    (`computeGAEBoot`, batch-normalized advantages), transposed once into the time-major SoA layout
    (`arr[t·N+n]`), then `epochs` passes of a BATCH-SUMMED BPTT PPO step: ONE call to
    `lstmPPOGradBatchBlasFFI` (BLAS GEMMs, the gate/head matmuls batched over all `N` sequences at each
    timestep, verified against `Σₙ lstmPPOGradSeqFFI` by `verify-lstm-blas`) sums the gradient over the
    WHOLE rollout, then one mean-scaled (÷`N·T`), grad-norm-clipped ascent step — same clip/scale FORM
    as `applyRecGrad`, but no longer bit-identical to it: this trades the original
    N-sequential-per-sequence-ascent-steps/epoch (each seeing the just-updated policy) for one
    batch-gradient-descent step/epoch (standard minibatch PPO, matching how every other trainer in this
    file already updates), a genuine training-algorithm change made deliberately for the speed win —
    confirmed via a learning-curve health check, not a byte-diff. (`updateRecSeqFFI`/`recPPOGradSeq`,
    and the scalar `lstmFwdStepBatchFFI`, remain for the exact original semantics if ever needed again.)
    Single-discrete only (LSTM head is one categorical); the plugin steps `N` copies in native C.
    Both BLAS steps have an opt-in **f32 tier** (`PUFFER_LSTM_F32=1`: `cblas_sgemm` throughout instead
    of `cblas_dgemm`, verified against the f64-BLAS kernels by `verify-lstm-fwd-f32`/
    `verify-lstm-grad-f32`, ~1e-6 relative; a 3-seed learning-curve check showed IDENTICAL traces to
    the f64-BLAS default) — default stays f64-BLAS, matching this project's convention of landing a
    new precision tier opt-in first. Net effect of all five swaps landed vs the original all-Lean
    trainer: ~16 SPS → ~42K SPS (f64-BLAS default) / ~72K SPS (f32 tier) at a 256-env/H64/T64 config
    (~2600× / ~4500×). -/
def trainPluginEnvRec (name config : String)
    (hidden numEnvs horizon totalTimesteps epochs : Nat)
    (lr gamma lam vfCoef entCoef clipEps maxGradNorm minLrRatio : Float)
    (logDash : Bool := false) (seed : UInt64) : IO Unit := do
  let u := USize.ofNat; let mk := FloatArray.mk
  -- BPTT's per-timestep GEMMs are small and strictly sequential (can't parallelize across time); the
  -- default all-cores OpenBLAS threading measured WORSE than a small fixed count here (see
  -- `lean_ffi_blas_set_threads`'s doc comment for the numbers).
  Puffer.Float.BLAS.blasSetThreadsFFI (u 8)
  -- Opt-in f32 tier (PUFFER_LSTM_F32=1): cblas_sgemm instead of cblas_dgemm throughout, verified
  -- against the f64-BLAS kernels by `verify-lstm-fwd-f32`/`verify-lstm-grad-f32` (~1e-6 relative).
  -- Default OFF, matching this project's convention of landing a new precision tier opt-in first.
  let useF32 := (← IO.getEnv "PUFFER_LSTM_F32").getD "0" != "0"
  let fwdStep := if useF32 then Puffer.Float.BLAS.lstmFwdStepBatchBlasF32FFI else Puffer.Float.BLAS.lstmFwdStepBatchBlasFFI
  let gradStep := if useF32 then Puffer.Float.BLAS.lstmPPOGradBatchBlasF32FFI else Puffer.Float.BLAS.lstmPPOGradBatchBlasFFI
  let h ← Puffer.Plugin.envOpen name (u numEnvs) seed config
  if h == 0 then IO.println s!"puffer train: env '{name}' not found — run ocean/build.sh {name}"; return
  let D := (Puffer.Plugin.envObsDim h).toNat
  let A := (Puffer.Plugin.envNumActions h).toNat
  let nAgents := (Puffer.Plugin.envNumAgents h).toNat
  let N := numEnvs * nAgents
  let O := A + 1; let H := hidden; let T := horizon
  let updates := max 1 (totalTimesteps / (N * T))
  let (p0, _) := initRec D H O seed
  let mut p := p0
  let P := (flattenRec p0).size                     -- param count for the dashboard Summary
  let mut rng := seed
  let mut hsF : FloatArray := mk (Array.replicate (N*H) 0.0)
  let mut csF : FloatArray := mk (Array.replicate (N*H) 0.0)
  IO.println s!"puffer train [{name}] RECURRENT (LSTM h={H}) — {numEnvs} envs × {nAgents} agents × {T} (batch {N}), {D}→LSTM{H}→{O}, BPTT PPO, {updates} updates"
  (← IO.getStdout).flush
  let mut obs ← Puffer.Plugin.envReset h            -- N·D
  let mut updNs : Nat := 0
  let mut lastLog : Array (String × Float) := #[]   -- env's own PufferLib `Log`, latest non-empty window
  let mut rollNs : Nat := 0; let mut bpttNs : Nat := 0; let mut tmNs : Nat := 0
  -- `--log` live dashboard state (opt-in; the ad-hoc line below stays byte-identical without it). The 7
  -- losses are surfaced from the batched BPTT grad (lstm_surface_losses in pufferblas.c) ONLY on the last
  -- epoch of a render frame, toggled by cudaMgLossEnableFFI — read-only, cannot perturb the update.
  let tStart ← IO.monoNanosNow
  let mut idx : Nat := 0; let mut lastRenderNs : Nat := 0
  let mut lrRoll : Nat := 0; let mut lrBptt : Nat := 0; let mut lrUpd : Nat := 0
  let G : UInt64 := 0x9E3779B97F4A7C15
  for upd in [0:updates] do
    let ub0 ← IO.monoNanosNow
    let willRender := logDash && (ub0 ≥ lastRenderNs + 600000000 || upd + 1 == updates)
    -- COSINE anneal, matching PufferLib (`cosineLr`); was a LINEAR ramp, the last of the three trainers
    -- still on the old shape (every non-recurrent path already used `cosineLr`).
    let lrNow := cosineLr lr minLrRatio upd updates
    let initHsF := hsF; let initCsF := csF
    let params := flattenRec p
    let mut trajs : Array (Array Transition) := Array.replicate N #[]
    let r0 ← IO.monoNanosNow
    for _s in [0:T] do
      -- batched native forward (replaces the per-env `lstmCellF` Lean glue) + batched native sampler
      let fwdOut := fwdStep params obs hsF csF (u N) (u D) (u H) (u A)
      let hN := Puffer.Float.FFI.sliceFFI fwdOut (u 0) (u (N*H))
      let cN := Puffer.Float.FFI.sliceFFI fwdOut (u (N*H)) (u (N*H))
      let Yb := Puffer.Float.FFI.sliceFFI fwdOut (u (2*N*H)) (u (N*O))
      let avl := Puffer.Float.FFI.sampleActionsBatchFFI Yb (u N) (u A) (u O) rng
      rng := rng + (UInt64.ofNat N) * G
      hsF := hN; csF := cN
      let mut actArr : Array Float := Array.replicate N 0.0
      for n in [0:N] do
        let xn := (Array.range D).map (fun j => obs[n*D + j]!)
        let a := (avl[n]!).toUInt64.toNat
        trajs := trajs.set! n ((trajs[n]!).push
          { obs := xn, action := a, reward := 0.0, value := avl[2*N+n]!, oldLogp := avl[N+n]!, terminal := false })
        actArr := actArr.set! n (Float.ofNat a)
      let out ← Puffer.Plugin.envStep h (mk actArr)
      for n in [0:N] do
        let r := out[N*D + n]!
        let term := out[N*D + N + n]! > 0.5
        let tr := trajs[n]!; let li := tr.size - 1
        trajs := trajs.set! n (tr.set! li { (tr[li]!) with reward := r, terminal := term })
        if term then
          for j in [0:H] do hsF := hsF.set! (n*H+j) 0.0; csF := csF.set! (n*H+j) 0.0
      obs := Puffer.Float.FFI.sliceFFI out (u 0) (u (N*D))
    -- bootstrap value per env from the current obs + carried hidden state (same batched forward)
    let fwdOutB := fwdStep params obs hsF csF (u N) (u D) (u H) (u A)
    let YbB := Puffer.Float.FFI.sliceFFI fwdOutB (u (2*N*H)) (u (N*O))
    let bootVals := (Array.range N).map (fun n => YbB[n*O + A]!)
    let segAdvRet := (Array.range N).map (fun n => computeGAEBoot trajs[n]! bootVals[n]! gamma lam)
    let allAdv := segAdvRet.foldl (fun acc ar => acc ++ ar.1) (#[] : Array Float)
    let nAdv := Float.ofNat (max allAdv.size 1)
    let mean := allAdv.foldl (·+·) 0.0 / nAdv
    let var := allAdv.foldl (fun s x => s + (x-mean)*(x-mean)) 0.0 / nAdv
    let std := Float.sqrt var
    let tm0 ← IO.monoNanosNow
    -- transpose the env-major rollout into the time-major SoA layout `lstmPPOGradBatchBlasFFI` wants
    -- (`arr[t*N+n]`, obs `arr[(t*N+n)*D+d]`) -- ONCE per update, reused by every epoch below (only `p`
    -- changes per epoch, not the collected rollout data).
    let mut obsTM : Array Float := Array.replicate (T*N*D) 0.0
    let mut actTM : Array Float := Array.replicate (T*N) 0.0
    let mut advTM : Array Float := Array.replicate (T*N) 0.0
    let mut retTM : Array Float := Array.replicate (T*N) 0.0
    let mut oldTM : Array Float := Array.replicate (T*N) 0.0
    let mut termTM : Array Float := Array.replicate (T*N) 0.0
    for n in [0:N] do
      let traj := trajs[n]!
      let (adv, ret) := segAdvRet[n]!
      for t in [0:T] do
        let tr := traj[t]!
        let tmIdx := t*N + n
        for d in [0:D] do obsTM := obsTM.set! (tmIdx*D+d) (tr.obs[d]!)
        actTM := actTM.set! tmIdx (Float.ofNat tr.action)
        advTM := advTM.set! tmIdx ((adv[t]! - mean) / (std + 1.0e-8))
        retTM := retTM.set! tmIdx (ret[t]!)
        oldTM := oldTM.set! tmIdx tr.oldLogp
        termTM := termTM.set! tmIdx (if tr.terminal then 1.0 else 0.0)
    let obsB := mk obsTM; let actB := mk actTM; let advB := mk advTM
    let retB := mk retTM; let oldB := mk oldTM; let termB := mk termTM
    let r1 ← IO.monoNanosNow; rollNs := rollNs + (r1 - r0); tmNs := tmNs + (r1 - tm0)
    -- one batch-summed BPTT gradient (all N sequences, multi-threaded BLAS GEMMs) per epoch, then one
    -- mean-scaled grad-norm-clipped ascent step -- same clip/scale FORM as `applyRecGrad` (now meaned
    -- over the whole N·T batch, not per-sequence), but no longer bit-identical to the old N-sequential-
    -- per-sequence-update reference: this is a genuine training-algorithm change (batch gradient descent
    -- vs online per-example ascent), verified via `verify-lstm-blas` (kernel itself, tolerance not
    -- bit-exact) plus a learning-curve health check, not a byte-diff against the prior commit.
    for ep in [0:epochs] do
      -- Enable the loss readback ONLY on the last epoch of a render frame; the BLAS BPTT grad then reduces
      -- the 7 losses into the shared g_mgLoss channel (read-only), read by cudaMgReadLossesFFI below.
      if logDash then Puffer.Float.CUDA.cudaMgLossEnableFFI (if willRender && ep + 1 == epochs then 1 else 0)
      let params := flattenRec p
      let g := gradStep params obsB actB advB retB oldB termB
                 initHsF initCsF (u N) (u T) (u H) (u D) (u A) vfCoef entCoef clipEps
      let mut sq := 0.0
      for i in [0:g.size] do sq := sq + g[i]! * g[i]!
      let mb := Float.ofNat (N * T)
      let meanNorm := Float.sqrt sq / mb
      let cc := if maxGradNorm > 0.0 && meanNorm > maxGradNorm then maxGradNorm / meanNorm else 1.0
      p := applyFlatGradRec p g (lrNow * cc / mb)
    let b1 ← IO.monoNanosNow; bpttNs := bpttNs + (b1 - r1)
    let ub1 ← IO.monoNanosNow; updNs := updNs + (ub1 - ub0)
    let lg ← Puffer.Plugin.envLogPairs h        -- env's own PufferLib `Log` (read + zero); metrics only
    if !lg.isEmpty then lastLog := lg
    -- Live dashboard (PufferLib's rich monitor): redraw in place on the render frame decided at update start.
    if willRender then
      let now ← IO.monoNanosNow
      let lossArr ← Puffer.Float.CUDA.cudaMgReadLossesFFI
      let du := max 1 (upd + 1 - lrUpd)
      let per := fun (cur last : Nat) => Float.ofNat ((cur - last) / du) / 1.0e9
      Puffer.RL.Dashboard.redrawFrom name P ((upd+1)*N*T) (upd+1) tStart now totalTimesteps
        (per rollNs lrRoll) (per bpttNs lrBptt) 0.0 0.0
        lossArr (lastLog.filter (fun kv => kv.1 != "n")) idx
      lrRoll := rollNs; lrBptt := bpttNs; lrUpd := upd + 1
      idx := (idx + 9) % 10; lastRenderNs := now
    if !logDash && ((upd+1) % 20 == 0 || upd == 0) then
      let mut epRet := 0.0; let mut nEps := 0
      for n in [0:N] do
        let mut run := 0.0
        for tr in trajs[n]! do
          run := run + tr.reward
          if tr.terminal then epRet := epRet + run; nEps := nEps + 1; run := 0.0
      let m := if nEps == 0 then 0.0 else epRet / Float.ofNat nEps
      IO.println s!"  update {upd+1}: {nEps} eps, mean ep return = {m}{fmtEnvLog lastLog}"
      (← IO.getStdout).flush
  Puffer.Plugin.envClose h
  let envSteps := updates * N * T
  let sps := if updNs == 0 then 0 else envSteps * 1000000000 / updNs
  IO.println s!"done. perf: {envSteps} env-steps ⇒ {sps} SPS   [rollout {rollNs/1000000}ms (of which SoA-transpose {tmNs/1000000}ms), BPTT {bpttNs/1000000}ms]"

/-! ### Gaussian (continuous) head — native gradient trainer -/

/-- Pack a continuous minibatch into the C kernel's flat arrays (`actsB` is `N·A`). -/
def mkBatchArraysCont (buf : Array ContTransition) (advN returns : Array Float) (idxs : Array Nat) :
    FloatArray × FloatArray × FloatArray × FloatArray × FloatArray := Id.run do
  let mut obsB : FloatArray := FloatArray.emptyWithCapacity 0
  let mut actsB : FloatArray := FloatArray.emptyWithCapacity 0
  let mut advs : FloatArray := FloatArray.emptyWithCapacity 0
  let mut rets : FloatArray := FloatArray.emptyWithCapacity 0
  let mut olps : FloatArray := FloatArray.emptyWithCapacity 0
  for t in idxs do
    let tr : ContTransition := buf[t]!
    for x in tr.obs do obsB := obsB.push x
    for x in tr.action do actsB := actsB.push x
    advs := advs.push advN[t]!
    rets := rets.push returns[t]!
    olps := olps.push tr.oldLogp
  return (obsB, actsB, advs, rets, olps)

def updatePPOIdxContFFI (p : MLP) (buf : Array ContTransition) (advN returns : Array Float)
    (idxs : Array Nat) (actionDim : Nat) (lr maxGradNorm vfCoef entCoef clipEps : Float) : MLP × Float := Id.run do
  let params := flattenMLP p
  let H := p.b1.size
  let D := (p.W1[0]!).size
  let (obsB, actsB, advs, rets, olps) := mkBatchArraysCont buf advN returns idxs
  let g := Puffer.Float.FFI.gaussPPOGradBatchFFI params obsB actsB advs rets olps
             (USize.ofNat idxs.size) (USize.ofNat H) (USize.ofNat D) (USize.ofNat actionDim) vfCoef entCoef clipEps
  let mut sq := 0.0
  for i in [0:g.size] do sq := sq + g[i]! * g[i]!
  let mb := Float.ofNat (max idxs.size 1)
  let meanNorm := Float.sqrt sq / mb
  let cc := if maxGradNorm > 0.0 && meanNorm > maxGradNorm then maxGradNorm / meanNorm else 1.0
  return (applyFlatGrad p g (lr * cc / mb), meanNorm)

/-- Muon variant of `updatePPOIdxContFFI` — Gaussian-head PPO gradient (native C kernel),
    then the same `applyMuon` step as the discrete MLP path (Nesterov → Newton–Schulz → wd),
    with persistent momentum `st`. PufferLib's optimizer on the continuous policy. -/
def updatePPOIdxContFFIMuon (p : MLP) (st : MuonState) (buf : Array ContTransition) (advN returns : Array Float)
    (idxs : Array Nat) (actionDim : Nat) (lr maxGradNorm vfCoef entCoef clipEps : Float) : MLP × MuonState × Float := Id.run do
  let params := flattenMLP p
  let H := p.b1.size
  let D := (p.W1[0]!).size
  let O := p.b2.size
  let (obsB, actsB, advs, rets, olps) := mkBatchArraysCont buf advN returns idxs
  let g := Puffer.Float.FFI.gaussPPOGradBatchFFI params obsB actsB advs rets olps
             (USize.ofNat idxs.size) (USize.ofNat H) (USize.ofNat D) (USize.ofNat actionDim) vfCoef entCoef clipEps
  let mut sq := 0.0
  for i in [0:g.size] do sq := sq + g[i]! * g[i]!
  let mb := Float.ofNat (max idxs.size 1)
  let meanNorm := Float.sqrt sq / mb
  let cc := if maxGradNorm > 0.0 && meanNorm > maxGradNorm then maxGradNorm / meanNorm else 1.0
  let structured := unflattenMLPGrad g (cc / mb) H D O
  let (p', st') := applyMuon p st structured lr 0.0 0.95 1.0e-7
  return (p', st', meanNorm)

/-- Gaussian Muon step from PRE-BUILT flat arrays with value-loss clipping (the continuous
    analogue of `muonStepFromArrays`; `actsB` is `n·actionDim` continuous actions). -/
def muonStepFromArraysCont (p : MLP) (st : MuonState) (obsB actsB advs rets olps oldvals : FloatArray)
    (n actionDim : Nat) (lr maxGradNorm vfCoef entCoef clipEps vfClip mu : Float) :
    MLP × MuonState × Float × FloatArray × FloatArray := Id.run do
  let params := flattenMLP p
  let H := p.b1.size
  let D := (p.W1[0]!).size
  let O := p.b2.size
  let P := params.size
  let g := Puffer.Float.FFI.gaussPPOGradBatchVclipFFI params obsB actsB advs rets olps oldvals
             (USize.ofNat n) (USize.ofNat H) (USize.ofNat D) (USize.ofNat actionDim) vfCoef entCoef clipEps vfClip
  let mut sq := 0.0
  for i in [0:P] do sq := sq + g[i]! * g[i]!
  let mb := Float.ofNat (max n 1)
  let meanNorm := Float.sqrt sq / mb
  let cc := if maxGradNorm > 0.0 && meanNorm > maxGradNorm then maxGradNorm / meanNorm else 1.0
  let structured := unflattenMLPGrad g (cc / mb) H D O
  let (p', st') := applyMuon p st structured lr 0.0 mu 1.0e-7
  let (lp, vv) := splitLV g P n
  return (p', st', meanNorm, lp, vv)

/-- Flatten a `CnnPolicy` to the params layout the CNN C kernel expects. -/
def flattenCnn (p : CnnPolicy) : FloatArray := Id.run do
  let mut a : FloatArray := FloatArray.emptyWithCapacity 0
  for row in p.convW do for x in row do a := a.push x
  for x in p.convB do a := a.push x
  for row in p.W1 do for x in row do a := a.push x
  for x in p.b1 do a := a.push x
  for row in p.W2 do for x in row do a := a.push x
  for x in p.b2 do a := a.push x
  return a

/-- Apply a flat CNN gradient (C-kernel layout) as `p := p + scale·grad`, reconstructing
    the conv + dense tensors in order (convW, convB, W1, b1, W2, b2). -/
def applyFlatGradCnn (p : CnnPolicy) (g : FloatArray) (scale : Float) : CnnPolicy := Id.run do
  let mut off := 0
  let mut convW := p.convW
  for f in [0:convW.size] do
    let mut row := convW[f]!
    let n := row.size
    for i in [0:n] do row := row.set! i (row[i]! + scale * g[off + i]!)
    off := off + n; convW := convW.set! f row
  let mut convB := p.convB
  for i in [0:convB.size] do convB := convB.set! i (convB[i]! + scale * g[off + i]!)
  off := off + convB.size
  let mut W1 := p.W1
  for j in [0:W1.size] do
    let mut row := W1[j]!
    let n := row.size
    for i in [0:n] do row := row.set! i (row[i]! + scale * g[off + i]!)
    off := off + n; W1 := W1.set! j row
  let mut b1 := p.b1
  for i in [0:b1.size] do b1 := b1.set! i (b1[i]! + scale * g[off + i]!)
  off := off + b1.size
  let mut W2 := p.W2
  for k in [0:W2.size] do
    let mut row := W2[k]!
    let n := row.size
    for i in [0:n] do row := row.set! i (row[i]! + scale * g[off + i]!)
    off := off + n; W2 := W2.set! k row
  let mut b2 := p.b2
  for i in [0:b2.size] do b2 := b2.set! i (b2[i]! + scale * g[off + i]!)
  return { p with convW := convW, convB := convB, W1 := W1, b1 := b1, W2 := W2, b2 := b2 }

/-- One minibatch ascent step for the CNN head via the native C gradient kernel (same
    mean-scale + global grad-norm clip as `updateCnnIdx`). -/
def updateCnnIdxFFI (p : CnnPolicy) (buf : Array Transition) (advN returns : Array Float)
    (idxs : Array Nat) (numActions : Nat) (lr maxGradNorm vfCoef entCoef clipEps : Float) : CnnPolicy × Float := Id.run do
  let params := flattenCnn p
  let (obsB, acts, advs, rets, olps) := mkBatchArrays buf advN returns idxs
  let g := Puffer.Float.FFI.cnnPPOGradBatchFFI params obsB acts advs rets olps
             (USize.ofNat idxs.size) (USize.ofNat p.chans) (USize.ofNat p.inH) (USize.ofNat p.inW)
             (USize.ofNat p.nFilters) (USize.ofNat p.kSize) (USize.ofNat p.stride)
             (USize.ofNat p.b1.size) (USize.ofNat numActions) vfCoef entCoef clipEps
  let mut sq := 0.0
  for i in [0:g.size] do sq := sq + g[i]! * g[i]!
  let mb := Float.ofNat (max idxs.size 1)
  let meanNorm := Float.sqrt sq / mb
  let cc := if maxGradNorm > 0.0 && meanNorm > maxGradNorm then maxGradNorm / meanNorm else 1.0
  return (applyFlatGradCnn p g (lr * cc / mb), meanNorm)

/-- Muon momentum buffers for a `CnnPolicy` (one per parameter tensor). -/
structure MuonStateCnn where
  mConvW : Array (Array Float)
  mConvB : Array Float
  mW1 : Array (Array Float)
  mb1 : Array Float
  mW2 : Array (Array Float)
  mb2 : Array Float

def MuonStateCnn.zeros (p : CnnPolicy) : MuonStateCnn :=
  { mConvW := p.convW.map (·.map (fun _ => 0.0)), mConvB := p.convB.map (fun _ => 0.0),
    mW1 := p.W1.map (·.map (fun _ => 0.0)), mb1 := p.b1.map (fun _ => 0.0),
    mW2 := p.W2.map (·.map (fun _ => 0.0)), mb2 := p.b2.map (fun _ => 0.0) }

/-- One Muon step for every `CnnPolicy` tensor: the three 2D weight matrices (`convW`, `W1`,
    `W2`) are orthogonalized (`Muon.stepMat`, as PufferLib does for any `ndim ≥ 2` param — the
    `convW` matrix is `nFilters × chans·k·k`, exactly PufferLib's `grad.view(out, -1)`), the
    three biases take the momentum-only step (`Muon.stepVec`). Returns `(policy', state')`. -/
def applyMuonCnn (p : CnnPolicy) (st : MuonStateCnn)
    (g : Array (Array Float) × Array Float × Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) : CnnPolicy × MuonStateCnn :=
  let (gConvW, gConvB, gW1, gb1, gW2, gb2) := g
  let (nConvW, mConvW) := Muon.stepMat p.convW gConvW st.mConvW lr wd mu eps
  let (nConvB, mConvB) := Muon.stepVec p.convB gConvB st.mConvB lr wd mu
  let (nW1, mW1) := Muon.stepMat p.W1 gW1 st.mW1 lr wd mu eps
  let (nb1, mb1) := Muon.stepVec p.b1 gb1 st.mb1 lr wd mu
  let (nW2, mW2) := Muon.stepMat p.W2 gW2 st.mW2 lr wd mu eps
  let (nb2, mb2) := Muon.stepVec p.b2 gb2 st.mb2 lr wd mu
  ({ p with convW := nConvW, convB := nConvB, W1 := nW1, b1 := nb1, W2 := nW2, b2 := nb2 },
   { mConvW := mConvW, mConvB := mConvB, mW1 := mW1, mb1 := mb1, mW2 := mW2, mb2 := mb2 })

/-- Unflatten the CNN C-kernel's summed flat gradient (the `flattenCnn`/`applyFlatGradCnn`
    layout: `convW, convB, W1, b1, W2, b2`) into structured tensors, scaling by `s`. -/
def unflattenCnnGrad (p : CnnPolicy) (g : FloatArray) (s : Float) :
    Array (Array Float) × Array Float × Array (Array Float) × Array Float × Array (Array Float) × Array Float := Id.run do
  let mut off := 0
  let mut gConvW : Array (Array Float) := #[]
  for f in [0:p.convW.size] do
    let n := p.convW[f]!.size
    let mut row : Array Float := #[]
    for i in [0:n] do row := row.push (s * g[off + i]!)
    off := off + n; gConvW := gConvW.push row
  let gConvB := (Array.range p.convB.size).map (fun i => s * g[off + i]!)
  off := off + p.convB.size
  let mut gW1 : Array (Array Float) := #[]
  for j in [0:p.W1.size] do
    let n := p.W1[j]!.size
    let mut row : Array Float := #[]
    for i in [0:n] do row := row.push (s * g[off + i]!)
    off := off + n; gW1 := gW1.push row
  let gb1 := (Array.range p.b1.size).map (fun i => s * g[off + i]!)
  off := off + p.b1.size
  let mut gW2 : Array (Array Float) := #[]
  for k in [0:p.W2.size] do
    let n := p.W2[k]!.size
    let mut row : Array Float := #[]
    for i in [0:n] do row := row.push (s * g[off + i]!)
    off := off + n; gW2 := gW2.push row
  let gb2 := (Array.range p.b2.size).map (fun i => s * g[off + i]!)
  return (gConvW, gConvB, gW1, gb1, gW2, gb2)

/-- Muon variant of `updateCnnIdxFFI` — same conv+dense PPO gradient (native C kernel), then
    `applyMuonCnn` (per-matrix Newton–Schulz orthogonalization) with persistent momentum `st`,
    and `clip_grad_norm_` on the mean gradient. Returns `(policy', state', gradNorm)`. -/
def updateCnnIdxFFIMuon (p : CnnPolicy) (st : MuonStateCnn) (buf : Array Transition) (advN returns : Array Float)
    (idxs : Array Nat) (numActions : Nat) (lr maxGradNorm vfCoef entCoef clipEps : Float) : CnnPolicy × MuonStateCnn × Float := Id.run do
  let params := flattenCnn p
  let (obsB, acts, advs, rets, olps) := mkBatchArrays buf advN returns idxs
  let g := Puffer.Float.FFI.cnnPPOGradBatchFFI params obsB acts advs rets olps
             (USize.ofNat idxs.size) (USize.ofNat p.chans) (USize.ofNat p.inH) (USize.ofNat p.inW)
             (USize.ofNat p.nFilters) (USize.ofNat p.kSize) (USize.ofNat p.stride)
             (USize.ofNat p.b1.size) (USize.ofNat numActions) vfCoef entCoef clipEps
  let mut sq := 0.0
  for i in [0:g.size] do sq := sq + g[i]! * g[i]!
  let mb := Float.ofNat (max idxs.size 1)
  let meanNorm := Float.sqrt sq / mb
  let cc := if maxGradNorm > 0.0 && meanNorm > maxGradNorm then maxGradNorm / meanNorm else 1.0
  let structured := unflattenCnnGrad p g (cc / mb)
  let (p', st') := applyMuonCnn p st structured lr 0.0 0.95 1.0e-7
  return (p', st', meanNorm)

/-- CNN Muon step from PRE-BUILT flat arrays with value-loss clipping (the CNN analogue of
    `muonStepFromArrays`; uses the conv value-clip kernel + `applyMuonCnn`). -/
def muonStepFromArraysCnn (p : CnnPolicy) (st : MuonStateCnn) (obsB acts advs rets olps oldvals : FloatArray)
    (n numActions : Nat) (lr maxGradNorm vfCoef entCoef clipEps vfClip mu : Float) :
    CnnPolicy × MuonStateCnn × Float × FloatArray × FloatArray := Id.run do
  let params := flattenCnn p
  let P := params.size
  let g := Puffer.Float.FFI.cnnPPOGradBatchVclipFFI params obsB acts advs rets olps oldvals
             (USize.ofNat n) (USize.ofNat p.chans) (USize.ofNat p.inH) (USize.ofNat p.inW)
             (USize.ofNat p.nFilters) (USize.ofNat p.kSize) (USize.ofNat p.stride)
             (USize.ofNat p.b1.size) (USize.ofNat numActions) vfCoef entCoef clipEps vfClip
  let mut sq := 0.0
  for i in [0:P] do sq := sq + g[i]! * g[i]!
  let mb := Float.ofNat (max n 1)
  let meanNorm := Float.sqrt sq / mb
  let cc := if maxGradNorm > 0.0 && meanNorm > maxGradNorm then maxGradNorm / meanNorm else 1.0
  let structured := unflattenCnnGrad p g (cc / mb)
  let (p', st') := applyMuonCnn p st structured lr 0.0 mu 1.0e-7
  let (lp, vv) := splitLV g P n
  return (p', st', meanNorm, lp, vv)

/-- One CNN minibatch ascent step via the BLAS gradient kernel — identical to
    `updateCnnIdxFFI` (same mean-scale + grad-norm clip + `applyFlatGradCnn`), but the
    conv (im2col) + dense forward/backward run through OpenBLAS (`cnnPPOGradBatchBlasFFI`).
    Matches the scalar CNN kernel to tolerance (~1e-11), not bit-exactly.

    SAFETY: the BLAS kernel casts its matrix extents to CBLAS's 32-bit `int`, so the
    im2col row count `R = |minibatch|·oH·oW` must fit in `INT_MAX`. This holds for every
    realistic config (small grids ⇒ `R` in the millions), but if a huge minibatch × large
    image would overflow, fall back to the size_t scalar kernel (`updateCnnIdxFFI`) so the
    result stays correct — the BLAS path is a true drop-in with no correctness cliff. -/
def updateCnnIdxBlasFFI (p : CnnPolicy) (buf : Array Transition) (advN returns : Array Float)
    (idxs : Array Nat) (numActions : Nat) (lr maxGradNorm vfCoef entCoef clipEps : Float) : CnnPolicy := Id.run do
  let rRows := idxs.size * cnnOutH p * cnnOutW p
  if rRows ≥ 2147483647 then
    return (updateCnnIdxFFI p buf advN returns idxs numActions lr maxGradNorm vfCoef entCoef clipEps).1
  let params := flattenCnn p
  let (obsB, acts, advs, rets, olps) := mkBatchArrays buf advN returns idxs
  let g := Puffer.Float.BLAS.cnnPPOGradBatchBlasFFI params obsB acts advs rets olps
             (USize.ofNat idxs.size) (USize.ofNat p.chans) (USize.ofNat p.inH) (USize.ofNat p.inW)
             (USize.ofNat p.nFilters) (USize.ofNat p.kSize) (USize.ofNat p.stride)
             (USize.ofNat p.b1.size) (USize.ofNat numActions) vfCoef entCoef clipEps (USize.ofNat p.nScalar)
  let mut sq := 0.0
  for i in [0:g.size] do sq := sq + g[i]! * g[i]!
  let mb := Float.ofNat (max idxs.size 1)
  let meanNorm := Float.sqrt sq / mb
  let cc := if maxGradNorm > 0.0 && meanNorm > maxGradNorm then maxGradNorm / meanNorm else 1.0
  return applyFlatGradCnn p g (lr * cc / mb)

/-- Muon momentum buffers for a `RecPolicy` (one per parameter tensor). -/
structure MuonStateRec where
  mWx : Array (Array Float)
  mWh : Array (Array Float)
  mbih : Array Float
  mWo : Array (Array Float)
  mbo : Array Float

def MuonStateRec.zeros (p : RecPolicy) : MuonStateRec :=
  { mWx := p.Wx.map (·.map (fun _ => 0.0)), mWh := p.Wh.map (·.map (fun _ => 0.0)),
    mbih := p.bih.map (fun _ => 0.0), mWo := p.Wo.map (·.map (fun _ => 0.0)), mbo := p.bo.map (fun _ => 0.0) }

/-- One Muon step for every `RecPolicy` tensor: the three 2D weight matrices (`Wx` `4H×D`,
    `Wh` `4H×H`, `Wo` `dout×H`) are orthogonalized as WHOLE matrices (`Muon.stepMat`, as
    PufferLib does for any `ndim ≥ 2` param — it is blind to the stacked-gate structure), the
    two biases take the momentum-only step (`Muon.stepVec`). Returns `(policy', state')`. -/
def applyMuonRec (p : RecPolicy) (st : MuonStateRec)
    (g : Array (Array Float) × Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) : RecPolicy × MuonStateRec :=
  let (gWx, gWh, gbih, gWo, gbo) := g
  let (nWx, mWx) := Muon.stepMat p.Wx gWx st.mWx lr wd mu eps
  let (nWh, mWh) := Muon.stepMat p.Wh gWh st.mWh lr wd mu eps
  let (nbih, mbih) := Muon.stepVec p.bih gbih st.mbih lr wd mu
  let (nWo, mWo) := Muon.stepMat p.Wo gWo st.mWo lr wd mu eps
  let (nbo, mbo) := Muon.stepVec p.bo gbo st.mbo lr wd mu
  ({ p with Wx := nWx, Wh := nWh, bih := nbih, Wo := nWo, bo := nbo },
   { mWx := mWx, mWh := mWh, mbih := mbih, mWo := mWo, mbo := mbo })

/-- Unflatten the LSTM C-kernel's summed flat gradient (the `flattenRec`/`applyFlatGradRec`
    layout: `Wx, Wh, bih, Wo, bo`) into structured tensors, scaling by `s`. -/
def unflattenRecGrad (p : RecPolicy) (g : FloatArray) (s : Float) :
    Array (Array Float) × Array (Array Float) × Array Float × Array (Array Float) × Array Float := Id.run do
  let mut off := 0
  let mut gWx : Array (Array Float) := #[]
  for k in [0:p.Wx.size] do
    let n := p.Wx[k]!.size
    let mut row : Array Float := #[]
    for i in [0:n] do row := row.push (s * g[off + i]!)
    off := off + n; gWx := gWx.push row
  let mut gWh : Array (Array Float) := #[]
  for k in [0:p.Wh.size] do
    let n := p.Wh[k]!.size
    let mut row : Array Float := #[]
    for i in [0:n] do row := row.push (s * g[off + i]!)
    off := off + n; gWh := gWh.push row
  let gbih := (Array.range p.bih.size).map (fun i => s * g[off + i]!)
  off := off + p.bih.size
  let mut gWo : Array (Array Float) := #[]
  for m in [0:p.Wo.size] do
    let n := p.Wo[m]!.size
    let mut row : Array Float := #[]
    for i in [0:n] do row := row.push (s * g[off + i]!)
    off := off + n; gWo := gWo.push row
  let gbo := (Array.range p.bo.size).map (fun i => s * g[off + i]!)
  return (gWx, gWh, gbih, gWo, gbo)

/-- Muon variant of `updateRecSeqFFI` — same BPTT PPO gradient (native C kernel), then
    `applyMuonRec` (per-matrix Newton–Schulz orthogonalization) with persistent momentum `st`,
    and `clip_grad_norm_` on the mean gradient (over `seqLen`). Returns `(policy', state', gradNorm)`. -/
def updateRecSeqFFIMuon (p : RecPolicy) (st : MuonStateRec) (traj : Array Transition) (h0 c0 advN returns : Array Float)
    (numActions : Nat) (lr maxGradNorm vfCoef entCoef clipEps : Float) : RecPolicy × MuonStateRec × Float := Id.run do
  let params := flattenRec p
  let H := p.hSize
  let D := (p.Wx[0]!).size
  let (obsSeq, acts, olps, terms) := mkSeqArrays traj
  let g := Puffer.Float.FFI.lstmPPOGradSeqFFI params obsSeq acts (FloatArray.mk advN)
             (FloatArray.mk returns) olps terms (FloatArray.mk h0) (FloatArray.mk c0)
             (USize.ofNat traj.size) (USize.ofNat H) (USize.ofNat D) (USize.ofNat numActions) vfCoef entCoef clipEps
  let mut sq := 0.0
  for i in [0:g.size] do sq := sq + g[i]! * g[i]!
  let sl := Float.ofNat (max traj.size 1)
  let meanNorm := Float.sqrt sq / sl
  let cc := if maxGradNorm > 0.0 && meanNorm > maxGradNorm then maxGradNorm / meanNorm else 1.0
  let structured := unflattenRecGrad p g (cc / sl)
  let (p', st') := applyMuonRec p st structured lr 0.0 0.95 1.0e-7
  return (p', st', meanNorm)

/-- Forward the LSTM over one sequence from `(h0,c0)` under the CURRENT policy, returning the
    per-timestep `(new_logp, new_value)` — the recurrent analogue of `policyAndValue`, used to
    iterate the ratio/value buffers in prioritized replay. Resets `(h,c)` to zero after a terminal
    step, matching the BPTT kernel's boundary handling. -/
def recSeqForward (p : RecPolicy) (traj : Array Transition) (h0 c0 : Array Float) (numActions : Nat) :
    Array Float × Array Float := Id.run do
  let mut h := h0; let mut c := c0
  let mut lps : Array Float := #[]
  let mut vs : Array Float := #[]
  for t in [0:traj.size] do
    let tr := traj[t]!
    let (h', c', out) := lstmCellF p tr.obs h c
    let (probs, v) := recProbsValue out numActions
    lps := lps.push (Float.log probs[tr.action]!)
    vs := vs.push v
    if tr.terminal then
      h := zeros p.hSize; c := zeros p.hSize
    else
      h := h'; c := c'
  return (lps, vs)

/-- Roll ONE multi-agent env instance under the native (agent-batched) forward. -/
def multiSegmentRolloutFFI {S : Type} (env : MultiEnv S) (params : FloatArray) (H : Nat)
    (horizon : Nat) (s0 : S) (rng0 : UInt64) :
    Array (Array Transition) × Array Float × S × UInt64 × Array Float := Id.run do
  let N := env.numAgents
  let D := env.obsDim
  let A := env.numActions
  let O := A + 1
  let u := USize.ofNat
  -- batched forward across the N agents ⇒ per-agent (probs, value)
  let fwd := fun (obsAll : Array (Array Float)) => Id.run do
    let mut xb : Array Float := #[]
    for o in obsAll do for x in o do xb := xb.push x
    let Yb := Puffer.Float.BLAS.mlpForwardBatchRefFFI params (FloatArray.mk xb) (u N) (u D) (u H) (u O)
    let mut pvs : Array (Array Float × Float) := #[]
    for a in [0:N] do
      let out := (Array.range O).map (fun k => Yb[a*O + k]!)
      pvs := pvs.push (softmax ((Array.range A).map (fun k => out[k]!)), out[A]!)
    return pvs
  let mut st := s0
  let mut rng := rng0
  let mut streams : Array (Array Transition) := Array.replicate N #[]
  let mut epReturns : Array Float := #[]
  let mut epRet := 0.0
  for _ in [0:horizon] do
    let obsAll := env.observe st
    let pvs := fwd obsAll
    let mut actions : Array Nat := #[]
    let mut vals : Array Float := #[]
    let mut logps : Array Float := #[]
    for a in [0:N] do
      let (probs, v) := pvs[a]!
      let (word, rng') := rngNext rng
      rng := rng'
      let act := sampleCat probs (uniform01 word)
      actions := actions.push act; vals := vals.push v; logps := logps.push (Float.log probs[act]!)
    let (st', rewards, term) := env.step st actions
    for a in [0:N] do
      streams := streams.set! a ((streams[a]!).push
        { obs := obsAll[a]!, action := actions[a]!, reward := rewards[a]!, value := vals[a]!,
          oldLogp := logps[a]!, terminal := term })
    epRet := epRet + rewards.foldl (· + ·) 0.0
    st := st'
    if term then
      epReturns := epReturns.push epRet; epRet := 0.0
      let (sReset, rng'') := env.reset rng; rng := rng''; st := sReset
  let pvsF := fwd (env.observe st)
  let mut bootVals : Array Float := #[]
  for a in [0:N] do bootVals := bootVals.push (pvsF[a]!).2
  return (streams, bootVals, st, rng, epReturns)

/-- Roll `numEnvs` multi-agent instances with the native forward; flatten the streams. -/
def multiVecRolloutFFI {S : Type} (env : MultiEnv S) (p : MLP) (horizon : Nat)
    (states : Array S) (rng0 : UInt64) :
    Array (Array Transition) × Array Float × Array S × UInt64 × Array Float := Id.run do
  let params := flattenMLP p
  let H := p.b1.size
  let mut rng := rng0
  let mut streams : Array (Array Transition) := #[]
  let mut bootVals : Array Float := #[]
  let mut newStates : Array S := #[]
  let mut epReturns : Array Float := #[]
  for st in states do
    let (strs, boots, st', rng', epRets) := multiSegmentRolloutFFI env params H horizon st rng
    rng := rng'
    streams := streams ++ strs; bootVals := bootVals ++ boots
    newStates := newStates.push st'; epReturns := epReturns ++ epRets
  return (streams, bootVals, newStates, rng, epReturns)

end Puffer.RL.NNTrain
