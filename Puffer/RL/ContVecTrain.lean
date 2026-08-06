/-
# Gaussian continuous-action PPO (Tier 2)

The discrete trainer (`VecTrain.lean`) has a softmax head over `numActions`. This
module adds the CONTINUOUS-action path: a diagonal-Gaussian policy for `ContEnv S`
envs (docking, squared_continuous, …). It mirrors the vectorized discrete trainer
(parallel envs → experience buffer → minibatched clipped-surrogate PPO, truncated-GAE
bootstrap, LR annealing, grad clipping) — only the policy HEAD, the sampling, and the
log-prob/entropy differ.

**Policy head.** The MLP output width is `2·actionDim + 1`: `out[0..d)` are the per-dim
MEANS `μ`, `out[d..2d)` the per-dim LOG-STDs (state-dependent, clamped to
`[logstdLo, logstdHi]`), and `out[2d]` the value. `σ = exp(logstd)`.

**Sampling** (Box–Muller): `aᵢ = μᵢ + σᵢ·zᵢ`, `zᵢ ~ N(0,1)`; then
`logp = Σᵢ (−½zᵢ² − logstdᵢ − ½log2π)` (since `(aᵢ−μᵢ)/σᵢ = zᵢ`).

**Objective on the AD tape.** The per-dim log-prob is
`−½·((aᵢ−μᵢ)·exp(−logstdᵢ))² − logstdᵢ − ½log2π` (no `div` needed — the reciprocal is
`exp(−logstd)`), and the per-dim differential entropy is `logstdᵢ + ½(1+log2π)`. The
ratio, clipped surrogate, value loss, and entropy bonus are assembled exactly as in the
discrete `mlpGradPPO`, reusing the same AD primitives (`clampC` for both the log-std
clamp and the ratio clip; `minV` for the pessimistic surrogate). The runtime error
bounds for this head are in `Puffer/RL/GaussianRuntime.lean`.

Mathlib-free (extends `NNTrain`/`VecTrain`); the binary links no Mathlib.
-/
import Puffer.RL.VecTrain

namespace Puffer.RL.NNTrain

open Puffer.RL (Env ContEnv)
open Puffer.FloatR
open Puffer.RL.Train (rngNext uniform01)

/-! ### Gaussian constants + clamp bounds -/

/-- `log(2π)`. -/
def log2piC : Float := Float.log (2.0 * 3.141592653589793)
/-- `½·log(2π)` — the Gaussian log-prob normalizer. -/
def halfLog2piC : Float := 0.5 * log2piC
/-- `½·(1 + log2π)` — the per-dim differential-entropy constant (`H = logstd + this`). -/
def halfLog2pieEC : Float := 0.5 * (1.0 + log2piC)
/-- Log-std clamp bounds (PufferLib-style stability clamp). -/
def logstdLo : Float := -20.0   -- PufferLib's safe_continuous_logstd bound (src/pufferlib.cu:449)
def logstdHi : Float := 2.0

@[inline] def clampF (x lo hi : Float) : Float := max lo (min hi x)

/-- A continuous-action rollout transition. -/
structure ContTransition where
  obs : Array Float
  action : Array Float
  reward : Float
  value : Float
  oldLogp : Float
  terminal : Bool
  deriving Inhabited

/-! ### Continuous policy forward + Gaussian sampling -/

/-- From the MLP output, read `(means, clamped logstds, value)` for a `d`-dim head. -/
def contPolicy (p : MLP) (actionDim : Nat) (obs : Array Float) :
    Array Float × Array Float × Float :=
  let (_, _, out) := forwardAll p obs
  let mean := (Array.range actionDim).map (fun i => out[i]!)
  let logstd := (Array.range actionDim).map (fun i => clampF (out[actionDim + i]!) logstdLo logstdHi)
  (mean, logstd, out[2 * actionDim]!)

/-- Sample `aᵢ = μᵢ + σᵢ·zᵢ` per dim (Box–Muller `z ~ N(0,1)`) and return the sampled
    action, its log-prob `Σ(−½zᵢ² − logstdᵢ − ½log2π)`, and the advanced RNG. -/
def gaussianSample (mean logstd : Array Float) (rng0 : UInt64) :
    Array Float × Float × UInt64 := Id.run do
  let d := mean.size
  let mut rng := rng0
  let mut action : Array Float := #[]
  let mut logp : Float := 0.0
  for i in [0:d] do
    let (w1, r1) := rngNext rng
    let (w2, r2) := rngNext r1
    rng := r2
    let u1 := max (uniform01 w1) 1.0e-7
    let u2 := uniform01 w2
    let z := Float.sqrt (-2.0 * Float.log u1) * Float.cos (2.0 * 3.141592653589793 * u2)
    let ls := logstd[i]!
    let a := mean[i]! + Float.exp ls * z
    action := action.push a
    logp := logp + (-0.5 * z * z - ls - halfLog2piC)
  return (action, logp, rng)

/-! ### Truncated-GAE over continuous transitions (mirror of `computeGAEBoot`) -/

def computeGAEBootC (traj : Array ContTransition) (bootValue gamma lam : Float) :
    Array Float × Array Float := Id.run do
  let n := traj.size
  let mut adv := Array.replicate n 0.0
  let mut lastA := 0.0
  for i in [0:n] do
    let t := n - 1 - i
    let nnt := if traj[t]!.terminal then 0.0 else 1.0
    let vNext := if t + 1 < n then traj[t+1]!.value else bootValue
    let delta := traj[t]!.reward + gamma * vNext * nnt - traj[t]!.value
    lastA := delta + gamma * lam * nnt * lastA
    adv := adv.set! t lastA
  return (adv, (Array.range n).map (fun t => adv[t]! + traj[t]!.value))

/-- `ContTransition` version of `computePuffAdvantage` — PufferLib's `compute_puff_advantage`
    (V-Trace/GAE, `A_{last}=0`, `δ = ρ·(r + γ·V'·nnt − V)`, `A_t = δ + γλ·c·A_{t+1}·nnt`; the GPU
    `puff_advantage_row_vec` form — ρ scales the whole TD error). `imp ≡ 1` reduces it to GAE.
    No bootstrap (PufferLib keeps the last advantage 0). -/
def computePuffAdvantageC (traj : Array ContTransition) (importance : Array Float)
    (gamma lam rhoClip cClip : Float) : Array Float × Array Float := Id.run do
  let n := traj.size
  let mut adv := Array.replicate n 0.0
  let mut lastA := 0.0
  for i in [0:n-1] do
    let t := n - 2 - i
    let nnt := if traj[t]!.terminal then 0.0 else 1.0
    let imp := importance[t]!
    let rho := if imp < rhoClip then imp else rhoClip
    let c := if imp < cClip then imp else cClip
    let rr := traj[t]!.reward
    let r := if rr > 1.0 then 1.0 else if rr < -1.0 then -1.0 else rr   -- PufferLib clamps rewards to [-1,1]
    let delta := rho * (r + gamma * traj[t+1]!.value * nnt - traj[t]!.value)
    lastA := delta + gamma * lam * c * lastA * nnt
    adv := adv.set! t lastA
  return (adv, (Array.range n).map (fun t => adv[t]! + traj[t]!.value))

/-- `computePuffAdvantageC` with EXTERNAL iterated values + live importance (for prioritized
    replay — mirrors `computePuffAdvantageV`). Returns just the advantages. -/
def computePuffAdvantageVC (traj : Array ContTransition) (values importance : Array Float)
    (gamma lam rhoClip cClip : Float) : Array Float := Id.run do
  let n := traj.size
  let mut adv := Array.replicate n 0.0
  let mut lastA := 0.0
  for i in [0:n-1] do
    let t := n - 2 - i
    let nnt := if traj[t]!.terminal then 0.0 else 1.0
    let imp := importance[t]!
    let rho := if imp < rhoClip then imp else rhoClip
    let c := if imp < cClip then imp else cClip
    let rr := traj[t]!.reward
    let r := if rr > 1.0 then 1.0 else if rr < -1.0 then -1.0 else rr
    let delta := rho * (r + gamma * values[t+1]! * nnt - values[t]!)
    lastA := delta + gamma * lam * c * lastA * nnt
    adv := adv.set! t lastA
  return adv

/-- Gaussian log-prob of a GIVEN continuous action + the value, under the current policy —
    the continuous analogue of `policyAndValue` (used to iterate the ratio/value buffers in
    prioritized replay). Reuses `contPolicy` (clamped logstd), matching the C kernel. -/
def gaussLogpValue (p : MLP) (d : Nat) (obs act : Array Float) : Float × Float := Id.run do
  let (mean, logstd, v) := contPolicy p d obs
  let mut logp := 0.0
  for i in [0:d] do
    let ls := logstd[i]!
    let z := (act[i]! - mean[i]!) * Float.exp (-ls)
    logp := logp + (-0.5 * z * z - ls - halfLog2piC)
  return (logp, v)

/-! ### Continuous rollout (mirror of `segmentRollout`/`vecRollout`) -/

def segmentRolloutCont {S : Type} (env : ContEnv S) (p : MLP) (horizon : Nat)
    (s0 : S) (rng0 : UInt64) :
    Array ContTransition × Float × S × UInt64 × Array Float := Id.run do
  let d := env.actionDim
  let mut st := s0
  let mut rng := rng0
  let mut traj : Array ContTransition := #[]
  let mut epReturns : Array Float := #[]
  let mut epRet := 0.0
  for _ in [0:horizon] do
    let obs := env.observe st
    let (mean, logstd, v) := contPolicy p d obs
    let (action, logp, rng') := gaussianSample mean logstd rng
    rng := rng'
    let (st', r, term) := env.step st action
    traj := traj.push { obs := obs, action := action, reward := r, value := v,
                        oldLogp := logp, terminal := term }
    epRet := epRet + r
    if term then
      epReturns := epReturns.push epRet
      epRet := 0.0
      let (sReset, rng'') := env.reset rng
      rng := rng''
      st := sReset
    else
      st := st'
  let (_, _, bootV) := contPolicy p d (env.observe st)
  return (traj, bootV, st, rng, epReturns)

def vecRolloutCont {S : Type} (env : ContEnv S) (p : MLP) (horizon : Nat)
    (states : Array S) (rng0 : UInt64) :
    Array (Array ContTransition) × Array Float × Array S × UInt64 × Array Float := Id.run do
  let mut rng := rng0
  let mut trajs : Array (Array ContTransition) := #[]
  let mut bootVals : Array Float := #[]
  let mut newStates : Array S := #[]
  let mut epReturns : Array Float := #[]
  for st in states do
    let (traj, bootV, st', rng', epRets) := segmentRolloutCont env p horizon st rng
    rng := rng'
    trajs := trajs.push traj
    bootVals := bootVals.push bootV
    newStates := newStates.push st'
    epReturns := epReturns ++ epRets
  return (trajs, bootVals, newStates, rng, epReturns)

def buildBatchC (trajs : Array (Array ContTransition)) (_bootVals : Array Float) (gamma lam : Float) :
    Array ContTransition × Array Float × Array Float := Id.run do
  let mut allTr : Array ContTransition := #[]
  let mut allAdv : Array Float := #[]
  let mut allRet : Array Float := #[]
  for traj in trajs do
    let imp := Array.replicate traj.size 1.0
    let (adv, ret) := computePuffAdvantageC traj imp gamma lam 1.0 1.0
    allTr := allTr ++ traj
    allAdv := allAdv ++ adv
    allRet := allRet ++ ret
  return (allTr, allAdv, allRet)

/-! ### Continuous PPO gradient (Gaussian log-prob + entropy on the AD tape) -/

/-- Gradient of the Gaussian-PPO objective for one transition, via autodiff:
    `min(ρ·A, clip(ρ,1−ε,1+ε)·A) − vf·½(V−R)² + ent·H`, with `ρ = exp(logp − logp_old)`,
    `logp = Σ −½((a−μ)e^{−logstd})² − logstd − ½log2π`, `H = Σ logstd + ½(1+log2π)`. -/
def mlpGradPPOCont (p : MLP) (x acts : Array Float) (d : Nat)
    (adv ret oldLogp vfCoef entCoef clipEps : Float) :
    Array (Array Float) × Array Float × Array (Array Float) × Array Float :=
  let build : AD.ADM (AD.V × (Array (Array AD.V) × Array AD.V × Array (Array AD.V) × Array AD.V)) := do
    let W1v ← p.W1.mapM (fun row => row.mapM AD.leaf)
    let b1v ← p.b1.mapM AD.leaf
    let W2v ← p.W2.mapM (fun row => row.mapM AD.leaf)
    let b2v ← p.b2.mapM AD.leaf
    let xv ← x.mapM AD.const
    let z1 ← (Array.range p.b1.size).mapM (fun j => do let dp ← AD.dotV W1v[j]! xv; AD.add b1v[j]! dp)
    let h ← z1.mapM AD.relu
    let out ← (Array.range p.b2.size).mapM (fun k => do let dp ← AD.dotV W2v[k]! h; AD.add b2v[k]! dp)
    let hLog2pi ← AD.const halfLog2piC
    let hLog2pieE ← AD.const halfLog2pieEC
    let zeroL ← AD.const 0.0
    -- Σ log-prob over the d action dims
    let logp ← (Array.range d).foldlM (fun acc i => do
      let ai ← AD.const acts[i]!
      let lsi ← AD.clampC logstdLo logstdHi out[d + i]!
      let diff ← AD.sub ai out[i]!
      let invStd ← AD.exp (← AD.neg lsi)
      let z ← AD.mul diff invStd
      let z2 ← AD.mul z z
      let li ← AD.sub (← AD.sub (← AD.scale (-0.5) z2) lsi) hLog2pi
      AD.add acc li) zeroL
    -- Σ differential entropy over the d dims
    let ent ← (Array.range d).foldlM (fun acc i => do
      let lsi ← AD.clampC logstdLo logstdHi out[d + i]!
      AD.add acc (← AD.add lsi hLog2pieE)) zeroL
    let ratio ← AD.exp (← AD.sub logp (← AD.const oldLogp))
    let t1 ← AD.scale adv ratio
    let t2 ← AD.scale adv (← AD.clampC (1.0 - clipEps) (1.0 + clipEps) ratio)
    let surr ← AD.minV t1 t2
    let dv ← AD.sub out[2 * d]! (← AD.const ret)
    let vl ← AD.scale (vfCoef * 0.5) (← AD.mul dv dv)
    let obj ← AD.sub (← AD.add surr (← AD.scale entCoef ent)) vl
    return (obj, (W1v, b1v, W2v, b2v))
  let (res, t) := build.run AD.Tape.empty
  let (root, W1v, b1v, W2v, b2v) := res
  let g := AD.grads t root
  (W1v.map (·.map (fun h => g[h]!)), b1v.map (fun h => g[h]!),
   W2v.map (·.map (fun h => g[h]!)), b2v.map (fun h => g[h]!))

def ppoGradIdxCont (p : MLP) (buf : Array ContTransition) (advN returns : Array Float)
    (idxs : Array Nat) (d : Nat) (vfCoef entCoef clipEps : Float) :
    Array (Array Float) × Array Float × Array (Array Float) × Array Float := Id.run do
  let mut gW1 := p.W1.map (·.map (fun _ => 0.0))
  let mut gb1 := p.b1.map (fun _ => 0.0)
  let mut gW2 := p.W2.map (·.map (fun _ => 0.0))
  let mut gb2 := p.b2.map (fun _ => 0.0)
  for t in idxs do
    let tr : ContTransition := buf[t]!
    let (dW1, db1, dW2, db2) :=
      mlpGradPPOCont p tr.obs tr.action d advN[t]! returns[t]! tr.oldLogp vfCoef entCoef clipEps
    gW1 := matAdd gW1 dW1; gb1 := vecAdd gb1 db1
    gW2 := matAdd gW2 dW2; gb2 := vecAdd gb2 db2
  return (gW1, gb1, gW2, gb2)

def updatePPOIdxCont (p : MLP) (buf : Array ContTransition) (advN returns : Array Float)
    (idxs : Array Nat) (d : Nat) (lr maxGradNorm vfCoef entCoef clipEps : Float) : MLP :=
  let (gW1, gb1, gW2, gb2) := ppoGradIdxCont p buf advN returns idxs d vfCoef entCoef clipEps
  let mb := Float.ofNat (max idxs.size 1)
  let meanNorm := Float.sqrt (gradSumSq gW1 gb1 gW2 gb2) / mb
  let clipCoef := if maxGradNorm > 0.0 && meanNorm > maxGradNorm then maxGradNorm / meanNorm else 1.0
  let scale := lr * clipCoef / mb
  { W1 := matAxpy scale p.W1 gW1, b1 := vecAxpy scale p.b1 gb1,
    W2 := matAxpy scale p.W2 gW2, b2 := vecAxpy scale p.b2 gb2 }

end Puffer.RL.NNTrain
