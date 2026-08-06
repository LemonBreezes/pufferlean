/-
# Vectorized PPO trainer (Tier 1: the PufferLib training-loop structure)

The single-env trainer in `NNTrain.lean` runs ONE env instance episode-by-episode
and does full-batch PPO. PufferLib's defining structure is different: many env
instances stepped in parallel for a fixed HORIZON (auto-resetting on terminal),
collected into one experience BUFFER, then several EPOCHS of MINIBATCHED SGD over
the shuffled buffer. This module rebuilds the loop that way, reusing the verified
kernels (`policyAndValue`, `mlpGradPPO`, `computeGAE`, `normalizeAdv`, the axpy
update) unchanged — only the surrounding data flow changes.

Pieces:
  * `computeGAEBoot` — `computeGAE` with a BOOTSTRAP value at the segment boundary,
    so a truncated horizon segment (ending mid-episode) bootstraps from `V(sₕ)`
    instead of assuming a terminal. Identical to `computeGAE` when the segment ends
    on a terminal (the bootstrap is then masked by `nnt = 0`).
  * `segmentRollout` — one env instance for `horizon` steps, auto-resetting on
    terminal; returns the horizon-length segment, the bootstrap value `V(final)`,
    the (persistent) final state, and the returns of episodes that completed.
  * `vecRollout` — `numEnvs` segments (the policy is fixed during collection, so
    stepping envs sequentially is identical to interleaving, up to RNG order).
  * `buildBatch` — GAE per segment (bootstrapped), flattened into one buffer.
  * `shuffleIdx` — Fisher–Yates permutation for minibatch sampling.
  * `ppoGradIdx`/`updatePPOIdx` — minibatch gradient (over an index set, advantages
    pre-normalized at batch level) and its mean-scaled ascent step.

These are the shared vectorized-PPO helpers the plugin trainers (`Exe/Puffer.lean`,
`trainPluginEnv*`) reuse. Mathlib-free (extends `NNTrain`); the binary links no Mathlib.
-/
import Puffer.RL.NNTrain

namespace Puffer.RL.NNTrain

open Puffer.RL (Env)
open Puffer.FloatR
open Puffer.RL.Train (rngNext uniform01 softmax sampleCat)

/-- `computeGAE` with a bootstrap value at the segment end: at the last step, if the
    transition is non-terminal, the next-state value is `bootValue = V(sₕ)` and the
    non-terminal mask is `1` (via the stored `terminal` flag). Reduces to `computeGAE`
    when the segment ends on a terminal (`nnt = 0` masks `bootValue`). -/
def computeGAEBoot (traj : Array Transition) (bootValue gamma lam : Float) :
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

/-- PufferLib's `compute_puff_advantage` — the GPU kernel `puff_advantage_row_vec`
    (`~/src/PufferLib/src/pufferlib.cu`, the path `puffer train` runs on GPU whenever the horizon
    is a multiple of 8) over ONE horizon segment. V-Trace / GAE with an importance-ratio
    correction. `A_{T-1} = 0`, and for `t = T-2 … 0`:
      `ρ_t = min(imp_t, ρ_clip)`,  `c_t = min(imp_t, c_clip)`,  `nnt_t = 1 − done_{t+1}`,
      `δ_t = ρ_t·(r_t + γ·V_{t+1}·nnt_t − V_t)`   (the vec kernel — ρ scales the WHOLE TD error),
      `A_t = δ_t + γλ·c_t·A_{t+1}·nnt_t`.
    Verified bit-for-bit against the ACTUAL compiled `pufferlib._C.puff_advantage` (bf16 CUDA), not
    just the source — see `tools/vtrace_ref.py`. PufferLib's CPU/scalar kernels use `δ = ρ·r + γV'·nnt − V`
    (ρ on the reward only); the two agree exactly iff `ρ = 1` (`GAE.deltaVec_eq_deltaScalar_iff`), which
    holds on the first minibatch before ratios drift. We match the GPU path that actually trains.
    Index mapping to a Lean `Transition` segment: PufferLib's `reward[t+1]`/`value[t+1]`/`done[t+1]`
    are `traj[t].reward`/`traj[t+1].value`/`traj[t].terminal` (PufferLib stores rewards/terminals one
    step ahead). The LAST transition keeps `A = 0` — its value only bootstraps `A_{T-2}`; PufferLib
    collects no separate bootstrap value (so `bootValue` is unused here, unlike `computeGAEBoot`).
    `imp ≡ 1` (⇒ `ρ = c = 1`) reduces this EXACTLY to standard GAE; the live per-minibatch importance
    ratios are threaded once prioritized replay is wired. Returns `(advantages, returns)` with
    `returns[t] = A_t + V_t`. -/
def computePuffAdvantage (traj : Array Transition) (importance : Array Float)
    (gamma lam rhoClip cClip : Float) : Array Float × Array Float := Id.run do
  let n := traj.size
  let mut adv := Array.replicate n 0.0
  let mut lastA := 0.0
  -- backward over t = n-2 … 0; the final transition (t = n-1) keeps A = 0
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

/-- `computePuffAdvantage` with the value estimates supplied EXTERNALLY (`values`) rather than
    read from the transitions — the iterated-value form PufferLib's `train()` uses (`val[idx]` is
    overwritten after each minibatch, so the advantage recompute sees updated values). `importance`
    is the live per-step ratio buffer (`self.ratio`). Returns just the advantages (the caller pairs
    them with `values` for the returns). Rewards/terminals still come from `traj`. -/
def computePuffAdvantageV (traj : Array Transition) (values importance : Array Float)
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
    let r := if rr > 1.0 then 1.0 else if rr < -1.0 then -1.0 else rr   -- PufferLib clamps rewards to [-1,1]
    let delta := rho * (r + gamma * values[t+1]! * nnt - values[t]!)
    lastA := delta + gamma * lam * c * lastA * nnt
    adv := adv.set! t lastA
  return adv

/-- Flat (B·T row-major) V-Trace for the MinGRU trainer — the ORACLE for the GPU kernel
    `cudaVtraceMinGRUFFI`. A line-for-line copy of `trainPluginEnvMinGRU`'s per-segment `vtrace` closure,
    reading flat arrays instead of `Array Transition` segments: the SCALAR delta `ρ·r + γV′·nnt − V`
    (ρ on the reward only, unlike the vec `computePuffAdvantageV` above) with the LAST step bootstrapped
    by `bootv[e]` (V(s_T)), scanning t = T-1 … 0. `term[e·T+t] > 0.5` = terminal. The GPU kernel matches
    this op-for-op (`--fmad=false`) ⇒ bit-exact; `min` vs the kernel's `<` ternary agree on finite ratios. -/
def vtraceMinGRUFlat (rew val term imp bootv : FloatArray) (B T : Nat)
    (gamma lam rhoClip cClip : Float) : FloatArray := Id.run do
  let mut adv : FloatArray := FloatArray.mk (Array.replicate (B*T) 0.0)
  for e in [0:B] do
    let mut lastA := 0.0
    for i in [0:T] do
      let t := T - 1 - i
      let idx := e*T + t
      let nnt := if term[idx]! > 0.5 then 0.0 else 1.0
      let vNext := if t + 1 < T then val[e*T + (t+1)]! else bootv[e]!
      let rho := min imp[idx]! rhoClip
      let c := min imp[idx]! cClip
      let rr := rew[idx]!
      let r := if rr > 1.0 then 1.0 else if rr < -1.0 then -1.0 else rr
      let delta := rho * r + gamma*vNext*nnt - val[idx]!
      lastA := delta + gamma*lam*c*lastA*nnt
      adv := adv.set! idx lastA
  return adv

/-- Sample `k` indices from `[0, probs.size)` WITH REPLACEMENT, each draw `∝ probs` — the
    `torch.multinomial(prio_probs, …, replacement=True)` of PufferLib's prioritized replay. Linear
    cumulative search (fine for the small per-rollout segment count). The RNG stream is Lean's
    `rngNext`, so the sampled minibatches are NOT the same draws as PyTorch's — the trajectories
    diverge stochastically (the same "chaotic full run" the project validates empirically), but the
    prioritized-sampling DISTRIBUTION is identical. -/
def weightedSampleReplace (probs : Array Float) (k : Nat) (rng0 : UInt64) : Array Nat × UInt64 := Id.run do
  let n := probs.size
  let mut cum : Array Float := Array.replicate n 0.0
  let mut acc := 0.0
  for i in [0:n] do
    acc := acc + probs[i]!
    cum := cum.set! i acc
  let total := acc
  let mut out : Array Nat := #[]
  let mut rng := rng0
  for _ in [0:k] do
    let (word, rng') := rngNext rng
    rng := rng'
    let u := uniform01 word * total
    let mut idx := n - 1
    let mut found := false
    for i in [0:n] do
      if !found && cum[i]! ≥ u then idx := i; found := true
    out := out.push idx
  return (out, rng)

/-- Roll ONE env instance for `horizon` steps under a fixed policy, auto-resetting on
    terminal. Returns `(segment, bootstrapValue, finalState, rng', episodeReturns)`.
    The final state is persistent (the next rollout continues from it); the bootstrap
    value is `V(observe finalState)` (masked in GAE if the last step was terminal). -/
def segmentRollout {S : Type} (env : Env S) (p : MLP) (horizon : Nat) (s0 : S) (rng0 : UInt64) :
    Array Transition × Float × S × UInt64 × Array Float := Id.run do
  let mut st := s0
  let mut rng := rng0
  let mut traj : Array Transition := #[]
  let mut epReturns : Array Float := #[]
  let mut epRet := 0.0
  for _ in [0:horizon] do
    let obs := env.observe st
    let (probs, v) := policyAndValue p obs
    let (word, rng') := rngNext rng
    rng := rng'
    let a := sampleCat probs (uniform01 word)
    let (st', r, term) := env.step st a
    traj := traj.push { obs := obs, action := a, reward := r, value := v,
                        oldLogp := Float.log probs[a]!, terminal := term }
    epRet := epRet + r
    if term then
      epReturns := epReturns.push epRet
      epRet := 0.0
      let (sReset, rng'') := env.reset rng
      rng := rng''
      st := sReset
    else
      st := st'
  let (_, bootV) := policyAndValue p (env.observe st)
  return (traj, bootV, st, rng, epReturns)

/-- Roll `numEnvs` env instances (one segment each) from persistent `states`. The
    policy is fixed during collection, so sequential per-env stepping is identical to
    interleaved stepping (up to RNG consumption order). -/
def vecRollout {S : Type} (env : Env S) (p : MLP) (horizon : Nat)
    (states : Array S) (rng0 : UInt64) :
    Array (Array Transition) × Array Float × Array S × UInt64 × Array Float := Id.run do
  let mut rng := rng0
  let mut trajs : Array (Array Transition) := #[]
  let mut bootVals : Array Float := #[]
  let mut newStates : Array S := #[]
  let mut epReturns : Array Float := #[]
  for st in states do
    let (traj, bootV, st', rng', epRets) := segmentRollout env p horizon st rng
    rng := rng'
    trajs := trajs.push traj
    bootVals := bootVals.push bootV
    newStates := newStates.push st'
    epReturns := epReturns ++ epRets
  return (trajs, bootVals, newStates, rng, epReturns)

/-- Flatten the per-env segments into one experience buffer, computing PufferLib's
    `compute_puff_advantage` (V-Trace/GAE, `computePuffAdvantage`) per segment. Returns
    `(buffer, rawAdvantages, valueTargets)`; advantages are normalized once at batch level by
    the caller. Importance ratios are `1` on this pass (⇒ standard GAE); the live ratios + the
    per-minibatch recompute arrive with prioritized replay. `_bootVals` is unused — PufferLib's
    advantage keeps `A_{last}=0` rather than bootstrapping (kept in the signature for callers). -/
def buildBatch (trajs : Array (Array Transition)) (_bootVals : Array Float) (gamma lam : Float) :
    Array Transition × Array Float × Array Float := Id.run do
  let mut allTr : Array Transition := #[]
  let mut allAdv : Array Float := #[]
  let mut allRet : Array Float := #[]
  for traj in trajs do
    let imp := Array.replicate traj.size 1.0
    let (adv, ret) := computePuffAdvantage traj imp gamma lam 1.0 1.0
    allTr := allTr ++ traj
    allAdv := allAdv ++ adv
    allRet := allRet ++ ret
  return (allTr, allAdv, allRet)

/-- Fisher–Yates shuffle of `[0, n)` using the LCG-style `rngNext`. -/
def shuffleIdx (n : Nat) (rng0 : UInt64) : Array Nat × UInt64 := Id.run do
  let mut a := Array.range n
  let mut rng := rng0
  for i in [0:n] do
    let j := n - 1 - i
    let (word, rng') := rngNext rng
    rng := rng'
    let k := word.toNat % (j + 1)
    let tmp := a[j]!
    a := a.set! j a[k]!
    a := a.set! k tmp
  return (a, rng)

/-- Minibatch PPO gradient: sum of `mlpGradPPO` over the given buffer indices, with
    advantages PRE-NORMALIZED (at batch level) — unlike `ppoGrad`, which normalizes
    the whole trajectory internally. -/
def ppoGradIdx (p : MLP) (buf : Array Transition) (advN returns : Array Float)
    (idxs : Array Nat) (vfCoef entCoef clipEps : Float) :
    Array (Array Float) × Array Float × Array (Array Float) × Array Float := Id.run do
  let mut gW1 := p.W1.map (·.map (fun _ => 0.0))
  let mut gb1 := p.b1.map (fun _ => 0.0)
  let mut gW2 := p.W2.map (·.map (fun _ => 0.0))
  let mut gb2 := p.b2.map (fun _ => 0.0)
  for t in idxs do
    let tr : Transition := buf[t]!
    let (dW1, db1, dW2, db2) :=
      mlpGradPPO p tr.obs tr.action advN[t]! returns[t]! tr.oldLogp vfCoef entCoef clipEps
    gW1 := matAdd gW1 dW1; gb1 := vecAdd gb1 db1
    gW2 := matAdd gW2 dW2; gb2 := vecAdd gb2 db2
  return (gW1, gb1, gW2, gb2)

/-- Sum of squares of every entry across the four gradient tensors. -/
def gradSumSq (gW1 : Array (Array Float)) (gb1 : Array Float)
    (gW2 : Array (Array Float)) (gb2 : Array Float) : Float :=
  let sq := fun (acc x : Float) => acc + x * x
  let vsq := fun (acc : Float) (v : Array Float) => v.foldl sq acc
  let msq := fun (acc : Float) (m : Array (Array Float)) => m.foldl vsq acc
  msq (msq (vsq (vsq 0.0 gb1) gb2) gW1) gW2

/-- One minibatch ascent step: mean gradient over the minibatch (scale `lr/|mb|`,
    PyTorch's mean-loss convention) with global gradient-norm CLIPPING to
    `maxGradNorm` (PufferLib's `max_grad_norm`, `≤ 0` disables) — the standard PPO
    stabilizer against destructive large updates. -/
def updatePPOIdx (p : MLP) (buf : Array Transition) (advN returns : Array Float)
    (idxs : Array Nat) (lr maxGradNorm vfCoef entCoef clipEps : Float) : MLP :=
  let (gW1, gb1, gW2, gb2) := ppoGradIdx p buf advN returns idxs vfCoef entCoef clipEps
  let mb := Float.ofNat (max idxs.size 1)
  -- L2 norm of the MEAN gradient (= sum-gradient norm / |mb|)
  let meanNorm := Float.sqrt (gradSumSq gW1 gb1 gW2 gb2) / mb
  let clipCoef := if maxGradNorm > 0.0 && meanNorm > maxGradNorm then maxGradNorm / meanNorm else 1.0
  let scale := lr * clipCoef / mb
  { W1 := matAxpy scale p.W1 gW1, b1 := vecAxpy scale p.b1 gb1,
    W2 := matAxpy scale p.W2 gW2, b2 := vecAxpy scale p.b2 gb2 }

/-- Index of the largest entry (the greedy action). -/
def argmaxArr (a : Array Float) : Nat := Id.run do
  let mut best := 0
  let mut bv := a[0]!
  for i in [1:a.size] do
    if a[i]! > bv then bv := a[i]!; best := i
  return best

/-- Evaluate the policy GREEDILY (argmax action) over `nEps` episodes. Returns
    `(meanReturn, wins, losses, draws, meanLength, rng')` — win/loss/draw classify each
    episode's total reward by sign (a faithful win-rate for the ±1-terminal envs; for
    dense/continuing envs it is just sign-of-net-return). -/
def evalGreedy {S : Type} (env : Env S) (p : MLP) (nEps : Nat) (rng0 : UInt64) :
    Float × Nat × Nat × Nat × Float × UInt64 := Id.run do
  let mut rng := rng0
  let mut retSum := 0.0
  let mut lenSum := 0
  let mut wins := 0
  let mut losses := 0
  let mut draws := 0
  for _ in [0:nEps] do
    let (s0, r1) := env.reset rng
    rng := r1
    let mut st := s0
    let mut ret := 0.0
    let mut steps := 0
    for _ in [0:env.maxSteps] do
      let (probs, _) := policyAndValue p (env.observe st)
      let (st', r, term) := env.step st (argmaxArr probs)
      ret := ret + r
      st := st'
      steps := steps + 1
      if term then break
    retSum := retSum + ret
    lenSum := lenSum + steps
    if ret > 0.0 then wins := wins + 1
    else if ret < 0.0 then losses := losses + 1
    else draws := draws + 1
  let d := Float.ofNat (max nEps 1)
  return (retSum / d, wins, losses, draws, Float.ofNat lenSum / d, rng)

end Puffer.RL.NNTrain
