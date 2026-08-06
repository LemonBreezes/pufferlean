/-
NN-policy REINFORCE — a 2-layer MLP policy trained by policy gradient over a generic
discrete-action `EnvSpec`, in native `Float`, Mathlib-free.

This replaces the tabular policy (`Puffer/RL/Train.lean`) with a real neural network:

    one-hot(state) → Linear(W1,b1) → ReLU → Linear(W2,b2) → softmax → action

The forward dots use the PROVEN kernel `dotF` (so `Puffer.FloatR.dotF_error` bounds
each pre-activation vs its ℝ value). The backward pass is hand-coded for this fixed
architecture — the seed instance of reverse-mode autodiff (PLAN.md M2): softmax
cross-entropy grad `π − 1[a]`, backprop through the linear/ReLU/linear stack. Same
REINFORCE objective, now with learned features.
-/
import Puffer.Float.Exec
import Puffer.Float.AutoDiff
import Puffer.Float.Muon
import Puffer.RL.Train
import Puffer.RL.EnvSpec

namespace Puffer.RL.NNTrain

open Puffer.RL (Env)

open Puffer.FloatR
open Puffer.RL.Train (rngNext uniform01 softmax sampleCat discountedReturns)

/-- 2-layer MLP parameters: `W1 : H×din`, `b1 : H`, `W2 : dout×H`, `b2 : dout`. -/
structure MLP where
  W1 : Array (Array Float)
  b1 : Array Float
  W2 : Array (Array Float)
  b2 : Array Float

/-- **Lossless MLP checkpoint (serialize).** A whitespace-separated token stream of the EXACT f64 bit
    patterns (`Float.toBits`), tagged `puffer-ckpt 1 <din> <H> <dout>`, so `train --save` / `eval --load`
    round-trip a policy bit-for-bit. Mathlib-free. -/
def mlpSerialize (p : MLP) : String :=
  let din := if 0 < p.W1.size then (p.W1[0]!).size else 0
  let hidden := p.W1.size
  let dout := p.W2.size
  let bits (x : Float) : String := toString x.toBits
  let flatMat (m : Array (Array Float)) : List String := m.toList.flatMap (fun row => row.toList.map bits)
  String.intercalate " "
    (["puffer-ckpt", "1", toString din, toString hidden, toString dout]
      ++ flatMat p.W1 ++ p.b1.toList.map bits ++ flatMat p.W2 ++ p.b2.toList.map bits)

/-- **Lossless MLP checkpoint (deserialize).** Inverse of `mlpSerialize`; validates the tag and that
    the token count matches the declared dims. Returns `none` on any malformed input. -/
def mlpDeserialize (s : String) : Option MLP := Id.run do
  let toks := (((s.replace "\n" " ").replace "\t" " ").replace "\r" " ").splitOn " " |>.filter (· ≠ "")
  match toks with
  | "puffer-ckpt" :: "1" :: dinS :: hidS :: doutS :: rest =>
      match dinS.toNat?, hidS.toNat?, doutS.toNat? with
      | some din, some hidden, some dout =>
          let need := hidden * din + hidden + dout * hidden + dout
          if rest.length ≠ need then return none
          -- all tokens must be valid Nats (bit patterns)
          if rest.any (fun t => t.toNat?.isNone) then return none
          let vals : Array Float := (rest.map (fun t => Float.ofBits (UInt64.ofNat t.toNat!))).toArray
          let W1 := (Array.range hidden).map (fun i => (Array.range din).map (fun j => vals[i * din + j]!))
          let o1 := hidden * din
          let b1 := (Array.range hidden).map (fun i => vals[o1 + i]!)
          let o2 := o1 + hidden
          let W2 := (Array.range dout).map (fun i => (Array.range hidden).map (fun j => vals[o2 + i * hidden + j]!))
          let o3 := o2 + dout * hidden
          let b2 := (Array.range dout).map (fun i => vals[o3 + i]!)
          return some { W1 := W1, b1 := b1, W2 := W2, b2 := b2 }
      | _, _, _ => return none
  | _ => return none

/-- Dot of two `Array`s via the proven `dotF` kernel (so its error bound applies). -/
def dotFA (w x : Array Float) : Float := dotF w.toList x.toList

/-- One-hot encoding of state `i` in dimension `size`. -/
def oneHot (size i : Nat) : Array Float :=
  (Array.range size).map (fun j => if j == i then 1.0 else 0.0)

/-- Forward pass, returning the pre-activation `z1`, hidden `h`, and output `logits`
    (all needed for backprop). -/
def forwardAll (p : MLP) (x : Array Float) : Array Float × Array Float × Array Float :=
  let z1 := (Array.range p.b1.size).map (fun j => p.b1[j]! + dotFA p.W1[j]! x)
  let h := z1.map reluF
  let logits := (Array.range p.b2.size).map (fun k => p.b2[k]! + dotFA p.W2[k]! h)
  (z1, h, logits)

/-- Action probabilities `π(·|s)`. -/
def policyProbs (p : MLP) (size s : Nat) : Array Float :=
  let (_, _, logits) := forwardAll p (oneHot size s)
  softmax logits

/-! ### Rollout and the REINFORCE update (with hand-coded backprop) -/

/-- REINFORCE update: accumulate `∇` of `Σ_t (G_t−b)·log π(a_t|s_t)` over the
    trajectory by backprop through the fixed MLP, then ascend `p += lr·∇`. -/
def updateMLP (p : MLP) (traj : Array (Nat × Nat × Float)) (returns : Array Float)
    (lr baseline : Float) (size : Nat) : MLP := Id.run do
  let H := p.b1.size
  let A := p.b2.size
  let mut gW1 := p.W1.map (·.map (fun _ => 0.0))
  let mut gb1 := p.b1.map (fun _ => 0.0)
  let mut gW2 := p.W2.map (·.map (fun _ => 0.0))
  let mut gb2 := p.b2.map (fun _ => 0.0)
  for t in [0:traj.size] do
    let (s, a, _) := traj[t]!
    let adv := returns[t]! - baseline
    let x := oneHot size s
    let (z1, h, logits) := forwardAll p x
    let probs := softmax logits
    -- output gradient for ASCENT on (adv)·log π(a|s):  dz2[k] = adv·(1[k=a] − π_k)
    let dz2 := (Array.range A).map (fun k => adv * ((if k == a then 1.0 else 0.0) - probs[k]!))
    -- layer 2 grads + upstream to hidden
    for k in [0:A] do
      gb2 := gb2.set! k (gb2[k]! + dz2[k]!)
      gW2 := gW2.set! k ((Array.range H).map (fun j => gW2[k]![j]! + dz2[k]! * h[j]!))
    let dz1 := (Array.range H).map (fun j => Id.run do
      let mut acc := 0.0
      for k in [0:A] do acc := acc + dz2[k]! * p.W2[k]![j]!
      -- ReLU derivative
      return acc * (if z1[j]! > 0.0 then 1.0 else 0.0))
    -- layer 1 grads
    for j in [0:H] do
      gb1 := gb1.set! j (gb1[j]! + dz1[j]!)
      gW1 := gW1.set! j ((Array.range size).map (fun i => gW1[j]![i]! + dz1[j]! * x[i]!))
  -- gradient ascent
  return {
    W1 := (Array.range H).map (fun j => (Array.range size).map (fun i => p.W1[j]![i]! + lr * gW1[j]![i]!)),
    b1 := (Array.range H).map (fun j => p.b1[j]! + lr * gb1[j]!),
    W2 := (Array.range A).map (fun k => (Array.range H).map (fun j => p.W2[k]![j]! + lr * gW2[k]![j]!)),
    b2 := (Array.range A).map (fun k => p.b2[k]! + lr * gb2[k]!) }

/-! ### Same policy gradient, via the GENERAL autodiff engine

`mlpGradAD` builds the REINFORCE objective `adv·log π(a|s)` on the AD tape (params as
leaves) and reads gradients by one reverse sweep — no architecture-specific backprop.
`updateMLPAD`/`trainAD` train with it, proving the general engine subsumes the
hand-coded pass. -/

def vecAxpy (s : Float) (a b : Array Float) : Array Float :=
  (Array.range a.size).map (fun i => a[i]! + s * b[i]!)
def matAxpy (s : Float) (a b : Array (Array Float)) : Array (Array Float) :=
  (Array.range a.size).map (fun i => vecAxpy s a[i]! b[i]!)
def vecAdd (a b : Array Float) : Array Float := vecAxpy 1.0 a b
def matAdd (a b : Array (Array Float)) : Array (Array Float) := matAxpy 1.0 a b

/-- Gradient of `adv·log π(a|s)` w.r.t. the MLP params, computed by reverse-mode AD. -/
def mlpGradAD (p : MLP) (x : Array Float) (a : Nat) (adv : Float) :
    Array (Array Float) × Array Float × Array (Array Float) × Array Float :=
  let build : AD.ADM (AD.V × (Array (Array AD.V) × Array AD.V × Array (Array AD.V) × Array AD.V)) := do
    let W1v ← p.W1.mapM (fun row => row.mapM AD.leaf)
    let b1v ← p.b1.mapM AD.leaf
    let W2v ← p.W2.mapM (fun row => row.mapM AD.leaf)
    let b2v ← p.b2.mapM AD.leaf
    let xv ← x.mapM AD.const
    let z1 ← (Array.range p.b1.size).mapM (fun j => do
      let d ← AD.dotV W1v[j]! xv; AD.add b1v[j]! d)
    let h ← z1.mapM AD.relu
    let logits ← (Array.range p.b2.size).mapM (fun k => do
      let d ← AD.dotV W2v[k]! h; AD.add b2v[k]! d)
    let lse ← AD.logSumExp logits
    let logp ← AD.sub logits[a]! lse
    let obj ← AD.scale adv logp
    return (obj, (W1v, b1v, W2v, b2v))
  let (res, t) := build.run AD.Tape.empty
  let (root, W1v, b1v, W2v, b2v) := res
  let g := AD.grads t root
  (W1v.map (·.map (fun h => g[h]!)), b1v.map (fun h => g[h]!),
   W2v.map (·.map (fun h => g[h]!)), b2v.map (fun h => g[h]!))

/-- REINFORCE update using AD-computed gradients. -/
def updateMLPAD (p : MLP) (traj : Array (Nat × Nat × Float)) (returns : Array Float)
    (lr baseline : Float) (size : Nat) : MLP := Id.run do
  let mut gW1 := p.W1.map (·.map (fun _ => 0.0))
  let mut gb1 := p.b1.map (fun _ => 0.0)
  let mut gW2 := p.W2.map (·.map (fun _ => 0.0))
  let mut gb2 := p.b2.map (fun _ => 0.0)
  for t in [0:traj.size] do
    let (s, a, _) := traj[t]!
    let (dW1, db1, dW2, db2) := mlpGradAD p (oneHot size s) a (returns[t]! - baseline)
    gW1 := matAdd gW1 dW1; gb1 := vecAdd gb1 db1
    gW2 := matAdd gW2 dW2; gb2 := vecAdd gb2 db2
  return { W1 := matAxpy lr p.W1 gW1, b1 := vecAxpy lr p.b1 gb1,
           W2 := matAxpy lr p.W2 gW2, b2 := vecAxpy lr p.b2 gb2 }

/-! ### Actor-critic with GAE(γ,λ)

Shared net now outputs `A` action logits + `1` value (`dout = A+1`). GAE advantages
and value targets are computed from the rollout; the combined objective
`adv·log π(a|s) − vf·½·(V(s)−R)²` is built on the AD tape so both the policy-gradient
and value-regression gradients come from the general autodiff. Advantages are
normalized (mean 0 / unit std) — exactly `Puffer/RL/Normalize`'s `advNorm`. -/

structure Transition where
  obs : Array Float   -- the encoded observation where the action was taken
  action : Nat
  reward : Float
  value : Float
  oldLogp : Float    -- log π_old(a|s) at rollout time (for the PPO ratio)
  terminal : Bool
  deriving Inhabited

/-- Action probabilities and the value estimate `V(s)` from an observation (outputs:
    first `A` are action logits, last is the value). -/
def policyAndValue (p : MLP) (obs : Array Float) : Array Float × Float :=
  let (_, _, out) := forwardAll p obs
  let A := p.b2.size - 1
  (softmax ((Array.range A).map (fun k => out[k]!)), out[A]!)

/-- Roll out one episode under the policy over ANY `Env S`, stopping at the first
    terminal or `env.maxSteps`. -/
def rolloutEnv {S : Type} (env : Env S) (p : MLP) (rng0 : UInt64) :
    Array Transition × Float × UInt64 := Id.run do
  let (s0, rng1) := env.reset rng0
  let mut st := s0
  let mut rng := rng1
  let mut traj : Array Transition := #[]
  let mut ret := 0.0
  for _ in [0:env.maxSteps] do
    let obs := env.observe st
    let (probs, v) := policyAndValue p obs
    let (word, rng') := rngNext rng
    rng := rng'
    let a := sampleCat probs (uniform01 word)
    let (st', r, term) := env.step st a
    traj := traj.push { obs := obs, action := a, reward := r, value := v,
                        oldLogp := Float.log probs[a]!, terminal := term }
    ret := ret + r
    st := st'
    if term then break
  return (traj, ret, rng)

/-- GAE: `A_t = δ_t + γλ·nnt·A_{t+1}`, `δ_t = r_t + γ·V_{t+1}·nnt − V_t`, and value
    targets `R_t = A_t + V_t`. -/
def computeGAE (traj : Array Transition) (gamma lam : Float) : Array Float × Array Float := Id.run do
  let n := traj.size
  let mut adv := Array.replicate n 0.0
  let mut lastA := 0.0
  for i in [0:n] do
    let t := n - 1 - i
    let nnt := if traj[t]!.terminal then 0.0 else 1.0
    let vNext := if t + 1 < n then traj[t+1]!.value else 0.0
    let delta := traj[t]!.reward + gamma * vNext * nnt - traj[t]!.value
    lastA := delta + gamma * lam * nnt * lastA
    adv := adv.set! t lastA
  return (adv, (Array.range n).map (fun t => adv[t]! + traj[t]!.value))

/-- Gradient of `adv·log π(a|s) − vf·½·(V(s)−R)²` w.r.t. the net, via autodiff. -/
def mlpGradAC (p : MLP) (x : Array Float) (a : Nat) (adv ret vfCoef : Float) :
    Array (Array Float) × Array Float × Array (Array Float) × Array Float :=
  let build : AD.ADM (AD.V × (Array (Array AD.V) × Array AD.V × Array (Array AD.V) × Array AD.V)) := do
    let W1v ← p.W1.mapM (fun row => row.mapM AD.leaf)
    let b1v ← p.b1.mapM AD.leaf
    let W2v ← p.W2.mapM (fun row => row.mapM AD.leaf)
    let b2v ← p.b2.mapM AD.leaf
    let xv ← x.mapM AD.const
    let z1 ← (Array.range p.b1.size).mapM (fun j => do let d ← AD.dotV W1v[j]! xv; AD.add b1v[j]! d)
    let h ← z1.mapM AD.relu
    let out ← (Array.range p.b2.size).mapM (fun k => do let d ← AD.dotV W2v[k]! h; AD.add b2v[k]! d)
    let A := p.b2.size - 1
    let lse ← AD.logSumExp ((Array.range A).map (fun k => out[k]!))
    let logp ← AD.sub out[a]! lse
    let term1 ← AD.scale adv logp
    let diff ← AD.sub out[A]! (← AD.const ret)
    let term2 ← AD.scale (vfCoef * 0.5) (← AD.mul diff diff)
    let obj ← AD.sub term1 term2
    return (obj, (W1v, b1v, W2v, b2v))
  let (res, t) := build.run AD.Tape.empty
  let (root, W1v, b1v, W2v, b2v) := res
  let g := AD.grads t root
  (W1v.map (·.map (fun h => g[h]!)), b1v.map (fun h => g[h]!),
   W2v.map (·.map (fun h => g[h]!)), b2v.map (fun h => g[h]!))

/-- Normalize advantages to mean 0 / unit std (Puffer's `adv_normalized`). -/
def normalizeAdv (adv : Array Float) : Array Float := Id.run do
  let n := Float.ofNat adv.size
  let mean := adv.foldl (· + ·) 0.0 / n
  let var := adv.foldl (fun s x => s + (x - mean) * (x - mean)) 0.0 / n
  let std := Float.sqrt var
  return adv.map (fun x => (x - mean) / (std + 1e-8))

def updateAC (p : MLP) (traj : Array Transition) (adv returns : Array Float)
    (lr vfCoef : Float) : MLP := Id.run do
  let advN := normalizeAdv adv
  let mut gW1 := p.W1.map (·.map (fun _ => 0.0))
  let mut gb1 := p.b1.map (fun _ => 0.0)
  let mut gW2 := p.W2.map (·.map (fun _ => 0.0))
  let mut gb2 := p.b2.map (fun _ => 0.0)
  for t in [0:traj.size] do
    let tr : Transition := traj[t]!
    let (dW1, db1, dW2, db2) := mlpGradAC p tr.obs tr.action advN[t]! returns[t]! vfCoef
    gW1 := matAdd gW1 dW1; gb1 := vecAdd gb1 db1
    gW2 := matAdd gW2 dW2; gb2 := vecAdd gb2 db2
  return { W1 := matAxpy lr p.W1 gW1, b1 := vecAxpy lr p.b1 gb1,
           W2 := matAxpy lr p.W2 gW2, b2 := vecAxpy lr p.b2 gb2 }

/-- Gradient of the PPO objective for one transition, via autodiff:
    `min(ρ·A, clip(ρ,1−ε,1+ε)·A) − vf·½(V−R)² + ent·H`,  `ρ = exp(logπ − logπ_old)`. -/
def mlpGradPPO (p : MLP) (x : Array Float) (a : Nat)
    (adv ret oldLogp vfCoef entCoef clipEps : Float) :
    Array (Array Float) × Array Float × Array (Array Float) × Array Float :=
  let build : AD.ADM (AD.V × (Array (Array AD.V) × Array AD.V × Array (Array AD.V) × Array AD.V)) := do
    let W1v ← p.W1.mapM (fun row => row.mapM AD.leaf)
    let b1v ← p.b1.mapM AD.leaf
    let W2v ← p.W2.mapM (fun row => row.mapM AD.leaf)
    let b2v ← p.b2.mapM AD.leaf
    let xv ← x.mapM AD.const
    let z1 ← (Array.range p.b1.size).mapM (fun j => do let d ← AD.dotV W1v[j]! xv; AD.add b1v[j]! d)
    let h ← z1.mapM AD.relu
    let out ← (Array.range p.b2.size).mapM (fun k => do let d ← AD.dotV W2v[k]! h; AD.add b2v[k]! d)
    let A := p.b2.size - 1
    let lse ← AD.logSumExp ((Array.range A).map (fun k => out[k]!))
    -- clipped surrogate:  min(ρ·A, clip(ρ)·A),  ρ = exp(logπ(a) − oldLogp)
    let logpA ← AD.sub out[a]! lse
    let oldC ← AD.const oldLogp
    let ratio ← AD.exp (← AD.sub logpA oldC)
    let surr1 ← AD.scale adv ratio
    let ratioC ← AD.clampC (1.0 - clipEps) (1.0 + clipEps) ratio
    let surr2 ← AD.scale adv ratioC
    let polObj ← AD.minV surr1 surr2
    -- value loss  ½(V − R)²
    let diff ← AD.sub out[A]! (← AD.const ret)
    let vloss ← AD.mul diff diff
    -- entropy  H = −Σ π_k logπ_k
    let mut ent ← AD.const 0.0
    for k in [0:A] do
      let logpk ← AD.sub out[k]! lse
      let pk ← AD.exp logpk
      let term ← AD.mul pk logpk
      ent ← AD.sub ent term
    -- objective = polObj − vf·½·vloss + ent·H
    let vterm ← AD.scale (vfCoef * 0.5) vloss
    let eterm ← AD.scale entCoef ent
    let obj ← AD.add (← AD.sub polObj vterm) eterm
    return (obj, (W1v, b1v, W2v, b2v))
  let (res, t) := build.run AD.Tape.empty
  let (root, W1v, b1v, W2v, b2v) := res
  let g := AD.grads t root
  (W1v.map (·.map (fun h => g[h]!)), b1v.map (fun h => g[h]!),
   W2v.map (·.map (fun h => g[h]!)), b2v.map (fun h => g[h]!))

/-- Accumulated PPO objective gradient over the rollout (advantages normalized). -/
def ppoGrad (p : MLP) (traj : Array Transition) (adv returns : Array Float)
    (vfCoef entCoef clipEps : Float) :
    Array (Array Float) × Array Float × Array (Array Float) × Array Float := Id.run do
  let advN := normalizeAdv adv
  let mut gW1 := p.W1.map (·.map (fun _ => 0.0))
  let mut gb1 := p.b1.map (fun _ => 0.0)
  let mut gW2 := p.W2.map (·.map (fun _ => 0.0))
  let mut gb2 := p.b2.map (fun _ => 0.0)
  for t in [0:traj.size] do
    let tr : Transition := traj[t]!
    let (dW1, db1, dW2, db2) :=
      mlpGradPPO p tr.obs tr.action advN[t]! returns[t]! tr.oldLogp vfCoef entCoef clipEps
    gW1 := matAdd gW1 dW1; gb1 := vecAdd gb1 db1
    gW2 := matAdd gW2 dW2; gb2 := vecAdd gb2 db2
  return (gW1, gb1, gW2, gb2)

/-- One PPO epoch with a plain SGD (ascent) step. -/
def updatePPO (p : MLP) (traj : Array Transition) (adv returns : Array Float)
    (lr vfCoef entCoef clipEps : Float) : MLP :=
  let (gW1, gb1, gW2, gb2) := ppoGrad p traj adv returns vfCoef entCoef clipEps
  { W1 := matAxpy lr p.W1 gW1, b1 := vecAxpy lr p.b1 gb1,
    W2 := matAxpy lr p.W2 gW2, b2 := vecAxpy lr p.b2 gb2 }

/-! ### Muon optimizer path (executable, from `Puffer/Float/Muon.lean`) -/

/-- Muon momentum buffers, one per parameter tensor. -/
structure MuonState where
  mW1 : Array (Array Float)
  mb1 : Array Float
  mW2 : Array (Array Float)
  mb2 : Array Float

def MuonState.zeros (p : MLP) : MuonState :=
  { mW1 := p.W1.map (·.map (fun _ => 0.0)), mb1 := p.b1.map (fun _ => 0.0),
    mW2 := p.W2.map (·.map (fun _ => 0.0)), mb2 := p.b2.map (fun _ => 0.0) }

/-- Apply one Muon step to all params (2D weights orthogonalized, 1D biases not). -/
def applyMuon (p : MLP) (st : MuonState)
    (g : Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) : MLP × MuonState :=
  let (gW1, gb1, gW2, gb2) := g
  let (nW1, mW1) := Muon.stepMat p.W1 gW1 st.mW1 lr wd mu eps
  let (nb1, mb1) := Muon.stepVec p.b1 gb1 st.mb1 lr wd mu
  let (nW2, mW2) := Muon.stepMat p.W2 gW2 st.mW2 lr wd mu eps
  let (nb2, mb2) := Muon.stepVec p.b2 gb2 st.mb2 lr wd mu
  ({ W1 := nW1, b1 := nb1, W2 := nW2, b2 := nb2 },
   { mW1 := mW1, mb1 := mb1, mW2 := mW2, mb2 := mb2 })

/-! ### Weight init and the training entry point -/

/-- A random `rows×cols` matrix with entries in `[-scale, scale)`. -/
def randMat (rows cols : Nat) (scale : Float) (rng0 : UInt64) : Array (Array Float) × UInt64 := Id.run do
  let mut rng := rng0
  let mut m : Array (Array Float) := #[]
  for _ in [0:rows] do
    let mut row : Array Float := #[]
    for _ in [0:cols] do
      let (word, rng') := rngNext rng
      rng := rng'
      row := row.push ((uniform01 word) * (2.0 * scale) - scale)
    m := m.push row
  return (m, rng)

/-- Random 2-layer MLP (`din → H → dout`), biases zero.

Kaiming-uniform init, `bound = gain/√fan_in`, matching PufferLib's own `puf_kaiming_init`
(`src/kernels.cu:383`) with their per-stage gains: `√2` for the ReLU encoder (`src/models.cu:431`)
and `1.0` for the decoder head (`src/models.cu:502`).

This replaces a FIXED 0.3 scale that ignored fan-in. Because the logit magnitude then grew like
`√H`, the initial softmax saturated at large hidden sizes: at `hidden=512` (upstream's setting for
maze, terraform, g2048, go, tetris, dino, nmmo3) the policy was effectively deterministic from step
one, so the agent repeated a single action, never reached a goal, and no reward signal ever appeared
— maze read a flat 0.0 forever while a uniform-random policy solves 58% of its episodes. Hidden 64
happened to stay under the saturation knee, which is why smaller envs masked the bug.
`initMinGRU` already did this correctly (see its comment naming the same failure mode). -/
def initMLP (din H dout : Nat) (rng0 : UInt64) : MLP × UInt64 := Id.run do
  let inv := fun (n : Nat) => 1.0 / Float.sqrt (Float.ofNat n)
  let (W1, rng1) := randMat H din (Float.sqrt 2.0 * inv din) rng0
  let (W2, rng2) := randMat dout H (inv H) rng1
  return ({ W1 := W1, b1 := Array.replicate H 0.0, W2 := W2, b2 := Array.replicate dout 0.0 }, rng2)

/-- Average episode return of policy `p` over `n` fresh episodes on `env`.
    Returns `(meanReturn, meanLength, advancedRng)`. -/
def evalPolicy {S : Type} (env : Env S) (p : MLP) (n : Nat) (rng0 : UInt64) :
    Float × Float × UInt64 := Id.run do
  let mut rng := rng0
  let mut retSum := 0.0
  let mut lenSum := 0.0
  for _ in [0:n] do
    let (traj, ret, rng') := rolloutEnv env p rng
    rng := rng'
    retSum := retSum + ret
    lenSum := lenSum + Float.ofNat traj.size
  let d := Float.ofNat (max n 1)
  return (retSum / d, lenSum / d, rng)

end Puffer.RL.NNTrain
