/-
# Recurrent (LSTM) PPO with truncated BPTT (Tier 2)

The feedforward trainer computes a per-transition gradient. A recurrent policy's
output at step `t` depends on the whole history through the LSTM hidden state, so its
gradient needs **backprop through time**: the entire sequence's forward unroll is
built on ONE autodiff tape, and reverse-mode over that tape IS BPTT.

**Policy.** A single LSTM layer `obs → (h,c) → linear head`. Weights: `Wx : 4H×obsDim`
and `Wh : 4H×H` (the input/forget/candidate/output gates stacked), a combined bias
`bih : 4H` (forget-gate bias initialised to 1), and an output head `Wo : dout×H`,
`bo : dout` (`dout = numActions+1`). The standard cell:
`i,f,o = σ(gate); g = tanh(gate); c' = f⊙c + i⊙g; h' = o⊙tanh(c')`
(σ/tanh are the new `AD.sigmoid`/`AD.tanh` primitives).

**Rollout.** The hidden state `(h,c)` is threaded through each env and PERSISTS across
rollouts, resetting to 0 at episode boundaries (truncated BPTT with window = horizon).

**Update.** Per env-sequence: rebuild the LSTM unroll on one tape from the segment's
initial `(h,c)` (a detached constant — the truncation point), resetting `(h,c)` to 0
after each terminal, summing the per-step PPO objective (identical to `mlpGradPPO`'s:
clipped surrogate − vf·½·value-loss + ent·H). Reverse-mode gives the gradient w.r.t.
all LSTM+head weights; one mean-scaled, grad-clipped ascent step per sequence.

This is what POMDP tasks need — with a *partially-observed* cue (shown once), a
feedforward net cannot solve them but an LSTM can.

Mathlib-free (extends `VecTrain`); the binary links no Mathlib.
-/
import Puffer.RL.VecTrain

namespace Puffer.RL.NNTrain

open Puffer.RL (Env)
open Puffer.FloatR
open Puffer.RL.Train (rngNext uniform01 softmax sampleCat)

/-! ### The LSTM policy -/

structure RecPolicy where
  Wx : Array (Array Float)   -- 4H × obsDim
  Wh : Array (Array Float)   -- 4H × H
  bih : Array Float           -- 4H
  Wo : Array (Array Float)   -- dout × H
  bo : Array Float            -- dout
  hSize : Nat

def initRec (obsDim hidden dout : Nat) (rng0 : UInt64) : RecPolicy × UInt64 := Id.run do
  let (wx, r1) := randMat (4 * hidden) obsDim 0.2 rng0
  let (wh, r2) := randMat (4 * hidden) hidden 0.2 r1
  let bih := (Array.range (4 * hidden)).map (fun k =>
    if hidden ≤ k && k < 2 * hidden then 1.0 else 0.0)   -- forget-gate bias = 1
  let (wo, r3) := randMat dout hidden 0.2 r2
  let bo := (Array.range dout).map (fun _ => 0.0)
  return ({ Wx := wx, Wh := wh, bih := bih, Wo := wo, bo := bo, hSize := hidden }, r3)

@[inline] def sigF (x : Float) : Float := 1.0 / (1.0 + Float.exp (-x))
/-- Left-fold dot (matches `AD.dotV`'s summation order, so rollout ≈ BPTT primal). -/
@[inline] def dotL (w x : Array Float) : Float :=
  (Array.range w.size).foldl (fun s i => s + w[i]! * x[i]!) 0.0
@[inline] def zeros (n : Nat) : Array Float := Array.replicate n 0.0

/-- One LSTM step in `Float` (rollout): returns `(h', c', out)`. -/
def lstmCellF (p : RecPolicy) (x h c : Array Float) : Array Float × Array Float × Array Float := Id.run do
  let H := p.hSize
  let gate := (Array.range (4 * H)).map (fun k => p.bih[k]! + dotL p.Wx[k]! x + dotL p.Wh[k]! h)
  let mut cN : Array Float := #[]
  let mut hN : Array Float := #[]
  for j in [0:H] do
    let ig := sigF gate[j]!
    let fg := sigF gate[H + j]!
    let gg := Float.tanh gate[2 * H + j]!
    let og := sigF gate[3 * H + j]!
    let cj := fg * c[j]! + ig * gg
    cN := cN.push cj
    hN := hN.push (og * Float.tanh cj)
  let out := (Array.range p.bo.size).map (fun m => p.bo[m]! + dotL p.Wo[m]! hN)
  return (hN, cN, out)

/-- Action probabilities + value from an LSTM output (`dout = numActions+1`). -/
def recProbsValue (out : Array Float) (numActions : Nat) : Array Float × Float :=
  (softmax ((Array.range numActions).map (fun k => out[k]!)), out[numActions]!)

/-! ### Recurrent rollout (hidden state threaded, reset on episode end) -/

def segmentRolloutRec {S : Type} (env : Env S) (p : RecPolicy) (horizon : Nat)
    (s0 : S) (h0 c0 : Array Float) (rng0 : UInt64) :
    Array Transition × Float × S × Array Float × Array Float × UInt64 × Array Float := Id.run do
  let H := p.hSize
  let mut st := s0
  let mut h := h0
  let mut c := c0
  let mut rng := rng0
  let mut traj : Array Transition := #[]
  let mut epReturns : Array Float := #[]
  let mut epRet := 0.0
  for _ in [0:horizon] do
    let obs := env.observe st
    let (h', c', out) := lstmCellF p obs h c
    let (probs, v) := recProbsValue out env.numActions
    let (word, rng') := rngNext rng
    rng := rng'
    let a := sampleCat probs (uniform01 word)
    let (st', r, term) := env.step st a
    traj := traj.push { obs := obs, action := a, reward := r, value := v,
                        oldLogp := Float.log probs[a]!, terminal := term }
    epRet := epRet + r
    h := h'; c := c'
    if term then
      epReturns := epReturns.push epRet
      epRet := 0.0
      let (sReset, rng'') := env.reset rng
      rng := rng''
      st := sReset
      h := zeros H; c := zeros H          -- reset LSTM state at episode boundary
    else
      st := st'
  let (_, _, outB) := lstmCellF p (env.observe st) h c
  let (_, bootV) := recProbsValue outB env.numActions
  return (traj, bootV, st, h, c, rng, epReturns)

/-- Roll `numEnvs` env instances, carrying each one's persistent `(h,c)`. Returns the
    per-env segments, bootstrap values, and the UPDATED states/hidden states. The
    caller keeps the OLD `hs`/`cs` (the segments' BPTT-initial states). -/
def vecRolloutRec {S : Type} (env : Env S) (p : RecPolicy) (horizon : Nat)
    (states : Array S) (hs cs : Array (Array Float)) (rng0 : UInt64) :
    Array (Array Transition) × Array Float × Array S × Array (Array Float) × Array (Array Float)
      × UInt64 × Array Float := Id.run do
  let mut rng := rng0
  let mut trajs : Array (Array Transition) := #[]
  let mut bootVals : Array Float := #[]
  let mut newStates : Array S := #[]
  let mut newHs : Array (Array Float) := #[]
  let mut newCs : Array (Array Float) := #[]
  let mut epReturns : Array Float := #[]
  for (st, hc) in states.zip (hs.zip cs) do
    let (h, c) := hc
    let (traj, bootV, st', h', c', rng', epRets) := segmentRolloutRec env p horizon st h c rng
    rng := rng'
    trajs := trajs.push traj
    bootVals := bootVals.push bootV
    newStates := newStates.push st'
    newHs := newHs.push h'
    newCs := newCs.push c'
    epReturns := epReturns ++ epRets
  return (trajs, bootVals, newStates, newHs, newCs, rng, epReturns)

/-! ### Per-step PPO objective (factored from `mlpGradPPO`) + BPTT gradient -/

/-- The per-step PPO objective from output logits `out` (`dout = A+1`), on the tape:
    `min(ρ·A, clip(ρ)·A) − vf·½(V−R)² + ent·H`. -/
def ppoStepObj (out : Array AD.V) (A a : Nat) (adv ret oldLogp vfCoef entCoef clipEps : Float) :
    AD.ADM AD.V := do
  let lse ← AD.logSumExp ((Array.range A).map (fun k => out[k]!))
  let logpA ← AD.sub out[a]! lse
  let ratio ← AD.exp (← AD.sub logpA (← AD.const oldLogp))
  let surr1 ← AD.scale adv ratio
  let surr2 ← AD.scale adv (← AD.clampC (1.0 - clipEps) (1.0 + clipEps) ratio)
  let polObj ← AD.minV surr1 surr2
  let diff ← AD.sub out[A]! (← AD.const ret)
  let vloss ← AD.mul diff diff
  let mut ent ← AD.const 0.0
  for k in [0:A] do
    let logpk ← AD.sub out[k]! lse
    ent ← AD.sub ent (← AD.mul (← AD.exp logpk) logpk)
  AD.add (← AD.sub polObj (← AD.scale (vfCoef * 0.5) vloss)) (← AD.scale entCoef ent)

/-- Gradient bundle for the LSTM policy. -/
structure RecGrad where
  gWx : Array (Array Float)
  gWh : Array (Array Float)
  gbih : Array Float
  gWo : Array (Array Float)
  gbo : Array Float

/-- The shared tape build for one env-sequence: the whole LSTM unroll (from the
    detached initial `(h0,c0)`, resetting `(h,c)` to 0 after each terminal) with the
    summed per-step PPO objective as the root. Returns the root and the weight-leaf
    handles (for gradient extraction). -/
def recSeqBuild (p : RecPolicy) (traj : Array Transition) (h0 c0 : Array Float)
    (advN returns : Array Float) (numActions : Nat) (vfCoef entCoef clipEps : Float) :
    AD.ADM (AD.V ×
      (Array (Array AD.V) × Array (Array AD.V) × Array AD.V × Array (Array AD.V) × Array AD.V)) := do
  let H := p.hSize
  let A := numActions
  let Wxv ← p.Wx.mapM (fun row => row.mapM AD.leaf)
  let Whv ← p.Wh.mapM (fun row => row.mapM AD.leaf)
  let bihv ← p.bih.mapM AD.leaf
  let Wov ← p.Wo.mapM (fun row => row.mapM AD.leaf)
  let bov ← p.bo.mapM AD.leaf
  let mut h ← h0.mapM AD.const
  let mut c ← c0.mapM AD.const
  let mut total ← AD.const 0.0
  for t in [0:traj.size] do
    let tr := traj[t]!
    if t > 0 && traj[t-1]!.terminal then
      h ← (zeros H).mapM AD.const
      c ← (zeros H).mapM AD.const
    let xv ← tr.obs.mapM AD.const
    let gate ← (Array.range (4 * H)).mapM (fun k => do
      let gx ← AD.dotV Wxv[k]! xv
      let gh ← AD.dotV Whv[k]! h
      AD.add bihv[k]! (← AD.add gx gh))
    let mut cN : Array AD.V := #[]
    let mut hN : Array AD.V := #[]
    for j in [0:H] do
      let ig ← AD.sigmoid gate[j]!
      let fg ← AD.sigmoid gate[H + j]!
      let gg ← AD.tanh gate[2 * H + j]!
      let og ← AD.sigmoid gate[3 * H + j]!
      let cj ← AD.add (← AD.mul fg c[j]!) (← AD.mul ig gg)
      cN := cN.push cj
      hN := hN.push (← AD.mul og (← AD.tanh cj))
    h := hN; c := cN
    let out ← (Array.range p.bo.size).mapM (fun m => do let d ← AD.dotV Wov[m]! h; AD.add bov[m]! d)
    let objT ← ppoStepObj out A tr.action advN[t]! returns[t]! tr.oldLogp vfCoef entCoef clipEps
    total ← AD.add total objT
  return (total, (Wxv, Whv, bihv, Wov, bov))

/-- BPTT gradient of the summed PPO objective over one env-sequence. -/
def recPPOGradSeq (p : RecPolicy) (traj : Array Transition) (h0 c0 : Array Float)
    (advN returns : Array Float) (numActions : Nat) (vfCoef entCoef clipEps : Float) : RecGrad :=
  let build := recSeqBuild p traj h0 c0 advN returns numActions vfCoef entCoef clipEps
  let (res, tp) := build.run AD.Tape.empty
  let (root, Wxv, Whv, bihv, Wov, bov) := res
  let g := AD.grads tp root
  { gWx := Wxv.map (·.map (fun v => g[v]!)), gWh := Whv.map (·.map (fun v => g[v]!)),
    gbih := bihv.map (fun v => g[v]!), gWo := Wov.map (·.map (fun v => g[v]!)),
    gbo := bov.map (fun v => g[v]!) }

/-- The summed-PPO-objective primal value over one sequence (for finite-difference
    gradient checks of `recPPOGradSeq`). -/
def recSeqObjPrimal (p : RecPolicy) (traj : Array Transition) (h0 c0 : Array Float)
    (advN returns : Array Float) (numActions : Nat) (vfCoef entCoef clipEps : Float) : Float :=
  let build := recSeqBuild p traj h0 c0 advN returns numActions vfCoef entCoef clipEps
  let (res, tp) := build.run AD.Tape.empty
  AD.valueAt tp res.1

/-- Sum of squares over all five LSTM gradient tensors. -/
def recGradSumSq (gr : RecGrad) : Float :=
  let sq := fun (acc x : Float) => acc + x * x
  let vsq := fun (acc : Float) (v : Array Float) => v.foldl sq acc
  let msq := fun (acc : Float) (m : Array (Array Float)) => m.foldl vsq acc
  msq (msq (msq (vsq (vsq 0.0 gr.gbih) gr.gbo) gr.gWx) gr.gWh) gr.gWo

/-- One BPTT ascent step for a sequence: mean gradient over the `seqLen` timesteps
    (scale `lr/seqLen`) with global grad-norm clipping. -/
def applyRecGrad (p : RecPolicy) (gr : RecGrad) (lr maxGradNorm : Float) (seqLen : Nat) : RecPolicy :=
  let sl := Float.ofNat (max seqLen 1)
  let meanNorm := Float.sqrt (recGradSumSq gr) / sl
  let cc := if maxGradNorm > 0.0 && meanNorm > maxGradNorm then maxGradNorm / meanNorm else 1.0
  let s := lr * cc / sl
  { p with
    Wx := matAxpy s p.Wx gr.gWx, Wh := matAxpy s p.Wh gr.gWh,
    bih := vecAxpy s p.bih gr.gbih, Wo := matAxpy s p.Wo gr.gWo, bo := vecAxpy s p.bo gr.gbo }

/-! ### The recurrent trainer -/

end Puffer.RL.NNTrain
