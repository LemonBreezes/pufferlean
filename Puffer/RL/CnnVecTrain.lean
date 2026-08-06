/-
# CNN-encoder PPO for spatial (grid) observations (Tier 2)

The MLP head flattens the observation into a vector, discarding 2D structure. For
grid/pixel envs (snake's head-centred crop, maze vision windows, board planes) a
CNN encoder — convolutions with weight sharing — matches the spatial structure (this
is what PufferLib's `puffernet` uses). This module adds that head.

**Policy.** The flat observation is read as `chans × inH × inW` (row-major). One conv
layer of `nFilters` filters (`chans × kSize × kSize`, given `stride`, VALID padding)
→ ReLU → flatten → a 1-hidden-layer MLP → `dout = numActions+1` logits/value:
`out = W2·relu(W1·flatten(relu(conv(x))) + b1) + b2`.

**Gradient.** Each conv output pixel is `bias + Σ filter⊙patch`, i.e. a `dotV` of the
filter row against the gathered input patch (both share the `(c,ky,kx)` index order via
`patchIdx`), then `relu`. Building the whole conv+dense forward on the autodiff tape and
running reverse-mode gives the exact gradient (FD-verified). The per-step PPO objective
is `ppoStepObj` (shared with the recurrent head). Reuses the vectorized machinery
(experience buffer, minibatched PPO, GAE, LR anneal, grad clip) via a higher-order
rollout parameterized by the policy forward.

Mathlib-free (extends `RecVecTrain`); the binary links no Mathlib.
-/
import Puffer.RL.RecVecTrain

namespace Puffer.RL.NNTrain

open Puffer.RL (Env)
open Puffer.FloatR
open Puffer.RL.Train (rngNext uniform01 softmax sampleCat)

/-! ### The CNN policy -/

structure CnnPolicy where
  convW : Array (Array Float)   -- nFilters × (chans·k·k)
  convB : Array Float            -- nFilters
  W1 : Array (Array Float)      -- hidden × flatDim
  b1 : Array Float
  W2 : Array (Array Float)      -- dout × hidden
  b2 : Array Float
  chans : Nat
  inH : Nat
  inW : Nat
  nFilters : Nat
  kSize : Nat
  stride : Nat
  /-- Number of leading obs scalars passed through (concatenated after the conv features,
      before the dense head) rather than fed to the conv. The conv reads `obs[nScalar:]`
      reshaped `chans×inH×inW`. `0` ⇒ pure CNN (the image is the whole obs). -/
  nScalar : Nat := 0

@[inline] def cnnOutH (p : CnnPolicy) : Nat := (p.inH - p.kSize) / p.stride + 1
@[inline] def cnnOutW (p : CnnPolicy) : Nat := (p.inW - p.kSize) / p.stride + 1
/-- Dense-head input dim = flattened conv map + passthrough scalars. -/
@[inline] def cnnFlatDim (p : CnnPolicy) : Nat := p.nFilters * cnnOutH p * cnnOutW p + p.nScalar

def initCnn (chans inH inW nFilters kSize stride hidden dout : Nat) (rng0 : UInt64)
    (nScalar : Nat := 0) : CnnPolicy × UInt64 := Id.run do
  let (cw, r1) := randMat nFilters (chans * kSize * kSize) 0.3 rng0
  let cb := (Array.range nFilters).map (fun _ => 0.0)
  let outH := (inH - kSize) / stride + 1
  let outW := (inW - kSize) / stride + 1
  let flatDim := nFilters * outH * outW + nScalar
  let (w1, r2) := randMat hidden flatDim 0.2 r1
  let b1 := (Array.range hidden).map (fun _ => 0.0)
  let (w2, r3) := randMat dout hidden 0.2 r2
  let b2 := (Array.range dout).map (fun _ => 0.0)
  return ({ convW := cw, convB := cb, W1 := w1, b1 := b1, W2 := w2, b2 := b2,
            chans := chans, inH := inH, inW := inW, nFilters := nFilters,
            kSize := kSize, stride := stride, nScalar := nScalar }, r3)

/-- Flat observation indices of the conv patch at output position `(oy,ox)`, in the
    `(c,ky,kx)` order matching a filter row of `convW`. Offset by `nScalar` so the conv
    reads the image part `obs[nScalar:]`. -/
def patchIdx (p : CnnPolicy) (oy ox : Nat) : Array Nat :=
  let C := p.chans; let H := p.inH; let W := p.inW; let k := p.kSize; let s := p.stride
  (Array.range (C * k * k)).map (fun idx =>
    let c := idx / (k * k); let rem := idx % (k * k); let ky := rem / k; let kx := rem % k
    p.nScalar + (c * H + (oy * s + ky)) * W + (ox * s + kx))

/-- Conv → ReLU feature map (flattened `nFilters·outH·outW`) then the `nScalar` passthrough
    scalars `obs[0:nScalar]` appended. -/
def convForwardF (p : CnnPolicy) (obs : Array Float) : Array Float := Id.run do
  let oH := cnnOutH p; let oW := cnnOutW p
  let mut feat : Array Float := #[]
  for f in [0:p.nFilters] do
    for oy in [0:oH] do
      for ox in [0:oW] do
        let pidx := patchIdx p oy ox
        let acc := p.convB[f]! + dotL p.convW[f]! (pidx.map (fun i => obs[i]!))
        feat := feat.push (if acc > 0.0 then acc else 0.0)
  for j in [0:p.nScalar] do feat := feat.push obs[j]!    -- passthrough scalars
  return feat

/-- CNN forward → output logits/value (`dout = numActions+1`). -/
def cnnForward (p : CnnPolicy) (obs : Array Float) : Array Float :=
  let feat := convForwardF p obs
  let z1 := (Array.range p.b1.size).map (fun j => p.b1[j]! + dotL p.W1[j]! feat)
  let h := z1.map (fun z => if z > 0.0 then z else 0.0)
  (Array.range p.b2.size).map (fun k => p.b2[k]! + dotL p.W2[k]! h)

def cnnProbsValue (p : CnnPolicy) (numActions : Nat) (obs : Array Float) : Array Float × Float :=
  let out := cnnForward p obs
  (softmax ((Array.range numActions).map (fun k => out[k]!)), out[numActions]!)

/-! ### Higher-order rollout (parameterized by the policy forward) -/

def segmentRolloutF {S : Type} (env : Env S) (pv : Array Float → Array Float × Float)
    (horizon : Nat) (s0 : S) (rng0 : UInt64) :
    Array Transition × Float × S × UInt64 × Array Float := Id.run do
  let mut st := s0
  let mut rng := rng0
  let mut traj : Array Transition := #[]
  let mut epReturns : Array Float := #[]
  let mut epRet := 0.0
  for _ in [0:horizon] do
    let obs := env.observe st
    let (probs, v) := pv obs
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
  let (_, bootV) := pv (env.observe st)
  return (traj, bootV, st, rng, epReturns)

def vecRolloutF {S : Type} (env : Env S) (pv : Array Float → Array Float × Float)
    (horizon : Nat) (states : Array S) (rng0 : UInt64) :
    Array (Array Transition) × Array Float × Array S × UInt64 × Array Float := Id.run do
  let mut rng := rng0
  let mut trajs : Array (Array Transition) := #[]
  let mut bootVals : Array Float := #[]
  let mut newStates : Array S := #[]
  let mut epReturns : Array Float := #[]
  for st in states do
    let (traj, bootV, st', rng', epRets) := segmentRolloutF env pv horizon st rng
    rng := rng'
    trajs := trajs.push traj
    bootVals := bootVals.push bootV
    newStates := newStates.push st'
    epReturns := epReturns ++ epRets
  return (trajs, bootVals, newStates, rng, epReturns)

/-! ### CNN PPO gradient (conv + dense on the AD tape) -/

/-- The conv+dense forward + summed PPO objective for one transition, on the tape.
    Returns the objective root and the weight-leaf handles. -/
def cnnBuild (p : CnnPolicy) (obs : Array Float) (a A : Nat)
    (adv ret oldLogp vfCoef entCoef clipEps : Float) :
    AD.ADM (AD.V ×
      (Array (Array AD.V) × Array AD.V × Array (Array AD.V) × Array AD.V × Array (Array AD.V) × Array AD.V)) := do
  let convWv ← p.convW.mapM (fun row => row.mapM AD.leaf)
  let convBv ← p.convB.mapM AD.leaf
  let W1v ← p.W1.mapM (fun row => row.mapM AD.leaf)
  let b1v ← p.b1.mapM AD.leaf
  let W2v ← p.W2.mapM (fun row => row.mapM AD.leaf)
  let b2v ← p.b2.mapM AD.leaf
  let xv ← obs.mapM AD.const
  let oH := cnnOutH p; let oW := cnnOutW p
  -- conv → relu feature map
  let mut feat : Array AD.V := #[]
  for f in [0:p.nFilters] do
    for oy in [0:oH] do
      for ox in [0:oW] do
        let xPatch := (patchIdx p oy ox).map (fun i => xv[i]!)
        let dp ← AD.dotV convWv[f]! xPatch
        feat := feat.push (← AD.relu (← AD.add convBv[f]! dp))
  for j in [0:p.nScalar] do feat := feat.push xv[j]!    -- passthrough scalars (consts)
  -- dense head
  let z1 ← (Array.range p.b1.size).mapM (fun j => do let d ← AD.dotV W1v[j]! feat; AD.add b1v[j]! d)
  let h ← z1.mapM AD.relu
  let out ← (Array.range p.b2.size).mapM (fun k => do let d ← AD.dotV W2v[k]! h; AD.add b2v[k]! d)
  let obj ← ppoStepObj out A a adv ret oldLogp vfCoef entCoef clipEps
  return (obj, (convWv, convBv, W1v, b1v, W2v, b2v))

structure CnnGrad where
  gConvW : Array (Array Float)
  gConvB : Array Float
  gW1 : Array (Array Float)
  gb1 : Array Float
  gW2 : Array (Array Float)
  gb2 : Array Float

def cnnZeroGrad (p : CnnPolicy) : CnnGrad :=
  { gConvW := p.convW.map (·.map (fun _ => 0.0)), gConvB := p.convB.map (fun _ => 0.0),
    gW1 := p.W1.map (·.map (fun _ => 0.0)), gb1 := p.b1.map (fun _ => 0.0),
    gW2 := p.W2.map (·.map (fun _ => 0.0)), gb2 := p.b2.map (fun _ => 0.0) }

def cnnGradAdd (a b : CnnGrad) : CnnGrad :=
  { gConvW := matAdd a.gConvW b.gConvW, gConvB := vecAdd a.gConvB b.gConvB,
    gW1 := matAdd a.gW1 b.gW1, gb1 := vecAdd a.gb1 b.gb1,
    gW2 := matAdd a.gW2 b.gW2, gb2 := vecAdd a.gb2 b.gb2 }

/-- Per-transition CNN-PPO gradient (conv + dense, via BPTT-free reverse mode). -/
def cnnGradPPO (p : CnnPolicy) (obs : Array Float) (a A : Nat)
    (adv ret oldLogp vfCoef entCoef clipEps : Float) : CnnGrad :=
  let build := cnnBuild p obs a A adv ret oldLogp vfCoef entCoef clipEps
  let (res, tp) := build.run AD.Tape.empty
  let (root, convWv, convBv, W1v, b1v, W2v, b2v) := res
  let g := AD.grads tp root
  { gConvW := convWv.map (·.map (fun v => g[v]!)), gConvB := convBv.map (fun v => g[v]!),
    gW1 := W1v.map (·.map (fun v => g[v]!)), gb1 := b1v.map (fun v => g[v]!),
    gW2 := W2v.map (·.map (fun v => g[v]!)), gb2 := b2v.map (fun v => g[v]!) }

/-- The single-transition objective primal (for finite-difference gradient checks). -/
def cnnObjPrimal (p : CnnPolicy) (obs : Array Float) (a A : Nat)
    (adv ret oldLogp vfCoef entCoef clipEps : Float) : Float :=
  let build := cnnBuild p obs a A adv ret oldLogp vfCoef entCoef clipEps
  let (res, tp) := build.run AD.Tape.empty
  AD.valueAt tp res.1

def cnnGradSumSq (gr : CnnGrad) : Float :=
  let sq := fun (acc x : Float) => acc + x * x
  let vsq := fun (acc : Float) (v : Array Float) => v.foldl sq acc
  let msq := fun (acc : Float) (m : Array (Array Float)) => m.foldl vsq acc
  msq (msq (msq (vsq (vsq (vsq 0.0 gr.gConvB) gr.gb1) gr.gb2) gr.gConvW) gr.gW1) gr.gW2

/-- Sum the CNN-PPO gradient over a minibatch of buffer indices. -/
def cnnGradIdx (p : CnnPolicy) (buf : Array Transition) (advN returns : Array Float)
    (idxs : Array Nat) (A : Nat) (vfCoef entCoef clipEps : Float) : CnnGrad := Id.run do
  let mut acc := cnnZeroGrad p
  for t in idxs do
    let tr : Transition := buf[t]!
    acc := cnnGradAdd acc (cnnGradPPO p tr.obs tr.action A advN[t]! returns[t]! tr.oldLogp vfCoef entCoef clipEps)
  return acc

/-- One minibatch ascent step (mean gradient `lr/|mb|`, global grad-norm clip). -/
def updateCnnIdx (p : CnnPolicy) (buf : Array Transition) (advN returns : Array Float)
    (idxs : Array Nat) (A : Nat) (lr maxGradNorm vfCoef entCoef clipEps : Float) : CnnPolicy :=
  let gr := cnnGradIdx p buf advN returns idxs A vfCoef entCoef clipEps
  let mb := Float.ofNat (max idxs.size 1)
  let meanNorm := Float.sqrt (cnnGradSumSq gr) / mb
  let cc := if maxGradNorm > 0.0 && meanNorm > maxGradNorm then maxGradNorm / meanNorm else 1.0
  let s := lr * cc / mb
  { p with
    convW := matAxpy s p.convW gr.gConvW, convB := vecAxpy s p.convB gr.gConvB,
    W1 := matAxpy s p.W1 gr.gW1, b1 := vecAxpy s p.b1 gr.gb1,
    W2 := matAxpy s p.W2 gr.gW2, b2 := vecAxpy s p.b2 gr.gb2 }

end Puffer.RL.NNTrain
