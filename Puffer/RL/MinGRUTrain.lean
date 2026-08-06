/-
MinGRU policy — gradient (reverse-mode AD ⇒ BPTT by construction), Muon step, recurrent
rollout, and a PufferLib-parity PER trainer. Wires PufferLib's DEFAULT network
(`DefaultEncoder → MinGRU(num_layers) → DefaultDecoder`, `Puffer/Net/MinGRU.lean`) into the
trainer, closing the last end-to-end-parity gap (the optimizer/advantage/loss/clipping are
already identical across the FFI trainers; this swaps the 2-layer MLP for the MinGRU net).

The gradient is built on the scalar reverse-mode AD tape (`Puffer/Float/AutoDiff.lean`): the
whole sequence forward + the summed per-timestep PPO objective are recorded on ONE tape, and a
single reverse sweep gives the gradient w.r.t. every weight — i.e. BPTT for free, correct by
construction (validated against finite differences, `verify-mingru-grad`). PufferLib's value-loss
clipping is `½·max((V−R)², (V_clip−R)²)` with `max(a,b)=a+relu(b−a)`; `_g`'s piecewise branch is
resolved on the node's primal value (`AD.valueAt`).
-/
import Puffer.Net.MinGRU
import Puffer.Float.AutoDiff
import Puffer.RL.FFITrain
import Puffer.RL.Dashboard

namespace Puffer.RL.NNTrain

open Puffer.Net.MinGRU (Weights sigmoid gAct stepForward seqForward)
open Puffer.RL (Env)
open Puffer.RL.Train (rngNext uniform01 softmax sampleCat)
open Puffer.FloatR

/-- Dimensions of a MinGRU `Weights` (inferred from the matrices). -/
@[inline] def mgH (w : Weights) : Nat := w.bEnc.size
@[inline] def mgObs (w : Weights) : Nat := if w.wEnc.size == 0 then 0 else (w.wEnc[0]!).size
@[inline] def mgLayers (w : Weights) : Nat := w.layers.size
@[inline] def mgA (w : Weights) : Nat := w.bDec.size

/-! ### Gradient of the summed per-timestep PPO objective over one sequence (BPTT via AD). -/

/-- Reverse-mode gradient (ascent form, in `Weights` shape) of the summed PPO objective over
    ONE sequence, w.r.t. every MinGRU weight. `advN`/`rets`/`oldvals` are per-timestep (the
    prio-weighted normalized advantage, the return target, and V at collection = the value-clip
    reference). The whole sequence recurrence lives on one tape, so a single reverse sweep is the
    truncated-BPTT gradient. `vfClip ≤ 0` disables value clipping. -/
def mingruGradSeq (w : Weights) (traj : Array Transition) (advN rets oldvals : Array Float)
    (numActions : Nat) (vfCoef entCoef clipEps vfClip : Float) : Weights := Id.run do
  let H := mgH w
  let numLayers := mgLayers w
  let build : AD.ADM (AD.V × (Array (Array AD.V) × Array AD.V × Array (Array (Array AD.V)) ×
      Array (Array AD.V) × Array AD.V × Array (Array AD.V) × Array AD.V)) := do
    let wEncV ← w.wEnc.mapM (fun row => row.mapM AD.leaf)
    let bEncV ← w.bEnc.mapM AD.leaf
    let layersV ← w.layers.mapM (fun L => L.mapM (fun row => row.mapM AD.leaf))
    let wDecV ← w.wDec.mapM (fun row => row.mapM AD.leaf)
    let bDecV ← w.bDec.mapM AD.leaf
    let wValV ← w.wVal.mapM (fun row => row.mapM AD.leaf)
    let bValV ← w.bVal.mapM AD.leaf
    let one ← AD.const 1.0
    let half ← AD.const 0.5
    let mut state : Array (Array AD.V) := #[]
    for _ in [0:numLayers] do
      let z ← (Array.range H).mapM (fun _ => AD.const 0.0)
      state := state.push z
    let mut obj ← AD.const 0.0
    for t in [0:traj.size] do
      let obs ← traj[t]!.obs.mapM AD.const
      let e ← (Array.range H).mapM (fun i => do AD.add (← AD.dotV wEncV[i]! obs) bEncV[i]!)
      let mut h := e
      let mut newState : Array (Array AD.V) := #[]
      for l in [0:numLayers] do
        let L := layersV[l]!
        let prev := state[l]!
        let y ← L.mapM (fun row => AD.dotV row h)
        let mut out : Array AD.V := #[]
        let mut hn : Array AD.V := #[]
        for j in [0:H] do
          let hid := y[j]!
          let gate := y[H + j]!
          let proj := y[2*H + j]!
          let zg ← AD.sigmoid gate
          let tp ← get
          let g ← if 0.0 ≤ AD.valueAt tp hid then AD.add hid half else AD.sigmoid hid
          let o ← AD.add (← AD.mul (← AD.sub one zg) prev[j]!) (← AD.mul zg g)
          let hg ← AD.sigmoid proj
          let hnj ← AD.add (← AD.mul hg o) (← AD.mul (← AD.sub one hg) h[j]!)
          out := out.push o
          hn := hn.push hnj
        h := hn
        newState := newState.push out
      state := newState
      let logits ← (Array.range numActions).mapM (fun k => do AD.add (← AD.dotV wDecV[k]! h) bDecV[k]!)
      let value ← AD.add (← AD.dotV wValV[0]! h) bValV[0]!
      let a := traj[t]!.action
      let lse ← AD.logSumExp logits
      let logpA ← AD.sub logits[a]! lse
      let ratio ← AD.exp (← AD.sub logpA (← AD.const traj[t]!.oldLogp))
      let t1 ← AD.scale advN[t]! ratio
      let t2 ← AD.scale advN[t]! (← AD.clampC (1.0 - clipEps) (1.0 + clipEps) ratio)
      let surr ← AD.minV t1 t2
      let mut entSum ← AD.const 0.0
      for k in [0:numActions] do
        let pk ← AD.exp (← AD.sub logits[k]! lse)
        entSum ← AD.add entSum (← AD.mul pk logits[k]!)
      let ent ← AD.sub lse entSum
      let dv ← AD.sub value (← AD.const rets[t]!)
      let uu ← AD.mul dv dv
      let vloss ← if vfClip > 0.0 then do
          let vold ← AD.const oldvals[t]!
          let vclip ← AD.add vold (← AD.clampC (-vfClip) vfClip (← AD.sub value vold))
          let dvc ← AD.sub vclip (← AD.const rets[t]!)
          let cc ← AD.mul dvc dvc
          AD.add uu (← AD.relu (← AD.sub cc uu))       -- max(uu, cc) = uu + relu(cc − uu)
        else pure uu
      let objt ← AD.sub (← AD.add surr (← AD.scale entCoef ent)) (← AD.scale (vfCoef * 0.5) vloss)
      obj ← AD.add obj objt
      if traj[t]!.terminal then                          -- reset state at an episode boundary (no BPTT across)
        let mut zs : Array (Array AD.V) := #[]
        for _ in [0:numLayers] do
          zs := zs.push (← (Array.range H).mapM (fun _ => AD.const 0.0))
        state := zs
    return (obj, (wEncV, bEncV, layersV, wDecV, bDecV, wValV, bValV))
  let (res, tape) := build.run AD.Tape.empty
  let (root, wEncV, bEncV, layersV, wDecV, bDecV, wValV, bValV) :=
    (res.1, res.2.1, res.2.2.1, res.2.2.2.1, res.2.2.2.2.1, res.2.2.2.2.2.1, res.2.2.2.2.2.2.1, res.2.2.2.2.2.2.2)
  let g := AD.grads tape root
  return {
    wEnc := wEncV.map (fun row => row.map (fun v => g[v]!)),
    bEnc := bEncV.map (fun v => g[v]!),
    layers := layersV.map (fun L => L.map (fun row => row.map (fun v => g[v]!))),
    wDec := wDecV.map (fun row => row.map (fun v => g[v]!)),
    bDec := bDecV.map (fun v => g[v]!),
    wVal := wValV.map (fun row => row.map (fun v => g[v]!)),
    bVal := bValV.map (fun v => g[v]!) }

/-! ### Multi-discrete head: the same MinGRU core, `W = Σ headSizes` logits + 1 value.

A multi-discrete policy is exactly the single-discrete one with a WIDER decoder: the head emits
`W = Σ_h headSizes[h]` logits, sliced into `K` contiguous per-head blocks, each softmaxed on its own.
The joint log-prob is `Σ_h log p_h(a_h)` and the entropy `Σ_h H_h`, so ONE PPO clip acts on the joint
ratio and the gradient decomposes per head — the exact convention of the multi-discrete MLP path
(`k_ppo_dout_md` / `trainPluginEnvMD`). Nothing in the recurrence, the weight layout (`flattenMG` with
`numActions := W`), the Muon or the BPTT changes; `mgA w` is `W`, not the number of actions. -/

/-- Reverse-mode (AD-BPTT) gradient of the summed MULTI-DISCRETE PPO objective over ONE sequence — the
    oracle twin of the `k_mg_ppo_b_md` GPU head. Inputs are flat/per-timestep: `obsSeq[t]` the observation,
    `acts[t][h]` the action of head `h`, `oldlps[t]` the JOINT old log-prob, plus the usual
    `advN`/`rets`/`oldvals`/`terms`. `headSizes` gives the `K` head widths (`W = Σ headSizes` decoder rows).
    `vfClip ≤ 0` disables value clipping. `K = 1` reduces exactly to `mingruGradSeq`. -/
def mingruGradSeqMD (w : Weights) (obsSeq : Array (Array Float)) (acts : Array (Array Nat))
    (oldlps advN rets oldvals : Array Float) (terms : Array Bool) (headSizes : Array Nat)
    (vfCoef entCoef clipEps vfClip : Float) : Weights := Id.run do
  let H := mgH w
  let numLayers := mgLayers w
  let Wtot := headSizes.foldl (·+·) 0
  let K := headSizes.size
  let build : AD.ADM (AD.V × (Array (Array AD.V) × Array AD.V × Array (Array (Array AD.V)) ×
      Array (Array AD.V) × Array AD.V × Array (Array AD.V) × Array AD.V)) := do
    let wEncV ← w.wEnc.mapM (fun row => row.mapM AD.leaf)
    let bEncV ← w.bEnc.mapM AD.leaf
    let layersV ← w.layers.mapM (fun L => L.mapM (fun row => row.mapM AD.leaf))
    let wDecV ← w.wDec.mapM (fun row => row.mapM AD.leaf)
    let bDecV ← w.bDec.mapM AD.leaf
    let wValV ← w.wVal.mapM (fun row => row.mapM AD.leaf)
    let bValV ← w.bVal.mapM AD.leaf
    let one ← AD.const 1.0
    let half ← AD.const 0.5
    let mut state : Array (Array AD.V) := #[]
    for _ in [0:numLayers] do
      let z ← (Array.range H).mapM (fun _ => AD.const 0.0)
      state := state.push z
    let mut obj ← AD.const 0.0
    for t in [0:obsSeq.size] do
      let obs ← obsSeq[t]!.mapM AD.const
      let e ← (Array.range H).mapM (fun i => do AD.add (← AD.dotV wEncV[i]! obs) bEncV[i]!)
      let mut h := e
      let mut newState : Array (Array AD.V) := #[]
      for l in [0:numLayers] do
        let L := layersV[l]!
        let prev := state[l]!
        let y ← L.mapM (fun row => AD.dotV row h)
        let mut out : Array AD.V := #[]
        let mut hn : Array AD.V := #[]
        for j in [0:H] do
          let hid := y[j]!
          let gate := y[H + j]!
          let proj := y[2*H + j]!
          let zg ← AD.sigmoid gate
          let tp ← get
          let g ← if 0.0 ≤ AD.valueAt tp hid then AD.add hid half else AD.sigmoid hid
          let o ← AD.add (← AD.mul (← AD.sub one zg) prev[j]!) (← AD.mul zg g)
          let hg ← AD.sigmoid proj
          let hnj ← AD.add (← AD.mul hg o) (← AD.mul (← AD.sub one hg) h[j]!)
          out := out.push o
          hn := hn.push hnj
        h := hn
        newState := newState.push out
      state := newState
      let logits ← (Array.range Wtot).mapM (fun k => do AD.add (← AD.dotV wDecV[k]! h) bDecV[k]!)
      let value ← AD.add (← AD.dotV wValV[0]! h) bValV[0]!
      -- per-head softmax: joint log-prob = Σ_h log p_h(a_h), entropy = Σ_h H_h
      let mut jointLogp ← AD.const 0.0
      let mut ent ← AD.const 0.0
      let mut off := 0
      for hh in [0:K] do
        let sz := headSizes[hh]!
        let hl := (Array.range sz).map (fun k => logits[off + k]!)
        let lse ← AD.logSumExp hl
        jointLogp ← AD.add jointLogp (← AD.sub hl[(acts[t]!)[hh]!]! lse)
        let mut entSum ← AD.const 0.0
        for k in [0:sz] do
          let pk ← AD.exp (← AD.sub hl[k]! lse)
          entSum ← AD.add entSum (← AD.mul pk hl[k]!)
        ent ← AD.add ent (← AD.sub lse entSum)
        off := off + sz
      let ratio ← AD.exp (← AD.sub jointLogp (← AD.const oldlps[t]!))
      let t1 ← AD.scale advN[t]! ratio
      let t2 ← AD.scale advN[t]! (← AD.clampC (1.0 - clipEps) (1.0 + clipEps) ratio)
      let surr ← AD.minV t1 t2
      let dv ← AD.sub value (← AD.const rets[t]!)
      let uu ← AD.mul dv dv
      let vloss ← if vfClip > 0.0 then do
          let vold ← AD.const oldvals[t]!
          let vclip ← AD.add vold (← AD.clampC (-vfClip) vfClip (← AD.sub value vold))
          let dvc ← AD.sub vclip (← AD.const rets[t]!)
          let cc ← AD.mul dvc dvc
          AD.add uu (← AD.relu (← AD.sub cc uu))
        else pure uu
      let objt ← AD.sub (← AD.add surr (← AD.scale entCoef ent)) (← AD.scale (vfCoef * 0.5) vloss)
      obj ← AD.add obj objt
      if terms[t]! then                                  -- episode boundary: no BPTT across it
        let mut zs : Array (Array AD.V) := #[]
        for _ in [0:numLayers] do
          zs := zs.push (← (Array.range H).mapM (fun _ => AD.const 0.0))
        state := zs
    return (obj, (wEncV, bEncV, layersV, wDecV, bDecV, wValV, bValV))
  let (res, tape) := build.run AD.Tape.empty
  let (root, wEncV, bEncV, layersV, wDecV, bDecV, wValV, bValV) :=
    (res.1, res.2.1, res.2.2.1, res.2.2.2.1, res.2.2.2.2.1, res.2.2.2.2.2.1, res.2.2.2.2.2.2.1, res.2.2.2.2.2.2.2)
  let g := AD.grads tape root
  return {
    wEnc := wEncV.map (fun row => row.map (fun v => g[v]!)),
    bEnc := bEncV.map (fun v => g[v]!),
    layers := layersV.map (fun L => L.map (fun row => row.map (fun v => g[v]!))),
    wDec := wDecV.map (fun row => row.map (fun v => g[v]!)),
    bDec := bDecV.map (fun v => g[v]!),
    wVal := wValV.map (fun row => row.map (fun v => g[v]!)),
    bVal := bValV.map (fun v => g[v]!) }

/-! ### Muon over the MinGRU weights (every 2D matrix orthogonalized; biases momentum-only). -/

/-- Muon momentum buffers for a MinGRU `Weights`. -/
structure MuonStateMG where
  mWEnc : Array (Array Float)
  mBEnc : Array Float
  mLayers : Array (Array (Array Float))
  mWDec : Array (Array Float)
  mBDec : Array Float
  mWVal : Array (Array Float)
  mBVal : Array Float

def MuonStateMG.zeros (w : Weights) : MuonStateMG :=
  { mWEnc := w.wEnc.map (·.map (fun _ => 0.0)), mBEnc := w.bEnc.map (fun _ => 0.0),
    mLayers := w.layers.map (·.map (·.map (fun _ => 0.0))),
    mWDec := w.wDec.map (·.map (fun _ => 0.0)), mBDec := w.bDec.map (fun _ => 0.0),
    mWVal := w.wVal.map (·.map (fun _ => 0.0)), mBVal := w.bVal.map (fun _ => 0.0) }

/-- One Muon step for every MinGRU tensor: the encoder / each MinGRU layer (`3H×H`) / decoder /
    value matrices are orthogonalized (`Muon.stepMat`), the three biases take the momentum-only
    step (`Muon.stepVec`). `g` is the ascent gradient in `Weights` shape. -/
def applyMuonMG (w : Weights) (st : MuonStateMG) (g : Weights) (lr wd mu eps : Float) :
    Weights × MuonStateMG :=
  let (nWEnc, mWEnc) := Muon.stepMat w.wEnc g.wEnc st.mWEnc lr wd mu eps
  let (nBEnc, mBEnc) := Muon.stepVec w.bEnc g.bEnc st.mBEnc lr wd mu
  let layerRes := (Array.range w.layers.size).map (fun i =>
    Muon.stepMat w.layers[i]! g.layers[i]! st.mLayers[i]! lr wd mu eps)
  let nLayers := layerRes.map (·.1)
  let mLayers := layerRes.map (·.2)
  let (nWDec, mWDec) := Muon.stepMat w.wDec g.wDec st.mWDec lr wd mu eps
  let (nBDec, mBDec) := Muon.stepVec w.bDec g.bDec st.mBDec lr wd mu
  let (nWVal, mWVal) := Muon.stepMat w.wVal g.wVal st.mWVal lr wd mu eps
  let (nBVal, mBVal) := Muon.stepVec w.bVal g.bVal st.mBVal lr wd mu
  ({ wEnc := nWEnc, bEnc := nBEnc, layers := nLayers, wDec := nWDec, bDec := nBDec, wVal := nWVal, bVal := nBVal },
   { mWEnc := mWEnc, mBEnc := mBEnc, mLayers := mLayers, mWDec := mWDec, mBDec := mBDec, mWVal := mWVal, mBVal := mBVal })

/-- Flatten all MinGRU weights to one `FloatArray` (`weight_norm` / grad-norm bookkeeping). -/
def flattenMG (w : Weights) : FloatArray := Id.run do
  let mut a : FloatArray := FloatArray.emptyWithCapacity 0
  for row in w.wEnc do for x in row do a := a.push x
  for x in w.bEnc do a := a.push x
  for L in w.layers do for row in L do for x in row do a := a.push x
  for row in w.wDec do for x in row do a := a.push x
  for x in w.bDec do a := a.push x
  for row in w.wVal do for x in row do a := a.push x
  for x in w.bVal do a := a.push x
  return a

/-- Native MinGRU BPTT gradient (flat, `flattenMG` layout) via the C kernel — the fast twin of
    `flattenMG (mingruGradSeq …)`, validated against it by `mingruKernelCheck`. -/
def mingruGradSeqFFI (w : Weights) (traj : Array Transition) (advN rets oldvals : Array Float)
    (numActions : Nat) (vfCoef entCoef clipEps vfClip : Float) : FloatArray :=
  let params := flattenMG w
  let (obsSeq, acts, olps, terms) := mkSeqArrays traj
  Puffer.Float.FFI.mingruPPOGradSeqFFI params obsSeq acts (FloatArray.mk advN) (FloatArray.mk rets)
    olps terms (FloatArray.mk oldvals) (USize.ofNat traj.size) (USize.ofNat (mgH w)) (USize.ofNat (mgObs w))
    (USize.ofNat (mgLayers w)) (USize.ofNat numActions) vfCoef entCoef clipEps vfClip

/-- L2 norm over a `Weights` gradient (for `clip_grad_norm_`). -/
def gradNormMG (g : Weights) : Float :=
  let fa := flattenMG g
  Id.run do
    let mut s := 0.0
    for i in [0:fa.size] do s := s + fa[i]! * fa[i]!
    return Float.sqrt s

/-- Random `[-scale,scale)` MinGRU weights (biases zero), splitmix64 from `seed`. -/
def initMinGRU (obsSize H numLayers numActions : Nat) (seed : UInt64) : Weights × UInt64 := Id.run do
  let rmat := fun (rows cols : Nat) (rng0 : UInt64) (scale : Float) => Id.run do
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
  let mut rng := seed
  -- PufferLib/PyTorch nn.Linear default init: uniform[-1/√fan_in, 1/√fan_in]. Scale-invariant, so it
  -- stays well-conditioned at hidden 128 × 4 layers (a fixed scale like 0.2 saturates the deep net → a
  -- near-deterministic, unexplorative initial policy that never learns).
  let inv := fun (n : Nat) => 1.0 / Float.sqrt (Float.ofNat n)
  -- ENCODER gain √2, matching PufferLib's `puf_kaiming_init(&wt, std::sqrt(2.0f), …)` for the encoder
  -- (src/models.cu:432); their decoder uses gain 1.0 (:502), which is what `inv` alone gives. Ours had
  -- gain 1.0 on the encoder too — a real parity gap, and the same fan-in/gain class already fixed in
  -- `initMLP`. Measured consequence: our untrained joint entropy on moba spans 6.508–6.988 nats across
  -- seeds where PufferLib's spans 6.9429–6.9448 (essentially exactly uniform), and our runs then have
  -- no entropy floor at all — 22/40 finish below PufferLib's 40-seed minimum and ~10% collapse, where
  -- theirs collapse 0/40.
  let (wEnc, r1) := rmat H obsSize rng (Float.sqrt 2.0 * inv obsSize); rng := r1
  let mut layers : Array (Array (Array Float)) := #[]
  for _ in [0:numLayers] do
    let (L, r) := rmat (3*H) H rng (inv H); rng := r
    layers := layers.push L
  let (wDec, r3) := rmat numActions H rng (inv H); rng := r3
  let (wVal, r5) := rmat 1 H rng (inv H); rng := r5
  return ({ wEnc := wEnc, bEnc := Array.replicate H 0.0, layers := layers,
            wDec := wDec, bDec := Array.replicate numActions 0.0,
            wVal := wVal, bVal := #[0.0] }, rng)

/-! ### Gradient validation vs finite differences (the project's AD-oracle discipline). -/

/-- Primal summed PPO objective over one sequence (plain `Float`), the value `mingruGradSeq`
    differentiates — for the finite-difference check. -/
def mingruObjSeq (w : Weights) (traj : Array Transition) (advN rets oldvals : Array Float)
    (numActions : Nat) (vfCoef entCoef clipEps vfClip : Float) : Float := Id.run do
  let H := mgH w; let numLayers := mgLayers w
  let mut state : Array (Array Float) := (Array.range numLayers).map (fun _ => Array.replicate H 0.0)
  let mut obj := 0.0
  for t in [0:traj.size] do
    let (logits, value, newState) := stepForward w H traj[t]!.obs state
    state := newState
    let a := traj[t]!.action
    let mut sumexp := 0.0
    for k in [0:numActions] do sumexp := sumexp + Float.exp logits[k]!
    let lse := Float.log sumexp
    let ratio := Float.exp ((logits[a]! - lse) - traj[t]!.oldLogp)
    let lo := 1.0 - clipEps; let hi := 1.0 + clipEps
    let ratioC := if ratio < lo then lo else if ratio > hi then hi else ratio
    let s1 := advN[t]! * ratio; let s2 := advN[t]! * ratioC
    let surr := if s1 ≤ s2 then s1 else s2
    let mut entSum := 0.0
    for k in [0:numActions] do entSum := entSum + Float.exp (logits[k]! - lse) * logits[k]!
    let ent := lse - entSum
    let dv := value - rets[t]!; let uu := dv * dv
    let vl := if vfClip > 0.0 then
        let d := value - oldvals[t]!
        let cl := if d < -vfClip then -vfClip else if d > vfClip then vfClip else d
        let dvc := (oldvals[t]! + cl) - rets[t]!; let cc := dvc * dvc
        if uu ≥ cc then uu else cc
      else uu
    obj := obj + surr + entCoef * ent - vfCoef * 0.5 * vl
    if traj[t]!.terminal then
      state := (Array.range numLayers).map (fun _ => Array.replicate H 0.0)
  return obj

/-- Read a `rows×cols` matrix from `fa` starting at `off`; returns `(matrix, nextOff)`. -/
def readMat (fa : FloatArray) (off rows cols : Nat) : Array (Array Float) × Nat := Id.run do
  let mut o := off
  let mut m : Array (Array Float) := #[]
  for _ in [0:rows] do
    let mut row : Array Float := #[]
    for _ in [0:cols] do
      row := row.push fa[o]!
      o := o + 1
    m := m.push row
  return (m, o)

/-- Read a length-`n` vector from `fa` at `off`; returns `(vector, nextOff)`. -/
def readVec (fa : FloatArray) (off n : Nat) : Array Float × Nat := Id.run do
  let mut o := off
  let mut v : Array Float := #[]
  for _ in [0:n] do
    v := v.push fa[o]!
    o := o + 1
  return (v, o)

/-- Rebuild `Weights` from a flat `FloatArray` (the `flattenMG` layout), using `template`'s dims. -/
def unflattenMG (template : Weights) (fa : FloatArray) : Weights := Id.run do
  let H := mgH template; let obsSize := mgObs template; let numLayers := mgLayers template; let A := mgA template
  let (wEnc, o1) := readMat fa 0 H obsSize
  let (bEnc, o2) := readVec fa o1 H
  let mut off := o2
  let mut layers : Array (Array (Array Float)) := #[]
  for _ in [0:numLayers] do
    let (L, o) := readMat fa off (3*H) H
    layers := layers.push L
    off := o
  let (wDec, o3) := readMat fa off A H
  let (bDec, o4) := readVec fa o3 A
  let (wVal, o5) := readMat fa o4 1 H
  let (bVal, _) := readVec fa o5 1
  return { wEnc, bEnc, layers, wDec, bDec, wVal, bVal }

/-- AD-vs-finite-difference gradient check on a fixed small MinGRU + sequence: returns
    `(max|Δ|, max relΔ)` over all parameters (should be at finite-difference roundoff scale). -/
def mingruGradCheck : Float × Float := Id.run do
  let numActions := 3
  let (w, _) := initMinGRU 4 4 2 numActions 0xABCDEF
  -- a fixed 3-step sequence (obs/action/oldLogp/reward arbitrary; adv/ret/oldval arbitrary)
  let mkObs := fun (s : Float) => #[s, -s, 0.5*s, 1.0-s]
  let traj : Array Transition := #[
    { obs := mkObs 0.3, action := 0, reward := 1.0, value := 0.4, oldLogp := -1.0, terminal := false },
    { obs := mkObs (-0.6), action := 2, reward := 0.0, value := 0.2, oldLogp := -1.2, terminal := true },
    { obs := mkObs 0.9, action := 1, reward := 1.0, value := 0.1, oldLogp := -0.9, terminal := false }]
  let advN : Array Float := #[0.7, -0.4, 0.5]
  let rets : Array Float := #[0.9, 0.3, 0.6]
  let oldvals : Array Float := #[0.4, 0.2, 0.1]
  let vfCoef := 0.5; let entCoef := 0.01; let clipEps := 0.2; let vfClip := 0.2
  let gAD := flattenMG (mingruGradSeq w traj advN rets oldvals numActions vfCoef entCoef clipEps vfClip)
  let base := flattenMG w
  let eps := 1.0e-5
  let mut maxAbs := 0.0; let mut maxRel := 0.0
  for i in [0:base.size] do
    let wp := unflattenMG w (base.set! i (base[i]! + eps))
    let wm := unflattenMG w (base.set! i (base[i]! - eps))
    let fd := (mingruObjSeq wp traj advN rets oldvals numActions vfCoef entCoef clipEps vfClip
             - mingruObjSeq wm traj advN rets oldvals numActions vfCoef entCoef clipEps vfClip) / (2.0 * eps)
    let d := Float.abs (gAD[i]! - fd)
    maxAbs := if d > maxAbs then d else maxAbs
    let rel := d / (Float.abs fd + 1.0e-12)
    maxRel := if rel > maxRel then rel else maxRel
  return (maxAbs, maxRel)

/-- Primal summed MULTI-DISCRETE PPO objective over one sequence (plain `Float`) — the value
    `mingruGradSeqMD` differentiates, for the finite-difference check. -/
def mingruObjSeqMD (w : Weights) (obsSeq : Array (Array Float)) (acts : Array (Array Nat))
    (oldlps advN rets oldvals : Array Float) (terms : Array Bool) (headSizes : Array Nat)
    (vfCoef entCoef clipEps vfClip : Float) : Float := Id.run do
  let H := mgH w; let numLayers := mgLayers w; let K := headSizes.size
  let mut state : Array (Array Float) := (Array.range numLayers).map (fun _ => Array.replicate H 0.0)
  let mut obj := 0.0
  for t in [0:obsSeq.size] do
    let (logits, value, newState) := stepForward w H obsSeq[t]! state
    state := newState
    let mut jointLogp := 0.0
    let mut ent := 0.0
    let mut off := 0
    for hh in [0:K] do
      let sz := headSizes[hh]!
      let mut sumexp := 0.0
      for k in [0:sz] do sumexp := sumexp + Float.exp logits[off+k]!
      let lse := Float.log sumexp
      jointLogp := jointLogp + (logits[off + (acts[t]!)[hh]!]! - lse)
      let mut entSum := 0.0
      for k in [0:sz] do entSum := entSum + Float.exp (logits[off+k]! - lse) * logits[off+k]!
      ent := ent + (lse - entSum)
      off := off + sz
    let ratio := Float.exp (jointLogp - oldlps[t]!)
    let lo := 1.0 - clipEps; let hi := 1.0 + clipEps
    let ratioC := if ratio < lo then lo else if ratio > hi then hi else ratio
    let s1 := advN[t]! * ratio; let s2 := advN[t]! * ratioC
    let surr := if s1 ≤ s2 then s1 else s2
    let dv := value - rets[t]!; let uu := dv * dv
    let vl := if vfClip > 0.0 then
        let d := value - oldvals[t]!
        let cl := if d < -vfClip then -vfClip else if d > vfClip then vfClip else d
        let dvc := (oldvals[t]! + cl) - rets[t]!; let cc := dvc * dvc
        if uu ≥ cc then uu else cc
      else uu
    obj := obj + surr + entCoef * ent - vfCoef * 0.5 * vl
    if terms[t]! then
      state := (Array.range numLayers).map (fun _ => Array.replicate H 0.0)
  return obj

/-- A small fixed multi-discrete MinGRU test case (`K=2` heads `[3,4]` ⇒ `W=7` decoder rows, 3 steps with a
    mid-sequence terminal) — shared by the AD-vs-finite-difference check and the GPU kernel check so both
    differentiate exactly the same problem. -/
def mdCase : Weights × Array (Array Float) × Array (Array Nat) × Array Float × Array Float ×
    Array Float × Array Float × Array Bool × Array Nat :=
  let headSizes : Array Nat := #[3, 4]
  let Wtot := headSizes.foldl (·+·) 0
  let (w, _) := initMinGRU 4 4 2 Wtot 0xABCDEF
  let mkObs := fun (s : Float) => #[s, -s, 0.5*s, 1.0-s]
  let obsSeq := #[mkObs 0.3, mkObs (-0.6), mkObs 0.9]
  let acts : Array (Array Nat) := #[#[0, 2], #[2, 0], #[1, 3]]
  -- the joint log-prob at this init is ≈ −log3 − log4 = −2.485, so oldLogp −2.5 puts t=1,2 INSIDE the PPO
  -- clip window (full policy gradient) while t=0 (−2.9 ⇒ ratio ≈ 1.5 > 1+ε with adv > 0) lands in the
  -- clipped branch whose policy gradient is exactly zero — both branches get differentiated.
  let oldlps : Array Float := #[-2.9, -2.5, -2.5]
  let advN : Array Float := #[0.7, -0.4, 0.5]
  let rets : Array Float := #[0.9, 0.3, 0.6]
  let oldvals : Array Float := #[0.4, 0.2, 0.1]
  let terms : Array Bool := #[false, true, false]
  (w, obsSeq, acts, oldlps, advN, rets, oldvals, terms, headSizes)

/-- AD-vs-finite-difference gradient check for the MULTI-DISCRETE MinGRU head on `mdCase`: returns
    `(max|Δ|, max relΔ)` over all parameters (finite-difference roundoff scale ⇒ the per-head
    softmax/joint-ratio gradient is correct). Pure CPU — no GPU involved. -/
def mingruMDGradCheck : Float × Float := Id.run do
  let (w, obsSeq, acts, oldlps, advN, rets, oldvals, terms, headSizes) := mdCase
  let vfCoef := 0.5; let entCoef := 0.01; let clipEps := 0.2; let vfClip := 0.2
  let gAD := flattenMG (mingruGradSeqMD w obsSeq acts oldlps advN rets oldvals terms headSizes
                          vfCoef entCoef clipEps vfClip)
  let base := flattenMG w
  let eps := 1.0e-5
  let mut maxAbs := 0.0; let mut maxRel := 0.0
  for i in [0:base.size] do
    let wp := unflattenMG w (base.set! i (base[i]! + eps))
    let wm := unflattenMG w (base.set! i (base[i]! - eps))
    let fd := (mingruObjSeqMD wp obsSeq acts oldlps advN rets oldvals terms headSizes vfCoef entCoef clipEps vfClip
             - mingruObjSeqMD wm obsSeq acts oldlps advN rets oldvals terms headSizes vfCoef entCoef clipEps vfClip) / (2.0 * eps)
    let d := Float.abs (gAD[i]! - fd)
    maxAbs := if d > maxAbs then d else maxAbs
    let rel := d / (Float.abs fd + 1.0e-12)
    maxRel := if rel > maxRel then rel else maxRel
  return (maxAbs, maxRel)

/-- Native C kernel vs the AD-tape oracle (`mingruGradSeq`) on the same fixed case: `max|Δ|`
    over all parameters (both f64; tolerance-scale, not bit-exact, due to summation order). -/
def mingruKernelCheck : Float := Id.run do
  let numActions := 3
  let (w, _) := initMinGRU 4 4 2 numActions 0xABCDEF
  let mkObs := fun (s : Float) => #[s, -s, 0.5*s, 1.0-s]
  let traj : Array Transition := #[
    { obs := mkObs 0.3, action := 0, reward := 1.0, value := 0.4, oldLogp := -1.0, terminal := false },
    { obs := mkObs (-0.6), action := 2, reward := 0.0, value := 0.2, oldLogp := -1.2, terminal := true },
    { obs := mkObs 0.9, action := 1, reward := 1.0, value := 0.1, oldLogp := -0.9, terminal := false }]
  let advN : Array Float := #[0.7, -0.4, 0.5]
  let rets : Array Float := #[0.9, 0.3, 0.6]
  let oldvals : Array Float := #[0.4, 0.2, 0.1]
  let vfCoef := 0.5; let entCoef := 0.01; let clipEps := 0.2; let vfClip := 0.2
  let gAD := flattenMG (mingruGradSeq w traj advN rets oldvals numActions vfCoef entCoef clipEps vfClip)
  let gC := mingruGradSeqFFI w traj advN rets oldvals numActions vfCoef entCoef clipEps vfClip
  let mut maxAbs := 0.0
  for i in [0:gAD.size] do
    let d := Float.abs (gAD[i]! - gC[i]!)
    maxAbs := if d > maxAbs then d else maxAbs
  return maxAbs

/-! ### Recurrent rollout + the PufferLib-parity PER trainer. -/

/-- One `forward_eval` step: `(softmax logits, value, new per-layer state)`. -/
@[inline] def mingruProbsValueState (w : Weights) (H : Nat) (obs : Array Float) (state : Array (Array Float)) :
    Array Float × Float × Array (Array Float) :=
  let (logits, value, newState) := stepForward w H obs state
  (softmax logits, value, newState)

/-- Roll ONE env for `horizon` steps from a ZERO MinGRU state (the per-segment `forward_train`
    convention), sampling actions and resetting the recurrent state on episode boundaries. -/
def mingruSegmentRollout {S : Type} (env : Env S) (w : Weights) (H numLayers horizon : Nat)
    (s0 : S) (rng0 : UInt64) : Array Transition × S × UInt64 × Array Float := Id.run do
  let mut st := s0; let mut rng := rng0
  let zero : Array (Array Float) := (Array.range numLayers).map (fun _ => Array.replicate H 0.0)
  let mut state := zero
  let mut traj : Array Transition := #[]
  let mut epReturns : Array Float := #[]; let mut epRet := 0.0
  for _ in [0:horizon] do
    let obs := env.observe st
    let (probs, v, newState) := mingruProbsValueState w H obs state
    let (word, rng') := rngNext rng; rng := rng'
    let a := sampleCat probs (uniform01 word)
    let (st', r, term) := env.step st a
    traj := traj.push { obs := obs, action := a, reward := r, value := v, oldLogp := Float.log probs[a]!, terminal := term }
    epRet := epRet + r
    if term then
      epReturns := epReturns.push epRet; epRet := 0.0
      state := zero
      let (sReset, rng'') := env.reset rng; rng := rng''; st := sReset
    else
      state := newState; st := st'
  return (traj, st, rng, epReturns)

/-- Forward the MinGRU over a sequence from ZERO state under the CURRENT policy, returning the
    per-timestep `(new_logp of the taken action, new_value)` — for iterating the ratio/value
    buffers in prioritized replay (resets state on terminals, matching the rollout + gradient). -/
def mingruSeqForwardRV (w : Weights) (traj : Array Transition) (H numLayers _numActions : Nat) :
    Array Float × Array Float := Id.run do
  let zero : Array (Array Float) := (Array.range numLayers).map (fun _ => Array.replicate H 0.0)
  let mut state := zero
  let mut lps : Array Float := #[]; let mut vs : Array Float := #[]
  for t in [0:traj.size] do
    let (probs, v, newState) := mingruProbsValueState w H traj[t]!.obs state
    lps := lps.push (Float.log probs[traj[t]!.action]!)
    vs := vs.push v
    if traj[t]!.terminal then state := zero else state := newState
  return (lps, vs)

/-- One Muon optimizer step on the FLAT MinGRU params (`flattenMG` layout), weights + momentum kept as flat
    `FloatArray`s: GPU Muon (`cudaMuonStepMatFFI`, matching `Muon.stepMat`) for each weight MATRIX, Nesterov
    (`Muon.stepVec`) for each bias. Avoids the O(P) flatten/unflatten + the pure-Lean Newton–Schulz that
    dominate at PufferLib's hidden 128 (P≈200k). `gFlat` is the (already mean-scaled + clipped) gradient. -/
def muonStepFlatMG (wFlat momFlat gFlat : FloatArray) (H D L A : Nat) (lr wd mu eps : Float) :
    FloatArray × FloatArray := Id.run do
  let u := USize.ofNat; let mk := FloatArray.mk; let slice := Puffer.Float.FFI.sliceFFI
  let mut nw : Array Float := #[]; let mut nm : Array Float := #[]; let mut off := 0
  -- segments in flattenMG order, tagged: true = matrix (GPU Muon), false = bias (Nesterov)
  let segs : Array (Bool × Nat × Nat) :=
    #[(true, H, D), (false, H, 1)]
    ++ (Array.replicate L (true, 3*H, H))
    ++ #[(true, A, H), (false, A, 1), (true, 1, H), (false, 1, 1)]
  for seg in segs do
    let (isMat, rows, cols) := seg
    let n := rows*cols
    if isMat then
      let res := Puffer.Float.CUDA.cudaMuonStepMatFFI (slice wFlat (u off) (u n)) (slice gFlat (u off) (u n))
                   (slice momFlat (u off) (u n)) (u rows) (u cols) lr wd mu eps
      nw := nw ++ (Array.range n).map (fun i => res[i]!)
      nm := nm ++ (Array.range n).map (fun i => res[n+i]!)
    else
      let mm := (Array.range n).map (fun i => mu * momFlat[off+i]! + gFlat[off+i]!)
      nw := nw ++ (Array.range n).map (fun i => wFlat[off+i]! * (1.0 - lr*wd) + lr * (gFlat[off+i]! + mu * mm[i]!))
      nm := nm ++ mm
    off := off + n
  return (mk nw, mk nm)

/-- Host twin of the on-GPU gradclip (`k_gclip_norm` + `k_gclip_scale`): `gnorm = √Σ(g·sc)²`,
    `cc = (if maxNorm > 0 ∧ gnorm > maxNorm then maxNorm/gnorm else 1)·sc`, `out[i] = g[i]·cc`.

    Only used on the HOST-GRADIENT FALLBACK of the MinGRU trainer — when the BPTT could not leave its
    summed gradient device-resident (bg 70 unavailable) it returns the gradient in the array instead, and
    the resident Muon then wants an already-clipped host gradient. The resident fast path never calls
    this (the identical formula runs on-GPU in tree order), so it cannot perturb the bit-identical path;
    the two differ by the usual ~1 ulp of summation order, which is fine for a degraded fallback. -/
def gradClipMG (g : FloatArray) (P : Nat) (maxNorm sc : Float) : FloatArray := Id.run do
  let mut ss := 0.0
  for i in [0:P] do
    let x := g[i]! * sc
    ss := ss + x * x
  let gnorm := ss.sqrt
  let cc := (if maxNorm > 0.0 && gnorm > maxNorm then maxNorm / gnorm else 1.0) * sc
  let mut out := FloatArray.emptyWithCapacity P
  for i in [0:P] do out := out.push (g[i]! * cc)
  return out

/-- **MinGRU recurrent PLUGIN trainer** — PufferLib's DEFAULT network (`[torch] network = MinGRU`) on the
    runtime C-plugin path. The architecture matches PufferLib exactly: `initMinGRU` builds a Linear encoder
    (obs→H, PufferLib's `DefaultEncoder`), `numLayers` MinGRU layers (`[policy] num_layers`), and linear
    action + value heads (`DefaultDecoder`). Per update: drive the plugin (`envReset`/`envStep`) for `T`
    steps threading each env's `numLayers·H` recurrent state (reset to 0 at episode boundaries), collect
    per-env sequences, per-segment GAE (batch-normalized advantages), then `epochs × numMB` minibatch BPTT
    ascent — each minibatch SUMS its sequences' gradients (the fast parallel-scan C twin `mingruGradSeqFFI`)
    into one grad-norm-clipped Muon step (`applyMuonMG`). Single-discrete. No env-specific code: the env is
    dlopen'd and stepped through the C ABI, the policy is PufferLib's. -/
def trainPluginEnvMinGRU (name config : String)
    (hidden numLayers numEnvs horizon totalTimesteps epochs numMB minibatchSize : Nat)
    (lr wd mu eps gamma lam vfCoef entCoef clipEps vfClip maxGradNorm replayRatio minLrRatio : Float)
    (rhoClip : Float := 1.0) (cClip : Float := 1.0) (prioAlpha : Float := 0.8)
    (prioBeta0 : Float := 0.2) (logDash : Bool := false)
    (annealEntCoef : Bool := false) (minEntCoefRatio : Float := 0.1)
    (checkpointInterval : Nat := 0)      -- STREAM 3: save every N updates (0 = final-only)
    (loadPath : Option String := none)   -- STREAM 3: seed weights from this checkpoint before training
    (seed : UInt64) : IO Unit := do
  let u := USize.ofNat; let mk := FloatArray.mk
  let h ← Puffer.Plugin.envOpen name (u numEnvs) seed config
  if h == 0 then IO.println s!"puffer train: env '{name}' not found — run ocean/build.sh {name}"; return
  let D := (Puffer.Plugin.envObsDim h).toNat
  let A := (Puffer.Plugin.envNumActions h).toNat
  let nAgents := (Puffer.Plugin.envNumAgents h).toNat
  let N := numEnvs * nAgents
  let H := hidden; let T := horizon
  let updates := max 1 (totalTimesteps / (N * T))
  let (w0, _) := initMinGRU D H numLayers A seed
  let mut wFlat := flattenMG w0                       -- weights as ONE flat array (host copy for the BPTT grad)
  let P := wFlat.size
  -- STREAM 3: seed from a checkpoint if --load was given (must match the current policy's param count).
  match loadPath with
  | some lp =>
    match ← loadPolicyCheckpoint lp with
    | some wl =>
      if wl.size == P then wFlat := wl; IO.println s!"puffer train: loaded checkpoint {lp} ({P} params)"
      else IO.println s!"puffer train: checkpoint {lp} has {wl.size} params, this policy needs {P} — ignoring, training fresh"
    | none => IO.println s!"puffer train: --load {lp} not found — training fresh"
  | none => pure ()
  -- Policy weights + momentum live device-RESIDENT in one handle: the rollout reads them and the Muon
  -- updates them IN PLACE, so [weights;mom] no longer round-trips PCIe per minibatch (only gClip uploads).
  let ph ← Puffer.Float.CUDA.policyLoadFFI (mk ((Array.range P).map (fun i => wFlat[i]!) ++ Array.replicate P 0.0)) (u P)
  if ph == 0 then
    IO.println "puffer train: policy weights device alloc failed (out of VRAM?)"
    Puffer.Plugin.envClose h; return
  let mut rng := seed
  let L := numLayers
  let G : UInt64 := 0x9E3779B97F4A7C15
  -- flat recurrent state N·L·H, PERSISTS across updates (truncated BPTT), reset at episode boundaries.
  let mut stateFlat : FloatArray := mk (Array.replicate (N*L*H) 0.0)
  IO.println s!"puffer train [{name}] MinGRU (recurrent {H}×{numLayers}L, PufferLib default net, GPU forward) — {numEnvs} envs × {nAgents} agents × {T} (batch {N}), {D}→enc{H}→MinGRU×{numLayers}→{A+1}, BPTT PPO+Muon, {updates} updates"
  -- STREAM 3: PufferLib's startup probe lines (src/pufferlib.cu:1987, vecenv.h:263). Single-discrete ⇒ 1 head.
  IO.println s!"Detected discrete action space with 1 heads"
  IO.println s!"Num workers: {← rollWorkers}"
  (← IO.getStdout).flush
  let plainLog := (← IO.getEnv "PUFFER_PLAIN_LOG").isSome   -- STREAM 3: suppress the checkpoint-saved line under plain-log
  let runId := s!"{seed}"                                   -- STREAM 3: checkpoints/<env>/<seed>/<step>.bin
  let mut obs ← Puffer.Plugin.envReset h
  let mut updNs : Nat := 0
  let prof := (← IO.getEnv "PUFFER_MG_PROFILE").isSome     -- phase wall-time attribution
  let mut tRoll : Nat := 0; let mut tVt : Nat := 0; let mut tGrad : Nat := 0; let mut tMuon : Nat := 0
  let mut tPrio : Nat := 0; let mut tGath : Nat := 0; let mut tIter : Nat := 0   -- "other"-phase sub-timers
  -- `--log` live dashboard state (opt-in; without it the ad-hoc per-update lines below are untouched).
  let t0 ← IO.monoNanosNow
  -- Losses are surfaced from the BPTT grad ONLY on the last minibatch of a render frame (0.6s cadence),
  -- toggled in the loop below — a per-minibatch D2H readback cost ~35% SPS, which would break SPS parity.
  let profT := prof || logDash                              -- phase timers on when the dashboard is live
  let mut idx : Nat := 0                                    -- blowfish spinner
  let mut lastRenderNs : Nat := 0                           -- 0.6s render rate-limit
  let mut lrRoll : Nat := 0; let mut lrVt : Nat := 0; let mut lrGrad : Nat := 0
  let mut lrMuon : Nat := 0; let mut lrUpd : Nat := 0       -- last-render phase-timer snapshots
  -- Resident chaining (Conn-lite): after the first chain-capable rollout, obs+state live INSIDE the
  -- native side (pinned ping-pong staging + device state buffer) — later calls pass EMPTY obs0/state0
  -- and get back only [rewCol; termCol] on logging updates (else an empty array), retiring the
  -- ~NT·(D+5)-sized per-update return and the f64 finalObs/finalState round trips. Bit-identical (the
  -- retired round trips were exact widen/narrow identities); PUFFER_MG_CHAIN=0 restores the old path.
  let mut chained := false
  -- env's OWN PufferLib `Log` (read+zero each update, printed on the progress line). MinGRU is the
  -- DEFAULT network, so without this the log channel is invisible on the default path.
  let mut lastLog : Array (String × Float) := #[]
  for upd in [0:updates] do
    let ub0 ← IO.monoNanosNow
    -- Decide the 0.6s render frame at update START so the loss readback (last minibatch) lines up with it.
    let willRender := logDash && (ub0 ≥ lastRenderNs + 600000000 || upd + 1 == updates)
    -- COSINE anneal, matching PufferLib (`torch_pufferl.py` train(): `lr_min + ½(lr − lr_min)(1 + cos(π·t/T))`,
    -- and `cosine_annealing()` in their `src/pufferlib.cu`). This was a LINEAR ramp — a different shape
    -- entirely: at the halfway point linear gives 0.5·lr where cosine gives 0.5·lr too, but the first
    -- and last quarters differ by up to ~10% of lr (cosine holds high early, decays fast mid, flattens late).
    let lrNow := cosineLr lr minLrRatio upd updates
    -- Entropy-coef anneal (PufferLib pufferlib.cu:1563-1566): same `cosine_annealing()` shape as lr, from the
    -- base `entCoef` (upd=0) to `entCoef·minEntCoefRatio` (upd=updates). OFF ⇒ entCoefNow == entCoef (default).
    let entCoefNow := if annealEntCoef then cosineLr entCoef minEntCoefRatio upd updates else entCoef
    let rolloutRng := rng; rng := rng + (UInt64.ofNat (N*T)) * G
    let NT := N*T
    -- Native per-update MinGRU rollout: resident weights (was ~1.7MB re-uploaded per step) + resident
    -- recurrent state (threaded + reset on device), device forward+sample, CPU env-step, C column scatter.
    let wantLog : UInt8 := if (upd+1) % 20 == 0 || upd == 0 then 1 else 0
    let wasChained := chained
    let r0 ← IO.monoNanosNow
    let roll ← Puffer.Float.CUDA.cudaPluginRolloutMinGRUFFI h ph
                 (if chained then FloatArray.mk #[] else obs)
                 (if chained then FloatArray.mk #[] else stateFlat)
                 (u N) (u D) (u H) (u L) (u A) (u T) wantLog rolloutRng
    if profT then tRoll := tRoll + ((← IO.monoNanosNow) - r0)
    if !wasChained then
      obs := Puffer.Float.FFI.sliceFFI roll (u (NT*D + 5*NT)) (u (N*D))
      stateFlat := Puffer.Float.FFI.sliceFFI roll (u (NT*D + 5*NT + N*D)) (u (N*L*H))
      chained ← Puffer.Float.CUDA.cudaMgChainReadyFFI (u N) (u D) (u (L*H))
    -- Trajectory is kept FULLY FLAT — no boxed `Transition` records are built at all (the per-update N·T
    -- allocation is gone). obs lives in obsCol (C-gathered per minibatch); the scalar per-step fields are
    -- read straight from the SoA columns (actCol/rewCol/valCol/logpCol/termCol, row e·T+s) where needed.
    -- === PufferLib train loop (torch_pufferl.py train()): V-Trace advantage with a LIVE per-segment
    -- importance-ratio buffer + iterated values, prioritized replay (sample ∝ |adv|^α, IS-corrected),
    -- reward clamped to [-1,1] inside computePuffAdvantageV. The V-Trace ratio correction is what stabilizes
    -- the off-policy replay that a plain PPO clip cannot. ===
    -- prioAlpha/prioBeta0/rhoClip/cClip are now PARAMETERS (PufferLib train.prio_alpha / prio_beta0 /
    -- vtrace_rho_clip / vtrace_c_clip). They used to be hardcoded local `let`s of 0.8/0.2/1.0/1.0 right
    -- here, which silently discarded every per-env value in config/*.ini — 28 envs set all four, and
    -- their sweep-tuned ρ̄ runs as high as 5.0 (maze) against the 1.0 that was actually being used.
    -- bootstrap V(s_T) per env (one read-only GPU forward) — the next-value for the segment's LAST step, so
    -- the terminal reward isn't dropped (computePuffAdvantageV zeros the last adv; the segment's final reward is there).
    -- bootVals V(s_T) is computed by the native rollout above (value at the final obs + recurrent state).
    let annealBeta := prioBeta0 + (1.0 - prioBeta0) * prioAlpha * Float.ofNat upd / Float.ofNat (max updates 1)
    -- V-Trace advantage now runs on the GPU (cudaVtraceMinGRUFFI, one thread/segment) — it was the
    -- dominant interpreted-Lean host loop (recomputed for all N segments every minibatch). valueBuf and
    -- ratioBuf are FLAT (N·T, row e·T+t) so they upload directly; the kernel is bit-exact vs the oracle
    -- `vtraceMinGRUFlat` (a line-for-line copy of this trainer's old per-segment closure).
    -- === DEVICE-RESIDENT minibatch prep (PufferLib's all-on-GPU train step) ===
    -- H2D the rollout's scalar columns ONCE per update; V-Trace, the scalar gather, and the value/ratio
    -- iterate then all run on the GPU (valueBuf/ratioBuf/advFlat stay device-resident). The only host↔device
    -- traffic per minibatch is Σ|adv| (for the host sampler), segIdx/mbPrio, and the gradient — killing the
    -- ~120ms/update of host minibatch prep (gather + iterate + per-mb V-Trace round-trips).
    -- Device-DIRECT columns: the buffered rollout scatters act/logp/val/rew/term/boot straight into the
    -- resident device columns (bit-identical values); when it stamped them, the six host slice copies AND
    -- the ~10.5MB pageable prep H2D below are skipped entirely.
    let colsReady ← Puffer.Float.CUDA.cudaMgColsReadyFFI (u N) (u T)
    if !colsReady then
      -- The full SoA layout below only exists when the rollout was NOT chained. If chaining is on (Lean
      -- latches `chained` once) but the rollout stopped stamping the resident columns — which is what a
      -- mid-run device-allocation failure looks like — `roll` is the compact [rewCol;termCol] form (or
      -- empty), and these offsets run off its end. `sliceFFI` memcpy's WITHOUT bounds checks, so pre-fix
      -- that was a heap over-read whose garbage was uploaded as the training columns and trained on,
      -- printing healthy updates/SPS and exiting 0. Refuse instead.
      if roll.size < NT*D + 5*NT + N*D + N*L*H + N then
        throw <| IO.userError s!"[puffer] *** MinGRU rollout FAILED: the resident scalar columns were not \
          stamped and the returned rollout is the compact chained form ({roll.size} doubles, need \
          {NT*D + 5*NT + N*D + N*L*H + N}) — a device allocation failed mid-run.\n\
          [puffer] *** Aborting: continuing would have trained on out-of-bounds garbage while printing \
          healthy-looking updates/SPS.\n\
          [puffer] *** Retry with fewer envs (--num-envs) or a shorter --train.horizon, or free VRAM."
      let actCol  := Puffer.Float.FFI.sliceFFI roll (u (NT*D)) (u NT)
      let logpCol := Puffer.Float.FFI.sliceFFI roll (u (NT*D + NT)) (u NT)
      let valCol  := Puffer.Float.FFI.sliceFFI roll (u (NT*D + 2*NT)) (u NT)
      let rewCol  := Puffer.Float.FFI.sliceFFI roll (u (NT*D + 3*NT)) (u NT)
      let termCol := Puffer.Float.FFI.sliceFFI roll (u (NT*D + 4*NT)) (u NT)
      let bvSlice := Puffer.Float.FFI.sliceFFI roll (u (NT*D + 5*NT + N*D + N*L*H)) (u N)  -- V(s_T) per env
      Puffer.Float.CUDA.cudaMgPrepFFI rewCol termCol actCol logpCol valCol bvSlice (u N) (u T) (u 1)
    let mbSegs := max 1 (min N (minibatchSize / T))
    let numMinibatches := max 1 (Float.toUInt64 (replayRatio * Float.ofNat (N*T) / Float.ofNat (max minibatchSize 1))).toNat
    for _mb in [0:numMinibatches] do
      let vt0 ← IO.monoNanosNow
      -- Device V-Trace + ON-DEVICE prioritized sampling (segIdx/mbPrio stay resident; the grad is then
      -- called with both arrays empty). Falls back to the host sampler when gated off.
      let devPrio ← Puffer.Float.CUDA.cudaMgVtracePrioFFI (u N) (u T) gamma lam rhoClip cClip
                      (u mbSegs) prioAlpha annealBeta rng
      if profT then tVt := tVt + ((← IO.monoNanosNow) - vt0)
      let p0 ← IO.monoNanosNow
      let (segIdxF, mbPrioF) ←
        if devPrio then pure (FloatArray.mk #[], FloatArray.mk #[])
        else do
          let advL1 ← Puffer.Float.CUDA.cudaMgVtraceFFI (u N) (u T) gamma lam rhoClip cClip
          pure (Puffer.Float.FFI.prioSampleFFI advL1 (u N) (u T) (u mbSegs) prioAlpha annealBeta rng)
      rng := rng + (UInt64.ofNat mbSegs) * (0x9E3779B97F4A7C15 : UInt64)
      let Bmb := if devPrio then mbSegs else segIdxF.size
      if prof then tPrio := tPrio + ((← IO.monoNanosNow) - p0)
      -- Enable the loss readback ONLY on the last minibatch of a render frame — set to 0 otherwise so a
      -- prior frame's enable does not leak into the next (non-render) update's minibatches.
      if logDash then Puffer.Float.CUDA.cudaMgLossEnableFFI (if willRender && _mb + 1 == numMinibatches then 1 else 0)
      let gd0 ← IO.monoNanosNow
      -- empty `scal` + nonempty `mbPrioF` ⇒ device-column mode: the BPTT gathers the 6 scalars on-GPU from the
      -- resident columns and iterates value/ratio on-GPU (no host gather, no host iterate). obs also device-resident.
      let g ← Puffer.Float.CUDA.cudaMinGRUPpoGradFFI wFlat (mk #[]) (mk #[])
                 (u Bmb) (u T) (u H) (u D) (u L) (u A) (u 1) (mk #[])
                 vfCoef entCoefNow clipEps vfClip segIdxF mbPrioF
      if profT then tGrad := tGrad + ((← IO.monoNanosNow) - gd0)
      let sc := 1.0 / Float.ofNat (max (Bmb*T) 1)
      let mu0 ← IO.monoNanosNow
      -- === GRADIENT MODE: the FFI is AUTHORITATIVE, Lean only READS its decision ===
      -- `g.size == 0` ⇒ the summed gradient stayed device-RESIDENT (bg 70): pass an empty gClip and the
      -- Muon gradclips + consumes it on-GPU. This is the fast path and is unchanged/bit-identical.
      -- `g.size  > 0` ⇒ HOST fallback: the resident buffer could not be allocated, so `g` carries the real
      -- gradient — clip it here (same formula) and hand the Muon a real gradient.
      -- Previously Lean assumed "resident" UNCONDITIONALLY and threw `g` away, so whenever the FFI could
      -- not produce a resident gradient the Muon stepped with a ZERO gradient: the run printed normal
      -- updates/episode returns/SPS, finished, and exited 0 having learned nothing. Deciding the mode on
      -- ONE side (C) removes the possibility of the two disagreeing; the FFI now raises an IO error —
      -- nonzero exit — when it can produce neither form, rather than returning an empty "resident" array.
      let gClip := if g.size == 0 then mk #[] else gradClipMG g P maxGradNorm sc
      wFlat ← Puffer.Float.CUDA.cudaMinGRUMuonResidentFFI ph gClip (u H) (u D) (u L) (u A) lrNow wd mu eps maxGradNorm sc
      if profT then tMuon := tMuon + ((← IO.monoNanosNow) - mu0)
    let ub1 ← IO.monoNanosNow; updNs := updNs + (ub1 - ub0)
    -- Live dashboard (PufferLib's rich monitor): redraw in place. `willRender` was decided at update
    -- start (0.6s cadence) so the loss readback on this update's last minibatch lines up with this frame.
    if willRender then
      let now ← IO.monoNanosNow
      let lossArr ← Puffer.Float.CUDA.cudaMgReadLossesFFI
      let lg ← Puffer.Plugin.envLogPairs h              -- read+zero; User Stats source
      if !lg.isEmpty then lastLog := lg
      let du := max 1 (upd + 1 - lrUpd)                 -- updates since last render (per-update avg)
      let per := fun (cur last : Nat) => Float.ofNat ((cur - last) / du) / 1.0e9
      Puffer.RL.Dashboard.redrawFrom name P ((upd+1)*N*T) (upd+1) t0 now totalTimesteps
        (per tRoll lrRoll) (per tVt lrVt) (per tGrad lrGrad) (per tMuon lrMuon)
        lossArr (lastLog.filter (fun kv => kv.1 != "n")) idx
      lrRoll := tRoll; lrVt := tVt; lrGrad := tGrad; lrMuon := tMuon; lrUpd := upd + 1
      idx := (idx + 9) % 10; lastRenderNs := now
    -- STREAM 3: checkpoint the resident policy weights — every `checkpointInterval` updates (if >0) and
    -- always on the final update. Cheap (a host-side wFlat write); the print is gated so PUFFER_PLAIN_LOG
    -- stays machine-parseable (the grepped lines are untouched).
    if (upd+1 == updates) || (checkpointInterval > 0 && (upd+1) % checkpointInterval == 0) then
      -- weights are DEVICE-RESIDENT (the resident Muon returns an empty host array), so DOWNLOAD them
      -- for the checkpoint; the host `wFlat` is empty after update 1 and would serialize 0 params.
      let wFull ← Puffer.Float.CUDA.policyDownloadFFI ph (u P)   -- 2P: [weights(P); momentum(P)]
      let wSave := Puffer.Float.FFI.sliceFFI wFull (u 0) (u P)   -- first P = weights (load re-zeros momentum)
      let ckptPath ← savePolicyCheckpoint name runId ((upd+1) * N * T) wSave
      if !plainLog then IO.println s!"  checkpoint saved: {ckptPath}"
    if !logDash && ((upd+1) % 20 == 0 || upd == 0) then
      -- lazy slices: rew/term are only read here (every 20th update); prep no longer needs them host-side.
      -- Chained calls return the compact [rewCol; termCol] layout; legacy calls the full SoA layout.
      let rewCol  := if wasChained then Puffer.Float.FFI.sliceFFI roll (u 0) (u NT)
                     else Puffer.Float.FFI.sliceFFI roll (u (NT*D + 3*NT)) (u NT)
      let termCol := if wasChained then Puffer.Float.FFI.sliceFFI roll (u NT) (u NT)
                     else Puffer.Float.FFI.sliceFFI roll (u (NT*D + 4*NT)) (u NT)
      let lg ← Puffer.Plugin.envLogPairs h   -- read+zero; metrics only, no learning path touched
      if !lg.isEmpty then lastLog := lg
      let mut epRet := 0.0; let mut nEps := 0
      for n in [0:N] do
        let mut run := 0.0
        for s in [0:T] do
          let row := n*T+s
          run := run + rewCol[row]!
          if termCol[row]! > 0.5 then epRet := epRet + run; nEps := nEps + 1; run := 0.0
      let m := if nEps == 0 then 0.0 else epRet / Float.ofNat nEps
      -- per-step mean too: envs that never set a terminal flag (upstream reports their episodes via a
      -- `Log` channel our plugin ABI does not carry) would otherwise print a permanent 0.0 here
      let mut rsum := 0.0
      for i in [0:N*T] do rsum := rsum + rewCol[i]!
      IO.println s!"  update {upd+1}: {nEps} eps, mean ep return = {m} (mean/step {rsum / Float.ofNat (N*T)}){fmtEnvLog lastLog}"
      (← IO.getStdout).flush
  Puffer.Float.CUDA.policyFreeFFI ph          -- release the device-resident policy buffer
  Puffer.Plugin.envClose h
  let envSteps := updates * N * T
  let sps := if updNs == 0 then 0 else envSteps * 1000000000 / updNs
  IO.println s!"done. perf: {envSteps} env-steps ⇒ {sps} SPS"
  if prof then
    let ms := fun (n : Nat) => Float.ofNat (n / 1000) / 1000.0
    let other := (if updNs ≥ tRoll+tVt+tGrad+tMuon then updNs - (tRoll+tVt+tGrad+tMuon) else 0)
    IO.println s!"[mg-prof] rollout={ms tRoll}  vtrace={ms tVt}  bptt-grad={ms tGrad}  muon={ms tMuon}  other={ms other}  ms (cumulative)"
    IO.println s!"[mg-prof]   other split: prio/sample={ms tPrio}  minibatch-gather={ms tGath}  ratio/value-iterate+gradclip={ms tIter}  ms"

/-- **MinGRU recurrent MULTI-DISCRETE plugin trainer** — PufferLib's default network (`[torch] network =
    MinGRU`, `[policy] num_layers`) for envs with `K > 1` categorical action heads (convert, drive, target,
    terraform, moba, robocode, slimevolley, minimal). Until this existed every MD env was routed to the
    single-hidden-layer MLP regardless of `network`, so `num_layers` was silently ignored for all of them.

    The policy is the SAME MinGRU core as `trainPluginEnvMinGRU`, only the head is wider: the decoder emits
    `W = Σ headSizes` logits (+ the value), sliced into `K` per-head softmaxes, sampled per head, with the
    joint log-prob `Σ_h log p_h(a_h)` driving one PPO clip — the convention of the multi-discrete MLP path
    (`trainPluginEnvMD` / `k_ppo_dout_md`). `initMinGRU … W` therefore gives the right weight shapes, and
    the BPTT / Muon / V-Trace / prioritized replay are the single-discrete ones with `A := W`.

    Per update: MD rollout (`cudaPluginRolloutMinGRUMDFFI` — the FUSED K-head arm: fused single-launch
    step kernel, concurrent stream-buffers, CUDA-graph replay, zero-copy pinned obs, K-wide pinned
    actions; it falls back to the legacy non-fused arm for shapes the fused kernel cannot express, see
    its docstring) → device-direct columns (so `cudaMgPrepFFI` and its ~24MB pageable H2D are SKIPPED
    whenever `cudaMgColsReadyFFI` says the rollout stamped them) → per minibatch device V-Trace +
    prioritized sampling → the MD BPTT (`k_mg_ppo_b_md`) → resident Muon. Like the single-discrete
    trainer it also chains obs/state resident across updates once the fused arm has run one update. -/
def trainPluginEnvMinGRUMD (name config : String)
    (hidden numLayers numEnvs horizon totalTimesteps epochs numMB minibatchSize : Nat)
    (lr wd mu eps gamma lam vfCoef entCoef clipEps vfClip maxGradNorm replayRatio minLrRatio : Float)
    (rhoClip : Float := 1.0) (cClip : Float := 1.0) (prioAlpha : Float := 0.8)
    (prioBeta0 : Float := 0.2) (logDash : Bool := false)
    (annealEntCoef : Bool := false) (minEntCoefRatio : Float := 0.1)
    (checkpointInterval : Nat := 0)      -- STREAM 3: save every N updates (0 = final-only)
    (loadPath : Option String := none)   -- STREAM 3: seed weights from this checkpoint before training
    (seed : UInt64) : IO Unit := do
  let _ := epochs; let _ := numMB                    -- PufferLib derives the minibatch count from replay_ratio
  let u := USize.ofNat; let mk := FloatArray.mk
  let h ← Puffer.Plugin.envOpen name (u numEnvs) seed config
  if h == 0 then IO.println s!"puffer train: env '{name}' not found — run ocean/build.sh {name}"; return
  let D := (Puffer.Plugin.envObsDim h).toNat
  let nAgents := (Puffer.Plugin.envNumAgents h).toNat
  let N := numEnvs * nAgents
  let K := (Puffer.Plugin.envNHeads h).toNat
  let headSizes := Puffer.Plugin.envHeadSizes h
  let Wtot := headSizes.foldl (·+·) 0                -- logits width = Σ head sizes (the decoder's rows)
  let hsF := mk (headSizes.map Float.ofNat)
  let H := hidden; let T := horizon; let L := numLayers
  let updates := max 1 (totalTimesteps / (N * T))
  if K ≤ 1 || Wtot == 0 then
    IO.println s!"puffer train: '{name}' is not multi-discrete ({K} head(s)) — use the single-discrete MinGRU trainer"
    Puffer.Plugin.envClose h; return
  let (w0, _) := initMinGRU D H L Wtot seed
  let mut wFlat := flattenMG w0
  let P := wFlat.size
  -- STREAM 3: seed from a checkpoint if --load was given (must match the current policy's param count).
  match loadPath with
  | some lp =>
    match ← loadPolicyCheckpoint lp with
    | some wl =>
      if wl.size == P then wFlat := wl; IO.println s!"puffer train: loaded checkpoint {lp} ({P} params)"
      else IO.println s!"puffer train: checkpoint {lp} has {wl.size} params, this policy needs {P} — ignoring, training fresh"
    | none => IO.println s!"puffer train: --load {lp} not found — training fresh"
  | none => pure ()
  let ph ← Puffer.Float.CUDA.policyLoadFFI (mk ((Array.range P).map (fun i => wFlat[i]!) ++ Array.replicate P 0.0)) (u P)
  if ph == 0 then
    IO.println "puffer train: policy weights device alloc failed (out of VRAM?)"
    Puffer.Plugin.envClose h; return
  let mut rng := seed
  let G : UInt64 := 0x9E3779B97F4A7C15
  -- flat recurrent state N·L·H, PERSISTS across updates (truncated BPTT), reset at episode boundaries.
  let mut stateFlat : FloatArray := mk (Array.replicate (N*L*H) 0.0)
  IO.println s!"puffer train [{name}] MULTI-DISCRETE ({K} heads {headSizes}) MinGRU (recurrent {H}×{numLayers}L, PufferLib default net, GPU forward) — {numEnvs} envs × {nAgents} agents × {T} (batch {N}), {D}→enc{H}→MinGRU×{numLayers}→{Wtot}+1, BPTT PPO+Muon, {updates} updates"
  -- STREAM 3: PufferLib's startup probe lines (src/pufferlib.cu:1987, vecenv.h:263). K categorical heads.
  IO.println s!"Detected discrete action space with {K} heads"
  IO.println s!"Num workers: {← rollWorkers}"
  (← IO.getStdout).flush
  let plainLog := (← IO.getEnv "PUFFER_PLAIN_LOG").isSome   -- STREAM 3: gate the checkpoint-saved line
  let runId := s!"{seed}"                                   -- STREAM 3: checkpoints/<env>/<seed>/<step>.bin
  let mut obs ← Puffer.Plugin.envReset h
  let mut updNs : Nat := 0
  -- DIAGNOSTIC ONLY (`PUFFER_LOG_EVERY`, default 20 = the shipped cadence): print-cadence override for
  -- the reward line. Needed to tell whether the moba entropy collapse LEADS or FOLLOWS the reward
  -- decline — at 183 updates the 20-update cadence gives 9 points, far too coarse to order the two.
  -- Touches ONLY when the block below runs: no rollout, gradient or RNG consumer reads it, and the
  -- rew/term columns it slices are returned by the rollout on every update either way. Unset ⇒
  -- byte-identical to the shipped path. NB at =1 the env-log window (`envLogPairs` is read+zero)
  -- shrinks to one update, so the `env log:` fields get noisier — that is the diagnostic's own cost,
  -- not a training change.
  let logEvery : Nat := max 1 (((← IO.getEnv "PUFFER_LOG_EVERY").bind String.toNat?).getD 20)
  let mut lastLog : Array (String × Float) := #[]     -- env's own PufferLib `Log`, latest non-empty window
  -- Resident chaining (Conn-lite), as on the single-discrete path: once the FUSED rollout arm has run
  -- an update, obs live in its pinned ping-pong staging and the recurrent state in the device buffer,
  -- so later calls pass EMPTY obs0/state0 and get back only [rewCol; termCol]. At the flagship MD
  -- config that retires a ~33MB state round trip + a ~4MB obs round trip (and their f32↔f64 conversion
  -- loops) per update. Bit-identical (the retired round trips were exact widen/narrow identities);
  -- PUFFER_MG_CHAIN=0 and PUFFER_MG_MD_FUSED=0 both restore the per-update round trip.
  let mut chained := false
  -- `--log` live dashboard state (opt-in; the ad-hoc lines below are untouched without it). The MD
  -- trainer had no phase timers — they are added here, taken only when `logDash` (zero cost otherwise).
  let t0 ← IO.monoNanosNow
  -- Losses surfaced from the BPTT grad ONLY on the last minibatch of a render frame (toggled in the loop),
  -- not every minibatch — a per-minibatch D2H readback cost ~35% SPS, which would break SPS parity.
  let mut tRoll : Nat := 0; let mut tVt : Nat := 0; let mut tGrad : Nat := 0; let mut tMuon : Nat := 0
  let mut idx : Nat := 0                                    -- blowfish spinner
  let mut lastRenderNs : Nat := 0                           -- 0.6s render rate-limit
  let mut lrRoll : Nat := 0; let mut lrVt : Nat := 0; let mut lrGrad : Nat := 0
  let mut lrMuon : Nat := 0; let mut lrUpd : Nat := 0       -- last-render phase-timer snapshots
  for upd in [0:updates] do
    let ub0 ← IO.monoNanosNow
    -- 0.6s render frame decided at update START so the last-minibatch loss readback lines up with it.
    let willRender := logDash && (ub0 ≥ lastRenderNs + 600000000 || upd + 1 == updates)
    -- COSINE anneal, matching PufferLib (see the single-discrete trainer above) — was a LINEAR ramp.
    let lrNow := cosineLr lr minLrRatio upd updates
    -- Entropy-coef anneal (PufferLib pufferlib.cu:1563-1566), same cosine shape as lr; OFF ⇒ == entCoef.
    let entCoefNow := if annealEntCoef then cosineLr entCoef minEntCoefRatio upd updates else entCoef
    let rolloutRng := rng; rng := rng + (UInt64.ofNat (N*T)) * G
    let NT := N*T
    -- [actCol(NT·K); logpCol; valCol; rewCol; termCol; finalObs(N·D); finalState(N·L·H); bootVals(N)]
    -- (obs are NOT returned: the BPTT gathers them from the device-resident trajectory the rollout wrote),
    -- or the compact CHAINED form [rewCol(NT); termCol(NT)] once the fused arm holds obs+state resident.
    let wasChained := chained
    let r0md ← if logDash then IO.monoNanosNow else pure (0 : Nat)
    let roll ← Puffer.Float.CUDA.cudaPluginRolloutMinGRUMDFFI h ph
                 (if chained then mk #[] else obs)
                 (if chained then mk #[] else stateFlat)
                 hsF (u N) (u D) (u H) (u L) (u Wtot) (u K) (u T) rolloutRng
    if logDash then tRoll := tRoll + ((← IO.monoNanosNow) - r0md)
    if !wasChained then
      obs       := Puffer.Float.FFI.sliceFFI roll (u (NT*K + 4*NT)) (u (N*D))
      stateFlat := Puffer.Float.FFI.sliceFFI roll (u (NT*K + 4*NT + N*D)) (u (N*L*H))
      -- only the FUSED rollout arm keeps obs/state resident; the legacy arm never stamps, so this
      -- latches false there and every update keeps passing the real arrays.
      chained ← Puffer.Float.CUDA.cudaMgChainReadyFFI (u N) (u D) (u (L*H))
    -- rew/term offsets differ between the two return layouts; the slices themselves are built lazily
    -- (only the logging updates and the non-device-column prep need them).
    let rewOff  := if wasChained then 0  else NT*K + 2*NT
    let termOff := if wasChained then NT else NT*K + 3*NT
    -- Device-DIRECT columns: the fused rollout arm scattered act/logp/val/rew/term/boot straight into
    -- the resident device columns (K-wide act included), so the host slices AND mg_prep's ~24MB
    -- pageable H2D are skipped entirely. The legacy arm does not stamp them and preps as before.
    let colsReady ← Puffer.Float.CUDA.cudaMgColsReadyFFI (u N) (u T)
    if !colsReady then
      -- Same guard as the single-discrete trainer: the full SoA layout only exists when the rollout was
      -- NOT chained. If chaining latched but the columns stopped being stamped (what a mid-run device
      -- allocation failure looks like), these offsets would run off the compact return — and sliceFFI
      -- memcpy's without bounds checks, so we would train on heap garbage while printing healthy SPS.
      if wasChained || roll.size < NT*K + 4*NT + N*D + N*L*H + N then
        throw <| IO.userError s!"[puffer] *** MD MinGRU rollout FAILED: the resident scalar columns were \
          not stamped and the returned rollout is the compact chained form ({roll.size} doubles, need \
          {NT*K + 4*NT + N*D + N*L*H + N}) — a device allocation failed mid-run.\n\
          [puffer] *** Aborting: continuing would have trained on out-of-bounds garbage.\n\
          [puffer] *** Retry with fewer envs (--num-envs) or a shorter --train.horizon, or free VRAM."
      let actCol  := Puffer.Float.FFI.sliceFFI roll (u 0) (u (NT*K))
      let logpCol := Puffer.Float.FFI.sliceFFI roll (u (NT*K)) (u NT)
      let valCol  := Puffer.Float.FFI.sliceFFI roll (u (NT*K + NT)) (u NT)
      let rewCol  := Puffer.Float.FFI.sliceFFI roll (u rewOff) (u NT)
      let termCol := Puffer.Float.FFI.sliceFFI roll (u termOff) (u NT)
      let bvSlice := Puffer.Float.FFI.sliceFFI roll (u (NT*K + 4*NT + N*D + N*L*H)) (u N)   -- V(s_T) per row
      Puffer.Float.CUDA.cudaMgPrepFFI rewCol termCol actCol logpCol valCol bvSlice (u N) (u T) (u K)
    -- prioAlpha/prioBeta0/rhoClip/cClip are PARAMETERS (PufferLib train.prio_alpha / prio_beta0 /
    -- vtrace_rho_clip / vtrace_c_clip), not the hardcoded 0.8/0.2/1.0/1.0 that used to sit here and
    -- silently discard every per-env config value.
    let annealBeta := prioBeta0 + (1.0 - prioBeta0) * prioAlpha * Float.ofNat upd / Float.ofNat (max updates 1)
    let mbSegs := max 1 (min N (minibatchSize / T))
    let numMinibatches := max 1 (Float.toUInt64 (replayRatio * Float.ofNat (N*T) / Float.ofNat (max minibatchSize 1))).toNat
    for _mb in [0:numMinibatches] do
      let vt0 ← if logDash then IO.monoNanosNow else pure (0 : Nat)
      let devPrio ← Puffer.Float.CUDA.cudaMgVtracePrioFFI (u N) (u T) gamma lam rhoClip cClip
                      (u mbSegs) prioAlpha annealBeta rng
      let (segIdxF, mbPrioF) ←
        if devPrio then pure (mk #[], mk #[])
        else do
          let advL1 ← Puffer.Float.CUDA.cudaMgVtraceFFI (u N) (u T) gamma lam rhoClip cClip
          pure (Puffer.Float.FFI.prioSampleFFI advL1 (u N) (u T) (u mbSegs) prioAlpha annealBeta rng)
      rng := rng + (UInt64.ofNat mbSegs) * G
      let Bmb := if devPrio then mbSegs else segIdxF.size
      if logDash then tVt := tVt + ((← IO.monoNanosNow) - vt0)
      -- Enable the loss readback ONLY on the last minibatch of a render frame; set 0 otherwise so a prior
      -- frame's enable does not leak into the next update's minibatches (a per-mb readback costs ~35% SPS).
      if logDash then Puffer.Float.CUDA.cudaMgLossEnableFFI (if willRender && _mb + 1 == numMinibatches then 1 else 0)
      let gd0 ← if logDash then IO.monoNanosNow else pure (0 : Nat)
      -- device-column mode (empty obs+scal): the BPTT gathers obs + the K-wide action column on-GPU.
      let g ← Puffer.Float.CUDA.cudaMinGRUPpoGradFFI wFlat (mk #[]) (mk #[])
                 (u Bmb) (u T) (u H) (u D) (u L) (u Wtot) (u K) hsF
                 vfCoef entCoefNow clipEps vfClip segIdxF mbPrioF
      if logDash then tGrad := tGrad + ((← IO.monoNanosNow) - gd0)
      let sc := 1.0 / Float.ofNat (max (Bmb*T) 1)
      -- gradient MODE is the FFI's decision, read off its return size (see cudaMinGRUPpoGradFFI)
      let gClip := if g.size == 0 then mk #[] else gradClipMG g P maxGradNorm sc
      let mu0 ← if logDash then IO.monoNanosNow else pure (0 : Nat)
      wFlat ← Puffer.Float.CUDA.cudaMinGRUMuonResidentFFI ph gClip (u H) (u D) (u L) (u Wtot) lrNow wd mu eps maxGradNorm sc
      if logDash then tMuon := tMuon + ((← IO.monoNanosNow) - mu0)
    let ub1 ← IO.monoNanosNow; updNs := updNs + (ub1 - ub0)
    -- Live dashboard (PufferLib's rich monitor): redraw in place. `willRender` was decided at update
    -- start (0.6s cadence) so the loss readback on this update's last minibatch lines up with this frame.
    if willRender then
      let now ← IO.monoNanosNow
      let lossArr ← Puffer.Float.CUDA.cudaMgReadLossesFFI
      let lg ← Puffer.Plugin.envLogPairs h              -- read+zero; User Stats source
      if !lg.isEmpty then lastLog := lg
      let du := max 1 (upd + 1 - lrUpd)                 -- updates since last render (per-update avg)
      let per := fun (cur last : Nat) => Float.ofNat ((cur - last) / du) / 1.0e9
      Puffer.RL.Dashboard.redrawFrom name P ((upd+1)*N*T) (upd+1) t0 now totalTimesteps
        (per tRoll lrRoll) (per tVt lrVt) (per tGrad lrGrad) (per tMuon lrMuon)
        lossArr (lastLog.filter (fun kv => kv.1 != "n")) idx
      lrRoll := tRoll; lrVt := tVt; lrGrad := tGrad; lrMuon := tMuon; lrUpd := upd + 1
      idx := (idx + 9) % 10; lastRenderNs := now
    -- STREAM 3: checkpoint the resident policy weights — every `checkpointInterval` updates (if >0) and
    -- always on the final update. Print gated so PUFFER_PLAIN_LOG stays machine-parseable.
    if (upd+1 == updates) || (checkpointInterval > 0 && (upd+1) % checkpointInterval == 0) then
      -- weights are DEVICE-RESIDENT (the resident Muon returns an empty host array), so DOWNLOAD them
      -- for the checkpoint; the host `wFlat` is empty after update 1 and would serialize 0 params.
      let wFull ← Puffer.Float.CUDA.policyDownloadFFI ph (u P)   -- 2P: [weights(P); momentum(P)]
      let wSave := Puffer.Float.FFI.sliceFFI wFull (u 0) (u P)   -- first P = weights (load re-zeros momentum)
      let ckptPath ← savePolicyCheckpoint name runId ((upd+1) * N * T) wSave
      if !plainLog then IO.println s!"  checkpoint saved: {ckptPath}"
    -- same cadence as the single-discrete trainer PLUS the final update: at PufferLib's MD config
    -- (8192 agents × 64) a 12M-step run is only ~22 updates, so a pure every-20th cadence would never
    -- print the score the run is judged on.
    if !logDash && ((upd+1) % logEvery == 0 || upd == 0 || upd + 1 == updates) then
      let lg ← Puffer.Plugin.envLogPairs h            -- read+zero; metrics only, no learning path touched
      if !lg.isEmpty then lastLog := lg
      -- lazy slices: rew/term are only read here, and the offsets follow the layout the rollout returned
      let rewCol  := Puffer.Float.FFI.sliceFFI roll (u rewOff) (u NT)
      let termCol := Puffer.Float.FFI.sliceFFI roll (u termOff) (u NT)
      let mut epRet := 0.0; let mut nEps := 0
      for n in [0:N] do
        let mut run := 0.0
        for s in [0:T] do
          let row := n*T+s
          run := run + rewCol[row]!
          if termCol[row]! > 0.5 then epRet := epRet + run; nEps := nEps + 1; run := 0.0
      let m := if nEps == 0 then 0.0 else epRet / Float.ofNat nEps
      let mut rsum := 0.0
      for i in [0:N*T] do rsum := rsum + rewCol[i]!
      IO.println s!"  update {upd+1}: {nEps} eps, mean ep return = {m} (mean/step {rsum / Float.ofNat (N*T)}){fmtEnvLog lastLog}"
      (← IO.getStdout).flush
  Puffer.Float.CUDA.policyFreeFFI ph          -- release the device-resident policy buffer
  Puffer.Plugin.envClose h
  let envSteps := updates * N * T
  let sps := if updNs == 0 then 0 else envSteps * 1000000000 / updNs
  IO.println s!"done. perf: {envSteps} env-steps ⇒ {sps} SPS"

end Puffer.RL.NNTrain
