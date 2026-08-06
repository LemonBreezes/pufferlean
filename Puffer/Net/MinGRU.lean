/-
Runnable, Mathlib-free forward pass of PufferLib's DEFAULT policy network:
`DefaultEncoder → MinGRU(num_layers) → DefaultDecoder`.

Faithful to `~/src/PufferLib/pufferlib/models.py` (the classes selected by
`config/default.ini [torch] network=MinGRU, encoder=DefaultEncoder, decoder=DefaultDecoder`;
`[policy] hidden_size=128, num_layers=4`, `expansion_factor` is ignored by MinGRU):

  DefaultEncoder (models.py:30)  e = W_enc·obs + b_enc                 (Linear, bias, NO activation)
  MinGRU        (models.py:105)  per layer i (Linear(H,3H,bias=False)):
      [hid, gate, proj] = W_i · h                                     (chunk into 3×H)
      z   = sigmoid(gate)
      g   = _g(hid),  _g(x) = x+0.5 if x≥0 else sigmoid(x)            (models.py:116)
      out = (1−z)·state_i + z·g                                       (torch.lerp, models.py:139)
      hg  = sigmoid(proj)
      h   = hg·out + (1−hg)·h                                         (highway, models.py:121-123)
      state_i ← out
  DefaultDecoder (models.py:60)  logits = W_dec·h + b_dec  (discrete);  value = W_val·h + b_val

There is NO LayerNorm/RMSNorm in this path (RMSNorm lives only in the separate `GRU` class,
models.py:206). The training path `forward_train` (models.py:144) computes the SAME recurrence
via a Heinsen log-space parallel scan — mathematically identical to this sequential
`forward_eval` recurrence (models.py:132), differing only in float summation order. We port the
sequential recurrence (the rollout/inference path) and verify it bit-for-bit against a NumPy
reference of the same math (`tools/mingru_ref.py`).

This is the FORWARD only — the backward/gradient kernel and trainer wiring are a later step.
-/
namespace Puffer.Net.MinGRU

abbrev Vec := Array Float
abbrev Mat := Array (Array Float)   -- rows

/-- Sequential left-to-right dot (index 0 upward). Both this and the NumPy reference use this
    exact order, so the two forwards agree bit-for-bit (torch's BLAS uses a different summation
    order → a documented roundoff-scale difference, not an algorithm difference). -/
def dot (w x : Vec) : Float := Id.run do
  let mut acc := 0.0
  for i in [0:w.size] do
    acc := acc + w[i]! * x[i]!
  return acc

/-- `W · x` for `W` a list of rows. -/
def matvec (W : Mat) (x : Vec) : Vec := W.map (fun row => dot row x)

/-- `W · x + b`. -/
def linear (W : Mat) (b : Vec) (x : Vec) : Vec :=
  (Array.range W.size).map (fun i => dot (W[i]!) x + b[i]!)

@[inline] def sigmoid (x : Float) : Float := 1.0 / (1.0 + Float.exp (-x))

/-- MinGRU's `_g` activation (`models.py:116`): `x+0.5` for `x ≥ 0`, else `sigmoid x`. -/
@[inline] def gAct (x : Float) : Float := if 0.0 ≤ x then x + 0.5 else sigmoid x

/-- Fixed weights for one policy (single discrete action head). -/
structure Weights where
  wEnc : Mat   -- [H × obsSize]   encoder
  bEnc : Vec   -- [H]
  layers : Array Mat  -- num_layers × [3H × H]   MinGRU (no bias)
  wDec : Mat   -- [A × H]        decoder logits
  bDec : Vec   -- [A]
  wVal : Mat   -- [1 × H]        value head
  bVal : Vec   -- [1]

/-- One MinGRU layer step: input `h` [H], previous state `prev` [H], weight `W` [3H × H].
    Returns `(h', newState)` — `h'` feeds the next layer, `newState` is carried to next t. -/
def layerStep (W : Mat) (h prev : Vec) (H : Nat) : Vec × Vec := Id.run do
  let y := matvec W h                 -- [3H]
  let mut out : Vec := Array.emptyWithCapacity H
  let mut hnew : Vec := Array.emptyWithCapacity H
  for j in [0:H] do
    let hid := y[j]!
    let gate := y[H + j]!
    let proj := y[2*H + j]!
    let z := sigmoid gate
    let o := (1.0 - z) * prev[j]! + z * gAct hid    -- lerp(state, g(hid), z)
    let hg := sigmoid proj
    let hn := hg * o + (1.0 - hg) * h[j]!            -- highway(h, out, proj)
    out := out.push o
    hnew := hnew.push hn
  return (hnew, out)

/-- Single-timestep `forward_eval`: `obs`, per-layer `state` → `(logits, value, newState)`. -/
def stepForward (w : Weights) (H : Nat) (obs : Vec) (state : Array Vec) : Vec × Float × Array Vec := Id.run do
  let e := linear w.wEnc w.bEnc obs
  let mut h := e
  let mut newState : Array Vec := Array.emptyWithCapacity w.layers.size
  for i in [0:w.layers.size] do
    let (hn, out) := layerStep (w.layers[i]!) h (state[i]!) H
    h := hn
    newState := newState.push out
  let logits := linear w.wDec w.bDec h
  let value := (linear w.wVal w.bVal h)[0]!
  return (logits, value, newState)

/-- Sequence forward (`forward_eval` carrying state from a ZERO initial state, matching
    `forward_train`'s per-segment convention). Returns `(logits, value)` per timestep. -/
def seqForward (w : Weights) (H numLayers : Nat) (obsSeq : Array Vec) : Array (Vec × Float) := Id.run do
  let mut state : Array Vec := (Array.range numLayers).map (fun _ => Array.replicate H 0.0)
  let mut outs : Array (Vec × Float) := Array.emptyWithCapacity obsSeq.size
  for obs in obsSeq do
    let (logits, value, newState) := stepForward w H obs state
    state := newState
    outs := outs.push (logits, value)
  return outs

/-! ### `forward_train` — the Heinsen log-space parallel scan (models.py:144).

PufferLib's TRAINING forward: the SAME MinGRU recurrence as `forward_eval`
(`out_t = (1−z_t)·out_{t−1} + z_t·g_t`, zero initial state), computed as a parallel scan
in log space (on a GPU this is `O(log T)` depth; on our sequential CPU it is the same `O(T)`
work, kept for parity with what PufferLib actually differentiates through). Numerically equal
to `seqForward` up to float summation order. -/

/-- `softplus x = log(1 + eˣ)`, in the stable branch form. -/
@[inline] def softplus (x : Float) : Float :=
  if x > 0.0 then x + Float.log (1.0 + Float.exp (-x)) else Float.log (1.0 + Float.exp x)

/-- `_log_g` (models.py:118): `log(x+0.5)` for `x ≥ 0` (= `log(_g x)`), else `−softplus(−x)` (= `log σ(x)`). -/
@[inline] def logG (x : Float) : Float :=
  if 0.0 ≤ x then Float.log (x + 0.5) else -(softplus (-x))

/-- Heinsen log-space associative scan (models.py:125-127) for `out_t = (1−z_t)·out_{t−1} + z_t·g_t`
    (`out₀ = 0`): given `log_coeffs = log(1−z)` and `log_values = log(z·g)`, returns
    `out = exp(a⋆ + logcumsumexp(log_values − a⋆))` with `a⋆ = cumsum(log_coeffs)`. The
    logcumsumexp is a single stable online pass (running max `m`, sum `s`). -/
def heinsenScan (logCoeffs logValues : Array Float) : Array Float := Id.run do
  let n := logCoeffs.size
  let mut out : Array Float := Array.replicate n 0.0
  let mut aStar := 0.0
  let mut m := -1.0e308         -- running max of `x = log_values − a⋆` (≈ −∞)
  let mut s := 0.0              -- Σ_{i≤t} exp(xᵢ − m)
  for t in [0:n] do
    aStar := aStar + logCoeffs[t]!
    let x := logValues[t]! - aStar
    if x > m then
      s := s * Float.exp (m - x) + 1.0
      m := x
    else
      s := s + Float.exp (x - m)
    out := out.set! t (Float.exp (aStar + m + Float.log s))   -- exp(a⋆ + (m + log s)) = exp(a⋆ + lcse)
  return out

/-- `forward_train` (models.py:144): the MinGRU recurrence via the Heinsen scan (per neuron, over
    time) + the highway. Zero initial state. `(logits, value)` per timestep, matching `seqForward`
    up to float order. -/
def seqForwardTrain (w : Weights) (H numLayers : Nat) (obsSeq : Array Vec) : Array (Vec × Float) := Id.run do
  let T := obsSeq.size
  let mut h : Array Vec := obsSeq.map (fun obs => linear w.wEnc w.bEnc obs)
  for l in [0:numLayers] do
    let W := w.layers[l]!
    let ys : Array Vec := h.map (fun ht => matvec W ht)                 -- [T][3H]  (hidden|gate|proj)
    let mut outCols : Array Vec := Array.emptyWithCapacity H            -- [H][T]  scanned state per neuron
    for j in [0:H] do
      let logCoeffs := (Array.range T).map (fun t => -(softplus (ys[t]![H + j]!)))
      let logValues := (Array.range T).map (fun t => -(softplus (-(ys[t]![H + j]!))) + logG (ys[t]![j]!))
      outCols := outCols.push (heinsenScan logCoeffs logValues)
    let mut newH : Array Vec := Array.emptyWithCapacity T
    for t in [0:T] do
      let mut row : Vec := Array.emptyWithCapacity H
      for j in [0:H] do
        let hg := sigmoid (ys[t]![2*H + j]!)
        row := row.push (hg * (outCols[j]![t]!) + (1.0 - hg) * (h[t]![j]!))     -- highway
      newH := newH.push row
    h := newH
  return (Array.range T).map (fun t => (linear w.wDec w.bDec (h[t]!), (linear w.wVal w.bVal (h[t]!))[0]!))

/-! ### Self-test: emit a forward on deterministic fixed inputs as f64 bits.
    Run with `lean --run Puffer/Net/MinGRU.lean`; `tools/mingru_ref.py` recomputes and checks. -/

/-- splitmix64 (same stream as the project's `rngNext`), for reproducible weight init. -/
def rngNext (s : UInt64) : UInt64 × UInt64 :=
  let s := s + 0x9E3779B97F4A7C15
  let z := s
  let z := (z ^^^ (z >>> 30)) * 0xBF58476D1CE4E5B9
  let z := (z ^^^ (z >>> 27)) * 0x94D049BB133111EB
  let z := z ^^^ (z >>> 31)
  (z, s)

/-- Uniform in `[-1, 1)` from a 64-bit word (top 53 bits → [0,1), then rescaled). -/
def uniform11 (word : UInt64) : Float :=
  (Float.ofNat (word >>> 11).toNat / 9007199254740992.0) * 2.0 - 1.0

/-- Random `[-1,1)` vector of length `n`. -/
def randVec (n : Nat) (rng0 : UInt64) : Vec × UInt64 := Id.run do
  let mut rng := rng0
  let mut v : Vec := Array.emptyWithCapacity n
  for _ in [0:n] do
    let (w, rng') := rngNext rng
    rng := rng'
    v := v.push (uniform11 w)
  return (v, rng)

/-- Random `[-1,1)` `rows×cols` matrix. -/
def randMat (rows cols : Nat) (rng0 : UInt64) : Mat × UInt64 := Id.run do
  let mut rng := rng0
  let mut m : Mat := Array.emptyWithCapacity rows
  for _ in [0:rows] do
    let (row, rng') := randVec cols rng
    rng := rng'
    m := m.push row
  return (m, rng)

@[inline] def b (x : Float) : String := toString (Float.toBits x)
def bV (xs : Vec) : String := "[" ++ String.intercalate "," (xs.toList.map b) ++ "]"
def bM (m : Mat) : String := "[" ++ String.intercalate "," (m.toList.map bV) ++ "]"

/-- Emit a `seqForward` on deterministic fixed inputs as one JSON line of f64 bit-patterns
    (weights, obs, logits, values). `Puffer/Net/MinGRUSelfTest.lean` wraps this as a `main` for
    `lean --run`; `tools/mingru_ref.py` recomputes and checks. Kept a plain `IO Unit` (not a
    top-level `main`) so the module can be imported by the trainer without a `main` clash. -/
def emitSelfTest : IO Unit := do
  let obsSize := 5
  let H := 4
  let numLayers := 2
  let A := 3
  let T := 3
  let mut rng : UInt64 := 0xC0FFEE123
  let (wEnc, r1) := randMat H obsSize rng; rng := r1
  let (bEnc, r2) := randVec H rng; rng := r2
  let mut layers : Array Mat := #[]
  for _ in [0:numLayers] do
    let (l, r) := randMat (3*H) H rng; rng := r
    layers := layers.push l
  let (wDec, r3) := randMat A H rng; rng := r3
  let (bDec, r4) := randVec A rng; rng := r4
  let (wVal, r5) := randMat 1 H rng; rng := r5
  let (bVal, r6) := randVec 1 rng; rng := r6
  let mut obsSeq : Array Vec := #[]
  for _ in [0:T] do
    let (o, r) := randVec obsSize rng; rng := r
    obsSeq := obsSeq.push o
  let w : Weights := { wEnc, bEnc, layers, wDec, bDec, wVal, bVal }
  let outs := seqForward w H numLayers obsSeq
  let logitsSeq := outs.map (·.1)
  let valueSeq := outs.map (·.2)
  let outsTrain := seqForwardTrain w H numLayers obsSeq       -- Heinsen parallel-scan forward_train
  let logitsTrain := outsTrain.map (·.1)
  let valueTrain := outsTrain.map (·.2)
  let layersJson := "[" ++ String.intercalate "," (layers.toList.map bM) ++ "]"
  IO.println ("{" ++
    "\"obsSize\":" ++ toString obsSize ++ ",\"H\":" ++ toString H ++
    ",\"numLayers\":" ++ toString numLayers ++ ",\"A\":" ++ toString A ++ ",\"T\":" ++ toString T ++
    ",\"wEnc\":" ++ bM wEnc ++ ",\"bEnc\":" ++ bV bEnc ++ ",\"layers\":" ++ layersJson ++
    ",\"wDec\":" ++ bM wDec ++ ",\"bDec\":" ++ bV bDec ++ ",\"wVal\":" ++ bM wVal ++ ",\"bVal\":" ++ bV bVal ++
    ",\"obsSeq\":" ++ bM obsSeq ++
    ",\"logits\":" ++ bM logitsSeq ++ ",\"values\":" ++ bV valueSeq ++
    ",\"logitsTrain\":" ++ bM logitsTrain ++ ",\"valuesTrain\":" ++ bV valueTrain ++ "}")

end Puffer.Net.MinGRU
