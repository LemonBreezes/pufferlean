/-
Executable Muon optimizer — Mathlib-free.

Mirrors `~/src/PufferLib/src/muon.cu`: Nesterov momentum, then (for 2D weight
matrices) orthogonalize the update by 5 Newton–Schulz iterations with the tuned
quintic schedule `muonCoeffs`, then a decoupled-weight-decay step. 1D params
(biases) skip orthogonalization.

The ℝ spec + convergence proof (`nsClassical_quadratic`, `nsScalar` singular-value
action) live in `Puffer/Optim/Muon.lean`; this is the runnable counterpart.

Sign note: `newtonSchulz` is odd, so ascending in `+ortho(∇obj)` (⟨g,ortho(g)⟩ = Σσ ≥ 0
increases the objective) equals PufferLib's `w − lr·ortho(∇loss)`. We keep the ascent form.
-/
namespace Puffer.FloatR.Muon

abbrev Mat := Array (Array Float)

def matmul (A B : Mat) : Mat := Id.run do
  let m := A.size
  let k := if m == 0 then 0 else A[0]!.size
  let n := if B.size == 0 then 0 else B[0]!.size
  let mut C : Mat := #[]
  for i in [0:m] do
    let mut row : Array Float := #[]
    for j in [0:n] do
      let mut s := 0.0
      for l in [0:k] do
        s := s + A[i]![l]! * B[l]![j]!
      row := row.push s
    C := C.push row
  return C

def transpose (A : Mat) : Mat := Id.run do
  let m := A.size
  let n := if m == 0 then 0 else A[0]!.size
  let mut T : Mat := #[]
  for j in [0:n] do
    let mut row : Array Float := #[]
    for i in [0:m] do
      row := row.push A[i]![j]!
    T := T.push row
  return T

def frobNorm (A : Mat) : Float :=
  Float.sqrt (A.foldl (fun s row => s + row.foldl (fun s2 x => s2 + x * x) 0.0) 0.0)

/-- The maximum absolute entry of a matrix (`max x (-x) = |x|` folded over all entries), seeded at `0.0`.
    A runnable magnitude bound computed from the matrix itself — no caller-supplied threshold. -/
def matMaxAbs (X : Mat) : Float :=
  X.foldl (fun acc row => row.foldl (fun a x => max a (max x (-x))) acc) 0.0

def scalarMul (c : Float) (X : Mat) : Mat := X.map (·.map (c * ·))

/-- `a·X + b·Y` (same shape). -/
def matLin (a : Float) (X : Mat) (b : Float) (Y : Mat) : Mat :=
  (Array.range X.size).map (fun i => (Array.range X[i]!.size).map (fun j => a * X[i]![j]! + b * Y[i]![j]!))

/-- `a·X + b·Y + c·Z` (same shape). -/
def lincomb3 (a : Float) (X : Mat) (b : Float) (Y : Mat) (c : Float) (Z : Mat) : Mat :=
  (Array.range X.size).map (fun i => (Array.range X[i]!.size).map (fun j =>
    a * X[i]![j]! + b * Y[i]![j]! + c * Z[i]![j]!))

/-- PufferLib's tuned 5-step Newton–Schulz coefficient schedule (muon.cu:78–84). -/
def muonCoeffs : Array (Float × Float × Float) :=
  #[(4.0848, -6.8946, 2.9270), (3.9505, -6.3029, 2.6377), (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046), (2.8366, -3.0525, 1.2012)]

/-- Newton–Schulz orthogonalization over an ARBITRARY coefficient schedule `coeffs` (same body as
    `newtonSchulz`, but the iteration table is a parameter). `newtonSchulz = newtonSchulzWith muonCoeffs`
    (`rfl`); the runnable `coeffListOk` check verifies a supplied `coeffs` bit-matches `muonCoeffs`, so the
    proven O(1) bound transports to any table that passes the check. -/
def newtonSchulzWith (coeffs : Array (Float × Float × Float)) (X0 : Mat) (eps : Float) : Mat := Id.run do
  let mut X := scalarMul (1.0 / (frobNorm X0 + eps)) X0
  for coef in coeffs do
    let (a, b, c) := coef
    let rows := X.size
    let cols := X[0]!.size
    if rows ≤ cols then
      let A := matmul X (transpose X)       -- rows×rows
      let AX := matmul A X
      let AAX := matmul A AX
      X := lincomb3 a X b AX c AAX
    else
      let A := matmul (transpose X) X       -- cols×cols
      let XA := matmul X A
      let XAA := matmul XA A
      X := lincomb3 a X b XA c XAA
  return X

def newtonSchulz (X0 : Mat) (eps : Float) : Mat := newtonSchulzWith muonCoeffs X0 eps

/-- Muon step for a 2D weight matrix: Nesterov momentum → orthogonalize → decoupled
    weight decay. Ascent form (feeds ∇objective). Returns `(newW, newMomentum)`. -/
def stepMat (W grad mom : Mat) (lr wd mu eps : Float) : Mat × Mat :=
  let newMom := matLin mu mom 1.0 grad          -- m ← μ·m + g
  let update := matLin 1.0 grad mu newMom       -- Nesterov: g + μ·m
  let ortho := newtonSchulz update eps
  let rows := Float.ofNat W.size
  let cols := Float.ofNat W[0]!.size
  let scale := Float.sqrt (max 1.0 (rows / cols))
  (matLin (1.0 - lr * wd) W (lr * scale) ortho, newMom)

/-- Muon step for a 1D param (bias): Nesterov momentum + weight decay, no orthogonalization. -/
def stepVec (b grad mom : Array Float) (lr wd mu : Float) : Array Float × Array Float :=
  let newMom := (Array.range b.size).map (fun i => mu * mom[i]! + grad[i]!)
  let update := (Array.range b.size).map (fun i => grad[i]! + mu * newMom[i]!)
  (Array.range b.size |>.map (fun i => b[i]! * (1.0 - lr * wd) + lr * update[i]!), newMom)

end Puffer.FloatR.Muon
