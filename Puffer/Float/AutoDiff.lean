/-
Reverse-mode automatic differentiation (scalar, tape-based), Mathlib-free.

Generalizes the hand-coded MLP backprop (`Puffer/RL/NNTrain.lean`) into a reusable
engine: build any expression from the primitive ops below (they record a Wengert
tape of `(value, [(parent, ∂self/∂parent)])`), then `grads` does one reverse sweep
to get the derivative w.r.t. every leaf. Correctness is validated against finite
differences (`puffer grad`); a formal `AD = derivative` theorem over ℝ is future work.

Executable and native-`Float`, so it plugs straight into the trainer.
-/
namespace Puffer.FloatR.AD

/-- Wengert tape: primal `val` of each node and its `(parent, localDeriv)` edges. -/
structure Tape where
  val : Array Float
  deps : Array (Array (Nat × Float))

def Tape.empty : Tape := { val := #[], deps := #[] }

/-- The build monad: threads the tape. -/
abbrev ADM := StateM Tape
/-- A node handle. -/
abbrev V := Nat

/-- Append a node with value `v` and derivative edges `ds`; return its handle. -/
def push (v : Float) (ds : Array (Nat × Float)) : ADM V := do
  let t ← get
  let id := t.val.size
  set (Tape.mk (t.val.push v) (t.deps.push ds))
  pure id

/-- A leaf (input / trainable parameter). -/
def leaf (v : Float) : ADM V := push v #[]
/-- A constant (a leaf whose gradient we ignore). -/
def const (c : Float) : ADM V := push c #[]

def add (a b : V) : ADM V := do let t ← get; push (t.val[a]! + t.val[b]!) #[(a, 1.0), (b, 1.0)]
def sub (a b : V) : ADM V := do let t ← get; push (t.val[a]! - t.val[b]!) #[(a, 1.0), (b, -1.0)]
def mul (a b : V) : ADM V := do
  let t ← get; let va := t.val[a]!; let vb := t.val[b]!
  push (va * vb) #[(a, vb), (b, va)]
def neg (a : V) : ADM V := do let t ← get; push (- t.val[a]!) #[(a, -1.0)]
/-- Multiply by a constant `c` (no gradient flows to `c`). -/
def scale (c : Float) (a : V) : ADM V := do let t ← get; push (c * t.val[a]!) #[(a, c)]
def relu (a : V) : ADM V := do
  let t ← get; let va := t.val[a]!; let g := if va > 0.0 then 1.0 else 0.0
  push (if va > 0.0 then va else 0.0) #[(a, g)]
def exp (a : V) : ADM V := do let t ← get; let e := Float.exp t.val[a]!; push e #[(a, e)]
def log (a : V) : ADM V := do let t ← get; let va := t.val[a]!; push (Float.log va) #[(a, 1.0 / va)]
/-- Logistic sigmoid `σ(x) = 1/(1+e^{-x})`; derivative `σ(1−σ)`. -/
def sigmoid (a : V) : ADM V := do
  let t ← get; let s := 1.0 / (1.0 + Float.exp (- t.val[a]!)); push s #[(a, s * (1.0 - s))]
/-- Hyperbolic tangent `tanh x`; derivative `1 − tanh²x`. -/
def tanh (a : V) : ADM V := do
  let t ← get; let th := Float.tanh t.val[a]!; push th #[(a, 1.0 - th * th)]
/-- `min(a,b)`; the gradient flows to the smaller argument (subgradient at ties). -/
def minV (a b : V) : ADM V := do
  let t ← get; let va := t.val[a]!; let vb := t.val[b]!
  if va ≤ vb then push va #[(a, 1.0), (b, 0.0)] else push vb #[(a, 0.0), (b, 1.0)]
/-- Clamp to the constant interval `[lo,hi]`; gradient is 1 strictly inside, else 0
    (as PufferLib's `fmaxf(lo, fminf(hi, x))` clip). -/
def clampC (lo hi : Float) (a : V) : ADM V := do
  let t ← get; let va := t.val[a]!
  let v := if va < lo then lo else if hi < va then hi else va
  let g := if lo < va ∧ va < hi then 1.0 else 0.0
  push v #[(a, g)]

/-- Dot product `Σ wᵢ·xᵢ` of two handle vectors. -/
def dotV (w x : Array V) : ADM V := do
  let mut acc ← const 0.0
  for i in [0:w.size] do
    let p ← mul w[i]! x[i]!
    acc ← add acc p
  return acc

/-- `log Σ exp xⱼ` (used for categorical log-probs). -/
def logSumExp (xs : Array V) : ADM V := do
  let mut s ← const 0.0
  for x in xs do
    let e ← exp x
    s ← add s e
  log s

/-- Run a tape-building computation from empty; return `(root handle, tape)`. -/
def run (m : ADM V) : V × Tape := m.run Tape.empty

/-- Primal value of a node. -/
def valueAt (t : Tape) (v : V) : Float := t.val[v]!

/-- One reverse sweep: returns `∂root/∂node` for every node (in particular, for the
    leaves = parameters). Assumes nodes were appended in evaluation order. -/
def grads (t : Tape) (root : V) : Array Float := Id.run do
  let n := t.val.size
  let mut adj := Array.replicate n 0.0
  adj := adj.set! root 1.0
  for i in [0:n] do
    let idx := n - 1 - i
    let a := adj[idx]!
    for e in t.deps[idx]! do
      adj := adj.set! e.1 (adj[e.1]! + a * e.2)
  return adj

/-! ### Validation helpers (AD vs finite differences)

For a `build : Array Float → ADM V` that creates its inputs as its first leaves
(in order), `adGrad` reads the reverse-mode gradient and `fdGrad` a central
finite-difference gradient; they should agree, validating the engine. -/

/-- Primal value of `build x`. -/
def evalPrimal (build : Array Float → ADM V) (x : Array Float) : Float :=
  let (root, t) := run (build x)
  valueAt t root

/-- Reverse-mode gradient w.r.t. the inputs (first `x.size` leaves). -/
def adGrad (build : Array Float → ADM V) (x : Array Float) : Array Float :=
  let (root, t) := run (build x)
  let g := grads t root
  (Array.range x.size).map (fun i => g[i]!)

/-- Central finite-difference gradient (a fresh, alias-free perturbation per input). -/
def fdGrad (build : Array Float → ADM V) (x : Array Float) (eps : Float) : Array Float :=
  (Array.range x.size).map (fun i =>
    let xp := (Array.range x.size).map (fun j => if j == i then x[j]! + eps else x[j]!)
    let xm := (Array.range x.size).map (fun j => if j == i then x[j]! - eps else x[j]!)
    (evalPrimal build xp - evalPrimal build xm) / (2.0 * eps))

end Puffer.FloatR.AD
