/-
Computable (Float) evaluators of the PROVEN error-bound formulas — the runtime side
of the real-number error-bound layer.

The theorems in `Puffer/RL/GAERuntime.lean`, `Puffer/Float/Net.lean`,
`Puffer/RL/MuonRuntime.lean` establish, over ℝ, that `|toReal(exec) − ℝspec| ≤ <bound>`.
Those bounds are `noncomputable` (they mention `toReal`). Here we evaluate the SAME
formulas in `Float` (with `toReal x` ↦ the value `x`), giving a runnable numeric error
interval. `Puffer/verify` emits these; `tools/verify_ref.py` checks the actual error
(vs exact ℝ, computed independently in Python) stays inside them.
-/
import Puffer.Float.Exec

namespace Puffer.FloatR

/-- Unit roundoff `2⁻⁵³` as a `Float`. -/
def u64F : Float := 1.0 / 9007199254740992.0

/-- Evaluator of `dotErrBnd` (see `Puffer/Float/Net.lean`). -/
def dotErrBndF : List Float → List Float → Float
  | [], _ => 0
  | _, [] => 0
  | x :: xs, w :: ws =>
      u64F * Float.abs (x * w + dotF xs ws) + u64F * Float.abs (x * w) + dotErrBndF xs ws

/-- Evaluator of `gaeErrBnd` (see `Puffer/RL/GAERuntime.lean`). -/
def gaeErrBndF (w : Float) : List Float → Float
  | [] => 0
  | δ :: rest =>
      u64F * Float.abs (δ + w * gaeHeadF w rest)
      + u64F * Float.abs (w * gaeHeadF w rest)
      + Float.abs w * gaeErrBndF w rest

/-- Evaluators of the `nsScalar` error chain (see `Puffer/RL/MuonRuntime.lean`). -/
def e2F (σ : Float) : Float := u64F * Float.abs (σ * σ)
def e4F (σ : Float) : Float :=
  u64F * Float.abs (σ * σ * (σ * σ)) + Float.abs (σ * σ) * e2F σ + Float.abs (σ * σ) * e2F σ
def eBSF (b σ : Float) : Float := u64F * Float.abs (b * (σ * σ)) + Float.abs b * e2F σ
def eCSF (c σ : Float) : Float := u64F * Float.abs (c * (σ * σ * (σ * σ))) + Float.abs c * e4F σ
def eT1F (a b σ : Float) : Float := u64F * Float.abs (a + b * (σ * σ)) + eBSF b σ
def eT2F (a b c σ : Float) : Float :=
  u64F * Float.abs (a + b * (σ * σ) + c * (σ * σ * (σ * σ))) + eT1F a b σ + eCSF c σ
def nsScalarErrBndF (a b c σ : Float) : Float :=
  u64F * Float.abs (σ * (a + b * (σ * σ) + c * (σ * σ * (σ * σ)))) + Float.abs σ * eT2F a b c σ

/-! ### Forward-pass composed bound evaluators (mirror `Puffer/RL/ForwardRuntime.lean`).

The MLP forward pass composes, per neuron, a linear unit `z = b + dotF w x`, a ReLU,
then a second linear unit for the logits. These evaluate the PROVEN composed error
bound in `Float` so `puffer verify-fwd` can emit it. -/

/-- Error of one pre-activation `z = b + dotF w x`: the dot bound plus the bias-add
    rounding. Mirrors `linZ_error`. -/
def z1ErrF (w : List Float) (b : Float) (x : List Float) : Float :=
  u64F * Float.abs (b + dotF w x) + dotErrBndF w x

/-- `Σᵢ |wᵢ| · eᵢ` — how per-entry input errors `e` on a vector propagate through a
    dot with weights `w` (each weight is exact, so it just scales its entry's error). -/
def sumAbsMulF : List Float → List Float → Float
  | [], _ => 0
  | _, [] => 0
  | w :: ws, e :: es => Float.abs w * e + sumAbsMulF ws es

/-- Error of one output logit `b₂ + dotF w₂ h` versus the ideal real logit
    `toReal b₂ + Σ toReal(w₂ᵢ)·hRᵢ` over the IDEAL real hidden `hR`: the layer-2 dot
    rounding + bias-add rounding, plus the propagated hidden error `εh`. Mirrors
    `logit_error`. -/
def logitErrF (w2 : List Float) (b2 : Float) (h εh : List Float) : Float :=
  u64F * Float.abs (b2 + dotF w2 h) + dotErrBndF w2 h + sumAbsMulF w2 εh

/-- The Float hidden layer `hⱼ = reluF (bⱼ + dotF Wⱼ x)`. -/
def fwdHidden (W1 : List (List Float)) (b1 : List Float) (x : List Float) : List Float :=
  (W1.zip b1).map (fun wb => reluF (wb.2 + dotF wb.1 x))

/-- Per-neuron hidden error `εhⱼ = z1ErrF Wⱼ bⱼ x` (ReLU propagates it with no growth). -/
def fwdHiddenErr (W1 : List (List Float)) (b1 : List Float) (x : List Float) : List Float :=
  (W1.zip b1).map (fun wb => z1ErrF wb.1 wb.2 x)

/-- The Float output logits `oₖ = b₂ₖ + dotF W₂ₖ h`. -/
def fwdLogits (W1 : List (List Float)) (b1 : List Float)
    (W2 : List (List Float)) (b2 : List Float) (x : List Float) : List Float :=
  (W2.zip b2).map (fun wb => wb.2 + dotF wb.1 (fwdHidden W1 b1 x))

/-- Per-logit end-to-end error bound (composes both layers + the propagated hidden error). -/
def fwdLogitErrs (W1 : List (List Float)) (b1 : List Float)
    (W2 : List (List Float)) (b2 : List Float) (x : List Float) : List Float :=
  (W2.zip b2).map (fun wb => logitErrF wb.1 wb.2 (fwdHidden W1 b1 x) (fwdHiddenErr W1 b1 x))

end Puffer.FloatR
