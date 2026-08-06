/-
# Symbolic-n MLP: arbitrary-depth active-region gradient-Lipschitz (uniform width)

C20 (`MLPDepthExpr`) proved the active-region gradient-Lipschitz for a fixed 3-layer ReLU MLP, noting that a
symbolic-`n`-layer statement was blocked by dependent types (each layer changes the `Fin` dimension). This module
removes that block by fixing a UNIFORM WIDTH `d`: every hidden layer maps `Fin d → Fin d`, so the network is a
`List` of layers `[(weights, bias)]` that can be FOLDED and INDUCTED over — a genuine arbitrary-depth result.

The whole net is `mlpNE layers x = List.foldl (relu ∘ linLayerE) x layers` (a ReLU after every layer);
`mlpNLin layers x` is the relu-stripped LINEARIZATION (a composition of linear layers, hence `Smooth`). The
active-region condition is the recursive predicate `AllActive σ layers x` — "at each layer the pre-activation over
the running (relu) activations is positive", threaded through the fold exactly as the forward pass runs.

The proof is C20's general step (`relu_linLayer_match`) iterated by list induction (`mlpN_active_eq_aux`): the
invariant `LayerMatch` (relu-net activations agree with their linearization in value AND gradient) is preserved
across every layer, so on the all-active region the whole net's value and gradient collapse to the `Smooth`
linearization — which inherits C4's `derivR_lip`. `mlpN_active_gradient_lipschitz` is the capstone, holding for a
list of ANY length:

    |derivR (mlpNE layers x i) σ k − derivR (mlpNE layers x i) σ' k|  ≤  dLip R (mlpNLin layers x i)·δ

given `AllActive` at both `σ` and `σ'`, `Smooth` input, and the region bounds.

**Scope (honestly disclosed):** UNIFORM width `d` — every layer is `Fin d → Fin d` (this is what lets a `List`
replace C20's dependently-typed heterogeneous stack; genuinely-varying widths still need dependent types). A ReLU
is applied after EVERY layer (a stack of relu-linLayers); a final linear readout would compose as one more
`linLayerE` (not covered here). The `AllActive` hypothesis at BOTH points is REAL and load-bearing: off it a
hidden ReLU crosses its kink and the gradient genuinely jumps (C7's intrinsic non-Lipschitzness), exactly as in
C11/C20. This is the true depth generalization: `layers` is an arbitrary `List`, so the capstone covers any depth.
-/
import Puffer.RL.MLPDepthExpr

open Puffer.FloatR.ADR
open Puffer.RL.LinearLayerExpr
open Puffer.RL.MLPExpr
open Puffer.RL.MLPDepthExpr

namespace Puffer.RL.MLPDepthNExpr

/-- A uniform-width MLP layer: a weight-index map `Fin d → Fin d → Nat` and a bias-index map `Fin d → Nat`. -/
abbrev MLPLayer (d : Nat) : Type := (Fin d → Fin d → Nat) × (Fin d → Nat)

/-- Apply one ReLU layer to the running activations: `aᵢ ↦ relu(Σⱼ W[i][j]·aⱼ + bᵢ)`. -/
def applyReluLayer {d : Nat} (a : Fin d → Expr) (L : MLPLayer d) : Fin d → Expr :=
  fun i => Expr.relu (linLayerE L.1 a L.2 i)

/-- Apply one LINEAR layer (no ReLU) to the running activations — the linearization's step. -/
def applyLinLayer {d : Nat} (a : Fin d → Expr) (L : MLPLayer d) : Fin d → Expr :=
  fun i => linLayerE L.1 a L.2 i

/-- **The `n`-layer ReLU MLP**: fold `relu ∘ linLayerE` over the layer list (a ReLU after every layer). -/
def mlpNE {d : Nat} (layers : List (MLPLayer d)) (x : Fin d → Expr) : Fin d → Expr :=
  layers.foldl applyReluLayer x

/-- The **relu-stripped linearization** of `mlpNE` — a composition of linear layers, hence `Smooth`. -/
def mlpNLin {d : Nat} (layers : List (MLPLayer d)) (x : Fin d → Expr) : Fin d → Expr :=
  layers.foldl applyLinLayer x

/-- **All-intermediate-active predicate**, threaded through the fold: at each layer the pre-activation over the
    running (relu) activations is strictly positive. Mirrors the forward pass — `applyReluLayer a L` is the next
    running activation after layer `L`. This is the arbitrary-depth analogue of C20's `hact1`/`hact2`. -/
def AllActive {d : Nat} (σ : Nat → ℝ) : List (MLPLayer d) → (Fin d → Expr) → Prop
  | [], _ => True
  | L :: rest, a => (∀ i, 0 < evalR (linLayerE L.1 a L.2 i) σ) ∧ AllActive σ rest (applyReluLayer a L)

/-- The linearization of an `n`-layer net over `Smooth` inputs is `Smooth` (induction over layers, each
    `linLayerE_smooth`). So C4's `derivR_lip` applies to it at any depth. -/
theorem mlpNLin_smooth {d : Nat} (layers : List (MLPLayer d)) :
    ∀ (x : Fin d → Expr) (i : Fin d), (∀ l, Smooth (x l)) → Smooth (mlpNLin layers x i) := by
  induction layers with
  | nil => intro x i hx; exact hx i
  | cons L rest ih =>
      intro x i hx
      exact ih (applyLinLayer x L) i (fun j => linLayerE_smooth L.1 x L.2 j hx)

/-- **The arbitrary-depth reduction (list induction).** Given a `LayerMatch` between the running relu activations
    `a` and their linearization `a'`, and `AllActive` for the remaining layers, the whole remaining net's relu
    activations match the linearization's — value AND gradient. Iterates C20's `relu_linLayer_match` once per
    layer via induction over the layer list; `LayerMatch.rfl'` seeds it at the input. -/
theorem mlpN_active_eq_aux {d : Nat} (σ : Nat → ℝ) (k : Nat) (layers : List (MLPLayer d)) :
    ∀ (a a' : Fin d → Expr), LayerMatch σ k a a' → AllActive σ layers a →
      LayerMatch σ k (mlpNE layers a) (mlpNLin layers a') := by
  induction layers with
  | nil => intro a a' hm _; exact hm
  | cons L rest ih =>
      intro a a' hm hact
      obtain ⟨h1, h2⟩ := hact
      exact ih (applyReluLayer a L) (applyLinLayer a' L)
        (relu_linLayer_match L.1 a a' L.2 σ k hm h1) h2

/-- **The `n`-layer net matches its linearization on the all-active region** (value + gradient), for a list of
    ANY length — instantiates the reduction at the input via `LayerMatch.rfl'`. -/
theorem mlpN_active_eq {d : Nat} (σ : Nat → ℝ) (k : Nat) (layers : List (MLPLayer d)) (x : Fin d → Expr)
    (hact : AllActive σ layers x) :
    LayerMatch σ k (mlpNE layers x) (mlpNLin layers x) :=
  mlpN_active_eq_aux σ k layers x x (LayerMatch.rfl' σ k x) hact

/-- **CAPSTONE: arbitrary-depth ReLU MLP is gradient-Lipschitz on the all-active region** — the symbolic-`n`
    generalization of C20 (which fixed 3 layers). For a layer list of ANY length, with `Smooth` input and every
    intermediate pre-activation positive at BOTH `σ` and `σ'`:
    `|derivR (mlpNE layers x i) σ k − derivR (mlpNE layers x i) σ' k| ≤ dLip R (mlpNLin layers x i)·δ`. On the
    all-active region the net collapses to its `Smooth` linearization (`mlpN_active_eq`), which inherits C4's
    `derivR_lip`. Uniform width `d`; the `AllActive` hypothesis is load-bearing (off it a ReLU crosses its kink). -/
theorem mlpN_active_gradient_lipschitz {d : Nat} (layers : List (MLPLayer d)) (x : Fin d → Expr) (i : Fin d)
    (σ σ' : Nat → ℝ) (R δ : ℝ) (k : Nat) (hx : ∀ l, Smooth (x l))
    (hσ : ∀ j, |σ j| ≤ R) (hσ' : ∀ j, |σ' j| ≤ R) (hδ : ∀ j, |σ j - σ' j| ≤ δ) (hR : 0 ≤ R)
    (hact : AllActive σ layers x) (hact' : AllActive σ' layers x) :
    |derivR (mlpNE layers x i) σ k - derivR (mlpNE layers x i) σ' k|
      ≤ dLip R (mlpNLin layers x i) * δ := by
  rw [(mlpN_active_eq σ k layers x hact).2 i, (mlpN_active_eq σ' k layers x hact').2 i]
  exact derivR_lip (mlpNLin_smooth layers x i hx) σ σ' R δ k hσ hσ' hδ hR

end Puffer.RL.MLPDepthNExpr
