/-
# Arbitrary-depth MLP: the active-region gradient-Lipschitz composition step

C11 (`Puffer.RL.MLPExpr`) proved the active-region gradient-Lipschitz for a 2-layer ReLU MLP and noted "the
congruence machinery is general (any depth); the capstone is stated for 2 layers … deeper stacks iterate the
same congruence + active-passthrough step." This module makes that iteration explicit: it isolates the general
composition step and instantiates it at 3 layers (beyond C11's 2).

The engine is `relu_linLayer_match`: bundle "activations `a` match their `Smooth` linearization `a'` in value AND
gradient" as `LayerMatch σ k a a'`; then on the active region an active-relu linear layer over `a` produces
activations that again match the linear layer over `a'` — via C11's layer congruence + active-relu pass-through.
This is a fixed-point-preserving step: starting from the `Smooth` input (which matches itself), applying it once
per hidden layer keeps the invariant, so a net of ANY depth collapses (value + gradient, on the all-active
region) to its fully-linear (relu-stripped) `Smooth` linearization, which inherits C4's gradient-Lipschitz.

`mlp3_active_gradient_lipschitz` is the 3-layer capstone: `|derivR (mlp3E …) σ k − derivR (mlp3E …) σ' k| ≤
dLip R (mlp3Lin …)·δ`, given every hidden pre-activation positive at both `σ` and `σ'`. Its proof applies the
step TWICE (two hidden layers) — the same two lines would extend to any depth by adding more applications.

**Scope (honestly disclosed):** the composition STEP (`relu_linLayer_match`) and the linearization/smoothness are
fully general (any depth, any dimensions); the gradient-Lipschitz CAPSTONE is instantiated at 3 layers (C11 was
2) as a concrete witness that depth composes — a fixed `n`-layer capstone for symbolic `n` would need a
dependently-typed heterogeneous layer stack (each layer changes the `Fin` dimension), which the explicit
instances sidestep. The active-region hypothesis (all hidden pre-activations `> 0` at both points) is REAL and
load-bearing: off it a hidden ReLU crosses its kink and the gradient genuinely jumps (C7's intrinsic
non-Lipschitzness), exactly as in C11.
-/
import Puffer.RL.MLPExpr

open Puffer.FloatR.ADR
open Puffer.RL.LinearLayerExpr
open Puffer.RL.MLPExpr

namespace Puffer.RL.MLPDepthExpr

/-- **Layer match.** "Activations `a` agree with their linearization `a'` at `(σ, k)`" — both value and gradient
    match pointwise. The invariant propagated layer-by-layer through a deep net on its active region. -/
def LayerMatch {n : Nat} (σ : Nat → ℝ) (k : Nat) (a a' : Fin n → Expr) : Prop :=
  (∀ j, evalR (a j) σ = evalR (a' j) σ) ∧ (∀ j, derivR (a j) σ k = derivR (a' j) σ k)

/-- The base case: the `Smooth` input matches itself. -/
theorem LayerMatch.rfl' {n : Nat} (σ : Nat → ℝ) (k : Nat) (a : Fin n → Expr) : LayerMatch σ k a a :=
  ⟨fun _ => rfl, fun _ => rfl⟩

/-- **THE GENERAL COMPOSITION STEP (iterates to any depth).** If activations `a` match their linearization `a'`
    (value + gradient at `σ`), then on the active region (`0 < evalR (linLayerE w a b i) σ` for every output `i`)
    the active-relu linear layer over `a` matches the linear layer over `a'` — the invariant is preserved across
    one more layer. Combines C11's layer congruence (`linLayerE_evalR_congr`/`linLayerE_derivR_congr`) with the
    active-relu pass-through (`relu_active_evalR`/`relu_active_derivR`). -/
theorem relu_linLayer_match {n m : Nat} (w : Fin n → Fin m → Nat) (a a' : Fin m → Expr) (b : Fin n → Nat)
    (σ : Nat → ℝ) (k : Nat) (hmatch : LayerMatch σ k a a')
    (hact : ∀ i, 0 < evalR (linLayerE w a b i) σ) :
    LayerMatch σ k (fun i => Expr.relu (linLayerE w a b i)) (fun i => linLayerE w a' b i) := by
  obtain ⟨hval, hder⟩ := hmatch
  refine ⟨fun i => ?_, fun i => ?_⟩
  · rw [relu_active_evalR _ σ (hact i)]; exact linLayerE_evalR_congr w a a' b i σ hval
  · rw [relu_active_derivR _ σ k (hact i)]; exact linLayerE_derivR_congr w a a' b i σ k hval hder

/-- **A 3-layer ReLU MLP** (input → relu-linLayer → relu-linLayer → output linLayer). -/
def mlp3E {m hd1 hd2 o : Nat}
    (w1 : Fin hd1 → Fin m → Nat) (b1 : Fin hd1 → Nat)
    (w2 : Fin hd2 → Fin hd1 → Nat) (b2 : Fin hd2 → Nat)
    (w3 : Fin o → Fin hd2 → Nat) (b3 : Fin o → Nat) (x : Fin m → Expr) : Fin o → Expr :=
  linLayerE w3 (fun j => Expr.relu (linLayerE w2 (fun l => Expr.relu (linLayerE w1 x b1 l)) b2 j)) b3

/-- The relu-stripped LINEARIZATION of `mlp3E` (a composition of three linear layers). This IS `Smooth`. -/
def mlp3Lin {m hd1 hd2 o : Nat}
    (w1 : Fin hd1 → Fin m → Nat) (b1 : Fin hd1 → Nat)
    (w2 : Fin hd2 → Fin hd1 → Nat) (b2 : Fin hd2 → Nat)
    (w3 : Fin o → Fin hd2 → Nat) (b3 : Fin o → Nat) (x : Fin m → Expr) : Fin o → Expr :=
  linLayerE w3 (fun j => linLayerE w2 (fun l => linLayerE w1 x b1 l) b2 j) b3

/-- The 3-layer linearization is `Smooth` (given `Smooth` inputs) — three nested `linLayerE_smooth`. -/
theorem mlp3Lin_smooth {m hd1 hd2 o : Nat}
    (w1 : Fin hd1 → Fin m → Nat) (b1 : Fin hd1 → Nat)
    (w2 : Fin hd2 → Fin hd1 → Nat) (b2 : Fin hd2 → Nat)
    (w3 : Fin o → Fin hd2 → Nat) (b3 : Fin o → Nat) (x : Fin m → Expr) (i : Fin o)
    (hx : ∀ l, Smooth (x l)) : Smooth (mlp3Lin w1 b1 w2 b2 w3 b3 x i) :=
  linLayerE_smooth w3 _ b3 i (fun j =>
    linLayerE_smooth w2 _ b2 j (fun l => linLayerE_smooth w1 x b1 l hx))

/-- **Active-region reduction (3 layers).** On the region where every hidden pre-activation is positive (layer 1
    over the input, layer 2 over the layer-1 relu outputs), the 3-layer ReLU MLP's gradient equals its
    linearization's — by applying the general composition step TWICE. -/
theorem mlp3_active_deriv_eq {m hd1 hd2 o : Nat}
    (w1 : Fin hd1 → Fin m → Nat) (b1 : Fin hd1 → Nat)
    (w2 : Fin hd2 → Fin hd1 → Nat) (b2 : Fin hd2 → Nat)
    (w3 : Fin o → Fin hd2 → Nat) (b3 : Fin o → Nat) (x : Fin m → Expr) (i : Fin o) (σ : Nat → ℝ) (k : Nat)
    (hact1 : ∀ l, 0 < evalR (linLayerE w1 x b1 l) σ)
    (hact2 : ∀ j, 0 < evalR (linLayerE w2 (fun l => Expr.relu (linLayerE w1 x b1 l)) b2 j) σ) :
    derivR (mlp3E w1 b1 w2 b2 w3 b3 x i) σ k = derivR (mlp3Lin w1 b1 w2 b2 w3 b3 x i) σ k := by
  have m1 := relu_linLayer_match w1 x x b1 σ k (LayerMatch.rfl' σ k x) hact1
  have m2 := relu_linLayer_match w2 _ _ b2 σ k m1 hact2
  exact linLayerE_derivR_congr w3 _ _ b3 i σ k m2.1 m2.2

/-- **CAPSTONE: the 3-layer ReLU MLP is gradient-Lipschitz on the all-hidden-active region** (beyond C11's 2
    layers), with the linearization's concrete C4 constant `dLip R (mlp3Lin i)`:
    `|derivR (mlp3E i) σ k − derivR (mlp3E i) σ' k| ≤ dLip R (mlp3Lin i)·δ`, provided every hidden pre-activation
    is positive at BOTH `σ` and `σ'`. Same shape as C11's `mlp2_active_gradient_lipschitz`, one layer deeper — a
    witness that depth composes by iterating `relu_linLayer_match`. -/
theorem mlp3_active_gradient_lipschitz {m hd1 hd2 o : Nat}
    (w1 : Fin hd1 → Fin m → Nat) (b1 : Fin hd1 → Nat)
    (w2 : Fin hd2 → Fin hd1 → Nat) (b2 : Fin hd2 → Nat)
    (w3 : Fin o → Fin hd2 → Nat) (b3 : Fin o → Nat) (x : Fin m → Expr) (i : Fin o)
    (σ σ' : Nat → ℝ) (R δ : ℝ) (k : Nat) (hx : ∀ l, Smooth (x l))
    (hσ : ∀ i, |σ i| ≤ R) (hσ' : ∀ i, |σ' i| ≤ R) (hδ : ∀ i, |σ i - σ' i| ≤ δ) (hR : 0 ≤ R)
    (hact1 : ∀ l, 0 < evalR (linLayerE w1 x b1 l) σ)
    (hact2 : ∀ j, 0 < evalR (linLayerE w2 (fun l => Expr.relu (linLayerE w1 x b1 l)) b2 j) σ)
    (hact1' : ∀ l, 0 < evalR (linLayerE w1 x b1 l) σ')
    (hact2' : ∀ j, 0 < evalR (linLayerE w2 (fun l => Expr.relu (linLayerE w1 x b1 l)) b2 j) σ') :
    |derivR (mlp3E w1 b1 w2 b2 w3 b3 x i) σ k - derivR (mlp3E w1 b1 w2 b2 w3 b3 x i) σ' k|
      ≤ dLip R (mlp3Lin w1 b1 w2 b2 w3 b3 x i) * δ := by
  rw [mlp3_active_deriv_eq w1 b1 w2 b2 w3 b3 x i σ k hact1 hact2,
      mlp3_active_deriv_eq w1 b1 w2 b2 w3 b3 x i σ' k hact1' hact2']
  exact derivR_lip (mlp3Lin_smooth w1 b1 w2 b2 w3 b3 x i hx) σ σ' R δ k hσ hσ' hδ hR

end Puffer.RL.MLPDepthExpr
