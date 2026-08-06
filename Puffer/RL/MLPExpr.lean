/-
# Multi-layer MLP composition → `Expr`, and its active-region gradient-Lipschitz

Stacks the C10 linear-layer builder with `relu` into a multi-layer MLP `Expr`, and settles its
gradient-Lipschitz — the point where C4 (linear, `Smooth`) and C7 (relu, away-from-kink) combine ACROSS layers.

A deep ReLU MLP is genuinely NOT globally `Smooth`: each `relu` leaves the fragment, so the next layer's
activations aren't `Smooth` and the recursive budgets don't pass through. The honest structural result is the
**active-region reduction**: on the region where every hidden ReLU is active (its pre-activation `> 0`), each
hidden unit is locally its linear pre-activation, so the whole MLP's value AND gradient collapse to those of
the LINEARIZED (relu-stripped) net — which IS `Smooth` (a composition of linear layers), inheriting C4's
concrete gradient-Lipschitz constant with no free hypotheses beyond "all hidden units active".

The machinery:
* **Layer congruence** (`linLayerE_evalR_congr`/`linLayerE_derivR_congr`): a linear layer depends on its input
  activations ONLY through their value and gradient — via the closed forms `evalR_linLayerE`/`derivR_linLayerE`.
* **Active relu pass-through** (`relu_active_evalR`/`relu_active_derivR`): the pointwise value/gradient of
  `relu e` equals `e`'s when `evalR e σ > 0`. (Unlike C7's two-point Lipschitz bound, the pointwise equality
  needs NO smoothness — just positivity at the point.)

Composed for a 2-layer MLP (`mlp2E = linLayerE ∘ relu ∘ linLayerE`, the puffer policy shape): on the
all-hidden-active region, `mlp2E` matches its linearization `mlp2Lin` (value + gradient), `mlp2Lin` is `Smooth`,
and hence (`mlp2_active_gradient_lipschitz`) the ReLU MLP is gradient-Lipschitz there with `dLip R (mlp2Lin i)`.

**Scope (honestly disclosed):** the active-region hypothesis (`∀ hidden j, 0 < evalR (pre-activation) σ`, at
BOTH points) is REAL and load-bearing — off it, a hidden ReLU crosses its kink and the gradient genuinely jumps
(C7's intrinsic non-Lipschitzness). The congruence machinery is general (any depth); the capstone is stated for
2 layers (the representative policy net); deeper stacks iterate the same congruence + active-passthrough step.
The `dLip R (mlp2Lin i)` constant is the C4 budget of the linear composition — concrete and computable.
-/
import Puffer.RL.LinearLayerExpr

open Puffer.FloatR.ADR
open Puffer.RL.LinearLayerExpr

namespace Puffer.RL.MLPExpr

/-- Closed form for a neuron's gradient: `derivR (dotBiasE wa b) σ k = Σ ([wⱼ=k]·aⱼ + Wⱼ·∂aⱼ) + [b=k]`
    (product rule on each `mul (var wⱼ) aⱼ` term). -/
theorem derivR_dotBiasE (wa : List (Nat × Expr)) (b : Nat) (σ : Nat → ℝ) (k : Nat) :
    derivR (dotBiasE wa b) σ k
      = (wa.map (fun p => (if p.1 = k then (1:ℝ) else 0) * evalR p.2 σ + σ p.1 * derivR p.2 σ k)).sum
        + (if b = k then (1:ℝ) else 0) := by
  induction wa with
  | nil => simp [dotBiasE, derivR]
  | cons p rest ih =>
      obtain ⟨wj, aj⟩ := p
      simp only [dotBiasE, derivR, evalR, List.map_cons, List.sum_cons, ih]
      ring

/-- Closed form for a layer neuron's gradient over `Fin m`. -/
theorem derivR_linLayerE {n m : Nat} (w : Fin n → Fin m → Nat) (a : Fin m → Expr) (b : Fin n → Nat)
    (i : Fin n) (σ : Nat → ℝ) (k : Nat) :
    derivR (linLayerE w a b i) σ k
      = (Finset.univ.sum (fun j : Fin m =>
          (if w i j = k then (1:ℝ) else 0) * evalR (a j) σ + σ (w i j) * derivR (a j) σ k))
        + (if b i = k then (1:ℝ) else 0) := by
  simp only [linLayerE, derivR_dotBiasE, List.map_ofFn, List.sum_ofFn, Function.comp]

/-- **Layer value-congruence.** A linear layer depends on its input activations only through their VALUE: if
    `evalR (a j) σ = evalR (a' j) σ` for all `j`, the layer outputs agree. -/
theorem linLayerE_evalR_congr {n m : Nat} (w : Fin n → Fin m → Nat) (a a' : Fin m → Expr)
    (b : Fin n → Nat) (i : Fin n) (σ : Nat → ℝ) (hval : ∀ j, evalR (a j) σ = evalR (a' j) σ) :
    evalR (linLayerE w a b i) σ = evalR (linLayerE w a' b i) σ := by
  rw [evalR_linLayerE, evalR_linLayerE]
  congr 1
  exact Finset.sum_congr rfl (fun j _ => by rw [hval j])

/-- **Layer gradient-congruence.** A linear layer depends on its input activations only through their VALUE and
    GRADIENT: if both agree pointwise, the layer's gradient agrees. This is the composition principle — it lets
    a substitution deep in the net (e.g. active-relu ↦ its pre-activation) propagate up through the next layer. -/
theorem linLayerE_derivR_congr {n m : Nat} (w : Fin n → Fin m → Nat) (a a' : Fin m → Expr)
    (b : Fin n → Nat) (i : Fin n) (σ : Nat → ℝ) (k : Nat)
    (hval : ∀ j, evalR (a j) σ = evalR (a' j) σ) (hder : ∀ j, derivR (a j) σ k = derivR (a' j) σ k) :
    derivR (linLayerE w a b i) σ k = derivR (linLayerE w a' b i) σ k := by
  rw [derivR_linLayerE, derivR_linLayerE]
  congr 1
  exact Finset.sum_congr rfl (fun j _ => by rw [hval j, hder j])

/-- **Active relu value pass-through.** `evalR (relu e) σ = evalR e σ` when `evalR e σ > 0` (relu is locally
    the identity on its active side). No smoothness needed — a pointwise equality. -/
theorem relu_active_evalR (e : Expr) (σ : Nat → ℝ) (hpos : 0 < evalR e σ) :
    evalR (.relu e) σ = evalR e σ := by
  simp only [evalR]; exact max_eq_left hpos.le

/-- **Active relu gradient pass-through.** `derivR (relu e) σ k = derivR e σ k` when `evalR e σ > 0`. No
    smoothness needed (contrast C7's two-point Lipschitz bound) — just positivity at the point. -/
theorem relu_active_derivR (e : Expr) (σ : Nat → ℝ) (k : Nat) (hpos : 0 < evalR e σ) :
    derivR (.relu e) σ k = derivR e σ k := by
  simp only [derivR, if_pos hpos]

/-- **A 2-layer ReLU MLP as an `Expr`**: `out_i = Σⱼ W2[i][j]·relu(Σₗ W1[j][l]·xₗ + b1_j) + b2_i` (one hidden
    layer + output layer — the puffer policy shape). Produces the logit vector `Fin o → Expr`. -/
def mlp2E {m h o : Nat} (w1 : Fin h → Fin m → Nat) (b1 : Fin h → Nat)
    (w2 : Fin o → Fin h → Nat) (b2 : Fin o → Nat) (x : Fin m → Expr) : Fin o → Expr :=
  linLayerE w2 (fun j => .relu (linLayerE w1 x b1 j)) b2

/-- The relu-stripped LINEARIZATION of `mlp2E` (composition of two linear layers). This IS `Smooth`. -/
def mlp2Lin {m h o : Nat} (w1 : Fin h → Fin m → Nat) (b1 : Fin h → Nat)
    (w2 : Fin o → Fin h → Nat) (b2 : Fin o → Nat) (x : Fin m → Expr) : Fin o → Expr :=
  linLayerE w2 (fun j => linLayerE w1 x b1 j) b2

/-- The linearization is `Smooth` (given `Smooth` inputs) — a composition of linear layers, so C4's `derivR_lip`
    gives it a concrete gradient-Lipschitz constant. -/
theorem mlp2Lin_smooth {m h o : Nat} (w1 : Fin h → Fin m → Nat) (b1 : Fin h → Nat)
    (w2 : Fin o → Fin h → Nat) (b2 : Fin o → Nat) (x : Fin m → Expr) (i : Fin o)
    (hx : ∀ l, Smooth (x l)) : Smooth (mlp2Lin w1 b1 w2 b2 x i) :=
  linLayerE_smooth w2 _ b2 i (fun j => linLayerE_smooth w1 x b1 j hx)

/-- **Active-region reduction.** On the region where every hidden pre-activation is positive
    (`∀ j, 0 < evalR (linLayerE w1 x b1 j) σ`), the ReLU MLP equals its linearization in BOTH value and gradient:
    each hidden `relu` is locally its pre-activation (active pass-through), and the output layer depends on the
    hidden units only through value+gradient (layer congruence). -/
theorem mlp2_active_eq {m h o : Nat} (w1 : Fin h → Fin m → Nat) (b1 : Fin h → Nat)
    (w2 : Fin o → Fin h → Nat) (b2 : Fin o → Nat) (x : Fin m → Expr) (i : Fin o) (σ : Nat → ℝ) (k : Nat)
    (hact : ∀ j, 0 < evalR (linLayerE w1 x b1 j) σ) :
    evalR (mlp2E w1 b1 w2 b2 x i) σ = evalR (mlp2Lin w1 b1 w2 b2 x i) σ ∧
    derivR (mlp2E w1 b1 w2 b2 x i) σ k = derivR (mlp2Lin w1 b1 w2 b2 x i) σ k := by
  refine ⟨?_, ?_⟩
  · exact linLayerE_evalR_congr w2 _ _ b2 i σ (fun j => relu_active_evalR _ σ (hact j))
  · exact linLayerE_derivR_congr w2 _ _ b2 i σ k
      (fun j => relu_active_evalR _ σ (hact j)) (fun j => relu_active_derivR _ σ k (hact j))

/-- **CAPSTONE: the 2-layer ReLU MLP is gradient-Lipschitz on the all-hidden-active region**, with the
    linearization's concrete C4 constant `dLip R (mlp2Lin i)`:
    `|derivR (mlp2E i) σ k − derivR (mlp2E i) σ' k| ≤ dLip R (mlp2Lin i) · δ`, provided every hidden
    pre-activation is positive at BOTH `σ` and `σ'`. Combines C10 (linear layers), the active-relu pass-through,
    and C4 (`derivR_lip` on the `Smooth` linearization) across layers. Off the active region a hidden ReLU
    crosses its kink and the gradient genuinely jumps (C7's intrinsic non-Lipschitzness) — hence the hypothesis. -/
theorem mlp2_active_gradient_lipschitz {m h o : Nat} (w1 : Fin h → Fin m → Nat) (b1 : Fin h → Nat)
    (w2 : Fin o → Fin h → Nat) (b2 : Fin o → Nat) (x : Fin m → Expr) (i : Fin o)
    (σ σ' : Nat → ℝ) (R δ : ℝ) (k : Nat)
    (hx : ∀ l, Smooth (x l))
    (hσ : ∀ i, |σ i| ≤ R) (hσ' : ∀ i, |σ' i| ≤ R) (hδ : ∀ i, |σ i - σ' i| ≤ δ) (hR : 0 ≤ R)
    (hact : ∀ j, 0 < evalR (linLayerE w1 x b1 j) σ)
    (hact' : ∀ j, 0 < evalR (linLayerE w1 x b1 j) σ') :
    |derivR (mlp2E w1 b1 w2 b2 x i) σ k - derivR (mlp2E w1 b1 w2 b2 x i) σ' k|
      ≤ dLip R (mlp2Lin w1 b1 w2 b2 x i) * δ := by
  rw [(mlp2_active_eq w1 b1 w2 b2 x i σ k hact).2, (mlp2_active_eq w1 b1 w2 b2 x i σ' k hact').2]
  exact derivR_lip (mlp2Lin_smooth w1 b1 w2 b2 x i hx) σ σ' R δ k hσ hσ' hδ hR

end Puffer.RL.MLPExpr
