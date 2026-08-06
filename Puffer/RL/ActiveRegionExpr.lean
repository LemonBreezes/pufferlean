/-
# Active-region sufficient condition for the MLP capstone

C11's multi-layer capstone (`MLPExpr.mlp2_active_gradient_lipschitz`) is gated on the ACTIVE-REGION hypothesis
`∀ j, 0 < evalR (linLayerE w1 x b1 j) σ` — every hidden pre-activation strictly positive (so each ReLU is on
its active side and the gradient collapses to the linearization). That hypothesis is left abstract there; this
module turns it into a CHECKABLE condition on the weights, biases, and inputs.

The sufficient condition is **bias dominance**: a hidden neuron `out_i = Σⱼ W[i][j]·aⱼ + b_i` is active whenever
its bias value strictly exceeds the total magnitude of its weighted activations,
`Σⱼ |σ(w i j)·evalR(aⱼ)σ| < σ(b i)`. Then the worst-case negative contribution of the weighted sum still cannot
pull the pre-activation below zero. `linLayerE_layer_active` lifts this per-neuron condition to the whole layer,
producing EXACTLY C11's `hact`.

**Scope (honestly disclosed):** this is a SUFFICIENT (not necessary) condition — bias dominance guarantees the
hidden unit is active, discharging C11's active-region hypothesis from a checkable weight/bias/input condition.
It does NOT claim the condition holds for any particular trained network (whether a given net's hidden units are
active on a given input is data-dependent); it provides the checkable premise under which C11's bound applies.
-/
import Puffer.RL.LinearLayerExpr

open Puffer.FloatR.ADR
open Puffer.RL.LinearLayerExpr

namespace Puffer.RL.ActiveRegionExpr

/-- **Bias dominance ⟹ active neuron.** If the bias value strictly exceeds the total magnitude of the weighted
    activations, `Σⱼ |σ(w i j)·evalR(aⱼ)σ| < σ(b i)`, then the hidden pre-activation is strictly positive:
    `0 < evalR (linLayerE w a b i) σ`. (The weighted sum is `≥ −Σⱼ |·|` by the triangle inequality, so adding a
    bias larger than that magnitude keeps the pre-activation positive.) -/
theorem linLayerE_pos_of_bias_dominant {n m : Nat} (w : Fin n → Fin m → Nat) (a : Fin m → Expr)
    (b : Fin n → Nat) (i : Fin n) (σ : Nat → ℝ)
    (hdom : (∑ j : Fin m, |σ (w i j) * evalR (a j) σ|) < σ (b i)) :
    0 < evalR (linLayerE w a b i) σ := by
  rw [evalR_linLayerE]
  have h1 : -(∑ j : Fin m, |σ (w i j) * evalR (a j) σ|)
      ≤ ∑ j : Fin m, σ (w i j) * evalR (a j) σ :=
    (neg_le_neg (Finset.abs_sum_le_sum_abs _ _)).trans (neg_abs_le _)
  linarith [hdom]

/-- **Whole-layer active-region condition.** If every neuron of the layer is bias-dominant, then every hidden
    pre-activation is positive — EXACTLY the `hact : ∀ j, 0 < evalR (linLayerE w a b j) σ` hypothesis of C11's
    `mlp2_active_gradient_lipschitz`. This reduces that hypothesis to the per-neuron checkable condition. -/
theorem linLayerE_layer_active {n m : Nat} (w : Fin n → Fin m → Nat) (a : Fin m → Expr)
    (b : Fin n → Nat) (σ : Nat → ℝ)
    (hdom : ∀ i, (∑ j : Fin m, |σ (w i j) * evalR (a j) σ|) < σ (b i)) :
    ∀ i, 0 < evalR (linLayerE w a b i) σ :=
  fun i => linLayerE_pos_of_bias_dominant w a b i σ (hdom i)

/-- **Magnitude-bound sufficient condition.** If each weight satisfies `|σ(w i j)| ≤ ω`, each activation
    `|evalR(aⱼ)σ| ≤ α`, and the bias beats the worst-case weighted sum `m·ω·α < σ(b i)`, then the neuron is
    active. A coarser but purely numeric premise (no per-term sum): the total weighted magnitude is at most
    `m·ω·α`, dominated by the bias. -/
theorem linLayerE_pos_of_mag_bound {n m : Nat} (w : Fin n → Fin m → Nat) (a : Fin m → Expr)
    (b : Fin n → Nat) (i : Fin n) (σ : Nat → ℝ) (ω α : ℝ)
    (hw : ∀ j, |σ (w i j)| ≤ ω) (ha : ∀ j, |evalR (a j) σ| ≤ α)
    (hdom : (m : ℝ) * ω * α < σ (b i)) :
    0 < evalR (linLayerE w a b i) σ := by
  refine linLayerE_pos_of_bias_dominant w a b i σ (lt_of_le_of_lt ?_ hdom)
  have hterm : ∀ j : Fin m, |σ (w i j) * evalR (a j) σ| ≤ ω * α := by
    intro j
    rw [abs_mul]
    exact mul_le_mul (hw j) (ha j) (abs_nonneg _) ((abs_nonneg _).trans (hw j))
  calc (∑ j : Fin m, |σ (w i j) * evalR (a j) σ|)
      ≤ ∑ _j : Fin m, ω * α := Finset.sum_le_sum (fun j _ => hterm j)
    _ = (m : ℝ) * ω * α := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]; ring

end Puffer.RL.ActiveRegionExpr
