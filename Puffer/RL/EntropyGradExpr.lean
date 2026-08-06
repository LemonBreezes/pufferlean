/-
# Entropy gradient-Lipschitz (the last per-term bound of C14)

The C14 total-objective gradient-Lipschitz assembly left the entropy term's bound `Lent` as a hypothesis. This
module supplies it. The categorical entropy `entropyCatE logps = −Σᵢ exp(logpᵢ)·logpᵢ` (C12) carries `log`
nodes (each log-prob `logpᵢ` is a log-softmax), so it is NOT `Smooth` and C4 does not apply directly; instead
the bound is assembled from the C4/C5 helpers (`abs_mul_sub_mul_le`, `exp_abs_sub_le`) applied to each `exp·log`
product, given per-log-prob value/derivative magnitude and Lipschitz budgets.

Two structural simplifications carry the proof:
* The negation cancels: `derivR (entropyCatE logps) = −derivR (crossTermE logps)`, so the entropy's gradient
  variation equals the cross term's.
* Each term's derivative collapses: `derivR (mul (exp lp) lp) = exp(lp)·(∂lp)·(lp+1)` (`derivR_expLogTerm`) —
  a triple product bounded via two `abs_mul_sub_mul_le` chains plus `exp_abs_sub_le` (the C5 `exp`-node
  local-Lipschitz).

Given uniform per-log-prob budgets `|lp| ≤ M`, `|Δlp| ≤ Lv·δ`, `|∂lp| ≤ Dm`, `|Δ∂lp| ≤ Dl·δ`, the per-term
bound (`expLogTerm_lip`) is `exp(M)·((M+1)·Dl + (M+2)·Dm·Lv)·δ`, and the entropy over `n = length logps` terms
is bounded by `n · C` (`entropyCatE_gradient_lipschitz`).

**Scope (honestly disclosed):** the four per-log-prob budgets `M`/`Lv`/`Dm`/`Dl` are taken as HYPOTHESES here —
they are the value/derivative magnitude and Lipschitz constants of each log-softmax `logpᵢ`, which decompose
downstream into C4 (the raw logit, `Smooth`) and C6 (the `log`-partition, with the positive floor supplied FREE
by C9's `expSumE_floor`). So this is the entropy's gradient-Lipschitz REDUCED to (and, per term, PROVEN from)
those budgets; discharging the budgets for a concrete softmax is the composition of C4 + C6 + C9. The uniform
`C` and `n·C` form is honest for a softmax (a shared partition floor gives all log-probs common budgets).
-/
import Puffer.RL.ValueEntropyExpr

open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)
open Puffer.RL.ValueEntropyExpr (crossTermE entropyCatE)

namespace Puffer.RL.EntropyGradExpr

/-- The derivative of a single entropy term collapses to a triple product:
    `derivR (exp(lp)·lp) = exp(lp)·(∂lp)·(lp + 1)` (product rule + `exp` chain rule). -/
theorem derivR_expLogTerm (lp : Expr) (σ : Nat → ℝ) (k : Nat) :
    derivR (.mul (.exp lp) lp) σ k = Real.exp (evalR lp σ) * derivR lp σ k * (evalR lp σ + 1) := by
  simp only [derivR, evalR]; ring

/-- **`exp(lp)·(∂lp)` is Lipschitz** in the parameters: `|exp(Lσ)·∂Lσ − exp(Lσ')·∂Lσ'| ≤ exp(M)·(Dl + Dm·Lv)·δ`,
    given `lp` capped by `M` and its derivative bounded by `Dm` with Lipschitz `Dl·δ` (the C5 `exp`-node
    derivative-Lipschitz, for an abstract — not necessarily `Smooth` — argument). -/
theorem expDeriv_lip (lp : Expr) (σ σ' : Nat → ℝ) (k : Nat) (M Lv Dm Dl δ : ℝ)
    (hMσ : evalR lp σ ≤ M) (hMσ' : evalR lp σ' ≤ M)
    (hLv : |evalR lp σ - evalR lp σ'| ≤ Lv * δ)
    (hDmσ' : |derivR lp σ' k| ≤ Dm)
    (hDl : |derivR lp σ k - derivR lp σ' k| ≤ Dl * δ)
    (hDmn : 0 ≤ Dm) :
    |Real.exp (evalR lp σ) * derivR lp σ k - Real.exp (evalR lp σ') * derivR lp σ' k|
      ≤ Real.exp M * (Dl + Dm * Lv) * δ := by
  have hExpσ : Real.exp (evalR lp σ) ≤ Real.exp M := Real.exp_le_exp.mpr hMσ
  have hExpdiff : |Real.exp (evalR lp σ) - Real.exp (evalR lp σ')| ≤ Real.exp M * (Lv * δ) := by
    calc |Real.exp (evalR lp σ) - Real.exp (evalR lp σ')|
        ≤ Real.exp M * |evalR lp σ - evalR lp σ'| := exp_abs_sub_le _ _ _ hMσ hMσ'
      _ ≤ Real.exp M * (Lv * δ) := mul_le_mul_of_nonneg_left hLv (Real.exp_pos _).le
  calc |Real.exp (evalR lp σ) * derivR lp σ k - Real.exp (evalR lp σ') * derivR lp σ' k|
      ≤ |Real.exp (evalR lp σ)| * |derivR lp σ k - derivR lp σ' k|
          + |derivR lp σ' k| * |Real.exp (evalR lp σ) - Real.exp (evalR lp σ')| := abs_mul_sub_mul_le _ _ _ _
    _ ≤ Real.exp M * (Dl * δ) + Dm * (Real.exp M * (Lv * δ)) := by
        apply add_le_add
        · rw [abs_of_pos (Real.exp_pos _)]; exact mul_le_mul hExpσ hDl (abs_nonneg _) (Real.exp_pos _).le
        · exact mul_le_mul hDmσ' hExpdiff (abs_nonneg _) hDmn
    _ = Real.exp M * (Dl + Dm * Lv) * δ := by ring

/-- **Per-term entropy gradient-Lipschitz.** For a log-prob `lp` with `|lp| ≤ M`, value-Lipschitz `Lv·δ`,
    derivative magnitude `Dm`, and derivative-Lipschitz `Dl·δ` on the region, the entropy term `exp(lp)·lp` is
    gradient-Lipschitz with constant `exp(M)·((M+1)·Dl + (M+2)·Dm·Lv)`:
    `|derivR (exp(lp)·lp) σ k − derivR (exp(lp)·lp) σ' k| ≤ exp(M)·((M+1)·Dl + (M+2)·Dm·Lv)·δ`. The triple
    product `exp(lp)·(∂lp)·(lp+1)` is bounded via `abs_mul_sub_mul_le` (grouping `(exp·∂lp)·(lp+1)`) with the
    `exp·∂lp` factor handled by `expDeriv_lip`. -/
theorem expLogTerm_lip (lp : Expr) (σ σ' : Nat → ℝ) (k : Nat) (M Lv Dm Dl δ : ℝ)
    (hMσ : |evalR lp σ| ≤ M) (hMσ' : |evalR lp σ'| ≤ M)
    (hLv : |evalR lp σ - evalR lp σ'| ≤ Lv * δ)
    (hDmσ : |derivR lp σ k| ≤ Dm) (hDmσ' : |derivR lp σ' k| ≤ Dm)
    (hDl : |derivR lp σ k - derivR lp σ' k| ≤ Dl * δ)
    (hDmn : 0 ≤ Dm) :
    |derivR (.mul (.exp lp) lp) σ k - derivR (.mul (.exp lp) lp) σ' k|
      ≤ Real.exp M * ((M + 1) * Dl + (M + 2) * Dm * Lv) * δ := by
  rw [derivR_expLogTerm, derivR_expLogTerm]
  have hM0 : 0 ≤ M := (abs_nonneg _).trans hMσ
  have hMσle : evalR lp σ ≤ M := (le_abs_self _).trans hMσ
  have hMσ'le : evalR lp σ' ≤ M := (le_abs_self _).trans hMσ'
  have hAσ : |Real.exp (evalR lp σ) * derivR lp σ k| ≤ Real.exp M * Dm := by
    rw [abs_mul, abs_of_pos (Real.exp_pos _)]
    exact mul_le_mul (Real.exp_le_exp.mpr hMσle) hDmσ (abs_nonneg _) (Real.exp_pos _).le
  have hBσ' : |evalR lp σ' + 1| ≤ M + 1 := by
    calc |evalR lp σ' + 1| ≤ |evalR lp σ'| + |(1:ℝ)| := abs_add_le _ _
      _ ≤ M + 1 := by rw [abs_one]; linarith [hMσ']
  have hBdiff : |(evalR lp σ + 1) - (evalR lp σ' + 1)| ≤ Lv * δ := by
    rw [show (evalR lp σ + 1) - (evalR lp σ' + 1) = evalR lp σ - evalR lp σ' from by ring]; exact hLv
  have hAdiff := expDeriv_lip lp σ σ' k M Lv Dm Dl δ hMσle hMσ'le hLv hDmσ' hDl hDmn
  calc |Real.exp (evalR lp σ) * derivR lp σ k * (evalR lp σ + 1)
        - Real.exp (evalR lp σ') * derivR lp σ' k * (evalR lp σ' + 1)|
      ≤ |Real.exp (evalR lp σ) * derivR lp σ k| * |(evalR lp σ + 1) - (evalR lp σ' + 1)|
        + |evalR lp σ' + 1|
          * |Real.exp (evalR lp σ) * derivR lp σ k - Real.exp (evalR lp σ') * derivR lp σ' k|
          := abs_mul_sub_mul_le _ _ _ _
    _ ≤ (Real.exp M * Dm) * (Lv * δ) + (M + 1) * (Real.exp M * (Dl + Dm * Lv) * δ) := by
        apply add_le_add
        · exact mul_le_mul hAσ hBdiff (abs_nonneg _) (mul_nonneg (Real.exp_pos _).le hDmn)
        · exact mul_le_mul hBσ' hAdiff (abs_nonneg _) (by linarith)
    _ = Real.exp M * ((M + 1) * Dl + (M + 2) * Dm * Lv) * δ := by ring

/-- **Sum assembly.** The cross term's gradient variation is bounded by the sum of its per-term variations (a
    triangle inequality over the `add`-fold that builds `crossTermE`). -/
theorem crossTermE_grad_diff_le (logps : List Expr) (σ σ' : Nat → ℝ) (k : Nat) :
    |derivR (crossTermE logps) σ k - derivR (crossTermE logps) σ' k|
      ≤ (logps.map (fun lp =>
          |derivR (.mul (.exp lp) lp) σ k - derivR (.mul (.exp lp) lp) σ' k|)).sum := by
  induction logps with
  | nil => simp [crossTermE, derivR]
  | cons lp rest ih =>
      have e1 : derivR (crossTermE (lp :: rest)) σ k
          = derivR (.mul (.exp lp) lp) σ k + derivR (crossTermE rest) σ k := rfl
      have e2 : derivR (crossTermE (lp :: rest)) σ' k
          = derivR (.mul (.exp lp) lp) σ' k + derivR (crossTermE rest) σ' k := rfl
      rw [e1, e2, List.map_cons, List.sum_cons]
      have key : (derivR (.mul (.exp lp) lp) σ k + derivR (crossTermE rest) σ k)
            - (derivR (.mul (.exp lp) lp) σ' k + derivR (crossTermE rest) σ' k)
          = (derivR (.mul (.exp lp) lp) σ k - derivR (.mul (.exp lp) lp) σ' k)
            + (derivR (crossTermE rest) σ k - derivR (crossTermE rest) σ' k) := by ring
      rw [key]
      exact (abs_add_le _ _).trans (add_le_add le_rfl ih)

/-- **Entropy gradient-Lipschitz.** Given a UNIFORM per-log-prob-term gradient-Lipschitz bound `C` (each
    `exp(lpᵢ)·lpᵢ` term varies by `≤ C`, e.g. `C = exp(M)·((M+1)·Dl + (M+2)·Dm·Lv)·δ` from `expLogTerm_lip` under
    shared softmax budgets), the categorical entropy `entropyCatE logps` (`= −Σ`) is gradient-Lipschitz with
    bound `n · C` over `n = length logps` terms — the negation of the cross term cancels, then the sum bound
    applies. This is exactly the `Lent`-hypothesis that C14's `ppoTotalObjE_gradient_lipschitz` requires. -/
theorem entropyCatE_gradient_lipschitz (logps : List Expr) (σ σ' : Nat → ℝ) (k : Nat) (C : ℝ)
    (hterm : ∀ lp ∈ logps,
      |derivR (.mul (.exp lp) lp) σ k - derivR (.mul (.exp lp) lp) σ' k| ≤ C) :
    |derivR (entropyCatE logps) σ k - derivR (entropyCatE logps) σ' k|
      ≤ (logps.length : ℝ) * C := by
  have hneg : derivR (entropyCatE logps) σ k - derivR (entropyCatE logps) σ' k
      = -(derivR (crossTermE logps) σ k - derivR (crossTermE logps) σ' k) := by
    simp only [entropyCatE, derivR]; ring
  rw [hneg, abs_neg]
  refine (crossTermE_grad_diff_le logps σ σ' k).trans ?_
  have hsum : (logps.map (fun lp =>
      |derivR (.mul (.exp lp) lp) σ k - derivR (.mul (.exp lp) lp) σ' k|)).sum
        ≤ (logps.map (fun _ => C)).sum :=
    List.sum_le_sum (fun x hx => hterm x hx)
  refine hsum.trans (le_of_eq ?_)
  rw [List.map_const', List.sum_replicate, nsmul_eq_mul]

end Puffer.RL.EntropyGradExpr
