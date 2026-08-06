/-
# Log-softmax VALUE budgets: completing the per-log-prob budget discharge (C17's partner)

C17 (`LogSoftmaxBudgetExpr`) discharged the log-softmax DERIVATIVE budgets `Dm`/`Dl` by composing C4 + C6 + C9.
This module discharges the remaining VALUE budgets — `M` (value magnitude) and `Lv` (value Lipschitz) — which
C15's entropy `expLogTerm_lip` and C16's ratio also consume. Together C17 + this file make all four per-log-prob
budgets (`M`, `Lv`, `Dm`, `Dl`) concrete for the actual softmax.

The decomposition is the value analogue of C17's: `evalR (logSoftmaxE chosen logits) σ = evalR chosen σ −
Real.log (evalR (expSumE logits) σ)`. So:
* `logSoftmaxE_value_lip` (the `Lv` budget, CLEAN — needs only the floor `c`): `≤ (vLip R chosen + vLip R
  (expSumE logits)/c)·δ`, via C4's `evalR_lip` on the chosen logit and on the `Smooth` partition, composed with
  the new `log_lipschitz_on_floor` (`Real.log` is `(1/c)`-Lipschitz on `[c,∞)`).
* `logSoftmaxE_value_mag` (the `M` budget): `≤ vMag R chosen + max |log c| |log U|`, needing BOTH the floor `c`
  and a partition UPPER bound `U` (since `|log(Σexp)|` needs the partition bounded away from `0` AND above).
* `expSumE_upper` supplies a concrete upper bound `Σexp ≤ Σⱼ exp(vMag R logitⱼ)` (each term via `evalR_mag`).
* `logSoftmaxE_value_lip_floored` — the FULLY CONCRETE `Lv`, with the floor discharged via C9's `expSumE_floor`
  (`c = exp(−vMag R e)`), so no free floor hypothesis remains.

**Scope (honestly disclosed):** the `Lv` budget needs only the floor (clean, and the floored corollary makes it
fully concrete via C9). The `M` budget additionally needs a partition UPPER bound `U` — supplied concretely by
`expSumE_upper` (`Σⱼ exp(vMag R logitⱼ)`); a fully-floored-and-ceiled `M` corollary composing both is a
mechanical instantiation (left to the caller — the constant `max |log c| |log U|` is not simplified here).
Budgets are over the region `|σ i| ≤ R` with `Smooth` logits (C10 linear layers). This completes the DERIVATIVE
budgets of C17 with the VALUE budgets, so all of `M`/`Lv`/`Dm`/`Dl` for the real log-softmax are now discharged.
-/
import Puffer.RL.LogSoftmaxBudgetExpr

open Puffer.FloatR.ADR
open Puffer.RL.SoftmaxExpr (expSumE logPartitionE logSoftmaxE)
open Puffer.RL.LogSoftmaxBudgetExpr (expSumE_smooth)

namespace Puffer.RL.LogSoftmaxValueBudgetExpr

/-- **`Real.log` is `(1/c)`-Lipschitz on the ray `[c, ∞)`**: `|log x − log y| ≤ |x − y|/c` for `x, y ≥ c > 0`.
    The value-Lipschitz analogue of C6's `1/x`-derivative bound. Proved from `Real.log_le_sub_one_of_pos`
    (`log t ≤ t − 1`): `log a − log b = log(a/b) ≤ a/b − 1 = (a−b)/b ≤ (a−b)/c` for `b ≤ a`, symmetrized. -/
theorem log_lipschitz_on_floor (x y c : ℝ) (hc : 0 < c) (hx : c ≤ x) (hy : c ≤ y) :
    |Real.log x - Real.log y| ≤ |x - y| / c := by
  have key : ∀ a b : ℝ, 0 < b → b ≤ a → c ≤ b → Real.log a - Real.log b ≤ (a - b) / c := by
    intro a b hb0 hba hcb
    have ha0 : 0 < a := lt_of_lt_of_le hb0 hba
    have hlog : Real.log a - Real.log b = Real.log (a / b) := (Real.log_div ha0.ne' hb0.ne').symm
    rw [hlog]
    have h1 : Real.log (a / b) ≤ a / b - 1 := Real.log_le_sub_one_of_pos (div_pos ha0 hb0)
    have h2 : a / b - 1 = (a - b) / b := by field_simp
    rw [h2] at h1
    refine h1.trans ?_
    have hab : 0 ≤ a - b := by linarith
    gcongr
  rcases le_total y x with h | h
  · have hxy : 0 ≤ x - y := by linarith
    rw [abs_of_nonneg hxy,
      abs_of_nonneg (by linarith [Real.log_le_log (lt_of_lt_of_le hc hy) h] : 0 ≤ Real.log x - Real.log y)]
    exact key x y (lt_of_lt_of_le hc hy) h hy
  · have hxy : x - y ≤ 0 := by linarith
    rw [abs_of_nonpos hxy,
      abs_of_nonpos (by linarith [Real.log_le_log (lt_of_lt_of_le hc hx) h] : Real.log x - Real.log y ≤ 0),
      neg_sub, neg_sub]
    exact key y x (lt_of_lt_of_le hc hx) h hx

/-- **Log-softmax value Lipschitz (`Lv` budget).** Over the region `|σ i|, |σ' i| ≤ R` with the partition
    floored `c ≤ evalR (expSumE logits) σ, σ'`, `|evalR (logSoftmaxE chosen logits) σ − evalR (logSoftmaxE chosen
    logits) σ'| ≤ (vLip R chosen + vLip R (expSumE logits)/c)·δ`. Composes C4's `evalR_lip` on the chosen logit
    and on the `Smooth` partition, with `log_lipschitz_on_floor`, via triangle on `evalR (log-softmax) = evalR
    chosen − log(partition)`. Needs only the floor (no upper bound). -/
theorem logSoftmaxE_value_lip (chosen : Expr) (logits : List Expr) (hch : Smooth chosen)
    (hlog : ∀ lp ∈ logits, Smooth lp)
    (σ σ' : Nat → ℝ) (R δ c : ℝ)
    (hσ : ∀ i, |σ i| ≤ R) (hσ' : ∀ i, |σ' i| ≤ R) (hδ : ∀ i, |σ i - σ' i| ≤ δ) (hR : 0 ≤ R)
    (hc : 0 < c)
    (hfloor : c ≤ evalR (expSumE logits) σ) (hfloor' : c ≤ evalR (expSumE logits) σ') :
    |evalR (logSoftmaxE chosen logits) σ - evalR (logSoftmaxE chosen logits) σ'|
      ≤ (vLip R chosen + vLip R (expSumE logits) / c) * δ := by
  have hchL : |evalR chosen σ - evalR chosen σ'| ≤ vLip R chosen * δ :=
    evalR_lip hch σ σ' R δ hσ hσ' hδ hR
  have hpartL : |evalR (expSumE logits) σ - evalR (expSumE logits) σ'| ≤ vLip R (expSumE logits) * δ :=
    evalR_lip (expSumE_smooth logits hlog) σ σ' R δ hσ hσ' hδ hR
  have hlogL : |Real.log (evalR (expSumE logits) σ) - Real.log (evalR (expSumE logits) σ')|
      ≤ (vLip R (expSumE logits) * δ) / c :=
    (log_lipschitz_on_floor _ _ c hc hfloor hfloor').trans (by gcongr)
  have hev : ∀ τ : Nat → ℝ, evalR (logSoftmaxE chosen logits) τ
      = evalR chosen τ - Real.log (evalR (expSumE logits) τ) := fun τ => by
    simp only [logSoftmaxE, logPartitionE, evalR]
  rw [hev σ, hev σ']
  have key : (evalR chosen σ - Real.log (evalR (expSumE logits) σ))
        - (evalR chosen σ' - Real.log (evalR (expSumE logits) σ'))
      = (evalR chosen σ - evalR chosen σ')
        + (-(Real.log (evalR (expSumE logits) σ) - Real.log (evalR (expSumE logits) σ'))) := by ring
  rw [key]
  calc |(evalR chosen σ - evalR chosen σ')
          + (-(Real.log (evalR (expSumE logits) σ) - Real.log (evalR (expSumE logits) σ')))|
      ≤ |evalR chosen σ - evalR chosen σ'|
          + |-(Real.log (evalR (expSumE logits) σ) - Real.log (evalR (expSumE logits) σ'))| := abs_add_le _ _
    _ = |evalR chosen σ - evalR chosen σ'|
          + |Real.log (evalR (expSumE logits) σ) - Real.log (evalR (expSumE logits) σ')| := by rw [abs_neg]
    _ ≤ vLip R chosen * δ + (vLip R (expSumE logits) * δ) / c := add_le_add hchL hlogL
    _ = (vLip R chosen + vLip R (expSumE logits) / c) * δ := by ring

/-- **Log-softmax value magnitude (`M` budget).** With the partition bounded `c ≤ evalR (expSumE logits) σ ≤ U`
    (`0 < c`), `|evalR (logSoftmaxE chosen logits) σ| ≤ vMag R chosen + max |log c| |log U|`. The chosen logit is
    bounded by C4's `evalR_mag`; the log-partition `|log(Σexp)|` is bounded by `max |log c| |log U|` since the
    partition lies in `[c, U]` and `log` is monotone. Needs a partition UPPER bound `U` (unlike `Lv`). -/
theorem logSoftmaxE_value_mag (chosen : Expr) (logits : List Expr) (hch : Smooth chosen)
    (σ : Nat → ℝ) (R c U : ℝ) (hσ : ∀ i, |σ i| ≤ R)
    (hc : 0 < c) (hfloor : c ≤ evalR (expSumE logits) σ) (hceil : evalR (expSumE logits) σ ≤ U) :
    |evalR (logSoftmaxE chosen logits) σ| ≤ vMag R chosen + max |Real.log c| |Real.log U| := by
  have hev : evalR (logSoftmaxE chosen logits) σ
      = evalR chosen σ - Real.log (evalR (expSumE logits) σ) := by
    simp only [logSoftmaxE, logPartitionE, evalR]
  rw [hev]
  have hchM : |evalR chosen σ| ≤ vMag R chosen := evalR_mag hch σ R hσ
  have hpartpos : 0 < evalR (expSumE logits) σ := lt_of_lt_of_le hc hfloor
  have hlogmag : |Real.log (evalR (expSumE logits) σ)| ≤ max |Real.log c| |Real.log U| := by
    have hlo : Real.log c ≤ Real.log (evalR (expSumE logits) σ) := Real.log_le_log hc hfloor
    have hhi : Real.log (evalR (expSumE logits) σ) ≤ Real.log U := Real.log_le_log hpartpos hceil
    rw [abs_le]
    refine ⟨?_, ?_⟩
    · calc -max |Real.log c| |Real.log U| ≤ -|Real.log c| := neg_le_neg (le_max_left _ _)
        _ ≤ Real.log c := neg_abs_le _
        _ ≤ _ := hlo
    · calc Real.log (evalR (expSumE logits) σ) ≤ Real.log U := hhi
        _ ≤ |Real.log U| := le_abs_self _
        _ ≤ _ := le_max_right _ _
  calc |evalR chosen σ - Real.log (evalR (expSumE logits) σ)|
      ≤ |evalR chosen σ| + |Real.log (evalR (expSumE logits) σ)| := abs_sub _ _
    _ ≤ vMag R chosen + max |Real.log c| |Real.log U| := add_le_add hchM hlogmag

/-- **Concrete partition upper bound.** `evalR (expSumE logits) σ ≤ Σⱼ exp(vMag R logitⱼ)` — each `exp(evalR
    logitⱼ σ) ≤ exp(vMag R logitⱼ)` (via `evalR_mag` on the `Smooth` logit). Supplies the `U` for
    `logSoftmaxE_value_mag`. -/
theorem expSumE_upper (logits : List Expr) (hlog : ∀ lp ∈ logits, Smooth lp)
    (σ : Nat → ℝ) (R : ℝ) (hσ : ∀ i, |σ i| ≤ R) :
    evalR (expSumE logits) σ ≤ (logits.map (fun e => Real.exp (vMag R e))).sum := by
  rw [Puffer.RL.SoftmaxExpr.evalR_expSumE]
  apply List.sum_le_sum
  intro e he
  exact Real.exp_le_exp.mpr ((le_abs_self _).trans (evalR_mag (hlog e he) σ R hσ))

/-- **Fully concrete `Lv`.** For a nonempty logit list `e :: es` with `Smooth` head `e`, the floor is discharged
    via C9's `expSumE_floor` (`c = exp(−vMag R e) > 0`), so the value-Lipschitz bound is fully computable from the
    network's `Smooth` logits with no free floor hypothesis. -/
theorem logSoftmaxE_value_lip_floored (chosen e : Expr) (es : List Expr) (hch : Smooth chosen)
    (he : Smooth e) (hes : ∀ lp ∈ es, Smooth lp)
    (σ σ' : Nat → ℝ) (R δ : ℝ)
    (hσ : ∀ i, |σ i| ≤ R) (hσ' : ∀ i, |σ' i| ≤ R) (hδ : ∀ i, |σ i - σ' i| ≤ δ) (hR : 0 ≤ R) :
    |evalR (logSoftmaxE chosen (e :: es)) σ - evalR (logSoftmaxE chosen (e :: es)) σ'|
      ≤ (vLip R chosen + vLip R (expSumE (e :: es)) / Real.exp (-(vMag R e))) * δ :=
  logSoftmaxE_value_lip chosen (e :: es) hch
    (fun lp hlp => (List.mem_cons.mp hlp).elim (fun h => h ▸ he) (fun h => hes lp h))
    σ σ' R δ (Real.exp (-(vMag R e))) hσ hσ' hδ hR (Real.exp_pos _)
    (Puffer.RL.SoftmaxExpr.expSumE_floor e es he σ R hσ)
    (Puffer.RL.SoftmaxExpr.expSumE_floor e es he σ' R hσ')

end Puffer.RL.LogSoftmaxValueBudgetExpr
