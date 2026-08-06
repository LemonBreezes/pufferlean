/-
# Concrete log-softmax budget bundle: all four M/Lv/Dm/Dl for a softmax log-prob

C17 (`LogSoftmaxBudgetExpr`) discharged the log-softmax DERIVATIVE budgets `Dm`/`Dl`, and C18
(`LogSoftmaxValueBudgetExpr`) the VALUE budgets `M`/`Lv` — but `M` was left with the partition floor `c` and
upper bound `U` as free hypotheses, and the four budgets were spread across separate lemmas with separate
hypotheses. This module (1) supplies the still-missing FULLY-CONCRETE `M` — floor `c = exp(−vMag R e)` via C9's
`expSumE_floor` and ceiling `U = Σⱼ exp(vMag R logitⱼ)` via C18's `expSumE_upper`, so no free `c`/`U` remains —
and (2) packages all four concrete budgets into a single `logSoftmaxE_budgets` lemma: the clean interface that
C15's entropy (`expLogTerm_lip`) and C19's ratio (`ratioE_deriv_lip`) consume.

The four budget constants (`budgetM`/`budgetLv`/`budgetDm`/`budgetDl`) are computed from the network's `Smooth`
logits over the input region `|σ i| ≤ R`, with a SHARED partition floor `c = exp(−vMag R e)` (`e` the head
logit) — the softmax discharging its own `log`-floor (C9's payoff). `logSoftmaxE_budgets` delivers, for a
softmax log-prob `logSoftmaxE chosen (e :: es)`, the value magnitude at BOTH points, the value-Lipschitz, the
derivative magnitude at BOTH points, and the derivative-Lipschitz — exactly the shapes C15/C19 need (C15's
`expLogTerm_lip` needs `|evalR| ≤ M`, `|Δevalr| ≤ Lv·δ`, `|derivR| ≤ Dm`, `|Δderivr| ≤ Dl·δ` at both σ and σ').

**Scope (honestly disclosed):** the budgets are concrete for a NONEMPTY logit list `e :: es` with `Smooth`
head `e` (needed by C9's floor) and `Smooth` chosen + tail. The constants are the raw nested expressions from
C17/C18 (not algebraically simplified — e.g. `budgetDl` carries the `/c²` from the log-node); the floor
`c = exp(−vMag R e)` is a valid but generally loose lower bound (any single `exp` term of the partition). These
compose downstream into C15's entropy and C19's ratio bounds for the actual softmax-MLP policy.
-/
import Puffer.RL.LogSoftmaxValueBudgetExpr

open Puffer.FloatR.ADR
open Puffer.RL.SoftmaxExpr (expSumE logSoftmaxE expSumE_floor)
open Puffer.RL.LogSoftmaxBudgetExpr (logSoftmaxE_deriv_mag logSoftmaxE_deriv_lip_floored)
open Puffer.RL.LogSoftmaxValueBudgetExpr
  (logSoftmaxE_value_mag expSumE_upper logSoftmaxE_value_lip_floored)

namespace Puffer.RL.LogSoftmaxBudgetBundle

/-- Membership helper: `∀ lp ∈ e :: es, Smooth lp` from `Smooth e` (head) + `Smooth` tail. -/
private theorem cons_smooth {e : Expr} {es : List Expr} (he : Smooth e) (hes : ∀ lp ∈ es, Smooth lp) :
    ∀ lp ∈ e :: es, Smooth lp :=
  fun lp hlp => (List.mem_cons.mp hlp).elim (fun h => h ▸ he) (fun h => hes lp h)

/-- **Concrete value-magnitude budget `M`** for `logSoftmaxE chosen (e :: es)`: `vMag R chosen + max (vMag R e)
    |log(Σⱼ exp(vMag R logitⱼ))|` — the shared partition floor `c = exp(−vMag R e)` (`|log c| = vMag R e`) and
    ceiling `U = Σⱼ exp(vMag R logitⱼ)`. -/
noncomputable def budgetM (chosen e : Expr) (es : List Expr) (R : ℝ) : ℝ :=
  vMag R chosen + max (vMag R e) |Real.log (((e :: es).map (fun l => Real.exp (vMag R l))).sum)|

/-- **Concrete value-Lipschitz budget `Lv`** (coefficient of `δ`): `vLip R chosen + vLip R (expSumE (e::es)) / c`
    with `c = exp(−vMag R e)`. -/
noncomputable def budgetLv (chosen e : Expr) (es : List Expr) (R : ℝ) : ℝ :=
  vLip R chosen + vLip R (expSumE (e :: es)) / Real.exp (-(vMag R e))

/-- **Concrete derivative-magnitude budget `Dm`**: `dMag R chosen + dMag R (expSumE (e::es)) / c`. -/
noncomputable def budgetDm (chosen e : Expr) (es : List Expr) (R : ℝ) : ℝ :=
  dMag R chosen + dMag R (expSumE (e :: es)) / Real.exp (-(vMag R e))

/-- **Concrete derivative-Lipschitz budget `Dl`** (coefficient of `δ`): `dLip R chosen + (dLip R (expSumE (e::es))
    / c + dMag R (expSumE (e::es))·vLip R (expSumE (e::es)) / c²)` with `c = exp(−vMag R e)`. -/
noncomputable def budgetDl (chosen e : Expr) (es : List Expr) (R : ℝ) : ℝ :=
  dLip R chosen + (dLip R (expSumE (e :: es)) / Real.exp (-(vMag R e))
    + dMag R (expSumE (e :: es)) * vLip R (expSumE (e :: es)) / Real.exp (-(vMag R e)) ^ 2)

/-- **Fully-concrete value magnitude** `|evalR (logSoftmaxE chosen (e :: es)) σ| ≤ budgetM chosen e es R`, with
    NO free floor/ceiling: `c = exp(−vMag R e)` discharged by C9's `expSumE_floor`, `U = Σⱼ exp(vMag R logitⱼ)` by
    C18's `expSumE_upper`. The `|log c| = vMag R e` simplification uses `Real.log_exp` + `vMag_nonneg`. -/
theorem logSoftmaxE_value_mag_concrete (chosen e : Expr) (es : List Expr) (hch : Smooth chosen)
    (he : Smooth e) (hes : ∀ lp ∈ es, Smooth lp)
    (σ : Nat → ℝ) (R : ℝ) (hσ : ∀ i, |σ i| ≤ R) (hR : 0 ≤ R) :
    |evalR (logSoftmaxE chosen (e :: es)) σ| ≤ budgetM chosen e es R := by
  have h := logSoftmaxE_value_mag chosen (e :: es) hch σ R (Real.exp (-(vMag R e)))
    (((e :: es).map (fun l => Real.exp (vMag R l))).sum) hσ (Real.exp_pos _)
    (expSumE_floor e es he σ R hσ)
    (expSumE_upper (e :: es) (cons_smooth he hes) σ R hσ)
  rwa [Real.log_exp, abs_neg, abs_of_nonneg (vMag_nonneg he R hR)] at h

/-- **The concrete budget bundle.** For a softmax log-prob `logSoftmaxE chosen (e :: es)` (`Smooth` chosen, head,
    and tail), over the region `|σ i|, |σ' i| ≤ R` with `|σ i − σ' i| ≤ δ`, all four concrete per-log-prob budgets
    hold simultaneously — value magnitude at BOTH `σ` and `σ'` (`≤ budgetM`), value-Lipschitz (`≤ budgetLv·δ`),
    derivative magnitude at BOTH points (`≤ budgetDm`), and derivative-Lipschitz (`≤ budgetDl·δ`). This is EXACTLY
    the interface C15's `expLogTerm_lip` (entropy) and C19's `ratioE_deriv_lip` (ratio) consume, with the floor
    `c = exp(−vMag R e)` discharged for free (C9). Composes C17 (`Dm`/`Dl`) + C18 (`M`/`Lv`) into one lemma. -/
theorem logSoftmaxE_budgets (chosen e : Expr) (es : List Expr) (hch : Smooth chosen)
    (he : Smooth e) (hes : ∀ lp ∈ es, Smooth lp)
    (σ σ' : Nat → ℝ) (R δ : ℝ) (k : Nat)
    (hσ : ∀ i, |σ i| ≤ R) (hσ' : ∀ i, |σ' i| ≤ R) (hδ : ∀ i, |σ i - σ' i| ≤ δ) (hR : 0 ≤ R) :
    |evalR (logSoftmaxE chosen (e :: es)) σ| ≤ budgetM chosen e es R
    ∧ |evalR (logSoftmaxE chosen (e :: es)) σ'| ≤ budgetM chosen e es R
    ∧ |evalR (logSoftmaxE chosen (e :: es)) σ - evalR (logSoftmaxE chosen (e :: es)) σ'|
        ≤ budgetLv chosen e es R * δ
    ∧ |derivR (logSoftmaxE chosen (e :: es)) σ k| ≤ budgetDm chosen e es R
    ∧ |derivR (logSoftmaxE chosen (e :: es)) σ' k| ≤ budgetDm chosen e es R
    ∧ |derivR (logSoftmaxE chosen (e :: es)) σ k - derivR (logSoftmaxE chosen (e :: es)) σ' k|
        ≤ budgetDl chosen e es R * δ := by
  refine ⟨logSoftmaxE_value_mag_concrete chosen e es hch he hes σ R hσ hR,
    logSoftmaxE_value_mag_concrete chosen e es hch he hes σ' R hσ' hR,
    logSoftmaxE_value_lip_floored chosen e es hch he hes σ σ' R δ hσ hσ' hδ hR,
    ?_, ?_,
    logSoftmaxE_deriv_lip_floored chosen e es hch he hes σ σ' R δ k hσ hσ' hδ hR⟩
  · exact logSoftmaxE_deriv_mag chosen (e :: es) hch (cons_smooth he hes) σ R
      (Real.exp (-(vMag R e))) k hσ hR (Real.exp_pos _) (expSumE_floor e es he σ R hσ)
  · exact logSoftmaxE_deriv_mag chosen (e :: es) hch (cons_smooth he hes) σ' R
      (Real.exp (-(vMag R e))) k hσ' hR (Real.exp_pos _) (expSumE_floor e es he σ' R hσ')

end Puffer.RL.LogSoftmaxBudgetBundle
