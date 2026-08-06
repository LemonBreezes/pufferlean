/-
# Concrete softmax entropy gradient-Lipschitz: the uniform-over-logits step of C24

C24 (`EntropyConcreteExpr`) proved the categorical entropy's gradient-Lipschitz from UNIFORM per-log-prob budgets
`M`/`Lv`/`Dm`/`Dl`, and disclosed that making those budgets uniform ACROSS the softmax's log-probs (a max over the
`n` logits) was a deferred mechanical step. This module performs it, composing C25's per-log-prob budget bundle
(`LogSoftmaxBudgetBundle.logSoftmaxE_budgets`) with a uniform bound across the log-probs.

For a softmax the log-probs are `logSoftmaxE (logit i) (e::es)` (`i : Fin n`, one per class), all sharing the SAME
partition `expSumE (e::es)` and differing only in the chosen logit `logit i`. C25's four concrete budgets
`budgetM`/`budgetLv`/`budgetDm`/`budgetDl` depend on the chosen logit ONLY through `vMag`/`vLip`/`dMag`/`dLip R
(logit i)` (added to a shared-partition common term), so:

* `budgetM_le`/`budgetLv_le`/`budgetDm_le`/`budgetDl_le` — a common bound on the per-logit `vMag`/`vLip`/`dMag`/
  `dLip R (logit i)` lifts C25's per-log-prob budget to a UNIFORM cap (pure monotonicity — the "max over logits").
* `entropyCatE_softmax_gradient_lipschitz` — given uniform caps `M`/`Lv`/`Dm`/`Dl` on the C25 budgets, the softmax
  entropy is gradient-Lipschitz with the concrete `n·exp(M)·((M+1)·Dl + (M+2)·Dm·Lv)·δ` — the C25 per-log-prob
  budgets discharged into C24's `entropyCatE_ofFn_gradient_lipschitz` per index.
* `entropyCatE_softmax_gradient_lipschitz_common` — FULLY CONCRETE: given a COMMON bound on every logit's budget
  (`∀ i, vMag R (logit i) ≤ Vm`, …), the softmax entropy gradient-Lipschitz with the caps written out explicitly in
  the logits' `Vm`/`Vlp`/`Dmg`/`Dlp` and the shared partition budgets — no free per-log-prob hypothesis.

**Scope (honestly disclosed):** the uniform caps come from a COMMON bound over the `n` logits' individual budgets
(`Vm`/`Vlp`/`Dmg`/`Dlp` — the "max over logits", supplied as hypotheses `∀ i, vMag R (logit i) ≤ Vm`, …). For a
concrete network these are the max of the per-logit `vMag`/… over the classes — a finite max, checkable but
network-dependent. The logits `logit i` and the partition head `e`/tail `es` must be `Smooth` (C10 linear layers),
over the region `|σ i| ≤ R`. This completes C24's deferred uniform-over-logits step: the softmax entropy's
gradient-Lipschitz is now concrete in the network's per-logit `Smooth` budgets.
-/
import Puffer.RL.EntropyConcreteExpr
import Puffer.RL.LogSoftmaxBudgetBundle
import Puffer.RL.SoftmaxExpr
open Puffer.FloatR.ADR
open Puffer.RL.SoftmaxExpr (logSoftmaxE expSumE)
open Puffer.RL.ValueEntropyExpr (entropyCatE)
open Puffer.RL.LogSoftmaxBudgetBundle (budgetM budgetLv budgetDm budgetDl logSoftmaxE_budgets)
open Puffer.RL.EntropyConcreteExpr (entropyCatE_ofFn_gradient_lipschitz)
open Puffer.RL.LogSoftmaxBudgetExpr (expSumE_smooth)

namespace Puffer.RL.EntropyBudgetUniform

/-- **Uniform value-magnitude cap.** A common bound `vMag R chosen ≤ Vm` lifts C25's `budgetM` to a uniform cap
    `Vm + max (vMag R e) |log Σ exp(vMag)|` (the chosen-logit term is the only per-log-prob variation). -/
theorem budgetM_le (chosen e : Expr) (es : List Expr) (R Vm : ℝ) (h : vMag R chosen ≤ Vm) :
    budgetM chosen e es R
      ≤ Vm + max (vMag R e) |Real.log (((e :: es).map (fun l => Real.exp (vMag R l))).sum)| := by
  simp only [budgetM]; linarith [h]

theorem budgetLv_le (chosen e : Expr) (es : List Expr) (R Vlp : ℝ) (h : vLip R chosen ≤ Vlp) :
    budgetLv chosen e es R ≤ Vlp + vLip R (expSumE (e :: es)) / Real.exp (-(vMag R e)) := by
  simp only [budgetLv]; linarith [h]

theorem budgetDm_le (chosen e : Expr) (es : List Expr) (R Dmg : ℝ) (h : dMag R chosen ≤ Dmg) :
    budgetDm chosen e es R ≤ Dmg + dMag R (expSumE (e :: es)) / Real.exp (-(vMag R e)) := by
  simp only [budgetDm]; linarith [h]

theorem budgetDl_le (chosen e : Expr) (es : List Expr) (R Dlp : ℝ) (h : dLip R chosen ≤ Dlp) :
    budgetDl chosen e es R ≤ Dlp + (dLip R (expSumE (e :: es)) / Real.exp (-(vMag R e))
      + dMag R (expSumE (e :: es)) * vLip R (expSumE (e :: es)) / Real.exp (-(vMag R e)) ^ 2) := by
  simp only [budgetDl]; linarith [h]


/-- **Softmax entropy gradient-Lipschitz from uniform budget caps.** Given uniform caps `M`/`Lv`/`Dm`/`Dl` on the
    C25 per-log-prob budgets (`∀ i, budgetM (logit i) e es R ≤ M`, …), the softmax entropy
    `entropyCatE (List.ofFn (fun i => logSoftmaxE (logit i) (e::es)))` is gradient-Lipschitz with
    `≤ n·exp(M)·((M+1)·Dl + (M+2)·Dm·Lv)·δ` — the C25 budgets discharged per index into C24's
    `entropyCatE_ofFn_gradient_lipschitz`. -/
theorem entropyCatE_softmax_gradient_lipschitz {n : Nat} (logit : Fin n → Expr) (e : Expr) (es : List Expr)
    (he : Smooth e) (hes : ∀ lp ∈ es, Smooth lp) (hlogit : ∀ i, Smooth (logit i))
    (σ σ' : Nat → ℝ) (R δ : ℝ) (k : Nat)
    (hσ : ∀ i, |σ i| ≤ R) (hσ' : ∀ i, |σ' i| ≤ R) (hδ : ∀ i, |σ i - σ' i| ≤ δ) (hR : 0 ≤ R)
    (M Lv Dm Dl : ℝ)
    (hM : ∀ i, budgetM (logit i) e es R ≤ M) (hLv : ∀ i, budgetLv (logit i) e es R ≤ Lv)
    (hDm : ∀ i, budgetDm (logit i) e es R ≤ Dm) (hDl : ∀ i, budgetDl (logit i) e es R ≤ Dl)
    (hDmn : 0 ≤ Dm) :
    |derivR (entropyCatE (List.ofFn (fun i => logSoftmaxE (logit i) (e :: es)))) σ k
        - derivR (entropyCatE (List.ofFn (fun i => logSoftmaxE (logit i) (e :: es)))) σ' k|
      ≤ (n : ℝ) * (Real.exp M * ((M + 1) * Dl + (M + 2) * Dm * Lv) * δ) := by
  have hδn : (0:ℝ) ≤ δ := (abs_nonneg _).trans (hδ 0)
  refine entropyCatE_ofFn_gradient_lipschitz (fun i => logSoftmaxE (logit i) (e :: es)) σ σ' k M Lv Dm Dl δ
    ?_ ?_ ?_ ?_ ?_ ?_ hDmn
  all_goals intro i
  all_goals obtain ⟨b1, b2, b3, b4, b5, b6⟩ :=
    logSoftmaxE_budgets (logit i) e es (hlogit i) he hes σ σ' R δ k hσ hσ' hδ hR
  · exact b1.trans (hM i)
  · exact b2.trans (hM i)
  · exact b3.trans (mul_le_mul_of_nonneg_right (hLv i) hδn)
  · exact b4.trans (hDm i)
  · exact b5.trans (hDm i)
  · exact b6.trans (mul_le_mul_of_nonneg_right (hDl i) hδn)

/-- **Fully concrete softmax entropy gradient-Lipschitz.** Given a COMMON bound on every logit's budget
    (`∀ i, vMag R (logit i) ≤ Vm`, `∀ i, vLip R (logit i) ≤ Vlp`, `∀ i, dMag R (logit i) ≤ Dmg`,
    `∀ i, dLip R (logit i) ≤ Dlp` — the max over the `n` logits), the softmax entropy is gradient-Lipschitz with
    the caps written out in the logits' `Vm`/`Vlp`/`Dmg`/`Dlp` and the shared partition budgets — no free
    per-log-prob hypothesis. Composes the main theorem with the `budget_le` uniform-bounding lemmas. -/
theorem entropyCatE_softmax_gradient_lipschitz_common {n : Nat} [NeZero n]
    (logit : Fin n → Expr) (e : Expr) (es : List Expr)
    (he : Smooth e) (hes : ∀ lp ∈ es, Smooth lp) (hlogit : ∀ i, Smooth (logit i))
    (σ σ' : Nat → ℝ) (R δ : ℝ) (k : Nat)
    (hσ : ∀ i, |σ i| ≤ R) (hσ' : ∀ i, |σ' i| ≤ R) (hδ : ∀ i, |σ i - σ' i| ≤ δ) (hR : 0 ≤ R)
    (Vm Vlp Dmg Dlp : ℝ)
    (hVm : ∀ i, vMag R (logit i) ≤ Vm) (hVlp : ∀ i, vLip R (logit i) ≤ Vlp)
    (hDmg : ∀ i, dMag R (logit i) ≤ Dmg) (hDlp : ∀ i, dLip R (logit i) ≤ Dlp)
    (hDmgn : 0 ≤ Dmg) :
    |derivR (entropyCatE (List.ofFn (fun i => logSoftmaxE (logit i) (e :: es)))) σ k
        - derivR (entropyCatE (List.ofFn (fun i => logSoftmaxE (logit i) (e :: es)))) σ' k|
      ≤ (n : ℝ) * (Real.exp (Vm + max (vMag R e)
            |Real.log (((e :: es).map (fun l => Real.exp (vMag R l))).sum)|)
          * (((Vm + max (vMag R e)
                |Real.log (((e :: es).map (fun l => Real.exp (vMag R l))).sum)|) + 1)
              * (Dlp + (dLip R (expSumE (e :: es)) / Real.exp (-(vMag R e))
                  + dMag R (expSumE (e :: es)) * vLip R (expSumE (e :: es)) / Real.exp (-(vMag R e)) ^ 2))
            + ((Vm + max (vMag R e)
                  |Real.log (((e :: es).map (fun l => Real.exp (vMag R l))).sum)|) + 2)
                * (Dmg + dMag R (expSumE (e :: es)) / Real.exp (-(vMag R e)))
                * (Vlp + vLip R (expSumE (e :: es)) / Real.exp (-(vMag R e)))) * δ) :=
  entropyCatE_softmax_gradient_lipschitz logit e es he hes hlogit σ σ' R δ k hσ hσ' hδ hR _ _ _ _
    (fun i => budgetM_le (logit i) e es R Vm (hVm i))
    (fun i => budgetLv_le (logit i) e es R Vlp (hVlp i))
    (fun i => budgetDm_le (logit i) e es R Dmg (hDmg i))
    (fun i => budgetDl_le (logit i) e es R Dlp (hDlp i))
    (add_nonneg hDmgn (div_nonneg (dMag_nonneg (expSumE_smooth (e :: es)
      (fun lp hlp => (List.mem_cons.mp hlp).elim (fun h => h ▸ he) (fun h => hes lp h))) R hR)
      (Real.exp_pos _).le))

end Puffer.RL.EntropyBudgetUniform
