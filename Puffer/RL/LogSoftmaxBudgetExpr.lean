/-
# Log-softmax derivative budgets: discharging C15/C16's per-log-prob hypotheses

C15 (entropy) reduced its bound to per-log-prob budgets `M`/`Lv`/`Dm`/`Dl`, and C16 (surrogate) reduced its to
the ratio's `Lr` — both left those budgets as hypotheses. This module discharges the DERIVATIVE budgets of the
actual log-softmax `logSoftmaxE chosen logits = chosen − log(Σⱼ exp(logitⱼ))` (C9) by composing C4 (the raw
logit, `Smooth`) + C6 (the `log`-partition's derivative Lipschitz) + C9 (the partition's positive floor).

The key decomposition: `derivR (logSoftmaxE chosen logits) = derivR chosen − derivR (log (expSumE logits))`, so:
* `logSoftmaxE_deriv_lip` (the derivative Lipschitz `Dl`): `≤ (dLip R chosen + [C6 log-partition bound])·δ`, where
  the log-partition term is EXACTLY C6's `derivR_log_lip` applied to `expSumE logits` — which is `Smooth`
  (`expSumE_smooth`, from the `Smooth` logits) with the positive floor `c` supplied by C9's `expSumE_floor`.
* `logSoftmaxE_deriv_mag` (the derivative magnitude `Dm`): `≤ dMag R chosen + dMag R (expSumE logits)/c` (C4 for
  the logit, the log-node derivative `∂/evalR` bounded by the floor).
* `logSoftmaxE_deriv_lip_floored`: the FULLY CONCRETE derivative Lipschitz — the floor discharged via C9's
  `expSumE_floor`, giving `c = exp(−vMag R e)` for a nonempty logit list with `Smooth` head `e`, so NO free floor
  hypothesis remains.

These are exactly the `Dm`/`Dl` that C15's `expLogTerm_lip` consumes and (via the exp-node) the ratio's `Lr`
that C16 consumes — computed from the network's `Smooth` logits.

**Scope (honestly disclosed):** this discharges the DERIVATIVE budgets (`Dm`, `Dl`) — the ones that use C6
directly and dominate C15's entropy constant. The VALUE budgets (`M` = value magnitude, `Lv` = value Lipschitz)
also feed C15/C16 and decompose similarly (`M` additionally needs a partition UPPER bound for `|log(Σexp)|`, a
further step). The floor `c` is the partition's positive lower bound; the floored corollary uses C9's concrete
`exp(−vMag R e)`. All budgets are over the region `|σ i| ≤ R` with the logits `Smooth` (C10 linear layers).
-/
import Puffer.RL.SoftmaxExpr

open Puffer.FloatR.ADR
open Puffer.RL.SoftmaxExpr (expSumE logPartitionE logSoftmaxE)

namespace Puffer.RL.LogSoftmaxBudgetExpr

/-- The softmax partition `Σⱼ exp(logitⱼ)` is `Smooth` whenever every logit is (`exp`/`add`/`const` are `Smooth`
    constructors) — so C4's `derivR_lip`/`derivR_mag` and C6's `derivR_log_lip` apply to it. -/
theorem expSumE_smooth (logits : List Expr) (hlog : ∀ lp ∈ logits, Smooth lp) :
    Smooth (expSumE logits) := by
  induction logits with
  | nil => exact Smooth.const 0
  | cons e es ih =>
      refine Smooth.add (Smooth.exp ?_) (ih ?_)
      · exact hlog e (List.mem_cons.mpr (Or.inl rfl))
      · exact fun q hq => hlog q (List.mem_cons.mpr (Or.inr hq))

/-- **Log-softmax derivative Lipschitz (`Dl` budget).** For a `Smooth` chosen logit and `Smooth` logits, over
    the region `|σ i|, |σ' i| ≤ R` with the partition floored `c ≤ evalR (expSumE logits) σ, σ'` (from C9),
    `|derivR (logSoftmaxE chosen logits) σ k − derivR (logSoftmaxE chosen logits) σ' k| ≤ (dLip R chosen +
    (dLip R (expSumE logits)/c + dMag R (expSumE logits)·vLip R (expSumE logits)/c²))·δ`. Composes C4 (chosen,
    `derivR_lip`) + C6 (the `log`-partition, `derivR_log_lip` on the `Smooth` partition) via triangle on
    `derivR (log-softmax) = derivR chosen − derivR (log-partition)`. -/
theorem logSoftmaxE_deriv_lip (chosen : Expr) (logits : List Expr) (hch : Smooth chosen)
    (hlog : ∀ lp ∈ logits, Smooth lp)
    (σ σ' : Nat → ℝ) (R δ c : ℝ) (k : Nat)
    (hσ : ∀ i, |σ i| ≤ R) (hσ' : ∀ i, |σ' i| ≤ R) (hδ : ∀ i, |σ i - σ' i| ≤ δ) (hR : 0 ≤ R)
    (hc : 0 < c)
    (hfloor : c ≤ evalR (expSumE logits) σ) (hfloor' : c ≤ evalR (expSumE logits) σ') :
    |derivR (logSoftmaxE chosen logits) σ k - derivR (logSoftmaxE chosen logits) σ' k|
      ≤ (dLip R chosen
         + (dLip R (expSumE logits) / c
            + dMag R (expSumE logits) * vLip R (expSumE logits) / c ^ 2)) * δ := by
  have hchLip : |derivR chosen σ k - derivR chosen σ' k| ≤ dLip R chosen * δ :=
    derivR_lip hch σ σ' R δ k hσ hσ' hδ hR
  have hpartLip : |derivR (.log (expSumE logits)) σ k - derivR (.log (expSumE logits)) σ' k|
      ≤ (dLip R (expSumE logits) / c
         + dMag R (expSumE logits) * vLip R (expSumE logits) / c ^ 2) * δ :=
    derivR_log_lip (expSumE_smooth logits hlog) σ σ' k R δ c hσ hσ' hδ hR hc hfloor hfloor'
  have hd : ∀ τ : Nat → ℝ, derivR (logSoftmaxE chosen logits) τ k
      = derivR chosen τ k - derivR (.log (expSumE logits)) τ k := fun τ => by
    simp only [logSoftmaxE, logPartitionE, derivR]
  rw [hd σ, hd σ']
  have key : (derivR chosen σ k - derivR (.log (expSumE logits)) σ k)
        - (derivR chosen σ' k - derivR (.log (expSumE logits)) σ' k)
      = (derivR chosen σ k - derivR chosen σ' k)
        + (-(derivR (.log (expSumE logits)) σ k - derivR (.log (expSumE logits)) σ' k)) := by ring
  rw [key]
  calc |(derivR chosen σ k - derivR chosen σ' k)
          + (-(derivR (.log (expSumE logits)) σ k - derivR (.log (expSumE logits)) σ' k))|
      ≤ |derivR chosen σ k - derivR chosen σ' k|
          + |-(derivR (.log (expSumE logits)) σ k - derivR (.log (expSumE logits)) σ' k)| := abs_add_le _ _
    _ = |derivR chosen σ k - derivR chosen σ' k|
          + |derivR (.log (expSumE logits)) σ k - derivR (.log (expSumE logits)) σ' k| := by rw [abs_neg]
    _ ≤ dLip R chosen * δ
          + (dLip R (expSumE logits) / c
             + dMag R (expSumE logits) * vLip R (expSumE logits) / c ^ 2) * δ := add_le_add hchLip hpartLip
    _ = (dLip R chosen
         + (dLip R (expSumE logits) / c
            + dMag R (expSumE logits) * vLip R (expSumE logits) / c ^ 2)) * δ := by ring

/-- **Log-softmax derivative magnitude (`Dm` budget).** `|derivR (logSoftmaxE chosen logits) σ k| ≤ dMag R
    chosen + dMag R (expSumE logits)/c` — C4 for the chosen logit plus the `log`-node derivative `∂/evalR`
    bounded by the partition floor `c`. -/
theorem logSoftmaxE_deriv_mag (chosen : Expr) (logits : List Expr) (hch : Smooth chosen)
    (hlog : ∀ lp ∈ logits, Smooth lp)
    (σ : Nat → ℝ) (R c : ℝ) (k : Nat) (hσ : ∀ i, |σ i| ≤ R) (hR : 0 ≤ R)
    (hc : 0 < c) (hfloor : c ≤ evalR (expSumE logits) σ) :
    |derivR (logSoftmaxE chosen logits) σ k|
      ≤ dMag R chosen + dMag R (expSumE logits) / c := by
  have hd : derivR (logSoftmaxE chosen logits) σ k
      = derivR chosen σ k - derivR (.log (expSumE logits)) σ k := by
    simp only [logSoftmaxE, logPartitionE, derivR]
  rw [hd]
  have hchM : |derivR chosen σ k| ≤ dMag R chosen := derivR_mag hch σ R k hσ hR
  have he : 0 < evalR (expSumE logits) σ := lt_of_lt_of_le hc hfloor
  have hpartM : |derivR (.log (expSumE logits)) σ k| ≤ dMag R (expSumE logits) / c := by
    simp only [derivR, abs_div, abs_of_pos he]
    gcongr
    · exact (abs_nonneg _).trans (derivR_mag (expSumE_smooth logits hlog) σ R k hσ hR)
    · exact derivR_mag (expSumE_smooth logits hlog) σ R k hσ hR
  exact (abs_sub _ _).trans (add_le_add hchM hpartM)

/-- **Fully concrete log-softmax derivative Lipschitz.** For a nonempty logit list `e :: es` with `Smooth` head
    `e` and `Smooth` tail, the partition floor is discharged by C9's `expSumE_floor` (`c = exp(−vMag R e) > 0`),
    so NO free floor hypothesis remains: the derivative Lipschitz is bounded by `(dLip R chosen + (dLip R
    (expSumE (e::es))/exp(−vMag R e) + dMag R (expSumE (e::es))·vLip R (expSumE (e::es))/exp(−vMag R e)²))·δ` —
    a fully computable constant from the network's `Smooth` logits. This is the softmax discharging its own
    `log`-floor (C9's payoff) applied to the log-softmax's gradient-Lipschitz. -/
theorem logSoftmaxE_deriv_lip_floored (chosen e : Expr) (es : List Expr) (hch : Smooth chosen)
    (he : Smooth e) (hes : ∀ lp ∈ es, Smooth lp)
    (σ σ' : Nat → ℝ) (R δ : ℝ) (k : Nat)
    (hσ : ∀ i, |σ i| ≤ R) (hσ' : ∀ i, |σ' i| ≤ R) (hδ : ∀ i, |σ i - σ' i| ≤ δ) (hR : 0 ≤ R) :
    |derivR (logSoftmaxE chosen (e :: es)) σ k - derivR (logSoftmaxE chosen (e :: es)) σ' k|
      ≤ (dLip R chosen
         + (dLip R (expSumE (e :: es)) / Real.exp (-(vMag R e))
            + dMag R (expSumE (e :: es)) * vLip R (expSumE (e :: es))
                / Real.exp (-(vMag R e)) ^ 2)) * δ :=
  logSoftmaxE_deriv_lip chosen (e :: es) hch
    (fun lp hlp => (List.mem_cons.mp hlp).elim (fun h => h ▸ he) (fun h => hes lp h))
    σ σ' R δ (Real.exp (-(vMag R e))) k hσ hσ' hδ hR (Real.exp_pos _)
    (Puffer.RL.SoftmaxExpr.expSumE_floor e es he σ R hσ)
    (Puffer.RL.SoftmaxExpr.expSumE_floor e es he σ' R hσ')

end Puffer.RL.LogSoftmaxBudgetExpr
