/-
# The entropy barrier: discharging the entropy clauses of C35's `hTrap` — UNCONDITIONALLY over the region

C35 (`TrajReachability`) reduced the ideal trajectory's region membership to a one-step trapping premise `hTrap`,
whose value-level predicate `InRegVal` has, besides the clip-interior clause (discharged by C36 `ClipBarrier`), two
ENTROPY clauses: the log-prob value budget `∀ lp ∈ logps, |evalR lp σ| ≤ Ment` and the derivative budget
`∀ k, ∀ lp ∈ logps, |derivR lp σ k| ≤ Dment`. This module discharges those two clauses at the projected step.

Unlike the clip interior — an OPEN interval a step can escape, so C36 needed a per-step MARGIN and a fixed clip band
is not trapping — the entropy conditions are MAGNITUDE BUDGETS that hold UNIFORMLY over the whole region `|σ i| ≤ R`
(the log-probs are bounded functions of bounded parameters). Since a projected step lands back in the region
(`projAscentE_mem`), the budgets transfer to the next point WITH NO MARGIN and NO expansiveness obstruction — the
entropy barrier is UNCONDITIONAL over the region (the easy half of `hTrap`).

* `entropy_barrier` — the structural core: given the log-prob value/derivative budgets hold at EVERY region point,
  they hold at `projAscentE O lr R σ` (which is a region point). One application of `projAscentE_mem`.
* `entropy_barrier_smooth` — concrete for `Smooth` log-probs: the uniform budgets are `vMag R lp` / `dMag R lp` via
  C4's `evalR_mag` / `derivR_mag`, so `Ment` / `Dment` are any common bounds over the log-probs.
* `entropy_barrier_softmax` — concrete for the SOFTMAX log-probs the real objective uses
  (`logps = List.ofFn (fun i => logSoftmaxE (logit i) (e::es))`): the uniform budgets are C25's `budgetM` / `budgetDm`
  (via `logSoftmaxE_value_mag_concrete` / `logSoftmaxE_budgets`), lifted to `Ment` / `Dment` by a common bound over
  the logits (the max over classes, as in C31). Discharges the entropy clauses of `hTrap` for the actual softmax-MLP
  entropy term.

**Scope (honestly disclosed).** This discharges the entropy clauses UNCONDITIONALLY over the `R`-region — no per-step
margin, unlike C36's clip barrier — because they are uniform magnitude budgets, not an open-interval trapping. `Ment`
/ `Dment` are the network's `Smooth` / log-softmax magnitude budget constants over the region (`vMag`/`dMag`, or
C25's `budgetM`/`budgetDm` with a common-over-logits cap), the same finite/checkable constants used in C24/C31/C26.
The clip-interior clause (C36) and the coupling budgets are separate. With C36 (clip) + this (entropy), both value-
level clauses of `hTrap`'s `InRegVal` are discharged at the projected step — the clip via a per-step margin, the
entropy unconditionally.
-/
import Puffer.RL.RegionInvariance
import Puffer.RL.LogSoftmaxBudgetBundle
open Puffer.FloatR.ADR
open Puffer.RL.RegionInvariance (projAscentE projAscentE_mem)
open Puffer.RL.SoftmaxExpr (logSoftmaxE)
open Puffer.RL.LogSoftmaxBudgetBundle (budgetM budgetDm logSoftmaxE_value_mag_concrete logSoftmaxE_budgets)

namespace Puffer.RL.EntropyBarrier

/-- **The entropy barrier (structural core).** If the log-prob value budget `Ment` and derivative budget `Dment`
    hold at EVERY point of the region `|ρ k| ≤ R`, then they hold at the projected step `projAscentE O lr R σ` — which
    is a region point by `projAscentE_mem`. Unconditional: the budgets are uniform magnitude bounds, so no per-step
    margin is needed (contrast C36's clip barrier). Discharges the two entropy clauses of C35's `InRegVal` at the
    projected step. -/
theorem entropy_barrier (O : Expr) (logps : List Expr) (lr : Float) (R Ment Dment : ℝ) (σ : Nat → ℝ)
    (hR : 0 ≤ R)
    (hMent : ∀ ρ : Nat → ℝ, (∀ k, |ρ k| ≤ R) → ∀ lp ∈ logps, |evalR lp ρ| ≤ Ment)
    (hDment : ∀ ρ : Nat → ℝ, (∀ k, |ρ k| ≤ R) → ∀ k, ∀ lp ∈ logps, |derivR lp ρ k| ≤ Dment) :
    (∀ lp ∈ logps, |evalR lp (projAscentE O lr R σ)| ≤ Ment)
    ∧ (∀ k, ∀ lp ∈ logps, |derivR lp (projAscentE O lr R σ) k| ≤ Dment) :=
  have hτ : ∀ k, |projAscentE O lr R σ k| ≤ R := projAscentE_mem O lr R σ hR
  ⟨hMent _ hτ, hDment _ hτ⟩

/-- **The entropy barrier, concrete for `Smooth` log-probs.** Each `lp ∈ logps` is `Smooth`, so its value and
    derivative magnitudes are uniformly bounded over the region by C4's `evalR_mag` (`≤ vMag R lp`) and `derivR_mag`
    (`≤ dMag R lp`). Given common caps `∀ lp ∈ logps, vMag R lp ≤ Ment` and `∀ lp ∈ logps, dMag R lp ≤ Dment` (the
    max over the log-probs), the entropy clauses hold at the projected step. -/
theorem entropy_barrier_smooth (O : Expr) (logps : List Expr) (lr : Float) (R Ment Dment : ℝ) (σ : Nat → ℝ)
    (hR : 0 ≤ R) (hlp : ∀ lp ∈ logps, Smooth lp)
    (hVm : ∀ lp ∈ logps, vMag R lp ≤ Ment) (hDm : ∀ lp ∈ logps, dMag R lp ≤ Dment) :
    (∀ lp ∈ logps, |evalR lp (projAscentE O lr R σ)| ≤ Ment)
    ∧ (∀ k, ∀ lp ∈ logps, |derivR lp (projAscentE O lr R σ) k| ≤ Dment) :=
  entropy_barrier O logps lr R Ment Dment σ hR
    (fun ρ hρ lp hlpm => (evalR_mag (hlp lp hlpm) ρ R hρ).trans (hVm lp hlpm))
    (fun ρ hρ k lp hlpm => (derivR_mag (hlp lp hlpm) ρ R k hρ hR).trans (hDm lp hlpm))

/-- **The entropy barrier, concrete for the SOFTMAX log-probs of the real objective.** For
    `logps = List.ofFn (fun i => logSoftmaxE (logit i) (e :: es))` (one log-prob per class, `Smooth` logits/head/tail),
    each log-prob's value and derivative magnitudes are uniformly bounded over the region by C25's `budgetM` /
    `budgetDm` (`logSoftmaxE_value_mag_concrete` / the deriv-magnitude half of `logSoftmaxE_budgets`). Given a common
    bound over the logits (`∀ i, budgetM (logit i) e es R ≤ Ment` and `∀ i, budgetDm (logit i) e es R ≤ Dment` — the
    max over classes, as in C31), the entropy clauses hold at the projected step. This discharges `hTrap`'s entropy
    clauses for the actual softmax-MLP entropy term. -/
theorem entropy_barrier_softmax {n : Nat} (O e : Expr) (logit : Fin n → Expr) (es : List Expr)
    (lr : Float) (R Ment Dment : ℝ) (σ : Nat → ℝ) (hR : 0 ≤ R)
    (he : Smooth e) (hes : ∀ lp ∈ es, Smooth lp) (hlogit : ∀ i, Smooth (logit i))
    (hMent : ∀ i, budgetM (logit i) e es R ≤ Ment) (hDment : ∀ i, budgetDm (logit i) e es R ≤ Dment) :
    (∀ lp ∈ List.ofFn (fun i => logSoftmaxE (logit i) (e :: es)),
        |evalR lp (projAscentE O lr R σ)| ≤ Ment)
    ∧ (∀ k, ∀ lp ∈ List.ofFn (fun i => logSoftmaxE (logit i) (e :: es)),
        |derivR lp (projAscentE O lr R σ) k| ≤ Dment) := by
  have hτ : ∀ k, |projAscentE O lr R σ k| ≤ R := projAscentE_mem O lr R σ hR
  refine ⟨?_, ?_⟩
  · rw [List.forall_mem_ofFn_iff]
    intro i
    exact (logSoftmaxE_value_mag_concrete (logit i) e es (hlogit i) he hes _ R hτ hR).trans (hMent i)
  · intro k
    rw [List.forall_mem_ofFn_iff]
    intro i
    have hb := logSoftmaxE_budgets (logit i) e es (hlogit i) he hes
      (projAscentE O lr R σ) (projAscentE O lr R σ) R 0 k hτ hτ (fun j => by simp) hR
    exact hb.2.2.2.1.trans (hDment i)

end Puffer.RL.EntropyBarrier
