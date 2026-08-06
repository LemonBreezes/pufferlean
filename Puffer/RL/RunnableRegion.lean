/-
# The runnable trajectory's region conditions: param-bound (weight-clamp) + entropy budgets (unconditional)

The whole-run error interval (C35 `TrajReachability.ppo_whole_run_reachable`, C38
`HTrapAssembly.ppo_whole_run_from_barriers`) takes the RUNNABLE trajectory `θ`'s per-step region conditions as
premises: `hRegθ : ∀ p i, |θ p i| ≤ R` (param-bound), `hMθ : ∀ p, ∀ lp ∈ logps, |evalR lp (θ p)| ≤ Ment` and
`hDmEθ : ∀ p k, ∀ lp ∈ logps, |derivR lp (θ p) k| ≤ Dment` (entropy budgets). The runnable `θ` is the actual Float
trainer trajectory, only known within `B`/step of the ideal (non-deterministic in ℝ), so — unlike the ideal
trajectory (C29/C35 projected ascent) — its region membership cannot be discharged by projection-of-the-ideal-update.
This module discharges the two that ARE structural, leaving only the clip interior as a per-step runtime check.

* `runnable_region_bounded` — param-bound by the trainer's WEIGHT-CLAMP: if every post-step weight lies in `[−R, R]`
  (`hclamp : ∀ n k, |θ (n+1) k| ≤ R`, a faithful model of real PPO/Muon implementations that clip/clamp weights) and
  the start is bounded, then `∀ n k, |θ n k| ≤ R`. This discharges `hRegθ` for a weight-clamped runnable trainer (the
  runnable analogue of C29's `projAscentE_traj_bounded`).
* `region_entropy_smooth` / `region_entropy_softmax` — the entropy budgets at ANY region point `ρ`, UNCONDITIONALLY
  (exactly as C37 `EntropyBarrier`, because they are uniform magnitude budgets, not open-interval trapping): for
  `Smooth` log-probs via C4's `evalR_mag`/`derivR_mag` (`≤ vMag R lp`/`dMag R lp`), for the softmax log-probs
  (`List.ofFn (fun i => logSoftmaxE (logit i) (e::es))`) via C25's `budgetM`/`budgetDm`. Stated at a RAW region point
  `ρ` (the runnable `θ p` is not a `projAscentE` image, so C37's projected-step version does not apply directly).
* `runnable_region_forall` / `runnable_region_forall_softmax` — the capstone: for a weight-clamped runnable trajectory
  in the region for all `p`, both the param-bound and the entropy budgets hold for all `p` (the entropy at each `θ p`
  by the uniform region budget). Discharges `hRegθ`, `hMθ`, `hDmEθ` together.

**Scope (honestly disclosed).** This discharges the runnable trajectory's param-bound (via the trainer's
weight-clamp — a faithful model of real implementations) and its entropy budgets (unconditional over the region,
uniform magnitude budgets exactly as C37). The clip-interior condition `hIntθ` (`ratio(θ p) ∈ (lo,hi)`) remains a
per-step runtime condition — the same quantity `puffer verify` checks each step: a fixed clip band is not
forward-invariant without a contraction (C36), and the runnable `θ` is not a projected-ascent point, so C36's clip
barrier does not directly apply to it. `Ment`/`Dment` are the network's `Smooth`/log-softmax magnitude budget
constants (`vMag`/`dMag`, or C25's `budgetM`/`budgetDm` with a common-over-logits cap), as in C24/C31/C37.
-/
import Puffer.RL.LogSoftmaxBudgetBundle
open Puffer.FloatR.ADR
open Puffer.RL.SoftmaxExpr (logSoftmaxE)
open Puffer.RL.LogSoftmaxBudgetBundle (budgetM budgetDm logSoftmaxE_value_mag_concrete logSoftmaxE_budgets)

namespace Puffer.RL.RunnableRegion

/-- **Runnable param-bound by weight-clamping.** A runnable trajectory `θ` whose start is bounded (`h0`) and whose
    every post-step weight is clamped into `[−R, R]` (`hclamp` — real PPO/Muon trainers clip/clamp weights) stays in
    the region for all steps: `∀ n k, |θ n k| ≤ R`. Discharges the whole-run interval's `hRegθ` for the runnable
    trajectory (the runnable analogue of C29's `projAscentE_traj_bounded`, but from the trainer's own clamp rather
    than projection of the ideal update). -/
theorem runnable_region_bounded (θ : Nat → (Nat → ℝ)) (R : ℝ)
    (h0 : ∀ k, |θ 0 k| ≤ R) (hclamp : ∀ n k, |θ (n + 1) k| ≤ R) :
    ∀ n k, |θ n k| ≤ R := by
  intro n
  cases n with
  | zero => exact h0
  | succ m => exact hclamp m

/-- **Entropy budgets at a raw region point, `Smooth` log-probs.** At any `ρ` in the region (`|ρ k| ≤ R`), each
    `Smooth` log-prob's value and derivative magnitudes are bounded by C4's `evalR_mag` (`≤ vMag R lp`) and
    `derivR_mag` (`≤ dMag R lp`); given common caps `Ment`/`Dment` over the log-probs, the entropy budgets hold at `ρ`.
    Unconditional over the region — no per-step margin (uniform magnitude budgets, as C37). -/
theorem region_entropy_smooth (logps : List Expr) (R Ment Dment : ℝ) (ρ : Nat → ℝ)
    (hρ : ∀ k, |ρ k| ≤ R) (hR : 0 ≤ R) (hlp : ∀ lp ∈ logps, Smooth lp)
    (hVm : ∀ lp ∈ logps, vMag R lp ≤ Ment) (hDm : ∀ lp ∈ logps, dMag R lp ≤ Dment) :
    (∀ lp ∈ logps, |evalR lp ρ| ≤ Ment) ∧ (∀ k, ∀ lp ∈ logps, |derivR lp ρ k| ≤ Dment) :=
  ⟨fun lp hlpm => (evalR_mag (hlp lp hlpm) ρ R hρ).trans (hVm lp hlpm),
   fun k lp hlpm => (derivR_mag (hlp lp hlpm) ρ R k hρ hR).trans (hDm lp hlpm)⟩

/-- **Entropy budgets at a raw region point, SOFTMAX log-probs.** At any `ρ` in the region, for the softmax log-prob
    list `List.ofFn (fun i => logSoftmaxE (logit i) (e::es))` the value/derivative magnitudes are bounded by C25's
    `budgetM`/`budgetDm` (`logSoftmaxE_value_mag_concrete` and the deriv-magnitude half of `logSoftmaxE_budgets`),
    lifted to common caps `Ment`/`Dment` over the logits (max over classes, as C31). -/
theorem region_entropy_softmax {n : Nat} (e : Expr) (logit : Fin n → Expr) (es : List Expr)
    (R Ment Dment : ℝ) (ρ : Nat → ℝ) (hρ : ∀ k, |ρ k| ≤ R) (hR : 0 ≤ R)
    (he : Smooth e) (hes : ∀ lp ∈ es, Smooth lp) (hlogit : ∀ i, Smooth (logit i))
    (hMent : ∀ i, budgetM (logit i) e es R ≤ Ment) (hDment : ∀ i, budgetDm (logit i) e es R ≤ Dment) :
    (∀ lp ∈ List.ofFn (fun i => logSoftmaxE (logit i) (e :: es)), |evalR lp ρ| ≤ Ment)
    ∧ (∀ k, ∀ lp ∈ List.ofFn (fun i => logSoftmaxE (logit i) (e :: es)), |derivR lp ρ k| ≤ Dment) := by
  refine ⟨?_, ?_⟩
  · rw [List.forall_mem_ofFn_iff]
    intro i
    exact (logSoftmaxE_value_mag_concrete (logit i) e es (hlogit i) he hes ρ R hρ hR).trans (hMent i)
  · intro k
    rw [List.forall_mem_ofFn_iff]
    intro i
    have hb := logSoftmaxE_budgets (logit i) e es (hlogit i) he hes ρ ρ R 0 k hρ hρ (fun j => by simp) hR
    exact hb.2.2.2.1.trans (hDment i)

/-- **CAPSTONE (`Smooth` log-probs): the runnable trajectory's `hRegθ`/`hMθ`/`hDmEθ` discharged together.** For a
    weight-clamped runnable trajectory (`h0`/`hclamp`) with `Smooth` log-probs and common magnitude caps, the
    param-bound holds for all `p` (`runnable_region_bounded`) and, at each in-region `θ p`, the entropy value and
    derivative budgets hold (`region_entropy_smooth`). This is exactly the runnable-side region slate the whole-run
    interval consumes — discharged, save the per-step clip interior. -/
theorem runnable_region_forall (θ : Nat → (Nat → ℝ)) (logps : List Expr) (R Ment Dment : ℝ)
    (h0 : ∀ k, |θ 0 k| ≤ R) (hclamp : ∀ n k, |θ (n + 1) k| ≤ R) (hR : 0 ≤ R)
    (hlp : ∀ lp ∈ logps, Smooth lp)
    (hVm : ∀ lp ∈ logps, vMag R lp ≤ Ment) (hDm : ∀ lp ∈ logps, dMag R lp ≤ Dment) :
    (∀ p i, |θ p i| ≤ R)
    ∧ (∀ p, ∀ lp ∈ logps, |evalR lp (θ p)| ≤ Ment)
    ∧ (∀ p k, ∀ lp ∈ logps, |derivR lp (θ p) k| ≤ Dment) :=
  have hreg : ∀ p i, |θ p i| ≤ R := runnable_region_bounded θ R h0 hclamp
  ⟨hreg,
   fun p => (region_entropy_smooth logps R Ment Dment (θ p) (hreg p) hR hlp hVm hDm).1,
   fun p k => (region_entropy_smooth logps R Ment Dment (θ p) (hreg p) hR hlp hVm hDm).2 k⟩

/-- **CAPSTONE (SOFTMAX log-probs): the runnable trajectory's region slate for the real objective.** As
    `runnable_region_forall`, but for the softmax log-prob list `List.ofFn (fun i => logSoftmaxE (logit i) (e::es))`
    the real objective uses, with the entropy budgets discharged by `region_entropy_softmax` (C25 `budgetM`/`budgetDm`
    with a common-over-logits cap). Discharges the runnable-side `hRegθ`/`hMθ`/`hDmEθ` for the softmax-MLP entropy
    term, save the per-step clip interior. -/
theorem runnable_region_forall_softmax {n : Nat} (θ : Nat → (Nat → ℝ)) (e : Expr) (logit : Fin n → Expr)
    (es : List Expr) (R Ment Dment : ℝ)
    (h0 : ∀ k, |θ 0 k| ≤ R) (hclamp : ∀ m k, |θ (m + 1) k| ≤ R) (hR : 0 ≤ R)
    (he : Smooth e) (hes : ∀ lp ∈ es, Smooth lp) (hlogit : ∀ i, Smooth (logit i))
    (hMent : ∀ i, budgetM (logit i) e es R ≤ Ment) (hDment : ∀ i, budgetDm (logit i) e es R ≤ Dment) :
    (∀ p i, |θ p i| ≤ R)
    ∧ (∀ p, ∀ lp ∈ List.ofFn (fun i => logSoftmaxE (logit i) (e :: es)), |evalR lp (θ p)| ≤ Ment)
    ∧ (∀ p k, ∀ lp ∈ List.ofFn (fun i => logSoftmaxE (logit i) (e :: es)),
        |derivR lp (θ p) k| ≤ Dment) :=
  have hreg : ∀ p i, |θ p i| ≤ R := runnable_region_bounded θ R h0 hclamp
  ⟨hreg,
   fun p => (region_entropy_softmax e logit es R Ment Dment (θ p) (hreg p) hR he hes hlogit hMent hDment).1,
   fun p k =>
     (region_entropy_softmax e logit es R Ment Dment (θ p) (hreg p) hR he hes hlogit hMent hDment).2 k⟩

end Puffer.RL.RunnableRegion
