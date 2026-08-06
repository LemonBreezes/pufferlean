/-
# Discharging the clip-barrier budget constants concretely: floor `c` and log-prob magnitude `Mlog`

C36 (`ClipBarrier`) / C38 (`HTrapAssembly`) left the clip barrier parameterized by three network budget constants —
`Gmag` (objective gradient magnitude), `Mlog` (log-prob magnitude), `c` (partition floor) — supplied as hypotheses.
Two of the three are pure log-softmax budget facts that hold over the whole region, so they need not be free: this
module discharges them to their concrete network values,
  * `c := exp(−vMag R e)` — the partition floor, via C9's `expSumE_floor` (the partition dominates a single `exp`
    term, and the logit is bounded by `vMag R e` over the region);
  * `Mlog := budgetM chosen e es R` — the log-prob value magnitude, via C18/C25's `logSoftmaxE_value_mag_concrete`
    (`evalR (logSoftmaxE …) σ ≤ |evalR …| ≤ budgetM`),
producing a FLOORED clip barrier and floored glue that no longer take the floor/magnitude hypotheses.

* `clipMarginC` — the clip margin with `c`/`Mlog` at their concrete values: `exp(budgetM − oldLogp)·(vLip R chosen +
  vLip R (expSumE (e::es))/exp(−vMag R e))·(|lr|·Gmag)`.
* `logProb_mag_le` — `evalR (logSoftmaxE chosen (e::es)) σ ≤ budgetM chosen e es R` over the region.
* `clip_barrier_floored` — C36's `clip_barrier_concrete` with `c`/`Mlog`/the four floor-and-magnitude hypotheses
  DISCHARGED (both at `σ` and at the projected image, in the region by `projAscentE_mem`); it takes only the `Smooth`
  structure, region membership, the objective gradient magnitude `Gmag`, and the concrete clip margin.
* `hTrap_step_floored` — C38's `hTrap_step` with the same discharge: produces `InRegVal (projAscentE … σ)` from the
  `Smooth` structure, region, `Gmag`, the concrete clip margin, and the (C37) uniform entropy budgets — no free
  floor/magnitude constants.

**Scope (honestly disclosed).** This discharges the log-softmax budget constants `c` and `Mlog` to concrete network
structural budgets (`exp(−vMag R e)`, `budgetM`). The entropy budgets `Ment`/`Dment` are separately dischargeable to
`budgetM`/`budgetDm` by C37's `entropy_barrier_softmax` (for the softmax log-prob list). The ONE remaining free
constant is `Gmag`, the gradient magnitude of the WHOLE PPO objective `ppoObjE` — which is NOT `Smooth` (it carries the
clip `min`/`max` and the log-partition), so C4's `derivR_mag` does not apply and a concrete `Gmag` needs a dedicated
gradient-magnitude budget for the objective (surrogate + value + entropy gradient magnitudes, the magnitude analogue of
C15/C16's gradient-Lipschitz budgets) — a separate development. With this module, the clip barrier's floor and
log-prob-magnitude constants are concrete; only the objective gradient magnitude remains a supplied budget.
-/
import Puffer.RL.ClipBarrier
import Puffer.RL.EntropyBarrier
import Puffer.RL.TrajReachability
import Puffer.RL.LogSoftmaxBudgetBundle
open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)
open Puffer.RL.RegionInvariance (projAscentE projAscentE_mem)
open Puffer.RL.WholeRunFromC26 (ppoObjE)
open Puffer.RL.SoftmaxExpr (logSoftmaxE expSumE expSumE_floor)
open Puffer.RL.SurrogateExpr (ratioE)
open Puffer.RL.TrajReachability (InRegVal)
open Puffer.RL.ClipBarrier (clip_barrier_concrete)
open Puffer.RL.EntropyBarrier (entropy_barrier)
open Puffer.RL.LogSoftmaxBudgetBundle (budgetM logSoftmaxE_value_mag_concrete)

namespace Puffer.RL.BudgetDischarge

/-- The clip margin with the floor `c` and log-prob magnitude `Mlog` at their concrete network values
    (`c = exp(−vMag R e)`, `Mlog = budgetM chosen e es R`): the C36 per-step ratio move
    `exp(budgetM − oldLogp)·(vLip R chosen + vLip R (expSumE (e::es))/exp(−vMag R e))·(|lr|·Gmag)`. Definitionally
    C36's `clipMargin` (= `ratioE_projStep_disp`'s bound) with those two constants substituted. -/
noncomputable def clipMarginC (chosen e : Expr) (es : List Expr) (oldLogp lr : Float) (R Gmag : ℝ) : ℝ :=
  Real.exp (budgetM chosen e es R - toReal oldLogp)
    * (vLip R chosen + vLip R (expSumE (e :: es)) / Real.exp (-(vMag R e))) * (|toReal lr| * Gmag)

/-- The log-prob value magnitude is bounded by C25's concrete `budgetM` over the region (one-sided, as the clip
    barrier needs): `evalR (logSoftmaxE chosen (e::es)) σ ≤ budgetM chosen e es R`. -/
theorem logProb_mag_le (chosen e : Expr) (es : List Expr) (hch : Smooth chosen) (he : Smooth e)
    (hes : ∀ lp ∈ es, Smooth lp) (σ : Nat → ℝ) (R : ℝ) (hσ : ∀ i, |σ i| ≤ R) (hR : 0 ≤ R) :
    evalR (logSoftmaxE chosen (e :: es)) σ ≤ budgetM chosen e es R :=
  (le_abs_self _).trans (logSoftmaxE_value_mag_concrete chosen e es hch he hes σ R hσ hR)

/-- **The floored clip barrier.** C36's `clip_barrier_concrete` with the partition floor `c = exp(−vMag R e)` and the
    log-prob magnitude `Mlog = budgetM chosen e es R` DISCHARGED (via `expSumE_floor` and `logProb_mag_le`, at both `σ`
    and its projected image — the latter in the region by `projAscentE_mem`). It takes only the `Smooth` structure,
    region membership at `σ`, the objective gradient magnitude `Gmag`, and the concrete clip margin `clipMarginC`; the
    four floor-and-magnitude hypotheses are gone. Concludes `ratio(projAscentE … σ) ∈ (lo, hi)`. -/
theorem clip_barrier_floored (chosen e V : Expr) (es logps : List Expr)
    (oldLogp g lo hi cv ce ret lr : Float) (R Gmag : ℝ) (σ : Nat → ℝ)
    (hch : Smooth chosen) (he : Smooth e) (hes : ∀ lp ∈ es, Smooth lp)
    (hσ : ∀ k, |σ k| ≤ R) (hR : 0 ≤ R)
    (hGmag : ∀ k, |derivR (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) σ k| ≤ Gmag)
    (hlo : toReal lo + clipMarginC chosen e es oldLogp lr R Gmag
          < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ)
    (hhi : evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ
          < toReal hi - clipMarginC chosen e es oldLogp lr R Gmag) :
    toReal lo < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp)
        (projAscentE (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr R σ)
      ∧ evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp)
        (projAscentE (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr R σ) < toReal hi :=
  have hτ : ∀ k, |projAscentE (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr R σ k| ≤ R :=
    projAscentE_mem _ lr R σ hR
  clip_barrier_concrete chosen e V es logps oldLogp g lo hi cv ce ret lr R Gmag
    (budgetM chosen e es R) (Real.exp (-(vMag R e))) σ hch he hes hσ hR hGmag
    (Real.exp_pos _)
    (expSumE_floor e es he σ R hσ)
    (expSumE_floor e es he _ R hτ)
    (logProb_mag_le chosen e es hch he hes σ R hσ hR)
    (logProb_mag_le chosen e es hch he hes _ R hτ hR)
    hlo hhi

/-- **The floored `hTrap_step`.** C38's glue with the floor `c` and log-prob magnitude `Mlog` discharged: from `σ` in
    the region with the objective gradient magnitude `Gmag`, the CONCRETE clip margin `clipMarginC`, and the (C37)
    uniform entropy budgets, one projected step lands with `InRegVal (projAscentE … σ)` — the clip clause via
    `clip_barrier_floored`, the entropy clauses via C37's `entropy_barrier`. No free floor/magnitude constants remain;
    only `Gmag` and the entropy budgets `Ment`/`Dment` are supplied (the latter dischargeable by C37's
    `entropy_barrier_softmax`). -/
theorem hTrap_step_floored (chosen e V : Expr) (es logps : List Expr)
    (oldLogp g lo hi cv ce ret lr : Float) (R Gmag Ment Dment : ℝ) (σ : Nat → ℝ)
    (hch : Smooth chosen) (he : Smooth e) (hes : ∀ lp ∈ es, Smooth lp)
    (hσ : ∀ k, |σ k| ≤ R) (hR : 0 ≤ R)
    (hGmag : ∀ k, |derivR (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) σ k| ≤ Gmag)
    (hlo : toReal lo + clipMarginC chosen e es oldLogp lr R Gmag
          < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ)
    (hhi : evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ
          < toReal hi - clipMarginC chosen e es oldLogp lr R Gmag)
    (hMent : ∀ ρ : Nat → ℝ, (∀ k, |ρ k| ≤ R) → ∀ lp ∈ logps, |evalR lp ρ| ≤ Ment)
    (hDment : ∀ ρ : Nat → ℝ, (∀ k, |ρ k| ≤ R) → ∀ k, ∀ lp ∈ logps, |derivR lp ρ k| ≤ Dment) :
    InRegVal chosen e es logps oldLogp lo hi Ment Dment
      (projAscentE (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr R σ) := by
  obtain ⟨hcliplo, hcliphi⟩ := clip_barrier_floored chosen e V es logps oldLogp g lo hi cv ce ret lr R
    Gmag σ hch he hes hσ hR hGmag hlo hhi
  obtain ⟨hev, hde⟩ := entropy_barrier (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) logps lr R
    Ment Dment σ hR hMent hDment
  exact ⟨hcliplo, hcliphi, hev, hde⟩

end Puffer.RL.BudgetDischarge
