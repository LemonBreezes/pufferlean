/-
# The objective gradient-magnitude budget: a concrete bound on `|derivR (ppoObjE …) σ k|`

C39 (`BudgetDischarge`) discharged the clip barrier's floor `c` and log-prob magnitude `Mlog` to concrete network
budgets, leaving `Gmag` — the gradient magnitude of the whole PPO objective `ppoObjE` — as the last free constant,
because `ppoObjE` is NOT `Smooth` (it carries the clip `min`/`max` and the log-partition), so C4's `derivR_mag` does
not apply. This module builds `Gmag` concretely by decomposing over the objective's `add`/`sub`/`scale` structure
(`ppoTotalObjE = surrogate − cv·value + ce·entropy`) and bounding each term:

* **Non-`Smooth` core** — `derivR (min a b)` / `derivR (max a b)` SELECT one branch (an `if` on `evalR`), so their
  magnitude is `≤ max` of the branch magnitudes (`derivR_min_mag` / `derivR_max_mag`), and `derivR_clampE_mag` gives
  `|derivR (clampE r lo hi) σ k| ≤ |derivR r σ k|` (a clamp's gradient is the inner gradient or 0).
* **Surrogate** — `ppoSurrogateE_grad_mag`: `|derivR (ppoSurrogateE r g lo hi) σ k| ≤ |g|·|derivR r σ k|` (the clip is
  a `min` of two `g`-scaled branches, each `≤ |g|·|derivR r|`). This handles the clip's non-smoothness exactly.
* **Value** — `valueSqErrE_grad_mag`: the value squared error is `Smooth` (`mul (sub V c) (sub V c)`, `V` a linear
  head), so `|derivR (valueSqErrE V ret) σ k| ≤ dMag R (valueSqErrE V ret)` by C4's `derivR_mag`.
* **Entropy** — `entropyCatE_grad_mag`: the cross-entropy term is `Σ mul (exp lp) lp`, each derivative
  `exp(lp)·derivR lp·(lp+1)`, so `|derivR (entropyCatE logps) σ k| ≤ (length logps)·exp(Ment)·(Ment+1)·Dment` from the
  per-log-prob value/derivative budgets `Ment`/`Dment`.
* **Ratio** — `ratioLS_grad_mag`: `|derivR (ratioE (logSoftmaxE chosen (e::es)) oldLogp) σ k| ≤
  exp(budgetM − oldLogp)·budgetDm` (via `derivR_ratioE` = `exp(logsoftmax − oldLogp)·derivR logsoftmax` and C25's
  value/derivative budgets).
* **Assembly** — `ppoObjE_grad_mag` (from per-term magnitude budgets `Dr`/`Dval`/`Dent`) and
  `ppoObjE_grad_mag_concrete` (all three discharged): `|derivR (ppoObjE …) σ k| ≤ |g|·(exp(budgetM − oldLogp)·budgetDm)
  + |cv|·dMag R (valueSqErrE V ret) + |ce|·(length logps · exp(Ment)·(Ment+1)·Dment)` — the concrete `Gmag`, entirely
  in the network's `Smooth`/log-softmax structural budgets.

**Scope (honestly disclosed).** This gives `Gmag` concretely over the region `|σ i| ≤ R` in terms of the network's
structural budgets (`budgetM`/`budgetDm` for the softmax ratio, `dMag` for the `Smooth` value head, and the entropy
log-prob budgets `Ment`/`Dment` — which for the softmax entropy are `budgetM`/`budgetDm`, as in C31/C37). It closes
C39's remaining free constant: the clip barrier's per-step slate is now concrete in the network's budgets. The
entropy budgets `Ment`/`Dment` are supplied as the per-log-prob magnitude/derivative caps (dischargeable to
`budgetM`/`budgetDm` for softmax log-probs).
-/
import Puffer.RL.WholeRunFromC26
import Puffer.RL.LogSoftmaxBudgetBundle
import Puffer.RL.RatioGradExpr
open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)
open Puffer.RL.WholeRunFromC26 (ppoObjE)
open Puffer.RL.PPOObjectiveGrad (derivR_ppoTotalObjE)
open Puffer.RL.SurrogateExpr (ppoSurrogateE ratioE)
open Puffer.RL.RatioGradExpr (derivR_ratioE)
open Puffer.RL.SoftmaxExpr (logSoftmaxE)
open Puffer.RL.ValueEntropyExpr (valueSqErrE entropyCatE crossTermE)
open Puffer.RL.LogSoftmaxBudgetBundle (budgetM budgetDm logSoftmaxE_value_mag_concrete logSoftmaxE_budgets)

namespace Puffer.RL.ObjectiveGradMag

/-- `derivR (min a b)` selects one branch, so its magnitude is at most the larger branch magnitude. -/
theorem derivR_min_mag (a b : Expr) (σ : Nat → ℝ) (k : Nat) :
    |derivR (.min a b) σ k| ≤ max |derivR a σ k| |derivR b σ k| := by
  simp only [derivR]
  split_ifs
  · exact le_max_left _ _
  · exact le_max_right _ _

/-- `derivR (max a b)` selects one branch, so its magnitude is at most the larger branch magnitude. -/
theorem derivR_max_mag (a b : Expr) (σ : Nat → ℝ) (k : Nat) :
    |derivR (.max a b) σ k| ≤ max |derivR a σ k| |derivR b σ k| := by
  simp only [derivR]
  split_ifs
  · exact le_max_right _ _
  · exact le_max_left _ _

/-- A clamp's gradient is the inner gradient or `0` (the constant bounds contribute `0`), so
    `|derivR (clampE r lo hi) σ k| ≤ |derivR r σ k|`. -/
theorem derivR_clampE_mag (r : Expr) (lo hi : Float) (σ : Nat → ℝ) (k : Nat) :
    |derivR (clampE r lo hi) σ k| ≤ |derivR r σ k| := by
  have h1 := derivR_min_mag (.max r (.const lo)) (.const hi) σ k
  have h2 := derivR_max_mag r (.const lo) σ k
  have hc : ∀ c : Float, derivR (.const c) σ k = 0 := fun c => rfl
  rw [hc hi, abs_zero, max_eq_left (abs_nonneg _)] at h1
  rw [hc lo, abs_zero, max_eq_left (abs_nonneg _)] at h2
  exact h1.trans h2

/-- **Surrogate gradient magnitude.** The clipped surrogate `min (g·r) (g·clamp(r,lo,hi))` has gradient magnitude
    `≤ |g|·|derivR r σ k|`: it selects a `g`-scaled branch, and both branches (`g·r` and `g·clamp(r)`) have magnitude
    `≤ |g|·|derivR r|` (the clamp branch by `derivR_clampE_mag`). Handles the clip's non-smoothness exactly. -/
theorem ppoSurrogateE_grad_mag (r : Expr) (g lo hi : Float) (σ : Nat → ℝ) (k : Nat) :
    |derivR (ppoSurrogateE r g lo hi) σ k| ≤ |toReal g| * |derivR r σ k| := by
  have h1 := derivR_min_mag (.scale g r) (.scale g (clampE r lo hi)) σ k
  have hs1 : derivR (.scale g r) σ k = toReal g * derivR r σ k := rfl
  have hs2 : derivR (.scale g (clampE r lo hi)) σ k = toReal g * derivR (clampE r lo hi) σ k := rfl
  have hcl := derivR_clampE_mag r lo hi σ k
  have hunfold : derivR (ppoSurrogateE r g lo hi) σ k
      = derivR (.min (.scale g r) (.scale g (clampE r lo hi))) σ k := rfl
  rw [hunfold]
  refine h1.trans ?_
  rw [hs1, hs2, abs_mul, abs_mul]
  refine max_le ?_ ?_
  · exact le_of_eq rfl
  · exact mul_le_mul_of_nonneg_left hcl (abs_nonneg _)

/-- **Value gradient magnitude (`Smooth`).** The value squared error `mul (sub V c) (sub V c)` is `Smooth` when `V`
    is (`V` a linear head), so C4's `derivR_mag` gives `|derivR (valueSqErrE V ret) σ k| ≤ dMag R (valueSqErrE V ret)`
    uniformly over the region. -/
theorem valueSqErrE_grad_mag (V : Expr) (ret : Float) (σ : Nat → ℝ) (R : ℝ) (k : Nat)
    (hV : Smooth V) (hσ : ∀ i, |σ i| ≤ R) (hR : 0 ≤ R) :
    |derivR (valueSqErrE V ret) σ k| ≤ dMag R (valueSqErrE V ret) :=
  have hsm : Smooth (valueSqErrE V ret) :=
    Smooth.mul (Smooth.sub hV (Smooth.const ret)) (Smooth.sub hV (Smooth.const ret))
  derivR_mag hsm σ R k hσ hR

/-- **Ratio gradient magnitude.** `derivR (ratioE lp oldLogp) = exp(lp − oldLogp)·derivR lp`, so for `lp =
    logSoftmaxE chosen (e::es)` the magnitude is `≤ exp(budgetM − oldLogp)·budgetDm` over the region (value bounded by
    C25's `budgetM`, derivative by `budgetDm`). -/
theorem ratioLS_grad_mag (chosen e : Expr) (es : List Expr) (oldLogp : Float) (σ : Nat → ℝ) (R : ℝ) (k : Nat)
    (hch : Smooth chosen) (he : Smooth e) (hes : ∀ lp ∈ es, Smooth lp)
    (hσ : ∀ i, |σ i| ≤ R) (hR : 0 ≤ R) :
    |derivR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ k|
      ≤ Real.exp (budgetM chosen e es R - toReal oldLogp) * budgetDm chosen e es R := by
  rw [derivR_ratioE, abs_mul, abs_of_pos (Real.exp_pos _)]
  have hev : evalR (logSoftmaxE chosen (e :: es)) σ ≤ budgetM chosen e es R :=
    (le_abs_self _).trans (logSoftmaxE_value_mag_concrete chosen e es hch he hes σ R hσ hR)
  have hder : |derivR (logSoftmaxE chosen (e :: es)) σ k| ≤ budgetDm chosen e es R :=
    (logSoftmaxE_budgets chosen e es hch he hes σ σ R 0 k hσ hσ (fun j => by simp) hR).2.2.2.1
  exact mul_le_mul (Real.exp_le_exp.mpr (by linarith [hev])) hder (abs_nonneg _) (Real.exp_pos _).le

/-- Per-log-prob cross-entropy-term gradient magnitude: `derivR (mul (exp lp) lp) = exp(lp)·derivR lp·(lp+1)`, so
    `≤ exp(Ment)·(Ment+1)·Dment` when `|evalR lp σ| ≤ Ment`, `|derivR lp σ k| ≤ Dment`. -/
theorem mulExpLp_grad_mag (lp : Expr) (σ : Nat → ℝ) (k : Nat) (Ment Dment : ℝ)
    (hM : |evalR lp σ| ≤ Ment) (hD : |derivR lp σ k| ≤ Dment) :
    |derivR (.mul (.exp lp) lp) σ k| ≤ Real.exp Ment * (Ment + 1) * Dment := by
  have hev : derivR (.mul (.exp lp) lp) σ k
      = Real.exp (evalR lp σ) * derivR lp σ k * (evalR lp σ + 1) := by
    simp only [derivR, evalR]; ring
  rw [hev, abs_mul, abs_mul, abs_of_pos (Real.exp_pos _)]
  have hM1 : |evalR lp σ + 1| ≤ Ment + 1 :=
    (abs_add_le _ _).trans (by rw [abs_one]; linarith [hM])
  have hexp : Real.exp (evalR lp σ) ≤ Real.exp Ment := Real.exp_le_exp.mpr ((le_abs_self _).trans hM)
  have hX : Real.exp (evalR lp σ) * |derivR lp σ k| ≤ Real.exp Ment * Dment :=
    mul_le_mul hexp hD (abs_nonneg _) (Real.exp_pos _).le
  calc Real.exp (evalR lp σ) * |derivR lp σ k| * |evalR lp σ + 1|
      ≤ Real.exp Ment * Dment * (Ment + 1) :=
        mul_le_mul hX hM1 (abs_nonneg _) (mul_nonneg (Real.exp_pos _).le ((abs_nonneg _).trans hD))
    _ = Real.exp Ment * (Ment + 1) * Dment := by ring

/-- **Entropy gradient magnitude.** The cross-entropy term `crossTermE logps = Σ mul (exp lp) lp` has gradient
    magnitude `≤ (length logps)·exp(Ment)·(Ment+1)·Dment` from the per-log-prob budgets (list induction, per-term by
    `mulExpLp_grad_mag`). -/
theorem crossTermE_grad_mag (logps : List Expr) (σ : Nat → ℝ) (k : Nat) (Ment Dment : ℝ)
    (hM : ∀ lp ∈ logps, |evalR lp σ| ≤ Ment) (hD : ∀ lp ∈ logps, |derivR lp σ k| ≤ Dment) :
    |derivR (crossTermE logps) σ k| ≤ (logps.length : ℝ) * (Real.exp Ment * (Ment + 1) * Dment) := by
  induction logps with
  | nil => simp [crossTermE, derivR]
  | cons lp rest ih =>
      have hhead := mulExpLp_grad_mag lp σ k Ment Dment
        (hM lp (List.mem_cons.mpr (Or.inl rfl))) (hD lp (List.mem_cons.mpr (Or.inl rfl)))
      have hrest := ih (fun q hq => hM q (List.mem_cons.mpr (Or.inr hq)))
        (fun q hq => hD q (List.mem_cons.mpr (Or.inr hq)))
      have he : derivR (crossTermE (lp :: rest)) σ k
          = derivR (.mul (.exp lp) lp) σ k + derivR (crossTermE rest) σ k := rfl
      rw [he]
      calc |derivR (.mul (.exp lp) lp) σ k + derivR (crossTermE rest) σ k|
          ≤ |derivR (.mul (.exp lp) lp) σ k| + |derivR (crossTermE rest) σ k| := abs_add_le _ _
        _ ≤ Real.exp Ment * (Ment + 1) * Dment
            + (rest.length : ℝ) * (Real.exp Ment * (Ment + 1) * Dment) := add_le_add hhead hrest
        _ = ((lp :: rest).length : ℝ) * (Real.exp Ment * (Ment + 1) * Dment) := by
            simp only [List.length_cons]; push_cast; ring

/-- The categorical entropy `entropyCatE logps = −crossTermE logps` has the same gradient magnitude as the cross
    term (the negation cancels under `|·|`). -/
theorem entropyCatE_grad_mag (logps : List Expr) (σ : Nat → ℝ) (k : Nat) (Ment Dment : ℝ)
    (hM : ∀ lp ∈ logps, |evalR lp σ| ≤ Ment) (hD : ∀ lp ∈ logps, |derivR lp σ k| ≤ Dment) :
    |derivR (entropyCatE logps) σ k| ≤ (logps.length : ℝ) * (Real.exp Ment * (Ment + 1) * Dment) := by
  have he : derivR (entropyCatE logps) σ k = -derivR (crossTermE logps) σ k := by
    simp only [entropyCatE, derivR]; ring
  rw [he, abs_neg]
  exact crossTermE_grad_mag logps σ k Ment Dment hM hD

/-- **The objective gradient magnitude, from per-term budgets.** Over the objective's linear `add`/`sub`/`scale`
    structure (`derivR_ppoTotalObjE`), from bounds `Dr`/`Dval`/`Dent` on the ratio/value/entropy sub-term gradients,
    `|derivR (ppoObjE …) σ k| ≤ |g|·Dr + |cv|·Dval + |ce|·Dent` (the surrogate bounded by `|g|·Dr` via
    `ppoSurrogateE_grad_mag`). -/
theorem ppoObjE_grad_mag (chosen e V : Expr) (es logps : List Expr)
    (oldLogp g lo hi cv ce ret : Float) (σ : Nat → ℝ) (k : Nat) (Dr Dval Dent : ℝ)
    (hDr : |derivR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ k| ≤ Dr)
    (hDval : |derivR (valueSqErrE V ret) σ k| ≤ Dval)
    (hDent : |derivR (entropyCatE logps) σ k| ≤ Dent) :
    |derivR (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) σ k|
      ≤ |toReal g| * Dr + |toReal cv| * Dval + |toReal ce| * Dent := by
  have hsurr : |derivR (ppoSurrogateE (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) g lo hi) σ k|
      ≤ |toReal g| * Dr :=
    (ppoSurrogateE_grad_mag _ g lo hi σ k).trans (mul_le_mul_of_nonneg_left hDr (abs_nonneg _))
  have hB : |toReal cv| * |derivR (valueSqErrE V ret) σ k| ≤ |toReal cv| * Dval :=
    mul_le_mul_of_nonneg_left hDval (abs_nonneg _)
  have hC : |toReal ce| * |derivR (entropyCatE logps) σ k| ≤ |toReal ce| * Dent :=
    mul_le_mul_of_nonneg_left hDent (abs_nonneg _)
  unfold ppoObjE
  rw [derivR_ppoTotalObjE]
  calc |derivR (ppoSurrogateE (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) g lo hi) σ k
          - toReal cv * derivR (valueSqErrE V ret) σ k
          + toReal ce * derivR (entropyCatE logps) σ k|
      ≤ |derivR (ppoSurrogateE (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) g lo hi) σ k
          - toReal cv * derivR (valueSqErrE V ret) σ k|
          + |toReal ce * derivR (entropyCatE logps) σ k| := abs_add_le _ _
    _ ≤ (|derivR (ppoSurrogateE (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) g lo hi) σ k|
            + |toReal cv * derivR (valueSqErrE V ret) σ k|)
          + |toReal ce * derivR (entropyCatE logps) σ k| := by gcongr; exact abs_sub _ _
    _ = |derivR (ppoSurrogateE (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) g lo hi) σ k|
          + |toReal cv| * |derivR (valueSqErrE V ret) σ k|
          + |toReal ce| * |derivR (entropyCatE logps) σ k| := by rw [abs_mul, abs_mul]
    _ ≤ |toReal g| * Dr + |toReal cv| * Dval + |toReal ce| * Dent :=
        add_le_add (add_le_add hsurr hB) hC

/-- **The concrete objective gradient magnitude (`Gmag`).** All three sub-term budgets discharged: the ratio via
    `ratioLS_grad_mag`, the value via `valueSqErrE_grad_mag`, the entropy via `entropyCatE_grad_mag`. Over the region
    `|σ i| ≤ R`, with per-log-prob budgets `Ment`/`Dment`,
    `|derivR (ppoObjE …) σ k| ≤ |g|·(exp(budgetM − oldLogp)·budgetDm) + |cv|·dMag R (valueSqErrE V ret)
      + |ce|·(length logps · exp(Ment)·(Ment+1)·Dment)` — the concrete `Gmag`, entirely in the network's structural
    budgets. Closes C39's remaining free constant. -/
theorem ppoObjE_grad_mag_concrete (chosen e V : Expr) (es logps : List Expr)
    (oldLogp g lo hi cv ce ret : Float) (σ : Nat → ℝ) (R : ℝ) (k : Nat) (Ment Dment : ℝ)
    (hch : Smooth chosen) (he : Smooth e) (hes : ∀ lp ∈ es, Smooth lp) (hV : Smooth V)
    (hσ : ∀ i, |σ i| ≤ R) (hR : 0 ≤ R)
    (hMent : ∀ lp ∈ logps, |evalR lp σ| ≤ Ment) (hDment : ∀ lp ∈ logps, |derivR lp σ k| ≤ Dment) :
    |derivR (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) σ k|
      ≤ |toReal g| * (Real.exp (budgetM chosen e es R - toReal oldLogp) * budgetDm chosen e es R)
        + |toReal cv| * dMag R (valueSqErrE V ret)
        + |toReal ce| * ((logps.length : ℝ) * (Real.exp Ment * (Ment + 1) * Dment)) :=
  ppoObjE_grad_mag chosen e V es logps oldLogp g lo hi cv ce ret σ k _ _ _
    (ratioLS_grad_mag chosen e es oldLogp σ R k hch he hes hσ hR)
    (valueSqErrE_grad_mag V ret σ R k hV hσ hR)
    (entropyCatE_grad_mag logps σ k Ment Dment hMent hDment)

end Puffer.RL.ObjectiveGradMag
