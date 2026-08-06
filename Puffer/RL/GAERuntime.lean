/-
Closing the trifecta on the RL core: the executable `Float` GAE recurrence is
within a certified bound of its ℝ spec.

`Puffer.RL.GAE.gaeHead` (ℝ) is proved equal to the closed-form discounted sum of
TD errors (`gaeHead_eq_geoSum`). `Puffer.FloatR.gaeHeadF` (Float) is the runnable
version the trainer uses. Here we prove they differ by at most a *computable*
accumulated-rounding bound `gaeErrBnd`, composing the IEEE (1+δ) axioms
(`Puffer/Float/Basic.lean`) through the backward recurrence — exactly the
`dotF_error` pattern, now on the RL advantage computation. Same object, three
layers: ℝ spec + closed form (proved) · running Float impl · proven error bound.
-/
import Mathlib
import Puffer.Float.Basic
import Puffer.Float.Exec
import Puffer.RL.GAE

namespace Puffer.RL.GAERuntime

open Puffer.FloatR
open Puffer.RL.GAE (gaeHead gaeHead_cons gaeHead_eq_geoSum)
open Finset

/-- Computable certified bound on the GAE rounding error: at each step, one add and
    one multiply rounding, plus the (discount-scaled) propagated tail error. -/
noncomputable def gaeErrBnd (w : Float) : List Float → ℝ
  | [] => 0
  | δ :: rest =>
      u64 * |toReal δ + toReal (w * gaeHeadF w rest)|
      + u64 * |toReal w * toReal (gaeHeadF w rest)|
      + |toReal w| * gaeErrBnd w rest

theorem gaeErrBnd_nonneg (w : Float) (ds : List Float) : 0 ≤ gaeErrBnd w ds := by
  induction ds with
  | nil => simp [gaeErrBnd]
  | cons δ rest ih =>
      simp only [gaeErrBnd]
      have hu := u64_pos.le
      exact add_nonneg (add_nonneg (mul_nonneg hu (abs_nonneg _)) (mul_nonneg hu (abs_nonneg _)))
        (mul_nonneg (abs_nonneg _) ih)

/-- **GAE runtime error bound.** The executable `gaeHeadF w ds` deviates from the
    exact real GAE `gaeHead (toReal w) (map toReal ds)` by at most `gaeErrBnd w ds`. -/
theorem gaeHeadF_error (w : Float) (ds : List Float) :
    |toReal (gaeHeadF w ds) - gaeHead (toReal w) (ds.map toReal)| ≤ gaeErrBnd w ds := by
  induction ds with
  | nil => simp [gaeHeadF, gaeErrBnd]
  | cons δ rest ih =>
      simp only [gaeHeadF, List.map_cons, gaeHead_cons, gaeErrBnd]
      have split :
          toReal (δ + w * gaeHeadF w rest)
              - (toReal δ + toReal w * gaeHead (toReal w) (rest.map toReal))
            = (toReal (δ + w * gaeHeadF w rest) - (toReal δ + toReal (w * gaeHeadF w rest)))
              + (toReal (w * gaeHeadF w rest) - toReal w * toReal (gaeHeadF w rest))
              + toReal w * (toReal (gaeHeadF w rest) - gaeHead (toReal w) (rest.map toReal)) := by
        ring
      rw [split]
      calc |(toReal (δ + w * gaeHeadF w rest) - (toReal δ + toReal (w * gaeHeadF w rest)))
              + (toReal (w * gaeHeadF w rest) - toReal w * toReal (gaeHeadF w rest))
              + toReal w * (toReal (gaeHeadF w rest) - gaeHead (toReal w) (rest.map toReal))|
          ≤ (|toReal (δ + w * gaeHeadF w rest) - (toReal δ + toReal (w * gaeHeadF w rest))|
              + |toReal (w * gaeHeadF w rest) - toReal w * toReal (gaeHeadF w rest)|)
              + |toReal w * (toReal (gaeHeadF w rest) - gaeHead (toReal w) (rest.map toReal))| :=
            (abs_add_le _ _).trans (add_le_add (abs_add_le _ _) le_rfl)
        _ ≤ (u64 * |toReal δ + toReal (w * gaeHeadF w rest)|
              + u64 * |toReal w * toReal (gaeHeadF w rest)|)
              + |toReal w| * gaeErrBnd w rest := by
            refine add_le_add (add_le_add (add_error δ (w * gaeHeadF w rest))
              (mul_error w (gaeHeadF w rest))) ?_
            rw [abs_mul]
            exact mul_le_mul_of_nonneg_left ih (abs_nonneg _)

/-- **Capstone (full trifecta on GAE).** The running `Float` GAE is within the
    certified bound of the *closed-form* discounted sum of TD errors — combining the
    runtime bound with the flagship `gaeHead_eq_geoSum` (recursive = closed form). -/
theorem gaeHeadF_closedForm_error (w : Float) (ds : List Float) :
    |toReal (gaeHeadF w ds)
        - ∑ i ∈ range (ds.map toReal).length, (toReal w) ^ i * (ds.map toReal).getD i 0|
      ≤ gaeErrBnd w ds := by
  rw [← gaeHead_eq_geoSum]
  exact gaeHeadF_error w ds

end Puffer.RL.GAERuntime
