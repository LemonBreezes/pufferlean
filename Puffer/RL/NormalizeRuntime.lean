/-
Closing the trifecta on the `√`/`÷` normalization kernel — shared by advantage
normalization `(adv−mean)/(√var+ε)` (`Puffer/RL/Normalize.lean`) and Muon's
Frobenius normalization `X/(‖X‖_F+ε)` (`Puffer/Float/Muon.lean`). Both are
`num / (√s + ε)`, so one runtime bound covers both.

The proof composes the extended Float axioms: `sqrt_error` + `add_error` bound the
denominator, `div_error` the final rounding, and the reusable `div_perturb_bound`
(from `Normalize`, over ℝ) the numerator/denominator perturbation. Well-conditioning
enters only as a denominator lower bound `dmin` (the `+ε` floor).
-/
import Mathlib
import Puffer.Float.Basic
import Puffer.RL.Normalize

namespace Puffer.RL.NormalizeRuntime

open Puffer.FloatR
open Puffer.RL.Normalize (div_perturb_bound advNorm)

/-- The executable normalization kernel `num / (√s + ε)`. -/
def divSqrtF (num s eps : Float) : Float := num / (Float.sqrt s + eps)

/-- **Normalization runtime error.** The running `num/(√s+ε)` deviates from the exact
    real `toReal num / (√(toReal s) + toReal ε)` by at most the div rounding plus the
    (sqrt+add) denominator error amplified by `1/dmin`. Instantiate `num = adv−mean,
    s = var` for advantage normalization, or `s = ‖X‖²` for Muon Frobenius normalize. -/
theorem divSqrtF_error (num s eps : Float) (dmin : ℝ) (hdmin : 0 < dmin)
    (hd : dmin ≤ |Real.sqrt (toReal s) + toReal eps|)
    (hdF : dmin ≤ |toReal (Float.sqrt s + eps)|) :
    |toReal (divSqrtF num s eps) - toReal num / (Real.sqrt (toReal s) + toReal eps)|
      ≤ u64 * |toReal num / toReal (Float.sqrt s + eps)|
        + |toReal num / (Real.sqrt (toReal s) + toReal eps)|
          * (u64 * |toReal (Float.sqrt s) + toReal eps| + u64 * Real.sqrt (toReal s)) / dmin := by
  -- denominator error: sqrt rounding + add rounding
  have hden : |toReal (Float.sqrt s + eps) - (Real.sqrt (toReal s) + toReal eps)|
      ≤ u64 * |toReal (Float.sqrt s) + toReal eps| + u64 * Real.sqrt (toReal s) := by
    calc |toReal (Float.sqrt s + eps) - (Real.sqrt (toReal s) + toReal eps)|
        = |(toReal (Float.sqrt s + eps) - (toReal (Float.sqrt s) + toReal eps))
            + (toReal (Float.sqrt s) - Real.sqrt (toReal s))| := by congr 1; ring
      _ ≤ |toReal (Float.sqrt s + eps) - (toReal (Float.sqrt s) + toReal eps)|
            + |toReal (Float.sqrt s) - Real.sqrt (toReal s)| := abs_add_le _ _
      _ ≤ u64 * |toReal (Float.sqrt s) + toReal eps| + u64 * Real.sqrt (toReal s) :=
            add_le_add (add_error (Float.sqrt s) eps) (sqrt_error s)
  -- numerator/denominator perturbation (numerator is exact here: εn = 0)
  have hpert := div_perturb_bound (toReal num) (toReal num)
    (Real.sqrt (toReal s) + toReal eps) (toReal (Float.sqrt s + eps))
    0 (u64 * |toReal (Float.sqrt s) + toReal eps| + u64 * Real.sqrt (toReal s)) dmin
    hdmin hd hdF (by simp) hden
  -- final rounding of the division, then triangle
  unfold divSqrtF
  calc |toReal (num / (Float.sqrt s + eps)) - toReal num / (Real.sqrt (toReal s) + toReal eps)|
      ≤ |toReal (num / (Float.sqrt s + eps)) - toReal num / toReal (Float.sqrt s + eps)|
          + |toReal num / toReal (Float.sqrt s + eps) - toReal num / (Real.sqrt (toReal s) + toReal eps)| :=
        abs_sub_le _ _ _
    _ ≤ u64 * |toReal num / toReal (Float.sqrt s + eps)|
          + (0 + |toReal num / (Real.sqrt (toReal s) + toReal eps)|
              * (u64 * |toReal (Float.sqrt s) + toReal eps| + u64 * Real.sqrt (toReal s))) / dmin :=
        add_le_add (div_error num (Float.sqrt s + eps)) hpert
    _ = u64 * |toReal num / toReal (Float.sqrt s + eps)|
          + |toReal num / (Real.sqrt (toReal s) + toReal eps)|
            * (u64 * |toReal (Float.sqrt s) + toReal eps| + u64 * Real.sqrt (toReal s)) / dmin := by
        rw [zero_add]

/-- The exact-real value normalized here is precisely `Normalize.advNorm num 0 s ε`
    (advantage normalization with the mean already subtracted into `num`). -/
theorem divSqrtF_target (num s eps : Float) :
    toReal num / (Real.sqrt (toReal s) + toReal eps) = advNorm (toReal num) 0 (toReal s) (toReal eps) := by
  unfold advNorm; rw [sub_zero]

end Puffer.RL.NormalizeRuntime
