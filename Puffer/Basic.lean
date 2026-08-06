import Mathlib

/-- Toolchain sanity check: Mathlib is available and the reals are in scope. -/
example : (0 : ℝ) ≤ 1 := by norm_num
