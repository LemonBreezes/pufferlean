/-
The advantage-NORMALIZATION map's sensitivity: how `advNorm a mean var ε = (a − mean)/(√var + ε)` responds
to perturbed value / mean / variance. The ℝ-level perturbation companion (in the a72/a77/a80 family) that
lifts the GAE→normalize STATISTICS bounds (`GAEMean` a99 center + `GAEVar` a100 scale, and `GAETrue` a91 the
per-slot value) to the actual NORMALIZED advantage `normalizeAdv` feeds the policy gradient.

`advNorm` is a quotient with a `√var` denominator, so its sensitivity has two parts:

  • a VARIANCE-shift term — via the Hölder-½ bound `abs_sqrt_sub_sqrt_le` (`|√v₁ − √v₂| ≤ √|v₁ − v₂|`),
    scaled by the centered magnitude `|a₁ − m₁|` over the product of the two denominators (`√vᵢ + ε ≥ ε > 0`);
  • a VALUE/MEAN-shift term — `(|Δa| + |Δmean|)` over the perturbed denominator.

`advNorm_perturb` combines them over a common denominator (`div_sub_div`, numerator split
`n₁d₂ − n₂d₁ = n₁(d₂−d₁) + d₁(n₁−n₂)`, then the fraction split `(X + d₁Y)/(d₁d₂) = X/(d₁d₂) + Y/d₂`).

Pure ℝ (Mathlib) — axiom-clean (`propext`/`Classical.choice`/`Quot.sound`; the `√` floor is the `+ε`).
Composed with a87's `fullNormalizeAdv_error` (Float↔ℝ at the entry statistics) and a91/a99/a100 (value/mean/
variance differences to the true-arithmetic GAE), it yields the full `normalizeAdv`-on-true-GAE bound.
-/
import Mathlib
import Puffer.Float.Basic
import Puffer.RL.Normalize

namespace Puffer.RL.AdvNormPerturb

open Puffer.FloatR
open Puffer.RL.Normalize (advNorm advNorm_denom_pos)

/-- **Advantage-normalization sensitivity.** The normalized advantage `(a − mean)/(√var + ε)` responds to
    perturbed value/mean/variance within a variance-shift term (Hölder-½ `√|Δvar|`, scaled by the centered
    magnitude over the two denominators) plus a value/mean-shift term (`(|Δa| + |Δmean|)` over the
    denominator). The `+ε` is the denominator floor. -/
theorem advNorm_perturb (a1 m1 v1 a2 m2 v2 ε : ℝ)
    (hv1 : 0 ≤ v1) (hv2 : 0 ≤ v2) (hε : 0 < ε) :
    |advNorm a1 m1 v1 ε - advNorm a2 m2 v2 ε|
      ≤ |a1 - m1| * Real.sqrt |v1 - v2| / ((Real.sqrt v1 + ε) * (Real.sqrt v2 + ε))
        + (|a1 - a2| + |m1 - m2|) / (Real.sqrt v2 + ε) := by
  set d1 := Real.sqrt v1 + ε with hd1
  set d2 := Real.sqrt v2 + ε with hd2
  have hd1p : 0 < d1 := advNorm_denom_pos v1 ε hε
  have hd2p : 0 < d2 := advNorm_denom_pos v2 ε hε
  have hprod : 0 < d1 * d2 := mul_pos hd1p hd2p
  have hcomb : advNorm a1 m1 v1 ε - advNorm a2 m2 v2 ε
      = ((a1 - m1) * (d2 - d1) + d1 * ((a1 - m1) - (a2 - m2))) / (d1 * d2) := by
    rw [advNorm, advNorm, ← hd1, ← hd2, div_sub_div _ _ (ne_of_gt hd1p) (ne_of_gt hd2p)]
    congr 1; ring
  rw [hcomb, abs_div, abs_of_pos hprod]
  have hnum : |(a1 - m1) * (d2 - d1) + d1 * ((a1 - m1) - (a2 - m2))|
      ≤ |a1 - m1| * Real.sqrt |v1 - v2| + d1 * (|a1 - a2| + |m1 - m2|) := by
    refine (abs_add_le _ _).trans (add_le_add ?_ ?_)
    · rw [abs_mul]
      refine mul_le_mul_of_nonneg_left ?_ (abs_nonneg _)
      have hdd : d2 - d1 = Real.sqrt v2 - Real.sqrt v1 := by rw [hd1, hd2]; ring
      rw [hdd]
      calc |Real.sqrt v2 - Real.sqrt v1| ≤ Real.sqrt |v2 - v1| := abs_sqrt_sub_sqrt_le hv2 hv1
        _ = Real.sqrt |v1 - v2| := by rw [abs_sub_comm]
    · rw [abs_mul, abs_of_pos hd1p]
      refine mul_le_mul_of_nonneg_left ?_ hd1p.le
      calc |(a1 - m1) - (a2 - m2)| = |(a1 - a2) - (m1 - m2)| := by ring_nf
        _ ≤ |a1 - a2| + |m1 - m2| := abs_sub _ _
  have heq : (|a1 - m1| * Real.sqrt |v1 - v2| + d1 * (|a1 - a2| + |m1 - m2|)) / (d1 * d2)
      = |a1 - m1| * Real.sqrt |v1 - v2| / (d1 * d2) + (|a1 - a2| + |m1 - m2|) / d2 := by
    rw [add_div]
    congr 1
    exact mul_div_mul_left _ d2 (ne_of_gt hd1p)
  exact (div_le_div_of_nonneg_right hnum hprod.le).trans (le_of_eq heq)

end Puffer.RL.AdvNormPerturb
