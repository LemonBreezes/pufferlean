/-
Advantage normalization over ℝ, with a division-perturbation error bound.

PufferLib normalizes advantages per minibatch (`pufferlib.cu` ~line 829):

    adv_normalized = (adv − adv_mean) / (sqrt(adv_var) + 1e-8)

where `adv_mean`, `adv_var` are (nonassociative) reductions over the minibatch.
The `+ε` floor guarantees a strictly positive denominator, which is exactly what
makes the error controllable: dividing by something `≥ ε` bounds the amplification
of the numerator/denominator errors. The centerpiece is `div_perturb_bound`, a
reusable quotient error lemma; `advNorm_perturb` specializes it to normalization.
-/
import Mathlib
import Puffer.Numeric.Bf16
import Puffer.Numeric.Reduction

namespace Puffer.RL.Normalize

open Finset

/-- **Division perturbation bound.** If `n̂,d̂` approximate `n,d` within `εₙ,ε_d` and
    both denominators have magnitude `≥ dmin > 0`, then the quotient error is
    `|n̂/d̂ − n/d| ≤ (εₙ + |n/d|·ε_d) / dmin`. -/
theorem div_perturb_bound (n nhat d dhat εn εd dmin : ℝ)
    (hdmin : 0 < dmin) (hd : dmin ≤ |d|) (hdhat : dmin ≤ |dhat|)
    (hn : |nhat - n| ≤ εn) (hdd : |dhat - d| ≤ εd) :
    |nhat / dhat - n / d| ≤ (εn + |n / d| * εd) / dmin := by
  have hdpos : (0 : ℝ) < |d| := lt_of_lt_of_le hdmin hd
  have hdhpos : (0 : ℝ) < |dhat| := lt_of_lt_of_le hdmin hdhat
  have hdne : d ≠ 0 := abs_pos.mp hdpos
  have hdhne : dhat ≠ 0 := abs_pos.mp hdhpos
  have hda : |d| ≠ 0 := ne_of_gt hdpos
  have hdha : |dhat| ≠ 0 := ne_of_gt hdhpos
  have hεn : 0 ≤ εn := le_trans (abs_nonneg _) hn
  have hεd : 0 ≤ εd := le_trans (abs_nonneg _) hdd
  have hsub : |d - dhat| ≤ εd := by rw [abs_sub_comm]; exact hdd
  have key : nhat / dhat - n / d = (nhat - n) / dhat + n * (d - dhat) / (dhat * d) := by
    field_simp; ring
  -- term 1: |(nhat - n)/dhat| ≤ εn/dmin
  have ht1 : |(nhat - n) / dhat| ≤ εn / dmin := by
    rw [abs_div, div_le_div_iff₀ hdhpos hdmin]
    calc |nhat - n| * dmin ≤ εn * dmin := mul_le_mul_of_nonneg_right hn hdmin.le
      _ ≤ εn * |dhat| := mul_le_mul_of_nonneg_left hdhat hεn
  -- term 2: |n·(d−dhat)/(dhat·d)| ≤ |n/d|·εd/dmin
  have ht2b : |d - dhat| / |dhat| ≤ εd / dmin := by
    rw [div_le_div_iff₀ hdhpos hdmin]
    calc |d - dhat| * dmin ≤ εd * dmin := mul_le_mul_of_nonneg_right hsub hdmin.le
      _ ≤ εd * |dhat| := mul_le_mul_of_nonneg_left hdhat hεd
  have ht2 : |n * (d - dhat) / (dhat * d)| ≤ |n / d| * εd / dmin := by
    have e1 : |n * (d - dhat) / (dhat * d)| = |n| / |d| * (|d - dhat| / |dhat|) := by
      rw [abs_div, abs_mul, abs_mul]; field_simp
    have e2 : |n / d| * εd / dmin = |n| / |d| * (εd / dmin) := by rw [abs_div]; ring
    rw [e1, e2]
    exact mul_le_mul_of_nonneg_left ht2b (by positivity)
  calc |nhat / dhat - n / d|
      = |(nhat - n) / dhat + n * (d - dhat) / (dhat * d)| := by rw [key]
    _ ≤ |(nhat - n) / dhat| + |n * (d - dhat) / (dhat * d)| := abs_add_le _ _
    _ ≤ εn / dmin + |n / d| * εd / dmin := add_le_add ht1 ht2
    _ = (εn + |n / d| * εd) / dmin := by rw [add_div]

/-- Normalized advantage: `(adv − mean) / (√var + ε)`. -/
noncomputable def advNorm (adv mean var ε : ℝ) : ℝ := (adv - mean) / (Real.sqrt var + ε)

/-- The `+ε` floor makes the denominator strictly positive — normalization is total. -/
theorem advNorm_denom_pos (var ε : ℝ) (hε : 0 < ε) : 0 < Real.sqrt var + ε := by
  have := Real.sqrt_nonneg var; linarith

/-- **Centering is exact.** Normalized advantages sum to zero over the minibatch
    (all share the same denominator, and the mean-subtracted numerators cancel). -/
theorem advNorm_sum_eq_zero {ι : Type*} (s : Finset ι) (adv : ι → ℝ) (var ε : ℝ)
    (hs : s.Nonempty) :
    ∑ i ∈ s, advNorm (adv i) ((∑ j ∈ s, adv j) / s.card) var ε = 0 := by
  have hcard : (s.card : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Finset.card_ne_zero.mpr hs)
  unfold advNorm
  rw [← Finset.sum_div]
  have hnum : ∑ i ∈ s, (adv i - (∑ j ∈ s, adv j) / s.card) = 0 := by
    rw [Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul]
    have : (s.card : ℝ) * ((∑ j ∈ s, adv j) / s.card) = ∑ j ∈ s, adv j := by field_simp
    rw [this, sub_self]
  rw [hnum, zero_div]

/-- **Unit-variance normalization (sum form).** With the `ε`-floor removed (`ε = 0`) and `var` set to the biased
    sample variance `(Σ(aᵢ−mean)²)/n`, the squared normalized advantages sum to the count: `Σ advNorm(aᵢ)² = n`.
    Each term is `(aᵢ−mean)²/var` (since `(√var)² = var`), and the sum of squared deviations is exactly `var·n`,
    so the `var` cancels. The scale half of normalization (its centering counterpart is `advNorm_sum_eq_zero`):
    dividing by `√var` rescales the spread to exactly 1. `hpos : 0 < var` rules out the degenerate all-equal
    minibatch (and forces `n > 0`). -/
theorem advNorm_sum_sq_eq_card {ι : Type*} (s : Finset ι) (adv : ι → ℝ) (mean var : ℝ)
    (hvar : var = (∑ i ∈ s, (adv i - mean) ^ 2) / s.card) (hpos : 0 < var) :
    ∑ i ∈ s, (advNorm (adv i) mean var 0) ^ 2 = s.card := by
  have hcardpos : 0 < s.card := by
    rcases Nat.eq_zero_or_pos s.card with h | h
    · exfalso; rw [hvar, h, Nat.cast_zero, div_zero] at hpos; exact lt_irrefl 0 hpos
    · exact h
  have hcard : (s.card : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hcardpos.ne'
  have hsum : ∑ i ∈ s, (adv i - mean) ^ 2 = var * s.card := by
    rw [hvar]; field_simp
  calc ∑ i ∈ s, (advNorm (adv i) mean var 0) ^ 2
      = ∑ i ∈ s, (adv i - mean) ^ 2 / var := by
        apply Finset.sum_congr rfl; intro i _
        unfold advNorm
        rw [add_zero, div_pow, Real.sq_sqrt hpos.le]
    _ = (∑ i ∈ s, (adv i - mean) ^ 2) / var := by rw [← Finset.sum_div]
    _ = (var * s.card) / var := by rw [hsum]
    _ = s.card := by field_simp

/-- **Unit-variance normalization (mean form).** The mean of the squared normalized advantages is exactly `1` —
    since they are zero-mean (`advNorm_sum_eq_zero`), their sample variance is `1`. This is the defining purpose of
    advantage normalization: rescale the minibatch to zero mean and unit variance. -/
theorem advNorm_meanSq_eq_one {ι : Type*} (s : Finset ι) (adv : ι → ℝ) (mean var : ℝ)
    (hvar : var = (∑ i ∈ s, (adv i - mean) ^ 2) / s.card) (hpos : 0 < var) :
    (∑ i ∈ s, (advNorm (adv i) mean var 0) ^ 2) / s.card = 1 := by
  have hcardpos : 0 < s.card := by
    rcases Nat.eq_zero_or_pos s.card with h | h
    · exfalso; rw [hvar, h, Nat.cast_zero, div_zero] at hpos; exact lt_irrefl 0 hpos
    · exact h
  have hcard : (s.card : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hcardpos.ne'
  rw [advNorm_sum_sq_eq_card s adv mean var hvar hpos, div_self hcard]

/-- **Advantage normalization is invariant under positive affine reparametrization.** Rescaling every
    advantage by `c > 0` and shifting by `b` (`aᵢ ↦ c·aᵢ + b`) sends the minibatch mean to `c·mean + b` and
    the sample variance to `c²·var`; with the `ε`-floor removed (`ε = 0`) the normalized advantage is
    UNCHANGED: `advNorm (c·a + b) (c·mean + b) (c²·var) 0 = advNorm a mean var 0`. This is the algebraic core
    of why advantage normalization makes the PPO objective invariant to reward scaling/shifting (the ratio
    `(aᵢ − mean)/√var` is scale-and-shift free). `0 < c` is load-bearing: for `c < 0` the sign flips (`advNorm`
    would negate); and `ε = 0` is essential — a nonzero denominator floor `√var + ε` does not rescale with `c`,
    so invariance fails for `ε > 0` (matching how the file's other exact-normalization theorems use `ε = 0`).
    Proof: `√(c²·var) = |c|·√var = c·√var` (`Real.sqrt_mul` + `Real.sqrt_sq`, needing `0 ≤ c`), then cancel the
    common factor `c` from numerator `c·(a − mean)` and denominator (`mul_div_mul_left`, needing `c ≠ 0`). -/
theorem advNorm_affine_invariant (a mean var b c : ℝ) (hc : 0 < c) :
    advNorm (c * a + b) (c * mean + b) (c ^ 2 * var) 0 = advNorm a mean var 0 := by
  unfold advNorm
  simp only [add_zero]
  rw [Real.sqrt_mul (sq_nonneg c) var, Real.sqrt_sq hc.le]
  rw [show c * a + b - (c * mean + b) = c * (a - mean) from by ring]
  rw [mul_div_mul_left _ _ hc.ne']

/-- **Advantage normalization preserves order.** `advNorm a1 mean var ε ≤ advNorm a2 mean var ε ↔ a1 ≤ a2` —
    dividing the centered advantages by the positive denominator `√var + ε` is order-preserving, so
    normalization keeps the relative ranking of samples (which are more/less advantageous). -/
theorem advNorm_le_advNorm_iff (a1 a2 mean var ε : ℝ) (hε : 0 < ε) :
    advNorm a1 mean var ε ≤ advNorm a2 mean var ε ↔ a1 ≤ a2 := by
  unfold advNorm
  rw [div_le_div_iff_of_pos_right (advNorm_denom_pos var ε hε)]
  constructor <;> intro h <;> linarith

/-- **Advantage normalization preserves sign.** `0 < advNorm a mean var ε ↔ mean < a` — a normalized advantage
    is positive exactly when the raw advantage is above the mean. So the policy-gradient sign (reinforce vs
    discourage an action) is preserved by normalization. -/
theorem advNorm_pos_iff (a mean var ε : ℝ) (hε : 0 < ε) :
    0 < advNorm a mean var ε ↔ mean < a := by
  unfold advNorm
  rw [lt_div_iff₀ (advNorm_denom_pos var ε hε)]
  constructor <;> intro h <;> nlinarith [advNorm_denom_pos var ε hε]

/-- **Advantage-normalization error bound.** Errors `εa` in the advantage, `εm` in the
    mean, and the sqrt-of-variance error propagate to the normalized advantage,
    amplified by at most `1/ε` (the denominator floor). -/
theorem advNorm_perturb (adv mean var advH meanH varH ε εa εm : ℝ) (hε : 0 < ε)
    (ha : |advH - adv| ≤ εa) (hm : |meanH - mean| ≤ εm) :
    |advNorm advH meanH varH ε - advNorm adv mean var ε|
      ≤ ((εa + εm) + |advNorm adv mean var ε| * |Real.sqrt varH - Real.sqrt var|) / ε := by
  unfold advNorm
  have hd : ε ≤ |Real.sqrt var + ε| := by
    rw [abs_of_pos (advNorm_denom_pos var ε hε)]; have := Real.sqrt_nonneg var; linarith
  have hdH : ε ≤ |Real.sqrt varH + ε| := by
    rw [abs_of_pos (advNorm_denom_pos varH ε hε)]; have := Real.sqrt_nonneg varH; linarith
  have hn : |(advH - meanH) - (adv - mean)| ≤ εa + εm := by
    have he : (advH - meanH) - (adv - mean) = (advH - adv) + (-(meanH - mean)) := by ring
    rw [he]
    calc |(advH - adv) + (-(meanH - mean))|
        ≤ |advH - adv| + |-(meanH - mean)| := abs_add_le _ _
      _ = |advH - adv| + |meanH - mean| := by rw [abs_neg]
      _ ≤ εa + εm := add_le_add ha hm
  have hdd : |(Real.sqrt varH + ε) - (Real.sqrt var + ε)| = |Real.sqrt varH - Real.sqrt var| := by
    congr 1; ring
  exact div_perturb_bound (adv - mean) (advH - meanH) (Real.sqrt var + ε) (Real.sqrt varH + ε)
    (εa + εm) |Real.sqrt varH - Real.sqrt var| ε hε hd hdH hn hdd.le

end Puffer.RL.Normalize
