/-
# The MATRIX Newton–Schulz map is operator-norm Lipschitz — via submultiplicativity + telescoping

C44 (`NewtonSchulzLipschitz`) bounded the SCALAR Newton–Schulz map's Lipschitz constant
`L_ns = |a| + 3|b|M² + 5|c|M⁴` on `|σ| ≤ M`, and (conservatively) left the MATRIX lift open, thinking it needed
SVD / singular-vector perturbation analysis. It does NOT: the matrix Newton–Schulz map is a matrix POLYNOMIAL in `X`
and `Xᵀ` (Muon's odd map is `X ↦ a·X + b·(XXᵀ)X + c·(XXᵀ)²X`), and a matrix polynomial is Lipschitz on a bounded
operator-norm ball by pure SUBMULTIPLICATIVITY (`‖A·B‖ ≤ ‖A‖·‖B‖`) + TELESCOPING (swap one factor at a time),
using only `‖Xᵀ‖ = ‖X‖`. No SVD, no singular-vector rotation, no eigenvalue perturbation.

This module proves that ABSTRACTLY, in a normed ∗-ring `[NormedRing R] [NormedAlgebra ℝ R] [StarRing R]
[NormedStarGroup R]` — the transpose is the `star` (`‖star x‖ = ‖x‖` is exactly `NormedStarGroup`), and the real
square matrices with the L2 operator norm and conjugate-transpose are an instance (as used throughout
`MuonGramBound`/`SpectralBridge`). Writing the odd Muon map as `nsStarStep a b c x = a•x + b•(x·xᴴ·x) +
c•(x·xᴴ·x·xᴴ·x)` (with `xᴴ = star x`):

* `sq3_lipschitz` — the cubic monomial `x·xᴴ·x` is `3M²`-Lipschitz on `‖x‖ ≤ M` (3-term telescope, each term
  `≤ M²·‖x−y‖`).
* `sq5_lipschitz` — the quintic monomial `x·xᴴ·x·xᴴ·x` is `5M⁴`-Lipschitz (5-term telescope, each `≤ M⁴·‖x−y‖`).
* `nsStarStep_lipschitz` — the whole odd Muon Newton–Schulz step: `‖nsStarStep a b c x − nsStarStep a b c y‖ ≤
  (|a| + 3|b|M² + 5|c|M⁴)·‖x − y‖` — the SAME constant `L_ns` as C44's scalar bound, now in the matrix operator norm.
* `nsStarStep_lipschitz_delta` — in C42's `hlip` shape (`≤ L·δ` given `‖x − y‖ ≤ δ`), so it directly feeds
  `MuonAscentBridge.muon_whole_run_opnorm_interval`.

**Scope (honestly disclosed).** This is the operator-norm Lipschitz of ONE Newton–Schulz step of the odd Muon map,
proved for the abstract normed ∗-ring that the repo's real square matrices (L2 operator norm + conjugate transpose)
instantiate — closing C44's deferred matrix lift with NO SVD, via submultiplicativity + telescoping. The constant
`L_ns = |a| + 3|b|M² + 5|c|M⁴` on `‖X‖ ≤ M` matches C44's scalar sup exactly. What remains toward C42's full Muon
whole-run `L`: (i) instantiating this abstract bound at the repo's concrete matrix type (the `[NormedRing]
[NormedStarGroup]` instance for the L2 matrix operator norm — a standard C*-structure, not established here); (ii)
composing the `k` Newton–Schulz iterations of one Muon step (a `k`-fold composition, each `L_ns`-Lipschitz on its
range, so the composite is `L_ns^k`-Lipschitz on the nested balls); (iii) composing with the momentum/parameter
update. This supplies the core per-iteration operator-norm Lipschitz — the ingredient C44 flagged as the hard part.
-/
import Mathlib

namespace Puffer.RL.NewtonSchulzMatrixLipschitz

variable {R : Type*} [NormedRing R] [NormedAlgebra ℝ R] [StarRing R] [NormedStarGroup R]

omit [NormedAlgebra ℝ R] [StarRing R] [NormedStarGroup R] in
/-- Norm submultiplicativity across a triple product: `‖a·b·c‖ ≤ ‖a‖·‖b‖·‖c‖`. -/
theorem norm_mul3_le (a b c : R) : ‖a * b * c‖ ≤ ‖a‖ * ‖b‖ * ‖c‖ :=
  (norm_mul_le _ _).trans (mul_le_mul_of_nonneg_right (norm_mul_le _ _) (norm_nonneg _))

omit [NormedAlgebra ℝ R] [StarRing R] [NormedStarGroup R] in
/-- Norm submultiplicativity across a quadruple product. -/
theorem norm_mul4_le (a b c d : R) : ‖a * b * c * d‖ ≤ ‖a‖ * ‖b‖ * ‖c‖ * ‖d‖ :=
  (norm_mul_le _ _).trans (mul_le_mul_of_nonneg_right (norm_mul3_le a b c) (norm_nonneg _))

omit [NormedAlgebra ℝ R] [StarRing R] [NormedStarGroup R] in
/-- Norm submultiplicativity across a quintuple product. -/
theorem norm_mul5_le (a b c d e : R) : ‖a * b * c * d * e‖ ≤ ‖a‖ * ‖b‖ * ‖c‖ * ‖d‖ * ‖e‖ :=
  (norm_mul_le _ _).trans (mul_le_mul_of_nonneg_right (norm_mul4_le a b c d) (norm_nonneg _))

omit [NormedAlgebra ℝ R] in
/-- The transpose (`star`) is norm-preserving on differences: `‖star x − star y‖ = ‖x − y‖`. -/
theorem norm_star_sub (x y : R) : ‖star x - star y‖ = ‖x - y‖ := by
  rw [← star_sub, norm_star]

omit [NormedAlgebra ℝ R] in
/-- **The cubic Newton–Schulz monomial `x·xᴴ·x` is `3M²`-Lipschitz on `‖x‖ ≤ M`.** Telescope
    `x·xᴴ·x − y·yᴴ·y = (x−y)·xᴴ·x + y·(xᴴ−yᴴ)·x + y·yᴴ·(x−y)`; each of the three terms has norm `≤ M²·‖x−y‖` by
    submultiplicativity + `‖star ·‖ = ‖·‖`. -/
theorem sq3_lipschitz (M : ℝ) (x y : R) (hM : 0 ≤ M) (hx : ‖x‖ ≤ M) (hy : ‖y‖ ≤ M) :
    ‖x * star x * x - y * star y * y‖ ≤ 3 * M ^ 2 * ‖x - y‖ := by
  have htel : x * star x * x - y * star y * y
      = (x - y) * star x * x + y * (star x - star y) * x + y * star y * (x - y) := by
    noncomm_ring
  have hxx : ‖x‖ * ‖x‖ ≤ M ^ 2 := by rw [sq]; exact mul_le_mul hx hx (norm_nonneg _) hM
  have hyx : ‖y‖ * ‖x‖ ≤ M ^ 2 := by rw [sq]; exact mul_le_mul hy hx (norm_nonneg _) hM
  have hyy : ‖y‖ * ‖y‖ ≤ M ^ 2 := by rw [sq]; exact mul_le_mul hy hy (norm_nonneg _) hM
  have h1 : ‖(x - y) * star x * x‖ ≤ M ^ 2 * ‖x - y‖ := by
    refine (norm_mul3_le _ _ _).trans ?_
    rw [norm_star]
    calc ‖x - y‖ * ‖x‖ * ‖x‖ = (‖x‖ * ‖x‖) * ‖x - y‖ := by ring
      _ ≤ M ^ 2 * ‖x - y‖ := mul_le_mul_of_nonneg_right hxx (norm_nonneg _)
  have h2 : ‖y * (star x - star y) * x‖ ≤ M ^ 2 * ‖x - y‖ := by
    refine (norm_mul3_le _ _ _).trans ?_
    rw [norm_star_sub]
    calc ‖y‖ * ‖x - y‖ * ‖x‖ = (‖y‖ * ‖x‖) * ‖x - y‖ := by ring
      _ ≤ M ^ 2 * ‖x - y‖ := mul_le_mul_of_nonneg_right hyx (norm_nonneg _)
  have h3 : ‖y * star y * (x - y)‖ ≤ M ^ 2 * ‖x - y‖ := by
    refine (norm_mul3_le _ _ _).trans ?_
    rw [norm_star]
    calc ‖y‖ * ‖y‖ * ‖x - y‖ = (‖y‖ * ‖y‖) * ‖x - y‖ := by ring
      _ ≤ M ^ 2 * ‖x - y‖ := mul_le_mul_of_nonneg_right hyy (norm_nonneg _)
  rw [htel]
  calc ‖(x - y) * star x * x + y * (star x - star y) * x + y * star y * (x - y)‖
      ≤ ‖(x - y) * star x * x‖ + ‖y * (star x - star y) * x‖ + ‖y * star y * (x - y)‖ := by
        refine (norm_add_le _ _).trans (add_le_add ?_ le_rfl)
        exact norm_add_le _ _
    _ ≤ M ^ 2 * ‖x - y‖ + M ^ 2 * ‖x - y‖ + M ^ 2 * ‖x - y‖ := add_le_add (add_le_add h1 h2) h3
    _ = 3 * M ^ 2 * ‖x - y‖ := by ring

omit [NormedAlgebra ℝ R] in
/-- **The quintic Newton–Schulz monomial `x·xᴴ·x·xᴴ·x` is `5M⁴`-Lipschitz on `‖x‖ ≤ M`.** Telescope over the five
    factors (swapping `x`→`y` one at a time, `xᴴ`→`yᴴ` at the star positions); each of the five terms has norm
    `≤ M⁴·‖x−y‖`. -/
theorem sq5_lipschitz (M : ℝ) (x y : R) (_hM : 0 ≤ M) (hx : ‖x‖ ≤ M) (hy : ‖y‖ ≤ M) :
    ‖x * star x * x * star x * x - y * star y * y * star y * y‖ ≤ 5 * M ^ 4 * ‖x - y‖ := by
  have htel : x * star x * x * star x * x - y * star y * y * star y * y
      = (x - y) * star x * x * star x * x
        + y * (star x - star y) * x * star x * x
        + y * star y * (x - y) * star x * x
        + y * star y * y * (star x - star y) * x
        + y * star y * y * star y * (x - y) := by
    noncomm_ring
  -- each of the four "M-factors" (any mix of ‖x‖/‖y‖) is ≤ M
  have hxM : ‖x‖ ≤ M := hx
  have hyM : ‖y‖ ≤ M := hy
  have hxy0 : (0:ℝ) ≤ ‖x - y‖ := norm_nonneg _
  -- helper: a product of four norms each ≤ M, times ‖x-y‖, is ≤ M^4 * ‖x-y‖
  have key : ∀ p q r s : ℝ, 0 ≤ p → 0 ≤ q → 0 ≤ r → 0 ≤ s → p ≤ M → q ≤ M → r ≤ M → s ≤ M →
      p * q * r * s * ‖x - y‖ ≤ M ^ 4 * ‖x - y‖ := by
    intro p q r s hp hq hr hs hpM hqM hrM hsM
    have hpqrs : p * q * r * s ≤ M ^ 4 := by
      have h1 : p * q ≤ M * M := mul_le_mul hpM hqM hq (le_trans hp hpM)
      have h2 : r * s ≤ M * M := mul_le_mul hrM hsM hs (le_trans hr hrM)
      calc p * q * r * s = (p * q) * (r * s) := by ring
        _ ≤ (M * M) * (M * M) :=
            mul_le_mul h1 h2 (mul_nonneg hr hs) (mul_nonneg (le_trans hp hpM) (le_trans hq hqM))
        _ = M ^ 4 := by ring
    exact mul_le_mul_of_nonneg_right hpqrs hxy0
  have h1 : ‖(x - y) * star x * x * star x * x‖ ≤ M ^ 4 * ‖x - y‖ := by
    refine (norm_mul5_le _ _ _ _ _).trans ?_
    simp only [norm_star]
    calc ‖x - y‖ * ‖x‖ * ‖x‖ * ‖x‖ * ‖x‖ = ‖x‖ * ‖x‖ * ‖x‖ * ‖x‖ * ‖x - y‖ := by ring
      _ ≤ M ^ 4 * ‖x - y‖ := key _ _ _ _ (norm_nonneg _) (norm_nonneg _) (norm_nonneg _)
          (norm_nonneg _) hxM hxM hxM hxM
  have h2 : ‖y * (star x - star y) * x * star x * x‖ ≤ M ^ 4 * ‖x - y‖ := by
    refine (norm_mul5_le _ _ _ _ _).trans ?_
    simp only [norm_star, norm_star_sub]
    calc ‖y‖ * ‖x - y‖ * ‖x‖ * ‖x‖ * ‖x‖ = ‖y‖ * ‖x‖ * ‖x‖ * ‖x‖ * ‖x - y‖ := by ring
      _ ≤ M ^ 4 * ‖x - y‖ := key _ _ _ _ (norm_nonneg _) (norm_nonneg _) (norm_nonneg _)
          (norm_nonneg _) hyM hxM hxM hxM
  have h3 : ‖y * star y * (x - y) * star x * x‖ ≤ M ^ 4 * ‖x - y‖ := by
    refine (norm_mul5_le _ _ _ _ _).trans ?_
    simp only [norm_star]
    calc ‖y‖ * ‖y‖ * ‖x - y‖ * ‖x‖ * ‖x‖ = ‖y‖ * ‖y‖ * ‖x‖ * ‖x‖ * ‖x - y‖ := by ring
      _ ≤ M ^ 4 * ‖x - y‖ := key _ _ _ _ (norm_nonneg _) (norm_nonneg _) (norm_nonneg _)
          (norm_nonneg _) hyM hyM hxM hxM
  have h4 : ‖y * star y * y * (star x - star y) * x‖ ≤ M ^ 4 * ‖x - y‖ := by
    refine (norm_mul5_le _ _ _ _ _).trans ?_
    simp only [norm_star, norm_star_sub]
    calc ‖y‖ * ‖y‖ * ‖y‖ * ‖x - y‖ * ‖x‖ = ‖y‖ * ‖y‖ * ‖y‖ * ‖x‖ * ‖x - y‖ := by ring
      _ ≤ M ^ 4 * ‖x - y‖ := key _ _ _ _ (norm_nonneg _) (norm_nonneg _) (norm_nonneg _)
          (norm_nonneg _) hyM hyM hyM hxM
  have h5 : ‖y * star y * y * star y * (x - y)‖ ≤ M ^ 4 * ‖x - y‖ := by
    refine (norm_mul5_le _ _ _ _ _).trans ?_
    simp only [norm_star]
    calc ‖y‖ * ‖y‖ * ‖y‖ * ‖y‖ * ‖x - y‖ = ‖y‖ * ‖y‖ * ‖y‖ * ‖y‖ * ‖x - y‖ := by ring
      _ ≤ M ^ 4 * ‖x - y‖ := key _ _ _ _ (norm_nonneg _) (norm_nonneg _) (norm_nonneg _)
          (norm_nonneg _) hyM hyM hyM hyM
  rw [htel]
  calc ‖(x - y) * star x * x * star x * x + y * (star x - star y) * x * star x * x
          + y * star y * (x - y) * star x * x + y * star y * y * (star x - star y) * x
          + y * star y * y * star y * (x - y)‖
      ≤ ‖(x - y) * star x * x * star x * x‖ + ‖y * (star x - star y) * x * star x * x‖
          + ‖y * star y * (x - y) * star x * x‖ + ‖y * star y * y * (star x - star y) * x‖
          + ‖y * star y * y * star y * (x - y)‖ := by
        refine (norm_add_le _ _).trans (add_le_add ?_ le_rfl)
        refine (norm_add_le _ _).trans (add_le_add ?_ le_rfl)
        refine (norm_add_le _ _).trans (add_le_add ?_ le_rfl)
        exact norm_add_le _ _
    _ ≤ M ^ 4 * ‖x - y‖ + M ^ 4 * ‖x - y‖ + M ^ 4 * ‖x - y‖ + M ^ 4 * ‖x - y‖ + M ^ 4 * ‖x - y‖ :=
        add_le_add (add_le_add (add_le_add (add_le_add h1 h2) h3) h4) h5
    _ = 5 * M ^ 4 * ‖x - y‖ := by ring

/-- **The odd Muon Newton–Schulz step** `a•x + b•(x·xᴴ·x) + c•(x·xᴴ·x·xᴴ·x)` (`xᴴ = star x`). Equals Muon's
    `a·X + b·(XXᵀ)X + c·(XXᵀ)²X` when `star` is the (conjugate) transpose. -/
def nsStarStep (a b c : ℝ) (x : R) : R :=
  a • x + b • (x * star x * x) + c • (x * star x * x * star x * x)

/-- **CAPSTONE: the matrix Newton–Schulz step is `L_ns`-Lipschitz in the operator norm.** On `‖x‖, ‖y‖ ≤ M`,
    `‖nsStarStep a b c x − nsStarStep a b c y‖ ≤ (|a| + 3|b|M² + 5|c|M⁴)·‖x − y‖` — the SAME `L_ns` as C44's scalar
    bound, now for the matrix operator norm, via `sq3_lipschitz`/`sq5_lipschitz` over the linear `add`/`smul`
    structure. No SVD: pure submultiplicativity + telescoping. -/
theorem nsStarStep_lipschitz (a b c M : ℝ) (x y : R) (hM : 0 ≤ M) (hx : ‖x‖ ≤ M) (hy : ‖y‖ ≤ M) :
    ‖nsStarStep a b c x - nsStarStep a b c y‖
      ≤ (|a| + 3 * |b| * M ^ 2 + 5 * |c| * M ^ 4) * ‖x - y‖ := by
  have hdiff : nsStarStep a b c x - nsStarStep a b c y
      = a • (x - y) + b • (x * star x * x - y * star y * y)
        + c • (x * star x * x * star x * x - y * star y * y * star y * y) := by
    simp only [nsStarStep, smul_sub]; abel
  have ha : ‖a • (x - y)‖ = |a| * ‖x - y‖ := by rw [norm_smul, Real.norm_eq_abs]
  have hb : ‖b • (x * star x * x - y * star y * y)‖ ≤ |b| * (3 * M ^ 2 * ‖x - y‖) := by
    rw [norm_smul, Real.norm_eq_abs]
    exact mul_le_mul_of_nonneg_left (sq3_lipschitz M x y hM hx hy) (abs_nonneg b)
  have hc : ‖c • (x * star x * x * star x * x - y * star y * y * star y * y)‖
      ≤ |c| * (5 * M ^ 4 * ‖x - y‖) := by
    rw [norm_smul, Real.norm_eq_abs]
    exact mul_le_mul_of_nonneg_left (sq5_lipschitz M x y hM hx hy) (abs_nonneg c)
  rw [hdiff]
  calc ‖a • (x - y) + b • (x * star x * x - y * star y * y)
          + c • (x * star x * x * star x * x - y * star y * y * star y * y)‖
      ≤ ‖a • (x - y)‖ + ‖b • (x * star x * x - y * star y * y)‖
          + ‖c • (x * star x * x * star x * x - y * star y * y * star y * y)‖ := by
        refine (norm_add_le _ _).trans (add_le_add ?_ le_rfl)
        exact norm_add_le _ _
    _ ≤ |a| * ‖x - y‖ + |b| * (3 * M ^ 2 * ‖x - y‖) + |c| * (5 * M ^ 4 * ‖x - y‖) :=
        add_le_add (add_le_add (le_of_eq ha) hb) hc
    _ = (|a| + 3 * |b| * M ^ 2 + 5 * |c| * M ^ 4) * ‖x - y‖ := by ring

/-- The capstone in C42's `hlip` shape: given `‖x − y‖ ≤ δ`, the step differs by `≤ L_ns·δ` — directly feeds
    `MuonAscentBridge.muon_whole_run_opnorm_interval` as the per-step operator-norm Lipschitz. -/
theorem nsStarStep_lipschitz_delta (a b c M δ : ℝ) (x y : R) (hM : 0 ≤ M) (hx : ‖x‖ ≤ M) (hy : ‖y‖ ≤ M)
    (hδ : ‖x - y‖ ≤ δ) :
    ‖nsStarStep a b c x - nsStarStep a b c y‖ ≤ (|a| + 3 * |b| * M ^ 2 + 5 * |c| * M ^ 4) * δ := by
  refine (nsStarStep_lipschitz a b c M x y hM hx hy).trans ?_
  refine mul_le_mul_of_nonneg_left hδ ?_
  positivity

end Puffer.RL.NewtonSchulzMatrixLipschitz
