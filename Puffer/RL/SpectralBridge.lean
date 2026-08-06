/-
The SVD spectral bridge for the l2 operator norm — the piece the tight Newton–Schulz tower
(`Puffer/RL/NewtonSchulzTight.lean`) flagged as "prohibitive in Mathlib".

The tight, O(1) Newton–Schulz constant is contingent on a genuine SPECTRAL fact, not a rounding
fact: for the Muon map `p(X) = a·X + b·X³ + c·X⁵` (an ODD polynomial, so `p(X) = X·q(XᵀX)` with
`q(t) = a + b·t + c·t²`), the operator norm `‖p(X)‖₂` equals `maxᵢ |p_scalar(σᵢ)|` where the `σᵢ`
are the singular values of `X`. That identity reduces the matrix norm bound to a SCALAR bound on
the singular values — the mechanism by which the singular values are driven to 1 and the norm stays
O(1) across iterations.

Mathlib (v4.28.0) has NO `singularValues` / SVD / polar-decomposition API, and its `Matrix` type is
not registered as a `CStarAlgebra` (the l2 operator norm is a *scoped* instance), so the abstract
C*-algebra spectral-radius / CFC machinery does not fire on `Matrix`. This file nonetheless PROVES
the operator-norm ≤-bound directly, routing through the tools Mathlib *does* have:

  • `Matrix.IsHermitian.spectral_theorem` : `H = U · diag(λ) · Uᴴ` for Hermitian `H`;
  • `l2_opNorm_conjTranspose_mul_self` : `‖Aᴴ A‖ = ‖A‖²` (the C*-identity for the l2 op norm);
  • `l2_opNorm_mul` / `l2_opNorm_diagonal` : submultiplicativity + diagonal norm = sup norm;
  • `eigenvalues_conjTranspose_mul_self_nonneg` : eigenvalues of `Aᴴ A` are `≥ 0` (= σ²).

Results (all axiom-clean — only `propext`/`Classical.choice`/`Quot.sound`, no `sorry`):

  • `norm_unitary`      : a unitary matrix has l2 operator norm `1`.
  • `norm_conj_unitary` : conjugation by a unitary preserves the l2 operator norm.
  • `opNorm_le_of_eigenvalue_bound`      : `‖H‖ ≤ c` from `∀ i, |λᵢ(H)| ≤ c`  (Hermitian `H`).
  • `opNorm_le_of_gram_eigenvalue_bound` : `‖A‖ ≤ c` from `∀ i, λᵢ(Aᴴ A) ≤ c²`  (any `A`).
      Since `λᵢ(Aᴴ A) = σᵢ²`, this is exactly `‖A‖₂ = maxᵢ σᵢ ≤ c` — the singular-value
      characterization of the spectral norm, in the ≤-form that propagates a norm bound.

WHAT REMAINS (honest scope) for the full O(1) Newton–Schulz retrofit:
  (a) Express the Muon Gram matrix `p(X)ᴴ p(X) = q(M)·M·q(M) = M·q(M)²` (`M = XᵀX`) and identify its
      eigenvalues with `λ·q(λ)² = p_scalar(√λ)²` — a polynomial-of-a-Hermitian-matrix eigenvalue
      computation (Mathlib `charpoly_cfc_eq` / spectral mapping is the likely lever).
  (b) The SCALAR bound `t·q(t)² ≤ C²` on `t ∈ [0, ‖X‖²]` — the razor-thin tuned-coefficient
      inequality that `nlinarith` could not previously discharge (max ≈ 1.273; genuinely hard).
  (c) Bridge Mathlib `Matrix` ↔ the runnable float `Mat`/`matmul` object.
This file delivers the spectral REDUCTION (a matrix operator-norm bound from a scalar/eigenvalue
bound) that (a)+(b) would feed; it removes "prohibitive in Mathlib" as the blocker.
-/
import Mathlib

namespace Puffer.RL.SpectralBridge

open scoped Matrix Matrix.Norms.L2Operator
open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n] {𝕜 : Type*} [RCLike 𝕜]

/-- A unitary matrix has l2 operator norm `1`. -/
lemma norm_unitary [Nonempty n] (U : Matrix.unitaryGroup n 𝕜) :
    ‖(U : Matrix n n 𝕜)‖ = 1 := by
  have hsq : ‖(U : Matrix n n 𝕜)‖ * ‖(U : Matrix n n 𝕜)‖ = 1 := by
    rw [← l2_opNorm_conjTranspose_mul_self]
    have : (U : Matrix n n 𝕜)ᴴ * (U : Matrix n n 𝕜) = 1 := by
      have := Unitary.coe_star_mul_self U
      rwa [Matrix.star_eq_conjTranspose] at this
    rw [this]; exact norm_one
  nlinarith [norm_nonneg (U : Matrix n n 𝕜), hsq]

/-- Conjugating by a unitary can only shrink (or preserve) the l2 operator norm: `‖U C Uᴴ‖ ≤ ‖C‖`. -/
lemma norm_conj_le [Nonempty n] (U : Matrix.unitaryGroup n 𝕜) (C : Matrix n n 𝕜) :
    ‖(U : Matrix n n 𝕜) * C * (U : Matrix n n 𝕜)ᴴ‖ ≤ ‖C‖ := by
  have hU : ‖(U : Matrix n n 𝕜)‖ = 1 := norm_unitary U
  have hUH : ‖(U : Matrix n n 𝕜)ᴴ‖ = 1 := by rw [l2_opNorm_conjTranspose]; exact hU
  calc ‖(U : Matrix n n 𝕜) * C * (U : Matrix n n 𝕜)ᴴ‖
      ≤ ‖(U : Matrix n n 𝕜) * C‖ * ‖(U : Matrix n n 𝕜)ᴴ‖ := l2_opNorm_mul _ _
    _ ≤ (‖(U : Matrix n n 𝕜)‖ * ‖C‖) * ‖(U : Matrix n n 𝕜)ᴴ‖ := by
        gcongr; exact l2_opNorm_mul _ _
    _ = ‖C‖ := by rw [hU, hUH]; ring

/-- Conjugation by a unitary preserves the l2 operator norm: `‖U B Uᴴ‖ = ‖B‖`. -/
lemma norm_conj_unitary [Nonempty n] (U : Matrix.unitaryGroup n 𝕜) (B : Matrix n n 𝕜) :
    ‖(U : Matrix n n 𝕜) * B * (U : Matrix n n 𝕜)ᴴ‖ = ‖B‖ := by
  refine le_antisymm (norm_conj_le U B) ?_
  -- lower bound: conjugate back by `star U`, which is `Uᴴ`
  have key := norm_conj_le (star U) ((U : Matrix n n 𝕜) * B * (U : Matrix n n 𝕜)ᴴ)
  have hs1 : ((star U : Matrix.unitaryGroup n 𝕜) : Matrix n n 𝕜) = (U : Matrix n n 𝕜)ᴴ := by
    rw [Unitary.coe_star, Matrix.star_eq_conjTranspose]
  rw [hs1] at key
  simp only [Matrix.conjTranspose_conjTranspose] at key
  have hBeq : (U : Matrix n n 𝕜)ᴴ * ((U : Matrix n n 𝕜) * B * (U : Matrix n n 𝕜)ᴴ)
      * (U : Matrix n n 𝕜) = B := by
    have h1 : (U : Matrix n n 𝕜)ᴴ * (U : Matrix n n 𝕜) = 1 := by
      have := Unitary.coe_star_mul_self U
      rwa [Matrix.star_eq_conjTranspose] at this
    calc (U : Matrix n n 𝕜)ᴴ * ((U : Matrix n n 𝕜) * B * (U : Matrix n n 𝕜)ᴴ) * (U : Matrix n n 𝕜)
        = ((U : Matrix n n 𝕜)ᴴ * (U : Matrix n n 𝕜)) * B
            * ((U : Matrix n n 𝕜)ᴴ * (U : Matrix n n 𝕜)) := by simp only [Matrix.mul_assoc]
      _ = B := by rw [h1, Matrix.one_mul, Matrix.mul_one]
  rw [hBeq] at key
  exact key

/-- **Spectral-norm ↔ eigenvalue bridge (Hermitian case).** For a Hermitian matrix `H`, if every
eigenvalue is bounded in absolute value by `c`, then the l2 operator norm `‖H‖ ≤ c`. This is the
matrix side of the spectral identity `‖H‖₂ = maxᵢ |λᵢ|` in the ≤-form needed to bound the norm:
it routes through the concrete spectral theorem `H = U·diag(λ)·Uᴴ`, unitary-conjugation invariance,
and `l2_opNorm_diagonal` — sidestepping the (absent) `Matrix` CStarAlgebra / CFC machinery. -/
theorem opNorm_le_of_eigenvalue_bound [Nonempty n] {H : Matrix n n 𝕜} (hH : H.IsHermitian)
    {c : ℝ} (hc : 0 ≤ c) (hbound : ∀ i, |hH.eigenvalues i| ≤ c) : ‖H‖ ≤ c := by
  have hspec := hH.spectral_theorem
  rw [Unitary.conjStarAlgAut_apply] at hspec
  rw [Matrix.star_eq_conjTranspose] at hspec
  rw [hspec, norm_conj_unitary, l2_opNorm_diagonal]
  rw [pi_norm_le_iff_of_nonneg hc]
  intro i
  simp only [Function.comp_apply, RCLike.norm_ofReal]
  exact hbound i

/-- **Spectral-norm ↔ eigenvalue characterization (Hermitian case), sharp form.** For a Hermitian
matrix `H` and any real `c`, the l2 operator norm satisfies `‖H‖ ≤ c` **iff** every eigenvalue is
bounded in absolute value by `c`. This is the two-sided (soundness + completeness) strengthening of
`opNorm_le_of_eigenvalue_bound`, which only supplied the `⟸` direction: it additionally proves the
bound is *necessary* (`‖H‖ ≤ c → ∀ i, |λᵢ| ≤ c`), i.e. that `‖H‖ = maxᵢ |λᵢ|` is exact and the norm
bound is tight. The forward direction routes through the concrete spectral theorem
`H = U·diag(λ)·Uᴴ`, unitary-conjugation invariance (`norm_conj_unitary`) and `l2_opNorm_diagonal` to
identify `‖H‖` with the sup-norm of the eigenvalue vector, then reads off each component via
`norm_le_pi_norm`. No `0 ≤ c` hypothesis is needed: with `n` nonempty it is forced by
`0 ≤ |λᵢ| ≤ c`. -/
theorem opNorm_le_iff_eigenvalue_bound [Nonempty n] {H : Matrix n n 𝕜} (hH : H.IsHermitian)
    {c : ℝ} : ‖H‖ ≤ c ↔ ∀ i, |hH.eigenvalues i| ≤ c := by
  -- `‖H‖` equals the sup-norm of the (ℝ↪𝕜-embedded) eigenvalue vector.
  have heq : ‖H‖ = ‖(RCLike.ofReal ∘ hH.eigenvalues : n → 𝕜)‖ := by
    have hspec := hH.spectral_theorem
    rw [Unitary.conjStarAlgAut_apply] at hspec
    rw [Matrix.star_eq_conjTranspose] at hspec
    conv_lhs => rw [hspec]
    rw [norm_conj_unitary, l2_opNorm_diagonal]
  constructor
  · intro h i
    have hcomp : ‖(RCLike.ofReal ∘ hH.eigenvalues : n → 𝕜) i‖
        ≤ ‖(RCLike.ofReal ∘ hH.eigenvalues : n → 𝕜)‖ := norm_le_pi_norm _ i
    simp only [Function.comp_apply, RCLike.norm_ofReal] at hcomp
    exact hcomp.trans (heq ▸ h)
  · intro h
    have hc : 0 ≤ c := (abs_nonneg _).trans (h (Classical.arbitrary n))
    exact opNorm_le_of_eigenvalue_bound hH hc h

/-- **SVD spectral bridge for the operator norm.** For an arbitrary (possibly rectangular) matrix
`A`, the l2 operator norm is bounded by `c` as soon as every eigenvalue of the Gram matrix `Aᴴ A`
is `≤ c²`. The eigenvalues of `Aᴴ A` are exactly the *squared singular values* of `A`, so this is
the statement `‖A‖₂ = maxᵢ σᵢ ≤ c` — the singular-value characterization of the spectral norm, in
the ≤-form that propagates a norm bound. Routes through `‖Aᴴ A‖ = ‖A‖²` and the Hermitian bridge. -/
theorem opNorm_le_of_gram_eigenvalue_bound {m : Type*} [Fintype m] [Nonempty n]
    {A : Matrix m n 𝕜} {c : ℝ} (hc : 0 ≤ c)
    (hbound : ∀ i, (isHermitian_conjTranspose_mul_self A).eigenvalues i ≤ c ^ 2) : ‖A‖ ≤ c := by
  have hH := isHermitian_conjTranspose_mul_self A
  have hgram : ‖Aᴴ * A‖ ≤ c ^ 2 := by
    refine opNorm_le_of_eigenvalue_bound hH (sq_nonneg c) fun i => ?_
    rw [abs_of_nonneg (eigenvalues_conjTranspose_mul_self_nonneg A i)]
    exact hbound i
  have hsq : ‖A‖ * ‖A‖ ≤ c ^ 2 := by rw [← l2_opNorm_conjTranspose_mul_self]; exact hgram
  nlinarith [norm_nonneg A, hsq, hc]

end Puffer.RL.SpectralBridge
