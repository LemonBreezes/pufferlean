/-
The Gram-eigenvalue mapping — the spectral-mapping step joining the SVD spectral bridge
(`Puffer/RL/SpectralBridge.lean`) to the scalar bound (`Puffer/RL/MuonScalarBound.lean`).

For the odd Muon map `p(X) = a·X + b·X³ + c·X⁵ = X·q(XᵀX)` with `q(u) = a + b·u + c·u²`, the Gram
matrix of one step is `p(X)ᴴ p(X) = q(M)·M·q(M) = M·q(M)²` where `M = XᵀX` is Hermitian PSD. The
analytic content is the SPECTRAL MAPPING

    λᵢ(M·q(M)²) = μᵢ·q(μᵢ)²          (μᵢ = eigenvalues of M),

which reduces the operator norm `‖M·q(M)²‖` to the scalar quantities `μᵢ·q(μᵢ)²` that
`muon_scalar_bound` controls on `[0,1]`.

`opNorm_gram_poly_le` proves the mapping in ≤-form for an arbitrary real symmetric coefficient triple:
it diagonalizes `M = U·diag(μ)·Uᴴ` (`spectral_theorem`), pushes the matrix polynomial through the
`*`-algebra automorphism `conjStarAlgAut U` (a genuine spectral mapping — `φ(M·q(M)²) = φ(M)·q(φ(M))²`
since `φ` is a ring/star hom), reads off the diagonal `diag(μ·q(μ)²)`, and strips the unitary
conjugation with `norm_conj_unitary` + `l2_opNorm_diagonal`. `opNorm_muon_gram_le` then instantiates
it at the actual tuned schedule: if every eigenvalue of `M` lies in `[0,1]`, one Muon step's Gram
matrix has operator norm `≤ 1.63`, i.e. `‖p(X)‖₂ ≤ √1.63 < 1.2767`.

Both axiom-clean (`propext`/`Classical.choice`/`Quot.sound`, no `sorry`). Over ℝ (Muon iterates are
real matrices).

REMAINING for the full O(1) Newton–Schulz tower: only piece (c) — the identity `p(X)ᴴp(X) = M·q(M)²`
relating the rectangular runnable iterate `X` (float `Mat`) to the abstract `M = XᵀX`, i.e. the
`Matrix`↔float-`Mat` bridge. The spectral reduction (bridge), the scalar bound, and the eigenvalue
mapping are now all proven on the Mathlib-`Matrix` side.
-/
import Mathlib
import Puffer.RL.SpectralBridge
import Puffer.RL.MuonScalarBound

namespace Puffer.RL.MuonGramBound

open scoped Matrix Matrix.Norms.L2Operator
open Matrix
open Puffer.RL.SpectralBridge
open Puffer.RL.MuonScalarBound (muon_scalar_bound)
open Puffer.Optim.Muon (muonCoeffs)

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- Diagonal computation: `D · (a•1 + b•D + c•D²)² = diagonal (i ↦ w i · (a + b·w i + c·w i²)²)`. -/
theorem diag_gram (w : n → ℝ) (a b c : ℝ) :
    diagonal w * (a • (1 : Matrix n n ℝ) + b • diagonal w + c • (diagonal w) ^ 2) ^ 2
      = diagonal (fun i => w i * (a + b * w i + c * w i ^ 2) ^ 2) := by
  rw [show (1 : Matrix n n ℝ) = diagonal (fun _ => (1 : ℝ)) from diagonal_one.symm]
  simp only [← diagonal_smul, diagonal_add, diagonal_pow, diagonal_mul_diagonal]
  congr 1
  funext i
  simp only [Pi.smul_apply, Pi.pow_apply, smul_eq_mul]
  ring

/-- **Gram-eigenvalue mapping (operator-norm form).** For a Hermitian `M` with eigenvalues `μ`, the
matrix `M·q(M)²` (`q(u) = a + b·u + c·u²`) has l2 operator norm `≤ C` as soon as `|μᵢ·q(μᵢ)²| ≤ C`
for all `i`. Its eigenvalues ARE exactly the `μᵢ·q(μᵢ)²` — established by diagonalizing
`M = U·diag(μ)·Uᴴ`, pushing the polynomial through the `*`-algebra automorphism `conjStarAlgAut U`,
and reading the diagonal. This is the spectral-mapping step `λᵢ(M·q(M)²) = μᵢ·q(μᵢ)²`, ≤-form. -/
theorem opNorm_gram_poly_le [Nonempty n] {M : Matrix n n ℝ} (hM : M.IsHermitian)
    (a b c C : ℝ) (hC : 0 ≤ C)
    (hbnd : ∀ i, |hM.eigenvalues i * (a + b * hM.eigenvalues i + c * hM.eigenvalues i ^ 2) ^ 2| ≤ C) :
    ‖M * (a • (1 : Matrix n n ℝ) + b • M + c • M ^ 2) ^ 2‖ ≤ C := by
  have hspec := hM.spectral_theorem
  set U := hM.eigenvectorUnitary with hU
  set w : n → ℝ := RCLike.ofReal ∘ hM.eigenvalues with hw
  -- push the polynomial through the *-algebra automorphism φ = conjugation by U
  have key : M * (a • (1 : Matrix n n ℝ) + b • M + c • M ^ 2) ^ 2
      = (Unitary.conjStarAlgAut ℝ (Matrix n n ℝ) U)
          (diagonal w * (a • (1 : Matrix n n ℝ) + b • diagonal w + c • (diagonal w) ^ 2) ^ 2) := by
    conv_lhs => rw [hspec]
    simp only [map_mul, map_pow, map_add, map_smul, map_one]
  rw [key, diag_gram]
  -- unfold `conjStarAlgAut` to `U · diag · Uᴴ` and strip the unitary conjugation
  rw [Unitary.conjStarAlgAut_apply, Matrix.star_eq_conjTranspose, norm_conj_unitary,
    l2_opNorm_diagonal, pi_norm_le_iff_of_nonneg hC]
  intro i
  simp only [hw, Function.comp_apply, RCLike.ofReal_real_eq_id, id_eq, Real.norm_eq_abs]
  exact hbnd i

/-- **Gram-eigenvalue mapping (operator-norm EQUALITY form).** For a Hermitian `M` with eigenvalues
`μ`, the l2 operator norm of the Muon Gram matrix `M·q(M)²` (`q(u) = a + b·u + c·u²`) *equals* the
sup-norm of the vector `i ↦ μᵢ·q(μᵢ)²`. This is the FULL spectral-mapping identity
`λᵢ(M·q(M)²) = μᵢ·q(μᵢ)²` in operator-norm form — strictly stronger than the ≤-form
`opNorm_gram_poly_le` (which it re-derives via `pi_norm_le_iff`/`norm_le_pi_norm`), and it also yields
the matching per-eigenvalue LOWER bound `|μⱼ·q(μⱼ)²| ≤ ‖M·q(M)²‖` for every index `j`. Proved by
diagonalizing `M = U·diag(μ)·Uᴴ`, pushing the polynomial through the `*`-algebra automorphism
`conjStarAlgAut U` (`map_mul`/`map_pow`/`map_add`/`map_smul`/`map_one`), reading off `diag(μ·q(μ)²)`
via `diag_gram`, then stripping the unitary conjugation (`norm_conj_unitary` + `l2_opNorm_diagonal`).
`[Nonempty n]` and `IsHermitian` are both load-bearing (the latter is what makes `‖·‖` equal the
eigenvalue sup — false for a general non-normal matrix). -/
theorem opNorm_gram_poly_eq [Nonempty n] {M : Matrix n n ℝ} (hM : M.IsHermitian) (a b c : ℝ) :
    ‖M * (a • (1 : Matrix n n ℝ) + b • M + c • M ^ 2) ^ 2‖
      = ‖(fun i => hM.eigenvalues i
            * (a + b * hM.eigenvalues i + c * hM.eigenvalues i ^ 2) ^ 2)‖ := by
  have hspec := hM.spectral_theorem
  set U := hM.eigenvectorUnitary with hU
  set w : n → ℝ := RCLike.ofReal ∘ hM.eigenvalues with hw
  have key : M * (a • (1 : Matrix n n ℝ) + b • M + c • M ^ 2) ^ 2
      = (Unitary.conjStarAlgAut ℝ (Matrix n n ℝ) U)
          (diagonal w * (a • (1 : Matrix n n ℝ) + b • diagonal w + c • (diagonal w) ^ 2) ^ 2) := by
    conv_lhs => rw [hspec]
    simp only [map_mul, map_pow, map_add, map_smul, map_one]
  rw [key, diag_gram]
  rw [Unitary.conjStarAlgAut_apply, Matrix.star_eq_conjTranspose, norm_conj_unitary,
    l2_opNorm_diagonal]
  have hfun : (fun i => w i * (a + b * w i + c * w i ^ 2) ^ 2)
      = fun i => hM.eigenvalues i
          * (a + b * hM.eigenvalues i + c * hM.eigenvalues i ^ 2) ^ 2 := by
    funext i
    simp only [hw, Function.comp_apply, RCLike.ofReal_real_eq_id, id_eq]
  rw [hfun]

/-- **One Muon Newton–Schulz step's Gram matrix is O(1).** For the actual tuned schedule
`muonCoeffs`: if every eigenvalue of the Hermitian `M` lies in `[0,1]`, then the step's Gram matrix
`M·q(M)²` has l2 operator norm `≤ 1.63`. Equivalently `‖p(X)‖₂ ≤ √1.63 < 1.2767`, since
`p(X)ᴴp(X) = M·q(M)²` with `M = XᵀX`. This fuses the eigenvalue mapping (`opNorm_gram_poly_le`) with
the scalar bound (`muon_scalar_bound`): the eigenvalues map to `μᵢ·q(μᵢ)² ≤ 1.63`. The bound is
DIMENSION-FREE and does NOT compound with the matrix size — the whole point of the tight tower. -/
theorem opNorm_muon_gram_le [Nonempty n] {M : Matrix n n ℝ} (hM : M.IsHermitian)
    {a b c : ℝ} (hmem : (a, b, c) ∈ muonCoeffs)
    (hev : ∀ i, 0 ≤ hM.eigenvalues i ∧ hM.eigenvalues i ≤ 1) :
    ‖M * (a • (1 : Matrix n n ℝ) + b • M + c • M ^ 2) ^ 2‖ ≤ 1.63 := by
  refine opNorm_gram_poly_le hM a b c 1.63 (by norm_num) fun i => ?_
  obtain ⟨h0, h1⟩ := hev i
  rw [abs_of_nonneg (mul_nonneg h0 (sq_nonneg _))]
  exact muon_scalar_bound hmem _ h0 h1

/-- **The Gram identity.** `p(X)ᴴ p(X) = M · q(M)²` where `p(X) = X·q(M)`, `M = XᵀX`, and
`q(M) = a•1 + b•M + c•M²`. Purely algebraic: `p(X)ᴴp(X) = q(M)ᴴ (XᵀX) q(M) = q(M)·M·q(M) = M·q(M)²`
using that `q(M)` is Hermitian (real coefficients) and commutes with `M` (both are polynomials in
`M`). This is the identity relating the rectangular iterate `X` to the square Gram matrix `M` — the
last algebraic step of the tight Newton–Schulz tower. -/
theorem gram_identity {m : Type*} [Fintype m] (X : Matrix m n ℝ) (a b c : ℝ) :
    (X * (a • 1 + b • (Xᴴ * X) + c • (Xᴴ * X) ^ 2))ᴴ
        * (X * (a • 1 + b • (Xᴴ * X) + c • (Xᴴ * X) ^ 2))
      = (Xᴴ * X) * (a • 1 + b • (Xᴴ * X) + c • (Xᴴ * X) ^ 2) ^ 2 := by
  set M := Xᴴ * X with hMdef
  set qM := a • (1 : Matrix n n ℝ) + b • M + c • M ^ 2 with hqM
  have hMherm : Mᴴ = M := by rw [hMdef, conjTranspose_mul, conjTranspose_conjTranspose]
  have hqMherm : qMᴴ = qM := by
    rw [hqM]
    simp only [conjTranspose_add, conjTranspose_smul, conjTranspose_one, star_trivial,
      conjTranspose_pow, hMherm]
  have hcomm : qM * M = M * qM := by
    have h : Commute qM M := by
      rw [hqM]
      exact (((Commute.one_left M).smul_left a).add_left ((Commute.refl M).smul_left b)).add_left
        (((Commute.refl M).pow_left 2).smul_left c)
    exact h
  rw [conjTranspose_mul, hqMherm, Matrix.mul_assoc qM Xᴴ (X * qM), ← Matrix.mul_assoc Xᴴ X qM,
    ← hMdef, ← Matrix.mul_assoc qM M qM, hcomm, Matrix.mul_assoc M qM qM, ← pow_two]

/-- **One full Muon Newton–Schulz step is O(1) in operator norm.** For the actual tuned schedule and
any rectangular real iterate `X` whose Gram matrix `XᵀX` has all eigenvalues `≤ 1` (equivalently
`‖X‖₂ ≤ 1`, the intended precondition), the next iterate `p(X) = X·q(XᵀX)` satisfies
`‖p(X)‖₂ ≤ √1.63 < 1.2767`. This is the end-to-end per-step bound on the Mathlib-`Matrix` side:
`‖p(X)‖² = ‖p(X)ᴴp(X)‖ = ‖M·q(M)²‖ ≤ 1.63`, chaining the Gram identity, the eigenvalue mapping, and
the scalar bound. DIMENSION-FREE — the whole point of the tight tower. -/
theorem opNorm_muon_step_le [Nonempty n] {m : Type*} [Fintype m] (X : Matrix m n ℝ) {a b c : ℝ}
    (hmem : (a, b, c) ∈ muonCoeffs)
    (hev : ∀ i, (isHermitian_conjTranspose_mul_self X).eigenvalues i ≤ 1) :
    ‖X * (a • (1 : Matrix n n ℝ) + b • (Xᴴ * X) + c • (Xᴴ * X) ^ 2)‖ ≤ Real.sqrt 1.63 := by
  have hM := isHermitian_conjTranspose_mul_self X
  set p := X * (a • (1 : Matrix n n ℝ) + b • (Xᴴ * X) + c • (Xᴴ * X) ^ 2) with hp
  have hgram : ‖pᴴ * p‖ ≤ 1.63 := by
    rw [hp, gram_identity]
    exact opNorm_muon_gram_le hM hmem
      (fun i => ⟨eigenvalues_conjTranspose_mul_self_nonneg X i, hev i⟩)
  have hsq : ‖p‖ ^ 2 ≤ 1.63 := by
    rw [pow_two, ← l2_opNorm_conjTranspose_mul_self]; exact hgram
  have := Real.sqrt_le_sqrt hsq
  rwa [Real.sqrt_sq (norm_nonneg p)] at this

end Puffer.RL.MuonGramBound
