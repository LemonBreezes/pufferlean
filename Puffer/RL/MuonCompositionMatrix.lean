/-
The matrix-level 5-iteration Newton–Schulz composition bound: `‖(5-fold Muon composition of X₀)‖₂ ≤ √1.3111`
for `‖X₀‖₂ ≤ 1`. Lifts the scalar composition boundedness (`MuonComposition`) to the abstract `Matrix`
operator norm, folding the per-step spectral reduction through the five tuned steps.

Ingredients (all reusing the spectral bridge + Gram machinery):
  • `eigenvalue_le_opNorm`   : each eigenvalue of a Hermitian matrix is `≤ ‖H‖` (spectral theorem +
      `l2_opNorm_diagonal` + `norm_le_pi_norm` — the reverse of `opNorm_le_of_eigenvalue_bound`).
  • `gram_eig_le_opNorm_sq`  : each eigenvalue of the Gram `Xᴴ X` is `≤ ‖X‖²`.
  • `muon_step_normsq_le`     : `‖X·q(XᵀX)‖² ≤ C` from the eigenvalue bound `∀i |μᵢ·q(μᵢ)²| ≤ C`
      (`gram_identity` + `opNorm_gram_poly_le`); the general-`C` form of `opNorm_muon_step_le`.
  • `muon_step_chain`        : `‖X‖² ≤ Bin` + a scalar bound on `[0,Bin]` ⟹ `‖X·q(XᵀX)‖² ≤ Bout`
      (eigenvalues of `XᵀX` are in `[0, ‖X‖²] ⊆ [0,Bin]`, so the scalar bound applies).
  • `newtonSchulz_comp_normsq`: the payoff — chaining the five `muon_comp_step1..5` scalar bounds through
      `muon_step_chain` with the schedule `[1,1.63,1.63,1.57,1.33,1.3111]`.

All axiom-clean (`propext`/`Classical.choice`/`Quot.sound`, no `sorry`). Over ℝ (the abstract exact-ℝ map;
the runnable-step transport to the Float iterate is `NewtonSchulzFloat`/`NewtonSchulzSeedClosed`).

The 5-step composition operator norm stays `≤ √1.63 ≈ 1.277` throughout and ends `≤ √1.3111 ≈ 1.145` —
DIMENSION-FREE O(1). This is the full-composition analogue of the per-step `‖p(X)‖₂ ≤ √1.64` bound.
-/
import Mathlib
import Puffer.RL.SpectralBridge
import Puffer.RL.MuonGramBound
import Puffer.RL.MuonComposition

namespace Puffer.RL.MuonCompositionMatrix

open scoped Matrix Matrix.Norms.L2Operator
open Matrix
open Puffer.RL.SpectralBridge
open Puffer.RL.MuonGramBound (opNorm_gram_poly_le gram_identity)
open Puffer.RL.MuonComposition

variable {n : Type*} [Fintype n] [DecidableEq n]

/-- Each eigenvalue of a Hermitian matrix is `≤` its l2 operator norm (reverse of
    `opNorm_le_of_eigenvalue_bound`). -/
theorem eigenvalue_le_opNorm {𝕜 : Type*} [RCLike 𝕜] [Nonempty n] {H : Matrix n n 𝕜}
    (hH : H.IsHermitian) (i : n) : hH.eigenvalues i ≤ ‖H‖ := by
  have hspec := hH.spectral_theorem
  rw [Unitary.conjStarAlgAut_apply, Matrix.star_eq_conjTranspose] at hspec
  calc hH.eigenvalues i ≤ |hH.eigenvalues i| := le_abs_self _
    _ = ‖(RCLike.ofReal (hH.eigenvalues i) : 𝕜)‖ := (RCLike.norm_ofReal _).symm
    _ = ‖(RCLike.ofReal ∘ hH.eigenvalues : n → 𝕜) i‖ := by rw [Function.comp_apply]
    _ ≤ ‖(RCLike.ofReal ∘ hH.eigenvalues : n → 𝕜)‖ := norm_le_pi_norm _ i
    _ = ‖(diagonal (RCLike.ofReal ∘ hH.eigenvalues) : Matrix n n 𝕜)‖ := (l2_opNorm_diagonal _).symm
    _ = ‖H‖ := by conv_rhs => rw [hspec, norm_conj_unitary]

/-- Each eigenvalue of the Gram `Xᴴ X` is `≤ ‖X‖²`. -/
theorem gram_eig_le_opNorm_sq [Nonempty n] {m : Type*} [Fintype m] (X : Matrix m n ℝ) (i : n) :
    (isHermitian_conjTranspose_mul_self X).eigenvalues i ≤ ‖X‖ ^ 2 := by
  have := eigenvalue_le_opNorm (isHermitian_conjTranspose_mul_self X) i
  rwa [l2_opNorm_conjTranspose_mul_self, ← sq] at this

/-- **Generalized per-step (squared).** `‖X·q(XᵀX)‖² ≤ C` from the eigenvalue bound `∀i |μᵢ·q(μᵢ)²| ≤ C`
    — the general-`C` form of `opNorm_muon_step_le`. -/
theorem muon_step_normsq_le [Nonempty n] {m : Type*} [Fintype m] (X : Matrix m n ℝ) (a b c C : ℝ)
    (hbnd : ∀ i, |(isHermitian_conjTranspose_mul_self X).eigenvalues i
      * (a + b * (isHermitian_conjTranspose_mul_self X).eigenvalues i
        + c * (isHermitian_conjTranspose_mul_self X).eigenvalues i ^ 2) ^ 2| ≤ C) :
    ‖X * (a • (1 : Matrix n n ℝ) + b • (Xᴴ * X) + c • (Xᴴ * X) ^ 2)‖ ^ 2 ≤ C := by
  have hM := isHermitian_conjTranspose_mul_self X
  have hC : 0 ≤ C := le_trans (abs_nonneg _) (hbnd (Classical.arbitrary n))
  rw [sq, ← l2_opNorm_conjTranspose_mul_self, gram_identity]
  exact opNorm_gram_poly_le hM a b c C hC hbnd

/-- **Chaining step.** `‖X‖² ≤ Bin` + a scalar bound on `[0,Bin]` ⟹ `‖X·q(XᵀX)‖² ≤ Bout` (eigenvalues of
    `XᵀX` lie in `[0, ‖X‖²] ⊆ [0,Bin]`, so the scalar bound applies pointwise). -/
theorem muon_step_chain [Nonempty n] {m : Type*} [Fintype m] (X : Matrix m n ℝ) (a b c Bin Bout : ℝ)
    (hBin : ‖X‖ ^ 2 ≤ Bin)
    (hscalar : ∀ t : ℝ, 0 ≤ t → t ≤ Bin → t * (a + b * t + c * t ^ 2) ^ 2 ≤ Bout) :
    ‖X * (a • (1 : Matrix n n ℝ) + b • (Xᴴ * X) + c • (Xᴴ * X) ^ 2)‖ ^ 2 ≤ Bout :=
  muon_step_normsq_le X a b c Bout (fun i => by
    have h0 := eigenvalues_conjTranspose_mul_self_nonneg X i
    rw [abs_of_nonneg (mul_nonneg h0 (sq_nonneg _))]
    exact hscalar _ h0 (le_trans (gram_eig_le_opNorm_sq X i) hBin))

/-- **Push-through identity** `q(A·Aᴴ)·A = A·q(Aᴴ·A)` for `q(u)=α+β·u+γ·u²` — the left- and right-Gram
    Muon step maps are the SAME matrix. Lets `nsIterR`'s left-Gram embedding be identified with `pstep`. -/
theorem muon_pushthrough {m : Type*} [Fintype m] [DecidableEq m] (A : Matrix m n ℝ) (α β γ : ℝ) :
    (α • (1 : Matrix m m ℝ) + β • (A * Aᴴ) + γ • (A * Aᴴ) ^ 2) * A
      = A * (α • (1 : Matrix n n ℝ) + β • (Aᴴ * A) + γ • (Aᴴ * A) ^ 2) := by
  have h1 : (A * Aᴴ) * A = A * (Aᴴ * A) := Matrix.mul_assoc A Aᴴ A
  have h2 : (A * Aᴴ) ^ 2 * A = A * (Aᴴ * A) ^ 2 := by
    rw [pow_two, pow_two]
    calc (A * Aᴴ) * (A * Aᴴ) * A = (A * Aᴴ) * ((A * Aᴴ) * A) := by rw [Matrix.mul_assoc]
      _ = (A * Aᴴ) * (A * (Aᴴ * A)) := by rw [h1]
      _ = ((A * Aᴴ) * A) * (Aᴴ * A) := by rw [← Matrix.mul_assoc]
      _ = (A * (Aᴴ * A)) * (Aᴴ * A) := by rw [h1]
      _ = A * ((Aᴴ * A) * (Aᴴ * A)) := by rw [Matrix.mul_assoc]
  simp only [Matrix.add_mul, Matrix.mul_add, Matrix.smul_mul, Matrix.mul_smul,
    Matrix.one_mul, Matrix.mul_one, h1, h2]

/-- One Muon step map `p(X) = X·q(XᵀX)`, `q(u) = a + b·u + c·u²`. -/
def pstep (a b c : ℝ) {m : Type*} [Fintype m] (X : Matrix m n ℝ) : Matrix m n ℝ :=
  X * (a • (1 : Matrix n n ℝ) + b • (Xᴴ * X) + c • (Xᴴ * X) ^ 2)

/-- **Left-isometry equivariance of the Muon step.** If `U` is a linear isometry on the output/row space
    (`Uᴴ · U = 1`), then one Muon Newton–Schulz step commutes with left-multiplication by `U`:
    `pstep a b c (U · X) = U · pstep a b c X`. The orthogonality hypothesis is load-bearing — it collapses the
    Gram matrix `(U·X)ᴴ·(U·X) = Xᴴ·(Uᴴ·U)·X` back to `Xᴴ·X`, so the entire polynomial factor
    `a•1 + b•(Xᴴ·X) + c•(Xᴴ·X)²` is unchanged and `U` slides out on the left. Dropping `hU` breaks it (e.g.
    `U = 2•1` gives `(U·X)ᴴ·(U·X) = 4·(Xᴴ·X)`, changing the factor). This is exactly the property that makes the
    Newton–Schulz orthogonalization iteration equivariant under the left action of the isometry group (a change
    of orthonormal basis on the output side): since one step commutes with `U`, so does the whole 5-fold
    composition — the structural reason NS is a basis-independent orthogonalization on the output frame. The
    hypothesis class is inhabited (the identity, and every genuine rotation/reflection, is an isometry). -/
theorem pstep_orthogonal_left {m : Type*} [Fintype m] [DecidableEq m]
    (U : Matrix m m ℝ) (hU : Uᴴ * U = 1) (a b c : ℝ) (X : Matrix m n ℝ) :
    pstep a b c (U * X) = U * pstep a b c X := by
  have hgram : (U * X)ᴴ * (U * X) = Xᴴ * X := by
    rw [Matrix.conjTranspose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc Uᴴ U X, hU,
      Matrix.one_mul]
  unfold pstep
  rw [hgram, Matrix.mul_assoc]

/-- **The 5-iteration Muon Newton–Schulz composition is O(1) at the matrix level.** For an input with
    `‖X₀‖₂ ≤ 1`, the 5-fold composition of the tuned Muon steps has `‖·‖² ≤ 1.3111`, i.e.
    `‖·‖₂ ≤ √1.3111 < 1.15`. Dimension-free: the operator norm stays `≤ √1.63 ≈ 1.277` throughout the five
    iterations and ends `≤ √1.3111 ≈ 1.145`. The full-composition analogue of the per-step `‖p(X)‖₂ ≤ √1.64`. -/
theorem newtonSchulz_comp_normsq [Nonempty n] {m : Type*} [Fintype m] (X0 : Matrix m n ℝ)
    (hX0 : ‖X0‖ ^ 2 ≤ 1) :
    ‖pstep 2.8366 (-3.0525) 1.2012 (pstep 2.8769 (-3.1427) 1.2046
        (pstep 3.7418 (-5.5913) 2.3037 (pstep 3.9505 (-6.3029) 2.6377
          (pstep 4.0848 (-6.8946) 2.9270 X0))))‖ ^ 2 ≤ 1.3111 := by
  have c1 : ‖pstep 4.0848 (-6.8946) 2.9270 X0‖ ^ 2 ≤ 1.63 :=
    muon_step_chain X0 _ _ _ 1 1.63 hX0 (fun t h0 h1 => by nlinarith [muon_comp_step1 t h0 h1])
  have c2 : ‖pstep 3.9505 (-6.3029) 2.6377 (pstep 4.0848 (-6.8946) 2.9270 X0)‖ ^ 2 ≤ 1.63 :=
    muon_step_chain _ _ _ _ 1.63 1.63 c1 (fun t h0 h1 => by nlinarith [muon_comp_step2 t h0 h1])
  have c3 : ‖pstep 3.7418 (-5.5913) 2.3037 (pstep 3.9505 (-6.3029) 2.6377
      (pstep 4.0848 (-6.8946) 2.9270 X0))‖ ^ 2 ≤ 1.57 :=
    muon_step_chain _ _ _ _ 1.63 1.57 c2 (fun t h0 h1 => by nlinarith [muon_comp_step3 t h0 h1])
  have c4 : ‖pstep 2.8769 (-3.1427) 1.2046 (pstep 3.7418 (-5.5913) 2.3037
      (pstep 3.9505 (-6.3029) 2.6377 (pstep 4.0848 (-6.8946) 2.9270 X0)))‖ ^ 2 ≤ 1.33 :=
    muon_step_chain _ _ _ _ 1.57 1.33 c3 (fun t h0 h1 => by nlinarith [muon_comp_step4 t h0 h1])
  exact muon_step_chain _ _ _ _ 1.33 1.3111 c4 (fun t h0 h1 => by nlinarith [muon_comp_step5 t h0 h1])

end Puffer.RL.MuonCompositionMatrix
