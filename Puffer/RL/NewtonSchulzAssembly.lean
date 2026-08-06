/-
Closing the runnable trainer: assembling one exact-ℝ Newton–Schulz step (`nsIterR`) through the
`Matrix` embedding into the abstract per-step operator-norm bound.

`nsIterR` (the ℝ mirror of the runnable `nsIter`, `NewtonSchulzError.lean`) in the `r ≤ c` branch is
`lincomb3R a X b (A·X) c (A·A·X)` with `A = X·Xᵀ` — i.e. the left-Gram polynomial `q(N)·X`,
`N = X·Xᵀ`, `q(u) = a + b·u + c·u²`. This file threads that computation through the three exact
op-homomorphisms of `MatrixEmbed` (`matmulR → *`, `transposeR → ᵀ`, `lincomb3R → •/+`):

  • `nsIterR_toMatrixR` : `toMatrixR (nsIterR X (a,b,c)) = q(N)·MX`  (`MX = toMatrixR X`, `N = MX·MXᴴ`),
    the EXACT embedding of one Newton–Schulz step as a Mathlib matrix polynomial;
  • `nsIterR_opNorm_le` : `‖toMatrixR (nsIterR X (a,b,c))‖₂ ≤ √C` given `C` bounds the scalar
    quantities `μᵢ·q(μᵢ)²` on the Gram eigenvalues `μ` of `N` — the abstract per-step bound
    (`MuonGramBound.opNorm_gram_poly_le`) transported to the runnable pipeline's exact-ℝ mirror.
    It routes the LEFT-Gram form through `‖A‖² = ‖A·Aᴴ‖` (the mirror of `gram_identity`).

Both axiom-clean beyond the trusted Float `toReal` axiom.

With `nsIterR_opNorm_le`, the Newton–Schulz per-step tight bound `‖p(X)‖₂ ≤ √1.63 < 1.2767` reaches the
runnable trainer's own iterate (via its exact-ℝ mirror). Two honest residual gaps remain, neither of
which is new mathematics on the matrix side:
  • COEFFICIENT ROUNDING. `nsIterR` uses `toReal` of the *Float* coefficients; `muon_scalar_bound` is
    proven for the exact ℝ literals. Discharging `C = 1.63` for the `toReal`-coefficients needs
    `|toReal (4.0848 : Float) − (4.0848 : ℝ)| ≤ tiny` etc., which the opaque `toReal` axiom on Float
    literals does not pin down (the 0.5 % margin of `muon_scalar_bound` would absorb it given a Float
    value model). Hence `nsIterR_opNorm_le` takes the scalar/eigenvalue bound as a hypothesis.
  • FLOAT ROUNDING & the `[0,1]` precondition: the Float→mirror gap is the `MatBnd` tower
    (`MatrixEmbed.toMatrixF_sub_toMatrixR_entry`); the eigenvalue-in-`[0,1]` precondition is the
    Frobenius-normalization step that seeds `newtonSchulz`.
-/
import Mathlib
import Puffer.RL.NewtonSchulzError
import Puffer.RL.NewtonSchulzTight
import Puffer.RL.MatrixEmbed
import Puffer.RL.MuonGramBound

namespace Puffer.RL.NewtonSchulzAssembly

open scoped Matrix Matrix.Norms.L2Operator
open Matrix
open Puffer.RL.NewtonSchulzError
open Puffer.RL.MatrixEmbed
open Puffer.RL.MuonGramBound (opNorm_gram_poly_le)
open Puffer.FloatR (toReal)

/-- One exact-ℝ Newton–Schulz step (`r ≤ c` branch) embeds as the left-Gram matrix polynomial
    `q(N)·MX` with `N = MX·MXᵀ`, `MX = toMatrixR X`, `q(u) = a + b·u + c·u²` (coefficients `toReal`
    of the Float schedule). Threads the three `MatrixEmbed` op-homomorphisms through `nsIterR`. -/
theorem nsIterR_toMatrixR (X : MatR) (a b c : Float) (r cc : Nat)
    (hXsz : X.size = r) (hXrow : ∀ i, i < r → (X[i]!).size = cc)
    (hr : 0 < r) (hrc : r ≤ cc) :
    toMatrixR r cc (nsIterR X (a, b, c))
      = ((toReal a) • (1 : Matrix (Fin r) (Fin r) ℝ)
          + (toReal b) • (toMatrixR r cc X * (toMatrixR r cc X)ᵀ)
          + (toReal c) • (toMatrixR r cc X * (toMatrixR r cc X)ᵀ) ^ 2)
        * toMatrixR r cc X := by
  have hcc : 0 < cc := lt_of_lt_of_le hr hrc
  have hXne : ¬ (X.size = 0) := by rw [hXsz]; omega
  have hcond : X.size ≤ (X[0]!).size := by rw [hXsz, hXrow 0 hr]; exact hrc
  have hTXsz : (transposeR X).size = cc := by rw [transposeR_size, if_neg hXne, hXrow 0 hr]
  have hTXrow : ∀ i, i < cc → ((transposeR X)[i]!).size = r := by
    intro i hi
    rw [transposeR_rowSize X i (by rw [if_neg hXne, hXrow 0 hr]; exact hi), hXsz]
  have hAsz : (matmulR X (transposeR X)).size = r := by rw [matmulR_size, hXsz]
  have hArow : ∀ i, i < r → ((matmulR X (transposeR X))[i]!).size = r := by
    intro i hi
    rw [matmulR_rowSize X (transposeR X) i (by rw [hXsz]; exact hi), if_neg (by rw [hTXsz]; omega),
      hTXrow 0 hcc]
  have hAXsz : (matmulR (matmulR X (transposeR X)) X).size = r := by rw [matmulR_size, hAsz]
  have hAXrow : ∀ i, i < r → ((matmulR (matmulR X (transposeR X)) X)[i]!).size = cc := by
    intro i hi
    rw [matmulR_rowSize (matmulR X (transposeR X)) X i (by rw [hAsz]; exact hi),
      if_neg hXne, hXrow 0 hr]
  have hN : toMatrixR r r (matmulR X (transposeR X))
      = toMatrixR r cc X * (toMatrixR r cc X)ᵀ := by
    rw [matmulR_toMatrixR X (transposeR X) r cc r hXsz hXrow hTXsz hTXrow hr hcc,
      transposeR_toMatrixR X r cc hXsz hXrow hr]
  have hAX : toMatrixR r cc (matmulR (matmulR X (transposeR X)) X)
      = (toMatrixR r cc X * (toMatrixR r cc X)ᵀ) * toMatrixR r cc X := by
    rw [matmulR_toMatrixR (matmulR X (transposeR X)) X r r cc hAsz hArow hXsz hXrow hr hr, hN]
  have hAAX : toMatrixR r cc (matmulR (matmulR X (transposeR X))
        (matmulR (matmulR X (transposeR X)) X))
      = (toMatrixR r cc X * (toMatrixR r cc X)ᵀ)
        * ((toMatrixR r cc X * (toMatrixR r cc X)ᵀ) * toMatrixR r cc X) := by
    rw [matmulR_toMatrixR (matmulR X (transposeR X)) (matmulR (matmulR X (transposeR X)) X)
      r r cc hAsz hArow hAXsz hAXrow hr hr, hN, hAX]
  rw [nsIterR]
  simp only [hcond, if_true]
  rw [lincomb3R_toMatrixR (toReal a) (toReal b) (toReal c) X
      (matmulR (matmulR X (transposeR X)) X)
      (matmulR (matmulR X (transposeR X)) (matmulR (matmulR X (transposeR X)) X)) r cc hXsz hXrow,
    hAX, hAAX]
  set MX := toMatrixR r cc X
  set N := MX * MXᵀ
  rw [← Matrix.mul_assoc N N MX, ← pow_two, Matrix.add_mul, Matrix.add_mul, Matrix.smul_mul,
    Matrix.smul_mul, Matrix.smul_mul, Matrix.one_mul]

/-- **Operator-norm bound for one runnable (mirror) Newton–Schulz step.** In the `r ≤ c` branch, the
    embedded iterate `toMatrixR (nsIterR X (a,b,c))` has l2 operator norm `≤ √C`, where `C` bounds
    `|μᵢ·q(μᵢ)²|` on the eigenvalues `μ` of the Gram matrix `N = MX·MXᴴ`. Via `nsIterR_toMatrixR`
    (embedding = `q(N)·MX`) and the LEFT-Gram identity `(q(N)·MX)·(q(N)·MX)ᴴ = N·q(N)²` fed to
    `opNorm_gram_poly_le`, then `‖A‖² = ‖A·Aᴴ‖` and `Real.sqrt` monotonicity. With the tuned schedule
    (eigenvalues in `[0,1]`, `C = 1.63`) this is the `√1.63 < 1.2767` per-step bound on the runnable
    pipeline's exact-ℝ mirror. -/
theorem nsIterR_opNorm_le (X : MatR) (a b c : Float) (r cc : Nat) (C : ℝ) (hC : 0 ≤ C)
    (hXsz : X.size = r) (hXrow : ∀ i, i < r → (X[i]!).size = cc)
    (hr : 0 < r) (hrc : r ≤ cc)
    (hbnd : ∀ i,
      |(isHermitian_mul_conjTranspose_self (toMatrixR r cc X)).eigenvalues i
        * (toReal a + toReal b * (isHermitian_mul_conjTranspose_self (toMatrixR r cc X)).eigenvalues i
          + toReal c * (isHermitian_mul_conjTranspose_self (toMatrixR r cc X)).eigenvalues i ^ 2) ^ 2|
        ≤ C) :
    ‖toMatrixR r cc (nsIterR X (a, b, c))‖ ≤ Real.sqrt C := by
  have : Nonempty (Fin r) := ⟨⟨0, hr⟩⟩
  set MX := toMatrixR r cc X with hMXdef
  have hHerm := isHermitian_mul_conjTranspose_self MX
  set N := MX * MXᴴ with hNdef
  set qN := (toReal a) • (1 : Matrix (Fin r) (Fin r) ℝ) + (toReal b) • N + (toReal c) • N ^ 2 with hqN
  have hstep : toMatrixR r cc (nsIterR X (a, b, c)) = qN * MX := by
    rw [nsIterR_toMatrixR X a b c r cc hXsz hXrow hr hrc, ← hMXdef,
      ← conjTranspose_eq_transpose_of_trivial MX, ← hNdef, ← hqN]
  rw [hstep]
  have hqNherm : qNᴴ = qN := by
    rw [hqN]
    simp only [conjTranspose_add, conjTranspose_smul, conjTranspose_one, star_trivial,
      conjTranspose_pow, hHerm.eq]
  have hcomm : qN * N = N * qN := by
    have h : Commute qN N := by
      rw [hqN]
      exact (((Commute.one_left N).smul_left _).add_left ((Commute.refl N).smul_left _)).add_left
        (((Commute.refl N).pow_left 2).smul_left _)
    exact h
  have hAAH : (qN * MX) * (qN * MX)ᴴ = N * qN ^ 2 := by
    rw [conjTranspose_mul, hqNherm, Matrix.mul_assoc qN MX (MXᴴ * qN),
      ← Matrix.mul_assoc MX MXᴴ qN, ← hNdef, ← Matrix.mul_assoc qN N qN, hcomm,
      Matrix.mul_assoc N qN qN, ← pow_two]
  have hnorm2 : ‖(qN * MX) * (qN * MX)ᴴ‖ = ‖qN * MX‖ * ‖qN * MX‖ := by
    have := l2_opNorm_conjTranspose_mul_self (qN * MX)ᴴ
    rwa [conjTranspose_conjTranspose, l2_opNorm_conjTranspose] at this
  have hgram : ‖N * qN ^ 2‖ ≤ C :=
    opNorm_gram_poly_le hHerm (toReal a) (toReal b) (toReal c) C hC hbnd
  have hsq : ‖qN * MX‖ ^ 2 ≤ C := by rw [pow_two, ← hnorm2, hAAH]; exact hgram
  have := Real.sqrt_le_sqrt hsq
  rwa [Real.sqrt_sq (norm_nonneg _)] at this

end Puffer.RL.NewtonSchulzAssembly
