/-
The float-`Mat` ↔ Mathlib-`Matrix` embedding — the bridge carrying the abstract operator-norm
results (`Puffer/RL/SpectralBridge.lean`, `MuonGramBound.lean`) back to the runnable pipeline.

The runnable Newton–Schulz operates on `Mat = Array (Array Float)` with ROUNDING; its exact-ℝ mirror
`MatR = Array (Array ℝ)` (`matmulR`/`transposeR`/`lincomb3R` in `NewtonSchulzError.lean`) does the
same index arithmetic in exact reals. This file embeds both into Mathlib `Matrix (Fin r) (Fin c) ℝ`:

  • `toMatrixR` : the exact-ℝ mirror → Mathlib matrix (index lookup);
  • `toMatrixF` : the runnable Float matrix → Mathlib matrix (via `toReal`).

and proves that the mirror's ops become Mathlib's ops EXACTLY (no rounding — that is the point of the
mirror):

  • `matmulR_toMatrixR`   : `toMatrixR (matmulR A B) = toMatrixR A * toMatrixR B`
  • `transposeR_toMatrixR`: `toMatrixR (transposeR X) = (toMatrixR X)ᵀ`
  • `lincomb3R_toMatrixR` : `toMatrixR (lincomb3R a X b Y c Z) = a•X + b•Y + c•Z`

These three are the complete operation set of `nsIterR` (the ℝ mirror of one Newton–Schulz step), so
any mirror computation transfers to Mathlib `Matrix` and the abstract spectral/operator-norm theorems
apply to it. The Float↔mirror ROUNDING gap is exactly the existing `MatBnd` tower, surfaced here at the
matrix level by `toMatrixF_sub_toMatrixR_entry` (`|toMatrixF X − toMatrixR XR|ᵢⱼ ≤ ε` from `MatBnd`).

All axiom-clean beyond the trusted Float `toReal` axiom (`propext`/`Classical.choice`/`Quot.sound`
[+`toReal` for `toMatrixF_sub_toMatrixR_entry`], no `sorry`).

The assembly through `nsIterR`'s shape-branch is done in `Puffer/RL/NewtonSchulzAssembly.lean`
(`nsIterR_toMatrixR` : `toMatrixR (nsIterR X coef) = q(N)·MX`; `nsIterR_opNorm_le` : the per-step
`√C` operator-norm bound), built on these three op-homomorphisms.
-/
import Mathlib
import Puffer.RL.NewtonSchulzError
import Puffer.RL.NewtonSchulzTight

namespace Puffer.RL.MatrixEmbed

open scoped Matrix
open Matrix
open Puffer.RL.NewtonSchulzError (MatR matmulR matmulR_getElem transposeR transposeR_getElem
  lincomb3R lincomb3R_getElem MatBnd)
open Puffer.RL.NewtonSchulzTight (realDotR_eq_sum)
open Puffer.FloatR (toReal)
open Puffer.FloatR.Muon (Mat)

/-- Embed an exact-ℝ mirror matrix (`Array (Array ℝ)`) as a Mathlib `r×c` matrix by index lookup. -/
noncomputable def toMatrixR (r c : Nat) (A : MatR) : Matrix (Fin r) (Fin c) ℝ :=
  Matrix.of fun i j => (A[i.1]!)[j.1]!

/-- Embed a runnable Float matrix (`Array (Array Float)`) as a Mathlib `r×c` real matrix via `toReal`. -/
noncomputable def toMatrixF (r c : Nat) (A : Mat) : Matrix (Fin r) (Fin c) ℝ :=
  Matrix.of fun i j => toReal ((A[i.1]!)[j.1]!)

/-- The Float embedding is entrywise within `ε` of the exact-ℝ mirror embedding, straight from
    `MatBnd` — the rounding gap, made explicit at the Mathlib-`Matrix` level. -/
theorem toMatrixF_sub_toMatrixR_entry (X : Mat) (XR : MatR) (r c : Nat) (M ε : ℝ)
    (h : MatBnd X XR r c M ε) (i : Fin r) (j : Fin c) :
    |toMatrixF r c X i j - toMatrixR r c XR i j| ≤ ε := by
  simp only [toMatrixF, toMatrixR, Matrix.of_apply]
  exact h.err i.1 i.2 j.1 j.2

/-- **Frobenius-distance bound from the entrywise rounding gap.** The entrywise `ε`-bound of `MatBnd` (surfaced
    at the Mathlib-`Matrix` level by `toMatrixF_sub_toMatrixR_entry`) aggregates to a global squared-Frobenius
    bound: the sum of squared entry differences between the runnable Float embedding and the exact-ℝ mirror
    embedding is at most `r·c·ε²`. This turns the local per-entry rounding gap into a single global L²/Frobenius
    error estimate for the matrix Newton–Schulz path. -/
theorem frobSq_toMatrixF_sub_toMatrixR_le (X : Mat) (XR : MatR) (r c : Nat) (M ε : ℝ)
    (h : MatBnd X XR r c M ε) :
    ∑ i : Fin r, ∑ j : Fin c, (toMatrixF r c X i j - toMatrixR r c XR i j) ^ 2
      ≤ (r : ℝ) * c * ε ^ 2 := by
  have hterm : ∀ i : Fin r, ∀ j : Fin c,
      (toMatrixF r c X i j - toMatrixR r c XR i j) ^ 2 ≤ ε ^ 2 := by
    intro i j
    have hb := toMatrixF_sub_toMatrixR_entry X XR r c M ε h i j
    have hle := abs_le.mp hb
    exact sq_le_sq' hle.1 hle.2
  calc ∑ i : Fin r, ∑ j : Fin c, (toMatrixF r c X i j - toMatrixR r c XR i j) ^ 2
      ≤ ∑ _i : Fin r, ∑ _j : Fin c, ε ^ 2 := by
        apply Finset.sum_le_sum; intro i _
        apply Finset.sum_le_sum; intro j _
        exact hterm i j
    _ = (r : ℝ) * c * ε ^ 2 := by
        simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
        ring

/-- `matmulR` becomes Mathlib matrix multiplication under the embedding (EXACT — no rounding). -/
theorem matmulR_toMatrixR (A B : MatR) (r k c : Nat)
    (hA : A.size = r) (hArow : ∀ i, i < r → (A[i]!).size = k)
    (hB : B.size = k) (hBrow : ∀ i, i < k → (B[i]!).size = c)
    (hr : 0 < r) (hk : 0 < k) :
    toMatrixR r c (matmulR A B) = toMatrixR r k A * toMatrixR k c B := by
  ext i j
  rw [toMatrixR, Matrix.of_apply, Matrix.mul_apply]
  have hjc : (j.1) < (if B.size = 0 then 0 else (B[0]!).size) := by
    rw [if_neg (by rw [hB]; omega), hBrow 0 hk]; exact j.2
  rw [matmulR_getElem A B i.1 j.1 (by rw [hA]; exact i.2) hjc]
  have hinner : (if A.size = 0 then 0 else (A[0]!).size) = k := by
    rw [if_neg (by rw [hA]; omega), hArow 0 hr]
  rw [hinner, realDotR_eq_sum, ← Fin.sum_univ_eq_sum_range
    (fun l => (A[i.1]!)[l]! * (B[l]!)[j.1]!) k]
  simp only [toMatrixR, Matrix.of_apply]

/-- `transposeR` becomes Mathlib transpose under the embedding. -/
theorem transposeR_toMatrixR (X : MatR) (r c : Nat)
    (hX : X.size = r) (hXrow : ∀ i, i < r → (X[i]!).size = c) (hr : 0 < r) :
    toMatrixR c r (transposeR X) = (toMatrixR r c X)ᵀ := by
  ext j i
  rw [toMatrixR, Matrix.of_apply, Matrix.transpose_apply, toMatrixR, Matrix.of_apply]
  have hc : (if X.size = 0 then 0 else (X[0]!).size) = c := by
    rw [if_neg (by rw [hX]; omega), hXrow 0 hr]
  rw [transposeR_getElem X i.1 j.1 (by rw [hc]; exact j.2) (by rw [hX]; exact i.2)]

/-- `lincomb3R` becomes the Mathlib linear combination `a•X + b•Y + c•Z` under the embedding. -/
theorem lincomb3R_toMatrixR (a b c : ℝ) (X Y Z : MatR) (r cc : Nat)
    (hX : X.size = r) (hXrow : ∀ i, i < r → (X[i]!).size = cc) :
    toMatrixR r cc (lincomb3R a X b Y c Z)
      = a • toMatrixR r cc X + b • toMatrixR r cc Y + c • toMatrixR r cc Z := by
  ext i j
  rw [toMatrixR, Matrix.of_apply,
    lincomb3R_getElem a X b Y c Z i.1 j.1 (by rw [hX]; exact i.2) (by rw [hXrow i.1 i.2]; exact j.2)]
  simp only [Matrix.add_apply, Matrix.smul_apply, toMatrixR, Matrix.of_apply, smul_eq_mul]

end Puffer.RL.MatrixEmbed
