/-
The norm-based error tower — piece (ii) of the tightening of the Newton–Schulz `FE` constant: the
additive-error replacement for the entrywise `MatBnd` tower, whose per-entry magnitude `idxDotMagBnd ≈ k·M²`
is the doubly-exponential culprit (~10²¹⁶⁰ after 5 iterations).

The entrywise `NewtonSchulzError.matmul_MatBnd` propagates a per-entry magnitude `M` that a matmul turns into
`idxDotMagBnd M M k ≈ k·M²` — SQUARING each matmul, so over 3 matmuls × 5 iterations the constant is
doubly-exponential. This file re-derives the matmul step against NORM-based quantities instead: the error is
measured in the l2 OPERATOR norm and the magnitude in the FROBENIUS norm, and the recurrence is ADDITIVE in
the input errors (no squaring), with O(1)/O(√dim) coefficients.

  • `frobRow` / `frobCol` : the Frobenius norm of a Float matrix as a row-sum / column-sum of `sumSqR` (these
      are the two forms the matmul rounding atom `NewtonSchulzFrobBridge.matmul_rounding_opNorm` produces).
  • `opNorm_le_frobRow` / `opNorm_le_frobCol` : `‖toMatrixF X‖₂ ≤ frob X` (operator ≤ Frobenius, via
      `opNorm_sq_le_frobenius_sq`).
  • `matmul_error_opNorm` — THE ATOM: for `A·B` (Float) with mirror `AR·BR`,

      ‖toMatrixF(matmul A B) − toMatrixR(matmulR AR BR)‖₂
        ≤ β(k)·‖A‖_F·‖B‖_F               (rounding, from `matmul_rounding_opNorm`)
          + εA·‖B‖_F + ‖toMatrixR AR‖₂·εB  (perturbation, submultiplicative + triangle),

      `β(k) = (3+u64)·((1+u64)ᵏ − 1) ≈ 3k·u64`, `εA = ‖toMatrixF A − toMatrixR AR‖₂`, likewise `εB`. The error
      accumulates ADDITIVELY in `εA`, `εB` with coefficients `‖B‖_F` (= O(√dim) by the proven mirror bound) and
      `‖toMatrixR AR‖₂` (= O(1), the mirror operator norm) — versus the entrywise tower's `k·M²` squaring. The
      proof splits `toMatrixF(matmul) − toMatrixR(matmulR)` into the rounding part (`matmul_rounding_opNorm`)
      and the perturbation part `A_f·B_f − AR·BR = (A_f−AR)·B_f + AR·(B_f−BR)` (`l2_opNorm_mul` +
      `matmulR_toMatrixR` for the mirror product).

  • `transpose_toMatrixF` / `transpose_error_opNorm` — the second op: `transpose` is a pure reindexing, hence
      an l2-operator ISOMETRY (`l2_opNorm_conjTranspose`, real `Aᴴ = Aᵀ`), so `‖toMatrixF(transpose X) −
      toMatrixR(transposeR XR)‖₂ ≤ εX` — its error equals the input error EXACTLY, contributing zero growth.

  • `scalarMul_error_opNorm` — the third op: `scalarMul c X` scales each entry (with rounding), so the error is
      the per-entry rounding aggregated to `u64·|toReal c|·‖X‖_F` (via `opNorm_sq_le_frobenius_sq` + `mul_error`)
      plus the scalar perturbation `|toReal c|·εX` (`norm_smul`).
  • `lincomb3_error_opNorm` — the fourth op: the 3-term combine `a·X + b·Y + c·Z` (each term rounding). The
      per-entry rounding (`lincomb3Entry_error`) is bounded by `K·(|a||x|+|b||y|+|c||z|)`
      (`lincomb3_entry_le`, `K = u64·(3+3u64+u64²)`), aggregated to `√3·K·(|a|‖X‖_F+|b|‖Y‖_F+|c|‖Z‖_F)`
      (`lincomb3_agg`, via `(p+q+r)² ≤ 3(p²+q²+r²)` + ℓ² subadditivity), plus the 3-term perturbation
      `|a|εX+|b|εY+|c|εZ` (`lincomb3R_toMatrixR` + `norm_smul`).

ALL FOUR `nsIter` ops — `matmul` (the crux, additive error), `transpose` (zero-growth isometry), `scalarMul`,
`lincomb3` — are re-derived in additive-error norm form, and ASSEMBLED into one `nsIter` step for BOTH shape
branches (`gram_error_opNorm`/`gram_gt_error_opNorm` + `matmul_error_opNorm` chain +
`nsIter_error_opNorm`/`nsIter_error_opNorm_gt`): the step error is ADDITIVE in `εX`, so one iteration is a
Lipschitz map on the error with polynomial coefficients. Axiom-clean modulo the trusted Float base
(`transpose_error_opNorm` is pure `toReal` + Mathlib).

The 5-iteration FOLD is then closed: `fold_affine_error` shows a UNIFORM per-step affine bound `ε' ≤ L·ε + C`
folds to `L^len·ε₀ + C·∑Lᵏ` (affine, hence POLYNOMIAL), and `newtonSchulz_opNorm_poly` adds the proven mirror
bound (`nsIterR_comp_normsq`, `≤ √1.3131`) to give `‖toMatrixF(newtonSchulz X0 eps)‖₂ ≤ √1.3131 + polynomial(FE)` —
the culminating tight-tower statement with a POLYNOMIAL `FE`, versus the doubly-exponential entrywise one.

The BOOTSTRAP's magnitude control is COMPLETE (row and column forms): `frobRow_le_opNorm`/`frobCol_le_opNorm`
(`frob ≤ √dim·‖·‖₂`, Frobenius ≤ √dim · operator via `trace(AᴴA) = ∑λ ≤ dim·‖·‖²`) and
`frobRow_le_mirror_add_err`/`frobCol_le_mirror_add_err` (`frob ≤ √dim·(‖mirror‖₂ + ε)`) bound every
`frobRow`/`frobCol` coefficient of the per-step bound by the proven-bounded mirror operator norm plus the
error. So while the error stays bounded the magnitudes stay `O(√dim)`, making the per-step `L`, `C` uniform.
`fold_region_affine_error` closes the COUPLING: the per-step affine bound need only hold in the self-bounding
region `{ε ≤ εmax}` (`L·εmax + C ≤ εmax`), which is then forward-INVARIANT (the `region_invariant` pattern on
the error), and `newtonSchulz_opNorm_region` is the culminating statement with that region-restricted per-step
bound.

The per-step composition is DISCHARGED: `gram_step_affine` (the Gram step is affine in `εX`) and
`matmul_step_affine` (one matmul with two affine operands composes to an affine output) chain, and
`nsIter_step_affine` collapses the whole `nsIter` step (`gram` + two matmuls + `nsIter_error_opNorm`) into a
single `ε' ≤ L·εX + C` with `L`, `C` explicit polynomials in the (bounded) Frobenius magnitudes and mirror
op-norms — adversarially verified sound, `L`/`C` matching the composition term-for-term. So one `nsIter` step
is genuinely a Lipschitz-affine map on the error.

Every `frobRow`/`frobCol` magnitude hypothesis of `nsIter_step_affine` is dischargeable from
mirror-operator-norm + error data (via the four `frob*_le_*` lemmas above). The coupled bootstrap is then
FULLY THREADED: `fold_affine_mirror` carries a per-prefix mirror-bound hypothesis `hmir` (the POSITION-SPECIFIC
op-norms `‖mirror(prefix-fold)‖ ≤ Rm` — there is NO uniform mirror invariant, since step 1 maps the scalar
interval `[0,1] → [0,1.63]`) through the fold, so the per-step bound may assume `‖mirror XR‖ ≤ Rm`;
`newtonSchulz_opNorm_mirror` is the culminating statement `‖toMatrixF(newtonSchulz)‖₂ ≤ √1.3131 +
polynomial(FE)` whose two remaining hypotheses are both genuinely dischargeable — `hstep` from
`nsIter_step_affine` (magnitudes bounded from `Rm` + the region via the `frob*_le_mirror_add_err` lemmas and
the intermediates' mirror op-norms by submultiplicativity), and `hmir` from `MuonComposition` (`Rm = √1.63`).

CORRECTION (the EXPANDING recurrence): the Newton–Schulz per-step Lipschitz factor is `L > 1` (the error
amplifies by ~`|a|+3|b|+5|c| ≈ 40` per step), so the SELF-BOUNDING region `L·εmax + C ≤ εmax` of
`fold_region_affine_error`/`fold_affine_mirror` is EMPTY — those are correctly proven but their region
hypothesis is not dischargeable for the real recurrence. `fold_affine_mirror_exp` is the fix (final error
`L^len·ε₀ + C·∑Lᵏ ≤ B`, threading a "final-error-from-here ≤ B" invariant), and `newtonSchulz_opNorm_mirror_exp`
is the corrected culminating statement — the fold error stays polynomial (`~40⁵·ε_seed ≈ 10⁸·u64`, still tiny)
even though it expands. This TIGHTENS an already-closed bound (`NewtonSchulzFull.newtonSchulz_opNorm_le`); it
changes no correctness claim. A fully hypothesis-free theorem still needs the (large but mechanical) discharge
of `hstep` (wire `nsIter_step_affine` + magnitude lemmas + submultiplicativity) and `hmir` (lift the per-
position `MuonComposition` bounds to `toMatrixR`) with concrete `L, C, B, Rm`.
-/
import Mathlib
import Puffer.RL.NewtonSchulzFrobBridge
import Puffer.RL.MuonStepBound

namespace Puffer.RL.NewtonSchulzNormTower

open scoped Matrix Matrix.Norms.L2Operator
open Matrix
open Puffer.FloatR (toReal u64 u64_pos mul_error mul_abs_le add_abs_le)
open Puffer.FloatR.Muon (Mat matmul newtonSchulz scalarMul frobNorm muonCoeffs)
open Puffer.RL.NewtonSchulzError (MatR matmulR transposeR scalarMulR scalarMulR_getElem scalarMul_getElem
  lincomb3R lincomb3_getElem nsIterR transposeR_size transposeR_rowSize matmulR_size matmulR_rowSize)
open Puffer.RL.NewtonSchulzTight (sumSqR sumSqR_nonneg)
open Puffer.RL.MuonMatrixRuntime (transpose_getElem matLinEntryErrBnd lincomb3EntryErrBnd lincomb3Entry_error
  nsIter transpose_size transpose_rowSize newtonSchulz_eq_foldl matmul_size matmul_rowSize)
open Puffer.RL.MuonStepBound (matLinEntryErrBnd_le)
open Puffer.RL.MatrixEmbed (toMatrixF toMatrixR matmulR_toMatrixR transposeR_toMatrixR lincomb3R_toMatrixR)
open Puffer.RL.NewtonSchulzCompMirror (nsIterR_comp_normsq nsIterR_step_normsq)
open Puffer.RL.NewtonSchulzTransport (nsIterR_size nsIterR_rowSize)
open Puffer.RL.MuonCompositionFloat (muon_comp_step1_float muon_comp_step2_float muon_comp_step3_float
  muon_comp_step4_float muon_comp_step5_float)
open Puffer.RL.MuonCompositionMatrix (gram_eig_le_opNorm_sq)
open Puffer.RL.NewtonSchulzRunnable (opNorm_sq_le_frobenius_sq)
open Puffer.RL.NewtonSchulzFrobBridge (matmul_rounding_opNorm)

/-- Frobenius norm of a Float matrix as a ROW-sum of `sumSqR` (over the first `a` rows, length-`b` prefixes) —
    the form the rounding atom produces for a LEFT matmul operand. -/
noncomputable def frobRow (X : Mat) (a b : Nat) : ℝ :=
  Real.sqrt (∑ i ∈ Finset.range a, sumSqR (fun l => toReal ((X[i]!)[l]!)) b)

/-- Frobenius norm of a Float matrix as a COLUMN-sum of `sumSqR` — the form the rounding atom produces for a
    RIGHT matmul operand. (Equal to `frobRow` of the transpose; the same Frobenius value, different index form.) -/
noncomputable def frobCol (X : Mat) (a b : Nat) : ℝ :=
  Real.sqrt (∑ j ∈ Finset.range a, sumSqR (fun l => toReal ((X[l]!)[j]!)) b)

/-- `‖toMatrixF r k A‖₂ ≤ frobRow A r k` — operator norm ≤ Frobenius norm (row form), via
    `opNorm_sq_le_frobenius_sq` (`‖·‖² ≤ ∑ᵢⱼ ·²`) with the double sum rewritten as the row-sum of `sumSqR`. -/
theorem opNorm_le_frobRow (A : Mat) (r k : Nat) [Nonempty (Fin k)] :
    ‖toMatrixF r k A‖ ≤ frobRow A r k := by
  have hsq := opNorm_sq_le_frobenius_sq (toMatrixF r k A)
  have hconv : (∑ i, ∑ j, (toMatrixF r k A i j) ^ 2)
      = ∑ i ∈ Finset.range r, sumSqR (fun l => toReal ((A[i]!)[l]!)) k := by
    rw [← Fin.sum_univ_eq_sum_range (fun i => sumSqR (fun l => toReal ((A[i]!)[l]!)) k) r]
    apply Finset.sum_congr rfl; intro i _
    rw [sumSqR, ← Fin.sum_univ_eq_sum_range (fun j => (toReal ((A[i.1]!)[j]!)) ^ 2) k]
    apply Finset.sum_congr rfl; intro j _; simp only [toMatrixF, Matrix.of_apply]
  rw [hconv] at hsq
  rw [frobRow, ← Real.sqrt_sq (norm_nonneg (toMatrixF r k A))]; exact Real.sqrt_le_sqrt hsq

/-- **Every Float entry is dominated by the Frobenius magnitude (ℓ∞ ≤ Frobenius).**
    For any in-range index `i < r`, `j < c`, the absolute real value of the `(i,j)` Float entry is at most
    `frobRow X r c`. This is the entrywise magnitude-control companion of the aggregate `opNorm_le_frobRow`:
    it lets the bootstrap's bound on `frobRow` (mirror op-norm + error) bound EVERY individual Float entry of
    every iterate. Proof extracts the single squared entry from the `frobRow` double sum via two
    `Finset.single_le_sum` steps (inner over columns `< c`, outer over rows `< r`), then transfers through
    `Real.sqrt` monotonicity and `Real.sqrt_sq_eq_abs`. Both hypotheses are load-bearing: an entry in a row
    `i ≥ r` (or column `j ≥ c`) is not summed into `frobRow`, so it can exceed it. -/
theorem abs_entry_le_frobRow (X : Mat) (r c i j : Nat) (hi : i < r) (hj : j < c) :
    |toReal ((X[i]!)[j]!)| ≤ frobRow X r c := by
  have hinner : (toReal ((X[i]!)[j]!)) ^ 2 ≤ sumSqR (fun l => toReal ((X[i]!)[l]!)) c := by
    rw [sumSqR]
    exact Finset.single_le_sum (f := fun l => (toReal ((X[i]!)[l]!)) ^ 2)
      (fun l _ => sq_nonneg _) (Finset.mem_range.mpr hj)
  have houter : sumSqR (fun l => toReal ((X[i]!)[l]!)) c
      ≤ ∑ i' ∈ Finset.range r, sumSqR (fun l => toReal ((X[i']!)[l]!)) c :=
    Finset.single_le_sum (f := fun i' => sumSqR (fun l => toReal ((X[i']!)[l]!)) c)
      (fun i' _ => sumSqR_nonneg _ _) (Finset.mem_range.mpr hi)
  have hsq : (toReal ((X[i]!)[j]!)) ^ 2
      ≤ ∑ i' ∈ Finset.range r, sumSqR (fun l => toReal ((X[i']!)[l]!)) c := le_trans hinner houter
  calc |toReal ((X[i]!)[j]!)| = Real.sqrt ((toReal ((X[i]!)[j]!)) ^ 2) := (Real.sqrt_sq_eq_abs _).symm
    _ ≤ Real.sqrt (∑ i' ∈ Finset.range r, sumSqR (fun l => toReal ((X[i']!)[l]!)) c) := Real.sqrt_le_sqrt hsq
    _ = frobRow X r c := by rw [frobRow]

/-- **Every entry of a real matrix is bounded by its l2 operator norm** — `|A i j| ≤ ‖A‖₂`. Apply `A` to the
    `j`ᵗʰ standard basis vector `eⱼ`: the result `A *ᵥ eⱼ` is column `j`, its l2 norm is `≤ ‖A‖·‖eⱼ‖ = ‖A‖`
    (`l2_opNorm_mulVec`, `EuclideanSpace.norm_single`), its `i`-th component is exactly `A i j`, and any single
    component of a Euclidean vector is `≤` its l2 norm (`PiLp.norm_apply_le`). -/
theorem abs_matrix_entry_le_l2_opNorm {r c : Nat} (A : Matrix (Fin r) (Fin c) ℝ)
    (i : Fin r) (j : Fin c) : |A i j| ≤ ‖A‖ := by
  have hx : ‖(EuclideanSpace.single j (1 : ℝ) : EuclideanSpace ℝ (Fin c))‖ = 1 := by
    rw [EuclideanSpace.norm_single, norm_one]
  have hmv := A.l2_opNorm_mulVec (EuclideanSpace.single j (1 : ℝ))
  rw [hx, mul_one] at hmv
  have hcomp : ‖((EuclideanSpace.equiv (Fin r) ℝ).symm
        (A *ᵥ (EuclideanSpace.single j (1 : ℝ)))) i‖
      ≤ ‖(EuclideanSpace.equiv (Fin r) ℝ).symm (A *ᵥ (EuclideanSpace.single j (1 : ℝ)))‖ :=
    PiLp.norm_apply_le _ i
  have hval : ((EuclideanSpace.equiv (Fin r) ℝ).symm
        (A *ᵥ (EuclideanSpace.single j (1 : ℝ)))) i = A i j := by
    show (A *ᵥ (EuclideanSpace.single j (1 : ℝ))) i = A i j
    simp only [Matrix.mulVec, dotProduct, EuclideanSpace.single_apply, mul_ite, mul_one, mul_zero]
    rw [Finset.sum_ite_eq' Finset.univ j (fun k => A i k)]
    simp
  rw [hval, Real.norm_eq_abs] at hcomp
  exact le_trans hcomp hmv

/-- **Every Float entry is dominated by the l2 OPERATOR norm (ℓ∞ ≤ operator norm).** For any in-range index
    `i < r`, `j < c`, the absolute real value of the `(i,j)` Float entry is at most `‖toMatrixF r c X‖₂`. This is
    the operator-norm counterpart of `abs_entry_le_frobRow` and STRICTLY SHARPENS it: since
    `‖toMatrixF r c X‖ ≤ frobRow X r c` (`opNorm_le_frobRow`), chaining gives
    `|entry| ≤ ‖toMatrixF‖ ≤ frobRow` — tighter by up to a factor `√c`. Because the mirror operator norm is proven
    `O(1)` (no `√dim`), this bounds every Float entry of every Newton–Schulz iterate by an `O(1)` quantity (mirror
    op-norm + error), versus the `O(√dim)` bound through `frobRow`. Proof: reduce to the general ℝ fact
    `abs_matrix_entry_le_l2_opNorm` on `toMatrixF r c X` at the `Fin`-cast indices. Both bounds `hi`, `hj` are
    load-bearing: an entry in a row `i ≥ r` (or column `j ≥ c`) is not part of the embedded matrix, so it can
    exceed the norm. -/
theorem abs_entry_le_opNorm (X : Mat) (r c i j : Nat) (hi : i < r) (hj : j < c) :
    |toReal ((X[i]!)[j]!)| ≤ ‖toMatrixF r c X‖ := by
  have h := abs_matrix_entry_le_l2_opNorm (toMatrixF r c X) ⟨i, hi⟩ ⟨j, hj⟩
  rwa [toMatrixF, Matrix.of_apply] at h

/-- `‖toMatrixF k n B‖₂ ≤ frobCol B n k` — operator norm ≤ Frobenius norm (column form); same as
    `opNorm_le_frobRow` but the double sum is reindexed (`Finset.sum_comm`) into the column-sum. -/
theorem opNorm_le_frobCol (B : Mat) (k n : Nat) [Nonempty (Fin n)] :
    ‖toMatrixF k n B‖ ≤ frobCol B n k := by
  have hsq := opNorm_sq_le_frobenius_sq (toMatrixF k n B)
  have hconv : (∑ i, ∑ j, (toMatrixF k n B i j) ^ 2)
      = ∑ j ∈ Finset.range n, sumSqR (fun l => toReal ((B[l]!)[j]!)) k := by
    rw [Finset.sum_comm, ← Fin.sum_univ_eq_sum_range (fun j => sumSqR (fun l => toReal ((B[l]!)[j]!)) k) n]
    apply Finset.sum_congr rfl; intro j _
    rw [sumSqR, ← Fin.sum_univ_eq_sum_range (fun l => (toReal ((B[l]!)[j.1]!)) ^ 2) k]
    apply Finset.sum_congr rfl; intro l _; simp only [toMatrixF, Matrix.of_apply]
  rw [hconv] at hsq
  rw [frobCol, ← Real.sqrt_sq (norm_nonneg (toMatrixF k n B))]; exact Real.sqrt_le_sqrt hsq

/-- **The matmul error atom (norm form) — the additive-error replacement for the entrywise `matmul_MatBnd`.**
    For `A·B` (Float, `r×k · k×n`) with exact-ℝ mirror `AR·BR`, the l2-OPERATOR-norm error is

      ‖toMatrixF(matmul A B) − toMatrixR(matmulR AR BR)‖₂
        ≤ β(k)·‖A‖_F·‖B‖_F + εA·‖B‖_F + ‖toMatrixR AR‖₂·εB,

    `β(k) = (3+u64)·((1+u64)ᵏ − 1) ≈ 3k·u64`, `εA = ‖toMatrixF A − toMatrixR AR‖₂`, `εB` likewise. The error is
    ADDITIVE in `εA`, `εB` (coefficients `‖B‖_F` and the mirror operator norm `‖toMatrixR AR‖₂`), plus the
    rounding `β·‖A‖_F·‖B‖_F` — NO squaring of a magnitude, unlike the entrywise `idxDotMagBnd`'s `k·M²`. Proof:
    split into rounding (`matmul_rounding_opNorm`) + perturbation `A_f·B_f − AR·BR = (A_f−AR)·B_f + AR·(B_f−BR)`
    (`l2_opNorm_mul` submultiplicativity + `matmulR_toMatrixR` for the mirror product), and
    `opNorm_le_frobCol` for `‖B_f‖₂ ≤ ‖B‖_F`. -/
theorem matmul_error_opNorm (A B : Mat) (AR BR : MatR) (r k n : Nat) [Nonempty (Fin n)]
    (εA εB : ℝ)
    (hAsz : A.size = r) (hAk : (if A = #[] then 0 else (A[0]!).size) = k)
    (hBn : (if B = #[] then 0 else B[0]!.size) = n)
    (hARsz : AR.size = r) (hARrow : ∀ i, i < r → (AR[i]!).size = k)
    (hBRsz : BR.size = k) (hBRrow : ∀ i, i < k → (BR[i]!).size = n)
    (hr : 0 < r) (hk : 0 < k)
    (hεA : ‖toMatrixF r k A - toMatrixR r k AR‖ ≤ εA)
    (hεB : ‖toMatrixF k n B - toMatrixR k n BR‖ ≤ εB) :
    ‖toMatrixF r n (matmul A B) - toMatrixR r n (matmulR AR BR)‖
      ≤ ((2 : ℝ) + u64) * (((1 : ℝ) + u64) ^ k - 1) * (frobRow A r k * frobCol B n k)
        + εA * ‖toMatrixF k n B‖ + ‖toMatrixR r k AR‖ * εB := by
  have hmir : toMatrixR r n (matmulR AR BR) = toMatrixR r k AR * toMatrixR k n BR :=
    matmulR_toMatrixR AR BR r k n hARsz hARrow hBRsz hBRrow hr hk
  haveI hinst : Nonempty (Fin (if B = #[] then 0 else B[0]!.size)) := by rw [hBn]; infer_instance
  have ha32 := matmul_rounding_opNorm A B
  rw [hAsz, hAk, hBn] at ha32
  have hsplit : toMatrixF r k A * toMatrixF k n B - toMatrixR r k AR * toMatrixR k n BR
      = (toMatrixF r k A - toMatrixR r k AR) * toMatrixF k n B
        + toMatrixR r k AR * (toMatrixF k n B - toMatrixR k n BR) := by
    rw [Matrix.sub_mul, Matrix.mul_sub]; abel
  have hpert : ‖toMatrixF r k A * toMatrixF k n B - toMatrixR r k AR * toMatrixR k n BR‖
      ≤ εA * ‖toMatrixF k n B‖ + ‖toMatrixR r k AR‖ * εB := by
    rw [hsplit]
    refine le_trans (norm_add_le _ _) ?_
    have h1 : ‖(toMatrixF r k A - toMatrixR r k AR) * toMatrixF k n B‖ ≤ εA * ‖toMatrixF k n B‖ := by
      refine le_trans (l2_opNorm_mul _ _) ?_
      exact mul_le_mul hεA le_rfl (norm_nonneg _) (le_trans (norm_nonneg _) hεA)
    have h2 : ‖toMatrixR r k AR * (toMatrixF k n B - toMatrixR k n BR)‖ ≤ ‖toMatrixR r k AR‖ * εB := by
      refine le_trans (l2_opNorm_mul _ _) ?_
      exact mul_le_mul_of_nonneg_left hεB (norm_nonneg _)
    linarith
  calc ‖toMatrixF r n (matmul A B) - toMatrixR r n (matmulR AR BR)‖
      = ‖(toMatrixF r n (matmul A B) - toMatrixF r k A * toMatrixF k n B)
          + (toMatrixF r k A * toMatrixF k n B - toMatrixR r n (matmulR AR BR))‖ := by
        congr 1; abel
    _ ≤ ‖toMatrixF r n (matmul A B) - toMatrixF r k A * toMatrixF k n B‖
          + ‖toMatrixF r k A * toMatrixF k n B - toMatrixR r n (matmulR AR BR)‖ := norm_add_le _ _
    _ ≤ ((2 : ℝ) + u64) * (((1 : ℝ) + u64) ^ k - 1) * (frobRow A r k * frobCol B n k)
          + (εA * ‖toMatrixF k n B‖ + ‖toMatrixR r k AR‖ * εB) := by
        refine add_le_add ?_ ?_
        · rw [frobRow, frobCol]; exact ha32
        · rw [hmir]; exact hpert
    _ = ((2 : ℝ) + u64) * (((1 : ℝ) + u64) ^ k - 1) * (frobRow A r k * frobCol B n k)
          + εA * ‖toMatrixF k n B‖ + ‖toMatrixR r k AR‖ * εB := by ring

/-- Float `transpose` embeds as the Mathlib transpose of the embedding (the `toMatrixF` analogue of
    `MatrixEmbed.transposeR_toMatrixR`), via `transpose_getElem` (`(transpose X)[j][i] = X[i][j]`). -/
theorem transpose_toMatrixF (X : Mat) (r c : Nat)
    (hX : X.size = r) (hXrow : ∀ i, i < r → (X[i]!).size = c) (hr : 0 < r) :
    toMatrixF c r (Puffer.FloatR.Muon.transpose X) = (toMatrixF r c X)ᵀ := by
  ext j i
  rw [toMatrixF, Matrix.of_apply, Matrix.transpose_apply, toMatrixF, Matrix.of_apply]
  have hc : (if X = #[] then 0 else (X[0]!).size) = c := by
    rw [if_neg (by intro h; rw [h] at hX; simp at hX; omega), hXrow 0 hr]
  rw [transpose_getElem X i.1 j.1 (by rw [hc]; exact j.2) (by rw [hX]; exact i.2)]

/-- **The transpose error atom — an l2-operator ISOMETRY.** `transpose` is pure reindexing (no arithmetic, no
    rounding), so its error equals the input error EXACTLY — no growth at all:
    `‖toMatrixF(transpose X) − toMatrixR(transposeR XR)‖₂ ≤ εX`. Proof: both embeddings become Mathlib
    transposes (`transpose_toMatrixF`, `transposeR_toMatrixR`), `(·)ᵀ` distributes over the difference
    (`Matrix.transpose_sub`), and the l2 operator norm is transpose-invariant (`l2_opNorm_conjTranspose`, with
    `conjTranspose_eq_transpose_of_trivial` since `star = id` on ℝ). This is the second retrofit op after
    `matmul_error_opNorm`; unlike matmul it contributes ZERO to the error growth. -/
theorem transpose_error_opNorm (X : Mat) (XR : MatR) (r c : Nat) (εX : ℝ)
    (hXsz : X.size = r) (hXrow : ∀ i, i < r → (X[i]!).size = c)
    (hXRsz : XR.size = r) (hXRrow : ∀ i, i < r → (XR[i]!).size = c) (hr : 0 < r)
    (hεX : ‖toMatrixF r c X - toMatrixR r c XR‖ ≤ εX) :
    ‖toMatrixF c r (Puffer.FloatR.Muon.transpose X) - toMatrixR c r (transposeR XR)‖ ≤ εX := by
  rw [transpose_toMatrixF X r c hXsz hXrow hr, transposeR_toMatrixR XR r c hXRsz hXRrow hr,
    ← Matrix.transpose_sub, ← conjTranspose_eq_transpose_of_trivial, l2_opNorm_conjTranspose]
  exact hεX

/-! ### The `scalarMul` op — entrywise rounding (aggregated) + a scalar perturbation

`scalarMul c X` scales every entry by `c` (Float, WITH rounding). Its mirror `scalarMulR (toReal c) XR` scales
exactly. The error is the per-entry rounding `|toReal(c·x) − toReal c·toReal x| ≤ u64·|toReal c|·|x|`
(`mul_error`), aggregated to a Frobenius bound `u64·|toReal c|·‖X‖_F` (via `opNorm_sq_le_frobenius_sq`), plus a
scalar perturbation `|toReal c|·εX` (`norm_smul`). -/

/-- Mirror of `scalarMul` embeds as the exact scalar multiple of the embedding. -/
theorem scalarMulR_toMatrixR (s : ℝ) (XR : MatR) (r cc : Nat)
    (hXRsz : XR.size = r) (hXRrow : ∀ i, i < r → (XR[i]!).size = cc) :
    toMatrixR r cc (scalarMulR s XR) = s • toMatrixR r cc XR := by
  ext i j
  rw [toMatrixR, Matrix.of_apply, Matrix.smul_apply, toMatrixR, Matrix.of_apply, smul_eq_mul,
    scalarMulR_getElem s XR i.1 j.1 (by rw [hXRsz]; exact i.2) (by rw [hXRrow i.1 i.2]; exact j.2)]

/-- The per-entry rounding of `scalarMul` aggregated to an operator-norm bound: `u64·|toReal c|·‖X‖_F`. -/
theorem scalarMul_rounding_opNorm (c : Float) (X : Mat) (r cc : Nat) [Nonempty (Fin cc)]
    (hXsz : X.size = r) (hXrow : ∀ i, i < r → (X[i]!).size = cc) :
    ‖toMatrixF r cc (Puffer.FloatR.Muon.scalarMul c X) - toReal c • toMatrixF r cc X‖
      ≤ u64 * |toReal c| * frobRow X r cc := by
  set D := toMatrixF r cc (Puffer.FloatR.Muon.scalarMul c X) - toReal c • toMatrixF r cc X with hD
  have hentry : ∀ (i : Fin r) (j : Fin cc),
      (D i j) ^ 2 ≤ (u64 * |toReal c|) ^ 2 * (toReal ((X[i.1]!)[j.1]!)) ^ 2 := by
    intro i j
    have hbnd : |D i j| ≤ u64 * |toReal c| * |toReal ((X[i.1]!)[j.1]!)| := by
      rw [hD, Matrix.sub_apply, Matrix.smul_apply, toMatrixF, toMatrixF, Matrix.of_apply, Matrix.of_apply,
        smul_eq_mul, scalarMul_getElem c X i.1 j.1 (by rw [hXsz]; exact i.2) (by rw [hXrow i.1 i.2]; exact j.2)]
      calc |toReal (c * (X[i.1]!)[j.1]!) - toReal c * toReal ((X[i.1]!)[j.1]!)|
          ≤ u64 * |toReal c * toReal ((X[i.1]!)[j.1]!)| := mul_error c ((X[i.1]!)[j.1]!)
        _ = u64 * |toReal c| * |toReal ((X[i.1]!)[j.1]!)| := by rw [abs_mul]; ring
    have h0 : 0 ≤ u64 * |toReal c| * |toReal ((X[i.1]!)[j.1]!)| :=
      mul_nonneg (mul_nonneg u64_pos.le (abs_nonneg _)) (abs_nonneg _)
    nlinarith [sq_abs (D i j), abs_nonneg (D i j), hbnd, h0, sq_abs (toReal ((X[i.1]!)[j.1]!))]
  have hCnn : 0 ≤ ∑ i ∈ Finset.range r, sumSqR (fun l => toReal ((X[i]!)[l]!)) cc :=
    Finset.sum_nonneg (fun i _ => sumSqR_nonneg _ _)
  have hsum : (∑ i : Fin r, ∑ j : Fin cc, (D i j) ^ 2)
      ≤ (u64 * |toReal c|) ^ 2 * (∑ i ∈ Finset.range r, sumSqR (fun l => toReal ((X[i]!)[l]!)) cc) := by
    have hconv : (∑ i : Fin r, ∑ j : Fin cc, (u64 * |toReal c|) ^ 2 * (toReal ((X[i.1]!)[j.1]!)) ^ 2)
        = (u64 * |toReal c|) ^ 2 * (∑ i ∈ Finset.range r, sumSqR (fun l => toReal ((X[i]!)[l]!)) cc) := by
      rw [Finset.mul_sum,
        ← Fin.sum_univ_eq_sum_range (fun i => (u64 * |toReal c|) ^ 2 * sumSqR (fun l => toReal ((X[i]!)[l]!)) cc) r]
      apply Finset.sum_congr rfl; intro i _
      rw [sumSqR, Finset.mul_sum,
        ← Fin.sum_univ_eq_sum_range (fun j => (u64 * |toReal c|) ^ 2 * (toReal ((X[i.1]!)[j]!)) ^ 2) cc]
    calc (∑ i : Fin r, ∑ j : Fin cc, (D i j) ^ 2)
        ≤ ∑ i : Fin r, ∑ j : Fin cc, (u64 * |toReal c|) ^ 2 * (toReal ((X[i.1]!)[j.1]!)) ^ 2 :=
          Finset.sum_le_sum (fun i _ => Finset.sum_le_sum (fun j _ => hentry i j))
      _ = _ := hconv
  have hfr : (frobRow X r cc) ^ 2 = ∑ i ∈ Finset.range r, sumSqR (fun l => toReal ((X[i]!)[l]!)) cc := by
    rw [frobRow, Real.sq_sqrt hCnn]
  have hsq : ‖D‖ ^ 2 ≤ ((u64 * |toReal c|) * frobRow X r cc) ^ 2 := by
    refine le_trans (opNorm_sq_le_frobenius_sq D) ?_
    rw [mul_pow, hfr]; exact hsum
  have hRHS : 0 ≤ (u64 * |toReal c|) * frobRow X r cc := by
    rw [frobRow]; exact mul_nonneg (mul_nonneg u64_pos.le (abs_nonneg _)) (Real.sqrt_nonneg _)
  calc ‖D‖ = Real.sqrt (‖D‖ ^ 2) := (Real.sqrt_sq (norm_nonneg _)).symm
    _ ≤ Real.sqrt (((u64 * |toReal c|) * frobRow X r cc) ^ 2) := Real.sqrt_le_sqrt hsq
    _ = u64 * |toReal c| * frobRow X r cc := by rw [Real.sqrt_sq hRHS]

/-- **The scalarMul error atom (norm form).** `‖toMatrixF(scalarMul c X) − toMatrixR(scalarMulR (toReal c)
    XR)‖₂ ≤ u64·|toReal c|·‖X‖_F + |toReal c|·εX` — the entrywise rounding aggregated to a Frobenius term plus
    the scalar perturbation. The error scales by `|toReal c|` (contraction if `|c| ≤ 1`, as for the
    Frobenius-normalization seed and the `nsIter` combine coefficients). -/
theorem scalarMul_error_opNorm (c : Float) (X : Mat) (XR : MatR) (r cc : Nat) [Nonempty (Fin cc)] (εX : ℝ)
    (hXsz : X.size = r) (hXrow : ∀ i, i < r → (X[i]!).size = cc)
    (hXRsz : XR.size = r) (hXRrow : ∀ i, i < r → (XR[i]!).size = cc)
    (hεX : ‖toMatrixF r cc X - toMatrixR r cc XR‖ ≤ εX) :
    ‖toMatrixF r cc (Puffer.FloatR.Muon.scalarMul c X) - toMatrixR r cc (scalarMulR (toReal c) XR)‖
      ≤ u64 * |toReal c| * frobRow X r cc + |toReal c| * εX := by
  rw [scalarMulR_toMatrixR (toReal c) XR r cc hXRsz hXRrow]
  have hround := scalarMul_rounding_opNorm c X r cc hXsz hXrow
  have hpert : ‖toReal c • toMatrixF r cc X - toReal c • toMatrixR r cc XR‖ ≤ |toReal c| * εX := by
    rw [← smul_sub, norm_smul, Real.norm_eq_abs]
    exact mul_le_mul_of_nonneg_left hεX (abs_nonneg _)
  calc ‖toMatrixF r cc (Puffer.FloatR.Muon.scalarMul c X) - toReal c • toMatrixR r cc XR‖
      = ‖(toMatrixF r cc (Puffer.FloatR.Muon.scalarMul c X) - toReal c • toMatrixF r cc X)
          + (toReal c • toMatrixF r cc X - toReal c • toMatrixR r cc XR)‖ := by congr 1; abel
    _ ≤ ‖toMatrixF r cc (Puffer.FloatR.Muon.scalarMul c X) - toReal c • toMatrixF r cc X‖
          + ‖toReal c • toMatrixF r cc X - toReal c • toMatrixR r cc XR‖ := norm_add_le _ _
    _ ≤ u64 * |toReal c| * frobRow X r cc + |toReal c| * εX := add_le_add hround hpert

/-! ### The `lincomb3` op — the 3-term Newton–Schulz combine `a·X + b·Y + c·Z`

Like `scalarMul` but with three rounding-carrying terms. The per-entry rounding (`lincomb3Entry_error`) is
massaged into a clean `K·(|a||x|+|b||y|+|c||z|)` form (`lincomb3_entry_le`, `K = u64·(3+3u64+u64²) ≈ 4·u64`),
then aggregated to an operator-norm bound (`lincomb3_agg`, via `opNorm_sq_le_frobenius_sq` + `(p+q+r)² ≤
3(p²+q²+r²)` + ℓ² subadditivity, giving the `√3·K·(|a|‖X‖_F+|b|‖Y‖_F+|c|‖Z‖_F)` rounding term), plus the
3-term scalar perturbation `|a|εX+|b|εY+|c|εZ` (`lincomb3R_toMatrixR` mirror homomorphism + `norm_smul`). -/

theorem frobRow_sq_finsum (X : Mat) (r cc : Nat) :
    (frobRow X r cc) ^ 2 = ∑ i : Fin r, ∑ j : Fin cc, (toReal ((X[i.1]!)[j.1]!)) ^ 2 := by
  have hCnn : 0 ≤ ∑ i ∈ Finset.range r, sumSqR (fun l => toReal ((X[i]!)[l]!)) cc :=
    Finset.sum_nonneg (fun i _ => sumSqR_nonneg _ _)
  rw [frobRow, Real.sq_sqrt hCnn,
    ← Fin.sum_univ_eq_sum_range (fun i => sumSqR (fun l => toReal ((X[i]!)[l]!)) cc) r]
  apply Finset.sum_congr rfl; intro i _
  rw [sumSqR, ← Fin.sum_univ_eq_sum_range (fun j => (toReal ((X[i.1]!)[j]!)) ^ 2) cc]

theorem lincomb3_agg {r cc : Nat} (D : Matrix (Fin r) (Fin cc) ℝ) [Nonempty (Fin cc)]
    (a b c : ℝ) (X Y Z : Mat) (Kl : ℝ) (hKl : 0 ≤ Kl)
    (hentry : ∀ (i : Fin r) (j : Fin cc), |D i j|
        ≤ Kl * (|a| * |toReal ((X[i.1]!)[j.1]!)| + |b| * |toReal ((Y[i.1]!)[j.1]!)| + |c| * |toReal ((Z[i.1]!)[j.1]!)|)) :
    ‖D‖ ≤ Real.sqrt 3 * Kl * (|a| * frobRow X r cc + |b| * frobRow Y r cc + |c| * frobRow Z r cc) := by
  set SX := ∑ i : Fin r, ∑ j : Fin cc, (toReal ((X[i.1]!)[j.1]!)) ^ 2 with hSX
  set SY := ∑ i : Fin r, ∑ j : Fin cc, (toReal ((Y[i.1]!)[j.1]!)) ^ 2 with hSY
  set SZ := ∑ i : Fin r, ∑ j : Fin cc, (toReal ((Z[i.1]!)[j.1]!)) ^ 2 with hSZ
  have hSXnn : 0 ≤ SX := Finset.sum_nonneg (fun i _ => Finset.sum_nonneg (fun j _ => sq_nonneg _))
  have hSYnn : 0 ≤ SY := Finset.sum_nonneg (fun i _ => Finset.sum_nonneg (fun j _ => sq_nonneg _))
  have hSZnn : 0 ≤ SZ := Finset.sum_nonneg (fun i _ => Finset.sum_nonneg (fun j _ => sq_nonneg _))
  have hentry2 : ∀ (i : Fin r) (j : Fin cc), (D i j) ^ 2
      ≤ 3 * Kl ^ 2 * ((a ^ 2 * (toReal ((X[i.1]!)[j.1]!)) ^ 2)
          + (b ^ 2 * (toReal ((Y[i.1]!)[j.1]!)) ^ 2) + (c ^ 2 * (toReal ((Z[i.1]!)[j.1]!)) ^ 2)) := by
    intro i j
    set x := toReal ((X[i.1]!)[j.1]!)
    set y := toReal ((Y[i.1]!)[j.1]!)
    set z := toReal ((Z[i.1]!)[j.1]!)
    set P := |a| * |x| with hPdef
    set Q := |b| * |y| with hQdef
    set R := |c| * |z| with hRdef
    have hb : |D i j| ≤ Kl * (P + Q + R) := hentry i j
    have hMsq : (P + Q + R) ^ 2 ≤ 3 * (P ^ 2 + Q ^ 2 + R ^ 2) := by
      nlinarith [sq_nonneg (P - Q), sq_nonneg (Q - R), sq_nonneg (P - R)]
    have hDsq : (D i j) ^ 2 ≤ (Kl * (P + Q + R)) ^ 2 := by
      rw [← sq_abs (D i j)]; exact pow_le_pow_left₀ (abs_nonneg _) hb 2
    have hPa : P ^ 2 = a ^ 2 * x ^ 2 := by rw [hPdef, mul_pow, sq_abs, sq_abs]
    have hQb : Q ^ 2 = b ^ 2 * y ^ 2 := by rw [hQdef, mul_pow, sq_abs, sq_abs]
    have hRc : R ^ 2 = c ^ 2 * z ^ 2 := by rw [hRdef, mul_pow, sq_abs, sq_abs]
    calc (D i j) ^ 2 ≤ (Kl * (P + Q + R)) ^ 2 := hDsq
      _ = Kl ^ 2 * (P + Q + R) ^ 2 := by ring
      _ ≤ Kl ^ 2 * (3 * (P ^ 2 + Q ^ 2 + R ^ 2)) := mul_le_mul_of_nonneg_left hMsq (sq_nonneg Kl)
      _ = 3 * Kl ^ 2 * (a ^ 2 * x ^ 2 + b ^ 2 * y ^ 2 + c ^ 2 * z ^ 2) := by rw [hPa, hQb, hRc]; ring
  have hsum : (∑ i : Fin r, ∑ j : Fin cc, (D i j) ^ 2)
      ≤ 3 * Kl ^ 2 * (a ^ 2 * SX + b ^ 2 * SY + c ^ 2 * SZ) := by
    refine le_trans (Finset.sum_le_sum (fun i _ => Finset.sum_le_sum (fun j _ => hentry2 i j))) (le_of_eq ?_)
    rw [hSX, hSY, hSZ]
    simp only [mul_add, Finset.sum_add_distrib, Finset.mul_sum, mul_assoc]
  have hsq : ‖D‖ ^ 2 ≤ 3 * Kl ^ 2 * (a ^ 2 * SX + b ^ 2 * SY + c ^ 2 * SZ) :=
    le_trans (Puffer.RL.NewtonSchulzRunnable.opNorm_sq_le_frobenius_sq D) hsum
  have hsub : Real.sqrt (a ^ 2 * SX + b ^ 2 * SY + c ^ 2 * SZ)
      ≤ |a| * Real.sqrt SX + |b| * Real.sqrt SY + |c| * Real.sqrt SZ := by
    rw [show a ^ 2 * SX + b ^ 2 * SY + c ^ 2 * SZ
        = (|a| * Real.sqrt SX) ^ 2 + (|b| * Real.sqrt SY) ^ 2 + (|c| * Real.sqrt SZ) ^ 2 by
      rw [mul_pow, mul_pow, mul_pow, Real.sq_sqrt hSXnn, Real.sq_sqrt hSYnn, Real.sq_sqrt hSZnn, sq_abs, sq_abs, sq_abs]]
    have h1 : 0 ≤ |a| * Real.sqrt SX := by positivity
    have h2 : 0 ≤ |b| * Real.sqrt SY := by positivity
    have h3 : 0 ≤ |c| * Real.sqrt SZ := by positivity
    rw [← Real.sqrt_sq (by positivity : (0:ℝ) ≤ |a| * Real.sqrt SX + |b| * Real.sqrt SY + |c| * Real.sqrt SZ)]
    apply Real.sqrt_le_sqrt
    nlinarith [mul_nonneg h1 h2, mul_nonneg h2 h3, mul_nonneg h1 h3]
  have hfX : Real.sqrt SX = frobRow X r cc := by
    rw [hSX, ← frobRow_sq_finsum X r cc, Real.sqrt_sq (by rw [frobRow]; exact Real.sqrt_nonneg _)]
  have hfY : Real.sqrt SY = frobRow Y r cc := by
    rw [hSY, ← frobRow_sq_finsum Y r cc, Real.sqrt_sq (by rw [frobRow]; exact Real.sqrt_nonneg _)]
  have hfZ : Real.sqrt SZ = frobRow Z r cc := by
    rw [hSZ, ← frobRow_sq_finsum Z r cc, Real.sqrt_sq (by rw [frobRow]; exact Real.sqrt_nonneg _)]
  have hDnn : 0 ≤ ‖D‖ := norm_nonneg _
  calc ‖D‖ = Real.sqrt (‖D‖ ^ 2) := (Real.sqrt_sq hDnn).symm
    _ ≤ Real.sqrt (3 * Kl ^ 2 * (a ^ 2 * SX + b ^ 2 * SY + c ^ 2 * SZ)) := Real.sqrt_le_sqrt hsq
    _ = Real.sqrt 3 * Kl * Real.sqrt (a ^ 2 * SX + b ^ 2 * SY + c ^ 2 * SZ) := by
        rw [show (3 : ℝ) * Kl ^ 2 * (a ^ 2 * SX + b ^ 2 * SY + c ^ 2 * SZ)
            = 3 * (Kl ^ 2 * (a ^ 2 * SX + b ^ 2 * SY + c ^ 2 * SZ)) by ring,
          Real.sqrt_mul (by norm_num), Real.sqrt_mul (sq_nonneg Kl), Real.sqrt_sq hKl]; ring
    _ ≤ Real.sqrt 3 * Kl * (|a| * Real.sqrt SX + |b| * Real.sqrt SY + |c| * Real.sqrt SZ) :=
        mul_le_mul_of_nonneg_left hsub (by positivity)
    _ = Real.sqrt 3 * Kl * (|a| * frobRow X r cc + |b| * frobRow Y r cc + |c| * frobRow Z r cc) := by
        rw [hfX, hfY, hfZ]

theorem lincomb3_entry_le (a x b y c z : Float) :
    lincomb3EntryErrBnd a x b y c z 0 0 0
      ≤ u64 * (3 + 3 * u64 + u64 ^ 2)
        * (|toReal a| * |toReal x| + |toReal b| * |toReal y| + |toReal c| * |toReal z|) := by
  have hu : (0:ℝ) ≤ u64 := u64_pos.le
  set P := |toReal a| * |toReal x| with hP
  set Q := |toReal b| * |toReal y| with hQ
  set R := |toReal c| * |toReal z| with hR
  have hP0 : 0 ≤ P := by rw [hP]; positivity
  have hQ0 : 0 ≤ Q := by rw [hQ]; positivity
  have hR0 : 0 ≤ R := by rw [hR]; positivity
  have hax : |toReal (a * x)| ≤ (1 + u64) * P := by
    rw [hP]; calc |toReal (a * x)| ≤ (1 + u64) * |toReal a * toReal x| := mul_abs_le a x
      _ = (1 + u64) * (|toReal a| * |toReal x|) := by rw [abs_mul]
  have hby : |toReal (b * y)| ≤ (1 + u64) * Q := by
    rw [hQ]; calc |toReal (b * y)| ≤ (1 + u64) * |toReal b * toReal y| := mul_abs_le b y
      _ = (1 + u64) * (|toReal b| * |toReal y|) := by rw [abs_mul]
  have hcz : |toReal (c * z)| ≤ (1 + u64) * R := by
    rw [hR]; calc |toReal (c * z)| ≤ (1 + u64) * |toReal c * toReal z| := mul_abs_le c z
      _ = (1 + u64) * (|toReal c| * |toReal z|) := by rw [abs_mul]
  have haxby : |toReal (a * x + b * y)| ≤ (1 + u64) ^ 2 * (P + Q) := by
    calc |toReal (a * x + b * y)| ≤ (1 + u64) * |toReal (a * x) + toReal (b * y)| := add_abs_le (a*x) (b*y)
      _ ≤ (1 + u64) * (|toReal (a * x)| + |toReal (b * y)|) :=
          mul_le_mul_of_nonneg_left (abs_add_le _ _) (by linarith)
      _ ≤ (1 + u64) * ((1 + u64) * P + (1 + u64) * Q) :=
          mul_le_mul_of_nonneg_left (add_le_add hax hby) (by linarith)
      _ = (1 + u64) ^ 2 * (P + Q) := by ring
  have ht1 : u64 * |toReal (a * x + b * y) + toReal (c * z)|
      ≤ u64 * ((1 + u64) ^ 2 * (P + Q) + (1 + u64) * R) := by
    apply mul_le_mul_of_nonneg_left _ hu
    calc |toReal (a * x + b * y) + toReal (c * z)|
        ≤ |toReal (a * x + b * y)| + |toReal (c * z)| := abs_add_le _ _
      _ ≤ (1 + u64) ^ 2 * (P + Q) + (1 + u64) * R := add_le_add haxby hcz
  have ht2 : matLinEntryErrBnd a x b y 0 0 ≤ u64 * (2 + u64) * (P + Q) := by
    rw [hP, hQ]; exact matLinEntryErrBnd_le a x b y
  have ht3 : u64 * |toReal c * toReal z| = u64 * R := by rw [hR, abs_mul]
  unfold lincomb3EntryErrBnd
  simp only [mul_zero, add_zero]
  nlinarith [ht1, ht2, ht3, hP0, hQ0, hR0, hu, sq_nonneg u64,
    mul_nonneg hu hR0, mul_nonneg (mul_nonneg hu hu) hR0]

theorem lincomb3_error_opNorm (a : Float) (X : Mat) (b : Float) (Y : Mat) (c : Float) (Z : Mat)
    (XR YR ZR : MatR) (r cc : Nat) [Nonempty (Fin cc)] (εX εY εZ : ℝ)
    (hXsz : X.size = r) (hXrow : ∀ i, i < r → (X[i]!).size = cc)
    (hXRsz : XR.size = r) (hXRrow : ∀ i, i < r → (XR[i]!).size = cc)
    (hεX : ‖toMatrixF r cc X - toMatrixR r cc XR‖ ≤ εX)
    (hεY : ‖toMatrixF r cc Y - toMatrixR r cc YR‖ ≤ εY)
    (hεZ : ‖toMatrixF r cc Z - toMatrixR r cc ZR‖ ≤ εZ) :
    ‖toMatrixF r cc (Puffer.FloatR.Muon.lincomb3 a X b Y c Z)
        - toMatrixR r cc (lincomb3R (toReal a) XR (toReal b) YR (toReal c) ZR)‖
      ≤ Real.sqrt 3 * (u64 * (3 + 3 * u64 + u64 ^ 2))
          * (|toReal a| * frobRow X r cc + |toReal b| * frobRow Y r cc + |toReal c| * frobRow Z r cc)
        + (|toReal a| * εX + |toReal b| * εY + |toReal c| * εZ) := by
  set S := toReal a • toMatrixF r cc X + toReal b • toMatrixF r cc Y + toReal c • toMatrixF r cc Z with hS
  set Kl := u64 * (3 + 3 * u64 + u64 ^ 2) with hKldef
  have hKl0 : 0 ≤ Kl := by rw [hKldef]; exact mul_nonneg u64_pos.le (by nlinarith [u64_pos.le, sq_nonneg u64])
  -- rounding
  have hround : ‖toMatrixF r cc (Puffer.FloatR.Muon.lincomb3 a X b Y c Z) - S‖
      ≤ Real.sqrt 3 * Kl * (|toReal a| * frobRow X r cc + |toReal b| * frobRow Y r cc + |toReal c| * frobRow Z r cc) := by
    apply lincomb3_agg _ (toReal a) (toReal b) (toReal c) X Y Z Kl hKl0
    intro i j
    have hDval : (toMatrixF r cc (Puffer.FloatR.Muon.lincomb3 a X b Y c Z) - S) i j
        = toReal (((Puffer.FloatR.Muon.lincomb3 a X b Y c Z)[i.1]!)[j.1]!)
          - (toReal a * toReal ((X[i.1]!)[j.1]!) + toReal b * toReal ((Y[i.1]!)[j.1]!)
             + toReal c * toReal ((Z[i.1]!)[j.1]!)) := by
      simp only [hS, Matrix.sub_apply, Matrix.add_apply, Matrix.smul_apply, toMatrixF, Matrix.of_apply, smul_eq_mul]
    rw [hDval, lincomb3_getElem a X b Y c Z i.1 j.1 (by rw [hXsz]; exact i.2) (by rw [hXrow i.1 i.2]; exact j.2)]
    refine le_trans (lincomb3Entry_error a ((X[i.1]!)[j.1]!) b ((Y[i.1]!)[j.1]!) c ((Z[i.1]!)[j.1]!)
      (toReal ((X[i.1]!)[j.1]!)) (toReal ((Y[i.1]!)[j.1]!)) (toReal ((Z[i.1]!)[j.1]!)) 0 0 0
      (by simp) (by simp) (by simp)) ?_
    exact lincomb3_entry_le a ((X[i.1]!)[j.1]!) b ((Y[i.1]!)[j.1]!) c ((Z[i.1]!)[j.1]!)
  -- perturbation
  have hpert : ‖S - toMatrixR r cc (lincomb3R (toReal a) XR (toReal b) YR (toReal c) ZR)‖
      ≤ |toReal a| * εX + |toReal b| * εY + |toReal c| * εZ := by
    rw [hS, lincomb3R_toMatrixR (toReal a) (toReal b) (toReal c) XR YR ZR r cc hXRsz hXRrow]
    have hsplit : (toReal a • toMatrixF r cc X + toReal b • toMatrixF r cc Y + toReal c • toMatrixF r cc Z)
        - (toReal a • toMatrixR r cc XR + toReal b • toMatrixR r cc YR + toReal c • toMatrixR r cc ZR)
        = toReal a • (toMatrixF r cc X - toMatrixR r cc XR) + toReal b • (toMatrixF r cc Y - toMatrixR r cc YR)
          + toReal c • (toMatrixF r cc Z - toMatrixR r cc ZR) := by simp only [smul_sub]; abel
    rw [hsplit]
    refine le_trans (norm_add_le _ _) ?_
    refine le_trans (add_le_add (norm_add_le _ _) le_rfl) ?_
    rw [norm_smul, norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs, Real.norm_eq_abs]
    exact add_le_add (add_le_add (mul_le_mul_of_nonneg_left hεX (abs_nonneg _))
      (mul_le_mul_of_nonneg_left hεY (abs_nonneg _))) (mul_le_mul_of_nonneg_left hεZ (abs_nonneg _))
  calc ‖toMatrixF r cc (Puffer.FloatR.Muon.lincomb3 a X b Y c Z)
        - toMatrixR r cc (lincomb3R (toReal a) XR (toReal b) YR (toReal c) ZR)‖
      = ‖(toMatrixF r cc (Puffer.FloatR.Muon.lincomb3 a X b Y c Z) - S)
          + (S - toMatrixR r cc (lincomb3R (toReal a) XR (toReal b) YR (toReal c) ZR))‖ := by congr 1; abel
    _ ≤ ‖toMatrixF r cc (Puffer.FloatR.Muon.lincomb3 a X b Y c Z) - S‖
          + ‖S - toMatrixR r cc (lincomb3R (toReal a) XR (toReal b) YR (toReal c) ZR)‖ := norm_add_le _ _
    _ ≤ Real.sqrt 3 * Kl * (|toReal a| * frobRow X r cc + |toReal b| * frobRow Y r cc + |toReal c| * frobRow Z r cc)
          + (|toReal a| * εX + |toReal b| * εY + |toReal c| * εZ) := add_le_add hround hpert

/-! ### Assembling the four ops into one `nsIter` step (`r ≤ c` branch)

`nsIter X = lincomb3 a X b AX c AAX` with `A = X·Xᵀ` (Gram), `AX = A·X`, `AAX = A·AX`. The step error is
composed modularly: `gram_error_opNorm` gives `A`'s error from `X`'s (transpose isometry + one matmul);
`matmul_error_opNorm` then gives `AX`'s error (from `A`, `X`) and `AAX`'s (from `A`, `AX`); and
`nsIter_error_opNorm` assembles the final `lincomb3` combine. The whole step error is ADDITIVE in `εX`, so one
`nsIter` step is a Lipschitz map on the error with polynomial (frobenius-magnitude) coefficients — the
qualitative property that makes the 5-iteration fold error polynomial rather than doubly-exponential. -/

/-- **The Gram-matrix error atom `A = X·Xᵀ`.** Composes the transpose isometry (`transpose_error_opNorm`, the
    input error `εX` passes through) with one matmul (`matmul_error_opNorm`, inner dim `c`): the error of the
    Gram matrix `X·Xᵀ` is ADDITIVE in `εX`. The first composition of the `nsIter` step. -/
theorem gram_error_opNorm (X : Mat) (XR : MatR) (r c : Nat) [Nonempty (Fin r)] (εX : ℝ)
    (hXsz : X.size = r) (hXrow : ∀ i, i < r → (X[i]!).size = c)
    (hXRsz : XR.size = r) (hXRrow : ∀ i, i < r → (XR[i]!).size = c)
    (hr : 0 < r) (hc : 0 < c)
    (hεX : ‖toMatrixF r c X - toMatrixR r c XR‖ ≤ εX) :
    ‖toMatrixF r r (matmul X (Puffer.FloatR.Muon.transpose X))
        - toMatrixR r r (matmulR XR (transposeR XR))‖
      ≤ ((2:ℝ) + u64) * (((1:ℝ) + u64) ^ c - 1)
          * (frobRow X r c * frobCol (Puffer.FloatR.Muon.transpose X) r c)
        + εX * ‖toMatrixF r c X‖ + ‖toMatrixR r c XR‖ * εX := by
  have hXne : X ≠ #[] := by intro h; rw [h] at hXsz; simp at hXsz; omega
  have hXRne : XR ≠ #[] := by intro h; rw [h] at hXRsz; simp at hXRsz; omega
  -- transpose error
  have htXt := transpose_error_opNorm X XR r c εX hXsz hXrow hXRsz hXRrow hr hεX
  -- Float shape of transpose X: c×r
  have htsz : (Puffer.FloatR.Muon.transpose X).size = c := by
    rw [transpose_size, if_neg hXne, hXrow 0 hr]
  have htne : Puffer.FloatR.Muon.transpose X ≠ #[] := by
    intro h; rw [h] at htsz; simp at htsz; omega
  have htrow : ∀ i, i < c → ((Puffer.FloatR.Muon.transpose X)[i]!).size = r := by
    intro i hi; rw [transpose_rowSize X i (by rw [if_neg hXne, hXrow 0 hr]; exact hi)]; exact hXsz
  -- mirror shape of transposeR XR: c×r
  have hTRsz : (transposeR XR).size = c := by
    rw [transposeR_size, if_neg (by rw [hXRsz]; omega), hXRrow 0 hr]
  have hTRrow : ∀ i, i < c → ((transposeR XR)[i]!).size = r := by
    intro i hi; rw [transposeR_rowSize XR i (by rw [if_neg (by rw [hXRsz]; omega), hXRrow 0 hr]; exact hi)]; exact hXRsz
  -- dimension expressions for matmul_error_opNorm
  have hAk : (if X = #[] then 0 else (X[0]!).size) = c := by rw [if_neg hXne, hXrow 0 hr]
  have hBn : (if Puffer.FloatR.Muon.transpose X = #[] then 0
      else ((Puffer.FloatR.Muon.transpose X)[0]!).size) = r := by rw [if_neg htne, htrow 0 hc]
  haveI : Nonempty (Fin c) := ⟨⟨0, hc⟩⟩
  have hiso : ‖toMatrixF c r (Puffer.FloatR.Muon.transpose X)‖ = ‖toMatrixF r c X‖ := by
    rw [transpose_toMatrixF X r c hXsz hXrow hr, ← conjTranspose_eq_transpose_of_trivial, l2_opNorm_conjTranspose]
  have h := matmul_error_opNorm X (Puffer.FloatR.Muon.transpose X) XR (transposeR XR) r c r εX εX
    hXsz hAk hBn hXRsz hXRrow hTRsz hTRrow hr hc hεX htXt
  rw [hiso] at h; exact h

/-- **One Newton–Schulz step error, assembled (`r ≤ c` branch).** `nsIter X = lincomb3 a X b AX c AAX` with
    `A = X·Xᵀ`, `AX = A·X`, `AAX = A·AX`. Given the input error `εX` and the two matmul-chain errors `εAX`,
    `εAAX` (each an application of `matmul_error_opNorm`; `εA` for `A` from `gram_error_opNorm`), the step error
    is the `lincomb3` combine — ADDITIVE in `εX`, `εAX`, `εAAX`. -/
theorem nsIter_error_opNorm (X : Mat) (XR : MatR) (r c : Nat) [Nonempty (Fin c)] (a b cc : Float)
    (εX εAX εAAX : ℝ)
    (hXsz : X.size = r) (hXrow : ∀ i, i < r → (X[i]!).size = c)
    (hXRsz : XR.size = r) (hXRrow : ∀ i, i < r → (XR[i]!).size = c)
    (hr : 0 < r) (hrc : r ≤ c)
    (hεX : ‖toMatrixF r c X - toMatrixR r c XR‖ ≤ εX)
    (hεAX : ‖toMatrixF r c (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X)
        - toMatrixR r c (matmulR (matmulR XR (transposeR XR)) XR)‖ ≤ εAX)
    (hεAAX : ‖toMatrixF r c (matmul (matmul X (Puffer.FloatR.Muon.transpose X))
          (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X))
        - toMatrixR r c (matmulR (matmulR XR (transposeR XR)) (matmulR (matmulR XR (transposeR XR)) XR))‖ ≤ εAAX) :
    ‖toMatrixF r c (nsIter X (a, b, cc)) - toMatrixR r c (nsIterR XR (a, b, cc))‖
      ≤ Real.sqrt 3 * (u64 * (3 + 3 * u64 + u64 ^ 2))
          * (|toReal a| * frobRow X r c
             + |toReal b| * frobRow (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X) r c
             + |toReal cc| * frobRow (matmul (matmul X (Puffer.FloatR.Muon.transpose X))
                 (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X)) r c)
        + (|toReal a| * εX + |toReal b| * εAX + |toReal cc| * εAAX) := by
  have hnsIter : nsIter X (a, b, cc)
      = Puffer.FloatR.Muon.lincomb3 a X b (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X) cc
          (matmul (matmul X (Puffer.FloatR.Muon.transpose X))
            (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X)) := by
    dsimp only [nsIter]; rw [if_pos (by rw [hXsz, hXrow 0 hr]; exact hrc)]
  have hnsIterR : nsIterR XR (a, b, cc)
      = Puffer.RL.NewtonSchulzError.lincomb3R (toReal a) XR (toReal b) (matmulR (matmulR XR (transposeR XR)) XR)
          (toReal cc) (matmulR (matmulR XR (transposeR XR)) (matmulR (matmulR XR (transposeR XR)) XR)) := by
    dsimp only [nsIterR]; rw [if_pos (by rw [hXRsz, hXRrow 0 hr]; exact hrc)]
  rw [hnsIter, hnsIterR]
  exact lincomb3_error_opNorm a X b (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X) cc
    (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X))
    XR (matmulR (matmulR XR (transposeR XR)) XR)
    (matmulR (matmulR XR (transposeR XR)) (matmulR (matmulR XR (transposeR XR)) XR)) r c εX εAX εAAX
    hXsz hXrow hXRsz hXRrow hεX hεAX hεAAX

/-! ### The symmetric `r > c` branch of the `nsIter` step

Identical structure to the `r ≤ c` branch with the Gram on the other side: `A = Xᵀ·X` (c×c), `XA = X·A`,
`nsIter X = lincomb3 a X b XA c (XA·A)`. `gram_gt_error_opNorm` is the `Xᵀ·X` Gram atom; `nsIter_error_opNorm_gt`
assembles the `lincomb3` combine (`if_neg`, `c < r`). -/

/-- **The Gram-matrix error atom `A = Xᵀ·X` (`r > c` branch).** Mirrors `gram_error_opNorm` with operands
    swapped: `transpose X` (c×r) on the left, `X` (r×c) on the right, inner dim `r`, output `c×c`. -/
theorem gram_gt_error_opNorm (X : Mat) (XR : MatR) (r c : Nat) [Nonempty (Fin c)] (εX : ℝ)
    (hXsz : X.size = r) (hXrow : ∀ i, i < r → (X[i]!).size = c)
    (hXRsz : XR.size = r) (hXRrow : ∀ i, i < r → (XR[i]!).size = c)
    (hr : 0 < r) (hc : 0 < c)
    (hεX : ‖toMatrixF r c X - toMatrixR r c XR‖ ≤ εX) :
    ‖toMatrixF c c (matmul (Puffer.FloatR.Muon.transpose X) X)
        - toMatrixR c c (matmulR (transposeR XR) XR)‖
      ≤ ((2:ℝ) + u64) * (((1:ℝ) + u64) ^ r - 1)
          * (frobRow (Puffer.FloatR.Muon.transpose X) c r * frobCol X c r)
        + εX * ‖toMatrixF r c X‖ + ‖toMatrixR c r (transposeR XR)‖ * εX := by
  have hXne : X ≠ #[] := by intro h; rw [h] at hXsz; simp at hXsz; omega
  have htXt := transpose_error_opNorm X XR r c εX hXsz hXrow hXRsz hXRrow hr hεX
  have htsz : (Puffer.FloatR.Muon.transpose X).size = c := by rw [transpose_size, if_neg hXne, hXrow 0 hr]
  have htne : Puffer.FloatR.Muon.transpose X ≠ #[] := by intro h; rw [h] at htsz; simp at htsz; omega
  have htrow : ∀ i, i < c → ((Puffer.FloatR.Muon.transpose X)[i]!).size = r := by
    intro i hi; rw [transpose_rowSize X i (by rw [if_neg hXne, hXrow 0 hr]; exact hi)]; exact hXsz
  have hTRsz : (transposeR XR).size = c := by rw [transposeR_size, if_neg (by rw [hXRsz]; omega), hXRrow 0 hr]
  have hTRrow : ∀ i, i < c → ((transposeR XR)[i]!).size = r := by
    intro i hi; rw [transposeR_rowSize XR i (by rw [if_neg (by rw [hXRsz]; omega), hXRrow 0 hr]; exact hi)]; exact hXRsz
  have hAk : (if Puffer.FloatR.Muon.transpose X = #[] then 0
      else ((Puffer.FloatR.Muon.transpose X)[0]!).size) = r := by rw [if_neg htne, htrow 0 hc]
  have hBn : (if X = #[] then 0 else (X[0]!).size) = c := by rw [if_neg hXne, hXrow 0 hr]
  haveI : Nonempty (Fin r) := ⟨⟨0, hr⟩⟩
  exact matmul_error_opNorm (Puffer.FloatR.Muon.transpose X) X (transposeR XR) XR c r c εX εX
    htsz hAk hBn hTRsz hTRrow hXRsz hXRrow hc hr htXt hεX

/-- **One Newton–Schulz step error, assembled (`r > c` branch).** `nsIter X = lincomb3 a X b XA c XAA` with
    `A = Xᵀ·X`, `XA = X·A`, `XAA = XA·A`. Given `εX` and the two matmul-chain errors `εXA`, `εXAA`, the step
    error is the `lincomb3` combine — ADDITIVE in `εX`, `εXA`, `εXAA`. -/
theorem nsIter_error_opNorm_gt (X : Mat) (XR : MatR) (r c : Nat) [Nonempty (Fin c)] (a b cc : Float)
    (εX εXA εXAA : ℝ)
    (hXsz : X.size = r) (hXrow : ∀ i, i < r → (X[i]!).size = c)
    (hXRsz : XR.size = r) (hXRrow : ∀ i, i < r → (XR[i]!).size = c)
    (hr : 0 < r) (hgt : c < r)
    (hεX : ‖toMatrixF r c X - toMatrixR r c XR‖ ≤ εX)
    (hεXA : ‖toMatrixF r c (matmul X (matmul (Puffer.FloatR.Muon.transpose X) X))
        - toMatrixR r c (matmulR XR (matmulR (transposeR XR) XR))‖ ≤ εXA)
    (hεXAA : ‖toMatrixF r c (matmul (matmul X (matmul (Puffer.FloatR.Muon.transpose X) X))
          (matmul (Puffer.FloatR.Muon.transpose X) X))
        - toMatrixR r c (matmulR (matmulR XR (matmulR (transposeR XR) XR)) (matmulR (transposeR XR) XR))‖ ≤ εXAA) :
    ‖toMatrixF r c (nsIter X (a, b, cc)) - toMatrixR r c (nsIterR XR (a, b, cc))‖
      ≤ Real.sqrt 3 * (u64 * (3 + 3 * u64 + u64 ^ 2))
          * (|toReal a| * frobRow X r c
             + |toReal b| * frobRow (matmul X (matmul (Puffer.FloatR.Muon.transpose X) X)) r c
             + |toReal cc| * frobRow (matmul (matmul X (matmul (Puffer.FloatR.Muon.transpose X) X))
                 (matmul (Puffer.FloatR.Muon.transpose X) X)) r c)
        + (|toReal a| * εX + |toReal b| * εXA + |toReal cc| * εXAA) := by
  have hnsIter : nsIter X (a, b, cc)
      = Puffer.FloatR.Muon.lincomb3 a X b (matmul X (matmul (Puffer.FloatR.Muon.transpose X) X)) cc
          (matmul (matmul X (matmul (Puffer.FloatR.Muon.transpose X) X))
            (matmul (Puffer.FloatR.Muon.transpose X) X)) := by
    dsimp only [nsIter]; rw [if_neg (by rw [hXsz, hXrow 0 hr]; omega)]
  have hnsIterR : nsIterR XR (a, b, cc)
      = Puffer.RL.NewtonSchulzError.lincomb3R (toReal a) XR (toReal b) (matmulR XR (matmulR (transposeR XR) XR))
          (toReal cc) (matmulR (matmulR XR (matmulR (transposeR XR) XR)) (matmulR (transposeR XR) XR)) := by
    dsimp only [nsIterR]; rw [if_neg (by rw [hXRsz, hXRrow 0 hr]; omega)]
  rw [hnsIter, hnsIterR]
  exact lincomb3_error_opNorm a X b (matmul X (matmul (Puffer.FloatR.Muon.transpose X) X)) cc
    (matmul (matmul X (matmul (Puffer.FloatR.Muon.transpose X) X)) (matmul (Puffer.FloatR.Muon.transpose X) X))
    XR (matmulR XR (matmulR (transposeR XR) XR))
    (matmulR (matmulR XR (matmulR (transposeR XR) XR)) (matmulR (transposeR XR) XR)) r c εX εXA εXAA
    hXsz hXrow hXRsz hXRrow hεX hεXA hεXAA

/-! ### The magnitude/error BOOTSTRAP — controlling the Float Frobenius magnitude

The per-step error bound's coefficients are the Frobenius magnitudes `frobRow`/`frobCol` of the iterate and
its intermediates. For a UNIFORM per-step bound (the `hstep` hypothesis of `newtonSchulz_opNorm_poly`), those
must stay bounded across the fold. `frobRow_le_opNorm` bounds Frobenius by `√dim · operator` (the reverse of
`opNorm_le_frobRow`), and `frobRow_le_mirror_add_err` bounds it by `√dim·(‖mirror‖₂ + ε)` — so the proven
mirror bound (`≤ √1.3131`, `nsIterR_comp_normsq` / the per-step `muon_comp_step` chain) keeps every Frobenius
magnitude `O(√dim)` as long as the error stays bounded. This is the magnitude half of the bootstrap; the
remaining coupling is the region-invariant fold threading magnitude and error jointly. -/

/-- **Frobenius ≤ √dim · operator norm.** `frobRow X r c ≤ √c · ‖toMatrixF X‖₂` — the reverse of
    `opNorm_le_frobRow`, via `‖A‖_F² = trace(AᴴA) = ∑ⱼ λⱼ(AᴴA) ≤ c·‖A‖₂²` (each Gram eigenvalue `≤ ‖A‖²`,
    `gram_eig_le_opNorm_sq`; trace = ∑ eigenvalues = ∑ᵢⱼ Aᵢⱼ²). Bounds the Float Frobenius magnitude by the
    operator norm, which the mirror bound keeps O(1). -/
theorem frobRow_le_opNorm (X : Mat) (r c : Nat) [Nonempty (Fin c)] :
    frobRow X r c ≤ Real.sqrt c * ‖toMatrixF r c X‖ := by
  set A := toMatrixF r c X with hA
  have hHerm := isHermitian_conjTranspose_mul_self A
  have htrace : (Aᴴ * A).trace = ∑ i, ∑ j, (A i j) ^ 2 := by
    simp only [Matrix.trace, Matrix.diag_apply, Matrix.mul_apply, Matrix.conjTranspose_apply, star_trivial]
    rw [Finset.sum_comm]; apply Finset.sum_congr rfl; intro i _; apply Finset.sum_congr rfl; intro j _; rw [sq]
  have hsum : ∑ j, hHerm.eigenvalues j = (Aᴴ * A).trace := by rw [hHerm.trace_eq_sum_eigenvalues]; simp
  have hfrobsq : (frobRow X r c) ^ 2 ≤ (c : ℝ) * ‖A‖ ^ 2 := by
    rw [frobRow_sq_finsum X r c]
    have hconv : (∑ i : Fin r, ∑ j : Fin c, (toReal ((X[i.1]!)[j.1]!)) ^ 2) = ∑ i, ∑ j, (A i j) ^ 2 := by
      apply Finset.sum_congr rfl; intro i _; apply Finset.sum_congr rfl; intro j _
      simp only [hA, toMatrixF, Matrix.of_apply]
    rw [hconv]
    calc ∑ i, ∑ j, (A i j) ^ 2 = (Aᴴ * A).trace := htrace.symm
      _ = ∑ j, hHerm.eigenvalues j := hsum.symm
      _ ≤ ∑ _j : Fin c, ‖A‖ ^ 2 := Finset.sum_le_sum (fun j _ => gram_eig_le_opNorm_sq A j)
      _ = (c : ℝ) * ‖A‖ ^ 2 := by rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hcnn : (0:ℝ) ≤ (c : ℝ) := Nat.cast_nonneg c
  calc frobRow X r c = Real.sqrt ((frobRow X r c) ^ 2) := (Real.sqrt_sq (by rw [frobRow]; exact Real.sqrt_nonneg _)).symm
    _ ≤ Real.sqrt ((c : ℝ) * ‖A‖ ^ 2) := Real.sqrt_le_sqrt hfrobsq
    _ = Real.sqrt c * ‖A‖ := by rw [Real.sqrt_mul hcnn, Real.sqrt_sq (norm_nonneg _)]

/-- **The Float Frobenius magnitude is controlled by the mirror + error.** `frobRow X r c ≤ √c·(‖toMatrixR
    XR‖₂ + ε)` — combines `frobRow_le_opNorm` with the triangle `‖toMatrixF X‖ ≤ ‖toMatrixR XR‖ + ‖diff‖`.
    Since the mirror operator norm is proven O(1) (`≤ √1.3131`), this keeps every per-step Frobenius magnitude
    O(√dim) as long as the error `ε` stays bounded — the magnitude half of the bootstrap. -/
theorem frobRow_le_mirror_add_err (X : Mat) (XR : MatR) (r c : Nat) [Nonempty (Fin c)] (ε : ℝ)
    (hε : ‖toMatrixF r c X - toMatrixR r c XR‖ ≤ ε) :
    frobRow X r c ≤ Real.sqrt c * (‖toMatrixR r c XR‖ + ε) := by
  refine le_trans (frobRow_le_opNorm X r c) ?_
  apply mul_le_mul_of_nonneg_left _ (Real.sqrt_nonneg _)
  calc ‖toMatrixF r c X‖ = ‖toMatrixR r c XR + (toMatrixF r c X - toMatrixR r c XR)‖ := by rw [add_sub_cancel]
    _ ≤ ‖toMatrixR r c XR‖ + ‖toMatrixF r c X - toMatrixR r c XR‖ := norm_add_le _ _
    _ ≤ ‖toMatrixR r c XR‖ + ε := by linarith

/-- **Frobenius (column form) ≤ √dim · operator norm.** `frobCol B n k ≤ √n · ‖toMatrixF k n B‖₂` — the
    column-form companion of `frobRow_le_opNorm` (same `trace(AᴴA) = ∑λ ≤ dim·‖·‖²` argument, over the `n`
    eigenvalues of the `n×n` Gram `(toMatrixF k n B)ᴴ(toMatrixF k n B)`). -/
theorem frobCol_le_opNorm (B : Mat) (k n : Nat) [Nonempty (Fin n)] :
    frobCol B n k ≤ Real.sqrt n * ‖toMatrixF k n B‖ := by
  set A := toMatrixF k n B with hA
  have hHerm := isHermitian_conjTranspose_mul_self A
  have htrace : (Aᴴ * A).trace = ∑ i, ∑ j, (A i j) ^ 2 := by
    simp only [Matrix.trace, Matrix.diag_apply, Matrix.mul_apply, Matrix.conjTranspose_apply, star_trivial]
    rw [Finset.sum_comm]; apply Finset.sum_congr rfl; intro i _; apply Finset.sum_congr rfl; intro j _; rw [sq]
  have hsum : ∑ j, hHerm.eigenvalues j = (Aᴴ * A).trace := by rw [hHerm.trace_eq_sum_eigenvalues]; simp
  have hconv : (∑ i, ∑ j, (A i j) ^ 2) = ∑ j ∈ Finset.range n, sumSqR (fun l => toReal ((B[l]!)[j]!)) k := by
    rw [Finset.sum_comm, ← Fin.sum_univ_eq_sum_range (fun j => sumSqR (fun l => toReal ((B[l]!)[j]!)) k) n]
    apply Finset.sum_congr rfl; intro j _
    rw [sumSqR, ← Fin.sum_univ_eq_sum_range (fun l => (toReal ((B[l]!)[j.1]!)) ^ 2) k]
    apply Finset.sum_congr rfl; intro l _; simp only [hA, toMatrixF, Matrix.of_apply]
  have hfrobsq : (frobCol B n k) ^ 2 ≤ (n : ℝ) * ‖A‖ ^ 2 := by
    rw [frobCol, Real.sq_sqrt (Finset.sum_nonneg (fun j _ => sumSqR_nonneg _ _)), ← hconv]
    calc ∑ i, ∑ j, (A i j) ^ 2 = (Aᴴ * A).trace := htrace.symm
      _ = ∑ j, hHerm.eigenvalues j := hsum.symm
      _ ≤ ∑ _j : Fin n, ‖A‖ ^ 2 := Finset.sum_le_sum (fun j _ => gram_eig_le_opNorm_sq A j)
      _ = (n : ℝ) * ‖A‖ ^ 2 := by rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  calc frobCol B n k = Real.sqrt ((frobCol B n k) ^ 2) := (Real.sqrt_sq (by rw [frobCol]; exact Real.sqrt_nonneg _)).symm
    _ ≤ Real.sqrt ((n : ℝ) * ‖A‖ ^ 2) := Real.sqrt_le_sqrt hfrobsq
    _ = Real.sqrt n * ‖A‖ := by rw [Real.sqrt_mul (Nat.cast_nonneg n), Real.sqrt_sq (norm_nonneg _)]

/-- **The Float Frobenius (column form) magnitude from mirror + error.** `frobCol B n k ≤ √n·(‖toMatrixR k n
    BR‖₂ + ε)` — the column-form companion of `frobRow_le_mirror_add_err`. Together they discharge every
    `frobRow`/`frobCol` hypothesis of `nsIter_step_affine` from the mirror operator norm + error. -/
theorem frobCol_le_mirror_add_err (B : Mat) (BR : MatR) (k n : Nat) [Nonempty (Fin n)] (ε : ℝ)
    (hε : ‖toMatrixF k n B - toMatrixR k n BR‖ ≤ ε) :
    frobCol B n k ≤ Real.sqrt n * (‖toMatrixR k n BR‖ + ε) := by
  refine le_trans (frobCol_le_opNorm B k n) ?_
  apply mul_le_mul_of_nonneg_left _ (Real.sqrt_nonneg _)
  calc ‖toMatrixF k n B‖ = ‖toMatrixR k n BR + (toMatrixF k n B - toMatrixR k n BR)‖ := by rw [add_sub_cancel]
    _ ≤ ‖toMatrixR k n BR‖ + ‖toMatrixF k n B - toMatrixR k n BR‖ := norm_add_le _ _
    _ ≤ ‖toMatrixR k n BR‖ + ε := by linarith

/-- **The Float OPERATOR-norm magnitude from mirror + error** — the `√dim`-free companion of
    `frobCol_le_mirror_add_err`. `‖toMatrixF k n B‖₂ ≤ ‖toMatrixR k n BR‖₂ + ε`, pure triangle inequality (no
    `√n` factor). This is the magnitude used in the PERTURBATION (Lipschitz) coefficient of the matmul error —
    keeping it in operator norm (not Frobenius) is what makes the per-step Lipschitz factor `L` tight (`3‖X‖²`,
    `5‖X‖⁴` instead of the `√dim`-inflated Frobenius versions). The `β` ROUNDING term still uses Frobenius. -/
theorem opNorm_le_mirror_add_err (B : Mat) (BR : MatR) (k n : Nat) (ε : ℝ)
    (hε : ‖toMatrixF k n B - toMatrixR k n BR‖ ≤ ε) :
    ‖toMatrixF k n B‖ ≤ ‖toMatrixR k n BR‖ + ε := by
  calc ‖toMatrixF k n B‖ = ‖toMatrixR k n BR + (toMatrixF k n B - toMatrixR k n BR)‖ := by rw [add_sub_cancel]
    _ ≤ ‖toMatrixR k n BR‖ + ‖toMatrixF k n B - toMatrixR k n BR‖ := norm_add_le _ _
    _ ≤ ‖toMatrixR k n BR‖ + ε := by linarith

/-! ### Per-step composition atoms — the step error is AFFINE in `εX`

The reusable atoms that compose the four op errors into a single per-step affine bound `ε' ≤ L·ε + C` (the
form the region fold's `hstep` consumes). `gram_step_affine`: the Gram `X·Xᵀ` error, affine in `εX` given
magnitude bounds. `matmul_step_affine`: one matmul in the chain — two affine-in-`εX` operand errors compose to
an affine output error. Chaining `gram_step_affine` + two `matmul_step_affine` + `nsIter_error_opNorm` gives
the whole `nsIter` step as `ε' ≤ L·ε + C` with `L`, `C` polynomial in the (bounded) magnitudes. -/

/-- **The Gram step, AFFINE.** `gram_error_opNorm` bounds `A = X·Xᵀ`'s error; the LIPSCHITZ coefficient uses the
    OPERATOR-norm magnitude of `X` (`‖toMatrixF X‖ ≤ MFX`, tight — no `√dim`) and `‖mirror X‖ ≤ MX`, while the
    `β` ROUNDING term keeps Frobenius (`frobRow X ≤ FX`, `frobCol Xᵀ ≤ FXt`). Affine in `εX`:
    `≤ (MFX + MX)·εX + β(c)·FX·FXt`. The `MFX + MX ≈ 2‖X‖` Lipschitz coefficient (vs the `√dim`-inflated
    Frobenius `FXt + MX`) is the gram link of the tight `L`. -/
theorem gram_step_affine (X : Mat) (XR : MatR) (r c : Nat) [Nonempty (Fin c)]
    (εX FX FXt MFX MX : ℝ)
    (hXsz : X.size = r) (hXrow : ∀ i, i < r → (X[i]!).size = c)
    (hXRsz : XR.size = r) (hXRrow : ∀ i, i < r → (XR[i]!).size = c)
    (hr : 0 < r) (hc : 0 < c)
    (hεX : ‖toMatrixF r c X - toMatrixR r c XR‖ ≤ εX)
    (hFX : frobRow X r c ≤ FX) (hFXt : frobCol (Puffer.FloatR.Muon.transpose X) r c ≤ FXt)
    (hMFX : ‖toMatrixF r c X‖ ≤ MFX) (hMX : ‖toMatrixR r c XR‖ ≤ MX)
    (hεX0 : 0 ≤ εX) (hFX0 : 0 ≤ FX) :
    ‖toMatrixF r r (matmul X (Puffer.FloatR.Muon.transpose X))
        - toMatrixR r r (matmulR XR (transposeR XR))‖
      ≤ (MFX + MX) * εX + ((2:ℝ) + u64) * (((1:ℝ) + u64) ^ c - 1) * FX * FXt := by
  haveI : Nonempty (Fin r) := ⟨⟨0, hr⟩⟩
  have hbc : 0 ≤ ((2:ℝ) + u64) * (((1:ℝ) + u64) ^ c - 1) := by
    have : (1:ℝ) ≤ (1 + u64) ^ c := one_le_pow₀ (by nlinarith [u64_pos.le])
    exact mul_nonneg (by nlinarith [u64_pos.le]) (by linarith)
  refine le_trans (gram_error_opNorm X XR r c εX hXsz hXrow hXRsz hXRrow hr hc hεX) ?_
  have hFXFXt : frobRow X r c * frobCol (Puffer.FloatR.Muon.transpose X) r c ≤ FX * FXt :=
    mul_le_mul hFX hFXt (le_trans (Real.sqrt_nonneg _) (le_of_eq (by rw [frobCol]))) hFX0
  nlinarith [hFXFXt, hMX, hεX0, hbc, mul_le_mul_of_nonneg_left hMFX hεX0,
    mul_le_mul_of_nonneg_right hMX hεX0, mul_le_mul_of_nonneg_left hFXFXt hbc]

/-- **One matmul in the chain, AFFINE composition.** For `matmul A B` where both operands' errors are affine
    in `εX` (`‖diffA‖ ≤ LA·εX+CA`, `‖diffB‖ ≤ LB·εX+CB`), the LIPSCHITZ side uses the OPERATOR-norm magnitude of
    `B` (`‖toMatrixF B‖ ≤ MB`, tight — no `√dim`) and `‖mirror A‖ ≤ MA`, while the `β` ROUNDING term keeps the
    Frobenius `frobRow A ≤ FA`, `frobCol B ≤ FCB`. The matmul error is again affine in `εX`:
    `≤ (LA·MB + MA·LB)·εX + (β(k)·FA·FCB + CA·MB + MA·CB)`. Using `MB` (operator norm) instead of `FCB`
    (Frobenius) in the Lipschitz coefficient is what keeps the per-step `L` tight (`3‖X‖²`, `5‖X‖⁴`). -/
theorem matmul_step_affine (A B : Mat) (AR BR : MatR) (r k n : Nat) [Nonempty (Fin n)]
    (εX LA CA LB CB FA FCB MB MA : ℝ)
    (hAsz : A.size = r) (hAk : (if A = #[] then 0 else (A[0]!).size) = k)
    (hBn : (if B = #[] then 0 else B[0]!.size) = n)
    (hARsz : AR.size = r) (hARrow : ∀ i, i < r → (AR[i]!).size = k)
    (hBRsz : BR.size = k) (hBRrow : ∀ i, i < k → (BR[i]!).size = n)
    (hr : 0 < r) (hk : 0 < k)
    (hεA : ‖toMatrixF r k A - toMatrixR r k AR‖ ≤ LA * εX + CA)
    (hεB : ‖toMatrixF k n B - toMatrixR k n BR‖ ≤ LB * εX + CB)
    (hFA : frobRow A r k ≤ FA) (hFCB : frobCol B n k ≤ FCB) (hMB : ‖toMatrixF k n B‖ ≤ MB)
    (hMA : ‖toMatrixR r k AR‖ ≤ MA)
    (hFA0 : 0 ≤ FA) (hLAeps0 : 0 ≤ LA * εX + CA) (hLBeps0 : 0 ≤ LB * εX + CB) :
    ‖toMatrixF r n (matmul A B) - toMatrixR r n (matmulR AR BR)‖
      ≤ (LA * MB + MA * LB) * εX
        + (((2:ℝ) + u64) * (((1:ℝ) + u64) ^ k - 1) * FA * FCB + CA * MB + MA * CB) := by
  have hbc : 0 ≤ ((2:ℝ) + u64) * (((1:ℝ) + u64) ^ k - 1) := by
    have : (1:ℝ) ≤ (1 + u64) ^ k := one_le_pow₀ (by nlinarith [u64_pos.le])
    have h3 : (0:ℝ) ≤ 2 + u64 := by nlinarith [u64_pos.le]
    exact mul_nonneg h3 (by linarith)
  refine le_trans (matmul_error_opNorm A B AR BR r k n (LA * εX + CA) (LB * εX + CB)
    hAsz hAk hBn hARsz hARrow hBRsz hBRrow hr hk hεA hεB) ?_
  -- β(k)·frobRow A·frobCol B + (LA εX+CA)·‖B‖₂ + ‖mirror A‖·(LB εX+CB)  ≤  affine RHS
  have hFAFCB : frobRow A r k * frobCol B n k ≤ FA * FCB :=
    mul_le_mul hFA hFCB (le_trans (Real.sqrt_nonneg _) (le_of_eq (by rw [frobCol]))) hFA0
  have hpB : (LA * εX + CA) * ‖toMatrixF k n B‖ ≤ (LA * εX + CA) * MB :=
    mul_le_mul_of_nonneg_left hMB hLAeps0
  nlinarith [hFAFCB, hMA, hbc, hLAeps0, hLBeps0, hpB,
    mul_le_mul_of_nonneg_left hLBeps0 (le_trans (norm_nonneg _) hMA),
    mul_le_mul_of_nonneg_left hFAFCB hbc]

/-- **One Newton–Schulz step error composed into a single AFFINE per-step bound** (`r ≤ c` branch).
    Chains the four proven op-error atoms (`gram_step_affine`, two `matmul_step_affine`, and
    `nsIter_error_opNorm`) into `ε' ≤ L·εX + C` with `L`, `C` the composed expressions. -/
theorem nsIter_step_affine (X : Puffer.FloatR.Muon.Mat) (XR : Puffer.RL.NewtonSchulzError.MatR)
    (r c : Nat) [Nonempty (Fin c)] (a b cc : Float)
    (εX FX FXt MX FA FCX MA FCAX FAX FAAX MFX MFAX : ℝ)
    (hXsz : X.size = r) (hXrow : ∀ i, i < r → (X[i]!).size = c)
    (hXRsz : XR.size = r) (hXRrow : ∀ i, i < r → (XR[i]!).size = c)
    (hr : 0 < r) (hrc : r ≤ c)
    (hεX : ‖toMatrixF r c X - toMatrixR r c XR‖ ≤ εX)
    (hFX : frobRow X r c ≤ FX) (hFXt : frobCol (Puffer.FloatR.Muon.transpose X) r c ≤ FXt)
    (hMX : ‖toMatrixR r c XR‖ ≤ MX)
    (hMFX : ‖toMatrixF r c X‖ ≤ MFX)
    (hMFAX : ‖toMatrixF r c (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X)‖ ≤ MFAX)
    (hFA : frobRow (matmul X (Puffer.FloatR.Muon.transpose X)) r r ≤ FA)
    (hFCX : frobCol X c r ≤ FCX)
    (hMA : ‖toMatrixR r r (matmulR XR (transposeR XR))‖ ≤ MA)
    (hFCAX : frobCol (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X) c r ≤ FCAX)
    (hFAX : frobRow (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X) r c ≤ FAX)
    (hFAAX : frobRow (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X)) r c ≤ FAAX)
    (hεX0 : 0 ≤ εX) (hFX0 : 0 ≤ FX) (hFXt0 : 0 ≤ FXt) (hMX0 : 0 ≤ MX)
    (hFA0 : 0 ≤ FA) (hFCX0 : 0 ≤ FCX) (hMA0 : 0 ≤ MA) :
    ‖toMatrixF r c (nsIter X (a, b, cc)) - toMatrixR r c (nsIterR XR (a, b, cc))‖
      ≤ (|toReal a| + |toReal b| * ((MFX + MX) * MFX + MA)
            + |toReal cc| * ((MFX + MX) * MFAX
                + MA * ((MFX + MX) * MFX + MA))) * εX
        + (Real.sqrt 3 * (u64 * (3 + 3 * u64 + u64 ^ 2))
              * (|toReal a| * FX + |toReal b| * FAX + |toReal cc| * FAAX)
            + |toReal b| * (((2:ℝ) + u64) * (((1:ℝ) + u64) ^ r - 1) * FA * FCX
                  + ((2:ℝ) + u64) * (((1:ℝ) + u64) ^ c - 1) * FX * FXt * MFX)
            + |toReal cc| * (((2:ℝ) + u64) * (((1:ℝ) + u64) ^ r - 1) * FA * FCAX
                  + ((2:ℝ) + u64) * (((1:ℝ) + u64) ^ c - 1) * FX * FXt * MFAX
                  + MA * (((2:ℝ) + u64) * (((1:ℝ) + u64) ^ r - 1) * FA * FCX
                      + ((2:ℝ) + u64) * (((1:ℝ) + u64) ^ c - 1) * FX * FXt * MFX))) := by
  have hc0 : 0 < c := lt_of_lt_of_le hr hrc
  haveI : Nonempty (Fin r) := ⟨⟨0, hr⟩⟩
  -- abbreviations
  set βc : ℝ := ((2:ℝ) + u64) * (((1:ℝ) + u64) ^ c - 1) with hβc
  set βr : ℝ := ((2:ℝ) + u64) * (((1:ℝ) + u64) ^ r - 1) with hβr
  set K : ℝ := u64 * (3 + 3 * u64 + u64 ^ 2) with hK
  have hu0 : (0:ℝ) ≤ u64 := u64_pos.le
  have hβc0 : 0 ≤ βc := by
    rw [hβc]; have : (1:ℝ) ≤ (1 + u64) ^ c := one_le_pow₀ (by linarith)
    exact mul_nonneg (by linarith) (by linarith)
  have hβr0 : 0 ≤ βr := by
    rw [hβr]; have : (1:ℝ) ≤ (1 + u64) ^ r := one_le_pow₀ (by linarith)
    exact mul_nonneg (by linarith) (by linarith)
  have hMFX0 : 0 ≤ MFX := le_trans (norm_nonneg _) hMFX
  set LAX : ℝ := (MFX + MX) * MFX + MA with hLAX
  set CAX : ℝ := βr * FA * FCX + βc * FX * FXt * MFX with hCAX
  set LAAX : ℝ := (MFX + MX) * MFAX + MA * LAX with hLAAX
  set CAAX : ℝ := βr * FA * FCAX + βc * FX * FXt * MFAX + MA * CAX with hCAAX
  ----------------------------------------------------------------------------
  -- SHAPES for A = matmul X (Puffer.FloatR.Muon.transpose X)  (r × r)
  ----------------------------------------------------------------------------
  have hXne : X ≠ #[] := by intro h; rw [h] at hXsz; simp at hXsz; omega
  have hXn : (if X = #[] then 0 else X[0]!.size) = c := by rw [if_neg hXne, hXrow 0 hr]
  have hAsz : (matmul X (Puffer.FloatR.Muon.transpose X)).size = r := by rw [matmul_size]; exact hXsz
  have htXsz : (Puffer.FloatR.Muon.transpose X).size = c := by rw [transpose_size, if_neg hXne, hXrow 0 hr]
  have htXne : Puffer.FloatR.Muon.transpose X ≠ #[] := by intro h; rw [h] at htXsz; simp at htXsz; omega
  have hAne : matmul X (Puffer.FloatR.Muon.transpose X) ≠ #[] := by
    intro h; rw [h] at hAsz; simp at hAsz; omega
  have hAk : (if matmul X (Puffer.FloatR.Muon.transpose X) = #[] then 0 else ((matmul X (Puffer.FloatR.Muon.transpose X))[0]!).size) = r := by
    rw [if_neg hAne, matmul_rowSize X (Puffer.FloatR.Muon.transpose X) 0 (by rw [hXsz]; exact hr), if_neg htXne,
      transpose_rowSize X 0 (by rw [if_neg hXne, hXrow 0 hr]; exact hc0)]
    exact hXsz
  -- mirror A_R = matmulR XR (transposeR XR)  (r × r)
  have hARsz : (matmulR XR (transposeR XR)).size = r := by rw [matmulR_size]; exact hXRsz
  have htXRsz : (transposeR XR).size = c := by
    rw [transposeR_size, if_neg (by rw [hXRsz]; omega), hXRrow 0 hr]
  have hARrow : ∀ i, i < r → ((matmulR XR (transposeR XR))[i]!).size = r := by
    intro i hi
    rw [matmulR_rowSize XR (transposeR XR) i (by rw [hXRsz]; exact hi),
      if_neg (by rw [htXRsz]; omega),
      transposeR_rowSize XR 0 (by rw [if_neg (by rw [hXRsz]; omega), hXRrow 0 hr]; exact hc0)]
    exact hXRsz
  ----------------------------------------------------------------------------
  -- SHAPES for AX = matmul A X  (r × c) and mirror AX_R = matmulR A_R XR
  ----------------------------------------------------------------------------
  have hAXsz : (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X).size = r := by rw [matmul_size]; exact hAsz
  have hAXne : matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X ≠ #[] := by
    intro h; rw [h] at hAXsz; simp at hAXsz; omega
  have hAXn : (if matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X = #[] then 0
      else ((matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X)[0]!).size) = c := by
    rw [if_neg hAXne, matmul_rowSize (matmul X (Puffer.FloatR.Muon.transpose X)) X 0 (by rw [hAsz]; exact hr),
      if_neg hXne, hXrow 0 hr]
  have hAXRsz : (matmulR (matmulR XR (transposeR XR)) XR).size = r := by rw [matmulR_size]; exact hARsz
  have hAXRrow : ∀ i, i < r → ((matmulR (matmulR XR (transposeR XR)) XR)[i]!).size = c := by
    intro i hi
    rw [matmulR_rowSize (matmulR XR (transposeR XR)) XR i (by rw [hARsz]; exact hi),
      if_neg (by rw [hXRsz]; omega), hXRrow 0 hr]
  ----------------------------------------------------------------------------
  -- STEP 2:  gram atom  ->  ‖A_f - A_R‖ ≤ (FXt+MX)·εX + βc·FX·FXt
  ----------------------------------------------------------------------------
  have hepsA : ‖toMatrixF r r (matmul X (Puffer.FloatR.Muon.transpose X))
      - toMatrixR r r (matmulR XR (transposeR XR))‖ ≤ (MFX + MX) * εX + βc * FX * FXt := by
    have h := gram_step_affine X XR r c εX FX FXt MFX MX hXsz hXrow hXRsz hXRrow hr hc0 hεX hFX hFXt hMFX hMX hεX0 hFX0
    rw [← hβc] at h; exact h
  -- rewrite hεX into the affine form 1·εX + 0
  have hεX' : ‖toMatrixF r c X - toMatrixR r c XR‖ ≤ (1:ℝ) * εX + 0 := by linarith [hεX]
  ----------------------------------------------------------------------------
  -- STEP 3:  AX matmul  A · X   (r,k=r,n=c),  operandA = A, operandB = X
  --   LA = FXt+MX, CA = βc·FX·FXt, LB = 1, CB = 0
  ----------------------------------------------------------------------------
  have hLAeps0 : 0 ≤ (MFX + MX) * εX + βc * FX * FXt := by
    have hA1 : 0 ≤ βc * FX * FXt := by positivity
    have hA2 : 0 ≤ (MFX + MX) * εX := mul_nonneg (by linarith) hεX0
    linarith
  have hLBeps0 : 0 ≤ (1:ℝ) * εX + 0 := by linarith [hεX0]
  have hepsAX' : ‖toMatrixF r c (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X)
      - toMatrixR r c (matmulR (matmulR XR (transposeR XR)) XR)‖ ≤ LAX * εX + CAX := by
    have h := matmul_step_affine (matmul X (Puffer.FloatR.Muon.transpose X)) X
      (matmulR XR (transposeR XR)) XR r r c εX
      (MFX + MX) (βc * FX * FXt) 1 0 FA FCX MFX MA
      hAsz hAk hXn hARsz hARrow hXRsz hXRrow hr hr
      hepsA hεX' hFA hFCX hMFX hMA hFA0 hLAeps0 hLBeps0
    rw [← hβr] at h
    refine le_trans h (le_of_eq ?_)
    rw [hLAX, hCAX]; ring
  ----------------------------------------------------------------------------
  -- STEP 4:  AAX matmul  A · AX   (r,k=r,n=c),  operandA = A, operandB = AX
  --   LA = FXt+MX, CA = βc·FX·FXt, LB = LAX, CB = CAX
  ----------------------------------------------------------------------------
  have hLAX0 : 0 ≤ LAX := by rw [hLAX]; exact add_nonneg (mul_nonneg (by linarith) hMFX0) hMA0
  have hCAX0 : 0 ≤ CAX := by
    rw [hCAX]; have h1 : 0 ≤ βr * FA * FCX := by positivity
    have h2 : 0 ≤ βc * FX * FXt * MFX := by positivity
    linarith
  have hLBeps0' : 0 ≤ LAX * εX + CAX := by
    have : 0 ≤ LAX * εX := mul_nonneg hLAX0 hεX0
    linarith
  have hepsAAX' : ‖toMatrixF r c (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X))
      - toMatrixR r c (matmulR (matmulR XR (transposeR XR)) (matmulR (matmulR XR (transposeR XR)) XR))‖
      ≤ LAAX * εX + CAAX := by
    have h := matmul_step_affine (matmul X (Puffer.FloatR.Muon.transpose X)) (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X)
      (matmulR XR (transposeR XR)) (matmulR (matmulR XR (transposeR XR)) XR) r r c εX
      (MFX + MX) (βc * FX * FXt) LAX CAX FA FCAX MFAX MA
      hAsz hAk hAXn hARsz hARrow hAXRsz hAXRrow hr hr
      hepsA hepsAX' hFA hFCAX hMFAX hMA hFA0 hLAeps0 hLBeps0'
    rw [← hβr] at h
    exact h
  ----------------------------------------------------------------------------
  -- STEP 5:  nsIter combine
  ----------------------------------------------------------------------------
  have hns := nsIter_error_opNorm X XR r c a b cc εX (LAX * εX + CAX) (LAAX * εX + CAAX)
    hXsz hXrow hXRsz hXRrow hr hrc hεX hepsAX' hepsAAX'
  rw [← hK] at hns
  -- bound the frobRow magnitudes by FX, FAX, FAAX inside the K-term
  have hKpos : 0 ≤ Real.sqrt 3 * K := by
    have : 0 ≤ K := by rw [hK]; positivity
    positivity
  -- the three frobRow terms are ≤ FX, FAX, FAAX
  have hfrX : frobRow X r c ≤ FX := hFX
  have hfrAX : frobRow (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X) r c ≤ FAX := hFAX
  have hfrAAX : frobRow (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X)) r c ≤ FAAX :=
    hFAAX
  ----------------------------------------------------------------------------
  -- FINAL:  combine everything into L·εX + C
  ----------------------------------------------------------------------------
  have ha0 : 0 ≤ |toReal a| := abs_nonneg _
  have hb0 : 0 ≤ |toReal b| := abs_nonneg _
  have hcc0 : 0 ≤ |toReal cc| := abs_nonneg _
  -- bound the K bracket monotonically
  have hbracket : |toReal a| * frobRow X r c
      + |toReal b| * frobRow (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X) r c
      + |toReal cc| * frobRow (matmul (matmul X (Puffer.FloatR.Muon.transpose X))
          (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X)) r c
      ≤ |toReal a| * FX + |toReal b| * FAX + |toReal cc| * FAAX := by
    have t1 := mul_le_mul_of_nonneg_left hfrX ha0
    have t2 := mul_le_mul_of_nonneg_left hfrAX hb0
    have t3 := mul_le_mul_of_nonneg_left hfrAAX hcc0
    linarith
  -- chain: hns RHS ≤ target
  have hKmono : Real.sqrt 3 * K * (|toReal a| * frobRow X r c
      + |toReal b| * frobRow (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X) r c
      + |toReal cc| * frobRow (matmul (matmul X (Puffer.FloatR.Muon.transpose X))
          (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X)) r c)
      ≤ Real.sqrt 3 * K * (|toReal a| * FX + |toReal b| * FAX + |toReal cc| * FAAX) :=
    mul_le_mul_of_nonneg_left hbracket hKpos
  -- the additive ε part: |a|·εX + |b|·(LAX εX+CAX) + |cc|·(LAAX εX+CAAX)
  --   = (|a| + |b|·LAX + |cc|·LAAX)·εX + (|b|·CAX + |cc|·CAAX)
  refine le_trans hns ?_
  linarith [hKmono]

/-! ### The 5-iteration fold — a uniform per-step affine bound folds to a POLYNOMIAL `FE`

The final piece: `newtonSchulz = muonCoeffs.foldl nsIter seed` (5 iterations). `fold_affine_error` is the
generic combinator — a UNIFORM per-step affine error bound `ε' ≤ L·ε + C` folds to `L^len·ε₀ + C·∑_{k<len}Lᵏ`
(affine, hence polynomial). `newtonSchulz_opNorm_poly` instantiates it and adds the proven mirror bound
(`nsIterR_comp_normsq`, `≤ √1.3131`) via triangle, giving the runnable `newtonSchulz`'s operator norm
`≤ √1.3131 + polynomial(FE)`. The uniform per-step bound `hstep` (the magnitude/error BOOTSTRAP) is the sole
hypothesis: establishing it — the full per-step composition (`gram_error_opNorm` + matmul chain +
`nsIter_error_opNorm`, giving `ε' ≤ L·ε + C`) with the magnitudes uniformly bounded (`frobRow ≤ √dim·‖·‖₂` +
the proven per-step mirror bounds) — is the remaining coupling. -/

theorem fold_affine_error {β : Type} (f : Mat → β → Mat) (fR : MatR → β → MatR)
    (r c : Nat) (L C : ℝ)
    (hstep : ∀ (X : Mat) (XR : MatR) (coef : β) (ε : ℝ),
       ‖toMatrixF r c X - toMatrixR r c XR‖ ≤ ε →
       ‖toMatrixF r c (f X coef) - toMatrixR r c (fR XR coef)‖ ≤ L * ε + C) :
    ∀ (coeffs : List β) (seed : Mat) (seedR : MatR) (ε0 : ℝ),
      ‖toMatrixF r c seed - toMatrixR r c seedR‖ ≤ ε0 →
      ‖toMatrixF r c (coeffs.foldl f seed) - toMatrixR r c (coeffs.foldl fR seedR)‖
        ≤ L ^ coeffs.length * ε0 + C * (∑ k ∈ Finset.range coeffs.length, L ^ k) := by
  intro coeffs
  induction coeffs with
  | nil => intro seed seedR ε0 h; simpa using h
  | cons coef rest ih =>
      intro seed seedR ε0 h
      have hrec := ih (f seed coef) (fR seedR coef) (L * ε0 + C) (hstep seed seedR coef ε0 h)
      simp only [List.foldl_cons, List.length_cons]
      refine le_trans hrec (le_of_eq ?_)
      rw [Finset.sum_range_succ, pow_succ]; ring

/-- **The whole `newtonSchulz` fold error is POLYNOMIAL — the culminating tight-tower statement with a
    polynomial `FE`.** For a Frobenius-normalized seed (`‖toMatrixR seedR‖² ≤ 1`) and a UNIFORM per-step affine
    error bound `ε' ≤ L·ε + C` (the magnitude/error bootstrap), the runnable `newtonSchulz X0 eps` has operator
    norm `≤ √1.3131 + (L^5·ε_seed + C·(L⁴+L³+L²+L+1))` — DIMENSION-FREE `√1.3131 < 1.15` PLUS a POLYNOMIAL (affine)
    accumulated rounding, versus the entrywise tower's doubly-exponential `FE`. Combines `fold_affine_error`
    (the fold error is affine in the per-step `L`, `C`) with `nsIterR_comp_normsq` (the mirror fold `≤ √1.3131`)
    via the triangle inequality. -/
theorem newtonSchulz_opNorm_poly (X0 : Mat) (eps : Float) (seedR : MatR) (r cc : Nat) (L C ε0 : ℝ)
    (hr : 0 < r) (hrc : r ≤ cc)
    (hSsz : seedR.size = r) (hSrow : ∀ i, i < r → (seedR[i]!).size = cc)
    (hseedNorm : ‖toMatrixR r cc seedR‖ ^ 2 ≤ 1)
    (hseedErr : ‖toMatrixF r cc (scalarMul (1.0 / (frobNorm X0 + eps)) X0) - toMatrixR r cc seedR‖ ≤ ε0)
    (hstep : ∀ (X : Mat) (XR : MatR) (coef : Float × Float × Float) (ε : ℝ),
       ‖toMatrixF r cc X - toMatrixR r cc XR‖ ≤ ε →
       ‖toMatrixF r cc (nsIter X coef) - toMatrixR r cc (nsIterR XR coef)‖ ≤ L * ε + C) :
    ‖toMatrixF r cc (newtonSchulz X0 eps)‖
      ≤ Real.sqrt 1.3131
        + (L ^ muonCoeffs.toList.length * ε0
           + C * (∑ k ∈ Finset.range muonCoeffs.toList.length, L ^ k)) := by
  have hns : newtonSchulz X0 eps
      = muonCoeffs.toList.foldl nsIter (scalarMul (1.0 / (frobNorm X0 + eps)) X0) := by
    rw [newtonSchulz_eq_foldl, ← Array.foldl_toList]
  have hfold := fold_affine_error nsIter nsIterR r cc L C hstep muonCoeffs.toList
    (scalarMul (1.0 / (frobNorm X0 + eps)) X0) seedR ε0 hseedErr
  rw [← hns] at hfold
  have hmir := nsIterR_comp_normsq seedR r cc hSsz hSrow hr hrc hseedNorm
  have hmirle : ‖toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR)‖ ≤ Real.sqrt 1.3131 := by
    rw [← Real.sqrt_sq (norm_nonneg _)]; exact Real.sqrt_le_sqrt hmir
  calc ‖toMatrixF r cc (newtonSchulz X0 eps)‖
      = ‖toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR)
          + (toMatrixF r cc (newtonSchulz X0 eps)
             - toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR))‖ := by rw [add_sub_cancel]
    _ ≤ ‖toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR)‖
          + ‖toMatrixF r cc (newtonSchulz X0 eps)
             - toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR)‖ := norm_add_le _ _
    _ ≤ Real.sqrt 1.3131
          + (L ^ muonCoeffs.toList.length * ε0
             + C * (∑ k ∈ Finset.range muonCoeffs.toList.length, L ^ k)) := add_le_add hmirle hfold

/-- **Region-restricted affine-fold error combinator (the coupled bootstrap fold).** The per-step affine
    bound `ε' ≤ L·ε + C` need only hold WITHIN the self-bounding region `{ε ≤ εmax}` (`L·εmax + C ≤ εmax`) —
    the region is then forward-INVARIANT (the error never leaves it), so the fold error is
    `≤ L^len·ε₀ + C·∑_{k<len} Lᵏ`. This resolves the coupling: within `{ε ≤ εmax}` the Float Frobenius
    magnitudes (the per-step coefficients) are bounded (`frobRow_le_mirror_add_err` + the mirror bound), so
    `L`, `C` are uniform; and `region_invariant`-style self-bounding keeps the error there. -/
theorem fold_region_affine_error {β : Type} (f : Mat → β → Mat) (fR : MatR → β → MatR)
    (r c : Nat) (L C εmax : ℝ) (hL : 0 ≤ L) (hself : L * εmax + C ≤ εmax)
    (hstep : ∀ (X : Mat) (XR : MatR) (coef : β) (ε : ℝ), ε ≤ εmax →
       ‖toMatrixF r c X - toMatrixR r c XR‖ ≤ ε →
       ‖toMatrixF r c (f X coef) - toMatrixR r c (fR XR coef)‖ ≤ L * ε + C) :
    ∀ (coeffs : List β) (seed : Mat) (seedR : MatR) (ε0 : ℝ), ε0 ≤ εmax →
      ‖toMatrixF r c seed - toMatrixR r c seedR‖ ≤ ε0 →
      ‖toMatrixF r c (coeffs.foldl f seed) - toMatrixR r c (coeffs.foldl fR seedR)‖
        ≤ L ^ coeffs.length * ε0 + C * (∑ k ∈ Finset.range coeffs.length, L ^ k) := by
  intro coeffs
  induction coeffs with
  | nil => intro seed seedR ε0 _ h; simpa using h
  | cons coef rest ih =>
      intro seed seedR ε0 hε0max h
      have hstep0 := hstep seed seedR coef ε0 hε0max h
      have hnext : L * ε0 + C ≤ εmax := le_trans (by nlinarith [hL, hε0max]) hself
      have hrec := ih (f seed coef) (fR seedR coef) (L * ε0 + C) hnext hstep0
      simp only [List.foldl_cons, List.length_cons]
      refine le_trans hrec (le_of_eq ?_)
      rw [Finset.sum_range_succ, pow_succ]; ring

/-- **`newtonSchulz` polynomial-`FE` bound from the coupled region fold.** Same culminating statement as
    `newtonSchulz_opNorm_poly` but the per-step affine bound need only hold in the self-bounding region
    `{ε ≤ εmax}` — the dischargeable form (the magnitudes are bounded there). -/
theorem newtonSchulz_opNorm_region (X0 : Mat) (eps : Float) (seedR : MatR) (r cc : Nat) (L C εmax ε0 : ℝ)
    (hr : 0 < r) (hrc : r ≤ cc) (hL : 0 ≤ L) (hself : L * εmax + C ≤ εmax)
    (hSsz : seedR.size = r) (hSrow : ∀ i, i < r → (seedR[i]!).size = cc)
    (hseedNorm : ‖toMatrixR r cc seedR‖ ^ 2 ≤ 1)
    (hε0max : ε0 ≤ εmax)
    (hseedErr : ‖toMatrixF r cc (scalarMul (1.0 / (frobNorm X0 + eps)) X0) - toMatrixR r cc seedR‖ ≤ ε0)
    (hstep : ∀ (X : Mat) (XR : MatR) (coef : Float × Float × Float) (ε : ℝ), ε ≤ εmax →
       ‖toMatrixF r cc X - toMatrixR r cc XR‖ ≤ ε →
       ‖toMatrixF r cc (nsIter X coef) - toMatrixR r cc (nsIterR XR coef)‖ ≤ L * ε + C) :
    ‖toMatrixF r cc (newtonSchulz X0 eps)‖
      ≤ Real.sqrt 1.3131
        + (L ^ muonCoeffs.toList.length * ε0
           + C * (∑ k ∈ Finset.range muonCoeffs.toList.length, L ^ k)) := by
  have hns : newtonSchulz X0 eps
      = muonCoeffs.toList.foldl nsIter (scalarMul (1.0 / (frobNorm X0 + eps)) X0) := by
    rw [newtonSchulz_eq_foldl, ← Array.foldl_toList]
  have hfold := fold_region_affine_error nsIter nsIterR r cc L C εmax hL hself hstep muonCoeffs.toList
    (scalarMul (1.0 / (frobNorm X0 + eps)) X0) seedR ε0 hε0max hseedErr
  rw [← hns] at hfold
  have hmirle : ‖toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR)‖ ≤ Real.sqrt 1.3131 := by
    rw [← Real.sqrt_sq (norm_nonneg _)]
    exact Real.sqrt_le_sqrt (nsIterR_comp_normsq seedR r cc hSsz hSrow hr hrc hseedNorm)
  calc ‖toMatrixF r cc (newtonSchulz X0 eps)‖
      = ‖toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR)
          + (toMatrixF r cc (newtonSchulz X0 eps)
             - toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR))‖ := by rw [add_sub_cancel]
    _ ≤ ‖toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR)‖
          + ‖toMatrixF r cc (newtonSchulz X0 eps)
             - toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR)‖ := norm_add_le _ _
    _ ≤ Real.sqrt 1.3131
          + (L ^ muonCoeffs.toList.length * ε0
             + C * (∑ k ∈ Finset.range muonCoeffs.toList.length, L ^ k)) := add_le_add hmirle hfold

/-! ### The mirror-bound-carrying fold — the coupled bootstrap fully threaded

`fold_region_affine_error` (a40) needed the per-step bound uniform in `{ε ≤ εmax}`, but the magnitudes need
`‖mirror XR‖ ≤ Rm` too, and that mirror bound is POSITION-SPECIFIC (no uniform invariant — step 1's scalar
maps `[0,1] → [0,1.63]`). `fold_affine_mirror` resolves this: it threads a per-prefix mirror-bound hypothesis
(`hmir`, the position-specific op-norms the composition supplies) through the fold, so the per-step bound may
assume `‖mirror XR‖ ≤ Rm`. `newtonSchulz_opNorm_mirror` is then the culminating statement whose TWO remaining
hypotheses are both genuinely dischargeable: `hstep` from `nsIter_step_affine` (magnitudes bounded via
`frobRow`/`frobCol_le_mirror_add_err` from `Rm` + the region), and `hmir` from `MuonComposition`. -/

/-- **Mirror-bound-carrying affine fold.** The per-step error bound `ε' ≤ L·ε + C` need only hold within the
    self-bounding error region `{ε ≤ εmax}` AND when the mirror iterate is bounded `‖mirror XR‖ ≤ Rm`. Given
    that the mirror bound holds at EVERY fold prefix (`hmir`, a POSITION-SPECIFIC fact the caller discharges
    from the mirror composition — there is no uniform mirror invariant), the fold error is `≤ L^len·ε₀ +
    C·∑Lᵏ`. This threads the position-specific mirror bounds through the fold (the resolution of the coupled
    bootstrap: the mirror magnitudes come from outside, the error accumulates affinely). -/
theorem fold_affine_mirror {β : Type} (f : Mat → β → Mat) (fR : MatR → β → MatR)
    (r c : Nat) (L C εmax Rm : ℝ) (hL : 0 ≤ L) (hself : L * εmax + C ≤ εmax)
    (hstep : ∀ (X : Mat) (XR : MatR) (coef : β) (ε : ℝ), ε ≤ εmax → ‖toMatrixR r c XR‖ ≤ Rm →
       ‖toMatrixF r c X - toMatrixR r c XR‖ ≤ ε →
       ‖toMatrixF r c (f X coef) - toMatrixR r c (fR XR coef)‖ ≤ L * ε + C) :
    ∀ (coeffs : List β) (seed : Mat) (seedR : MatR) (ε0 : ℝ), ε0 ≤ εmax →
      ‖toMatrixF r c seed - toMatrixR r c seedR‖ ≤ ε0 →
      (∀ m, m ≤ coeffs.length → ‖toMatrixR r c ((coeffs.take m).foldl fR seedR)‖ ≤ Rm) →
      ‖toMatrixF r c (coeffs.foldl f seed) - toMatrixR r c (coeffs.foldl fR seedR)‖
        ≤ L ^ coeffs.length * ε0 + C * (∑ k ∈ Finset.range coeffs.length, L ^ k) := by
  intro coeffs
  induction coeffs with
  | nil => intro seed seedR ε0 _ h _; simpa using h
  | cons coef rest ih =>
      intro seed seedR ε0 hε0max h hmir
      -- mirror bound at position 0 = seedR
      have hmir0 : ‖toMatrixR r c seedR‖ ≤ Rm := by
        have := hmir 0 (Nat.zero_le _); simpa using this
      have hstep0 := hstep seed seedR coef ε0 hε0max hmir0 h
      have hnext : L * ε0 + C ≤ εmax := le_trans (by nlinarith [hL, hε0max]) hself
      -- shifted mirror hypothesis for the tail
      have hmir' : ∀ m, m ≤ rest.length →
          ‖toMatrixR r c ((rest.take m).foldl fR (fR seedR coef))‖ ≤ Rm := by
        intro m hm
        have h2 := hmir (m + 1) (Nat.succ_le_succ hm)
        rwa [List.take_succ_cons, List.foldl_cons] at h2
      have hrec := ih (f seed coef) (fR seedR coef) (L * ε0 + C) hnext hstep0 hmir'
      simp only [List.foldl_cons, List.length_cons]
      refine le_trans hrec (le_of_eq ?_)
      rw [Finset.sum_range_succ, pow_succ]; ring

/-- **`newtonSchulz` polynomial-`FE` bound with the mirror-bound-carrying fold — the coupled bootstrap fully
    threaded.** Same `√1.3131 + polynomial(FE)` conclusion, but the per-step affine bound holds only in-region
    `{ε ≤ εmax}` AND when the mirror iterate is bounded (`‖mirror XR‖ ≤ Rm`), and the mirror bound is supplied
    at EVERY fold prefix (`hmir`) — the position-specific mirror op-norms the composition provides. This is the
    DISCHARGEABLE form: `hstep` follows from `nsIter_step_affine` (the magnitudes bounded via
    `frobRow/frobCol_le_mirror_add_err` from `Rm` + the region), and `hmir` from `MuonComposition`. -/
theorem newtonSchulz_opNorm_mirror (X0 : Mat) (eps : Float) (seedR : MatR) (r cc : Nat)
    (L C εmax Rm ε0 : ℝ)
    (hr : 0 < r) (hrc : r ≤ cc) (hL : 0 ≤ L) (hself : L * εmax + C ≤ εmax)
    (hSsz : seedR.size = r) (hSrow : ∀ i, i < r → (seedR[i]!).size = cc)
    (hseedNorm : ‖toMatrixR r cc seedR‖ ^ 2 ≤ 1)
    (hε0max : ε0 ≤ εmax)
    (hseedErr : ‖toMatrixF r cc (scalarMul (1.0 / (frobNorm X0 + eps)) X0) - toMatrixR r cc seedR‖ ≤ ε0)
    (hmir : ∀ m, m ≤ muonCoeffs.toList.length →
       ‖toMatrixR r cc ((muonCoeffs.toList.take m).foldl nsIterR seedR)‖ ≤ Rm)
    (hstep : ∀ (X : Mat) (XR : MatR) (coef : Float × Float × Float) (ε : ℝ), ε ≤ εmax →
       ‖toMatrixR r cc XR‖ ≤ Rm →
       ‖toMatrixF r cc X - toMatrixR r cc XR‖ ≤ ε →
       ‖toMatrixF r cc (nsIter X coef) - toMatrixR r cc (nsIterR XR coef)‖ ≤ L * ε + C) :
    ‖toMatrixF r cc (newtonSchulz X0 eps)‖
      ≤ Real.sqrt 1.3131
        + (L ^ muonCoeffs.toList.length * ε0
           + C * (∑ k ∈ Finset.range muonCoeffs.toList.length, L ^ k)) := by
  have hns : newtonSchulz X0 eps
      = muonCoeffs.toList.foldl nsIter (scalarMul (1.0 / (frobNorm X0 + eps)) X0) := by
    rw [newtonSchulz_eq_foldl, ← Array.foldl_toList]
  have hfold := fold_affine_mirror nsIter nsIterR r cc L C εmax Rm hL hself hstep muonCoeffs.toList
    (scalarMul (1.0 / (frobNorm X0 + eps)) X0) seedR ε0 hε0max hseedErr hmir
  rw [← hns] at hfold
  have hmirle : ‖toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR)‖ ≤ Real.sqrt 1.3131 := by
    rw [← Real.sqrt_sq (norm_nonneg _)]
    exact Real.sqrt_le_sqrt (nsIterR_comp_normsq seedR r cc hSsz hSrow hr hrc hseedNorm)
  calc ‖toMatrixF r cc (newtonSchulz X0 eps)‖
      = ‖toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR)
          + (toMatrixF r cc (newtonSchulz X0 eps)
             - toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR))‖ := by rw [add_sub_cancel]
    _ ≤ ‖toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR)‖
          + ‖toMatrixF r cc (newtonSchulz X0 eps)
             - toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR)‖ := norm_add_le _ _
    _ ≤ Real.sqrt 1.3131
          + (L ^ muonCoeffs.toList.length * ε0
             + C * (∑ k ∈ Finset.range muonCoeffs.toList.length, L ^ k)) := add_le_add hmirle hfold

/-! ### The EXPANDING-recurrence correction (`L ≥ 1`)

The Newton–Schulz per-step Lipschitz factor is `L > 1` (the Float-vs-mirror error amplifies by ~`|a|+3|b|+5|c|
≈ 40` per step), so the SELF-BOUNDING region `L·εmax + C ≤ εmax` of `fold_region_affine_error`/
`fold_affine_mirror` is EMPTY — those combinators are correctly proven but their region hypothesis is not
dischargeable for the real recurrence. `fold_affine_mirror_exp` is the fix: the per-step bound holds for `ε ≤
B` (a fixed magnitude-bounding threshold), the caller supplies `L^len·ε₀ + C·∑Lᵏ ≤ B` (the FINAL error stays
under `B`), and the threaded "final-error-from-here ≤ B" invariant — preserved EXACTLY by each step — gives
`ε ≤ B` at every position. `newtonSchulz_opNorm_mirror_exp` is the corrected culminating statement. -/

/-- **Mirror-bound-carrying affine fold for an EXPANDING recurrence (`L ≥ 1`).** The per-step Lipschitz factor
    of the Newton–Schulz error is `> 1` (the error amplifies ~`|a|+3|b|+5|c|` per step), so the self-bounding
    region `L·εmax+C ≤ εmax` is empty. Instead: the per-step bound holds for `ε ≤ B` (a FIXED magnitude-bounding
    threshold), and the caller supplies `L^len·ε₀ + C·∑Lᵏ ≤ B` (the FINAL error stays under `B`). The threaded
    invariant "final-error-from-here ≤ B" is preserved EXACTLY by each step (`L^{d}·(L·ε+C) + C·geom_{d-1} =
    L^{d+1}·ε + C·geom_d`), and gives `ε ≤ B` at every position (via `L^len·ε ≥ ε` for `L ≥ 1`). So the fold
    error is `≤ L^len·ε₀ + C·∑Lᵏ`. The correct fold for the real (expanding) Newton–Schulz error. -/
theorem fold_affine_mirror_exp {β : Type} (f : Mat → β → Mat) (fR : MatR → β → MatR)
    (r c : Nat) (L C B Rm : ℝ) (hL1 : 1 ≤ L) (hC0 : 0 ≤ C)
    (hstep : ∀ (X : Mat) (XR : MatR) (coef : β) (ε : ℝ), ε ≤ B → ‖toMatrixR r c XR‖ ≤ Rm →
       ‖toMatrixF r c X - toMatrixR r c XR‖ ≤ ε →
       ‖toMatrixF r c (f X coef) - toMatrixR r c (fR XR coef)‖ ≤ L * ε + C) :
    ∀ (coeffs : List β) (seed : Mat) (seedR : MatR) (ε0 : ℝ),
      L ^ coeffs.length * ε0 + C * (∑ k ∈ Finset.range coeffs.length, L ^ k) ≤ B →
      ‖toMatrixF r c seed - toMatrixR r c seedR‖ ≤ ε0 →
      (∀ m, m ≤ coeffs.length → ‖toMatrixR r c ((coeffs.take m).foldl fR seedR)‖ ≤ Rm) →
      ‖toMatrixF r c (coeffs.foldl f seed) - toMatrixR r c (coeffs.foldl fR seedR)‖
        ≤ L ^ coeffs.length * ε0 + C * (∑ k ∈ Finset.range coeffs.length, L ^ k) := by
  intro coeffs
  induction coeffs with
  | nil => intro seed seedR ε0 _ h _; simpa using h
  | cons coef rest ih =>
      intro seed seedR ε0 hB h hmir
      have hε00 : 0 ≤ ε0 := le_trans (norm_nonneg _) h
      have hLm : (1:ℝ) ≤ L ^ (rest.length + 1) := one_le_pow₀ hL1
      have hgeom0 : 0 ≤ C * (∑ k ∈ Finset.range (rest.length + 1), L ^ k) :=
        mul_nonneg hC0 (Finset.sum_nonneg (fun k _ => pow_nonneg (by linarith) k))
      have hε0B : ε0 ≤ B := by
        have h1 : ε0 ≤ L ^ (rest.length + 1) * ε0 := le_mul_of_one_le_left hε00 hLm
        rw [List.length_cons] at hB; linarith
      have hmir0 : ‖toMatrixR r c seedR‖ ≤ Rm := by
        have := hmir 0 (Nat.zero_le _); simpa using this
      have hstep0 := hstep seed seedR coef ε0 hε0B hmir0 h
      have hBtail : L ^ rest.length * (L * ε0 + C) + C * (∑ k ∈ Finset.range rest.length, L ^ k) ≤ B := by
        have heq : L ^ rest.length * (L * ε0 + C) + C * (∑ k ∈ Finset.range rest.length, L ^ k)
            = L ^ (rest.length + 1) * ε0 + C * (∑ k ∈ Finset.range (rest.length + 1), L ^ k) := by
          rw [Finset.sum_range_succ, pow_succ]; ring
        rw [heq]; rw [List.length_cons] at hB; exact hB
      have hmir' : ∀ m, m ≤ rest.length →
          ‖toMatrixR r c ((rest.take m).foldl fR (fR seedR coef))‖ ≤ Rm := by
        intro m hm
        have h2 := hmir (m + 1) (Nat.succ_le_succ hm)
        rwa [List.take_succ_cons, List.foldl_cons] at h2
      have hrec := ih (f seed coef) (fR seedR coef) (L * ε0 + C) hBtail hstep0 hmir'
      simp only [List.foldl_cons, List.length_cons]
      refine le_trans hrec (le_of_eq ?_)
      rw [Finset.sum_range_succ, pow_succ]; ring

/-- **`newtonSchulz` polynomial-`FE` bound via the EXPANDING mirror fold — the corrected coupled bootstrap.**
    The per-step Lipschitz factor `L > 1`, so this uses `fold_affine_mirror_exp` (final error `≤ B`) rather than
    a self-bounding region. `‖toMatrixF(newtonSchulz X0 eps)‖₂ ≤ √1.3131 + (L⁵·ε_seed + C·(L⁴+…+1))` given the
    per-step bound `hstep` (for `ε ≤ B`, dischargeable from `nsIter_step_affine`), the per-prefix mirror bounds
    `hmir` (from `MuonComposition`), the final-error bound `L⁵·ε_seed + C·geom ≤ B`, and the normalized seed. -/
theorem newtonSchulz_opNorm_mirror_exp (X0 : Mat) (eps : Float) (seedR : MatR) (r cc : Nat)
    (L C B Rm ε0 : ℝ) (hr : 0 < r) (hrc : r ≤ cc) (hL1 : 1 ≤ L) (hC0 : 0 ≤ C)
    (hSsz : seedR.size = r) (hSrow : ∀ i, i < r → (seedR[i]!).size = cc)
    (hseedNorm : ‖toMatrixR r cc seedR‖ ^ 2 ≤ 1)
    (hB : L ^ muonCoeffs.toList.length * ε0
        + C * (∑ k ∈ Finset.range muonCoeffs.toList.length, L ^ k) ≤ B)
    (hseedErr : ‖toMatrixF r cc (scalarMul (1.0 / (frobNorm X0 + eps)) X0) - toMatrixR r cc seedR‖ ≤ ε0)
    (hmir : ∀ m, m ≤ muonCoeffs.toList.length →
       ‖toMatrixR r cc ((muonCoeffs.toList.take m).foldl nsIterR seedR)‖ ≤ Rm)
    (hstep : ∀ (X : Mat) (XR : MatR) (coef : Float × Float × Float) (ε : ℝ), ε ≤ B →
       ‖toMatrixR r cc XR‖ ≤ Rm →
       ‖toMatrixF r cc X - toMatrixR r cc XR‖ ≤ ε →
       ‖toMatrixF r cc (nsIter X coef) - toMatrixR r cc (nsIterR XR coef)‖ ≤ L * ε + C) :
    ‖toMatrixF r cc (newtonSchulz X0 eps)‖
      ≤ Real.sqrt 1.3131
        + (L ^ muonCoeffs.toList.length * ε0
           + C * (∑ k ∈ Finset.range muonCoeffs.toList.length, L ^ k)) := by
  have hns : newtonSchulz X0 eps
      = muonCoeffs.toList.foldl nsIter (scalarMul (1.0 / (frobNorm X0 + eps)) X0) := by
    rw [newtonSchulz_eq_foldl, ← Array.foldl_toList]
  have hfold := fold_affine_mirror_exp nsIter nsIterR r cc L C B Rm hL1 hC0 hstep muonCoeffs.toList
    (scalarMul (1.0 / (frobNorm X0 + eps)) X0) seedR ε0 hB hseedErr hmir
  rw [← hns] at hfold
  have hmirle : ‖toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR)‖ ≤ Real.sqrt 1.3131 := by
    rw [← Real.sqrt_sq (norm_nonneg _)]
    exact Real.sqrt_le_sqrt (nsIterR_comp_normsq seedR r cc hSsz hSrow hr hrc hseedNorm)
  calc ‖toMatrixF r cc (newtonSchulz X0 eps)‖
      = ‖toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR)
          + (toMatrixF r cc (newtonSchulz X0 eps)
             - toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR))‖ := by rw [add_sub_cancel]
    _ ≤ ‖toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR)‖
          + ‖toMatrixF r cc (newtonSchulz X0 eps)
             - toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR)‖ := norm_add_le _ _
    _ ≤ Real.sqrt 1.3131
          + (L ^ muonCoeffs.toList.length * ε0
             + C * (∑ k ∈ Finset.range muonCoeffs.toList.length, L ^ k)) := add_le_add hmirle hfold

/-! ### Concrete-constants discharge — eliminating the two structural hypotheses

`newtonSchulz_opNorm_mirror_exp` leaves two structural (∀-quantified) hypotheses: `hstep` (the per-step affine
composition) and `hmir` (the per-prefix mirror op-norm bounds). This section discharges BOTH into concrete
form, leaving only scalar side-conditions.

`hmir` is discharged by `nsIterR_prefix_normle`: every prefix of the mirror fold has operator norm `≤ √1.63`
(the schedule maximum), directly from the `nsIterR_step_normsq` chain — the same per-step bounds
`nsIterR_comp_normsq` uses, but exposed at every prefix rather than only the final one. So `Rm := √1.63`,
unconditionally (given a normalized seed).

`hstep` needs the 9 Frobenius/operator magnitudes of `nsIter_step_affine` bounded uniformly. The mirror
op-norm powers (`Rm²`, `Rm³`, `Rm⁵`) come from submultiplicativity (`mirror_mul_opNorm` +
`mirror_transpose_opNorm`); the Float Frobenius magnitudes come from `frobRow/frobCol_le_mirror_add_err`
(mirror op-norm + intermediate error), and the intermediate errors from `gram_step_affine`/`matmul_step_affine`
with `ε ≤ B`. Assembled in `nsIter_step_uniform`. -/

/-- **Mirror product op-norm is submultiplicative.** `‖toMatrixR(matmulR A B)‖ ≤ ‖toMatrixR A‖·‖toMatrixR B‖`
    — the mirror product embeds EXACTLY as the Mathlib matrix product (`matmulR_toMatrixR`), so l2-operator
    submultiplicativity (`l2_opNorm_mul`) applies. The atom for the `Rm²/Rm³/Rm⁵` op-norm powers. -/
theorem mirror_mul_opNorm (A B : MatR) (r k c : Nat)
    (hA : A.size = r) (hArow : ∀ i, i < r → (A[i]!).size = k)
    (hB : B.size = k) (hBrow : ∀ i, i < k → (B[i]!).size = c)
    (hr : 0 < r) (hk : 0 < k) :
    ‖toMatrixR r c (matmulR A B)‖ ≤ ‖toMatrixR r k A‖ * ‖toMatrixR k c B‖ := by
  rw [matmulR_toMatrixR A B r k c hA hArow hB hBrow hr hk]; exact l2_opNorm_mul _ _

/-- **Mirror transpose is an l2-operator isometry.** `‖toMatrixR(transposeR X)‖ = ‖toMatrixR X‖` — the mirror
    transpose embeds as the Mathlib transpose (`transposeR_toMatrixR`), which over ℝ equals the conjugate
    transpose (`conjTranspose_eq_transpose_of_trivial`, `star = id`), and the l2 operator norm is
    conjTranspose-invariant (`l2_opNorm_conjTranspose`). -/
theorem mirror_transpose_opNorm (X : MatR) (r c : Nat)
    (hX : X.size = r) (hXrow : ∀ i, i < r → (X[i]!).size = c) (hr : 0 < r) :
    ‖toMatrixR c r (transposeR X)‖ = ‖toMatrixR r c X‖ := by
  rw [transposeR_toMatrixR X r c hX hXrow hr, ← conjTranspose_eq_transpose_of_trivial, l2_opNorm_conjTranspose]

/-- **`hmir` discharged: every prefix of the mirror fold has operator norm `≤ √1.63`.** For a normalized seed
    (`‖toMatrixR seed‖² ≤ 1`), each of the six prefixes `(muonCoeffs.toList.take m).foldl nsIterR seed`
    (`m ≤ 5`) has `‖·‖₂ ≤ √1.63` — the schedule maximum. Proof: the `nsIterR_step_normsq` chain gives
    per-prefix norm-squared bounds `[1, 1.63, 1.63, 1.57, 1.33, 1.3131]` (the same bounds `nsIterR_comp_normsq`
    threads), all `≤ 1.63`; `interval_cases m` reduces each prefix (via the concrete list) and `√·` monotonicity
    finishes. So `Rm := √1.63` discharges the `hmir` hypothesis of `newtonSchulz_opNorm_mirror_exp`. -/
theorem nsIterR_prefix_normle (seed : MatR) (r cc : Nat)
    (hsz : seed.size = r) (hrow : ∀ i, i < r → (seed[i]!).size = cc) (hr : 0 < r) (hrc : r ≤ cc)
    (hseed : ‖toMatrixR r cc seed‖ ^ 2 ≤ 1) :
    ∀ m, m ≤ muonCoeffs.toList.length →
      ‖toMatrixR r cc ((muonCoeffs.toList.take m).foldl nsIterR seed)‖ ≤ Real.sqrt 1.63 := by
  have shape : ∀ (Y : MatR) (coef : Float × Float × Float), Y.size = r →
      (∀ i, i < r → (Y[i]!).size = cc) →
      (nsIterR Y coef).size = r ∧ (∀ i, i < r → ((nsIterR Y coef)[i]!).size = cc) := by
    intro Y coef hYsz hYrow
    exact ⟨by rw [nsIterR_size, hYsz],
      fun i hi => by rw [nsIterR_rowSize Y coef i (by rw [hYsz]; exact hi), hYrow i hi]⟩
  obtain ⟨s1, r1⟩ := shape seed _ hsz hrow
  obtain ⟨s2, r2⟩ := shape _ _ s1 r1
  obtain ⟨s3, r3⟩ := shape _ _ s2 r2
  obtain ⟨s4, r4⟩ := shape _ _ s3 r3
  have c1 := nsIterR_step_normsq seed 4.0848 (-6.8946) 2.9270 r cc hsz hrow hr hrc hseed
    (fun t h0 h1 => muon_comp_step1_float t h0 h1)
  have c2 := nsIterR_step_normsq _ 3.9505 (-6.3029) 2.6377 r cc s1 r1 hr hrc c1
    (fun t h0 h1 => muon_comp_step2_float t h0 h1)
  have c3 := nsIterR_step_normsq _ 3.7418 (-5.5913) 2.3037 r cc s2 r2 hr hrc c2
    (fun t h0 h1 => muon_comp_step3_float t h0 h1)
  have c4 := nsIterR_step_normsq _ 2.8769 (-3.1427) 1.2046 r cc s3 r3 hr hrc c3
    (fun t h0 h1 => muon_comp_step4_float t h0 h1)
  have c5 := nsIterR_step_normsq _ 2.8366 (-3.0525) 1.2012 r cc s4 r4 hr hrc c4
    (fun t h0 h1 => muon_comp_step5_float t h0 h1)
  intro m hm
  rw [show muonCoeffs.toList.length = 5 from by decide] at hm
  interval_cases m <;>
    rw [show muonCoeffs.toList = [(4.0848, -6.8946, 2.9270), (3.9505, -6.3029, 2.6377),
      (3.7418, -5.5913, 2.3037), (2.8769, -3.1427, 1.2046), (2.8366, -3.0525, 1.2012)] from rfl] <;>
    simp only [List.take_succ_cons, List.take_zero, List.take_nil, List.foldl_cons, List.foldl_nil]
  · rw [← Real.sqrt_sq (norm_nonneg _)]; exact Real.sqrt_le_sqrt (le_trans hseed (by norm_num))
  · rw [← Real.sqrt_sq (norm_nonneg _)]; exact Real.sqrt_le_sqrt (le_trans c1 (by norm_num))
  · rw [← Real.sqrt_sq (norm_nonneg _)]; exact Real.sqrt_le_sqrt (le_trans c2 (by norm_num))
  · rw [← Real.sqrt_sq (norm_nonneg _)]; exact Real.sqrt_le_sqrt (le_trans c3 (by norm_num))
  · rw [← Real.sqrt_sq (norm_nonneg _)]; exact Real.sqrt_le_sqrt (le_trans c4 (by norm_num))
  · rw [← Real.sqrt_sq (norm_nonneg _)]; exact Real.sqrt_le_sqrt (le_trans c5 (by norm_num))

/-! ### The concrete-constant tower for the uniform per-step bound

`nsuStepL`/`nsuStepC` are the explicit (dimension- and `Rm,B`-parameterized) per-step Lipschitz factor and
additive constant, built from the 9 magnitude bounds of `nsIter_step_affine` with `ε ≤ B` substituted. The
intermediate defs mirror the internal `set` locals of `nsIter_step_affine` (`βc`, `βr`, `K`, the `FX/FXt/FA/…`
magnitudes, the `LAX/CAX/LAAX/CAAX` combos) so `nsIter_step_uniform` can discharge `hstep` with these constants
in place of the coefficient- and matrix-dependent bounds. -/

noncomputable def nsuβ (n : Nat) : ℝ := ((2:ℝ) + u64) * (((1:ℝ) + u64) ^ n - 1)
noncomputable def nsuK : ℝ := u64 * (3 + 3 * u64 + u64 ^ 2)
noncomputable def nsuFX (Rm B : ℝ) (c : Nat) : ℝ := Real.sqrt c * (Rm + B)
noncomputable def nsuFXt (Rm B : ℝ) (r : Nat) : ℝ := Real.sqrt r * (Rm + B)
/-- OPERATOR-norm magnitude of `X` (`√dim`-free): `‖toMatrixF X‖₂ ≤ Rm + B`. The Lipschitz-side magnitude. -/
noncomputable def nsuMFX (Rm B : ℝ) : ℝ := Rm + B
noncomputable def nsuEA (Rm B : ℝ) (r c : Nat) : ℝ :=
  (nsuMFX Rm B + Rm) * B + nsuβ c * nsuFX Rm B c * nsuFXt Rm B r
noncomputable def nsuFA (Rm B : ℝ) (r c : Nat) : ℝ := Real.sqrt r * (Rm ^ 2 + nsuEA Rm B r c)
noncomputable def nsuLAX (Rm B : ℝ) (r c : Nat) : ℝ := (nsuMFX Rm B + Rm) * nsuMFX Rm B + Rm ^ 2
noncomputable def nsuCAX (Rm B : ℝ) (r c : Nat) : ℝ :=
  nsuβ r * nsuFA Rm B r c * nsuFX Rm B c + nsuβ c * nsuFX Rm B c * nsuFXt Rm B r * nsuMFX Rm B
noncomputable def nsuEAX (Rm B : ℝ) (r c : Nat) : ℝ := nsuLAX Rm B r c * B + nsuCAX Rm B r c
/-- OPERATOR-norm magnitude of `AX = (X Xᵀ)X` (`√dim`-free): `‖toMatrixF(AX)‖₂ ≤ Rm³ + errAX`. -/
noncomputable def nsuMFAX (Rm B : ℝ) (r c : Nat) : ℝ := Rm ^ 3 + nsuEAX Rm B r c
noncomputable def nsuFCAX (Rm B : ℝ) (r c : Nat) : ℝ := Real.sqrt c * (Rm ^ 3 + nsuEAX Rm B r c)
noncomputable def nsuLAAX (Rm B : ℝ) (r c : Nat) : ℝ :=
  (nsuMFX Rm B + Rm) * nsuMFAX Rm B r c + Rm ^ 2 * nsuLAX Rm B r c
noncomputable def nsuCAAX (Rm B : ℝ) (r c : Nat) : ℝ :=
  nsuβ r * nsuFA Rm B r c * nsuFCAX Rm B r c + nsuβ c * nsuFX Rm B c * nsuFXt Rm B r * nsuMFAX Rm B r c
    + Rm ^ 2 * nsuCAX Rm B r c
noncomputable def nsuEAAX (Rm B : ℝ) (r c : Nat) : ℝ := nsuLAAX Rm B r c * B + nsuCAAX Rm B r c
noncomputable def nsuFAAX (Rm B : ℝ) (r c : Nat) : ℝ := Real.sqrt c * (Rm ^ 5 + nsuEAAX Rm B r c)
noncomputable def nsuStepL (A' B' C' Rm B : ℝ) (r c : Nat) : ℝ :=
  A' + B' * nsuLAX Rm B r c + C' * nsuLAAX Rm B r c
noncomputable def nsuStepC (A' B' C' Rm B : ℝ) (r c : Nat) : ℝ :=
  Real.sqrt 3 * nsuK * (A' * nsuFX Rm B c + B' * nsuFCAX Rm B r c + C' * nsuFAAX Rm B r c)
    + B' * nsuCAX Rm B r c + C' * nsuCAAX Rm B r c

set_option maxHeartbeats 4000000 in
theorem nsIter_step_uniform (X : Mat) (XR : MatR) (r c : Nat) [Nonempty (Fin c)]
    (a b cc : Float) (A' B' C' Rm Bd ε : ℝ)
    (hXsz : X.size = r) (hXrow : ∀ i, i < r → (X[i]!).size = c)
    (hXRsz : XR.size = r) (hXRrow : ∀ i, i < r → (XR[i]!).size = c)
    (hr : 0 < r) (hrc : r ≤ c)
    (hRm : ‖toMatrixR r c XR‖ ≤ Rm) (hRm0 : 0 ≤ Rm) (hBd0 : 0 ≤ Bd)
    (hε : ‖toMatrixF r c X - toMatrixR r c XR‖ ≤ ε) (hε0 : 0 ≤ ε) (hεB : ε ≤ Bd)
    (ha : |toReal a| ≤ A') (hb : |toReal b| ≤ B') (hcp : |toReal cc| ≤ C') :
    ‖toMatrixF r c (nsIter X (a, b, cc)) - toMatrixR r c (nsIterR XR (a, b, cc))‖
      ≤ nsuStepL A' B' C' Rm Bd r c * ε + nsuStepC A' B' C' Rm Bd r c := by
  have hc0 : 0 < c := lt_of_lt_of_le hr hrc
  haveI : Nonempty (Fin r) := ⟨⟨0, hr⟩⟩
  have hu0 : (0:ℝ) ≤ u64 := u64_pos.le
  have hβeq : ∀ n : Nat, ((2:ℝ) + u64) * (((1:ℝ) + u64) ^ n - 1) = nsuβ n := fun _ => rfl
  have hKeq : u64 * (3 + 3 * u64 + u64 ^ 2) = nsuK := rfl
  -- nonnegativity
  have hFX0 : 0 ≤ nsuFX Rm Bd c := by rw [nsuFX]; positivity
  have hFXt0 : 0 ≤ nsuFXt Rm Bd r := by rw [nsuFXt]; positivity
  have hβc0 : 0 ≤ nsuβ c := by rw [nsuβ]; have h1 : (1:ℝ) ≤ (1+u64)^c := one_le_pow₀ (by linarith); nlinarith
  have hβr0 : 0 ≤ nsuβ r := by rw [nsuβ]; have h1 : (1:ℝ) ≤ (1+u64)^r := one_le_pow₀ (by linarith); nlinarith
  have hRm2 : 0 ≤ Rm ^ 2 := pow_nonneg hRm0 2
  have hRm3 : 0 ≤ Rm ^ 3 := pow_nonneg hRm0 3
  have hRm5 : 0 ≤ Rm ^ 5 := pow_nonneg hRm0 5
  have hMFX0 : 0 ≤ nsuMFX Rm Bd := by rw [nsuMFX]; linarith
  have hMFXRm : 0 ≤ nsuMFX Rm Bd + Rm := by rw [nsuMFX]; linarith
  have hEA0 : 0 ≤ nsuEA Rm Bd r c := by
    rw [nsuEA]; exact add_nonneg (mul_nonneg hMFXRm hBd0) (mul_nonneg (mul_nonneg hβc0 hFX0) hFXt0)
  have hFA0 : 0 ≤ nsuFA Rm Bd r c := by
    rw [nsuFA]; exact mul_nonneg (Real.sqrt_nonneg _) (add_nonneg hRm2 hEA0)
  have hLAX0 : 0 ≤ nsuLAX Rm Bd r c := by rw [nsuLAX]; exact add_nonneg (mul_nonneg hMFXRm hMFX0) hRm2
  have hCAX0 : 0 ≤ nsuCAX Rm Bd r c := by
    rw [nsuCAX]
    exact add_nonneg (mul_nonneg (mul_nonneg hβr0 hFA0) hFX0)
      (mul_nonneg (mul_nonneg (mul_nonneg hβc0 hFX0) hFXt0) hMFX0)
  have hEAX0 : 0 ≤ nsuEAX Rm Bd r c := by rw [nsuEAX]; exact add_nonneg (mul_nonneg hLAX0 hBd0) hCAX0
  have hMFAX0 : 0 ≤ nsuMFAX Rm Bd r c := by rw [nsuMFAX]; exact add_nonneg hRm3 hEAX0
  have hFCAX0 : 0 ≤ nsuFCAX Rm Bd r c := by
    rw [nsuFCAX]; exact mul_nonneg (Real.sqrt_nonneg _) (add_nonneg hRm3 hEAX0)
  have hLAAX0 : 0 ≤ nsuLAAX Rm Bd r c := by
    rw [nsuLAAX]; exact add_nonneg (mul_nonneg hMFXRm hMFAX0) (mul_nonneg hRm2 hLAX0)
  have hCAAX0 : 0 ≤ nsuCAAX Rm Bd r c := by
    rw [nsuCAAX]
    exact add_nonneg (add_nonneg (mul_nonneg (mul_nonneg hβr0 hFA0) hFCAX0)
      (mul_nonneg (mul_nonneg (mul_nonneg hβc0 hFX0) hFXt0) hMFAX0)) (mul_nonneg hRm2 hCAX0)
  have hEAAX0 : 0 ≤ nsuEAAX Rm Bd r c := by rw [nsuEAAX]; exact add_nonneg (mul_nonneg hLAAX0 hBd0) hCAAX0
  have hFAAX0 : 0 ≤ nsuFAAX Rm Bd r c := by
    rw [nsuFAAX]; exact mul_nonneg (Real.sqrt_nonneg _) (add_nonneg hRm5 hEAAX0)
  have hK0 : 0 ≤ nsuK := by rw [nsuK]; positivity
  ----- shapes -----
  have hXne : X ≠ #[] := by intro h; rw [h] at hXsz; simp at hXsz; omega
  have hXn : (if X = #[] then 0 else X[0]!.size) = c := by rw [if_neg hXne, hXrow 0 hr]
  have hAsz : (matmul X (Puffer.FloatR.Muon.transpose X)).size = r := by rw [matmul_size]; exact hXsz
  have htXsz : (Puffer.FloatR.Muon.transpose X).size = c := by rw [transpose_size, if_neg hXne, hXrow 0 hr]
  have htXne : Puffer.FloatR.Muon.transpose X ≠ #[] := by intro h; rw [h] at htXsz; simp at htXsz; omega
  have hAne : matmul X (Puffer.FloatR.Muon.transpose X) ≠ #[] := by intro h; rw [h] at hAsz; simp at hAsz; omega
  have hAk : (if matmul X (Puffer.FloatR.Muon.transpose X) = #[] then 0 else ((matmul X (Puffer.FloatR.Muon.transpose X))[0]!).size) = r := by
    rw [if_neg hAne, matmul_rowSize X (Puffer.FloatR.Muon.transpose X) 0 (by rw [hXsz]; exact hr), if_neg htXne,
      transpose_rowSize X 0 (by rw [if_neg hXne, hXrow 0 hr]; exact hc0)]
    exact hXsz
  have hARsz : (matmulR XR (transposeR XR)).size = r := by rw [matmulR_size]; exact hXRsz
  have htXRsz : (transposeR XR).size = c := by rw [transposeR_size, if_neg (by rw [hXRsz]; omega), hXRrow 0 hr]
  have htXRrow : ∀ i, i < c → ((transposeR XR)[i]!).size = r := by
    intro i hi
    rw [transposeR_rowSize XR i (by rw [if_neg (by rw [hXRsz]; omega), hXRrow 0 hr]; exact hi)]; exact hXRsz
  have hARrow : ∀ i, i < r → ((matmulR XR (transposeR XR))[i]!).size = r := by
    intro i hi
    rw [matmulR_rowSize XR (transposeR XR) i (by rw [hXRsz]; exact hi), if_neg (by rw [htXRsz]; omega),
      transposeR_rowSize XR 0 (by rw [if_neg (by rw [hXRsz]; omega), hXRrow 0 hr]; exact hc0)]
    exact hXRsz
  have hAXsz : (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X).size = r := by rw [matmul_size]; exact hAsz
  have hAXne : matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X ≠ #[] := by intro h; rw [h] at hAXsz; simp at hAXsz; omega
  have hAXn : (if matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X = #[] then 0 else ((matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X)[0]!).size) = c := by
    rw [if_neg hAXne, matmul_rowSize (matmul X (Puffer.FloatR.Muon.transpose X)) X 0 (by rw [hAsz]; exact hr), if_neg hXne, hXrow 0 hr]
  have hAXRsz : (matmulR (matmulR XR (transposeR XR)) XR).size = r := by rw [matmulR_size]; exact hARsz
  have hAXRrow : ∀ i, i < r → ((matmulR (matmulR XR (transposeR XR)) XR)[i]!).size = c := by
    intro i hi
    rw [matmulR_rowSize (matmulR XR (transposeR XR)) XR i (by rw [hARsz]; exact hi), if_neg (by rw [hXRsz]; omega), hXRrow 0 hr]
  ----- easy magnitudes -----
  have hFX : frobRow X r c ≤ nsuFX Rm Bd c := by
    rw [nsuFX]; refine le_trans (frobRow_le_mirror_add_err X XR r c ε hε) ?_; gcongr
  have hFCX : frobCol X c r ≤ nsuFX Rm Bd c := by
    rw [nsuFX]; refine le_trans (frobCol_le_mirror_add_err X XR r c ε hε) ?_; gcongr
  have hdiffT : ‖toMatrixF c r (Puffer.FloatR.Muon.transpose X) - toMatrixR c r (transposeR XR)‖ ≤ ε :=
    transpose_error_opNorm X XR r c ε hXsz hXrow hXRsz hXRrow hr hε
  have hFXt : frobCol (Puffer.FloatR.Muon.transpose X) r c ≤ nsuFXt Rm Bd r := by
    rw [nsuFXt]
    refine le_trans (frobCol_le_mirror_add_err (Puffer.FloatR.Muon.transpose X) (transposeR XR) c r ε hdiffT) ?_
    rw [mirror_transpose_opNorm XR r c hXRsz hXRrow hr]; gcongr
  have hMA : ‖toMatrixR r r (matmulR XR (transposeR XR))‖ ≤ Rm ^ 2 := by
    refine le_trans (mirror_mul_opNorm XR (transposeR XR) r c r hXRsz hXRrow htXRsz htXRrow hr hc0) ?_
    rw [mirror_transpose_opNorm XR r c hXRsz hXRrow hr]
    calc ‖toMatrixR r c XR‖ * ‖toMatrixR r c XR‖ ≤ Rm * Rm := mul_le_mul hRm hRm (norm_nonneg _) hRm0
      _ = Rm ^ 2 := by ring
  have hMFX : ‖toMatrixF r c X‖ ≤ nsuMFX Rm Bd := by
    rw [nsuMFX]; refine le_trans (opNorm_le_mirror_add_err X XR r c ε hε) ?_; linarith [hRm, hεB]
  ----- gram affine error + FA -----
  have hepsA_aff : ‖toMatrixF r r (matmul X (Puffer.FloatR.Muon.transpose X)) - toMatrixR r r (matmulR XR (transposeR XR))‖
      ≤ (nsuMFX Rm Bd + Rm) * ε + nsuβ c * nsuFX Rm Bd c * nsuFXt Rm Bd r := by
    have h := gram_step_affine X XR r c ε (nsuFX Rm Bd c) (nsuFXt Rm Bd r) (nsuMFX Rm Bd) Rm hXsz hXrow hXRsz hXRrow hr hc0 hε hFX hFXt hMFX hRm hε0 hFX0
    rwa [hβeq c] at h
  have herrA : ‖toMatrixF r r (matmul X (Puffer.FloatR.Muon.transpose X)) - toMatrixR r r (matmulR XR (transposeR XR))‖
      ≤ nsuEA Rm Bd r c := by
    refine le_trans hepsA_aff ?_; rw [nsuEA]; gcongr
  have hFA : frobRow (matmul X (Puffer.FloatR.Muon.transpose X)) r r ≤ nsuFA Rm Bd r c := by
    rw [nsuFA]
    refine le_trans (frobRow_le_mirror_add_err (matmul X (Puffer.FloatR.Muon.transpose X)) (matmulR XR (transposeR XR)) r r (nsuEA Rm Bd r c) herrA) ?_
    gcongr
  ----- AX affine error + FCAX/FAX -----
  have hXerr_aff : ‖toMatrixF r c X - toMatrixR r c XR‖ ≤ (1:ℝ) * ε + 0 := by linarith
  have hMFX0 : 0 ≤ nsuMFX Rm Bd := by rw [nsuMFX]; linarith
  have hLAeps0 : 0 ≤ (nsuMFX Rm Bd + Rm) * ε + nsuβ c * nsuFX Rm Bd c * nsuFXt Rm Bd r := by positivity
  have hLBeps0 : 0 ≤ (1:ℝ) * ε + 0 := by linarith
  have hepsAX_aff : ‖toMatrixF r c (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X)
      - toMatrixR r c (matmulR (matmulR XR (transposeR XR)) XR)‖ ≤ nsuLAX Rm Bd r c * ε + nsuCAX Rm Bd r c := by
    have h := matmul_step_affine (matmul X (Puffer.FloatR.Muon.transpose X)) X (matmulR XR (transposeR XR)) XR r r c ε
      (nsuMFX Rm Bd + Rm) (nsuβ c * nsuFX Rm Bd c * nsuFXt Rm Bd r) 1 0
      (nsuFA Rm Bd r c) (nsuFX Rm Bd c) (nsuMFX Rm Bd) (Rm ^ 2)
      hAsz hAk hXn hARsz hARrow hXRsz hXRrow hr hr hepsA_aff hXerr_aff hFA hFCX hMFX hMA hFA0 hLAeps0 hLBeps0
    rw [hβeq r] at h
    refine le_trans h (le_of_eq ?_); rw [nsuLAX, nsuCAX]; ring
  have herrAX : ‖toMatrixF r c (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X)
      - toMatrixR r c (matmulR (matmulR XR (transposeR XR)) XR)‖ ≤ nsuEAX Rm Bd r c := by
    refine le_trans hepsAX_aff ?_; rw [nsuEAX]; gcongr
  have hM_AX : ‖toMatrixR r c (matmulR (matmulR XR (transposeR XR)) XR)‖ ≤ Rm ^ 3 := by
    refine le_trans (mirror_mul_opNorm (matmulR XR (transposeR XR)) XR r r c hARsz hARrow hXRsz hXRrow hr hr) ?_
    calc ‖toMatrixR r r (matmulR XR (transposeR XR))‖ * ‖toMatrixR r c XR‖
        ≤ Rm ^ 2 * Rm := mul_le_mul hMA hRm (norm_nonneg _) hRm2
      _ = Rm ^ 3 := by ring
  have hFCAX : frobCol (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X) c r ≤ nsuFCAX Rm Bd r c := by
    rw [nsuFCAX]
    refine le_trans (frobCol_le_mirror_add_err (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X) (matmulR (matmulR XR (transposeR XR)) XR) r c (nsuEAX Rm Bd r c) herrAX) ?_
    gcongr
  have hFAX : frobRow (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X) r c ≤ nsuFCAX Rm Bd r c := by
    rw [nsuFCAX]
    refine le_trans (frobRow_le_mirror_add_err (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X) (matmulR (matmulR XR (transposeR XR)) XR) r c (nsuEAX Rm Bd r c) herrAX) ?_
    gcongr
  have hMFAX : ‖toMatrixF r c (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X)‖ ≤ nsuMFAX Rm Bd r c := by
    rw [nsuMFAX]
    refine le_trans (opNorm_le_mirror_add_err (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X)
      (matmulR (matmulR XR (transposeR XR)) XR) r c (nsuEAX Rm Bd r c) herrAX) ?_
    linarith [hM_AX]
  ----- AAX affine error + FAAX -----
  have hepsAAX_aff : ‖toMatrixF r c (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X))
      - toMatrixR r c (matmulR (matmulR XR (transposeR XR)) (matmulR (matmulR XR (transposeR XR)) XR))‖
      ≤ nsuLAAX Rm Bd r c * ε + nsuCAAX Rm Bd r c := by
    have h := matmul_step_affine (matmul X (Puffer.FloatR.Muon.transpose X)) (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X)
      (matmulR XR (transposeR XR)) (matmulR (matmulR XR (transposeR XR)) XR) r r c ε
      (nsuMFX Rm Bd + Rm) (nsuβ c * nsuFX Rm Bd c * nsuFXt Rm Bd r) (nsuLAX Rm Bd r c) (nsuCAX Rm Bd r c)
      (nsuFA Rm Bd r c) (nsuFCAX Rm Bd r c) (nsuMFAX Rm Bd r c) (Rm ^ 2)
      hAsz hAk hAXn hARsz hARrow hAXRsz hAXRrow hr hr hepsA_aff hepsAX_aff hFA hFCAX hMFAX hMA hFA0 hLAeps0
      (by positivity)
    rw [hβeq r] at h
    refine le_trans h (le_of_eq ?_); rw [nsuLAAX, nsuCAAX]
  have herrAAX : ‖toMatrixF r c (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X))
      - toMatrixR r c (matmulR (matmulR XR (transposeR XR)) (matmulR (matmulR XR (transposeR XR)) XR))‖
      ≤ nsuEAAX Rm Bd r c := by
    refine le_trans hepsAAX_aff ?_; rw [nsuEAAX]; gcongr
  have hFAAX : frobRow (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X)) r c ≤ nsuFAAX Rm Bd r c := by
    rw [nsuFAAX]
    refine le_trans (frobRow_le_mirror_add_err (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) (matmul (matmul X (Puffer.FloatR.Muon.transpose X)) X)) (matmulR (matmulR XR (transposeR XR)) (matmulR (matmulR XR (transposeR XR)) XR)) r c (nsuEAAX Rm Bd r c) herrAAX) ?_
    have hM_AAX : ‖toMatrixR r c (matmulR (matmulR XR (transposeR XR)) (matmulR (matmulR XR (transposeR XR)) XR))‖ ≤ Rm ^ 5 := by
      refine le_trans (mirror_mul_opNorm (matmulR XR (transposeR XR)) (matmulR (matmulR XR (transposeR XR)) XR) r r c hARsz hARrow hAXRsz hAXRrow hr hr) ?_
      calc ‖toMatrixR r r (matmulR XR (transposeR XR))‖ * ‖toMatrixR r c (matmulR (matmulR XR (transposeR XR)) XR)‖
          ≤ Rm ^ 2 * Rm ^ 3 := mul_le_mul hMA hM_AX (norm_nonneg _) hRm2
        _ = Rm ^ 5 := by ring
    gcongr
  ----- combine via nsIter_step_affine -----
  have hmain := nsIter_step_affine X XR r c a b cc ε
    (nsuFX Rm Bd c) (nsuFXt Rm Bd r) Rm (nsuFA Rm Bd r c) (nsuFX Rm Bd c) (Rm ^ 2)
    (nsuFCAX Rm Bd r c) (nsuFCAX Rm Bd r c) (nsuFAAX Rm Bd r c) (nsuMFX Rm Bd) (nsuMFAX Rm Bd r c)
    hXsz hXrow hXRsz hXRrow hr hrc hε hFX hFXt hRm hMFX hMFAX hFA hFCX hMA hFCAX hFAX hFAAX
    hε0 hFX0 hFXt0 hRm0 hFA0 hFX0 hRm2
  simp only [hβeq, hKeq] at hmain
  refine le_trans hmain ?_
  simp only [nsuStepL, nsuStepC, nsuLAX, nsuLAAX, nsuCAX, nsuCAAX]
  gcongr


/-! ### Shape preservation + the shape/predicate-threaded expanding fold

`nsIter`/`nsIterR` preserve the `r × c` shape (both branches end in `lincomb3 a X b _ c _`), so the shape is a
fold invariant. `fold_affine_mirror_exp_shape` extends `fold_affine_mirror_exp` with (i) a coefficient
predicate `P` (threaded via `List.mem_cons`, so the per-step bound may assume the schedule's coefficient bounds
rather than holding for all coefficients) and (ii) shape threading (so the per-step bound receives the current
iterate's shape — which `nsIter_step_uniform` needs). This is exactly the interface the concrete capstone
consumes: `hstep` from `nsIter_step_uniform`, `hmir` from `nsIterR_prefix_normle`. -/

theorem nsIter_size (X : Mat) (coef : Float × Float × Float) : (nsIter X coef).size = X.size := by
  obtain ⟨a, b, cc⟩ := coef
  dsimp only [nsIter]
  split <;> exact Puffer.RL.NewtonSchulzError.lincomb3_size _ _ _ _ _ _

theorem nsIter_rowSize (X : Mat) (coef : Float × Float × Float) (i : Nat) (hi : i < X.size) :
    ((nsIter X coef)[i]!).size = (X[i]!).size := by
  obtain ⟨a, b, cc⟩ := coef
  dsimp only [nsIter]
  split <;> exact Puffer.RL.NewtonSchulzError.lincomb3_rowSize _ _ _ _ _ _ i hi

theorem fold_affine_mirror_exp_shape {β : Type} (f : Mat → β → Mat) (fR : MatR → β → MatR)
    (r c : Nat) (L C B Rm : ℝ) (P : β → Prop) (hL1 : 1 ≤ L) (hC0 : 0 ≤ C)
    (hfpres : ∀ (X : Mat) (coef : β), X.size = r → (∀ i, i < r → (X[i]!).size = c) →
       (f X coef).size = r ∧ (∀ i, i < r → ((f X coef)[i]!).size = c))
    (hfRpres : ∀ (XR : MatR) (coef : β), XR.size = r → (∀ i, i < r → (XR[i]!).size = c) →
       (fR XR coef).size = r ∧ (∀ i, i < r → ((fR XR coef)[i]!).size = c))
    (hstep : ∀ (X : Mat) (XR : MatR) (coef : β) (ε : ℝ), P coef →
       X.size = r → (∀ i, i < r → (X[i]!).size = c) →
       XR.size = r → (∀ i, i < r → (XR[i]!).size = c) →
       ε ≤ B → ‖toMatrixR r c XR‖ ≤ Rm → ‖toMatrixF r c X - toMatrixR r c XR‖ ≤ ε →
       ‖toMatrixF r c (f X coef) - toMatrixR r c (fR XR coef)‖ ≤ L * ε + C) :
    ∀ (coeffs : List β) (seed : Mat) (seedR : MatR) (ε0 : ℝ),
      (∀ coef ∈ coeffs, P coef) →
      seed.size = r → (∀ i, i < r → (seed[i]!).size = c) →
      seedR.size = r → (∀ i, i < r → (seedR[i]!).size = c) →
      L ^ coeffs.length * ε0 + C * (∑ k ∈ Finset.range coeffs.length, L ^ k) ≤ B →
      ‖toMatrixF r c seed - toMatrixR r c seedR‖ ≤ ε0 →
      (∀ m, m ≤ coeffs.length → ‖toMatrixR r c ((coeffs.take m).foldl fR seedR)‖ ≤ Rm) →
      ‖toMatrixF r c (coeffs.foldl f seed) - toMatrixR r c (coeffs.foldl fR seedR)‖
        ≤ L ^ coeffs.length * ε0 + C * (∑ k ∈ Finset.range coeffs.length, L ^ k) := by
  intro coeffs
  induction coeffs with
  | nil => intro seed seedR ε0 _ _ _ _ _ _ h _; simpa using h
  | cons coef rest ih =>
      intro seed seedR ε0 hP hSsz hSrow hRsz hRrow hB h hmir
      have hPhead : P coef := hP coef (List.mem_cons_self ..)
      have hPtail : ∀ c ∈ rest, P c := fun c hc => hP c (List.mem_cons_of_mem coef hc)
      have hε00 : 0 ≤ ε0 := le_trans (norm_nonneg _) h
      have hLm : (1:ℝ) ≤ L ^ (rest.length + 1) := one_le_pow₀ hL1
      have hgeom0 : 0 ≤ C * (∑ k ∈ Finset.range (rest.length + 1), L ^ k) :=
        mul_nonneg hC0 (Finset.sum_nonneg (fun k _ => pow_nonneg (by linarith) k))
      have hε0B : ε0 ≤ B := by
        have h1 : ε0 ≤ L ^ (rest.length + 1) * ε0 := le_mul_of_one_le_left hε00 hLm
        rw [List.length_cons] at hB; linarith
      have hmir0 : ‖toMatrixR r c seedR‖ ≤ Rm := by
        have := hmir 0 (Nat.zero_le _); simpa using this
      have hstep0 := hstep seed seedR coef ε0 hPhead hSsz hSrow hRsz hRrow hε0B hmir0 h
      obtain ⟨hFsz', hFrow'⟩ := hfpres seed coef hSsz hSrow
      obtain ⟨hRsz', hRrow'⟩ := hfRpres seedR coef hRsz hRrow
      have hBtail : L ^ rest.length * (L * ε0 + C) + C * (∑ k ∈ Finset.range rest.length, L ^ k) ≤ B := by
        have heq : L ^ rest.length * (L * ε0 + C) + C * (∑ k ∈ Finset.range rest.length, L ^ k)
            = L ^ (rest.length + 1) * ε0 + C * (∑ k ∈ Finset.range (rest.length + 1), L ^ k) := by
          rw [Finset.sum_range_succ, pow_succ]; ring
        rw [heq]; rw [List.length_cons] at hB; exact hB
      have hmir' : ∀ m, m ≤ rest.length →
          ‖toMatrixR r c ((rest.take m).foldl fR (fR seedR coef))‖ ≤ Rm := by
        intro m hm
        have h2 := hmir (m + 1) (Nat.succ_le_succ hm)
        rwa [List.take_succ_cons, List.foldl_cons] at h2
      have hrec := ih (f seed coef) (fR seedR coef) (L * ε0 + C) hPtail hFsz' hFrow' hRsz' hRrow' hBtail hstep0 hmir'
      simp only [List.foldl_cons, List.length_cons]
      refine le_trans hrec (le_of_eq ?_)
      rw [Finset.sum_range_succ, pow_succ]; ring


/-! ### The per-step NON-UNIFORM fold — tighter than the uniform `L⁵·ε₀ + C·∑Lᵏ`

The uniform fold pays the worst-case `L` on every step. But the later Newton–Schulz steps have much smaller
coefficients (`|a|,|b|,|c|`), hence smaller per-step Lipschitz factors. `fold_affine_mirror_exp_perstep` carries
per-step functions `Lf, Cf : β → ℝ` (evaluated at each step's own coefficient), and folds the recurrence
`εₖ ≤ Lf cₖ · εₖ₋₁ + Cf cₖ` EXACTLY to `coeffs.foldl (fun acc coef => Lf coef · acc + Cf coef) ε₀` — the
`(∏ Lf cᵢ)·ε₀ + ∑ⱼ Cf cⱼ·∏_{i>j} Lf cᵢ` form, no worst-case inflation. `foldl_affine_ge` (monotonicity of the
affine fold when `Lf ≥ 1`, `Cf ≥ 0`) gives the `εₖ ≤ B` threshold from the final bound. -/

theorem foldl_affine_ge {β : Type} (Lf Cf : β → ℝ) :
    ∀ (l : List β) (x : ℝ), (∀ coef ∈ l, 1 ≤ Lf coef) → (∀ coef ∈ l, 0 ≤ Cf coef) → 0 ≤ x →
      x ≤ l.foldl (fun acc coef => Lf coef * acc + Cf coef) x := by
  intro l
  induction l with
  | nil => intro x _ _ hx; simpa using le_refl x
  | cons coef rest ih =>
      intro x hL hC hx
      have hLhead : 1 ≤ Lf coef := hL coef (List.mem_cons_self ..)
      have hChead : 0 ≤ Cf coef := hC coef (List.mem_cons_self ..)
      have hgx : x ≤ Lf coef * x + Cf coef := by nlinarith [hLhead, hChead, hx]
      have hgx0 : 0 ≤ Lf coef * x + Cf coef := by nlinarith [hLhead, hChead, hx]
      have hrec := ih (Lf coef * x + Cf coef)
        (fun c hc => hL c (List.mem_cons_of_mem coef hc)) (fun c hc => hC c (List.mem_cons_of_mem coef hc)) hgx0
      simp only [List.foldl_cons]
      linarith [hgx, hrec]

theorem fold_affine_mirror_exp_perstep {β : Type} (f : Mat → β → Mat) (fR : MatR → β → MatR)
    (r c : Nat) (B Rm : ℝ) (Lf Cf : β → ℝ) (P : β → Prop)
    (hL1 : ∀ coef, P coef → 1 ≤ Lf coef) (hC0 : ∀ coef, P coef → 0 ≤ Cf coef)
    (hfpres : ∀ (X : Mat) (coef : β), X.size = r → (∀ i, i < r → (X[i]!).size = c) →
       (f X coef).size = r ∧ (∀ i, i < r → ((f X coef)[i]!).size = c))
    (hfRpres : ∀ (XR : MatR) (coef : β), XR.size = r → (∀ i, i < r → (XR[i]!).size = c) →
       (fR XR coef).size = r ∧ (∀ i, i < r → ((fR XR coef)[i]!).size = c))
    (hstep : ∀ (X : Mat) (XR : MatR) (coef : β) (ε : ℝ), P coef →
       X.size = r → (∀ i, i < r → (X[i]!).size = c) →
       XR.size = r → (∀ i, i < r → (XR[i]!).size = c) →
       ε ≤ B → ‖toMatrixR r c XR‖ ≤ Rm → ‖toMatrixF r c X - toMatrixR r c XR‖ ≤ ε →
       ‖toMatrixF r c (f X coef) - toMatrixR r c (fR XR coef)‖ ≤ Lf coef * ε + Cf coef) :
    ∀ (coeffs : List β) (seed : Mat) (seedR : MatR) (ε0 : ℝ),
      (∀ coef ∈ coeffs, P coef) →
      seed.size = r → (∀ i, i < r → (seed[i]!).size = c) →
      seedR.size = r → (∀ i, i < r → (seedR[i]!).size = c) →
      coeffs.foldl (fun acc coef => Lf coef * acc + Cf coef) ε0 ≤ B →
      ‖toMatrixF r c seed - toMatrixR r c seedR‖ ≤ ε0 →
      (∀ m, m ≤ coeffs.length → ‖toMatrixR r c ((coeffs.take m).foldl fR seedR)‖ ≤ Rm) →
      ‖toMatrixF r c (coeffs.foldl f seed) - toMatrixR r c (coeffs.foldl fR seedR)‖
        ≤ coeffs.foldl (fun acc coef => Lf coef * acc + Cf coef) ε0 := by
  intro coeffs
  induction coeffs with
  | nil => intro seed seedR ε0 _ _ _ _ _ _ h _; simpa using h
  | cons coef rest ih =>
      intro seed seedR ε0 hP hSsz hSrow hRsz hRrow hB h hmir
      have hPhead : P coef := hP coef (List.mem_cons_self ..)
      have hPtail : ∀ c ∈ rest, P c := fun c hc => hP c (List.mem_cons_of_mem coef hc)
      have hLhead : 1 ≤ Lf coef := hL1 coef hPhead
      have hChead : 0 ≤ Cf coef := hC0 coef hPhead
      have hε00 : 0 ≤ ε0 := le_trans (norm_nonneg _) h
      have hgε0 : 0 ≤ Lf coef * ε0 + Cf coef := by nlinarith [hLhead, hChead, hε00]
      -- B threshold: (coef::rest).foldl = rest.foldl (Lf coef * ε0 + Cf coef)
      rw [List.foldl_cons] at hB
      have hBtail : rest.foldl (fun acc coef => Lf coef * acc + Cf coef) (Lf coef * ε0 + Cf coef) ≤ B := hB
      have hge := foldl_affine_ge Lf Cf rest (Lf coef * ε0 + Cf coef)
        (fun c hc => hL1 c (hPtail c hc)) (fun c hc => hC0 c (hPtail c hc)) hgε0
      have hε0B : ε0 ≤ B := by nlinarith [hLhead, hChead, hε00, hge, hBtail]
      have hmir0 : ‖toMatrixR r c seedR‖ ≤ Rm := by
        have := hmir 0 (Nat.zero_le _); simpa using this
      have hstep0 := hstep seed seedR coef ε0 hPhead hSsz hSrow hRsz hRrow hε0B hmir0 h
      obtain ⟨hFsz', hFrow'⟩ := hfpres seed coef hSsz hSrow
      obtain ⟨hRsz', hRrow'⟩ := hfRpres seedR coef hRsz hRrow
      have hmir' : ∀ m, m ≤ rest.length →
          ‖toMatrixR r c ((rest.take m).foldl fR (fR seedR coef))‖ ≤ Rm := by
        intro m hm
        have h2 := hmir (m + 1) (Nat.succ_le_succ hm)
        rwa [List.take_succ_cons, List.foldl_cons] at h2
      have hrec := ih (f seed coef) (fR seedR coef) (Lf coef * ε0 + Cf coef) hPtail hFsz' hFrow' hRsz' hRrow' hBtail hstep0 hmir'
      simp only [List.foldl_cons]
      exact hrec


/-! ### The concrete-constants capstone — no structural hypotheses

`newtonSchulz_opNorm_concrete` is `newtonSchulz_opNorm_mirror_exp` with BOTH structural (∀-over-arbitrary-matrix)
hypotheses discharged: `hmir` by `nsIterR_prefix_normle` (fixing `Rm := √1.63`) and `hstep` by
`nsIter_step_uniform` (fixing `L := nsuStepL …`, `C := nsuStepC …`), wired through
`fold_affine_mirror_exp_shape` (which threads the iterate shapes `nsIter_step_uniform` needs and restricts the
per-step bound to the schedule coefficients via the predicate `P`). What remains are SCALAR / shape side
conditions only: the seed shape + normalization, the seed rounding error `ε0`, the coefficient-schedule bounds
`hcoef` (bound the 5 concrete `muonCoeffs` tuples by `A',B',C'`), `hL1` (`1 ≤ L`), and the fixpoint
`hB` (`L⁵·ε0 + C·∑Lᵏ ≤ B`). `nsuStepC_nonneg` discharges `hC0` internally. The bound is the runnable
`newtonSchulz`'s operator norm `≤ √1.3131 + (L⁵·ε0 + C·(L⁴+…+1))`, polynomial in the Float-rounding unit — no
∀-quantified matrix hypothesis survives. -/

theorem nsuStepC_nonneg (A' B' C' Rm Bd : ℝ) (r c : Nat)
    (hA'0 : 0 ≤ A') (hB'0 : 0 ≤ B') (hC'0 : 0 ≤ C') (hRm0 : 0 ≤ Rm) (hBd0 : 0 ≤ Bd) :
    0 ≤ nsuStepC A' B' C' Rm Bd r c := by
  have hu0 : (0:ℝ) ≤ u64 := u64_pos.le
  have hFX0 : 0 ≤ nsuFX Rm Bd c := by rw [nsuFX]; positivity
  have hFXt0 : 0 ≤ nsuFXt Rm Bd r := by rw [nsuFXt]; positivity
  have hβc0 : 0 ≤ nsuβ c := by rw [nsuβ]; have h1 : (1:ℝ) ≤ (1+u64)^c := one_le_pow₀ (by linarith); nlinarith
  have hβr0 : 0 ≤ nsuβ r := by rw [nsuβ]; have h1 : (1:ℝ) ≤ (1+u64)^r := one_le_pow₀ (by linarith); nlinarith
  have hRm2 : 0 ≤ Rm ^ 2 := pow_nonneg hRm0 2
  have hMFX0 : 0 ≤ nsuMFX Rm Bd := by rw [nsuMFX]; linarith
  have hMFXRm : 0 ≤ nsuMFX Rm Bd + Rm := by rw [nsuMFX]; linarith
  have hEA0 : 0 ≤ nsuEA Rm Bd r c := by
    rw [nsuEA]; exact add_nonneg (mul_nonneg hMFXRm hBd0) (mul_nonneg (mul_nonneg hβc0 hFX0) hFXt0)
  have hFA0 : 0 ≤ nsuFA Rm Bd r c := by rw [nsuFA]; exact mul_nonneg (Real.sqrt_nonneg _) (add_nonneg hRm2 hEA0)
  have hLAX0 : 0 ≤ nsuLAX Rm Bd r c := by rw [nsuLAX]; exact add_nonneg (mul_nonneg hMFXRm hMFX0) hRm2
  have hCAX0 : 0 ≤ nsuCAX Rm Bd r c := by
    rw [nsuCAX]; exact add_nonneg (mul_nonneg (mul_nonneg hβr0 hFA0) hFX0)
      (mul_nonneg (mul_nonneg (mul_nonneg hβc0 hFX0) hFXt0) hMFX0)
  have hEAX0 : 0 ≤ nsuEAX Rm Bd r c := by rw [nsuEAX]; exact add_nonneg (mul_nonneg hLAX0 hBd0) hCAX0
  have hRm3 : 0 ≤ Rm ^ 3 := pow_nonneg hRm0 3
  have hMFAX0 : 0 ≤ nsuMFAX Rm Bd r c := by rw [nsuMFAX]; exact add_nonneg hRm3 hEAX0
  have hFCAX0 : 0 ≤ nsuFCAX Rm Bd r c := by
    rw [nsuFCAX]; exact mul_nonneg (Real.sqrt_nonneg _) (add_nonneg hRm3 hEAX0)
  have hLAAX0 : 0 ≤ nsuLAAX Rm Bd r c := by
    rw [nsuLAAX]; exact add_nonneg (mul_nonneg hMFXRm hMFAX0) (mul_nonneg hRm2 hLAX0)
  have hCAAX0 : 0 ≤ nsuCAAX Rm Bd r c := by
    rw [nsuCAAX]; exact add_nonneg (add_nonneg (mul_nonneg (mul_nonneg hβr0 hFA0) hFCAX0)
      (mul_nonneg (mul_nonneg (mul_nonneg hβc0 hFX0) hFXt0) hMFAX0)) (mul_nonneg hRm2 hCAX0)
  have hRm5 : 0 ≤ Rm ^ 5 := pow_nonneg hRm0 5
  have hEAAX0 : 0 ≤ nsuEAAX Rm Bd r c := by rw [nsuEAAX]; exact add_nonneg (mul_nonneg hLAAX0 hBd0) hCAAX0
  have hFAAX0 : 0 ≤ nsuFAAX Rm Bd r c := by
    rw [nsuFAAX]; exact mul_nonneg (Real.sqrt_nonneg _) (add_nonneg hRm5 hEAAX0)
  have hK0 : 0 ≤ nsuK := by rw [nsuK]; positivity
  rw [nsuStepC]
  refine add_nonneg (add_nonneg ?_ (mul_nonneg hB'0 hCAX0)) (mul_nonneg hC'0 hCAAX0)
  exact mul_nonneg (mul_nonneg (Real.sqrt_nonneg _) hK0)
    (add_nonneg (add_nonneg (mul_nonneg hA'0 hFX0) (mul_nonneg hB'0 hFCAX0)) (mul_nonneg hC'0 hFAAX0))

set_option maxHeartbeats 1000000 in
theorem newtonSchulz_opNorm_concrete (X0 : Mat) (eps : Float) (seedR : MatR) (r cc : Nat)
    (A' B' C' Bd ε0 : ℝ) (hr : 0 < r) (hrc : r ≤ cc) (hBd0 : 0 ≤ Bd)
    (hA'0 : 0 ≤ A') (hB'0 : 0 ≤ B') (hC'0 : 0 ≤ C')
    (hX0sz : X0.size = r) (hX0row : ∀ i, i < r → (X0[i]!).size = cc)
    (hSsz : seedR.size = r) (hSrow : ∀ i, i < r → (seedR[i]!).size = cc)
    (hseedNorm : ‖toMatrixR r cc seedR‖ ^ 2 ≤ 1)
    (hcoef : ∀ coef ∈ muonCoeffs.toList,
       |toReal coef.1| ≤ A' ∧ |toReal coef.2.1| ≤ B' ∧ |toReal coef.2.2| ≤ C')
    (hL1 : 1 ≤ nsuStepL A' B' C' (Real.sqrt 1.63) Bd r cc)
    (hB : nsuStepL A' B' C' (Real.sqrt 1.63) Bd r cc ^ muonCoeffs.toList.length * ε0
        + nsuStepC A' B' C' (Real.sqrt 1.63) Bd r cc
            * (∑ k ∈ Finset.range muonCoeffs.toList.length, nsuStepL A' B' C' (Real.sqrt 1.63) Bd r cc ^ k) ≤ Bd)
    (hseedErr : ‖toMatrixF r cc (scalarMul (1.0 / (frobNorm X0 + eps)) X0) - toMatrixR r cc seedR‖ ≤ ε0) :
    ‖toMatrixF r cc (newtonSchulz X0 eps)‖
      ≤ Real.sqrt 1.3131
        + (nsuStepL A' B' C' (Real.sqrt 1.63) Bd r cc ^ muonCoeffs.toList.length * ε0
           + nsuStepC A' B' C' (Real.sqrt 1.63) Bd r cc
               * (∑ k ∈ Finset.range muonCoeffs.toList.length, nsuStepL A' B' C' (Real.sqrt 1.63) Bd r cc ^ k)) := by
  haveI : Nonempty (Fin cc) := ⟨⟨0, lt_of_lt_of_le hr hrc⟩⟩
  set L := nsuStepL A' B' C' (Real.sqrt 1.63) Bd r cc with hLdef
  set Cst := nsuStepC A' B' C' (Real.sqrt 1.63) Bd r cc with hCdef
  have hRm0 : (0:ℝ) ≤ Real.sqrt 1.63 := Real.sqrt_nonneg _
  have hC0 : 0 ≤ Cst := by
    rw [hCdef]; exact nsuStepC_nonneg A' B' C' (Real.sqrt 1.63) Bd r cc hA'0 hB'0 hC'0 hRm0 hBd0
  have hFSsz : (scalarMul (1.0 / (frobNorm X0 + eps)) X0).size = r := by
    rw [Puffer.RL.NewtonSchulzError.scalarMul_size]; exact hX0sz
  have hFSrow : ∀ i, i < r → ((scalarMul (1.0 / (frobNorm X0 + eps)) X0)[i]!).size = cc := by
    intro i hi
    rw [Puffer.RL.NewtonSchulzError.scalarMul_rowSize _ X0 i (by rw [hX0sz]; exact hi)]; exact hX0row i hi
  have hfpres : ∀ (X : Mat) (coef : Float × Float × Float), X.size = r → (∀ i, i < r → (X[i]!).size = cc) →
      (nsIter X coef).size = r ∧ (∀ i, i < r → ((nsIter X coef)[i]!).size = cc) := by
    intro X coef hXsz hXrow
    exact ⟨by rw [nsIter_size, hXsz], fun i hi => by rw [nsIter_rowSize X coef i (by rw [hXsz]; exact hi), hXrow i hi]⟩
  have hfRpres : ∀ (XR : MatR) (coef : Float × Float × Float), XR.size = r → (∀ i, i < r → (XR[i]!).size = cc) →
      (nsIterR XR coef).size = r ∧ (∀ i, i < r → ((nsIterR XR coef)[i]!).size = cc) := by
    intro XR coef hXRsz hXRrow
    exact ⟨by rw [nsIterR_size, hXRsz], fun i hi => by rw [nsIterR_rowSize XR coef i (by rw [hXRsz]; exact hi), hXRrow i hi]⟩
  have hstep : ∀ (X : Mat) (XR : MatR) (coef : Float × Float × Float) (ε : ℝ),
      (fun coef : Float × Float × Float => |toReal coef.1| ≤ A' ∧ |toReal coef.2.1| ≤ B' ∧ |toReal coef.2.2| ≤ C') coef →
      X.size = r → (∀ i, i < r → (X[i]!).size = cc) →
      XR.size = r → (∀ i, i < r → (XR[i]!).size = cc) →
      ε ≤ Bd → ‖toMatrixR r cc XR‖ ≤ Real.sqrt 1.63 →
      ‖toMatrixF r cc X - toMatrixR r cc XR‖ ≤ ε →
      ‖toMatrixF r cc (nsIter X coef) - toMatrixR r cc (nsIterR XR coef)‖ ≤ L * ε + Cst := by
    rintro X XR ⟨a, b, e⟩ ε ⟨ha, hb, he⟩ hXsz hXrow hXRsz hXRrow hεB hRm hdiff
    rw [hLdef, hCdef]
    exact nsIter_step_uniform X XR r cc a b e A' B' C' (Real.sqrt 1.63) Bd ε hXsz hXrow hXRsz hXRrow hr hrc
      hRm hRm0 hBd0 hdiff (le_trans (norm_nonneg _) hdiff) hεB ha hb he
  have hns : newtonSchulz X0 eps = muonCoeffs.toList.foldl nsIter (scalarMul (1.0 / (frobNorm X0 + eps)) X0) := by
    rw [newtonSchulz_eq_foldl, ← Array.foldl_toList]
  have hfold := fold_affine_mirror_exp_shape nsIter nsIterR r cc L Cst Bd (Real.sqrt 1.63)
    (fun coef : Float × Float × Float => |toReal coef.1| ≤ A' ∧ |toReal coef.2.1| ≤ B' ∧ |toReal coef.2.2| ≤ C')
    hL1 hC0 hfpres hfRpres hstep muonCoeffs.toList (scalarMul (1.0 / (frobNorm X0 + eps)) X0) seedR ε0
    hcoef hFSsz hFSrow hSsz hSrow hB hseedErr
    (nsIterR_prefix_normle seedR r cc hSsz hSrow hr hrc hseedNorm)
  rw [← hns] at hfold
  have hmirle : ‖toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR)‖ ≤ Real.sqrt 1.3131 := by
    rw [← Real.sqrt_sq (norm_nonneg _)]
    exact Real.sqrt_le_sqrt (nsIterR_comp_normsq seedR r cc hSsz hSrow hr hrc hseedNorm)
  calc ‖toMatrixF r cc (newtonSchulz X0 eps)‖
      = ‖toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR)
          + (toMatrixF r cc (newtonSchulz X0 eps)
             - toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR))‖ := by rw [add_sub_cancel]
    _ ≤ ‖toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR)‖
          + ‖toMatrixF r cc (newtonSchulz X0 eps)
             - toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR)‖ := norm_add_le _ _
    _ ≤ Real.sqrt 1.3131 + (L ^ muonCoeffs.toList.length * ε0
          + Cst * (∑ k ∈ Finset.range muonCoeffs.toList.length, L ^ k)) := add_le_add hmirle hfold


/-! ### The per-step NON-UNIFORM capstone — tighter additive constant

`newtonSchulz_opNorm_perstep` is `newtonSchulz_opNorm_concrete` on the per-step fold: each step's `L`/`C` are
`nsuStepL`/`nsuStepC` evaluated at THAT step's own coefficient magnitudes (`|toReal a|` etc., via
`nsIter_step_uniform` with `A' = |toReal a|` exactly), so the bound is the exact per-step recurrence
`muonCoeffs.foldl (fun acc coef => Lf coef · acc + Cf coef) ε0` rather than the worst-case `L⁵·ε0 + C·∑Lᵏ`.
`nsuStepL_ge` (`A' ≤ nsuStepL A' …`) gives `hL1` (`1 ≤ Lf coef` from `1 ≤ |toReal a|`); `nsuStepC_nonneg` gives
`hC0`. The later Newton–Schulz steps have small coefficients, so their per-step `L` is `~34` (vs the worst `~77`),
making the product `∏Lᵢ` and tail `∑ⱼ Cⱼ∏_{i>j}Lᵢ` much smaller than the uniform bound. -/

theorem nsuStepL_ge (A' B' C' Rm Bd : ℝ) (r c : Nat)
    (hB'0 : 0 ≤ B') (hC'0 : 0 ≤ C') (hRm0 : 0 ≤ Rm) (hBd0 : 0 ≤ Bd) :
    A' ≤ nsuStepL A' B' C' Rm Bd r c := by
  have hu0 : (0:ℝ) ≤ u64 := u64_pos.le
  have hFX0 : 0 ≤ nsuFX Rm Bd c := by rw [nsuFX]; positivity
  have hFXt0 : 0 ≤ nsuFXt Rm Bd r := by rw [nsuFXt]; positivity
  have hβc0 : 0 ≤ nsuβ c := by rw [nsuβ]; have h1 : (1:ℝ) ≤ (1+u64)^c := one_le_pow₀ (by linarith); nlinarith
  have hβr0 : 0 ≤ nsuβ r := by rw [nsuβ]; have h1 : (1:ℝ) ≤ (1+u64)^r := one_le_pow₀ (by linarith); nlinarith
  have hRm2 : 0 ≤ Rm ^ 2 := pow_nonneg hRm0 2
  have hMFX0 : 0 ≤ nsuMFX Rm Bd := by rw [nsuMFX]; linarith
  have hMFXRm : 0 ≤ nsuMFX Rm Bd + Rm := by rw [nsuMFX]; linarith
  have hEA0 : 0 ≤ nsuEA Rm Bd r c := by
    rw [nsuEA]; exact add_nonneg (mul_nonneg hMFXRm hBd0) (mul_nonneg (mul_nonneg hβc0 hFX0) hFXt0)
  have hFA0 : 0 ≤ nsuFA Rm Bd r c := by rw [nsuFA]; exact mul_nonneg (Real.sqrt_nonneg _) (add_nonneg hRm2 hEA0)
  have hLAX0 : 0 ≤ nsuLAX Rm Bd r c := by rw [nsuLAX]; exact add_nonneg (mul_nonneg hMFXRm hMFX0) hRm2
  have hCAX0 : 0 ≤ nsuCAX Rm Bd r c := by
    rw [nsuCAX]; exact add_nonneg (mul_nonneg (mul_nonneg hβr0 hFA0) hFX0)
      (mul_nonneg (mul_nonneg (mul_nonneg hβc0 hFX0) hFXt0) hMFX0)
  have hEAX0 : 0 ≤ nsuEAX Rm Bd r c := by rw [nsuEAX]; exact add_nonneg (mul_nonneg hLAX0 hBd0) hCAX0
  have hRm3 : 0 ≤ Rm ^ 3 := pow_nonneg hRm0 3
  have hMFAX0 : 0 ≤ nsuMFAX Rm Bd r c := by rw [nsuMFAX]; exact add_nonneg hRm3 hEAX0
  have hLAAX0 : 0 ≤ nsuLAAX Rm Bd r c := by
    rw [nsuLAAX]; exact add_nonneg (mul_nonneg hMFXRm hMFAX0) (mul_nonneg hRm2 hLAX0)
  rw [nsuStepL]; nlinarith [mul_nonneg hB'0 hLAX0, mul_nonneg hC'0 hLAAX0]

set_option maxHeartbeats 1000000 in
theorem newtonSchulz_opNorm_perstep (X0 : Mat) (eps : Float) (seedR : MatR) (r cc : Nat)
    (Bd ε0 : ℝ) (hr : 0 < r) (hrc : r ≤ cc) (hBd0 : 0 ≤ Bd)
    (hX0sz : X0.size = r) (hX0row : ∀ i, i < r → (X0[i]!).size = cc)
    (hSsz : seedR.size = r) (hSrow : ∀ i, i < r → (seedR[i]!).size = cc)
    (hseedNorm : ‖toMatrixR r cc seedR‖ ^ 2 ≤ 1)
    (hcoef1 : ∀ coef ∈ muonCoeffs.toList, 1 ≤ |toReal coef.1|)
    (hB : muonCoeffs.toList.foldl (fun acc coef =>
        nsuStepL |toReal coef.1| |toReal coef.2.1| |toReal coef.2.2| (Real.sqrt 1.63) Bd r cc * acc
          + nsuStepC |toReal coef.1| |toReal coef.2.1| |toReal coef.2.2| (Real.sqrt 1.63) Bd r cc) ε0 ≤ Bd)
    (hseedErr : ‖toMatrixF r cc (scalarMul (1.0 / (frobNorm X0 + eps)) X0) - toMatrixR r cc seedR‖ ≤ ε0) :
    ‖toMatrixF r cc (newtonSchulz X0 eps)‖
      ≤ Real.sqrt 1.3131
        + muonCoeffs.toList.foldl (fun acc coef =>
            nsuStepL |toReal coef.1| |toReal coef.2.1| |toReal coef.2.2| (Real.sqrt 1.63) Bd r cc * acc
              + nsuStepC |toReal coef.1| |toReal coef.2.1| |toReal coef.2.2| (Real.sqrt 1.63) Bd r cc) ε0 := by
  haveI : Nonempty (Fin cc) := ⟨⟨0, lt_of_lt_of_le hr hrc⟩⟩
  have hRm0 : (0:ℝ) ≤ Real.sqrt 1.63 := Real.sqrt_nonneg _
  have hfpres : ∀ (X : Mat) (coef : Float × Float × Float), X.size = r → (∀ i, i < r → (X[i]!).size = cc) →
      (nsIter X coef).size = r ∧ (∀ i, i < r → ((nsIter X coef)[i]!).size = cc) := by
    intro X coef hXsz hXrow
    exact ⟨by rw [nsIter_size, hXsz], fun i hi => by rw [nsIter_rowSize X coef i (by rw [hXsz]; exact hi), hXrow i hi]⟩
  have hfRpres : ∀ (XR : MatR) (coef : Float × Float × Float), XR.size = r → (∀ i, i < r → (XR[i]!).size = cc) →
      (nsIterR XR coef).size = r ∧ (∀ i, i < r → ((nsIterR XR coef)[i]!).size = cc) := by
    intro XR coef hXRsz hXRrow
    exact ⟨by rw [nsIterR_size, hXRsz], fun i hi => by rw [nsIterR_rowSize XR coef i (by rw [hXRsz]; exact hi), hXRrow i hi]⟩
  have hFSsz : (scalarMul (1.0 / (frobNorm X0 + eps)) X0).size = r := by
    rw [Puffer.RL.NewtonSchulzError.scalarMul_size]; exact hX0sz
  have hFSrow : ∀ i, i < r → ((scalarMul (1.0 / (frobNorm X0 + eps)) X0)[i]!).size = cc := by
    intro i hi
    rw [Puffer.RL.NewtonSchulzError.scalarMul_rowSize _ X0 i (by rw [hX0sz]; exact hi)]; exact hX0row i hi
  have hL1 : ∀ coef : Float × Float × Float, (1 ≤ |toReal coef.1|) →
      1 ≤ nsuStepL |toReal coef.1| |toReal coef.2.1| |toReal coef.2.2| (Real.sqrt 1.63) Bd r cc :=
    fun coef hp => le_trans hp (nsuStepL_ge _ _ _ _ _ r cc (abs_nonneg _) (abs_nonneg _) hRm0 hBd0)
  have hC0 : ∀ coef : Float × Float × Float, (1 ≤ |toReal coef.1|) →
      0 ≤ nsuStepC |toReal coef.1| |toReal coef.2.1| |toReal coef.2.2| (Real.sqrt 1.63) Bd r cc :=
    fun coef _ => nsuStepC_nonneg _ _ _ _ _ r cc (abs_nonneg _) (abs_nonneg _) (abs_nonneg _) hRm0 hBd0
  have hstep : ∀ (X : Mat) (XR : MatR) (coef : Float × Float × Float) (ε : ℝ),
      (1 ≤ |toReal coef.1|) →
      X.size = r → (∀ i, i < r → (X[i]!).size = cc) →
      XR.size = r → (∀ i, i < r → (XR[i]!).size = cc) →
      ε ≤ Bd → ‖toMatrixR r cc XR‖ ≤ Real.sqrt 1.63 →
      ‖toMatrixF r cc X - toMatrixR r cc XR‖ ≤ ε →
      ‖toMatrixF r cc (nsIter X coef) - toMatrixR r cc (nsIterR XR coef)‖
        ≤ nsuStepL |toReal coef.1| |toReal coef.2.1| |toReal coef.2.2| (Real.sqrt 1.63) Bd r cc * ε
          + nsuStepC |toReal coef.1| |toReal coef.2.1| |toReal coef.2.2| (Real.sqrt 1.63) Bd r cc := by
    rintro X XR ⟨a, b, e⟩ ε _ hXsz hXrow hXRsz hXRrow hεB hRm hdiff
    exact nsIter_step_uniform X XR r cc a b e |toReal a| |toReal b| |toReal e| (Real.sqrt 1.63) Bd ε
      hXsz hXrow hXRsz hXRrow hr hrc hRm hRm0 hBd0 hdiff (le_trans (norm_nonneg _) hdiff) hεB (le_refl _) (le_refl _) (le_refl _)
  have hns : newtonSchulz X0 eps = muonCoeffs.toList.foldl nsIter (scalarMul (1.0 / (frobNorm X0 + eps)) X0) := by
    rw [newtonSchulz_eq_foldl, ← Array.foldl_toList]
  have hfold := fold_affine_mirror_exp_perstep nsIter nsIterR r cc Bd (Real.sqrt 1.63)
    (fun coef : Float × Float × Float => nsuStepL |toReal coef.1| |toReal coef.2.1| |toReal coef.2.2| (Real.sqrt 1.63) Bd r cc)
    (fun coef : Float × Float × Float => nsuStepC |toReal coef.1| |toReal coef.2.1| |toReal coef.2.2| (Real.sqrt 1.63) Bd r cc)
    (fun coef : Float × Float × Float => 1 ≤ |toReal coef.1|)
    hL1 hC0 hfpres hfRpres hstep muonCoeffs.toList (scalarMul (1.0 / (frobNorm X0 + eps)) X0) seedR ε0
    hcoef1 hFSsz hFSrow hSsz hSrow hB hseedErr
    (nsIterR_prefix_normle seedR r cc hSsz hSrow hr hrc hseedNorm)
  rw [← hns] at hfold
  have hmirle : ‖toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR)‖ ≤ Real.sqrt 1.3131 := by
    rw [← Real.sqrt_sq (norm_nonneg _)]
    exact Real.sqrt_le_sqrt (nsIterR_comp_normsq seedR r cc hSsz hSrow hr hrc hseedNorm)
  calc ‖toMatrixF r cc (newtonSchulz X0 eps)‖
      = ‖toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR)
          + (toMatrixF r cc (newtonSchulz X0 eps)
             - toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR))‖ := by rw [add_sub_cancel]
    _ ≤ ‖toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR)‖
          + ‖toMatrixF r cc (newtonSchulz X0 eps)
             - toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seedR)‖ := norm_add_le _ _
    _ ≤ Real.sqrt 1.3131 + muonCoeffs.toList.foldl (fun acc coef =>
            nsuStepL |toReal coef.1| |toReal coef.2.1| |toReal coef.2.2| (Real.sqrt 1.63) Bd r cc * acc
              + nsuStepC |toReal coef.1| |toReal coef.2.1| |toReal coef.2.2| (Real.sqrt 1.63) Bd r cc) ε0 :=
        add_le_add hmirle hfold


/-! ### Concrete-dimension instantiation (4×4) — `hB` and `hcoef` discharged

At `r = cc = 4` (`√4 = 2`) with `A'=5, B'=7, C'=3, Bd=0.0007, ε0=2e-16`, both scalar side conditions of
`newtonSchulz_opNorm_concrete` are discharged numerically. `nsu_4x4_bounds` bounds the constant tower by clean
rationals (`nsuStepL ≤ 80`, `nsuStepC ≤ 3.2e-13`, `1 ≤ nsuStepL`) — using `√1.63 ≤ 1.277`, `(√1.63)² = 1.63`
exactly, `u64 = 2⁻⁵³ ≤ 1.12e-16`, `√4 = 2` — so the fixpoint (`L⁵·ε0 + C·∑Lᵏ`, here
`≈ 6.6e-7 + 1.3e-5 ≈ 1.4e-5`) reduces to a rational inequality. `toReal_lit_abs_le` (via `lit_close`)
bounds each `muonCoeffs` coefficient's `toReal`, discharging `hcoef`. The result `newtonSchulz_opNorm_4x4`:
`‖toMatrixF 4 4 (newtonSchulz X0 eps)‖₂ ≤ √1.3131 + 0.00002`, fully closed except for the input-data hypotheses
(`X0`/seed shape, seed normalization, seed rounding error `≤ 2e-16`).

The per-step Lipschitz factor `L` was tightened from the Frobenius-inflated `~189` to the OPERATOR-norm value
`~79`: the per-step error splits into a ROUNDING part (`β·frobRow·frobCol`, genuinely Frobenius) and a
PERTURBATION part (`Lipschitz·‖diff‖`); the latter was bounded via `frobCol B` (a `√dim` inflation of the tight
`‖B‖₂`) in `matmul_error_opNorm`. Using the operator norm there (region-bounded by `Rm+ε`) makes the Lipschitz
chain tight — `gram: 2‖X‖`, `AX: 3‖X‖²`, `AAX: 5‖X‖⁴` — so `L = |a|+3|b|M²+5|c|M⁴ ≈ 79` (`M ≤ √1.63`), the tight
operator-norm derivative bound. Since the accumulated-rounding floor is `C·∑Lᵏ ~ C·L⁴`, dropping `L` `192→80`
shrinks the floor `~39×` (`5.6e-4 → ~1.4e-5`), taking the bound `0.0007 → 0.00002`. This is a constant-factor
tightening of `L`, NOT a contraction: `L ≈ 79 > 1` because the degree-5 Newton–Schulz polynomial with the
schedule coefficients is genuinely steep (the region-invariant `L < 1` remains un-dischargeable, cf. (a44)).
The seed-error threshold `2e-16` is `~1.8×` the actual `scalarMul` rounding `~u64 ≈ 1.1e-16`. -/

theorem toReal_lit_abs_le (m e : Nat) (K : ℝ)
    (h7 : |(OfScientific.ofScientific m true e : ℝ)| ≤ 7)
    (hK : |(OfScientific.ofScientific m true e : ℝ)| ≤ K - 1e-6) :
    |toReal (OfScientific.ofScientific m true e : Float)| ≤ K := by
  have hc := Puffer.RL.MuonCoeffFloat.lit_close m e h7
  have ht := abs_sub_abs_le_abs_sub (toReal (OfScientific.ofScientific m true e : Float))
    ((OfScientific.ofScientific m true e : ℝ))
  linarith [hc, ht]

set_option maxHeartbeats 4000000 in
theorem nsu_4x4_bounds :
    1 ≤ nsuStepL 5 7 3 (Real.sqrt 1.63) 0.0007 4 4
    ∧ nsuStepL 5 7 3 (Real.sqrt 1.63) 0.0007 4 4 ≤ 80
    ∧ nsuStepC 5 7 3 (Real.sqrt 1.63) 0.0007 4 4 ≤ 3.2e-13 := by
  set Rm := Real.sqrt 1.63 with hRmdef
  have h4 : Real.sqrt ((4:ℕ):ℝ) = 2 := by
    rw [show ((4:ℕ):ℝ) = 2^2 by norm_num, Real.sqrt_sq (by norm_num)]
  have hRmnn : (0:ℝ) ≤ Rm := Real.sqrt_nonneg _
  have hRmub : Rm ≤ 1.277 := by
    rw [hRmdef, show (1.277:ℝ) = Real.sqrt (1.277^2) from (Real.sqrt_sq (by norm_num)).symm]
    exact Real.sqrt_le_sqrt (by norm_num)
  have hRm2 : Rm^2 = 1.63 := by rw [hRmdef]; exact Real.sq_sqrt (by norm_num)
  have hRm3 : Rm^3 ≤ 2.082 := by
    have : Rm^3 = Rm^2 * Rm := by ring
    rw [this, hRm2]; nlinarith [hRmub, hRmnn]
  have hRm30 : 0 ≤ Rm^3 := pow_nonneg hRmnn 3
  have hRm5 : Rm^5 ≤ 3.393 := by
    have : Rm^5 = Rm^2 * Rm^2 * Rm := by ring
    rw [this, hRm2]; nlinarith [hRmub, hRmnn]
  have hRm50 : 0 ≤ Rm^5 := pow_nonneg hRmnn 5
  have hRm20 : (0:ℝ) ≤ Rm^2 := by rw [hRm2]; norm_num
  have hu0 : (0:ℝ) ≤ u64 := u64_pos.le
  have hu16 : u64 ≤ 1.12e-16 := by unfold u64; norm_num
  have hsqrt3 : Real.sqrt 3 ≤ 1.74 := by
    rw [show (1.74:ℝ) = Real.sqrt (1.74^2) from (Real.sqrt_sq (by norm_num)).symm]
    exact Real.sqrt_le_sqrt (by norm_num)
  have hsqrt30 : (0:ℝ) ≤ Real.sqrt 3 := Real.sqrt_nonneg _
  have hFX0 : 0 ≤ nsuFX Rm 0.0007 4 := by rw [nsuFX, h4]; positivity
  have hFXt0 : 0 ≤ nsuFXt Rm 0.0007 4 := by rw [nsuFXt, h4]; positivity
  have hFXub : nsuFX Rm 0.0007 4 ≤ 2.556 := by rw [nsuFX, h4]; nlinarith [hRmub]
  have hFXtub : nsuFXt Rm 0.0007 4 ≤ 2.556 := by rw [nsuFXt, h4]; nlinarith [hRmub]
  have hβ40 : 0 ≤ nsuβ 4 := by
    rw [nsuβ]
    have h := pow_le_pow_left₀ (show (0:ℝ) ≤ 1 by norm_num) (show (1:ℝ) ≤ 1+u64 by linarith) 4
    simp only [one_pow] at h; nlinarith [h, hu0]
  have hβ4ub : nsuβ 4 ≤ 9e-16 := by
    rw [nsuβ]
    have h1 : (1:ℝ) + u64 ≤ 1 + 1.12e-16 := by linarith
    have h2 : (1+u64)^4 ≤ (1+1.12e-16)^4 := pow_le_pow_left₀ (by linarith) h1 4
    have h4nn : (0:ℝ) ≤ (1+u64)^4 - 1 := by
      have h := pow_le_pow_left₀ (show (0:ℝ) ≤ 1 by norm_num) (show (1:ℝ) ≤ 1+u64 by linarith) 4
      simp only [one_pow] at h; linarith
    calc (2+u64)*((1+u64)^4-1) ≤ (2+1.12e-16)*((1+1.12e-16)^4-1) := by
          apply mul_le_mul (by linarith) (by linarith [h2]) h4nn (by norm_num)
      _ ≤ 9e-16 := by norm_num
  have hK0 : 0 ≤ nsuK := by rw [nsuK]; positivity
  have hKub : nsuK ≤ 3.4e-16 := by rw [nsuK]; nlinarith [hu16, hu0]
  have hMFX0 : 0 ≤ nsuMFX Rm 0.0007 := by rw [nsuMFX]; linarith
  have hMFXub : nsuMFX Rm 0.0007 ≤ 1.278 := by rw [nsuMFX]; linarith [hRmub]
  have hMFXRm0 : 0 ≤ nsuMFX Rm 0.0007 + Rm := by linarith
  have hMFXRmub : nsuMFX Rm 0.0007 + Rm ≤ 2.556 := by linarith [hMFXub, hRmub]
  have hFXtRm0 : 0 ≤ nsuFXt Rm 0.0007 4 + Rm := by linarith
  have hFXtRmub : nsuFXt Rm 0.0007 4 + Rm ≤ 3.834 := by linarith [hFXtub, hRmub]
  have hLAX0 : 0 ≤ nsuLAX Rm 0.0007 4 4 := by rw [nsuLAX]; exact add_nonneg (mul_nonneg hMFXRm0 hMFX0) hRm20
  have hLAXub : nsuLAX Rm 0.0007 4 4 ≤ 4.9 := by
    rw [nsuLAX]
    have := mul_le_mul hMFXRmub hMFXub hMFX0 (by norm_num : (0:ℝ) ≤ 2.556)
    rw [hRm2]; nlinarith [this]
  have hEA0 : 0 ≤ nsuEA Rm 0.0007 4 4 := by
    rw [nsuEA]; exact add_nonneg (mul_nonneg hMFXRm0 (by norm_num)) (mul_nonneg (mul_nonneg hβ40 hFX0) hFXt0)
  have hEAub : nsuEA Rm 0.0007 4 4 ≤ 0.004 := by
    rw [nsuEA]
    have t1 : (nsuMFX Rm 0.0007 + Rm) * 0.0007 ≤ 2.556 * 0.0007 := by nlinarith [hMFXRmub, hMFXRm0]
    have t2 : nsuβ 4 * nsuFX Rm 0.0007 4 * nsuFXt Rm 0.0007 4 ≤ 9e-16 * 2.556 * 2.556 := by
      have a1 := mul_le_mul hβ4ub hFXub hFX0 (by norm_num : (0:ℝ) ≤ 9e-16)
      have a2 := mul_le_mul a1 hFXtub hFXt0 (by positivity)
      linarith [a2]
    nlinarith [t1, t2]
  have hFA0 : 0 ≤ nsuFA Rm 0.0007 4 4 := by rw [nsuFA, h4]; exact mul_nonneg (by norm_num) (add_nonneg hRm20 hEA0)
  have hFAub : nsuFA Rm 0.0007 4 4 ≤ 3.27 := by rw [nsuFA, h4, hRm2]; nlinarith [hEAub]
  have hCAX0 : 0 ≤ nsuCAX Rm 0.0007 4 4 := by
    rw [nsuCAX]; exact add_nonneg (mul_nonneg (mul_nonneg hβ40 hFA0) hFX0)
      (mul_nonneg (mul_nonneg (mul_nonneg hβ40 hFX0) hFXt0) hMFX0)
  have hCAXub : nsuCAX Rm 0.0007 4 4 ≤ 1.6e-14 := by
    have hbr : nsuFA Rm 0.0007 4 4 * nsuFX Rm 0.0007 4 + nsuFX Rm 0.0007 4 * nsuFXt Rm 0.0007 4 * nsuMFX Rm 0.0007 ≤ 17 := by
      have a1 := mul_le_mul hFAub hFXub hFX0 (by norm_num : (0:ℝ) ≤ 3.27)
      have a2 := mul_le_mul (mul_le_mul hFXub hFXtub hFXt0 (by norm_num)) hMFXub hMFX0 (by positivity)
      nlinarith [a1, a2]
    have hbr0 : 0 ≤ nsuFA Rm 0.0007 4 4 * nsuFX Rm 0.0007 4 + nsuFX Rm 0.0007 4 * nsuFXt Rm 0.0007 4 * nsuMFX Rm 0.0007 :=
      add_nonneg (mul_nonneg hFA0 hFX0) (mul_nonneg (mul_nonneg hFX0 hFXt0) hMFX0)
    have hfact : nsuCAX Rm 0.0007 4 4
        = nsuβ 4 * (nsuFA Rm 0.0007 4 4 * nsuFX Rm 0.0007 4 + nsuFX Rm 0.0007 4 * nsuFXt Rm 0.0007 4 * nsuMFX Rm 0.0007) := by
      rw [nsuCAX]; ring
    rw [hfact]
    calc nsuβ 4 * _ ≤ 9e-16 * 17 := mul_le_mul hβ4ub hbr hbr0 (by norm_num)
      _ ≤ 1.6e-14 := by norm_num
  have hEAX0 : 0 ≤ nsuEAX Rm 0.0007 4 4 := by rw [nsuEAX]; exact add_nonneg (mul_nonneg hLAX0 (by norm_num)) hCAX0
  have hEAXub : nsuEAX Rm 0.0007 4 4 ≤ 0.012 := by rw [nsuEAX]; nlinarith [hLAXub, hLAX0, hCAXub]
  have hMFAX0 : 0 ≤ nsuMFAX Rm 0.0007 4 4 := by rw [nsuMFAX]; exact add_nonneg hRm30 hEAX0
  have hMFAXub : nsuMFAX Rm 0.0007 4 4 ≤ 2.1 := by rw [nsuMFAX]; nlinarith [hRm3, hEAXub]
  have hFCAX0 : 0 ≤ nsuFCAX Rm 0.0007 4 4 := by rw [nsuFCAX, h4]; exact mul_nonneg (by norm_num) (add_nonneg hRm30 hEAX0)
  have hFCAXub : nsuFCAX Rm 0.0007 4 4 ≤ 4.19 := by rw [nsuFCAX, h4]; nlinarith [hRm3, hEAXub]
  have hLAAX0 : 0 ≤ nsuLAAX Rm 0.0007 4 4 := by rw [nsuLAAX]; exact add_nonneg (mul_nonneg hMFXRm0 hMFAX0) (mul_nonneg hRm20 hLAX0)
  have hLAAXub : nsuLAAX Rm 0.0007 4 4 ≤ 13.4 := by
    rw [nsuLAAX, hRm2]
    have t1 := mul_le_mul hMFXRmub hMFAXub hMFAX0 (by norm_num : (0:ℝ) ≤ 2.556)
    have t2 : (1.63:ℝ) * nsuLAX Rm 0.0007 4 4 ≤ 1.63 * 4.9 := by nlinarith [hLAXub, hLAX0]
    nlinarith [t1, t2]
  have hCAAX0 : 0 ≤ nsuCAAX Rm 0.0007 4 4 := by
    rw [nsuCAAX]; exact add_nonneg (add_nonneg (mul_nonneg (mul_nonneg hβ40 hFA0) hFCAX0)
      (mul_nonneg (mul_nonneg (mul_nonneg hβ40 hFX0) hFXt0) hMFAX0)) (mul_nonneg hRm20 hCAX0)
  have hCAAXub : nsuCAAX Rm 0.0007 4 4 ≤ 5.5e-14 := by
    have hbr : nsuFA Rm 0.0007 4 4 * nsuFCAX Rm 0.0007 4 4 + nsuFX Rm 0.0007 4 * nsuFXt Rm 0.0007 4 * nsuMFAX Rm 0.0007 4 4 ≤ 28 := by
      have a1 := mul_le_mul hFAub hFCAXub hFCAX0 (by norm_num : (0:ℝ) ≤ 3.27)
      have a2 := mul_le_mul (mul_le_mul hFXub hFXtub hFXt0 (by norm_num)) hMFAXub hMFAX0 (by positivity)
      nlinarith [a1, a2]
    have hbr0 : 0 ≤ nsuFA Rm 0.0007 4 4 * nsuFCAX Rm 0.0007 4 4 + nsuFX Rm 0.0007 4 * nsuFXt Rm 0.0007 4 * nsuMFAX Rm 0.0007 4 4 :=
      add_nonneg (mul_nonneg hFA0 hFCAX0) (mul_nonneg (mul_nonneg hFX0 hFXt0) hMFAX0)
    have hfact : nsuCAAX Rm 0.0007 4 4
        = nsuβ 4 * (nsuFA Rm 0.0007 4 4 * nsuFCAX Rm 0.0007 4 4 + nsuFX Rm 0.0007 4 * nsuFXt Rm 0.0007 4 * nsuMFAX Rm 0.0007 4 4)
          + Rm^2 * nsuCAX Rm 0.0007 4 4 := by rw [nsuCAAX]; ring
    rw [hfact, hRm2]
    have p1 : nsuβ 4 * (nsuFA Rm 0.0007 4 4 * nsuFCAX Rm 0.0007 4 4 + nsuFX Rm 0.0007 4 * nsuFXt Rm 0.0007 4 * nsuMFAX Rm 0.0007 4 4) ≤ 9e-16 * 28 :=
      mul_le_mul hβ4ub hbr hbr0 (by norm_num)
    have p2 : (1.63:ℝ) * nsuCAX Rm 0.0007 4 4 ≤ 1.63 * 1.6e-14 := by nlinarith [hCAXub, hCAX0]
    nlinarith [p1, p2]
  have hEAAX0 : 0 ≤ nsuEAAX Rm 0.0007 4 4 := by rw [nsuEAAX]; exact add_nonneg (mul_nonneg hLAAX0 (by norm_num)) hCAAX0
  have hEAAXub : nsuEAAX Rm 0.0007 4 4 ≤ 0.036 := by rw [nsuEAAX]; nlinarith [hLAAXub, hLAAX0, hCAAXub]
  have hFAAX0 : 0 ≤ nsuFAAX Rm 0.0007 4 4 := by rw [nsuFAAX, h4]; exact mul_nonneg (by norm_num) (add_nonneg hRm50 hEAAX0)
  have hFAAXub : nsuFAAX Rm 0.0007 4 4 ≤ 6.86 := by rw [nsuFAAX, h4]; nlinarith [hRm5, hEAAXub]
  have hStepL0 : 1 ≤ nsuStepL 5 7 3 Rm 0.0007 4 4 := by rw [nsuStepL]; nlinarith [hLAX0, hLAAX0]
  have hStepLub : nsuStepL 5 7 3 Rm 0.0007 4 4 ≤ 80 := by
    rw [nsuStepL]; nlinarith [hLAXub, hLAAXub, hLAX0, hLAAX0]
  have hStepCub : nsuStepC 5 7 3 Rm 0.0007 4 4 ≤ 3.2e-13 := by
    rw [nsuStepC]
    have hbr : (5:ℝ) * nsuFX Rm 0.0007 4 + 7 * nsuFCAX Rm 0.0007 4 4 + 3 * nsuFAAX Rm 0.0007 4 4 ≤ 62.7 := by
      nlinarith [hFXub, hFCAXub, hFAAXub]
    have hbr0 : (0:ℝ) ≤ 5 * nsuFX Rm 0.0007 4 + 7 * nsuFCAX Rm 0.0007 4 4 + 3 * nsuFAAX Rm 0.0007 4 4 := by
      have := mul_nonneg (by norm_num : (0:ℝ) ≤ 5) hFX0
      have := mul_nonneg (by norm_num : (0:ℝ) ≤ 7) hFCAX0
      have := mul_nonneg (by norm_num : (0:ℝ) ≤ 3) hFAAX0
      linarith
    have hKt : Real.sqrt 3 * nsuK * (5 * nsuFX Rm 0.0007 4 + 7 * nsuFCAX Rm 0.0007 4 4 + 3 * nsuFAAX Rm 0.0007 4 4)
        ≤ 1.74 * 3.4e-16 * 62.7 := by
      have m1 : Real.sqrt 3 * nsuK ≤ 1.74 * 3.4e-16 := mul_le_mul hsqrt3 hKub hK0 (by norm_num)
      exact mul_le_mul m1 hbr hbr0 (by norm_num)
    nlinarith [hKt, hCAXub, hCAAXub, hCAX0, hCAAX0]
  exact ⟨hStepL0, hStepLub, hStepCub⟩

set_option maxHeartbeats 2000000 in
theorem nsu_4x4_step_le (a' b' c' : ℝ) (ha0 : 0 ≤ a') (hb0 : 0 ≤ b') (hc0 : 0 ≤ c') :
    nsuStepL a' b' c' (Real.sqrt 1.63) 0.0007 4 4 ≤ a' + b' * 4.9 + c' * 13.4
    ∧ nsuStepC a' b' c' (Real.sqrt 1.63) 0.0007 4 4
        ≤ 6e-16 * (a' * 2.6 + b' * 4.2 + c' * 6.9) + b' * 1.6e-14 + c' * 5.5e-14 := by
  set Rm := Real.sqrt 1.63 with hRmdef
  have h4 : Real.sqrt ((4:ℕ):ℝ) = 2 := by rw [show ((4:ℕ):ℝ) = 2^2 by norm_num, Real.sqrt_sq (by norm_num)]
  have hRmnn : (0:ℝ) ≤ Rm := Real.sqrt_nonneg _
  have hRmub : Rm ≤ 1.277 := by
    rw [hRmdef, show (1.277:ℝ) = Real.sqrt (1.277^2) from (Real.sqrt_sq (by norm_num)).symm]
    exact Real.sqrt_le_sqrt (by norm_num)
  have hRm2 : Rm^2 = 1.63 := by rw [hRmdef]; exact Real.sq_sqrt (by norm_num)
  have hRm3 : Rm^3 ≤ 2.082 := by
    have : Rm^3 = Rm^2 * Rm := by ring
    rw [this, hRm2]; nlinarith [hRmub, hRmnn]
  have hRm30 : 0 ≤ Rm^3 := pow_nonneg hRmnn 3
  have hRm5 : Rm^5 ≤ 3.393 := by
    have : Rm^5 = Rm^2 * Rm^2 * Rm := by ring
    rw [this, hRm2]; nlinarith [hRmub, hRmnn]
  have hRm50 : 0 ≤ Rm^5 := pow_nonneg hRmnn 5
  have hRm20 : (0:ℝ) ≤ Rm^2 := by rw [hRm2]; norm_num
  have hu0 : (0:ℝ) ≤ u64 := u64_pos.le
  have hu16 : u64 ≤ 1.12e-16 := by unfold u64; norm_num
  have hsqrt3 : Real.sqrt 3 ≤ 1.74 := by
    rw [show (1.74:ℝ) = Real.sqrt (1.74^2) from (Real.sqrt_sq (by norm_num)).symm]
    exact Real.sqrt_le_sqrt (by norm_num)
  have hsqrt30 : (0:ℝ) ≤ Real.sqrt 3 := Real.sqrt_nonneg _
  have hFX0 : 0 ≤ nsuFX Rm 0.0007 4 := by rw [nsuFX, h4]; positivity
  have hFXt0 : 0 ≤ nsuFXt Rm 0.0007 4 := by rw [nsuFXt, h4]; positivity
  have hFXub : nsuFX Rm 0.0007 4 ≤ 2.556 := by rw [nsuFX, h4]; nlinarith [hRmub]
  have hFXtub : nsuFXt Rm 0.0007 4 ≤ 2.556 := by rw [nsuFXt, h4]; nlinarith [hRmub]
  have hβ40 : 0 ≤ nsuβ 4 := by
    rw [nsuβ]; have h := pow_le_pow_left₀ (show (0:ℝ) ≤ 1 by norm_num) (show (1:ℝ) ≤ 1+u64 by linarith) 4
    simp only [one_pow] at h; nlinarith [h, hu0]
  have hβ4ub : nsuβ 4 ≤ 9e-16 := by
    rw [nsuβ]
    have h1 : (1:ℝ) + u64 ≤ 1 + 1.12e-16 := by linarith
    have h2 : (1+u64)^4 ≤ (1+1.12e-16)^4 := pow_le_pow_left₀ (by linarith) h1 4
    have h4nn : (0:ℝ) ≤ (1+u64)^4 - 1 := by
      have h := pow_le_pow_left₀ (show (0:ℝ) ≤ 1 by norm_num) (show (1:ℝ) ≤ 1+u64 by linarith) 4
      simp only [one_pow] at h; linarith
    calc (2+u64)*((1+u64)^4-1) ≤ (2+1.12e-16)*((1+1.12e-16)^4-1) := by
          apply mul_le_mul (by linarith) (by linarith [h2]) h4nn (by norm_num)
      _ ≤ 9e-16 := by norm_num
  have hK0 : 0 ≤ nsuK := by rw [nsuK]; positivity
  have hKub : nsuK ≤ 3.4e-16 := by rw [nsuK]; nlinarith [hu16, hu0]
  have hMFX0 : 0 ≤ nsuMFX Rm 0.0007 := by rw [nsuMFX]; linarith
  have hMFXub : nsuMFX Rm 0.0007 ≤ 1.278 := by rw [nsuMFX]; linarith [hRmub]
  have hMFXRm0 : 0 ≤ nsuMFX Rm 0.0007 + Rm := by linarith
  have hMFXRmub : nsuMFX Rm 0.0007 + Rm ≤ 2.556 := by linarith [hMFXub, hRmub]
  have hLAX0 : 0 ≤ nsuLAX Rm 0.0007 4 4 := by rw [nsuLAX]; exact add_nonneg (mul_nonneg hMFXRm0 hMFX0) hRm20
  have hLAXub : nsuLAX Rm 0.0007 4 4 ≤ 4.9 := by
    rw [nsuLAX]; have := mul_le_mul hMFXRmub hMFXub hMFX0 (by norm_num : (0:ℝ) ≤ 2.556); rw [hRm2]; nlinarith [this]
  have hEA0 : 0 ≤ nsuEA Rm 0.0007 4 4 := by
    rw [nsuEA]; exact add_nonneg (mul_nonneg hMFXRm0 (by norm_num)) (mul_nonneg (mul_nonneg hβ40 hFX0) hFXt0)
  have hEAub : nsuEA Rm 0.0007 4 4 ≤ 0.004 := by
    rw [nsuEA]
    have t1 : (nsuMFX Rm 0.0007 + Rm) * 0.0007 ≤ 2.556 * 0.0007 := by nlinarith [hMFXRmub, hMFXRm0]
    have t2 : nsuβ 4 * nsuFX Rm 0.0007 4 * nsuFXt Rm 0.0007 4 ≤ 9e-16 * 2.556 * 2.556 := by
      have a1 := mul_le_mul hβ4ub hFXub hFX0 (by norm_num : (0:ℝ) ≤ 9e-16)
      have a2 := mul_le_mul a1 hFXtub hFXt0 (by positivity)
      linarith [a2]
    nlinarith [t1, t2]
  have hFA0 : 0 ≤ nsuFA Rm 0.0007 4 4 := by rw [nsuFA, h4]; exact mul_nonneg (by norm_num) (add_nonneg hRm20 hEA0)
  have hFAub : nsuFA Rm 0.0007 4 4 ≤ 3.27 := by rw [nsuFA, h4, hRm2]; nlinarith [hEAub]
  have hCAX0 : 0 ≤ nsuCAX Rm 0.0007 4 4 := by
    rw [nsuCAX]; exact add_nonneg (mul_nonneg (mul_nonneg hβ40 hFA0) hFX0)
      (mul_nonneg (mul_nonneg (mul_nonneg hβ40 hFX0) hFXt0) hMFX0)
  have hCAXub : nsuCAX Rm 0.0007 4 4 ≤ 1.6e-14 := by
    have hbr : nsuFA Rm 0.0007 4 4 * nsuFX Rm 0.0007 4 + nsuFX Rm 0.0007 4 * nsuFXt Rm 0.0007 4 * nsuMFX Rm 0.0007 ≤ 17 := by
      have a1 := mul_le_mul hFAub hFXub hFX0 (by norm_num : (0:ℝ) ≤ 3.27)
      have a2 := mul_le_mul (mul_le_mul hFXub hFXtub hFXt0 (by norm_num)) hMFXub hMFX0 (by positivity)
      nlinarith [a1, a2]
    have hbr0 : 0 ≤ nsuFA Rm 0.0007 4 4 * nsuFX Rm 0.0007 4 + nsuFX Rm 0.0007 4 * nsuFXt Rm 0.0007 4 * nsuMFX Rm 0.0007 :=
      add_nonneg (mul_nonneg hFA0 hFX0) (mul_nonneg (mul_nonneg hFX0 hFXt0) hMFX0)
    have hfact : nsuCAX Rm 0.0007 4 4
        = nsuβ 4 * (nsuFA Rm 0.0007 4 4 * nsuFX Rm 0.0007 4 + nsuFX Rm 0.0007 4 * nsuFXt Rm 0.0007 4 * nsuMFX Rm 0.0007) := by
      rw [nsuCAX]; ring
    rw [hfact]; calc nsuβ 4 * _ ≤ 9e-16 * 17 := mul_le_mul hβ4ub hbr hbr0 (by norm_num)
      _ ≤ 1.6e-14 := by norm_num
  have hEAX0 : 0 ≤ nsuEAX Rm 0.0007 4 4 := by rw [nsuEAX]; exact add_nonneg (mul_nonneg hLAX0 (by norm_num)) hCAX0
  have hEAXub : nsuEAX Rm 0.0007 4 4 ≤ 0.012 := by rw [nsuEAX]; nlinarith [hLAXub, hLAX0, hCAXub]
  have hMFAX0 : 0 ≤ nsuMFAX Rm 0.0007 4 4 := by rw [nsuMFAX]; exact add_nonneg hRm30 hEAX0
  have hMFAXub : nsuMFAX Rm 0.0007 4 4 ≤ 2.1 := by rw [nsuMFAX]; nlinarith [hRm3, hEAXub]
  have hFCAX0 : 0 ≤ nsuFCAX Rm 0.0007 4 4 := by rw [nsuFCAX, h4]; exact mul_nonneg (by norm_num) (add_nonneg hRm30 hEAX0)
  have hFCAXub : nsuFCAX Rm 0.0007 4 4 ≤ 4.2 := by rw [nsuFCAX, h4]; nlinarith [hRm3, hEAXub]
  have hLAAX0 : 0 ≤ nsuLAAX Rm 0.0007 4 4 := by rw [nsuLAAX]; exact add_nonneg (mul_nonneg hMFXRm0 hMFAX0) (mul_nonneg hRm20 hLAX0)
  have hLAAXub : nsuLAAX Rm 0.0007 4 4 ≤ 13.4 := by
    rw [nsuLAAX, hRm2]
    have t1 := mul_le_mul hMFXRmub hMFAXub hMFAX0 (by norm_num : (0:ℝ) ≤ 2.556)
    have t2 : (1.63:ℝ) * nsuLAX Rm 0.0007 4 4 ≤ 1.63 * 4.9 := by nlinarith [hLAXub, hLAX0]
    nlinarith [t1, t2]
  have hCAAX0 : 0 ≤ nsuCAAX Rm 0.0007 4 4 := by
    rw [nsuCAAX]; exact add_nonneg (add_nonneg (mul_nonneg (mul_nonneg hβ40 hFA0) hFCAX0)
      (mul_nonneg (mul_nonneg (mul_nonneg hβ40 hFX0) hFXt0) hMFAX0)) (mul_nonneg hRm20 hCAX0)
  have hCAAXub : nsuCAAX Rm 0.0007 4 4 ≤ 5.5e-14 := by
    have hbr : nsuFA Rm 0.0007 4 4 * nsuFCAX Rm 0.0007 4 4 + nsuFX Rm 0.0007 4 * nsuFXt Rm 0.0007 4 * nsuMFAX Rm 0.0007 4 4 ≤ 28 := by
      have a1 := mul_le_mul hFAub hFCAXub hFCAX0 (by norm_num : (0:ℝ) ≤ 3.27)
      have a2 := mul_le_mul (mul_le_mul hFXub hFXtub hFXt0 (by norm_num)) hMFAXub hMFAX0 (by positivity)
      nlinarith [a1, a2]
    have hbr0 : 0 ≤ nsuFA Rm 0.0007 4 4 * nsuFCAX Rm 0.0007 4 4 + nsuFX Rm 0.0007 4 * nsuFXt Rm 0.0007 4 * nsuMFAX Rm 0.0007 4 4 :=
      add_nonneg (mul_nonneg hFA0 hFCAX0) (mul_nonneg (mul_nonneg hFX0 hFXt0) hMFAX0)
    have hfact : nsuCAAX Rm 0.0007 4 4
        = nsuβ 4 * (nsuFA Rm 0.0007 4 4 * nsuFCAX Rm 0.0007 4 4 + nsuFX Rm 0.0007 4 * nsuFXt Rm 0.0007 4 * nsuMFAX Rm 0.0007 4 4)
          + Rm^2 * nsuCAX Rm 0.0007 4 4 := by rw [nsuCAAX]; ring
    rw [hfact, hRm2]
    have p1 : nsuβ 4 * (nsuFA Rm 0.0007 4 4 * nsuFCAX Rm 0.0007 4 4 + nsuFX Rm 0.0007 4 * nsuFXt Rm 0.0007 4 * nsuMFAX Rm 0.0007 4 4) ≤ 9e-16 * 28 :=
      mul_le_mul hβ4ub hbr hbr0 (by norm_num)
    have p2 : (1.63:ℝ) * nsuCAX Rm 0.0007 4 4 ≤ 1.63 * 1.6e-14 := by nlinarith [hCAXub, hCAX0]
    nlinarith [p1, p2]
  have hFAAX0 : 0 ≤ nsuFAAX Rm 0.0007 4 4 := by rw [nsuFAAX, h4]
                                                exact mul_nonneg (by norm_num) (add_nonneg hRm50 (by rw [nsuEAAX]; exact add_nonneg (mul_nonneg hLAAX0 (by norm_num)) hCAAX0))
  have hEAAXub : nsuEAAX Rm 0.0007 4 4 ≤ 0.036 := by
    rw [nsuEAAX]; nlinarith [hLAAXub, hLAAX0, hCAAXub]
  have hFAAXub : nsuFAAX Rm 0.0007 4 4 ≤ 6.9 := by rw [nsuFAAX, h4]; nlinarith [hRm5, hEAAXub]
  refine ⟨by rw [nsuStepL]; nlinarith [hLAXub, hLAAXub, hb0, hc0, mul_nonneg hb0 hLAX0, mul_nonneg hc0 hLAAX0], ?_⟩
  rw [nsuStepC]
  have hbr : a' * nsuFX Rm 0.0007 4 + b' * nsuFCAX Rm 0.0007 4 4 + c' * nsuFAAX Rm 0.0007 4 4 ≤ a' * 2.6 + b' * 4.2 + c' * 6.9 := by
    nlinarith [hFXub, hFCAXub, hFAAXub, ha0, hb0, hc0]
  have hbr0 : 0 ≤ a' * nsuFX Rm 0.0007 4 + b' * nsuFCAX Rm 0.0007 4 4 + c' * nsuFAAX Rm 0.0007 4 4 :=
    add_nonneg (add_nonneg (mul_nonneg ha0 hFX0) (mul_nonneg hb0 hFCAX0)) (mul_nonneg hc0 hFAAX0)
  have hKt : Real.sqrt 3 * nsuK ≤ 6e-16 := by nlinarith [hsqrt3, hKub, hK0, hsqrt30]
  have hKt0 : 0 ≤ Real.sqrt 3 * nsuK := mul_nonneg hsqrt30 hK0
  have hbrRHS0 : 0 ≤ a' * 2.6 + b' * 4.2 + c' * 6.9 := by positivity
  have hprod : Real.sqrt 3 * nsuK * (a' * nsuFX Rm 0.0007 4 + b' * nsuFCAX Rm 0.0007 4 4 + c' * nsuFAAX Rm 0.0007 4 4)
      ≤ 6e-16 * (a' * 2.6 + b' * 4.2 + c' * 6.9) := mul_le_mul hKt hbr hbr0 (by norm_num)
  nlinarith [hprod, hCAXub, hCAAXub, hCAX0, hCAAX0, hb0, hc0, mul_le_mul_of_nonneg_left hCAXub hb0, mul_le_mul_of_nonneg_left hCAAXub hc0]

set_option maxHeartbeats 1000000 in
theorem newtonSchulz_opNorm_4x4 (X0 : Mat) (eps : Float) (seedR : MatR)
    (hX0sz : X0.size = 4) (hX0row : ∀ i, i < 4 → (X0[i]!).size = 4)
    (hSsz : seedR.size = 4) (hSrow : ∀ i, i < 4 → (seedR[i]!).size = 4)
    (hseedNorm : ‖toMatrixR 4 4 seedR‖ ^ 2 ≤ 1)
    (hseedErr : ‖toMatrixF 4 4 (scalarMul (1.0 / (frobNorm X0 + eps)) X0) - toMatrixR 4 4 seedR‖ ≤ 2e-16) :
    ‖toMatrixF 4 4 (newtonSchulz X0 eps)‖ ≤ Real.sqrt 1.3131 + 0.00002 := by
  obtain ⟨hL1, hLub, hCub⟩ := nsu_4x4_bounds
  have hC0 := nsuStepC_nonneg 5 7 3 (Real.sqrt 1.63) 0.0007 4 4 (by norm_num) (by norm_num) (by norm_num)
    (Real.sqrt_nonneg _) (by norm_num)
  have hLpos : (0:ℝ) ≤ nsuStepL 5 7 3 (Real.sqrt 1.63) 0.0007 4 4 := by linarith
  -- hcoef
  have hcoef : ∀ coef ∈ muonCoeffs.toList,
      |toReal coef.1| ≤ 5 ∧ |toReal coef.2.1| ≤ 7 ∧ |toReal coef.2.2| ≤ 3 := by
    rw [show muonCoeffs.toList = [(4.0848, -6.8946, 2.9270), (3.9505, -6.3029, 2.6377),
      (3.7418, -5.5913, 2.3037), (2.8769, -3.1427, 1.2046), (2.8366, -3.0525, 1.2012)] from rfl]
    intro coef hmem
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hmem
    rcases hmem with h|h|h|h|h <;> subst h <;> refine ⟨?_, ?_, ?_⟩ <;> dsimp only [] <;>
      first
        | exact toReal_lit_abs_le _ _ _ (by norm_num) (by norm_num)
        | (rw [show ((-6.8946 : Float)) = -(6.8946 : Float) from rfl, Puffer.FloatR.toReal_neg, abs_neg]
           exact toReal_lit_abs_le _ _ _ (by norm_num) (by norm_num))
        | (rw [show ((-6.3029 : Float)) = -(6.3029 : Float) from rfl, Puffer.FloatR.toReal_neg, abs_neg]
           exact toReal_lit_abs_le _ _ _ (by norm_num) (by norm_num))
        | (rw [show ((-5.5913 : Float)) = -(5.5913 : Float) from rfl, Puffer.FloatR.toReal_neg, abs_neg]
           exact toReal_lit_abs_le _ _ _ (by norm_num) (by norm_num))
        | (rw [show ((-3.1427 : Float)) = -(3.1427 : Float) from rfl, Puffer.FloatR.toReal_neg, abs_neg]
           exact toReal_lit_abs_le _ _ _ (by norm_num) (by norm_num))
        | (rw [show ((-3.0525 : Float)) = -(3.0525 : Float) from rfl, Puffer.FloatR.toReal_neg, abs_neg]
           exact toReal_lit_abs_le _ _ _ (by norm_num) (by norm_num))
  -- hB : the fixpoint at concrete dims
  have hlen : muonCoeffs.toList.length = 5 := by decide
  have hfin : nsuStepL 5 7 3 (Real.sqrt 1.63) 0.0007 4 4 ^ muonCoeffs.toList.length * 2e-16
      + nsuStepC 5 7 3 (Real.sqrt 1.63) 0.0007 4 4
          * (∑ k ∈ Finset.range muonCoeffs.toList.length, nsuStepL 5 7 3 (Real.sqrt 1.63) 0.0007 4 4 ^ k) ≤ 0.00002 := by
    rw [hlen, Finset.sum_range_succ, Finset.sum_range_succ, Finset.sum_range_succ, Finset.sum_range_succ,
      Finset.sum_range_one]
    set L := nsuStepL 5 7 3 (Real.sqrt 1.63) 0.0007 4 4
    set C := nsuStepC 5 7 3 (Real.sqrt 1.63) 0.0007 4 4
    have p2 : L^2 ≤ 80^2 := pow_le_pow_left₀ hLpos hLub 2
    have p3 : L^3 ≤ 80^3 := pow_le_pow_left₀ hLpos hLub 3
    have p4 : L^4 ≤ 80^4 := pow_le_pow_left₀ hLpos hLub 4
    have p5 : L^5 ≤ 80^5 := pow_le_pow_left₀ hLpos hLub 5
    have hgeom0 : (0:ℝ) ≤ 1 + L + L^2 + L^3 + L^4 := by positivity
    have hgeom : (1:ℝ) + L + L^2 + L^3 + L^4 ≤ 1 + 80 + 80^2 + 80^3 + 80^4 := by
      nlinarith [hLub, p2, p3, p4]
    have hCgeom : C * (1 + L + L^2 + L^3 + L^4) ≤ 3.2e-13 * (1 + 80 + 80^2 + 80^3 + 80^4) :=
      mul_le_mul hCub hgeom hgeom0 (by norm_num)
    have hL5 : L^5 * 2e-16 ≤ 80^5 * 2e-16 := by nlinarith [p5]
    have hnum : (80:ℝ)^5 * 2e-16 + 3.2e-13 * (1 + 80 + 80^2 + 80^3 + 80^4) ≤ 0.00002 := by norm_num
    simp only [pow_zero, pow_one]
    linarith [hL5, hCgeom, hnum]
  have hB := le_trans hfin (by norm_num : (0.00002:ℝ) ≤ 0.0007)
  -- apply the capstone
  have hmain := newtonSchulz_opNorm_concrete X0 eps seedR 4 4 5 7 3 0.0007 2e-16
    (by norm_num) (le_refl 4) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    hX0sz hX0row hSsz hSrow hseedNorm hcoef hL1 hB hseedErr
  refine le_trans hmain ?_
  linarith [hfin]


/-- Per-step `nsuStepL`/`nsuStepC` bounds at concrete coefficient bounds (packages `nsu_4x4_step_le` +
    monotonicity in the coefficient magnitudes + `nsuStepL_ge` for nonnegativity). -/
theorem nsu_4x4_stepbound (a b c : Float) (A' B' C' Lb Cb : ℝ)
    (hA : |toReal a| ≤ A') (hB : |toReal b| ≤ B') (hC : |toReal c| ≤ C')
    (hLb : A' + B' * 4.9 + C' * 13.4 ≤ Lb)
    (hCb : 6e-16 * (A' * 2.6 + B' * 4.2 + C' * 6.9) + B' * 1.6e-14 + C' * 5.5e-14 ≤ Cb) :
    nsuStepL |toReal a| |toReal b| |toReal c| (Real.sqrt 1.63) 0.0007 4 4 ≤ Lb
    ∧ nsuStepC |toReal a| |toReal b| |toReal c| (Real.sqrt 1.63) 0.0007 4 4 ≤ Cb
    ∧ 0 ≤ nsuStepL |toReal a| |toReal b| |toReal c| (Real.sqrt 1.63) 0.0007 4 4 := by
  obtain ⟨hL, hCc⟩ := nsu_4x4_step_le |toReal a| |toReal b| |toReal c| (abs_nonneg _) (abs_nonneg _) (abs_nonneg _)
  refine ⟨le_trans hL (by nlinarith [hA, hB, hC, hLb]),
    le_trans hCc (by nlinarith [hA, hB, hC, hCb]),
    le_trans (abs_nonneg _) (nsuStepL_ge _ _ _ _ _ 4 4 (abs_nonneg _) (abs_nonneg _) (Real.sqrt_nonneg _) (by norm_num))⟩

-- helper: a positive scientific literal's |toReal| ≥ 1
theorem toReal_lit_ge_one (m e : Nat) (h7 : |(OfScientific.ofScientific m true e : ℝ)| ≤ 7)
    (hge : (1:ℝ) ≤ (OfScientific.ofScientific m true e : ℝ) - 1e-6) :
    1 ≤ |toReal (OfScientific.ofScientific m true e : Float)| := by
  have hc := Puffer.RL.MuonCoeffFloat.lit_close m e h7
  have : (1:ℝ) ≤ toReal (OfScientific.ofScientific m true e : Float) := by
    rw [abs_le] at hc; linarith [hc.1]
  exact le_trans this (le_abs_self _)

set_option maxHeartbeats 1600000 in
theorem newtonSchulz_opNorm_4x4_perstep (X0 : Mat) (eps : Float) (seedR : MatR)
    (hX0sz : X0.size = 4) (hX0row : ∀ i, i < 4 → (X0[i]!).size = 4)
    (hSsz : seedR.size = 4) (hSrow : ∀ i, i < 4 → (seedR[i]!).size = 4)
    (hseedNorm : ‖toMatrixR 4 4 seedR‖ ^ 2 ≤ 1)
    (hseedErr : ‖toMatrixF 4 4 (scalarMul (1.0 / (frobNorm X0 + eps)) X0) - toMatrixR 4 4 seedR‖ ≤ 2e-16) :
    ‖toMatrixF 4 4 (newtonSchulz X0 eps)‖ ≤ Real.sqrt 1.3131 + 0.000002 := by
  -- per-step L,C bounds (nsu_4x4_stepbound + lit_close coefficient bounds)
  have hneg : ∀ (mm ee : Nat) (K : ℝ), |(OfScientific.ofScientific mm true ee : ℝ)| ≤ 7 →
      |(OfScientific.ofScientific mm true ee : ℝ)| ≤ K - 1e-6 →
      |toReal (-(OfScientific.ofScientific mm true ee : Float))| ≤ K := by
    intro mm ee K h7 hK
    rw [Puffer.FloatR.toReal_neg, abs_neg]; exact toReal_lit_abs_le mm ee K h7 hK
  obtain ⟨hL1b, hC1b, hL1n⟩ := nsu_4x4_stepbound 4.0848 (-6.8946) 2.9270 4.085 6.895 2.928 77.2 3.1e-13
    (toReal_lit_abs_le _ _ _ (by norm_num) (by norm_num)) (hneg 68946 4 _ (by norm_num) (by norm_num))
    (toReal_lit_abs_le _ _ _ (by norm_num) (by norm_num)) (by norm_num) (by norm_num)
  obtain ⟨hL2b, hC2b, hL2n⟩ := nsu_4x4_stepbound 3.9505 (-6.3029) 2.6377 3.951 6.303 2.638 70.2 2.8e-13
    (toReal_lit_abs_le _ _ _ (by norm_num) (by norm_num)) (hneg 63029 4 _ (by norm_num) (by norm_num))
    (toReal_lit_abs_le _ _ _ (by norm_num) (by norm_num)) (by norm_num) (by norm_num)
  obtain ⟨hL3b, hC3b, hL3n⟩ := nsu_4x4_stepbound 3.7418 (-5.5913) 2.3037 3.742 5.592 2.304 62.1 2.5e-13
    (toReal_lit_abs_le _ _ _ (by norm_num) (by norm_num)) (hneg 55913 4 _ (by norm_num) (by norm_num))
    (toReal_lit_abs_le _ _ _ (by norm_num) (by norm_num)) (by norm_num) (by norm_num)
  obtain ⟨hL4b, hC4b, hL4n⟩ := nsu_4x4_stepbound 2.8769 (-3.1427) 1.2046 2.877 3.143 1.205 34.5 1.4e-13
    (toReal_lit_abs_le _ _ _ (by norm_num) (by norm_num)) (hneg 31427 4 _ (by norm_num) (by norm_num))
    (toReal_lit_abs_le _ _ _ (by norm_num) (by norm_num)) (by norm_num) (by norm_num)
  obtain ⟨hL5b, hC5b, hL5n⟩ := nsu_4x4_stepbound 2.8366 (-3.0525) 1.2012 2.837 3.053 1.202 34 1.4e-13
    (toReal_lit_abs_le _ _ _ (by norm_num) (by norm_num)) (hneg 30525 4 _ (by norm_num) (by norm_num))
    (toReal_lit_abs_le _ _ _ (by norm_num) (by norm_num)) (by norm_num) (by norm_num)
  have hcoef1 : ∀ coef ∈ muonCoeffs.toList, 1 ≤ |toReal coef.1| := by
    rw [show muonCoeffs.toList = [(4.0848, -6.8946, 2.9270), (3.9505, -6.3029, 2.6377),
      (3.7418, -5.5913, 2.3037), (2.8769, -3.1427, 1.2046), (2.8366, -3.0525, 1.2012)] from rfl]
    intro coef hmem
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hmem
    rcases hmem with h|h|h|h|h <;> subst h <;> dsimp only [] <;>
      exact toReal_lit_ge_one _ _ (by norm_num) (by norm_num)
  -- the per-step foldl bound, via monotone accumulation
  have hfin : muonCoeffs.toList.foldl (fun acc coef =>
      nsuStepL |toReal coef.1| |toReal coef.2.1| |toReal coef.2.2| (Real.sqrt 1.63) 0.0007 4 4 * acc
        + nsuStepC |toReal coef.1| |toReal coef.2.1| |toReal coef.2.2| (Real.sqrt 1.63) 0.0007 4 4) 2e-16 ≤ 0.000002 := by
    rw [show muonCoeffs.toList = [(4.0848, -6.8946, 2.9270), (3.9505, -6.3029, 2.6377),
      (3.7418, -5.5913, 2.3037), (2.8769, -3.1427, 1.2046), (2.8366, -3.0525, 1.2012)] from rfl]
    simp only [List.foldl_cons, List.foldl_nil]
    set L1 := nsuStepL |toReal (4.0848:Float)| |toReal (-6.8946:Float)| |toReal (2.9270:Float)| (Real.sqrt 1.63) 0.0007 4 4
    set C1 := nsuStepC |toReal (4.0848:Float)| |toReal (-6.8946:Float)| |toReal (2.9270:Float)| (Real.sqrt 1.63) 0.0007 4 4
    set L2 := nsuStepL |toReal (3.9505:Float)| |toReal (-6.3029:Float)| |toReal (2.6377:Float)| (Real.sqrt 1.63) 0.0007 4 4
    set C2 := nsuStepC |toReal (3.9505:Float)| |toReal (-6.3029:Float)| |toReal (2.6377:Float)| (Real.sqrt 1.63) 0.0007 4 4
    set L3 := nsuStepL |toReal (3.7418:Float)| |toReal (-5.5913:Float)| |toReal (2.3037:Float)| (Real.sqrt 1.63) 0.0007 4 4
    set C3 := nsuStepC |toReal (3.7418:Float)| |toReal (-5.5913:Float)| |toReal (2.3037:Float)| (Real.sqrt 1.63) 0.0007 4 4
    set L4 := nsuStepL |toReal (2.8769:Float)| |toReal (-3.1427:Float)| |toReal (1.2046:Float)| (Real.sqrt 1.63) 0.0007 4 4
    set C4 := nsuStepC |toReal (2.8769:Float)| |toReal (-3.1427:Float)| |toReal (1.2046:Float)| (Real.sqrt 1.63) 0.0007 4 4
    set L5 := nsuStepL |toReal (2.8366:Float)| |toReal (-3.0525:Float)| |toReal (1.2012:Float)| (Real.sqrt 1.63) 0.0007 4 4
    set C5 := nsuStepC |toReal (2.8366:Float)| |toReal (-3.0525:Float)| |toReal (1.2012:Float)| (Real.sqrt 1.63) 0.0007 4 4
    have s1 : L1 * 2e-16 + C1 ≤ 3.3e-13 := by nlinarith [hL1b, hC1b]
    have s2 : L2 * (L1 * 2e-16 + C1) + C2 ≤ 2.4e-11 := by
      nlinarith [mul_le_mul_of_nonneg_left s1 hL2n,
        mul_le_mul_of_nonneg_right hL2b (show (0:ℝ) ≤ 3.3e-13 by norm_num), hC2b]
    have s3 : L3 * (L2 * (L1 * 2e-16 + C1) + C2) + C3 ≤ 1.5e-9 := by
      nlinarith [mul_le_mul_of_nonneg_left s2 hL3n,
        mul_le_mul_of_nonneg_right hL3b (show (0:ℝ) ≤ 2.4e-11 by norm_num), hC3b]
    have s4 : L4 * (L3 * (L2 * (L1 * 2e-16 + C1) + C2) + C3) + C4 ≤ 5.2e-8 := by
      nlinarith [mul_le_mul_of_nonneg_left s3 hL4n,
        mul_le_mul_of_nonneg_right hL4b (show (0:ℝ) ≤ 1.5e-9 by norm_num), hC4b]
    nlinarith [mul_le_mul_of_nonneg_left s4 hL5n,
      mul_le_mul_of_nonneg_right hL5b (show (0:ℝ) ≤ 5.2e-8 by norm_num), hC5b]
  -- apply the per-step capstone with the tight fixpoint
  have hB := le_trans hfin (by norm_num : (0.000002:ℝ) ≤ 0.0007)
  have hmain := newtonSchulz_opNorm_perstep X0 eps seedR 4 4 0.0007 2e-16
    (by norm_num) (le_refl 4) (by norm_num) hX0sz hX0row hSsz hSrow hseedNorm hcoef1 hB hseedErr
  refine le_trans hmain ?_
  linarith [hfin]

end Puffer.RL.NewtonSchulzNormTower
