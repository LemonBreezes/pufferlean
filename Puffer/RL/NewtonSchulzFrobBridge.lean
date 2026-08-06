/-
The Frobenius→operator bridge for the matmul rounding — piece (i) of the norm-based tightening of the
Newton–Schulz `FE` constant.

`NewtonSchulzTight.matmul_rounding_frob_sq` bounds the TOTAL squared rounding error of a runnable `matmul A B`
(Float product vs the exact real dot of the Float entries), summed over all entries:

    ∑ᵢⱼ (fl(AB)ᵢⱼ − (A_f·B_f)ᵢⱼ)²  ≤  β(k)² · ‖A‖_F² · ‖B‖_F²,   β(k) = (3+u64)·((1+u64)ᵏ − 1) ≈ 2k·u64.

This file lifts that `∑ Dᵢⱼ²` bound to an OPERATOR-norm bound on the rounding matrix, via the already-proven
`NewtonSchulzRunnable.opNorm_sq_le_frobenius_sq` (`‖D‖₂² ≤ ∑ Dᵢⱼ²`):

    ‖toMatrixF (matmul A B) − toMatrixF A · toMatrixF B‖₂  ≤  β(k) · ‖A‖_F · ‖B‖_F.

  • `toMatrixF_mul_apply` : the EXACT `Matrix` product of the Float-embedded matrices has entries equal to the
      exact real dot of the Float entries (`idxDotR`) — the "unrounded" matmul, distinct from the rounded
      runnable `matmul`. (Analogue of `MatrixEmbed.matmulR_toMatrixR`, for `toMatrixF` and the exact product.)
  • `matmul_rounding_opNorm` : the operator-norm rounding bound — MULTIPLICATIVE in the two Frobenius norms
      `√(∑ᵢ ∑ₗ Aᵢₗ²)`, `√(∑ⱼ ∑ₗ Bₗⱼ²)`, with the polynomial factor `β(k) ≈ 2k·u64`. Numerically β(4)≈1.3e−15
      versus the entrywise tower's ~10²¹⁶⁰: the per-matmul rounding is now O(k·u64)·‖A‖_F·‖B‖_F.

Together with the proven mirror operator-norm invariant (`NewtonSchulzCompMirror.nsIterR_comp_normsq`,
`‖·‖₂ ≤ √1.3131`), which keeps the iterates `O(√dim)` in Frobenius norm, this is the operator-norm rounding
atom a full norm-based `MatBnd` retrofit consumes; the remaining piece is the tower reassembly (carrying an
operator-norm magnitude + Frobenius error with a magnitude/error bootstrap) — no new mathematics.

Axiom-clean modulo the trusted Float base (`toReal` + the `add_model`/`mul_model` inherited via
`matmul_rounding_frob_sq`). This TIGHTENS an already-closed bound (`NewtonSchulzFull.newtonSchulz_opNorm_le`);
it changes no correctness claim.
-/
import Mathlib
import Puffer.RL.NewtonSchulzRunnable

namespace Puffer.RL.NewtonSchulzFrobBridge

open scoped Matrix Matrix.Norms.L2Operator
open Matrix
open Puffer.FloatR (toReal u64 u64_pos)
open Puffer.FloatR.Muon (Mat matmul)
open Puffer.RL.MuonMatrixRuntime (idxDotR)
open Puffer.RL.NewtonSchulzTight (sumSqR sumSqR_nonneg idxDotR_eq_sum matmul_rounding_frob_sq)
open Puffer.RL.MatrixEmbed (toMatrixF)
open Puffer.RL.NewtonSchulzRunnable (opNorm_sq_le_frobenius_sq)

/-- **The exact `Matrix` product of the Float-embedded matrices reads off `idxDotR`.** Each entry of
    `toMatrixF A · toMatrixF B` is the exact real dot of the Float entries of row `i` of `A` and column `j` of
    `B` — the "unrounded" matmul, distinct from the rounded runnable `matmul A B`. This is what the rounding
    matrix subtracts. (Analogue of `MatrixEmbed.matmulR_toMatrixR` for `toMatrixF` and the exact product.) -/
theorem toMatrixF_mul_apply (A B : Mat) (r k n : Nat) (i : Fin r) (j : Fin n) :
    (toMatrixF r k A * toMatrixF k n B) i j
      = idxDotR (fun l => (A[i.1]!)[l]!) (fun l => (B[l]!)[j.1]!) k := by
  rw [Matrix.mul_apply]
  simp only [toMatrixF, Matrix.of_apply]
  rw [idxDotR_eq_sum, Fin.sum_univ_eq_sum_range (fun l => toReal ((A[i.1]!)[l]!) * toReal ((B[l]!)[j.1]!)) k]

/-- **Frobenius→operator bridge for the matmul rounding.** The OPERATOR norm of the runnable `matmul A B`'s
    rounding (the Float product minus the exact real product `toMatrixF A · toMatrixF B` of the Float entries)
    is `≤ β(k) · ‖A‖_F · ‖B‖_F`, where `β(k) = (3+u64)·((1+u64)ᵏ − 1) ≈ 2k·u64` and the two Frobenius norms are
    `√(∑ᵢ ∑ₗ Aᵢₗ²)`, `√(∑ⱼ ∑ₗ Bₗⱼ²)` (over the matmul's inner dimension `k`). This is the operator-norm form of
    `matmul_rounding_frob_sq`: `‖D‖₂² ≤ ∑ᵢⱼ Dᵢⱼ² ≤ β²·‖A‖_F²·‖B‖_F²` (via `opNorm_sq_le_frobenius_sq`), then
    `√`. MULTIPLICATIVE in the Frobenius norms with a factor merely LINEAR in the inner dim — the polynomial,
    non-compounding per-matmul rounding, versus the entrywise tower's doubly-exponential magnitude growth
    (β(4)≈1.3e−15 vs ~10²¹⁶⁰). Dimensions are read from the operands (`r = A.size`, `n = B[0].size`,
    `k = A[0].size`), so no shape hypotheses are needed; only `Nonempty (Fin n)` (nonempty output width). -/
theorem matmul_rounding_opNorm (A B : Mat)
    [Nonempty (Fin (if B = #[] then 0 else B[0]!.size))] :
    ‖toMatrixF A.size (if B = #[] then 0 else B[0]!.size) (matmul A B)
       - toMatrixF A.size (if A = #[] then 0 else (A[0]!).size) A
         * toMatrixF (if A = #[] then 0 else (A[0]!).size) (if B = #[] then 0 else B[0]!.size) B‖
      ≤ ((2 : ℝ) + u64) * (((1 : ℝ) + u64) ^ (if A = #[] then 0 else (A[0]!).size) - (1 : ℝ))
        * (Real.sqrt (∑ i ∈ Finset.range A.size,
              sumSqR (fun l => toReal ((A[i]!)[l]!)) (if A = #[] then 0 else (A[0]!).size))
           * Real.sqrt (∑ j ∈ Finset.range (if B = #[] then 0 else B[0]!.size),
                sumSqR (fun l => toReal ((B[l]!)[j]!)) (if A = #[] then 0 else (A[0]!).size))) := by
  set r := A.size with hr
  set k := if A = #[] then 0 else (A[0]!).size with hk
  set n := if B = #[] then 0 else B[0]!.size with hn
  set β := ((2 : ℝ) + u64) * (((1 : ℝ) + u64) ^ k - 1) with hβ
  set SA := ∑ i ∈ Finset.range r, sumSqR (fun l => toReal ((A[i]!)[l]!)) k with hSA
  set SB := ∑ j ∈ Finset.range n, sumSqR (fun l => toReal ((B[l]!)[j]!)) k with hSB
  set D := toMatrixF r n (matmul A B) - toMatrixF r k A * toMatrixF k n B with hD
  have hDentry : ∀ (i : Fin r) (j : Fin n),
      D i j = toReal (((matmul A B)[i.1]!)[j.1]!)
        - idxDotR (fun l => (A[i.1]!)[l]!) (fun l => (B[l]!)[j.1]!) k := by
    intro i j
    rw [hD, Matrix.sub_apply, toMatrixF_mul_apply A B r k n i j]
    simp only [toMatrixF, Matrix.of_apply]
  have hfrob : ‖D‖ ^ 2 ≤ β ^ 2 * (SA * SB) := by
    refine le_trans (opNorm_sq_le_frobenius_sq D) ?_
    have hconv : (∑ i, ∑ j, (D i j) ^ 2)
        = ∑ i ∈ Finset.range r, ∑ j ∈ Finset.range n,
            (toReal (((matmul A B)[i]!)[j]!)
              - idxDotR (fun l => (A[i]!)[l]!) (fun l => (B[l]!)[j]!) k) ^ 2 := by
      rw [← Fin.sum_univ_eq_sum_range
        (fun i => ∑ j ∈ Finset.range n,
          (toReal (((matmul A B)[i]!)[j]!) - idxDotR (fun l => (A[i]!)[l]!) (fun l => (B[l]!)[j]!) k) ^ 2) r]
      apply Finset.sum_congr rfl; intro i _
      rw [← Fin.sum_univ_eq_sum_range
        (fun j => (toReal (((matmul A B)[i.1]!)[j]!)
          - idxDotR (fun l => (A[i.1]!)[l]!) (fun l => (B[l]!)[j]!) k) ^ 2) n]
      apply Finset.sum_congr rfl; intro j _
      rw [hDentry i j]
    rw [hconv]
    exact matmul_rounding_frob_sq A B
  have hβ0 : 0 ≤ β := by
    rw [hβ]
    exact mul_nonneg (by nlinarith [u64_pos.le])
      (by nlinarith [one_le_pow₀ (show (1:ℝ) ≤ 1 + u64 by nlinarith [u64_pos.le]) (n := k)])
  have hSA0 : 0 ≤ SA := Finset.sum_nonneg (fun i _ => sumSqR_nonneg _ _)
  have hSB0 : 0 ≤ SB := Finset.sum_nonneg (fun j _ => sumSqR_nonneg _ _)
  have hDnn : 0 ≤ ‖D‖ := norm_nonneg _
  have hfin : ‖D‖ ≤ Real.sqrt (β ^ 2 * (SA * SB)) := by
    rw [← Real.sqrt_sq hDnn]; exact Real.sqrt_le_sqrt hfrob
  rw [Real.sqrt_mul (sq_nonneg β), Real.sqrt_sq hβ0, Real.sqrt_mul hSA0] at hfin
  exact hfin

end Puffer.RL.NewtonSchulzFrobBridge
