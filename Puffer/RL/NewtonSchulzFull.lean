/-
The whole-algorithm capstone: the runnable `newtonSchulz X0 eps` (all 5 Muon iterations, actual Float
coefficients, real hardware Float arithmetic) has output operator norm

    ‖toMatrixF (newtonSchulz X0 eps)‖₂  ≤  √1.3131 + √(r·c)·(accumulated rounding)

— DIMENSION-FREE O(1) up to the explicit accumulated Float rounding, for a normalized input.

This transports the exact-ℝ mirror composition bound (`NewtonSchulzCompMirror.nsIterR_comp_normsq`,
`‖·‖₂ ≤ √1.3131`) to the runnable Float output, folding in the Float→mirror rounding via the existing
whole-fold `MatBnd` tower (`NewtonSchulzError.newtonSchulz_MatBnd`):

  • `fold_opNorm_le` : the generic transport — a whole-fold `MatBnd A AR r c FM FE` plus the mirror bound
      `‖toMatrixR AR‖² ≤ 1.3131` give `‖toMatrixF A‖₂ ≤ √1.3131 + √(r·c)·FE`. Triangle inequality on
      `toMatrixF A = toMatrixR AR + (toMatrixF A − toMatrixR AR)`, with the difference bounded entrywise by
      `FE` (`toMatrixF_sub_toMatrixR_entry`) and carried into the operator norm by `l2_opNorm_le_of_entrywise`
      — the same shape as the per-step `nsIter_opNorm_le_muon`.
  • `newtonSchulz_opNorm_le` : the payoff — instantiates `fold_opNorm_le` with `newtonSchulz_MatBnd`
      (Float↔mirror over the whole fold) and `nsIterR_comp_normsq` (the mirror composition), for the
      Frobenius-normalized seed `scalarMul (1.0/(‖X₀‖_F+eps)) X₀`.

Axiom-clean modulo the trusted Float base. This is the terminal statement of the tight Newton–Schulz
tower: the full runnable Muon orthogonalization's spectral bound, from IEEE Float arithmetic through the
5-iteration composition, with every rounding layer explicit and no `sorry`. The first term `√1.3131 < 1.15`
is the tight spectral constant (dimension-free); the second is the accumulated per-op Float rounding
(astronomically loose in constant but genuine and closed). The only external precondition is the normalized
seed `‖toMatrixR (mirror seed)‖ ≤ 1` — the Frobenius-normalization property (`NewtonSchulzSeedClosed`,
discharged under the explicit eps design margin).
-/
import Mathlib
import Puffer.RL.NewtonSchulzFloat
import Puffer.RL.NewtonSchulzCompMirror

namespace Puffer.RL.NewtonSchulzFull

open scoped Matrix Matrix.Norms.L2Operator
open Matrix
open Puffer.FloatR (toReal u64)
open Puffer.FloatR.Muon (Mat frobNorm scalarMul muonCoeffs)
open Puffer.RL.NewtonSchulzError
open Puffer.RL.MatrixEmbed (toMatrixR toMatrixF toMatrixF_sub_toMatrixR_entry)
open Puffer.RL.NewtonSchulzFloat (l2_opNorm_le_of_entrywise)
open Puffer.RL.NewtonSchulzCompMirror (nsIterR_comp_normsq)

/-- **Generic fold transport.** A whole-fold `MatBnd A AR r c FM FE` + the mirror composition bound
    `‖toMatrixR AR‖² ≤ 1.3131` give `‖toMatrixF A‖₂ ≤ √1.3131 + √(r·c)·FE`. Triangle inequality; the difference
    is entrywise `≤ FE` (`MatBnd`) carried into the operator norm by `l2_opNorm_le_of_entrywise`. -/
theorem fold_opNorm_le (A : Mat) (AR : MatR) (r c : Nat) (FM FE : ℝ) (hr : 0 < r) (hc : 0 < c)
    (hAB : MatBnd A AR r c FM FE) (hmirror : ‖toMatrixR r c AR‖ ^ 2 ≤ 1.3131) :
    ‖toMatrixF r c A‖ ≤ Real.sqrt 1.3131 + Real.sqrt ((r : ℝ) * c) * FE := by
  have : Nonempty (Fin c) := ⟨⟨0, hc⟩⟩
  have hentry := fun (i : Fin r) (j : Fin c) => toMatrixF_sub_toMatrixR_entry A AR r c FM FE hAB i j
  have hFEnn : 0 ≤ FE := le_trans (abs_nonneg _) (hentry ⟨0, hr⟩ ⟨0, hc⟩)
  have hdiff : ‖toMatrixF r c A - toMatrixR r c AR‖ ≤ Real.sqrt ((r : ℝ) * c) * FE := by
    refine l2_opNorm_le_of_entrywise _ _ hFEnn (fun i j => ?_)
    rw [Matrix.sub_apply]; exact hentry i j
  have hmir : ‖toMatrixR r c AR‖ ≤ Real.sqrt 1.3131 := by
    rw [← Real.sqrt_sq (norm_nonneg _)]; exact Real.sqrt_le_sqrt hmirror
  calc ‖toMatrixF r c A‖
      = ‖toMatrixR r c AR + (toMatrixF r c A - toMatrixR r c AR)‖ := by rw [add_sub_cancel]
    _ ≤ ‖toMatrixR r c AR‖ + ‖toMatrixF r c A - toMatrixR r c AR‖ := norm_add_le _ _
    _ ≤ Real.sqrt 1.3131 + Real.sqrt ((r : ℝ) * c) * FE := add_le_add hmir hdiff

/-- **The whole runnable `newtonSchulz`'s output operator norm is O(1).** For a Frobenius-normalized input
    (`‖toMatrixR (mirror seed)‖ ≤ 1`), the runnable Float `newtonSchulz X0 eps` (5 Muon iterations, actual
    Float coefficients) satisfies `‖toMatrixF (newtonSchulz X0 eps)‖₂ ≤ √1.3131 + √(r·c)·(accumulated MatBnd
    rounding)` — DIMENSION-FREE O(1) up to the explicit Float rounding. The terminal tight-tower statement. -/
theorem newtonSchulz_opNorm_le (X0 : Mat) (X0R : MatR) (eps : Float) (r c : Nat) (M ε : ℝ)
    (hr : 0 < r) (hc : 0 < c) (hM : 0 ≤ M) (hrc : r ≤ c) (hX0 : MatBnd X0 X0R r c M ε)
    (hseed : ‖toMatrixR r c (scalarMulR (toReal (1.0 / (frobNorm X0 + eps))) X0R)‖ ^ 2 ≤ 1) :
    ‖toMatrixF r c (Puffer.FloatR.Muon.newtonSchulz X0 eps)‖
      ≤ Real.sqrt 1.3131 + Real.sqrt ((r : ℝ) * c)
          * (muonCoeffs.toList.foldl (fun e a => nsIterBnd r c a e)
              ((1 + u64) * (|toReal (1.0 / (frobNorm X0 + eps))| * M),
               u64 * (|toReal (1.0 / (frobNorm X0 + eps))| * M)
                 + |toReal (1.0 / (frobNorm X0 + eps))| * ε)).2 := by
  refine fold_opNorm_le _ _ r c _ _ hr hc (newtonSchulz_MatBnd X0 X0R eps r c M ε hr hc hM hX0) ?_
  exact nsIterR_comp_normsq (scalarMulR (toReal (1.0 / (frobNorm X0 + eps))) X0R) r c
    (by rw [scalarMulR_size, hX0.sizeXR]) (fun i hi => by
      rw [scalarMulR_rowSize _ _ i (by rw [hX0.sizeXR]; exact hi), hX0.rowXR i hi]) hr hrc hseed

end Puffer.RL.NewtonSchulzFull
