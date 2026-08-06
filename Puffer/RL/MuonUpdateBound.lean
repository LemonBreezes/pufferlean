/-
Operator-norm bound for one FULL Muon optimizer update across all layers (`NNTrain.applyMuon`): both 2D
weight matrices `W1`, `W2` of the MLP are Muon-stepped (Nesterov → Newton–Schulz orthogonalize → weight
decay). `applyMuon.1.W1` and `.1.W2` are DEFEQ to the respective `stepMat`, so each new weight inherits the
single-layer bound (`MuonStepBound.stepMat_opNorm_le`):

    ‖toMatrixF W_ℓ'‖ ≤ |toReal(1−lr·wd)|·‖toMatrixF W_ℓ‖ + |toReal(lr·scale_ℓ)|·B_ℓ + √(rₗ·cₗ)·E_ℓ

with `B_ℓ = √1.3131 + rounding` from the Newton–Schulz capstone. `applyMuon_opNorm_le` bundles BOTH weight
layers into one statement (the 1D biases go through `stepVec`, no orthogonalization — a separate, simpler
Nesterov+decay bound, not covered here). Axiom-clean modulo the trusted Float base.

This lifts the per-layer weight-norm-growth recurrence to the whole network's parameter update: after one Muon
step EVERY weight matrix has grown by at most the weight-decay contraction plus a dimension-free O(1)
orthogonalized term — the norm-growth recurrence for the trained model, proved to the trusted Float model.
-/
import Mathlib
import Puffer.RL.NNTrain
import Puffer.RL.MuonStepBound

namespace Puffer.RL.MuonUpdateBound

open scoped Matrix Matrix.Norms.L2Operator
open Puffer.FloatR (toReal u64)
open Puffer.FloatR.Muon (Mat matLin stepMat newtonSchulz)
open Puffer.RL.MatrixEmbed (toMatrixF)
open Puffer.RL.MuonMatrixRuntime (matLinEntryErrBnd)
open Puffer.RL.NNTrain (MLP MuonState applyMuon)
open Puffer.RL.MuonStepBound (stepMat_opNorm_le)

/-- **Full Muon update, both weight layers bounded.** After one `applyMuon`, each new weight matrix `W1'`,
    `W2'` satisfies the single-layer `stepMat` operator-norm bound — weight-decay contraction
    `|toReal(1−lr·wd)|·‖W_ℓ‖` plus a dimension-free O(1) orthogonalized term `|toReal(lr·scaleₗ)|·B_ℓ` plus
    the layer's `matLin` rounding `√(rₗ·cₗ)·E_ℓ`. Each conjunct is `stepMat_opNorm_le` on the defeq
    `applyMuon.1.W_ℓ = (stepMat p.W_ℓ …).1`. `B_ℓ = √1.3131 + rounding` from `newtonSchulz_opNorm`. -/
theorem applyMuon_opNorm_le
    (p : MLP) (st : MuonState) (gW1 : Mat) (gb1 : Array Float) (gW2 : Mat) (gb2 : Array Float)
    (lr wd mu eps : Float) {r1 c1 r2 c2 : Nat} [Nonempty (Fin c1)] [Nonempty (Fin c2)]
    (E1 B1 E2 B2 : ℝ) (hE1 : 0 ≤ E1) (hE2 : 0 ≤ E2)
    -- layer 1 (W1 : r1×c1)
    (hW1sz : p.W1.size = r1) (hW1row : ∀ i, i < r1 → (p.W1[i]!).size = c1)
    (hentry1 : ∀ (i : Fin r1) (j : Fin c1),
      matLinEntryErrBnd (1.0 - lr * wd) ((p.W1[i.1]!)[j.1]!)
        (lr * Float.sqrt (max 1.0 (Float.ofNat p.W1.size / Float.ofNat (p.W1[0]!).size)))
        (((newtonSchulz (matLin 1.0 gW1 mu (matLin mu st.mW1 1.0 gW1)) eps)[i.1]!)[j.1]!) 0 0 ≤ E1)
    (hortho1 : ‖toMatrixF r1 c1 (newtonSchulz (matLin 1.0 gW1 mu (matLin mu st.mW1 1.0 gW1)) eps)‖ ≤ B1)
    -- layer 2 (W2 : r2×c2)
    (hW2sz : p.W2.size = r2) (hW2row : ∀ i, i < r2 → (p.W2[i]!).size = c2)
    (hentry2 : ∀ (i : Fin r2) (j : Fin c2),
      matLinEntryErrBnd (1.0 - lr * wd) ((p.W2[i.1]!)[j.1]!)
        (lr * Float.sqrt (max 1.0 (Float.ofNat p.W2.size / Float.ofNat (p.W2[0]!).size)))
        (((newtonSchulz (matLin 1.0 gW2 mu (matLin mu st.mW2 1.0 gW2)) eps)[i.1]!)[j.1]!) 0 0 ≤ E2)
    (hortho2 : ‖toMatrixF r2 c2 (newtonSchulz (matLin 1.0 gW2 mu (matLin mu st.mW2 1.0 gW2)) eps)‖ ≤ B2) :
    ‖toMatrixF r1 c1 (applyMuon p st (gW1, gb1, gW2, gb2) lr wd mu eps).1.W1‖
        ≤ |toReal (1.0 - lr * wd)| * ‖toMatrixF r1 c1 p.W1‖
          + |toReal (lr * Float.sqrt (max 1.0 (Float.ofNat p.W1.size / Float.ofNat (p.W1[0]!).size)))| * B1
          + Real.sqrt ((r1 : ℝ) * c1) * E1
      ∧ ‖toMatrixF r2 c2 (applyMuon p st (gW1, gb1, gW2, gb2) lr wd mu eps).1.W2‖
        ≤ |toReal (1.0 - lr * wd)| * ‖toMatrixF r2 c2 p.W2‖
          + |toReal (lr * Float.sqrt (max 1.0 (Float.ofNat p.W2.size / Float.ofNat (p.W2[0]!).size)))| * B2
          + Real.sqrt ((r2 : ℝ) * c2) * E2 :=
  ⟨stepMat_opNorm_le p.W1 gW1 st.mW1 lr wd mu eps E1 B1 hE1 hW1sz hW1row hentry1 hortho1,
   stepMat_opNorm_le p.W2 gW2 st.mW2 lr wd mu eps E2 B2 hE2 hW2sz hW2row hentry2 hortho2⟩

end Puffer.RL.MuonUpdateBound
