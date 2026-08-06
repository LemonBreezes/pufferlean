/-
Transporting the abstract-`Matrix` composition bound (`MuonCompositionMatrix.newtonSchulz_comp_normsq`)
to the runnable pipeline's exact-ℝ mirror fold.

The key connection: one `nsIterR` step (the ℝ mirror of the runnable `nsIter`) embeds, under `toMatrixR`,
as exactly one `pstep` — the abstract Muon step of the composition. `nsIterR_toMatrixR` gives the LEFT-Gram
form `q(N)·MX` (`N = MX·MXᵀ`); the PUSH-THROUGH identity (`muon_pushthrough`, `q(A·Aᴴ)·A = A·q(Aᴴ·A)`)
turns that into the right-Gram `pstep = MX·q(MXᴴ·MX)`. So:

  • `nsIterR_toMatrixR_pstep` : `toMatrixR (nsIterR X (a,b,c)) = pstep (toReal a) (toReal b) (toReal c)
      (toMatrixR X)` — ONE mirror step IS one `pstep` (with the `toReal` coefficients).
  • `nsIterR_size` / `nsIterR_rowSize` : `nsIterR` preserves the matrix shape (both branches end in
      `lincomb3R _ X …`), so the shape hypotheses hold through a fold of iterations.

Both axiom-clean beyond the trusted Float `toReal` axiom. With these, folding `nsIterR_toMatrixR_pstep`
over the 5-step schedule identifies `toMatrixR (foldl nsIterR seed muonCoeffs)` with the 5-fold `pstep`
composition of `newtonSchulz_comp_normsq`, giving `‖·‖₂ ≤ √1.3131` on the runnable mirror.

REMAINING to that fold: (a) a COEFFICIENT-ROBUST composition — `newtonSchulz_comp_normsq` uses the exact ℝ
literals `4.0848…`, whereas the mirror carries `toReal (4.0848 : Float)`; the five `muon_comp_step` scalar
bounds must be re-proved tolerant of the `≈10⁻¹⁶` coefficient perturbation (the `float_coeff_bound`
technique from `MuonCoeffFloat`, now on the growing intervals). (b) the shape-tracked 5-fold nesting +
`newtonSchulz_eq_foldl`. Both are mechanical on top of the transport core proved here — the push-through
insight (mirror step = abstract `pstep`) is the substantive content.
-/
import Mathlib
import Puffer.RL.NewtonSchulzAssembly
import Puffer.RL.MuonCompositionMatrix

namespace Puffer.RL.NewtonSchulzTransport

open scoped Matrix
open Matrix
open Puffer.FloatR (toReal)
open Puffer.RL.NewtonSchulzError (MatR nsIterR lincomb3R_size lincomb3R_rowSize)
open Puffer.RL.MatrixEmbed (toMatrixR)
open Puffer.RL.NewtonSchulzAssembly (nsIterR_toMatrixR)
open Puffer.RL.MuonCompositionMatrix (pstep muon_pushthrough)

/-- **One mirror step IS one `pstep`.** `toMatrixR (nsIterR X coef) = pstep (toReal-coeffs) (toMatrixR X)`
    — via `nsIterR_toMatrixR` (left-Gram `q(N)·MX`) + `muon_pushthrough` (left = right = `pstep`). -/
theorem nsIterR_toMatrixR_pstep (X : MatR) (a b c : Float) (r cc : Nat)
    (hXsz : X.size = r) (hXrow : ∀ i, i < r → (X[i]!).size = cc) (hr : 0 < r) (hrc : r ≤ cc) :
    toMatrixR r cc (nsIterR X (a, b, c))
      = pstep (toReal a) (toReal b) (toReal c) (toMatrixR r cc X) := by
  rw [nsIterR_toMatrixR X a b c r cc hXsz hXrow hr hrc,
    ← conjTranspose_eq_transpose_of_trivial (toMatrixR r cc X), pstep]
  exact muon_pushthrough (toMatrixR r cc X) (toReal a) (toReal b) (toReal c)

/-- `nsIterR` preserves the row count (both shape-branches end in `lincomb3R _ X …`). -/
theorem nsIterR_size (X : MatR) (coef : Float × Float × Float) : (nsIterR X coef).size = X.size := by
  obtain ⟨a, b, c⟩ := coef
  simp only [nsIterR]
  split_ifs <;> exact lincomb3R_size _ _ _ _ _ _

/-- `nsIterR` preserves each row's length. -/
theorem nsIterR_rowSize (X : MatR) (coef : Float × Float × Float) (i : Nat) (hi : i < X.size) :
    ((nsIterR X coef)[i]!).size = (X[i]!).size := by
  obtain ⟨a, b, c⟩ := coef
  simp only [nsIterR]
  split_ifs <;> exact lincomb3R_rowSize _ _ _ _ _ _ i hi

end Puffer.RL.NewtonSchulzTransport
