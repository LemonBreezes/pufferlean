/-
The runnable mirror's 5-iteration composition bound: `‖toMatrixR (foldl nsIterR seed muonCoeffs)‖₂ ≤ √1.3131`
for a normalized seed. This is the shape-tracked 5-fold nesting that lifts the coefficient-robust scalar
composition (`MuonCompositionFloat`) to the abstract-`Matrix` operator norm of the runnable pipeline's
exact-ℝ mirror fold — the terminal composition capstone.

Assembly:
  • `nsIterR_step_normsq` : the per-step mirror bound — `‖toMatrixR X‖² ≤ Bin` + the Float scalar bound for
      `coef` on `[0,Bin]` ⟹ `‖toMatrixR (nsIterR X coef)‖² ≤ Bout`. Combines `nsIterR_toMatrixR_pstep`
      (mirror step = abstract `pstep`) with `muon_step_chain` (the abstract per-step operator-norm bound).
  • `nsIterR_comp_normsq` : chaining the five `muon_comp_step1..5_float` bounds through `nsIterR_step_normsq`
      with the schedule `[1,1.63,1.63,1.57,1.33,1.3131]`, tracking shape via `nsIterR_size`/`nsIterR_rowSize`.
      For `‖toMatrixR seed‖ ≤ 1`, the 5-fold `nsIterR` has `‖·‖² ≤ 1.3131`.

`muonCoeffs.toList.foldl nsIterR seed` reduces (`rfl`) to that 5-fold nesting, so the bound holds for the
literal mirror fold — the exact-ℝ mirror of `newtonSchulz X0 eps = muonCoeffs.foldl nsIter seed`
(`newtonSchulz_eq_foldl`).

Axiom-clean beyond the trusted Float base. The runnable Muon Newton–Schulz's full 5-iteration output has
operator norm `≤ √1.3131 < 1.15` — DIMENSION-FREE O(1), for the ACTUAL Float coefficients — provided the seed
is normalized (`‖toMatrixR seed‖ ≤ 1`, the Frobenius-normalization precondition from `NewtonSchulzSeedClosed`).
This is the full-composition analogue of the per-step runnable bound, on the exact-ℝ mirror; the residual
Float→mirror rounding of the whole fold is the accumulated `MatBnd` (per-step machinery already proved).
-/
import Mathlib
import Puffer.RL.NewtonSchulzTransport
import Puffer.RL.MuonCompositionMatrix
import Puffer.RL.MuonCompositionFloat

namespace Puffer.RL.NewtonSchulzCompMirror

open scoped Matrix Matrix.Norms.L2Operator
open Matrix
open Puffer.FloatR (toReal)
open Puffer.FloatR.Muon (muonCoeffs)
open Puffer.RL.NewtonSchulzError (MatR nsIterR)
open Puffer.RL.MatrixEmbed (toMatrixR)
open Puffer.RL.NewtonSchulzTransport (nsIterR_toMatrixR_pstep nsIterR_size nsIterR_rowSize)
open Puffer.RL.MuonCompositionMatrix (muon_step_chain)
open Puffer.RL.MuonCompositionFloat

/-- **Per-step mirror bound.** `‖toMatrixR X‖² ≤ Bin` + the Float scalar bound for `coef` on `[0,Bin]` ⟹
    `‖toMatrixR (nsIterR X coef)‖² ≤ Bout`. Via `nsIterR_toMatrixR_pstep` (mirror step = `pstep`) +
    `muon_step_chain`. -/
theorem nsIterR_step_normsq (X : MatR) (a b c : Float) (r cc : Nat) {Bin Bout : ℝ}
    (hXsz : X.size = r) (hXrow : ∀ i, i < r → (X[i]!).size = cc) (hr : 0 < r) (hrc : r ≤ cc)
    (hBin : ‖toMatrixR r cc X‖ ^ 2 ≤ Bin)
    (hscalar : ∀ t : ℝ, 0 ≤ t → t ≤ Bin →
      t * (toReal a + toReal b * t + toReal c * t ^ 2) ^ 2 ≤ Bout) :
    ‖toMatrixR r cc (nsIterR X (a, b, c))‖ ^ 2 ≤ Bout := by
  have : Nonempty (Fin cc) := ⟨⟨0, lt_of_lt_of_le hr hrc⟩⟩
  rw [nsIterR_toMatrixR_pstep X a b c r cc hXsz hXrow hr hrc]
  exact muon_step_chain (toMatrixR r cc X) (toReal a) (toReal b) (toReal c) Bin Bout hBin hscalar

set_option maxHeartbeats 1000000 in
/-- **The runnable mirror's 5-iteration composition is O(1).** For a normalized seed
    (`‖toMatrixR seed‖ ≤ 1`), the literal mirror fold `foldl nsIterR seed muonCoeffs` has `‖·‖² ≤ 1.3131`,
    i.e. `‖·‖₂ ≤ √1.3131 < 1.15` — dimension-free, for the ACTUAL Float coefficients. -/
theorem nsIterR_comp_normsq (seed : MatR) (r cc : Nat)
    (hsz : seed.size = r) (hrow : ∀ i, i < r → (seed[i]!).size = cc) (hr : 0 < r) (hrc : r ≤ cc)
    (hseed : ‖toMatrixR r cc seed‖ ^ 2 ≤ 1) :
    ‖toMatrixR r cc (muonCoeffs.toList.foldl nsIterR seed)‖ ^ 2 ≤ 1.3131 := by
  show ‖toMatrixR r cc (nsIterR (nsIterR (nsIterR (nsIterR (nsIterR seed
    (4.0848, -6.8946, 2.9270)) (3.9505, -6.3029, 2.6377)) (3.7418, -5.5913, 2.3037))
    (2.8769, -3.1427, 1.2046)) (2.8366, -3.0525, 1.2012))‖ ^ 2 ≤ 1.3131
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
  exact nsIterR_step_normsq _ 2.8366 (-3.0525) 1.2012 r cc s4 r4 hr hrc c4
    (fun t h0 h1 => muon_comp_step5_float t h0 h1)

end Puffer.RL.NewtonSchulzCompMirror
