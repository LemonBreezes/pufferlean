/-
The bounded-region invariant, threaded into the actual training-loop iterate.

`NNTrain.trainPPOMuon` is an imperative loop `p ← applyMuon p st (ppoGrad p …) …` over episodes. Its
PARAMETER DYNAMICS are modelled purely by `muonTraj`: iterate `applyMuon` from `(p₀, st₀)` using a gradient
oracle `grad : MLP → grads` (the stochastic rollout gradient of the real loop is abstracted here as a
function of the current params — its effect is carried by the per-step forcing bound). `muonTraj … (n+1)` is
DEFEQ to one `applyMuon` on `muonTraj … n`, so each weight matrix along the trajectory obeys the Muon per-step
recurrence (`MuonUpdateBound.applyMuon_opNorm_le` / `stepMat_opNorm_le`).

  • `muonTraj`            : the functional training iterate `ℕ → MLP × MuonState`.
  • `muonTraj_W1_invariant`, `muonTraj_W2_invariant` : the bounded-region LOOP INVARIANT on the ACTUAL
      iterate — for a self-bounding radius `R` (`ρ·R + C ≤ R`, from `self_bounding_radius`) and the per-step
      recurrence along the trajectory, `‖toMatrixF (muonTraj … n).1.W_ℓ‖ ≤ R` for EVERY step `n`
      (`region_invariant` on `a n = ‖toMatrixF (muonTraj … n).1.W_ℓ‖`; `a 0` is defeq to `‖toMatrixF p₀.W_ℓ‖`).

Axiom-clean modulo `toReal`. The parameter trajectory of the Muon training loop stays inside the bounded
region for all time — the training-stability invariant now lives on the actual iterate, not just an abstract
sequence. The per-step recurrence hypothesis is exactly what `applyMuon_opNorm_le` supplies at each step
(with the uniform O(1) orthogonalized forcing from the dimension-free Newton–Schulz kernel).
-/
import Mathlib
import Puffer.RL.NNTrain
import Puffer.RL.MuonTrainBound
import Puffer.RL.NewtonSchulzFull
import Puffer.RL.NewtonSchulzRunnable

namespace Puffer.RL.MuonTrainLoop

open scoped Matrix Matrix.Norms.L2Operator
open Puffer.FloatR (toReal u64 u64_pos)
open Puffer.FloatR.Muon (Mat matLin newtonSchulz stepMat stepVec frobNorm muonCoeffs matMaxAbs)
open Puffer.RL.MatrixEmbed (toMatrixF toMatrixR)
open Puffer.RL.MuonMatrixRuntime (matLinEntryErrBnd)
open Puffer.RL.NewtonSchulzError (MatR MatBnd scalarMulR nsIterBnd)
open Puffer.RL.NNTrain (MLP MuonState applyMuon)
open Puffer.RL.MuonTrainBound (region_invariant network_region_invariant self_bounding_radius)
open Puffer.RL.MuonStepBound (stepMat_opNorm_le stepVec_entry_le nesterov_upd_abs_le matLinEntryErrBnd_le
  mulAdd_abs_le stepVec_snd_get matLin_opNorm_le)
open Puffer.RL.NewtonSchulzFull (newtonSchulz_opNorm_le)
open Puffer.RL.NewtonSchulzSeedClosed (matSizeOk matShapeOk matEntryBnd epsCheckB cfConst)
open Puffer.RL.NewtonSchulzRunnable (epsDefault)
open Puffer.RL.NewtonSchulzRunnable (newtonSchulz_opNorm_runnable)

/-- **The functional training iterate.** Repeatedly apply `applyMuon` from `(p₀, st₀)`, taking the gradient
    at each step from the oracle `grad` (the pure model of the loop's rollout gradient). `muonTraj … (n+1)` is
    defeq to one `applyMuon` step on `muonTraj … n`. -/
def muonTraj (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) : Nat → MLP × MuonState
  | 0 => (p0, st0)
  | n + 1 =>
      let s := muonTraj grad lr wd mu eps p0 st0 n
      applyMuon s.1 s.2 (grad s.1) lr wd mu eps

/-- **The Muon iterate is a discrete flow (semigroup composition law).** Running the trajectory for `m + n`
    steps is the same as running `m` steps and then running `n` more from the resulting `(MLP, MuonState)` pair:
    `muonTraj … p0 st0 (m + n) = muonTraj … (muonTraj … p0 st0 m).1 (muonTraj … p0 st0 m).2 n`. This is the
    time-additivity / semigroup property of the training iterate — the trajectory is a genuine deterministic flow,
    so checkpoint-and-resume is exact. Proved by induction on `n` (`applyMuon` step composes with the induction
    hypothesis; base case is `Prod` eta + `m + 0 = m`). -/
theorem muonTraj_add
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (m n : Nat) :
    muonTraj grad lr wd mu eps p0 st0 (m + n)
      = muonTraj grad lr wd mu eps (muonTraj grad lr wd mu eps p0 st0 m).1
          (muonTraj grad lr wd mu eps p0 st0 m).2 n := by
  induction n with
  | zero => rfl
  | succ k ih =>
      show muonTraj grad lr wd mu eps p0 st0 (m + k + 1) = _
      simp only [muonTraj]
      rw [ih]

/-- **The Muon training flow reads its gradient oracle only along its own orbit (gradient-oracle locality /
    congruence).** If two gradient oracles `grad1`, `grad2` agree on every parameter iterate visited by
    `grad1`'s trajectory up to time `n` (`∀ k < n, grad1 (muonTraj grad1 … k).1 = grad2 (muonTraj grad1 … k).1`),
    then the two trajectories coincide at time `n`: `muonTraj grad1 … n = muonTraj grad2 … n`. Hence the whole
    `applyMuon` iterate — parameters and momentum buffers — depends on the oracle ONLY through its restriction to
    the visited MLPs; any oracle agreeing with `grad` on the orbit produces the identical run. This is the
    locality companion of the semigroup law `muonTraj_add`, and it is exactly what justifies modelling the real
    stochastic-rollout gradient by an abstract `grad : MLP → grads`. Proved by induction: the step unfolds one
    `applyMuon`, the orbit-agreement hypothesis rewrites `grad1` to `grad2` at the visited point, and the IH
    transports along the equal states. The hypothesis is load-bearing: two oracles that differ at a visited
    iterate produce different trajectories. -/
theorem muonTraj_grad_congr
    (grad1 grad2 : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) :
    ∀ n, (∀ k, k < n →
        grad1 (muonTraj grad1 lr wd mu eps p0 st0 k).1
          = grad2 (muonTraj grad1 lr wd mu eps p0 st0 k).1) →
      muonTraj grad1 lr wd mu eps p0 st0 n = muonTraj grad2 lr wd mu eps p0 st0 n := by
  intro n
  induction n with
  | zero => intro _; rfl
  | succ m ih =>
    intro h
    have ihm : muonTraj grad1 lr wd mu eps p0 st0 m = muonTraj grad2 lr wd mu eps p0 st0 m :=
      ih (fun k hk => h k (by omega))
    simp only [muonTraj]
    rw [h m (by omega), ihm]

/-! ### Shape is preserved along the trajectory (uniform `r×c`)

`stepMat W … .1 = matLin (1−lr·wd) W (lr·scale) ortho`, and `matLin` preserves `X`'s shape exactly
(`matLin_size`/`matLin_rowSize`), so each weight matrix keeps its dimensions through every update. Hence the
shape hypotheses of `muonTraj_W_step` are DISCHARGED uniformly from the initial MLP's shape — no per-step
shape assumption is needed (unlike the genuinely data-dependent `E`/`B` forcing). -/

theorem matLin_size (a : Float) (X : Mat) (b : Float) (Y : Mat) : (matLin a X b Y).size = X.size := by
  simp [matLin]

theorem matLin_rowSize (a : Float) (X : Mat) (b : Float) (Y : Mat) (i : Nat) (hi : i < X.size) :
    ((matLin a X b Y)[i]!).size = (X[i]!).size := by
  rw [matLin, getElem!_pos _ i (by simpa using hi), Array.getElem_map, Array.getElem_range,
    Array.size_map, Array.size_range]

theorem stepMat_fst_size (W g m : Mat) (lr wd mu eps : Float) :
    (stepMat W g m lr wd mu eps).1.size = W.size := matLin_size _ _ _ _

theorem stepMat_fst_rowSize (W g m : Mat) (lr wd mu eps : Float) (i : Nat) (hi : i < W.size) :
    ((stepMat W g m lr wd mu eps).1[i]!).size = (W[i]!).size := matLin_rowSize _ _ _ _ i hi

/-- `W1`'s outer size is preserved along the trajectory: `(muonTraj … n).1.W1.size = p₀.W1.size`. -/
theorem muonTraj_W1_size
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) :
    ∀ n, (muonTraj grad lr wd mu eps p0 st0 n).1.W1.size = p0.W1.size := by
  intro n
  induction n with
  | zero => rfl
  | succ m ih =>
    rw [show (muonTraj grad lr wd mu eps p0 st0 (m + 1)).1.W1
        = (stepMat (muonTraj grad lr wd mu eps p0 st0 m).1.W1 _ _ lr wd mu eps).1 from rfl,
      stepMat_fst_size, ih]

/-- `W2`'s outer size is preserved along the trajectory. -/
theorem muonTraj_W2_size
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) :
    ∀ n, (muonTraj grad lr wd mu eps p0 st0 n).1.W2.size = p0.W2.size := by
  intro n
  induction n with
  | zero => rfl
  | succ m ih =>
    rw [show (muonTraj grad lr wd mu eps p0 st0 (m + 1)).1.W2
        = (stepMat (muonTraj grad lr wd mu eps p0 st0 m).1.W2 _ _ lr wd mu eps).1 from rfl,
      stepMat_fst_size, ih]

/-- `W1`'s row sizes are preserved along the trajectory. -/
theorem muonTraj_W1_rowSize
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (i : Nat) (hi : i < p0.W1.size) :
    ∀ n, ((muonTraj grad lr wd mu eps p0 st0 n).1.W1[i]!).size = (p0.W1[i]!).size := by
  intro n
  induction n with
  | zero => rfl
  | succ m ih =>
    rw [show (muonTraj grad lr wd mu eps p0 st0 (m + 1)).1.W1
        = (stepMat (muonTraj grad lr wd mu eps p0 st0 m).1.W1 _ _ lr wd mu eps).1 from rfl,
      stepMat_fst_rowSize _ _ _ lr wd mu eps i (by rw [muonTraj_W1_size]; exact hi), ih]

/-- `W2`'s row sizes are preserved along the trajectory. -/
theorem muonTraj_W2_rowSize
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (i : Nat) (hi : i < p0.W2.size) :
    ∀ n, ((muonTraj grad lr wd mu eps p0 st0 n).1.W2[i]!).size = (p0.W2[i]!).size := by
  intro n
  induction n with
  | zero => rfl
  | succ m ih =>
    rw [show (muonTraj grad lr wd mu eps p0 st0 (m + 1)).1.W2
        = (stepMat (muonTraj grad lr wd mu eps p0 st0 m).1.W2 _ _ lr wd mu eps).1 from rfl,
      stepMat_fst_rowSize _ _ _ lr wd mu eps i (by rw [muonTraj_W2_size]; exact hi), ih]

set_option maxHeartbeats 1000000 in
/-- **Per-step recurrence for `W1`, DISCHARGED from `applyMuon`.** `muonTraj … (n+1)).1.W1` is defeq to one
    `stepMat` on `muonTraj … n`, so `stepMat_opNorm_le` gives the Muon per-step operator-norm recurrence at
    step `n` directly — `ρ = |toReal(1−lr·wd)|`, forcing `|toReal(lr·scaleₙ)|·B + √(r·c)·E`. The `hstep`
    hypothesis of `muonTraj_W1_invariant` is exactly `∀ n`, this. -/
theorem muonTraj_W1_step {r c : Nat} [Nonempty (Fin c)]
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (n : Nat) (E B : ℝ) (hE : 0 ≤ E)
    (hWsz : ((muonTraj grad lr wd mu eps p0 st0 n).1.W1).size = r)
    (hWrow : ∀ i, i < r → (((muonTraj grad lr wd mu eps p0 st0 n).1.W1)[i]!).size = c)
    (hentry : ∀ (i : Fin r) (j : Fin c),
      matLinEntryErrBnd (1.0 - lr * wd) (((( muonTraj grad lr wd mu eps p0 st0 n).1.W1)[i.1]!)[j.1]!)
        (lr * Float.sqrt (max 1.0 (Float.ofNat ((muonTraj grad lr wd mu eps p0 st0 n).1.W1).size
          / Float.ofNat (((muonTraj grad lr wd mu eps p0 st0 n).1.W1)[0]!).size)))
        (((newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
          (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
            ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) eps)[i.1]!)[j.1]!) 0 0 ≤ E)
    (hortho : ‖toMatrixF r c (newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
      (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
        ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) eps)‖ ≤ B) :
    ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.W1‖
      ≤ |toReal (1.0 - lr * wd)| * ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 n).1.W1‖
        + |toReal (lr * Float.sqrt (max 1.0 (Float.ofNat ((muonTraj grad lr wd mu eps p0 st0 n).1.W1).size
            / Float.ofNat (((muonTraj grad lr wd mu eps p0 st0 n).1.W1)[0]!).size)))| * B
        + Real.sqrt ((r : ℝ) * c) * E := by
  have heq : (muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.W1
      = (stepMat (muonTraj grad lr wd mu eps p0 st0 n).1.W1
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1)
          ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) lr wd mu eps).1 := rfl
  rw [heq]
  exact stepMat_opNorm_le _ _ _ lr wd mu eps E B hE hWsz hWrow hentry hortho

/-- **Bounded-region loop invariant for `W1` along the actual iterate.** For a self-bounding radius `R`
    (`ρ·R + C ≤ R`) and the Muon per-step recurrence along the trajectory (supplied by `applyMuon_opNorm_le`
    at each step), the first weight matrix stays `‖toMatrixF (muonTraj … n).1.W1‖ ≤ R` for EVERY training
    step `n`, once the initial weight is inside the region. -/
theorem muonTraj_W1_invariant {r c : Nat}
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (ρ C R : ℝ) (hρ0 : 0 ≤ ρ)
    (hself : ρ * R + C ≤ R) (h0 : ‖toMatrixF r c p0.W1‖ ≤ R)
    (hstep : ∀ n, ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.W1‖
        ≤ ρ * ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 n).1.W1‖ + C) :
    ∀ n, ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 n).1.W1‖ ≤ R :=
  region_invariant (fun n => ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 n).1.W1‖)
    ρ C R hρ0 hself h0 hstep

set_option maxHeartbeats 1000000 in
/-- **Per-step recurrence for `W2`, DISCHARGED from `applyMuon`.** As `muonTraj_W1_step`, for the second
    weight matrix (`applyMuon.1.W2 = (stepMat p.W2 (grad …).2.2.1 st.mW2 …).1`). -/
theorem muonTraj_W2_step {r c : Nat} [Nonempty (Fin c)]
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (n : Nat) (E B : ℝ) (hE : 0 ≤ E)
    (hWsz : ((muonTraj grad lr wd mu eps p0 st0 n).1.W2).size = r)
    (hWrow : ∀ i, i < r → (((muonTraj grad lr wd mu eps p0 st0 n).1.W2)[i]!).size = c)
    (hentry : ∀ (i : Fin r) (j : Fin c),
      matLinEntryErrBnd (1.0 - lr * wd) (((( muonTraj grad lr wd mu eps p0 st0 n).1.W2)[i.1]!)[j.1]!)
        (lr * Float.sqrt (max 1.0 (Float.ofNat ((muonTraj grad lr wd mu eps p0 st0 n).1.W2).size
          / Float.ofNat (((muonTraj grad lr wd mu eps p0 st0 n).1.W2)[0]!).size)))
        (((newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
          (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
            ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) eps)[i.1]!)[j.1]!) 0 0 ≤ E)
    (hortho : ‖toMatrixF r c (newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
      (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
        ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) eps)‖ ≤ B) :
    ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.W2‖
      ≤ |toReal (1.0 - lr * wd)| * ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 n).1.W2‖
        + |toReal (lr * Float.sqrt (max 1.0 (Float.ofNat ((muonTraj grad lr wd mu eps p0 st0 n).1.W2).size
            / Float.ofNat (((muonTraj grad lr wd mu eps p0 st0 n).1.W2)[0]!).size)))| * B
        + Real.sqrt ((r : ℝ) * c) * E := by
  have heq : (muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.W2
      = (stepMat (muonTraj grad lr wd mu eps p0 st0 n).1.W2
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1)
          ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) lr wd mu eps).1 := rfl
  rw [heq]
  exact stepMat_opNorm_le _ _ _ lr wd mu eps E B hE hWsz hWrow hentry hortho

/-- **Bounded-region loop invariant for `W2` along the actual iterate.** As `muonTraj_W1_invariant`, for the
    second weight matrix. -/
theorem muonTraj_W2_invariant {r c : Nat}
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (ρ C R : ℝ) (hρ0 : 0 ≤ ρ)
    (hself : ρ * R + C ≤ R) (h0 : ‖toMatrixF r c p0.W2‖ ≤ R)
    (hstep : ∀ n, ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.W2‖
        ≤ ρ * ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 n).1.W2‖ + C) :
    ∀ n, ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 n).1.W2‖ ≤ R :=
  region_invariant (fun n => ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 n).1.W2‖)
    ρ C R hρ0 hself h0 hstep

/-! ### The assembled training-stability theorem: bounded gradients ⟹ trajectory bounded

Combining everything — shape discharge (`muonTraj_W1_size`/`rowSize`, so the per-step scale is the FIXED
`√(max 1 (r/c))`), the per-step recurrence (`muonTraj_W1_step`, from `stepMat_opNorm_le`), and the
forward-invariant region (`region_invariant`) — under ONE honest hypothesis: the per-step forcing is
uniformly bounded (`E`, `B` uniform along the trajectory — the effect of bounded gradients). No shape
assumption, no abstract recurrence, no assumed invariant survive; the only inputs are the initial MLP shape,
the initial weight in the region, a self-bounding radius, and the uniform-forcing (bounded-gradient) data. -/

set_option maxHeartbeats 1000000 in
/-- **Muon training stability, fully assembled.** For the actual training iterate `muonTraj`, if (i) the
    initial `W1` has shape `r×c` and lies in `{‖·‖ ≤ R}`, (ii) the per-step rounding and orthogonalized-output
    are uniformly bounded by `E`, `B` (the effect of BOUNDED GRADIENTS), and (iii) `R` is self-bounding for
    the resulting forcing (`|toReal(1−lr·wd)|·R + (|toReal(lr·scale)|·B + √(r·c)·E) ≤ R`, i.e. weight decay
    dominates), then the first weight matrix stays `‖toMatrixF (muonTraj … n).1.W1‖ ≤ R` for EVERY training
    step `n`. Shape is discharged internally (`muonTraj_W1_size`/`rowSize` fix the scale); the per-step
    recurrence is `muonTraj_W1_step`; the invariant is `region_invariant`. The Muon training-stability
    theorem grounded in the concrete iterated update, with only the bounded-gradient forcing as hypothesis. -/
theorem muonTraj_W1_bounded {r c : Nat} [Nonempty (Fin c)]
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (E B R : ℝ) (hE : 0 ≤ E) (hr : 0 < r)
    (hp0sz : p0.W1.size = r) (hp0row : ∀ i, i < r → (p0.W1[i]!).size = c)
    (h0 : ‖toMatrixF r c p0.W1‖ ≤ R)
    (hself : |toReal (1.0 - lr * wd)| * R
        + (|toReal (lr * Float.sqrt (max 1.0
              (Float.ofNat p0.W1.size / Float.ofNat (p0.W1[0]!).size)))| * B
            + Real.sqrt ((r : ℝ) * c) * E) ≤ R)
    (hentry : ∀ n, ∀ (i : Fin r) (j : Fin c),
      matLinEntryErrBnd (1.0 - lr * wd) (((( muonTraj grad lr wd mu eps p0 st0 n).1.W1)[i.1]!)[j.1]!)
        (lr * Float.sqrt (max 1.0 (Float.ofNat ((muonTraj grad lr wd mu eps p0 st0 n).1.W1).size
          / Float.ofNat (((muonTraj grad lr wd mu eps p0 st0 n).1.W1)[0]!).size)))
        (((newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
          (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
            ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) eps)[i.1]!)[j.1]!) 0 0 ≤ E)
    (hortho : ∀ n, ‖toMatrixF r c (newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
      (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
        ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) eps)‖ ≤ B) :
    ∀ n, ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 n).1.W1‖ ≤ R := by
  refine region_invariant (fun n => ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 n).1.W1‖)
    |toReal (1.0 - lr * wd)|
    (|toReal (lr * Float.sqrt (max 1.0 (Float.ofNat p0.W1.size / Float.ofNat (p0.W1[0]!).size)))| * B
      + Real.sqrt ((r : ℝ) * c) * E) R (abs_nonneg _) hself h0 ?_
  intro n
  have hstep := muonTraj_W1_step grad lr wd mu eps p0 st0 n E B hE
    (by rw [muonTraj_W1_size]; exact hp0sz)
    (fun i hi => by
      rw [muonTraj_W1_rowSize grad lr wd mu eps p0 st0 i (by rw [hp0sz]; exact hi)]; exact hp0row i hi)
    (hentry n) (hortho n)
  rw [muonTraj_W1_size grad lr wd mu eps p0 st0 n,
    muonTraj_W1_rowSize grad lr wd mu eps p0 st0 0 (by rw [hp0sz]; exact hr) n] at hstep
  linarith [hstep]

set_option maxHeartbeats 1000000 in
/-- **Muon training stability for `W2`, fully assembled.** As `muonTraj_W1_bounded`, for the second weight
    matrix — under the same bounded-gradient forcing and dominant-weight-decay condition, the trajectory of
    `W2` stays in `{‖·‖ ≤ R}` for every training step. Shape discharged via `muonTraj_W2_size`/`rowSize`;
    per-step recurrence `muonTraj_W2_step`; invariant `region_invariant`. -/
theorem muonTraj_W2_bounded {r c : Nat} [Nonempty (Fin c)]
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (E B R : ℝ) (hE : 0 ≤ E) (hr : 0 < r)
    (hp0sz : p0.W2.size = r) (hp0row : ∀ i, i < r → (p0.W2[i]!).size = c)
    (h0 : ‖toMatrixF r c p0.W2‖ ≤ R)
    (hself : |toReal (1.0 - lr * wd)| * R
        + (|toReal (lr * Float.sqrt (max 1.0
              (Float.ofNat p0.W2.size / Float.ofNat (p0.W2[0]!).size)))| * B
            + Real.sqrt ((r : ℝ) * c) * E) ≤ R)
    (hentry : ∀ n, ∀ (i : Fin r) (j : Fin c),
      matLinEntryErrBnd (1.0 - lr * wd) (((( muonTraj grad lr wd mu eps p0 st0 n).1.W2)[i.1]!)[j.1]!)
        (lr * Float.sqrt (max 1.0 (Float.ofNat ((muonTraj grad lr wd mu eps p0 st0 n).1.W2).size
          / Float.ofNat (((muonTraj grad lr wd mu eps p0 st0 n).1.W2)[0]!).size)))
        (((newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
          (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
            ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) eps)[i.1]!)[j.1]!) 0 0 ≤ E)
    (hortho : ∀ n, ‖toMatrixF r c (newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
      (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
        ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) eps)‖ ≤ B) :
    ∀ n, ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 n).1.W2‖ ≤ R := by
  refine region_invariant (fun n => ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 n).1.W2‖)
    |toReal (1.0 - lr * wd)|
    (|toReal (lr * Float.sqrt (max 1.0 (Float.ofNat p0.W2.size / Float.ofNat (p0.W2[0]!).size)))| * B
      + Real.sqrt ((r : ℝ) * c) * E) R (abs_nonneg _) hself h0 ?_
  intro n
  have hstep := muonTraj_W2_step grad lr wd mu eps p0 st0 n E B hE
    (by rw [muonTraj_W2_size]; exact hp0sz)
    (fun i hi => by
      rw [muonTraj_W2_rowSize grad lr wd mu eps p0 st0 i (by rw [hp0sz]; exact hi)]; exact hp0row i hi)
    (hentry n) (hortho n)
  rw [muonTraj_W2_size grad lr wd mu eps p0 st0 n,
    muonTraj_W2_rowSize grad lr wd mu eps p0 st0 0 (by rw [hp0sz]; exact hr) n] at hstep
  linarith [hstep]

/-! ### Bias assembled stability (`b1`): the 1D per-entry analogue

The bias update `applyMuon.1.b1 = (stepVec p.b1 (grad …).2.1 st.mb1 lr wd mu).1` has no orthogonalization, so
each entry follows the affine per-entry recurrence `|b'[i]| ≤ ρ·|b[i]| + C` with `ρ = (1+u64)²·
|toReal(1−lr·wd)|` (`stepVec_entry_le`). `stepVec` preserves `b`'s size, so an entry stays valid along the
trajectory. The assembled bias-stability theorem is `region_invariant` on `a n = |toReal (bₙ[i])|`. -/

theorem stepVec_fst_size (b grad mom : Array Float) (lr wd mu : Float) :
    (stepVec b grad mom lr wd mu).1.size = b.size := by simp [stepVec]

/-- `b1`'s size is preserved along the trajectory. -/
theorem muonTraj_b1_size
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) :
    ∀ n, (muonTraj grad lr wd mu eps p0 st0 n).1.b1.size = p0.b1.size := by
  intro n
  induction n with
  | zero => rfl
  | succ m ih =>
    rw [show (muonTraj grad lr wd mu eps p0 st0 (m + 1)).1.b1
        = (stepVec (muonTraj grad lr wd mu eps p0 st0 m).1.b1 _ _ lr wd mu).1 from rfl,
      stepVec_fst_size, ih]

set_option maxHeartbeats 1000000 in
/-- **Per-step recurrence for a `b1` entry, DISCHARGED from `applyMuon`.** `stepVec_entry_le` on the defeq
    `muonTraj … (n+1)).1.b1 = (stepVec (muonTraj … n).1.b1 …).1`. -/
theorem muonTraj_b1_step
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (i : Nat) (hi : i < p0.b1.size) (n : Nat) :
    |toReal ((muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.b1[i]!)|
      ≤ (1 + u64) ^ 2 * (|toReal (1.0 - lr * wd)| * |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b1[i]!)|
          + |toReal lr| * |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.1)[i]!
              + mu * (mu * ((muonTraj grad lr wd mu eps p0 st0 n).2.mb1)[i]!
                + ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.1)[i]!))|) := by
  have heq : (muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.b1
      = (stepVec (muonTraj grad lr wd mu eps p0 st0 n).1.b1
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.1)
          ((muonTraj grad lr wd mu eps p0 st0 n).2.mb1) lr wd mu).1 := rfl
  rw [heq]
  exact stepVec_entry_le _ _ _ lr wd mu i (by rw [muonTraj_b1_size]; exact hi)

set_option maxHeartbeats 1000000 in
/-- **Muon bias stability, fully assembled (`b1` entry).** If the initial bias entry is in `{|·| ≤ R}` and the
    per-step Nesterov update magnitude is uniformly bounded (bounded gradients, hypothesis `hforce`), and `R`
    is self-bounding for `ρ = (1+u64)²·|toReal(1−lr·wd)|` and forcing `C`, then `|toReal (bₙ[i])| ≤ R` for
    EVERY training step `n`. The 1D-bias analogue of `muonTraj_W1_bounded` — size preserved (`muonTraj_b1_
    size`), per-step recurrence `muonTraj_b1_step`, invariant `region_invariant`. -/
theorem muonTraj_b1_bounded
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (i : Nat) (hi : i < p0.b1.size) (C R : ℝ)
    (h0 : |toReal (p0.b1[i]!)| ≤ R)
    (hself : (1 + u64) ^ 2 * |toReal (1.0 - lr * wd)| * R + C ≤ R)
    (hforce : ∀ n, (1 + u64) ^ 2 * (|toReal lr| * |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.1)[i]!
        + mu * (mu * ((muonTraj grad lr wd mu eps p0 st0 n).2.mb1)[i]!
          + ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.1)[i]!))|) ≤ C) :
    ∀ n, |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b1[i]!)| ≤ R := by
  refine region_invariant (fun n => |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b1[i]!)|)
    ((1 + u64) ^ 2 * |toReal (1.0 - lr * wd)|) C R (by positivity) hself h0 ?_
  intro n
  nlinarith [muonTraj_b1_step grad lr wd mu eps p0 st0 i hi n, hforce n]

/-! ### The `b1` momentum stays bounded across training (bounded gradients ⟹ bounded momentum)

The `Mm` (momentum bound) hypothesis of `muonTraj_b1_force_le` is itself a training-loop quantity: the
momentum entry obeys `mb1_{n+1}[i] = μ·mb1_n[i] + grad_n[i]` (`stepVec.2`), an affine recurrence with
contraction `(1+u64)²·|toReal μ|`. So under bounded gradients (`G`) and `μ` small enough that the radius is
self-bounding, the momentum stays bounded for all training time — discharging `Mm`. -/

set_option maxHeartbeats 1000000 in
/-- **Per-step recurrence for the `b1` momentum entry, DISCHARGED from `applyMuon`.** `mb1_{n+1}[i] =
    μ·mb1_n[i] + grad_n[i]` (via `stepVec.2` = `newMom`), bounded by `mulAdd_abs_le`. -/
theorem muonTraj_mb1_step
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (i : Nat) (hi : i < p0.b1.size) (n : Nat) :
    |toReal ((muonTraj grad lr wd mu eps p0 st0 (n + 1)).2.mb1[i]!)|
      ≤ (1 + u64) ^ 2 * (|toReal mu| * |toReal ((muonTraj grad lr wd mu eps p0 st0 n).2.mb1[i]!)|)
        + (1 + u64) * |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.1)[i]!)| := by
  have heq : (muonTraj grad lr wd mu eps p0 st0 (n + 1)).2.mb1
      = (stepVec (muonTraj grad lr wd mu eps p0 st0 n).1.b1
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.1)
          ((muonTraj grad lr wd mu eps p0 st0 n).2.mb1) lr wd mu).2 := rfl
  rw [heq, stepVec_snd_get _ _ _ lr wd mu i (by rw [muonTraj_b1_size]; exact hi)]
  exact mulAdd_abs_le mu _ _

/-- **The `b1` momentum entry stays bounded across training.** Under bounded gradients (`|toReal grad_n[i]| ≤
    G`) and a self-bounding radius (`(1+u64)²·|toReal μ|·R + (1+u64)·G ≤ R`, i.e. `μ` small enough), the
    momentum entry `|toReal (mb1_n[i])| ≤ R` for EVERY training step — the affine-recurrence bound on the
    momentum. Discharges the `Mm` hypothesis of `muonTraj_b1_force_le` (`Mm := R`). -/
theorem muonTraj_mb1_bounded
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (i : Nat) (hi : i < p0.b1.size) (G R : ℝ)
    (h0 : |toReal (st0.mb1[i]!)| ≤ R)
    (hself : (1 + u64) ^ 2 * |toReal mu| * R + (1 + u64) * G ≤ R)
    (hG : ∀ n, |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.1)[i]!)| ≤ G) :
    ∀ n, |toReal ((muonTraj grad lr wd mu eps p0 st0 n).2.mb1[i]!)| ≤ R := by
  have hu : (0 : ℝ) ≤ u64 := u64_pos.le
  refine region_invariant (fun n => |toReal ((muonTraj grad lr wd mu eps p0 st0 n).2.mb1[i]!)|)
    ((1 + u64) ^ 2 * |toReal mu|) ((1 + u64) * G) R (by positivity) hself h0 ?_
  intro n
  nlinarith [muonTraj_mb1_step grad lr wd mu eps p0 st0 i hi n, hG n, hu]

/-! ### Bias assembled stability (`b2`): the second bias, mirroring `b1`

`applyMuon.1.b2 = (stepVec p.b2 (grad …).2.2.2 st.mb2 lr wd mu).1` — the fourth parameter tensor. Identical
structure to `b1` with the gradient component `.2.2.2` and momentum `.2.mb2`. -/

/-- `b2`'s size is preserved along the trajectory. -/
theorem muonTraj_b2_size
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) :
    ∀ n, (muonTraj grad lr wd mu eps p0 st0 n).1.b2.size = p0.b2.size := by
  intro n
  induction n with
  | zero => rfl
  | succ m ih =>
    rw [show (muonTraj grad lr wd mu eps p0 st0 (m + 1)).1.b2
        = (stepVec (muonTraj grad lr wd mu eps p0 st0 m).1.b2 _ _ lr wd mu).1 from rfl,
      stepVec_fst_size, ih]

set_option maxHeartbeats 1000000 in
/-- **Per-step recurrence for a `b2` entry, DISCHARGED from `applyMuon`.** As `muonTraj_b1_step`, for the
    second bias (`(grad …).2.2.2`, `.2.mb2`). -/
theorem muonTraj_b2_step
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (i : Nat) (hi : i < p0.b2.size) (n : Nat) :
    |toReal ((muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.b2[i]!)|
      ≤ (1 + u64) ^ 2 * (|toReal (1.0 - lr * wd)| * |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b2[i]!)|
          + |toReal lr| * |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.2)[i]!
              + mu * (mu * ((muonTraj grad lr wd mu eps p0 st0 n).2.mb2)[i]!
                + ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.2)[i]!))|) := by
  have heq : (muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.b2
      = (stepVec (muonTraj grad lr wd mu eps p0 st0 n).1.b2
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.2)
          ((muonTraj grad lr wd mu eps p0 st0 n).2.mb2) lr wd mu).1 := rfl
  rw [heq]
  exact stepVec_entry_le _ _ _ lr wd mu i (by rw [muonTraj_b2_size]; exact hi)

set_option maxHeartbeats 1000000 in
/-- **Muon bias stability, fully assembled (`b2` entry).** As `muonTraj_b1_bounded`, for the second bias. With
    all four tensors covered (`W1`, `W2`, `b1`, `b2`), EVERY MLP parameter has the assembled training-stability
    certificate. -/
theorem muonTraj_b2_bounded
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (i : Nat) (hi : i < p0.b2.size) (C R : ℝ)
    (h0 : |toReal (p0.b2[i]!)| ≤ R)
    (hself : (1 + u64) ^ 2 * |toReal (1.0 - lr * wd)| * R + C ≤ R)
    (hforce : ∀ n, (1 + u64) ^ 2 * (|toReal lr| * |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.2)[i]!
        + mu * (mu * ((muonTraj grad lr wd mu eps p0 st0 n).2.mb2)[i]!
          + ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.2)[i]!))|) ≤ C) :
    ∀ n, |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b2[i]!)| ≤ R := by
  refine region_invariant (fun n => |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b2[i]!)|)
    ((1 + u64) ^ 2 * |toReal (1.0 - lr * wd)|) C R (by positivity) hself h0 ?_
  intro n
  nlinarith [muonTraj_b2_step grad lr wd mu eps p0 st0 i hi n, hforce n]

/-! ### Whole-network assembled stability: all four tensors, one bounded region

The per-tensor recurrences (from `muonTraj_W_step`/`muonTraj_b_step`, each relaxed to a COMMON contraction
`ρ` — the biases' `(1+u64)²·|toReal(1−lr·wd)|` dominates the weights' `|toReal(1−lr·wd)|`) sum: the
whole-network norm `‖W1‖ + ‖W2‖ + |b1[i]| + |b2[j]|` obeys the same affine recurrence with combined forcing,
so it stays inside one bounded region for all training time (`network_region_invariant`, `Fin 4`). -/

/-- **Muon whole-network stability, fully assembled.** For the actual training iterate `muonTraj`, if each of
    the four tensor measures obeys the Muon per-step recurrence with a COMMON contraction `ρ < 1` and forcing
    `C_ℓ` (from `muonTraj_W_step`/`muonTraj_b_step`), the initial summed norm is `≤ R`, and `R` is
    self-bounding for the total forcing, then the whole-network norm `‖toMatrixF W1‖ + ‖toMatrixF W2‖ +
    |toReal b1[i]| + |toReal b2[j]|` stays `≤ R` for EVERY training step `n`. The `Fin 4` instance of
    `network_region_invariant` on the concrete `muonTraj` tensor sequences — every parameter of the network
    bounded simultaneously by one region. -/
theorem muonTraj_network_bounded {r1 c1 r2 c2 : Nat}
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (i j : Nat) (ρ CW1 CW2 Cb1 Cb2 R : ℝ)
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1)
    (hR : (CW1 + CW2 + Cb1 + Cb2) / (1 - ρ) ≤ R)
    (hS0 : ‖toMatrixF r1 c1 p0.W1‖ + ‖toMatrixF r2 c2 p0.W2‖
        + |toReal (p0.b1[i]!)| + |toReal (p0.b2[j]!)| ≤ R)
    (hW1 : ∀ n, ‖toMatrixF r1 c1 (muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.W1‖
        ≤ ρ * ‖toMatrixF r1 c1 (muonTraj grad lr wd mu eps p0 st0 n).1.W1‖ + CW1)
    (hW2 : ∀ n, ‖toMatrixF r2 c2 (muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.W2‖
        ≤ ρ * ‖toMatrixF r2 c2 (muonTraj grad lr wd mu eps p0 st0 n).1.W2‖ + CW2)
    (hb1 : ∀ n, |toReal ((muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.b1[i]!)|
        ≤ ρ * |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b1[i]!)| + Cb1)
    (hb2 : ∀ n, |toReal ((muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.b2[j]!)|
        ≤ ρ * |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b2[j]!)| + Cb2) :
    ∀ n, ‖toMatrixF r1 c1 (muonTraj grad lr wd mu eps p0 st0 n).1.W1‖
        + ‖toMatrixF r2 c2 (muonTraj grad lr wd mu eps p0 st0 n).1.W2‖
        + |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b1[i]!)|
        + |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b2[j]!)| ≤ R := by
  intro n
  have h := network_region_invariant
    ![fun n => ‖toMatrixF r1 c1 (muonTraj grad lr wd mu eps p0 st0 n).1.W1‖,
      fun n => ‖toMatrixF r2 c2 (muonTraj grad lr wd mu eps p0 st0 n).1.W2‖,
      fun n => |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b1[i]!)|,
      fun n => |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b2[j]!)|]
    ρ R ![CW1, CW2, Cb1, Cb2] hρ0 hρ1
    (by simpa [Fin.sum_univ_four] using hR)
    (by simpa [Fin.sum_univ_four] using hS0)
    (fun k m => by fin_cases k <;> simp only [] <;> apply_assumption) n
  simpa [Fin.sum_univ_four] using h

/-! ### Whole-network stability from the NATURAL per-tensor recurrences

The four per-tensor recurrences come out of the step lemmas at DIFFERENT contractions — weights at
`ρ_W = |toReal(1−lr·wd)|` (`muonTraj_W1_step`/`muonTraj_W2_step`), biases at `ρ_b = (1+u64)²·|toReal(1−lr·wd)|`
(`muonTraj_b1_step`/`muonTraj_b2_step`). `recur_relax` lifts a nonnegative-sequence recurrence to a larger
contraction, so the weight recurrences relax to the common `ρ_b`; `muonTraj_network_bounded'` then takes the
four recurrences AS THEY COME FROM THE STEP LEMMAS and unifies them internally. -/

/-- Relax an affine recurrence to a larger contraction (for nonnegative sequences): `a(n+1) ≤ ρ₁·a n + C`,
    `ρ₁ ≤ ρ₂`, `a n ≥ 0` ⟹ `a(n+1) ≤ ρ₂·a n + C`. -/
theorem recur_relax (a : Nat → ℝ) (ρ1 ρ2 C : ℝ) (hρ : ρ1 ≤ ρ2) (ha : ∀ n, 0 ≤ a n)
    (h : ∀ n, a (n + 1) ≤ ρ1 * a n + C) : ∀ n, a (n + 1) ≤ ρ2 * a n + C := by
  intro n; nlinarith [h n, ha n, hρ]

/-- **Whole-network stability from the natural per-tensor recurrences.** Takes the four recurrences exactly
    as the step lemmas produce them — weights at `ρ_W = |toReal(1−lr·wd)|`, biases at `ρ_b = (1+u64)²·
    |toReal(1−lr·wd)|` — and unifies to the common `ρ_b` (weights relaxed via `recur_relax`, valid since
    `ρ_W ≤ ρ_b`). Given the initial summed norm `≤ R` and `R` self-bounding for the total forcing at `ρ_b`,
    the whole-network norm stays `≤ R` for every training step. So the four `applyMuon`-discharged recurrences
    (`muonTraj_{W1,W2,b1,b2}_step`) plug straight in — the whole-network bound is grounded in the concrete
    update with no manual ρ-matching. -/
theorem muonTraj_network_bounded' {r1 c1 r2 c2 : Nat}
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (i j : Nat) (CW1 CW2 Cb1 Cb2 R : ℝ)
    (hρ1 : (1 + u64) ^ 2 * |toReal (1.0 - lr * wd)| < 1)
    (hR : (CW1 + CW2 + Cb1 + Cb2) / (1 - (1 + u64) ^ 2 * |toReal (1.0 - lr * wd)|) ≤ R)
    (hS0 : ‖toMatrixF r1 c1 p0.W1‖ + ‖toMatrixF r2 c2 p0.W2‖
        + |toReal (p0.b1[i]!)| + |toReal (p0.b2[j]!)| ≤ R)
    (hW1 : ∀ n, ‖toMatrixF r1 c1 (muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.W1‖
        ≤ |toReal (1.0 - lr * wd)| * ‖toMatrixF r1 c1 (muonTraj grad lr wd mu eps p0 st0 n).1.W1‖ + CW1)
    (hW2 : ∀ n, ‖toMatrixF r2 c2 (muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.W2‖
        ≤ |toReal (1.0 - lr * wd)| * ‖toMatrixF r2 c2 (muonTraj grad lr wd mu eps p0 st0 n).1.W2‖ + CW2)
    (hb1 : ∀ n, |toReal ((muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.b1[i]!)|
        ≤ (1 + u64) ^ 2 * |toReal (1.0 - lr * wd)| * |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b1[i]!)| + Cb1)
    (hb2 : ∀ n, |toReal ((muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.b2[j]!)|
        ≤ (1 + u64) ^ 2 * |toReal (1.0 - lr * wd)| * |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b2[j]!)| + Cb2) :
    ∀ n, ‖toMatrixF r1 c1 (muonTraj grad lr wd mu eps p0 st0 n).1.W1‖
        + ‖toMatrixF r2 c2 (muonTraj grad lr wd mu eps p0 st0 n).1.W2‖
        + |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b1[i]!)|
        + |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b2[j]!)| ≤ R := by
  have hle : |toReal (1.0 - lr * wd)| ≤ (1 + u64) ^ 2 * |toReal (1.0 - lr * wd)| := by
    nlinarith [abs_nonneg (toReal (1.0 - lr * wd)), u64_pos.le, sq_nonneg u64]
  refine muonTraj_network_bounded grad lr wd mu eps p0 st0 i j
    ((1 + u64) ^ 2 * |toReal (1.0 - lr * wd)|) CW1 CW2 Cb1 Cb2 R (by positivity) hρ1 hR hS0
    (recur_relax (fun n => ‖toMatrixF r1 c1 (muonTraj grad lr wd mu eps p0 st0 n).1.W1‖)
      |toReal (1.0 - lr * wd)| ((1 + u64) ^ 2 * |toReal (1.0 - lr * wd)|) CW1 hle
      (fun n => norm_nonneg _) hW1)
    (recur_relax (fun n => ‖toMatrixF r2 c2 (muonTraj grad lr wd mu eps p0 st0 n).1.W2‖)
      |toReal (1.0 - lr * wd)| ((1 + u64) ^ 2 * |toReal (1.0 - lr * wd)|) CW2 hle
      (fun n => norm_nonneg _) hW2) hb1 hb2

/-! ### The four recurrences, discharged from the per-step DATA

`muonTraj_{W1,W2}_recur` / `muonTraj_{b1,b2}_recur` package the per-step lemmas into the exact recurrence
forms `muonTraj_network_bounded'` consumes — from the per-tensor per-step data (shape + uniform `E`/`B` for
weights, uniform update forcing for biases), no recurrence hypothesis. `muonTraj_network_bounded_full` then
takes ONLY that data and produces the whole-network bound: every recurrence is discharged from the concrete
`applyMuon` update, and the whole-network stability rests on nothing but the initial state, the self-bounding
radius, and the bounded-gradient forcing. -/

set_option maxHeartbeats 1000000 in
/-- `W1` recurrence discharged from the per-step data (shape + uniform `E`, `B`). -/
theorem muonTraj_W1_recur {r c : Nat} [Nonempty (Fin c)]
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (E B : ℝ) (hE : 0 ≤ E) (hr : 0 < r)
    (hp0sz : p0.W1.size = r) (hp0row : ∀ i, i < r → (p0.W1[i]!).size = c)
    (hentry : ∀ n, ∀ (i : Fin r) (j : Fin c),
      matLinEntryErrBnd (1.0 - lr * wd) (((( muonTraj grad lr wd mu eps p0 st0 n).1.W1)[i.1]!)[j.1]!)
        (lr * Float.sqrt (max 1.0 (Float.ofNat ((muonTraj grad lr wd mu eps p0 st0 n).1.W1).size
          / Float.ofNat (((muonTraj grad lr wd mu eps p0 st0 n).1.W1)[0]!).size)))
        (((newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
          (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
            ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) eps)[i.1]!)[j.1]!) 0 0 ≤ E)
    (hortho : ∀ n, ‖toMatrixF r c (newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
      (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
        ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) eps)‖ ≤ B) :
    ∀ n, ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.W1‖
      ≤ |toReal (1.0 - lr * wd)| * ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 n).1.W1‖
        + (|toReal (lr * Float.sqrt (max 1.0
            (Float.ofNat p0.W1.size / Float.ofNat (p0.W1[0]!).size)))| * B + Real.sqrt ((r : ℝ) * c) * E) := by
  intro n
  have hstep := muonTraj_W1_step grad lr wd mu eps p0 st0 n E B hE
    (by rw [muonTraj_W1_size]; exact hp0sz)
    (fun i hi => by
      rw [muonTraj_W1_rowSize grad lr wd mu eps p0 st0 i (by rw [hp0sz]; exact hi)]; exact hp0row i hi)
    (hentry n) (hortho n)
  rw [muonTraj_W1_size grad lr wd mu eps p0 st0 n,
    muonTraj_W1_rowSize grad lr wd mu eps p0 st0 0 (by rw [hp0sz]; exact hr) n] at hstep
  linarith [hstep]

set_option maxHeartbeats 1000000 in
/-- **Sharp geometric convergence of the trained `W1` operator norm to its attractor.** For the actual Muon
    training iterate `muonTraj`, under (i) the initial `W1` shape `r×c`, (ii) uniform per-step rounding /
    orthogonalized-output bounds `E`, `B` (the effect of BOUNDED GRADIENTS), and (iii) *strict* weight-decay
    contraction `ρ_W = |toReal(1−lr·wd)| < 1`, the first weight matrix's operator norm satisfies the SHARP
    transient bound `‖W1_n‖ ≤ C/(1−ρ_W) + ρ_W^n · (‖W1_0‖ − C/(1−ρ_W))`, with forcing
    `C = |toReal(lr·√(max 1 (r/c)))|·B + √(r·c)·E`. The excess over the attractor `C/(1−ρ_W)` decays
    geometrically at rate `ρ_W`; as `n → ∞` the bound → `C/(1−ρ_W)`, the self-bounding region boundary. This
    STRICTLY SHARPENS the fixed-point invariant `muonTraj_W1_bounded` (whose `≤ R` is exactly the `ρ_W^n ≤ 1`
    slackening of this): it pins the exact `n`-dependence AND the attractor, not just boundedness — the first time
    the sharp geometric transient (`affine_recur_geom`, previously applied only to abstract sequences) is brought
    to the concrete trainer iterate. The per-step recurrence is discharged from the concrete `applyMuon` via
    `muonTraj_W1_recur`, then unrolled by the sharp geometric closed form
    `Puffer.RL.MuonTrainBound.affine_recur_geom` (fully qualified — not in this file's `open` list). Both
    hypotheses are load-bearing: without the bounded-gradient data `E`/`B` the norm can grow, and with `ρ_W ≥ 1`
    the attractor `C/(1−ρ_W)` is meaningless and the bound fails. -/
theorem muonTraj_W1_geometric_convergence {r c : Nat} [Nonempty (Fin c)]
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (E B : ℝ) (hE : 0 ≤ E) (hr : 0 < r)
    (hp0sz : p0.W1.size = r) (hp0row : ∀ i, i < r → (p0.W1[i]!).size = c)
    (hρ1 : |toReal (1.0 - lr * wd)| < 1)
    (hentry : ∀ n, ∀ (i : Fin r) (j : Fin c),
      matLinEntryErrBnd (1.0 - lr * wd) (((( muonTraj grad lr wd mu eps p0 st0 n).1.W1)[i.1]!)[j.1]!)
        (lr * Float.sqrt (max 1.0 (Float.ofNat ((muonTraj grad lr wd mu eps p0 st0 n).1.W1).size
          / Float.ofNat (((muonTraj grad lr wd mu eps p0 st0 n).1.W1)[0]!).size)))
        (((newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
          (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
            ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) eps)[i.1]!)[j.1]!) 0 0 ≤ E)
    (hortho : ∀ n, ‖toMatrixF r c (newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
      (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
        ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) eps)‖ ≤ B) :
    ∀ n, ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 n).1.W1‖
      ≤ (|toReal (lr * Float.sqrt (max 1.0
              (Float.ofNat p0.W1.size / Float.ofNat (p0.W1[0]!).size)))| * B
            + Real.sqrt ((r : ℝ) * c) * E) / ((1 : ℝ) - |toReal (1.0 - lr * wd)|)
        + |toReal (1.0 - lr * wd)| ^ n
          * (‖toMatrixF r c p0.W1‖
              - (|toReal (lr * Float.sqrt (max 1.0
                    (Float.ofNat p0.W1.size / Float.ofNat (p0.W1[0]!).size)))| * B
                  + Real.sqrt ((r : ℝ) * c) * E) / ((1 : ℝ) - |toReal (1.0 - lr * wd)|)) := by
  intro n
  have hrec := muonTraj_W1_recur grad lr wd mu eps p0 st0 E B hE hr hp0sz hp0row hentry hortho
  have key := Puffer.RL.MuonTrainBound.affine_recur_geom
    (fun k => ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 k).1.W1‖)
    |toReal (1.0 - lr * wd)|
    (|toReal (lr * Float.sqrt (max 1.0 (Float.ofNat p0.W1.size / Float.ofNat (p0.W1[0]!).size)))| * B
      + Real.sqrt ((r : ℝ) * c) * E) (abs_nonneg _) hρ1 hrec n
  -- `(muonTraj … 0).1.W1` is defeq to `p0.W1`; match the goal.
  exact key

set_option maxHeartbeats 1000000 in
/-- `W2` recurrence discharged from the per-step data. -/
theorem muonTraj_W2_recur {r c : Nat} [Nonempty (Fin c)]
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (E B : ℝ) (hE : 0 ≤ E) (hr : 0 < r)
    (hp0sz : p0.W2.size = r) (hp0row : ∀ i, i < r → (p0.W2[i]!).size = c)
    (hentry : ∀ n, ∀ (i : Fin r) (j : Fin c),
      matLinEntryErrBnd (1.0 - lr * wd) (((( muonTraj grad lr wd mu eps p0 st0 n).1.W2)[i.1]!)[j.1]!)
        (lr * Float.sqrt (max 1.0 (Float.ofNat ((muonTraj grad lr wd mu eps p0 st0 n).1.W2).size
          / Float.ofNat (((muonTraj grad lr wd mu eps p0 st0 n).1.W2)[0]!).size)))
        (((newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
          (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
            ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) eps)[i.1]!)[j.1]!) 0 0 ≤ E)
    (hortho : ∀ n, ‖toMatrixF r c (newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
      (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
        ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) eps)‖ ≤ B) :
    ∀ n, ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.W2‖
      ≤ |toReal (1.0 - lr * wd)| * ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 n).1.W2‖
        + (|toReal (lr * Float.sqrt (max 1.0
            (Float.ofNat p0.W2.size / Float.ofNat (p0.W2[0]!).size)))| * B + Real.sqrt ((r : ℝ) * c) * E) := by
  intro n
  have hstep := muonTraj_W2_step grad lr wd mu eps p0 st0 n E B hE
    (by rw [muonTraj_W2_size]; exact hp0sz)
    (fun i hi => by
      rw [muonTraj_W2_rowSize grad lr wd mu eps p0 st0 i (by rw [hp0sz]; exact hi)]; exact hp0row i hi)
    (hentry n) (hortho n)
  rw [muonTraj_W2_size grad lr wd mu eps p0 st0 n,
    muonTraj_W2_rowSize grad lr wd mu eps p0 st0 0 (by rw [hp0sz]; exact hr) n] at hstep
  linarith [hstep]

/-- `b1` recurrence discharged from the per-step forcing bound. -/
theorem muonTraj_b1_recur
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (i : Nat) (hi : i < p0.b1.size) (C : ℝ)
    (hforce : ∀ n, (1 + u64) ^ 2 * (|toReal lr| * |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.1)[i]!
        + mu * (mu * ((muonTraj grad lr wd mu eps p0 st0 n).2.mb1)[i]!
          + ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.1)[i]!))|) ≤ C) :
    ∀ n, |toReal ((muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.b1[i]!)|
      ≤ (1 + u64) ^ 2 * |toReal (1.0 - lr * wd)| * |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b1[i]!)| + C := by
  intro n
  nlinarith [muonTraj_b1_step grad lr wd mu eps p0 st0 i hi n, hforce n]

/-- `b2` recurrence discharged from the per-step forcing bound. -/
theorem muonTraj_b2_recur
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (j : Nat) (hj : j < p0.b2.size) (C : ℝ)
    (hforce : ∀ n, (1 + u64) ^ 2 * (|toReal lr| * |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.2)[j]!
        + mu * (mu * ((muonTraj grad lr wd mu eps p0 st0 n).2.mb2)[j]!
          + ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.2)[j]!))|) ≤ C) :
    ∀ n, |toReal ((muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.b2[j]!)|
      ≤ (1 + u64) ^ 2 * |toReal (1.0 - lr * wd)| * |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b2[j]!)| + C := by
  intro n
  nlinarith [muonTraj_b2_step grad lr wd mu eps p0 st0 j hj n, hforce n]

/-- **Full whole-network Muon training stability, from per-step DATA.** THE capstone: for the actual training
    iterate `muonTraj`, given ONLY (i) the initial MLP shape and initial summed norm `≤ R`, (ii) the per-tensor
    per-step bounded-gradient DATA — uniform `E`/`B` with per-step entry/ortho hypotheses for `W1`, `W2`, and
    uniform update-forcing bounds `Cb1`, `Cb2` for `b1`, `b2` — (iii) weight decay dominant (`ρ_b =
    (1+u64)²·|toReal(1−lr·wd)| < 1`) and `R` self-bounding for the total forcing, the whole-network norm
    `‖toMatrixF W1‖ + ‖toMatrixF W2‖ + |toReal b1[i]| + |toReal b2[j]|` stays `≤ R` for EVERY training step. The
    four recurrences are discharged from the concrete `applyMuon` (`muonTraj_*_recur`), unified to the common
    contraction and summed (`muonTraj_network_bounded'`). No recurrence, invariance, shape, or ρ-matching
    hypothesis survives — only the initial state, the self-bounding radius, and the bounded-gradient forcing.
    The complete, self-contained Muon whole-network training-stability certificate, grounded end-to-end in the
    running trainer's actual update and resting on the dimension-free O(1) Newton–Schulz spectral kernel. -/
theorem muonTraj_network_bounded_full {r1 c1 r2 c2 : Nat} [Nonempty (Fin c1)] [Nonempty (Fin c2)]
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (i j : Nat)
    (EW1 BW1 EW2 BW2 Cb1 Cb2 R : ℝ) (hEW1 : 0 ≤ EW1) (hEW2 : 0 ≤ EW2)
    (hr1 : 0 < r1) (hr2 : 0 < r2) (hi : i < p0.b1.size) (hj : j < p0.b2.size)
    (hp0W1sz : p0.W1.size = r1) (hp0W1row : ∀ i, i < r1 → (p0.W1[i]!).size = c1)
    (hp0W2sz : p0.W2.size = r2) (hp0W2row : ∀ i, i < r2 → (p0.W2[i]!).size = c2)
    (hρ1 : (1 + u64) ^ 2 * |toReal (1.0 - lr * wd)| < 1)
    (hR : ((|toReal (lr * Float.sqrt (max 1.0 (Float.ofNat p0.W1.size / Float.ofNat (p0.W1[0]!).size)))| * BW1
            + Real.sqrt ((r1 : ℝ) * c1) * EW1)
          + (|toReal (lr * Float.sqrt (max 1.0 (Float.ofNat p0.W2.size / Float.ofNat (p0.W2[0]!).size)))| * BW2
            + Real.sqrt ((r2 : ℝ) * c2) * EW2) + Cb1 + Cb2)
        / (1 - (1 + u64) ^ 2 * |toReal (1.0 - lr * wd)|) ≤ R)
    (hS0 : ‖toMatrixF r1 c1 p0.W1‖ + ‖toMatrixF r2 c2 p0.W2‖
        + |toReal (p0.b1[i]!)| + |toReal (p0.b2[j]!)| ≤ R)
    (hentryW1 : ∀ n, ∀ (a : Fin r1) (b : Fin c1),
      matLinEntryErrBnd (1.0 - lr * wd) (((( muonTraj grad lr wd mu eps p0 st0 n).1.W1)[a.1]!)[b.1]!)
        (lr * Float.sqrt (max 1.0 (Float.ofNat ((muonTraj grad lr wd mu eps p0 st0 n).1.W1).size
          / Float.ofNat (((muonTraj grad lr wd mu eps p0 st0 n).1.W1)[0]!).size)))
        (((newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
          (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
            ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) eps)[a.1]!)[b.1]!) 0 0 ≤ EW1)
    (horthoW1 : ∀ n, ‖toMatrixF r1 c1 (newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
      (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
        ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) eps)‖ ≤ BW1)
    (hentryW2 : ∀ n, ∀ (a : Fin r2) (b : Fin c2),
      matLinEntryErrBnd (1.0 - lr * wd) (((( muonTraj grad lr wd mu eps p0 st0 n).1.W2)[a.1]!)[b.1]!)
        (lr * Float.sqrt (max 1.0 (Float.ofNat ((muonTraj grad lr wd mu eps p0 st0 n).1.W2).size
          / Float.ofNat (((muonTraj grad lr wd mu eps p0 st0 n).1.W2)[0]!).size)))
        (((newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
          (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
            ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) eps)[a.1]!)[b.1]!) 0 0 ≤ EW2)
    (horthoW2 : ∀ n, ‖toMatrixF r2 c2 (newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
      (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
        ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) eps)‖ ≤ BW2)
    (hforceb1 : ∀ n, (1 + u64) ^ 2 * (|toReal lr| * |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.1)[i]!
        + mu * (mu * ((muonTraj grad lr wd mu eps p0 st0 n).2.mb1)[i]!
          + ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.1)[i]!))|) ≤ Cb1)
    (hforceb2 : ∀ n, (1 + u64) ^ 2 * (|toReal lr| * |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.2)[j]!
        + mu * (mu * ((muonTraj grad lr wd mu eps p0 st0 n).2.mb2)[j]!
          + ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.2)[j]!))|) ≤ Cb2) :
    ∀ n, ‖toMatrixF r1 c1 (muonTraj grad lr wd mu eps p0 st0 n).1.W1‖
        + ‖toMatrixF r2 c2 (muonTraj grad lr wd mu eps p0 st0 n).1.W2‖
        + |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b1[i]!)|
        + |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b2[j]!)| ≤ R :=
  muonTraj_network_bounded' grad lr wd mu eps p0 st0 i j
    (|toReal (lr * Float.sqrt (max 1.0 (Float.ofNat p0.W1.size / Float.ofNat (p0.W1[0]!).size)))| * BW1
      + Real.sqrt ((r1 : ℝ) * c1) * EW1)
    (|toReal (lr * Float.sqrt (max 1.0 (Float.ofNat p0.W2.size / Float.ofNat (p0.W2[0]!).size)))| * BW2
      + Real.sqrt ((r2 : ℝ) * c2) * EW2) Cb1 Cb2 R hρ1 hR hS0
    (muonTraj_W1_recur grad lr wd mu eps p0 st0 EW1 BW1 hEW1 hr1 hp0W1sz hp0W1row hentryW1 horthoW1)
    (muonTraj_W2_recur grad lr wd mu eps p0 st0 EW2 BW2 hEW2 hr2 hp0W2sz hp0W2row hentryW2 horthoW2)
    (muonTraj_b1_recur grad lr wd mu eps p0 st0 i hi Cb1 hforceb1)
    (muonTraj_b2_recur grad lr wd mu eps p0 st0 j hj Cb2 hforceb2)

/-! ### Grinding the `b2` forcing down to gradient/momentum magnitudes

The `hforceb2` hypothesis of the capstone bounds the per-step Nesterov update magnitude. `nesterov_upd_abs_le`
bounds that update (`g + mu·(mu·m + g)`) by the gradient (`g`) and momentum (`m`) entry magnitudes, so
`muonTraj_b2_force_le` discharges `hforceb2` from UNIFORM bounds `G` (gradient) and `Mm` (momentum) on the
`j`-th entry along the trajectory — with the concrete forcing constant `Cb2 = (1+u64)⁶·|toReal lr|·(|toReal
mu|²·Mm + |toReal mu|·G + G)`. -/

/-- **The `b2` per-step forcing, discharged from uniform gradient/momentum bounds.** Given uniform bounds `G`
    on the `j`-th gradient entry and `Mm` on the `j`-th momentum entry along the trajectory, the `b2` forcing
    is `≤ (1+u64)⁶·|toReal lr|·(|toReal mu|²·Mm + |toReal mu|·G + G)` — so `hforceb2` (with this `Cb2`) follows
    from the bounded-gradient/bounded-momentum data via `nesterov_upd_abs_le`. -/
theorem muonTraj_b2_force_le
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (j : Nat) (G Mm : ℝ)
    (hG : ∀ n, |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.2)[j]!)| ≤ G)
    (hM : ∀ n, |toReal (((muonTraj grad lr wd mu eps p0 st0 n).2.mb2)[j]!)| ≤ Mm) :
    ∀ n, (1 + u64) ^ 2 * (|toReal lr| * |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.2)[j]!
        + mu * (mu * ((muonTraj grad lr wd mu eps p0 st0 n).2.mb2)[j]!
          + ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.2)[j]!))|)
      ≤ (1 + u64) ^ 6 * (|toReal lr| * (|toReal mu| ^ 2 * Mm + |toReal mu| * G + G)) := by
  intro n
  have hlr : (0 : ℝ) ≤ |toReal lr| := abs_nonneg _
  have hupd := nesterov_upd_abs_le (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.2)[j]!)
    mu (((muonTraj grad lr wd mu eps p0 st0 n).2.mb2)[j]!)
  have hmm : (0 : ℝ) ≤ |toReal mu| := abs_nonneg _
  have hbnd : |toReal mu| ^ 2 * |toReal (((muonTraj grad lr wd mu eps p0 st0 n).2.mb2)[j]!)|
        + |toReal mu| * |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.2)[j]!)|
        + |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.2)[j]!)|
      ≤ |toReal mu| ^ 2 * Mm + |toReal mu| * G + G := by
    nlinarith [hG n, hM n, hmm, abs_nonneg (toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.2)[j]!))]
  set upd := |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.2)[j]!
      + mu * (mu * ((muonTraj grad lr wd mu eps p0 st0 n).2.mb2)[j]!
        + ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.2)[j]!))| with hupddef
  set S := |toReal mu| ^ 2 * |toReal (((muonTraj grad lr wd mu eps p0 st0 n).2.mb2)[j]!)|
        + |toReal mu| * |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.2)[j]!)|
        + |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.2)[j]!)| with hSdef
  calc (1 + u64) ^ 2 * (|toReal lr| * upd)
      = |toReal lr| * ((1 + u64) ^ 2 * upd) := by ring
    _ ≤ |toReal lr| * ((1 + u64) ^ 2 * ((1 + u64) ^ 4 * S)) :=
        mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hupd (by positivity)) hlr
    _ = (1 + u64) ^ 6 * (|toReal lr| * S) := by ring
    _ ≤ (1 + u64) ^ 6 * (|toReal lr| * (|toReal mu| ^ 2 * Mm + |toReal mu| * G + G)) :=
        mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hbnd hlr) (by positivity)

/-! ### Grinding the weight-side `E` to fundamental entry magnitudes

`matLinEntryErrBnd_le` reduces the per-step weight-update rounding `E` (`matLinEntryErrBnd (1−lr·wd) W[i][j]
(lr·scale) ortho[i][j] 0 0`) to the coefficient·entry magnitudes. `muonTraj_W1_entry_le` discharges the `E`
hypothesis of `muonTraj_W1_step` from UNIFORM bounds on the weight entry (`WM`), orthogonalized-update entry
(`OM`), and scale coefficient (`SC`) along the trajectory — with `E = u64·(2+u64)·(|toReal(1−lr·wd)|·WM +
SC·OM)`. (`WM` is bounded by the operator-norm invariant `R` since `|A_ij| ≤ ‖A‖₂`; `OM` by
`newtonSchulz_opNorm`; `SC` by the fixed scale.) So the weight-side `E` grinds to the same kind of fundamental
entry-magnitude data as the bias forcing. -/

/-- **The `W1` per-step entry rounding, discharged from uniform entry/scale bounds.** Given uniform bounds `WM`
    on the weight entries, `OM` on the orthogonalized-update entries, and `SC` on the scale coefficient
    `|toReal(lr·scaleₙ)|` along the trajectory, the `matLinEntryErrBnd` per-entry rounding is `≤ u64·(2+u64)·
    (|toReal(1−lr·wd)|·WM + SC·OM)` — so the `hentry` hypothesis of `muonTraj_W1_step`/`muonTraj_W1_recur` (with
    this `E`) follows from fundamental entry-magnitude data via `matLinEntryErrBnd_le`. -/
theorem muonTraj_W1_entry_le {r c : Nat}
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (WM OM SC : ℝ) (hSCnn : 0 ≤ SC)
    (hWM : ∀ n, ∀ (i : Fin r) (j : Fin c),
      |toReal (((muonTraj grad lr wd mu eps p0 st0 n).1.W1)[i.1]!)[j.1]!| ≤ WM)
    (hOM : ∀ n, ∀ (i : Fin r) (j : Fin c),
      |toReal (((newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
        (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) eps)[i.1]!)[j.1]!)| ≤ OM)
    (hSC : ∀ n, |toReal (lr * Float.sqrt (max 1.0 (Float.ofNat ((muonTraj grad lr wd mu eps p0 st0 n).1.W1).size
        / Float.ofNat (((muonTraj grad lr wd mu eps p0 st0 n).1.W1)[0]!).size)))| ≤ SC) :
    ∀ n, ∀ (i : Fin r) (j : Fin c),
      matLinEntryErrBnd (1.0 - lr * wd) (((( muonTraj grad lr wd mu eps p0 st0 n).1.W1)[i.1]!)[j.1]!)
        (lr * Float.sqrt (max 1.0 (Float.ofNat ((muonTraj grad lr wd mu eps p0 st0 n).1.W1).size
          / Float.ofNat (((muonTraj grad lr wd mu eps p0 st0 n).1.W1)[0]!).size)))
        (((newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
          (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
            ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) eps)[i.1]!)[j.1]!) 0 0
      ≤ u64 * (2 + u64) * (|toReal (1.0 - lr * wd)| * WM + SC * OM) := by
  intro n i j
  refine le_trans (matLinEntryErrBnd_le _ _ _ _) ?_
  have hu : (0 : ℝ) ≤ u64 * (2 + u64) := mul_nonneg u64_pos.le (by linarith [u64_pos.le])
  have h1 : |toReal (1.0 - lr * wd)| * |toReal (((muonTraj grad lr wd mu eps p0 st0 n).1.W1)[i.1]!)[j.1]!|
      ≤ |toReal (1.0 - lr * wd)| * WM := mul_le_mul_of_nonneg_left (hWM n i j) (abs_nonneg _)
  have h2 : |toReal (lr * Float.sqrt (max 1.0 (Float.ofNat ((muonTraj grad lr wd mu eps p0 st0 n).1.W1).size
        / Float.ofNat (((muonTraj grad lr wd mu eps p0 st0 n).1.W1)[0]!).size)))|
      * |toReal (((newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
        (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) eps)[i.1]!)[j.1]!)|
      ≤ SC * OM := mul_le_mul (hSC n) (hOM n i j) (abs_nonneg _) hSCnn
  nlinarith [h1, h2, hu]

/-! ### Grinding the weight-side `B` to `newtonSchulz_opNorm`

`hortho`'s bound on `‖toMatrixF ortho‖` (the orthogonalized-update operator norm) is exactly what
`NewtonSchulzFull.newtonSchulz_opNorm_le` bounds by `√1.3131 + √(r·c)·rounding` — DIMENSION-FREE O(1) in the
tight spectral constant. `muonTraj_W1_ortho_le` maps it over the trajectory: each step's orthogonalized update
has operator norm `≤ √1.3131 + √(r·c)·(rounding)`, discharging `B` to the Newton–Schulz spectral kernel — given
the update's `MatBnd` mirror (`M`, `ε`) and its seed normalization (`‖·‖² ≤ 1`). -/

/-- **The `W1` orthogonalized-update norm, ground to `newtonSchulz_opNorm_le`.** Each step's ortho matrix
    `newtonSchulz update_n eps` (the Muon iteration on the Nesterov update) has operator norm `≤ √1.3131 +
    √(r·c)·(rounding)` — the dimension-free O(1) spectral bound — given the per-step `MatBnd` mirror and seed
    normalization for `update_n`. Discharges `hortho`'s `B` to the Newton–Schulz kernel: `B` is the tight
    spectral constant `√1.3131 < 1.15` plus explicit rounding. -/
theorem muonTraj_W1_ortho_le {r c : Nat}
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (updateR : Nat → MatR) (M ε : ℝ)
    (hr : 0 < r) (hc : 0 < c) (hM : 0 ≤ M) (hrc : r ≤ c)
    (hX0 : ∀ n, MatBnd (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
        (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) (updateR n) r c M ε)
    (hseed : ∀ n, ‖toMatrixR r c (scalarMulR (toReal (1.0 / (frobNorm
        (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
          (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
            ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) + eps))) (updateR n))‖ ^ 2 ≤ 1) :
    ∀ n, ‖toMatrixF r c (newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
        (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) eps)‖
      ≤ Real.sqrt 1.3131 + Real.sqrt ((r : ℝ) * c)
          * (muonCoeffs.toList.foldl (fun e a => nsIterBnd r c a e)
              ((1 + u64) * (|toReal (1.0 / (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) + eps))| * M),
               u64 * (|toReal (1.0 / (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) + eps))| * M)
                 + |toReal (1.0 / (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) + eps))| * ε)).2 :=
  fun n => newtonSchulz_opNorm_le _ (updateR n) eps r c M ε hr hc hM hrc (hX0 n) (hseed n)

/-- **The `W2` orthogonalized-update norm, ground to `newtonSchulz_opNorm_le`.** As `muonTraj_W1_ortho_le`, for
    the second weight matrix (gradient component `.2.2.1`, momentum `.2.mW2`): each step's ortho matrix has
    operator norm `≤ √1.3131 + √(r·c)·(rounding)` — the dimension-free O(1) spectral bound. -/
theorem muonTraj_W2_ortho_le {r c : Nat}
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (updateR : Nat → MatR) (M ε : ℝ)
    (hr : 0 < r) (hc : 0 < c) (hM : 0 ≤ M) (hrc : r ≤ c)
    (hX0 : ∀ n, MatBnd (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
        (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) (updateR n) r c M ε)
    (hseed : ∀ n, ‖toMatrixR r c (scalarMulR (toReal (1.0 / (frobNorm
        (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
          (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
            ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) + eps))) (updateR n))‖ ^ 2 ≤ 1) :
    ∀ n, ‖toMatrixF r c (newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
        (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) eps)‖
      ≤ Real.sqrt 1.3131 + Real.sqrt ((r : ℝ) * c)
          * (muonCoeffs.toList.foldl (fun e a => nsIterBnd r c a e)
              ((1 + u64) * (|toReal (1.0 / (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) + eps))| * M),
               u64 * (|toReal (1.0 / (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) + eps))| * M)
                 + |toReal (1.0 / (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) + eps))| * ε)).2 :=
  fun n => newtonSchulz_opNorm_le _ (updateR n) eps r c M ε hr hc hM hrc (hX0 n) (hseed n)

/-! ### Grinding the `W1` ortho seed normalization to runnable `Bool` checks

`muonTraj_W1_ortho_le` still takes the abstract `MatBnd` mirror and the seed normalization `‖·‖² ≤ 1`.
`NewtonSchulzRunnable.newtonSchulz_opNorm_runnable` discharges BOTH from four runnable `Bool` checks on the
update matrix (`matSizeOk`, `matShapeOk`, `matEntryBnd`, `epsCheckB`). `muonTraj_W1_ortho_runnable` maps it
over the trajectory: the ortho bound holds given only those decidable checks on each step's update — the seed
normalization is now a runtime check, the faithful mirror is built (`mirrorOf`), and the magnitude is a Float
`Mf`. -/

/-- **The `W1` orthogonalized-update norm, from RUNNABLE `Bool` checks.** As `muonTraj_W1_ortho_le`, but the
    `MatBnd` mirror and seed normalization are discharged from the four runnable checks on the update matrix:
    `matSizeOk r c` (dims), `matShapeOk r c update_n` (shape), `matEntryBnd Mf update_n` (finiteness),
    `epsCheckB (cfConst r c) eps (frobNorm update_n)` (eps margin). The ortho bound `≤ √1.3131 + √(r·c)·rounding`
    (dimension-free O(1)) then follows per step with `M := toReal Mf` — no abstract seed normalization. -/
theorem muonTraj_W1_ortho_runnable {r c : Nat}
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps Mf : Float) (p0 : MLP) (st0 : MuonState)
    (hr : 0 < r) (hc : 0 < c) (hrc : r ≤ c) (hsize : matSizeOk r c = true)
    (hshape : ∀ n, matShapeOk r c (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
        (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) = true)
    (hbnd : ∀ n, matEntryBnd Mf (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
        (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) = true)
    (hb : ∀ n, epsCheckB (cfConst r c) eps (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
        (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1)))) = true) :
    ∀ n, ‖toMatrixF r c (newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
        (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) eps)‖
      ≤ Real.sqrt 1.3131 + Real.sqrt ((r : ℝ) * c)
          * (muonCoeffs.toList.foldl (fun e a => nsIterBnd r c a e)
              ((1 + u64) * (|toReal (1.0 / (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) + eps))| * toReal Mf),
               u64 * (|toReal (1.0 / (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) + eps))| * toReal Mf)
                 + |toReal (1.0 / (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) + eps))| * 0)).2 :=
  fun n => newtonSchulz_opNorm_runnable _ eps Mf r c hr hc hrc hsize (hshape n) (hbnd n) (hb n)

/-- **The `W1` ortho norm with the magnitude COMPUTED (`matMaxAbs`), no free `Mf`.** As
    `muonTraj_W1_ortho_runnable`, but the per-step magnitude is `matMaxAbs update_n` (the folded max `|entry|`
    of each step's update), so there is no free magnitude parameter; `matEntryBnd (matMaxAbs update_n)
    update_n` is the runtime finiteness check, and the ortho bound uses `M := toReal (matMaxAbs update_n)`. -/
theorem muonTraj_W1_ortho_selfM {r c : Nat}
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState)
    (hr : 0 < r) (hc : 0 < c) (hrc : r ≤ c) (hsize : matSizeOk r c = true)
    (hshape : ∀ n, matShapeOk r c (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
        (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) = true)
    (hbnd : ∀ n, matEntryBnd (matMaxAbs (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
        (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))))
        (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
          (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
            ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) = true)
    (hb : ∀ n, epsCheckB (cfConst r c) eps (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
        (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1)))) = true) :
    ∀ n, ‖toMatrixF r c (newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
        (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) eps)‖
      ≤ Real.sqrt 1.3131 + Real.sqrt ((r : ℝ) * c)
          * (muonCoeffs.toList.foldl (fun e a => nsIterBnd r c a e)
              ((1 + u64) * (|toReal (1.0 / (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) + eps))|
                * toReal (matMaxAbs (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))))),
               u64 * (|toReal (1.0 / (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) + eps))|
                * toReal (matMaxAbs (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1)))))
                 + |toReal (1.0 / (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) + eps))| * 0)).2 :=
  fun n => newtonSchulz_opNorm_runnable _ eps _ r c hr hc hrc hsize (hshape n) (hbnd n) (hb n)

/-- **The `W1` ortho norm with BOTH magnitude and eps COMPUTED, no free `Mf`/`eps` on the ortho.** As
    `muonTraj_W1_ortho_selfM`, but the orthogonalizer's eps is the per-step computed `epsDefault update_n r c`;
    so neither magnitude nor ortho eps is a free parameter. (The trajectory `eps` still drives `muonTraj`.) -/
theorem muonTraj_W1_ortho_autoEps {r c : Nat}
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState)
    (hr : 0 < r) (hc : 0 < c) (hrc : r ≤ c) (hsize : matSizeOk r c = true)
    (hshape : ∀ n, matShapeOk r c (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
        (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) = true)
    (hbnd : ∀ n, matEntryBnd (matMaxAbs (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
        (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))))
        (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
          (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
            ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) = true)
    (hb : ∀ n, epsCheckB (cfConst r c)
        (epsDefault (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
          (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
            ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) r c)
        (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
          (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
            ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1)))) = true) :
    ∀ n, ‖toMatrixF r c (newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
        (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1)))
        (epsDefault (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
          (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
            ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) r c))‖
      ≤ Real.sqrt 1.3131 + Real.sqrt ((r : ℝ) * c)
          * (muonCoeffs.toList.foldl (fun e a => nsIterBnd r c a e)
              ((1 + u64) * (|toReal (1.0 / (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1)))
                  + epsDefault (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
                    (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
                      ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) r c))|
                * toReal (matMaxAbs (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))))),
               u64 * (|toReal (1.0 / (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1)))
                  + epsDefault (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
                    (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
                      ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) r c))|
                * toReal (matMaxAbs (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1)))))
                 + |toReal (1.0 / (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1)))
                  + epsDefault (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
                    (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
                      ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) r c))| * 0)).2 :=
  fun n => newtonSchulz_opNorm_runnable _ _ _ r c hr hc hrc hsize (hshape n) (hbnd n) (hb n)

/-- **The `W2` orthogonalized-update norm, from RUNNABLE `Bool` checks.** As `muonTraj_W1_ortho_runnable`, for
    the second weight matrix (gradient component `.2.2.1`, momentum `.2.mW2`): the `MatBnd` mirror and seed
    normalization are discharged from `matSizeOk`/`matShapeOk`/`matEntryBnd`/`epsCheckB` on the update matrix,
    giving the ortho bound `≤ √1.3131 + √(r·c)·rounding` per step. -/
theorem muonTraj_W2_ortho_runnable {r c : Nat}
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps Mf : Float) (p0 : MLP) (st0 : MuonState)
    (hr : 0 < r) (hc : 0 < c) (hrc : r ≤ c) (hsize : matSizeOk r c = true)
    (hshape : ∀ n, matShapeOk r c (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
        (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) = true)
    (hbnd : ∀ n, matEntryBnd Mf (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
        (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) = true)
    (hb : ∀ n, epsCheckB (cfConst r c) eps (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
        (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1)))) = true) :
    ∀ n, ‖toMatrixF r c (newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
        (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) eps)‖
      ≤ Real.sqrt 1.3131 + Real.sqrt ((r : ℝ) * c)
          * (muonCoeffs.toList.foldl (fun e a => nsIterBnd r c a e)
              ((1 + u64) * (|toReal (1.0 / (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) + eps))| * toReal Mf),
               u64 * (|toReal (1.0 / (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) + eps))| * toReal Mf)
                 + |toReal (1.0 / (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) + eps))| * 0)).2 :=
  fun n => newtonSchulz_opNorm_runnable _ eps Mf r c hr hc hrc hsize (hshape n) (hbnd n) (hb n)

/-- **The `W2` ortho norm with the magnitude COMPUTED (`matMaxAbs`), no free `Mf`.** As
    `muonTraj_W1_ortho_selfM`, for the second weight matrix (gradient component `.2.2.1`, momentum `.2.mW2`):
    the per-step magnitude is `matMaxAbs update_n`, so no free magnitude parameter; `matEntryBnd (matMaxAbs
    update_n) update_n` is the runtime finiteness check, and the ortho bound uses `M := toReal (matMaxAbs
    update_n)`. -/
theorem muonTraj_W2_ortho_selfM {r c : Nat}
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState)
    (hr : 0 < r) (hc : 0 < c) (hrc : r ≤ c) (hsize : matSizeOk r c = true)
    (hshape : ∀ n, matShapeOk r c (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
        (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) = true)
    (hbnd : ∀ n, matEntryBnd (matMaxAbs (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
        (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))))
        (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
          (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
            ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) = true)
    (hb : ∀ n, epsCheckB (cfConst r c) eps (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
        (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1)))) = true) :
    ∀ n, ‖toMatrixF r c (newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
        (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) eps)‖
      ≤ Real.sqrt 1.3131 + Real.sqrt ((r : ℝ) * c)
          * (muonCoeffs.toList.foldl (fun e a => nsIterBnd r c a e)
              ((1 + u64) * (|toReal (1.0 / (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) + eps))|
                * toReal (matMaxAbs (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))))),
               u64 * (|toReal (1.0 / (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) + eps))|
                * toReal (matMaxAbs (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1)))))
                 + |toReal (1.0 / (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) + eps))| * 0)).2 :=
  fun n => newtonSchulz_opNorm_runnable _ eps _ r c hr hc hrc hsize (hshape n) (hbnd n) (hb n)

/-- **The `W2` ortho norm with BOTH magnitude and eps COMPUTED, no free `Mf`/`eps` on the ortho.** As
    `muonTraj_W2_ortho_selfM`, but the orthogonalizer's eps is the per-step computed `epsDefault update_n r c`
    (`= cfConst r c · ‖update_n‖_F`), so neither the magnitude nor the ortho eps is a free parameter — the
    orthogonalized update `newtonSchulz update_n (epsDefault update_n r c)` has operator norm `≤ √1.3131 +
    √(r·c)·rounding` given only the runnable `matShapeOk`/`matEntryBnd (matMaxAbs …)`/`epsCheckB` checks on the
    update matrix. (The trajectory `eps` still drives `muonTraj`; only the orthogonalizer's eps is computed.) -/
theorem muonTraj_W2_ortho_autoEps {r c : Nat}
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState)
    (hr : 0 < r) (hc : 0 < c) (hrc : r ≤ c) (hsize : matSizeOk r c = true)
    (hshape : ∀ n, matShapeOk r c (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
        (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) = true)
    (hbnd : ∀ n, matEntryBnd (matMaxAbs (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
        (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))))
        (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
          (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
            ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) = true)
    (hb : ∀ n, epsCheckB (cfConst r c)
        (epsDefault (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
          (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
            ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) r c)
        (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
          (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
            ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1)))) = true) :
    ∀ n, ‖toMatrixF r c (newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
        (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1)))
        (epsDefault (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
          (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
            ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) r c))‖
      ≤ Real.sqrt 1.3131 + Real.sqrt ((r : ℝ) * c)
          * (muonCoeffs.toList.foldl (fun e a => nsIterBnd r c a e)
              ((1 + u64) * (|toReal (1.0 / (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1)))
                  + epsDefault (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
                    (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
                      ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) r c))|
                * toReal (matMaxAbs (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))))),
               u64 * (|toReal (1.0 / (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1)))
                  + epsDefault (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
                    (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
                      ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) r c))|
                * toReal (matMaxAbs (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1)))))
                 + |toReal (1.0 / (frobNorm (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
                  (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
                    ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1)))
                  + epsDefault (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
                    (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
                      ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) r c))| * 0)).2 :=
  fun n => newtonSchulz_opNorm_runnable _ _ _ r c hr hc hrc hsize (hshape n) (hbnd n) (hb n)

/-- **The `b1` per-step forcing, discharged from uniform gradient/momentum bounds.** As `muonTraj_b2_force_le`,
    for the first bias (gradient component `.2.1`, momentum `.2.mb1`): given uniform bounds `G` on the `i`-th
    gradient entry and `Mm` on the `i`-th momentum entry, the `b1` forcing is `≤ (1+u64)⁶·|toReal lr|·
    (|toReal mu|²·Mm + |toReal mu|·G + G)`, so `hforceb1` follows from bounded-gradient/momentum data. -/
theorem muonTraj_b1_force_le
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (i : Nat) (G Mm : ℝ)
    (hG : ∀ n, |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.1)[i]!)| ≤ G)
    (hM : ∀ n, |toReal (((muonTraj grad lr wd mu eps p0 st0 n).2.mb1)[i]!)| ≤ Mm) :
    ∀ n, (1 + u64) ^ 2 * (|toReal lr| * |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.1)[i]!
        + mu * (mu * ((muonTraj grad lr wd mu eps p0 st0 n).2.mb1)[i]!
          + ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.1)[i]!))|)
      ≤ (1 + u64) ^ 6 * (|toReal lr| * (|toReal mu| ^ 2 * Mm + |toReal mu| * G + G)) := by
  intro n
  have hlr : (0 : ℝ) ≤ |toReal lr| := abs_nonneg _
  have hupd := nesterov_upd_abs_le (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.1)[i]!)
    mu (((muonTraj grad lr wd mu eps p0 st0 n).2.mb1)[i]!)
  have hmm : (0 : ℝ) ≤ |toReal mu| := abs_nonneg _
  have hbnd : |toReal mu| ^ 2 * |toReal (((muonTraj grad lr wd mu eps p0 st0 n).2.mb1)[i]!)|
        + |toReal mu| * |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.1)[i]!)|
        + |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.1)[i]!)|
      ≤ |toReal mu| ^ 2 * Mm + |toReal mu| * G + G := by
    nlinarith [hG n, hM n, hmm, abs_nonneg (toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.1)[i]!))]
  set upd := |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.1)[i]!
      + mu * (mu * ((muonTraj grad lr wd mu eps p0 st0 n).2.mb1)[i]!
        + ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.1)[i]!))| with hupddef
  set S := |toReal mu| ^ 2 * |toReal (((muonTraj grad lr wd mu eps p0 st0 n).2.mb1)[i]!)|
        + |toReal mu| * |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.1)[i]!)|
        + |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.1)[i]!)| with hSdef
  calc (1 + u64) ^ 2 * (|toReal lr| * upd)
      = |toReal lr| * ((1 + u64) ^ 2 * upd) := by ring
    _ ≤ |toReal lr| * ((1 + u64) ^ 2 * ((1 + u64) ^ 4 * S)) :=
        mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hupd (by positivity)) hlr
    _ = (1 + u64) ^ 6 * (|toReal lr| * S) := by ring
    _ ≤ (1 + u64) ^ 6 * (|toReal lr| * (|toReal mu| ^ 2 * Mm + |toReal mu| * G + G)) :=
        mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hbnd hlr) (by positivity)

set_option maxHeartbeats 1000000 in
/-- **Per-step recurrence for the `b2` momentum entry, DISCHARGED from `applyMuon`.** As `muonTraj_mb1_step`,
    for the second bias momentum (`stepVec.2` on `.2.2.2`, `.2.mb2`). -/
theorem muonTraj_mb2_step
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (j : Nat) (hj : j < p0.b2.size) (n : Nat) :
    |toReal ((muonTraj grad lr wd mu eps p0 st0 (n + 1)).2.mb2[j]!)|
      ≤ (1 + u64) ^ 2 * (|toReal mu| * |toReal ((muonTraj grad lr wd mu eps p0 st0 n).2.mb2[j]!)|)
        + (1 + u64) * |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.2)[j]!)| := by
  have heq : (muonTraj grad lr wd mu eps p0 st0 (n + 1)).2.mb2
      = (stepVec (muonTraj grad lr wd mu eps p0 st0 n).1.b2
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.2)
          ((muonTraj grad lr wd mu eps p0 st0 n).2.mb2) lr wd mu).2 := rfl
  rw [heq, stepVec_snd_get _ _ _ lr wd mu j (by rw [muonTraj_b2_size]; exact hj)]
  exact mulAdd_abs_le mu _ _

/-- **The `b2` momentum entry stays bounded across training.** As `muonTraj_mb1_bounded`, for the second bias
    momentum: under bounded gradients (`G`) and a self-bounding radius, `|toReal (mb2_n[j])| ≤ R` for EVERY
    step — discharges the `Mm` hypothesis of `muonTraj_b2_force_le` (`Mm := R`). With `muonTraj_mb1_bounded`,
    BOTH bias momenta stay bounded under bounded gradients. -/
theorem muonTraj_mb2_bounded
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (j : Nat) (hj : j < p0.b2.size) (G R : ℝ)
    (h0 : |toReal (st0.mb2[j]!)| ≤ R)
    (hself : (1 + u64) ^ 2 * |toReal mu| * R + (1 + u64) * G ≤ R)
    (hG : ∀ n, |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.2)[j]!)| ≤ G) :
    ∀ n, |toReal ((muonTraj grad lr wd mu eps p0 st0 n).2.mb2[j]!)| ≤ R := by
  have hu : (0 : ℝ) ≤ u64 := u64_pos.le
  refine region_invariant (fun n => |toReal ((muonTraj grad lr wd mu eps p0 st0 n).2.mb2[j]!)|)
    ((1 + u64) ^ 2 * |toReal mu|) ((1 + u64) * G) R (by positivity) hself h0 ?_
  intro n
  nlinarith [muonTraj_mb2_step grad lr wd mu eps p0 st0 j hj n, hG n, hu]

/-! ### The `mW1` weight momentum stays bounded across training

The 2D weight momentum obeys `mW1_{n+1} = matLin μ mW1_n 1.0 grad_n` (`stepMat.2` = `newMom = μ·mom + grad`),
so its OPERATOR NORM follows an affine recurrence with contraction `|toReal μ|` (`matLin_opNorm_le`). Under a
bounded gradient operator norm (`GN`) and bounded per-entry `matLin` rounding (`E`), the weight momentum stays
bounded for all training time. -/

/-- `mW1`'s outer size is preserved along the trajectory (`matLin` preserves it). -/
theorem muonTraj_mW1_size
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) :
    ∀ n, (muonTraj grad lr wd mu eps p0 st0 n).2.mW1.size = st0.mW1.size := by
  intro n
  induction n with
  | zero => rfl
  | succ m ih =>
    rw [show (muonTraj grad lr wd mu eps p0 st0 (m + 1)).2.mW1
        = matLin mu (muonTraj grad lr wd mu eps p0 st0 m).2.mW1 1.0 _ from rfl, matLin_size, ih]

/-- `mW1`'s row sizes are preserved along the trajectory. -/
theorem muonTraj_mW1_rowSize
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (i : Nat) (hi : i < st0.mW1.size) :
    ∀ n, ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1[i]!).size = (st0.mW1[i]!).size := by
  intro n
  induction n with
  | zero => rfl
  | succ m ih =>
    rw [show (muonTraj grad lr wd mu eps p0 st0 (m + 1)).2.mW1
        = matLin mu (muonTraj grad lr wd mu eps p0 st0 m).2.mW1 1.0 _ from rfl,
      matLin_rowSize _ _ _ _ i (by rw [muonTraj_mW1_size]; exact hi), ih]

set_option maxHeartbeats 1000000 in
/-- **Per-step operator-norm recurrence for the `mW1` weight momentum, DISCHARGED from `applyMuon`.**
    `mW1_{n+1} = matLin μ mW1_n 1.0 grad_n` (`stepMat.2`), bounded by `matLin_opNorm_le`:
    `‖mW1_{n+1}‖ ≤ |toReal μ|·‖mW1_n‖ + |toReal 1.0|·‖grad_n‖ + √(r·c)·E`. -/
theorem muonTraj_mW1_step {r c : Nat} [Nonempty (Fin c)]
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (E : ℝ) (hE : 0 ≤ E)
    (hmsz : st0.mW1.size = r) (hmrow : ∀ i, i < r → (st0.mW1[i]!).size = c) (n : Nat)
    (hentry : ∀ (i : Fin r) (j : Fin c),
      matLinEntryErrBnd mu (((muonTraj grad lr wd mu eps p0 st0 n).2.mW1[i.1]!)[j.1]!)
        1.0 (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1[i.1]!)[j.1]!) 0 0 ≤ E) :
    ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 (n + 1)).2.mW1‖
      ≤ |toReal mu| * ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 n).2.mW1‖
        + |toReal 1.0| * ‖toMatrixF r c ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1)‖
        + Real.sqrt ((r : ℝ) * c) * E := by
  have heq : (muonTraj grad lr wd mu eps p0 st0 (n + 1)).2.mW1
      = matLin mu (muonTraj grad lr wd mu eps p0 st0 n).2.mW1 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) := rfl
  rw [heq]
  exact matLin_opNorm_le mu _ 1.0 _ E hE (by rw [muonTraj_mW1_size]; exact hmsz)
    (fun i hi => by
      rw [muonTraj_mW1_rowSize grad lr wd mu eps p0 st0 i (by rw [hmsz]; exact hi)]; exact hmrow i hi)
    hentry

/-- **The `mW1` weight momentum's operator norm stays bounded across training.** Under a bounded gradient
    operator norm (`‖toMatrixF grad_n‖ ≤ GN`), bounded per-entry `matLin` rounding (`E`), and a self-bounding
    radius (`|toReal μ|·R + (|toReal 1.0|·GN + √(r·c)·E) ≤ R`, i.e. `μ` small enough), the weight momentum
    stays `‖toMatrixF (mW1_n)‖ ≤ R` for EVERY training step — the operator-norm affine-recurrence bound on the
    weight momentum. So the optimizer's 2D momentum buffer is certified bounded, not just the parameters. -/
theorem muonTraj_mW1_bounded {r c : Nat} [Nonempty (Fin c)]
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (E GN R : ℝ) (hE : 0 ≤ E)
    (hmsz : st0.mW1.size = r) (hmrow : ∀ i, i < r → (st0.mW1[i]!).size = c)
    (h0 : ‖toMatrixF r c st0.mW1‖ ≤ R)
    (hself : |toReal mu| * R + (|toReal (1.0 : Float)| * GN + Real.sqrt ((r : ℝ) * c) * E) ≤ R)
    (hentry : ∀ n, ∀ (i : Fin r) (j : Fin c),
      matLinEntryErrBnd mu (((muonTraj grad lr wd mu eps p0 st0 n).2.mW1[i.1]!)[j.1]!)
        1.0 (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1[i.1]!)[j.1]!) 0 0 ≤ E)
    (hGN : ∀ n, ‖toMatrixF r c ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1)‖ ≤ GN) :
    ∀ n, ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 n).2.mW1‖ ≤ R := by
  refine region_invariant (fun n => ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 n).2.mW1‖)
    |toReal mu| (|toReal (1.0 : Float)| * GN + Real.sqrt ((r : ℝ) * c) * E) R (abs_nonneg _) hself h0 ?_
  intro n
  nlinarith [muonTraj_mW1_step grad lr wd mu eps p0 st0 E hE hmsz hmrow n (hentry n), hGN n,
    abs_nonneg (toReal (1.0 : Float))]

/-! ### The `mW2` weight momentum stays bounded across training (mirrors `mW1`) -/

/-- `mW2`'s outer size is preserved along the trajectory. -/
theorem muonTraj_mW2_size
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) :
    ∀ n, (muonTraj grad lr wd mu eps p0 st0 n).2.mW2.size = st0.mW2.size := by
  intro n
  induction n with
  | zero => rfl
  | succ m ih =>
    rw [show (muonTraj grad lr wd mu eps p0 st0 (m + 1)).2.mW2
        = matLin mu (muonTraj grad lr wd mu eps p0 st0 m).2.mW2 1.0 _ from rfl, matLin_size, ih]

/-- `mW2`'s row sizes are preserved along the trajectory. -/
theorem muonTraj_mW2_rowSize
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (i : Nat) (hi : i < st0.mW2.size) :
    ∀ n, ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2[i]!).size = (st0.mW2[i]!).size := by
  intro n
  induction n with
  | zero => rfl
  | succ m ih =>
    rw [show (muonTraj grad lr wd mu eps p0 st0 (m + 1)).2.mW2
        = matLin mu (muonTraj grad lr wd mu eps p0 st0 m).2.mW2 1.0 _ from rfl,
      matLin_rowSize _ _ _ _ i (by rw [muonTraj_mW2_size]; exact hi), ih]

set_option maxHeartbeats 1000000 in
/-- **Per-step operator-norm recurrence for the `mW2` weight momentum.** As `muonTraj_mW1_step`, for the
    second weight momentum (`stepMat.2` on `.2.2.1`, `.2.mW2`). -/
theorem muonTraj_mW2_step {r c : Nat} [Nonempty (Fin c)]
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (E : ℝ) (hE : 0 ≤ E)
    (hmsz : st0.mW2.size = r) (hmrow : ∀ i, i < r → (st0.mW2[i]!).size = c) (n : Nat)
    (hentry : ∀ (i : Fin r) (j : Fin c),
      matLinEntryErrBnd mu (((muonTraj grad lr wd mu eps p0 st0 n).2.mW2[i.1]!)[j.1]!)
        1.0 (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1[i.1]!)[j.1]!) 0 0 ≤ E) :
    ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 (n + 1)).2.mW2‖
      ≤ |toReal mu| * ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 n).2.mW2‖
        + |toReal 1.0| * ‖toMatrixF r c ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1)‖
        + Real.sqrt ((r : ℝ) * c) * E := by
  have heq : (muonTraj grad lr wd mu eps p0 st0 (n + 1)).2.mW2
      = matLin mu (muonTraj grad lr wd mu eps p0 st0 n).2.mW2 1.0
          ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) := rfl
  rw [heq]
  exact matLin_opNorm_le mu _ 1.0 _ E hE (by rw [muonTraj_mW2_size]; exact hmsz)
    (fun i hi => by
      rw [muonTraj_mW2_rowSize grad lr wd mu eps p0 st0 i (by rw [hmsz]; exact hi)]; exact hmrow i hi)
    hentry

/-- **The `mW2` weight momentum's operator norm stays bounded across training.** As `muonTraj_mW1_bounded`,
    for the second weight momentum. With `muonTraj_mW1_bounded` and the two bias-momentum bounds, ALL FOUR of
    the Muon optimizer's momentum buffers are certified bounded under bounded gradients. -/
theorem muonTraj_mW2_bounded {r c : Nat} [Nonempty (Fin c)]
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (E GN R : ℝ) (hE : 0 ≤ E)
    (hmsz : st0.mW2.size = r) (hmrow : ∀ i, i < r → (st0.mW2[i]!).size = c)
    (h0 : ‖toMatrixF r c st0.mW2‖ ≤ R)
    (hself : |toReal mu| * R + (|toReal (1.0 : Float)| * GN + Real.sqrt ((r : ℝ) * c) * E) ≤ R)
    (hentry : ∀ n, ∀ (i : Fin r) (j : Fin c),
      matLinEntryErrBnd mu (((muonTraj grad lr wd mu eps p0 st0 n).2.mW2[i.1]!)[j.1]!)
        1.0 (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1[i.1]!)[j.1]!) 0 0 ≤ E)
    (hGN : ∀ n, ‖toMatrixF r c ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1)‖ ≤ GN) :
    ∀ n, ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 n).2.mW2‖ ≤ R := by
  refine region_invariant (fun n => ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 n).2.mW2‖)
    |toReal mu| (|toReal (1.0 : Float)| * GN + Real.sqrt ((r : ℝ) * c) * E) R (abs_nonneg _) hself h0 ?_
  intro n
  nlinarith [muonTraj_mW2_step grad lr wd mu eps p0 st0 E hE hmsz hmrow n (hentry n), hGN n,
    abs_nonneg (toReal (1.0 : Float))]

/-! ### The four momentum recurrences, discharged from the per-step DATA

`muonTraj_{mW1,mW2}_recur` / `muonTraj_{mb1,mb2}_recur` package the momentum step lemmas into uniform-forcing
recurrence forms (the momentum analogue of `muonTraj_{W1,W2,b1,b2}_recur`) — from the per-step data (shape +
uniform gradient op-norm `GN` / rounding `E` for the weight momenta, uniform update forcing for the bias
momenta), no recurrence hypothesis. These feed the whole-optimizer-state discharge below. -/

/-- `mW1` recurrence discharged from the per-step data (shape + uniform gradient op-norm `GN`, rounding `E`). -/
theorem muonTraj_mW1_recur {r c : Nat} [Nonempty (Fin c)]
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (E GN : ℝ) (hE : 0 ≤ E)
    (hmsz : st0.mW1.size = r) (hmrow : ∀ i, i < r → (st0.mW1[i]!).size = c)
    (hentry : ∀ n, ∀ (i : Fin r) (j : Fin c),
      matLinEntryErrBnd mu (((muonTraj grad lr wd mu eps p0 st0 n).2.mW1[i.1]!)[j.1]!)
        1.0 (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1[i.1]!)[j.1]!) 0 0 ≤ E)
    (hGN : ∀ n, ‖toMatrixF r c ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1)‖ ≤ GN) :
    ∀ n, ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 (n + 1)).2.mW1‖
      ≤ |toReal mu| * ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 n).2.mW1‖
        + (|toReal (1.0 : Float)| * GN + Real.sqrt ((r : ℝ) * c) * E) := by
  intro n
  have hstep := muonTraj_mW1_step grad lr wd mu eps p0 st0 E hE hmsz hmrow n (hentry n)
  nlinarith [hstep, hGN n, abs_nonneg (toReal (1.0 : Float))]

/-- `mW2` recurrence discharged from the per-step data. -/
theorem muonTraj_mW2_recur {r c : Nat} [Nonempty (Fin c)]
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (E GN : ℝ) (hE : 0 ≤ E)
    (hmsz : st0.mW2.size = r) (hmrow : ∀ i, i < r → (st0.mW2[i]!).size = c)
    (hentry : ∀ n, ∀ (i : Fin r) (j : Fin c),
      matLinEntryErrBnd mu (((muonTraj grad lr wd mu eps p0 st0 n).2.mW2[i.1]!)[j.1]!)
        1.0 (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1[i.1]!)[j.1]!) 0 0 ≤ E)
    (hGN : ∀ n, ‖toMatrixF r c ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1)‖ ≤ GN) :
    ∀ n, ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 (n + 1)).2.mW2‖
      ≤ |toReal mu| * ‖toMatrixF r c (muonTraj grad lr wd mu eps p0 st0 n).2.mW2‖
        + (|toReal (1.0 : Float)| * GN + Real.sqrt ((r : ℝ) * c) * E) := by
  intro n
  have hstep := muonTraj_mW2_step grad lr wd mu eps p0 st0 E hE hmsz hmrow n (hentry n)
  nlinarith [hstep, hGN n, abs_nonneg (toReal (1.0 : Float))]

/-- `mb1` recurrence discharged from the per-step forcing bound. -/
theorem muonTraj_mb1_recur
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (i : Nat) (hi : i < p0.b1.size) (C : ℝ)
    (hforce : ∀ n, (1 + u64) * |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.1)[i]!)| ≤ C) :
    ∀ n, |toReal ((muonTraj grad lr wd mu eps p0 st0 (n + 1)).2.mb1[i]!)|
      ≤ (1 + u64) ^ 2 * |toReal mu| * |toReal ((muonTraj grad lr wd mu eps p0 st0 n).2.mb1[i]!)| + C := by
  intro n
  nlinarith [muonTraj_mb1_step grad lr wd mu eps p0 st0 i hi n, hforce n]

/-- `mb2` recurrence discharged from the per-step forcing bound. -/
theorem muonTraj_mb2_recur
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (j : Nat) (hj : j < p0.b2.size) (C : ℝ)
    (hforce : ∀ n, (1 + u64) * |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.2)[j]!)| ≤ C) :
    ∀ n, |toReal ((muonTraj grad lr wd mu eps p0 st0 (n + 1)).2.mb2[j]!)|
      ≤ (1 + u64) ^ 2 * |toReal mu| * |toReal ((muonTraj grad lr wd mu eps p0 st0 n).2.mb2[j]!)| + C := by
  intro n
  nlinarith [muonTraj_mb2_step grad lr wd mu eps p0 st0 j hj n, hforce n]

/-! ### The whole optimizer state stays bounded across training

Summing all EIGHT quantities — the four parameter norms (`W1`, `W2` operator norms; `b1`, `b2` entries) and
the four momentum norms (`mW1`, `mW2` operator norms; `mb1`, `mb2` entries) — gives a single measure of the
whole optimizer state. Each obeys an affine recurrence at its own contraction; relaxed to a common `ρ` (the
caller supplies each recurrence at `ρ`), the sum obeys the same recurrence, so the ENTIRE optimizer state
(parameters + momentum buffers) stays in one bounded region for all training time. -/

set_option maxHeartbeats 1600000 in
/-- **Whole-optimizer-state bound: parameters + all momentum buffers in one region.** For the actual training
    iterate `muonTraj`, if each of the EIGHT optimizer-state quantities obeys the Muon per-step recurrence with
    a COMMON contraction `ρ < 1` and forcing `C_k`, the initial summed state is `≤ R`, and `R` is self-bounding,
    then the whole-optimizer-state measure `‖W1‖ + ‖W2‖ + |b1[i]| + |b2[j]| + ‖mW1‖ + ‖mW2‖ + |mb1[i]| +
    |mb2[j]|` stays `≤ R` for EVERY training step — a single scalar capturing the parameters AND the four
    momentum buffers, uniformly bounded across the entire training trajectory. -/
theorem muonTraj_optimizer_bounded {r1 c1 r2 c2 : Nat}
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (i j : Nat)
    (ρ CW1 CW2 Cb1 Cb2 CmW1 CmW2 Cmb1 Cmb2 R : ℝ) (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1)
    (hR : (CW1 + CW2 + Cb1 + Cb2 + CmW1 + CmW2 + Cmb1 + Cmb2) / (1 - ρ) ≤ R)
    (hS0 : ‖toMatrixF r1 c1 p0.W1‖ + ‖toMatrixF r2 c2 p0.W2‖ + |toReal (p0.b1[i]!)| + |toReal (p0.b2[j]!)|
        + ‖toMatrixF r1 c1 st0.mW1‖ + ‖toMatrixF r2 c2 st0.mW2‖ + |toReal (st0.mb1[i]!)| + |toReal (st0.mb2[j]!)| ≤ R)
    (hW1 : ∀ n, ‖toMatrixF r1 c1 (muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.W1‖ ≤ ρ * ‖toMatrixF r1 c1 (muonTraj grad lr wd mu eps p0 st0 n).1.W1‖ + CW1)
    (hW2 : ∀ n, ‖toMatrixF r2 c2 (muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.W2‖ ≤ ρ * ‖toMatrixF r2 c2 (muonTraj grad lr wd mu eps p0 st0 n).1.W2‖ + CW2)
    (hb1 : ∀ n, |toReal ((muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.b1[i]!)| ≤ ρ * |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b1[i]!)| + Cb1)
    (hb2 : ∀ n, |toReal ((muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.b2[j]!)| ≤ ρ * |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b2[j]!)| + Cb2)
    (hmW1 : ∀ n, ‖toMatrixF r1 c1 (muonTraj grad lr wd mu eps p0 st0 (n + 1)).2.mW1‖ ≤ ρ * ‖toMatrixF r1 c1 (muonTraj grad lr wd mu eps p0 st0 n).2.mW1‖ + CmW1)
    (hmW2 : ∀ n, ‖toMatrixF r2 c2 (muonTraj grad lr wd mu eps p0 st0 (n + 1)).2.mW2‖ ≤ ρ * ‖toMatrixF r2 c2 (muonTraj grad lr wd mu eps p0 st0 n).2.mW2‖ + CmW2)
    (hmb1 : ∀ n, |toReal ((muonTraj grad lr wd mu eps p0 st0 (n + 1)).2.mb1[i]!)| ≤ ρ * |toReal ((muonTraj grad lr wd mu eps p0 st0 n).2.mb1[i]!)| + Cmb1)
    (hmb2 : ∀ n, |toReal ((muonTraj grad lr wd mu eps p0 st0 (n + 1)).2.mb2[j]!)| ≤ ρ * |toReal ((muonTraj grad lr wd mu eps p0 st0 n).2.mb2[j]!)| + Cmb2) :
    ∀ n, ‖toMatrixF r1 c1 (muonTraj grad lr wd mu eps p0 st0 n).1.W1‖ + ‖toMatrixF r2 c2 (muonTraj grad lr wd mu eps p0 st0 n).1.W2‖
        + |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b1[i]!)| + |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b2[j]!)|
        + ‖toMatrixF r1 c1 (muonTraj grad lr wd mu eps p0 st0 n).2.mW1‖ + ‖toMatrixF r2 c2 (muonTraj grad lr wd mu eps p0 st0 n).2.mW2‖
        + |toReal ((muonTraj grad lr wd mu eps p0 st0 n).2.mb1[i]!)| + |toReal ((muonTraj grad lr wd mu eps p0 st0 n).2.mb2[j]!)| ≤ R := by
  refine region_invariant
    (fun n => ‖toMatrixF r1 c1 (muonTraj grad lr wd mu eps p0 st0 n).1.W1‖ + ‖toMatrixF r2 c2 (muonTraj grad lr wd mu eps p0 st0 n).1.W2‖
        + |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b1[i]!)| + |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b2[j]!)|
        + ‖toMatrixF r1 c1 (muonTraj grad lr wd mu eps p0 st0 n).2.mW1‖ + ‖toMatrixF r2 c2 (muonTraj grad lr wd mu eps p0 st0 n).2.mW2‖
        + |toReal ((muonTraj grad lr wd mu eps p0 st0 n).2.mb1[i]!)| + |toReal ((muonTraj grad lr wd mu eps p0 st0 n).2.mb2[j]!)|)
    ρ (CW1 + CW2 + Cb1 + Cb2 + CmW1 + CmW2 + Cmb1 + Cmb2) R hρ0
    (self_bounding_radius ρ _ R hρ1 hR) hS0 ?_
  intro n
  have e1 := hW1 n; have e2 := hW2 n; have e3 := hb1 n; have e4 := hb2 n
  have e5 := hmW1 n; have e6 := hmW2 n; have e7 := hmb1 n; have e8 := hmb2 n
  simp only []
  nlinarith [e1, e2, e3, e4, e5, e6, e7, e8]

/-! ### Whole-optimizer-state stability from the NATURAL per-tensor recurrences

The eight recurrences come out of the step/recur lemmas at FOUR different contractions — weights at
`ρ_W = |toReal(1−lr·wd)|`, biases at `ρ_b = (1+u64)²·|toReal(1−lr·wd)|`, weight momenta at `ρ_{mW} =
|toReal μ|`, bias momenta at `ρ_{mb} = (1+u64)²·|toReal μ|` — while `muonTraj_optimizer_bounded` needs a common
`ρ`. Unlike the whole-network case (weights and biases shared the `|toReal(1−lr·wd)|` base, so a single
expression dominated), here the two independent bases `|toReal(1−lr·wd)|` and `|toReal μ|` mean `ρ` must be a
parameter dominating both bias-shaped contractions; `recur_relax` then relaxes each to `ρ`. -/

set_option maxHeartbeats 1600000 in
/-- **Whole-optimizer-state stability from the natural per-tensor recurrences.** Takes the eight recurrences
    exactly as the step/recur lemmas produce them — weights at `ρ_W = |toReal(1−lr·wd)|`, biases at `ρ_b =
    (1+u64)²·|toReal(1−lr·wd)|`, weight momenta at `ρ_{mW} = |toReal μ|`, bias momenta at `ρ_{mb} =
    (1+u64)²·|toReal μ|` — and unifies to a common `ρ`. It suffices that `ρ` dominates the two "bias-shaped"
    contractions `(1+u64)²·|toReal(1−lr·wd)|` (`hρb`) and `(1+u64)²·|toReal μ|` (`hρmb`), since each dominates
    the corresponding weight-shaped one (`(1+u64)² ≥ 1`); every recurrence is then relaxed to `ρ` via
    `recur_relax`. Given the initial summed state `≤ R` and `R` self-bounding for the total forcing at `ρ`, the
    whole-optimizer-state measure stays `≤ R` for every training step — the eight `applyMuon`-discharged
    recurrences (`muonTraj_{W1,W2,b1,b2}_recur` + `muonTraj_{mW1,mW2,mb1,mb2}_recur`) plug straight in with no
    manual ρ-matching. -/
theorem muonTraj_optimizer_bounded' {r1 c1 r2 c2 : Nat}
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (i j : Nat)
    (CW1 CW2 Cb1 Cb2 CmW1 CmW2 Cmb1 Cmb2 ρ R : ℝ) (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1)
    (hρb : (1 + u64) ^ 2 * |toReal (1.0 - lr * wd)| ≤ ρ)
    (hρmb : (1 + u64) ^ 2 * |toReal mu| ≤ ρ)
    (hR : (CW1 + CW2 + Cb1 + Cb2 + CmW1 + CmW2 + Cmb1 + Cmb2) / (1 - ρ) ≤ R)
    (hS0 : ‖toMatrixF r1 c1 p0.W1‖ + ‖toMatrixF r2 c2 p0.W2‖ + |toReal (p0.b1[i]!)| + |toReal (p0.b2[j]!)|
        + ‖toMatrixF r1 c1 st0.mW1‖ + ‖toMatrixF r2 c2 st0.mW2‖ + |toReal (st0.mb1[i]!)| + |toReal (st0.mb2[j]!)| ≤ R)
    (hW1 : ∀ n, ‖toMatrixF r1 c1 (muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.W1‖
        ≤ |toReal (1.0 - lr * wd)| * ‖toMatrixF r1 c1 (muonTraj grad lr wd mu eps p0 st0 n).1.W1‖ + CW1)
    (hW2 : ∀ n, ‖toMatrixF r2 c2 (muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.W2‖
        ≤ |toReal (1.0 - lr * wd)| * ‖toMatrixF r2 c2 (muonTraj grad lr wd mu eps p0 st0 n).1.W2‖ + CW2)
    (hb1 : ∀ n, |toReal ((muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.b1[i]!)|
        ≤ (1 + u64) ^ 2 * |toReal (1.0 - lr * wd)| * |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b1[i]!)| + Cb1)
    (hb2 : ∀ n, |toReal ((muonTraj grad lr wd mu eps p0 st0 (n + 1)).1.b2[j]!)|
        ≤ (1 + u64) ^ 2 * |toReal (1.0 - lr * wd)| * |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b2[j]!)| + Cb2)
    (hmW1 : ∀ n, ‖toMatrixF r1 c1 (muonTraj grad lr wd mu eps p0 st0 (n + 1)).2.mW1‖
        ≤ |toReal mu| * ‖toMatrixF r1 c1 (muonTraj grad lr wd mu eps p0 st0 n).2.mW1‖ + CmW1)
    (hmW2 : ∀ n, ‖toMatrixF r2 c2 (muonTraj grad lr wd mu eps p0 st0 (n + 1)).2.mW2‖
        ≤ |toReal mu| * ‖toMatrixF r2 c2 (muonTraj grad lr wd mu eps p0 st0 n).2.mW2‖ + CmW2)
    (hmb1 : ∀ n, |toReal ((muonTraj grad lr wd mu eps p0 st0 (n + 1)).2.mb1[i]!)|
        ≤ (1 + u64) ^ 2 * |toReal mu| * |toReal ((muonTraj grad lr wd mu eps p0 st0 n).2.mb1[i]!)| + Cmb1)
    (hmb2 : ∀ n, |toReal ((muonTraj grad lr wd mu eps p0 st0 (n + 1)).2.mb2[j]!)|
        ≤ (1 + u64) ^ 2 * |toReal mu| * |toReal ((muonTraj grad lr wd mu eps p0 st0 n).2.mb2[j]!)| + Cmb2) :
    ∀ n, ‖toMatrixF r1 c1 (muonTraj grad lr wd mu eps p0 st0 n).1.W1‖ + ‖toMatrixF r2 c2 (muonTraj grad lr wd mu eps p0 st0 n).1.W2‖
        + |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b1[i]!)| + |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b2[j]!)|
        + ‖toMatrixF r1 c1 (muonTraj grad lr wd mu eps p0 st0 n).2.mW1‖ + ‖toMatrixF r2 c2 (muonTraj grad lr wd mu eps p0 st0 n).2.mW2‖
        + |toReal ((muonTraj grad lr wd mu eps p0 st0 n).2.mb1[i]!)| + |toReal ((muonTraj grad lr wd mu eps p0 st0 n).2.mb2[j]!)| ≤ R := by
  have hleW : |toReal (1.0 - lr * wd)| ≤ ρ := by
    have hle : |toReal (1.0 - lr * wd)| ≤ (1 + u64) ^ 2 * |toReal (1.0 - lr * wd)| := by
      nlinarith [abs_nonneg (toReal (1.0 - lr * wd)), u64_pos.le, sq_nonneg u64]
    exact le_trans hle hρb
  have hlemW : |toReal mu| ≤ ρ := by
    have hle : |toReal mu| ≤ (1 + u64) ^ 2 * |toReal mu| := by
      nlinarith [abs_nonneg (toReal mu), u64_pos.le, sq_nonneg u64]
    exact le_trans hle hρmb
  exact muonTraj_optimizer_bounded grad lr wd mu eps p0 st0 i j
    ρ CW1 CW2 Cb1 Cb2 CmW1 CmW2 Cmb1 Cmb2 R hρ0 hρ1 hR hS0
    (recur_relax (fun n => ‖toMatrixF r1 c1 (muonTraj grad lr wd mu eps p0 st0 n).1.W1‖)
      |toReal (1.0 - lr * wd)| ρ CW1 hleW (fun n => norm_nonneg _) hW1)
    (recur_relax (fun n => ‖toMatrixF r2 c2 (muonTraj grad lr wd mu eps p0 st0 n).1.W2‖)
      |toReal (1.0 - lr * wd)| ρ CW2 hleW (fun n => norm_nonneg _) hW2)
    (recur_relax (fun n => |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b1[i]!)|)
      ((1 + u64) ^ 2 * |toReal (1.0 - lr * wd)|) ρ Cb1 hρb (fun n => abs_nonneg _) hb1)
    (recur_relax (fun n => |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b2[j]!)|)
      ((1 + u64) ^ 2 * |toReal (1.0 - lr * wd)|) ρ Cb2 hρb (fun n => abs_nonneg _) hb2)
    (recur_relax (fun n => ‖toMatrixF r1 c1 (muonTraj grad lr wd mu eps p0 st0 n).2.mW1‖)
      |toReal mu| ρ CmW1 hlemW (fun n => norm_nonneg _) hmW1)
    (recur_relax (fun n => ‖toMatrixF r2 c2 (muonTraj grad lr wd mu eps p0 st0 n).2.mW2‖)
      |toReal mu| ρ CmW2 hlemW (fun n => norm_nonneg _) hmW2)
    (recur_relax (fun n => |toReal ((muonTraj grad lr wd mu eps p0 st0 n).2.mb1[i]!)|)
      ((1 + u64) ^ 2 * |toReal mu|) ρ Cmb1 hρmb (fun n => abs_nonneg _) hmb1)
    (recur_relax (fun n => |toReal ((muonTraj grad lr wd mu eps p0 st0 n).2.mb2[j]!)|)
      ((1 + u64) ^ 2 * |toReal mu|) ρ Cmb2 hρmb (fun n => abs_nonneg _) hmb2)

/-! ### Full whole-optimizer-state stability, discharged entirely from per-step DATA

`muonTraj_optimizer_bounded_full` is the "step discharge" analogue of `muonTraj_network_bounded_full` for the
FULL optimizer state (parameters + all four momentum buffers). Every one of the eight recurrences is discharged
from the concrete `applyMuon` update (`muonTraj_{W1,W2,b1,b2}_recur` + `muonTraj_{mW1,mW2,mb1,mb2}_recur`),
unified to a common `ρ` and summed (`muonTraj_optimizer_bounded'`). The caller supplies ONLY the initial state,
the self-bounding radius, the two ρ-domination facts, and the per-tensor bounded-gradient DATA — no recurrence,
invariance, shape, or ρ-matching hypothesis survives. -/

set_option maxHeartbeats 1600000 in
/-- **Full whole-optimizer-state Muon training stability, from per-step DATA.** THE optimizer-state capstone:
    for the actual training iterate `muonTraj`, given ONLY (i) the initial MLP/momentum shapes and initial
    summed state `≤ R`, (ii) the per-tensor per-step bounded-gradient DATA — uniform `E`/`B` with per-step
    entry/ortho hypotheses for `W1`, `W2`; uniform `E`/gradient-op-norm `GN` with per-step entry hypotheses for
    the weight momenta `mW1`, `mW2`; uniform update-forcing `Cb1`, `Cb2` for the biases and `Cmb1`, `Cmb2` for
    the bias momenta — (iii) weight decay / momentum dominant (`ρ` dominating `(1+u64)²·|toReal(1−lr·wd)|` and
    `(1+u64)²·|toReal μ|`, `ρ < 1`) and `R` self-bounding for the total forcing, the whole-optimizer-state
    measure `‖W1‖ + ‖W2‖ + |b1[i]| + |b2[j]| + ‖mW1‖ + ‖mW2‖ + |mb1[i]| + |mb2[j]|` stays `≤ R` for EVERY
    training step. The eight recurrences are discharged from the concrete `applyMuon` (`muonTraj_*_recur`),
    unified to the common contraction and summed (`muonTraj_optimizer_bounded'`). No recurrence, invariance,
    shape, or ρ-matching hypothesis survives — only the initial state, the self-bounding radius, the two
    ρ-domination facts, and the bounded-gradient forcing. The complete, self-contained Muon whole-optimizer-state
    training-stability certificate, grounded end-to-end in the running trainer's actual update and resting on
    the dimension-free O(1) Newton–Schulz spectral kernel. -/
theorem muonTraj_optimizer_bounded_full {r1 c1 r2 c2 : Nat} [Nonempty (Fin c1)] [Nonempty (Fin c2)]
    (grad : MLP → Array (Array Float) × Array Float × Array (Array Float) × Array Float)
    (lr wd mu eps : Float) (p0 : MLP) (st0 : MuonState) (i j : Nat)
    (EW1 BW1 EW2 BW2 Cb1 Cb2 EmW1 GN1 EmW2 GN2 Cmb1 Cmb2 ρ R : ℝ)
    (hEW1 : 0 ≤ EW1) (hEW2 : 0 ≤ EW2) (hEmW1 : 0 ≤ EmW1) (hEmW2 : 0 ≤ EmW2)
    (hr1 : 0 < r1) (hr2 : 0 < r2) (hi : i < p0.b1.size) (hj : j < p0.b2.size)
    (hp0W1sz : p0.W1.size = r1) (hp0W1row : ∀ i, i < r1 → (p0.W1[i]!).size = c1)
    (hp0W2sz : p0.W2.size = r2) (hp0W2row : ∀ i, i < r2 → (p0.W2[i]!).size = c2)
    (hmW1sz : st0.mW1.size = r1) (hmW1row : ∀ i, i < r1 → (st0.mW1[i]!).size = c1)
    (hmW2sz : st0.mW2.size = r2) (hmW2row : ∀ i, i < r2 → (st0.mW2[i]!).size = c2)
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1)
    (hρb : (1 + u64) ^ 2 * |toReal (1.0 - lr * wd)| ≤ ρ)
    (hρmb : (1 + u64) ^ 2 * |toReal mu| ≤ ρ)
    (hR : ((|toReal (lr * Float.sqrt (max 1.0 (Float.ofNat p0.W1.size / Float.ofNat (p0.W1[0]!).size)))| * BW1
            + Real.sqrt ((r1 : ℝ) * c1) * EW1)
          + (|toReal (lr * Float.sqrt (max 1.0 (Float.ofNat p0.W2.size / Float.ofNat (p0.W2[0]!).size)))| * BW2
            + Real.sqrt ((r2 : ℝ) * c2) * EW2) + Cb1 + Cb2
          + (|toReal (1.0 : Float)| * GN1 + Real.sqrt ((r1 : ℝ) * c1) * EmW1)
          + (|toReal (1.0 : Float)| * GN2 + Real.sqrt ((r2 : ℝ) * c2) * EmW2) + Cmb1 + Cmb2)
        / (1 - ρ) ≤ R)
    (hS0 : ‖toMatrixF r1 c1 p0.W1‖ + ‖toMatrixF r2 c2 p0.W2‖ + |toReal (p0.b1[i]!)| + |toReal (p0.b2[j]!)|
        + ‖toMatrixF r1 c1 st0.mW1‖ + ‖toMatrixF r2 c2 st0.mW2‖ + |toReal (st0.mb1[i]!)| + |toReal (st0.mb2[j]!)| ≤ R)
    (hentryW1 : ∀ n, ∀ (a : Fin r1) (b : Fin c1),
      matLinEntryErrBnd (1.0 - lr * wd) (((( muonTraj grad lr wd mu eps p0 st0 n).1.W1)[a.1]!)[b.1]!)
        (lr * Float.sqrt (max 1.0 (Float.ofNat ((muonTraj grad lr wd mu eps p0 st0 n).1.W1).size
          / Float.ofNat (((muonTraj grad lr wd mu eps p0 st0 n).1.W1)[0]!).size)))
        (((newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
          (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
            ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) eps)[a.1]!)[b.1]!) 0 0 ≤ EW1)
    (horthoW1 : ∀ n, ‖toMatrixF r1 c1 (newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1) mu
      (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW1) 1.0
        ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1))) eps)‖ ≤ BW1)
    (hentryW2 : ∀ n, ∀ (a : Fin r2) (b : Fin c2),
      matLinEntryErrBnd (1.0 - lr * wd) (((( muonTraj grad lr wd mu eps p0 st0 n).1.W2)[a.1]!)[b.1]!)
        (lr * Float.sqrt (max 1.0 (Float.ofNat ((muonTraj grad lr wd mu eps p0 st0 n).1.W2).size
          / Float.ofNat (((muonTraj grad lr wd mu eps p0 st0 n).1.W2)[0]!).size)))
        (((newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
          (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
            ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) eps)[a.1]!)[b.1]!) 0 0 ≤ EW2)
    (horthoW2 : ∀ n, ‖toMatrixF r2 c2 (newtonSchulz (matLin 1.0 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1) mu
      (matLin mu ((muonTraj grad lr wd mu eps p0 st0 n).2.mW2) 1.0
        ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1))) eps)‖ ≤ BW2)
    (hentrymW1 : ∀ n, ∀ (a : Fin r1) (b : Fin c1),
      matLinEntryErrBnd mu (((muonTraj grad lr wd mu eps p0 st0 n).2.mW1[a.1]!)[b.1]!)
        1.0 (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1[a.1]!)[b.1]!) 0 0 ≤ EmW1)
    (hGN1 : ∀ n, ‖toMatrixF r1 c1 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).1)‖ ≤ GN1)
    (hentrymW2 : ∀ n, ∀ (a : Fin r2) (b : Fin c2),
      matLinEntryErrBnd mu (((muonTraj grad lr wd mu eps p0 st0 n).2.mW2[a.1]!)[b.1]!)
        1.0 (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1[a.1]!)[b.1]!) 0 0 ≤ EmW2)
    (hGN2 : ∀ n, ‖toMatrixF r2 c2 ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.1)‖ ≤ GN2)
    (hforceb1 : ∀ n, (1 + u64) ^ 2 * (|toReal lr| * |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.1)[i]!
        + mu * (mu * ((muonTraj grad lr wd mu eps p0 st0 n).2.mb1)[i]!
          + ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.1)[i]!))|) ≤ Cb1)
    (hforceb2 : ∀ n, (1 + u64) ^ 2 * (|toReal lr| * |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.2)[j]!
        + mu * (mu * ((muonTraj grad lr wd mu eps p0 st0 n).2.mb2)[j]!
          + ((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.2)[j]!))|) ≤ Cb2)
    (hforcemb1 : ∀ n, (1 + u64) * |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.1)[i]!)| ≤ Cmb1)
    (hforcemb2 : ∀ n, (1 + u64) * |toReal (((grad (muonTraj grad lr wd mu eps p0 st0 n).1).2.2.2)[j]!)| ≤ Cmb2) :
    ∀ n, ‖toMatrixF r1 c1 (muonTraj grad lr wd mu eps p0 st0 n).1.W1‖ + ‖toMatrixF r2 c2 (muonTraj grad lr wd mu eps p0 st0 n).1.W2‖
        + |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b1[i]!)| + |toReal ((muonTraj grad lr wd mu eps p0 st0 n).1.b2[j]!)|
        + ‖toMatrixF r1 c1 (muonTraj grad lr wd mu eps p0 st0 n).2.mW1‖ + ‖toMatrixF r2 c2 (muonTraj grad lr wd mu eps p0 st0 n).2.mW2‖
        + |toReal ((muonTraj grad lr wd mu eps p0 st0 n).2.mb1[i]!)| + |toReal ((muonTraj grad lr wd mu eps p0 st0 n).2.mb2[j]!)| ≤ R :=
  muonTraj_optimizer_bounded' grad lr wd mu eps p0 st0 i j
    (|toReal (lr * Float.sqrt (max 1.0 (Float.ofNat p0.W1.size / Float.ofNat (p0.W1[0]!).size)))| * BW1
      + Real.sqrt ((r1 : ℝ) * c1) * EW1)
    (|toReal (lr * Float.sqrt (max 1.0 (Float.ofNat p0.W2.size / Float.ofNat (p0.W2[0]!).size)))| * BW2
      + Real.sqrt ((r2 : ℝ) * c2) * EW2) Cb1 Cb2
    (|toReal (1.0 : Float)| * GN1 + Real.sqrt ((r1 : ℝ) * c1) * EmW1)
    (|toReal (1.0 : Float)| * GN2 + Real.sqrt ((r2 : ℝ) * c2) * EmW2) Cmb1 Cmb2 ρ R hρ0 hρ1 hρb hρmb hR hS0
    (muonTraj_W1_recur grad lr wd mu eps p0 st0 EW1 BW1 hEW1 hr1 hp0W1sz hp0W1row hentryW1 horthoW1)
    (muonTraj_W2_recur grad lr wd mu eps p0 st0 EW2 BW2 hEW2 hr2 hp0W2sz hp0W2row hentryW2 horthoW2)
    (muonTraj_b1_recur grad lr wd mu eps p0 st0 i hi Cb1 hforceb1)
    (muonTraj_b2_recur grad lr wd mu eps p0 st0 j hj Cb2 hforceb2)
    (muonTraj_mW1_recur grad lr wd mu eps p0 st0 EmW1 GN1 hEmW1 hmW1sz hmW1row hentrymW1 hGN1)
    (muonTraj_mW2_recur grad lr wd mu eps p0 st0 EmW2 GN2 hEmW2 hmW2sz hmW2row hentrymW2 hGN2)
    (muonTraj_mb1_recur grad lr wd mu eps p0 st0 i hi Cmb1 hforcemb1)
    (muonTraj_mb2_recur grad lr wd mu eps p0 st0 j hj Cmb2 hforcemb2)

end Puffer.RL.MuonTrainLoop
