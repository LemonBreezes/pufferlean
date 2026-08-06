/-
# The unit-ball Muon whole-run interval: C59's proven `3^k` composed into the Muon step

C53 (`MuonStepLipschitz`) closed C42's open Muon step-Lipschitz `L = 1 + |lr|·Lu^k·G` — but with the NS
uniform per-ball bound `Lu` as a HYPOTHESIS (`hLu`, quantified over C50's crude nested radii, which GROW:
`nsMagBound 1 = 2`, so that hypothesis is not dischargeable along the crude radii). C59 (`NewtonSchulzBall`)
then PROVED the unit ball invariant under the classical NS step and the `k`-fold map `3^k`-Lipschitz there.
This module is the C59→C53 composition — the payoff of the invariant ball:

* `muonStep_ball_lipschitz` — for the CLASSICAL coefficients `(3/2, −1/2, 0)` and UNIT-BALL-VALUED gradients
  (`∀ x, ‖grad x‖ ≤ 1`), C53's `muonStep` is `(1 + |lr|·3^k·G)`-Lipschitz: both `grad x` and `grad y` lie in
  the invariant ball, so C59's `nsIter_lipschitz_ball` bounds the NS difference by `3^k·‖grad x − grad y‖`.
  NO `hLu` hypothesis — the NS factor `3^k` is now a THEOREM (C59), not an assumption.
* `muon_ball_whole_run` — the capstone: fed into C42's `muon_whole_run_opnorm_interval`, the runnable
  trajectory stays within `L^n·‖θ 0 − θ' 0‖ + B·Σ_{j<n} L^j` of the ideal Muon trajectory, with the concrete
  `L = 1 + |lr|·3^k·G` — the Muon whole-run error interval with the TRUE, PROVEN NS Lipschitz constant.
* `muonStep_ball_mag` / `muon_traj_drift` — the NS output lies in the unit ball (`nsIter_ball_invariant`),
  so one Muon step moves the parameter by at most `|lr|` (`‖muonStep … x‖ ≤ ‖x‖ + |lr|`), and a Muon
  trajectory drifts at most linearly (`‖θ n‖ ≤ ‖θ 0‖ + n·|lr|`) — the Muon analogue of a step-size bound,
  the param-boundedness ingredient (a UNIFORM bound would need weight decay, cf. C32/C29).

**Scope (honestly disclosed).** This eliminates C53's `hLu` hypothesis for the classical coefficients: the
`3^k` NS Lipschitz on the unit ball is C59's theorem. The unit-ball gradient hypothesis `∀ x, ‖grad x‖ ≤ 1`
is the honest model of Muon's SPECTRAL NORMALIZATION — the real optimizer normalizes the gradient/momentum
before Newton–Schulz (that is the point of NS: polish an approximately-normalized matrix toward its
orthogonal factor), so unit-ball-valued gradients are faithful, not a convenience. Remaining honest inputs:
`G` — the gradient map's operator-norm Lipschitz (C60 delivers the SUP-metric version concretely; the
sup↔opnorm dimension-factor wiring is the sibling residual); `B` — the Float-vs-ℝ Muon per-step error
(dischargeable by the Muon runtime machinery, `nsScalarF_error`-lineage). The setting is C59's C*-order
typeclass `[CStarAlgebra A][PartialOrder A][StarOrderedRing A]` (richer than C53's bare normed ∗-ring,
necessarily — the invariant ball is spectral/order-theoretic; `CStarMatrix n n ℂ` instantiates it out of the
box, the ℝ-matrix complexification being the remaining instance wiring). `L = 1 + |lr|·3^k·G ≥ 1` is
EXPANSIVE (like plain ascent, C27), so the interval is geometric — a horizon-free Muon bound would need
weight decay (cf. C32).
-/
import Puffer.RL.NewtonSchulzBall
import Puffer.RL.MuonStepLipschitz

open Puffer.RL.NewtonSchulzIterate (nsIter)
open Puffer.RL.NewtonSchulzBall (nsIter_ball_invariant nsIter_lipschitz_ball)
open Puffer.RL.MuonStepLipschitz (muonStep)
open Puffer.RL.MuonAscentBridge (muon_whole_run_opnorm_interval)

namespace Puffer.RL.MuonBallWholeRun

variable {A : Type*} [CStarAlgebra A] [PartialOrder A] [StarOrderedRing A]

/-- **The unit-ball Muon step Lipschitz — C53's `hLu` eliminated.** For the classical NS coefficients and a
    `G`-Lipschitz, UNIT-BALL-VALUED gradient map (`∀ x, ‖grad x‖ ≤ 1` — Muon's spectral normalization), the
    full Muon step `muonStep grad lr (3/2) (−1/2) 0 k` is globally `(1 + |lr|·3^k·G)`-Lipschitz. Mirrors
    C53's `muonStep_lipschitz`, with C59's `nsIter_lipschitz_ball` (both gradients in the invariant ball)
    replacing the abstract `hLu` chain — the NS factor `3^k` is a theorem here, not a hypothesis. -/
theorem muonStep_ball_lipschitz (grad : A → A) (lr : ℝ) (k : ℕ) (G : ℝ)
    (hgradLip : ∀ x y, ‖grad x - grad y‖ ≤ G * ‖x - y‖)
    (hgrad1 : ∀ x, ‖grad x‖ ≤ 1) :
    ∀ x y, ‖muonStep grad lr (3 / 2) (-(1 / 2)) 0 k x
        - muonStep grad lr (3 / 2) (-(1 / 2)) 0 k y‖
      ≤ (1 + |lr| * 3 ^ k * G) * ‖x - y‖ := by
  intro x y
  have hrw : muonStep grad lr (3 / 2) (-(1 / 2)) 0 k x
        - muonStep grad lr (3 / 2) (-(1 / 2)) 0 k y
      = (x - y) - lr • (nsIter (3 / 2) (-(1 / 2)) 0 k (grad x)
          - nsIter (3 / 2) (-(1 / 2)) 0 k (grad y)) := by
    simp only [muonStep, smul_sub]; abel
  rw [hrw]
  calc ‖(x - y) - lr • (nsIter (3 / 2) (-(1 / 2)) 0 k (grad x)
          - nsIter (3 / 2) (-(1 / 2)) 0 k (grad y))‖
      ≤ ‖x - y‖ + ‖lr • (nsIter (3 / 2) (-(1 / 2)) 0 k (grad x)
          - nsIter (3 / 2) (-(1 / 2)) 0 k (grad y))‖ := norm_sub_le _ _
    _ = ‖x - y‖ + |lr| * ‖nsIter (3 / 2) (-(1 / 2)) 0 k (grad x)
          - nsIter (3 / 2) (-(1 / 2)) 0 k (grad y)‖ := by
        rw [norm_smul, Real.norm_eq_abs]
    _ ≤ ‖x - y‖ + |lr| * (3 ^ k * ‖grad x - grad y‖) := by
        gcongr
        exact nsIter_lipschitz_ball k (grad x) (grad y) (hgrad1 x) (hgrad1 y)
    _ ≤ ‖x - y‖ + |lr| * (3 ^ k * (G * ‖x - y‖)) := by
        gcongr
        exact hgradLip x y
    _ = (1 + |lr| * 3 ^ k * G) * ‖x - y‖ := by ring

/-- **THE UNIT-BALL MUON WHOLE-RUN ERROR INTERVAL.** For the runnable trajectory `θ` within `B` per step of
    the ideal `θ'` (which follows the exact classical-coefficient Muon step on unit-ball gradients), the
    geometric interval holds with the CONCRETE, fully-proven step-Lipschitz `L = 1 + |lr|·3^k·G`:
    `‖θ n − θ' n‖ ≤ L^n·‖θ 0 − θ' 0‖ + B·Σ_{j<n} L^j`. C42's `hlip` is discharged by
    `muonStep_ball_lipschitz` — no `hLu` hypothesis anywhere; the remaining honest inputs are `G` (gradient
    Lipschitz), the unit-ball gradient bound (spectral normalization), and `B` (per-step Float error). -/
theorem muon_ball_whole_run (grad : A → A) (lr : ℝ) (k : ℕ) (G B : ℝ)
    (θ θ' : Nat → A)
    (hgradLip : ∀ x y, ‖grad x - grad y‖ ≤ G * ‖x - y‖)
    (hgrad1 : ∀ x, ‖grad x‖ ≤ 1) (hG : 0 ≤ G)
    (hstep : ∀ n, ‖θ (n + 1) - muonStep grad lr (3 / 2) (-(1 / 2)) 0 k (θ n)‖ ≤ B)
    (hideal : ∀ n, θ' (n + 1) = muonStep grad lr (3 / 2) (-(1 / 2)) 0 k (θ' n)) (n : Nat) :
    ‖θ n - θ' n‖ ≤ (1 + |lr| * 3 ^ k * G) ^ n * ‖θ 0 - θ' 0‖
      + B * ∑ j ∈ Finset.range n, (1 + |lr| * 3 ^ k * G) ^ j := by
  have hL0 : (0 : ℝ) ≤ 1 + |lr| * 3 ^ k * G := by
    have h : (0 : ℝ) ≤ |lr| * 3 ^ k * G :=
      mul_nonneg (mul_nonneg (abs_nonneg _) (by positivity)) hG
    linarith
  exact muon_whole_run_opnorm_interval (muonStep grad lr (3 / 2) (-(1 / 2)) 0 k) θ θ'
    (1 + |lr| * 3 ^ k * G) B hL0
    (muonStep_ball_lipschitz grad lr k G hgradLip hgrad1) hstep hideal n

/-- **One Muon step moves the parameter by at most `|lr|`.** The NS output lies in the unit ball
    (`nsIter_ball_invariant` on the unit-ball-valued gradient), so
    `‖muonStep … x‖ ≤ ‖x‖ + |lr|·1` — the Muon-step analogue of a step-size bound. -/
theorem muonStep_ball_mag (grad : A → A) (lr : ℝ) (k : ℕ)
    (hgrad1 : ∀ x, ‖grad x‖ ≤ 1) (x : A) :
    ‖muonStep grad lr (3 / 2) (-(1 / 2)) 0 k x‖ ≤ ‖x‖ + |lr| := by
  simp only [muonStep]
  calc ‖x - lr • nsIter (3 / 2) (-(1 / 2)) 0 k (grad x)‖
      ≤ ‖x‖ + ‖lr • nsIter (3 / 2) (-(1 / 2)) 0 k (grad x)‖ := norm_sub_le _ _
    _ = ‖x‖ + |lr| * ‖nsIter (3 / 2) (-(1 / 2)) 0 k (grad x)‖ := by
        rw [norm_smul, Real.norm_eq_abs]
    _ ≤ ‖x‖ + |lr| * 1 := by
        gcongr
        exact nsIter_ball_invariant k (grad x) (hgrad1 x)
    _ = ‖x‖ + |lr| := by ring

/-- **Linear parameter drift under Muon.** A trajectory following the unit-ball Muon step drifts at most
    linearly: `‖θ n‖ ≤ ‖θ 0‖ + n·|lr|` (each step moves the parameter by ≤ `|lr|`, `muonStep_ball_mag`).
    The Muon param-boundedness ingredient — a UNIFORM (horizon-free) bound would need weight decay
    (cf. C32/C29). -/
theorem muon_traj_drift (grad : A → A) (lr : ℝ) (k : ℕ) (θ : Nat → A)
    (hgrad1 : ∀ x, ‖grad x‖ ≤ 1)
    (htraj : ∀ n, θ (n + 1) = muonStep grad lr (3 / 2) (-(1 / 2)) 0 k (θ n)) :
    ∀ n, ‖θ n‖ ≤ ‖θ 0‖ + n * |lr| := by
  intro n
  induction n with
  | zero => simp
  | succ m ih =>
      rw [htraj m]
      calc ‖muonStep grad lr (3 / 2) (-(1 / 2)) 0 k (θ m)‖
          ≤ ‖θ m‖ + |lr| := muonStep_ball_mag grad lr k hgrad1 (θ m)
        _ ≤ (‖θ 0‖ + m * |lr|) + |lr| := by linarith
        _ = ‖θ 0‖ + ((m : ℝ) + 1) * |lr| := by ring
        _ = ‖θ 0‖ + ((m + 1 : ℕ) : ℝ) * |lr| := by push_cast; ring

end Puffer.RL.MuonBallWholeRun

-- Non-vacuity: `ℂ` (a unital C*-algebra with its canonical order) instantiates the whole-run interval out
-- of the box, exactly as C59's ball invariance does.
open Puffer.RL.MuonBallWholeRun Puffer.RL.MuonStepLipschitz in
open scoped ComplexOrder in
example (grad : ℂ → ℂ) (lr : ℝ) (k : ℕ) (G B : ℝ) (θ θ' : Nat → ℂ)
    (hgradLip : ∀ x y, ‖grad x - grad y‖ ≤ G * ‖x - y‖)
    (hgrad1 : ∀ x, ‖grad x‖ ≤ 1) (hG : 0 ≤ G)
    (hstep : ∀ n, ‖θ (n + 1) - muonStep grad lr (3 / 2) (-(1 / 2)) 0 k (θ n)‖ ≤ B)
    (hideal : ∀ n, θ' (n + 1) = muonStep grad lr (3 / 2) (-(1 / 2)) 0 k (θ' n)) (n : Nat) :
    ‖θ n - θ' n‖ ≤ (1 + |lr| * 3 ^ k * G) ^ n * ‖θ 0 - θ' 0‖
      + B * ∑ j ∈ Finset.range n, (1 + |lr| * 3 ^ k * G) ^ j :=
  muon_ball_whole_run grad lr k G B θ θ' hgradLip hgrad1 hG hstep hideal n

