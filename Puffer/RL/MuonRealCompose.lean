/-
# The routine compositions: the real-matrix Muon whole-run, and Muon-shaped horizon-free finiteness

Two compositions the C62–C67 batches noted as enabled-but-not-restated, delivered:

**Part A — the Muon whole-run interval on REAL matrices.** C62 (`MuonBallWholeRun`) proved the Muon whole-run
interval with the TRUE `3^k` NS constant in the abstract C*-order setting; C66 (`RealMatrixBall`) pulled the
unit-ball NS results back to `Matrix (Fin d) (Fin d) ℝ` with the L2 operator norm. Composing them:

* `muonStep_real_lipschitz` — on real square matrices, C53's `muonStep` (DIRECTLY reused: the four typeclasses
  `NormedRing`/`NormedAlgebra ℝ`/`StarRing`/`NormedStarGroup` all resolve under the scoped L2-operator norm, the
  last via `CStarRing.to_normedStarGroup`) with the classical coefficients on UNIT-BALL gradients is
  `(1 + |lr|·3^k·G)`-Lipschitz — C62's proof with C66's `nsIter_lipschitz_ball_real` in place of the abstract
  ball lemma. No `hLu` hypothesis; the `3^k` is a theorem on the repo's actual trainer shape.
* `muon_real_whole_run` — the capstone: C42's `muon_whole_run_opnorm_interval` at `E := Matrix (Fin d) (Fin d) ℝ`
  gives `‖θ n − θ' n‖ ≤ L^n·‖θ 0 − θ' 0‖ + B·Σ_{j<n} L^j` with `L = 1 + |lr|·3^k·G` — the Muon whole-run error
  interval ON REAL TRAINER MATRICES with the proven NS constant.
* `muonStep_real_mag` / `muon_real_traj_drift` — the real-matrix step-size bound (`‖muonStep … x‖ ≤ ‖x‖ + |lr|`,
  the NS output in the real unit ball by `nsIter_ball_invariant_real`) and the linear parameter drift
  `‖θ n‖ ≤ ‖θ 0‖ + n·|lr|`.

**Part B — the horizon-free certificate for the MUON-shaped weight-decay update.** C67 (`WdRunFinite`) delivered
the horizon-free overflow-free run for the SGD-shaped decayed update `d·w − lr·g` and noted the Muon-shaped
variant (`d·w − lr·NS(g)`) composes identically; C61 (`MuonUpdateFinite`) certified the Float NS circuit
`nsScalarF`. Composing them:

* `muonWdUpdateF w g lr d a b c := wdUpdateF w (nsScalarF a b c g) lr d` — the Muon + weight-decay update IS
  C67's decayed update at the NS-orthogonalized gradient (direct reuse, no new op tree).
* `muonWdUpdateF_mag_le` / `muonWdTrainStep_mag` — the step budget map stays AFFINE-CONTRACTIVE with the SAME
  slope `wdStepRho Bd = (1+u64)²·Bd` (the weight path is untouched by the NS circuit); only the constant changes
  to `muonWdStepC`, threading C61's `nsScalarFBound` at the `gradW` gradient budget.
* `muonWdTrainRun_all_finite_uniform` (capstone) — **the horizon-free certificate for the Muon-shaped run**:
  EVERY weight at EVERY step of an UNBOUNDED run is `isFinite`, from ONE `n`-independent budget check —
  C67's `wdRunBound`/`wdRunBound_uniform` machinery reused verbatim (it is generic in the slope/constant).
* `muonWdTrainRun_forward_all_finite_uniform` — the forward pass finite at every step, likewise `n`-independent.

**Scope (honestly disclosed).** Part A: the remaining honest inputs are exactly C62's — `G` (the gradient map's
operator-norm Lipschitz; C60 delivers the sup-metric version, C63/C65 the dimension-factor transfer, regional)
and `B` (the per-step Float error); the unit-ball gradient hypothesis is Muon's spectral normalization;
`L ≥ 1` is expansive → geometric interval. Part B: the fixed-batch model and C57 lineage persist (self-contained
linear-layer `gradW`, scalar per-singular-value NS circuit as C61, not the full `ADReverse` tape or matrix NS);
the contraction `wdStepRho Bd < 1` is C67's mild condition (any real `wd ≳ 10⁻¹⁵`). Both parts are pure
compositions of verified pieces (C42/C53/C61/C66/C67 lemmas reused by name); NO new axiom.
-/
import Puffer.RL.MuonBallWholeRun
import Puffer.RL.RealMatrixBall
import Puffer.RL.WdRunFinite
import Puffer.RL.MuonUpdateFinite

open Puffer.FloatR
open Puffer.RL.NewtonSchulzIterate (nsIter)
open Puffer.RL.RealMatrixBall (nsIter_ball_invariant_real nsIter_lipschitz_ball_real)
open Puffer.RL.MuonStepLipschitz (muonStep)
open Puffer.RL.MuonAscentBridge (muon_whole_run_opnorm_interval)
open Puffer.RL.FiniteBound (isFinite_of_bounded overflowBound dotBound dotF_isFinite)
open Puffer.RL.ForwardFinite (dotBound_mono)
open Puffer.RL.BackwardFinite (gradW gradW_mag)
open Puffer.RL.MuonUpdateFinite (nsScalarFBound nsScalarF_mag_le)
open Puffer.RL.WholeRunFinite (nsScalarFBound_nonneg)
open Puffer.RL.WdRunFinite (wdStepRho wdStepRho_nonneg wdStepRho_lt_one wdUpdateF wdUpdateF_mag_le
  wdUpdateVec wdUpdateVec_mag wdRunBound wdRunBound_uniform wdUniformBound)

namespace Puffer.RL.MuonRealCompose

/-! ## Part A — the Muon whole-run interval on REAL matrices -/

open scoped Matrix.Norms.L2Operator

variable {d : ℕ}

/-- **The Muon step is `(1 + |lr|·3^k·G)`-Lipschitz on REAL matrices.** C53's `muonStep` (directly reused — the
    scoped L2-operator-norm instances give real square matrices the full normed-∗-ring structure) with the
    classical NS coefficients and a `G`-Lipschitz, UNIT-BALL-VALUED gradient map is globally Lipschitz with the
    PROVEN NS factor `3^k` — C62's `muonStep_ball_lipschitz` transported to the repo's real trainer shape via
    C66's `nsIter_lipschitz_ball_real`. -/
theorem muonStep_real_lipschitz (grad : Matrix (Fin d) (Fin d) ℝ → Matrix (Fin d) (Fin d) ℝ)
    (lr : ℝ) (k : ℕ) (G : ℝ)
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
        exact nsIter_lipschitz_ball_real k (grad x) (grad y) (hgrad1 x) (hgrad1 y)
    _ ≤ ‖x - y‖ + |lr| * (3 ^ k * (G * ‖x - y‖)) := by
        gcongr
        exact hgradLip x y
    _ = (1 + |lr| * 3 ^ k * G) * ‖x - y‖ := by ring

/-- **THE MUON WHOLE-RUN ERROR INTERVAL ON REAL TRAINER MATRICES.** For the runnable trajectory `θ` within `B`
    per step of the ideal `θ'` (following the exact classical-coefficient Muon step on unit-ball gradients),
    `‖θ n − θ' n‖ ≤ L^n·‖θ 0 − θ' 0‖ + B·Σ_{j<n} L^j` with the concrete, fully-proven `L = 1 + |lr|·3^k·G` —
    C42's opnorm interval at `E := Matrix (Fin d) (Fin d) ℝ`, its `hlip` discharged by
    `muonStep_real_lipschitz`. Remaining honest inputs: `G`, the unit-ball gradient bound (spectral
    normalization), and `B`. -/
theorem muon_real_whole_run (grad : Matrix (Fin d) (Fin d) ℝ → Matrix (Fin d) (Fin d) ℝ)
    (lr : ℝ) (k : ℕ) (G B : ℝ) (θ θ' : Nat → Matrix (Fin d) (Fin d) ℝ)
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
    (muonStep_real_lipschitz grad lr k G hgradLip hgrad1) hstep hideal n

/-- One real-matrix Muon step moves the parameter by at most `|lr|` (the NS output lies in the REAL unit ball
    by C66's `nsIter_ball_invariant_real`). -/
theorem muonStep_real_mag (grad : Matrix (Fin d) (Fin d) ℝ → Matrix (Fin d) (Fin d) ℝ)
    (lr : ℝ) (k : ℕ) (hgrad1 : ∀ x, ‖grad x‖ ≤ 1) (x : Matrix (Fin d) (Fin d) ℝ) :
    ‖muonStep grad lr (3 / 2) (-(1 / 2)) 0 k x‖ ≤ ‖x‖ + |lr| := by
  simp only [muonStep]
  calc ‖x - lr • nsIter (3 / 2) (-(1 / 2)) 0 k (grad x)‖
      ≤ ‖x‖ + ‖lr • nsIter (3 / 2) (-(1 / 2)) 0 k (grad x)‖ := norm_sub_le _ _
    _ = ‖x‖ + |lr| * ‖nsIter (3 / 2) (-(1 / 2)) 0 k (grad x)‖ := by
        rw [norm_smul, Real.norm_eq_abs]
    _ ≤ ‖x‖ + |lr| * 1 := by
        gcongr
        exact nsIter_ball_invariant_real k (grad x) (hgrad1 x)
    _ = ‖x‖ + |lr| := by ring

/-- **Linear parameter drift under real-matrix Muon**: `‖θ n‖ ≤ ‖θ 0‖ + n·|lr|` — the param-boundedness
    ingredient on the trainer shape (a uniform bound would need weight decay, cf. C32/C29). -/
theorem muon_real_traj_drift (grad : Matrix (Fin d) (Fin d) ℝ → Matrix (Fin d) (Fin d) ℝ)
    (lr : ℝ) (k : ℕ) (θ : Nat → Matrix (Fin d) (Fin d) ℝ)
    (hgrad1 : ∀ x, ‖grad x‖ ≤ 1)
    (htraj : ∀ n, θ (n + 1) = muonStep grad lr (3 / 2) (-(1 / 2)) 0 k (θ n)) :
    ∀ n, ‖θ n‖ ≤ ‖θ 0‖ + n * |lr| := by
  intro n
  induction n with
  | zero => simp
  | succ m ih =>
      rw [htraj m]
      calc ‖muonStep grad lr (3 / 2) (-(1 / 2)) 0 k (θ m)‖
          ≤ ‖θ m‖ + |lr| := muonStep_real_mag grad lr k hgrad1 (θ m)
        _ ≤ (‖θ 0‖ + m * |lr|) + |lr| := by linarith
        _ = ‖θ 0‖ + ((m : ℝ) + 1) * |lr| := by ring
        _ = ‖θ 0‖ + ((m + 1 : ℕ) : ℝ) * |lr| := by push_cast; ring

/-! ## Part B — the horizon-free certificate for the Muon-shaped weight-decay update -/

/-- **The Muon + weight-decay step constant**: C67's `wdStepC` with the raw gradient budget replaced by C61's
    NS-circuit budget at the `gradW` gradient bound — `(1+u64)²·(Blr·nsScalarFBound Ba Bb Bc ((1+u64)·(Bgs·Bx)))`.
    Step-invariant (fixed-batch gradient), so C67's contractive-run machinery applies verbatim. -/
noncomputable def muonWdStepC (Bx Bgs Blr Ba Bb Bc : ℝ) : ℝ :=
  (1 + u64) ^ 2 * (Blr * nsScalarFBound Ba Bb Bc ((1 + u64) * (Bgs * Bx)))

theorem muonWdStepC_nonneg (Bx Bgs Blr Ba Bb Bc : ℝ) (hx : 0 ≤ Bx) (hgs : 0 ≤ Bgs)
    (hlr : 0 ≤ Blr) (ha : 0 ≤ Ba) (hb : 0 ≤ Bb) (hc : 0 ≤ Bc) :
    0 ≤ muonWdStepC Bx Bgs Blr Ba Bb Bc := by
  have h1 : (0 : ℝ) ≤ 1 + u64 := by linarith [u64_pos]
  exact mul_nonneg (sq_nonneg _) (mul_nonneg hlr
    (nsScalarFBound_nonneg Ba Bb Bc _ ha hb hc (mul_nonneg h1 (mul_nonneg hgs hx))))

/-- **The Muon + weight-decay update** `w ← d·w − lr·nsScalarF(g)`: C67's decayed update at the
    NS-orthogonalized gradient (direct reuse — no new op tree). -/
def muonWdUpdateF (w g lr dc a b c : Float) : Float := wdUpdateF w (nsScalarF a b c g) lr dc

/-- **The Muon-wd update stays AFFINE-CONTRACTIVE in the weight bound** with C67's slope `wdStepRho Bd`
    (the weight path is untouched by the NS circuit); the constant threads C61's `nsScalarFBound`. -/
theorem muonWdUpdateF_mag_le (w g lr dc a b c : Float) (Bw Bg Blr Bd Ba Bb Bc : ℝ)
    (hw : |toReal w| ≤ Bw) (hg : |toReal g| ≤ Bg) (hlr : |toReal lr| ≤ Blr)
    (hd : |toReal dc| ≤ Bd) (ha : |toReal a| ≤ Ba) (hb : |toReal b| ≤ Bb)
    (hc : |toReal c| ≤ Bc) :
    |toReal (muonWdUpdateF w g lr dc a b c)|
      ≤ wdStepRho Bd * Bw + (1 + u64) ^ 2 * (Blr * nsScalarFBound Ba Bb Bc Bg) :=
  wdUpdateF_mag_le w (nsScalarF a b c g) lr dc Bw (nsScalarFBound Ba Bb Bc Bg) Blr Bd
    hw (nsScalarF_mag_le a b c g Ba Bb Bc Bg ha hb hc hg) hlr hd

/-- **One Muon-shaped weight-decay training step**: backward gradient `gradW` (C57), per-coordinate NS
    orthogonalization (C61's circuit), then the decayed vector update (C67's `wdUpdateVec`). -/
def muonWdTrainStep (x w : List Float) (gseed lr dc a b c : Float) : List Float :=
  wdUpdateVec w ((gradW gseed x).map (nsScalarF a b c)) lr dc

/-- One step never lengthens the weight row. -/
theorem muonWdTrainStep_length (x w : List Float) (gseed lr dc a b c : Float) :
    (muonWdTrainStep x w gseed lr dc a b c).length ≤ w.length := by
  simp only [muonWdTrainStep, wdUpdateVec, List.length_zipWith]
  exact min_le_left _ _

/-- Every NS-orthogonalized gradient entry is bounded by C61's circuit budget at the `gradW` bound. -/
theorem mapNS_gradW_mag (x : List Float) (gseed a b c : Float) (Bx Bgs Ba Bb Bc : ℝ)
    (hx : ∀ xi ∈ x, |toReal xi| ≤ Bx) (hgs : |toReal gseed| ≤ Bgs)
    (ha : |toReal a| ≤ Ba) (hb : |toReal b| ≤ Bb) (hc : |toReal c| ≤ Bc) :
    ∀ u ∈ (gradW gseed x).map (nsScalarF a b c),
      |toReal u| ≤ nsScalarFBound Ba Bb Bc ((1 + u64) * (Bgs * Bx)) := by
  intro u hu
  rw [List.mem_map] at hu
  obtain ⟨gw, hgw, rfl⟩ := hu
  exact nsScalarF_mag_le a b c gw Ba Bb Bc _ ha hb hc (gradW_mag gseed x Bgs Bx hgs hx gw hgw)

/-- **One Muon-wd step maps a `B`-budget to `wdStepRho Bd·B + muonWdStepC …`** — the contractive affine step
    map with the Muon constant (C67's `wdUpdateVec_mag` at the NS-mapped gradient row). -/
theorem muonWdTrainStep_mag (x w : List Float) (gseed lr dc a b c : Float)
    (Bx B Bgs Blr Bd Ba Bb Bc : ℝ)
    (hx : ∀ xi ∈ x, |toReal xi| ≤ Bx) (hw : ∀ wi ∈ w, |toReal wi| ≤ B)
    (hgs : |toReal gseed| ≤ Bgs) (hlr : |toReal lr| ≤ Blr) (hd : |toReal dc| ≤ Bd)
    (ha : |toReal a| ≤ Ba) (hb : |toReal b| ≤ Bb) (hc : |toReal c| ≤ Bc) :
    ∀ u ∈ muonWdTrainStep x w gseed lr dc a b c,
      |toReal u| ≤ wdStepRho Bd * B + muonWdStepC Bx Bgs Blr Ba Bb Bc := by
  intro u hu
  have h := wdUpdateVec_mag lr dc B (nsScalarFBound Ba Bb Bc ((1 + u64) * (Bgs * Bx))) Blr Bd
    hlr hd w ((gradW gseed x).map (nsScalarF a b c)) hw
    (mapNS_gradW_mag x gseed a b c Bx Bgs Ba Bb Bc hx hgs ha hb hc) u hu
  unfold muonWdStepC
  exact h

/-- **The n-step Muon-shaped weight-decay training run** (fixed batch per step, as C64/C67). -/
def muonWdTrainRun (x : List Float) (gseed lr dc a b c : Float) : Nat → List Float → List Float
  | 0, w => w
  | n + 1, w => muonWdTrainRun x gseed lr dc a b c n (muonWdTrainStep x w gseed lr dc a b c)

/-- The run never lengthens the weight row. -/
theorem muonWdTrainRun_length (x : List Float) (gseed lr dc a b c : Float) :
    ∀ (n : Nat) (w : List Float), (muonWdTrainRun x gseed lr dc a b c n w).length ≤ w.length
  | 0, _ => le_refl _
  | n + 1, w => (muonWdTrainRun_length x gseed lr dc a b c n _).trans
      (muonWdTrainStep_length x w gseed lr dc a b c)

/-- **The run invariant**: initial weights `≤ B` ⟹ every weight after `n` steps is bounded by C67's budget
    iterate `wdRunBound (wdStepRho Bd) (muonWdStepC …) n B` — the SAME contractive machinery, the Muon constant. -/
theorem muonWdTrainRun_mag (x : List Float) (gseed lr dc a b c : Float)
    (Bx Bgs Blr Bd Ba Bb Bc : ℝ)
    (hx : ∀ xi ∈ x, |toReal xi| ≤ Bx) (hgs : |toReal gseed| ≤ Bgs) (hlr : |toReal lr| ≤ Blr)
    (hd : |toReal dc| ≤ Bd) (ha : |toReal a| ≤ Ba) (hb : |toReal b| ≤ Bb)
    (hc : |toReal c| ≤ Bc) :
    ∀ (n : Nat) (w : List Float) (B : ℝ), (∀ wi ∈ w, |toReal wi| ≤ B) →
      ∀ u ∈ muonWdTrainRun x gseed lr dc a b c n w,
        |toReal u| ≤ wdRunBound (wdStepRho Bd) (muonWdStepC Bx Bgs Blr Ba Bb Bc) n B
  | 0, _, _, hw => hw
  | n + 1, w, B, hw =>
      muonWdTrainRun_mag x gseed lr dc a b c Bx Bgs Blr Bd Ba Bb Bc hx hgs hlr hd ha hb hc n
        (muonWdTrainStep x w gseed lr dc a b c)
        (wdStepRho Bd * B + muonWdStepC Bx Bgs Blr Ba Bb Bc)
        (muonWdTrainStep_mag x w gseed lr dc a b c Bx B Bgs Blr Bd Ba Bb Bc hx hw hgs hlr hd
          ha hb hc)

/-- **THE MUON-SHAPED WEIGHT-DECAY WHOLE-RUN IS OVERFLOW-FREE — HORIZON-FREE.** Under C67's mild contraction
    `wdStepRho Bd < 1`, with the SINGLE `n`-independent budget condition
    `wdUniformBound (wdStepRho Bd) (muonWdStepC …) B0 ≤ overflowBound`, EVERY weight at EVERY step of an
    ARBITRARILY LONG Muon-shaped (NS-orthogonalized, weight-decayed) run is `isFinite` — C67's horizon-free
    certificate with the actual Muon update, its `wdRunBound_uniform` machinery reused verbatim. -/
theorem muonWdTrainRun_all_finite_uniform (x w : List Float) (gseed lr dc a b c : Float)
    (Bx B0 Bgs Blr Bd Ba Bb Bc : ℝ) (hBx0 : 0 ≤ Bx) (hB00 : 0 ≤ B0)
    (hx : ∀ xi ∈ x, |toReal xi| ≤ Bx) (hw : ∀ wi ∈ w, |toReal wi| ≤ B0)
    (hgs : |toReal gseed| ≤ Bgs) (hlr : |toReal lr| ≤ Blr) (hd : |toReal dc| ≤ Bd)
    (ha : |toReal a| ≤ Ba) (hb : |toReal b| ≤ Bb) (hc : |toReal c| ≤ Bc)
    (hcontract : wdStepRho Bd < 1)
    (hbound : wdUniformBound (wdStepRho Bd) (muonWdStepC Bx Bgs Blr Ba Bb Bc) B0
        ≤ overflowBound) :
    ∀ m, ∀ u ∈ muonWdTrainRun x gseed lr dc a b c m w, u.isFinite = true := by
  intro m u hu
  have hρ0 : 0 ≤ wdStepRho Bd := wdStepRho_nonneg Bd ((abs_nonneg _).trans hd)
  have hC : 0 ≤ muonWdStepC Bx Bgs Blr Ba Bb Bc :=
    muonWdStepC_nonneg Bx Bgs Blr Ba Bb Bc hBx0 ((abs_nonneg _).trans hgs)
      ((abs_nonneg _).trans hlr) ((abs_nonneg _).trans ha) ((abs_nonneg _).trans hb)
      ((abs_nonneg _).trans hc)
  exact isFinite_of_bounded _
    (((muonWdTrainRun_mag x gseed lr dc a b c Bx Bgs Blr Bd Ba Bb Bc hx hgs hlr hd ha hb hc
        m w B0 hw u hu).trans
      (wdRunBound_uniform (wdStepRho Bd) (muonWdStepC Bx Bgs Blr Ba Bb Bc) hρ0 hcontract hC B0
        hB00 m)).trans hbound)

/-- **The forward pass is overflow-free at every step of the Muon-shaped run — horizon-free**: one
    `n`-independent `dotBound` budget at the max of the input bound and the uniform weight budget. -/
theorem muonWdTrainRun_forward_all_finite_uniform (x w : List Float) (gseed lr dc a b c : Float)
    (Bx B0 Bgs Blr Bd Ba Bb Bc : ℝ) (hBx0 : 0 ≤ Bx) (hB00 : 0 ≤ B0)
    (hx : ∀ xi ∈ x, |toReal xi| ≤ Bx) (hw : ∀ wi ∈ w, |toReal wi| ≤ B0)
    (hgs : |toReal gseed| ≤ Bgs) (hlr : |toReal lr| ≤ Blr) (hd : |toReal dc| ≤ Bd)
    (ha : |toReal a| ≤ Ba) (hb : |toReal b| ≤ Bb) (hc : |toReal c| ≤ Bc)
    (hcontract : wdStepRho Bd < 1)
    (hfwd : dotBound (min x.length w.length)
        (max Bx (wdUniformBound (wdStepRho Bd) (muonWdStepC Bx Bgs Blr Ba Bb Bc) B0))
      ≤ overflowBound) :
    ∀ m, (Puffer.RL.FiniteBound.dotF x
        (muonWdTrainRun x gseed lr dc a b c m w)).isFinite = true := by
  intro m
  have hρ0 : 0 ≤ wdStepRho Bd := wdStepRho_nonneg Bd ((abs_nonneg _).trans hd)
  have hC : 0 ≤ muonWdStepC Bx Bgs Blr Ba Bb Bc :=
    muonWdStepC_nonneg Bx Bgs Blr Ba Bb Bc hBx0 ((abs_nonneg _).trans hgs)
      ((abs_nonneg _).trans hlr) ((abs_nonneg _).trans ha) ((abs_nonneg _).trans hb)
      ((abs_nonneg _).trans hc)
  have hBmax0 : 0 ≤ max Bx (wdUniformBound (wdStepRho Bd) (muonWdStepC Bx Bgs Blr Ba Bb Bc) B0) :=
    hBx0.trans (le_max_left _ _)
  have hx' : ∀ xi ∈ x, |toReal xi|
      ≤ max Bx (wdUniformBound (wdStepRho Bd) (muonWdStepC Bx Bgs Blr Ba Bb Bc) B0) :=
    fun xi hxi => (hx xi hxi).trans (le_max_left _ _)
  have hw' : ∀ wi ∈ muonWdTrainRun x gseed lr dc a b c m w, |toReal wi|
      ≤ max Bx (wdUniformBound (wdStepRho Bd) (muonWdStepC Bx Bgs Blr Ba Bb Bc) B0) := fun wi hwi =>
    ((muonWdTrainRun_mag x gseed lr dc a b c Bx Bgs Blr Bd Ba Bb Bc hx hgs hlr hd ha hb hc
        m w B0 hw wi hwi).trans
      (wdRunBound_uniform (wdStepRho Bd) (muonWdStepC Bx Bgs Blr Ba Bb Bc) hρ0 hcontract hC B0
        hB00 m)).trans (le_max_right _ _)
  have hlen : min x.length (muonWdTrainRun x gseed lr dc a b c m w).length
      ≤ min x.length w.length :=
    min_le_min (le_refl _) (muonWdTrainRun_length x gseed lr dc a b c m w)
  exact dotF_isFinite _ hBmax0 x (muonWdTrainRun x gseed lr dc a b c m w) hx' hw'
    ((dotBound_mono _ hlen).trans hfwd)

end Puffer.RL.MuonRealCompose
