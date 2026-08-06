/-
# C89: Horizon-free Muon via the invariant ball — weight decay closes C62's disclosure

C62 (`MuonBallWholeRun`) proved the unit-ball Muon step `(1 + |lr|·3^k·G)`-Lipschitz with C59's
PROVEN NS constant and the linear drift `‖θ n‖ ≤ ‖θ 0‖ + n·|lr|`, but disclosed twice that a
UNIFORM (horizon-free) bound "would need weight decay (cf. C32/C29)". C88 (`MuonRunEval`)
re-disclosed the same at the Float-budget level. This module supplies the weight-decay side:

* **The recurrence is the REAL one.** PufferLib's `muon.cu:58–65` fused update is
  `wb = wb * (1 - lr*wd) - lr * scale * update` — decoupled weight decay IS in the actual
  optimizer. `muonWdStep grad lr wd a b c k x := (1 − wd) • x − lr • nsIter a b c k (grad x)`
  models it with `wd` standing for the literal decay amount (muon.cu's `lr*wd`; its per-matrix
  `scale` folds into the learning rate's role). C53's bare `muonStep` is the `wd = 0` case
  (`muonWdStep_zero_eq`), so the linear-drift twin transfers back verbatim
  (`muonWd_traj_drift_zero` ∘ C62's `muon_traj_drift`).
* **`muonWdStep_ball_mag`** — one step: `‖θ'‖ ≤ (1 − wd)·‖θ‖ + |lr|`. The NS output lies in
  C59's invariant unit ball (`nsIter_ball_invariant` on the spectrally-normalized gradient,
  `∀ x, ‖grad x‖ ≤ 1` — the honest model of Muon's normalization, exactly as C62), so the
  increment is uniformly `|lr|` REGARDLESS of gradient magnitude or iteration count `k`, and the
  weight term carries the contraction `(1 − wd)`.
* **`muonWd_traj_uniform`** (capstone, ℝ-side) — THE HORIZON-FREE BOUND: for `0 < wd ≤ 1`,
  `‖θ n‖ ≤ ‖θ 0‖ + |lr|/wd` for ALL `n` — no horizon anywhere in the statement. C2/C32's
  `affine_recur_uniform` on the recurrence above (`ρ = 1 − wd`, `C = |lr|`, attractor
  `|lr|/wd`). Contrast C62's `muon_traj_drift` (bare recurrence: linear growth `n·|lr|`) and
  C88 (scalar Float circuit: budget grows with `n`).
* **`muonWdStep_ball_lipschitz` / `muonWd_whole_run_uniform`** — the step is
  `((1 − wd) + |lr|·3^k·G)`-Lipschitz on ball-valued gradients (C62's constant with the identity
  part shrunk from `1` to `1 − wd`), so when the decay dominates the NS-amplified gradient
  Lipschitz (`|lr|·3^k·G < wd`, the Muon analogue of C32's `wd > |lr|·G`), the runnable-vs-ideal
  trajectory error is HORIZON-FREE: `‖θ n − θ' n‖ ≤ ‖θ 0 − θ' 0‖ + B/(1 − L)` for all `n` —
  the Muon counterpart of C32's `wd_whole_run_uniform_interval`, with the NS factor `3^k` a
  THEOREM (C59), not a hypothesis.
* **The runnable radius** — `muonBallRadF B0F lrF wdLoF := slackF·(B0F + slackF·(lrF/wdLoF))`
  Float-evaluates the uniform radius, with `muonBallRadF_dominates`:
  `‖θ 0‖ + |lr|/wd ≤ toReal (muonBallRadF …)` from Float witnesses (`B0F`, `lrF` upper;
  `wdLoF` a LOWER witness for the decay — division is antitone in the denominator, the same
  direction-reversal discipline as C83's floor). The new `unit_div_dominates` extends C81's unit
  family to division (one `slackF_key` per node, C78's key unchanged). Capstone
  **`muonWd_traj_uniform_runnable`**: the strict positivity `0 < toReal wdLoF` comes from ONE
  runtime Bool `checkLe minWdF wdLoF` against the pinned literal `minWdF = 1e-9`
  (`minWdF_pos` via `toReal_ofScientific_close` — C81's `contractThresholdF` one-time-gap
  pattern), so the horizon-free radius is a computed Float.

**Scope (honestly disclosed).** The ball results hold for the CLASSICAL NS coefficients
`(3/2, −1/2, 0)` — the only ones with proven unit-ball invariance (C59) — on
spectrally-normalized (ball-valued) gradients. The SHIPPED optimizer uses the tuned quintic
schedule (`muon.cu`'s `ns_coeffs`, e.g. `(4.0848, −6.8946, 2.9270)`; `Puffer/Float/Muon.lean`),
which is deliberately NOT unit-ball-invariant and is not covered here; only the weight-decay
FUSION structure `(1−wd)•· − lr•(·)` is faithful to the shipped kernel for any coefficients
(`muonWdStep` is coefficient-agnostic). This is C62's scope exactly. The ℝ couplings
`‖θ 0‖ ≤ toReal B0F`, `|lr| ≤ toReal lrF`,
`toReal wdLoF ≤ wd ≤ 1`, the ball-valued-gradient hypothesis, and (for the error interval) `G`
and the per-step Float error `B` remain the caller's, exactly as in C62 — the trajectory lives
in an abstract C*-algebra, so there is no Float data to check them against (contrast C81/C88,
where weights ARE Float lists). The contraction margin `|lr|·3^k·G < wd` of
`muonWd_whole_run_uniform` is an ℝ hypothesis here; its Bool mirror needs a `3^k` Float
evaluator (a double-`slackF` per doubling — `3 ≤ toReal 3.0` is NOT derivable from the
`ofScientific` pin, so each step spends two keys) plus a `GF` coupling that is caller-side
anyway — mechanical, deferred, shape disclosed. The setting is C59/C62's C*-order typeclass;
`ℂ` instantiates everything (demonstrated), real matrices via C66's complexification as before.
Demo constants (`lr = 0.1`, `wd ≥ 0.009`) are illustrative, not from a shipped config. NO new
axiom, no `sorryAx`: everything routes through C59's ball, C2's `affine_recur_uniform`, C78's
`slackF_key`, C70's `checkLe`, and the pre-existing `(1+δ)`/`ofScientific` trusted base.
-/
import Puffer.RL.MuonBallWholeRun
import Puffer.RL.MuonTrainBound
import Puffer.RL.RunConstEval

open Puffer.FloatR (toReal u64 u64_pos u64_lt_one mul_model add_model div_model
  toReal_ofScientific_close)
open Puffer.RL.NewtonSchulzIterate (nsIter)
open Puffer.RL.NewtonSchulzBall (nsIter_ball_invariant nsIter_lipschitz_ball)
open Puffer.RL.MuonStepLipschitz (muonStep)
open Puffer.RL.MuonBallWholeRun (muon_traj_drift)
open Puffer.RL.MuonTrainBound (affine_recur_uniform)
open Puffer.RL.BudgetEval (slackF slackF_key slackF_nonneg)
open Puffer.RL.MarginCheck (checkLe checkLe_sound)

namespace Puffer.RL.MuonBallRunFinite

/-! ### The weight-decay-augmented Muon step (muon.cu:58–65's fused update) -/

section Def

variable {R : Type*} [NormedRing R] [NormedAlgebra ℝ R] [StarRing R] [NormedStarGroup R]

/-- **The weight-decay Muon update** `θ ↦ (1 − wd)·θ − lr·NS(∇θ)` — the model of muon.cu's fused
    `wb = wb * (1 - lr*wd) - lr * scale * update` (`wd` here is the literal decay amount
    `lr*wd` there; `scale` folds into the learning-rate slot). C53's bare `muonStep` is the
    `wd = 0` case. -/
def muonWdStep (grad : R → R) (lr wd : ℝ) (a b c : ℝ) (k : ℕ) (x : R) : R :=
  (1 - wd) • x - lr • nsIter a b c k (grad x)

omit [NormedStarGroup R] in
/-- At `wd = 0` the weight-decay step IS C53's bare `muonStep` — the connector that transfers
    C62's linear-drift twin back to this module's recurrence. -/
theorem muonWdStep_zero_eq (grad : R → R) (lr : ℝ) (a b c : ℝ) (k : ℕ) (x : R) :
    muonWdStep grad lr 0 a b c k x = muonStep grad lr a b c k x := by
  simp [muonWdStep, muonStep]

end Def

/-! ### The ball theorems: uniform step increment, horizon-free norm, horizon-free interval -/

section Ball

variable {A : Type*} [CStarAlgebra A] [PartialOrder A] [StarOrderedRing A]

/-- **One weight-decay Muon step: `‖θ'‖ ≤ (1 − wd)·‖θ‖ + |lr|`.** The NS output lies in C59's
    invariant unit ball (spectrally-normalized gradient), so the increment is uniformly `|lr|`
    — independent of the gradient magnitude and of `k` — while the weight term carries the
    contraction factor `(1 − wd)`. The affine-contraction recurrence in norm. -/
theorem muonWdStep_ball_mag (grad : A → A) (lr wd : ℝ) (k : ℕ)
    (hgrad1 : ∀ x, ‖grad x‖ ≤ 1) (hwd1 : wd ≤ 1) (x : A) :
    ‖muonWdStep grad lr wd (3 / 2) (-(1 / 2)) 0 k x‖ ≤ (1 - wd) * ‖x‖ + |lr| := by
  have h1wd : (0 : ℝ) ≤ 1 - wd := by linarith
  simp only [muonWdStep]
  calc ‖(1 - wd) • x - lr • nsIter (3 / 2) (-(1 / 2)) 0 k (grad x)‖
      ≤ ‖(1 - wd) • x‖ + ‖lr • nsIter (3 / 2) (-(1 / 2)) 0 k (grad x)‖ := norm_sub_le _ _
    _ = |1 - wd| * ‖x‖ + |lr| * ‖nsIter (3 / 2) (-(1 / 2)) 0 k (grad x)‖ := by
        rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs]
    _ ≤ (1 - wd) * ‖x‖ + |lr| * 1 := by
        rw [abs_of_nonneg h1wd]
        gcongr
        exact nsIter_ball_invariant k (grad x) (hgrad1 x)
    _ = (1 - wd) * ‖x‖ + |lr| := by ring

/-- **THE HORIZON-FREE MUON BOUND (C62's disclosure closed).** Under weight decay `0 < wd ≤ 1`
    and spectrally-normalized gradients, EVERY point of a weight-decay Muon trajectory satisfies
    `‖θ n‖ ≤ ‖θ 0‖ + |lr|/wd` — no horizon in the statement. `affine_recur_uniform` on the
    per-step recurrence `‖θ (n+1)‖ ≤ (1 − wd)·‖θ n‖ + |lr|` (`ρ = 1 − wd < 1`, attractor
    `|lr|/wd`). Contrast: the bare recurrence only gets linear drift (C62 `muon_traj_drift`,
    transferred below), and the scalar Float circuit's budget grows with `n` (C88). -/
theorem muonWd_traj_uniform (grad : A → A) (lr wd : ℝ) (k : ℕ) (θ : Nat → A)
    (hgrad1 : ∀ x, ‖grad x‖ ≤ 1) (hwd0 : 0 < wd) (hwd1 : wd ≤ 1)
    (htraj : ∀ n, θ (n + 1) = muonWdStep grad lr wd (3 / 2) (-(1 / 2)) 0 k (θ n)) (n : ℕ) :
    ‖θ n‖ ≤ ‖θ 0‖ + |lr| / wd := by
  have h := affine_recur_uniform (fun m => ‖θ m‖) (1 - wd) |lr|
    (by linarith) (by linarith) (abs_nonneg _) (norm_nonneg _)
    (fun m => by
      show ‖θ (m + 1)‖ ≤ (1 - wd) * ‖θ m‖ + |lr|
      rw [htraj m]
      exact muonWdStep_ball_mag grad lr wd k hgrad1 hwd1 (θ m)) n
  have hrw : (1 : ℝ) - (1 - wd) = wd := by ring
  rwa [hrw] at h

/-- **The wd-step Lipschitz constant `L = (1 − wd) + |lr|·3^k·G`** — C62's
    `muonStep_ball_lipschitz` with the identity part shrunk from `1` to `1 − wd` by the decay.
    The NS factor `3^k` is C59's THEOREM (`nsIter_lipschitz_ball`), not a hypothesis. `L < 1`
    exactly when the decay dominates the NS-amplified gradient Lipschitz (`|lr|·3^k·G < wd`) —
    the Muon analogue of C32's contraction condition `wd > |lr|·G`. -/
theorem muonWdStep_ball_lipschitz (grad : A → A) (lr wd : ℝ) (k : ℕ) (G : ℝ)
    (hgradLip : ∀ x y, ‖grad x - grad y‖ ≤ G * ‖x - y‖)
    (hgrad1 : ∀ x, ‖grad x‖ ≤ 1) (hwd1 : wd ≤ 1) (x y : A) :
    ‖muonWdStep grad lr wd (3 / 2) (-(1 / 2)) 0 k x
        - muonWdStep grad lr wd (3 / 2) (-(1 / 2)) 0 k y‖
      ≤ ((1 - wd) + |lr| * 3 ^ k * G) * ‖x - y‖ := by
  have h1wd : (0 : ℝ) ≤ 1 - wd := by linarith
  have hrw : muonWdStep grad lr wd (3 / 2) (-(1 / 2)) 0 k x
        - muonWdStep grad lr wd (3 / 2) (-(1 / 2)) 0 k y
      = (1 - wd) • (x - y) - lr • (nsIter (3 / 2) (-(1 / 2)) 0 k (grad x)
          - nsIter (3 / 2) (-(1 / 2)) 0 k (grad y)) := by
    simp only [muonWdStep, smul_sub]; abel
  rw [hrw]
  calc ‖(1 - wd) • (x - y) - lr • (nsIter (3 / 2) (-(1 / 2)) 0 k (grad x)
          - nsIter (3 / 2) (-(1 / 2)) 0 k (grad y))‖
      ≤ ‖(1 - wd) • (x - y)‖ + ‖lr • (nsIter (3 / 2) (-(1 / 2)) 0 k (grad x)
          - nsIter (3 / 2) (-(1 / 2)) 0 k (grad y))‖ := norm_sub_le _ _
    _ = |1 - wd| * ‖x - y‖ + |lr| * ‖nsIter (3 / 2) (-(1 / 2)) 0 k (grad x)
          - nsIter (3 / 2) (-(1 / 2)) 0 k (grad y)‖ := by
        rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs]
    _ ≤ (1 - wd) * ‖x - y‖ + |lr| * (3 ^ k * ‖grad x - grad y‖) := by
        rw [abs_of_nonneg h1wd]
        gcongr
        exact nsIter_lipschitz_ball k (grad x) (grad y) (hgrad1 x) (hgrad1 y)
    _ ≤ (1 - wd) * ‖x - y‖ + |lr| * (3 ^ k * (G * ‖x - y‖)) := by
        gcongr
        exact hgradLip x y
    _ = ((1 - wd) + |lr| * 3 ^ k * G) * ‖x - y‖ := by ring

/-- **THE HORIZON-FREE MUON WHOLE-RUN ERROR INTERVAL — the Muon counterpart of C32.** When the
    decay dominates (`L = (1 − wd) + |lr|·3^k·G < 1`), the runnable trajectory `θ` (within `B`
    per step of the weight-decay Muon step) NEVER drifts more than the fixed neighborhood
    `‖θ 0 − θ' 0‖ + B/(1 − L)` from the ideal trajectory `θ'`, however long training runs —
    with the NS factor `3^k` proven (C59), not assumed. `affine_recur_uniform` on the error
    recurrence `d(n+1) ≤ L·d(n) + B`. -/
theorem muonWd_whole_run_uniform (grad : A → A) (lr wd : ℝ) (k : ℕ) (G B : ℝ)
    (θ θ' : Nat → A)
    (hgradLip : ∀ x y, ‖grad x - grad y‖ ≤ G * ‖x - y‖)
    (hgrad1 : ∀ x, ‖grad x‖ ≤ 1) (hG : 0 ≤ G) (hwd1 : wd ≤ 1) (hB : 0 ≤ B)
    (hL : (1 - wd) + |lr| * 3 ^ k * G < 1)
    (hstep : ∀ n, ‖θ (n + 1) - muonWdStep grad lr wd (3 / 2) (-(1 / 2)) 0 k (θ n)‖ ≤ B)
    (hideal : ∀ n, θ' (n + 1) = muonWdStep grad lr wd (3 / 2) (-(1 / 2)) 0 k (θ' n)) (n : ℕ) :
    ‖θ n - θ' n‖ ≤ ‖θ 0 - θ' 0‖ + B / (1 - ((1 - wd) + |lr| * 3 ^ k * G)) := by
  have h1wd : (0 : ℝ) ≤ 1 - wd := by linarith
  have hL0 : (0 : ℝ) ≤ (1 - wd) + |lr| * 3 ^ k * G :=
    add_nonneg h1wd (mul_nonneg (mul_nonneg (abs_nonneg _) (by positivity)) hG)
  have hrec : ∀ m, ‖θ (m + 1) - θ' (m + 1)‖
      ≤ ((1 - wd) + |lr| * 3 ^ k * G) * ‖θ m - θ' m‖ + B := by
    intro m
    have hsplit : θ (m + 1) - θ' (m + 1)
        = (θ (m + 1) - muonWdStep grad lr wd (3 / 2) (-(1 / 2)) 0 k (θ m))
          + (muonWdStep grad lr wd (3 / 2) (-(1 / 2)) 0 k (θ m) - θ' (m + 1)) := by abel
    rw [hsplit, hideal m]
    calc ‖(θ (m + 1) - muonWdStep grad lr wd (3 / 2) (-(1 / 2)) 0 k (θ m))
            + (muonWdStep grad lr wd (3 / 2) (-(1 / 2)) 0 k (θ m)
              - muonWdStep grad lr wd (3 / 2) (-(1 / 2)) 0 k (θ' m))‖
        ≤ ‖θ (m + 1) - muonWdStep grad lr wd (3 / 2) (-(1 / 2)) 0 k (θ m)‖
            + ‖muonWdStep grad lr wd (3 / 2) (-(1 / 2)) 0 k (θ m)
              - muonWdStep grad lr wd (3 / 2) (-(1 / 2)) 0 k (θ' m)‖ := norm_add_le _ _
      _ ≤ B + ((1 - wd) + |lr| * 3 ^ k * G) * ‖θ m - θ' m‖ :=
          add_le_add (hstep m)
            (muonWdStep_ball_lipschitz grad lr wd k G hgradLip hgrad1 hwd1 (θ m) (θ' m))
      _ = ((1 - wd) + |lr| * 3 ^ k * G) * ‖θ m - θ' m‖ + B := by ring
  exact affine_recur_uniform (fun m => ‖θ m - θ' m‖) ((1 - wd) + |lr| * 3 ^ k * G) B
    hL0 hL hB (norm_nonneg _) hrec n

/-- **The bare-recurrence linear-growth twin, transferred.** A `wd = 0` trajectory of
    `muonWdStep` is a `muonStep` trajectory (`muonWdStep_zero_eq`), so C62's `muon_traj_drift`
    applies verbatim: `‖θ n‖ ≤ ‖θ 0‖ + n·|lr|` — the honest contrast with the horizon-free
    `‖θ 0‖ + |lr|/wd` above. -/
theorem muonWd_traj_drift_zero (grad : A → A) (lr : ℝ) (k : ℕ) (θ : Nat → A)
    (hgrad1 : ∀ x, ‖grad x‖ ≤ 1)
    (htraj : ∀ n, θ (n + 1) = muonWdStep grad lr 0 (3 / 2) (-(1 / 2)) 0 k (θ n)) :
    ∀ n, ‖θ n‖ ≤ ‖θ 0‖ + n * |lr| :=
  muon_traj_drift grad lr k θ hgrad1 fun n => by rw [htraj n, muonWdStep_zero_eq]

end Ball

/-! ### The runnable radius: Float evaluator + one-Bool positivity gate -/

private theorem one_sub_u64_nonneg : (0 : ℝ) ≤ 1 - u64 := by linarith [u64_lt_one]

/-- The positivity floor for the decay witness: `1e-9`, pinned strictly positive below. Any real
    weight-decay amount (muon.cu's `lr*wd`, typically `≳ 10⁻⁶`) clears it comfortably. -/
def minWdF : Float := 1e-9

/-- `0 < toReal minWdF` — the one-time pin via `toReal_ofScientific_close`
    (`1e-9 = ofScientific 1 true 9`; `toReal minWdF ≥ 10⁻⁹·(1 − u64) > 0`). C81's
    `contractThresholdF` one-time-gap pattern: ONE runtime `checkLe minWdF wdLoF` then yields
    the STRICT `0 < toReal wdLoF`. -/
theorem minWdF_pos : 0 < toReal minWdF := by
  have h := toReal_ofScientific_close 1 true 9
  have habs : |(1e-9 : ℝ)| = (1e-9 : ℝ) := abs_of_pos (by norm_num)
  rw [habs] at h
  have h1 := (abs_le.mp h).1
  have hnum : (0 : ℝ) < 1e-9 := by norm_num
  unfold minWdF
  nlinarith [u64_lt_one]

/-- **Division unit — C81's unit family extended to `/`.** `x / y ≤ toReal (slackF * (xF / yF))`
    when the numerator is dominated ABOVE (`x ≤ toReal xF`) and the denominator BELOW
    (`0 < toReal yF ≤ y`): division is antitone in the denominator, so the Float mirror divides
    by the LOWER witness (the C83 direction-reversal discipline). One `slackF_key` absorbs the
    division's rounding and the slack multiply. -/
theorem unit_div_dominates (xF yF : Float) (x y : ℝ)
    (hx0 : 0 ≤ x) (hxf : x ≤ toReal xF)
    (hyF0 : 0 < toReal yF) (hyf : toReal yF ≤ y) :
    x / y ≤ toReal (slackF * (xF / yF)) := by
  have hy0 : 0 < y := lt_of_lt_of_le hyF0 hyf
  have hq0 : 0 ≤ toReal xF / toReal yF := div_nonneg (hx0.trans hxf) hyF0.le
  have h1 : x / y ≤ toReal xF / toReal yF := by
    have ha : 1 / y ≤ 1 / toReal yF := one_div_le_one_div_of_le hyF0 hyf
    calc x / y = x * (1 / y) := div_eq_mul_one_div x y
      _ ≤ toReal xF * (1 / toReal yF) :=
          mul_le_mul hxf ha (one_div_nonneg.mpr hy0.le) (hx0.trans hxf)
      _ = toReal xF / toReal yF := (div_eq_mul_one_div _ _).symm
  obtain ⟨δ₁, hδ₁, e₁⟩ := div_model xF yF
  obtain ⟨δ₂, hδ₂, e₂⟩ := mul_model slackF (xF / yF)
  have hd₁ : 1 - u64 ≤ 1 + δ₁ := by have := (abs_le.mp hδ₁).1; linarith
  have hd₂ : 1 - u64 ≤ 1 + δ₂ := by have := (abs_le.mp hδ₂).1; linarith
  have hu1 := one_sub_u64_nonneg
  have hs := slackF_nonneg
  have hq1 : 0 ≤ toReal (xF / yF) := by
    rw [e₁]; exact mul_nonneg hq0 (by linarith)
  have h2 : (toReal xF / toReal yF) * ((1 - u64) * (1 - u64))
      ≤ toReal (xF / yF) * (1 - u64) := by
    rw [e₁]
    have hmid : (toReal xF / toReal yF) * (1 - u64)
        ≤ (toReal xF / toReal yF) * (1 + δ₁) :=
      mul_le_mul_of_nonneg_left hd₁ hq0
    calc (toReal xF / toReal yF) * ((1 - u64) * (1 - u64))
        = ((toReal xF / toReal yF) * (1 - u64)) * (1 - u64) := by ring
      _ ≤ ((toReal xF / toReal yF) * (1 + δ₁)) * (1 - u64) :=
          mul_le_mul_of_nonneg_right hmid hu1
  rw [e₂]
  have hkey := mul_le_mul_of_nonneg_right slackF_key hq0
  calc x / y
      ≤ toReal xF / toReal yF := h1
    _ ≤ (toReal slackF * ((1 - u64) * (1 - u64))) * (toReal xF / toReal yF) := by
        nlinarith [u64_pos, hq0]
    _ = toReal slackF * ((toReal xF / toReal yF) * ((1 - u64) * (1 - u64))) := by ring
    _ ≤ toReal slackF * (toReal (xF / yF) * (1 - u64)) := mul_le_mul_of_nonneg_left h2 hs
    _ ≤ toReal slackF * (toReal (xF / yF) * (1 + δ₂)) :=
        mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hd₂ hq1) hs
    _ = toReal slackF * toReal (xF / yF) * (1 + δ₂) := by ring

/-- **The runnable horizon-free radius**: `slackF·(B0F + slackF·(lrF/wdLoF))` — the Float
    evaluation of `‖θ 0‖ + |lr|/wd`'s upper bound, one `slackF` per Float node. -/
def muonBallRadF (B0F lrF wdLoF : Float) : Float :=
  slackF * (B0F + slackF * (lrF / wdLoF))

/-- The evaluator dominates the ℝ radius: `B0 + |lr|/wd ≤ toReal (muonBallRadF B0F lrF wdLoF)`
    from the Float witnesses (`B0F`, `lrF` above; `wdLoF` below, strictly positive). -/
theorem muonBallRadF_dominates (B0F lrF wdLoF : Float) (B0 lr wd : ℝ)
    (hB00 : 0 ≤ B0) (hB0f : B0 ≤ toReal B0F)
    (hlrf : |lr| ≤ toReal lrF)
    (hwdF0 : 0 < toReal wdLoF) (hwdf : toReal wdLoF ≤ wd) :
    B0 + |lr| / wd ≤ toReal (muonBallRadF B0F lrF wdLoF) := by
  have hwd0 : 0 < wd := lt_of_lt_of_le hwdF0 hwdf
  have hq0 : 0 ≤ |lr| / wd := div_nonneg (abs_nonneg _) hwd0.le
  have h1 : |lr| / wd ≤ toReal (slackF * (lrF / wdLoF)) :=
    unit_div_dominates lrF wdLoF |lr| wd (abs_nonneg _) hlrf hwdF0 hwdf
  have h2 := Puffer.RL.RunConstEval.unit_add_dominates B0F (slackF * (lrF / wdLoF))
    B0 (|lr| / wd) hB00 hB0f hq0 h1
  have hsum0 : 0 ≤ B0 + |lr| / wd := add_nonneg hB00 hq0
  unfold muonBallRadF
  nlinarith [u64_pos]

/-- **THE RUNNABLE HORIZON-FREE MUON CERTIFICATE.** Float witnesses for the initial norm, the
    learning rate, and the decay (lower); ONE runtime Bool (`checkLe minWdF wdLoF`) supplying
    strict positivity; conclusion: EVERY point of the weight-decay Muon trajectory has norm at
    most the COMPUTED Float `muonBallRadF B0F lrF wdLoF` — horizon-free. The ℝ couplings
    (`hB0f`/`hlrf`/`hwdlo`/`hwd1`, ball-valued gradients) are the caller's, as in C62: the
    trajectory lives in an abstract C*-algebra with no Float data to check. -/
theorem muonWd_traj_uniform_runnable {A : Type*} [CStarAlgebra A] [PartialOrder A]
    [StarOrderedRing A] (grad : A → A) (lr wd : ℝ) (k : ℕ) (θ : Nat → A)
    (B0F lrF wdLoF : Float)
    (hgrad1 : ∀ x, ‖grad x‖ ≤ 1)
    (hB0f : ‖θ 0‖ ≤ toReal B0F) (hlrf : |lr| ≤ toReal lrF)
    (hwdlo : toReal wdLoF ≤ wd) (hwd1 : wd ≤ 1)
    (hpos : checkLe minWdF wdLoF = true)
    (htraj : ∀ n, θ (n + 1) = muonWdStep grad lr wd (3 / 2) (-(1 / 2)) 0 k (θ n)) (n : ℕ) :
    ‖θ n‖ ≤ toReal (muonBallRadF B0F lrF wdLoF) := by
  have hwdF0 : 0 < toReal wdLoF := lt_of_lt_of_le minWdF_pos (checkLe_sound hpos)
  have hwd0 : 0 < wd := lt_of_lt_of_le hwdF0 hwdlo
  exact (muonWd_traj_uniform grad lr wd k θ hgrad1 hwd0 hwd1 htraj n).trans
    (muonBallRadF_dominates B0F lrF wdLoF ‖θ 0‖ lr wd (norm_nonneg _) hB0f hlrf hwdF0 hwdlo)

/-! ### Measured demos (`lr = 0.1`, decay witness `0.009`, unit initial norm) -/

-- The positivity gate passes on a real decay witness and rejects a zero one.
/-- info: true -/
#guard_msgs in #eval checkLe minWdF 0.009

/-- info: false -/
#guard_msgs in #eval checkLe minWdF 0.0

-- The computed horizon-free radius: ≈ 1.001·(1 + 1.001·(0.1/0.009)) ≈ 12.13.
/-- info: 12.134344 -/
#guard_msgs in #eval muonBallRadF 1.0 0.1 0.009

-- Non-vacuity: `ℂ` instantiates the runnable capstone (Bool hypothesis in hypothesis form —
-- Float comparison is kernel-opaque, the established C70/C79 split; `#eval` above supplies the
-- witness) at the repo's 5 NS iterations.
open scoped ComplexOrder in
example (grad : ℂ → ℂ) (lr wd : ℝ) (θ : Nat → ℂ)
    (hgrad1 : ∀ x, ‖grad x‖ ≤ 1)
    (hB0f : ‖θ 0‖ ≤ toReal (1.0 : Float)) (hlrf : |lr| ≤ toReal (0.1 : Float))
    (hwdlo : toReal (0.009 : Float) ≤ wd) (hwd1 : wd ≤ 1)
    (hpos : checkLe minWdF (0.009 : Float) = true)
    (htraj : ∀ n, θ (n + 1) = muonWdStep grad lr wd (3 / 2) (-(1 / 2)) 0 5 (θ n)) (n : ℕ) :
    ‖θ n‖ ≤ toReal (muonBallRadF 1.0 0.1 0.009) :=
  muonWd_traj_uniform_runnable grad lr wd 5 θ 1.0 0.1 0.009
    hgrad1 hB0f hlrf hwdlo hwd1 hpos htraj n

-- Non-vacuity: the horizon-free error interval, `ℂ` (contraction hypothesis in ℝ form).
open scoped ComplexOrder in
example (grad : ℂ → ℂ) (lr wd : ℝ) (G B : ℝ) (θ θ' : Nat → ℂ)
    (hgradLip : ∀ x y, ‖grad x - grad y‖ ≤ G * ‖x - y‖)
    (hgrad1 : ∀ x, ‖grad x‖ ≤ 1) (hG : 0 ≤ G) (hwd1 : wd ≤ 1) (hB : 0 ≤ B)
    (hL : (1 - wd) + |lr| * 3 ^ 5 * G < 1)
    (hstep : ∀ n, ‖θ (n + 1) - muonWdStep grad lr wd (3 / 2) (-(1 / 2)) 0 5 (θ n)‖ ≤ B)
    (hideal : ∀ n, θ' (n + 1) = muonWdStep grad lr wd (3 / 2) (-(1 / 2)) 0 5 (θ' n)) (n : ℕ) :
    ‖θ n - θ' n‖ ≤ ‖θ 0 - θ' 0‖ + B / (1 - ((1 - wd) + |lr| * 3 ^ 5 * G)) :=
  muonWd_whole_run_uniform grad lr wd 5 G B θ θ' hgradLip hgrad1 hG hwd1 hB hL hstep hideal n

end Puffer.RL.MuonBallRunFinite
