/-
# Pipeline demo: the whole verification chain, instantiated end-to-end on concrete data

The formalization program (items C1–C76, see `PLAN.md`) built a continuous pipeline from network
budgets to runtime checks:

  * **Theory** — the whole-run error interval for the actual Muon optimizer on real trainer
    matrices, with the Newton–Schulz Lipschitz constant a THEOREM (C42→C53→C59→C62→C66), trajectory
    reachability via trapping regions and concrete Banach fixed points (C35–C38, C46–C49, C55, C58).
  * **Float certificates** — overflow-freedom from ONE trusted axiom (`isFinite_of_bounded`, C43)
    through the forward pass, the PPO loss scalar, the ACTUAL reverse-mode AD tape, and the
    optimizer update, over finite and unbounded horizons (C45, C51, C54, C57, C61, C64, C67, C68,
    C71, C72, C75).
  * **Runtime checks** — the irreducibly data-dependent premises behind computable, NaN-conservative,
    soundness-PROVEN `Bool` checks: per step (C70), per recorded run (C73), feeding the whole-run
    interval over the recorded horizon (C74's `trace_feeds_whole_run`).

This module is the demo in the repo's `NewtonSchulzDemo`/`MuonStepDemo` tradition: it INSTANTIATES
the pipeline's pieces on tiny concrete data, and the instances are THEOREMS — kernel-checked
evidence that the pieces compose on real inputs, not just abstractly.

  1. `#guard_msgs`-checked `#eval`s RUN the C70/C73 checkers on concrete traces (build-verified):
     a passing run, a region breach, and NaN data (rejected conservatively — IEEE comparisons with
     NaN are all-false, so a poisoned trace can only FAIL the check).
  2. A PROVEN trace-soundness instance: from a passing whole-trace `Bool` (hypothesis form — Float
     comparison is kernel-opaque, so the `Bool` is witnessed by native evaluation in the harness,
     exactly the C70/C73 honest split), every recorded parameter is certified in `[−1, 1]`.
  3. A PROVEN one-step trainer finiteness instance (C57): a concrete linear-layer training step —
     forward `dotF` and the SGD-updated weights — is overflow-free, with the budget checks
     discharged NUMERICALLY (the `(1+u64)`-budgets collapse to small rationals ≪ `overflowBound`).
  4. A PROVEN tape-certificate instance (C71): the ACTUAL `comp`-compiled tape of
     `mul (var 0) (var 1)` on inputs `1.0` has every gradient the ACTUAL `grads` engine produces
     overflow-free — the tape size reduces by `rfl`, the weight budget by `simp`, and the sweep
     budget by explicit per-node chaining (`≤ 4096 ≤ overflowBound`).
  5. A PROVEN whole-run connector instance (C74): `trace_feeds_whole_run` — the full pipeline
     statement — instantiated at the empty trace with a genuinely-satisfiable region setup
     (the PPO ratio at the all-zeros point is `exp(−1) ∈ (−1, 1)`): every hypothesis is discharged
     (vacuously, by literal `toReal` lemmas, or by real arithmetic), concluding the error interval.

**Scope (honestly disclosed).** The demos are tiny/degenerate BY DESIGN — the point is kernel-checked
COMPOSITION of the verified pieces on concrete data, not scale. Runtime `Bool`s appear in hypothesis
form where kernel-opaque (the harness evaluates them natively — the C70 split); the `#eval` guards
are build-time native evaluations, the theorems kernel proofs. Nothing new is proven beyond the
instances.
-/
import Puffer.RL.FiniteHorizonRun
import Puffer.RL.CompTapeWeights
import Puffer.RL.BackwardFinite

open Puffer.FloatR.ADR
open Puffer.FloatR (toReal u64 u64_pos u64_lt_one toReal_zeroLit toReal_oneLit toReal_neg
  toReal_zero)
open Puffer.FloatR.AD (Tape V grads)
open Puffer.FloatR.ADReverse (comp)
open Puffer.RL.FiniteBound (overflowBound dotBound dotF)
open Puffer.RL.ADTapeFinite (LogFree fwdBound sweepBound nodeBound edgeBound edgeBound_nonneg)
open Puffer.RL.CompTapeWeights (compWeightBound adGrad_isFinite_comp)
open Puffer.RL.BackwardFinite (linearTrainStep linear_train_step_all_finite)
open Puffer.RL.MarginCheck (checkRegion checkClipMargin)
open Puffer.RL.TraceCheck (Trace checkTrace checkTrace_sound)
open Puffer.RL.WholeRunFromC26 (Gtot)
open Puffer.RL.TrajReachability (InRegVal)
open Puffer.RL.SoftmaxExpr (logSoftmaxE logPartitionE expSumE)
open Puffer.RL.SurrogateExpr (ratioE)
open Puffer.RL.FiniteHorizonRun (trace_feeds_whole_run)

namespace Puffer.RL.PipelineDemo

/-! ### 1. The runtime checkers on concrete traces (build-verified `#eval`s) -/

-- A 2-step recorded run — parameter rows inside [−1, 1], ratios inside the [0.25, 0.8]
-- thresholds — passes the whole-trace check: ONE Bool certifies the run's premise slate (C73).
/-- info: true -/
#guard_msgs in #eval checkTrace [([0.5, -0.25], 0.75), ([0.9, 0.1], 0.5)] 1.0 0.25 0.8

-- A parameter excursion (1.5 ∉ [−1, 1]) fails the region check — the run is NOT certified.
/-- info: false -/
#guard_msgs in #eval checkTrace [([0.5, 1.5], 0.75)] 1.0 0.25 0.8

-- A ratio outside the margin thresholds (0.9 > 0.8) fails the clip check.
/-- info: false -/
#guard_msgs in #eval checkTrace [([0.5, -0.25], 0.9)] 1.0 0.25 0.8

-- NaN data (0/0) is rejected CONSERVATIVELY: IEEE comparisons with NaN are all-false, so a
-- poisoned trace can only fail — `true` is never wrongly returned (C70's soundness design).
/-- info: false -/
#guard_msgs in #eval checkTrace [([0.5, 0.0 / 0.0], 0.75)] 1.0 0.25 0.8

/-- info: false -/
#guard_msgs in #eval checkClipMargin (0.0 / 0.0) 0.25 0.8

/-! ### 2. A proven trace-soundness instance (kernel-checked consequence of the runtime Bool) -/

/-- From a passing whole-trace check on a concrete 1-step trace (the `Bool` in hypothesis form —
    Float comparison is kernel-opaque, so the witness is the harness's native evaluation; this
    trace is the 1-step prefix of §1's displayed 2-step passing trace, and its own check likewise
    evaluates `true` natively), EVERY recorded parameter is certified in `[−toReal 1.0, toReal 1.0]`
    — the machine-checked guarantee behind the passing check (C70 → C73). -/
example (h : checkTrace [([0.5, -0.25], 0.75)] 1.0 0.25 0.8 = true) :
    ∀ x ∈ [(0.5 : Float), -0.25], |toReal x| ≤ toReal (1.0 : Float) :=
  fun x hx =>
    ((checkTrace_sound h) _ (List.mem_singleton_self _)).1 x hx

/-! ### 3. Budget helpers: the `(1+u64)` budgets collapse to small numbers ≪ `overflowBound` -/

/-- `4096 ≤ overflowBound` — the demo budgets sit twelve binary orders of magnitude into a
    threshold of ≈ 1.8·10³⁰⁸: `overflowBound = (2 − 2⁻⁵²)·2¹⁰²³ ≥ 2¹⁰²³ ≥ 2¹² = 4096`. -/
theorem demo_overflow : (4096 : ℝ) ≤ overflowBound := by
  have h2 : (4096 : ℝ) ≤ 2 ^ (1023 : ℕ) := by
    calc (4096 : ℝ) = 2 ^ (12 : ℕ) := by norm_num
      _ ≤ 2 ^ (1023 : ℕ) := pow_le_pow_right₀ one_le_two (by norm_num)
  refine h2.trans ?_
  unfold overflowBound
  exact le_mul_of_one_le_left (by positivity) (by norm_num)

/-- One reverse-sweep node (`E = 2` edges, weight budget `D = 1`) grows the adjoint budget by at
    most `16×` — the C68 `nodeBound` collapses to integer arithmetic once `1 + u64 ≤ 2`. -/
theorem demo_nodeBound_le (C M : ℝ) (h0 : 0 ≤ C) (hCM : C ≤ M) :
    nodeBound 2 1 C ≤ 16 * M := by
  have hu0 : (0 : ℝ) < u64 := u64_pos
  have hu1 : u64 < 1 := u64_lt_one
  simp only [nodeBound, edgeBound]
  nlinarith [mul_nonneg h0 hu0.le, sq_nonneg u64, mul_nonneg (mul_nonneg h0 hu0.le) hu0.le]

theorem demo_nodeBound_nonneg (C : ℝ) (h0 : 0 ≤ C) : 0 ≤ nodeBound 2 1 C := by
  have := edgeBound_nonneg C 1 C h0 zero_le_one h0 2
  simpa [nodeBound] using this

/-- The 3-node reverse sweep of the demo tape stays under `4096` — three `16×` node steps from the
    seed budget `1`. -/
theorem demo_sweep : sweepBound 2 1 3 1 ≤ 4096 := by
  have n1 : nodeBound 2 1 1 ≤ 16 := by
    simpa using demo_nodeBound_le 1 1 zero_le_one le_rfl
  have p1 : (0 : ℝ) ≤ nodeBound 2 1 1 := demo_nodeBound_nonneg 1 zero_le_one
  have n2 : nodeBound 2 1 (nodeBound 2 1 1) ≤ 16 * 16 := by
    have := demo_nodeBound_le (nodeBound 2 1 1) 16 p1 n1
    exact this.trans (by norm_num)
  have p2 : (0 : ℝ) ≤ nodeBound 2 1 (nodeBound 2 1 1) := demo_nodeBound_nonneg _ p1
  have n3 : nodeBound 2 1 (nodeBound 2 1 (nodeBound 2 1 1)) ≤ 16 * 256 := by
    have := demo_nodeBound_le _ (16 * 16) p2 (by linarith)
    exact this.trans (by norm_num)
  show sweepBound 2 1 3 1 ≤ 4096
  simp only [sweepBound]
  linarith

/-! ### 4. A proven one-step trainer finiteness instance (C57) -/

/-- **One concrete training step is overflow-free, kernel-checked** (C57's
    `linear_train_step_all_finite` at `x = w = [1.0]`, `gseed = lr = 1.0`, all budgets `1`): the
    forward `dotF` output AND every SGD-updated weight are `isFinite`. Both budget checks are
    discharged numerically — `dotBound 1 1 = (1+u64)² ≤ 4096` and the backward-plus-update budget
    `(1+u64)·(1 + (1+u64)·(1+u64)) ≤ 4096`, twelve binary orders below `overflowBound`. -/
theorem demo_train_step :
    (dotF [1.0] [1.0]).isFinite = true ∧
      ∀ u ∈ linearTrainStep [1.0] [1.0] 1.0 1.0, u.isFinite = true := by
  have hu0 : (0 : ℝ) < u64 := u64_pos
  have hu1 : u64 < 1 := u64_lt_one
  refine linear_train_step_all_finite [1.0] [1.0] 1.0 1.0 1 1 1 zero_le_one
    (by simp) (by simp) (by simp) (by simp) ?_ ?_
  · -- forward budget: dotBound (min 1 1) 1 = (1+u64)² ≤ 4096 ≤ overflowBound
    have h4 : dotBound (min 1 1) 1 ≤ 4096 := by
      show dotBound 1 1 ≤ 4096
      simp only [dotBound]
      nlinarith
    exact h4.trans demo_overflow
  · -- backward + update budget ≤ 4096 ≤ overflowBound
    have h4 : (1 + u64) * (1 + (1 + u64) * (1 * ((1 + u64) * (1 * 1)))) ≤ (4096 : ℝ) := by
      have hu2 : u64 * u64 < 1 := by nlinarith
      have hu3 : u64 * u64 * u64 < 1 := by nlinarith
      nlinarith
    exact h4.trans demo_overflow

/-! ### 5. A proven tape-certificate instance (C68 + C71) -/

/-- The demo circuit `x₀ · x₁` and the all-ones input. -/
noncomputable def demoE : Expr := .mul (.var 0) (.var 1)

def demoσ : Nat → Float := fun _ => 1.0

/-- **The ACTUAL AD engine's gradients on a concrete compiled tape are overflow-free,
    kernel-checked** (C71's `adGrad_isFinite_comp` on the `comp`-compiled `mul (var 0) (var 1)` at
    inputs `1.0`): the compiled tape has 3 nodes (by `rfl` — the compilation reduces in the
    kernel), the weight budget collapses to `1` (by `simp` — the sibling forward values are
    `fwdBound 1 (var _) = 1`), and the 3-node sweep budget is `≤ 4096 ≤ overflowBound`
    (`demo_sweep`). Every adjoint the imperative `grads` sweep produces is `isFinite`. -/
theorem demo_tape (root : V) :
    ∀ j, j < (comp demoσ demoE Tape.empty).2.val.size →
      ((grads (comp demoσ demoE Tape.empty).2 root)[j]!).isFinite = true := by
  refine adGrad_isFinite_comp demoσ 1 (fun i => by simp [demoσ]) demoE ⟨trivial, trivial⟩ root ?_
  have hD : compWeightBound 1 demoE = 1 := by
    simp [demoE, compWeightBound, fwdBound]
  have hsz : (comp demoσ demoE Tape.empty).2.val.size = 3 := rfl
  rw [hD, hsz]
  exact demo_sweep.trans demo_overflow

/-! ### 6. A proven whole-run connector instance (C74) -/

/-- The all-zeros trajectory (runnable and ideal coincide — the degenerate exact run). -/
noncomputable def demoθ : Nat → Nat → ℝ := fun _ _ => 0

/-- The connector's Lipschitz constant at the demo instances — `lr = 0.0` kills the `Gtot` term,
    so `demoL = 1`. -/
noncomputable def demoL : ℝ :=
  1 + |toReal (0.0 : Float)| * Gtot (.const (-1.0)) (.const 0.0) (.const 0.0) [] []
    0.0 0.0 0.0 0.0 0.0 1 0 0 0 0

/-- **The full pipeline statement, instantiated end-to-end, kernel-checked** (C74's
    `trace_feeds_whole_run` at the empty trace): the runtime check is `rfl` on `[]`, the
    region/margin constants are the pinned literals (`toReal` of `±1.0`/`0.0`), the ideal
    start-in-region condition is GENUINE real arithmetic — the PPO ratio of the demo network at the
    all-zeros point is `exp(−1) ∈ (−1, 1)` — and every trajectory-side premise discharges
    (vacuously at horizon 0, or by `|0| ≤ 1`). The conclusion is the whole-run error interval over
    the recorded horizon. Tiny by design: what is demonstrated is that the theory-side theorems,
    the Float-side constants, and the runtime-check layer genuinely COMPOSE. -/
theorem demo_connector (k : Nat) :
    |demoθ 0 k - demoθ 0 k| ≤ demoL ^ 0 * 0 + 0 * ∑ j ∈ Finset.range 0, demoL ^ j := by
  have hval : evalR (ratioE (logSoftmaxE (.const (-1.0)) [(.const 0.0)]) 0.0) (demoθ 0)
      = Real.exp (-1) := by
    simp [ratioE, logSoftmaxE, logPartitionE, expSumE, evalR]
  refine trace_feeds_whole_run (.const (-1.0)) (.const 0.0) (.const 0.0) [] []
    0.0 0.0 (-1.0) 1.0 0.0 0.0 0.0 0.0
    0 0 demoL 1 0 0 1 0 0 0 0
    (Smooth.const _) (Smooth.const _) (by simp) (Smooth.const _)
    zero_le_one one_pos
    demoθ demoθ ([] : Trace) 1.0 0.0 0.0 0
    rfl ?_ rfl ?_ ?_
    (fun p hp => absurd hp (Nat.not_lt_zero p))
    (fun p hp => absurd hp (Nat.not_lt_zero p))
    ?_ ?_ (by simp) (by simp) (by simp)
    (fun n hn => absurd hn (Nat.not_lt_zero n))
    (fun n hn => absurd hn (Nat.not_lt_zero n))
    (by simp [demoθ]) (by simp [demoθ]) ?_ (by simp) (by simp)
    (fun p hp => absurd hp (Nat.not_lt_zero p))
    (fun p hp => absurd hp (Nat.not_lt_zero p))
    (fun p hp => absurd hp (Nat.not_lt_zero p))
    (fun p hp => absurd hp (Nat.not_lt_zero p))
    (fun p hp => absurd hp (Nat.not_lt_zero p))
    (fun p hp => absurd hp (Nat.not_lt_zero p))
    le_rfl 0 (Nat.zero_le _) k
  -- hL0 : 0 ≤ demoL (= 1, the lr = 0.0 factor killing Gtot)
  · unfold demoL; rw [toReal_zeroLit]; simp
  -- hR0 : 0 ≤ toReal 1.0
  · rw [toReal_oneLit]; exact zero_le_one
  -- hRfR : toReal 1.0 ≤ R = 1
  · rw [toReal_oneLit]
  -- the one-time gap constants: toReal (−1.0) + 0 < toReal 0.0, toReal 0.0 + 0 < toReal 1.0
  · simp
  · simp
  -- hVal0 : the ideal starts in the region — the ratio at the all-zeros point is exp(−1) ∈ (−1, 1)
  · refine ⟨?_, ?_, by simp, by simp⟩
    · rw [hval]; simp only [toReal_neg, toReal_oneLit]
      have := Real.exp_pos (-1 : ℝ); linarith
    · rw [hval, toReal_oneLit]
      calc Real.exp (-1) < Real.exp 0 := Real.exp_lt_exp.mpr (by norm_num)
        _ = 1 := Real.exp_zero

end Puffer.RL.PipelineDemo
