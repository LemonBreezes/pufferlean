/-
# Runnable-layer demo: the all-Bool certificates instantiated end-to-end on concrete data

The runnable layer (C78 `BudgetEval`, C80 `SweepEval`, C81 `RunConstEval`, C82 `VerifyTrace`)
made the ℝ-side budget hypotheses of the finiteness certificates EXECUTABLE: upward-slacked
computable Float evaluators whose `toReal` provably dominates the exact ℝ budget (`slackF =
1.001`, one factor absorbing each ℝ-side `(1+u64)` plus two adverse Float roundings via
`slackF_key`), so ONE Float comparison (`checkLe … capF`, sound by `le_of_float_le`) certifies
the ℝ-side hypothesis outright. This module is the demo in the house `PipelineDemo` tradition:
it RUNS those evaluators on tiny concrete data (`#guard_msgs`-asserted, so the displays cannot
rot) and INSTANTIATES the all-Bool capstones on the SAME data as kernel-checked theorems.

**The measured numbers** (asserted below; the displays print 6 decimals):

* `sgdStepCF 1.0 1.0 0.01 = 0.010020` — vs the exact ℝ `sgdStepC ≈ 0.01`: ≈0.2% compounded
  slack (2 units). `wdStepRhoF 0.9 = 0.901801` (2 units over `wdStepRho 0.9 ≈ 0.9`),
  `wdStepCF 1.0 1.0 0.01 = 0.010030` (3 units), and the horizon-free budget
  `wdUniformBoundF … = 1.103344` (vs `wdUniformBound ≈ 1.1`, ≈0.3%) — C81's constant mirrors.
* `compWeightBoundF 1.0 (mul (var 0) (var 1)) = 1.000000` — the all-`max` weight-budget mirror
  is EXACT on this expression (no rounded op on its spine) — and the 3-node sweep budget
  `sweepBoundF 2 1.0 3 1.0 = 27.189577` — vs the exact ℝ `sweepBound ≈ 27.000`: ≈0.7% over ~7
  compounded slack units, NEGLIGIBLE against `capF = 1e300` — C80's tape-side evaluators.
* The contraction check ACCEPTS `wdStepRhoF 0.9 = 0.901801 ≤ 0.999` and REJECTS
  `wdStepRhoF 0.998 = 0.999997 > 0.999` though `0.998` is truly contractive — the ≈0.3%
  sound-not-complete band quantified at `contractThresholdF` (acceptance requires
  `BdF ≲ 0.9970`).

**The proven instances** (runtime `Bool`s in hypothesis form — Float comparison is
kernel-opaque, the C70/C79 split; the `#guard_msgs` displays are the harness's native
witnesses that each hypothesis is satisfiable):

1. `demo_horizon_free` — C81's NINE-Bool horizon-free crown at the witness
   `x = w = [0.5, −0.25]`, `gseed = 1.0`, `lr = 0.01`, `d = 0.9` (thresholds
   `BxF = B0F = BgsF = 1.0`, `BlrF = 0.01`, `BdF = 0.9`): an ARBITRARILY LONG weight-decay run
   with every weight finite — no horizon in any hypothesis.
2. `demo_tape_runnable` — C80's THREE-Bool log-free tape certificate at the `comp`-compiled
   `mul (var 0) (var 1)` on the recorded inputs `[0.5, −0.25]` (tape size 3 by `rfl`): every
   gradient the ACTUAL `grads` engine produces is finite.
3. `demo_allOk_feeds_horizon_free` — C82's ONE aggregate `allOk` (its budget list carrying THIS
   demo's two evaluated budgets) supplies the budget Bool of the nine-Bool capstone: one
   passing report line feeds the whole-run certificate, composed end-to-end.

**Scope (honestly disclosed).** Tiny by design — what is demonstrated is that the runnable
layer's pieces COMPOSE on concrete data with kernel-checked consequences, not scale. Runtime
Bools appear in hypothesis form where kernel-opaque (the native `#eval` guards supply the
witnesses — the C70 split). Nothing new is proven beyond the instances; NO new axiom.
-/
import Puffer.RL.SweepEval
import Puffer.RL.RunConstEval
import Puffer.RL.VerifyTrace

open Puffer.FloatR.ADR (Expr)
open Puffer.FloatR.AD (Tape V grads)
open Puffer.FloatR.ADReverse (comp)
open Puffer.RL.MarginCheck (checkLe checkRegion checkAbsLe)
open Puffer.RL.TraceCheck (padRow)
open Puffer.RL.WdRunFinite (wdTrainRun)
open Puffer.RL.BudgetEval (capF wdUniformBoundF)
open Puffer.RL.SweepEval (sweepBoundF compWeightBoundF adGrad_isFinite_comp_runnable)
open Puffer.RL.RunConstEval (sgdStepCF wdStepRhoF wdStepCF contractThresholdF
  wdTrainRun_all_finite_uniform_runnable)
open Puffer.RL.VerifyTrace (runTraceChecks formatReport runTraceChecks_sound)

namespace Puffer.RL.RunnableDemo

/-! ### 1. C81's constant mirrors on concrete constants (build-verified `#eval`s) -/

-- The SGD step constant at unit data bounds and `lr = 0.01`: two slack units over the ℝ
-- `sgdStepC ≈ 0.01` (≈0.2%).
/-- info: 0.010020 -/
#guard_msgs in #eval sgdStepCF 1.0 1.0 0.01

-- The weight-decay step constant: one more unit (`slackF * sgdStepCF`, ≈0.3%).
/-- info: 0.010030 -/
#guard_msgs in #eval wdStepCF 1.0 1.0 0.01

-- The weight-decay slope at `Bd = 0.9`: two scale units over `wdStepRho 0.9 ≈ 0.9`.
/-- info: 0.901801 -/
#guard_msgs in #eval wdStepRhoF 0.9

-- The contraction check accepts: `0.901801 ≤ 0.999` (strictness below `1` from the offline
-- `contractThresholdF_lt_one`).
/-- info: true -/
#guard_msgs in #eval checkLe (wdStepRhoF 0.9) contractThresholdF

-- The horizon-free budget: `max`-free closed form `slackF·(B0 + slackF·(C/(1−ρ)))` at the
-- mirrored slope/constant — ≈0.3% over the ℝ `wdUniformBound ≈ 1.1`.
/-- info: 1.103344 -/
#guard_msgs in #eval wdUniformBoundF (wdStepRhoF 0.9) (wdStepCF 1.0 1.0 0.01) 1.0

-- The ONE n-independent budget Bool of the horizon-free capstone.
/-- info: true -/
#guard_msgs in #eval checkLe (wdUniformBoundF (wdStepRhoF 0.9) (wdStepCF 1.0 1.0 0.01) 1.0) capF

-- The sound-not-complete band, measured (see `contractThresholdF`'s docstring): `0.998` is
-- truly contractive in ℝ (`(1+u64)²·0.998 < 1`) but its two slack units land at `0.999997 >
-- 0.999` — REJECTED. Conservative, never unsound.
/-- info: 0.999997 -/
#guard_msgs in #eval wdStepRhoF 0.998

/-- info: false -/
#guard_msgs in #eval checkLe (wdStepRhoF 0.998) contractThresholdF

/-! ### 2. C80's tape-side evaluators on the compiled demo circuit -/

/-- The demo circuit `x₀ · x₁` (log-free: `LogFree rdE = True ∧ True`). -/
def rdE : Expr := .mul (.var 0) (.var 1)

/-- The recorded inputs — the runtime data row the region check binds. -/
def rdInputs : List Float := [0.5, -0.25]

/-- C73's zero-padded representation of the recorded row (the `hrep` plumbing is `rfl`). -/
def rdσ : Nat → Float := padRow rdInputs

/-- The compiled tape has 3 nodes — the compilation reduces in the kernel. -/
theorem rd_tape_size : (comp rdσ rdE Tape.empty).2.val.size = 3 := rfl

-- The log-free weight-budget mirror is EXACT here: `1.000000` (var/const leaves and `max`
-- spine only — no rounded op, no slack spent).
/-- info: 1.000000 -/
#guard_msgs in #eval compWeightBoundF 1.0 rdE

-- The 3-node sweep budget: ≈0.7% above the exact ℝ `sweepBound 2 1 3 1 ≈ 27.000` — and
-- ≈298 orders of magnitude below `capF`.
/-- info: 27.189577 -/
#guard_msgs in #eval
  sweepBoundF 2 (compWeightBoundF 1.0 rdE) (comp rdσ rdE Tape.empty).2.val.size 1.0

-- The ONE sweep-budget Bool of the tape capstone (the weight budget COMPUTED from the
-- expression — nothing plumbed).
/-- info: true -/
#guard_msgs in #eval
  checkLe (sweepBoundF 2 (compWeightBoundF 1.0 rdE) (comp rdσ rdE Tape.empty).2.val.size 1.0)
    capF

/-! ### 3. C81's nine-Bool horizon-free crown, instantiated (kernel-checked) -/

-- All NINE Bools of the horizon-free capstone hold at the witness — displayed as one list
-- (region ×2, seed/lr/decay magnitude ×3, threshold nonnegativity ×2, contraction, budget).
/-- info: true -/
#guard_msgs in #eval
  [checkRegion rdInputs 1.0, checkRegion rdInputs 1.0,
   checkAbsLe 1.0 1.0, checkAbsLe 0.01 0.01, checkAbsLe 0.9 0.9,
   checkLe 0.0 1.0, checkLe 0.0 1.0,
   checkLe (wdStepRhoF 0.9) contractThresholdF,
   checkLe (wdUniformBoundF (wdStepRhoF 0.9) (wdStepCF 1.0 1.0 0.01) 1.0) capF].all id

/-- **C81's horizon-free crown at the concrete witness, kernel-checked**: from the nine runtime
    Bools (hypothesis form — each displayed `true` above), EVERY weight of the weight-decay run
    `x = w = [0.5, −0.25]`, `gseed = 1.0`, `lr = 0.01`, `d = 0.9` is finite at EVERY horizon
    `m` — no hypothesis mentions `m`. -/
theorem demo_horizon_free
    (hx : checkRegion [0.5, -0.25] 1.0 = true) (hw : checkRegion [0.5, -0.25] 1.0 = true)
    (hgs : checkAbsLe 1.0 1.0 = true) (hlr : checkAbsLe 0.01 0.01 = true)
    (hd : checkAbsLe 0.9 0.9 = true)
    (hBx0 : checkLe 0.0 1.0 = true) (hB00 : checkLe 0.0 1.0 = true)
    (hcontr : checkLe (wdStepRhoF 0.9) contractThresholdF = true)
    (hbudget : checkLe (wdUniformBoundF (wdStepRhoF 0.9) (wdStepCF 1.0 1.0 0.01) 1.0) capF
      = true) :
    ∀ m, ∀ u ∈ wdTrainRun [0.5, -0.25] 1.0 0.01 0.9 m [0.5, -0.25], u.isFinite = true :=
  wdTrainRun_all_finite_uniform_runnable [0.5, -0.25] [0.5, -0.25] 1.0 0.01 0.9
    1.0 1.0 1.0 0.01 0.9 hx hw hgs hlr hd hBx0 hB00 hcontr hbudget

/-! ### 4. C80's three-Bool tape certificate, instantiated (kernel-checked) -/

/-- **C80's log-free tape certificate at the compiled demo circuit, kernel-checked**: from the
    THREE runtime Bools (region on the recorded inputs, threshold nonnegativity, the sweep
    budget — each displayed `true` above), every gradient the ACTUAL `grads` engine produces on
    the `comp`-compiled tape of `x₀ · x₁` at `[0.5, −0.25]` is finite. The `hrep` plumbing is
    `rfl` (`rdσ` IS the zero-padded row) and `LogFree rdE` is `⟨trivial, trivial⟩`. -/
theorem demo_tape_runnable (root : V)
    (hin : checkRegion rdInputs 1.0 = true) (hB0 : checkLe 0.0 1.0 = true)
    (hbudget : checkLe (sweepBoundF 2 (compWeightBoundF 1.0 rdE)
        (comp rdσ rdE Tape.empty).2.val.size 1.0) capF = true) :
    ∀ j, j < (comp rdσ rdE Tape.empty).2.val.size →
      ((grads (comp rdσ rdE Tape.empty).2 root)[j]!).isFinite = true :=
  adGrad_isFinite_comp_runnable rdInputs rdσ 1.0 (fun _ => rfl) hin hB0 rdE
    ⟨trivial, trivial⟩ root hbudget

/-! ### 5. C82's aggregate ties in: one report, the budgets of THIS demo -/

-- The C82 report over a passing 1-step trace, its budget list carrying this demo's TWO
-- evaluated budgets (the horizon-free run budget and the tape sweep budget): all PASS.
/--
info: puffer verify-trace — verified runtime checks
  PASS  region + clip margin, every recorded step  [TraceCheck.trace_to_traj_premises; FiniteHorizonRun.trace_feeds_whole_run]
  PASS  budget caps (Float-dominated ℝ budgets)  [BudgetEval.*_le_overflow_of_check]
  => allOk = true
-/
#guard_msgs in #eval
  IO.println (formatReport (runTraceChecks [([0.5, -0.25], 0.75)] 1.0 0.25 0.8
    [(wdUniformBoundF (wdStepRhoF 0.9) (wdStepCF 1.0 1.0 0.01) 1.0, capF),
     (sweepBoundF 2 (compWeightBoundF 1.0 rdE) 3 1.0, capF)]))

/-- info: true -/
#guard_msgs in #eval
  (runTraceChecks [([0.5, -0.25], 0.75)] 1.0 0.25 0.8
    [(wdUniformBoundF (wdStepRhoF 0.9) (wdStepCF 1.0 1.0 0.01) 1.0, capF),
     (sweepBoundF 2 (compWeightBoundF 1.0 rdE) 3 1.0, capF)]).allOk

/-- **C82 feeds C81, kernel-checked**: ONE passing aggregate `allOk` — whose budget list carries
    this demo's evaluated horizon-free budget — supplies the ninth Bool (`hbudget`) of the
    horizon-free capstone via `runTraceChecks_sound`'s budget projection; the other eight remain
    their own checks. Report → whole-run finiteness certificate, composed end-to-end. -/
theorem demo_allOk_feeds_horizon_free
    (hall : (runTraceChecks [([0.5, -0.25], 0.75)] 1.0 0.25 0.8
        [(wdUniformBoundF (wdStepRhoF 0.9) (wdStepCF 1.0 1.0 0.01) 1.0, capF),
         (sweepBoundF 2 (compWeightBoundF 1.0 rdE) 3 1.0, capF)]).allOk = true)
    (hx : checkRegion [0.5, -0.25] 1.0 = true) (hw : checkRegion [0.5, -0.25] 1.0 = true)
    (hgs : checkAbsLe 1.0 1.0 = true) (hlr : checkAbsLe 0.01 0.01 = true)
    (hd : checkAbsLe 0.9 0.9 = true)
    (hBx0 : checkLe 0.0 1.0 = true) (hB00 : checkLe 0.0 1.0 = true)
    (hcontr : checkLe (wdStepRhoF 0.9) contractThresholdF = true) :
    ∀ m, ∀ u ∈ wdTrainRun [0.5, -0.25] 1.0 0.01 0.9 m [0.5, -0.25], u.isFinite = true :=
  wdTrainRun_all_finite_uniform_runnable [0.5, -0.25] [0.5, -0.25] 1.0 0.01 0.9
    1.0 1.0 1.0 0.01 0.9 hx hw hgs hlr hd hBx0 hB00 hcontr
    ((runTraceChecks_sound hall).2 _ (List.Mem.head _))

end Puffer.RL.RunnableDemo
