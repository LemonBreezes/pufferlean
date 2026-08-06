import Puffer.Float.Expr

/-!
# C83: Mathlib-free checker core for the `puffer` executable

The verified runtime checkers (C70 `MarginCheck`, C73 `TraceCheck`, C78
`BudgetEval`, C80 `SweepEval`, C81 `RunConstEval`, C82 `VerifyTrace`) live in
Mathlib-importing modules: their *soundness theorems* mention `toReal : Float → ℝ`
and the ℝ-side trainer model, so they cannot be imported by `Exe/Puffer.lean`
without linking Mathlib into the binary — breaking the exe's documented
"all executable code here is Mathlib-free" invariant.

This module is the resolution: a **Mathlib-free duplicate of the computable
surface only**.  Every definition below is a *syntactically faithful copy* of
its original (same body, token for token, modulo namespace), so each one is
definitionally equal to the original and `Puffer/RL/CheckBridge.lean` proves

  `Puffer.Check.checkLe = Puffer.RL.MarginCheck.checkLe := rfl`

(and likewise for every other def) on the Mathlib side, then *transfers* the
original soundness theorems to these names.  The `rfl`s are the safety net: if
a copy here ever drifts from its original, the bridge fails to compile and the
build goes red — the duplication cannot silently diverge.

Import discipline: this file imports **only** `Puffer.Float.Expr` (itself
Mathlib-free — its transitive closure is `Expr → ErrBnd → Exec → ∅`), which
provides the 11-constructor AD grammar `Expr` needed by the C80 per-expression
budget evaluators `fwdBoundF`/`compWeightBoundF`.

Contents (checker layer, then budget-evaluator layer, then report layer):
* C70 margin checkers: `checkLe`, `checkGe`, `checkInterval`, `checkAbsLe`,
  `checkRegion`, `checkClipMargin`;
* C73 trace layer: `StepRec`, `Trace`, `checkTrace`;
* C78 budget evaluators: `slackF`, `capF`, `dotBoundF`, `stepBoundF`,
  `runBoundF`, `wdUniformBoundF`;
* C81 run-constant evaluators: `sgdStepCF`, `nsScalarFBoundF`, `muonStepCF`,
  `wdStepRhoF`, `wdStepCF`, `contractThresholdF`;
* C80 sweep/expression evaluators: `edgeBoundF`, `nodeBoundF`, `sweepBoundF`,
  `absF`, `fwdBoundF`, `compWeightBoundF`;
* C82 report surface: `VerifyReport`, `VerifyReport.allOk`, `runTraceChecks`,
  `formatReport`, `verifyTraceIO`.

What each Bool *means* (the theorems live with the originals and are
transferred by the bridge): a passing `checkTrace` plus C74's disclosed
plumbing yields the whole-run error interval
(`FiniteHorizonRun.trace_feeds_whole_run`); a passing `checkLe bound cap`
with a C78-style Float bound dominates the corresponding ℝ budget
(`BudgetEval.*_le_overflow_of_check`); and `runTraceChecks`'s single `allOk`
Bool feeds both at once (`VerifyTrace.allOk_feeds_whole_run`).
-/

namespace Puffer.Check

open Puffer.FloatR.ADR (Expr)

/-! ### C70 margin checkers (duplicates of `Puffer.RL.MarginCheck`)

NaN-conservative by construction: every comparison with a NaN input is
`false`, so a passing check never lies about an ill-formed Float. -/

/-- `x ≤ b` as a runnable Bool.  Duplicate of `MarginCheck.checkLe`; sound by
    the transferred `checkLe_sound : checkLe x b = true → toReal x ≤ toReal b`. -/
def checkLe (x b : Float) : Bool := decide (x ≤ b)

/-- `b ≤ x` as a runnable Bool.  Duplicate of `MarginCheck.checkGe`. -/
def checkGe (x b : Float) : Bool := decide (b ≤ x)

/-- Two-sided interval membership `lo ≤ x ≤ hi`.  Duplicate of
    `MarginCheck.checkInterval`. -/
def checkInterval (x lo hi : Float) : Bool := checkGe x lo && checkLe x hi

/-- Symmetric bound `|x| ≤ R`, phrased as interval membership in `[-R, R]`.
    Duplicate of `MarginCheck.checkAbsLe`. -/
def checkAbsLe (x R : Float) : Bool := checkInterval x (-R) R

/-- Every recorded parameter of one step lies in the `R`-ball: the runnable
    region check behind C68's region-invariance premise.  Duplicate of
    `MarginCheck.checkRegion`. -/
def checkRegion (θrow : List Float) (R : Float) : Bool :=
  θrow.all (fun x => checkAbsLe x R)

/-- The recorded PPO ratio sits inside the strict clip margin `[tLo, tHi]`:
    the runnable clip-interior check behind C69's barrier premise.  Duplicate
    of `MarginCheck.checkClipMargin`. -/
def checkClipMargin (x tLo tHi : Float) : Bool := checkInterval x tLo tHi

/-! ### C73 trace layer (duplicates of `Puffer.RL.TraceCheck`) -/

/-- One recorded training step: the parameter row and the recorded PPO ratio.
    Duplicate of `TraceCheck.StepRec`. -/
abbrev StepRec : Type := List Float × Float

/-- A recorded training trace: one `StepRec` per step.  Duplicate of
    `TraceCheck.Trace`. -/
abbrev Trace : Type := List StepRec

/-- Every step of the trace passes both the region check and the clip-margin
    check.  Duplicate of `TraceCheck.checkTrace`; via the bridge, a `true`
    here supplies the trace-Bool hypothesis of C74's
    `trace_feeds_whole_run` whole-run interval. -/
def checkTrace (tr : Trace) (R tLo tHi : Float) : Bool :=
  tr.all (fun s => checkRegion s.1 R && checkClipMargin s.2 tLo tHi)

/-! ### C78 budget evaluators (duplicates of `Puffer.RL.BudgetEval`)

Each `…F` evaluator is an upward-slacked Float mirror of an ℝ-side budget:
C78/C80/C81 prove `budget_ℝ ≤ toReal (…F args)` whenever the recursion stays
below `capF`, so `checkLe (…F args) cap = true` discharges the ℝ budget
hypothesis at runtime. -/

/-- Per-operation slack factor absorbing Float rounding in the evaluators.
    Duplicate of `BudgetEval.slackF`. -/
def slackF : Float := 1.001

/-- Overflow guard: evaluator outputs are compared against this cap, far below
    `Float.inf` but far above any realistic budget.  Duplicate of
    `BudgetEval.capF`. -/
def capF : Float := 1e300

/-- Slacked dot-product budget: `n` accumulations of a `B·B` product.
    Duplicate of `BudgetEval.dotBoundF`. -/
def dotBoundF (B : Float) : Nat → Float
  | 0 => 0.0
  | n + 1 => slackF * (slackF * (B * B) + dotBoundF B n)

/-- One slacked training-step radius growth: `B ↦ slackF·(B + C)`.  Duplicate
    of `BudgetEval.stepBoundF`. -/
def stepBoundF (C B : Float) : Float := slackF * (B + C)

/-- `n`-step iterated radius budget: fold `stepBoundF C` from `B`.  Duplicate
    of `BudgetEval.runBoundF`. -/
def runBoundF (C : Float) : Nat → Float → Float
  | 0, B => B
  | n + 1, B => runBoundF C n (stepBoundF C B)

/-- Horizon-free weight-decay radius budget `slackF·(B + slackF·(C/(1-ρ)))`.
    Duplicate of `BudgetEval.wdUniformBoundF`. -/
def wdUniformBoundF (ρ C B : Float) : Float :=
  slackF * (B + slackF * (C / (1.0 - ρ)))

/-! ### C81 run-constant evaluators (duplicates of `Puffer.RL.RunConstEval`) -/

/-- Slacked SGD per-step displacement constant `lr·(gs·x)`.  Duplicate of
    `RunConstEval.sgdStepCF`. -/
def sgdStepCF (Bx Bgs Blr : Float) : Float :=
  slackF * (Blr * (slackF * (Bgs * Bx)))

/-- Slacked Newton–Schulz scalar-step output bound
    `σ·(a + b·σ² + c·σ⁴)` with per-operation slack.  Duplicate of
    `RunConstEval.nsScalarFBoundF`. -/
def nsScalarFBoundF (Ba Bb Bc Bσ : Float) : Float :=
  slackF * (Bσ * (slackF *
    (slackF * (Ba + slackF * (Bb * (slackF * (Bσ * Bσ))))
      + slackF * (Bc * (slackF * ((slackF * (Bσ * Bσ)) * (slackF * (Bσ * Bσ))))))))

/-- Slacked Muon per-step displacement constant: learning rate times the
    NS-orthogonalized gradient bound.  Duplicate of `RunConstEval.muonStepCF`. -/
def muonStepCF (Bx Bgs Blr Ba Bb Bc : Float) : Float :=
  slackF * (Blr * nsScalarFBoundF Ba Bb Bc (slackF * (Bgs * Bx)))

/-- Slacked weight-decay contraction factor `slackF²·Bd`.  Duplicate of
    `RunConstEval.wdStepRhoF`. -/
def wdStepRhoF (Bd : Float) : Float := slackF * (slackF * Bd)

/-- Slacked weight-decay per-step displacement constant.  Duplicate of
    `RunConstEval.wdStepCF`. -/
def wdStepCF (Bx Bgs Blr : Float) : Float := slackF * sgdStepCF Bx Bgs Blr

/-- Strict contraction threshold for the runnable `ρ < 1` check.  Duplicate of
    `RunConstEval.contractThresholdF`. -/
def contractThresholdF : Float := 0.999

/-! ### C80 sweep/expression evaluators (duplicates of `Puffer.RL.SweepEval`) -/

/-- Slacked per-edge reverse-sweep adjoint budget after `k` edges.  Duplicate
    of `SweepEval.edgeBoundF`. -/
def edgeBoundF (A D C : Float) : Nat → Float
  | 0 => C
  | k + 1 => slackF * (edgeBoundF A D C k + slackF * (A * D))

/-- Per-node adjoint budget: `E` edges of `edgeBoundF` seeded at `C`.
    Duplicate of `SweepEval.nodeBoundF`. -/
def nodeBoundF (E : Nat) (D C : Float) : Float := edgeBoundF C D C E

/-- Whole-sweep adjoint budget: `m` nodes of `nodeBoundF`.  Duplicate of
    `SweepEval.sweepBoundF`. -/
def sweepBoundF (E : Nat) (D : Float) : Nat → Float → Float
  | 0, C => C
  | m + 1, C => sweepBoundF E D m (nodeBoundF E D C)

/-- Runnable absolute value `max c (-c)` (avoids `Float.abs`'s bit-level
    definition).  Duplicate of `SweepEval.absF`. -/
def absF (c : Float) : Float := max c (-c)

/-- Slacked structural forward-value budget of an expression over the `B`-ball,
    by recursion on the AD grammar.  Duplicate of `SweepEval.fwdBoundF`. -/
def fwdBoundF (B : Float) : Expr → Float
  | .var _ => B
  | .const c => absF c
  | .add a b => slackF * (fwdBoundF B a + fwdBoundF B b)
  | .sub a b => slackF * (fwdBoundF B a + fwdBoundF B b)
  | .mul a b => slackF * (fwdBoundF B a * fwdBoundF B b)
  | .scale c a => slackF * (absF c * fwdBoundF B a)
  | .exp a => slackF * Float.exp (fwdBoundF B a)
  | .log _ => 0.0
  | .relu a => fwdBoundF B a
  | .max a b => max (fwdBoundF B a) (fwdBoundF B b)
  | .min a b => max (fwdBoundF B a) (fwdBoundF B b)

/-- Slacked per-node composition-weight budget of an expression over the
    `B`-ball (the local-derivative magnitude cap the reverse sweep multiplies
    by).  Duplicate of `SweepEval.compWeightBoundF`. -/
def compWeightBoundF (B : Float) : Expr → Float
  | .var _ => 1.0
  | .const _ => 1.0
  | .add a b => max 1.0 (max (compWeightBoundF B a) (compWeightBoundF B b))
  | .sub a b => max 1.0 (max (compWeightBoundF B a) (compWeightBoundF B b))
  | .mul a b => max (max (fwdBoundF B a) (fwdBoundF B b))
      (max (compWeightBoundF B a) (compWeightBoundF B b))
  | .scale c a => max (absF c) (compWeightBoundF B a)
  | .exp a => max (slackF * Float.exp (fwdBoundF B a)) (compWeightBoundF B a)
  | .log _ => 0.0
  | .relu a => max 1.0 (compWeightBoundF B a)
  | .max a b => max 1.0 (max (compWeightBoundF B a) (compWeightBoundF B b))
  | .min a b => max 1.0 (max (compWeightBoundF B a) (compWeightBoundF B b))

/-! ### C82 report surface (duplicates of `Puffer.RL.VerifyTrace`)

`VerifyReport` is duplicated as a *new structure type* — structure types
cannot be `rfl`-equated across modules, so the bridge equates the Bool
*outputs* instead: `(Check.runTraceChecks …).allOk =
(VerifyTrace.runTraceChecks …).allOk` is `rfl` because both sides reduce to
the same `checkTrace … && budgets.all …` Bool. -/

/-- The two-field verification report: trace-margin half and budget half.
    Field-for-field duplicate of `VerifyTrace.VerifyReport`. -/
structure VerifyReport where
  /-- Every recorded step passed the region + clip-margin checks. -/
  regionMarginOk : Bool
  /-- Every supplied Float budget passed its cap check. -/
  budgetsOk : Bool
  deriving Repr, DecidableEq

/-- The single aggregate Bool.  Duplicate of `VerifyTrace.VerifyReport.allOk`;
    via the bridge, `allOk = true` feeds C74's whole-run interval *and* the
    per-pair budget dominations at once. -/
def VerifyReport.allOk (r : VerifyReport) : Bool :=
  r.regionMarginOk && r.budgetsOk

/-- Run every check on a recorded trace plus a list of `(bound, cap)` budget
    pairs.  Duplicate of `VerifyTrace.runTraceChecks`. -/
def runTraceChecks (tr : Trace) (R tLo tHi : Float)
    (budgets : List (Float × Float)) : VerifyReport :=
  { regionMarginOk := checkTrace tr R tLo tHi
    budgetsOk := budgets.all fun bc => checkLe bc.1 bc.2 }

/-- A human-readable pass/fail table naming the soundness theorems behind each
    line.  Duplicate of `VerifyTrace.formatReport` (same strings, so the exe
    output matches C82's build-asserted demo format). -/
def formatReport (r : VerifyReport) : String :=
  let line (ok : Bool) (name thm : String) : String :=
    s!"  {if ok then "PASS" else "FAIL"}  {name}  [{thm}]"
  String.intercalate "\n"
    [ "puffer verify-trace — verified runtime checks"
    , line r.regionMarginOk "region + clip margin, every recorded step"
        "TraceCheck.trace_to_traj_premises; FiniteHorizonRun.trace_feeds_whole_run"
    , line r.budgetsOk "budget caps (Float-dominated ℝ budgets)"
        "BudgetEval.*_le_overflow_of_check"
    , s!"  => allOk = {r.allOk}" ]

/-- Run the checks, print the report, return the aggregate Bool.  Duplicate of
    `VerifyTrace.verifyTraceIO` — the entry point `Exe/Puffer.lean`'s
    `verify-trace` subcommand calls. -/
def verifyTraceIO (tr : Trace) (R tLo tHi : Float)
    (budgets : List (Float × Float)) : IO Bool := do
  let r := runTraceChecks tr R tLo tHi budgets
  IO.println (formatReport r)
  pure r.allOk

/-! ### Build-asserted demos (mirroring C82's, now Mathlib-free) -/

-- The C82 demo aggregate, recomputed in the Mathlib-free core: a two-step
-- trace inside the region and clip margins, plus two passing budget pairs.
/--
info: true
-/
#guard_msgs in
#eval (runTraceChecks [([0.5, -0.25], 0.75), ([0.9, 0.1], 0.5)]
  1.0 0.25 0.8 [(1.0, capF), (2048.0, capF)]).allOk

-- A failing case: ratio `0.9` breaches the clip margin `[0.25, 0.8]`.
/--
info: false
-/
#guard_msgs in
#eval (runTraceChecks [([0.5, -0.25], 0.9)] 1.0 0.25 0.8 []).allOk

-- The budget evaluators run in the core: a 64-term dot budget stays far
-- below the overflow cap.
/--
info: true
-/
#guard_msgs in
#eval checkLe (dotBoundF 1.0 64) capF

end Puffer.Check
