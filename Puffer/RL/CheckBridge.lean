/-
# C83: the bridge — the Mathlib-free checker core equals the verified checkers

`Puffer/Check/Core.lean` duplicates the computable surface of the verified runtime
checkers (C70 `MarginCheck`, C73 `TraceCheck`, C78 `BudgetEval`, C80 `SweepEval`,
C81 `RunConstEval`, C82 `VerifyTrace`) in a Mathlib-free module so the `puffer`
executable can call them without linking Mathlib — the CLI wiring C82's module
docstring disclosed as the remainder ("splits the computable checkers into a
Mathlib-free module first").  This module is the other half of that split, on the
Mathlib side:

* **Definitional equalities** — every core def is proved EQUAL to its original
  (`Puffer.Check.checkLe = Puffer.RL.MarginCheck.checkLe := rfl`, …).  All
  non-recursive defs close by `rfl`; seven evaluators (`dotBoundF`,
  `runBoundF`, `edgeBoundF`, `nodeBoundF`, `sweepBoundF`, `fwdBoundF`,
  `compWeightBoundF` — six recursive, plus `nodeBoundF` which merely calls
  one) are compiled with per-module equation auxiliaries the `rfl` unifier
  does not cross, so those close by `funext` + induction over the (identical)
  equation lemmas.  Either way the proofs compile only because the core bodies
  are token-for-token faithful copies; if the core ever drifts from the
  verified originals, the equalities here fail and this module's build goes
  red (the module is registered in `Puffer.lean`, so a plain `lake build`
  catches the drift).  The duplication cannot silently diverge — this file is
  the tripwire.
  `VerifyReport` is a structure, so its surface is bridged at the Bool level:
  `(Check.runTraceChecks …).allOk = (VerifyTrace.runTraceChecks …).allOk := rfl`.
* **Transferred soundness** — the original theorems restated with the CORE's
  Bools as hypotheses, proved by the equalities: `checkLe_sound`,
  `checkInterval_sound`, `checkAbsLe_sound`, `checkRegion_sound`,
  `checkTrace_sound`, the aggregate decomposition `runTraceChecks_sound` /
  `runTraceChecks_budgets_toReal`, the three C78 budget discharges
  (`dotBound_le_overflow_of_check` etc.), and the crown: C82's
  `allOk_feeds_whole_run` with the ONE aggregate Bool now computed by the
  Mathlib-free binary — a passing `puffer verify-trace` (plus C74's disclosed
  plumbing/gap/theory-side slate) feeds the machine-checked whole-run error
  interval.

**Scope (honestly disclosed).** No new checking logic and no new analysis: every
proof here is either `rfl` (faithful duplication) or an application of the
existing C70/C73/C74/C78/C82 theorems to a definitionally-equal hypothesis.  The
axiom footprint of each transferred theorem is exactly its original's — checked
externally with `#print axioms` against the built oleans (all 12 pairs matched
verbatim); the file itself carries no `#print axioms` commands, so re-run that
check after any edit here.  C74's plumbing hypotheses (trace representation,
forward error, gap constants, ideal/coupling slate) remain the caller's, exactly
as disclosed there.  The core's `formatReport`/`verifyTraceIO` are presentation
and IO — no soundness claims attach to strings, so no bridge theorems for them;
the Bool they report is `runTraceChecks`'s, which is bridged.
-/
import Puffer.Check.Core
import Puffer.RL.VerifyTrace
import Puffer.RL.SweepEval
import Puffer.RL.RunConstEval

open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)
open Puffer.RL.RegionInvariance (projAscentE)
open Puffer.RL.WholeRunFromC26 (ppoObjE Gtot)
open Puffer.RL.SoftmaxExpr (logSoftmaxE expSumE)
open Puffer.RL.SurrogateExpr (ratioE)
open Puffer.RL.TrajReachability (InRegVal)
open Puffer.RL.HTrapAssembly (clipMargin)
open Puffer.RL.FiniteBound (dotBound overflowBound)
open Puffer.RL.WholeRunFinite (runBound)
open Puffer.RL.WdRunFinite (wdUniformBound)

namespace Puffer.RL.CheckBridge

/-! ### The definitional equalities (every one closes by `rfl`)

C70 margin checkers. -/

theorem checkLe_eq : Puffer.Check.checkLe = Puffer.RL.MarginCheck.checkLe := rfl
theorem checkGe_eq : Puffer.Check.checkGe = Puffer.RL.MarginCheck.checkGe := rfl
theorem checkInterval_eq :
    Puffer.Check.checkInterval = Puffer.RL.MarginCheck.checkInterval := rfl
theorem checkAbsLe_eq :
    Puffer.Check.checkAbsLe = Puffer.RL.MarginCheck.checkAbsLe := rfl
theorem checkRegion_eq :
    Puffer.Check.checkRegion = Puffer.RL.MarginCheck.checkRegion := rfl
theorem checkClipMargin_eq :
    Puffer.Check.checkClipMargin = Puffer.RL.MarginCheck.checkClipMargin := rfl

/-- C73 trace fold (the `Trace` types are `abbrev`s of the same underlying
    `List (List Float × Float)`, so the function equality typechecks). -/
theorem checkTrace_eq :
    Puffer.Check.checkTrace = Puffer.RL.TraceCheck.checkTrace := rfl

/-! C78 budget evaluators. -/

theorem slackF_eq : Puffer.Check.slackF = Puffer.RL.BudgetEval.slackF := rfl
theorem capF_eq : Puffer.Check.capF = Puffer.RL.BudgetEval.capF := rfl
/-- Recursive defs are compiled with per-module auxiliaries the `rfl` unifier
    does not cross, so the recursive equalities close by `funext` + induction
    over the (identical) equation lemmas instead — same faithfulness tripwire:
    any body drift breaks the `simp` closure. -/
theorem dotBoundF_eq :
    Puffer.Check.dotBoundF = Puffer.RL.BudgetEval.dotBoundF := by
  funext B n
  induction n with
  | zero => simp [Puffer.Check.dotBoundF, Puffer.RL.BudgetEval.dotBoundF]
  | succ n ih =>
    simp [Puffer.Check.dotBoundF, Puffer.RL.BudgetEval.dotBoundF, ih, slackF_eq]
theorem stepBoundF_eq :
    Puffer.Check.stepBoundF = Puffer.RL.BudgetEval.stepBoundF := rfl
theorem runBoundF_eq :
    Puffer.Check.runBoundF = Puffer.RL.BudgetEval.runBoundF := by
  funext C n B
  induction n generalizing B with
  | zero => simp [Puffer.Check.runBoundF, Puffer.RL.BudgetEval.runBoundF]
  | succ n ih =>
    simp [Puffer.Check.runBoundF, Puffer.RL.BudgetEval.runBoundF, ih, stepBoundF_eq]
theorem wdUniformBoundF_eq :
    Puffer.Check.wdUniformBoundF = Puffer.RL.BudgetEval.wdUniformBoundF := rfl

/-! C81 run-constant evaluators. -/

theorem sgdStepCF_eq :
    Puffer.Check.sgdStepCF = Puffer.RL.RunConstEval.sgdStepCF := rfl
theorem nsScalarFBoundF_eq :
    Puffer.Check.nsScalarFBoundF = Puffer.RL.RunConstEval.nsScalarFBoundF := rfl
theorem muonStepCF_eq :
    Puffer.Check.muonStepCF = Puffer.RL.RunConstEval.muonStepCF := rfl
theorem wdStepRhoF_eq :
    Puffer.Check.wdStepRhoF = Puffer.RL.RunConstEval.wdStepRhoF := rfl
theorem wdStepCF_eq :
    Puffer.Check.wdStepCF = Puffer.RL.RunConstEval.wdStepCF := rfl
theorem contractThresholdF_eq :
    Puffer.Check.contractThresholdF = Puffer.RL.RunConstEval.contractThresholdF := rfl

/-! C80 sweep/expression evaluators. -/

theorem absF_eq : Puffer.Check.absF = Puffer.RL.SweepEval.absF := rfl
theorem edgeBoundF_eq :
    Puffer.Check.edgeBoundF = Puffer.RL.SweepEval.edgeBoundF := by
  funext A D C k
  induction k with
  | zero => simp [Puffer.Check.edgeBoundF, Puffer.RL.SweepEval.edgeBoundF]
  | succ k ih =>
    simp [Puffer.Check.edgeBoundF, Puffer.RL.SweepEval.edgeBoundF, ih, slackF_eq]
theorem nodeBoundF_eq :
    Puffer.Check.nodeBoundF = Puffer.RL.SweepEval.nodeBoundF := by
  funext E D C
  simp [Puffer.Check.nodeBoundF, Puffer.RL.SweepEval.nodeBoundF, edgeBoundF_eq]
theorem sweepBoundF_eq :
    Puffer.Check.sweepBoundF = Puffer.RL.SweepEval.sweepBoundF := by
  funext E D m C
  induction m generalizing C with
  | zero => simp [Puffer.Check.sweepBoundF, Puffer.RL.SweepEval.sweepBoundF]
  | succ m ih =>
    simp [Puffer.Check.sweepBoundF, Puffer.RL.SweepEval.sweepBoundF, ih, nodeBoundF_eq]
theorem fwdBoundF_eq :
    Puffer.Check.fwdBoundF = Puffer.RL.SweepEval.fwdBoundF := by
  funext B ex
  induction ex <;>
    simp [Puffer.Check.fwdBoundF, Puffer.RL.SweepEval.fwdBoundF, slackF_eq, absF_eq, *]
theorem compWeightBoundF_eq :
    Puffer.Check.compWeightBoundF = Puffer.RL.SweepEval.compWeightBoundF := by
  funext B ex
  induction ex <;>
    simp [Puffer.Check.compWeightBoundF, Puffer.RL.SweepEval.compWeightBoundF,
      fwdBoundF_eq, slackF_eq, absF_eq, *]

/-- C82 aggregate, bridged at the Bool level: `VerifyReport` is a structure and
    structure types cannot be `rfl`-equated across modules, but both `allOk`
    compositions reduce to the same `checkTrace … && budgets.all (checkLe ·.1 ·.2)`
    Bool, so the OUTPUT equality is `rfl`. -/
theorem runTraceChecks_allOk_eq (tr : Puffer.Check.Trace) (R tLo tHi : Float)
    (budgets : List (Float × Float)) :
    (Puffer.Check.runTraceChecks tr R tLo tHi budgets).allOk
      = (Puffer.RL.VerifyTrace.runTraceChecks tr R tLo tHi budgets).allOk := rfl

/-! ### Transferred soundness: the core's Bools carry the original theorems

Each proof applies the original theorem to the (definitionally equal) core
hypothesis — no new analysis, identical axiom footprint. -/

/-- The core `checkLe` is sound: C70's `checkLe_sound` transferred. -/
theorem checkLe_sound {x b : Float} (h : Puffer.Check.checkLe x b = true) :
    toReal x ≤ toReal b :=
  Puffer.RL.MarginCheck.checkLe_sound h

/-- The core `checkGe` is sound: C70's `checkGe_sound` transferred. -/
theorem checkGe_sound {x b : Float} (h : Puffer.Check.checkGe x b = true) :
    toReal b ≤ toReal x :=
  Puffer.RL.MarginCheck.checkGe_sound h

/-- The core `checkInterval` is sound: C70's `checkInterval_sound` transferred. -/
theorem checkInterval_sound {x lo hi : Float}
    (h : Puffer.Check.checkInterval x lo hi = true) :
    toReal lo ≤ toReal x ∧ toReal x ≤ toReal hi :=
  Puffer.RL.MarginCheck.checkInterval_sound h

/-- The core `checkAbsLe` is sound: C70's `checkAbsLe_sound` transferred. -/
theorem checkAbsLe_sound {x R : Float} (h : Puffer.Check.checkAbsLe x R = true) :
    |toReal x| ≤ toReal R :=
  Puffer.RL.MarginCheck.checkAbsLe_sound h

/-- The core `checkRegion` is sound: C70's `checkRegion_sound` transferred —
    a passing region check bounds every recorded parameter in ℝ. -/
theorem checkRegion_sound {θrow : List Float} {R : Float}
    (h : Puffer.Check.checkRegion θrow R = true) :
    ∀ x ∈ θrow, |toReal x| ≤ toReal R :=
  Puffer.RL.MarginCheck.checkRegion_sound h

/-- The core `checkTrace` is sound: C73's `checkTrace_sound` transferred — a
    passing trace check yields, per recorded step, the ℝ-side region bound and
    margin interval. -/
theorem checkTrace_sound {tr : Puffer.Check.Trace} {R tLo tHi : Float}
    (h : Puffer.Check.checkTrace tr R tLo tHi = true) :
    ∀ s ∈ tr, (∀ x ∈ s.1, |toReal x| ≤ toReal R)
      ∧ (toReal tLo ≤ toReal s.2 ∧ toReal s.2 ≤ toReal tHi) :=
  Puffer.RL.TraceCheck.checkTrace_sound h

/-- The core aggregate decomposes into the ORIGINAL verified checkers' Bools:
    C82's `runTraceChecks_sound` transferred.  One passing core `allOk` (the
    Bool the Mathlib-free binary computes) yields C73's `checkTrace` Bool and
    C70's per-pair `checkLe` Bools, hence every downstream theorem. -/
theorem runTraceChecks_sound {tr : Puffer.Check.Trace} {R tLo tHi : Float}
    {budgets : List (Float × Float)}
    (h : (Puffer.Check.runTraceChecks tr R tLo tHi budgets).allOk = true) :
    Puffer.RL.TraceCheck.checkTrace tr R tLo tHi = true
      ∧ ∀ bc ∈ budgets, Puffer.RL.MarginCheck.checkLe bc.1 bc.2 = true :=
  Puffer.RL.VerifyTrace.runTraceChecks_sound h

/-- The budget projection in semantic form: C82's
    `runTraceChecks_budgets_toReal` transferred to the core aggregate. -/
theorem runTraceChecks_budgets_toReal {tr : Puffer.Check.Trace} {R tLo tHi : Float}
    {budgets : List (Float × Float)}
    (h : (Puffer.Check.runTraceChecks tr R tLo tHi budgets).allOk = true) :
    ∀ bc ∈ budgets, toReal bc.1 ≤ toReal bc.2 :=
  Puffer.RL.VerifyTrace.runTraceChecks_budgets_toReal h

/-! ### The C78 budget discharges, from the core's Bools -/

/-- C78's `dotBound_le_overflow_of_check` transferred: one core `checkLe` on
    the core-evaluated dot budget certifies the ℝ-side overflow hypothesis. -/
theorem dotBound_le_overflow_of_check (Bf : Float) (B : ℝ) (n : Nat)
    (hB0 : 0 ≤ B) (hBb : B ≤ toReal Bf)
    (hchk : Puffer.Check.checkLe (Puffer.Check.dotBoundF Bf n) Puffer.Check.capF = true) :
    dotBound n B ≤ overflowBound :=
  Puffer.RL.BudgetEval.dotBound_le_overflow_of_check Bf B n hB0 hBb
    (by rw [checkLe_eq, dotBoundF_eq, capF_eq] at hchk; exact hchk)

/-- C78's `runBound_le_overflow_of_check` transferred: one core `checkLe` on
    the core-evaluated whole-run budget certifies `runBound C n B ≤ overflowBound`
    — the hypothesis `trainRun_all_finite` consumes. -/
theorem runBound_le_overflow_of_check (Cf Bf : Float) (C B : ℝ) (n : Nat)
    (hC0 : 0 ≤ C) (hCc : C ≤ toReal Cf) (hB0 : 0 ≤ B) (hBb : B ≤ toReal Bf)
    (hchk : Puffer.Check.checkLe (Puffer.Check.runBoundF Cf n Bf) Puffer.Check.capF = true) :
    runBound C n B ≤ overflowBound :=
  Puffer.RL.BudgetEval.runBound_le_overflow_of_check Cf Bf C B n hC0 hCc hB0 hBb
    (by rw [checkLe_eq, runBoundF_eq, capF_eq] at hchk; exact hchk)

/-- C78's `wdUniformBound_le_overflow_of_check` transferred: the horizon-free
    weight-decay budget discharged by one core `checkLe`. -/
theorem wdUniformBound_le_overflow_of_check (ρf Cf Bf : Float) (ρ C B : ℝ)
    (hρr : ρ ≤ toReal ρf) (hr1 : toReal ρf < 1)
    (hC0 : 0 ≤ C) (hCc : C ≤ toReal Cf) (hB0 : 0 ≤ B) (hBb : B ≤ toReal Bf)
    (hchk : Puffer.Check.checkLe (Puffer.Check.wdUniformBoundF ρf Cf Bf)
      Puffer.Check.capF = true) :
    wdUniformBound ρ C B ≤ overflowBound :=
  Puffer.RL.BudgetEval.wdUniformBound_le_overflow_of_check ρf Cf Bf ρ C B
    hρr hr1 hC0 hCc hB0 hBb hchk

/-! ### The crown: the binary's `allOk` feeds the whole-run interval -/

/-- **C82's `allOk_feeds_whole_run` transferred to the Mathlib-free core**: the
    ONE aggregate Bool computed by `lake exe puffer verify-trace` — via the core
    duplicate `Puffer.Check.runTraceChecks` — supplies the trace-Bool hypothesis
    of C74's `trace_feeds_whole_run`.  A passing report (plus C74's disclosed
    plumbing: representation `hrep`, forward error `herr`, one-time gap
    constants, C41's structural entropy caps, and the theory-side
    ideal/step/coupling slate restricted to the recorded horizon) yields the
    machine-checked whole-run error interval `L^n·d0 + B·Σ_{j<n} L^j` at every
    `n ≤ tr.length`.  Statement identical to C82's except the hypothesis
    `hall` now names the core's runner — the exact Bool the Mathlib-free
    binary prints. -/
theorem allOk_feeds_whole_run (chosen eE V : Expr) (es logps : List Expr)
    (oldLogp g lo hi cv ce ret lr : Float)
    (B d0 L R Gmag Mlog c Ment Lvent Dment Dlent : ℝ)
    (hch : Smooth chosen) (he : Smooth eE) (hes : ∀ lp ∈ es, Smooth lp) (hV : Smooth V)
    (hR : 0 ≤ R) (hc : 0 < c)
    (θ θ' : Nat → (Nat → ℝ)) (tr : Puffer.Check.Trace) (Rf tLo tHi : Float) (e : ℝ)
    (budgets : List (Float × Float))
    (hL : L = 1 + |toReal lr| * Gtot chosen eE V es logps oldLogp g cv ce ret R Ment Lvent Dment Dlent)
    (hL0 : 0 ≤ L)
    -- ONE aggregate runtime Bool — the CORE's, computed by the Mathlib-free binary:
    (hall : (Puffer.Check.runTraceChecks tr Rf tLo tHi budgets).allOk = true)
    (hR0 : 0 ≤ toReal Rf) (hRfR : toReal Rf ≤ R)
    (hrep : ∀ p (hp : p < tr.length), ∀ i,
      θ p i = toReal (Puffer.RL.TraceCheck.padRow (tr[p].1) i))
    (herr : ∀ p (hp : p < tr.length),
      |evalR (ratioE (logSoftmaxE chosen (eE :: es)) oldLogp) (θ p) - toReal (tr[p].2)| ≤ e)
    (hLoGap : toReal lo + e < toReal tLo) (hHiGap : toReal tHi + e < toReal hi)
    (hlp : ∀ lp ∈ logps, Smooth lp)
    (hVm : ∀ lp ∈ logps, vMag R lp ≤ Ment) (hDm : ∀ lp ∈ logps, dMag R lp ≤ Dment)
    (hideal : ∀ n, n < tr.length → θ' (n + 1)
        = projAscentE (ppoObjE chosen eE V es logps oldLogp g lo hi cv ce ret) lr R (θ' n))
    (hstep : ∀ n, n < tr.length → ∀ k, |θ (n + 1) k
        - projAscentE (ppoObjE chosen eE V es logps oldLogp g lo hi cv ce ret) lr R (θ n) k| ≤ B)
    (hd0 : ∀ k, |θ 0 k - θ' 0 k| ≤ d0)
    (h0θ'reg : ∀ k, |θ' 0 k| ≤ R)
    (hVal0 : InRegVal chosen eE es logps oldLogp lo hi Ment Dment (θ' 0))
    (hMent : ∀ ρ : Nat → ℝ, (∀ k, |ρ k| ≤ R) → ∀ lp ∈ logps, |evalR lp ρ| ≤ Ment)
    (hDment : ∀ ρ : Nat → ℝ, (∀ k, |ρ k| ≤ R) → ∀ k, ∀ lp ∈ logps, |derivR lp ρ k| ≤ Dment)
    (hGmagθ' : ∀ p, p < tr.length →
        ∀ k, |derivR (ppoObjE chosen eE V es logps oldLogp g lo hi cv ce ret) (θ' p) k| ≤ Gmag)
    (hfloorθ' : ∀ p, p < tr.length → c ≤ evalR (expSumE (eE :: es)) (θ' p))
    (hMθ' : ∀ p, p < tr.length → evalR (logSoftmaxE chosen (eE :: es)) (θ' p) ≤ Mlog)
    (hmarginθ' : ∀ p, p < tr.length →
        toReal lo + clipMargin chosen eE es oldLogp lr R Gmag Mlog c
          < evalR (ratioE (logSoftmaxE chosen (eE :: es)) oldLogp) (θ' p)
        ∧ evalR (ratioE (logSoftmaxE chosen (eE :: es)) oldLogp) (θ' p)
          < toReal hi - clipMargin chosen eE es oldLogp lr R Gmag Mlog c)
    (hLvE : ∀ p, p < tr.length → ∀ (δ : ℝ), (∀ i, |θ p i - θ' p i| ≤ δ) →
        ∀ lp ∈ logps, |evalR lp (θ p) - evalR lp (θ' p)| ≤ Lvent * δ)
    (hDlE : ∀ p, p < tr.length → ∀ (δ : ℝ) (k : Nat), (∀ i, |θ p i - θ' p i| ≤ δ) →
        ∀ lp ∈ logps, |derivR lp (θ p) k - derivR lp (θ' p) k| ≤ Dlent * δ)
    (hDmEn : 0 ≤ Dment) (n : Nat) (hn : n ≤ tr.length) (k : Nat) :
    |θ n k - θ' n k| ≤ L ^ n * d0 + B * ∑ j ∈ Finset.range n, L ^ j :=
  Puffer.RL.VerifyTrace.allOk_feeds_whole_run chosen eE V es logps oldLogp g lo hi cv ce ret lr
    B d0 L R Gmag Mlog c Ment Lvent Dment Dlent hch he hes hV hR hc θ θ' tr Rf tLo tHi e budgets
    hL hL0 hall hR0 hRfR hrep herr hLoGap hHiGap hlp hVm hDm hideal hstep hd0 h0θ'reg hVal0
    hMent hDment hGmagθ' hfloorθ' hMθ' hmarginθ' hLvE hDlE hDmEn n hn k

end Puffer.RL.CheckBridge
