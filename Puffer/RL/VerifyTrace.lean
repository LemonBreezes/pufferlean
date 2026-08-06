/-
# The verify-harness entry point: one report, one `allOk` Bool, every theorem attached

The session's runtime layer ships as separate verified checkers: C70 (`MarginCheck`) — per-step
region/margin Bools with soundness; C73 (`TraceCheck`) — the whole-trace fold `checkTrace`; C78
(`BudgetEval`) — runnable budget comparisons `checkLe b cap` with proven domination; and C74
(`FiniteHorizonRun.trace_feeds_whole_run`) — the pipeline connector consuming the trace Bool. This
module is the HARNESS ENTRY POINT that composes them into one structured report the `puffer`
executable can call:

* `VerifyReport` — one Bool per check family, each field documented with the soundness theorem
  standing behind it, and the aggregate `VerifyReport.allOk`.
* `runTraceChecks` — the PURE runner: `checkTrace` (C73) over the recorded trace + `checkLe`
  (C70/C78) over the supplied `(budget, cap)` pairs. No IO, no new checking logic — the verified
  checkers verbatim.
* `runTraceChecks_sound` — the aggregate soundness (pure Bool plumbing): one passing `allOk`
  yields every constituent Bool, hence every constituent soundness theorem. `runTraceChecks_checkTrace`
  and `runTraceChecks_budgets_toReal` are the two projections in consumable form.
* `allOk_feeds_whole_run` — the composed consequence: C74's `trace_feeds_whole_run` restated with
  its trace-Bool hypothesis supplied BY the aggregate — one passing `allOk` (plus C74's disclosed
  plumbing/gap/theory-side hypotheses) yields the machine-checked whole-run error interval over
  the recorded horizon.
* `formatReport`/`verifyTraceIO` — the IO surface: a human-readable pass/fail table naming the
  theorems, and the `IO Bool` an exe target calls.
* `#guard_msgs` demos — build-asserted passing/breaching reports (the house Demo pattern; these
  cannot rot).

**Scope (honestly disclosed).** The runner composes EXISTING verified checkers — all soundness
lives in C70/C73/C74/C78's theorems (pointed to per field); the aggregate theorem is Bool plumbing
that lets a single `allOk` carry them all. The budget pairs' SEMANTIC meaning (which ℝ budget each
Float dominates) is the caller's pairing, discharged by C78's `*_dominates`/`*_le_overflow_of_check`
per pair. C74's plumbing hypotheses (trace representation `hrep`, forward error `herr`, one-time
gap constants, and the theory-side ideal/coupling slate) remain exactly as disclosed there — this
module certifies precisely the Bool layer. CLI WIRING: the `puffer` executable (`Exe/Puffer.lean`)
has a clean subcommand match (`"verify" :: _ => runVerify`, line ~550), and the one-line wiring is
`| "verify-trace" :: _ => <build a Trace from emitted data; Puffer.RL.VerifyTrace.verifyTraceIO …>`
— but that file documents a DELIBERATE invariant ("All executable code here is Mathlib-free, so
the binary links no Mathlib"), and this module (via the checkers' soundness theorems) transitively
imports Mathlib. Wiring it would silently break that documented design property, so the dispatch
line is disclosed here as the remainder, to be added only if the maintainer accepts a
Mathlib-linked binary (or splits the computable checkers into a Mathlib-free module first).
-/
import Puffer.RL.FiniteHorizonRun
import Puffer.RL.BudgetEval

open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)
open Puffer.RL.RegionInvariance (projAscentE)
open Puffer.RL.WholeRunFromC26 (ppoObjE Gtot)
open Puffer.RL.SoftmaxExpr (logSoftmaxE expSumE)
open Puffer.RL.SurrogateExpr (ratioE)
open Puffer.RL.TrajReachability (InRegVal)
open Puffer.RL.HTrapAssembly (clipMargin)
open Puffer.RL.TraceCheck (Trace checkTrace padRow)
open Puffer.RL.MarginCheck (checkLe checkLe_sound)
open Puffer.RL.BudgetEval (capF)
open Puffer.RL.FiniteHorizonRun (trace_feeds_whole_run)

namespace Puffer.RL.VerifyTrace

/-- **The verify report** — one Bool per check family, each backed by a soundness theorem:

* `regionMarginOk` — C73's `checkTrace` over the recorded trace: every step's parameter row in
  `[−R, R]` and its ratio inside the margin thresholds. Behind a `true`: C70's `step_certificate`
  per step, C73's `trace_to_traj_premises` for the whole-trace `∀ p < tr.length` premise slate,
  and (with C74's plumbing) `trace_feeds_whole_run`'s whole-run error interval.
* `budgetsOk` — C78's runnable budget comparisons: every supplied `(budget, cap)` Float pair
  passes `checkLe`. Behind a `true` (per pair, with the caller's pairing): C78's
  `dotBound_le_overflow_of_check` / `runBound_le_overflow_of_check` /
  `wdUniformBound_le_overflow_of_check` — the ℝ-side overflow hypotheses of the finiteness
  certificates (C43/C57/C61/C64/C67/C68/C71/C75/C77). -/
structure VerifyReport where
  /-- C73 `checkTrace`: region + clip margin at every recorded step (see `runTraceChecks_checkTrace`). -/
  regionMarginOk : Bool
  /-- C70/C78 `checkLe` over every supplied `(budget, cap)` pair (see `runTraceChecks_budgets_toReal`). -/
  budgetsOk : Bool
  deriving Repr, DecidableEq

/-- The aggregate: every check family passed. -/
def VerifyReport.allOk (r : VerifyReport) : Bool := r.regionMarginOk && r.budgetsOk

/-- **The pure runner** — the verified checkers verbatim, no new checking logic: C73's
    `checkTrace` on the trace, C70/C78's `checkLe` on each `(budget, cap)` pair. -/
def runTraceChecks (tr : Trace) (R tLo tHi : Float) (budgets : List (Float × Float)) :
    VerifyReport :=
  { regionMarginOk := checkTrace tr R tLo tHi
    budgetsOk := budgets.all fun bc => checkLe bc.1 bc.2 }

/-- **Aggregate soundness (pure Bool plumbing).** One passing `allOk` yields every constituent
    Bool — hence every constituent soundness theorem (C70's/C73's for the trace half, C78's for
    each budget pair). -/
theorem runTraceChecks_sound {tr : Trace} {R tLo tHi : Float} {budgets : List (Float × Float)}
    (h : (runTraceChecks tr R tLo tHi budgets).allOk = true) :
    checkTrace tr R tLo tHi = true ∧ ∀ bc ∈ budgets, checkLe bc.1 bc.2 = true := by
  simp only [runTraceChecks, VerifyReport.allOk, Bool.and_eq_true, List.all_eq_true] at h
  exact h

/-- The trace projection: a passing `allOk` supplies exactly the Bool hypothesis C73's
    `trace_to_traj_premises` and C74's `trace_feeds_whole_run` consume. -/
theorem runTraceChecks_checkTrace {tr : Trace} {R tLo tHi : Float}
    {budgets : List (Float × Float)}
    (h : (runTraceChecks tr R tLo tHi budgets).allOk = true) :
    checkTrace tr R tLo tHi = true :=
  (runTraceChecks_sound h).1

/-- The budget projection, in semantic form: every supplied Float budget's `toReal` is below its
    cap's. With the caller's pairing `(dotBoundF …, capF)`-style, C78's
    `*_le_overflow_of_check` lemmas turn each pair into the ℝ-side
    `budget ≤ overflowBound` hypothesis the finiteness certificates consume. -/
theorem runTraceChecks_budgets_toReal {tr : Trace} {R tLo tHi : Float}
    {budgets : List (Float × Float)}
    (h : (runTraceChecks tr R tLo tHi budgets).allOk = true) :
    ∀ bc ∈ budgets, toReal bc.1 ≤ toReal bc.2 :=
  fun bc hbc => checkLe_sound ((runTraceChecks_sound h).2 bc hbc)

/-- **The composed consequence: one `allOk` feeds the whole-run interval.** C74's
    `trace_feeds_whole_run` with its trace-Bool hypothesis supplied by the aggregate — a passing
    `runTraceChecks … |>.allOk` (plus C74's disclosed plumbing: representation `hrep`, forward
    error `herr`, one-time gap constants, C41's structural entropy caps, and the theory-side
    ideal/step/coupling slate restricted to the recorded horizon) yields the machine-checked
    whole-run error interval `L^n·d0 + B·Σ_{j<n} L^j` at every `n ≤ tr.length`. The budget half
    of the report rides along untouched (its consequences are per-pair, via
    `runTraceChecks_budgets_toReal`). -/
theorem allOk_feeds_whole_run (chosen eE V : Expr) (es logps : List Expr)
    (oldLogp g lo hi cv ce ret lr : Float)
    (B d0 L R Gmag Mlog c Ment Lvent Dment Dlent : ℝ)
    (hch : Smooth chosen) (he : Smooth eE) (hes : ∀ lp ∈ es, Smooth lp) (hV : Smooth V)
    (hR : 0 ≤ R) (hc : 0 < c)
    (θ θ' : Nat → (Nat → ℝ)) (tr : Trace) (Rf tLo tHi : Float) (e : ℝ)
    (budgets : List (Float × Float))
    (hL : L = 1 + |toReal lr| * Gtot chosen eE V es logps oldLogp g cv ce ret R Ment Lvent Dment Dlent)
    (hL0 : 0 ≤ L)
    -- ONE aggregate runtime Bool in place of the raw trace Bool:
    (hall : (runTraceChecks tr Rf tLo tHi budgets).allOk = true)
    (hR0 : 0 ≤ toReal Rf) (hRfR : toReal Rf ≤ R)
    (hrep : ∀ p (hp : p < tr.length), ∀ i, θ p i = toReal (padRow (tr[p].1) i))
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
  trace_feeds_whole_run chosen eE V es logps oldLogp g lo hi cv ce ret lr
    B d0 L R Gmag Mlog c Ment Lvent Dment Dlent hch he hes hV hR hc θ θ' tr Rf tLo tHi e
    hL hL0 (runTraceChecks_checkTrace hall) hR0 hRfR hrep herr hLoGap hHiGap
    hlp hVm hDm hideal hstep hd0 h0θ'reg hVal0 hMent hDment
    hGmagθ' hfloorθ' hMθ' hmarginθ' hLvE hDlE hDmEn n hn k

/-! ### The IO surface -/

/-- A human-readable pass/fail table naming the soundness theorems behind each line. -/
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

/-- The exe-callable entry: run the pure checks, print the report, return the aggregate. -/
def verifyTraceIO (tr : Trace) (R tLo tHi : Float) (budgets : List (Float × Float)) :
    IO Bool := do
  let r := runTraceChecks tr R tLo tHi budgets
  IO.println (formatReport r)
  pure r.allOk

/-! ### Build-asserted demos (the house `#guard_msgs` pattern — these cannot rot) -/

/-- info: true -/
#guard_msgs in #eval
  (runTraceChecks [([0.5, -0.25], 0.75), ([0.9, 0.1], 0.5)] 1.0 0.25 0.8
    [(1.0, capF), (2048.0, capF)]).allOk

/-- info: { regionMarginOk := true, budgetsOk := true } -/
#guard_msgs in #eval
  runTraceChecks [([0.5, -0.25], 0.75)] 1.0 0.25 0.8 [(1.0, capF)]

/-- info: false -/
#guard_msgs in #eval  -- region breach (1.5 ∉ [−1,1]) fails the trace half
  (runTraceChecks [([0.5, 1.5], 0.75)] 1.0 0.25 0.8 [(1.0, capF)]).allOk

/-- info: { regionMarginOk := true, budgetsOk := false } -/
#guard_msgs in #eval  -- an over-cap budget fails ONLY the budget half
  runTraceChecks [([0.5, -0.25], 0.75)] 1.0 0.25 0.8 [(1.0e301, capF)]

/-- info: false -/
#guard_msgs in #eval  -- NaN data is rejected conservatively (IEEE comparisons all-false)
  (runTraceChecks [([0.5, 0.0 / 0.0], 0.75)] 1.0 0.25 0.8 []).allOk

/--
info: puffer verify-trace — verified runtime checks
  PASS  region + clip margin, every recorded step  [TraceCheck.trace_to_traj_premises; FiniteHorizonRun.trace_feeds_whole_run]
  FAIL  budget caps (Float-dominated ℝ budgets)  [BudgetEval.*_le_overflow_of_check]
  => allOk = false
-/
#guard_msgs in #eval
  IO.println (formatReport (runTraceChecks [([0.5, -0.25], 0.75)] 1.0 0.25 0.8 [(1.0e301, capF)]))

end Puffer.RL.VerifyTrace
