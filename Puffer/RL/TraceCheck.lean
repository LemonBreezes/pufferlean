/-
# Whole-trace checker: the `∀ p` lift of the per-step margin certificates over a recorded run

C70 (`MarginCheck`) certified ONE step's runtime data — a passing `checkRegion`/`checkClipMargin`
`Bool` implies that step's `hRegθ p`-row / `hIntθ p` premise. The whole-run theorems (C35's
`ppo_whole_run_reachable`, C38's `ppo_whole_run_from_barriers`) consume those premises quantified
over the run (`hRegθ : ∀ p i, |θ p i| ≤ R`; `hIntθ : ∀ p, ratio(θ p) ∈ (lo, hi)`). This module
delivers the lift: a COMPUTABLE whole-trace checker over a recorded trajectory — one fold of C70's
per-step `Bool`s — with the soundness theorem that a passing trace check yields the premise slate at
EVERY recorded step.

* `Trace` — the recorded run: per step, the parameter row and the computed Float ratio (exactly the
  per-step data the `puffer verify` harness already collects).
* `checkTrace` — `tr.all (checkRegion ∧ checkClipMargin)`: one computable `Bool` for the whole run.
* `checkTrace_step` / `checkTrace_sound` / `checkTrace_sound_idx` — the per-step `Bool` and semantic
  facts at every member / every position of the trace.
* `padRow` + `trace_to_traj_premises` (capstone) — the bridge to the ACTUAL premise shapes: for a
  trajectory `θ : Nat → (Nat → ℝ)` REPRESENTED by the trace (each `θ p` the ZERO-PADDED recorded
  row, and the recorded Float ratios within the proven forward error `e` of the exact ratios), a
  passing `checkTrace` plus C70's one-time gap hypotheses yields, for EVERY `p < tr.length`,
  `(∀ i, |θ p i| ≤ toReal R) ∧ (toReal lo < ratio(θ p) ∧ ratio(θ p) < toReal hi)` — verbatim the
  `hRegθ`/`hIntθ` conjuncts of C35/C38, restricted to the recorded horizon. Indices beyond the
  recorded row are zero-padded, sound via `0 ≤ toReal R` (the caller obligation C70 disclosed, now
  discharged in-theorem). `trace_region_slate`/`trace_interior_slate` are the two projections.

**Scope (honestly disclosed).** The lift is over the RECORDED horizon: a finite trace certifies the
slate for `p < tr.length` — the honest reach of runtime checking (a run of `n` steps needs a trace
of length `≥ n`). The existing whole-run theorems STATE their runnable-side premises as unbounded
`∀ p`; their accumulation proofs only ever consume steps `p < n` below the queried step, so
consuming the restricted slate needs the mechanical restricted-horizon restatement of those
theorems — not done here (the IDEAL-side `∀ p` conditions are already theorems via the C35/C38/C58
trapping machinery; this module is about the RUNNABLE trajectory's data). The representation
hypotheses (`θ p` = the zero-padded recorded row; per-step ratio forward error `≤ e`) are the
caller's data plumbing — the harness records exactly this data, and `e` is the proven forward-error
budget (e.g. `RatioForward`). The one-time gap hypotheses (`toReal lo + e < toReal tLo`,
`toReal tHi + e < toReal hi`) remain offline constants-only obligations (C70). Sound-not-complete
inherited: Float comparisons are conservative (and kernel-opaque — a passing `checkTrace` arises
from native evaluation at runtime, as in `puffer verify`); a borderline-true run may fail the
check, but a passing check always certifies.
-/
import Puffer.RL.MarginCheck

open Puffer.FloatR
open Puffer.FloatR.ADR
open Puffer.RL.SoftmaxExpr (logSoftmaxE)
open Puffer.RL.SurrogateExpr (ratioE)
open Puffer.RL.MarginCheck

namespace Puffer.RL.TraceCheck

/-- One recorded step: the parameter row and the computed Float ratio at that step. -/
abbrev StepRec : Type := List Float × Float

/-- A recorded trajectory — the per-step data the `puffer verify` harness already collects. -/
abbrev Trace : Type := List StepRec

/-- **The whole-trace checker**: every recorded step passes BOTH of C70's per-step checks — the
    parameter row inside `[−R, R]` and the Float ratio inside the `[tLo, tHi]` thresholds. One
    computable `Bool` for the entire recorded run. -/
def checkTrace (tr : Trace) (R tLo tHi : Float) : Bool :=
  tr.all (fun s => checkRegion s.1 R && checkClipMargin s.2 tLo tHi)

/-- A passing trace check gives BOTH per-step `Bool` facts at every member — the form downstream
    per-step soundness theorems (C70) consume directly. -/
theorem checkTrace_step {tr : Trace} {R tLo tHi : Float}
    (h : checkTrace tr R tLo tHi = true) :
    ∀ s ∈ tr, checkRegion s.1 R = true ∧ checkClipMargin s.2 tLo tHi = true := by
  rw [checkTrace, List.all_eq_true] at h
  intro s hs
  have hs' := h s hs
  rw [Bool.and_eq_true] at hs'
  exact hs'

/-- Semantic member form: at every recorded step, every parameter entry is in `[−R, R]` and the
    recorded Float ratio is (non-strictly) inside the thresholds. The STRICT/exact-value margin
    facts come from C70's gap-hypothesis soundness at the capstone below. -/
theorem checkTrace_sound {tr : Trace} {R tLo tHi : Float}
    (h : checkTrace tr R tLo tHi = true) :
    ∀ s ∈ tr, (∀ x ∈ s.1, |toReal x| ≤ toReal R)
      ∧ (toReal tLo ≤ toReal s.2 ∧ toReal s.2 ≤ toReal tHi) :=
  fun s hs =>
    ⟨checkRegion_sound (checkTrace_step h s hs).1,
     checkInterval_sound (checkTrace_step h s hs).2⟩

/-- Positional form: the per-step `Bool` facts at every position `p < tr.length` — the index-aligned
    shape the trajectory bridge consumes. -/
theorem checkTrace_sound_idx {tr : Trace} {R tLo tHi : Float}
    (h : checkTrace tr R tLo tHi = true) :
    ∀ p (hp : p < tr.length),
      checkRegion (tr[p].1) R = true ∧ checkClipMargin (tr[p].2) tLo tHi = true :=
  fun _ hp => checkTrace_step h _ (tr.getElem_mem hp)

/-! ### The zero-padded row and the trajectory bridge -/

/-- The zero-padded total reading of a finite parameter row: entry `i` for `i < row.length`, `0.0`
    beyond — C35's `hRegθ` quantifies over ALL `Nat` indices, so the finite recorded row is read as
    a total `Nat → Float` this way (the padding C70 disclosed as the caller's obligation). -/
def padRow (row : List Float) (i : Nat) : Float := row.getD i 0.0

theorem padRow_lt {row : List Float} {i : Nat} (h : i < row.length) :
    padRow row i = row[i] := by
  simp [padRow, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]

theorem padRow_ge {row : List Float} {i : Nat} (h : row.length ≤ i) :
    padRow row i = 0.0 := by
  simp [padRow, List.getD_eq_getElem?_getD, List.getElem?_eq_none h]

/-- Every zero-padded entry of a checked row is bounded: on the row by `checkRegion`'s soundness,
    beyond it by `|toReal 0.0| = 0 ≤ toReal R`. -/
theorem padRow_abs_le {row : List Float} {R : Float}
    (hreg : checkRegion row R = true) (hR0 : 0 ≤ toReal R) :
    ∀ i, |toReal (padRow row i)| ≤ toReal R := by
  intro i
  by_cases hi : i < row.length
  · rw [padRow_lt hi]
    exact checkRegion_sound_idx hreg i hi
  · rw [padRow_ge (Nat.le_of_not_lt hi), toReal_zeroLit, abs_zero]
    exact hR0

/-- **The trajectory-premise bridge (capstone).** For a trajectory `θ` represented by the recorded
    trace — each `θ p` the zero-padded recorded row (`hrep`), each recorded Float ratio within the
    proven forward error `e` of the exact ratio at `θ p` (`herr`) — a passing `checkTrace`, the
    padding bound `0 ≤ toReal Rf`, and C70's one-time gap hypotheses yield the whole-run premise
    slate at EVERY recorded step: for all `p < tr.length`,
    `(∀ i, |θ p i| ≤ toReal Rf) ∧ (toReal lo < ratio(θ p) ∧ ratio(θ p) < toReal hi)` — verbatim the
    `hRegθ`/`hIntθ` conjuncts of C35's `ppo_whole_run_reachable` / C38's
    `ppo_whole_run_from_barriers`, restricted to the recorded horizon `p < tr.length` (the honest
    reach of a finite trace; see the module scope note on the restricted-horizon consumption). -/
theorem trace_to_traj_premises (chosen eE : Expr) (es : List Expr) (oldLogp lo hi : Float)
    (θ : Nat → (Nat → ℝ)) (tr : Trace) (Rf tLo tHi : Float) (e : ℝ)
    (hcheck : checkTrace tr Rf tLo tHi = true)
    (hR0 : 0 ≤ toReal Rf)
    (hrep : ∀ p (hp : p < tr.length), ∀ i, θ p i = toReal (padRow (tr[p].1) i))
    (herr : ∀ p (hp : p < tr.length),
      |evalR (ratioE (logSoftmaxE chosen (eE :: es)) oldLogp) (θ p) - toReal (tr[p].2)| ≤ e)
    (hLo : toReal lo + e < toReal tLo) (hHi : toReal tHi + e < toReal hi) :
    ∀ p (_hp : p < tr.length),
      (∀ i, |θ p i| ≤ toReal Rf)
        ∧ (toReal lo < evalR (ratioE (logSoftmaxE chosen (eE :: es)) oldLogp) (θ p)
            ∧ evalR (ratioE (logSoftmaxE chosen (eE :: es)) oldLogp) (θ p) < toReal hi) := by
  intro p hp
  obtain ⟨hreg, hclip⟩ := checkTrace_sound_idx hcheck p hp
  refine ⟨fun i => ?_, checkClipMargin_sound_exact hclip (herr p hp) hLo hHi⟩
  rw [hrep p hp i]
  exact padRow_abs_le hreg hR0 i

/-- Projection: the region slate `∀ p < tr.length, ∀ i, |θ p i| ≤ toReal Rf` — the restricted-horizon
    `hRegθ`. -/
theorem trace_region_slate (chosen eE : Expr) (es : List Expr) (oldLogp lo hi : Float)
    (θ : Nat → (Nat → ℝ)) (tr : Trace) (Rf tLo tHi : Float) (e : ℝ)
    (hcheck : checkTrace tr Rf tLo tHi = true) (hR0 : 0 ≤ toReal Rf)
    (hrep : ∀ p (hp : p < tr.length), ∀ i, θ p i = toReal (padRow (tr[p].1) i))
    (herr : ∀ p (hp : p < tr.length),
      |evalR (ratioE (logSoftmaxE chosen (eE :: es)) oldLogp) (θ p) - toReal (tr[p].2)| ≤ e)
    (hLo : toReal lo + e < toReal tLo) (hHi : toReal tHi + e < toReal hi) :
    ∀ p (_hp : p < tr.length), ∀ i, |θ p i| ≤ toReal Rf :=
  fun p hp => (trace_to_traj_premises chosen eE es oldLogp lo hi θ tr Rf tLo tHi e
    hcheck hR0 hrep herr hLo hHi p hp).1

/-- Projection: the clip-interior slate `∀ p < tr.length, ratio(θ p) ∈ (toReal lo, toReal hi)` —
    the restricted-horizon `hIntθ`. -/
theorem trace_interior_slate (chosen eE : Expr) (es : List Expr) (oldLogp lo hi : Float)
    (θ : Nat → (Nat → ℝ)) (tr : Trace) (Rf tLo tHi : Float) (e : ℝ)
    (hcheck : checkTrace tr Rf tLo tHi = true) (hR0 : 0 ≤ toReal Rf)
    (hrep : ∀ p (hp : p < tr.length), ∀ i, θ p i = toReal (padRow (tr[p].1) i))
    (herr : ∀ p (hp : p < tr.length),
      |evalR (ratioE (logSoftmaxE chosen (eE :: es)) oldLogp) (θ p) - toReal (tr[p].2)| ≤ e)
    (hLo : toReal lo + e < toReal tLo) (hHi : toReal tHi + e < toReal hi) :
    ∀ p (_hp : p < tr.length),
      toReal lo < evalR (ratioE (logSoftmaxE chosen (eE :: es)) oldLogp) (θ p)
        ∧ evalR (ratioE (logSoftmaxE chosen (eE :: es)) oldLogp) (θ p) < toReal hi :=
  fun p hp => (trace_to_traj_premises chosen eE es oldLogp lo hi θ tr Rf tLo tHi e
    hcheck hR0 hrep herr hLo hHi p hp).2

/-- Non-vacuity (conditional): on a singleton trace, a passing whole-trace check reduces to the
    two per-step `Bool`s. (Float comparison is kernel-opaque, so a concrete passing trace is
    witnessed by NATIVE evaluation at runtime — `#eval checkTrace …` in the harness — not by
    kernel reduction; this in-file example shows the reduction pipeline instantiates.) -/
example (row : List Float) (r R tLo tHi : Float)
    (h : checkTrace [(row, r)] R tLo tHi = true) :
    checkRegion row R = true ∧ checkClipMargin r tLo tHi = true :=
  checkTrace_step h _ (List.mem_singleton_self _)

end Puffer.RL.TraceCheck
