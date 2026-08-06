/-
# C91: the whole-run connector FULLY INSTANTIATED on concrete data

C74's `trace_feeds_whole_run` is the end-to-end connector: one passing runtime `Bool` + the
plumbing slate ⟹ the whole-run error interval. C79's `demo_connector` instantiated it at the
EMPTY trace (every per-step premise vacuous, horizon 0); C85's demos exercised the Bool layer
with the theory slate left in hypothesis form. This module closes the remaining distance: ONE
concrete toy instance with a NONEMPTY trace (horizon 1) in which EVERY hypothesis of the
connector is discharged in-file — the per-step premises are genuine obligations at `p = 0`,
none vacuous — and the conclusion is a genuinely nonzero deviation bounded by a genuinely
nonzero interval.

## The toy

A frozen two-logit uniform policy: `chosen = eE = es.head = const 0.0` (so the exact PPO ratio
is `exp(−log 2) = 1/2` at EVERY parameter point), `V = const 0.0`, `logps = []` (no entropy
terms), `lr = 0.0` (the projected update is pure clamp, `L = 1`), clip `(lo, hi) = (0.0, 1.0)`
with thresholds `(tLo, tHi) = (0.25, 0.75)` and forward-error budget `e = u64`. The recorded
trace is one step, row `[0.25]`, recorded Float ratio `0.5`; the runnable trajectory is the
zero-padded recorded row (constant in time), the ideal trajectory is all-zeros. So
`d0 = 1/2 ≥ |toReal 0.25|` genuinely separates the two trajectories and the conclusion at
`n = 1` bounds the nonzero deviation `|toReal 0.25|` by `L^1·d0 + B·Σ = 1/2`
(`full_instance_nonvacuous`).

## The accounting table (hypothesis → discharge route → status)

| hypothesis | route | status |
|---|---|---|
| `hch`/`he`/`hes`/`hV` (Smooth) | `Smooth.const` (syntactic) | closed |
| `hR`, `hc`, `hDmEn` | `norm_num` | closed |
| `hL` | `rfl` (`L91` defined verbatim) | closed |
| `hL0` | `toReal_zeroLit` kills the `Gtot` term: `L91 = 1` | closed |
| `hcheck` (the runtime `Bool`) | kernel-opaque Float comparison — hypothesis form + `#guard_msgs` native witness (the C70/C79 split) | closed at the established boundary |
| `hR0`/`hRfR` | `toReal_oneLit` | closed |
| `hrep` | by construction (`θrun` IS the zero-padded recorded row) — `rfl` | closed |
| `herr` | exact ratio `1/2` vs literal `0.5` via `toReal_ofScientific_close`, `e := u64` | closed |
| `hLoGap`/`hHiGap` | literal arithmetic with `u64 ≤ 10⁻⁶` | closed |
| `hlp`/`hVm`/`hDm`/`hMent`/`hDment`/`hLvE`/`hDlE` | vacuous (`logps = []`) | closed |
| `hideal`/`hstep` | clamp arithmetic at `lr = 0.0` (`projAscentE = clamp`) | closed |
| `hd0`/`h0θ'reg` | `|toReal 0.25| ≤ 1/2`, `|0| ≤ 1` | closed |
| `hVal0` | genuine real arithmetic: ratio `= 1/2 ∈ (0, 1)` | closed |
| `hGmagθ'` | **C40** `ppoObjE_grad_mag_concrete` (`Gmag91` := C40's closed form) | closed |
| `hfloorθ'` | `evalR (expSumE …) = 2 ≥ c = 1` | closed |
| `hMθ'` | `evalR (logSoftmaxE …) = −log 2 ≤ Mlog = 0` | closed |
| `hmarginθ'` | `clipMargin = 0` at `lr = 0.0` (the `|toReal lr|·Gmag` factor) + ratio interior | closed |

**Remainder: NONE** beyond the single kernel-opacity boundary — `hcheck` stays in hypothesis
form because Float comparison is opaque to the kernel (no `native_decide` in this development);
its truth at the concrete trace is witnessed by the build-checked `#guard_msgs` `#eval` below,
exactly the C70/C79 split used throughout.

## Honest scope

* `lr = 0.0` (a frozen policy) is what makes `hideal`/`hstep` exact clamp arithmetic and
  `clipMargin` collapse to `0` without evaluating `vLip`/`Gtot` numerically. A nonzero-`lr`
  instance is NOT structurally blocked: `Gtot`/`clipMargin`/`Gmag` are closed-form reals
  (C39/C40 routes) and would need interval arithmetic on `exp`/`log` values — more work, no
  new machinery. The margins/`hL0` here never need those numbers because the killing factor
  `|toReal 0.0| = 0` is exact.
* The toy's ratio is parameter-INDEPENDENT (all-const logits), which is what lets `herr` hold
  with the tiny budget `e = u64` at every recorded step. A parameter-dependent policy would
  discharge `herr` via the C-series forward-error theorems (`RatioForward`) instead — again a
  route that exists; here the exactness makes the discharge self-contained.
* The conclusion's LHS at `n = 1` is `|toReal 0.25 - 0| ∈ (0.24, 0.26)` — genuinely nonzero
  (`full_instance_nonvacuous`), so the instantiated interval is not vacuously `0 ≤ 0`.
-/
import Puffer.RL.FiniteHorizonRun
import Puffer.RL.ObjectiveGradMag

open Puffer.FloatR (toReal toReal_zeroLit toReal_oneLit toReal_zero toReal_ofScientific_close
  u64 u64_pos u64_lt_one)
open Puffer.FloatR.ADR
open Puffer.RL.WholeRunInterval (gradAscentE)
open Puffer.RL.RegionInvariance (projAscentE)
open Puffer.RL.WholeRunFromC26 (ppoObjE Gtot)
open Puffer.RL.SoftmaxExpr (logSoftmaxE logPartitionE expSumE)
open Puffer.RL.SurrogateExpr (ratioE)
open Puffer.RL.TrajReachability (InRegVal)
open Puffer.RL.HTrapAssembly (clipMargin)
open Puffer.RL.TraceCheck (Trace checkTrace padRow padRow_lt padRow_ge)
open Puffer.RL.FiniteHorizonRun (trace_feeds_whole_run)
open Puffer.RL.ObjectiveGradMag (ppoObjE_grad_mag_concrete)
open Puffer.RL.LogSoftmaxBudgetBundle (budgetM budgetDm)
open Puffer.RL.ValueEntropyExpr (valueSqErrE)

namespace Puffer.RL.FullInstantiation

/-! ### The toy data -/

/-- The constant-zero logit — every logit of the two-logit uniform toy policy. -/
noncomputable def zlogit : Expr := .const 0.0

/-- The recorded trace: ONE step, parameter row `[0.25]`, recorded Float ratio `0.5`. -/
def theTrace : Trace := [([0.25], 0.5)]

/-- The runnable trajectory: the zero-padded recorded row, constant in time (the recorded
    parameters do not move — `lr = 0.0`). -/
noncomputable def θrun : Nat → Nat → ℝ := fun _ i => toReal (padRow [(0.25 : Float)] i)

/-- The ideal trajectory: all-zeros (a projected-ascent fixed point at `lr = 0.0`). -/
noncomputable def θideal : Nat → Nat → ℝ := fun _ _ => 0

/-- The connector's Lipschitz constant, verbatim (`lr = 0.0` kills the `Gtot` term: `L91 = 1`). -/
noncomputable def L91 : ℝ :=
  1 + |toReal (0.0 : Float)| * Gtot zlogit zlogit (.const 0.0) [zlogit] []
    0.0 1.0 1.0 1.0 0.0 1 0 0 0 0

/-- The gradient-magnitude budget: C40's closed form (`ppoObjE_grad_mag_concrete`'s RHS) at the
    toy's data, `Ment = Dment = 0`. Never evaluated numerically — `hmarginθ'` needs only
    `clipMargin = 0` (the `|toReal 0.0|` factor), and `hGmagθ'` is C40's theorem itself. -/
noncomputable def Gmag91 : ℝ :=
  |toReal (1.0 : Float)| * (Real.exp (budgetM zlogit zlogit [zlogit] 1 - toReal (0.0 : Float))
      * budgetDm zlogit zlogit [zlogit] 1)
    + |toReal (1.0 : Float)| * dMag 1 (valueSqErrE (.const 0.0) 0.0)
    + |toReal (1.0 : Float)| * ((([] : List Expr).length : ℝ) * (Real.exp 0 * ((0 : ℝ) + 1) * 0))

/-! ### The runtime-`Bool` witnesses (native evaluation — the C70/C79 kernel-opacity split) -/

/-- info: true -/
#guard_msgs in #eval checkTrace theTrace 1.0 0.25 0.75

-- Negative control: a ratio outside the thresholds is rejected.
/-- info: false -/
#guard_msgs in #eval checkTrace [([0.25], 0.9)] 1.0 0.25 0.75

-- Negative control: a parameter row outside the region is rejected.
/-- info: false -/
#guard_msgs in #eval checkTrace [([1.5], 0.5)] 1.0 0.25 0.75

/-! ### Literal arithmetic (`toReal_ofScientific_close` + `u64 ≤ 10⁻⁶`) -/

private theorem hu6 : u64 ≤ (1 : ℝ) / 1000000 := by unfold u64; norm_num

private theorem quarter_close : |toReal (0.25 : Float) - 0.25| ≤ u64 * 0.25 := by
  have h := toReal_ofScientific_close 25 true 2
  have habs : |(0.25 : ℝ)| = (0.25 : ℝ) := abs_of_pos (by norm_num)
  rw [habs] at h
  exact h

private theorem half_close : |toReal (0.5 : Float) - 0.5| ≤ u64 * 0.5 := by
  have h := toReal_ofScientific_close 5 true 1
  have habs : |(0.5 : ℝ)| = (0.5 : ℝ) := abs_of_pos (by norm_num)
  rw [habs] at h
  exact h

private theorem threeq_close : |toReal (0.75 : Float) - 0.75| ≤ u64 * 0.75 := by
  have h := toReal_ofScientific_close 75 true 2
  have habs : |(0.75 : ℝ)| = (0.75 : ℝ) := abs_of_pos (by norm_num)
  rw [habs] at h
  exact h

private theorem quarter_pos : 0 < toReal (0.25 : Float) := by
  have h := (abs_le.mp quarter_close).1
  nlinarith [hu6, u64_pos]

private theorem quarter_le_half : toReal (0.25 : Float) ≤ 1 / 2 := by
  have h := (abs_le.mp quarter_close).2
  nlinarith [hu6, u64_pos]

/-! ### Exact evaluation of the toy policy (parameter-independent: all-const logits) -/

/-- The exact PPO ratio of the two-logit uniform policy is `1/2` at EVERY parameter point. -/
theorem ratio_eval (ρ : Nat → ℝ) :
    evalR (ratioE (logSoftmaxE zlogit (zlogit :: [zlogit])) 0.0) ρ = 1 / 2 := by
  simp only [ratioE, logSoftmaxE, logPartitionE, expSumE, evalR, zlogit, toReal_zeroLit,
    toReal_zero, Real.exp_zero]
  have h2 : (0 : ℝ) - Real.log (1 + (1 + 0)) - 0 = -Real.log 2 := by norm_num
  rw [h2, Real.exp_neg, Real.exp_log (by norm_num : (0 : ℝ) < 2)]
  norm_num

/-- The partition of the two-logit toy is exactly `2` at every parameter point. -/
theorem expsum_eval (ρ : Nat → ℝ) : evalR (expSumE (zlogit :: [zlogit])) ρ = 2 := by
  simp only [expSumE, evalR, zlogit, toReal_zeroLit, toReal_zero, Real.exp_zero]
  norm_num

/-- The log-softmax of the toy is exactly `−log 2` at every parameter point. -/
theorem lsm_eval (ρ : Nat → ℝ) :
    evalR (logSoftmaxE zlogit (zlogit :: [zlogit])) ρ = -Real.log 2 := by
  simp only [logSoftmaxE, logPartitionE, expSumE, evalR, zlogit, toReal_zeroLit, toReal_zero,
    Real.exp_zero]
  norm_num

/-! ### The frozen update: `projAscentE` at `lr = 0.0` is pure clamp -/

theorem projAscentE_zero_lr (e : Expr) (R : ℝ) (σ : Nat → ℝ) (k : Nat) :
    projAscentE e 0.0 R σ k = max (-R) (min (σ k) R) := by
  simp only [projAscentE, gradAscentE, toReal_zeroLit, zero_mul, add_zero]

/-- The clip margin collapses to `0` at `lr = 0.0`: the `|toReal lr| · Gmag` factor is exactly
    zero, so no `vLip`/`exp` budget is ever evaluated. -/
theorem margin_zero : clipMargin zlogit zlogit [zlogit] 0.0 0.0 1 Gmag91 0 1 = 0 := by
  unfold clipMargin
  simp only [toReal_zeroLit, abs_zero, zero_mul, mul_zero]

/-! ### The runnable row's entries -/

theorem θrun_head (p : Nat) : θrun p 0 = toReal (0.25 : Float) := by
  show toReal (padRow [(0.25 : Float)] 0) = _
  rw [padRow_lt (by norm_num)]
  simp

theorem θrun_tail (p j : Nat) : θrun p (j + 1) = 0 := by
  show toReal (padRow [(0.25 : Float)] (j + 1)) = 0
  rw [padRow_ge (by simp)]
  exact toReal_zeroLit

/-- Every runnable entry is a clamp fixed point in `[−1, 1]`. -/
theorem θrun_clamp (p k : Nat) : max (-(1 : ℝ)) (min (θrun p k) 1) = θrun p k := by
  cases k with
  | zero =>
      rw [θrun_head]
      have h1 := quarter_pos
      have h2 := quarter_le_half
      rw [min_eq_left (by linarith), max_eq_right (by linarith)]
  | succ j =>
      rw [θrun_tail]
      norm_num

/-! ### The capstone -/

/-- **The fully-instantiated whole-run interval (C91).** Every hypothesis of C74's
    `trace_feeds_whole_run` is discharged in-file at the concrete toy — the ONLY hypothesis-form
    input is the kernel-opaque runtime `Bool`, witnessed `true` by the `#guard_msgs` above. The
    conclusion is the whole-run error interval over the recorded horizon (`n ≤ 1`), and at
    `n = 1, k = 0` it bounds the genuinely nonzero deviation `|toReal 0.25|` by
    `L91^1·(1/2) + 0 = 1/2` (`L91 = 1`). -/
theorem full_instance (hcheck : checkTrace theTrace 1.0 0.25 0.75 = true)
    (n : Nat) (hn : n ≤ 1) (k : Nat) :
    |θrun n k - θideal n k| ≤ L91 ^ n * (1 / 2) + 0 * ∑ j ∈ Finset.range n, L91 ^ j := by
  -- Smooth (syntactic)
  have hes : ∀ lp ∈ [zlogit], Smooth lp := by
    intro lp hlp
    rw [List.mem_singleton.mp hlp]
    exact Smooth.const _
  -- hL0 : lr = 0.0 kills the Gtot term
  have hL0 : (0 : ℝ) ≤ L91 := by
    unfold L91
    rw [toReal_zeroLit]
    simp
  -- gap constants: 0 + u64 < toReal 0.25 ; toReal 0.75 + u64 < toReal 1.0 = 1
  have hLoGap : toReal (0.0 : Float) + u64 < toReal (0.25 : Float) := by
    rw [toReal_zeroLit]
    have h := (abs_le.mp quarter_close).1
    nlinarith [hu6, u64_pos]
  have hHiGap : toReal (0.75 : Float) + u64 < toReal (1.0 : Float) := by
    rw [toReal_oneLit]
    have h := (abs_le.mp threeq_close).2
    nlinarith [hu6, u64_pos]
  -- representation: θrun IS the zero-padded recorded row
  have hrep : ∀ p (hp : p < theTrace.length), ∀ i,
      θrun p i = toReal (padRow (theTrace[p].1) i) := by
    intro p hp i
    obtain rfl : p = 0 := Nat.lt_one_iff.mp hp
    rfl
  -- forward error: exact ratio 1/2 vs the recorded literal 0.5, within u64
  have herr : ∀ p (hp : p < theTrace.length),
      |evalR (ratioE (logSoftmaxE zlogit (zlogit :: [zlogit])) 0.0) (θrun p)
        - toReal (theTrace[p].2)| ≤ u64 := by
    intro p hp
    obtain rfl : p = 0 := Nat.lt_one_iff.mp hp
    show |evalR (ratioE (logSoftmaxE zlogit (zlogit :: [zlogit])) 0.0) (θrun 0)
      - toReal (0.5 : Float)| ≤ u64
    rw [ratio_eval]
    have h : |(1 : ℝ) / 2 - toReal (0.5 : Float)| = |toReal (0.5 : Float) - 0.5| := by
      rw [abs_sub_comm]
      norm_num
    rw [h]
    have := half_close
    nlinarith [u64_pos]
  -- ideal dynamics: all-zeros is a fixed point of the frozen projected update
  have hideal : ∀ n', n' < theTrace.length → θideal (n' + 1)
      = projAscentE (ppoObjE zlogit zlogit (.const 0.0) [zlogit] [] 0.0 1.0 0.0 1.0 1.0 1.0 0.0)
        0.0 1 (θideal n') := by
    intro n' _
    funext k'
    rw [projAscentE_zero_lr]
    show (0 : ℝ) = max (-1) (min 0 1)
    norm_num
  -- runnable step error: the recorded row is a clamp fixed point, so the step error is 0
  have hstep : ∀ n', n' < theTrace.length → ∀ k', |θrun (n' + 1) k'
      - projAscentE (ppoObjE zlogit zlogit (.const 0.0) [zlogit] [] 0.0 1.0 0.0 1.0 1.0 1.0 0.0)
        0.0 1 (θrun n') k'| ≤ 0 := by
    intro n' _ k'
    rw [projAscentE_zero_lr, θrun_clamp]
    show |θrun (n' + 1) k' - θrun n' k'| ≤ 0
    have : θrun (n' + 1) k' = θrun n' k' := rfl
    rw [this, sub_self, abs_zero]
  -- initial distance: |toReal 0.25| ≤ 1/2 at k = 0, 0 elsewhere
  have hd0 : ∀ k', |θrun 0 k' - θideal 0 k'| ≤ 1 / 2 := by
    intro k'
    show |θrun 0 k' - 0| ≤ 1 / 2
    rw [sub_zero]
    cases k' with
    | zero =>
        rw [θrun_head, abs_of_pos quarter_pos]
        exact quarter_le_half
    | succ j =>
        rw [θrun_tail]
        norm_num
  -- ideal start: |0| ≤ 1
  have h0reg : ∀ k', |θideal 0 k'| ≤ 1 := by
    intro k'
    show |(0 : ℝ)| ≤ 1
    norm_num
  -- ideal start in the value region: ratio = 1/2 ∈ (toReal 0.0, toReal 1.0) = (0, 1)
  have hVal0 : InRegVal zlogit zlogit [zlogit] [] 0.0 0.0 1.0 0 0 (θideal 0) := by
    refine ⟨?_, ?_, by simp, by simp⟩
    · rw [ratio_eval, toReal_zeroLit]; norm_num
    · rw [ratio_eval, toReal_oneLit]; norm_num
  -- ideal gradient magnitude: C40's closed-form budget
  have hGmag : ∀ p, p < theTrace.length → ∀ k',
      |derivR (ppoObjE zlogit zlogit (.const 0.0) [zlogit] [] 0.0 1.0 0.0 1.0 1.0 1.0 0.0)
        (θideal p) k'| ≤ Gmag91 := by
    intro p _ k'
    show _ ≤ Gmag91
    unfold Gmag91
    exact ppoObjE_grad_mag_concrete zlogit zlogit (.const 0.0) [zlogit] []
      0.0 1.0 0.0 1.0 1.0 1.0 0.0 (θideal p) 1 k' 0 0
      (Smooth.const _) (Smooth.const _) hes (Smooth.const _)
      (fun i => by show |(0 : ℝ)| ≤ 1; norm_num) zero_le_one (by simp) (by simp)
  -- ideal partition floor: 1 ≤ 2
  have hfloor : ∀ p, p < theTrace.length →
      (1 : ℝ) ≤ evalR (expSumE (zlogit :: [zlogit])) (θideal p) := by
    intro p _
    rw [expsum_eval]
    norm_num
  -- ideal log-softmax cap: −log 2 ≤ 0
  have hM : ∀ p, p < theTrace.length →
      evalR (logSoftmaxE zlogit (zlogit :: [zlogit])) (θideal p) ≤ 0 := by
    intro p _
    rw [lsm_eval]
    have := Real.log_nonneg (by norm_num : (1 : ℝ) ≤ 2)
    linarith
  -- ideal margins: clipMargin = 0, ratio = 1/2 strictly inside (0, 1)
  have hmargin : ∀ p, p < theTrace.length →
      toReal (0.0 : Float) + clipMargin zlogit zlogit [zlogit] 0.0 0.0 1 Gmag91 0 1
        < evalR (ratioE (logSoftmaxE zlogit (zlogit :: [zlogit])) 0.0) (θideal p)
      ∧ evalR (ratioE (logSoftmaxE zlogit (zlogit :: [zlogit])) 0.0) (θideal p)
        < toReal (1.0 : Float) - clipMargin zlogit zlogit [zlogit] 0.0 0.0 1 Gmag91 0 1 := by
    intro p _
    rw [margin_zero, ratio_eval, toReal_zeroLit, toReal_oneLit]
    norm_num
  exact trace_feeds_whole_run zlogit zlogit (.const 0.0) [zlogit] []
    0.0 1.0 0.0 1.0 1.0 1.0 0.0 0.0
    0 (1 / 2) L91 1 Gmag91 0 1 0 0 0 0
    (Smooth.const _) (Smooth.const _) hes (Smooth.const _)
    zero_le_one one_pos
    θrun θideal theTrace 1.0 0.25 0.75 u64
    rfl hL0
    hcheck
    (by rw [toReal_oneLit]; exact zero_le_one) (by rw [toReal_oneLit])
    hrep herr hLoGap hHiGap
    (by simp) (by simp) (by simp)
    hideal hstep hd0 h0reg hVal0
    (fun ρ _ => by simp) (fun ρ _ => by simp)
    hGmag hfloor hM hmargin
    (fun p _ δ _ => by simp) (fun p _ δ k' _ => by simp)
    le_rfl n (by simpa [theTrace] using hn) k

/-- The instantiated interval is NOT vacuous: at `n = 1, k = 0` the bounded deviation is
    genuinely nonzero — the runnable trajectory sits at `toReal 0.25 ∈ (0.24, 0.26)` while the
    ideal sits at `0`. -/
theorem full_instance_nonvacuous : 0 < |θrun 1 0 - θideal 1 0| := by
  show 0 < |θrun 1 0 - 0|
  rw [sub_zero, θrun_head, abs_of_pos quarter_pos]
  exact quarter_pos

/-- `L91 = 1`: the frozen learning rate kills the `Gtot` term exactly. -/
theorem L91_eq_one : L91 = 1 := by
  unfold L91
  rw [toReal_zeroLit]
  simp

end Puffer.RL.FullInstantiation
