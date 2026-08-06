/-
# Log-inclusive AD-tape finiteness: extending C68's forward certificate to the FULL grammar

C68 (`ADTapeFinite`) certified the actual AD engine's forward pass (`evalF`) on the LOG-FREE
sub-grammar, excluding `log` because its value (`log x → −∞` as `x → 0⁺`) and its backward edge
weight (`1/x`) need a domain floor. This module supplies the excluded op via the FLOOR pattern
C51 (`LossForwardFinite.logPart_isFinite`) established:

* **`LogFloored σ c e`** — the honest per-input semantic condition: at every `log` subterm, the
  argument's Float VALUE at this input is floored (`c ≤ toReal (evalF a σ)`, with a single global
  floor `0 < c`). For the PPO loss's only `log` — the log-partition `log(Σ exp(logits))` — the
  floor is STRUCTURAL: C9's `expSumE_floor` gives the ℝ-side partition `≥ exp(−vMag R e) > 0`
  (and `sumExpF_pos` gives Float-side strict positivity); the quantitative Float-side floor `c`
  transfers from the ℝ-side one within the proven forward error (the `evalF`-vs-`evalR` error
  bounds) — that transfer is the caller's (disclosed) glue, exactly as in C51.
* **`fwdBoundL`** — the log-inclusive forward budget: C68's `fwdBound` with the log case
  `max |log c| |log (fwdBoundL … a)|·(1+logEps)` (C51's bound, the argument's own budget serving
  as the ceiling `U`).
* **`evalFL_mag` / `evalFL_isFinite`** — the forward magnitude invariant and no-overflow
  certificate over the FULL 11-op grammar (extending C68's `evalF_mag` induction with the log
  case via C51's `logMag_le`). `logFree_logFloored`/`fwdBoundL_eq_fwdBound` show the extension is
  CONSERVATIVE: on log-free terms the predicate is vacuous and the budget coincides with C68's.
* **`log_edge_weight_mag`** — the BACKWARD half's missing piece: the `comp` compiler records the
  log node's edge weight as the Float `1.0 / x` (`ADReverse.comp`, the `.log` case) — under the
  floor `c ≤ toReal x` its magnitude is `≤ (1/c)·(1+u64)`, via the PRE-EXISTING `div_model` +
  `toReal_oneLit`. **`log_node_bounds`** packages both halves (value + edge weight) for one
  floored log node — the interface C68's tape hypothesis `hDw` consumes for log-inclusive
  comp-built tapes (`D ≥ max (log-free weight budget) ((1/c)·(1+u64))`).

**Scope (honestly disclosed).** The forward certificate now covers the FULL grammar; the floor
is a per-input semantic hypothesis (dischargeable for the actual PPO loss via the structural
partition floor, C9/C51 — the ℝ→Float floor transfer through the forward error is the caller's
glue, not formalized here). The backward edge weight for `log` is bounded (so C68's `hDw` is
dischargeable for log-inclusive comp tapes once the per-node value-readout glue C68 already
disclosed lands — that glue is unchanged by this module and remains the one composition step).
NO new axiom: `log_model`/`div_model`/`toReal_oneLit` are pre-existing trusted-base facts; the
finiteness axiom is C43's `isFinite_of_bounded`, reused in the certificates only.
-/
import Puffer.RL.ADTapeFinite

namespace Puffer.RL.LogTapeFinite

open Puffer.FloatR (toReal u64 u64_pos expEps logEps toReal_oneLit toReal_reluF div_model)
open Puffer.FloatR.ADR (Expr evalF)
open Puffer.RL.ADTapeFinite (LogFree fwdBound)
open Puffer.RL.FiniteBound (isFinite_of_bounded overflowBound)
open Puffer.RL.LossFinite (expF_mag_le)
open Puffer.RL.LossForwardFinite (mul_bound sub_bound min_mag_le max_mag_le logMag_le)
open Puffer.RL.PPOLossScalar (add_bound)

/-! ### The floored-log predicate: every `log` argument's VALUE is floored at this input -/

/-- **Per-input floored-log condition.** Structural on every op; at `log a` it additionally
    requires the argument's Float value at THIS input to be `≥ c` (the single global floor,
    `0 < c` supplied by the theorems). This is the honest semantic side-condition of C51's
    `logPart_isFinite`, lifted to the whole grammar. For the PPO loss's log-partition the floor
    is structural (C9's `expSumE_floor` / C51's `sumExpF_pos`). -/
def LogFloored (σ : Nat → Float) (c : ℝ) : Expr → Prop
  | .var _ => True
  | .const _ => True
  | .add a b => LogFloored σ c a ∧ LogFloored σ c b
  | .sub a b => LogFloored σ c a ∧ LogFloored σ c b
  | .mul a b => LogFloored σ c a ∧ LogFloored σ c b
  | .scale _ a => LogFloored σ c a
  | .exp a => LogFloored σ c a
  | .log a => LogFloored σ c a ∧ c ≤ toReal (evalF a σ)
  | .relu a => LogFloored σ c a
  | .max a b => LogFloored σ c a ∧ LogFloored σ c b
  | .min a b => LogFloored σ c a ∧ LogFloored σ c b

/-- Log-free terms are vacuously floored (the extension is conservative). -/
theorem logFree_logFloored (σ : Nat → Float) (c : ℝ) :
    ∀ e, LogFree e → LogFloored σ c e := by
  intro e
  induction e with
  | var i => intro _; trivial
  | const cc => intro _; trivial
  | add a b iha ihb => intro h; exact ⟨iha h.1, ihb h.2⟩
  | sub a b iha ihb => intro h; exact ⟨iha h.1, ihb h.2⟩
  | mul a b iha ihb => intro h; exact ⟨iha h.1, ihb h.2⟩
  | scale cc a iha => intro h; exact iha h
  | exp a iha => intro h; exact iha h
  | log a _ => intro h; exact h.elim
  | relu a iha => intro h; exact iha h
  | max a b iha ihb => intro h; exact ⟨iha h.1, ihb h.2⟩
  | min a b iha ihb => intro h; exact ⟨iha h.1, ihb h.2⟩

/-! ### The log-inclusive forward budget -/

/-- **Log-inclusive forward budget**: C68's `fwdBound` extended at `log` with C51's floored
    bound `max |log c| |log U|·(1+logEps)`, the ceiling `U` being the argument's own recursive
    budget. All other cases are verbatim C68. -/
noncomputable def fwdBoundL (B c : ℝ) : Expr → ℝ
  | .var _ => B
  | .const cc => |toReal cc|
  | .add a b => (1 + u64) * (fwdBoundL B c a + fwdBoundL B c b)
  | .sub a b => (1 + u64) * (fwdBoundL B c a + fwdBoundL B c b)
  | .mul a b => (1 + u64) * (fwdBoundL B c a * fwdBoundL B c b)
  | .scale cc a => (1 + u64) * (|toReal cc| * fwdBoundL B c a)
  | .exp a => Real.exp (fwdBoundL B c a) * (1 + expEps)
  | .log a => max |Real.log c| |Real.log (fwdBoundL B c a)| * (1 + logEps)
  | .relu a => fwdBoundL B c a
  | .max a b => max (fwdBoundL B c a) (fwdBoundL B c b)
  | .min a b => max (fwdBoundL B c a) (fwdBoundL B c b)

/-- On the log-free sub-grammar the log-inclusive budget coincides with C68's (conservative
    extension). -/
theorem fwdBoundL_eq_fwdBound (B c : ℝ) :
    ∀ e, LogFree e → fwdBoundL B c e = fwdBound B e := by
  intro e
  induction e with
  | var i => intro _; rfl
  | const cc => intro _; rfl
  | add a b iha ihb => intro h; simp only [fwdBoundL, fwdBound, iha h.1, ihb h.2]
  | sub a b iha ihb => intro h; simp only [fwdBoundL, fwdBound, iha h.1, ihb h.2]
  | mul a b iha ihb => intro h; simp only [fwdBoundL, fwdBound, iha h.1, ihb h.2]
  | scale cc a iha => intro h; simp only [fwdBoundL, fwdBound, iha h]
  | exp a iha => intro h; simp only [fwdBoundL, fwdBound, iha h]
  | log a _ => intro h; exact h.elim
  | relu a iha => intro h; simp only [fwdBoundL, fwdBound, iha h]
  | max a b iha ihb => intro h; simp only [fwdBoundL, fwdBound, iha h.1, ihb h.2]
  | min a b iha ihb => intro h; simp only [fwdBoundL, fwdBound, iha h.1, ihb h.2]

/-! ### The full-grammar forward magnitude invariant -/

/-- **Forward magnitude over the actual `evalF`, FULL grammar.** With every input
    `|toReal (σ i)| ≤ B` and every `log` argument floored at `c > 0` (`LogFloored`), the Float
    forward value is bounded by the log-inclusive recursive budget. Extends C68's `evalF_mag`
    with the log case via C51's `logMag_le` (the argument's budget as the ceiling). -/
theorem evalFL_mag (σ : Nat → Float) (B c : ℝ) (hc : 0 < c)
    (hσ : ∀ i, |toReal (σ i)| ≤ B) :
    ∀ (e : Expr), LogFloored σ c e → |toReal (evalF e σ)| ≤ fwdBoundL B c e := by
  intro e
  induction e with
  | var i => intro _; simpa only [evalF, fwdBoundL] using hσ i
  | const cc => intro _; simp only [evalF, fwdBoundL]; exact le_refl _
  | add a b iha ihb =>
      intro hlf
      simp only [evalF, fwdBoundL]
      exact add_bound _ _ _ _ (iha hlf.1) (ihb hlf.2)
  | sub a b iha ihb =>
      intro hlf
      simp only [evalF, fwdBoundL]
      exact sub_bound _ _ _ _ (iha hlf.1) (ihb hlf.2)
  | mul a b iha ihb =>
      intro hlf
      simp only [evalF, fwdBoundL]
      exact mul_bound _ _ _ _ (iha hlf.1) (ihb hlf.2)
  | scale cc a iha =>
      intro hlf
      simp only [evalF, fwdBoundL]
      exact mul_bound _ _ _ _ (le_refl |toReal cc|) (iha hlf)
  | exp a iha =>
      intro hlf
      simp only [evalF, fwdBoundL]
      exact expF_mag_le _ _ ((le_abs_self _).trans (iha hlf))
  | log a iha =>
      intro hlf
      simp only [evalF, fwdBoundL]
      exact logMag_le _ c _ hc hlf.2 ((le_abs_self _).trans (iha hlf.1))
  | relu a iha =>
      intro hlf
      simp only [evalF, fwdBoundL]
      rw [toReal_reluF]
      have h : |max (toReal (evalF a σ)) 0| ≤ |toReal (evalF a σ)| := by
        rcases le_total (toReal (evalF a σ)) 0 with h | h
        · rw [max_eq_right h, abs_zero]; exact abs_nonneg _
        · rw [max_eq_left h]
      exact h.trans (iha hlf)
  | max a b iha ihb =>
      intro hlf
      simp only [evalF, fwdBoundL]
      exact (max_mag_le _ _).trans (max_le_max (iha hlf.1) (ihb hlf.2))
  | min a b iha ihb =>
      intro hlf
      simp only [evalF, fwdBoundL]
      exact (min_mag_le _ _).trans (max_le_max (iha hlf.1) (ihb hlf.2))

/-- **Forward finiteness over the actual `evalF`, FULL grammar**: bounded inputs + floored logs
    + the budget under `overflowBound` ⟹ the Float forward value is overflow-free. -/
theorem evalFL_isFinite (σ : Nat → Float) (B c : ℝ) (hc : 0 < c)
    (hσ : ∀ i, |toReal (σ i)| ≤ B) (e : Expr) (hlf : LogFloored σ c e)
    (hbound : fwdBoundL B c e ≤ overflowBound) :
    (evalF e σ).isFinite = true :=
  isFinite_of_bounded _ ((evalFL_mag σ B c hc hσ e hlf).trans hbound)

/-! ### The log node's backward edge weight, floored -/

/-- **The log backward edge weight is bounded under the floor.** The `comp` compiler records the
    log node's single edge weight as the Float `1.0 / x` (`ADReverse.comp`, `.log` case; `x` the
    argument's tape value = its `evalF`). Under the floor `0 < c ≤ toReal x`, its magnitude is
    `≤ (1/c)·(1+u64)` — via the PRE-EXISTING `div_model` + the exact `toReal_oneLit`. This is the
    missing log entry for C68's edge-weight budget `D`. -/
theorem log_edge_weight_mag (x : Float) (c : ℝ) (hc : 0 < c) (hx : c ≤ toReal x) :
    |toReal ((1.0 : Float) / x)| ≤ (1 / c) * (1 + u64) := by
  obtain ⟨δ, hδ, he⟩ := div_model 1.0 x
  rw [he, toReal_oneLit, abs_mul]
  have hxpos : 0 < toReal x := lt_of_lt_of_le hc hx
  have h1 : |1 / toReal x| = 1 / toReal x := abs_of_pos (by positivity)
  have h2 : (1 : ℝ) / toReal x ≤ 1 / c := one_div_le_one_div_of_le hc hx
  have h3 : |1 + δ| ≤ 1 + u64 := (abs_add_le _ _).trans (by simpa using add_le_add_left hδ 1)
  have hcpos : (0 : ℝ) < 1 / c := by positivity
  rw [h1]
  exact mul_le_mul h2 h3 (abs_nonneg _) hcpos.le

/-- **One floored log node, both halves bounded** — the value (via C51's `logMag_le`, ceiling
    `U`) AND the backward edge weight (via `log_edge_weight_mag`). This is the per-node interface
    for discharging C68's tape hypotheses (`hDw` with `D ≥ (1/c)·(1+u64)` on the log edges) for
    log-inclusive `comp`-built tapes; the per-tape value-readout glue C68 disclosed is the
    remaining (unchanged) composition step. -/
theorem log_node_bounds (x : Float) (c U : ℝ) (hc : 0 < c) (hx : c ≤ toReal x)
    (hU : toReal x ≤ U) :
    |toReal (Float.log x)| ≤ max |Real.log c| |Real.log U| * (1 + logEps)
      ∧ |toReal ((1.0 : Float) / x)| ≤ (1 / c) * (1 + u64) :=
  ⟨logMag_le x c U hc hx hU, log_edge_weight_mag x c hc hx⟩

end Puffer.RL.LogTapeFinite
