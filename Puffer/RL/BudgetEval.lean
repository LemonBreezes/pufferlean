/-
# Runnable budget evaluators with PROVEN domination — making the ℝ-side budget checks executable

The finiteness certificates consume ℝ-side budget hypotheses — C43's `dotBound (min |xs| |ws|) B ≤
overflowBound`, C64's `runBound C n B₀ ≤ overflowBound` — but those budgets are `noncomputable`
(they mention `u64 : ℝ`), so a runtime harness cannot evaluate them. The repo's `ErrBnd.lean`
pattern evaluates proven formulas in `Float` but leaves the Float↔ℝ gap to the external harness;
this module CLOSES that gap with proven DOMINATION: an upward-slacked computable Float evaluator
whose `toReal` provably dominates the exact ℝ budget, so a runtime Float comparison (C70's
`checkLe`) against a proven-safe threshold certifies the ℝ-side hypothesis outright.

**The slack design (the crux, proved once).** Each ℝ budget step carries a `(1+u64)` factor; the
Float evaluation of the same step rounds each op by `(1±u64)` — possibly DOWN. Replacing the
ℝ-side `(1+u64)` by a Float constant `slackF = 1.001` with the PROVEN inflation
`slackF_key : (1+u64) ≤ toReal slackF · (1−u64)²` makes every evaluator step dominate its ℝ
counterpart: one `slackF` factor absorbs the ℝ-side `(1+u64)` AND two Float roundings — exactly
the per-step rounding count of both recursions below (`·`/`+` inside, `·` outside). `slackF`'s
`toReal` is pinned by the pre-existing `toReal_ofScientific_close` (decimal literals round to
nearest), so NO new axiom.

* `dotBoundF` — the computable evaluator of C43's `dotBound`, with **`dotBoundF_dominates`**:
  `0 ≤ B ≤ toReal Bf → dotBound n B ≤ toReal (dotBoundF Bf n)`.
* `stepBoundF`/`runBoundF` — the computable evaluators of C64's `stepBound`/`runBound`, with
  **`runBoundF_dominates`** (the induction generalizes over the running budget, so no
  monotonicity lemma is needed).
* `capF = 1e300` — a proven-safe Float threshold: **`capF_le : toReal capF ≤ overflowBound`**
  (via `toReal_ofScientific_close` + `2·10³⁰⁰ ≤ 2¹⁰²³`).
* **`dotF_isFinite_runnable`** (capstone) — the first finiteness certificate whose EVERY
  hypothesis is a runtime `Bool`: `checkRegion xs Bf` ∧ `checkRegion ws Bf` ∧ `checkLe 0.0 Bf` ∧
  `checkLe (dotBoundF Bf n) capF` all `true` ⟹ `(dotF xs ws).isFinite = true`.
* `runBound_le_overflow_of_check` / `wdUniformBound_le_overflow_of_check` — the same runnable
  discharge for C64's whole-run and C67's horizon-free budget hypothesis shapes (each from one
  `checkLe`), via `runBoundF_dominates` / `wdUniformBoundF_dominates`.

**Scope (honestly disclosed).** The evaluators are UPWARD-SLACKED — conservative: they may
reject configurations the ℝ budgets would accept (sound-not-complete, C70's philosophy; the
slack is ≈0.1% per step, negligible against `overflowBound ≈ 1.8·10³⁰⁸`). Evaluators are
provided for `dotBound`, `stepBound`/`runBound`, AND C67's `wdUniformBound` (the division's
domination handles the denominator's adverse rounding — a Float denominator LARGER than `1−ρ`
shrinks the quotient — via the same `slackF_key`). C68's `sweepBound` (the two-level
`edgeBound`/`nodeBound` recursion — the same slack pattern applies, more plumbing) remains
ℝ-side, dischargeable by the same design. For C64/C67's concrete runs the step constants
(e.g. `sgdStepC`, `wdStepRho`/`wdStepC`) also need dominating Float representatives — the
domination theorems take them as `C ≤ toReal Cf`-style hypotheses (the caller's constant
plumbing). NO new axiom: comparisons via C70's derived `le_of_float_le`, literals via the
pre-existing `toReal_ofScientific_close`.
-/
import Puffer.RL.WholeRunFinite
import Puffer.RL.WdRunFinite
import Puffer.RL.MarginCheck
open Puffer.FloatR
open Puffer.RL.FiniteBound (dotBound overflowBound dotF_isFinite)
open Puffer.RL.ForwardFinite (dotBound_nonneg)
open Puffer.RL.WholeRunFinite (stepBound runBound)
open Puffer.RL.WdRunFinite (wdUniformBound)
open Puffer.RL.MarginCheck (checkLe checkLe_sound checkRegion checkRegion_sound)

namespace Puffer.RL.BudgetEval

/-! ### The slack constant -/

/-- The per-step inflation constant, as a Float literal. Its `toReal` provably dominates
    `(1+u64)/(1−u64)²` (see `slackF_key`), so one `slackF` factor absorbs the ℝ budget's
    `(1+u64)` plus two Float roundings. -/
def slackF : Float := 1.001

/-- `toReal slackF ≥ 1.001·(1−u64)` — the literal's rounding gap, via the pre-existing
    `toReal_ofScientific_close` (`1.001 = ofScientific 1001 true 3`). -/
theorem slackF_lower : (1.001 : ℝ) * (1 - u64) ≤ toReal slackF := by
  have h := toReal_ofScientific_close 1001 true 3
  have habs : |(1.001 : ℝ)| = (1.001 : ℝ) := abs_of_pos (by norm_num)
  rw [habs] at h
  have h1 := (abs_le.mp h).1
  unfold slackF
  nlinarith [h1]

/-- `toReal slackF ≥ 0`. -/
theorem slackF_nonneg : 0 ≤ toReal slackF := by
  have h := slackF_lower
  nlinarith [u64_lt_one, u64_pos]

/-- **THE SLACK KEY** (proved once, reused by every evaluator step):
    `(1+u64) ≤ toReal slackF · (1−u64)²` — one `slackF` factor covers the ℝ-side `(1+u64)`
    and two downward Float roundings. Numerically: `1.001·(1−u64)³ ≥ 1+u64` since `u64 = 2⁻⁵³`. -/
theorem slackF_key : (1 + u64) ≤ toReal slackF * ((1 - u64) * (1 - u64)) := by
  have h := slackF_lower
  have hsq : (0 : ℝ) ≤ (1 - u64) * (1 - u64) :=
    mul_nonneg (by linarith [u64_lt_one]) (by linarith [u64_lt_one])
  have step : (1.001 : ℝ) * (1 - u64) * ((1 - u64) * (1 - u64))
      ≤ toReal slackF * ((1 - u64) * (1 - u64)) := mul_le_mul_of_nonneg_right h hsq
  refine le_trans ?_ step
  have hu : u64 ≤ (1 : ℝ) / 1000000 := by unfold u64; norm_num
  nlinarith [u64_pos, hu, sq_nonneg u64, mul_nonneg u64_pos.le u64_pos.le]

/-- `1 − u64 ≥ 0` (used throughout). -/
private theorem one_sub_u64_nonneg : (0 : ℝ) ≤ 1 - u64 := by linarith [u64_lt_one]

/-! ### The proven-safe threshold -/

/-- A Float threshold provably below the overflow bound: `1e300`. -/
def capF : Float := 1e300

set_option exponentiation.threshold 1100 in
/-- `toReal capF ≤ overflowBound` — the threshold is safe: the literal's `toReal` is within
    relative `u64` of `10³⁰⁰` (`toReal_ofScientific_close`), and `2·10³⁰⁰ ≤ 2¹⁰²³ ≤ overflowBound`. -/
theorem capF_le : toReal capF ≤ overflowBound := by
  have h := toReal_ofScientific_close 1 false 300
  have habs : |(1e300 : ℝ)| = (1e300 : ℝ) := abs_of_pos (by norm_num)
  rw [habs] at h
  have h2 := (abs_le.mp h).2
  have h1 : toReal capF ≤ (1e300 : ℝ) + u64 * (1e300 : ℝ) := by unfold capF; linarith
  have hmul : u64 * (1e300 : ℝ) ≤ 1 * (1e300 : ℝ) :=
    mul_le_mul_of_nonneg_right u64_lt_one.le (by norm_num)
  have h3 : (2 : ℝ) * (1e300 : ℝ) ≤ overflowBound := by
    unfold Puffer.RL.FiniteBound.overflowBound
    have hco : (1 : ℝ) ≤ (2 : ℝ) - (2 : ℝ) ^ (-52 : ℤ) := by
      have : (2 : ℝ) ^ (-52 : ℤ) ≤ 1 := by norm_num
      linarith
    calc (2 : ℝ) * (1e300 : ℝ) ≤ (2 : ℝ) ^ (1023 : ℕ) := by norm_num
      _ = 1 * (2 : ℝ) ^ (1023 : ℕ) := by ring
      _ ≤ ((2 : ℝ) - (2 : ℝ) ^ (-52 : ℤ)) * (2 : ℝ) ^ (1023 : ℕ) :=
          mul_le_mul_of_nonneg_right hco (by positivity)
  linarith

/-! ### The dot-product budget evaluator -/

/-- Computable Float evaluator of C43's `dotBound`, with `slackF` in place of `(1+u64)`:
    `dotBoundF Bf 0 = 0`; `dotBoundF Bf (n+1) = slackF·(slackF·Bf² + dotBoundF Bf n)`. -/
def dotBoundF (B : Float) : Nat → Float
  | 0 => 0.0
  | n + 1 => slackF * (slackF * (B * B) + dotBoundF B n)

/-- **DOMINATION** (the payoff): the Float evaluator's `toReal` dominates the exact ℝ budget —
    `0 ≤ B ≤ toReal Bf ⟹ dotBound n B ≤ toReal (dotBoundF Bf n)`. Induction; each step spends
    `slackF_key` twice (once absorbing the inner `(1+u64)·B²`'s roundings, once the outer's). -/
theorem dotBoundF_dominates (Bf : Float) (B : ℝ) (hB0 : 0 ≤ B) (hBb : B ≤ toReal Bf) :
    ∀ n, dotBound n B ≤ toReal (dotBoundF Bf n)
  | 0 => by simp [dotBound, dotBoundF]
  | n + 1 => by
      have IH := dotBoundF_dominates Bf B hB0 hBb n
      have hPr : 0 ≤ dotBound n B := dotBound_nonneg B n
      have hP : 0 ≤ toReal (dotBoundF Bf n) := le_trans hPr IH
      have hs := slackF_nonneg
      have hu1 := one_sub_u64_nonneg
      obtain ⟨δ₁, hδ₁, e₁⟩ := mul_model Bf Bf
      obtain ⟨δ₂, hδ₂, e₂⟩ := mul_model slackF (Bf * Bf)
      obtain ⟨δ₃, hδ₃, e₃⟩ := add_model (slackF * (Bf * Bf)) (dotBoundF Bf n)
      obtain ⟨δ₄, hδ₄, e₄⟩ := mul_model slackF (slackF * (Bf * Bf) + dotBoundF Bf n)
      have hd₁ : 1 - u64 ≤ 1 + δ₁ := by have := (abs_le.mp hδ₁).1; linarith
      have hd₂ : 1 - u64 ≤ 1 + δ₂ := by have := (abs_le.mp hδ₂).1; linarith
      have hd₃ : 1 - u64 ≤ 1 + δ₃ := by have := (abs_le.mp hδ₃).1; linarith
      have hd₄ : 1 - u64 ≤ 1 + δ₄ := by have := (abs_le.mp hδ₄).1; linarith
      have hb0 : 0 ≤ toReal Bf := le_trans hB0 hBb
      have hBB : B * B ≤ toReal Bf * toReal Bf := mul_le_mul hBb hBb hB0 hb0
      have hbb0 : 0 ≤ toReal Bf * toReal Bf := mul_nonneg hb0 hb0
      -- inner square: B²·(1−u64) ≤ toReal (Bf*Bf), and it is nonneg
      have h₁ : (B * B) * (1 - u64) ≤ toReal (Bf * Bf) := by
        rw [e₁]
        calc (B * B) * (1 - u64) ≤ (toReal Bf * toReal Bf) * (1 - u64) :=
              mul_le_mul_of_nonneg_right hBB hu1
          _ ≤ (toReal Bf * toReal Bf) * (1 + δ₁) := mul_le_mul_of_nonneg_left hd₁ hbb0
      have h₁0 : 0 ≤ toReal (Bf * Bf) := by
        rw [e₁]; exact mul_nonneg hbb0 (by linarith)
      -- slack absorbs the inner (1+u64): (1+u64)·B² ≤ toReal (slackF·(Bf*Bf))
      have h₂ : (1 + u64) * (B * B) ≤ toReal (slackF * (Bf * Bf)) := by
        rw [e₂]
        calc (1 + u64) * (B * B)
            ≤ (toReal slackF * ((1 - u64) * (1 - u64))) * (B * B) :=
              mul_le_mul_of_nonneg_right slackF_key (mul_nonneg hB0 hB0)
          _ = (toReal slackF * ((B * B) * (1 - u64))) * (1 - u64) := by ring
          _ ≤ (toReal slackF * toReal (Bf * Bf)) * (1 - u64) :=
              mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left h₁ hs) hu1
          _ ≤ (toReal slackF * toReal (Bf * Bf)) * (1 + δ₂) :=
              mul_le_mul_of_nonneg_left hd₂ (mul_nonneg hs h₁0)
      have h₂0 : 0 ≤ toReal (slackF * (Bf * Bf)) := by
        rw [e₂]; exact mul_nonneg (mul_nonneg hs h₁0) (by linarith)
      -- the sum survives its rounding
      have hsum : (1 + u64) * (B * B) + dotBound n B
          ≤ toReal (slackF * (Bf * Bf)) + toReal (dotBoundF Bf n) := add_le_add h₂ IH
      have hsum0 : 0 ≤ toReal (slackF * (Bf * Bf)) + toReal (dotBoundF Bf n) :=
        add_nonneg h₂0 hP
      have hsum0' : 0 ≤ (1 + u64) * (B * B) + dotBound n B :=
        add_nonneg (mul_nonneg (by linarith [u64_pos]) (mul_nonneg hB0 hB0)) hPr
      have h₃ : ((1 + u64) * (B * B) + dotBound n B) * (1 - u64)
          ≤ toReal (slackF * (Bf * Bf) + dotBoundF Bf n) := by
        rw [e₃]
        calc ((1 + u64) * (B * B) + dotBound n B) * (1 - u64)
            ≤ (toReal (slackF * (Bf * Bf)) + toReal (dotBoundF Bf n)) * (1 - u64) :=
              mul_le_mul_of_nonneg_right hsum hu1
          _ ≤ (toReal (slackF * (Bf * Bf)) + toReal (dotBoundF Bf n)) * (1 + δ₃) :=
              mul_le_mul_of_nonneg_left hd₃ hsum0
      have h₃0 : 0 ≤ toReal (slackF * (Bf * Bf) + dotBoundF Bf n) := by
        rw [e₃]; exact mul_nonneg hsum0 (by linarith)
      -- the outer step: slack absorbs the outer (1+u64)
      show dotBound (n + 1) B ≤ toReal (dotBoundF Bf (n + 1))
      have hR : dotBound (n + 1) B = (1 + u64) * ((1 + u64) * (B * B) + dotBound n B) := rfl
      have hF : dotBoundF Bf (n + 1) = slackF * (slackF * (Bf * Bf) + dotBoundF Bf n) := rfl
      rw [hR, hF, e₄]
      calc (1 + u64) * ((1 + u64) * (B * B) + dotBound n B)
          ≤ (toReal slackF * ((1 - u64) * (1 - u64)))
              * ((1 + u64) * (B * B) + dotBound n B) :=
            mul_le_mul_of_nonneg_right slackF_key hsum0'
        _ = (toReal slackF * (((1 + u64) * (B * B) + dotBound n B) * (1 - u64))) * (1 - u64) := by
            ring
        _ ≤ (toReal slackF * toReal (slackF * (Bf * Bf) + dotBoundF Bf n)) * (1 - u64) :=
            mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left h₃ hs) hu1
        _ ≤ (toReal slackF * toReal (slackF * (Bf * Bf) + dotBoundF Bf n)) * (1 + δ₄) :=
            mul_le_mul_of_nonneg_left hd₄ (mul_nonneg hs h₃0)

/-! ### The whole-run budget evaluator (C64's `stepBound`/`runBound`) -/

/-- Computable evaluator of C64's `stepBound C B = (1+u64)·(B+C)`, with `slackF` as the factor. -/
def stepBoundF (C B : Float) : Float := slackF * (B + C)

/-- Computable evaluator of C64's `runBound` (the n-fold iterate, inside-first as the original). -/
def runBoundF (C : Float) : Nat → Float → Float
  | 0, B => B
  | n + 1, B => runBoundF C n (stepBoundF C B)

/-- One evaluator step dominates one ℝ budget step (the same two-rounding slack spend). -/
theorem stepBoundF_dominates (Cf Bf : Float) (C B : ℝ) (hC0 : 0 ≤ C) (hCc : C ≤ toReal Cf)
    (hB0 : 0 ≤ B) (hBb : B ≤ toReal Bf) :
    stepBound C B ≤ toReal (stepBoundF Cf Bf) := by
  obtain ⟨δ₁, hδ₁, e₁⟩ := add_model Bf Cf
  obtain ⟨δ₂, hδ₂, e₂⟩ := mul_model slackF (Bf + Cf)
  have hd₁ : 1 - u64 ≤ 1 + δ₁ := by have := (abs_le.mp hδ₁).1; linarith
  have hd₂ : 1 - u64 ≤ 1 + δ₂ := by have := (abs_le.mp hδ₂).1; linarith
  have hs := slackF_nonneg
  have hu1 := one_sub_u64_nonneg
  have hbc : B + C ≤ toReal Bf + toReal Cf := add_le_add hBb hCc
  have hbc0 : 0 ≤ toReal Bf + toReal Cf :=
    add_nonneg (le_trans hB0 hBb) (le_trans hC0 hCc)
  have h₁ : (B + C) * (1 - u64) ≤ toReal (Bf + Cf) := by
    rw [e₁]
    calc (B + C) * (1 - u64) ≤ (toReal Bf + toReal Cf) * (1 - u64) :=
          mul_le_mul_of_nonneg_right hbc hu1
      _ ≤ (toReal Bf + toReal Cf) * (1 + δ₁) := mul_le_mul_of_nonneg_left hd₁ hbc0
  have h₁0 : 0 ≤ toReal (Bf + Cf) := by
    rw [e₁]; exact mul_nonneg hbc0 (by linarith)
  show (1 + u64) * (B + C) ≤ toReal (stepBoundF Cf Bf)
  have hF : stepBoundF Cf Bf = slackF * (Bf + Cf) := rfl
  rw [hF, e₂]
  calc (1 + u64) * (B + C)
      ≤ (toReal slackF * ((1 - u64) * (1 - u64))) * (B + C) :=
        mul_le_mul_of_nonneg_right slackF_key (add_nonneg hB0 hC0)
    _ = (toReal slackF * ((B + C) * (1 - u64))) * (1 - u64) := by ring
    _ ≤ (toReal slackF * toReal (Bf + Cf)) * (1 - u64) :=
        mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left h₁ hs) hu1
    _ ≤ (toReal slackF * toReal (Bf + Cf)) * (1 + δ₂) :=
        mul_le_mul_of_nonneg_left hd₂ (mul_nonneg hs h₁0)

/-- **DOMINATION for the whole-run budget**: `runBound C n B ≤ toReal (runBoundF Cf n Bf)` given
    the constants' domination. The induction generalizes over the running budget, so no
    `runBound` monotonicity is needed: each step passes the dominated Float budget as the new seed. -/
theorem runBoundF_dominates (Cf : Float) (C : ℝ) (hC0 : 0 ≤ C) (hCc : C ≤ toReal Cf) :
    ∀ (n : Nat) (Bf : Float) (B : ℝ), 0 ≤ B → B ≤ toReal Bf →
      runBound C n B ≤ toReal (runBoundF Cf n Bf)
  | 0, _, _, _, hBb => hBb
  | n + 1, Bf, B, hB0, hBb => by
      have hstep := stepBoundF_dominates Cf Bf C B hC0 hCc hB0 hBb
      have hstep0 : 0 ≤ stepBound C B := by
        unfold Puffer.RL.WholeRunFinite.stepBound
        exact mul_nonneg (by linarith [u64_pos]) (add_nonneg hB0 hC0)
      show runBound C n (stepBound C B) ≤ toReal (runBoundF Cf n (stepBoundF Cf Bf))
      exact runBoundF_dominates Cf C hC0 hCc n (stepBoundF Cf Bf) (stepBound C B) hstep0 hstep

/-! ### The runnable certificates -/

/-- A passing budget check certifies the ℝ-side budget hypothesis: `checkLe (dotBoundF Bf n) capF
    = true ⟹ dotBound n B ≤ overflowBound` (domination → Float comparison → safe threshold). -/
theorem dotBound_le_overflow_of_check (Bf : Float) (B : ℝ) (n : Nat)
    (hB0 : 0 ≤ B) (hBb : B ≤ toReal Bf)
    (hchk : checkLe (dotBoundF Bf n) capF = true) :
    dotBound n B ≤ overflowBound :=
  ((dotBoundF_dominates Bf B hB0 hBb n).trans (checkLe_sound hchk)).trans capF_le

/-- The C64-shaped whole-run discharge: one `checkLe` on the evaluated run budget certifies
    `runBound C n B ≤ overflowBound` — the exact hypothesis `trainRun_all_finite` consumes
    (the constants' domination `C ≤ toReal Cf`, `B ≤ toReal Bf` is the caller's plumbing). -/
theorem runBound_le_overflow_of_check (Cf Bf : Float) (C B : ℝ) (n : Nat)
    (hC0 : 0 ≤ C) (hCc : C ≤ toReal Cf) (hB0 : 0 ≤ B) (hBb : B ≤ toReal Bf)
    (hchk : checkLe (runBoundF Cf n Bf) capF = true) :
    runBound C n B ≤ overflowBound :=
  ((runBoundF_dominates Cf C hC0 hCc n Bf B hB0 hBb).trans (checkLe_sound hchk)).trans capF_le

/-- **THE RUNNABLE FINITENESS CERTIFICATE** (capstone): every hypothesis is a runtime `Bool`.
    If the input rows pass the region check at `Bf`, `0.0 ≤ Bf` passes, and the evaluated dot
    budget passes the threshold check, then the Float dot product is overflow-free. The first
    certificate in the chain whose ENTIRE hypothesis set is computed by the harness at runtime
    (the threshold's safety `capF_le` and the domination are proved offline, once). -/
theorem dotF_isFinite_runnable (xs ws : List Float) (Bf : Float)
    (hxs : checkRegion xs Bf = true) (hws : checkRegion ws Bf = true)
    (hB0 : checkLe 0.0 Bf = true)
    (hbudget : checkLe (dotBoundF Bf (min xs.length ws.length)) capF = true) :
    (Puffer.RL.FiniteBound.dotF xs ws).isFinite = true := by
  have hB0' : (0 : ℝ) ≤ toReal Bf := by
    have := checkLe_sound hB0
    rwa [toReal_zeroLit] at this
  exact dotF_isFinite (toReal Bf) hB0' xs ws (checkRegion_sound hxs) (checkRegion_sound hws)
    (dotBound_le_overflow_of_check Bf (toReal Bf) _ hB0' le_rfl hbudget)

/-! ### The horizon-free weight-decay budget evaluator (C67's `wdUniformBound`) -/

/-- Computable evaluator of C67's `wdUniformBound ρ C B = B + C/(1−ρ)`. The denominator's and
    the division's roundings are absorbed by an inner `slackF` on the quotient; the final add's
    rounding by an outer `slackF` on the sum. -/
def wdUniformBoundF (ρ C B : Float) : Float := slackF * (B + slackF * (C / (1.0 - ρ)))

/-- **DOMINATION for the horizon-free budget**: `wdUniformBound ρ C B ≤ toReal (wdUniformBoundF
    ρf Cf Bf)` given the constants' domination (`ρ ≤ toReal ρf < 1`, `C ≤ toReal Cf`,
    `B ≤ toReal Bf`). The denominator computed in Float can be LARGER than `1−ρ` (shrinking the
    quotient); the inner `slackF` covers that `(1+u64)` plus the division's own rounding via
    `slackF_key`, and the outer `slackF` covers the sum's rounding. -/
theorem wdUniformBoundF_dominates (ρf Cf Bf : Float) (ρ C B : ℝ)
    (hρr : ρ ≤ toReal ρf) (hr1 : toReal ρf < 1)
    (hC0 : 0 ≤ C) (hCc : C ≤ toReal Cf) (hB0 : 0 ≤ B) (hBb : B ≤ toReal Bf) :
    wdUniformBound ρ C B ≤ toReal (wdUniformBoundF ρf Cf Bf) := by
  obtain ⟨δ₁, hδ₁, e₁⟩ := sub_model 1.0 ρf
  obtain ⟨δ₂, hδ₂, e₂⟩ := div_model Cf (1.0 - ρf)
  obtain ⟨δ₃, hδ₃, e₃⟩ := mul_model slackF (Cf / (1.0 - ρf))
  obtain ⟨δ₄, hδ₄, e₄⟩ := add_model Bf (slackF * (Cf / (1.0 - ρf)))
  obtain ⟨δ₅, hδ₅, e₅⟩ := mul_model slackF (Bf + slackF * (Cf / (1.0 - ρf)))
  have hd₁' : 1 + δ₁ ≤ 1 + u64 := by have := (abs_le.mp hδ₁).2; linarith
  have hd₂ : 1 - u64 ≤ 1 + δ₂ := by have := (abs_le.mp hδ₂).1; linarith
  have hd₃ : 1 - u64 ≤ 1 + δ₃ := by have := (abs_le.mp hδ₃).1; linarith
  have hd₄ : 1 - u64 ≤ 1 + δ₄ := by have := (abs_le.mp hδ₄).1; linarith
  have hd₅ : 1 - u64 ≤ 1 + δ₅ := by have := (abs_le.mp hδ₅).1; linarith
  have h1r : (0 : ℝ) < 1 - toReal ρf := by linarith
  have h1ρ : (0 : ℝ) < 1 - ρ := by linarith [le_trans hρr hr1.le]
  have hc0 : 0 ≤ toReal Cf := le_trans hC0 hCc
  have hb0' : 0 ≤ toReal Bf := le_trans hB0 hBb
  -- the Float denominator: positive, and at most (1−r)(1+u64)
  have ed : toReal (1.0 - ρf) = (1 - toReal ρf) * (1 + δ₁) := by rw [e₁, toReal_oneLit]
  have hdpos : 0 < toReal (1.0 - ρf) := by
    rw [ed]
    have : (0 : ℝ) < 1 + δ₁ := by have := (abs_le.mp hδ₁).1; linarith [u64_lt_one]
    exact mul_pos h1r this
  have hdle : toReal (1.0 - ρf) ≤ (1 - toReal ρf) * (1 + u64) := by
    rw [ed]; exact mul_le_mul_of_nonneg_left hd₁' h1r.le
  -- the quotient's real value is nonneg
  have hcd0 : 0 ≤ toReal Cf / toReal (1.0 - ρf) := div_nonneg hc0 hdpos.le
  have hQ0 : 0 ≤ toReal (Cf / (1.0 - ρf)) := by
    rw [e₂]
    exact mul_nonneg hcd0 (by have := (abs_le.mp hδ₂).1; linarith [u64_lt_one])
  -- key quotient inequality: C/(1−ρ) ≤ (1+u64)·(c/d)
  have hkeyq : C / (1 - ρ) ≤ (1 + u64) * (toReal Cf / toReal (1.0 - ρf)) := by
    have h1 : C * (1 - toReal ρf) ≤ toReal Cf * (1 - ρ) :=
      mul_le_mul hCc (by linarith) h1r.le hc0
    have hCd : C * toReal (1.0 - ρf) ≤ toReal Cf * (1 + u64) * (1 - ρ) := by
      calc C * toReal (1.0 - ρf)
          ≤ C * ((1 - toReal ρf) * (1 + u64)) := mul_le_mul_of_nonneg_left hdle hC0
        _ = (C * (1 - toReal ρf)) * (1 + u64) := by ring
        _ ≤ (toReal Cf * (1 - ρ)) * (1 + u64) :=
            mul_le_mul_of_nonneg_right h1 (by linarith [u64_pos])
        _ = toReal Cf * (1 + u64) * (1 - ρ) := by ring
    rw [div_le_iff₀ h1ρ]
    have hrw : (1 + u64) * (toReal Cf / toReal (1.0 - ρf)) * (1 - ρ)
        = (toReal Cf * (1 + u64) * (1 - ρ)) / toReal (1.0 - ρf) := by ring
    rw [hrw]
    exact (le_div_iff₀ hdpos).mpr hCd
  -- the inner slacked quotient dominates C/(1−ρ)
  have hQge : (toReal Cf / toReal (1.0 - ρf)) * (1 - u64) ≤ toReal (Cf / (1.0 - ρf)) := by
    rw [e₂]; exact mul_le_mul_of_nonneg_left hd₂ hcd0
  have hI : C / (1 - ρ) ≤ toReal (slackF * (Cf / (1.0 - ρf))) := by
    rw [e₃]
    calc C / (1 - ρ)
        ≤ (1 + u64) * (toReal Cf / toReal (1.0 - ρf)) := hkeyq
      _ ≤ (toReal slackF * ((1 - u64) * (1 - u64))) * (toReal Cf / toReal (1.0 - ρf)) :=
          mul_le_mul_of_nonneg_right slackF_key hcd0
      _ = (toReal slackF * ((toReal Cf / toReal (1.0 - ρf)) * (1 - u64))) * (1 - u64) := by ring
      _ ≤ (toReal slackF * toReal (Cf / (1.0 - ρf))) * (1 - u64) :=
          mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hQge slackF_nonneg)
            one_sub_u64_nonneg
      _ ≤ (toReal slackF * toReal (Cf / (1.0 - ρf))) * (1 + δ₃) :=
          mul_le_mul_of_nonneg_left hd₃ (mul_nonneg slackF_nonneg hQ0)
  have hI0 : 0 ≤ toReal (slackF * (Cf / (1.0 - ρf))) := by
    rw [e₃]
    exact mul_nonneg (mul_nonneg slackF_nonneg hQ0)
      (by have := (abs_le.mp hδ₃).1; linarith [u64_lt_one])
  -- the outer slacked sum
  show wdUniformBound ρ C B ≤ toReal (wdUniformBoundF ρf Cf Bf)
  have hRdef : wdUniformBound ρ C B = B + C / (1 - ρ) := rfl
  have hFdef : wdUniformBoundF ρf Cf Bf = slackF * (Bf + slackF * (Cf / (1.0 - ρf))) := rfl
  rw [hRdef, hFdef, e₅, e₄]
  have hsumI : B + C / (1 - ρ) ≤ toReal Bf + toReal (slackF * (Cf / (1.0 - ρf))) :=
    add_le_add hBb hI
  have hsumI0 : 0 ≤ toReal Bf + toReal (slackF * (Cf / (1.0 - ρf))) := add_nonneg hb0' hI0
  have hone : (1 : ℝ) ≤ toReal slackF * ((1 - u64) * (1 - u64)) := by
    have := slackF_key; linarith [u64_pos]
  calc B + C / (1 - ρ)
      ≤ toReal Bf + toReal (slackF * (Cf / (1.0 - ρf))) := hsumI
    _ ≤ (toReal slackF * ((1 - u64) * (1 - u64)))
          * (toReal Bf + toReal (slackF * (Cf / (1.0 - ρf)))) :=
        le_mul_of_one_le_left hsumI0 hone
    _ = (toReal slackF * ((toReal Bf + toReal (slackF * (Cf / (1.0 - ρf)))) * (1 - u64)))
          * (1 - u64) := by ring
    _ ≤ (toReal slackF * ((toReal Bf + toReal (slackF * (Cf / (1.0 - ρf)))) * (1 + δ₄)))
          * (1 - u64) :=
        mul_le_mul_of_nonneg_right
          (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hd₄ hsumI0) slackF_nonneg)
          one_sub_u64_nonneg
    _ ≤ (toReal slackF * ((toReal Bf + toReal (slackF * (Cf / (1.0 - ρf)))) * (1 + δ₄)))
          * (1 + δ₅) :=
        mul_le_mul_of_nonneg_left hd₅
          (mul_nonneg slackF_nonneg (mul_nonneg hsumI0
            (by have := (abs_le.mp hδ₄).1; linarith [u64_lt_one])))

/-- The C67-shaped horizon-free discharge: one `checkLe` on the evaluated uniform budget
    certifies `wdUniformBound ρ C B ≤ overflowBound` — the exact hypothesis
    `wdTrainRun_all_finite_uniform` consumes. -/
theorem wdUniformBound_le_overflow_of_check (ρf Cf Bf : Float) (ρ C B : ℝ)
    (hρr : ρ ≤ toReal ρf) (hr1 : toReal ρf < 1)
    (hC0 : 0 ≤ C) (hCc : C ≤ toReal Cf) (hB0 : 0 ≤ B) (hBb : B ≤ toReal Bf)
    (hchk : checkLe (wdUniformBoundF ρf Cf Bf) capF = true) :
    wdUniformBound ρ C B ≤ overflowBound :=
  ((wdUniformBoundF_dominates ρf Cf Bf ρ C B hρr hr1 hC0 hCc hB0 hBb).trans
    (checkLe_sound hchk)).trans capF_le

end Puffer.RL.BudgetEval
