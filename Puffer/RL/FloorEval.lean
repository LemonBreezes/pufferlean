/-
# The downward-slacked floor evaluator: C80's last plumbed constant eliminated

C80 (`SweepEval`) delivered the log-inclusive runnable tape certificate
`adGrad_isFinite_comp_partition_runnable` with ONE plumbed constant left: the hypothesis
`hDf : compWeightBoundL B (floorC (logArgBound B e)) e ≤ toReal Df` — because the log entry
`(1/c)·(1+u64)` of the weight budget needs a LOWER Float representative of the floor
`c = floorC (logArgBound B e)`, the disclosed DIRECTION REVERSAL: every other representative in
the chain is an upper one. This module builds the downward-slacked evaluator and eliminates `hDf`:

* **`shrinkF = 0.998`** — the downward slack constant, pinned in BOTH directions by
  `toReal_ofScientific_close` (`shrinkF_lower` for positivity, `shrinkF_upper` for the anti-key),
  with the **ANTI-KEY** `shrinkF_key :
  `toReal shrinkF · toReal shrinkF · ((1+expEps)·(1+u64)²) ≤ (1−expEps)·(1−u64)` — in the
  downward direction the UPWARD roundings are adverse, so two `shrinkF` factors absorb the exp
  model's `(1+expEps)` and two multiplications' `(1+u64)` while staying under `floorC`'s own
  `(1−expEps)(1−u64)` haircut.
* **`logArgBoundF`** — the computable UPPER mirror of C77's `logArgBound` (exact `max` spine,
  C80's `fwdBoundF` at the log nodes), with `logArgBoundF_dominates` for `PartitionLogs`
  expressions. `floorC` is antitone, so a lower floor comes from an UPPER argument-bound mirror.
* **`floorCF Mf = shrinkF·(shrinkF·Float.exp (−Mf))`** — the computable LOWER floor:
  `floorCF_lower : toReal (floorCF Mf) ≤ floorC M` whenever `M ≤ toReal Mf`, and
  `floorCF_pos : 0 < toReal (floorCF Mf)` — together exactly the `0 < toReal cF ≤ c` that makes
  `1/toReal cF` DOMINATE `1/c`.
* **`fwdBoundLF`/`compWeightBoundLF`** — the log-inclusive weight-budget mirror. C75's
  `compWeightBoundL` routes its `mul`/`exp` sibling budgets through the log-inclusive
  `fwdBoundL`, whose log entry `max |log c| |log U|·(1+logEps)` needs `|log c| ≤ M+1`
  (`abs_log_floorC_le` — `floorC`'s haircut costs less than one nat) and `|log U| ≤ U` (a
  partition-shaped log argument's budget is `≥ 1`, `one_le_fwdBound_addExp`); the mirror spends
  the third key `slackF_key_log` on the `(1+logEps)` factor. `compWeightBoundLF_dominates` is
  the full log-inclusive domination at the concrete floor — the exact statement `hDf` demanded.
* **`adGrad_isFinite_comp_partition_runnable'`** (capstone) — C80's partition certificate with
  `hDf` GONE: the weight budget `compWeightBoundLF Bf e` is COMPUTED by the harness from the
  expression, and the hypotheses are the three runtime Bools (`checkRegion`, `checkLe 0.0 Bf`,
  one sweep-budget `checkLe`) plus C73's representation plumbing and the `PartitionLogs` shape.
  THE FULLY-RUNNABLE PARTITION TAPE CERTIFICATE.

**Scope (honestly disclosed).** Downward slack is conservative in the same sound-not-complete
sense as C78/C80's upward slack: `floorCF` undershoots the true floor by ≈0.4%, inflating the
mirrored log edge budget accordingly — a runtime check may reject configurations the ℝ budget
accepts, never the converse. The floor genuinely underflows to `0.0` for large argument budgets
(`Float.exp (−M) = 0` beyond `M ≈ 745`), upon which `1.0/0.0 = inf` fails the budget check —
the evaluator REJECTS instead of certifying, still sound (see the negative-control demo). NO new
axiom: the anti-key and both pinnings come from `toReal_ofScientific_close` and the pre-existing
`(1+δ)` models; everything else composes C75/C77/C78/C80.
-/
import Puffer.RL.SweepEval

namespace Puffer.RL.FloorEval

open Puffer.FloatR (toReal u64 u64_pos u64_lt_one expEps expEps_pos logEps logEps_pos
  exp_model add_model mul_model div_model toReal_neg toReal_oneLit toReal_zeroLit
  toReal_ofScientific_close toReal_max)
open Puffer.FloatR.AD (Tape V grads)
open Puffer.FloatR.ADR (Expr)
open Puffer.FloatR.ADReverse (comp)
open Puffer.RL.ADTapeFinite (LogFree fwdBound)
open Puffer.RL.LogTapeFinite (fwdBoundL fwdBoundL_eq_fwdBound)
open Puffer.RL.CompTapeLog (compWeightBoundL compWeightBoundL_nonneg)
open Puffer.RL.CompTapeFloor (PartitionLogs NonnegShape nonnegShape_logFree logArgBound
  fwdBound_nonneg adGrad_isFinite_comp_partition)
open Puffer.RL.FloatFloor (floorC floorC_pos one_sub_expEps_pos one_sub_u64_pos)
open Puffer.RL.BudgetEval (slackF slackF_lower slackF_nonneg slackF_key capF)
open Puffer.RL.SweepEval (slack_step slack_exp_step expEps_le_small expEps_lt_one
  absF toReal_absF fwdBoundF fwdBoundF_dominates sweepBoundF sweepBound_le_overflow_of_check)
open Puffer.RL.MarginCheck (checkLe checkLe_sound checkRegion)
open Puffer.RL.TraceCheck (padRow padRow_abs_le)

/-! ### Small shared facts -/

private theorem one_sub_u64' : (0 : ℝ) ≤ 1 - u64 := by linarith [u64_lt_one]

private theorem u64_le_small : u64 ≤ (1 : ℝ) / 1000000 := by
  unfold Puffer.FloatR.u64; norm_num

/-- `logEps = 2⁻⁵² ≤ 10⁻⁶` (the log analogue of C80's `expEps_le_small`). -/
theorem logEps_le_small : logEps ≤ (1 : ℝ) / 1000000 := by
  unfold Puffer.FloatR.logEps; norm_num

/-! ### The downward slack constant and its anti-key -/

/-- **The downward slack constant** `0.998`, the shrink mirror of C78's `slackF = 1.001`: each
    downward-slacked evaluation step spends one `shrinkF` factor to absorb one adverse UPWARD
    rounding while staying under the ℝ-side floor. -/
def shrinkF : Float := 0.998

/-- `toReal shrinkF ≥ 0.998·(1−u64)` — the literal's rounding gap (positivity direction), via the
    pre-existing `toReal_ofScientific_close` (`0.998 = ofScientific 998 true 3`). -/
theorem shrinkF_lower : (0.998 : ℝ) * (1 - u64) ≤ toReal shrinkF := by
  have h := toReal_ofScientific_close 998 true 3
  have habs : |(0.998 : ℝ)| = (0.998 : ℝ) := abs_of_pos (by norm_num)
  rw [habs] at h
  have h1 := (abs_le.mp h).1
  unfold shrinkF
  nlinarith [h1]

/-- `toReal shrinkF ≤ 0.998·(1+u64)` — the OTHER direction of the pin (the anti-key needs the
    Float constant bounded ABOVE, the reversal of C78's `slackF_lower`). -/
theorem shrinkF_upper : toReal shrinkF ≤ (0.998 : ℝ) * (1 + u64) := by
  have h := toReal_ofScientific_close 998 true 3
  have habs : |(0.998 : ℝ)| = (0.998 : ℝ) := abs_of_pos (by norm_num)
  rw [habs] at h
  have h2 := (abs_le.mp h).2
  unfold shrinkF
  nlinarith [h2]

/-- `toReal shrinkF > 0` (from the lower pin; `0.998·(1−u64) > 0`). -/
theorem shrinkF_pos : 0 < toReal shrinkF := by
  have h := shrinkF_lower
  nlinarith [u64_lt_one]

/-- **THE ANTI-KEY** (the direction reversal of C78's `slackF_key`, proved once): two `shrinkF`
    factors absorb the exp model's adverse `(1+expEps)` and two multiplications' adverse
    `(1+u64)` while staying under `floorC`'s own `(1−expEps)(1−u64)` haircut. Numerically:
    `0.998² ≈ 0.996` against `≈ 1 − 4·10⁻¹⁶` — comfortable. -/
theorem shrinkF_key :
    toReal shrinkF * toReal shrinkF * ((1 + expEps) * ((1 + u64) * (1 + u64)))
      ≤ (1 - expEps) * (1 - u64) := by
  have hu := u64_le_small
  have he := expEps_le_small
  have hs999 : toReal shrinkF ≤ 0.999 := by nlinarith [shrinkF_upper, hu]
  have hs0 := shrinkF_pos.le
  have b1 : u64 * u64 ≤ (1 : ℝ) / 1000000 * (1 / 1000000) :=
    mul_le_mul hu hu u64_pos.le (by norm_num)
  have b2 : u64 * expEps ≤ (1 : ℝ) / 1000000 * (1 / 1000000) :=
    mul_le_mul hu he expEps_pos.le (by norm_num)
  have b3 : u64 * u64 * expEps ≤ (1 : ℝ) / 1000000 * (1 / 1000000) * (1 / 1000000) :=
    mul_le_mul b1 he expEps_pos.le (by norm_num)
  have hfac : (1 + expEps) * ((1 + u64) * (1 + u64)) ≤ 1 + 4 / 1000000 := by
    nlinarith [b1, b2, b3, hu, he, u64_pos.le, expEps_pos.le]
  have hfac0 : (0 : ℝ) ≤ (1 + expEps) * ((1 + u64) * (1 + u64)) := by
    have h1 : (0 : ℝ) ≤ 1 + expEps := by linarith [expEps_pos]
    have h2 : (0 : ℝ) ≤ 1 + u64 := by linarith [u64_pos]
    exact mul_nonneg h1 (mul_nonneg h2 h2)
  have hss : toReal shrinkF * toReal shrinkF ≤ 0.999 * 0.999 :=
    mul_le_mul hs999 hs999 hs0 (by norm_num)
  have hrhs : (1 : ℝ) - 2 / 1000000 ≤ (1 - expEps) * (1 - u64) := by
    nlinarith [he, hu, mul_nonneg expEps_pos.le u64_pos.le]
  calc toReal shrinkF * toReal shrinkF * ((1 + expEps) * ((1 + u64) * (1 + u64)))
      ≤ (0.999 * 0.999) * (1 + 4 / 1000000) :=
        mul_le_mul hss hfac hfac0 (by norm_num)
    _ ≤ (1 : ℝ) - 2 / 1000000 := by norm_num
    _ ≤ (1 - expEps) * (1 - u64) := hrhs

/-! ### The third slack key: the `(1+logEps)` factor -/

/-- **The log slack key** (the `logEps` sibling of C78's `slackF_key`): one `slackF` factor
    covers the ℝ-side `(1+logEps)` log-budget factor plus two adverse Float roundings. -/
theorem slackF_key_log : (1 + logEps) ≤ toReal slackF * ((1 - u64) * (1 - u64)) := by
  have h := slackF_lower
  have hsq : (0 : ℝ) ≤ (1 - u64) * (1 - u64) := mul_nonneg one_sub_u64' one_sub_u64'
  have step : (1.001 : ℝ) * (1 - u64) * ((1 - u64) * (1 - u64))
      ≤ toReal slackF * ((1 - u64) * (1 - u64)) := mul_le_mul_of_nonneg_right h hsq
  refine le_trans ?_ step
  have hu := u64_le_small
  have hl := logEps_le_small
  nlinarith [u64_pos, logEps_pos, hu, hl, sq_nonneg u64,
    mul_nonneg u64_pos.le u64_pos.le,
    mul_nonneg (mul_nonneg u64_pos.le u64_pos.le) u64_pos.le]

/-- The generic log slack step (C80's `slack_step` with `(1+logEps)` in place of `(1+u64)` on the
    ℝ side, via `slackF_key_log`): a dominated pre-rounding value survives one adverse rounding
    and the outer multiplication under one `slackF`. -/
theorem slack_step_log (opF : Float) (y Y : ℝ) (hy0 : 0 ≤ y) (hyY : y ≤ Y)
    (hop : ∃ δ : ℝ, |δ| ≤ u64 ∧ toReal opF = Y * (1 + δ)) :
    (1 + logEps) * y ≤ toReal (slackF * opF) := by
  obtain ⟨δ₁, hδ₁, e₁⟩ := hop
  obtain ⟨δ₂, hδ₂, e₂⟩ := mul_model slackF opF
  have hd₁ : 1 - u64 ≤ 1 + δ₁ := by have := (abs_le.mp hδ₁).1; linarith
  have hd₂ : 1 - u64 ≤ 1 + δ₂ := by have := (abs_le.mp hδ₂).1; linarith
  have hY0 : 0 ≤ Y := le_trans hy0 hyY
  have h1δ : (0 : ℝ) ≤ 1 + δ₁ := by
    have := (abs_le.mp hδ₁).1; linarith [u64_lt_one]
  have hop0 : 0 ≤ toReal opF := by rw [e₁]; exact mul_nonneg hY0 h1δ
  rw [e₂]
  calc (1 + logEps) * y
      ≤ (toReal slackF * ((1 - u64) * (1 - u64))) * y :=
        mul_le_mul_of_nonneg_right slackF_key_log hy0
    _ = (toReal slackF * (y * (1 - u64))) * (1 - u64) := by ring
    _ ≤ (toReal slackF * (Y * (1 + δ₁))) * (1 - u64) := by
        have h1 : y * (1 - u64) ≤ Y * (1 + δ₁) :=
          le_trans (mul_le_mul_of_nonneg_right hyY one_sub_u64')
            (mul_le_mul_of_nonneg_left hd₁ hY0)
        exact mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left h1 slackF_nonneg)
          one_sub_u64'
    _ = (toReal slackF * toReal opF) * (1 - u64) := by rw [← e₁]
    _ ≤ (toReal slackF * toReal opF) * (1 + δ₂) :=
        mul_le_mul_of_nonneg_left hd₂ (mul_nonneg slackF_nonneg hop0)

/-! ### The log-argument bound's upper mirror -/

/-- The log-argument bound is nonnegative (inputs bounded by a nonnegative `B`) — the base the
    slack accounting needs (`M ≥ 0` makes `floorC M ≤ 1` and `M + 1 ≥ 1`). -/
theorem logArgBound_nonneg (B : ℝ) (hB : 0 ≤ B) : ∀ e : Expr, 0 ≤ logArgBound B e := by
  intro e
  induction e with
  | var i => simp only [logArgBound]; exact le_refl 0
  | const c => simp only [logArgBound]; exact le_refl 0
  | add a b iha _ => simp only [logArgBound]; exact iha.trans (le_max_left _ _)
  | sub a b iha _ => simp only [logArgBound]; exact iha.trans (le_max_left _ _)
  | mul a b iha _ => simp only [logArgBound]; exact iha.trans (le_max_left _ _)
  | scale c a iha => simpa only [logArgBound] using iha
  | exp a iha => simpa only [logArgBound] using iha
  | log arg _ => simp only [logArgBound]; exact fwdBound_nonneg B hB arg
  | relu a iha => simpa only [logArgBound] using iha
  | max a b iha _ => simp only [logArgBound]; exact iha.trans (le_max_left _ _)
  | min a b iha _ => simp only [logArgBound]; exact iha.trans (le_max_left _ _)

/-- **The computable UPPER mirror of C77's `logArgBound`**: the same exact `max` spine
    (`toReal_max` — no rounding), with C80's upward-slacked `fwdBoundF` at the log nodes.
    `floorC` is ANTITONE, so the lower floor `floorCF` is fed by this UPPER mirror. -/
def logArgBoundF (B : Float) : Expr → Float
  | .var _ => 0.0
  | .const _ => 0.0
  | .add a b => max (logArgBoundF B a) (logArgBoundF B b)
  | .sub a b => max (logArgBoundF B a) (logArgBoundF B b)
  | .mul a b => max (logArgBoundF B a) (logArgBoundF B b)
  | .scale _ a => logArgBoundF B a
  | .exp a => logArgBoundF B a
  | .log arg => fwdBoundF B arg
  | .relu a => logArgBoundF B a
  | .max a b => max (logArgBoundF B a) (logArgBoundF B b)
  | .min a b => max (logArgBoundF B a) (logArgBoundF B b)

/-- **Upper domination of the log-argument bound** (for `PartitionLogs` expressions — each log
    argument is the partition shape, hence `LogFree`, and C80's `fwdBoundF_dominates` applies):
    `logArgBound B e ≤ toReal (logArgBoundF Bf e)`. The `max` spine transports exactly. -/
theorem logArgBoundF_dominates (Bf : Float) (B : ℝ) (hB0 : 0 ≤ B) (hBb : B ≤ toReal Bf) :
    ∀ e : Expr, PartitionLogs e → logArgBound B e ≤ toReal (logArgBoundF Bf e) := by
  intro e
  induction e with
  | var i => intro _; exact le_of_eq toReal_zeroLit.symm
  | const c => intro _; exact le_of_eq toReal_zeroLit.symm
  | add a b iha ihb =>
      intro hp
      simp only [logArgBound, logArgBoundF]
      rw [toReal_max]
      exact max_le_max (iha hp.1) (ihb hp.2)
  | sub a b iha ihb =>
      intro hp
      simp only [logArgBound, logArgBoundF]
      rw [toReal_max]
      exact max_le_max (iha hp.1) (ihb hp.2)
  | mul a b iha ihb =>
      intro hp
      simp only [logArgBound, logArgBoundF]
      rw [toReal_max]
      exact max_le_max (iha hp.1) (ihb hp.2)
  | scale c a iha => intro hp; simpa only [logArgBound, logArgBoundF] using iha hp
  | exp a iha => intro hp; simpa only [logArgBound, logArgBoundF] using iha hp
  | log arg _ =>
      intro hp
      obtain ⟨a, b, rfl, hlfa, hnb⟩ := hp
      simp only [logArgBound, logArgBoundF]
      exact fwdBoundF_dominates Bf B hB0 hBb _ ⟨hlfa, nonnegShape_logFree b hnb⟩
  | relu a iha => intro hp; simpa only [logArgBound, logArgBoundF] using iha hp
  | max a b iha ihb =>
      intro hp
      simp only [logArgBound, logArgBoundF]
      rw [toReal_max]
      exact max_le_max (iha hp.1) (ihb hp.2)
  | min a b iha ihb =>
      intro hp
      simp only [logArgBound, logArgBoundF]
      rw [toReal_max]
      exact max_le_max (iha hp.1) (ihb hp.2)

/-! ### The computable lower floor -/

/-- **The computable LOWER floor**: `shrinkF·(shrinkF·Float.exp (−Mf))` — the downward-slacked
    Float evaluation of `floorC`'s head `exp(−M)`, two `shrinkF` factors paying for the exp
    model's and the two multiplications' adverse upward roundings (the anti-key). -/
def floorCF (Mf : Float) : Float := shrinkF * (shrinkF * Float.exp (-Mf))

/-- The Float floor is strictly positive — the `0 < toReal cF` half of the log edge budget's
    domination interface (every `(1+δ)` factor is `> 0`, `shrinkF` is pinned positive). -/
theorem floorCF_pos (Mf : Float) : 0 < toReal (floorCF Mf) := by
  obtain ⟨δe, hδe, ee⟩ := exp_model (-Mf)
  obtain ⟨δ₂, hδ₂, e₂⟩ := mul_model shrinkF (Float.exp (-Mf))
  obtain ⟨δ₁, hδ₁, e₁⟩ := mul_model shrinkF (shrinkF * Float.exp (-Mf))
  have h1e : (0 : ℝ) < 1 + δe := by
    have := (abs_le.mp hδe).1; linarith [expEps_lt_one]
  have h12 : (0 : ℝ) < 1 + δ₂ := by
    have := (abs_le.mp hδ₂).1; linarith [u64_lt_one]
  have h11 : (0 : ℝ) < 1 + δ₁ := by
    have := (abs_le.mp hδ₁).1; linarith [u64_lt_one]
  have hE : 0 < toReal (Float.exp (-Mf)) := by
    rw [ee]; exact mul_pos (Real.exp_pos _) h1e
  have hX : 0 < toReal (shrinkF * Float.exp (-Mf)) := by
    rw [e₂]; exact mul_pos (mul_pos shrinkF_pos hE) h12
  show 0 < toReal (shrinkF * (shrinkF * Float.exp (-Mf)))
  rw [e₁]
  exact mul_pos (mul_pos shrinkF_pos hX) h11

/-- **THE LOWER DOMINATION** (the direction reversal's payoff): the Float floor sits BELOW the
    exact `floorC M` whenever the argument-bound mirror sits ABOVE `M` — the adverse roundings
    are the UPWARD ones (`exp_model`'s `(1+expEps)`, each multiplication's `(1+u64)`), absorbed
    together with `floorC`'s `(1−expEps)(1−u64)` haircut by the anti-key. -/
theorem floorCF_lower (Mf : Float) (M : ℝ) (hMm : M ≤ toReal Mf) :
    toReal (floorCF Mf) ≤ floorC M := by
  obtain ⟨δe, hδe, ee⟩ := exp_model (-Mf)
  obtain ⟨δ₂, hδ₂, e₂⟩ := mul_model shrinkF (Float.exp (-Mf))
  obtain ⟨δ₁, hδ₁, e₁⟩ := mul_model shrinkF (shrinkF * Float.exp (-Mf))
  have hde : 1 + δe ≤ 1 + expEps := by have := (abs_le.mp hδe).2; linarith
  have hd₂ : 1 + δ₂ ≤ 1 + u64 := by have := (abs_le.mp hδ₂).2; linarith
  have hd₁ : 1 + δ₁ ≤ 1 + u64 := by have := (abs_le.mp hδ₁).2; linarith
  have hs0 := shrinkF_pos.le
  have hexp_mono : Real.exp (toReal (-Mf)) ≤ Real.exp (-M) := by
    rw [toReal_neg]; exact Real.exp_le_exp.mpr (neg_le_neg hMm)
  have hE : toReal (Float.exp (-Mf)) ≤ Real.exp (-M) * (1 + expEps) := by
    rw [ee]
    calc Real.exp (toReal (-Mf)) * (1 + δe)
        ≤ Real.exp (toReal (-Mf)) * (1 + expEps) :=
          mul_le_mul_of_nonneg_left hde (Real.exp_pos _).le
      _ ≤ Real.exp (-M) * (1 + expEps) :=
          mul_le_mul_of_nonneg_right hexp_mono (by linarith [expEps_pos])
  have hE0 : 0 ≤ toReal (Float.exp (-Mf)) := by
    rw [ee]
    have : (0 : ℝ) ≤ 1 + δe := by have := (abs_le.mp hδe).1; linarith [expEps_lt_one]
    exact mul_nonneg (Real.exp_pos _).le this
  have hX : toReal (shrinkF * Float.exp (-Mf))
      ≤ toReal shrinkF * (Real.exp (-M) * (1 + expEps)) * (1 + u64) := by
    rw [e₂]
    calc toReal shrinkF * toReal (Float.exp (-Mf)) * (1 + δ₂)
        ≤ toReal shrinkF * toReal (Float.exp (-Mf)) * (1 + u64) :=
          mul_le_mul_of_nonneg_left hd₂ (mul_nonneg hs0 hE0)
      _ ≤ toReal shrinkF * (Real.exp (-M) * (1 + expEps)) * (1 + u64) :=
          mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hE hs0)
            (by linarith [u64_pos])
  have hX0 : 0 ≤ toReal (shrinkF * Float.exp (-Mf)) := by
    rw [e₂]
    have : (0 : ℝ) ≤ 1 + δ₂ := by have := (abs_le.mp hδ₂).1; linarith [u64_lt_one]
    exact mul_nonneg (mul_nonneg hs0 hE0) this
  have hfinal : toReal (floorCF Mf)
      ≤ toReal shrinkF * (toReal shrinkF * (Real.exp (-M) * (1 + expEps)) * (1 + u64))
          * (1 + u64) := by
    show toReal (shrinkF * (shrinkF * Float.exp (-Mf))) ≤ _
    rw [e₁]
    calc toReal shrinkF * toReal (shrinkF * Float.exp (-Mf)) * (1 + δ₁)
        ≤ toReal shrinkF * toReal (shrinkF * Float.exp (-Mf)) * (1 + u64) :=
          mul_le_mul_of_nonneg_left hd₁ (mul_nonneg hs0 hX0)
      _ ≤ toReal shrinkF * (toReal shrinkF * (Real.exp (-M) * (1 + expEps)) * (1 + u64))
            * (1 + u64) :=
          mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hX hs0)
            (by linarith [u64_pos])
  refine hfinal.trans ?_
  unfold Puffer.RL.FloatFloor.floorC
  calc toReal shrinkF * (toReal shrinkF * (Real.exp (-M) * (1 + expEps)) * (1 + u64))
        * (1 + u64)
      = Real.exp (-M)
          * (toReal shrinkF * toReal shrinkF * ((1 + expEps) * ((1 + u64) * (1 + u64)))) := by
        ring
    _ ≤ Real.exp (-M) * ((1 - expEps) * (1 - u64)) :=
        mul_le_mul_of_nonneg_left shrinkF_key (Real.exp_pos _).le
    _ = Real.exp (-M) * (1 - expEps) * (1 - u64) := by ring

/-- **The floor's log costs at most one more than the argument bound**: `|log (floorC M)| ≤ M+1`
    for `M ≥ 0` — `floorC M = exp(−M)·(1−expEps)(1−u64)` sits in `[exp(−(M+1)), 1]` because the
    haircut `(1−expEps)(1−u64) ≥ 1/2 ≥ exp(−1)`. This is what prices the `|log c|` entry of the
    log-inclusive forward budget in `Mf` units. -/
theorem abs_log_floorC_le (M : ℝ) (hM : 0 ≤ M) : |Real.log (floorC M)| ≤ M + 1 := by
  have hpos := floorC_pos M
  have hu := u64_le_small
  have he := expEps_le_small
  have h2 : (2 : ℝ) ≤ Real.exp 1 := by linarith [Real.add_one_le_exp (1 : ℝ)]
  have hexp1 : Real.exp (-(1 : ℝ)) ≤ 1 / 2 := by
    rw [Real.exp_neg]
    have := one_div_le_one_div_of_le (by norm_num : (0 : ℝ) < 2) h2
    simpa [one_div] using this
  have hprod : Real.exp (-(1 : ℝ)) ≤ (1 - expEps) * (1 - u64) := by
    nlinarith [he, hu, mul_nonneg expEps_pos.le u64_pos.le, hexp1]
  -- upper: floorC M ≤ 1, so log(floorC M) ≤ 0 ≤ M + 1
  have hexpM : Real.exp (-M) ≤ 1 := by
    rw [← Real.exp_zero]
    exact Real.exp_le_exp.mpr (by linarith)
  have hle1 : floorC M ≤ 1 := by
    unfold Puffer.RL.FloatFloor.floorC
    nlinarith [hexpM, (Real.exp_pos (-M)).le, one_sub_expEps_pos.le, one_sub_u64_pos.le,
      expEps_pos, u64_pos, mul_nonneg one_sub_expEps_pos.le one_sub_u64_pos.le,
      mul_nonneg expEps_pos.le u64_pos.le]
  have hub : Real.log (floorC M) ≤ 0 := by
    have := Real.log_le_sub_one_of_pos hpos
    linarith
  -- lower: exp(−(M+1)) ≤ floorC M, so −(M+1) ≤ log(floorC M)
  have hlb : Real.exp (-(M + 1)) ≤ floorC M := by
    have heq : Real.exp (-(M + 1)) = Real.exp (-M) * Real.exp (-(1 : ℝ)) := by
      rw [← Real.exp_add]; ring_nf
    rw [heq]
    unfold Puffer.RL.FloatFloor.floorC
    calc Real.exp (-M) * Real.exp (-(1 : ℝ))
        ≤ Real.exp (-M) * ((1 - expEps) * (1 - u64)) :=
          mul_le_mul_of_nonneg_left hprod (Real.exp_pos _).le
      _ = Real.exp (-M) * (1 - expEps) * (1 - u64) := by ring
  have hlow : -(M + 1) ≤ Real.log (floorC M) := by
    have hlog := Real.log_le_log (Real.exp_pos _) hlb
    rwa [Real.log_exp] at hlog
  exact abs_le.mpr ⟨hlow, by linarith⟩

/-! ### The log-inclusive forward-budget mirror -/

/-- C72's log-inclusive forward budget is nonnegative (the log entry is a scaled max of absolute
    values; every other case routes through the subtrees as in C77's `fwdBound_nonneg`). -/
theorem fwdBoundL_nonneg (B c : ℝ) (hB : 0 ≤ B) : ∀ e : Expr, 0 ≤ fwdBoundL B c e := by
  intro e
  induction e with
  | var i => exact hB
  | const cc => exact abs_nonneg _
  | add a b iha ihb =>
      simp only [fwdBoundL]
      exact mul_nonneg (by linarith [u64_pos]) (by linarith)
  | sub a b iha ihb =>
      simp only [fwdBoundL]
      exact mul_nonneg (by linarith [u64_pos]) (by linarith)
  | mul a b iha ihb =>
      simp only [fwdBoundL]
      exact mul_nonneg (by linarith [u64_pos]) (mul_nonneg iha ihb)
  | scale cc a iha =>
      simp only [fwdBoundL]
      exact mul_nonneg (by linarith [u64_pos]) (mul_nonneg (abs_nonneg _) iha)
  | exp a _ =>
      simp only [fwdBoundL]
      exact mul_nonneg (Real.exp_pos _).le (by linarith [expEps_pos])
  | log a _ =>
      simp only [fwdBoundL]
      exact mul_nonneg ((abs_nonneg _).trans (le_max_left _ _)) (by linarith [logEps_pos])
  | relu a iha => simpa only [fwdBoundL] using iha
  | max a b iha _ => simp only [fwdBoundL]; exact iha.trans (le_max_left _ _)
  | min a b iha _ => simp only [fwdBoundL]; exact iha.trans (le_max_left _ _)

/-- **A partition-shaped log argument's budget is at least `1`** (`exp` of a nonnegative budget
    is `≥ 1`; the inflations and the tail only grow it) — so its log is nonnegative and at most
    the budget itself, the `|log U| ≤ U` half of the log entry's pricing. -/
theorem one_le_fwdBound_addExp (B : ℝ) (hB : 0 ≤ B) (a b : Expr) :
    1 ≤ fwdBound B (.add (.exp a) b) := by
  show (1 : ℝ) ≤ (1 + u64) * (Real.exp (fwdBound B a) * (1 + expEps) + fwdBound B b)
  have hexp1 : (1 : ℝ) ≤ Real.exp (fwdBound B a) := by
    have := Real.add_one_le_exp (fwdBound B a)
    linarith [fwdBound_nonneg B hB a]
  have hb0 : 0 ≤ fwdBound B b := fwdBound_nonneg B hB b
  have hsum0 : (0 : ℝ) ≤ Real.exp (fwdBound B a) * (1 + expEps) + fwdBound B b :=
    add_nonneg (mul_nonneg (Real.exp_pos _).le (by linarith [expEps_pos])) hb0
  calc (1 : ℝ) ≤ Real.exp (fwdBound B a) := hexp1
    _ ≤ Real.exp (fwdBound B a) * (1 + expEps) :=
        le_mul_of_one_le_right (Real.exp_pos _).le (by linarith [expEps_pos])
    _ ≤ Real.exp (fwdBound B a) * (1 + expEps) + fwdBound B b := le_add_of_nonneg_right hb0
    _ ≤ (1 + u64) * (Real.exp (fwdBound B a) * (1 + expEps) + fwdBound B b) :=
        le_mul_of_one_le_left hsum0 (by linarith [u64_pos])

/-- **The computable UPPER mirror of C72's `fwdBoundL`** (for `PartitionLogs` expressions):
    C80's `fwdBoundF` cases everywhere except `log`, whose entry prices the floor's log by
    `slackF·(slackF·(Mf + 1.0) + fwdBoundF B arg)` — the SUM dominates the ℝ side's max of
    `|log c| ≤ M+1` (via `abs_log_floorC_le`) and `|log U| ≤ U` (via `one_le_fwdBound_addExp`),
    and the outer `slackF` pays the `(1+logEps)` factor through `slackF_key_log`. -/
def fwdBoundLF (B Mf : Float) : Expr → Float
  | .var _ => B
  | .const c => absF c
  | .add a b => slackF * (fwdBoundLF B Mf a + fwdBoundLF B Mf b)
  | .sub a b => slackF * (fwdBoundLF B Mf a + fwdBoundLF B Mf b)
  | .mul a b => slackF * (fwdBoundLF B Mf a * fwdBoundLF B Mf b)
  | .scale c a => slackF * (absF c * fwdBoundLF B Mf a)
  | .exp a => slackF * Float.exp (fwdBoundLF B Mf a)
  | .log arg => slackF * (slackF * (Mf + 1.0) + fwdBoundF B arg)
  | .relu a => fwdBoundLF B Mf a
  | .max a b => max (fwdBoundLF B Mf a) (fwdBoundLF B Mf b)
  | .min a b => max (fwdBoundLF B Mf a) (fwdBoundLF B Mf b)

/-- **Log-inclusive forward-budget domination** at the concrete floor interface: for a
    `PartitionLogs` expression, `fwdBoundL B c e ≤ toReal (fwdBoundLF Bf Mf e)` given the
    upper mirrors `B ≤ toReal Bf`, `M ≤ toReal Mf` and the floor's log price `|log c| ≤ M+1`.
    The non-log cases are C80's `fwdBoundF_dominates` verbatim (one `slack_step` per rounded op,
    `slack_exp_step` for exp); the log case prices `max |log c| |log U|` by
    `(1+u64)(M+1) + U` and spends `slackF_key_log` on the `(1+logEps)` factor. -/
theorem fwdBoundLF_dominates (Bf Mf : Float) (B c M : ℝ)
    (hB0 : 0 ≤ B) (hBb : B ≤ toReal Bf) (hM0 : 0 ≤ M) (hMm : M ≤ toReal Mf)
    (hc : |Real.log c| ≤ M + 1) :
    ∀ e : Expr, PartitionLogs e → fwdBoundL B c e ≤ toReal (fwdBoundLF Bf Mf e) := by
  intro e
  induction e with
  | var i => intro _; exact hBb
  | const cc => intro _; exact le_of_eq (toReal_absF cc).symm
  | add a b iha ihb =>
      intro hp
      simp only [fwdBoundL, fwdBoundLF]
      exact slack_step _ _ _
        (add_nonneg (fwdBoundL_nonneg B c hB0 a) (fwdBoundL_nonneg B c hB0 b))
        (add_le_add (iha hp.1) (ihb hp.2)) (add_model _ _)
  | sub a b iha ihb =>
      intro hp
      simp only [fwdBoundL, fwdBoundLF]
      exact slack_step _ _ _
        (add_nonneg (fwdBoundL_nonneg B c hB0 a) (fwdBoundL_nonneg B c hB0 b))
        (add_le_add (iha hp.1) (ihb hp.2)) (add_model _ _)
  | mul a b iha ihb =>
      intro hp
      simp only [fwdBoundL, fwdBoundLF]
      exact slack_step _ _ _
        (mul_nonneg (fwdBoundL_nonneg B c hB0 a) (fwdBoundL_nonneg B c hB0 b))
        (mul_le_mul (iha hp.1) (ihb hp.2) (fwdBoundL_nonneg B c hB0 b)
          ((fwdBoundL_nonneg B c hB0 a).trans (iha hp.1)))
        (mul_model _ _)
  | scale cc a iha =>
      intro hp
      simp only [fwdBoundL, fwdBoundLF]
      refine slack_step _ _ _
        (mul_nonneg (abs_nonneg _) (fwdBoundL_nonneg B c hB0 a)) ?_ (mul_model _ _)
      rw [toReal_absF]
      exact mul_le_mul_of_nonneg_left (iha hp) (abs_nonneg _)
  | exp a iha =>
      intro hp
      simp only [fwdBoundL, fwdBoundLF]
      exact slack_exp_step _ _ (iha hp)
  | log arg _ =>
      intro hp
      obtain ⟨a, b, rfl, hlfa, hnb⟩ := hp
      have hlf : LogFree (Expr.add (.exp a) b) := ⟨hlfa, nonnegShape_logFree b hnb⟩
      show max |Real.log c| |Real.log (fwdBoundL B c (Expr.add (.exp a) b))| * (1 + logEps)
          ≤ toReal (slackF * (slackF * (Mf + 1.0) + fwdBoundF Bf (Expr.add (.exp a) b)))
      rw [fwdBoundL_eq_fwdBound B c _ hlf]
      set U : ℝ := fwdBound B (Expr.add (.exp a) b) with hUdef
      have hU1 : 1 ≤ U := one_le_fwdBound_addExp B hB0 a b
      have hUf : U ≤ toReal (fwdBoundF Bf (Expr.add (.exp a) b)) :=
        fwdBoundF_dominates Bf B hB0 hBb _ hlf
      have hlogU : |Real.log U| ≤ U := by
        rw [abs_of_nonneg (Real.log_nonneg hU1)]
        have := Real.log_le_sub_one_of_pos (lt_of_lt_of_le one_pos hU1)
        linarith
      have hstep1 : (1 + u64) * (M + 1) ≤ toReal (slackF * (Mf + 1.0)) := by
        refine slack_step (Mf + 1.0) (M + 1) (toReal Mf + 1) (by linarith) (by linarith) ?_
        obtain ⟨δ, hδ, he⟩ := add_model Mf 1.0
        rw [toReal_oneLit] at he
        exact ⟨δ, hδ, he⟩
      have hmax : max |Real.log c| |Real.log U| ≤ (1 + u64) * (M + 1) + U := by
        refine max_le ?_ ?_
        · nlinarith [hc, hM0, u64_pos, hU1]
        · nlinarith [hlogU, hM0, u64_pos]
      have hy0 : (0 : ℝ) ≤ (1 + u64) * (M + 1) + U := by nlinarith [hM0, u64_pos, hU1]
      have hyY : (1 + u64) * (M + 1) + U
          ≤ toReal (slackF * (Mf + 1.0)) + toReal (fwdBoundF Bf (Expr.add (.exp a) b)) :=
        add_le_add hstep1 hUf
      calc max |Real.log c| |Real.log U| * (1 + logEps)
          = (1 + logEps) * max |Real.log c| |Real.log U| := by ring
        _ ≤ (1 + logEps) * ((1 + u64) * (M + 1) + U) :=
            mul_le_mul_of_nonneg_left hmax (by linarith [logEps_pos])
        _ ≤ toReal (slackF * (slackF * (Mf + 1.0) + fwdBoundF Bf (Expr.add (.exp a) b))) :=
            slack_step_log _ _ _ hy0 hyY (add_model _ _)
  | relu a iha => intro hp; simpa only [fwdBoundL, fwdBoundLF] using iha hp
  | max a b iha ihb =>
      intro hp
      simp only [fwdBoundL, fwdBoundLF]
      rw [toReal_max]
      exact max_le_max (iha hp.1) (ihb hp.2)
  | min a b iha ihb =>
      intro hp
      simp only [fwdBoundL, fwdBoundLF]
      rw [toReal_max]
      exact max_le_max (iha hp.1) (ihb hp.2)

/-! ### The log-inclusive weight-budget mirror -/

/-- Log-free terms are (vacuously) `PartitionLogs` — the bridge that lets the weight budget's
    recursion descend INTO a partition-shaped log argument (which is log-free). -/
theorem logFree_partitionLogs : ∀ e : Expr, LogFree e → PartitionLogs e := by
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

/-- The recursion behind `compWeightBoundLF`, with the global floor representative `cF` (and the
    argument-bound mirror `Mf`) as parameters: C80's `compWeightBoundF` cases everywhere except
    (i) the `mul`/`exp` sibling budgets routed through `fwdBoundLF` and (ii) the log entry
    `slackF·(1.0/cF)` — the Float mirror of `(1/c)·(1+u64)`, dominating BECAUSE `cF` is a lower
    representative (a smaller positive denominator gives a larger quotient). -/
def compWeightBoundLFCore (B Mf cF : Float) : Expr → Float
  | .var _ => 1.0
  | .const _ => 1.0
  | .add a b => max 1.0 (max (compWeightBoundLFCore B Mf cF a) (compWeightBoundLFCore B Mf cF b))
  | .sub a b => max 1.0 (max (compWeightBoundLFCore B Mf cF a) (compWeightBoundLFCore B Mf cF b))
  | .mul a b => max (max (fwdBoundLF B Mf a) (fwdBoundLF B Mf b))
      (max (compWeightBoundLFCore B Mf cF a) (compWeightBoundLFCore B Mf cF b))
  | .scale c a => max (absF c) (compWeightBoundLFCore B Mf cF a)
  | .exp a => max (slackF * Float.exp (fwdBoundLF B Mf a)) (compWeightBoundLFCore B Mf cF a)
  | .log a => max (slackF * (1.0 / cF)) (compWeightBoundLFCore B Mf cF a)
  | .relu a => max 1.0 (compWeightBoundLFCore B Mf cF a)
  | .max a b => max 1.0 (max (compWeightBoundLFCore B Mf cF a) (compWeightBoundLFCore B Mf cF b))
  | .min a b => max 1.0 (max (compWeightBoundLFCore B Mf cF a) (compWeightBoundLFCore B Mf cF b))

/-- **Core domination**: `compWeightBoundL B c e ≤ toReal (compWeightBoundLFCore Bf Mf cF e)`
    for a `PartitionLogs` expression, given the two upper mirrors (`B`, `M`), the floor's log
    price, and the floor's LOWER representative (`0 < toReal cF ≤ c`). The `max` spine
    transports exactly; the log entry is one `slack_step` on `div_model`; the `mul`/`exp`
    sibling budgets are `fwdBoundLF_dominates`. -/
theorem compWeightBoundLFCore_dominates (Bf Mf cF : Float) (B c M : ℝ)
    (hB0 : 0 ≤ B) (hBb : B ≤ toReal Bf) (hM0 : 0 ≤ M) (hMm : M ≤ toReal Mf)
    (hc : |Real.log c| ≤ M + 1) (hcF0 : 0 < toReal cF) (hcFc : toReal cF ≤ c) :
    ∀ e : Expr, PartitionLogs e →
      compWeightBoundL B c e ≤ toReal (compWeightBoundLFCore Bf Mf cF e) := by
  intro e
  induction e with
  | var i => intro _; exact le_of_eq toReal_oneLit.symm
  | const cc => intro _; exact le_of_eq toReal_oneLit.symm
  | add a b iha ihb =>
      intro hp
      simp only [compWeightBoundL, compWeightBoundLFCore]
      rw [toReal_max, toReal_max, toReal_oneLit]
      exact max_le_max le_rfl (max_le_max (iha hp.1) (ihb hp.2))
  | sub a b iha ihb =>
      intro hp
      simp only [compWeightBoundL, compWeightBoundLFCore]
      rw [toReal_max, toReal_max, toReal_oneLit]
      exact max_le_max le_rfl (max_le_max (iha hp.1) (ihb hp.2))
  | mul a b iha ihb =>
      intro hp
      simp only [compWeightBoundL, compWeightBoundLFCore]
      rw [toReal_max, toReal_max, toReal_max]
      exact max_le_max
        (max_le_max (fwdBoundLF_dominates Bf Mf B c M hB0 hBb hM0 hMm hc a hp.1)
          (fwdBoundLF_dominates Bf Mf B c M hB0 hBb hM0 hMm hc b hp.2))
        (max_le_max (iha hp.1) (ihb hp.2))
  | scale cc a iha =>
      intro hp
      simp only [compWeightBoundL, compWeightBoundLFCore]
      rw [toReal_max]
      exact max_le_max (le_of_eq (toReal_absF cc).symm) (iha hp)
  | exp a iha =>
      intro hp
      simp only [compWeightBoundL, compWeightBoundLFCore]
      rw [toReal_max]
      exact max_le_max
        (slack_exp_step _ _ (fwdBoundLF_dominates Bf Mf B c M hB0 hBb hM0 hMm hc a hp))
        (iha hp)
  | log a iha =>
      intro hp
      obtain ⟨aa, bb, rfl, hlfa, hnb⟩ := hp
      simp only [compWeightBoundL, compWeightBoundLFCore]
      rw [toReal_max]
      refine max_le_max ?_
        (iha (logFree_partitionLogs _ ⟨hlfa, nonnegShape_logFree bb hnb⟩))
      have hc0 : (0 : ℝ) < c := lt_of_lt_of_le hcF0 hcFc
      have hop : ∃ δ : ℝ, |δ| ≤ u64
          ∧ toReal ((1.0 : Float) / cF) = (1 / toReal cF) * (1 + δ) := by
        obtain ⟨δ, hδ, he⟩ := div_model 1.0 cF
        rw [toReal_oneLit] at he
        exact ⟨δ, hδ, he⟩
      have h := slack_step ((1.0 : Float) / cF) (1 / c) (1 / toReal cF)
        (by positivity) (one_div_le_one_div_of_le hcF0 hcFc) hop
      calc (1 / c) * (1 + u64) = (1 + u64) * (1 / c) := by ring
        _ ≤ toReal (slackF * (1.0 / cF)) := h
  | relu a iha =>
      intro hp
      simp only [compWeightBoundL, compWeightBoundLFCore]
      rw [toReal_max, toReal_oneLit]
      exact max_le_max le_rfl (iha hp)
  | max a b iha ihb =>
      intro hp
      simp only [compWeightBoundL, compWeightBoundLFCore]
      rw [toReal_max, toReal_max, toReal_oneLit]
      exact max_le_max le_rfl (max_le_max (iha hp.1) (ihb hp.2))
  | min a b iha ihb =>
      intro hp
      simp only [compWeightBoundL, compWeightBoundLFCore]
      rw [toReal_max, toReal_max, toReal_oneLit]
      exact max_le_max le_rfl (max_le_max (iha hp.1) (ihb hp.2))

/-- **The log-inclusive weight-budget evaluator** (C80's disclosed remainder, closed): the core
    recursion seeded with the computable argument-bound mirror `logArgBoundF B e` and the
    computable lower floor `floorCF (logArgBoundF B e)` — everything the harness needs is the
    input bound `B` and the expression itself. -/
def compWeightBoundLF (B : Float) (e : Expr) : Float :=
  compWeightBoundLFCore B (logArgBoundF B e) (floorCF (logArgBoundF B e)) e

/-- **THE `hDf` STATEMENT, PROVED**: the computable evaluator dominates the exact log-inclusive
    weight budget at the concrete floor `floorC (logArgBound B e)` — exactly the ONE plumbed
    hypothesis of C80's partition capstone, now a theorem. Composes the upper argument-bound
    mirror (`logArgBoundF_dominates`), the lower floor (`floorCF_lower`/`floorCF_pos`), the
    floor's log price (`abs_log_floorC_le`), and the core domination. -/
theorem compWeightBoundLF_dominates (Bf : Float) (B : ℝ) (hB0 : 0 ≤ B) (hBb : B ≤ toReal Bf)
    (e : Expr) (hp : PartitionLogs e) :
    compWeightBoundL B (floorC (logArgBound B e)) e ≤ toReal (compWeightBoundLF Bf e) := by
  have hM0 : 0 ≤ logArgBound B e := logArgBound_nonneg B hB0 e
  have hMm : logArgBound B e ≤ toReal (logArgBoundF Bf e) :=
    logArgBoundF_dominates Bf B hB0 hBb e hp
  exact compWeightBoundLFCore_dominates Bf (logArgBoundF Bf e) (floorCF (logArgBoundF Bf e))
    B (floorC (logArgBound B e)) (logArgBound B e) hB0 hBb hM0 hMm
    (abs_log_floorC_le _ hM0) (floorCF_pos _) (floorCF_lower _ _ hMm) e hp

/-! ### The capstone: the fully-runnable partition tape certificate -/

/-- **CAPSTONE — C80's LAST PLUMBED CONSTANT ELIMINATED.** The partition-shaped (PPO-shaped)
    tape certificate whose weight budget is COMPUTED by the harness: for a `PartitionLogs`
    expression compiled by the ACTUAL `comp` from `Tape.empty`, with the inputs recorded as a
    list (C73's zero-padded representation `hrep`), THREE runtime Bools — `checkRegion inputs
    Bf`, `checkLe 0.0 Bf`, and ONE sweep-budget check at `compWeightBoundLF Bf e` — certify
    every gradient of the ACTUAL `grads` engine overflow-free. C80's `hDf` hypothesis is GONE,
    replaced by `compWeightBoundLF_dominates`; nothing about the budget is plumbed. -/
theorem adGrad_isFinite_comp_partition_runnable' (inputs : List Float) (σ : Nat → Float)
    (Bf : Float)
    (hrep : ∀ i, σ i = padRow inputs i)
    (hin : checkRegion inputs Bf = true) (hB0 : checkLe 0.0 Bf = true)
    (e : Expr) (hp : PartitionLogs e) (root : V)
    (hbudget : checkLe (sweepBoundF 2 (compWeightBoundLF Bf e)
        (comp σ e Tape.empty).2.val.size 1.0) capF = true) :
    ∀ j, j < (comp σ e Tape.empty).2.val.size →
      ((grads (comp σ e Tape.empty).2 root)[j]!).isFinite = true := by
  have hB0' : (0 : ℝ) ≤ toReal Bf := by
    have := checkLe_sound hB0
    rwa [toReal_zeroLit] at this
  have hσ : ∀ i, |toReal (σ i)| ≤ toReal Bf := by
    intro i
    rw [hrep i]
    exact padRow_abs_le hin hB0' i
  exact adGrad_isFinite_comp_partition σ (toReal Bf) hσ e hp root
    (sweepBound_le_overflow_of_check 2 (compWeightBoundLF Bf e) 1.0
      (compWeightBoundL (toReal Bf) (floorC (logArgBound (toReal Bf) e)) e) 1 _
      (compWeightBoundL_nonneg _ _ e)
      (compWeightBoundLF_dominates Bf (toReal Bf) hB0' le_rfl e hp)
      zero_le_one (le_of_eq toReal_oneLit.symm) hbudget)

/-! ### Non-vacuity: the genuine two-logit log-partition, checked at runtime -/

/-- The demo shape: the genuine two-logit log-partition `log (exp x₀ + exp x₁)` — the PPO
    loss's log usage (C77's canonical `PartitionLogs` witness). -/
private def demoE : Expr := .log (.add (.exp (.var 0)) (.exp (.var 1)))

/-- The capstone instantiates on the two-logit log-partition: the runnable Bools alone certify
    every compiled gradient finite (nothing plumbed — the weight budget is computed). -/
example (inputs : List Float) (σ : Nat → Float)
    (hrep : ∀ i, σ i = padRow inputs i)
    (hin : checkRegion inputs 2.0 = true) (hB0 : checkLe 0.0 2.0 = true) (root : V)
    (hbudget : checkLe (sweepBoundF 2 (compWeightBoundLF 2.0 demoE)
        (comp σ demoE Tape.empty).2.val.size 1.0) capF = true) :
    ∀ j, j < (comp σ demoE Tape.empty).2.val.size →
      ((grads (comp σ demoE Tape.empty).2 root)[j]!).isFinite = true :=
  adGrad_isFinite_comp_partition_runnable' inputs σ 2.0 hrep hin hB0 demoE
    ⟨.var 0, .exp (.var 1), rfl, trivial, trivial⟩ root hbudget

-- POSITIVE CONTROL (the check is not vacuously false): at input budget `Bf = 2.0` the
-- computed log-inclusive sweep budget (≈ 2.6·10⁴⁰, floor ≈ e⁻¹⁵) passes `capF`.
/-- info: true -/
#guard_msgs in
#eval checkLe (sweepBoundF 2 (compWeightBoundLF 2.0 demoE)
  (comp (fun _ => (0.5 : Float)) demoE Tape.empty).2.val.size 1.0) capF

-- NEGATIVE CONTROL (the check is not vacuously true): at `Bf = 100.0` the argument budget
-- ≈ 2e¹⁰⁰ drives `Float.exp (−Mf)` to underflow, the floor to `0.0`, the log edge budget to
-- `inf` — and the evaluator REJECTS (sound: it never certifies what it cannot floor).
/-- info: false -/
#guard_msgs in
#eval checkLe (sweepBoundF 2 (compWeightBoundLF 100.0 demoE)
  (comp (fun _ => (0.5 : Float)) demoE Tape.empty).2.val.size 1.0) capF

end Puffer.RL.FloorEval
