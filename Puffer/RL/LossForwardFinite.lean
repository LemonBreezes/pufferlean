/-
# PPO loss forward finiteness: carrying the no-overflow certificate through the loss forward terms

C43 (`FiniteBound`) certified a bounded linear kernel overflow-free (the one trusted no-overflow axiom
`isFinite_of_bounded` + the `(1+δ)` base); C45 (`ForwardFinite`) lifted it to the MLP forward pass; C48
(`LossFinite`) reached the readout head + the softmax `exp` numerator. This module carries the certificate through
the PPO LOSS FORWARD terms computed from bounded logits/activations: the **softmax partition** `Σ exp(logit)` (and,
via `log_model`, the **log-partition**), the **ratio** `exp(newLogp − oldLogp)`, the **clip** `min(g·r,
g·clamp(r,lo,hi))`, and the **value squared error** `(V − ret)²`. Each is a magnitude bound propagated from the
`(1+δ)` base under its honest side-condition, then `isFinite_of_bounded` for overflow-freedom — exactly C43/C45/C48's
pattern. **NO new axiom** (reuses C43's `isFinite_of_bounded` + the pre-existing `add`/`sub`/`mul`/`exp`/`log`
models and the exact `toReal_min`/`toReal_max`).

* `sub_bound` / `mul_bound` — magnitude-bound helpers (`(1+u)` growth) composing C43's `sub_mag_le`/`mul_mag_le`.
* `min_mag_le` / `max_mag_le` — Float `min`/`max` PRESERVE magnitude (`|toReal (min a b)| ≤ max |a| |b|`), from the
  EXACT `toReal_min`/`toReal_max`. The clip's core (min/max are magnitude-non-increasing — no roundoff).
* `valSqF` / `valSqF_mag_le` / `valSqF_isFinite` — the value squared error `(V − ret)²` is overflow-free from bounded
  `V`, `ret`.
* `ratioF` / `ratioF_mag_le` / `ratioF_isFinite` — the PPO ratio `exp(newLogp − oldLogp)` is overflow-free given the
  log-prob difference bounded (composes `sub` → `exp`, needing the exponent bounded above, C48's honest condition).
* `clampF` / `clampF_mag_le`, `ppoSurrF` / `ppoSurrF_mag_le` / `ppoSurrF_isFinite` — the clipped surrogate
  `min(g·r, g·clamp(r,lo,hi))` is overflow-free from bounded `g`, `r`, `lo`, `hi` (min/max preserve magnitude, so the
  clip adds no growth beyond the `g`-scalings).
* `sumExpF` / `sumBound` / `sumExpF_mag_le`, `expF_pos` / `sumExpF_nonneg` / `sumExpF_pos` / `sumExpF_isFinite` — the
  softmax partition `Σ exp(logit)` is magnitude-bounded and overflow-free from logits bounded ABOVE, and is strictly
  POSITIVE (so it can floor the log).
* `logMag_le` / `logPart_isFinite` — the **log-partition** `log(Σ exp)` is overflow-free given the partition floored
  `≥ c > 0` and bounded (`|log| ≤ max |log c| |log U|`, finite). Uses the pre-existing `log_model` — reaching the one
  loss term with a genuine domain side-condition (`log` needs its argument away from `0`).

**Scope (honestly disclosed).** This extends C48 through the PPO loss FORWARD terms above (partition, log-partition,
ratio, clip, value squared error), reusing C43's `isFinite_of_bounded` + the `(1+δ)` base — **NO new axiom**. The
honest side-conditions are stated explicitly: logits bounded ABOVE (for `exp`; unbounded logits genuinely overflow),
the partition floored `> 0` (for `log`; the floor `c` is a checkable input — `sumExpF_pos` shows the partition is
positive, but a concrete floor from the min logit is left as the hypothesis), and every operand magnitude-bounded.
It does NOT assemble the terms into the single scalar loss value, and does NOT cover the BACKWARD / AD pass or the
OPTIMIZER — a full-trainer finiteness would compose those next. Magnitude bounds grow with the operand bounds
(`exp(M)` for the exp/ratio, `(1+u)` per op) — realistic; `… ≤ overflowBound` is the checkable no-overflow condition.
-/
import Puffer.RL.LossFinite
open Puffer.FloatR
open Puffer.RL.FiniteBound
open Puffer.RL.LossFinite

namespace Puffer.RL.LossForwardFinite

/-- `0 ≤ 1 + u64` (roundoff factor positive). -/
private theorem one_add_u64_nonneg : (0 : ℝ) ≤ 1 + u64 := by have := u64_pos; linarith

/-- **Subtraction magnitude bound** to explicit operand caps: `|toReal (a − b)| ≤ (1+u)·(Ba + Bb)`. -/
theorem sub_bound (a b : Float) (Ba Bb : ℝ) (ha : |toReal a| ≤ Ba) (hb : |toReal b| ≤ Bb) :
    |toReal (a - b)| ≤ (1 + u64) * (Ba + Bb) :=
  (sub_mag_le a b).trans (mul_le_mul_of_nonneg_left (add_le_add ha hb) one_add_u64_nonneg)

/-- **Product magnitude bound** to explicit operand caps: `|toReal (a · b)| ≤ (1+u)·(Ba · Bb)`. -/
theorem mul_bound (a b : Float) (Ba Bb : ℝ) (ha : |toReal a| ≤ Ba) (hb : |toReal b| ≤ Bb) :
    |toReal (a * b)| ≤ (1 + u64) * (Ba * Bb) :=
  (mul_mag_le a b).trans (mul_le_mul_of_nonneg_left
    (mul_le_mul ha hb (abs_nonneg _) ((abs_nonneg _).trans ha)) one_add_u64_nonneg)

/-- **Float `min` preserves magnitude** (EXACT, `toReal_min`; no roundoff): `|toReal (min a b)| ≤ max |a| |b|`. -/
theorem min_mag_le (a b : Float) : |toReal (min a b)| ≤ max |toReal a| |toReal b| := by
  rw [toReal_min]
  rcases le_total (toReal a) (toReal b) with h | h
  · rw [min_eq_left h]; exact le_max_left _ _
  · rw [min_eq_right h]; exact le_max_right _ _

/-- **Float `max` preserves magnitude** (EXACT, `toReal_max`; no roundoff): `|toReal (max a b)| ≤ max |a| |b|`. -/
theorem max_mag_le (a b : Float) : |toReal (max a b)| ≤ max |toReal a| |toReal b| := by
  rw [toReal_max]
  rcases le_total (toReal a) (toReal b) with h | h
  · rw [max_eq_right h]; exact le_max_right _ _
  · rw [max_eq_left h]; exact le_max_left _ _

/-- **The value squared error** `(V − ret)²`. -/
def valSqF (V ret : Float) : Float := (V - ret) * (V - ret)

/-- **Value squared error magnitude** from bounded `V`, `ret`: `≤ (1+u)·((1+u)(BV+Br))²` (sub then square). -/
theorem valSqF_mag_le (V ret : Float) (BV Br : ℝ) (hV : |toReal V| ≤ BV) (hr : |toReal ret| ≤ Br) :
    |toReal (valSqF V ret)| ≤ (1 + u64) * (((1 + u64) * (BV + Br)) * ((1 + u64) * (BV + Br))) := by
  have hd : |toReal (V - ret)| ≤ (1 + u64) * (BV + Br) := sub_bound V ret BV Br hV hr
  show |toReal ((V - ret) * (V - ret))| ≤ _
  exact mul_bound (V - ret) (V - ret) _ _ hd hd

/-- **The value squared error is overflow-free** when its bound is within threshold. -/
theorem valSqF_isFinite (V ret : Float) (BV Br : ℝ) (hV : |toReal V| ≤ BV) (hr : |toReal ret| ≤ Br)
    (hbound : (1 + u64) * (((1 + u64) * (BV + Br)) * ((1 + u64) * (BV + Br))) ≤ overflowBound) :
    (valSqF V ret).isFinite = true :=
  isFinite_of_bounded _ ((valSqF_mag_le V ret BV Br hV hr).trans hbound)

/-- **The PPO ratio** `exp(newLogp − oldLogp)`. -/
def ratioF (newLogp oldLogp : Float) : Float := Float.exp (newLogp - oldLogp)

/-- **Ratio magnitude** from bounded log-probs: `≤ exp((1+u)(Bn+Bo))·(1+expEps)` — the log-prob difference is bounded
    above by `(1+u)(Bn+Bo)`, feeding `expF_mag_le` (C48). -/
theorem ratioF_mag_le (newLogp oldLogp : Float) (Bn Bo : ℝ)
    (hn : |toReal newLogp| ≤ Bn) (ho : |toReal oldLogp| ≤ Bo) :
    |toReal (ratioF newLogp oldLogp)| ≤ Real.exp ((1 + u64) * (Bn + Bo)) * (1 + expEps) := by
  have hle : toReal (newLogp - oldLogp) ≤ (1 + u64) * (Bn + Bo) :=
    (le_abs_self _).trans (sub_bound newLogp oldLogp Bn Bo hn ho)
  show |toReal (Float.exp (newLogp - oldLogp))| ≤ _
  exact expF_mag_le (newLogp - oldLogp) ((1 + u64) * (Bn + Bo)) hle

/-- **The PPO ratio is overflow-free** when `exp((1+u)(Bn+Bo))·(1+expEps) ≤ overflowBound`. -/
theorem ratioF_isFinite (newLogp oldLogp : Float) (Bn Bo : ℝ)
    (hn : |toReal newLogp| ≤ Bn) (ho : |toReal oldLogp| ≤ Bo)
    (hbound : Real.exp ((1 + u64) * (Bn + Bo)) * (1 + expEps) ≤ overflowBound) :
    (ratioF newLogp oldLogp).isFinite = true :=
  isFinite_of_bounded _ ((ratioF_mag_le newLogp oldLogp Bn Bo hn ho).trans hbound)

/-- **The clip** `clamp(a, lo, hi) = min (max a lo) hi` (as in `clampE`). -/
def clampF (a lo hi : Float) : Float := min (max a lo) hi

/-- **Clip magnitude** preserved (min/max add no roundoff): `|toReal (clampF a lo hi)| ≤ max (max |a| |lo|) |hi|`. -/
theorem clampF_mag_le (a lo hi : Float) :
    |toReal (clampF a lo hi)| ≤ max (max |toReal a| |toReal lo|) |toReal hi| :=
  (min_mag_le (max a lo) hi).trans (max_le_max (max_mag_le a lo) (le_refl _))

/-- **The clipped PPO surrogate** `min (g·r) (g·clamp(r,lo,hi))` (as in `ppoSurrogateE`). -/
def ppoSurrF (g r lo hi : Float) : Float := min (g * r) (g * clampF r lo hi)

/-- **Clipped surrogate magnitude** from bounded `g`, `r`, `lo`, `hi`: bounded by the max of the two `g`-scaled
    branches (the clip preserves magnitude, so its branch is `≤ (1+u)·Bg·(max (max Br Bl) Bh)`). -/
theorem ppoSurrF_mag_le (g r lo hi : Float) (Bg Br Bl Bh : ℝ)
    (hg : |toReal g| ≤ Bg) (hr : |toReal r| ≤ Br) (hl : |toReal lo| ≤ Bl) (hh : |toReal hi| ≤ Bh) :
    |toReal (ppoSurrF g r lo hi)|
      ≤ max ((1 + u64) * (Bg * Br)) ((1 + u64) * (Bg * (max (max Br Bl) Bh))) := by
  have hb1 : |toReal (g * r)| ≤ (1 + u64) * (Bg * Br) := mul_bound g r Bg Br hg hr
  have hclamp : |toReal (clampF r lo hi)| ≤ max (max Br Bl) Bh :=
    (clampF_mag_le r lo hi).trans (max_le_max (max_le_max hr hl) hh)
  have hb2 : |toReal (g * clampF r lo hi)| ≤ (1 + u64) * (Bg * (max (max Br Bl) Bh)) :=
    mul_bound g (clampF r lo hi) Bg (max (max Br Bl) Bh) hg hclamp
  exact (min_mag_le (g * r) (g * clampF r lo hi)).trans (max_le_max hb1 hb2)

/-- **The clipped PPO surrogate is overflow-free** when its bound is within threshold. -/
theorem ppoSurrF_isFinite (g r lo hi : Float) (Bg Br Bl Bh : ℝ)
    (hg : |toReal g| ≤ Bg) (hr : |toReal r| ≤ Br) (hl : |toReal lo| ≤ Bl) (hh : |toReal hi| ≤ Bh)
    (hbound : max ((1 + u64) * (Bg * Br)) ((1 + u64) * (Bg * (max (max Br Bl) Bh))) ≤ overflowBound) :
    (ppoSurrF g r lo hi).isFinite = true :=
  isFinite_of_bounded _ ((ppoSurrF_mag_le g r lo hi Bg Br Bl Bh hg hr hl hh).trans hbound)

/-- **The softmax partition** `Σ exp(logit)`, accumulated as `exp x + rest`. -/
def sumExpF : List Float → Float
  | [] => 0
  | x :: xs => Float.exp x + sumExpF xs

/-- The propagated partition magnitude bound: `n` `exp`-terms each `≤ t`, each add inflating by `(1+u)`. -/
noncomputable def sumBound : ℕ → ℝ → ℝ
  | 0, _ => 0
  | n + 1, t => (1 + u64) * (t + sumBound n t)

/-- **Partition magnitude propagation** from logits bounded ABOVE (`toReal a ≤ M`): each `exp` term `≤ exp(M)·
    (1+expEps)` (C48's `expF_mag_le`), summed through the `(1+u)`-inflating fold. -/
theorem sumExpF_mag_le (M : ℝ) : ∀ (logits : List Float), (∀ a ∈ logits, toReal a ≤ M) →
    |toReal (sumExpF logits)| ≤ sumBound logits.length (Real.exp M * (1 + expEps))
  | [], _ => by simp [sumExpF, sumBound]
  | x :: xs, hlog => by
      have hhead : toReal x ≤ M := hlog x (List.mem_cons.mpr (Or.inl rfl))
      have htail := sumExpF_mag_le M xs (fun a ha => hlog a (List.mem_cons.mpr (Or.inr ha)))
      have hx : |toReal (Float.exp x)| ≤ Real.exp M * (1 + expEps) := expF_mag_le x M hhead
      show |toReal (Float.exp x + sumExpF xs)| ≤ sumBound (xs.length + 1) (Real.exp M * (1 + expEps))
      rw [sumBound]
      calc |toReal (Float.exp x + sumExpF xs)|
          ≤ (1 + u64) * (|toReal (Float.exp x)| + |toReal (sumExpF xs)|) := add_mag_le _ _
        _ ≤ (1 + u64) * (Real.exp M * (1 + expEps) + sumBound xs.length (Real.exp M * (1 + expEps))) :=
            mul_le_mul_of_nonneg_left (add_le_add hx htail) one_add_u64_nonneg

/-- Each `Float.exp` term is strictly positive (`exp > 0`, `1+δ > 0` since `expEps < 1`). -/
theorem expF_pos (x : Float) : 0 < toReal (Float.exp x) := by
  obtain ⟨δ, hδ, he⟩ := exp_model x
  rw [he]
  have h1 : 0 < 1 + δ := by
    have hlt : expEps < 1 := by unfold expEps; norm_num
    have := abs_le.mp hδ; linarith [this.1]
  exact mul_pos (Real.exp_pos _) h1

/-- The partition is nonnegative (sum of positive `exp` terms; `add_model`'s `1+δ > 0`). -/
theorem sumExpF_nonneg : ∀ (xs : List Float), 0 ≤ toReal (sumExpF xs)
  | [] => by simp [sumExpF]
  | x :: xs => by
      show 0 ≤ toReal (Float.exp x + sumExpF xs)
      obtain ⟨δ, hδ, he⟩ := add_model (Float.exp x) (sumExpF xs)
      rw [he]
      have h1 : 0 ≤ 1 + δ := by have := u64_lt_one; have := abs_le.mp hδ; linarith [this.1]
      exact mul_nonneg (add_nonneg (expF_pos x).le (sumExpF_nonneg xs)) h1

/-- **The partition is strictly positive** for a non-empty logit list — the floor the `log` needs. -/
theorem sumExpF_pos (x : Float) (xs : List Float) : 0 < toReal (sumExpF (x :: xs)) := by
  show 0 < toReal (Float.exp x + sumExpF xs)
  obtain ⟨δ, hδ, he⟩ := add_model (Float.exp x) (sumExpF xs)
  rw [he]
  have h1 : 0 < 1 + δ := by have := u64_lt_one; have := abs_le.mp hδ; linarith [this.1]
  exact mul_pos (add_pos_of_pos_of_nonneg (expF_pos x) (sumExpF_nonneg xs)) h1

/-- **The softmax partition is overflow-free** from logits bounded above, when the bound is within threshold. -/
theorem sumExpF_isFinite (M : ℝ) (logits : List Float) (hlog : ∀ a ∈ logits, toReal a ≤ M)
    (hbound : sumBound logits.length (Real.exp M * (1 + expEps)) ≤ overflowBound) :
    (sumExpF logits).isFinite = true :=
  isFinite_of_bounded _ ((sumExpF_mag_le M logits hlog).trans hbound)

/-- A value in `[lo, hi]` has `|·| ≤ max |lo| |hi|`. -/
theorem abs_le_max_of_mem {lo hi y : ℝ} (h1 : lo ≤ y) (h2 : y ≤ hi) : |y| ≤ max |lo| |hi| := by
  rw [abs_le]
  refine ⟨?_, ?_⟩
  · calc -max |lo| |hi| ≤ -|lo| := neg_le_neg (le_max_left _ _)
      _ ≤ lo := neg_abs_le lo
      _ ≤ y := h1
  · calc y ≤ hi := h2
      _ ≤ |hi| := le_abs_self hi
      _ ≤ max |lo| |hi| := le_max_right _ _

/-- **Log magnitude on a floored, bounded argument** (`log_model`, needs `0 < toReal a`): with `c ≤ toReal a ≤ U`,
    `c > 0`, `|toReal (Float.log a)| ≤ max |log c| |log U|·(1+logEps)` (`log` monotone on positives). -/
theorem logMag_le (a : Float) (c U : ℝ) (hc : 0 < c) (hca : c ≤ toReal a) (haU : toReal a ≤ U) :
    |toReal (Float.log a)| ≤ max |Real.log c| |Real.log U| * (1 + logEps) := by
  have hapos : 0 < toReal a := lt_of_lt_of_le hc hca
  obtain ⟨δ, hδ, he⟩ := log_model a hapos
  rw [he, abs_mul]
  have h1 : |1 + δ| ≤ 1 + logEps := (abs_add_le _ _).trans (by simpa using add_le_add_left hδ 1)
  have hlogbound : |Real.log (toReal a)| ≤ max |Real.log c| |Real.log U| :=
    abs_le_max_of_mem (Real.log_le_log hc hca) (Real.log_le_log hapos haU)
  exact mul_le_mul hlogbound h1 (abs_nonneg _) ((abs_nonneg _).trans hlogbound)

/-- **THE LOG-PARTITION NO-OVERFLOW CERTIFICATE.** `log(Σ exp(logit))` is overflow-free given the logits bounded
    ABOVE (`toReal a ≤ M`, so the partition `≤ U = sumBound …`) and the partition floored `≥ c > 0` (the checkable
    side-condition `log` needs — `sumExpF_pos` shows the partition is positive), when `max |log c| |log U|·(1+logEps)
    ≤ overflowBound`. Reaches the one PPO loss term with a genuine domain side-condition, via the pre-existing
    `log_model`. NO new axiom. -/
theorem logPart_isFinite (logits : List Float) (M c : ℝ)
    (hlog : ∀ a ∈ logits, toReal a ≤ M) (hc : 0 < c) (hfloor : c ≤ toReal (sumExpF logits))
    (hbound : max |Real.log c| |Real.log (sumBound logits.length (Real.exp M * (1 + expEps)))| * (1 + logEps)
        ≤ overflowBound) :
    (Float.log (sumExpF logits)).isFinite = true := by
  have hU : toReal (sumExpF logits) ≤ sumBound logits.length (Real.exp M * (1 + expEps)) :=
    (le_abs_self _).trans (sumExpF_mag_le M logits hlog)
  exact isFinite_of_bounded _ ((logMag_le (sumExpF logits) c _ hc hfloor hU).trans hbound)

end Puffer.RL.LossForwardFinite
