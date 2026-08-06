/-
PPO losses over ℝ — faithful specs of PufferLib's policy/value/entropy math
(`~/src/PufferLib/src/pufferlib.cu`, loss kernel ~lines 900–960).

Definitions mirror the code exactly (clamp = `fmaxf(lo, fminf(hi, x))`); theorems
capture the meaningful structural properties: the clipped surrogate is pessimistic
and caps the ratio, the value loss is the pessimistic (max) of clipped/unclipped,
and the Schulman-k3 KL estimator is provably nonnegative.
-/
import Mathlib

namespace Puffer.RL.PPO

/-- `clamp x` to `[lo, hi]`, i.e. `fmaxf(lo, fminf(hi, x))`. -/
def clampR (x lo hi : ℝ) : ℝ := max lo (min x hi)

/-- The clamp never falls below its lower bound (`max` picks `≥ lo` unconditionally). -/
theorem clampR_ge_lo (x lo hi : ℝ) : lo ≤ clampR x lo hi := le_max_left _ _

/-- The clamp never exceeds its upper bound, provided `lo ≤ hi` (`min _ hi ≤ hi` and `lo ≤ hi`). -/
theorem clampR_le_hi (x lo hi : ℝ) (h : lo ≤ hi) : clampR x lo hi ≤ hi :=
  max_le h (min_le_right _ _)

/-- **Clamp range.** For `lo ≤ hi`, `clampR x lo hi ∈ [lo, hi]` — the fundamental clamp guarantee. -/
theorem clampR_mem (x lo hi : ℝ) (h : lo ≤ hi) :
    lo ≤ clampR x lo hi ∧ clampR x lo hi ≤ hi :=
  ⟨clampR_ge_lo x lo hi, clampR_le_hi x lo hi h⟩

/-- **Clamp is the identity in-range.** `lo ≤ x ≤ hi → clampR x lo hi = x` — clipping only activates outside
    the interval. -/
theorem clampR_eq_of_mem (x lo hi : ℝ) (h1 : lo ≤ x) (h2 : x ≤ hi) : clampR x lo hi = x := by
  unfold clampR; rw [min_eq_left h2, max_eq_right h1]

/-- Policy ratio `r = exp(new_logp − old_logp)`. -/
noncomputable def ratio (newLogp oldLogp : ℝ) : ℝ := Real.exp (newLogp - oldLogp)

theorem ratio_pos (newLogp oldLogp : ℝ) : 0 < ratio newLogp oldLogp := Real.exp_pos _

/-- **PPO ratio trust region.** The clipped ratio `clip(r, 1−ε, 1+ε)` always lies in `[1−ε, 1+ε]` for `ε ≥ 0`
    — the core PPO guarantee that the surrogate's clipped branch caps how far the policy ratio can move the
    objective. Instance of `clampR_mem` (with `1−ε ≤ 1+ε` from `ε ≥ 0`). -/
theorem ratio_clip_range (r ε : ℝ) (hε : 0 ≤ ε) :
    1 - ε ≤ clampR r (1 - ε) (1 + ε) ∧ clampR r (1 - ε) (1 + ε) ≤ 1 + ε :=
  clampR_mem r (1 - ε) (1 + ε) (by linarith)

/-! ### Clipped surrogate objective

PufferLib computes `pg_loss = max(-g·r, -g·clip(r))` with `g = w·adv_normalized`
(`w ≥ 0`), i.e. it minimizes the negation of the surrogate objective below. -/

/-- The PPO clipped surrogate objective (maximized): `min(g·r, g·clip(r, 1−ε, 1+ε))`. -/
def ppoObjective (g r ε : ℝ) : ℝ := min (g * r) (g * clampR r (1 - ε) (1 + ε))

/-- The clipped surrogate with explicit clip bounds `[lo,hi]` (generalizes
    `ppoObjective`, which is `lo = 1−ε, hi = 1+ε`). -/
def ppoObjLoHi (g r lo hi : ℝ) : ℝ := min (g * r) (g * clampR r lo hi)

theorem ppoObjective_eq_LoHi (g r ε : ℝ) : ppoObjective g r ε = ppoObjLoHi g r (1 - ε) (1 + ε) := rfl

/-- **The surrogate is unclipped when the ratio is in range.** For `lo ≤ r ≤ hi`, the clip is inactive so
    `ppoObjLoHi g r lo hi = g·r` — both `min` branches coincide (`clampR_eq_of_mem` + `min_self`). -/
theorem ppoObjLoHi_eq_of_mem (g r lo hi : ℝ) (h1 : lo ≤ r) (h2 : r ≤ hi) :
    ppoObjLoHi g r lo hi = g * r := by
  unfold ppoObjLoHi
  rw [clampR_eq_of_mem r lo hi h1 h2, min_self]

/-- **The PPO surrogate equals the plain objective inside the trust region.** When the ratio `r ∈ [1−ε, 1+ε]`
    the clip does nothing: `ppoObjective g r ε = g·r`. So clipping only alters the objective once the policy has
    moved the ratio outside `1 ± ε` — the value-side analogue is `valueClipped_eq_of_close`. -/
theorem ppoObjective_eq_of_close (g r ε : ℝ) (h1 : 1 - ε ≤ r) (h2 : r ≤ 1 + ε) :
    ppoObjective g r ε = g * r := by
  rw [ppoObjective_eq_LoHi]
  exact ppoObjLoHi_eq_of_mem g r (1 - ε) (1 + ε) h1 h2

/-- The objective is **pessimistic**: never above the unclipped surrogate `g·r`. -/
theorem ppoObjective_le_unclipped (g r ε : ℝ) : ppoObjective g r ε ≤ g * r :=
  min_le_left _ _

/-- Lattice identity: clamping inside a `min` with the same argument collapses to the
    upper cap. `min r (clamp r lo hi) = min r hi` whenever `lo ≤ hi`. -/
theorem min_clamp (r lo hi : ℝ) (h : lo ≤ hi) :
    min r (clampR r lo hi) = min r hi := by
  unfold clampR
  simp only [min_def, max_def]
  split_ifs <;> linarith

/-- **Clip characterization (positive advantage).** With `g ≥ 0` and `ε ≥ 0`, the
    clipped objective is `g · min(r, 1+ε)` — the upside of the ratio is capped at
    `1+ε`, removing the incentive to push the policy ratio arbitrarily high. -/
theorem ppoObjective_of_nonneg (g r ε : ℝ) (hg : 0 ≤ g) (hε : 0 ≤ ε) :
    ppoObjective g r ε = g * min r (1 + ε) := by
  unfold ppoObjective
  rw [← mul_min_of_nonneg r (clampR r (1 - ε) (1 + ε)) hg, min_clamp r (1 - ε) (1 + ε) (by linarith)]

/-- **Clipping flattens the objective above the trust region (positive advantage).** For `g ≥ 0` and a ratio
    that has run past the upper clip `r ≥ 1+ε`, the surrogate is pinned at `g·(1+ε)` — INDEPENDENT of `r`. So once
    the policy has moved the probability of a good action too far up, pushing it further yields no additional
    objective (zero gradient). This is the core PPO mechanism that removes the incentive for destructive updates
    on the upside (`ppoObjective_of_nonneg` + `min_eq_right`). -/
theorem ppoObjective_clipped_above (g r ε : ℝ) (hg : 0 ≤ g) (hε : 0 ≤ ε) (hr : 1 + ε ≤ r) :
    ppoObjective g r ε = g * (1 + ε) := by
  rw [ppoObjective_of_nonneg g r ε hg hε, min_eq_right hr]

/-- **Clipping flattens the objective below the trust region (negative advantage).** For `g ≤ 0` and a ratio
    below the lower clip `r ≤ 1−ε`, the surrogate is pinned at `g·(1−ε)` — INDEPENDENT of `r`. The symmetric PPO
    mechanism: for a bad action (`g < 0`) the objective wants the ratio small, but once it drops past `1−ε` the
    clip caps the reward, killing the incentive to keep shrinking it. No `hε` needed — `min(r,1+ε) ≤ r ≤ 1−ε`
    pins the clamp regardless of `ε`'s sign. -/
theorem ppoObjective_clipped_below (g r ε : ℝ) (hg : g ≤ 0) (hr : r ≤ 1 - ε) :
    ppoObjective g r ε = g * (1 - ε) := by
  unfold ppoObjective
  have hle : min r (1 + ε) ≤ 1 - ε := le_trans (min_le_left _ _) hr
  have hclamp : clampR r (1 - ε) (1 + ε) = 1 - ε := by
    unfold clampR; rw [max_eq_left hle]
  rw [hclamp]
  have hge : g * (1 - ε) ≤ g * r := mul_le_mul_of_nonpos_left hr hg
  rw [min_eq_right hge]

/-! ### Value loss -/

/-- Value clipped to a trust region around the old value: `val + clamp(Δ, −c, c)`. -/
def valueClipped (val valPred vfClip : ℝ) : ℝ :=
  val + clampR (valPred - val) (-vfClip) vfClip

/-- **Value-clip trust region.** For `vfClip ≥ 0`, the clipped value stays within `±vfClip` of the old value:
    `valueClipped val valPred vfClip ∈ [val − vfClip, val + vfClip]` — the update to the value prediction is
    capped (`clampR_mem`), the value-side counterpart of `ratio_clip_range`. -/
theorem valueClipped_mem (val valPred vfClip : ℝ) (h : 0 ≤ vfClip) :
    val - vfClip ≤ valueClipped val valPred vfClip
      ∧ valueClipped val valPred vfClip ≤ val + vfClip := by
  unfold valueClipped
  have hc := clampR_mem (valPred - val) (-vfClip) vfClip (by linarith)
  exact ⟨by linarith [hc.1], by linarith [hc.2]⟩

/-- **Value clip is inactive when the prediction is close.** If `|valPred − val| ≤ vfClip`, the clip is a
    no-op: `valueClipped val valPred vfClip = valPred` — clipping only changes predictions that move more than
    `vfClip` from the old value (`clampR_eq_of_mem`). -/
theorem valueClipped_eq_of_close (val valPred vfClip : ℝ) (h : |valPred - val| ≤ vfClip) :
    valueClipped val valPred vfClip = valPred := by
  unfold valueClipped
  rw [abs_le] at h
  rw [clampR_eq_of_mem (valPred - val) (-vfClip) vfClip (by linarith [h.1]) h.2]
  ring

/-- `v_loss = 0.5 · max((val_pred−ret)², (v_clipped−ret)²)`. -/
def valueLoss (val valPred ret vfClip : ℝ) : ℝ :=
  0.5 * max ((valPred - ret) ^ 2) ((valueClipped val valPred vfClip - ret) ^ 2)

theorem valueLoss_nonneg (val valPred ret vfClip : ℝ) :
    0 ≤ valueLoss val valPred ret vfClip := by
  unfold valueLoss
  exact mul_nonneg (by norm_num) (le_max_of_le_left (sq_nonneg _))

/-- The value loss is pessimistic: at least the (halved) unclipped squared error. -/
theorem valueLoss_ge_half_unclipped (val valPred ret vfClip : ℝ) :
    0.5 * (valPred - ret) ^ 2 ≤ valueLoss val valPred ret vfClip := by
  unfold valueLoss
  exact mul_le_mul_of_nonneg_left (le_max_left _ _) (by norm_num)

/-- **The value loss vanishes exactly when both value predictions hit the return.**
    `valueLoss = 0 ↔ valPred = ret ∧ valueClipped = ret` — since the loss is `½·max` of two squared errors, it is
    zero iff *both* the unclipped prediction and the clipped prediction equal the target return. This pins the
    global minimum of PufferLib's pessimistic (max-of-clipped) value objective: the second conjunct is
    load-bearing — `valPred = ret` alone does not force zero loss (the clipped branch can still miss, e.g.
    `val=0, valPred=ret=5, vfClip=1` gives clipped value `1` and loss `8`). The value counterpart of
    `approxKLNew_eq_zero_iff` on the KL side. -/
theorem valueLoss_eq_zero_iff (val valPred ret vfClip : ℝ) :
    valueLoss val valPred ret vfClip = 0 ↔
      valPred = ret ∧ valueClipped val valPred vfClip = ret := by
  unfold valueLoss
  rw [mul_eq_zero]
  have hA := sq_nonneg (valPred - ret)
  have hB := sq_nonneg (valueClipped val valPred vfClip - ret)
  constructor
  · rintro (h | h)
    · norm_num at h
    · have hA0 : (valPred - ret) ^ 2 = 0 := le_antisymm (h ▸ le_max_left _ _) hA
      have hB0 : (valueClipped val valPred vfClip - ret) ^ 2 = 0 :=
        le_antisymm (h ▸ le_max_right _ _) hB
      refine ⟨?_, ?_⟩
      · have : valPred - ret = 0 := pow_eq_zero_iff (by norm_num) |>.mp hA0
        linarith
      · have : valueClipped val valPred vfClip - ret = 0 := pow_eq_zero_iff (by norm_num) |>.mp hB0
        linarith
  · rintro ⟨h1, h2⟩
    right
    subst h1
    rw [h2]
    simp

/-! ### KL estimators and entropy -/

/-- Old approximate KL (single-sample term): `−logratio`. -/
def approxKLOld (logratio : ℝ) : ℝ := -logratio

/-- New approximate KL, Schulman's k3 estimator (single-sample): `(exp(lr) − 1) − lr`. -/
noncomputable def approxKLNew (logratio : ℝ) : ℝ := (Real.exp logratio - 1) - logratio

/-- **The k3 KL estimator is always nonnegative** — a provable soundness property of
    the estimator PufferLib uses, from `1 + x ≤ exp x`. -/
theorem approxKLNew_nonneg (logratio : ℝ) : 0 ≤ approxKLNew logratio := by
  unfold approxKLNew
  have := Real.add_one_le_exp logratio
  linarith

/-- **The k3 KL estimator is strictly positive off the identity** — for any nonzero log-ratio it is `> 0`
    (`1 + x < exp x` strictly when `x ≠ 0`). -/
theorem approxKLNew_pos_of_ne (logratio : ℝ) (h : logratio ≠ 0) : 0 < approxKLNew logratio := by
  unfold approxKLNew
  have := Real.add_one_lt_exp h
  linarith

/-- **The k3 KL estimator vanishes exactly at zero log-ratio.** `approxKLNew lr = 0 ↔ lr = 0` — together with
    `approxKLNew_nonneg`, this makes it a genuine divergence: nonnegative, and zero iff the new and old
    policies agree (`lr = new_logp − old_logp = 0`). -/
theorem approxKLNew_eq_zero_iff (logratio : ℝ) : approxKLNew logratio = 0 ↔ logratio = 0 := by
  constructor
  · intro heq
    by_contra hx
    exact absurd heq (ne_of_gt (approxKLNew_pos_of_ne logratio hx))
  · intro h; subst h; unfold approxKLNew; simp

/-- **The k3 KL estimator is monotone on nonnegative log-ratios.** For `0 ≤ lr1 ≤ lr2`,
    `approxKLNew lr1 ≤ approxKLNew lr2` — the estimator grows as the log-ratio moves away from `0` on the positive
    side (its derivative `exp lr − 1 ≥ 0` there). Together with `approxKLNew_eq_zero_iff` (minimum `0` at `lr = 0`),
    this shows the estimator increases monotonically with the policy's forward divergence: a larger positive
    log-ratio yields a larger measured KL, so PPO's early-stopping trigger fires monotonically as the new policy
    pulls ahead of the old. `h0 : 0 ≤ lr1` is load-bearing — on the negative branch the estimator *decreases*
    toward `0` (it is convex with its minimum at `0`). -/
theorem approxKLNew_mono_of_nonneg (lr1 lr2 : ℝ) (h0 : 0 ≤ lr1) (hle : lr1 ≤ lr2) :
    approxKLNew lr1 ≤ approxKLNew lr2 := by
  unfold approxKLNew
  have hsub : 0 ≤ lr2 - lr1 := by linarith
  have key : (lr2 - lr1) + 1 ≤ Real.exp (lr2 - lr1) := Real.add_one_le_exp (lr2 - lr1)
  have one_le : (1 : ℝ) ≤ Real.exp lr1 := Real.one_le_exp h0
  have e1pos : 0 < Real.exp lr1 := Real.exp_pos lr1
  have prod : Real.exp lr1 * Real.exp (lr2 - lr1) = Real.exp lr2 := by
    rw [← Real.exp_add]; ring_nf
  nlinarith [mul_le_mul_of_nonneg_left key (le_of_lt e1pos),
    mul_nonneg (sub_nonneg.2 one_le) hsub]

/-- Continuous (Gaussian) differential entropy: `½(1 + log 2π) + log σ` (`σ = e^{logStd}`). -/
noncomputable def gaussianEntropy (logStd : ℝ) : ℝ := 0.5 * (1 + Real.log (2 * Real.pi)) + logStd

/-- **Gaussian entropy is monotone in `logStd`.** A wider policy (larger `logStd` ⇒ larger `σ`) has more
    differential entropy — `gaussianEntropy` is `logStd` plus a constant, so it is strictly increasing. -/
theorem gaussianEntropy_mono (logStd1 logStd2 : ℝ) (h : logStd1 ≤ logStd2) :
    gaussianEntropy logStd1 ≤ gaussianEntropy logStd2 := by
  unfold gaussianEntropy; linarith

/-- Gaussian log-prob, exactly as in the C: `−½·((a−μ)/σ)² − ½log 2π − logStd`. -/
noncomputable def gaussianLogp (action mean logStd : ℝ) : ℝ :=
  -0.5 * ((action - mean) / Real.exp logStd) ^ 2 - 0.5 * Real.log (2 * Real.pi) - logStd

/-- **The Gaussian log-density is maximized at the mean.** For any action, `gaussianLogp action mean logStd ≤
    gaussianLogp mean mean logStd` — the density peaks at `μ` (the subtracted squared term is `≥ 0` and
    vanishes there). -/
theorem gaussianLogp_le_at_mean (action mean logStd : ℝ) :
    gaussianLogp action mean logStd ≤ gaussianLogp mean mean logStd := by
  unfold gaussianLogp
  simp only [sub_self, zero_div]
  nlinarith [sq_nonneg ((action - mean) / Real.exp logStd)]

/-- **The Gaussian log-density peaks UNIQUELY at the mean.** `gaussianLogp action mean logStd =
    gaussianLogp mean mean logStd ↔ action = mean` — the mode is exactly `μ` (`σ = e^{logStd} > 0`, so the
    squared term is zero iff `action = mean`). -/
theorem gaussianLogp_eq_at_mean_iff (action mean logStd : ℝ) :
    gaussianLogp action mean logStd = gaussianLogp mean mean logStd ↔ action = mean := by
  have hexp : Real.exp logStd ≠ 0 := Real.exp_ne_zero logStd
  unfold gaussianLogp
  constructor
  · intro h
    have hm : ((mean - mean) / Real.exp logStd) ^ 2 = 0 := by simp
    have hsq : ((action - mean) / Real.exp logStd) ^ 2 = 0 := by
      nlinarith [sq_nonneg ((action - mean) / Real.exp logStd), h, hm]
    have h0 : (action - mean) / Real.exp logStd = 0 := by
      simpa using pow_eq_zero_iff (n := 2) (by norm_num) |>.mp hsq
    rw [div_eq_zero_iff] at h0
    rcases h0 with h0 | h0
    · linarith
    · exact absurd h0 hexp
  · intro h; subst h; rfl

/-- **The Gaussian log-density is symmetric about the mean.** `gaussianLogp (mean + d) mean logStd =
    gaussianLogp (mean − d) mean logStd` — equal deviations either side of `μ` have equal log-probability, the
    reflection symmetry of the Gaussian (the density depends on `action` only through `(action − mean)²`). -/
theorem gaussianLogp_symm (mean logStd d : ℝ) :
    gaussianLogp (mean + d) mean logStd = gaussianLogp (mean - d) mean logStd := by
  unfold gaussianLogp; ring

/-- **The Gaussian log-density decreases with distance from the mean.** If `a1` is at least as close to `μ` as
    `a2` (`|a1 − mean| ≤ |a2 − mean|`), then `gaussianLogp a2 ≤ gaussianLogp a1` — the density is a monotone
    function of `|action − μ|`. Generalizes `gaussianLogp_le_at_mean` (the `a1 = mean` case). The squared
    deviation `(·−mean)²` is monotone in `|·−mean|`, and it enters `gaussianLogp` with a negative coefficient. -/
theorem gaussianLogp_le_of_abs_le (a1 a2 mean logStd : ℝ) (h : |a1 - mean| ≤ |a2 - mean|) :
    gaussianLogp a2 mean logStd ≤ gaussianLogp a1 mean logStd := by
  have hsq : (a1 - mean) ^ 2 ≤ (a2 - mean) ^ 2 := by
    rw [← sq_abs (a1 - mean), ← sq_abs (a2 - mean)]
    exact pow_le_pow_left₀ (abs_nonneg _) h 2
  have hσ : (0:ℝ) < Real.exp logStd ^ 2 := by positivity
  unfold gaussianLogp
  have key : ((a1 - mean) / Real.exp logStd) ^ 2 ≤ ((a2 - mean) / Real.exp logStd) ^ 2 := by
    rw [div_pow, div_pow]
    gcongr
  linarith

end Puffer.RL.PPO
