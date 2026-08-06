/-
Closing the trifecta on PPO: the executable `Float` PPO quantities are within a
certified bound of their ℝ specs.

Pieces, using the extended Float axioms:
* the **ratio** `exp(logratio)` — bounded via the `Float.exp` (1+δ) model
  (`ratioF_error`); and the FULL ratio `exp(newLogp − oldLogp)` from the two log-probs
  (`ratioFullF_error`, the sub + exp the trainer actually computes); and the ratio vs the
  IDEAL ratio (`ratioFullF_ideal_error`, composing the `exp` rounding with the `exp`
  local-Lipschitz `abs_exp_sub_exp_le` over the log-prob error — discharge `ε` via `logpF_ideal_error`, a110);
* the **clipped surrogate** `min(g·r, g·clip(r,lo,hi))` — min/max are exact
  (`toReal_min`/`toReal_max`), so only the two products round; bounded via `mul_error`
  and the 1-Lipschitzness of `min` (`ppoSurrF_error`);
* the **clipped value** `val + clip(valPred−val, −c, c)` — again min/max exact, only the
  inner sub and outer add round (`valueClippedF_error`, via `clampR_lipschitz`);
* the **value loss** `0.5·max((valPred−ret)², (vClipped−ret)²)` — the whole squared-error
  circuit (two subs, two `x*x` squares, exact `max`, and the `0.5` scale whose
  `ofScientific` literal gap folds in via `toReal_ofScientific_close`) — `valueLossF_error`,
  composing `valueClippedF_error` through the second square; and the UNCLIPPED `½·(val−ret)²`
  (`valueSqLossF_error`) — the plain value term `mlpGradPPO`'s tape uses (`scale (vfCoef·0.5) diff²`);
* the **approximate-KL estimators** (Schulman) — `approxKLOld = −logratio` is EXACT
  (`approxKLOldF_error`, an equality via `toReal_neg`); `approxKLNew = (exp lr − 1) − lr`
  rounds only at `exp` and the two subs (`approxKLNewF_error`). Both also bounded vs their IDEAL-policy
  values given the log-ratio error `ε` (`approxKLOldF_ideal_error` — exact, so `≤ ε`; `approxKLNewF_ideal_error`
  — rounding budget `+` the `exp`-Lipschitz perturbation `exp(lrIdeal)·(eᵋ−1)+ε`, via `approxKLNew_perturb`),
  the early-stopping metric's accuracy; discharge `ε` via the forward log-ratio bound (like a115→a119).

Beyond the error bounds, several **runtime soundness** guarantees hold for ALL Float inputs (no error budget):
`ppoSurrF_le_clipped` — the surrogate never exceeds the CLIPPED branch `g·clamp(r,lo,hi)` (companion to
`ppoSurrF_le_unclipped`, so it sits below both `min` arguments); `valueSqLossF_nonneg` / `valueLossF_nonneg` —
the unclipped `½·(val−ret)²` and the clipped `0.5·max(·²,·²)` value losses stay `≥ 0` despite the roundings
(`0.5 > 0`, `1+δ > 0` since `|δ| ≤ u64 < 1`; the clipped case adds `max`-of-nonneg-squares); and
`valueClip_offset_range` — the value-clip correction stays within `[−vfClip, vfClip]` (the trust region).

Together with `Puffer/RL/PPO.lean` (ℝ spec + structural theorems), the PPO objective
now has all three layers: ℝ spec · running Float · proven error bound.
-/
import Mathlib
import Puffer.Float.Basic
import Puffer.RL.PPO
import Puffer.RL.PPOPerturb

namespace Puffer.RL.PPORuntime

open Puffer.FloatR
open Puffer.RL.PPO (clampR ppoObjLoHi valueClipped valueLoss approxKLOld approxKLNew)
open Puffer.RL.PPOPerturb (clampR_lipschitz)

/-! ### PPO ratio (uses the `exp` axiom) -/

/-- Executable PPO ratio `ρ = exp(logratio)`. -/
def ratioF (lr : Float) : Float := Float.exp lr

/-- **PPO ratio error**: the running ratio is within relative `expEps` of the exact
    real ratio `exp(logratio)`. -/
theorem ratioF_error (lr : Float) :
    |toReal (ratioF lr) - Real.exp (toReal lr)| ≤ expEps * Real.exp (toReal lr) :=
  exp_error lr

/-- Executable FULL PPO ratio from the two log-probs: `ρ = exp(newLogp − oldLogp)` — what the trainer
    computes (vs `ratioF`, which takes the log-ratio precomputed). -/
def ratioFullF (newLogp oldLogp : Float) : Float := Float.exp (newLogp - oldLogp)

/-- **Full PPO ratio error.** The running `exp(newLogp − oldLogp)` deviates from the ideal
    `exp(toReal newLogp − toReal oldLogp)` by at most the outer `exp` rounding (`expEps`) plus `exp`'s
    Lipschitz factor over the inner subtraction's rounding `u64·|Δ|` (`sub_error` + `expApprox_error`). -/
theorem ratioFullF_error (newLogp oldLogp : Float) :
    |toReal (ratioFullF newLogp oldLogp) - Real.exp (toReal newLogp - toReal oldLogp)|
      ≤ expEps * Real.exp (toReal (newLogp - oldLogp))
        + Real.exp (toReal (newLogp - oldLogp))
            * (Real.exp (u64 * |toReal newLogp - toReal oldLogp|) - 1) := by
  have hd : |toReal (newLogp - oldLogp) - (toReal newLogp - toReal oldLogp)|
      ≤ u64 * |toReal newLogp - toReal oldLogp| := sub_error newLogp oldLogp
  exact expApprox_error (newLogp - oldLogp) (toReal newLogp - toReal oldLogp)
    (u64 * |toReal newLogp - toReal oldLogp|) hd

/-! ### Ratio vs the ideal ratio (exp perturbation) -/

/-- **`exp` local-Lipschitz.** `|exp x − exp y| ≤ exp y · (exp ε − 1)` when `|x − y| ≤ ε` — `exp x − exp y =
    exp y · (exp(x−y) − 1)` and `|exp(x−y) − 1| ≤ exp ε − 1` (`abs_one_sub_exp_le`). -/
theorem abs_exp_sub_exp_le (x y ε : ℝ) (h : |x - y| ≤ ε) :
    |Real.exp x - Real.exp y| ≤ Real.exp y * (Real.exp ε - 1) := by
  have hfac : Real.exp x - Real.exp y = Real.exp y * (Real.exp (x - y) - 1) := by
    rw [mul_sub, mul_one, ← Real.exp_add]; ring_nf
  rw [hfac, abs_mul, abs_of_pos (Real.exp_pos y)]
  refine mul_le_mul_of_nonneg_left ?_ (Real.exp_pos y).le
  have h1 := abs_one_sub_exp_le h
  rwa [abs_sub_comm] at h1

/-- **Runnable PPO ratio vs the ideal ratio.** The ratio the trainer computes — `ratioFullF newLogp oldLogp`
    — is within a107's `exp`-rounding budget `+ exp`-perturbation of the IDEAL ratio `exp(logpIdeal − oldLogp)`,
    given `newLogp` is within `ε` of the ideal log-policy `logpIdeal` (discharge `ε` via `logpF_ideal_error`,
    a110). Composes `ratioFullF_error` (a107) with `abs_exp_sub_exp_le` through the exponent. -/
theorem ratioFullF_ideal_error (newLogp oldLogp : Float) (logpIdeal ε : ℝ)
    (h : |toReal newLogp - logpIdeal| ≤ ε) :
    |toReal (ratioFullF newLogp oldLogp) - Real.exp (logpIdeal - toReal oldLogp)|
      ≤ (expEps * Real.exp (toReal (newLogp - oldLogp))
          + Real.exp (toReal (newLogp - oldLogp))
              * (Real.exp (u64 * |toReal newLogp - toReal oldLogp|) - 1))
        + Real.exp (logpIdeal - toReal oldLogp) * (Real.exp ε - 1) := by
  have h107 := ratioFullF_error newLogp oldLogp
  have hexp : |Real.exp (toReal newLogp - toReal oldLogp) - Real.exp (logpIdeal - toReal oldLogp)|
      ≤ Real.exp (logpIdeal - toReal oldLogp) * (Real.exp ε - 1) := by
    apply abs_exp_sub_exp_le
    calc |(toReal newLogp - toReal oldLogp) - (logpIdeal - toReal oldLogp)|
        = |toReal newLogp - logpIdeal| := by ring_nf
      _ ≤ ε := h
  calc |toReal (ratioFullF newLogp oldLogp) - Real.exp (logpIdeal - toReal oldLogp)|
      ≤ |toReal (ratioFullF newLogp oldLogp) - Real.exp (toReal newLogp - toReal oldLogp)|
        + |Real.exp (toReal newLogp - toReal oldLogp) - Real.exp (logpIdeal - toReal oldLogp)| :=
        abs_sub_le _ _ _
    _ ≤ _ := add_le_add h107 hexp

/-- **Runtime ratio positivity.** The running ratio `exp(logratio)` is strictly positive — the runtime
    counterpart of `PPO.ratio_pos`. (`exp` is positive and the `(1+δ)` rounding factor stays `> 0`.) -/
theorem ratioF_pos (lr : Float) : 0 < toReal (ratioF lr) := by
  unfold ratioF
  obtain ⟨δ, hδ, he⟩ := exp_model lr
  rw [he]
  refine mul_pos (Real.exp_pos _) ?_
  have h1 : -expEps ≤ δ := (abs_le.mp hδ).1
  have h2 : expEps < 1 := by unfold expEps; norm_num
  linarith

/-- **Full ratio positivity.** The running `exp(newLogp − oldLogp)` is strictly positive. -/
theorem ratioFullF_pos (newLogp oldLogp : Float) : 0 < toReal (ratioFullF newLogp oldLogp) := by
  unfold ratioFullF
  obtain ⟨δ, hδ, he⟩ := exp_model (newLogp - oldLogp)
  rw [he]
  refine mul_pos (Real.exp_pos _) ?_
  have h1 : -expEps ≤ δ := (abs_le.mp hδ).1
  have h2 : expEps < 1 := by unfold expEps; norm_num
  linarith

/-! ### PPO clipped surrogate (min/max exact; only products round) -/

/-- Executable clipped surrogate `min(g·r, g·clamp(r,lo,hi))`, `clamp = max lo (min r hi)`. -/
def ppoSurrF (g r lo hi : Float) : Float := min (g * r) (g * max lo (min r hi))

/-- **PPO clipped-surrogate error**: the running surrogate deviates from the ℝ spec
    `ppoObjLoHi` by at most `u·max(|g·r|, |g·clip(r)|)`. -/
theorem ppoSurrF_error (g r lo hi : Float) :
    |toReal (ppoSurrF g r lo hi) - ppoObjLoHi (toReal g) (toReal r) (toReal lo) (toReal hi)|
      ≤ u64 * max |toReal g * toReal r| |toReal g * clampR (toReal r) (toReal lo) (toReal hi)| := by
  have hclamp : toReal (max lo (min r hi)) = clampR (toReal r) (toReal lo) (toReal hi) := by
    unfold clampR; rw [toReal_max, toReal_min]
  have h1 : |toReal (g * r) - toReal g * toReal r| ≤ u64 * |toReal g * toReal r| := mul_error g r
  have h2 : |toReal (g * (max lo (min r hi))) - toReal g * clampR (toReal r) (toReal lo) (toReal hi)|
              ≤ u64 * |toReal g * clampR (toReal r) (toReal lo) (toReal hi)| := by
    have h := mul_error g (max lo (min r hi))
    rwa [hclamp] at h
  unfold ppoSurrF ppoObjLoHi
  rw [toReal_min]
  calc |min (toReal (g * r)) (toReal (g * max lo (min r hi)))
          - min (toReal g * toReal r) (toReal g * clampR (toReal r) (toReal lo) (toReal hi))|
      ≤ max |toReal (g * r) - toReal g * toReal r|
            |toReal (g * max lo (min r hi)) - toReal g * clampR (toReal r) (toReal lo) (toReal hi)| :=
        abs_min_sub_min_le_max _ _ _ _
    _ ≤ max (u64 * |toReal g * toReal r|)
            (u64 * |toReal g * clampR (toReal r) (toReal lo) (toReal hi)|) := max_le_max h1 h2
    _ = u64 * max |toReal g * toReal r| |toReal g * clampR (toReal r) (toReal lo) (toReal hi)| :=
        (mul_max_of_nonneg _ _ u64_pos.le).symm

/-- **Runtime clip inactivity ⇒ single-rounding error (sharp in-region bound).** Inside the trust region —
    when the Float ratio satisfies the *Float-order* bounds `lo ≤ r` and `r ≤ hi` — the running clipped
    surrogate `ppoSurrF g r lo hi = min (g·r) (g·clamp(r,lo,hi))` collapses **exactly** (as a `Float`, no
    rounding) to the plain product `g·r`: the clamp `max lo (min r hi)` reduces to `r`, so both `min`
    branches become the identical `Float` `g·r` and `min x x = x`. Hence the only rounding left is the
    single multiply, and the surrogate is within one unit-roundoff of the ideal `toReal g · toReal r` — a
    genuine **sharpening** of `ppoSurrF_error` (which for general inputs pays `u·max(|g·r|, |g·clip(r)|)`
    and drags in the clamp term). Both hypotheses are load-bearing: dropping `r ≤ hi` (take `r=10, hi=5,
    lo=0, g=1`) makes the surrogate `g·hi = 5 ≠ 10 = g·r`; dropping `lo ≤ r` with a negative
    advantage-weight (take `r=1, lo=5, hi=20, g=-1`) makes it `g·lo = -5 ≠ -1 = g·r`. The runtime
    counterpart of the ℝ-spec `ppoObjLoHi_eq_of_mem`, carried to the executable `Float` layer. -/
theorem ppoSurrF_error_of_mem (g r lo hi : Float) (hlo : lo ≤ r) (hhi : r ≤ hi) :
    |toReal (ppoSurrF g r lo hi) - toReal g * toReal r| ≤ u64 * |toReal g * toReal r| := by
  have heq : ppoSurrF g r lo hi = g * r := by
    unfold ppoSurrF
    rw [show min r hi = if r ≤ hi then r else hi from rfl, if_pos hhi,
        show max lo r = if lo ≤ r then r else lo from rfl, if_pos hlo,
        show min (g * r) (g * r) = if (g * r) ≤ (g * r) then (g * r) else (g * r) from rfl,
        ite_self]
  rw [heq]
  exact mul_error g r

/-- **Surrogate is pessimistic** (runtime). The running clipped surrogate is at most the unclipped
    objective `g·r` — the runtime counterpart of `PPO.ppoObjective_le_unclipped` (`min` picks the smaller;
    `min`/`max` are exact under `toReal`). -/
theorem ppoSurrF_le_unclipped (g r lo hi : Float) :
    toReal (ppoSurrF g r lo hi) ≤ toReal (g * r) := by
  unfold ppoSurrF; rw [toReal_min]; exact min_le_left _ _

/-- **Runtime soundness: surrogate ≤ clipped side.** The running PPO surrogate never exceeds the CLIPPED
    branch `g·clamp(r,lo,hi)` either — the companion of `ppoSurrF_le_unclipped` for the other `min` argument
    (`min` picks the smaller; `min`/`max` exact under `toReal`). Together they pin the surrogate below both
    branches, matching `PPO.ppoObjective_le_unclipped`/`_le_clipped`. -/
theorem ppoSurrF_le_clipped (g r lo hi : Float) :
    toReal (ppoSurrF g r lo hi) ≤ toReal (g * max lo (min r hi)) := by
  unfold ppoSurrF; rw [toReal_min]; exact min_le_right _ _

/-! ### PPO value loss (clipped value + squared error) -/

/-- Executable clipped value `val + clamp(valPred − val, −c, c)` (min/max exact). -/
def valueClippedF (val valPred vfClip : Float) : Float :=
  val + max (-vfClip) (min (valPred - val) vfClip)

theorem valueClippedF_error (val valPred vfClip : Float) :
    |toReal (valueClippedF val valPred vfClip)
        - valueClipped (toReal val) (toReal valPred) (toReal vfClip)|
      ≤ u64 * |toReal val + toReal (max (-vfClip) (min (valPred - val) vfClip))|
        + u64 * |toReal valPred - toReal val| := by
  -- the clamp is exact up to the inner `valPred - val` sub rounding
  have hd : |toReal (valPred - val) - (toReal valPred - toReal val)|
      ≤ u64 * |toReal valPred - toReal val| := sub_error valPred val
  have hclampToReal : toReal (max (-vfClip) (min (valPred - val) vfClip))
      = clampR (toReal (valPred - val)) (-toReal vfClip) (toReal vfClip) := by
    unfold clampR; rw [toReal_max, toReal_min, toReal_neg]
  have hclamp : |toReal (max (-vfClip) (min (valPred - val) vfClip))
        - clampR (toReal valPred - toReal val) (-toReal vfClip) (toReal vfClip)|
      ≤ u64 * |toReal valPred - toReal val| := by
    rw [hclampToReal]
    exact (clampR_lipschitz (toReal (valPred - val)) (toReal valPred - toReal val)
      (-toReal vfClip) (toReal vfClip)).trans hd
  have hres := addApprox_error val (max (-vfClip) (min (valPred - val) vfClip))
    (toReal val) (clampR (toReal valPred - toReal val) (-toReal vfClip) (toReal vfClip))
    0 (u64 * |toReal valPred - toReal val|) (by simp) hclamp
  rw [valueClippedF, valueClipped]
  simpa using hres

/-- **Runtime value-clip inactivity ⇒ prediction-tracking (in-region sharpening).** Inside the value trust
    region — when the Float residual satisfies the *Float-order* bounds `−vfClip ≤ valPred − val` and
    `valPred − val ≤ vfClip` — the running clipped value `valueClippedF val valPred vfClip =
    val + clamp(valPred−val, −vfClip, vfClip)` collapses (as a `Float`, no clamp rounding) to `val + (valPred − val)`:
    the clamp `max (−vfClip) (min (valPred − val) vfClip)` reduces to the residual `valPred − val`. Hence the ONLY
    roundings left are the inner subtraction and the outer addition, and the running clipped value is within
    `(sub + add)` unit-roundoff of the raw prediction `toReal valPred` — the value-side counterpart of
    `ppoSurrF_error_of_mem` (which pins the surrogate to `g·r` in-region), and the runtime layer of the ℝ-spec
    `valueClipped_eq_of_close`. Sharpens the general `valueClippedF_error` (which routes through `clampR_lipschitz`)
    to the exact in-region residual. Both hypotheses are load-bearing: dropping `valPred − val ≤ vfClip` (take
    `valPred − val = 10, vfClip = 5`) makes the clamp `5`, so the value is `val + 5 ≠ val + 10 = valPred`
    (`#eval valueClippedF 1.0 11.0 5.0 = 6.0 ≠ 11.0`); dropping `−vfClip ≤ valPred − val` (a residual below `−vfClip`)
    makes it `val − vfClip` (`#eval valueClippedF 1.0 (−9.0) 5.0 = −4.0 ≠ −9.0`), each an `O(1)` gap dwarfing the
    `u64` rounding bound. -/
theorem valueClippedF_error_of_mem (val valPred vfClip : Float)
    (hlo : -vfClip ≤ valPred - val) (hhi : valPred - val ≤ vfClip) :
    |toReal (valueClippedF val valPred vfClip) - toReal valPred|
      ≤ u64 * |toReal val + toReal (valPred - val)| + u64 * |toReal valPred - toReal val| := by
  have heq : valueClippedF val valPred vfClip = val + (valPred - val) := by
    unfold valueClippedF
    rw [show min (valPred - val) vfClip
          = if (valPred - val) ≤ vfClip then (valPred - val) else vfClip from rfl, if_pos hhi,
        show max (-vfClip) (valPred - val)
          = if (-vfClip) ≤ (valPred - val) then (valPred - val) else (-vfClip) from rfl, if_pos hlo]
  rw [heq]
  have hx : |toReal val - toReal val| ≤ 0 := by simp
  have hy : |toReal (valPred - val) - (toReal valPred - toReal val)| ≤ u64 * |toReal valPred - toReal val| :=
    sub_error valPred val
  have hres := addApprox_error val (valPred - val) (toReal val) (toReal valPred - toReal val)
    0 (u64 * |toReal valPred - toReal val|) hx hy
  have hsimp : toReal val + (toReal valPred - toReal val) = toReal valPred := by ring
  rw [hsimp] at hres
  simpa using hres

/-- **Runtime soundness: value-clip offset stays within ±vfClip.** The clamp `max(−c, min(valPred−val, c))`
    that `valueClippedF` adds to `val` lies in `[−c, c]` whenever `c ≥ 0` (min/max exact under `toReal`, then
    ℝ lattice: `le_max_left` for the floor, `max_le` for the ceiling). This is why PPO value clipping caps the
    predicted-value correction at ±vfClip — the trust-region guarantee, at the runtime layer. -/
theorem valueClip_offset_range (val valPred vfClip : Float) (hc : 0 ≤ toReal vfClip) :
    -toReal vfClip ≤ toReal (max (-vfClip) (min (valPred - val) vfClip))
      ∧ toReal (max (-vfClip) (min (valPred - val) vfClip)) ≤ toReal vfClip := by
  rw [toReal_max, toReal_min, toReal_neg]
  exact ⟨le_max_left _ _, max_le (by linarith) (min_le_right _ _)⟩

/-- Executable value loss `0.5·max((valPred−ret)², (vClipped−ret)²)` (squares via `x*x`). -/
def valueLossF (val valPred ret vfClip : Float) : Float :=
  0.5 * max ((valPred - ret) * (valPred - ret))
    ((valueClippedF val valPred vfClip - ret) * (valueClippedF val valPred vfClip - ret))

/-- Per-square error budget helper. -/
noncomputable def sqBnd (d : Float) (dR εd : ℝ) : ℝ :=
  u64 * |toReal d * toReal d| + |toReal d| * εd + |dR| * εd

theorem valueLossF_error (val valPred ret vfClip : Float) :
    |toReal (valueLossF val valPred ret vfClip)
        - valueLoss (toReal val) (toReal valPred) (toReal ret) (toReal vfClip)|
      ≤ u64 * |toReal (0.5 : Float) * toReal (max ((valPred - ret) * (valPred - ret))
            ((valueClippedF val valPred vfClip - ret) * (valueClippedF val valPred vfClip - ret)))|
        + |toReal (0.5 : Float)| * max (sqBnd (valPred - ret) (toReal valPred - toReal ret)
              (u64 * |toReal valPred - toReal ret|))
            (sqBnd (valueClippedF val valPred vfClip - ret)
              (valueClipped (toReal val) (toReal valPred) (toReal vfClip) - toReal ret)
              (u64 * |toReal (valueClippedF val valPred vfClip) - toReal ret|
                + (u64 * |toReal val + toReal (max (-vfClip) (min (valPred - val) vfClip))|
                  + u64 * |toReal valPred - toReal val|)))
        + |max ((toReal valPred - toReal ret) ^ 2)
              ((valueClipped (toReal val) (toReal valPred) (toReal vfClip) - toReal ret) ^ 2)|
            * (u64 * |(0.5 : ℝ)|) := by
  have h0 : ∀ z : Float, |toReal z - toReal z| ≤ 0 := fun z => by simp
  -- square 1: (valPred - ret)²
  have hd1 : |toReal (valPred - ret) - (toReal valPred - toReal ret)|
      ≤ u64 * |toReal valPred - toReal ret| := sub_error valPred ret
  have hs1 : |toReal ((valPred - ret) * (valPred - ret)) - (toReal valPred - toReal ret) ^ 2|
      ≤ sqBnd (valPred - ret) (toReal valPred - toReal ret) (u64 * |toReal valPred - toReal ret|) := by
    have := mulApprox_error (valPred - ret) (valPred - ret) (toReal valPred - toReal ret)
      (toReal valPred - toReal ret) (u64 * |toReal valPred - toReal ret|)
      (u64 * |toReal valPred - toReal ret|) hd1 hd1
    simpa [sqBnd, sq] using this
  -- valueClipped error
  have hvc := valueClippedF_error val valPred vfClip
  set εvc := u64 * |toReal val + toReal (max (-vfClip) (min (valPred - val) vfClip))|
    + u64 * |toReal valPred - toReal val| with hεvc
  -- square 2: (vc - ret)²
  have hd2 : |toReal (valueClippedF val valPred vfClip - ret)
        - (valueClipped (toReal val) (toReal valPred) (toReal vfClip) - toReal ret)|
      ≤ u64 * |toReal (valueClippedF val valPred vfClip) - toReal ret| + εvc := by
    have := subApprox_error (valueClippedF val valPred vfClip) ret
      (valueClipped (toReal val) (toReal valPred) (toReal vfClip)) (toReal ret) εvc 0 hvc (h0 ret)
    simpa using this
  have hs2 : |toReal ((valueClippedF val valPred vfClip - ret) * (valueClippedF val valPred vfClip - ret))
        - (valueClipped (toReal val) (toReal valPred) (toReal vfClip) - toReal ret) ^ 2|
      ≤ sqBnd (valueClippedF val valPred vfClip - ret)
          (valueClipped (toReal val) (toReal valPred) (toReal vfClip) - toReal ret)
          (u64 * |toReal (valueClippedF val valPred vfClip) - toReal ret| + εvc) := by
    have := mulApprox_error (valueClippedF val valPred vfClip - ret)
      (valueClippedF val valPred vfClip - ret)
      (valueClipped (toReal val) (toReal valPred) (toReal vfClip) - toReal ret)
      (valueClipped (toReal val) (toReal valPred) (toReal vfClip) - toReal ret)
      (u64 * |toReal (valueClippedF val valPred vfClip) - toReal ret| + εvc)
      (u64 * |toReal (valueClippedF val valPred vfClip) - toReal ret| + εvc) hd2 hd2
    simpa [sqBnd, sq] using this
  -- max is exact; Lipschitz gives max of the two square errors
  have hmax : |toReal (max ((valPred - ret) * (valPred - ret))
          ((valueClippedF val valPred vfClip - ret) * (valueClippedF val valPred vfClip - ret)))
        - max ((toReal valPred - toReal ret) ^ 2)
            ((valueClipped (toReal val) (toReal valPred) (toReal vfClip) - toReal ret) ^ 2)|
      ≤ max (sqBnd (valPred - ret) (toReal valPred - toReal ret) (u64 * |toReal valPred - toReal ret|))
          (sqBnd (valueClippedF val valPred vfClip - ret)
            (valueClipped (toReal val) (toReal valPred) (toReal vfClip) - toReal ret)
            (u64 * |toReal (valueClippedF val valPred vfClip) - toReal ret| + εvc)) := by
    rw [toReal_max]
    exact (abs_max_sub_max_le_max _ _ _ _).trans (max_le_max hs1 hs2)
  -- outer 0.5 multiply (ofScientific gap on 0.5)
  have h05 : |toReal (0.5 : Float) - (0.5 : ℝ)| ≤ u64 * |(0.5 : ℝ)| :=
    toReal_ofScientific_close 5 true 1
  have hres := mulApprox_error (0.5 : Float)
    (max ((valPred - ret) * (valPred - ret))
      ((valueClippedF val valPred vfClip - ret) * (valueClippedF val valPred vfClip - ret)))
    (0.5 : ℝ)
    (max ((toReal valPred - toReal ret) ^ 2)
      ((valueClipped (toReal val) (toReal valPred) (toReal vfClip) - toReal ret) ^ 2))
    (u64 * |(0.5 : ℝ)|) _ h05 hmax
  rw [valueLossF, valueLoss]
  simpa [εvc] using hres

/-- A `Float` self-product is nonnegative at the `toReal` level: `0 ≤ toReal (x·x)` — the rounding factor
    `1+δ` stays positive (`|δ| ≤ u64 < 1`), so `mul_self_nonneg` survives the multiply. -/
theorem toReal_mul_self_nonneg (x : Float) : 0 ≤ toReal (x * x) := by
  obtain ⟨δ, hδ, he⟩ := mul_model x x
  rw [he]
  have hu1 : (u64:ℝ) < 1 := by unfold u64; norm_num
  have hd : (0:ℝ) < 1 + δ := by have := (abs_le.mp hδ).1; linarith
  have hsq : (0:ℝ) ≤ toReal x * toReal x := mul_self_nonneg _
  positivity

/-- **Runtime soundness: clipped value loss is nonnegative.** The running `0.5·max((valPred−ret)²,
    (vClipped−ret)²)` is `≥ 0` for ALL Float inputs — each squared branch is nonneg (`toReal_mul_self_nonneg`),
    so their `max` is nonneg (`le_max_of_le_left`), and the `0.5` scale (`> 0`) plus the `1+δ` rounding
    (`> 0`) preserve the sign (`positivity`). The `max`-of-squares counterpart to `valueSqLossF_nonneg`. -/
theorem valueLossF_nonneg (val valPred ret vfClip : Float) :
    0 ≤ toReal (valueLossF val valPred ret vfClip) := by
  unfold valueLossF
  obtain ⟨δ, hδ, he⟩ := mul_model (0.5 : Float)
    (max ((valPred - ret) * (valPred - ret))
      ((valueClippedF val valPred vfClip - ret) * (valueClippedF val valPred vfClip - ret)))
  rw [he, toReal_max]
  have h05 : (0:ℝ) < toReal (0.5 : Float) := by
    have h : |toReal (0.5:Float) - (0.5:ℝ)| ≤ u64 * |(0.5:ℝ)| := toReal_ofScientific_close 5 true 1
    have hu : (u64:ℝ) < 1 := by unfold u64; norm_num
    rw [abs_le] at h; norm_num at h ⊢; nlinarith [h.1, u64_pos]
  have hu1 : (u64:ℝ) < 1 := by unfold u64; norm_num
  have hd : (0:ℝ) < 1 + δ := by have := (abs_le.mp hδ).1; linarith
  have hmax : (0:ℝ) ≤ max (toReal ((valPred - ret) * (valPred - ret)))
      (toReal ((valueClippedF val valPred vfClip - ret) * (valueClippedF val valPred vfClip - ret))) :=
    le_max_of_le_left (toReal_mul_self_nonneg _)
  positivity

/-- Executable UNCLIPPED value loss `½·(val − ret)²` — the value term `mlpGradPPO`'s AD tape actually uses
    (`scale (vfCoef·0.5) (diff·diff)`), the counterpart to the clipped `valueLossF`. -/
def valueSqLossF (val ret : Float) : Float := 0.5 * ((val - ret) * (val - ret))

/-- **Unclipped value-loss error.** The running `½·(val−ret)²` (`x*x` for the square, `0.5` via
    `toReal_ofScientific_close`) deviates from the ℝ `½·(val−ret)²` by the composed `sub`/square/`0.5·`
    rounding. -/
theorem valueSqLossF_error (val ret : Float) :
    |toReal (valueSqLossF val ret) - (0.5 : ℝ) * (toReal val - toReal ret) ^ 2|
      ≤ u64 * |toReal (0.5 : Float) * toReal ((val - ret) * (val - ret))|
        + |toReal (0.5 : Float)| * sqBnd (val - ret) (toReal val - toReal ret) (u64 * |toReal val - toReal ret|)
        + |(toReal val - toReal ret) ^ 2| * (u64 * |(0.5 : ℝ)|) := by
  have hd : |toReal (val - ret) - (toReal val - toReal ret)| ≤ u64 * |toReal val - toReal ret| :=
    sub_error val ret
  have hsq : |toReal ((val - ret) * (val - ret)) - (toReal val - toReal ret) ^ 2|
      ≤ sqBnd (val - ret) (toReal val - toReal ret) (u64 * |toReal val - toReal ret|) := by
    have := mulApprox_error (val - ret) (val - ret) (toReal val - toReal ret) (toReal val - toReal ret)
      (u64 * |toReal val - toReal ret|) (u64 * |toReal val - toReal ret|) hd hd
    simpa [sqBnd, sq] using this
  have h05 : |toReal (0.5 : Float) - (0.5 : ℝ)| ≤ u64 * |(0.5 : ℝ)| :=
    toReal_ofScientific_close 5 true 1
  have hres := mulApprox_error (0.5 : Float) ((val - ret) * (val - ret)) (0.5 : ℝ)
    ((toReal val - toReal ret) ^ 2) (u64 * |(0.5 : ℝ)|)
    (sqBnd (val - ret) (toReal val - toReal ret) (u64 * |toReal val - toReal ret|)) h05 hsq
  rw [valueSqLossF]
  simpa using hres

/-- **Runtime soundness: value loss is nonnegative.** The running `½·(val−ret)²` is `≥ 0` for ALL Float
    inputs — the `0.5` multiplier stays positive (`toReal_ofScientific_close`, `u64 < 1`) and both rounding
    factors `1+δ` stay positive (`|δ| ≤ u64 < 1`), so the squared core `mul_self_nonneg` survives the two
    roundings. A structural guarantee that the runtime value term never goes negative from arithmetic error. -/
theorem valueSqLossF_nonneg (val ret : Float) : 0 ≤ toReal (valueSqLossF val ret) := by
  unfold valueSqLossF
  obtain ⟨δ1, hδ1, he1⟩ := mul_model (val - ret) (val - ret)
  obtain ⟨δ2, hδ2, he2⟩ := mul_model (0.5 : Float) ((val - ret) * (val - ret))
  rw [he2, he1]
  have h05 : (0:ℝ) < toReal (0.5 : Float) := by
    have h : |toReal (0.5:Float) - (0.5:ℝ)| ≤ u64 * |(0.5:ℝ)| := toReal_ofScientific_close 5 true 1
    have hu : (u64:ℝ) < 1 := by unfold u64; norm_num
    rw [abs_le] at h; norm_num at h ⊢; nlinarith [h.1, u64_pos]
  have hu1 : (u64:ℝ) < 1 := by unfold u64; norm_num
  have hd1 : (0:ℝ) < 1 + δ1 := by have := (abs_le.mp hδ1).1; linarith
  have hd2 : (0:ℝ) < 1 + δ2 := by have := (abs_le.mp hδ2).1; linarith
  have hsq : (0:ℝ) ≤ toReal (val - ret) * toReal (val - ret) := mul_self_nonneg _
  positivity

/-! ### PPO approximate-KL estimators (Schulman) -/

/-- Executable old-KL estimator `−logratio`. -/
def approxKLOldF (lr : Float) : Float := -lr

/-- The old-KL estimator is EXACT (negation is exact under `toReal`). -/
theorem approxKLOldF_error (lr : Float) :
    toReal (approxKLOldF lr) = approxKLOld (toReal lr) := by
  rw [approxKLOldF, approxKLOld, toReal_neg]

/-- Executable Schulman k3 KL estimator `(exp(lr) − 1) − lr`. -/
def approxKLNewF (lr : Float) : Float := (Float.exp lr - 1) - lr

noncomputable def approxKLNewErrBnd (lr : Float) : ℝ :=
  u64 * |toReal (Float.exp lr - 1) - toReal lr|
    + (u64 * |toReal (Float.exp lr) - toReal (1 : Float)| + expEps * Real.exp (toReal lr))

theorem approxKLNewF_error (lr : Float) :
    |toReal (approxKLNewF lr) - approxKLNew (toReal lr)| ≤ approxKLNewErrBnd lr := by
  have h0 : ∀ z : Float, |toReal z - toReal z| ≤ 0 := fun z => by simp
  have he : |toReal (Float.exp lr) - Real.exp (toReal lr)| ≤ expEps * Real.exp (toReal lr) :=
    exp_error lr
  have hs1 : |toReal (Float.exp lr - 1) - (Real.exp (toReal lr) - 1)|
      ≤ u64 * |toReal (Float.exp lr) - toReal (1 : Float)| + expEps * Real.exp (toReal lr) := by
    have := subApprox_error (Float.exp lr) (1 : Float) (Real.exp (toReal lr)) 1
      (expEps * Real.exp (toReal lr)) 0 he (by simp)
    simpa using this
  have hs2 := subApprox_error (Float.exp lr - 1) lr (Real.exp (toReal lr) - 1) (toReal lr)
    (u64 * |toReal (Float.exp lr) - toReal (1 : Float)| + expEps * Real.exp (toReal lr)) 0 hs1 (h0 lr)
  rw [approxKLNewF, approxKLNew]
  simpa [approxKLNewErrBnd] using hs2

/-! ### KL estimators vs the ideal log-ratio (early-stopping accuracy) -/

/-- **Old-KL estimator vs the ideal.** Since `approxKLOld = −logratio` is EXACT, the running estimator on a
    Float log-ratio `lr` is within the log-ratio's own error `ε` of the ideal `approxKLOld lrIdeal` — no
    rounding added. Discharge `ε` via the forward log-ratio bound (a110-style), like a115→a119. -/
theorem approxKLOldF_ideal_error (lr : Float) (lrIdeal ε : ℝ) (h : |toReal lr - lrIdeal| ≤ ε) :
    |toReal (approxKLOldF lr) - approxKLOld lrIdeal| ≤ ε := by
  rw [approxKLOldF_error, approxKLOld, approxKLOld]
  rw [show -(toReal lr) - -lrIdeal = -(toReal lr - lrIdeal) by ring, abs_neg]
  exact h

/-- **k3 KL estimator perturbation.** `approxKLNew x = (eˣ − 1) − x` differs between two log-ratios by at most
    the `exp` gap plus the linear gap (`approxKLNew a − approxKLNew b = (eᵃ − eᵇ) − (a − b)`). Pure ℝ. -/
theorem approxKLNew_perturb (a b : ℝ) :
    |approxKLNew a - approxKLNew b| ≤ |Real.exp a - Real.exp b| + |a - b| := by
  rw [approxKLNew, approxKLNew]
  calc |Real.exp a - 1 - a - (Real.exp b - 1 - b)|
      = |(Real.exp a - Real.exp b) - (a - b)| := by ring_nf
    _ ≤ |Real.exp a - Real.exp b| + |a - b| := abs_sub _ _

/-- **k3 KL estimator vs the ideal.** The running Schulman k3 estimator `(exp lr − 1) − lr` on a Float
    log-ratio `lr` is within its own rounding budget (`approxKLNewF_error`, a107) plus the `exp`-Lipschitz
    perturbation `exp(lrIdeal)·(eᵋ − 1) + ε` of the ideal `approxKLNew lrIdeal`, given `lr` is within `ε` of
    `lrIdeal`. Composes `approxKLNewF_error` with `approxKLNew_perturb`/`abs_exp_sub_exp_le`. Discharge `ε`
    via the forward log-ratio bound (like a115→a119). -/
theorem approxKLNewF_ideal_error (lr : Float) (lrIdeal ε : ℝ) (h : |toReal lr - lrIdeal| ≤ ε) :
    |toReal (approxKLNewF lr) - approxKLNew lrIdeal|
      ≤ approxKLNewErrBnd lr + (Real.exp lrIdeal * (Real.exp ε - 1) + ε) := by
  have h107 := approxKLNewF_error lr
  have hpert : |approxKLNew (toReal lr) - approxKLNew lrIdeal|
      ≤ Real.exp lrIdeal * (Real.exp ε - 1) + ε := by
    refine (approxKLNew_perturb (toReal lr) lrIdeal).trans ?_
    exact add_le_add (abs_exp_sub_exp_le (toReal lr) lrIdeal ε h) h
  calc |toReal (approxKLNewF lr) - approxKLNew lrIdeal|
      ≤ |toReal (approxKLNewF lr) - approxKLNew (toReal lr)|
        + |approxKLNew (toReal lr) - approxKLNew lrIdeal| := abs_sub_le _ _ _
    _ ≤ _ := add_le_add h107 hpert

end Puffer.RL.PPORuntime
