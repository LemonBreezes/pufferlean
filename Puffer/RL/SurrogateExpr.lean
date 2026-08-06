/-
# Forward-objective → `Expr` compiler for the PPO clipped surrogate

This module is the missing FRONT-END of the composition chain: it compiles the PPO clipped surrogate
objective into the AD grammar `Expr` (`Puffer/Float/Expr.lean`), and proves the compiled term's exact-ℝ
evaluation `evalR` equals the pre-existing real-valued spec `Puffer.RL.PPO.ppoObjLoHi`. Until now the
`Expr` gradient-Lipschitz machinery (C4–C7 in `AutoDiffR.lean`) and the ℝ surrogate error theory
(`SurrogateForward.lean`, around `ppoObjLoHi`/`ppoObjective`/`clampR`) lived in separate worlds with
NOTHING connecting an `Expr` to the actual PPO objective. This file bridges them.

The clipped surrogate `min(g·r, g·clip(r, lo, hi))` (advantage-scale `g`, policy ratio `r`, clip window
`[lo,hi]`) compiles to `ppoSurrogateE r g lo hi = min (scale g r) (scale g (clampE r lo hi))`, reusing the
existing `clampE` clip builder. The policy ratio itself compiles via `ratioE newLogp oldLogp =
exp (newLogp − oldLogp)` (matching `PPO.ratio`). Three correctness levels are proven:

* `evalR_ppoSurrogateE` — the FORWARD correctness: `evalR (ppoSurrogateE …) σ = ppoObjLoHi (toReal g)
  (evalR r σ) (toReal lo) (toReal hi)`. The compiled objective's real evaluation is EXACTLY the ℝ spec.
* `evalR_ppoSurrogateE_ppoObjective` — instantiating the clip bounds to `1∓ε` recovers the single-`ε`
  `PPO.ppoObjective`.
* `ppoSurrogateE_interior` — the GRADIENT connection: on the strict clip interior (`lo < r < hi`) the
  compiled surrogate's value AND gradient collapse to the unclipped `g·r` / `g·(derivR r)`. So on the
  clip-active-interior region the compiled PPO gradient IS `g · (policy gradient derivR r)`, and its
  gradient-Lipschitz reduces (via `derivR_lip` on a `Smooth` ratio) to `|g|·dLip R r·δ`.

**Scope (honestly disclosed):** the clip surrogate's `min`/`max` kink branches are OUTSIDE the `Smooth`
gradient-Lipschitz fragment (`derivR` jumps at the clip boundary — intrinsic, matching C7's `relu`
disclosure), so the compiled objective is gradient-Lipschitz only on the clip interior (`ppoSurrogateE_interior`,
via `clampE_interior_id`), where it collapses to the smooth inner ratio. The advantage-scale `g`, old
log-prob `oldLogp`, and clip bounds `lo/hi` are treated as `Float` constants (detached / hyperparameters at
a training step), which is faithful to PPO. The clip bounds enter as `toReal lo, toReal hi`; the gap between
`toReal (1 - eps)` and `1 - toReal eps` (Float rounding of the bound) is a separate bridge, not this file's
concern.
-/
import Puffer.Float.AutoDiffR
import Puffer.RL.PPO

open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)

namespace Puffer.RL.SurrogateExpr

/-- **Clamp-form commutation.** `min (max x lo) hi = max lo (min x hi)` when `lo ≤ hi` — the two standard
    clamp encodings agree (both compute the median). Bridges the `Expr` clip `clampE = min (max · lo) hi`
    to the ℝ spec `clampR = max lo (min · hi)`. -/
theorem clamp_minmax_comm (x lo hi : ℝ) (h : lo ≤ hi) :
    min (max x lo) hi = max lo (min x hi) := by
  rw [max_min_distrib_left, max_eq_right h, max_comm x lo]

/-- **`clampE` evaluates to `clampR`.** The `Expr` clip builder `clampE a lo hi` (`= min (max a lo) hi`)
    evaluates, under `toReal lo ≤ toReal hi`, to the real-valued clamp `clampR (evalR a σ) (toReal lo)
    (toReal hi)` used by the PPO spec. The general companion to `clampE_interior_id` (which only covered the
    strict interior); this holds for ALL argument values (including the saturated branches). -/
theorem evalR_clampE (a : Expr) (lo hi : Float) (σ : Nat → ℝ)
    (h : toReal lo ≤ toReal hi) :
    evalR (clampE a lo hi) σ = Puffer.RL.PPO.clampR (evalR a σ) (toReal lo) (toReal hi) := by
  simp only [clampE, evalR, Puffer.RL.PPO.clampR]
  exact clamp_minmax_comm _ _ _ h

/-- **Ratio compiler.** The PPO policy ratio `r = exp(logp_new − logp_old)` as an `Expr`, parameterized by
    the new-policy log-prob sub-expression `newLogp` (which carries the θ-dependence — a log-softmax of the
    net's logits) and the detached old log-prob `oldLogp : Float`. -/
def ratioE (newLogp : Expr) (oldLogp : Float) : Expr := .exp (.sub newLogp (.const oldLogp))

/-- `evalR (ratioE newLogp oldLogp) σ = PPO.ratio (evalR newLogp σ) (toReal oldLogp)` — the compiled ratio
    evaluates to the ℝ spec `ratio = exp(new − old)`. -/
theorem evalR_ratioE (newLogp : Expr) (oldLogp : Float) (σ : Nat → ℝ) :
    evalR (ratioE newLogp oldLogp) σ = Puffer.RL.PPO.ratio (evalR newLogp σ) (toReal oldLogp) := by
  simp only [ratioE, evalR, Puffer.RL.PPO.ratio]

/-- **PPO clipped-surrogate compiler.** `min(g·r, g·clip(r, lo, hi))` as an `Expr`, with advantage-scale
    `g`, clip bounds `lo/hi : Float`, over the ratio sub-expression `r`. Reuses the existing `clampE` clip
    builder; the two surrogate branches are `scale g` of `r` and of `clampE r lo hi`. -/
def ppoSurrogateE (r : Expr) (g lo hi : Float) : Expr :=
  .min (.scale g r) (.scale g (clampE r lo hi))

/-- **FORWARD CORRECTNESS.** The compiled clipped surrogate's exact-ℝ evaluation equals the pre-existing ℝ
    spec `PPO.ppoObjLoHi`: `evalR (ppoSurrogateE r g lo hi) σ = ppoObjLoHi (toReal g) (evalR r σ) (toReal lo)
    (toReal hi)` (under `toReal lo ≤ toReal hi`, i.e. a well-formed clip window). This is the bridge that lets
    the `Expr` gradient-Lipschitz machinery act on the ACTUAL PPO objective, and keeps it consistent with the
    whole `SurrogateForward` Float↔ℝ error theory (all phrased over the same `ppoObjLoHi`). -/
theorem evalR_ppoSurrogateE (r : Expr) (g lo hi : Float) (σ : Nat → ℝ)
    (h : toReal lo ≤ toReal hi) :
    evalR (ppoSurrogateE r g lo hi) σ
      = Puffer.RL.PPO.ppoObjLoHi (toReal g) (evalR r σ) (toReal lo) (toReal hi) := by
  have hc : evalR (clampE r lo hi) σ = Puffer.RL.PPO.clampR (evalR r σ) (toReal lo) (toReal hi) :=
    evalR_clampE r lo hi σ h
  simp only [ppoSurrogateE, evalR, Puffer.RL.PPO.ppoObjLoHi]
  rw [hc]

/-- Instantiating the clip window to `[1−ε, 1+ε]` recovers the single-`ε` `PPO.ppoObjective`:
    `evalR (ppoSurrogateE r g lo hi) σ = ppoObjective (toReal g) (evalR r σ) ε` when `toReal lo = 1−ε` and
    `toReal hi = 1+ε` (with `ε ≥ 0` making the window well-formed). -/
theorem evalR_ppoSurrogateE_ppoObjective (r : Expr) (g lo hi : Float) (σ : Nat → ℝ)
    (ε : ℝ) (hlo : toReal lo = 1 - ε) (hhi : toReal hi = 1 + ε) (hε : 0 ≤ ε) :
    evalR (ppoSurrogateE r g lo hi) σ
      = Puffer.RL.PPO.ppoObjective (toReal g) (evalR r σ) ε := by
  rw [evalR_ppoSurrogateE r g lo hi σ (by rw [hlo, hhi]; linarith),
      Puffer.RL.PPO.ppoObjective_eq_LoHi, hlo, hhi]

/-- **GRADIENT connection on the clip interior.** On the strict clip interior (`toReal lo < evalR r σ <
    toReal hi`, the unclipped region) both the value AND the gradient of the compiled surrogate collapse to
    the unclipped `g·r`: `evalR = toReal g · evalR r σ` and `derivR = toReal g · derivR r σ k`. Consequence:
    on the clip-active-interior region the compiled PPO gradient IS `g · (policy gradient derivR r)`, so its
    gradient-Lipschitz reduces to the ratio's — e.g. `|g|·dLip R r·δ` when `r` is `Smooth` (via `derivR_lip`).
    The clip's `min`/`max` kinks matter only at the clip boundary; there (as with `relu` in C7) the gradient
    genuinely jumps and is intrinsically non-Lipschitz — hence the interior hypothesis. Uses
    `clampE_interior_id`: on the interior both surrogate branches have equal value and gradient. -/
theorem ppoSurrogateE_interior (r : Expr) (g lo hi : Float) (σ : Nat → ℝ) (k : Nat)
    (hlo : toReal lo < evalR r σ) (hhi : evalR r σ < toReal hi) :
    evalR (ppoSurrogateE r g lo hi) σ = toReal g * evalR r σ ∧
    derivR (ppoSurrogateE r g lo hi) σ k = toReal g * derivR r σ k := by
  obtain ⟨hv, hd⟩ := clampE_interior_id r lo hi σ k hlo hhi
  refine ⟨?_, ?_⟩
  · simp only [ppoSurrogateE, evalR, hv, min_self]
  · simp only [ppoSurrogateE, derivR, evalR, hv, hd, le_refl, if_true]

end Puffer.RL.SurrogateExpr
