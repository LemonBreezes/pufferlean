/-
# Clip-interior sufficient condition: the PPO ratio is in the clip window when the policy hasn't moved far

C26/C28's whole-objective gradient-Lipschitz holds on the CLIP INTERIOR — the hypothesis
`toReal lo < evalR (ratioE (logSoftmaxE chosen (e::es)) oldLogp) σ < toReal hi` (the policy ratio strictly
inside the clip window `(lo, hi)`). C29 established the PARAM-BOUNDEDNESS part of C26's region conditions but
flagged the clip-interior as a VALUE-level condition it did NOT discharge. This module discharges it from a
checkable bound on the new-policy log-prob, analogous to C21 (the active-region sufficient condition).

The ratio is `ratioE newLogp oldLogp = exp(newLogp − oldLogp)` (C8), so `evalR (ratioE …) σ =
Real.exp (evalR newLogp σ − toReal oldLogp)`. Since `Real.exp` is strictly monotone and inverts `Real.log` on
positives, the ratio lies in `(toReal lo, toReal hi)` exactly when the log-prob lies in the LOG-window shifted by
the old log-prob: `toReal oldLogp + log(toReal lo) < evalR newLogp σ < toReal oldLogp + log(toReal hi)`. That is
precisely the "policy hasn't moved too far from the old policy" condition PPO's clipping is designed to keep the
gradient in — here it is the checkable premise under which C26/C28's clip-interior hypothesis holds.

* `ratioE_mem_clip_interior` — the core: the log-prob in the log-window ⟹ the ratio in the clip window.
* `logSoftmaxE_ratio_clip_interior` — specialized to a softmax `newLogp = logSoftmaxE chosen (e::es)`.
* `ratioE_clip_interior_invariant` — the clip-interior INVARIANCE along a whole trajectory, reduced (by a trivial
  ∀-lift) to the per-step log-window bound; this supplies C28's `hIntσ`/`hIntσ'` from a checkable value condition.

**Scope (honestly disclosed):** this is a SUFFICIENT (checkable) condition — the ratio is in the clip window
WHENEVER the log-prob stays within the log-window of the old policy. It does NOT claim a trained policy always
satisfies it (whether the log-prob stays in-window is data/trajectory-dependent — a reachability question, like
C21's active region). It discharges C26/C28's clip-interior hypothesis from the value bound; combined with C29's
param-boundedness, the region conditions reduce to (param-bound: C29 projected) + (log-prob-in-window: here) +
(entropy budgets: separate).
-/
import Puffer.RL.SurrogateExpr
import Puffer.RL.SoftmaxExpr

open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)
open Puffer.RL.SurrogateExpr (ratioE evalR_ratioE)
open Puffer.RL.SoftmaxExpr (logSoftmaxE)

namespace Puffer.RL.ClipInteriorExpr

/-- **Clip-interior sufficient condition.** If the new-policy log-prob lies strictly inside the log-window of the
    old policy — `toReal oldLogp + Real.log (toReal lo) < evalR newLogp σ < toReal oldLogp + Real.log (toReal hi)`
    (with `0 < toReal lo` and `0 < toReal hi`) — then the PPO ratio `evalR (ratioE newLogp oldLogp) σ` lies
    strictly inside the clip window `(toReal lo, toReal hi)`, discharging C26/C28's clip-interior hypothesis.
    Proof: `evalR (ratioE …) σ = exp(evalR newLogp σ − toReal oldLogp)`; `exp` is strictly monotone and inverts
    `log` on positives (`Real.exp_log`), so the log-window bound transports to the ratio window (`Real.exp_lt_exp`). -/
theorem ratioE_mem_clip_interior (newLogp : Expr) (oldLogp lo hi : Float) (σ : Nat → ℝ)
    (hlo : 0 < toReal lo) (hhi : 0 < toReal hi)
    (hLoBound : toReal oldLogp + Real.log (toReal lo) < evalR newLogp σ)
    (hHiBound : evalR newLogp σ < toReal oldLogp + Real.log (toReal hi)) :
    toReal lo < evalR (ratioE newLogp oldLogp) σ
    ∧ evalR (ratioE newLogp oldLogp) σ < toReal hi := by
  rw [evalR_ratioE, Puffer.RL.PPO.ratio]
  refine ⟨?_, ?_⟩
  · calc toReal lo = Real.exp (Real.log (toReal lo)) := (Real.exp_log hlo).symm
      _ < Real.exp (evalR newLogp σ - toReal oldLogp) := Real.exp_lt_exp.mpr (by linarith)
  · calc Real.exp (evalR newLogp σ - toReal oldLogp)
        < Real.exp (Real.log (toReal hi)) := Real.exp_lt_exp.mpr (by linarith)
      _ = toReal hi := Real.exp_log hhi

/-- **Softmax specialization.** The clip-interior condition for `newLogp = logSoftmaxE chosen (e::es)` (a softmax
    log-prob): the log-softmax value in the log-window of the old policy ⟹ the ratio in the clip window. This is
    the "the softmax policy hasn't moved too far" condition PPO's clipping enforces. -/
theorem logSoftmaxE_ratio_clip_interior (chosen e : Expr) (es : List Expr) (oldLogp lo hi : Float) (σ : Nat → ℝ)
    (hlo : 0 < toReal lo) (hhi : 0 < toReal hi)
    (hLoBound : toReal oldLogp + Real.log (toReal lo) < evalR (logSoftmaxE chosen (e :: es)) σ)
    (hHiBound : evalR (logSoftmaxE chosen (e :: es)) σ < toReal oldLogp + Real.log (toReal hi)) :
    toReal lo < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ
    ∧ evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ < toReal hi :=
  ratioE_mem_clip_interior (logSoftmaxE chosen (e :: es)) oldLogp lo hi σ hlo hhi hLoBound hHiBound

/-- **Clip-interior invariance along a trajectory.** If every step `θ n` keeps the log-prob in the log-window,
    then the ratio is in the clip interior at every step — the clip-interior INVARIANCE C28's `hIntσ`/`hIntσ'`
    require, reduced to a per-step checkable value bound (a trivial ∀-lift of `ratioE_mem_clip_interior`). -/
theorem ratioE_clip_interior_invariant (newLogp : Expr) (oldLogp lo hi : Float) (θ : Nat → (Nat → ℝ))
    (hlo : 0 < toReal lo) (hhi : 0 < toReal hi)
    (hLoBound : ∀ n, toReal oldLogp + Real.log (toReal lo) < evalR newLogp (θ n))
    (hHiBound : ∀ n, evalR newLogp (θ n) < toReal oldLogp + Real.log (toReal hi)) :
    ∀ n, toReal lo < evalR (ratioE newLogp oldLogp) (θ n)
      ∧ evalR (ratioE newLogp oldLogp) (θ n) < toReal hi :=
  fun n => ratioE_mem_clip_interior newLogp oldLogp lo hi (θ n) hlo hhi (hLoBound n) (hHiBound n)

end Puffer.RL.ClipInteriorExpr
