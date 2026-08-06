/-
The COMPOSED per-weight update bound — closing the `εg` gap in `axpyStep_error`.

`UpdateRuntime.axpyStep_error` bounds one SGD (gradient-ascent) weight update
`p' = fl(p + fl(lr·g))` against the ideal real step `toReal p + toReal lr · Rg`, but it is
*parameterized* by an abstract gradient-error bound `εg` with hypothesis `|toReal g − Rg| ≤ εg`
(PLAN.md M8: "the one piece the reverse-mode AD does NOT yet certify").

That piece is now certified. `AutoDiffR.dF_error` proves the running forward-mode-AD Float
derivative `dF e σ k` is within the *computable* `derivErrBnd e σ k` of the exact-ℝ symbolic
derivative `derivR e (envR σ) k`, and `AutoDiffR.derivR_eq_deriv` proves that symbolic
derivative IS the true Mathlib derivative `∂(evalR e)/∂(var k)`. Feeding `dF_error` as the
`hg` hypothesis and `derivErrBnd` as `εg`, and rewriting the ideal step against the true
derivative, discharges the parameter entirely:

  `axpyStep_AD_error` : one gradient-ascent step on the AD-computed Float gradient `dF e σ k`
  is within `u64·|…| + u64·|lr·g| + |lr|·derivErrBnd e σ k` of the ideal REAL ascent step
  `toReal p + toReal lr · ∂(evalR e)/∂(var k)` — a fully-closed, machine-checked bound with NO
  free gradient-error parameter, against the genuine mathematical derivative.

The learning rate scales the gradient error by exactly `|lr|` (no roundoff amplification), and
`derivErrBnd`/the two `u64` terms are all computable. `axpyStep_AD_error_le` packages it against
any handles `Bp Blrg` for the two magnitude terms, for the vector/whole-net assembly. Axiom-clean
beyond the trusted Float base (via `dF_error` + `axpyStep_error`).
-/
import Puffer.RL.UpdateRuntime
import Puffer.Float.AutoDiffR

namespace Puffer.FloatR.ADR

open Puffer.FloatR

/-- **Composed AD + SGD-step error (per weight).** One gradient-ascent update `p + lr · (dF e σ k)`
    on the forward-mode-AD gradient component is within
    `u64·|toReal p + toReal (lr·g)| + (u64·|toReal lr · toReal g| + |toReal lr|·derivErrBnd e σ k)`
    of the ideal real ascent step against the TRUE derivative `∂(evalR e)/∂(var k)`. The open `εg`
    parameter of `axpyStep_error` is discharged by `dF_error` (gradient within `derivErrBnd`) and the
    ideal target is the genuine Mathlib derivative via `derivR_eq_deriv`. `hwd` (`WD`): away from every
    `log`/kink so the Float engine stays well-defined; `hp` (`PosR`): every `log` argument `> 0` so the
    real derivative exists. -/
theorem axpyStep_AD_error (e : Expr) (σ : Nat → Float) (k : Nat) (p lr : Float)
    (hwd : WD e σ) (hp : PosR e (envR σ)) :
    |toReal (p + lr * dF e σ k)
        - (toReal p + toReal lr
            * deriv (fun t => evalR e (Function.update (envR σ) k t)) (envR σ k))|
      ≤ u64 * |toReal p + toReal (lr * dF e σ k)|
          + (u64 * |toReal lr * toReal (dF e σ k)| + |toReal lr| * derivErrBnd e σ k) := by
  have h := axpyStep_error p lr (dF e σ k) (derivR e (envR σ) k) (derivErrBnd e σ k)
    (dF_error e σ k hwd)
  rwa [derivR_eq_deriv e (envR σ) k hp] at h

/-- `axpyStep_AD_error` with the two computable magnitude terms replaced by any upper handles
    `Bp ≥ |toReal p + toReal (lr·g)|`, `Blrg ≥ |toReal lr · toReal g|` — the form the whole-net
    update assembly folds over (each weight contributes `u64·Bp + u64·Blrg + |lr|·derivErrBnd`). -/
theorem axpyStep_AD_error_le (e : Expr) (σ : Nat → Float) (k : Nat) (p lr : Float)
    (hwd : WD e σ) (hp : PosR e (envR σ)) (Bp Blrg : ℝ)
    (hBp : |toReal p + toReal (lr * dF e σ k)| ≤ Bp)
    (hBlrg : |toReal lr * toReal (dF e σ k)| ≤ Blrg) :
    |toReal (p + lr * dF e σ k)
        - (toReal p + toReal lr
            * deriv (fun t => evalR e (Function.update (envR σ) k t)) (envR σ k))|
      ≤ u64 * Bp + (u64 * Blrg + |toReal lr| * derivErrBnd e σ k) := by
  refine (axpyStep_AD_error e σ k p lr hwd hp).trans ?_
  have hu : (0:ℝ) ≤ u64 := le_of_lt u64_pos
  gcongr

end Puffer.FloatR.ADR
