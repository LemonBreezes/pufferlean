/-
The SGD (gradient-ascent) weight-update step, composed error bound — the OPTIMIZER end
of the per-step update bound.

`updatePPO` finishes each update with `matAxpy lr p g` / `vecAxpy lr p g`, i.e. per weight

    p'ᵢ = fl(pᵢ + fl(lr · gᵢ)).

Given a bound `εg` on the gradient error (|toReal gᵢ − Rgᵢ| ≤ εg — the one piece the
reverse-mode AD does NOT yet certify; see PLAN.md M8), this bounds each updated weight
against the ideal real ascent step `toReal pᵢ + toReal lr · Rgᵢ`. Two unit-roundoff terms
(the multiply and the add) plus the gradient error scaled by exactly `|lr|` — the learning
rate does not amplify roundoff. This is the skeleton the whole update bound hangs on: it is
fully proven with NO new axioms, parameterized by the still-open `εg`.
-/
import Puffer.Float.Basic

namespace Puffer.FloatR

/-- **SGD-step composed error.** One gradient-ascent weight update `p + lr·g` is within
    `u64·|…| + u64·|lr·g| + |lr|·εg` of the ideal real step `toReal p + toReal lr · Rg`,
    given the gradient error bound `|toReal g − Rg| ≤ εg`. Proof: `mulApprox_error` on the
    `lr·g` scale (numerator error 0 on the exact `lr`), then `addApprox_error` on the
    `p + …` accumulate (numerator error 0 on the exact weight `p`). -/
theorem axpyStep_error (p lr g : Float) (Rg εg : ℝ) (hg : |toReal g - Rg| ≤ εg) :
    |toReal (p + lr * g) - (toReal p + toReal lr * Rg)|
      ≤ u64 * |toReal p + toReal (lr * g)| + (u64 * |toReal lr * toReal g| + |toReal lr| * εg) := by
  have hs : |toReal (lr * g) - toReal lr * Rg| ≤ u64 * |toReal lr * toReal g| + |toReal lr| * εg := by
    simpa using mulApprox_error lr g (toReal lr) Rg 0 εg (by simp) hg
  simpa using addApprox_error p (lr * g) (toReal p) (toReal lr * Rg) 0
    (u64 * |toReal lr * toReal g| + |toReal lr| * εg) (by simp) hs

end Puffer.FloatR
