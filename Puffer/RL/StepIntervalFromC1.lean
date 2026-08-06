/-
# The whole-run per-step interval `B` from the C1 reverse-mode gradient error

C27/C28/C29's whole-run theorems take the per-step interval as the hypothesis
`hstep : |θ(n+1) k − gradAscentE e lr (θ n) k| ≤ B` — how far one runnable (`Float`) training step strays from the
ideal exact-ℝ gradient-ascent step. This module reduces `B` to the per-coordinate GRADIENT error, so `B` is
concrete rather than a free constant: **`B = |toReal lr| · Bgrad`**, where `Bgrad` bounds the error between the
runnable Float gradient and the ideal `derivR`.

The reduction is elementary: the ideal ascent is `gradAscentE e lr σ k = σ k + lr·derivR e σ k`, and the runnable
step is `σ k + lr·(Float gradient)`; their difference is `lr·(Float gradient − derivR e σ k)`, so a per-coordinate
gradient error `≤ Bgrad` gives a per-step error `≤ |lr|·Bgrad`.

**`Bgrad` is concretely dischargeable by C1** (not merely a hypothesis): the C1 reverse-mode composition bound
`Puffer.RL.ReverseUpdateBound.reverseGrad_errorF_bounded` proves exactly
`|revGrad σ e k − derivR e (envR σ) k| ≤ revGradBndF σ e k W + bridgeBnd e σ k` — the ℓ∞ error between the Float
reverse-mode gradient `revGrad` and the ideal FORWARD `derivR` (the two AD modes agree by
`Puffer.FloatR.ADReverse.revE_R_eq_derivR : revE_R e σ 1 k = derivR e σ k`). So `Bgrad = revGradBndF σ e k W +
bridgeBnd e σ k` — the reverse-mode rounding term plus the bridge term — a concrete per-coordinate bound from the
trusted Float model. A single UNIFORM `Bgrad` for the sup-metric whole-run is the sup over coordinates `k` of that
per-coordinate bound.

`ascentStep_error_from_gradient_error` proves the single-step reduction; `hstep_from_gradient_error` lifts it to a
whole trajectory, giving exactly the `hstep` hypothesis of C27/C28/C29 with `B = |toReal lr|·Bgrad`.

**Scope (honestly disclosed):** the runnable gradient `grad` is taken abstractly with its error `≤ Bgrad`
(discharged concretely by C1's `reverseGrad_errorF_bounded`, whose ideal is `derivR e (envR σ)` — so the trajectory
params are the `toReal`-embedding of `Float` params, `envR σ`). The reduction assumes the runnable step is exactly
`σ k + lr·grad k` (the Float `axpy` update) up to the gradient error; the `lr` multiply's own rounding is folded
into `Bgrad`/the C1 bound's domain. A uniform `Bgrad` is the sup over coordinates of C1's per-coordinate bound.
-/
import Puffer.RL.WholeRunInterval
import Puffer.Float.AutoDiffR

open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)
open Puffer.RL.WholeRunInterval (gradAscentE)

namespace Puffer.RL.StepIntervalFromC1

/-- The runnable gradient-ascent step: `σ k ↦ σ k + lr · grad k`, using the Float-computed gradient `grad` in
    place of the ideal `derivR` (the `axpy` parameter update). -/
noncomputable def runnableStep (lr : Float) (σ grad : Nat → ℝ) : Nat → ℝ :=
  fun k => σ k + toReal lr * grad k

/-- **Single-step interval from the gradient error.** If the runnable gradient `grad` is within `Bgrad` of the
    ideal `derivR e σ` per coordinate (`∀ k, |grad k − derivR e σ k| ≤ Bgrad`), then one runnable step is within
    `|toReal lr|·Bgrad` of the ideal exact-ℝ ascent: `∀ k, |runnableStep lr σ grad k − gradAscentE e lr σ k| ≤
    |toReal lr|·Bgrad`. (`Bgrad` is discharged concretely by C1's `reverseGrad_errorF_bounded`.) -/
theorem ascentStep_error_from_gradient_error (e : Expr) (lr : Float) (σ grad : Nat → ℝ) (Bgrad : ℝ)
    (hgrad : ∀ k, |grad k - derivR e σ k| ≤ Bgrad) :
    ∀ k, |runnableStep lr σ grad k - gradAscentE e lr σ k| ≤ |toReal lr| * Bgrad := by
  intro k
  simp only [runnableStep, gradAscentE]
  calc |σ k + toReal lr * grad k - (σ k + toReal lr * derivR e σ k)|
      = |toReal lr * (grad k - derivR e σ k)| := by ring_nf
    _ = |toReal lr| * |grad k - derivR e σ k| := by rw [abs_mul]
    _ ≤ |toReal lr| * Bgrad := mul_le_mul_of_nonneg_left (hgrad k) (abs_nonneg _)

/-- **The whole-run `hstep` with `B = |lr|·Bgrad`.** For a runnable trajectory `θ` whose each step is the `axpy`
    update `θ(n+1) k = θ n k + lr·grad n k` with the per-coordinate gradient error `|grad n k − derivR e (θ n) k|
    ≤ Bgrad` (the C1 reverse-mode single-step interval), the whole-run's per-step hypothesis holds:
    `∀ n k, |θ(n+1) k − gradAscentE e lr (θ n) k| ≤ |toReal lr|·Bgrad`. This is exactly the `hstep` that
    C27/C28/C29 consume, with the concrete `B = |toReal lr|·Bgrad`. -/
theorem hstep_from_gradient_error (e : Expr) (lr : Float) (Bgrad : ℝ)
    (θ grad : Nat → (Nat → ℝ))
    (hθ : ∀ n k, θ (n + 1) k = θ n k + toReal lr * grad n k)
    (hgrad : ∀ n k, |grad n k - derivR e (θ n) k| ≤ Bgrad) :
    ∀ n k, |θ (n + 1) k - gradAscentE e lr (θ n) k| ≤ |toReal lr| * Bgrad := by
  intro n k
  have h := ascentStep_error_from_gradient_error e lr (θ n) (grad n) Bgrad (hgrad n) k
  simp only [runnableStep] at h
  rw [hθ n k]
  exact h

end Puffer.RL.StepIntervalFromC1
