/-
# PPO ratio gradient-Lipschitz: C16's `Lr` from the log-prob budgets

C16's surrogate gradient-Lipschitz (`SurrogateGradExpr.ppoSurrogateE_interior_gradient_lipschitz`) reduced its
bound to `|toReal g| · Lr`, where `Lr` is the policy ratio's gradient variation, left as a hypothesis. This
module supplies `Lr` from the new-policy log-prob's budgets, reusing C15's exp-node machinery.

The ratio is `ratioE newLogp oldLogp = exp (newLogp − oldLogp)` (C8, `SurrogateExpr`). Setting `A = sub newLogp
(const oldLogp)`, its derivative is `derivR (ratioE newLogp oldLogp) = exp(evalR A)·derivR A` (`derivR_ratioE`) —
EXACTLY the `exp(lp)·∂lp` shape of C15's `expDeriv_lip`. Since `const oldLogp` is `σ`-independent, `evalR A =
evalR newLogp − toReal oldLogp` and `derivR A = derivR newLogp`, so the ratio's gradient-Lipschitz `Lr` is
`expDeriv_lip` at `A` with the exp cap shifted by `−toReal oldLogp`:

    |Δ∇(ratioE newLogp oldLogp)|  ≤  exp(M − toReal oldLogp)·(Dl + Dm·Lv)·δ   (`ratioE_deriv_lip`)

given the new-policy log-prob budgets `evalR newLogp ≤ M`, `|Δ evalR newLogp| ≤ Lv·δ`, `|derivR newLogp| ≤ Dm`,
`|Δ derivR newLogp| ≤ Dl·δ`. Combined with C16, the surrogate's gradient-Lipschitz on the clip interior is
`|toReal g| · exp(M − toReal oldLogp)·(Dl + Dm·Lv)·δ`.

**Scope (honestly disclosed):** the new-policy log-prob budgets `M`/`Lv`/`Dm`/`Dl` are HYPOTHESES here — for a
softmax `newLogp` they are discharged by C17 (the derivative budgets `Dm`/`Dl`, `LogSoftmaxBudgetExpr`) plus the
log-softmax value budgets `M`/`Lv` (the latter needing a partition upper bound, a further step). This reduces the
ratio's `Lr` to those budgets; `oldLogp` is the detached old-policy log-prob (a `Float` constant), which only
shifts the exp cap.
-/
import Puffer.RL.SurrogateExpr
import Puffer.RL.EntropyGradExpr

open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)
open Puffer.RL.SurrogateExpr (ratioE)
open Puffer.RL.EntropyGradExpr (expDeriv_lip)

namespace Puffer.RL.RatioGradExpr

/-- The ratio's derivative: `derivR (ratioE newLogp oldLogp) σ k = exp(evalR newLogp σ − toReal oldLogp)·derivR
    newLogp σ k` (the `exp` chain rule; the detached `oldLogp` `const` contributes `0` to the inner derivative). -/
theorem derivR_ratioE (newLogp : Expr) (oldLogp : Float) (σ : Nat → ℝ) (k : Nat) :
    derivR (ratioE newLogp oldLogp) σ k
      = Real.exp (evalR newLogp σ - toReal oldLogp) * derivR newLogp σ k := by
  simp only [ratioE, derivR, evalR, sub_zero]

/-- **Ratio gradient-Lipschitz (`Lr`).** Given the new-policy log-prob's budgets (value cap `M`, value-Lipschitz
    `Lv·δ`, derivative magnitude `Dm`, derivative-Lipschitz `Dl·δ`), the PPO ratio's gradient variation is
    `|derivR (ratioE newLogp oldLogp) σ k − derivR (ratioE newLogp oldLogp) σ' k| ≤ exp(M − toReal oldLogp)·(Dl +
    Dm·Lv)·δ`. Reduces to C15's `expDeriv_lip` applied to `A = sub newLogp (const oldLogp)` (the `const oldLogp`
    cancels in the value difference and vanishes in the derivative, only shifting the exp cap to `M − toReal
    oldLogp`). This is exactly the `Lr` that C16's `ppoSurrogateE_interior_gradient_lipschitz` consumes. -/
theorem ratioE_deriv_lip (newLogp : Expr) (oldLogp : Float) (σ σ' : Nat → ℝ) (k : Nat)
    (M Lv Dm Dl δ : ℝ)
    (hMσ : evalR newLogp σ ≤ M) (hMσ' : evalR newLogp σ' ≤ M)
    (hLv : |evalR newLogp σ - evalR newLogp σ'| ≤ Lv * δ)
    (hDmσ' : |derivR newLogp σ' k| ≤ Dm)
    (hDl : |derivR newLogp σ k - derivR newLogp σ' k| ≤ Dl * δ)
    (hDmn : 0 ≤ Dm) :
    |derivR (ratioE newLogp oldLogp) σ k - derivR (ratioE newLogp oldLogp) σ' k|
      ≤ Real.exp (M - toReal oldLogp) * (Dl + Dm * Lv) * δ := by
  have hA : ∀ τ : Nat → ℝ, derivR (ratioE newLogp oldLogp) τ k
      = Real.exp (evalR (.sub newLogp (.const oldLogp)) τ) * derivR (.sub newLogp (.const oldLogp)) τ k :=
    fun τ => by simp only [ratioE, derivR]
  rw [hA σ, hA σ']
  apply expDeriv_lip (.sub newLogp (.const oldLogp)) σ σ' k (M - toReal oldLogp) Lv Dm Dl δ
  · simp only [evalR]; linarith [hMσ]
  · simp only [evalR]; linarith [hMσ']
  · simp only [evalR]
    rw [show (evalR newLogp σ - toReal oldLogp) - (evalR newLogp σ' - toReal oldLogp)
        = evalR newLogp σ - evalR newLogp σ' from by ring]
    exact hLv
  · simp only [derivR, sub_zero]; exact hDmσ'
  · simp only [derivR, sub_zero]; exact hDl
  · exact hDmn

end Puffer.RL.RatioGradExpr
