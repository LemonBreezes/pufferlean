/-
# Value-loss and entropy terms → `Expr`

The PPO total objective is `ppoTotalObj = ppoObjective − vfCoef·((V−ret)²/2) + entCoef·H`
(`PPOTotalPerturb.ppoTotalObj`): the clipped surrogate (compiled in C8) minus a value loss plus an entropy
bonus. This module compiles the remaining two terms into the AD grammar `Expr`.

* **Value loss** `valueSqErrE V ret = (V − ret)²` (value-head output `V` vs return target `ret`). Being a `sub`
  of a `Smooth` `V` and a `const`, squared by `mul`, it lands in the `Smooth` fragment — so unlike the softmax
  (`log`) and hidden ReLUs, the value loss's gradient-Lipschitz is DIRECTLY C4's `derivR_lip`, with NO free
  hypotheses (no floor, no active region). `valueSqErrE_gradient_lipschitz` states exactly that.
* **Entropy** `entropyCatE logps = −Σᵢ exp(logpᵢ)·logpᵢ`, the categorical (Shannon) entropy
  `H = −Σᵢ πᵢ log πᵢ` in terms of the log-probabilities `logpᵢ` (with `πᵢ = exp(logpᵢ)`) — which are exactly the
  C9 log-softmax `Expr`s. The partition-sum `Σᵢ exp(logpᵢ)·logpᵢ` (`crossTermE`) is a fold of `add` (no n-ary
  sum in the grammar), negated via `sub (const 0) ·`.

**Scope (honestly disclosed):** value loss = value correctness + smoothness + C4 gradient-Lipschitz (the clean,
strongest case). Entropy = VALUE correctness only; it contains `log` nodes (inside each log-softmax `logpᵢ`) and
`exp·log` products, so it is NOT `Smooth` — its gradient-Lipschitz needs C5 (`exp`) + C6 (`log`, floor
discharged by C9's `expSumE_floor`), assembled downstream. The coefficients `vfCoef`, `1/2`, `entCoef` are
applied by the caller (via `scale`) when assembling `ppoTotalObj = surrogate − scale (vfCoef/2) valueSqErrE +
scale entCoef entropyCatE`; they are kept out of these core lemmas (`toReal` of a Float coefficient like `0.5`
is only approximately `0.5`, a separate rounding bridge). `ret` is the detached return target (`Float`).
-/
import Puffer.Float.AutoDiffR

open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)

namespace Puffer.RL.ValueEntropyExpr

/-- **Value squared-error `(V − ret)²`** as an `Expr`: the value head output `V` against the return target
    `ret` (a detached `Float`). The core of the PPO value loss (`vfCoef·((V−ret)²/2)`, coefficient applied by
    the caller). -/
def valueSqErrE (V : Expr) (ret : Float) : Expr := .mul (.sub V (.const ret)) (.sub V (.const ret))

/-- `evalR (valueSqErrE V ret) σ = (evalR V σ − toReal ret)²`. -/
theorem evalR_valueSqErrE (V : Expr) (ret : Float) (σ : Nat → ℝ) :
    evalR (valueSqErrE V ret) σ = (evalR V σ - toReal ret) ^ 2 := by
  simp only [valueSqErrE, evalR]; ring

/-- The value loss is `Smooth` whenever the value head `V` is (`sub`/`mul`/`const` are `Smooth` constructors) —
    so its gradient-Lipschitz is directly C4, with NO floor or active-region side condition. -/
theorem valueSqErrE_smooth (V : Expr) (ret : Float) (hV : Smooth V) :
    Smooth (valueSqErrE V ret) :=
  Smooth.mul (Smooth.sub hV (Smooth.const ret)) (Smooth.sub hV (Smooth.const ret))

/-- **Value-loss gradient-Lipschitz (directly via C4).** For a `Smooth` value head `V`, over the region
    `|σ i| ≤ R` (both points), `|derivR (valueSqErrE V ret) σ k − derivR (valueSqErrE V ret) σ' k| ≤
    dLip R (valueSqErrE V ret) · δ` — no positive-floor / active hypotheses (the value loss is polynomial). -/
theorem valueSqErrE_gradient_lipschitz (V : Expr) (ret : Float) (hV : Smooth V)
    (σ σ' : Nat → ℝ) (R δ : ℝ) (k : Nat)
    (hσ : ∀ i, |σ i| ≤ R) (hσ' : ∀ i, |σ' i| ≤ R) (hδ : ∀ i, |σ i - σ' i| ≤ δ) (hR : 0 ≤ R) :
    |derivR (valueSqErrE V ret) σ k - derivR (valueSqErrE V ret) σ' k|
      ≤ dLip R (valueSqErrE V ret) * δ :=
  derivR_lip (valueSqErrE_smooth V ret hV) σ σ' R δ k hσ hσ' hδ hR

/-- **The cross term `Σᵢ exp(logpᵢ)·logpᵢ`** (`= Σᵢ πᵢ log πᵢ`) as an `Expr`: a fold of `add` over
    `mul (exp logpᵢ) logpᵢ` (the grammar has no n-ary sum). Negated, this is the categorical entropy. -/
def crossTermE : List Expr → Expr
  | [] => .const 0
  | lp :: rest => .add (.mul (.exp lp) lp) (crossTermE rest)

theorem evalR_crossTermE (logps : List Expr) (σ : Nat → ℝ) :
    evalR (crossTermE logps) σ = (logps.map (fun lp => Real.exp (evalR lp σ) * evalR lp σ)).sum := by
  induction logps with
  | nil => simp [crossTermE, evalR]
  | cons lp rest ih => simp only [crossTermE, evalR, List.map_cons, List.sum_cons, ih]

/-- **Categorical (Shannon) entropy `H = −Σᵢ exp(logpᵢ)·logpᵢ`** as an `Expr`, over the log-probabilities
    `logpᵢ` (with `πᵢ = exp(logpᵢ)`) — exactly the C9 log-softmax outputs. Built as `sub (const 0) (crossTermE)`. -/
def entropyCatE (logps : List Expr) : Expr := .sub (.const 0) (crossTermE logps)

theorem evalR_entropyCatE (logps : List Expr) (σ : Nat → ℝ) :
    evalR (entropyCatE logps) σ = -(logps.map (fun lp => Real.exp (evalR lp σ) * evalR lp σ)).sum := by
  simp only [entropyCatE, evalR, evalR_crossTermE, Puffer.FloatR.toReal_zero, zero_sub]

end Puffer.RL.ValueEntropyExpr
