/-
# Surrogate gradient-Lipschitz (the surrogate per-term bound of C14)

C14's total-objective gradient-Lipschitz assembly left the surrogate term's `Lsurr` as a hypothesis; C15
supplied the entropy's, and the value's is concrete via C4. This module supplies the surrogate's, on the clip
interior.

C8's `ppoSurrogateE_interior` established that on the strict clip interior (`toReal lo < evalR r σ < toReal hi`)
the surrogate's gradient COLLAPSES to the unclipped `derivR (ppoSurrogateE r g lo hi) σ k = toReal g · derivR r
σ k`. So when BOTH parameter points are on the interior, the surrogate's gradient VARIATION reduces to the
ratio's, scaled by `|toReal g|`:

    |Δ∇(ppoSurrogateE r g lo hi)|  ≤  |toReal g| · |Δ∇r|   (`ppoSurrogateE_interior_gradient_lipschitz`)

and, when the ratio `r` is `Smooth`, C4's `derivR_lip` makes it fully concrete
(`ppoSurrogateE_interior_gradient_lipschitz_smooth`):

    |Δ∇(ppoSurrogateE r g lo hi)|  ≤  |toReal g| · dLip R r · δ.

This is exactly the `Lsurr` hypothesis of C14's `ppoTotalObjE_gradient_lipschitz`, with
`Lsurr = |toReal g| · dLip R r · δ` for a `Smooth` ratio.

**Scope (honestly disclosed):** the bound holds on the clip INTERIOR (both σ and σ' strictly between the clip
bounds) — off the interior the clip kink makes the surrogate gradient jump (C8's disclosure; intrinsic, like
C7's relu). The `Smooth`-ratio corollary is fully concrete via C4; the actual PPO ratio `r = exp(logp_new −
logp_old)` with a softmax `logp_new` is NOT `Smooth` (its log-partition is a `log` node), so its `|Δ∇r|` bound
composes C5 (`exp`) + C6 (`log`, floor from C9) on the log-softmax (the same budget machinery as C15's entropy),
which is a further step — here the surrogate is REDUCED to whatever the ratio's gradient-Lipschitz `Lr` is.
-/
import Puffer.RL.SurrogateExpr

open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)
open Puffer.RL.SurrogateExpr (ppoSurrogateE ppoSurrogateE_interior)

namespace Puffer.RL.SurrogateGradExpr

/-- **Surrogate gradient-Lipschitz on the clip interior (reduced to the ratio).** When both parameter points are
    strictly inside the clip window (`toReal lo < evalR r σ, evalR r σ' < toReal hi`), the surrogate's gradient
    variation is `|toReal g|` times the ratio's: `|derivR (ppoSurrogateE r g lo hi) σ k − derivR (ppoSurrogateE r
    g lo hi) σ' k| ≤ |toReal g| · Lr`, given the ratio's gradient variation `|derivR r σ k − derivR r σ' k| ≤ Lr`.
    Uses C8's `ppoSurrogateE_interior` (gradient collapses to `toReal g · derivR r`) at both points. -/
theorem ppoSurrogateE_interior_gradient_lipschitz (r : Expr) (g lo hi : Float)
    (σ σ' : Nat → ℝ) (k : Nat) (Lr : ℝ)
    (hloσ : toReal lo < evalR r σ) (hhiσ : evalR r σ < toReal hi)
    (hloσ' : toReal lo < evalR r σ') (hhiσ' : evalR r σ' < toReal hi)
    (hr : |derivR r σ k - derivR r σ' k| ≤ Lr) :
    |derivR (ppoSurrogateE r g lo hi) σ k - derivR (ppoSurrogateE r g lo hi) σ' k|
      ≤ |toReal g| * Lr := by
  rw [(ppoSurrogateE_interior r g lo hi σ k hloσ hhiσ).2,
      (ppoSurrogateE_interior r g lo hi σ' k hloσ' hhiσ').2, ← mul_sub, abs_mul]
  exact mul_le_mul_of_nonneg_left hr (abs_nonneg _)

/-- **Surrogate gradient-Lipschitz, fully concrete for a `Smooth` ratio.** With the ratio `r` `Smooth`, C4's
    `derivR_lip` gives `|Δ∇r| ≤ dLip R r · δ`, so on the clip interior `|Δ∇(ppoSurrogateE r g lo hi)| ≤ |toReal g|
    · dLip R r · δ` — exactly the `Lsurr = |toReal g| · dLip R r · δ` that C14's `ppoTotalObjE_gradient_lipschitz`
    consumes. -/
theorem ppoSurrogateE_interior_gradient_lipschitz_smooth (r : Expr) (g lo hi : Float) (hrs : Smooth r)
    (σ σ' : Nat → ℝ) (R δ : ℝ) (k : Nat)
    (hσ : ∀ i, |σ i| ≤ R) (hσ' : ∀ i, |σ' i| ≤ R) (hδ : ∀ i, |σ i - σ' i| ≤ δ) (hR : 0 ≤ R)
    (hloσ : toReal lo < evalR r σ) (hhiσ : evalR r σ < toReal hi)
    (hloσ' : toReal lo < evalR r σ') (hhiσ' : evalR r σ' < toReal hi) :
    |derivR (ppoSurrogateE r g lo hi) σ k - derivR (ppoSurrogateE r g lo hi) σ' k|
      ≤ |toReal g| * (dLip R r * δ) :=
  ppoSurrogateE_interior_gradient_lipschitz r g lo hi σ σ' k _ hloσ hhiσ hloσ' hhiσ'
    (derivR_lip hrs σ σ' R δ k hσ hσ' hδ hR)

end Puffer.RL.SurrogateGradExpr
