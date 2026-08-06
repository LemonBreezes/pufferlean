/-
# PPO surrogate gradient-Lipschitz, made CONCRETE for a softmax policy

C16 (`SurrogateGradExpr`) reduced the surrogate's clip-interior gradient-Lipschitz to the ratio's `Lr`; C19
(`RatioGradExpr`) reduced the ratio's `Lr` to the new-policy log-prob's budgets `M`/`Lv`/`Dm`/`Dl`; and C17
(`LogSoftmaxBudgetExpr`) + C18 (`LogSoftmaxValueBudgetExpr`) discharged those four budgets for the actual
log-softmax. This module STITCHES the chain, giving the surrogate's gradient-Lipschitz purely in terms of the
network's `Smooth` logits (over a region, on the clip interior).

* `ppoSurrogateE_ratio_gradient_lipschitz` — the CLEAN composition (C16 ∘ C19): for `r = ratioE newLogp oldLogp`,
  given the clip-interior hypotheses and `newLogp`'s budgets, `|Δ∇(ppoSurrogateE (ratioE newLogp oldLogp) g lo
  hi)| ≤ |toReal g|·exp(M − toReal oldLogp)·(Dl + Dm·Lv)·δ` (budgets abstract).
* `ppoSurrogateE_softmax_gradient_lipschitz` — the FULLY CONCRETE version: for `newLogp = logSoftmaxE chosen
  (e :: es)` (a softmax over `Smooth` logits), the four budgets are discharged by C17/C18 with the partition floor
  `c = exp(−vMag R e)` (C9's `expSumE_floor`) and upper bound `U = Σⱼ exp(vMag R logitⱼ)` (`expSumE_upper`), so
  the bound is a computable constant in the network's logit budgets `vMag`/`vLip`/`dMag`/`dLip` — see `sfM`/`sfLv`/
  `sfDm`/`sfDl` (the concrete `M`/`Lv`/`Dm`/`Dl`). No free budget hypotheses remain.

**Scope (honestly disclosed):** the bound holds on the CLIP INTERIOR (both `σ`, `σ'` strictly inside the clip
window `(lo, hi)` at the ratio value) — off it the clip kink is intrinsically non-Lipschitz (C16/C7). The logits
`chosen`, `e`, `es` must be `Smooth` (C10 linear layers), over the region `|σ i| ≤ R`. The concrete constant
(`sfM`/`sfLv`/`sfDm`/`sfDl`) is large and nested (not algebraically simplified) — it is the honest composition of
the per-node budgets, computable from the network. `oldLogp` is the detached old-policy log-prob (a `Float`).
-/
import Puffer.RL.SurrogateGradExpr
import Puffer.RL.RatioGradExpr
import Puffer.RL.LogSoftmaxValueBudgetExpr

open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)
open Puffer.RL.SurrogateExpr (ppoSurrogateE ratioE)
open Puffer.RL.SurrogateGradExpr (ppoSurrogateE_interior_gradient_lipschitz)
open Puffer.RL.RatioGradExpr (ratioE_deriv_lip)
open Puffer.RL.SoftmaxExpr (logSoftmaxE expSumE expSumE_floor)
open Puffer.RL.LogSoftmaxBudgetExpr (logSoftmaxE_deriv_lip logSoftmaxE_deriv_mag expSumE_smooth)
open Puffer.RL.LogSoftmaxValueBudgetExpr (logSoftmaxE_value_lip logSoftmaxE_value_mag expSumE_upper)

namespace Puffer.RL.SurrogateConcreteExpr

/-- **Surrogate gradient-Lipschitz reduced to the ratio's log-prob budgets (C16 ∘ C19).** For `r = ratioE
    newLogp oldLogp`, on the clip interior (both points) and given the new-policy log-prob's budgets (value cap
    `M`, value-Lipschitz `Lv·δ`, derivative magnitude `Dm`, derivative-Lipschitz `Dl·δ`), the surrogate's
    gradient variation is `≤ |toReal g|·exp(M − toReal oldLogp)·(Dl + Dm·Lv)·δ`. Composes C16's
    `ppoSurrogateE_interior_gradient_lipschitz` with C19's `ratioE_deriv_lip`. -/
theorem ppoSurrogateE_ratio_gradient_lipschitz (newLogp : Expr) (oldLogp g lo hi : Float)
    (σ σ' : Nat → ℝ) (k : Nat) (M Lv Dm Dl δ : ℝ)
    (hloσ : toReal lo < evalR (ratioE newLogp oldLogp) σ)
    (hhiσ : evalR (ratioE newLogp oldLogp) σ < toReal hi)
    (hloσ' : toReal lo < evalR (ratioE newLogp oldLogp) σ')
    (hhiσ' : evalR (ratioE newLogp oldLogp) σ' < toReal hi)
    (hMσ : evalR newLogp σ ≤ M) (hMσ' : evalR newLogp σ' ≤ M)
    (hLv : |evalR newLogp σ - evalR newLogp σ'| ≤ Lv * δ)
    (hDmσ' : |derivR newLogp σ' k| ≤ Dm)
    (hDl : |derivR newLogp σ k - derivR newLogp σ' k| ≤ Dl * δ)
    (hDmn : 0 ≤ Dm) :
    |derivR (ppoSurrogateE (ratioE newLogp oldLogp) g lo hi) σ k
        - derivR (ppoSurrogateE (ratioE newLogp oldLogp) g lo hi) σ' k|
      ≤ |toReal g| * (Real.exp (M - toReal oldLogp) * (Dl + Dm * Lv) * δ) :=
  ppoSurrogateE_interior_gradient_lipschitz (ratioE newLogp oldLogp) g lo hi σ σ' k _
    hloσ hhiσ hloσ' hhiσ'
    (ratioE_deriv_lip newLogp oldLogp σ σ' k M Lv Dm Dl δ hMσ hMσ' hLv hDmσ' hDl hDmn)

/-- Concrete partition floor for a softmax over head-logit `e`: `c = exp(−vMag R e)` (C9's `expSumE_floor`). -/
noncomputable def sfC (R : ℝ) (e : Expr) : ℝ := Real.exp (-(vMag R e))

/-- Concrete partition upper bound: `U = Σⱼ exp(vMag R logitⱼ)` (`expSumE_upper`). -/
noncomputable def sfU (R : ℝ) (logits : List Expr) : ℝ := (logits.map (fun t => Real.exp (vMag R t))).sum

/-- Concrete log-softmax value cap `M = vMag R chosen + max |log c| |log U|` (C18's `logSoftmaxE_value_mag`). -/
noncomputable def sfM (R : ℝ) (chosen e : Expr) (logits : List Expr) : ℝ :=
  vMag R chosen + max |Real.log (sfC R e)| |Real.log (sfU R logits)|

/-- Concrete log-softmax value-Lipschitz `Lv = vLip R chosen + vLip R (expSumE logits)/c` (C18's
    `logSoftmaxE_value_lip`). -/
noncomputable def sfLv (R : ℝ) (chosen e : Expr) (logits : List Expr) : ℝ :=
  vLip R chosen + vLip R (expSumE logits) / sfC R e

/-- Concrete log-softmax derivative magnitude `Dm = dMag R chosen + dMag R (expSumE logits)/c` (C17's
    `logSoftmaxE_deriv_mag`). -/
noncomputable def sfDm (R : ℝ) (chosen e : Expr) (logits : List Expr) : ℝ :=
  dMag R chosen + dMag R (expSumE logits) / sfC R e

/-- Concrete log-softmax derivative-Lipschitz `Dl = dLip R chosen + (dLip R (expSumE logits)/c + dMag R (expSumE
    logits)·vLip R (expSumE logits)/c²)` (C17's `logSoftmaxE_deriv_lip`). -/
noncomputable def sfDl (R : ℝ) (chosen e : Expr) (logits : List Expr) : ℝ :=
  dLip R chosen + (dLip R (expSumE logits) / sfC R e
    + dMag R (expSumE logits) * vLip R (expSumE logits) / sfC R e ^ 2)

/-- **FULLY CONCRETE PPO surrogate gradient-Lipschitz for a softmax policy.** For `newLogp = logSoftmaxE chosen
    (e :: es)` (a softmax over `Smooth` logits, nonempty with `Smooth` head `e`), on the clip interior and over
    the region `|σ i| ≤ R`, the surrogate's gradient variation is bounded by a COMPUTABLE constant in the
    network's logit budgets:
    `|Δ∇(ppoSurrogateE (ratioE (logSoftmaxE chosen (e::es)) oldLogp) g lo hi)| ≤ |toReal g|·exp(sfM − toReal
    oldLogp)·(sfDl + sfDm·sfLv)·δ`, where `sfM`/`sfLv`/`sfDm`/`sfDl` are the concrete C17/C18 budgets with the
    partition floor `c = exp(−vMag R e)` and upper bound `U = Σⱼ exp(vMag R logitⱼ)` discharged automatically (C9
    `expSumE_floor` + `expSumE_upper`). NO free budget/floor hypotheses remain — this is the surrogate's
    gradient-Lipschitz stitched end-to-end (C16 ∘ C19 ∘ C17/C18) down to the `Smooth` logits. -/
theorem ppoSurrogateE_softmax_gradient_lipschitz (chosen e : Expr) (es : List Expr)
    (oldLogp g lo hi : Float) (hch : Smooth chosen) (he : Smooth e) (hes : ∀ lp ∈ es, Smooth lp)
    (σ σ' : Nat → ℝ) (R δ : ℝ) (k : Nat)
    (hσ : ∀ i, |σ i| ≤ R) (hσ' : ∀ i, |σ' i| ≤ R) (hδ : ∀ i, |σ i - σ' i| ≤ δ) (hR : 0 ≤ R)
    (hloσ : toReal lo < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ)
    (hhiσ : evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ < toReal hi)
    (hloσ' : toReal lo < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ')
    (hhiσ' : evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ' < toReal hi) :
    |derivR (ppoSurrogateE (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) g lo hi) σ k
        - derivR (ppoSurrogateE (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) g lo hi) σ' k|
      ≤ |toReal g| * (Real.exp (sfM R chosen e (e :: es) - toReal oldLogp)
          * (sfDl R chosen e (e :: es) + sfDm R chosen e (e :: es) * sfLv R chosen e (e :: es)) * δ) := by
  have hlog : ∀ lp ∈ (e :: es), Smooth lp :=
    fun lp hlp => (List.mem_cons.mp hlp).elim (fun h => h ▸ he) (fun h => hes lp h)
  have hc : (0:ℝ) < sfC R e := Real.exp_pos _
  have hfloorσ : sfC R e ≤ evalR (expSumE (e :: es)) σ := expSumE_floor e es he σ R hσ
  have hfloorσ' : sfC R e ≤ evalR (expSumE (e :: es)) σ' := expSumE_floor e es he σ' R hσ'
  have hceilσ : evalR (expSumE (e :: es)) σ ≤ sfU R (e :: es) := expSumE_upper (e :: es) hlog σ R hσ
  have hceilσ' : evalR (expSumE (e :: es)) σ' ≤ sfU R (e :: es) := expSumE_upper (e :: es) hlog σ' R hσ'
  apply ppoSurrogateE_ratio_gradient_lipschitz (logSoftmaxE chosen (e :: es)) oldLogp g lo hi σ σ' k
    (sfM R chosen e (e :: es)) (sfLv R chosen e (e :: es)) (sfDm R chosen e (e :: es))
    (sfDl R chosen e (e :: es)) δ hloσ hhiσ hloσ' hhiσ'
  · exact (le_abs_self _).trans (logSoftmaxE_value_mag chosen (e :: es) hch σ R (sfC R e)
      (sfU R (e :: es)) hσ hc hfloorσ hceilσ)
  · exact (le_abs_self _).trans (logSoftmaxE_value_mag chosen (e :: es) hch σ' R (sfC R e)
      (sfU R (e :: es)) hσ' hc hfloorσ' hceilσ')
  · exact logSoftmaxE_value_lip chosen (e :: es) hch hlog σ σ' R δ (sfC R e)
      hσ hσ' hδ hR hc hfloorσ hfloorσ'
  · exact logSoftmaxE_deriv_mag chosen (e :: es) hch hlog σ' R (sfC R e) k hσ' hR hc hfloorσ'
  · exact logSoftmaxE_deriv_lip chosen (e :: es) hch hlog σ σ' R δ (sfC R e) k
      hσ hσ' hδ hR hc hfloorσ hfloorσ'
  · exact add_nonneg (dMag_nonneg hch R hR)
      (div_nonneg (dMag_nonneg (expSumE_smooth (e :: es) hlog) R hR) hc.le)

end Puffer.RL.SurrogateConcreteExpr
