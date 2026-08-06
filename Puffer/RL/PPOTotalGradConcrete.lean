/-
# GRAND CAPSTONE: the whole PPO total objective's gradient-Lipschitz, concrete for a softmax-MLP policy

This is the culmination of the C1→C25 composition chain. C14 assembled the total PPO objective's
gradient-Lipschitz into three per-term pieces (`Lsurr + |cv|·LvalSq + |ce|·Lent`); C23 made the surrogate term
concrete (stitched end-to-end to the network's `Smooth` logit budgets), C12/C4 made the value term concrete
(polynomial ⟹ `Smooth` ⟹ `dLip`), and C24 reduced the entropy term to uniform per-log-prob budgets. This module
wires them into ONE theorem: the gradient variation of the FULL PPO total objective
`ppoTotalObjE (ppoSurrogateE (ratioE (logSoftmaxE chosen (e::es)) oldLogp) g lo hi) (valueSqErrE V ret)
(entropyCatE logps) cv ce` — the clipped surrogate over a softmax-log-prob ratio, minus the value squared error,
plus the entropy bonus — is bounded by a COMPUTABLE constant:

    |Δ∇(total)|  ≤  |g|·exp(sfM − oldLogp)·(sfDl + sfDm·sfLv)·δ         (surrogate, C23 concrete)
                  + |cv|·dLip R (valueSqErrE V ret)·δ                    (value, C12/C4 concrete)
                  + |ce|·(length logps)·exp(M)·((M+1)·Dl + (M+2)·Dm·Lv)·δ  (entropy, C24)

where `sfM`/`sfLv`/`sfDm`/`sfDl` are C23's concrete log-softmax budgets (partition floor `c = exp(−vMag R e)` and
ceiling `U = Σⱼ exp(vMag R logitⱼ)` discharged). The surrogate and value contributions are FULLY concrete in the
network's `Smooth` logit budgets; the entropy contribution is in its uniform per-log-prob budgets `M`/`Lv`/`Dm`/
`Dl` (each log-prob discharged by C17/C18/C25, made uniform across log-probs by a max — the one remaining
mechanical step).

**Scope (honestly disclosed):** the bound holds on the clip INTERIOR (both `σ`, `σ'` strictly inside `(lo,hi)` at
the ratio value — off it the clip kink is intrinsically non-Lipschitz, C7/C16). The policy logits (`chosen`, `e`,
`es`) and the value head `V` must be `Smooth` (C10 linear layers; for a deep MLP, C22's active-region reduction
supplies `Smooth`-linearized gradients). The entropy's uniform log-prob budgets `Ment`/`Lvent`/`Dment`/`Dlent`
are hypotheses (discharged per-log-prob by C25's bundle, made uniform by a max over the logits — the last
mechanical step). The coefficients `g`/`cv`/`ce`/`oldLogp`/`ret` are `Float` (the advantage, value/entropy
coefficients with the `1/2` folded into `cv`, old log-prob, and return target). This is the whole-objective
gradient-Lipschitz constant `G` that C3 turns into the ideal-step Lipschitz `L = 1 + |lr|·G` and C2 accumulates
into the whole-training-run error interval — now CONCRETE for the actual softmax-MLP PPO loss (on the clip
interior, hidden units active for a deep net, entropy budgets uniform).
-/
import Puffer.RL.PPOObjectiveGrad
import Puffer.RL.SurrogateConcreteExpr
import Puffer.RL.EntropyConcreteExpr

open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)
open Puffer.RL.PPOObjectiveExpr (ppoTotalObjE)
open Puffer.RL.PPOObjectiveGrad (ppoTotalObjE_gradient_lipschitz_value)
open Puffer.RL.ValueEntropyExpr (valueSqErrE entropyCatE)
open Puffer.RL.SurrogateExpr (ppoSurrogateE ratioE)
open Puffer.RL.SoftmaxExpr (logSoftmaxE)
open Puffer.RL.SurrogateConcreteExpr (sfM sfLv sfDm sfDl ppoSurrogateE_softmax_gradient_lipschitz)
open Puffer.RL.EntropyConcreteExpr (entropyCatE_gradient_lipschitz_uniform)

namespace Puffer.RL.PPOTotalGradConcrete

/-- **THE GRAND CAPSTONE — the whole PPO total objective's gradient-Lipschitz, concrete for a softmax-MLP.** For
    the full compiled PPO loss `ppoTotalObjE (ppoSurrogateE (ratioE (logSoftmaxE chosen (e::es)) oldLogp) g lo hi)
    (valueSqErrE V ret) (entropyCatE logps) cv ce` — clipped surrogate over the softmax-log-prob ratio, minus the
    value squared error, plus the entropy bonus — the gradient variation between any two parameter points on the
    clip interior is bounded by a COMPUTABLE constant, the sum of the three per-term contributions:
    the surrogate's (C23, fully concrete in the network's `Smooth` logit budgets `sfM/sfLv/sfDm/sfDl`), the value's
    (C12/C4, the concrete `dLip R (valueSqErrE V ret)·δ`), and the entropy's (C24, in its uniform per-log-prob
    budgets). Composes C14's `ppoTotalObjE_gradient_lipschitz_value` (the assembly, value discharged) with C23's
    surrogate and C24's entropy bounds. This is the concrete whole-objective gradient-Lipschitz `G` that C3→C2
    turn into the whole-training-run error interval. -/
theorem ppoTotalObj_softmax_gradient_lipschitz
    (chosen e V : Expr) (es logps : List Expr) (oldLogp g lo hi cv ce ret : Float)
    (hch : Smooth chosen) (he : Smooth e) (hes : ∀ lp ∈ es, Smooth lp) (hV : Smooth V)
    (σ σ' : Nat → ℝ) (R δ : ℝ) (k : Nat)
    (hσ : ∀ i, |σ i| ≤ R) (hσ' : ∀ i, |σ' i| ≤ R) (hδ : ∀ i, |σ i - σ' i| ≤ δ) (hR : 0 ≤ R)
    (hloσ : toReal lo < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ)
    (hhiσ : evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ < toReal hi)
    (hloσ' : toReal lo < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ')
    (hhiσ' : evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ' < toReal hi)
    (Ment Lvent Dment Dlent : ℝ)
    (hMσ : ∀ lp ∈ logps, |evalR lp σ| ≤ Ment) (hMσ' : ∀ lp ∈ logps, |evalR lp σ'| ≤ Ment)
    (hLvE : ∀ lp ∈ logps, |evalR lp σ - evalR lp σ'| ≤ Lvent * δ)
    (hDmEσ : ∀ lp ∈ logps, |derivR lp σ k| ≤ Dment) (hDmEσ' : ∀ lp ∈ logps, |derivR lp σ' k| ≤ Dment)
    (hDlE : ∀ lp ∈ logps, |derivR lp σ k - derivR lp σ' k| ≤ Dlent * δ) (hDmEn : 0 ≤ Dment) :
    |derivR (ppoTotalObjE (ppoSurrogateE (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) g lo hi)
              (valueSqErrE V ret) (entropyCatE logps) cv ce) σ k
        - derivR (ppoTotalObjE (ppoSurrogateE (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) g lo hi)
              (valueSqErrE V ret) (entropyCatE logps) cv ce) σ' k|
      ≤ |toReal g| * (Real.exp (sfM R chosen e (e :: es) - toReal oldLogp)
            * (sfDl R chosen e (e :: es) + sfDm R chosen e (e :: es) * sfLv R chosen e (e :: es)) * δ)
        + |toReal cv| * (dLip R (valueSqErrE V ret) * δ)
        + |toReal ce| * ((logps.length : ℝ)
            * (Real.exp Ment * ((Ment + 1) * Dlent + (Ment + 2) * Dment * Lvent) * δ)) :=
  ppoTotalObjE_gradient_lipschitz_value
    (ppoSurrogateE (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) g lo hi)
    (entropyCatE logps) V cv ce ret hV σ σ' R δ k hσ hσ' hδ hR _ _
    (ppoSurrogateE_softmax_gradient_lipschitz chosen e es oldLogp g lo hi hch he hes
      σ σ' R δ k hσ hσ' hδ hR hloσ hhiσ hloσ' hhiσ')
    (entropyCatE_gradient_lipschitz_uniform logps σ σ' k Ment Lvent Dment Dlent δ
      hMσ hMσ' hLvE hDmEσ hDmEσ' hDlE hDmEn)

end Puffer.RL.PPOTotalGradConcrete
