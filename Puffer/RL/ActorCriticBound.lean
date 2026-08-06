/-
Both heads of the ACTOR-CRITIC net — the policy AND the value — within proven bounds on the runnable net.

The actor-critic trainer (`NNTrain.policyAndValue`, used by `train-gae`/`train-ppo`) runs ONE shared net whose
`dout = A + 1` outputs split into `A` action logits + `1` value: `policyAndValue p obs = (softmax (first A
logits), out[A])`. This is subtly different from `policyProbs` (a72/a73, which softmaxes ALL `b2.size`
logits): the actor-critic POLICY is the softmax over only the FIRST `A = |b2| − 1` logits (the "shorter
softmax" the a73 audit flagged — this is the policy whose `oldLogp` PPO actually stores), and the last logit
is the CRITIC's value estimate `V(s)`. This file bounds both, closing that gap:

  • `acLogits` — the first-`A` logits as an array (`(Array.range (|b2|−1)).map (out[·])`), with the exec
    reductions `policyAndValue_fst` (`= Train.softmax acLogits`), `policyAndValue_snd` (`= out[A]`).
  • `value_error` — the CRITIC's value estimate `(policyAndValue p obs).2` within the forward-pass bound of
    its ideal real value `idealLogit p obs (|b2|−1)` — directly `forwardAll_logit_error` at the value index.
  • `sq_loss_perturb` / `value_loss_error` — the value LOSS `½(V − R)²` (the PPO critic objective, target `R`
    fixed) within `Bv·(|V−R| + |V*−R|)/2` of its ideal — pure `½(u−R)² − ½(v−R)² = ½(u−v)((u−R)+(v−R))`.
  • `acPolicy_error` — the ACTOR policy `(policyAndValue p obs).1[i]!` within `Bsm + (e^{2ε} − 1)` of the
    ideal ℝ softmax of the first-`A` ideal logits, reusing `PolicyBound.softmax_logits_error` (the softmax
    Float-rounding budget `Bsm` + input perturbation) with the per-logit error from `forwardAll_logit_error`.

Per in-range index; `A = |b2| − 1 > 0`. Axiom-clean beyond the trusted Float base — the reductions are pure
logic, the bounds inherit `ForwardExec`/`PolicyBound`'s footprint.
-/
import Puffer.RL.PolicyBound

namespace Puffer.RL.ActorCriticBound

open Puffer.FloatR
open Puffer.Net
open Puffer.RL.NNTrain
open Puffer.RL.PolicyBound
open Puffer.RL.ForwardExec (forwardAll_logit_error hRList rangeMapGetElem!)

/-! ### The actor-critic split: first-`A` logits (actor) and the value index (critic) -/

/-- The actor's logits — the first `A = |b2| − 1` outputs (the value head `out[A]` dropped). -/
def acLogits (p : MLP) (obs : Array Float) : Array Float :=
  (Array.range (p.b2.size - 1)).map (fun k => (forwardAll p obs).2.2[k]!)

/-- The ideal REAL logit at index `k` (layer-2 with the exact real hidden). -/
noncomputable def idealLogit (p : MLP) (obs : Array Float) (k : Nat) : ℝ :=
  toReal p.b2[k]! + dotRm (p.W2[k]!).toList (hRList p obs)

theorem policyAndValue_fst (p : MLP) (obs : Array Float) :
    (policyAndValue p obs).1 = Puffer.RL.Train.softmax (acLogits p obs) := rfl

theorem policyAndValue_snd (p : MLP) (obs : Array Float) :
    (policyAndValue p obs).2 = (forwardAll p obs).2.2[p.b2.size - 1]! := rfl

theorem acLogits_size (p : MLP) (obs : Array Float) : (acLogits p obs).size = p.b2.size - 1 := by
  rw [acLogits, Array.size_map, Array.size_range]

theorem acLogits_getElem! (p : MLP) (obs : Array Float) (k : Nat) (hk : k < p.b2.size - 1) :
    (acLogits p obs)[k]! = (forwardAll p obs).2.2[k]! := by
  rw [acLogits, rangeMapGetElem! _ _ k hk]

/-! ### The critic: value estimate and value loss -/

/-- **Critic value-estimate error.** The value head `(policyAndValue p obs).2 = out[A]` is within the
    forward-pass bound of its ideal real value — `forwardAll_logit_error` at the value index `A = |b2| − 1`. -/
theorem value_error (p : MLP) (obs : Array Float) (hpos : 0 < p.b2.size) :
    |toReal ((policyAndValue p obs).2) - idealLogit p obs (p.b2.size - 1)|
      ≤ z1ErrBnd (p.W2[p.b2.size - 1]!).toList p.b2[p.b2.size - 1]! ((forwardAll p obs).2.1).toList
        + dotDiffBnd (p.W2[p.b2.size - 1]!).toList ((forwardAll p obs).2.1).toList (hRList p obs) := by
  show |toReal ((forwardAll p obs).2.2[p.b2.size - 1]!) - idealLogit p obs (p.b2.size - 1)| ≤ _
  exact forwardAll_logit_error p obs (p.b2.size - 1) (by omega)

/-- **Half-squared-error perturbation.** `|½(u−R)² − ½(v−R)²| ≤ Bv·(|u−R| + |v−R|)/2` when `|u−v| ≤ Bv`
    (pure algebra: `½(u−R)² − ½(v−R)² = ½(u−v)((u−R)+(v−R))`). -/
theorem sq_loss_perturb (u v R Bv : ℝ) (h : |u - v| ≤ Bv) :
    |(u - R)^2 / 2 - (v - R)^2 / 2| ≤ Bv * (|u - R| + |v - R|) / 2 := by
  have hfac : (u - R)^2 / 2 - (v - R)^2 / 2 = (u - v) * ((u - R) + (v - R)) / 2 := by ring
  rw [hfac, abs_div, abs_mul, abs_two]
  have h2 : |u - v| * |(u - R) + (v - R)| ≤ Bv * (|u - R| + |v - R|) :=
    mul_le_mul h (abs_add_le _ _) (abs_nonneg _) (le_trans (abs_nonneg _) h)
  linarith [h2]

/-- **Critic value-loss error.** The value loss `½(V − R)²` (target `R` fixed) is within
    `Bv·(|V−R| + |V*−R|)/2` of its ideal `½(V* − R)²`, `Bv` the value-estimate error. -/
theorem value_loss_error (p : MLP) (obs : Array Float) (R : ℝ) (hpos : 0 < p.b2.size) :
    |(toReal ((policyAndValue p obs).2) - R)^2 / 2
        - (idealLogit p obs (p.b2.size - 1) - R)^2 / 2|
      ≤ (z1ErrBnd (p.W2[p.b2.size - 1]!).toList p.b2[p.b2.size - 1]! ((forwardAll p obs).2.1).toList
          + dotDiffBnd (p.W2[p.b2.size - 1]!).toList ((forwardAll p obs).2.1).toList (hRList p obs))
        * (|toReal ((policyAndValue p obs).2) - R| + |idealLogit p obs (p.b2.size - 1) - R|) / 2 :=
  sq_loss_perturb (toReal ((policyAndValue p obs).2)) (idealLogit p obs (p.b2.size - 1)) R _
    (value_error p obs hpos)

/-! ### The actor: policy over the first-`A` logits -/

/-- **Actor policy error.** The actor-critic policy `(policyAndValue p obs).1[i]!` (the softmax over the
    first `A = |b2| − 1` logits — the policy whose `oldLogp` PPO stores) is within `Bsm + (e^{2ε} − 1)` of the
    ideal ℝ softmax of the first-`A` ideal logits, via `softmax_logits_error` (softmax Float-rounding budget
    `Bsm` + input perturbation `ε` from `forwardAll_logit_error`). -/
theorem acPolicy_error (p : MLP) (obs : Array Float) (i : Nat) (ε Bsm : ℝ)
    (hi : i < p.b2.size - 1) (hpos1 : 0 < p.b2.size - 1)
    (hsm : |toReal ((policyAndValue p obs).1[i]!)
             - softmax (Finset.range (p.b2.size - 1)) (fun j => toReal (acLogits p obs)[j]!) i| ≤ Bsm)
    (hε : 0 ≤ ε)
    (hεbnd : ∀ k, k < p.b2.size - 1 →
      z1ErrBnd (p.W2[k]!).toList p.b2[k]! ((forwardAll p obs).2.1).toList
        + dotDiffBnd (p.W2[k]!).toList ((forwardAll p obs).2.1).toList (hRList p obs) ≤ ε) :
    |toReal ((policyAndValue p obs).1[i]!)
        - softmax (Finset.range (p.b2.size - 1)) (idealLogit p obs) i| ≤ Bsm + (Real.exp (2*ε) - 1) := by
  have hfst : (policyAndValue p obs).1 = Puffer.RL.Train.softmax (acLogits p obs) := rfl
  rw [hfst]; rw [hfst] at hsm
  have hsize : (acLogits p obs).size = p.b2.size - 1 := acLogits_size p obs
  have hpert : ∀ k ∈ Finset.range (acLogits p obs).size,
      |toReal (acLogits p obs)[k]! - idealLogit p obs k| ≤ ε := by
    intro k hk
    rw [hsize] at hk
    have hk' : k < p.b2.size - 1 := Finset.mem_range.mp hk
    rw [acLogits_getElem! p obs k hk']
    exact (forwardAll_logit_error p obs k (by omega)).trans (hεbnd k hk')
  have := softmax_logits_error (acLogits p obs) (idealLogit p obs) i ε Bsm
    (by rw [hsize]; exact hi) (by rw [hsize]; exact hpos1) (by rw [hsize]; exact hsm) hε hpert
  rw [hsize] at this
  exact this

/-- **Sharp actor policy error.** The actor-critic policy `(policyAndValue p obs).1[i]!` (the softmax over
    the first `A = |b2| − 1` logits — the policy whose `oldLogp` PPO stores) is within
    `Bsm + π*ᵢ·(e^{2ε} − 1)` of the ideal ℝ softmax `π*` of the first-`A` ideal logits — the SAME softmax
    Float-rounding budget `Bsm` as `acPolicy_error`, but with the input-perturbation term SHARPENED by the
    ideal probability factor `π*ᵢ = softmax (range A) (idealLogit) i ≤ 1`. Strictly refines `acPolicy_error`
    (whose term is the uniform `e^{2ε} − 1`) by using the TIGHT softmax Lipschitz sandwich
    `softmax_input_perturb` instead of its `≤ 1`-relaxed corollary — every actor prob lies strictly in
    `(0,1)`, so the factor genuinely tightens the bound. `ε` is the per-logit forward-pass error (from
    `forwardAll_logit_error`); no `0 ≤ ε` hypothesis is needed for the sharp form. -/
theorem acPolicy_error_sharp (p : MLP) (obs : Array Float) (i : Nat) (ε Bsm : ℝ)
    (hi : i < p.b2.size - 1) (hpos1 : 0 < p.b2.size - 1)
    (hsm : |toReal ((policyAndValue p obs).1[i]!)
             - softmax (Finset.range (p.b2.size - 1)) (fun j => toReal (acLogits p obs)[j]!) i| ≤ Bsm)
    (hεbnd : ∀ k, k < p.b2.size - 1 →
      z1ErrBnd (p.W2[k]!).toList p.b2[k]! ((forwardAll p obs).2.1).toList
        + dotDiffBnd (p.W2[k]!).toList ((forwardAll p obs).2.1).toList (hRList p obs) ≤ ε) :
    |toReal ((policyAndValue p obs).1[i]!)
        - softmax (Finset.range (p.b2.size - 1)) (idealLogit p obs) i|
      ≤ Bsm + softmax (Finset.range (p.b2.size - 1)) (idealLogit p obs) i * (Real.exp (2*ε) - 1) := by
  set A := p.b2.size - 1 with hA
  have hs : (Finset.range A).Nonempty := Finset.nonempty_range_iff.mpr (by omega)
  have hi' : i ∈ Finset.range A := Finset.mem_range.mpr hi
  have hpert : ∀ k ∈ Finset.range A,
      |toReal (acLogits p obs)[k]! - idealLogit p obs k| ≤ ε := by
    intro k hk
    have hk' : k < A := Finset.mem_range.mp hk
    rw [acLogits_getElem! p obs k (by omega)]
    exact (forwardAll_logit_error p obs k (by omega)).trans (hεbnd k (by omega))
  have htight := softmax_input_perturb (Finset.range A) (fun j => toReal (acLogits p obs)[j]!)
    (idealLogit p obs) i ε hs hi' hpert
  calc |toReal ((policyAndValue p obs).1[i]!) - softmax (Finset.range A) (idealLogit p obs) i|
      ≤ |toReal ((policyAndValue p obs).1[i]!)
            - softmax (Finset.range A) (fun j => toReal (acLogits p obs)[j]!) i|
          + |softmax (Finset.range A) (fun j => toReal (acLogits p obs)[j]!) i
            - softmax (Finset.range A) (idealLogit p obs) i| := abs_sub_le _ _ _
    _ ≤ Bsm + softmax (Finset.range A) (idealLogit p obs) i * (Real.exp (2*ε) - 1) :=
        add_le_add hsm htight

end Puffer.RL.ActorCriticBound
