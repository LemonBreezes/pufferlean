/-
The full actor-critic (A2C) objective under forward-pass perturbation — combining the policy and value
bounds into the ONE objective the `train-gae` trainer ascends.

`NNTrain.mlpGradAC` builds `adv · log π(a|s) − vfCoef · ½(V(s) − ret)²` on the AD tape (`term1 − term2`):
the policy-gradient term plus the (unclipped) critic regression. This file bounds how that combined objective
drifts when the shared net's forward pass rounds — composing the two halves already proven:

  • `acObjective` / `acObjective_perturb` — the ℝ objective within `|adv|·|Δlogp| + vfCoef·(|ΔV|·(|V−ret|+
    |V*−ret|)/2)` of a shifted objective (triangle: the policy term `|adv|`-Lipschitz in `logp`, the value
    term via a78's `abs_sq_sub_sq_le`).
  • `acObjective_run_perturb` — on the RUNNABLE net: the objective of the running actor log-policy (softmax
    over the first-`A` logits, a75) + critic value `V(s)` is within `|adv|·2ε + vfCoef·(Bv·(|V−ret|+|V*−ret|)
    /2)` of the ideal-logit objective, `ε` the forward-pass logit error (→ a73's log-policy 2-Lipschitz) and
    `Bv` the value-estimate error (a75's `value_error`). The whole A2C training objective, bounded end-to-end
    against forward-pass drift.

All ℝ (Mathlib) — `acObjective_perturb` is axiom-clean; the capstone inherits the trusted Float base via the
forward-pass logit bound.
-/
import Puffer.RL.LogPolicyBound
import Puffer.RL.PPOValuePerturb
import Puffer.RL.ActorCriticBound

namespace Puffer.RL.ACObjPerturb

open Puffer.FloatR
open Puffer.RL.PPOValuePerturb (abs_sq_sub_sq_le)

/-! ### The ℝ actor-critic objective perturbation -/

/-- The A2C / actor-critic objective (maximized): `adv · logπ − vfCoef · ½(V − ret)²`. -/
noncomputable def acObjective (adv logp V ret vfCoef : ℝ) : ℝ :=
  adv * logp - vfCoef * ((V - ret)^2 / 2)

/-- **AC objective perturbation.** Under a log-policy shift `|logpA − logpB|` and a value shift `|VA − VB|`,
    the objective moves within `|adv|·|Δlogp| + vfCoef·(|ΔV|·(|VA−ret|+|VB−ret|)/2)` — the policy term
    `|adv|`-Lipschitz in `logp`, the value term via the squared-error perturbation `abs_sq_sub_sq_le`. -/
theorem acObjective_perturb (adv logpA logpB VA VB ret vfCoef : ℝ) (hvf : 0 ≤ vfCoef) :
    |acObjective adv logpA VA ret vfCoef - acObjective adv logpB VB ret vfCoef|
      ≤ |adv| * |logpA - logpB| + vfCoef * (|VA - VB| * (|VA - ret| + |VB - ret|) / 2) := by
  have hval : |(VA - ret)^2/2 - (VB - ret)^2/2| ≤ |VA - VB| * (|VA - ret| + |VB - ret|) / 2 := by
    rw [show (VA - ret)^2/2 - (VB - ret)^2/2 = ((VA - ret)^2 - (VB - ret)^2)/2 by ring, abs_div,
      show |(2:ℝ)| = 2 by norm_num]
    linarith [abs_sq_sub_sq_le VA VB ret]
  rw [acObjective, acObjective]
  calc |adv * logpA - vfCoef * ((VA - ret)^2/2) - (adv * logpB - vfCoef * ((VB - ret)^2/2))|
      = |adv * (logpA - logpB) - vfCoef * ((VA - ret)^2/2 - (VB - ret)^2/2)| := by ring_nf
    _ ≤ |adv * (logpA - logpB)| + |vfCoef * ((VA - ret)^2/2 - (VB - ret)^2/2)| := abs_sub _ _
    _ ≤ |adv| * |logpA - logpB| + vfCoef * (|VA - VB| * (|VA - ret| + |VB - ret|) / 2) := by
        rw [abs_mul, abs_mul, abs_of_nonneg hvf]
        exact add_le_add le_rfl (mul_le_mul_of_nonneg_left hval hvf)

/-! ### Capstone: the running A2C objective under forward-pass error -/

open Puffer.Net
open Puffer.RL.NNTrain
open Puffer.RL.LogPolicyBound (logSoftmax_perturb)
open Puffer.RL.ActorCriticBound (value_error idealLogit acLogits acLogits_getElem!)
open Puffer.RL.ForwardExec (forwardAll_logit_error hRList)

/-- **Running-net A2C objective perturbation.** The objective of the running net — actor log-policy over the
    first-`A` logits + critic value `V(s)` — is within `|adv|·2ε + vfCoef·(Bv·(|V−ret|+|V*−ret|)/2)` of the
    ideal-logit objective, `ε` a uniform forward-pass logit-error bound (→ log-policy 2-Lipschitz) and `Bv`
    the value-estimate error (from `value_error`). The whole `train-gae` objective bounded end-to-end. -/
theorem acObjective_run_perturb (p : MLP) (obs : Array Float) (act : Nat) (adv ret vfCoef ε : ℝ)
    (ha : act < p.b2.size - 1) (hpos : 0 < p.b2.size) (hvf : 0 ≤ vfCoef)
    (hεbnd : ∀ k, k < p.b2.size - 1 →
      z1ErrBnd (p.W2[k]!).toList p.b2[k]! ((forwardAll p obs).2.1).toList
        + dotDiffBnd (p.W2[k]!).toList ((forwardAll p obs).2.1).toList (hRList p obs) ≤ ε) :
    |acObjective adv
          (Real.log (softmax (Finset.range (p.b2.size - 1)) (fun j => toReal (acLogits p obs)[j]!) act))
          (toReal ((policyAndValue p obs).2)) ret vfCoef
        - acObjective adv
          (Real.log (softmax (Finset.range (p.b2.size - 1)) (idealLogit p obs) act))
          (idealLogit p obs (p.b2.size - 1)) ret vfCoef|
      ≤ |adv| * (2*ε)
        + vfCoef * ((z1ErrBnd (p.W2[p.b2.size - 1]!).toList p.b2[p.b2.size - 1]! ((forwardAll p obs).2.1).toList
              + dotDiffBnd (p.W2[p.b2.size - 1]!).toList ((forwardAll p obs).2.1).toList (hRList p obs))
            * (|toReal ((policyAndValue p obs).2) - ret| + |idealLogit p obs (p.b2.size - 1) - ret|) / 2) := by
  have hlogp : |Real.log (softmax (Finset.range (p.b2.size - 1))
        (fun j => toReal (acLogits p obs)[j]!) act)
      - Real.log (softmax (Finset.range (p.b2.size - 1)) (idealLogit p obs) act)| ≤ 2*ε := by
    apply logSoftmax_perturb (Finset.range (p.b2.size - 1)) (fun j => toReal (acLogits p obs)[j]!)
      (idealLogit p obs) act ε (Finset.nonempty_range_iff.mpr (by omega)) (Finset.mem_range.mpr ha)
    intro k hk
    have hk' : k < p.b2.size - 1 := Finset.mem_range.mp hk
    rw [acLogits_getElem! p obs k hk']
    exact (forwardAll_logit_error p obs k (by omega)).trans (hεbnd k hk')
  have hV := value_error p obs hpos
  refine (acObjective_perturb adv _ _ _ _ ret vfCoef hvf).trans (add_le_add ?_ ?_)
  · exact mul_le_mul_of_nonneg_left hlogp (abs_nonneg _)
  · refine mul_le_mul_of_nonneg_left ?_ hvf
    have hF : (0:ℝ) ≤ |toReal ((policyAndValue p obs).2) - ret|
        + |idealLogit p obs (p.b2.size - 1) - ret| := by positivity
    nlinarith [mul_le_mul_of_nonneg_right hV hF, hF]

end Puffer.RL.ACObjPerturb
