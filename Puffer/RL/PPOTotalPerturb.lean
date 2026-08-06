/-
The WHOLE PPO training objective under forward-pass perturbation — the clipped surrogate, the value loss, and
the entropy bonus combined into the one scalar `mlpGradPPO` (the `train-ppo` trainer) ascends, bounded
end-to-end. The capstone of the objective-perturbation story (a77–a80).

`NNTrain.mlpGradPPO` builds `obj = min(adv·ρ, adv·clip(ρ)) − vfCoef·½(V − ret)² + entCoef·H(π)` on the AD
tape — the PPO clipped surrogate (`ρ = exp(logπ(a) − oldLogp)`) minus the critic regression plus the entropy
bonus. This file bounds its drift under forward-pass rounding by composing the three already-proven halves:

  • `ppoTotalObj` / `ppoTotalObj_perturb` — the ℝ objective within `|adv|·|Δρ| + vfCoef·(|ΔV|·(|V−ret|+
    |V*−ret|)/2) + entCoef·|ΔH|` (triangle over the three terms: `ppoObjective` `|adv|`-Lipschitz in `ρ` via
    a77, the value loss via a78's `abs_sq_sub_sq_le`, entropy `entCoef`-scaled).
  • `ppoTotalObj_run_perturb` — on the RUNNABLE net: the whole PPO objective of the running actor policy is
    within `|adv|·ratio_ideal·(e^{2ε}−1) + vfCoef·(Bv·(|V−ret|+|V*−ret|)/2) + entCoef·(2ε + (e^{2ε}−1)·H_ideal)`
    of the ideal, `ε` the forward-pass logit error. Composes a77's `ratio_perturb`, a75's `value_error`, and
    a80's `entropy_perturb` — one forward-pass error budget `ε` driving all three terms.

All ℝ (Mathlib) — `ppoTotalObj_perturb` is axiom-clean; the capstone inherits the trusted Float base via the
forward-pass logit bound. Scope (as in a77/a80): this is the ℝ-level objective drift — the ratio/entropy are
the ℝ log-softmax of the ACTUAL logits and `V = toReal(out[A])`; the AD tape's OWN Float rounding of
`exp`/`logSumExp`/`mul` is not modelled here (it composes separately, like a72's softmax `Bsm`).
-/
import Puffer.RL.PPOPerturb
import Puffer.RL.PPOValuePerturb
import Puffer.RL.EntropyPerturb
import Puffer.RL.ActorCriticBound

namespace Puffer.RL.PPOTotalPerturb

open Puffer.FloatR
open Puffer.RL.PPOValuePerturb (abs_sq_sub_sq_le)

/-! ### The ℝ full-PPO-objective perturbation -/

/-- The full PPO per-sample objective: clipped surrogate `−` value loss `+` entropy bonus. -/
noncomputable def ppoTotalObj (adv ρ clipEps V ret vfCoef entCoef H : ℝ) : ℝ :=
  Puffer.RL.PPO.ppoObjective adv ρ clipEps - vfCoef * ((V - ret)^2 / 2) + entCoef * H

/-- **Full PPO objective perturbation.** Under ratio/value/entropy shifts, the total objective moves within
    `|adv|·|Δρ| + vfCoef·(|ΔV|·(|V−ret|+|V*−ret|)/2) + entCoef·|ΔH|` — triangle over the three terms
    (`ppoObjective` `|adv|`-Lipschitz in `ρ`, the value loss via `abs_sq_sub_sq_le`, entropy linear). -/
theorem ppoTotalObj_perturb (adv ρa ρb clipEps Va Vb ret vfCoef entCoef Ha Hb : ℝ)
    (hvf : 0 ≤ vfCoef) (hent : 0 ≤ entCoef) :
    |ppoTotalObj adv ρa clipEps Va ret vfCoef entCoef Ha
        - ppoTotalObj adv ρb clipEps Vb ret vfCoef entCoef Hb|
      ≤ |adv| * |ρa - ρb| + vfCoef * (|Va - Vb| * (|Va - ret| + |Vb - ret|) / 2)
        + entCoef * |Ha - Hb| := by
  have hval : |(Va - ret)^2/2 - (Vb - ret)^2/2| ≤ |Va - Vb| * (|Va - ret| + |Vb - ret|) / 2 := by
    rw [show (Va - ret)^2/2 - (Vb - ret)^2/2 = ((Va - ret)^2 - (Vb - ret)^2)/2 by ring, abs_div,
      show |(2:ℝ)| = 2 by norm_num]
    linarith [abs_sq_sub_sq_le Va Vb ret]
  rw [ppoTotalObj, ppoTotalObj]
  calc |Puffer.RL.PPO.ppoObjective adv ρa clipEps - vfCoef * ((Va - ret)^2/2) + entCoef * Ha
          - (Puffer.RL.PPO.ppoObjective adv ρb clipEps - vfCoef * ((Vb - ret)^2/2) + entCoef * Hb)|
      = |(Puffer.RL.PPO.ppoObjective adv ρa clipEps - Puffer.RL.PPO.ppoObjective adv ρb clipEps)
          - vfCoef * ((Va - ret)^2/2 - (Vb - ret)^2/2) + entCoef * (Ha - Hb)| := by ring_nf
    _ ≤ |(Puffer.RL.PPO.ppoObjective adv ρa clipEps - Puffer.RL.PPO.ppoObjective adv ρb clipEps)
            - vfCoef * ((Va - ret)^2/2 - (Vb - ret)^2/2)| + |entCoef * (Ha - Hb)| := abs_add_le _ _
    _ ≤ (|Puffer.RL.PPO.ppoObjective adv ρa clipEps - Puffer.RL.PPO.ppoObjective adv ρb clipEps|
            + |vfCoef * ((Va - ret)^2/2 - (Vb - ret)^2/2)|) + |entCoef * (Ha - Hb)| :=
          add_le_add (abs_sub _ _) le_rfl
    _ ≤ |adv| * |ρa - ρb| + vfCoef * (|Va - Vb| * (|Va - ret| + |Vb - ret|) / 2)
          + entCoef * |Ha - Hb| := by
        rw [abs_mul, abs_of_nonneg hvf, abs_mul, abs_of_nonneg hent]
        exact add_le_add (add_le_add (Puffer.RL.PPOPerturb.ppoObjective_perturb adv ρa ρb clipEps)
          (mul_le_mul_of_nonneg_left hval hvf)) le_rfl

/-! ### Capstone: the running-net PPO objective under forward-pass error -/

open Puffer.Net
open Puffer.RL.NNTrain
open Puffer.RL.PPOPerturb (ratio_perturb)
open Puffer.RL.EntropyPerturb (entropy entropy_perturb)
open Puffer.RL.ActorCriticBound (value_error idealLogit acLogits acLogits_getElem!)
open Puffer.RL.ForwardExec (forwardAll_logit_error hRList)

/-- **Running-net full PPO objective perturbation.** The whole PPO objective (`mlpGradPPO`'s clipped
    surrogate − value loss + entropy bonus) of the running actor policy (first-`A` softmax) + critic value is
    within `|adv|·ratio_ideal·(e^{2ε}−1) + vfCoef·(Bv·(|V−ret|+|V*−ret|)/2) + entCoef·(2ε + (e^{2ε}−1)·H_ideal)`
    of the ideal-logit objective, `ε` a uniform forward-pass logit-error bound driving all three terms. -/
theorem ppoTotalObj_run_perturb (p : MLP) (obs : Array Float) (act : Nat)
    (adv ret oldLogp clipEps vfCoef entCoef ε : ℝ)
    (ha : act < p.b2.size - 1) (hpos : 0 < p.b2.size) (hvf : 0 ≤ vfCoef) (hent : 0 ≤ entCoef)
    (hεbnd : ∀ k, k < p.b2.size - 1 →
      z1ErrBnd (p.W2[k]!).toList p.b2[k]! ((forwardAll p obs).2.1).toList
        + dotDiffBnd (p.W2[k]!).toList ((forwardAll p obs).2.1).toList (hRList p obs) ≤ ε) :
    |ppoTotalObj adv (Puffer.RL.PPO.ratio (Real.log
            (softmax (Finset.range (p.b2.size - 1)) (fun j => toReal (acLogits p obs)[j]!) act)) oldLogp)
          clipEps (toReal ((policyAndValue p obs).2)) ret vfCoef entCoef
          (entropy (Finset.range (p.b2.size - 1)) (fun j => toReal (acLogits p obs)[j]!))
        - ppoTotalObj adv (Puffer.RL.PPO.ratio (Real.log
            (softmax (Finset.range (p.b2.size - 1)) (idealLogit p obs) act)) oldLogp)
          clipEps (idealLogit p obs (p.b2.size - 1)) ret vfCoef entCoef
          (entropy (Finset.range (p.b2.size - 1)) (idealLogit p obs))|
      ≤ |adv| * (Puffer.RL.PPO.ratio (Real.log
            (softmax (Finset.range (p.b2.size - 1)) (idealLogit p obs) act)) oldLogp * (Real.exp (2*ε) - 1))
        + vfCoef * ((z1ErrBnd (p.W2[p.b2.size - 1]!).toList p.b2[p.b2.size - 1]! ((forwardAll p obs).2.1).toList
              + dotDiffBnd (p.W2[p.b2.size - 1]!).toList ((forwardAll p obs).2.1).toList (hRList p obs))
            * (|toReal ((policyAndValue p obs).2) - ret| + |idealLogit p obs (p.b2.size - 1) - ret|) / 2)
        + entCoef * (2*ε + (Real.exp (2*ε) - 1)
            * entropy (Finset.range (p.b2.size - 1)) (idealLogit p obs)) := by
  have hpert : ∀ k ∈ Finset.range (p.b2.size - 1),
      |toReal (acLogits p obs)[k]! - idealLogit p obs k| ≤ ε := by
    intro k hk
    have hk' : k < p.b2.size - 1 := Finset.mem_range.mp hk
    rw [acLogits_getElem! p obs k hk']
    exact (forwardAll_logit_error p obs k (by omega)).trans (hεbnd k hk')
  have hne : (Finset.range (p.b2.size - 1)).Nonempty := Finset.nonempty_range_iff.mpr (by omega)
  have hV := value_error p obs hpos
  refine (ppoTotalObj_perturb adv _ _ clipEps _ _ ret vfCoef entCoef _ _ hvf hent).trans
    (add_le_add (add_le_add ?_ ?_) ?_)
  · exact mul_le_mul_of_nonneg_left
      (ratio_perturb (Finset.range (p.b2.size - 1)) (fun j => toReal (acLogits p obs)[j]!)
        (idealLogit p obs) act oldLogp ε hne (Finset.mem_range.mpr ha) hpert) (abs_nonneg _)
  · refine mul_le_mul_of_nonneg_left ?_ hvf
    have hF : (0:ℝ) ≤ |toReal ((policyAndValue p obs).2) - ret|
        + |idealLogit p obs (p.b2.size - 1) - ret| := by positivity
    nlinarith [mul_le_mul_of_nonneg_right hV hF, hF]
  · exact mul_le_mul_of_nonneg_left
      (entropy_perturb (Finset.range (p.b2.size - 1)) (fun j => toReal (acLogits p obs)[j]!)
        (idealLogit p obs) ε hne hpert) hent

end Puffer.RL.PPOTotalPerturb
