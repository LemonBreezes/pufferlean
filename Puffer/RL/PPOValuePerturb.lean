/-
The PPO clipped VALUE loss under a value-prediction perturbation — the critic's trust-region objective's
sensitivity to a value-estimate shift, the value-head companion to a77's ratio/surrogate perturbation.

PPO's critic minimizes the pessimistic clipped value loss `½·max((V − ret)², (V_clipped − ret)²)`
(`PPO.valueLoss`, with `V_clipped = old + clip(V − old, −c, c)`). The value prediction `V = V(s)` is the
critic head's output, which carries the forward-pass error (a75's `value_error`). How far does this loss
drift when `V` moves? The bound composes three 1-Lipschitz facts with the squared-error perturbation:

  • `abs_sq_sub_sq_le` — `|(u−r)² − (v−r)²| ≤ |u−v|·(|u−r| + |v−r|)` (pure algebra).
  • `valueClipped_lipschitz` — `V_clipped` is 1-Lipschitz in `V` (reusing a77's `clampR_lipschitz`).
  • `valueLoss_perturb` — the clipped loss moves within `½·max(Bv·(|u−ret|+|v−ret|), Bv·(|vc(u)−ret|+
    |vc(v)−ret|))` (max-of-squares 1-Lipschitz, each branch bounded by the squared-error perturbation).
  • `valueLoss_run_perturb` — on the RUNNABLE net: the clipped value loss of the running critic's estimate
    `(policyAndValue p obs).2 = V(s)` is within that bound of the ideal-value loss, `Bv` the forward-pass
    value-estimate error (from a75's `value_error`).

All ℝ (Mathlib) — the perturbation lemmas are axiom-clean; `valueLoss_run_perturb` inherits `value_error`'s
footprint (the trusted Float base via the forward-pass logit bound).
-/
import Puffer.RL.PPOPerturb
import Puffer.RL.ActorCriticBound

namespace Puffer.RL.PPOValuePerturb

open Puffer.FloatR
open Puffer.RL.PPOPerturb (clampR_lipschitz)

/-! ### The ℝ value-loss perturbation -/

/-- `|(u−r)² − (v−r)²| ≤ |u−v|·(|u−r| + |v−r|)` (pure algebra: `(u−r)²−(v−r)² = (u−v)((u−r)+(v−r))`). -/
theorem abs_sq_sub_sq_le (u v r : ℝ) :
    |(u - r)^2 - (v - r)^2| ≤ |u - v| * (|u - r| + |v - r|) := by
  rw [show (u - r)^2 - (v - r)^2 = (u - v) * ((u - r) + (v - r)) by ring, abs_mul]
  exact mul_le_mul_of_nonneg_left (abs_add_le _ _) (abs_nonneg _)

/-- `valueClipped` is 1-Lipschitz in the value prediction (via `clampR`'s 1-Lipschitzness). -/
theorem valueClipped_lipschitz (val u v c : ℝ) :
    |Puffer.RL.PPO.valueClipped val u c - Puffer.RL.PPO.valueClipped val v c| ≤ |u - v| := by
  rw [Puffer.RL.PPO.valueClipped, Puffer.RL.PPO.valueClipped,
    show val + Puffer.RL.PPO.clampR (u - val) (-c) c - (val + Puffer.RL.PPO.clampR (v - val) (-c) c)
      = Puffer.RL.PPO.clampR (u - val) (-c) c - Puffer.RL.PPO.clampR (v - val) (-c) c from by ring]
  refine (clampR_lipschitz (u - val) (v - val) (-c) c).trans ?_
  rw [show u - val - (v - val) = u - v from by ring]

/-- **PPO value-loss perturbation.** When the value prediction moves within `Bv`, the clipped value loss moves
    within `½·max(Bv·(|u−ret|+|v−ret|), Bv·(|vc(u)−ret|+|vc(v)−ret|))` — max-of-squares (1-Lipschitz) composed
    with the per-branch squared-error perturbation and `valueClipped`'s 1-Lipschitzness. -/
theorem valueLoss_perturb (val u v ret c Bv : ℝ) (h : |u - v| ≤ Bv) :
    |Puffer.RL.PPO.valueLoss val u ret c - Puffer.RL.PPO.valueLoss val v ret c|
      ≤ 0.5 * max (Bv * (|u - ret| + |v - ret|))
          (Bv * (|Puffer.RL.PPO.valueClipped val u c - ret|
                  + |Puffer.RL.PPO.valueClipped val v c - ret|)) := by
  rw [Puffer.RL.PPO.valueLoss, Puffer.RL.PPO.valueLoss,
    show (0.5 : ℝ) * max ((u - ret)^2) ((Puffer.RL.PPO.valueClipped val u c - ret)^2)
        - 0.5 * max ((v - ret)^2) ((Puffer.RL.PPO.valueClipped val v c - ret)^2)
      = 0.5 * (max ((u - ret)^2) ((Puffer.RL.PPO.valueClipped val u c - ret)^2)
              - max ((v - ret)^2) ((Puffer.RL.PPO.valueClipped val v c - ret)^2)) from by ring,
    abs_mul, abs_of_pos (by norm_num : (0:ℝ) < 0.5)]
  refine mul_le_mul_of_nonneg_left ?_ (by norm_num)
  refine (abs_max_sub_max_le_max _ _ _ _).trans ?_
  apply max_le_max
  · exact (abs_sq_sub_sq_le u v ret).trans (mul_le_mul_of_nonneg_right h (by positivity))
  · exact (abs_sq_sub_sq_le _ _ ret).trans
      (mul_le_mul_of_nonneg_right ((valueClipped_lipschitz val u v c).trans h) (by positivity))

/-! ### Capstone: the running critic's value loss under forward-pass error -/

open Puffer.RL.NNTrain
open Puffer.RL.ActorCriticBound (value_error idealLogit)
open Puffer.RL.ForwardExec (hRList)

/-- **Running-net PPO value-loss perturbation.** The clipped value loss of the running critic's value estimate
    `(policyAndValue p obs).2 = V(s)` is within the perturbation bound of the ideal-value loss, `Bv` the
    forward-pass value-estimate error (from `value_error`). The critic-side companion to a77's
    `ppoObjective_run_perturb`. -/
theorem valueLoss_run_perturb (p : MLP) (obs : Array Float) (val ret c : ℝ) (hpos : 0 < p.b2.size) :
    |Puffer.RL.PPO.valueLoss val (toReal ((policyAndValue p obs).2)) ret c
        - Puffer.RL.PPO.valueLoss val (idealLogit p obs (p.b2.size - 1)) ret c|
      ≤ 0.5 * max
          ((z1ErrBnd (p.W2[p.b2.size - 1]!).toList p.b2[p.b2.size - 1]! ((forwardAll p obs).2.1).toList
              + dotDiffBnd (p.W2[p.b2.size - 1]!).toList ((forwardAll p obs).2.1).toList (hRList p obs))
            * (|toReal ((policyAndValue p obs).2) - ret| + |idealLogit p obs (p.b2.size - 1) - ret|))
          ((z1ErrBnd (p.W2[p.b2.size - 1]!).toList p.b2[p.b2.size - 1]! ((forwardAll p obs).2.1).toList
              + dotDiffBnd (p.W2[p.b2.size - 1]!).toList ((forwardAll p obs).2.1).toList (hRList p obs))
            * (|Puffer.RL.PPO.valueClipped val (toReal ((policyAndValue p obs).2)) c - ret|
                + |Puffer.RL.PPO.valueClipped val (idealLogit p obs (p.b2.size - 1)) c - ret|)) :=
  valueLoss_perturb val (toReal ((policyAndValue p obs).2)) (idealLogit p obs (p.b2.size - 1)) ret c _
    (value_error p obs hpos)

end Puffer.RL.PPOValuePerturb
