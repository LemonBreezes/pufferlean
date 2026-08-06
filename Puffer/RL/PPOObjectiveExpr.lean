/-
# PPO total-objective assembly → `Expr`

Ties the three compiled objective terms — the clipped surrogate (C8, `ppoSurrogateE`), the value loss
(C12, `valueSqErrE`), and the entropy (C12, `entropyCatE`) — into ONE `Expr` whose exact-ℝ evaluation equals
the pre-existing spec `PPOTotalPerturb.ppoTotalObj = ppoObjective − vfCoef·((V−ret)²/2) + entCoef·H`.

`ppoTotalObjE surr valSq ent cv ce = add (sub surr (scale cv valSq)) (scale ce ent)` — surrogate minus a scaled
value term plus a scaled entropy term. The value coefficient `cv` FOLDS `vfCoef` and the `1/2` into a single
`scale` coefficient (`vfCoef·(x/2) = (vfCoef/2)·x`), matching the spec exactly when `toReal cv = vfCoef/2`.

Three correctness levels:
* `evalR_ppoTotalObjE` — the structural evaluation `evalR surr σ − toReal cv·evalR valSq σ + toReal ce·evalR ent σ`.
* `evalR_ppoTotalObjE_eq` — the ℝ-spec match: `= ppoTotalObj adv ρ clipEps V ret vfCoef entCoef H`, given the
  three sub-`Expr`s realize the surrogate / `(V−ret)²` / `H` and the coefficients realize `vfCoef/2` and `entCoef`.
* `evalR_ppoTotalObjE_concrete` — the END-TO-END compile: plugging the actual C8 `ppoSurrogateE ratio g lo hi`
  and C12 `valueSqErrE V ret` builders, the assembled objective evaluates to `ppoTotalObj (toReal g) (evalR ratio σ)
  clipEps (evalR V σ) (toReal ret) vfCoef entCoef H` — the compiled Expr IS the real total PPO objective.

**Scope (honestly disclosed):** the coefficient hypotheses `toReal cv = vfCoef/2`, `toReal ce = entCoef` and the
clip-window `toReal lo = 1−clipEps`, `toReal hi = 1+clipEps` are where the Float-hyperparameter values enter as
their real counterparts; the rounding of `vfCoef/2` (etc.) into the Float `cv` is a separate bridge, not this
file's concern (kept as clean hypotheses, matching the C8 clip-bound pattern). The entropy sub-`Expr` `ent` is
taken abstractly with `hent : evalR ent σ = H` (its `H` is whatever the policy's entropy is — C12's `entropyCatE`
supplies the categorical Shannon `H = −Σ πᵢ log πᵢ`). This is the objective's VALUE compile; the objective's
gradient-Lipschitz combines the per-term results (value loss: C4 unconditional; surrogate: C8 interior; entropy:
C5+C6) downstream.
-/
import Puffer.RL.SurrogateExpr
import Puffer.RL.ValueEntropyExpr
import Puffer.RL.PPOTotalPerturb

open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)
open Puffer.RL.SurrogateExpr (ppoSurrogateE evalR_ppoSurrogateE_ppoObjective)
open Puffer.RL.ValueEntropyExpr (valueSqErrE evalR_valueSqErrE)

namespace Puffer.RL.PPOObjectiveExpr

/-- **The assembled PPO total objective as an `Expr`**: `surr − cv·valSq + ce·ent` (surrogate minus a scaled
    value term plus a scaled entropy term). `cv` folds `vfCoef` and the `1/2` of the value loss. -/
def ppoTotalObjE (surr valSq ent : Expr) (cv ce : Float) : Expr :=
  .add (.sub surr (.scale cv valSq)) (.scale ce ent)

/-- Structural evaluation of the assembled objective. -/
theorem evalR_ppoTotalObjE (surr valSq ent : Expr) (cv ce : Float) (σ : Nat → ℝ) :
    evalR (ppoTotalObjE surr valSq ent cv ce) σ
      = evalR surr σ - toReal cv * evalR valSq σ + toReal ce * evalR ent σ := by
  simp only [ppoTotalObjE, evalR]

/-- **ℝ-spec match (modular).** When the three sub-`Expr`s realize the surrogate `ppoObjective adv ρ clipEps`,
    the value squared error `(V−ret)²`, and the entropy `H`, and the coefficients realize `vfCoef/2` and
    `entCoef`, the assembled objective evaluates to exactly `PPOTotalPerturb.ppoTotalObj`. The `vfCoef·(x/2) =
    (vfCoef/2)·x` folding is discharged by `ring`. -/
theorem evalR_ppoTotalObjE_eq (surr valSq ent : Expr) (cv ce : Float) (σ : Nat → ℝ)
    (adv ρ clipEps V ret vfCoef entCoef H : ℝ)
    (hsurr : evalR surr σ = Puffer.RL.PPO.ppoObjective adv ρ clipEps)
    (hval : evalR valSq σ = (V - ret) ^ 2)
    (hent : evalR ent σ = H)
    (hcv : toReal cv = vfCoef / 2) (hce : toReal ce = entCoef) :
    evalR (ppoTotalObjE surr valSq ent cv ce) σ
      = Puffer.RL.PPOTotalPerturb.ppoTotalObj adv ρ clipEps V ret vfCoef entCoef H := by
  rw [evalR_ppoTotalObjE, hsurr, hval, hent, hcv, hce, Puffer.RL.PPOTotalPerturb.ppoTotalObj]
  ring

/-- **END-TO-END compile.** Plugging the actual C8 surrogate builder `ppoSurrogateE ratio g lo hi` and the C12
    value-loss builder `valueSqErrE V ret` (entropy abstract via `hent`), the assembled objective evaluates to
    the real ℝ total PPO objective `ppoTotalObj (toReal g) (evalR ratio σ) clipEps (evalR V σ) (toReal ret)
    vfCoef entCoef H`. The compiled `Expr` IS the PPO total objective. Discharges the surrogate term via C8's
    `evalR_ppoSurrogateE_ppoObjective` (clip window realizes `1∓clipEps`) and the value term via C12's
    `evalR_valueSqErrE`. -/
theorem evalR_ppoTotalObjE_concrete (ratio V ent : Expr) (g lo hi ret cv ce : Float) (σ : Nat → ℝ)
    (clipEps H vfCoef entCoef : ℝ)
    (hlo : toReal lo = 1 - clipEps) (hhi : toReal hi = 1 + clipEps) (hε : 0 ≤ clipEps)
    (hent : evalR ent σ = H)
    (hcv : toReal cv = vfCoef / 2) (hce : toReal ce = entCoef) :
    evalR (ppoTotalObjE (ppoSurrogateE ratio g lo hi) (valueSqErrE V ret) ent cv ce) σ
      = Puffer.RL.PPOTotalPerturb.ppoTotalObj (toReal g) (evalR ratio σ) clipEps
          (evalR V σ) (toReal ret) vfCoef entCoef H := by
  apply evalR_ppoTotalObjE_eq
  · exact evalR_ppoSurrogateE_ppoObjective ratio g lo hi σ clipEps hlo hhi hε
  · exact evalR_valueSqErrE V ret σ
  · exact hent
  · exact hcv
  · exact hce

end Puffer.RL.PPOObjectiveExpr
