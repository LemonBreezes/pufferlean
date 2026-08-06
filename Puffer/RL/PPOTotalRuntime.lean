/-
The WHOLE PPO per-sample objective assembled on runnable `Float` — the clipped surrogate, the value loss,
and the entropy bonus combined into the one scalar `mlpGradPPO` ascends — bounded end-to-end. The RUNTIME
(Float↔ℝ) counterpart of `PPOTotalPerturb` (a81, the ℝ-level perturbation capstone), composing the
per-component runtime bounds (`ppoSurrF_error`, `valueLossF_error` a93, `entropyF_error` a97).

`ppoTotalObjF = ppoSurrF − vfCoef·valueLossF + entCoef·entropyF` (clipped surrogate `min(g·r, g·clip)` minus
the critic value loss plus the entropy bonus) — the standard PPO objective shape (`surrogate − vfCoef·vloss +
entCoef·H`, left-assoc). Here the value term is the CLIPPED value loss `valueLossF` (`½·max` of the two
squared errors); an `mlpGradPPO` body using the unclipped `½·(V−ret)²` swaps `valueLossF`/`valueLoss` for that
form — the composition is identical. `ppoTotalObjF_error` bounds its drift from the ℝ meaning
`ppoTotalObjR` (the same combination of the ℝ specs) by composing the three component budgets `Bsurr`/`Bvl`/
`Bent` through the two coefficient MULTIPLIES (`mulApprox_error`) and the SUB/ADD (`subApprox`/`addApprox`).

MODULAR (like a76's `normalizeAdv_error`): the three component budgets are hypotheses, discharged by the
respective runtime lemmas — `ppoSurrF_error`/`valueLossF_error` (a93) and `entropyF_error` (a97, needs the
`pᵢ > 0` log floor). The `vfCoef`/`entCoef` coefficients are exact `Float` inputs (their `toReal` is the ℝ
reference); the AD-tape's own logit-rounding is out of scope (as in a81).

Axiom-clean beyond the trusted Float base (`add/mul/sub_model` + `toReal`).
-/
import Puffer.RL.PPORuntime
import Puffer.RL.EntropyRuntime

namespace Puffer.RL.PPOTotalRuntime

open Puffer.FloatR
open Puffer.RL.PPO (ppoObjLoHi valueLoss)
open Puffer.RL.PPORuntime (ppoSurrF ppoSurrF_error valueLossF valueLossF_error)
open Puffer.RL.EntropyRuntime (entropyF entropyProbR entropyF_error)

/-- The full PPO per-sample objective on runnable `Float`: clipped surrogate − vfCoef·value loss +
    entCoef·entropy. Mirrors `mlpGradPPO`'s AD-tape objective as a standalone scalar. -/
def ppoTotalObjF (g r lo hi val valPred ret vfClip vfCoef entCoef : Float) (p : Nat → Float) (A : Nat) :
    Float :=
  ppoSurrF g r lo hi - vfCoef * valueLossF val valPred ret vfClip + entCoef * entropyF p A

/-- The ℝ meaning: the same combination of the ℝ specs. -/
noncomputable def ppoTotalObjR (g r lo hi val valPred ret vfClip vfCoef entCoef : ℝ) (pR : Nat → ℝ) (A : Nat) :
    ℝ :=
  ppoObjLoHi g r lo hi - vfCoef * valueLoss val valPred ret vfClip + entCoef * entropyProbR pR A

/-- **Full PPO objective runtime error.** The running `ppoTotalObjF` deviates from its ℝ meaning
    `ppoTotalObjR` by at most the two coefficient-multiply + sub/add roundings composed with the three
    component budgets `Bsurr` (surrogate), `Bvl` (value loss), `Bent` (entropy). Discharge the budgets with
    `ppoSurrF_error` / `valueLossF_error` (a93) and `entropyF_error` (a97). -/
theorem ppoTotalObjF_error (g r lo hi val valPred ret vfClip vfCoef entCoef : Float)
    (p : Nat → Float) (A : Nat) (Bsurr Bvl Bent : ℝ)
    (hsurr : |toReal (ppoSurrF g r lo hi) - ppoObjLoHi (toReal g) (toReal r) (toReal lo) (toReal hi)| ≤ Bsurr)
    (hvl : |toReal (valueLossF val valPred ret vfClip)
              - valueLoss (toReal val) (toReal valPred) (toReal ret) (toReal vfClip)| ≤ Bvl)
    (hent : |toReal (entropyF p A) - entropyProbR (fun i => toReal (p i)) A| ≤ Bent) :
    |toReal (ppoTotalObjF g r lo hi val valPred ret vfClip vfCoef entCoef p A)
        - ppoTotalObjR (toReal g) (toReal r) (toReal lo) (toReal hi) (toReal val) (toReal valPred)
            (toReal ret) (toReal vfClip) (toReal vfCoef) (toReal entCoef) (fun i => toReal (p i)) A|
      ≤ u64 * |toReal (ppoSurrF g r lo hi - vfCoef * valueLossF val valPred ret vfClip)
              + toReal (entCoef * entropyF p A)|
        + (u64 * |toReal (ppoSurrF g r lo hi) - toReal (vfCoef * valueLossF val valPred ret vfClip)|
            + Bsurr
            + (u64 * |toReal vfCoef * toReal (valueLossF val valPred ret vfClip)|
                + |toReal vfCoef| * Bvl))
          + (u64 * |toReal entCoef * toReal (entropyF p A)|
              + |toReal entCoef| * Bent) := by
  have h0 : ∀ z : Float, |toReal z - toReal z| ≤ 0 := fun z => by simp
  have hq1 : |toReal (vfCoef * valueLossF val valPred ret vfClip)
        - toReal vfCoef * valueLoss (toReal val) (toReal valPred) (toReal ret) (toReal vfClip)|
      ≤ u64 * |toReal vfCoef * toReal (valueLossF val valPred ret vfClip)| + |toReal vfCoef| * Bvl := by
    have := mulApprox_error vfCoef (valueLossF val valPred ret vfClip) (toReal vfCoef)
      (valueLoss (toReal val) (toReal valPred) (toReal ret) (toReal vfClip)) 0 Bvl (h0 vfCoef) hvl
    simpa using this
  have hq2 : |toReal (entCoef * entropyF p A)
        - toReal entCoef * entropyProbR (fun i => toReal (p i)) A|
      ≤ u64 * |toReal entCoef * toReal (entropyF p A)| + |toReal entCoef| * Bent := by
    have := mulApprox_error entCoef (entropyF p A) (toReal entCoef)
      (entropyProbR (fun i => toReal (p i)) A) 0 Bent (h0 entCoef) hent
    simpa using this
  have hs := subApprox_error (ppoSurrF g r lo hi) (vfCoef * valueLossF val valPred ret vfClip)
    (ppoObjLoHi (toReal g) (toReal r) (toReal lo) (toReal hi))
    (toReal vfCoef * valueLoss (toReal val) (toReal valPred) (toReal ret) (toReal vfClip))
    Bsurr _ hsurr hq1
  have hres := addApprox_error (ppoSurrF g r lo hi - vfCoef * valueLossF val valPred ret vfClip)
    (entCoef * entropyF p A)
    (ppoObjLoHi (toReal g) (toReal r) (toReal lo) (toReal hi)
      - toReal vfCoef * valueLoss (toReal val) (toReal valPred) (toReal ret) (toReal vfClip))
    (toReal entCoef * entropyProbR (fun i => toReal (p i)) A) _ _ hs hq2
  rw [ppoTotalObjF, ppoTotalObjR]
  simpa using hres

end Puffer.RL.PPOTotalRuntime
