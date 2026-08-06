/-
CAPSTONE of the GAE→normalize story (a84–a101): the runnable NORMALIZED ADVANTAGE — the value the trainer
actually feeds the policy gradient — bounded against the ideal normalization of the TRUE-arithmetic GAE.

`updatePPO`/`updateAC` normalize the GAE advantages (`normalizeAdv (computeGAE …).1`) before the
policy-gradient step. This file lands a single bound on `(normalizeAdv (computeGAE traj gamma lam).1)[i]!`
versus `advNorm (trueₜ) (trueMean) (trueVar) ε` — the exact-arithmetic GAE advantage, centered and scaled by
the exact-arithmetic mean/variance — by the triangle through the Float-rounded entry statistics:

  • `fullNormalizeAdv_error` (a87) — the Float↔ℝ rounding at the ENTRY statistics: the runnable normalized
    advantage vs `advNorm (toReal advᵢ) (lnMuR) (lnVarR) ε` (the ℝ normalization at the array's own
    entry-mean/variance).
  • `advNorm_perturb` (a101) — the normalization map's SENSITIVITY: from the entry statistics
    (`toReal advᵢ`, `lnMuR`, `lnVarR`) to the true-GAE statistics (`trueₜ`, `trueMean`, `trueVar`).

`normalizeAdv_trueGAE_error` is the `add_le_add` of the two. The three shift magnitudes inside the a101 term —
`|toReal advᵢ − trueₜ|`, `|lnMuR − trueMean|`, `√|lnVarR − trueVar|` — are themselves bounded by a91
(per-slot), a99 (mean), a100 (variance): the value, center, and scale differences to the true-arithmetic GAE.
So this is the honest end-to-end statement; the fully-substituted computable bound is those chained in.

Axiom-clean beyond the trusted Float base (inherits a87's `add/sub/mul/div/sqrt` footprint; a101 is pure ℝ).
-/
import Puffer.RL.GAEVar
import Puffer.RL.AdvNormPerturb

namespace Puffer.RL.GAENormTrue

open Puffer.FloatR
open Puffer.RL.NNTrain
open Puffer.RL.GAEInvariant (gadvListR)
open Puffer.RL.GAETrue (trueGaeSuffix)
open Puffer.RL.GAEVar (trueMean trueVar)
open Puffer.RL.LayerNormExec (lMean)
open Puffer.RL.LayerNormBound (lnMuR lnVarR lnMeanErr lnVarErr)
open Puffer.RL.NNTrain (normalizeAdv)
open Puffer.RL.AdvNormBound (advVar fullNormalizeAdv_error)
open Puffer.RL.AdvNormPerturb (advNorm_perturb)
open Puffer.RL.Normalize (advNorm)

/-- **CAPSTONE: the runnable normalized advantage on the true-arithmetic GAE.** The advantage the trainer
    actually feeds the policy gradient — `(normalizeAdv (computeGAE …).1)[i]!` — is within
    `a87 (Float↔ℝ at the entry statistics) + a101 (advNorm sensitivity to the true-GAE value/mean/variance)`
    of the ideal `advNorm` of the true-arithmetic GAE. Ties together the whole GAE→normalize story (a84–a101). -/
theorem normalizeAdv_trueGAE_error (traj : Array Transition) (gamma lam : Float) (cR dmin : ℝ)
    (i : Nat) (hi : i < (computeGAE traj gamma lam).1.size)
    (hc : toReal (Float.ofNat (computeGAE traj gamma lam).1.size) = cR) (hcpos : 0 < cR)
    (hvnn : 0 ≤ toReal (advVar (computeGAE traj gamma lam).1))
    (hvRnn : 0 ≤ lnVarR (fun k => (computeGAE traj gamma lam).1[k]!) (computeGAE traj gamma lam).1.size cR)
    (hdmin : 0 < dmin)
    (hden : dmin ≤ |toReal (Float.sqrt (advVar (computeGAE traj gamma lam).1) + 1e-8)|)
    (hyR : Real.sqrt (lnVarR (fun k => (computeGAE traj gamma lam).1[k]!) (computeGAE traj gamma lam).1.size cR)
              + toReal (1e-8 : Float) ≠ 0)
    (htvnn : 0 ≤ trueVar traj gamma lam traj.size cR)
    (hεpos : 0 < toReal (1e-8 : Float)) :
    |toReal ((normalizeAdv (computeGAE traj gamma lam).1)[i]!)
        - advNorm (gadvListR (trueGaeSuffix traj gamma lam traj.size i))
            (trueMean traj gamma lam traj.size cR) (trueVar traj gamma lam traj.size cR)
            (toReal (1e-8 : Float))|
      ≤ (u64 * |toReal ((computeGAE traj gamma lam).1[i]! - lMean (computeGAE traj gamma lam).1)
              / toReal (Float.sqrt (advVar (computeGAE traj gamma lam).1) + 1e-8)|
          + ((u64 * |toReal (computeGAE traj gamma lam).1[i]! - toReal (lMean (computeGAE traj gamma lam).1)|
                + lnMeanErr (fun k => (computeGAE traj gamma lam).1[k]!) (Float.ofNat (computeGAE traj gamma lam).1.size) (computeGAE traj gamma lam).1.size cR)
              + |(toReal (computeGAE traj gamma lam).1[i]! - lnMuR (fun k => (computeGAE traj gamma lam).1[k]!) (computeGAE traj gamma lam).1.size cR)
                  / (Real.sqrt (lnVarR (fun k => (computeGAE traj gamma lam).1[k]!) (computeGAE traj gamma lam).1.size cR) + toReal (1e-8 : Float))|
                * (u64 * |toReal (Float.sqrt (advVar (computeGAE traj gamma lam).1)) + toReal (1e-8 : Float)|
                    + (u64 * Real.sqrt (toReal (advVar (computeGAE traj gamma lam).1))
                        + Real.sqrt (lnVarErr (fun k => (computeGAE traj gamma lam).1[k]!) (Float.ofNat (computeGAE traj gamma lam).1.size) (computeGAE traj gamma lam).1.size cR)))) / dmin)
        + (|toReal (computeGAE traj gamma lam).1[i]! - lnMuR (fun k => (computeGAE traj gamma lam).1[k]!) (computeGAE traj gamma lam).1.size cR|
            * Real.sqrt |lnVarR (fun k => (computeGAE traj gamma lam).1[k]!) (computeGAE traj gamma lam).1.size cR - trueVar traj gamma lam traj.size cR|
            / ((Real.sqrt (lnVarR (fun k => (computeGAE traj gamma lam).1[k]!) (computeGAE traj gamma lam).1.size cR) + toReal (1e-8 : Float))
                * (Real.sqrt (trueVar traj gamma lam traj.size cR) + toReal (1e-8 : Float)))
          + (|toReal (computeGAE traj gamma lam).1[i]! - gadvListR (trueGaeSuffix traj gamma lam traj.size i)|
              + |lnMuR (fun k => (computeGAE traj gamma lam).1[k]!) (computeGAE traj gamma lam).1.size cR - trueMean traj gamma lam traj.size cR|)
            / (Real.sqrt (trueVar traj gamma lam traj.size cR) + toReal (1e-8 : Float))) := by
  refine (abs_sub_le (toReal ((normalizeAdv (computeGAE traj gamma lam).1)[i]!))
      (advNorm (toReal (computeGAE traj gamma lam).1[i]!)
        (lnMuR (fun k => (computeGAE traj gamma lam).1[k]!) (computeGAE traj gamma lam).1.size cR)
        (lnVarR (fun k => (computeGAE traj gamma lam).1[k]!) (computeGAE traj gamma lam).1.size cR)
        (toReal (1e-8 : Float)))
      (advNorm (gadvListR (trueGaeSuffix traj gamma lam traj.size i))
        (trueMean traj gamma lam traj.size cR) (trueVar traj gamma lam traj.size cR)
        (toReal (1e-8 : Float)))).trans ?_
  refine add_le_add (fullNormalizeAdv_error (computeGAE traj gamma lam).1 i cR dmin hi hc hcpos hvnn hvRnn hdmin hden hyR) ?_
  exact advNorm_perturb (toReal (computeGAE traj gamma lam).1[i]!)
    (lnMuR (fun k => (computeGAE traj gamma lam).1[k]!) (computeGAE traj gamma lam).1.size cR)
    (lnVarR (fun k => (computeGAE traj gamma lam).1[k]!) (computeGAE traj gamma lam).1.size cR)
    (gadvListR (trueGaeSuffix traj gamma lam traj.size i))
    (trueMean traj gamma lam traj.size cR) (trueVar traj gamma lam traj.size cR)
    (toReal (1e-8 : Float)) hvRnn htvnn hεpos

end Puffer.RL.GAENormTrue
