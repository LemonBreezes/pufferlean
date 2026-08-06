/-
The advantage-normalization CENTER on the true-arithmetic GAE: the mean of the runnable `computeGAE`
advantages (the value `normalizeAdv` subtracts) tracks the mean of the TRUE-arithmetic GAE advantages. The
first step of composing the GAE accuracy (a85–a98) with the advantage-normalization bound (a76/a87).

`normalizeAdv adv` centers by `lMean adv = (Σ advₖ)/c`. This file bounds that Float mean against the ℝ mean
of the true GAE advantages `(Σₜ gadvListR(trueGaeSuffix t))/cR`, by the triangle:

  • `advMean_error` (a76) — the arithmetic-mean ROUNDING: `lMean adv` vs the ℝ mean of the array entries
    `lnMuR = (Σ toReal advₖ)/cR`.
  • `computeGAE_totalAdv_error` (a98) divided by `cR` — the entry mean vs the TRUE-GAE mean: since
    `lnMuR − trueMean = (Σ toReal advₖ − Σ trueₜ)/cR`, its absolute value is the a98 aggregate over `cR`.

`computeGAE_mean_error` composes them: `|toReal(lMean (computeGAE …).1) − (Σₜ gadvListR(trueGaeSuffix t))/cR|
≤ lnMeanErr + (Σₜ gaeSlotBnd)/cR`. The count exactness (`hc : toReal ⌊|adv|⌋ = cR`, `hcpos`) is a hypothesis
(no `toReal_ofNat` lemma), as in a76.

This is the CENTER only; the normalization SCALE (variance / `√var + ε`) tracking the true-GAE variance is the
disclosed follow-up (the variance is quadratic in the entries).

Axiom-clean beyond the trusted Float base (inherits a76's `div`/mean footprint + a98's `add/mul/sub_model`).
-/
import Puffer.RL.GAETrue
import Puffer.RL.AdvNormBound

namespace Puffer.RL.GAEMean

open Puffer.FloatR
open Puffer.RL.NNTrain
open Puffer.RL.GAEInvariant (gadvListR)
open Puffer.RL.GAETrue (trueGaeSuffix gaeSlotBnd computeGAE_totalAdv_error)
open Puffer.RL.BackwardLoopReduction (computeGAE_fst_size)
open Puffer.RL.LayerNormExec (lMean)
open Puffer.RL.LayerNormBound (lnMuR lnMeanErr)
open Puffer.RL.AdvNormBound (advMean_error)

/-- **Advantage-normalization center on true GAE.** The mean of the runnable advantages (what `normalizeAdv`
    subtracts) is within `lnMeanErr + (Σ gaeSlotBnd)/cR` of the mean of the TRUE-arithmetic GAE advantages —
    a76's mean rounding `+` a98's aggregate δ/w error averaged over the trajectory. -/
theorem computeGAE_mean_error (traj : Array Transition) (gamma lam : Float) (cR : ℝ)
    (hc : toReal (Float.ofNat (computeGAE traj gamma lam).1.size) = cR) (hcpos : 0 < cR) :
    |toReal (lMean (computeGAE traj gamma lam).1)
        - (∑ t ∈ Finset.range traj.size, gadvListR (trueGaeSuffix traj gamma lam traj.size t)) / cR|
      ≤ lnMeanErr (fun k => (computeGAE traj gamma lam).1[k]!)
            (Float.ofNat (computeGAE traj gamma lam).1.size) (computeGAE traj gamma lam).1.size cR
        + (∑ t ∈ Finset.range traj.size, gaeSlotBnd traj gamma lam traj.size t) / cR := by
  set adv := (computeGAE traj gamma lam).1 with hadv
  have hsz : adv.size = traj.size := computeGAE_fst_size traj gamma lam
  have ha76 := advMean_error adv cR hc hcpos
  have hmuR : lnMuR (fun k => adv[k]!) adv.size cR
      = (∑ t ∈ Finset.range traj.size, toReal (adv[t]!)) / cR := by
    rw [lnMuR, hsz]
  have hagg := computeGAE_totalAdv_error traj gamma lam
  have hleg2 : |lnMuR (fun k => adv[k]!) adv.size cR
        - (∑ t ∈ Finset.range traj.size, gadvListR (trueGaeSuffix traj gamma lam traj.size t)) / cR|
      ≤ (∑ t ∈ Finset.range traj.size, gaeSlotBnd traj gamma lam traj.size t) / cR := by
    rw [hmuR, div_sub_div_same, abs_div, abs_of_pos hcpos]
    exact div_le_div_of_nonneg_right hagg hcpos.le
  calc |toReal (lMean adv)
          - (∑ t ∈ Finset.range traj.size, gadvListR (trueGaeSuffix traj gamma lam traj.size t)) / cR|
      ≤ |toReal (lMean adv) - lnMuR (fun k => adv[k]!) adv.size cR|
        + |lnMuR (fun k => adv[k]!) adv.size cR
            - (∑ t ∈ Finset.range traj.size, gadvListR (trueGaeSuffix traj gamma lam traj.size t)) / cR| :=
        abs_sub_le _ _ _
    _ ≤ lnMeanErr (fun k => adv[k]!) (Float.ofNat adv.size) adv.size cR
        + (∑ t ∈ Finset.range traj.size, gaeSlotBnd traj gamma lam traj.size t) / cR :=
        add_le_add ha76 hleg2

/-- **True-GAE advantage mean lies within the pointwise bounds.** When every per-slot exact-arithmetic GAE
    advantage `gadvListR (trueGaeSuffix … t)` sits in `[L, U]`, and the denominator `cR` equals the trajectory
    count `n`, the true-GAE advantage MEAN `(∑ₜ gadvListR (trueGaeSuffix …)) / cR` (the ℝ normalization center
    that `computeGAE_mean_error` tracks) also sits in `[L, U]`. The canonical "arithmetic mean stays between the
    sample min and max", specialised to the exact GAE advantages — so any per-slot advantage bound transfers to
    the normalization center. Complements the error bound: this bounds the true mean's RANGE, not the Float gap. -/
theorem trueGaeMean_mem_Icc (traj : Array Transition) (gamma lam : Float) (n : Nat) (cR : ℝ) (L U : ℝ)
    (hn : (n : ℝ) = cR) (hcpos : 0 < cR)
    (hlo : ∀ t ∈ Finset.range n, L ≤ gadvListR (trueGaeSuffix traj gamma lam n t))
    (hhi : ∀ t ∈ Finset.range n, gadvListR (trueGaeSuffix traj gamma lam n t) ≤ U) :
    L ≤ (∑ t ∈ Finset.range n, gadvListR (trueGaeSuffix traj gamma lam n t)) / cR
      ∧ (∑ t ∈ Finset.range n, gadvListR (trueGaeSuffix traj gamma lam n t)) / cR ≤ U := by
  have hsumL : (∑ _t ∈ Finset.range n, L) = L * cR := by
    rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul, hn, mul_comm]
  have hsumU : (∑ _t ∈ Finset.range n, U) = U * cR := by
    rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul, hn, mul_comm]
  constructor
  · rw [le_div_iff₀ hcpos]
    calc L * cR = ∑ _t ∈ Finset.range n, L := hsumL.symm
      _ ≤ ∑ t ∈ Finset.range n, gadvListR (trueGaeSuffix traj gamma lam n t) := Finset.sum_le_sum hlo
  · rw [div_le_iff₀ hcpos]
    calc (∑ t ∈ Finset.range n, gadvListR (trueGaeSuffix traj gamma lam n t))
        ≤ ∑ _t ∈ Finset.range n, U := Finset.sum_le_sum hhi
      _ = U * cR := hsumU

end Puffer.RL.GAEMean
