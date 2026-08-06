/-
The advantage-normalization SCALE on the true-arithmetic GAE: the VARIANCE of the runnable `computeGAE`
advantages (the spread `normalizeAdv` divides by, via `√var + ε`) tracks the variance of the TRUE-arithmetic
GAE advantages. The variance companion to `GAEMean` (a99's center), completing the mean+variance connection
of the GAE accuracy (a85–a98) with advantage-normalization (a76).

The variance is QUADRATIC in the entries, so — unlike the mean's linear sum — the per-slot budget is a
difference of squares. `varTerm_le` bounds `|(a − μ)² − (b − μ')²|` (different centers `μ ≠ μ'`) by routing
through the intermediate `(b − μ)²` and applying `abs_sq_sub_sq_le` (a78) TWICE: a value-shift `|a − b|·…`
and a mean-shift `|μ − μ'|·…`.

  • `trueMean` / `trueVar` — the ℝ mean/variance of the true-arithmetic GAE advantages (matching
    `lnMuR`/`lnVarR`'s shape).
  • `varTerm_le` — the per-term difference-of-squares bound (pure ℝ, axiom-clean).
  • `computeGAE_var_error` — the composite: `|toReal(advVar (computeGAE …).1) − trueVar|
    ≤ lnVarErr + (Σₜ varTermBnd)/cR` — a76's variance rounding `+` the per-term variance-difference budgets
    (value-shift = the a91 per-slot δ/w error, mean-shift = the a99 mean error) averaged over the trajectory.

The count exactness (`hc`/`hcpos`) is a hypothesis (no `toReal_ofNat`), as in a76/a99. With the center (a99)
and scale (here), the full `normalizeAdv`-on-true-GAE follows by composing `fullNormalizeAdv_error` (a87).

Axiom-clean beyond the trusted Float base (inherits a76's variance footprint; `varTerm_le` is pure ℝ).
-/
import Puffer.RL.GAETrue
import Puffer.RL.AdvNormBound
import Puffer.RL.PPOValuePerturb

namespace Puffer.RL.GAEVar

open Puffer.FloatR
open Puffer.RL.NNTrain
open Puffer.RL.GAEInvariant (gadvListR)
open Puffer.RL.GAETrue (trueGaeSuffix gaeSlotBnd computeGAE_totalAdv_error)
open Puffer.RL.BackwardLoopReduction (computeGAE_fst_size)
open Puffer.RL.LayerNormBound (lnMuR lnVarR lnVarErr)
open Puffer.RL.AdvNormBound (advVar advVar_error)
open Puffer.RL.PPOValuePerturb (abs_sq_sub_sq_le)

/-- The true-GAE mean and variance (matching `lnMuR`/`lnVarR`'s shape). -/
noncomputable def trueMean (traj : Array Transition) (gamma lam : Float) (n : Nat) (cR : ℝ) : ℝ :=
  (∑ t ∈ Finset.range n, gadvListR (trueGaeSuffix traj gamma lam n t)) / cR

noncomputable def trueVar (traj : Array Transition) (gamma lam : Float) (n : Nat) (cR : ℝ) : ℝ :=
  (∑ t ∈ Finset.range n,
      (gadvListR (trueGaeSuffix traj gamma lam n t) - trueMean traj gamma lam n cR)
        * (gadvListR (trueGaeSuffix traj gamma lam n t) - trueMean traj gamma lam n cR)) / cR

/-- Per-slot variance-difference budget: the value-shift factor and the mean-shift factor. -/
noncomputable def varTermBnd (adv : Array Float) (bk lnMu trueMu : ℝ) (k : Nat) : ℝ :=
  |toReal (adv[k]!) - bk| * (|toReal (adv[k]!) - lnMu| + |bk - lnMu|)
    + |lnMu - trueMu| * (|bk - lnMu| + |bk - trueMu|)

/-- **Per-term variance-difference bound** — `|(a − μ)² − (b − μ')²|` split through the intermediate center
    `(b − μ)²` and bounded by two `abs_sq_sub_sq_le` (value-shift `|a − b|` + mean-shift `|μ − μ'|`). -/
theorem varTerm_le (a b lnMu trueMu : ℝ) :
    |(a - lnMu) * (a - lnMu) - (b - trueMu) * (b - trueMu)|
      ≤ |a - b| * (|a - lnMu| + |b - lnMu|) + |lnMu - trueMu| * (|b - lnMu| + |b - trueMu|) := by
  have e1 : (a - lnMu) * (a - lnMu) - (b - trueMu) * (b - trueMu)
      = ((a - lnMu)^2 - (b - lnMu)^2) + ((lnMu - b)^2 - (trueMu - b)^2) := by ring
  rw [e1]
  refine (abs_add_le _ _).trans (add_le_add ?_ ?_)
  · exact abs_sq_sub_sq_le a b lnMu
  · have := abs_sq_sub_sq_le lnMu trueMu b
    rw [show |lnMu - b| + |trueMu - b| = |b - lnMu| + |b - trueMu| by
      rw [abs_sub_comm lnMu b, abs_sub_comm trueMu b]] at this
    exact this

/-- **Advantage-normalization SCALE on true GAE.** The variance of the runnable advantages tracks the
    variance of the TRUE-arithmetic GAE within `lnVarErr` (a76 rounding) + the per-term variance-difference
    budgets averaged over the trajectory (`varTerm_le`). Center (a99) + scale (here) ⇒ full normalize on
    true GAE via `fullNormalizeAdv_error`. -/
theorem computeGAE_var_error (traj : Array Transition) (gamma lam : Float) (cR : ℝ)
    (hc : toReal (Float.ofNat traj.size) = cR) (hcpos : 0 < cR) :
    |toReal (advVar (computeGAE traj gamma lam).1) - trueVar traj gamma lam traj.size cR|
      ≤ lnVarErr (fun k => (computeGAE traj gamma lam).1[k]!) (Float.ofNat traj.size) traj.size cR
        + (∑ t ∈ Finset.range traj.size,
            varTermBnd (computeGAE traj gamma lam).1 (gadvListR (trueGaeSuffix traj gamma lam traj.size t))
              (lnMuR (fun k => (computeGAE traj gamma lam).1[k]!) traj.size cR)
              (trueMean traj gamma lam traj.size cR) t) / cR := by
  set adv := (computeGAE traj gamma lam).1 with hadv
  have hsz : adv.size = traj.size := computeGAE_fst_size traj gamma lam
  have ha76 := advVar_error adv cR (by rw [hsz]; exact hc) hcpos
  rw [hsz] at ha76
  have hleg2 : |lnVarR (fun k => adv[k]!) traj.size cR - trueVar traj gamma lam traj.size cR|
      ≤ (∑ t ∈ Finset.range traj.size,
          varTermBnd adv (gadvListR (trueGaeSuffix traj gamma lam traj.size t))
            (lnMuR (fun k => adv[k]!) traj.size cR) (trueMean traj gamma lam traj.size cR) t) / cR := by
    rw [lnVarR, trueVar, div_sub_div_same, abs_div, abs_of_pos hcpos, ← Finset.sum_sub_distrib]
    refine div_le_div_of_nonneg_right ((Finset.abs_sum_le_sum_abs _ _).trans
      (Finset.sum_le_sum (fun t _ => ?_))) hcpos.le
    exact varTerm_le (toReal (adv[t]!)) (gadvListR (trueGaeSuffix traj gamma lam traj.size t))
      (lnMuR (fun k => adv[k]!) traj.size cR) (trueMean traj gamma lam traj.size cR)
  calc |toReal (advVar adv) - trueVar traj gamma lam traj.size cR|
      ≤ |toReal (advVar adv) - lnVarR (fun k => adv[k]!) traj.size cR|
        + |lnVarR (fun k => adv[k]!) traj.size cR - trueVar traj gamma lam traj.size cR| := abs_sub_le _ _ _
    _ ≤ lnVarErr (fun k => adv[k]!) (Float.ofNat traj.size) traj.size cR
        + (∑ t ∈ Finset.range traj.size,
            varTermBnd adv (gadvListR (trueGaeSuffix traj gamma lam traj.size t))
              (lnMuR (fun k => adv[k]!) traj.size cR) (trueMean traj gamma lam traj.size cR) t) / cR :=
        add_le_add ha76 hleg2

end Puffer.RL.GAEVar
