/-
The advantage-normalization step — `(adv − mean)/(√var + ε)`, Puffer's `adv_normalized` — with a proven
Float↔ℝ accuracy bound landing on the RUNNABLE `NNTrain.normalizeAdv`.

PPO/GAE normalize the minibatch advantages to mean 0 / unit std before the policy-gradient step
(`normalizeAdv`); this is `(advᵢ − μ)/(√σ² + ε)` — the SAME mean/variance reductions as LayerNorm, but with
the epsilon OUTSIDE the sqrt (`√var + ε`, not `√(var + ε)`), matching the ℝ spec `Normalize.advNorm`. This
file closes that trifecta on the code that runs:

  • `advNormF` / `advNorm_error` — the functional model `(a − mean)/(√var + ε)` within the composed
    `sub`/`sqrt`/`add`/`div` bound of `Normalize.advNorm (toReal a) meanR varR (toReal ε)`, given the mean/var
    error budgets. The NEW denominator shape `√var + ε` (`sqrtApprox_error` then `addApprox_error` — ε added
    AFTER the root, vs LayerNorm's `add`-then-`sqrt`).
  • `advVar` / `advVar_eq_lVar` — `normalizeAdv`'s direct square-accumulating fold IS `LayerNormExec.lVar`
    (via `Array.foldl_map`), so the mean/variance reuse LayerNorm's proven machinery.
  • `advMean_error` / `advVar_error` — the concrete mean/variance errors to the ℝ references (`lnMuR`/`lnVarR`),
    reusing `LayerNormBound.lnMean_error`/`lnVar_error` (the count `c = |adv|` exact via `hc`).
  • `normalizeAdv_getElem!` / `normalizeAdv_error` — the exec connection: `(normalizeAdv adv)[i]!` equals
    `advNormF adv[i]! (mean) (var) 1e-8` (pure `Array.map` unfolding), so the composed bound lands on the
    actual normalized advantage the trainer feeds the gradient. Discharge the mean/var budgets via
    `advMean_error`/`advVar_error` for a fully concrete bound.
  • `fullNormalizeAdv_error` — the a66-analog: chains `advMean_error`/`advVar_error` into `normalizeAdv_error`
    so the ℝ references are the concrete `lnMuR`/`lnVarR` and the budgets the concrete `lnMeanErr`/`lnVarErr`,
    leaving NO abstract error hypotheses (only count-exactness, variance-nonneg, and the denominator floor).

Axiom-clean beyond the trusted Float base — the exec reduction is pure logic; the bounds inherit
LayerNorm's footprint (`add/sub/mul/div/sqrt_model` + `toReal`).
-/
import Puffer.RL.Normalize
import Puffer.RL.LayerNormExec
import Puffer.RL.NNTrain
import Puffer.RL.FrobFoldAccuracy

namespace Puffer.RL.AdvNormBound

open Puffer.FloatR
open Puffer.RL.NNTrain
open Puffer.RL.LayerNormBound
open Puffer.RL.LayerNormExec (lMean lVar lMean_eq lVar_eq)
open Puffer.RL.SoftmaxExec (arrMapGetElem!)
open Puffer.RL.FrobFoldAccuracy (listFoldAdd_lb)

/-! ### The functional model and its bound (denominator `√var + ε`) -/

/-- Float normalized advantage `(a − mean)/(√var + ε)` (ε OUTSIDE the sqrt). -/
def advNormF (a mean var eps : Float) : Float := (a - mean) / (Float.sqrt var + eps)

/-- **Advantage-normalization error.** `advNormF` is within the composed `sub`/`sqrt`/`add`/`div` bound of
    `Normalize.advNorm (toReal a) meanR varR (toReal eps)`, given the mean/var errors `εmean`,`εvar` and the
    denominator floor `dmin` (`√var + ε ≥ ε > 0`). `a` exact (the input advantage). -/
theorem advNorm_error (a mean var eps : Float) (meanR varR εmean εvar dmin : ℝ)
    (hmean : |toReal mean - meanR| ≤ εmean) (hvar : |toReal var - varR| ≤ εvar)
    (hvnn : 0 ≤ toReal var) (hvRnn : 0 ≤ varR)
    (hdmin : 0 < dmin) (hden : dmin ≤ |toReal (Float.sqrt var + eps)|)
    (hyR : Real.sqrt varR + toReal eps ≠ 0) :
    |toReal (advNormF a mean var eps) - Puffer.RL.Normalize.advNorm (toReal a) meanR varR (toReal eps)|
      ≤ u64 * |toReal (a - mean) / toReal (Float.sqrt var + eps)|
        + ((u64 * |toReal a - toReal mean| + εmean)
            + |(toReal a - meanR) / (Real.sqrt varR + toReal eps)|
              * (u64 * |toReal (Float.sqrt var) + toReal eps|
                  + (u64 * Real.sqrt (toReal var) + Real.sqrt εvar))) / dmin := by
  have hnum : |toReal (a - mean) - (toReal a - meanR)| ≤ u64 * |toReal a - toReal mean| + εmean := by
    simpa using subApprox_error a mean (toReal a) meanR 0 εmean (by simp) hmean
  have hsq : |toReal (Float.sqrt var) - Real.sqrt varR| ≤ u64 * Real.sqrt (toReal var) + Real.sqrt εvar :=
    sqrtApprox_error var varR εvar hvar hvnn hvRnn
  have hden' : |toReal (Float.sqrt var + eps) - (Real.sqrt varR + toReal eps)|
      ≤ u64 * |toReal (Float.sqrt var) + toReal eps| + (u64 * Real.sqrt (toReal var) + Real.sqrt εvar) := by
    simpa using addApprox_error (Float.sqrt var) eps (Real.sqrt varR) (toReal eps)
      (u64 * Real.sqrt (toReal var) + Real.sqrt εvar) 0 hsq (by simp)
  have hdiv := divApprox_error (a - mean) (Float.sqrt var + eps) (toReal a - meanR)
    (Real.sqrt varR + toReal eps) (u64 * |toReal a - toReal mean| + εmean)
    (u64 * |toReal (Float.sqrt var) + toReal eps| + (u64 * Real.sqrt (toReal var) + Real.sqrt εvar))
    dmin hnum hden' hdmin hden hyR
  simpa [advNormF, Puffer.RL.Normalize.advNorm] using hdiv

/-! ### The runnable mean/variance reuse LayerNorm's machinery -/

/-- The variance `normalizeAdv` computes (direct square-fold around `lMean`). -/
def advVar (adv : Array Float) : Float :=
  adv.foldl (fun s x => s + (x - lMean adv) * (x - lMean adv)) 0.0 / Float.ofNat adv.size

/-- `normalizeAdv`'s direct square-fold IS `LayerNormExec.lVar` (map-fold, via `Array.foldl_map`). -/
theorem advVar_eq_lVar (adv : Array Float) : advVar adv = lVar adv := by
  rw [advVar, lVar, Array.foldl_map]

/-- **The runnable advantage variance is nonnegative.** The Float value the trainer actually computes for
    `advVar adv = (Σᵢ (advᵢ − μ)²)/|adv|` embeds to a nonnegative real, so `0 ≤ toReal (advVar adv)` — the
    honest Float-level witness of the `hvnn : 0 ≤ toReal (advVar adv)` side condition otherwise carried as a
    HYPOTHESIS throughout the advantage-normalization error chain (`advNorm_error`, `normalizeAdv_error`,
    `fullNormalizeAdv_error`). Despite two roundings per accumulation step (the centered square `mul_model`,
    then the running-sum `add_model`) and the final `div_model`, the invariant survives: each centered square
    rounds to `≥ 0` (its `(1+δ)` factor stays `≥ 1 − u64 > 0`), the running sum of nonnegatives stays `≥ 0`
    (`listFoldAdd_lb`), and dividing by the nonnegative count keeps it `≥ 0`. The hypothesis `hN` is
    load-bearing: `toReal` is deliberately unconstrained on `Float.ofNat` (there is no `toReal_ofNat` axiom),
    so a model with `toReal (Float.ofNat |adv|) < 0` and a non-constant minibatch (positive sum of squared
    deviations) would make the divided variance negative — this matches how the codebase carries count facts
    about `Float.ofNat` as explicit hypotheses rather than assuming them. -/
theorem advVar_nonneg (adv : Array Float) (hN : 0 ≤ toReal (Float.ofNat adv.size)) :
    0 ≤ toReal (advVar adv) := by
  rw [advVar]
  set mean := lMean adv with hmean
  -- each centered square rounds to a nonnegative real
  have hg : ∀ x ∈ adv.toList, 0 ≤ toReal ((x - mean) * (x - mean)) := by
    intro x _
    obtain ⟨δ, hδ, heq⟩ := mul_model (x - mean) (x - mean)
    rw [heq, abs_le] at *
    exact mul_nonneg (mul_self_nonneg _) (by linarith [hδ.1, u64_lt_one])
  -- the running fold of nonnegatives stays nonnegative (via listFoldAdd_lb)
  have hfold : 0 ≤ toReal (adv.foldl (fun s x => s + (x - mean) * (x - mean)) 0.0) := by
    rw [← Array.foldl_toList]
    have hlb := listFoldAdd_lb (fun x => (x - mean) * (x - mean)) adv.toList 0.0
      (by rw [toReal_zeroLit]) hg
    refine le_trans ?_ hlb
    refine mul_nonneg (pow_nonneg (by linarith [u64_lt_one]) _) ?_
    rw [toReal_zeroLit, zero_add]
    apply List.sum_nonneg
    intro y hy
    rw [List.mem_map] at hy
    obtain ⟨z, hz, rfl⟩ := hy
    exact hg z hz
  -- dividing the nonnegative sum by the nonnegative count keeps it nonnegative
  obtain ⟨δ, hδ, heq⟩ := div_model (adv.foldl (fun s x => s + (x - mean) * (x - mean)) 0.0)
    (Float.ofNat adv.size)
  rw [heq, abs_le] at *
  exact mul_nonneg (div_nonneg hfold hN) (by linarith [hδ.1, u64_lt_one])

/-- **Runnable mean error** to the ℝ reference `lnMuR` (reuses `lnMean_error`; `c = |adv|` exact via `hc`). -/
theorem advMean_error (adv : Array Float) (cR : ℝ)
    (hc : toReal (Float.ofNat adv.size) = cR) (hcpos : 0 < cR) :
    |toReal (lMean adv) - lnMuR (fun k => adv[k]!) adv.size cR|
      ≤ lnMeanErr (fun k => adv[k]!) (Float.ofNat adv.size) adv.size cR := by
  rw [lMean_eq]
  exact lnMean_error (fun k => adv[k]!) (Float.ofNat adv.size) adv.size cR hc hcpos

/-- **Runnable variance error** to the ℝ reference `lnVarR` (reuses `lnVar_error`; the runnable mean threads
    into the model mean via `lMean_eq`). -/
theorem advVar_error (adv : Array Float) (cR : ℝ)
    (hc : toReal (Float.ofNat adv.size) = cR) (hcpos : 0 < cR) :
    |toReal (advVar adv) - lnVarR (fun k => adv[k]!) adv.size cR|
      ≤ lnVarErr (fun k => adv[k]!) (Float.ofNat adv.size) adv.size cR := by
  rw [advVar_eq_lVar, lVar_eq, lMean_eq]
  exact lnVar_error (fun k => adv[k]!)
    (lnMeanF (fun k => adv[k]!) (Float.ofNat adv.size) adv.size) (Float.ofNat adv.size) adv.size
    (lnMuR (fun k => adv[k]!) adv.size cR) cR (lnMeanErr (fun k => adv[k]!) (Float.ofNat adv.size) adv.size cR)
    (lnMean_error (fun k => adv[k]!) (Float.ofNat adv.size) adv.size cR hc hcpos) hc hcpos

/-! ### The exec connection: the runnable `normalizeAdv` output -/

/-- The runnable normalized advantage `(normalizeAdv adv)[i]! = (adv[i]! − mean)/(√var + 1e-8)`. -/
theorem normalizeAdv_getElem! (adv : Array Float) (i : Nat) (hi : i < adv.size) :
    (normalizeAdv adv)[i]! = advNormF adv[i]! (lMean adv) (advVar adv) (1e-8) := by
  show (adv.map (fun x => (x - lMean adv) / (Float.sqrt (advVar adv) + 1e-8)))[i]! = _
  rw [arrMapGetElem! adv _ i hi]; rfl

/-- **Runnable advantage-normalization error.** The trainer's actual `(normalizeAdv adv)[i]!` output is within
    the composed bound of `Normalize.advNorm`, given the mean/var error budgets (discharge via
    `advMean_error`/`advVar_error`). Caps the advantage-normalization trifecta on the code that runs. -/
theorem normalizeAdv_error (adv : Array Float) (i : Nat) (meanR varR εmean εvar dmin : ℝ)
    (hi : i < adv.size)
    (hmean : |toReal (lMean adv) - meanR| ≤ εmean) (hvar : |toReal (advVar adv) - varR| ≤ εvar)
    (hvnn : 0 ≤ toReal (advVar adv)) (hvRnn : 0 ≤ varR)
    (hdmin : 0 < dmin) (hden : dmin ≤ |toReal (Float.sqrt (advVar adv) + 1e-8)|)
    (hyR : Real.sqrt varR + toReal (1e-8 : Float) ≠ 0) :
    |toReal ((normalizeAdv adv)[i]!)
        - Puffer.RL.Normalize.advNorm (toReal adv[i]!) meanR varR (toReal (1e-8 : Float))|
      ≤ u64 * |toReal (adv[i]! - lMean adv) / toReal (Float.sqrt (advVar adv) + 1e-8)|
        + ((u64 * |toReal adv[i]! - toReal (lMean adv)| + εmean)
            + |(toReal adv[i]! - meanR) / (Real.sqrt varR + toReal (1e-8 : Float))|
              * (u64 * |toReal (Float.sqrt (advVar adv)) + toReal (1e-8 : Float)|
                  + (u64 * Real.sqrt (toReal (advVar adv)) + Real.sqrt εvar))) / dmin := by
  rw [normalizeAdv_getElem! adv i hi]
  exact advNorm_error adv[i]! (lMean adv) (advVar adv) (1e-8) meanR varR εmean εvar dmin
    hmean hvar hvnn hvRnn hdmin hden hyR

/-- **Fully-chained runnable advantage-normalization error** — the a66-analog (`fullLayerNorm_error`) for
    `normalizeAdv`. Chains `advMean_error`/`advVar_error` INTO `normalizeAdv_error`, discharging the abstract
    `εmean`/`εvar`/`meanR`/`varR` budgets: the ℝ references become the concrete `lnMuR`/`lnVarR` (the exact-ℝ
    mean/variance of the advantage entries) and the error budgets the concrete `lnMeanErr`/`lnVarErr`. The
    only remaining inputs are the count exactness (`hc : toReal ⌊|adv|⌋ = cR`, `hcpos`), the variance
    nonnegativity (Float `hvnn` + ℝ `hvRnn`), and the denominator floor (`hdmin`/`hden`/`hyR` for `√var + ε`).
    Caps the advantage-normalization trifecta with NO abstract error hypotheses. -/
theorem fullNormalizeAdv_error (adv : Array Float) (i : Nat) (cR dmin : ℝ) (hi : i < adv.size)
    (hc : toReal (Float.ofNat adv.size) = cR) (hcpos : 0 < cR)
    (hvnn : 0 ≤ toReal (advVar adv))
    (hvRnn : 0 ≤ lnVarR (fun k => adv[k]!) adv.size cR)
    (hdmin : 0 < dmin) (hden : dmin ≤ |toReal (Float.sqrt (advVar adv) + 1e-8)|)
    (hyR : Real.sqrt (lnVarR (fun k => adv[k]!) adv.size cR) + toReal (1e-8 : Float) ≠ 0) :
    |toReal ((normalizeAdv adv)[i]!)
        - Puffer.RL.Normalize.advNorm (toReal adv[i]!)
            (lnMuR (fun k => adv[k]!) adv.size cR) (lnVarR (fun k => adv[k]!) adv.size cR)
            (toReal (1e-8 : Float))|
      ≤ u64 * |toReal (adv[i]! - lMean adv) / toReal (Float.sqrt (advVar adv) + 1e-8)|
        + ((u64 * |toReal adv[i]! - toReal (lMean adv)|
              + lnMeanErr (fun k => adv[k]!) (Float.ofNat adv.size) adv.size cR)
            + |(toReal adv[i]! - lnMuR (fun k => adv[k]!) adv.size cR)
                / (Real.sqrt (lnVarR (fun k => adv[k]!) adv.size cR) + toReal (1e-8 : Float))|
              * (u64 * |toReal (Float.sqrt (advVar adv)) + toReal (1e-8 : Float)|
                  + (u64 * Real.sqrt (toReal (advVar adv))
                      + Real.sqrt (lnVarErr (fun k => adv[k]!) (Float.ofNat adv.size) adv.size cR)))) / dmin := by
  exact normalizeAdv_error adv i (lnMuR (fun k => adv[k]!) adv.size cR)
    (lnVarR (fun k => adv[k]!) adv.size cR)
    (lnMeanErr (fun k => adv[k]!) (Float.ofNat adv.size) adv.size cR)
    (lnVarErr (fun k => adv[k]!) (Float.ofNat adv.size) adv.size cR) dmin hi
    (advMean_error adv cR hc hcpos) (advVar_error adv cR hc hcpos) hvnn hvRnn hdmin hden hyR

end Puffer.RL.AdvNormBound
