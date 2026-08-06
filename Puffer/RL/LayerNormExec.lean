/-
Connecting the RUNNABLE LayerNorm to its bounded functional model.

`LayerNormBound.fullLayerNormF` is the faithful functional model of LayerNorm, and `fullLayerNorm_error`
chains its four stages (mean → variance → denominator → affine) into ONE bound against the ℝ reference
`layerNormRval`. This file adds a genuinely RUNNABLE `Array Float` LayerNorm — `Puffer.RL.Train.layerNorm`,
an `Id.run do` block folding/mapping over honest arrays (`c = |x|`, `μ = foldl(+)/c`,
`σ² = foldl(+ of (xᵢ−μ)²)/c`, `denom = √(σ²+ε)`, output `map (·−μ)/denom·wᵢ+bᵢ`) — and proves the exec's
`i`-th output *equals* the model, transporting the whole error bound to the code that runs.

Pure structural plumbing, reusing the softmax-exec bridge (`SoftmaxExec.arrFoldl_add_eq_sumIdxFGo`,
`sumIdxFGo_congr`, `arrMapGetElem!`):

  • `lMean`/`lVar`/`lDenom` — the exact `Float` mean/variance/denominator the runnable computes.
  • `lMean_eq`/`lVar_eq`/`lDenom_eq` — each equals the model's `lnMeanF`/`lnVarF`/`lnDenomF` (both nested
    reductions are `Array.foldl (·+·) 0` = `sumIdxFGo` over `List.range`; the variance's centered-square
    summands match via `sumIdxFGo_congr`, and the mean threads in by `lMean_eq`).
  • `train_layerNorm_getElem!` — `(Train.layerNorm x w b eps)[i]! = fullLayerNormF (x[·]!) (w[·]!) (b[·]!)
    |x| eps |x| i` for every in-range `i` (`Id.run`/`Array.range`-map unfolding + `rangeMapGetElem!`).
  • `train_layerNorm_error` — the fully-chained Float↔ℝ bound, now on the ACTUAL runnable output, against
    `layerNormRval`. Same hypotheses as `fullLayerNorm_error`: the count is exact (`toReal (Float.ofNat |x|)
    = cR > 0`), `epsR = toReal eps`, radicands nonneg, denominator floor.

`train_layerNorm_error` lands against `layerNormRval` (count `cR`, epsilon `toReal eps`); the FINAL section
closes both gaps to land directly against the canonical `Net.layerNorm` (with `card` and the real `1e-5`):

  • `lnMuR_eq_net`/`lnVarR_eq_net` — with `cR = |x|` the ℝ-reference mean/variance ARE `Net.lnMean`/`lnVar`
    (`Finset.card_range`, `a*a = a²`).
  • `eps_gap` — the ONE genuinely-folded discrepancy: fixing `eps := (1e-5 : Float)`, the Float literal's
    `toReal` differs from the real `1e-5` by `≤ u64·1e-5` (`toReal_ofScientific_close`), which propagates
    through the `√` as `≤ √(u64·1e-5)` (the Hölder-½ bound `abs_sqrt_sub_sqrt_le`). This term is ADDED to the
    denominator error — the `1e-5` gap becomes an explicit, bounded contribution, not a parameter mismatch.
  • `lMean_error_net`/`lDenom_error_net` — the Float mean/denom errors to the CANONICAL `Net.lnMean`/`lnDenom`
    (the denom's being `lnDenomErr + √(u64·1e-5)`).
  • `train_layerNorm_error_net` — `|toReal ((Train.layerNorm x w b 1e-5)[i]!) − Net.layerNorm (range |x|)
    (x[·]!) (w[·]!) (b[·]!) i| ≤ netErrBnd` via `layerNorm_error_spec`. The only remaining hypothesis beyond
    the denom floor is `hcard : toReal (Float.ofNat |x|) = |x|` — genuine count-exactness (integers ≤ 2⁵³ are
    exactly representable, but no `toReal_ofNat` axiom exists, so it is carried honestly, not assumed away).

Axiom-clean beyond the trusted Float (1+δ) base — no `native_decide`, no `sorry`.
-/
import Puffer.RL.LayerNormBound
import Puffer.RL.SoftmaxExec
import Puffer.RL.Train

namespace Puffer.RL.LayerNormExec

open Puffer.FloatR
open Puffer.RL.LayerNormBound
open Puffer.RL.SoftmaxExec (arrFoldl_add_eq_sumIdxFGo sumIdxFGo_congr arrMapGetElem!)
open Puffer.RL.SoftmaxBound (sumIdxFGo)

/-! ### `getElem!` through a `range`-map, and the runnable's exact mean/variance/denominator -/

/-- `getElem!` through `(Array.range m).map g` (in range). -/
theorem rangeMapGetElem! {β} [Inhabited β] (m : Nat) (g : Nat → β) (i : Nat) (hi : i < m) :
    ((Array.range m).map g)[i]! = g i := by
  rw [Array.getElem!_eq_getD, Array.getD]; simp [hi, Array.getElem_map, Array.getElem_range]

/-- The exact `Float` mean `μ = (Σ xⱼ)/|x|` computed by `Train.layerNorm`. -/
def lMean (x : Array Float) : Float := x.foldl (· + ·) 0.0 / Float.ofNat x.size

/-- The exact `Float` variance `σ² = (Σ (xⱼ − μ)²)/|x|` computed by `Train.layerNorm`. -/
def lVar (x : Array Float) : Float :=
  (x.map (fun xi => (xi - lMean x) * (xi - lMean x))).foldl (· + ·) 0.0 / Float.ofNat x.size

/-- The exact `Float` normalizing denominator `√(σ² + ε)` computed by `Train.layerNorm`. -/
def lDenom (x : Array Float) (eps : Float) : Float := Float.sqrt (lVar x + eps)

/-- The runnable mean IS the model's `lnMeanF` (its `Array.foldl (·+·) 0` is `sumIdxFGo` over the range). -/
theorem lMean_eq (x : Array Float) :
    lMean x = lnMeanF (fun k => x[k]!) (Float.ofNat x.size) x.size := by
  rw [lMean, lnMeanF, arrFoldl_add_eq_sumIdxFGo]

/-- The runnable variance IS the model's `lnVarF` on the runnable mean (the centered-square array fold
    equals the `sumIdxFGo` sum, summand-by-summand via `sumIdxFGo_congr`). -/
theorem lVar_eq (x : Array Float) :
    lVar x = lnVarF (fun k => x[k]!) (lMean x) (Float.ofNat x.size) x.size := by
  have hnum : (x.map (fun xi => (xi - lMean x) * (xi - lMean x))).foldl (· + ·) 0.0
      = sumIdxFGo (fun k => (x[k]! - lMean x) * (x[k]! - lMean x)) 0.0 (List.range x.size) := by
    rw [arrFoldl_add_eq_sumIdxFGo, Array.size_map]
    apply sumIdxFGo_congr
    intro k hk; simp only [List.mem_range] at hk
    exact arrMapGetElem! x (fun xi => (xi - lMean x) * (xi - lMean x)) k hk
  rw [lVar, lnVarF, hnum]

/-- The runnable denominator IS the model's `lnDenomF` on the model's variance (mean threaded in). -/
theorem lDenom_eq (x : Array Float) (eps : Float) :
    lDenom x eps = lnDenomF (lnVarF (fun k => x[k]!)
        (lnMeanF (fun k => x[k]!) (Float.ofNat x.size) x.size) (Float.ofNat x.size) x.size) eps := by
  rw [lDenom, lnDenomF, lVar_eq, lMean_eq]

/-- **The runnable LayerNorm denominator is strictly positive (in ℝ).** Whenever the Float radicand `σ² + ε` has
    positive real value, the trainer's actual normalizing denominator `lDenom x eps = Float.sqrt (σ² + ε)`
    satisfies `0 < toReal (lDenom x eps)`. This is the honest Float-level witness of the `denom ≠ 0` /
    denominator-floor side conditions carried as hypotheses throughout the error chain: the rounded square root of
    a positive radicand cannot round to zero, because the trusted `sqrt_model` factor `(1+δ)` stays `≥ 1 − u64 > 0`.
    The hypothesis is load-bearing — with a zero radicand the sqrt is `0`, so positivity genuinely requires it. The
    LayerNorm-runtime counterpart of `SoftmaxExec.sDenom_pos`. -/
theorem lDenom_toReal_pos (x : Array Float) (eps : Float)
    (h : 0 < toReal (lVar x + eps)) : 0 < toReal (lDenom x eps) := by
  rw [lDenom]
  obtain ⟨δ, hδ, hmodel⟩ := sqrt_model (lVar x + eps)
  rw [hmodel]
  have hsqrt : 0 < Real.sqrt (toReal (lVar x + eps)) := Real.sqrt_pos.mpr h
  have hδ1 : 0 < 1 + δ := by
    have := (abs_le.mp hδ).1
    have := u64_lt_one
    linarith
  exact mul_pos hsqrt hδ1

/-! ### The exec ↔ functional-model identity, and the transported fully-chained error bound -/

/-- **The runnable LayerNorm equals the functional model, componentwise.** For every in-range `i`,
    `(Train.layerNorm x w b eps)[i]!` is exactly `fullLayerNormF (x[·]!) (w[·]!) (b[·]!) |x| eps |x| i`.
    Pure `Id.run`/`Array.range`-map unfolding — the runnable and the bounded model compute the same `Float`. -/
theorem train_layerNorm_getElem! (x w b : Array Float) (eps : Float) (i : Nat) (hi : i < x.size) :
    (Puffer.RL.Train.layerNorm x w b eps)[i]!
      = fullLayerNormF (fun k => x[k]!) (fun k => w[k]!) (fun k => b[k]!)
          (Float.ofNat x.size) eps x.size i := by
  have hunfold : Puffer.RL.Train.layerNorm x w b eps
      = (Array.range x.size).map (fun i => (x[i]! - lMean x) / lDenom x eps * w[i]! + b[i]!) := rfl
  rw [hunfold, rangeMapGetElem! x.size _ i hi, fullLayerNormF, layerNormF, lMean_eq, lDenom_eq]

/-- **Runnable LayerNorm Float↔ℝ accuracy.** The trainer's actual `Train.layerNorm` output `[i]!` is within
    the fully-chained `mean/var/denom/affine` bound of the ℝ reference `layerNormRval`. The whole
    `LayerNormBound` error analysis (`fullLayerNorm_error`), transported to the code that runs. Hypotheses,
    verbatim from `fullLayerNorm_error`: the count is exact (`toReal (Float.ofNat |x|) = cR > 0`), the epsilon
    matches (`epsR = toReal eps`), the radicands are nonneg, and the denominator floor `dmin ≤ |toReal denom|`
    (`denom ≥ √ε > 0`). -/
theorem train_layerNorm_error (x w b : Array Float) (eps : Float) (i : Nat) (hi : i < x.size)
    (cR epsR dmin : ℝ)
    (hc : toReal (Float.ofNat x.size) = cR) (hcpos : 0 < cR) (heps : toReal eps = epsR)
    (hnn : 0 ≤ toReal (lnVarF (fun k => x[k]!) (lnMeanF (fun k => x[k]!) (Float.ofNat x.size) x.size)
                  (Float.ofNat x.size) x.size + eps))
    (hRnn : 0 ≤ lnVarR (fun k => x[k]!) x.size cR + epsR)
    (hdmin : 0 < dmin)
    (hdd : dmin ≤ |toReal (lnDenomF (lnVarF (fun k => x[k]!)
              (lnMeanF (fun k => x[k]!) (Float.ofNat x.size) x.size) (Float.ofNat x.size) x.size) eps)|)
    (hyR : lnDenomR (fun k => x[k]!) x.size cR epsR ≠ 0) :
    |toReal ((Puffer.RL.Train.layerNorm x w b eps)[i]!)
        - layerNormRval (fun k => x[k]!) (fun k => w[k]!) (fun k => b[k]!) x.size i cR epsR|
      ≤ fullLayerNormErrBnd (fun k => x[k]!) (fun k => w[k]!) (fun k => b[k]!)
          (Float.ofNat x.size) eps x.size i cR epsR dmin := by
  rw [train_layerNorm_getElem! x w b eps i hi]
  exact fullLayerNorm_error (fun k => x[k]!) (fun k => w[k]!) (fun k => b[k]!)
    (Float.ofNat x.size) eps x.size i cR epsR dmin hc hcpos heps hnn hRnn hdmin hdd hyR

/-! ### Closing the `card`/`1e-5` gaps: landing directly on the canonical `Net.layerNorm` -/

/-- With `cR = |x|` the ℝ-reference mean IS `Net.lnMean`. -/
theorem lnMuR_eq_net (x : Array Float) (n : Nat) :
    lnMuR (fun k => x[k]!) n (n : ℝ) = Puffer.Net.lnMean (Finset.range n) (fun j => toReal x[j]!) := by
  rw [lnMuR, Puffer.Net.lnMean, Finset.card_range]

/-- With `cR = |x|` the ℝ-reference variance IS `Net.lnVar` (`a*a = a²`, matched mean/count). -/
theorem lnVarR_eq_net (x : Array Float) (n : Nat) :
    lnVarR (fun k => x[k]!) n (n : ℝ) = Puffer.Net.lnVar (Finset.range n) (fun j => toReal x[j]!) := by
  rw [lnVarR, Puffer.Net.lnVar, Finset.card_range]
  congr 1
  apply Finset.sum_congr rfl
  intro j _
  rw [lnMuR, Puffer.Net.lnMean, Finset.card_range]; ring

/-- The Float `1e-5` literal's `toReal` is nonnegative (`≥ 1e-5·(1 − u64) > 0`). -/
theorem eps_nn : 0 ≤ toReal (1e-5 : Float) := by
  have h := toReal_ofScientific_close 1 true 5
  have he : (OfScientific.ofScientific 1 true 5 : Float) = (1e-5 : Float) := rfl
  have heR : (OfScientific.ofScientific 1 true 5 : ℝ) = (1e-5 : ℝ) := rfl
  rw [he, heR] at h
  have hu : u64 < 1 := u64_lt_one
  have h1 : (0:ℝ) < 1e-5 := by norm_num
  rw [abs_of_pos h1] at h
  have := abs_le.mp h
  nlinarith [this.1, this.2]

/-- **The `1e-5` gap, folded through the `√`.** `√(V + toReal(1e-5:Float))` differs from the canonical
    `√(V + (1e-5:ℝ))` by at most `√(u64·|1e-5|)` — the literal-rounding gap `toReal_ofScientific_close`
    carried through the Hölder-½ bound `abs_sqrt_sub_sqrt_le`. -/
theorem eps_gap (V : ℝ) (hV : 0 ≤ V) :
    |Real.sqrt (V + toReal (1e-5 : Float)) - Real.sqrt (V + (1e-5 : ℝ))|
      ≤ Real.sqrt (u64 * |(1e-5 : ℝ)|) := by
  have hclose : |toReal (1e-5 : Float) - (1e-5 : ℝ)| ≤ u64 * |(1e-5 : ℝ)| := by
    have h := toReal_ofScientific_close 1 true 5
    simpa using h
  have ha : 0 ≤ V + toReal (1e-5 : Float) := by have := eps_nn; linarith
  have hb : 0 ≤ V + (1e-5 : ℝ) := by
    have h5 : (0:ℝ) < 1e-5 := by norm_num
    linarith
  calc |Real.sqrt (V + toReal (1e-5 : Float)) - Real.sqrt (V + (1e-5 : ℝ))|
      ≤ Real.sqrt |(V + toReal (1e-5 : Float)) - (V + (1e-5 : ℝ))| :=
        abs_sqrt_sub_sqrt_le ha hb
    _ = Real.sqrt |toReal (1e-5 : Float) - (1e-5 : ℝ)| := by ring_nf
    _ ≤ Real.sqrt (u64 * |(1e-5 : ℝ)|) := Real.sqrt_le_sqrt hclose

/-- **Float mean error to the canonical `Net.lnMean`** (count exact via `hcard`). -/
theorem lMean_error_net (x : Array Float)
    (hcard : toReal (Float.ofNat x.size) = (x.size : ℝ)) (hpos : 0 < x.size) :
    |toReal (lMean x) - Puffer.Net.lnMean (Finset.range x.size) (fun j => toReal x[j]!)|
      ≤ lnMeanErr (fun k => x[k]!) (Float.ofNat x.size) x.size (x.size : ℝ) := by
  rw [lMean_eq, ← lnMuR_eq_net]
  exact lnMean_error (fun k => x[k]!) (Float.ofNat x.size) x.size (x.size : ℝ) hcard
    (by exact_mod_cast hpos)

/-- **Float denominator error to the canonical `Net.lnDenom`** (real `1e-5`): the chained `lnDenomErr` PLUS
    the folded literal gap `√(u64·|1e-5|)` (`eps_gap`). -/
theorem lDenom_error_net (x : Array Float)
    (hcard : toReal (Float.ofNat x.size) = (x.size : ℝ)) (hpos : 0 < x.size)
    (hnn : 0 ≤ toReal (lnVarF (fun k => x[k]!)
              (lnMeanF (fun k => x[k]!) (Float.ofNat x.size) x.size) (Float.ofNat x.size) x.size
              + (1e-5 : Float))) :
    |toReal (lDenom x (1e-5 : Float))
        - Puffer.Net.lnDenom (Finset.range x.size) (fun j => toReal x[j]!)|
      ≤ lnDenomErr (fun k => x[k]!) (Float.ofNat x.size) (1e-5 : Float) x.size (x.size : ℝ)
        + Real.sqrt (u64 * |(1e-5 : ℝ)|) := by
  have hposR : (0:ℝ) < (x.size : ℝ) := by exact_mod_cast hpos
  have hμ' : |toReal (lnMeanF (fun k => x[k]!) (Float.ofNat x.size) x.size)
        - lnMuR (fun k => x[k]!) x.size (x.size : ℝ)|
      ≤ lnMeanErr (fun k => x[k]!) (Float.ofNat x.size) x.size (x.size : ℝ) :=
    lnMean_error (fun k => x[k]!) (Float.ofNat x.size) x.size (x.size : ℝ) hcard hposR
  have hvar : |toReal (lnVarF (fun k => x[k]!)
          (lnMeanF (fun k => x[k]!) (Float.ofNat x.size) x.size) (Float.ofNat x.size) x.size)
        - lnVarR (fun k => x[k]!) x.size (x.size : ℝ)|
      ≤ lnVarErr (fun k => x[k]!) (Float.ofNat x.size) x.size (x.size : ℝ) :=
    lnVar_error (fun k => x[k]!) (lnMeanF (fun k => x[k]!) (Float.ofNat x.size) x.size)
      (Float.ofNat x.size) x.size (lnMuR (fun k => x[k]!) x.size (x.size : ℝ)) (x.size : ℝ)
      (lnMeanErr (fun k => x[k]!) (Float.ofNat x.size) x.size (x.size : ℝ)) hμ' hcard hposR
  have hRnn : 0 ≤ lnVarR (fun k => x[k]!) x.size (x.size : ℝ) + toReal (1e-5 : Float) := by
    rw [lnVarR_eq_net]
    have h1 := Puffer.Net.lnVar_nonneg (Finset.range x.size) (fun j => toReal x[j]!)
    have h2 := eps_nn
    linarith
  have hden0 : |toReal (lnDenomF (lnVarF (fun k => x[k]!)
            (lnMeanF (fun k => x[k]!) (Float.ofNat x.size) x.size) (Float.ofNat x.size) x.size)
            (1e-5 : Float))
        - Real.sqrt (lnVarR (fun k => x[k]!) x.size (x.size : ℝ) + toReal (1e-5 : Float))|
      ≤ lnDenomErr (fun k => x[k]!) (Float.ofNat x.size) (1e-5 : Float) x.size (x.size : ℝ) :=
    lnDenom_error (lnVarF (fun k => x[k]!)
        (lnMeanF (fun k => x[k]!) (Float.ofNat x.size) x.size) (Float.ofNat x.size) x.size)
      (1e-5 : Float) (lnVarR (fun k => x[k]!) x.size (x.size : ℝ)) (toReal (1e-5 : Float))
      (lnVarErr (fun k => x[k]!) (Float.ofNat x.size) x.size (x.size : ℝ)) hvar rfl hnn hRnn
  rw [lDenom_eq]
  refine le_trans (abs_sub_le _
      (Real.sqrt (lnVarR (fun k => x[k]!) x.size (x.size : ℝ) + toReal (1e-5 : Float))) _)
    (add_le_add hden0 ?_)
  rw [lnVarR_eq_net, Puffer.Net.lnDenom]
  exact eps_gap (Puffer.Net.lnVar (Finset.range x.size) (fun j => toReal x[j]!))
    (Puffer.Net.lnVar_nonneg _ _)

/-- The fully-chained error bound on the runnable output against the canonical `Net.layerNorm`
    (`layerNorm_error_spec`'s RHS with the concrete `Net`-referenced mean/denom errors). -/
noncomputable def netErrBnd (x w b : Array Float) (i : Nat) (dmin : ℝ) : ℝ :=
  u64 * |toReal ((x[i]! - lMean x) / lDenom x (1e-5 : Float) * w[i]!) + toReal (b[i]!)|
    + (u64 * (|toReal ((x[i]! - lMean x) / lDenom x (1e-5 : Float))| * |toReal (w[i]!)|)
        + |toReal (w[i]!)|
          * (u64 * |toReal (x[i]! - lMean x) / toReal (lDenom x (1e-5 : Float))|
              + (u64 * |toReal (x[i]!) - toReal (lMean x)|
                  + lnMeanErr (fun k => x[k]!) (Float.ofNat x.size) x.size (x.size : ℝ)
                  + |(toReal (x[i]!) - Puffer.Net.lnMean (Finset.range x.size) (fun j => toReal x[j]!))
                      / Puffer.Net.lnDenom (Finset.range x.size) (fun j => toReal x[j]!)|
                    * (lnDenomErr (fun k => x[k]!) (Float.ofNat x.size) (1e-5 : Float) x.size (x.size : ℝ)
                        + Real.sqrt (u64 * |(1e-5 : ℝ)|)))
                / dmin))

/-- **Runnable LayerNorm Float↔ℝ accuracy, against the CANONICAL `Net.layerNorm`.** The trainer's actual
    `Train.layerNorm x w b 1e-5` output `[i]!` is within `netErrBnd` of `Puffer.Net.layerNorm` — the library's
    official ℝ LayerNorm with `s.card` and the real `1e-5`. Both a66/a68 gaps closed: the count via
    `hcard : toReal (Float.ofNat |x|) = |x|` (genuine exact-integer-representability, carried honestly as no
    `toReal_ofNat` axiom exists), and the `1e-5` literal folded into the denom error as `√(u64·|1e-5|)`. The
    only other hypotheses are the Float-radicand nonneg `hnn` and the denominator floor `dmin ≤ |toReal denom|`
    (`Net.lnDenom ≠ 0` is FREE, from `Net.lnDenom_pos`). -/
theorem train_layerNorm_error_net (x w b : Array Float) (i : Nat) (hi : i < x.size)
    (hcard : toReal (Float.ofNat x.size) = (x.size : ℝ)) (hpos : 0 < x.size)
    (dmin : ℝ) (hdmin : 0 < dmin) (hdd : dmin ≤ |toReal (lDenom x (1e-5 : Float))|)
    (hnn : 0 ≤ toReal (lnVarF (fun k => x[k]!)
              (lnMeanF (fun k => x[k]!) (Float.ofNat x.size) x.size) (Float.ofNat x.size) x.size
              + (1e-5 : Float))) :
    |toReal ((Puffer.RL.Train.layerNorm x w b (1e-5 : Float))[i]!)
        - Puffer.Net.layerNorm (Finset.range x.size) (fun j => toReal x[j]!)
            (fun j => toReal w[j]!) (fun j => toReal b[j]!) i|
      ≤ netErrBnd x w b i dmin := by
  have hlhs : (Puffer.RL.Train.layerNorm x w b (1e-5 : Float))[i]!
      = layerNormF (fun k => x[k]!) (lMean x) (lDenom x (1e-5 : Float))
          ((fun k => w[k]!) i) ((fun k => b[k]!) i) i := by
    rw [train_layerNorm_getElem! x w b (1e-5 : Float) i hi, fullLayerNormF, ← lDenom_eq, ← lMean_eq]
  rw [hlhs, netErrBnd]
  exact layerNorm_error_spec (fun k => x[k]!) (fun k => w[k]!) (fun k => b[k]!)
    (lMean x) (lDenom x (1e-5 : Float)) x.size i _ _ dmin
    (lMean_error_net x hcard hpos) (lDenom_error_net x hcard hpos hnn) hdmin hdd
    (ne_of_gt (Puffer.Net.lnDenom_pos _ _))

end Puffer.RL.LayerNormExec
