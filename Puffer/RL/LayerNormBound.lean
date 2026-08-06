/-
LayerNorm — `((xᵢ − μ)/√(σ² + ε))·wᵢ + bᵢ` — with a proven Float↔ℝ accuracy bound.

The last forward-pass reduction kernel without a bound (softmax was the other). LayerNorm nests TWO reductions
— the mean `μ = (Σ xⱼ)/c` and the variance `σ² = (Σ (xⱼ − μ)²)/c` (the variance summands themselves carry
the mean's error) — then a `√`, a division, and the affine `·w + b`. Bounded op-by-op, reusing the
approximate-summand sum core `SoftmaxBound.sumIdxFGo_error` for both reductions:

  • `lnMean_error` — `(Σ xⱼ)/c`: the sum (exact summands) then `divApprox_error` by the count `c`.
  • `lnVar_error`  — `(Σ (xⱼ − μ)²)/c`: each summand `(xⱼ − μ)²` carries the MEAN error `εμ` (via `sub`+`mul`),
    summed, then divided by `c`. The nested reduction.
  • `lnDenom_error`— `√(σ² + ε)`: `addApprox_error` then `sqrtApprox_error`.
  • `layerNorm_error` — `(xᵢ − μ)/denom · wᵢ + bᵢ`: `sub`, `div` (denom floored by `√ε`), `mul`, `add`; lands
    against `Puffer.Net.layerNorm`.

The count `c` is a `Float` with `toReal c = card` (exact for `card ≤ 2⁵³`), supplied as a hypothesis; each
division's denominator floor is likewise a hypothesis (`c > 0`; `denom ≥ √ε > 0`). The `+1e-5` epsilon is a
PARAMETER (`toReal eps = epsR`), so the bound is against `√(σ²R + epsR)`; instantiating `Net.lnDenom`'s literal
`1e-5` folds the tiny `toReal_ofScientific_close` gap `|toReal(1e-5:Float) − 1e-5|` into the denom error `εden`.
No executable Float LayerNorm exists in the repo yet; `lnMeanF`/`lnVarF`/`lnDenomF`/`layerNormF` are the
functional model (like `softmaxFval` for softmax). The four stages are MODULAR (chain `εμ → εσ² → εden`); a
single fully-composed end-to-end theorem with concrete `ε`s is a scoped follow-up. Axiom-clean beyond the
trusted Float (1+δ) base.
-/
import Puffer.RL.SoftmaxBound
import Puffer.Net.Forward

namespace Puffer.RL.LayerNormBound

open Puffer.FloatR
open Puffer.RL.SoftmaxBound (sumIdxFGo sumIdxRGo sumIdxErrBnd sumIdxFGo_error sumIdxRGo_eq_sum)

/-! ### The mean `μ = (Σ xⱼ)/c` -/

/-- Float batch mean `(Σ xⱼ)/c` (`c` = the feature count). -/
def lnMeanF (x : Nat → Float) (c : Float) (n : Nat) : Float := sumIdxFGo x 0.0 (List.range n) / c

/-- **Mean error.** `(Σ xⱼ)/c` is within the `sum`+`div` bound of `(Σ toReal xⱼ)/card`. The inputs `xⱼ`
    are exact (error 0); `c` exact with `toReal c = cR > 0`, the (known positive) count. -/
theorem lnMean_error (x : Nat → Float) (c : Float) (n : Nat) (cR : ℝ)
    (hc : toReal c = cR) (hcpos : 0 < cR) :
    |toReal (lnMeanF x c n) - (∑ j ∈ Finset.range n, toReal (x j)) / cR|
      ≤ u64 * |toReal (sumIdxFGo x 0.0 (List.range n)) / toReal c|
        + (sumIdxErrBnd x (fun _ => 0) 0.0 (List.range n) 0
            + |(∑ j ∈ Finset.range n, toReal (x j)) / cR| * 0) / cR := by
  have h0acc : |toReal (0.0 : Float) - (0 : ℝ)| ≤ 0 := by rw [toReal_zeroLit]; simp
  have hsum : |toReal (sumIdxFGo x 0.0 (List.range n)) - (∑ j ∈ Finset.range n, toReal (x j))|
      ≤ sumIdxErrBnd x (fun _ => 0) 0.0 (List.range n) 0 := by
    have h := sumIdxFGo_error x (fun k => toReal (x k)) (fun _ => 0) (List.range n) 0.0 0 0 h0acc
      (fun k _ => by simp)
    rwa [sumIdxRGo_eq_sum] at h
  have hcerr : |toReal c - cR| ≤ 0 := by rw [hc]; simp
  have hdmin : (0 : ℝ) < cR := hcpos
  have hden : cR ≤ |toReal c| := by rw [hc, abs_of_pos hcpos]
  have hyR : cR ≠ 0 := ne_of_gt hcpos
  simpa [lnMeanF] using
    divApprox_error (sumIdxFGo x 0.0 (List.range n)) c (∑ j ∈ Finset.range n, toReal (x j)) cR
      (sumIdxErrBnd x (fun _ => 0) 0.0 (List.range n) 0) 0 cR hsum hcerr hdmin hden hyR

/-! ### The variance `σ² = (Σ (xⱼ − μ)²)/c` (nested on the mean) -/

/-- The centered-square summand error bound (one `sub` carrying `εμ`, then one `mul` squaring it). -/
noncomputable def centeredSqBnd (x : Nat → Float) (μ : Float) (μR εμ : ℝ) (k : Nat) : ℝ :=
  u64 * |toReal (x k - μ) * toReal (x k - μ)|
    + |toReal (x k - μ)| * (u64 * |toReal (x k) - toReal μ| + εμ)
    + |toReal (x k) - μR| * (u64 * |toReal (x k) - toReal μ| + εμ)

/-- **Centered-square error.** `(xₖ − μ)²` (Float) is within `centeredSqBnd` of `(toReal xₖ − μR)²`, given
    the mean error `|toReal μ − μR| ≤ εμ`. -/
theorem centeredSq_error (x : Nat → Float) (μ : Float) (μR εμ : ℝ) (k : Nat)
    (hμ : |toReal μ - μR| ≤ εμ) :
    |toReal ((x k - μ) * (x k - μ)) - (toReal (x k) - μR) * (toReal (x k) - μR)|
      ≤ centeredSqBnd x μ μR εμ k := by
  have hd : |toReal (x k - μ) - (toReal (x k) - μR)| ≤ u64 * |toReal (x k) - toReal μ| + εμ := by
    simpa using subApprox_error (x k) μ (toReal (x k)) μR 0 εμ (by simp) hμ
  simpa [centeredSqBnd] using
    mulApprox_error (x k - μ) (x k - μ) (toReal (x k) - μR) (toReal (x k) - μR)
      (u64 * |toReal (x k) - toReal μ| + εμ) (u64 * |toReal (x k) - toReal μ| + εμ) hd hd

/-- Float variance `(Σ (xⱼ − μ)²)/c`. -/
def lnVarF (x : Nat → Float) (μ c : Float) (n : Nat) : Float :=
  sumIdxFGo (fun k => (x k - μ) * (x k - μ)) 0.0 (List.range n) / c

/-- **Variance error.** `(Σ (xⱼ − μ)²)/c` is within the composed `sum`+`div` bound of
    `(Σ (toReal xⱼ − μR)²)/cR`, given the mean error `εμ` and the (exact, positive) count `c`. -/
theorem lnVar_error (x : Nat → Float) (μ c : Float) (n : Nat) (μR cR εμ : ℝ)
    (hμ : |toReal μ - μR| ≤ εμ) (hc : toReal c = cR) (hcpos : 0 < cR) :
    |toReal (lnVarF x μ c n)
        - (∑ j ∈ Finset.range n, (toReal (x j) - μR) * (toReal (x j) - μR)) / cR|
      ≤ u64 * |toReal (sumIdxFGo (fun k => (x k - μ) * (x k - μ)) 0.0 (List.range n)) / toReal c|
        + (sumIdxErrBnd (fun k => (x k - μ) * (x k - μ)) (centeredSqBnd x μ μR εμ) 0.0 (List.range n) 0
            + |(∑ j ∈ Finset.range n, (toReal (x j) - μR) * (toReal (x j) - μR)) / cR| * 0) / cR := by
  have h0acc : |toReal (0.0 : Float) - (0 : ℝ)| ≤ 0 := by rw [toReal_zeroLit]; simp
  have hsum : |toReal (sumIdxFGo (fun k => (x k - μ) * (x k - μ)) 0.0 (List.range n))
        - (∑ j ∈ Finset.range n, (toReal (x j) - μR) * (toReal (x j) - μR))|
      ≤ sumIdxErrBnd (fun k => (x k - μ) * (x k - μ)) (centeredSqBnd x μ μR εμ) 0.0 (List.range n) 0 := by
    have h := sumIdxFGo_error (fun k => (x k - μ) * (x k - μ))
      (fun k => (toReal (x k) - μR) * (toReal (x k) - μR)) (centeredSqBnd x μ μR εμ)
      (List.range n) 0.0 0 0 h0acc (fun k _ => centeredSq_error x μ μR εμ k hμ)
    rwa [sumIdxRGo_eq_sum] at h
  have hcerr : |toReal c - cR| ≤ 0 := by rw [hc]; simp
  simpa [lnVarF] using
    divApprox_error (sumIdxFGo (fun k => (x k - μ) * (x k - μ)) 0.0 (List.range n)) c
      (∑ j ∈ Finset.range n, (toReal (x j) - μR) * (toReal (x j) - μR)) cR
      (sumIdxErrBnd (fun k => (x k - μ) * (x k - μ)) (centeredSqBnd x μ μR εμ) 0.0 (List.range n) 0) 0 cR
      hsum hcerr hcpos (by rw [hc, abs_of_pos hcpos]) (ne_of_gt hcpos)

/-- **The Float LayerNorm variance is nonnegative.** For any features `x`, centering point `μ`, and count
    `c` with `0 ≤ toReal c`, the executable Float variance `lnVarF x μ c n = (Σ (xⱼ − μ)²)/c` embeds to a
    nonnegative real: `0 ≤ toReal (lnVarF x μ c n)`. Each centered-square summand `(xⱼ − μ)·(xⱼ − μ)` is a Float
    self-product, hence `≥ 0` in ℝ (the rounding factor `1+δ` stays positive since `|δ| ≤ u64 < 1`); the running
    fold of nonnegatives from the `0.0` seed stays `≥ 0` (each `add_model` step preserves the sign); and dividing
    by the nonnegative count keeps it (`div_model` + `div_nonneg`). This is the Float-side counterpart of the ℝ
    theorem `Puffer.Net.lnVar_nonneg`, and it DISCHARGES the radicand-nonnegativity assumption `hnn` that
    `lnDenom_error`/`fullLayerNorm_error` currently take as a hypothesis (combined with `0 ≤ toReal eps` and one
    `add_model` step): the LayerNorm variance is provably a valid `√` radicand rather than an assumed one.
    `0 ≤ toReal c` is load-bearing — a negative count flips the sign
    (`lnVarF [1,2,3,4] 2.5 (−4.0) 4 = −1.25 < 0`, while `c = 4.0` gives the true population variance `1.25`). -/
theorem lnVarF_nonneg (x : Nat → Float) (μ c : Float) (n : Nat) (hc : 0 ≤ toReal c) :
    0 ≤ toReal (lnVarF x μ c n) := by
  have hself : ∀ a : Float, 0 ≤ toReal (a * a) := by
    intro a
    obtain ⟨δ, hδ, he⟩ := mul_model a a
    rw [he]
    exact mul_nonneg (mul_self_nonneg _) (by linarith [u64_lt_one, (abs_le.mp hδ).1])
  have hsum : ∀ (is : List Nat) (acc : Float), 0 ≤ toReal acc →
      0 ≤ toReal (sumIdxFGo (fun k => (x k - μ) * (x k - μ)) acc is) := by
    intro is
    induction is with
    | nil => intro acc h; simpa [sumIdxFGo] using h
    | cons i is ih =>
        intro acc h
        simp only [sumIdxFGo]
        refine ih (acc + (x i - μ) * (x i - μ)) ?_
        obtain ⟨δ, hδ, he⟩ := add_model acc ((x i - μ) * (x i - μ))
        rw [he]
        exact mul_nonneg (add_nonneg h (hself (x i - μ)))
          (by linarith [u64_lt_one, (abs_le.mp hδ).1])
  have hS : 0 ≤ toReal (sumIdxFGo (fun k => (x k - μ) * (x k - μ)) 0.0 (List.range n)) :=
    hsum (List.range n) 0.0 (le_of_eq toReal_zeroLit.symm)
  rw [lnVarF]
  obtain ⟨δ, hδ, he⟩ := div_model (sumIdxFGo (fun k => (x k - μ) * (x k - μ)) 0.0 (List.range n)) c
  rw [he]
  exact mul_nonneg (div_nonneg hS hc) (by linarith [u64_lt_one, (abs_le.mp hδ).1])

/-! ### The denominator `√(σ² + ε)` and the affine output -/

/-- Float normalizing denominator `√(σ² + ε)`. -/
def lnDenomF (var eps : Float) : Float := Float.sqrt (var + eps)

/-- **Denominator error.** `√(σ² + ε)` is within `addApprox`+`sqrtApprox` of `√(σ²R + εR)`, given the
    variance error `εvar` and (nonneg) radicands. -/
theorem lnDenom_error (var eps : Float) (varR epsR εvar : ℝ)
    (hvar : |toReal var - varR| ≤ εvar) (heps : toReal eps = epsR)
    (hnn : 0 ≤ toReal (var + eps)) (hRnn : 0 ≤ varR + epsR) :
    |toReal (lnDenomF var eps) - Real.sqrt (varR + epsR)|
      ≤ u64 * Real.sqrt (toReal (var + eps)) + Real.sqrt (u64 * |toReal var + toReal eps| + εvar) := by
  have hadd : |toReal (var + eps) - (varR + epsR)| ≤ u64 * |toReal var + toReal eps| + εvar := by
    simpa using addApprox_error var eps varR epsR εvar 0 hvar (by rw [heps]; simp)
  simpa [lnDenomF] using
    sqrtApprox_error (var + eps) (varR + epsR) (u64 * |toReal var + toReal eps| + εvar) hadd hnn hRnn

/-- Float LayerNorm output `(xᵢ − μ)/denom · wᵢ + bᵢ`. -/
def layerNormF (x : Nat → Float) (μ denom w b : Float) (i : Nat) : Float :=
  (x i - μ) / denom * w + b

/-- **LayerNorm output error.** The affine-normalized output is within the composed `sub`/`div`/`mul`/`add`
    bound of `(toReal xᵢ − μR)/denomR · toReal wᵢ + toReal bᵢ`, given the mean error `εμ`, denom error `εden`,
    and the denominator floor `dmin` (`denom ≥ √ε > 0`). `w`,`b` exact. -/
theorem layerNorm_error (x : Nat → Float) (μ denom w b : Float) (i : Nat) (μR denomR εμ εden dmin : ℝ)
    (hμ : |toReal μ - μR| ≤ εμ) (hden : |toReal denom - denomR| ≤ εden)
    (hdmin : 0 < dmin) (hdd : dmin ≤ |toReal denom|) (hyR : denomR ≠ 0) :
    |toReal (layerNormF x μ denom w b i)
        - ((toReal (x i) - μR) / denomR * toReal w + toReal b)|
      ≤ u64 * |toReal ((x i - μ) / denom * w) + toReal b|
        + (u64 * (|toReal ((x i - μ) / denom)| * |toReal w|)
            + |toReal w|
              * (u64 * |toReal (x i - μ) / toReal denom|
                  + (u64 * |toReal (x i) - toReal μ| + εμ
                      + |(toReal (x i) - μR) / denomR| * εden) / dmin)) := by
  have hnum : |toReal (x i - μ) - (toReal (x i) - μR)| ≤ u64 * |toReal (x i) - toReal μ| + εμ := by
    simpa using subApprox_error (x i) μ (toReal (x i)) μR 0 εμ (by simp) hμ
  have hq : |toReal ((x i - μ) / denom) - (toReal (x i) - μR) / denomR|
      ≤ u64 * |toReal (x i - μ) / toReal denom|
          + (u64 * |toReal (x i) - toReal μ| + εμ + |(toReal (x i) - μR) / denomR| * εden) / dmin := by
    simpa using divApprox_error (x i - μ) denom (toReal (x i) - μR) denomR
      (u64 * |toReal (x i) - toReal μ| + εμ) εden dmin hnum hden hdmin hdd hyR
  have hqw : |toReal ((x i - μ) / denom * w) - (toReal (x i) - μR) / denomR * toReal w|
      ≤ u64 * (|toReal ((x i - μ) / denom)| * |toReal w|)
          + |toReal w|
            * (u64 * |toReal (x i - μ) / toReal denom|
                + (u64 * |toReal (x i) - toReal μ| + εμ + |(toReal (x i) - μR) / denomR| * εden) / dmin) := by
    simpa using mulApprox_error ((x i - μ) / denom) w ((toReal (x i) - μR) / denomR) (toReal w)
      (u64 * |toReal (x i - μ) / toReal denom|
        + (u64 * |toReal (x i) - toReal μ| + εμ + |(toReal (x i) - μR) / denomR| * εden) / dmin) 0
      hq (by simp)
  simpa [layerNormF] using
    addApprox_error ((x i - μ) / denom * w) b ((toReal (x i) - μR) / denomR * toReal w) (toReal b) _ 0
      hqw (by simp)

/-- **LayerNorm output error, against the canonical ℝ spec** `Net.layerNorm`. With the Float mean/denom
    within `εμ`/`εden` of the ℝ `lnMean`/`lnDenom`, `layerNormF` is within the composed bound of
    `Puffer.Net.layerNorm (Finset.range n) (toReal ∘ x) (toReal ∘ w) (toReal ∘ b) i` — the library's official
    ℝ LayerNorm. (Chain `lnMean_error`/`lnVar_error`/`lnDenom_error` to supply `εμ`/`εden`.) -/
theorem layerNorm_error_spec (x w b : Nat → Float) (μ denom : Float) (n i : Nat) (εμ εden dmin : ℝ)
    (hμ : |toReal μ - Puffer.Net.lnMean (Finset.range n) (fun j => toReal (x j))| ≤ εμ)
    (hden : |toReal denom - Puffer.Net.lnDenom (Finset.range n) (fun j => toReal (x j))| ≤ εden)
    (hdmin : 0 < dmin) (hdd : dmin ≤ |toReal denom|)
    (hyR : Puffer.Net.lnDenom (Finset.range n) (fun j => toReal (x j)) ≠ 0) :
    |toReal (layerNormF x μ denom (w i) (b i) i)
        - Puffer.Net.layerNorm (Finset.range n) (fun j => toReal (x j)) (fun j => toReal (w j))
            (fun j => toReal (b j)) i|
      ≤ u64 * |toReal ((x i - μ) / denom * w i) + toReal (b i)|
        + (u64 * (|toReal ((x i - μ) / denom)| * |toReal (w i)|)
            + |toReal (w i)|
              * (u64 * |toReal (x i - μ) / toReal denom|
                  + (u64 * |toReal (x i) - toReal μ| + εμ
                      + |(toReal (x i) - Puffer.Net.lnMean (Finset.range n) (fun j => toReal (x j)))
                          / Puffer.Net.lnDenom (Finset.range n) (fun j => toReal (x j))| * εden)
                    / dmin)) := by
  have hspec : Puffer.Net.layerNorm (Finset.range n) (fun j => toReal (x j)) (fun j => toReal (w j))
        (fun j => toReal (b j)) i
      = (toReal (x i) - Puffer.Net.lnMean (Finset.range n) (fun j => toReal (x j)))
          / Puffer.Net.lnDenom (Finset.range n) (fun j => toReal (x j)) * toReal (w i) + toReal (b i) := rfl
  rw [hspec]
  exact layerNorm_error x μ denom (w i) (b i) i
    (Puffer.Net.lnMean (Finset.range n) (fun j => toReal (x j)))
    (Puffer.Net.lnDenom (Finset.range n) (fun j => toReal (x j))) εμ εden dmin hμ hden hdmin hdd hyR

/-! ### Fully-chained: the whole LayerNorm with all errors composed concretely

`fullLayerNorm_error` chains the four stages into ONE bound with NO abstract `εμ`/`εden` hypotheses — the
stage errors are computed from the inputs. The mean/variance/denom ℝ references and error bounds are named
(`lnMuR`/`lnVarR`/`lnDenomR`, `lnMeanErr`/`lnVarErr`/`lnDenomErr`), landing against the faithful ℝ LayerNorm
`layerNormRval` (using the actual `toReal eps`, i.e. `Net.layerNorm` with `1e-5` replaced by `toReal eps`). -/

/-- ℝ reference mean/variance/denominator (`= Net.lnMean`/`lnVar`/`√(lnVar+epsR)` when `cR = card`). -/
noncomputable def lnMuR (x : Nat → Float) (n : Nat) (cR : ℝ) : ℝ :=
  (∑ j ∈ Finset.range n, toReal (x j)) / cR
noncomputable def lnVarR (x : Nat → Float) (n : Nat) (cR : ℝ) : ℝ :=
  (∑ j ∈ Finset.range n, (toReal (x j) - lnMuR x n cR) * (toReal (x j) - lnMuR x n cR)) / cR
noncomputable def lnDenomR (x : Nat → Float) (n : Nat) (cR epsR : ℝ) : ℝ :=
  Real.sqrt (lnVarR x n cR + epsR)

/-- Concrete stage error bounds (each `= lnMean_error`/`lnVar_error`/`lnDenom_error`'s RHS). -/
noncomputable def lnMeanErr (x : Nat → Float) (c : Float) (n : Nat) (cR : ℝ) : ℝ :=
  u64 * |toReal (sumIdxFGo x 0.0 (List.range n)) / toReal c|
    + (sumIdxErrBnd x (fun _ => 0) 0.0 (List.range n) 0 + |lnMuR x n cR| * 0) / cR
noncomputable def lnVarErr (x : Nat → Float) (c : Float) (n : Nat) (cR : ℝ) : ℝ :=
  u64 * |toReal (sumIdxFGo (fun k => (x k - lnMeanF x c n) * (x k - lnMeanF x c n)) 0.0 (List.range n))
        / toReal c|
    + (sumIdxErrBnd (fun k => (x k - lnMeanF x c n) * (x k - lnMeanF x c n))
          (centeredSqBnd x (lnMeanF x c n) (lnMuR x n cR) (lnMeanErr x c n cR)) 0.0 (List.range n) 0
        + |lnVarR x n cR| * 0) / cR
noncomputable def lnDenomErr (x : Nat → Float) (c eps : Float) (n : Nat) (cR : ℝ) : ℝ :=
  u64 * Real.sqrt (toReal (lnVarF x (lnMeanF x c n) c n + eps))
    + Real.sqrt (u64 * |toReal (lnVarF x (lnMeanF x c n) c n) + toReal eps| + lnVarErr x c n cR)

/-- The whole LayerNorm over `Float` from the raw inputs `x`, `w`, `b`, count `c`, epsilon `eps`. -/
def fullLayerNormF (x w b : Nat → Float) (c eps : Float) (n i : Nat) : Float :=
  layerNormF x (lnMeanF x c n) (lnDenomF (lnVarF x (lnMeanF x c n) c n) eps) (w i) (b i) i

/-- The faithful ℝ LayerNorm reference (same `toReal`'d inputs, `cR` count, `epsR = toReal eps`). -/
noncomputable def layerNormRval (x w b : Nat → Float) (n i : Nat) (cR epsR : ℝ) : ℝ :=
  (toReal (x i) - lnMuR x n cR) / lnDenomR x n cR epsR * toReal (w i) + toReal (b i)

/-- The whole concrete error bound (`layerNorm_error`'s RHS with `εμ := lnMeanErr`, `εden := lnDenomErr`). -/
noncomputable def fullLayerNormErrBnd (x w b : Nat → Float) (c eps : Float) (n i : Nat)
    (cR epsR dmin : ℝ) : ℝ :=
  u64 * |toReal ((x i - lnMeanF x c n) / lnDenomF (lnVarF x (lnMeanF x c n) c n) eps * w i) + toReal (b i)|
    + (u64 * (|toReal ((x i - lnMeanF x c n) / lnDenomF (lnVarF x (lnMeanF x c n) c n) eps)| * |toReal (w i)|)
        + |toReal (w i)|
          * (u64 * |toReal (x i - lnMeanF x c n) / toReal (lnDenomF (lnVarF x (lnMeanF x c n) c n) eps)|
              + (u64 * |toReal (x i) - toReal (lnMeanF x c n)| + lnMeanErr x c n cR
                  + |(toReal (x i) - lnMuR x n cR) / lnDenomR x n cR epsR| * lnDenomErr x c eps n cR)
                / dmin))

/-- **The fully-chained LayerNorm error.** No abstract error hypotheses: the mean, variance, and denom errors
    are the concrete `lnMeanErr`/`lnVarErr`/`lnDenomErr`, composed. The only inputs are the exact count
    (`toReal c = cR > 0`), `epsR = toReal eps`, the radicand-nonneg facts, and the denominator floor. -/
theorem fullLayerNorm_error (x w b : Nat → Float) (c eps : Float) (n i : Nat) (cR epsR dmin : ℝ)
    (hc : toReal c = cR) (hcpos : 0 < cR) (heps : toReal eps = epsR)
    (hnn : 0 ≤ toReal (lnVarF x (lnMeanF x c n) c n + eps)) (hRnn : 0 ≤ lnVarR x n cR + epsR)
    (hdmin : 0 < dmin) (hdd : dmin ≤ |toReal (lnDenomF (lnVarF x (lnMeanF x c n) c n) eps)|)
    (hyR : lnDenomR x n cR epsR ≠ 0) :
    |toReal (fullLayerNormF x w b c eps n i) - layerNormRval x w b n i cR epsR|
      ≤ fullLayerNormErrBnd x w b c eps n i cR epsR dmin := by
  have hμ : |toReal (lnMeanF x c n) - lnMuR x n cR| ≤ lnMeanErr x c n cR :=
    lnMean_error x c n cR hc hcpos
  have hvar : |toReal (lnVarF x (lnMeanF x c n) c n) - lnVarR x n cR| ≤ lnVarErr x c n cR :=
    lnVar_error x (lnMeanF x c n) c n (lnMuR x n cR) cR (lnMeanErr x c n cR) hμ hc hcpos
  have hden : |toReal (lnDenomF (lnVarF x (lnMeanF x c n) c n) eps) - lnDenomR x n cR epsR|
      ≤ lnDenomErr x c eps n cR :=
    lnDenom_error (lnVarF x (lnMeanF x c n) c n) eps (lnVarR x n cR) epsR (lnVarErr x c n cR)
      hvar heps hnn hRnn
  exact layerNorm_error x (lnMeanF x c n) (lnDenomF (lnVarF x (lnMeanF x c n) c n) eps) (w i) (b i) i
    (lnMuR x n cR) (lnDenomR x n cR epsR) (lnMeanErr x c n cR) (lnDenomErr x c eps n cR) dmin
    hμ hden hdmin hdd hyR

end Puffer.RL.LayerNormBound
