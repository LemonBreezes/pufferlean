/-
The softmax kernel — the policy output — with a proven Float↔ℝ accuracy bound.

`softmax` is THE thing an RL policy computes (action probabilities); PufferLib runs the numerically-stable
form `exp(lᵢ − m) / Σⱼ exp(lⱼ − m)` (max-subtracted, `m = maxⱼ lⱼ`), whose ℝ meaning is the plain
`exp(lᵢ)/Σⱼ exp(lⱼ)` (the `exp(−m)` cancels). This file bounds the Float computation against that ℝ spec,
composing four layers:

  • `sumIdxFGo_error` — the running sum `Σ` (a left `foldl` of rounded adds) over summands that are themselves
    only APPROXIMATE (each within `εₖ` of its ℝ target — not merely `toReal`-rounded). The reusable core.
  • `expShift_error` — one shifted exponential `exp(lᵢ − m)`: `sub_error` on `lᵢ − m` then `expApprox_error`.
  • `divApprox_error` — the final `eᵢ / z`, with the denominator floored by the (positive) `i`-th term.
  • `softmaxShift_eq` — `exp(lᵢ − m)/Σⱼ exp(lⱼ − m) = exp(lᵢ)/Σⱼ exp(lⱼ)` (the max cancels), so the bound
    lands against the canonical ℝ softmax.

`softmaxFval_error` is the per-component bound; `softmaxFval_error_spec` states it directly against the
library's official ℝ softmax `Puffer.Net.softmax`. `softmaxFval` is the faithful functional model of the
runnable `Train.softmax` (`m = maxⱼ lⱼ`, `eₖ = exp(lₖ − m)`, `z = foldl(+) 0`, `eᵢ/z`); connecting the
`Array`/`Id.run` executable to `softmaxFval` is pure structural plumbing (`Array.map`/`foldl` unfolding, no
new math), a scoped follow-up. `dmin ≤ |toReal z|`, `dmin > 0` is the denominator floor (the sum of positive
exponentials is `≳ 1`), supplied as a hypothesis. Axiom-clean beyond the trusted Float (1+δ) base.
-/
import Puffer.Float.Basic
import Puffer.Net.Forward

namespace Puffer.RL.SoftmaxBound

open Puffer.FloatR

/-- Running Float sum: `foldl (·+·)` of `e i` over an index list, from `acc`. -/
def sumIdxFGo (e : Nat → Float) : Float → List Nat → Float
  | acc, [] => acc
  | acc, i :: is => sumIdxFGo e (acc + e i) is

/-- ℝ running sum with the ideal summands `eR i`. -/
noncomputable def sumIdxRGo (eR : Nat → ℝ) : ℝ → List Nat → ℝ
  | acc, [] => acc
  | acc, i :: is => sumIdxRGo eR (acc + eR i) is

/-- The accumulated error bound for `sumIdxFGo` (one `addApprox_error` per index). -/
noncomputable def sumIdxErrBnd (e : Nat → Float) (εe : Nat → ℝ) : Float → List Nat → ℝ → ℝ
  | _, [], εacc => εacc
  | acc, i :: is, εacc =>
      sumIdxErrBnd e εe (acc + e i) is (u64 * |toReal acc + toReal (e i)| + εacc + εe i)

/-- **Approximate-summand sum error.** The running Float sum tracks the ℝ sum of the ideal summands `eR`
    within `sumIdxErrBnd`, given the accumulator error and each summand's error `|toReal(e k) − eR k| ≤ εe k`. -/
theorem sumIdxFGo_error (e : Nat → Float) (eR εe : Nat → ℝ) :
    ∀ (is : List Nat) (acc : Float) (accR εacc : ℝ),
      |toReal acc - accR| ≤ εacc → (∀ k ∈ is, |toReal (e k) - eR k| ≤ εe k) →
      |toReal (sumIdxFGo e acc is) - sumIdxRGo eR accR is| ≤ sumIdxErrBnd e εe acc is εacc := by
  intro is
  induction is with
  | nil => intro acc accR εacc hacc _; simpa [sumIdxFGo, sumIdxRGo, sumIdxErrBnd] using hacc
  | cons i is ih =>
      intro acc accR εacc hacc hsum
      simp only [sumIdxFGo, sumIdxRGo, sumIdxErrBnd]
      have hei : |toReal (e i) - eR i| ≤ εe i := hsum i (List.mem_cons_self)
      have hstep : |toReal (acc + e i) - (accR + eR i)| ≤ u64 * |toReal acc + toReal (e i)| + εacc + εe i :=
        addApprox_error acc (e i) accR (eR i) εacc (εe i) hacc hei
      exact ih (acc + e i) (accR + eR i) _ hstep
        (fun k hk => hsum k (List.mem_cons_of_mem _ hk))

/-! ### The shifted exponential and the max-cancellation -/

/-- The shifted-exponential error bound (one `sub` rounding through `exp`). -/
noncomputable def expShiftBnd (l m : Float) : ℝ :=
  expEps * Real.exp (toReal (l - m))
    + Real.exp (toReal (l - m)) * (Real.exp (u64 * |toReal l - toReal m|) - 1)

/-- **Shifted-exponential error** `exp(l − m)`: `sub_error` on `l − m`, then `expApprox_error`. -/
theorem expShift_error (l m : Float) :
    |toReal (Float.exp (l - m)) - Real.exp (toReal l - toReal m)| ≤ expShiftBnd l m :=
  expApprox_error (l - m) (toReal l - toReal m) (u64 * |toReal l - toReal m|) (sub_error l m)

/-- **Max-subtraction cancels.** `exp(lᵢ − m)/Σⱼ exp(lⱼ − m) = exp(lᵢ)/Σⱼ exp(lⱼ)` for ANY shift `m`
    (the `exp(−m)` factors out of numerator and denominator) — so the stable Float softmax targets the
    canonical ℝ softmax `Real.exp(lᵢ)/Σⱼ Real.exp(lⱼ)`. -/
theorem softmaxShift_eq (n : Nat) (l : Nat → ℝ) (m : ℝ) (i : Nat) :
    Real.exp (l i - m) / (∑ j ∈ Finset.range n, Real.exp (l j - m))
      = Real.exp (l i) / (∑ j ∈ Finset.range n, Real.exp (l j)) := by
  have hfac : ∀ j, Real.exp (l j - m) = Real.exp (-m) * Real.exp (l j) := by
    intro j; rw [show l j - m = -m + l j from by ring, Real.exp_add]
  rw [hfac i, Finset.sum_congr rfl (fun j _ => hfac j), ← Finset.mul_sum,
    mul_div_mul_left _ _ (Real.exp_pos (-m)).ne']

/-- `sumIdxRGo` from `0` over `List.range n` is the `Finset.range n` sum. -/
theorem sumIdxRGo_eq_sum (eR : Nat → ℝ) (n : Nat) :
    sumIdxRGo eR 0 (List.range n) = ∑ j ∈ Finset.range n, eR j := by
  have hgo : ∀ (is : List Nat) (accR : ℝ), sumIdxRGo eR accR is = accR + (is.map eR).sum := by
    intro is; induction is with
    | nil => intro accR; simp [sumIdxRGo]
    | cons i is ih => intro accR; simp only [sumIdxRGo, ih, List.map_cons, List.sum_cons]; ring
  rw [hgo, zero_add]
  induction n with
  | zero => simp
  | succ k ih =>
      rw [List.range_succ, List.map_append, List.sum_append, ih, Finset.sum_range_succ]; simp

/-! ### The per-component softmax bound -/

/-- The stable Float softmax's `i`-th output: `exp(lᵢ − m) / Σⱼ exp(lⱼ − m)` (max `m`, sum from `0`). -/
def softmaxFval (l : Nat → Float) (m : Float) (n i : Nat) : Float :=
  Float.exp (l i - m) / sumIdxFGo (fun k => Float.exp (l k - m)) 0.0 (List.range n)

/-- **Per-component softmax error.** `softmaxFval` is within the composed `div/sum/exp` bound of the ℝ
    softmax `Real.exp(lᵢ)/Σⱼ Real.exp(lⱼ)` (via `softmaxShift_eq`). `dmin > 0`, `dmin ≤ |toReal z|` is the
    denominator floor (the sum of positive exponentials, `≳ 1`); supplied as a hypothesis. -/
theorem softmaxFval_error (l : Nat → Float) (m : Float) (n i : Nat) (dmin : ℝ)
    (hdmin : 0 < dmin)
    (hden : dmin ≤ |toReal (sumIdxFGo (fun k => Float.exp (l k - m)) 0.0 (List.range n))|)
    (hyR : (∑ j ∈ Finset.range n, Real.exp (toReal (l j) - toReal m)) ≠ 0) :
    |toReal (softmaxFval l m n i)
        - Real.exp (toReal (l i)) / (∑ j ∈ Finset.range n, Real.exp (toReal (l j)))|
      ≤ u64 * |toReal (Float.exp (l i - m))
              / toReal (sumIdxFGo (fun k => Float.exp (l k - m)) 0.0 (List.range n))|
        + (expShiftBnd (l i) m
            + |Real.exp (toReal (l i)) / (∑ j ∈ Finset.range n, Real.exp (toReal (l j)))|
              * sumIdxErrBnd (fun k => Float.exp (l k - m)) (fun k => expShiftBnd (l k) m) 0.0
                  (List.range n) 0) / dmin := by
  have h0acc : |toReal (0.0 : Float) - (0 : ℝ)| ≤ 0 := by rw [toReal_zeroLit]; simp
  have hden' : |toReal (sumIdxFGo (fun k => Float.exp (l k - m)) 0.0 (List.range n))
        - (∑ j ∈ Finset.range n, Real.exp (toReal (l j) - toReal m))|
      ≤ sumIdxErrBnd (fun k => Float.exp (l k - m)) (fun k => expShiftBnd (l k) m) 0.0
          (List.range n) 0 := by
    have h := sumIdxFGo_error (fun k => Float.exp (l k - m))
      (fun k => Real.exp (toReal (l k) - toReal m)) (fun k => expShiftBnd (l k) m)
      (List.range n) 0.0 0 0 h0acc (fun k _ => expShift_error (l k) m)
    rwa [sumIdxRGo_eq_sum] at h
  have hdiv := divApprox_error (Float.exp (l i - m))
    (sumIdxFGo (fun k => Float.exp (l k - m)) 0.0 (List.range n))
    (Real.exp (toReal (l i) - toReal m)) (∑ j ∈ Finset.range n, Real.exp (toReal (l j) - toReal m))
    (expShiftBnd (l i) m)
    (sumIdxErrBnd (fun k => Float.exp (l k - m)) (fun k => expShiftBnd (l k) m) 0.0 (List.range n) 0)
    dmin (expShift_error (l i) m) hden' hdmin hden hyR
  rw [softmaxShift_eq n (fun j => toReal (l j)) (toReal m) i] at hdiv
  exact hdiv

/-- **Per-component softmax error, against the canonical ℝ spec** `Net.Forward.softmax`. The stable Float
    softmax `softmaxFval` is within the composed bound of `Puffer.Net.softmax (Finset.range n)
    (toReal ∘ logits) i` — the library's official ℝ softmax. -/
theorem softmaxFval_error_spec (l : Nat → Float) (m : Float) (n i : Nat) (dmin : ℝ)
    (hdmin : 0 < dmin)
    (hden : dmin ≤ |toReal (sumIdxFGo (fun k => Float.exp (l k - m)) 0.0 (List.range n))|)
    (hyR : (∑ j ∈ Finset.range n, Real.exp (toReal (l j) - toReal m)) ≠ 0) :
    |toReal (softmaxFval l m n i)
        - Puffer.Net.softmax (Finset.range n) (fun j => toReal (l j)) i|
      ≤ u64 * |toReal (Float.exp (l i - m))
              / toReal (sumIdxFGo (fun k => Float.exp (l k - m)) 0.0 (List.range n))|
        + (expShiftBnd (l i) m
            + |Puffer.Net.softmax (Finset.range n) (fun j => toReal (l j)) i|
              * sumIdxErrBnd (fun k => Float.exp (l k - m)) (fun k => expShiftBnd (l k) m) 0.0
                  (List.range n) 0) / dmin := by
  have hspec : Puffer.Net.softmax (Finset.range n) (fun j => toReal (l j)) i
      = Real.exp (toReal (l i)) / (∑ j ∈ Finset.range n, Real.exp (toReal (l j))) := rfl
  rw [hspec]
  exact softmaxFval_error l m n i dmin hdmin hden hyR

/-! ### Runtime soundness: the executed softmax probabilities sum to ≈ 1 -/

/-- The per-component softmax error budget (the RHS of `softmaxFval_error_spec`), named so the
    sum-to-one deviation reads cleanly. -/
noncomputable def softmaxFvalErrBnd (l : Nat → Float) (m : Float) (n i : Nat) (dmin : ℝ) : ℝ :=
  u64 * |toReal (Float.exp (l i - m))
          / toReal (sumIdxFGo (fun k => Float.exp (l k - m)) 0.0 (List.range n))|
    + (expShiftBnd (l i) m
        + |Puffer.Net.softmax (Finset.range n) (fun j => toReal (l j)) i|
          * sumIdxErrBnd (fun k => Float.exp (l k - m)) (fun k => expShiftBnd (l k) m) 0.0
              (List.range n) 0) / dmin

/-- **The stable-softmax probabilities sum to within the summed per-component error of 1.** Since the ℝ
    softmax `Puffer.Net.softmax` sums to EXACTLY `1` (`softmax_sum_one`), the deviation of `Σᵢ softmaxFval`
    from `1` is bounded by the sum of the per-component `softmaxFval_error_spec` budgets (Finset triangle
    inequality). Together with `softmaxFval_pos` (a125), the executed policy is an approximate distribution:
    positive and normalized. -/
theorem softmaxFval_sum_error (l : Nat → Float) (m : Float) (n : Nat) (dmin : ℝ)
    (hn : 0 < n) (hdmin : 0 < dmin)
    (hden : dmin ≤ |toReal (sumIdxFGo (fun k => Float.exp (l k - m)) 0.0 (List.range n))|)
    (hyR : (∑ j ∈ Finset.range n, Real.exp (toReal (l j) - toReal m)) ≠ 0) :
    |(∑ i ∈ Finset.range n, toReal (softmaxFval l m n i)) - 1|
      ≤ ∑ i ∈ Finset.range n, softmaxFvalErrBnd l m n i dmin := by
  have hsum1 : (∑ i ∈ Finset.range n, Puffer.Net.softmax (Finset.range n) (fun j => toReal (l j)) i) = 1 :=
    Puffer.Net.softmax_sum_one (Finset.range n) (fun j => toReal (l j))
      (Finset.nonempty_range_iff.mpr (by omega))
  calc |(∑ i ∈ Finset.range n, toReal (softmaxFval l m n i)) - 1|
      = |∑ i ∈ Finset.range n,
          (toReal (softmaxFval l m n i)
            - Puffer.Net.softmax (Finset.range n) (fun j => toReal (l j)) i)| := by
        rw [Finset.sum_sub_distrib, hsum1]
    _ ≤ ∑ i ∈ Finset.range n,
          |toReal (softmaxFval l m n i)
            - Puffer.Net.softmax (Finset.range n) (fun j => toReal (l j)) i| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i ∈ Finset.range n, softmaxFvalErrBnd l m n i dmin :=
        Finset.sum_le_sum (fun i _ => softmaxFval_error_spec l m n i dmin hdmin hden hyR)

end Puffer.RL.SoftmaxBound
