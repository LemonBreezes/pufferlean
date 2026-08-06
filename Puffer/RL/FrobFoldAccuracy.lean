/-
The fold-accuracy sub-tower: bounding the runnable `frobNorm`'s Float fold sum-of-squares `Q` against
the exact real sum of squares. This is the irreducible primitive under the eps-domination inequality
(`NewtonSchulzFloat.domination_of_fold` reduces the seed condition to a bound on `toReal Q`).

`frobNorm X₀ = Float.sqrt Q` with `Q = X₀.foldl (fun s row => s + row.foldl (fun s2 x => s2 + x·x) 0) 0`
— a NESTED left-fold accumulating squares with a rounding at every add and every product. The core fact
is that accumulating NONNEGATIVE summands with round-to-nearest addition shrinks the exact sum by at most
`(1-u64)` per step:

  • `listFoldAdd_lb` : `(1-u64)^n · (init + ∑ g) ≤ toReal (foldl (·+g·) init)` — the reusable fold
    lower-bound induction (nonneg summands `g`, `n` = length). The crux; a clean `List.foldl` induction
    over the trusted `add_model`.
  • `arrSumSq_lb` : `(1-u64)^(n+1) · ∑ (toReal x)² ≤ toReal (∑ x·x fold)` for a single Float array —
    `listFoldAdd_lb` composed with `mul_model` on each product `x·x`.
  • `foldSumSq2D_lb` : the OUTER row-fold composition — for a matrix `X₀` with uniform row length `c`,
    `toReal Q ≥ (1-u64)^(r+c+1) · ∑_row ∑_x (toReal x)²`. `listFoldAdd_lb` on the outer row-fold with
    `arrSumSq_lb` per row.

All three axiom-clean beyond the trusted Float base (`add_model`, `mul_model`, `toReal`, `toReal_zeroLit`).

The list↔`Fin` conversion turning the flattened double sum into `Fin`-indexed sums is:
  • `arr_toList_map_sum`  : `(a.toList.map g).sum = ∑ i : Fin a.size, g a[i]`  (single array);
  • `arr2D_toList_map_sum`: `(X.toList.map (row ↦ (row.toList.map g).sum)).sum
      = ∑ i : Fin X.size, ∑ j : Fin (X[i]).size, g (X[i][j])`  (nested).
Both pure-logic axiom-clean (`List.ofFn_getElem_eq_map` + `List.sum_ofFn` + `finCongr` reindex).

The shape-normalization to the `getElem!`-`Fin` form (`∑ i : Fin r, ∑ j : Fin c, g (X₀[i]![j]!)`) matching
the mirror `S` is:
  • `arr_fin_sum_normalize`  : `∑ j : Fin a.size, g a[j] = ∑ j : Fin c, g a[j]!`  (`a.size = c`);
  • `fold_sum_normalize`     : the nested version (`∑ i : Fin X.size, ∑ j : Fin (X[i]).size` ↦
      `∑ i : Fin r, ∑ j : Fin c`);
  • `foldSumSq2D_finSum_lb`  : the CLOSED fold accuracy in that form —
      `(1-u64)^(r+c+1) · ∑ i : Fin r, ∑ j : Fin c, (toReal X₀[i]![j]!)² ≤ toReal Q`.

All axiom-clean beyond the trusted Float base. `foldSumSq2D_finSum_lb` IS `toReal Q ≥ (1-u64)^(r+c+1)·S`
in the FAITHFUL case: the mirror `X₀Rᵢⱼ = toReal X₀[i]![j]!` (input `ε = 0`) gives
`toMatrixR X₀R i j = X₀R[i]![j]! = toReal X₀[i]![j]!`, so its double sum is exactly the one bounded here —
a definitional substitution to plug into `domination_of_fold`. A nonzero input `ε` adds the (already
`MatBnd`-bounded) entrywise perturbation. The substantive rounding inductions, the list↔`Fin` core, and
the shape normalization are all done here.
-/
import Mathlib
import Puffer.Float.Basic
import Puffer.Float.Muon

namespace Puffer.RL.FrobFoldAccuracy

open Puffer.FloatR

/-- **Fold lower bound.** Accumulating nonnegative summands `g x` with add-rounding shrinks the real
    value by at most `(1-u64)^n` (`n` = list length): the rounded left-fold is `≥ (1-u64)^n·(init+∑g)`. -/
theorem listFoldAdd_lb {α} (g : α → Float) (l : List α) (init : Float)
    (hinit : 0 ≤ toReal init) (hg : ∀ x ∈ l, 0 ≤ toReal (g x)) :
    (1 - u64) ^ l.length * (toReal init + (l.map (fun x => toReal (g x))).sum)
      ≤ toReal (l.foldl (fun s x => s + g x) init) := by
  induction l generalizing init with
  | nil => simp
  | cons x xs ih =>
      simp only [List.foldl_cons, List.length_cons, List.map_cons, List.sum_cons, pow_succ]
      obtain ⟨δ, hδ, heq⟩ := add_model init (g x)
      rw [abs_le] at hδ
      have hgx : 0 ≤ toReal (g x) := hg x (List.mem_cons_self ..)
      have hu1 : (0 : ℝ) ≤ 1 - u64 := by linarith [u64_lt_one]
      have hinit' : 0 ≤ toReal (init + g x) := by rw [heq]; nlinarith [hinit, hgx, hδ.1, u64_lt_one]
      have hsum_nn : 0 ≤ (xs.map (fun x => toReal (g x))).sum := by
        apply List.sum_nonneg; intro y hy; rw [List.mem_map] at hy
        obtain ⟨z, hz, rfl⟩ := hy; exact hg z (List.mem_cons_of_mem _ hz)
      have hIH := ih (init + g x) hinit' (fun y hy => hg y (List.mem_cons_of_mem _ hy))
      refine le_trans ?_ hIH
      rw [heq, mul_assoc]
      apply mul_le_mul_of_nonneg_left _ (pow_nonneg hu1 _)
      nlinarith [hδ.1, hsum_nn, hinit, hgx, u64_pos,
        mul_nonneg (add_nonneg hinit hgx) (by linarith [hδ.1] : (0 : ℝ) ≤ δ + u64)]

/-- **Fold upper bound.** The dual of `listFoldAdd_lb`: accumulating nonnegative summands `g x`
    with add-rounding grows the real value by at most `(1+u64)^n` (`n` = list length): the rounded
    left-fold is `≤ (1+u64)^n·(init+∑g)`. Together with `listFoldAdd_lb` this sandwiches the runnable
    fold's real value between `(1-u64)^n·(init+∑g)` and `(1+u64)^n·(init+∑g)` — the upper half needed
    to bound the runnable `frobNorm`'s sum-of-squares accumulator `Q` from ABOVE (the whole tower
    below establishes only lower bounds). Both hypotheses are load-bearing: with `init < 0` or a
    negative summand, the per-step `(1+δ) ≤ (1+u64)` growth-factor inequality flips (it needs a
    nonnegative multiplicand), so the envelope fails. -/
theorem listFoldAdd_ub {α} (g : α → Float) (l : List α) (init : Float)
    (hinit : 0 ≤ toReal init) (hg : ∀ x ∈ l, 0 ≤ toReal (g x)) :
    toReal (l.foldl (fun s x => s + g x) init)
      ≤ (1 + u64) ^ l.length * (toReal init + (l.map (fun x => toReal (g x))).sum) := by
  induction l generalizing init with
  | nil => simp
  | cons x xs ih =>
      simp only [List.foldl_cons, List.length_cons, List.map_cons, List.sum_cons, pow_succ]
      obtain ⟨δ, hδ, heq⟩ := add_model init (g x)
      rw [abs_le] at hδ
      have hgx : 0 ≤ toReal (g x) := hg x (List.mem_cons_self ..)
      have hu1 : (0 : ℝ) ≤ 1 + u64 := by linarith [u64_pos]
      have hinit' : 0 ≤ toReal (init + g x) := by rw [heq]; nlinarith [hinit, hgx, hδ.1, u64_lt_one]
      have hsum_nn : 0 ≤ (xs.map (fun x => toReal (g x))).sum := by
        apply List.sum_nonneg; intro y hy; rw [List.mem_map] at hy
        obtain ⟨z, hz, rfl⟩ := hy; exact hg z (List.mem_cons_of_mem _ hz)
      have hIH := ih (init + g x) hinit' (fun y hy => hg y (List.mem_cons_of_mem _ hy))
      refine le_trans hIH ?_
      rw [mul_assoc]
      apply mul_le_mul_of_nonneg_left _ (pow_nonneg hu1 _)
      rw [heq]
      nlinarith [hδ.2, hsum_nn, hinit, hgx, u64_pos,
        mul_nonneg (add_nonneg hinit hgx) (by linarith [hδ.2] : (0 : ℝ) ≤ u64 - δ),
        mul_nonneg u64_pos.le hsum_nn]

/-- Single Float array sum-of-squares lower bound: `toReal(∑ x·x fold) ≥ (1-u64)^(n+1)·∑(toReal x)²`. -/
theorem arrSumSq_lb (a : Array Float) :
    (1 - u64) ^ (a.size + 1) * (a.toList.map (fun x => (toReal x) ^ 2)).sum
      ≤ toReal (a.foldl (fun s x => s + x * x) (0.0 : Float)) := by
  rw [← Array.foldl_toList]
  have hu1 : (0 : ℝ) ≤ 1 - u64 := by linarith [u64_lt_one]
  have hterm : ∀ x : Float, (1 - u64) * (toReal x) ^ 2 ≤ toReal (x * x) := by
    intro x; obtain ⟨δ, hδ, heq⟩ := mul_model x x
    rw [heq]; rw [abs_le] at hδ; nlinarith [sq_nonneg (toReal x), hδ.1]
  have hgnn : ∀ x ∈ a.toList, 0 ≤ toReal (x * x) :=
    fun x _ => le_trans (mul_nonneg hu1 (sq_nonneg _)) (hterm x)
  have hbase := listFoldAdd_lb (fun x => x * x) a.toList 0.0 (by rw [toReal_zeroLit]) hgnn
  rw [toReal_zeroLit, zero_add, Array.length_toList] at hbase
  refine le_trans ?_ hbase
  rw [pow_succ, mul_assoc]
  apply mul_le_mul_of_nonneg_left _ (pow_nonneg hu1 _)
  rw [← List.sum_map_mul_left]
  apply List.sum_le_sum
  intro x _
  simpa using hterm x

/-- **Outer row-fold composition.** For a matrix `X₀` with uniform row length `c`, the nested Float
    sum-of-squares fold obeys `toReal Q ≥ (1-u64)^(r+c+1) · ∑_row ∑_x (toReal x)²` — `listFoldAdd_lb`
    applied to the outer row-fold, with `arrSumSq_lb` per row. This is the full nested fold accuracy of
    `frobNorm`'s argument `Q` (`frobNorm X₀ = Float.sqrt Q`). -/
theorem foldSumSq2D_lb (X0 : Puffer.FloatR.Muon.Mat) (c : Nat)
    (hrow : ∀ row ∈ X0.toList, row.size = c) :
    (1 - u64) ^ (X0.size + c + 1)
        * (X0.toList.map (fun row => (row.toList.map (fun x => (toReal x) ^ 2)).sum)).sum
      ≤ toReal (X0.foldl (fun s row => s + row.foldl (fun s2 x => s2 + x * x) 0.0) 0.0) := by
  rw [← Array.foldl_toList]
  have hu1 : (0 : ℝ) ≤ 1 - u64 := by linarith [u64_lt_one]
  have hgnn : ∀ row ∈ X0.toList, 0 ≤ toReal (row.foldl (fun s2 x => s2 + x * x) 0.0) := by
    intro row _
    refine le_trans (mul_nonneg (pow_nonneg hu1 _) ?_) (arrSumSq_lb row)
    apply List.sum_nonneg; intro y hy; rw [List.mem_map] at hy; obtain ⟨z, _, rfl⟩ := hy; positivity
  have hbase := listFoldAdd_lb (fun row => row.foldl (fun s2 x => s2 + x * x) 0.0) X0.toList 0.0
    (by rw [toReal_zeroLit]) hgnn
  rw [toReal_zeroLit, zero_add, Array.length_toList] at hbase
  refine le_trans ?_ hbase
  rw [show X0.size + c + 1 = X0.size + (c + 1) from by ring, pow_add, mul_assoc]
  apply mul_le_mul_of_nonneg_left _ (pow_nonneg hu1 _)
  rw [← List.sum_map_mul_left]
  apply List.sum_le_sum
  intro row hrow_mem
  rw [← hrow row hrow_mem]
  exact arrSumSq_lb row

/-! ### List↔`Fin` conversion for the flattened fold sum -/

/-- `Array.toList`-map-sum equals the `Fin`-indexed sum: `(a.toList.map g).sum = ∑ i : Fin a.size, g a[i]`. -/
theorem arr_toList_map_sum {α M} [AddCommMonoid M] (a : Array α) (g : α → M) :
    (a.toList.map g).sum = ∑ i : Fin a.size, g (a[i]) := by
  rw [← List.ofFn_getElem_eq_map a.toList g, List.sum_ofFn]
  refine Finset.sum_equiv (finCongr Array.length_toList) (by simp) (fun i _ => ?_)
  simp only [finCongr_apply, Array.getElem_toList, Fin.getElem_fin, Fin.val_cast]

/-- Nested `Array.toList` double-map-sum equals the `Fin × Fin` double sum
    `∑ i : Fin X.size, ∑ j : Fin (X[i]).size, g (X[i][j])`. -/
theorem arr2D_toList_map_sum {α M} [AddCommMonoid M] (X : Array (Array α)) (g : α → M) :
    (X.toList.map (fun row => (row.toList.map g).sum)).sum
      = ∑ i : Fin X.size, ∑ j : Fin (X[i]).size, g ((X[i])[j]) := by
  rw [arr_toList_map_sum X (fun row => (row.toList.map g).sum)]
  exact Finset.sum_congr rfl (fun i _ => arr_toList_map_sum (X[i]) g)

/-! ### Shape normalization to the `getElem!`-`Fin` form matching the mirror `S` -/

/-- Reindex a single-array `Fin`-sum to `Fin c` with `getElem!`: `∑ j : Fin a.size, g a[j] =
    ∑ j : Fin c, g a[j]!` (`a.size = c`). -/
theorem arr_fin_sum_normalize {α M} [AddCommMonoid M] [Inhabited α] (a : Array α) (c : Nat) (g : α → M)
    (hc : a.size = c) : ∑ j : Fin a.size, g (a[j]) = ∑ j : Fin c, g (a[j.1]!) := by
  subst hc
  refine Finset.sum_congr rfl (fun j _ => ?_)
  congr 1; rw [Fin.getElem_fin, getElem!_pos a j.1 j.2]

/-- Nested shape normalization: `∑ i : Fin X.size, ∑ j : Fin (X[i]).size, g X[i][j] =
    ∑ i : Fin r, ∑ j : Fin c, g (X[i]![j]!)` under `X.size = r` and uniform `(X[i]).size = c`. -/
theorem fold_sum_normalize {α M} [AddCommMonoid M] [Inhabited α] (X : Array (Array α))
    (r c : Nat) (g : α → M) (hr : X.size = r)
    (hc : ∀ (i : Nat) (hi : i < X.size), (X[i]).size = c) :
    ∑ i : Fin X.size, ∑ j : Fin (X[i]).size, g ((X[i])[j])
      = ∑ i : Fin r, ∑ j : Fin c, g ((X[i.1]!)[j.1]!) := by
  subst hr
  refine Finset.sum_congr rfl (fun i _ => ?_)
  have hXi : X[i] = X[i.1]! := by rw [Fin.getElem_fin, getElem!_pos X i.1 i.2]
  rw [arr_fin_sum_normalize (X[i]) c g (hc i.1 i.2), hXi]

/-- **The closed nested fold accuracy, in the `getElem!`-`Fin` form matching the mirror `S`.** For a
    matrix `X₀` of shape `r×c`, `toReal Q ≥ (1-u64)^(r+c+1) · ∑ i : Fin r, ∑ j : Fin c, (toReal X₀[i]![j]!)²`
    where `Q = X₀.foldl (…nested sum of squares…)`. In the faithful case (`toMatrixR X₀R i j =
    toReal X₀[i]![j]!`, input `ε = 0`) the double sum is exactly `S = ∑ᵢⱼ(toMatrixR X₀R)²`, so this is
    `toReal Q ≥ (1-u64)^(r+c+1)·S` — the fold accuracy in the form `domination_of_fold` consumes. -/
theorem foldSumSq2D_finSum_lb (X0 : Puffer.FloatR.Muon.Mat) (r c : Nat) (hr : X0.size = r)
    (hc : ∀ (i : Nat) (hi : i < X0.size), (X0[i]).size = c) :
    (1 - u64) ^ (r + c + 1) * (∑ i : Fin r, ∑ j : Fin c, (toReal ((X0[i.1]!)[j.1]!)) ^ 2)
      ≤ toReal (X0.foldl (fun s row => s + row.foldl (fun s2 x => s2 + x * x) 0.0) 0.0) := by
  have hrow : ∀ row ∈ X0.toList, row.size = c := by
    intro row hmem
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.mp (Array.mem_def.mpr hmem)
    exact hc i hi
  have hconv : (X0.toList.map (fun row => (row.toList.map (fun x => (toReal x) ^ 2)).sum)).sum
      = ∑ i : Fin r, ∑ j : Fin c, (toReal ((X0[i.1]!)[j.1]!)) ^ 2 := by
    rw [arr2D_toList_map_sum X0 (fun x => (toReal x) ^ 2),
      fold_sum_normalize X0 r c (fun x => (toReal x) ^ 2) hr hc]
  rw [← hconv, ← hr]
  exact foldSumSq2D_lb X0 c hrow

end Puffer.RL.FrobFoldAccuracy
