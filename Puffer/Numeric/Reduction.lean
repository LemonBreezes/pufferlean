/-
Nonassociative reduction error — the rigorous bound for parallel float sums.

PufferLib's 66 reduction-order-dependent operations (advantage mean/var, grad
norm, softmax denominators, layernorm, Muon, …) sum many floats in a
hardware-determined tree order. Float addition is nonassociative, so no
real-number model can be bit-exact. But the *error* is bounded independent of the
tree shape: a summation tree of depth `d`, evaluated with each addition rounded by
any `rnd` satisfying `|rnd x − x| ≤ u·|x|`, deviates from the exact sum by at most

    ((1+u)^d − 1) · Σ|xᵢ|        (Higham-style, order/shape-agnostic).

This is exactly the worst-case-over-orderings guarantee the "full error-bound
theorems" track needs. Instantiated at `rnd = bf16, u = 2^{-7}` (or f32,
`u = 2^{-24}`) it turns any parallel reduction into a proven error bar.
-/
import Mathlib
import Puffer.Numeric.Bf16

namespace Puffer.Numeric

/-- A binary summation tree: leaves are exact real summands, nodes are additions. -/
inductive SumTree where
  | leaf : ℝ → SumTree
  | node : SumTree → SumTree → SumTree

namespace SumTree

/-- The exact (order-independent) sum of the leaves. -/
def exactSum : SumTree → ℝ
  | leaf x => x
  | node l r => exactSum l + exactSum r

/-- Sum of leaf magnitudes — the natural error scale. -/
def sumAbs : SumTree → ℝ
  | leaf x => |x|
  | node l r => sumAbs l + sumAbs r

/-- Tree depth = number of roundings on the longest root-to-leaf path. -/
def depth : SumTree → ℕ
  | leaf _ => 0
  | node l r => max (depth l) (depth r) + 1

/-- Evaluation where every addition is rounded by `rnd` (leaves exact). -/
def roundedEval (rnd : ℝ → ℝ) : SumTree → ℝ
  | leaf x => x
  | node l r => rnd (roundedEval rnd l + roundedEval rnd r)

theorem sumAbs_nonneg : ∀ t : SumTree, 0 ≤ sumAbs t
  | leaf x => abs_nonneg x
  | node l r => add_nonneg (sumAbs_nonneg l) (sumAbs_nonneg r)

variable (rnd : ℝ → ℝ) (u : ℝ)

/-- Magnitude growth: a rounded tree sum is at most `(1+u)^depth · Σ|xᵢ|`. -/
theorem roundedEval_abs_le (hu : 0 ≤ u) (hr : ∀ x, |rnd x - x| ≤ u * |x|) (t : SumTree) :
    |roundedEval rnd t| ≤ (1 + u) ^ depth t * sumAbs t := by
  induction t with
  | leaf x => simp [roundedEval, depth, sumAbs]
  | node l r ihl ihr =>
      show |rnd (roundedEval rnd l + roundedEval rnd r)| ≤ (1 + u) ^ depth (node l r) * sumAbs (node l r)
      set a := roundedEval rnd l
      set b := roundedEval rnd r
      have h1u : (1 : ℝ) ≤ 1 + u := by linarith
      have hround : |rnd (a + b)| ≤ (1 + u) * |a + b| := by
        have hr' := hr (a + b)
        have hsplit : |rnd (a + b)| ≤ |rnd (a + b) - (a + b)| + |a + b| := by
          calc |rnd (a + b)| = |(rnd (a + b) - (a + b)) + (a + b)| := by congr 1; ring
            _ ≤ |rnd (a + b) - (a + b)| + |a + b| := abs_add_le _ _
        nlinarith [hr', hsplit]
      have hab : |a + b| ≤ (1 + u) ^ depth l * sumAbs l + (1 + u) ^ depth r * sumAbs r :=
        le_trans (abs_add_le _ _) (add_le_add ihl ihr)
      have hPl : (1 + u) ^ depth l ≤ (1 + u) ^ max (depth l) (depth r) :=
        pow_le_pow_right₀ h1u (le_max_left _ _)
      have hPr : (1 + u) ^ depth r ≤ (1 + u) ^ max (depth l) (depth r) :=
        pow_le_pow_right₀ h1u (le_max_right _ _)
      calc |rnd (a + b)|
          ≤ (1 + u) * |a + b| := hround
        _ ≤ (1 + u) * ((1 + u) ^ depth l * sumAbs l + (1 + u) ^ depth r * sumAbs r) :=
              mul_le_mul_of_nonneg_left hab (by linarith)
        _ ≤ (1 + u) * ((1 + u) ^ max (depth l) (depth r) * sumAbs l
              + (1 + u) ^ max (depth l) (depth r) * sumAbs r) := by
              apply mul_le_mul_of_nonneg_left _ (by linarith)
              exact add_le_add (mul_le_mul_of_nonneg_right hPl (sumAbs_nonneg l))
                (mul_le_mul_of_nonneg_right hPr (sumAbs_nonneg r))
        _ = (1 + u) ^ depth (node l r) * sumAbs (node l r) := by
              simp only [depth, sumAbs, pow_succ]; ring

/-- **Reduction error bound.** For any summation tree of depth `d`, evaluating it
    with each addition rounded (`|rnd x − x| ≤ u|x|`) deviates from the exact sum by
    at most `((1+u)^d − 1)·Σ|xᵢ|` — independent of the tree/ordering. -/
theorem roundedEval_error (hu : 0 ≤ u) (hr : ∀ x, |rnd x - x| ≤ u * |x|) (t : SumTree) :
    |roundedEval rnd t - exactSum t| ≤ ((1 + u) ^ depth t - 1) * sumAbs t := by
  induction t with
  | leaf x => simp [roundedEval, exactSum, depth]
  | node l r ihl ihr =>
      show |rnd (roundedEval rnd l + roundedEval rnd r) - (exactSum l + exactSum r)|
            ≤ ((1 + u) ^ depth (node l r) - 1) * sumAbs (node l r)
      set a := roundedEval rnd l
      set b := roundedEval rnd r
      have hA := sumAbs_nonneg l
      have hB := sumAbs_nonneg r
      have h1u : (1 : ℝ) ≤ 1 + u := by linarith
      set P := (1 + u) ^ max (depth l) (depth r) with hP
      -- decompose the error: rounding of (a+b) + propagation from the subtrees
      have hround : |rnd (a + b) - (a + b)| ≤ u * |a + b| := hr (a + b)
      have habs_l : |a| ≤ P * sumAbs l := by
        rw [hP]
        exact le_trans (roundedEval_abs_le rnd u hu hr l)
          (mul_le_mul_of_nonneg_right (pow_le_pow_right₀ h1u (le_max_left _ _)) hA)
      have habs_r : |b| ≤ P * sumAbs r := by
        rw [hP]
        exact le_trans (roundedEval_abs_le rnd u hu hr r)
          (mul_le_mul_of_nonneg_right (pow_le_pow_right₀ h1u (le_max_right _ _)) hB)
      have hPl : (1 + u) ^ depth l - 1 ≤ P - 1 := by
        rw [hP]; exact sub_le_sub_right (pow_le_pow_right₀ h1u (le_max_left _ _)) 1
      have hPr : (1 + u) ^ depth r - 1 ≤ P - 1 := by
        rw [hP]; exact sub_le_sub_right (pow_le_pow_right₀ h1u (le_max_right _ _)) 1
      -- triangle: |rnd(a+b) - (sₗ+sᵣ)| ≤ u|a+b| + |a-sₗ| + |b-sᵣ|
      have htri : |rnd (a + b) - (exactSum l + exactSum r)|
                    ≤ u * |a + b| + (|a - exactSum l| + |b - exactSum r|) := by
        calc |rnd (a + b) - (exactSum l + exactSum r)|
            ≤ |rnd (a + b) - (a + b)| + |(a + b) - (exactSum l + exactSum r)| := by
                  rw [show rnd (a + b) - (exactSum l + exactSum r)
                        = (rnd (a + b) - (a + b)) + ((a + b) - (exactSum l + exactSum r)) by ring]
                  exact abs_add_le _ _
          _ ≤ u * |a + b| + (|a - exactSum l| + |b - exactSum r|) := by
                have : |(a + b) - (exactSum l + exactSum r)| ≤ |a - exactSum l| + |b - exactSum r| := by
                  rw [show (a + b) - (exactSum l + exactSum r)
                        = (a - exactSum l) + (b - exactSum r) by ring]
                  exact abs_add_le _ _
                linarith [hround]
      have hab : |a + b| ≤ P * (sumAbs l + sumAbs r) := by
        calc |a + b| ≤ |a| + |b| := abs_add_le _ _
          _ ≤ P * sumAbs l + P * sumAbs r := add_le_add habs_l habs_r
          _ = P * (sumAbs l + sumAbs r) := by ring
      -- assemble; RHS target = ((1+u)P − 1)(A+B)
      calc |rnd (a + b) - (exactSum l + exactSum r)|
          ≤ u * |a + b| + (|a - exactSum l| + |b - exactSum r|) := htri
        _ ≤ u * (P * (sumAbs l + sumAbs r))
              + (((1 + u) ^ depth l - 1) * sumAbs l + ((1 + u) ^ depth r - 1) * sumAbs r) :=
              add_le_add (mul_le_mul_of_nonneg_left hab hu) (add_le_add ihl ihr)
        _ ≤ u * (P * (sumAbs l + sumAbs r)) + ((P - 1) * sumAbs l + (P - 1) * sumAbs r) :=
              add_le_add le_rfl
                (add_le_add (mul_le_mul_of_nonneg_right hPl hA)
                  (mul_le_mul_of_nonneg_right hPr hB))
        _ = ((1 + u) ^ depth (node l r) - 1) * sumAbs (node l r) := by
              simp only [depth, sumAbs, pow_succ]; rw [← hP]; ring

/-- Every leaf of the summation tree is nonnegative — the "no cancellation" condition. Under it the
    exact sum equals `Σ|xᵢ|`, so the absolute error scale `sumAbs` coincides with the answer's own
    magnitude and the absolute bound upgrades to a *relative* one. -/
def LeafNonneg : SumTree → Prop
  | leaf x => 0 ≤ x
  | node l r => LeafNonneg l ∧ LeafNonneg r

/-- **Relative reduction error (no cancellation).** For a summation tree whose leaves are all
    nonnegative, the rounded reduction has *relative* error at most `(1+u)^d − 1`:
        `|roundedEval rnd t − exactSum t| ≤ ((1+u)^depth t − 1) · exactSum t`.
    Reason: for nonneg leaves `exactSum t = Σ|xᵢ| = sumAbs t` (proved inline by induction), so the
    order/shape-agnostic absolute bound `roundedEval_error` becomes relative to the answer itself —
    the practically meaningful guarantee for the trainer's nonnegative reductions (variance and
    grad-norm sums of squares, softmax/`logSumExp` denominators). The hypothesis is load-bearing:
    with cancellation `exactSum t` can be small or negative while `sumAbs t` (hence the true error)
    stays large, making the right-hand side too small — even negative — to bound `|·|`. -/
theorem roundedEval_relative_error (hu : 0 ≤ u) (hr : ∀ x, |rnd x - x| ≤ u * |x|)
    (t : SumTree) (h : LeafNonneg t) :
    |roundedEval rnd t - exactSum t| ≤ ((1 + u) ^ depth t - 1) * exactSum t := by
  -- No cancellation ⇒ the exact sum equals the magnitude sum (real induction over the tree).
  have heq : ∀ s : SumTree, LeafNonneg s → exactSum s = sumAbs s := by
    intro s
    induction s with
    | leaf x => intro hs; exact (abs_of_nonneg hs).symm
    | node l r ihl ihr =>
        intro hs
        obtain ⟨hl, hr'⟩ := hs
        simp only [exactSum, sumAbs]
        rw [ihl hl, ihr hr']
  -- Feed that into the absolute, order-agnostic error bound.
  have hbd := roundedEval_error rnd u hu hr t
  rw [← heq t h] at hbd
  exact hbd

/-- bf16 specialization: any bf16-rounded reduction of depth `d` is within
    `((1+2^{-7})^d − 1)·Σ|xᵢ|` of the exact real sum. -/
theorem roundedEval_bf16_error (t : SumTree) :
    |roundedEval bf16 t - exactSum t|
      ≤ ((1 + (2 : ℝ) ^ (-7 : ℤ)) ^ depth t - 1) * sumAbs t :=
  roundedEval_error bf16 ((2 : ℝ) ^ (-7 : ℤ)) (by positivity) bf16_error_bound t

end SumTree
end Puffer.Numeric
