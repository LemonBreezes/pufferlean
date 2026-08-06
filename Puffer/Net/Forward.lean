/-
Neural-net forward pass over ℝ — faithful specs of PufferLib's CPU inference
reference (`~/src/PufferLib/src/puffernet.h`).

Layers mirror the C exactly. Theorems capture each layer's defining property plus,
for the dot product (the linear/GEMM core), a bf16 input-rounding error bound that
composes `bf16_error_bound` through the accumulation — the forward-pass analogue of
the GAE/reduction error bounds.
-/
import Mathlib
import Puffer.Numeric.Bf16

namespace Puffer.Net

open Finset Puffer.Numeric

/-! ### Linear layer / dot product (`_linear`) -/

/-- Dot product `Σ xᵢ·wᵢ` (the inner loop of `_linear`). -/
def dot : List ℝ → List ℝ → ℝ
  | [], _ => 0
  | _, [] => 0
  | x :: xs, w :: ws => x * w + dot xs ws

/-- `Σ |xᵢ·wᵢ|` — the natural error scale for the dot product. -/
def dotAbs : List ℝ → List ℝ → ℝ
  | [], _ => 0
  | _, [] => 0
  | x :: xs, w :: ws => |x * w| + dotAbs xs ws

/-- One linear-layer output: `bias + Σ xᵢ·wᵢ` (`_linear`, puffernet.h:101). -/
def linearUnit (x w : List ℝ) (bias : ℝ) : ℝ := dot x w + bias

/-- **Dot-product bf16 input error.** Feeding bf16-truncated inputs perturbs the dot
    product by at most `2⁻⁷·Σ|xᵢwᵢ|` — the linear layer's forward error bound. -/
theorem dot_bf16_error : ∀ (x w : List ℝ),
    |dot x w - dot (x.map bf16) w| ≤ (2 : ℝ) ^ (-7 : ℤ) * dotAbs x w := by
  intro x
  induction x with
  | nil => intro w; simp [dot, dotAbs]
  | cons x xs ih =>
      intro w
      cases w with
      | nil => simp [dot, dotAbs]
      | cons w ws =>
          simp only [List.map_cons, dot, dotAbs]
          have hbf : |x - bf16 x| ≤ (2 : ℝ) ^ (-7 : ℤ) * |x| := by
            rw [abs_sub_comm]; exact bf16_error_bound x
          calc |x * w + dot xs ws - (bf16 x * w + dot (xs.map bf16) ws)|
              = |(x - bf16 x) * w + (dot xs ws - dot (xs.map bf16) ws)| := by congr 1; ring
            _ ≤ |(x - bf16 x) * w| + |dot xs ws - dot (xs.map bf16) ws| := abs_add_le _ _
            _ ≤ |x - bf16 x| * |w| + (2 : ℝ) ^ (-7 : ℤ) * dotAbs xs ws := by
                  rw [abs_mul]; exact add_le_add le_rfl (ih ws)
            _ ≤ (2 : ℝ) ^ (-7 : ℤ) * |x| * |w| + (2 : ℝ) ^ (-7 : ℤ) * dotAbs xs ws :=
                  add_le_add (mul_le_mul_of_nonneg_right hbf (abs_nonneg w)) le_rfl
            _ = (2 : ℝ) ^ (-7 : ℤ) * (|x * w| + dotAbs xs ws) := by rw [abs_mul]; ring

/-- `dotAbs` is nonnegative — a sum of absolute term-products. -/
theorem dotAbs_nonneg : ∀ (x w : List ℝ), 0 ≤ dotAbs x w := by
  intro x
  induction x with
  | nil => intro w; simp [dotAbs]
  | cons x xs ih =>
      intro w
      cases w with
      | nil => simp [dotAbs]
      | cons w ws => simp only [dotAbs]; exact add_nonneg (abs_nonneg _) (ih ws)

/-- **Dot product is symmetric.** `dot x w = dot w x` — the linear/GEMM inner product commutes in its two
    argument lists (each term `xᵢ·wᵢ = wᵢ·xᵢ`). Handles unequal lengths (both truncate at the shorter). -/
theorem dot_comm : ∀ (x w : List ℝ), dot x w = dot w x := by
  intro x
  induction x with
  | nil => intro w; cases w <;> simp [dot]
  | cons x xs ih =>
      intro w
      cases w with
      | nil => simp [dot]
      | cons w ws => simp only [dot]; rw [mul_comm x w, ih ws]

/-- **Dot product is block-additive (concatenation law).** If the left blocks have equal length
    (`a.length = b.length`), the dot product of concatenations splits into the block dot products:
    `dot (a ++ c) (b ++ d) = dot a b + dot c d`. This is the structural backbone of blocked GEMM — an inner
    product over concatenated feature/weight vectors is the sum of the per-block inner products. The length
    hypothesis is load-bearing: since `dot` truncates at the shorter list, a length mismatch lets the second
    block "slide into" the gap left by the first (e.g. `dot ([a]++[c']) ([]++[d']) = a·d'`, but
    `dot [a] [] + dot [c'] [d'] = c'·d'`). -/
theorem dot_append : ∀ (a b : List ℝ), a.length = b.length → ∀ (c d : List ℝ),
    dot (a ++ c) (b ++ d) = dot a b + dot c d := by
  intro a
  induction a with
  | nil =>
      intro b hlen c d
      cases b with
      | nil => simp [dot]
      | cons y ys => simp at hlen
  | cons x xs ih =>
      intro b hlen c d
      cases b with
      | nil => simp at hlen
      | cons y ys =>
          simp only [List.cons_append, dot]
          rw [List.length_cons, List.length_cons] at hlen
          have hlen' : xs.length = ys.length := by omega
          rw [ih ys hlen' c d]
          ring

/-- **Dot product is homogeneous in the left input.** `dot (c·x) w = c · dot x w` — scaling every input feature
    by `c` scales the linear pre-activation by `c` (degree-1 linearity of the GEMM core). -/
theorem dot_smul_left (c : ℝ) : ∀ (x w : List ℝ),
    dot (x.map (fun a => c * a)) w = c * dot x w := by
  intro x
  induction x with
  | nil => intro w; simp [dot]
  | cons x xs ih =>
      intro w
      cases w with
      | nil => simp [dot]
      | cons w ws => simp only [List.map_cons, dot]; rw [ih ws]; ring

/-- **Dot product is homogeneous in the right input** (`dot_smul_left` transported by `dot_comm`):
    `dot x (c·w) = c · dot x w` — scaling the weights by `c` scales the output by `c`. -/
theorem dot_smul_right (c : ℝ) (x w : List ℝ) :
    dot x (w.map (fun a => c * a)) = c * dot x w := by
  rw [dot_comm x, dot_smul_left, dot_comm w x]

/-- **Dot-product triangle inequality.** `|dot x w| ≤ dotAbs x w` — the linear layer's output magnitude never
    exceeds the sum of absolute term-products `Σ|xᵢwᵢ|`. The foundational magnitude bound for the GEMM core (the
    reason `dotAbs` is the natural error scale in `dot_bf16_error`), proved by induction with `abs_add_le` at each
    accumulation step. -/
theorem dot_abs_le_dotAbs : ∀ (x w : List ℝ), |dot x w| ≤ dotAbs x w := by
  intro x
  induction x with
  | nil => intro w; simp [dot, dotAbs]
  | cons x xs ih =>
      intro w
      cases w with
      | nil => simp [dot, dotAbs]
      | cons w ws =>
          simp only [dot, dotAbs]
          calc |x * w + dot xs ws| ≤ |x * w| + |dot xs ws| := abs_add_le _ _
            _ ≤ |x * w| + dotAbs xs ws := add_le_add le_rfl (ih ws)

/-- **Linear-unit magnitude bound.** `|linearUnit x w bias| ≤ dotAbs x w + |bias|` — a single linear-layer
    output is bounded by the absolute weighted sum plus the bias magnitude (`dot_abs_le_dotAbs` + triangle). -/
theorem linearUnit_abs_le (x w : List ℝ) (bias : ℝ) :
    |linearUnit x w bias| ≤ dotAbs x w + |bias| := by
  unfold linearUnit
  calc |dot x w + bias| ≤ |dot x w| + |bias| := abs_add_le _ _
    _ ≤ dotAbs x w + |bias| := add_le_add (dot_abs_le_dotAbs x w) le_rfl

/-! ### Activations (`_relu`, `_gelu`, `_sigmoid`) -/

/-- ReLU `max(x, 0)` (`_relu`, fmaxf(0,x)). -/
def relu (x : ℝ) : ℝ := max x 0

theorem relu_nonneg (x : ℝ) : 0 ≤ relu x := le_max_right _ _
theorem relu_eq_self_of_nonneg (x : ℝ) (h : 0 ≤ x) : relu x = x := max_eq_left h

/-- ReLU zeroes out non-positive inputs. -/
theorem relu_eq_zero_of_nonpos (x : ℝ) (h : x ≤ 0) : relu x = 0 := max_eq_right h

/-- **ReLU is monotone.** `x ≤ y → relu x ≤ relu y` — the activation preserves the order of its inputs. -/
theorem relu_mono (x y : ℝ) (h : x ≤ y) : relu x ≤ relu y := max_le_max h le_rfl

/-- **ReLU is idempotent.** `relu (relu x) = relu x` — applying ReLU to an already-rectified value is a no-op
    (its output is nonnegative, so ReLU fixes it). -/
theorem relu_idem (x : ℝ) : relu (relu x) = relu x := relu_eq_self_of_nonneg (relu x) (relu_nonneg x)

/-- **ReLU is 1-Lipschitz** — it never amplifies error through the layer. -/
theorem relu_lipschitz (x y : ℝ) : |relu x - relu y| ≤ |x - y| :=
  abs_max_sub_max_le_abs x y 0

/-- **Positive/negative-part decomposition.** `relu x − relu (−x) = x` — every real is the difference of its
    positive part `x⁺ = relu x` and negative part `x⁻ = relu (−x)`. Two ReLUs with input negation reconstruct the
    identity, the basis of the signed split (and why a leaky/absolute unit can be built from ReLUs). -/
theorem relu_sub_relu_neg (x : ℝ) : relu x - relu (-x) = x := by
  unfold relu
  rcases le_or_gt 0 x with h | h
  · rw [max_eq_left h, max_eq_right (by linarith : -x ≤ 0)]; ring
  · rw [max_eq_right (le_of_lt h), max_eq_left (by linarith : 0 ≤ -x)]; ring

/-- **ReLU builds the absolute value.** `relu x + relu (−x) = |x|` — the SUM of the positive and negative parts
    is the magnitude (`|x| = x⁺ + x⁻`), the companion of `relu_sub_relu_neg`. So `|x|` is expressible as a
    two-ReLU network. -/
theorem relu_add_relu_neg (x : ℝ) : relu x + relu (-x) = |x| := by
  unfold relu
  rcases le_or_gt 0 x with h | h
  · rw [max_eq_left h, max_eq_right (by linarith : -x ≤ 0), abs_of_nonneg h]; ring
  · rw [max_eq_right (le_of_lt h), max_eq_left (by linarith : 0 ≤ -x), abs_of_neg h]; ring

/-- Logistic sigmoid `1/(1+e^{-x})` (`_sigmoid`). -/
noncomputable def sigmoid (x : ℝ) : ℝ := 1 / (1 + Real.exp (-x))

theorem sigmoid_pos (x : ℝ) : 0 < sigmoid x := by unfold sigmoid; positivity

theorem sigmoid_lt_one (x : ℝ) : sigmoid x < 1 := by
  unfold sigmoid
  rw [div_lt_one (by positivity)]
  have := Real.exp_pos (-x); linarith

/-- Sigmoid passes through `½` at the origin. -/
theorem sigmoid_half : sigmoid 0 = 1/2 := by
  unfold sigmoid; rw [neg_zero, Real.exp_zero]; norm_num

/-- **Sigmoid is monotone.** `x ≤ y → sigmoid x ≤ sigmoid y` — larger input, larger activation (`exp(−·)`
    decreasing shrinks the denominator). -/
theorem sigmoid_mono (x y : ℝ) (h : x ≤ y) : sigmoid x ≤ sigmoid y := by
  unfold sigmoid
  apply one_div_le_one_div_of_le
  · positivity
  · have : Real.exp (-y) ≤ Real.exp (-x) := Real.exp_le_exp.mpr (by linarith)
    linarith

/-- **Sigmoid reflection symmetry.** `sigmoid (−x) = 1 − sigmoid x` — the logistic curve is point-symmetric
    about `(0, ½)`. -/
theorem sigmoid_symm (x : ℝ) : sigmoid (-x) = 1 - sigmoid x := by
  unfold sigmoid
  rw [neg_neg]
  have h1 : (0:ℝ) < 1 + Real.exp (-x) := by positivity
  have h2 : (0:ℝ) < 1 + Real.exp x := by positivity
  have hexp : Real.exp x * Real.exp (-x) = 1 := by
    rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
  field_simp
  nlinarith [hexp]

/-- **Sigmoid odds ratio.** `sigmoid x / (1 − sigmoid x) = eˣ` — the odds of the logistic probability are the
    exponential of the logit. `1 − sigmoid x = e^{−x}/(1+e^{−x})` is the reflected sigmoid, so the ratio
    collapses to `1/e^{−x} = eˣ` (via `eˣ·e^{−x} = 1`). -/
theorem sigmoid_odds (x : ℝ) : sigmoid x / (1 - sigmoid x) = Real.exp x := by
  have h1 : (0:ℝ) < 1 + Real.exp (-x) := by positivity
  have hexp : Real.exp x * Real.exp (-x) = 1 := by
    rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
  have hlt : sigmoid x < 1 := sigmoid_lt_one x
  have hne : (1:ℝ) - sigmoid x ≠ 0 := ne_of_gt (by linarith)
  rw [div_eq_iff hne]
  unfold sigmoid
  field_simp
  nlinarith [hexp]

/-- **Sigmoid inverts the logit.** `log(sigmoid x / (1 − sigmoid x)) = x` — the log-odds of the sigmoid output
    recover the pre-activation exactly. This is the defining inverse relationship: sigmoid maps a real logit to a
    probability `p ∈ (0,1)`, and `log(p/(1−p))` maps it back. Immediate from `sigmoid_odds` and `log∘exp = id`. -/
theorem sigmoid_logit_inv (x : ℝ) : Real.log (sigmoid x / (1 - sigmoid x)) = x := by
  rw [sigmoid_odds, Real.log_exp]

/-- **Sigmoid is a rescaled tanh.** `sigmoid x = (1 + tanh(x/2))/2` — the logistic and hyperbolic-tangent
    activations are the *same curve*, related by an affine reparametrization. Writing `A = e^{x/2}`, `B = e^{−x/2}`
    with `A·B = 1`: `tanh(x/2) = (A−B)/(A+B)` and `1/(1+e^{−x}) = 1/(1+B²)`, and the two collapse to `A/(A+B)`.
    This is why GELU's tanh gate (`gelu`, below) and the logistic policy head are interchangeable up to scaling. -/
theorem sigmoid_eq_tanh (x : ℝ) : sigmoid x = (1 + Real.tanh (x / 2)) / 2 := by
  have hnx : Real.exp (-x) = Real.exp (-(x / 2)) * Real.exp (-(x / 2)) := by
    rw [← Real.exp_add]; ring_nf
  unfold sigmoid
  rw [Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq, hnx]
  set A := Real.exp (x / 2) with hA
  set B := Real.exp (-(x / 2)) with hB
  have hAB : A * B = 1 := by rw [hA, hB, ← Real.exp_add]; ring_nf; exact Real.exp_zero
  have hApos : 0 < A := Real.exp_pos _
  have hBpos : 0 < B := Real.exp_pos _
  have hab : 0 < A + B := by linarith
  have hbb : 0 < 1 + B * B := by positivity
  field_simp
  nlinarith [hAB, hApos, hBpos]

/-- **Tanh is a rescaled sigmoid** (the inverse form): `tanh(x/2) = 2·sigmoid x − 1`. Maps the logistic output
    range `(0,1)` onto tanh's `(−1,1)`. -/
theorem tanh_half_eq (x : ℝ) : Real.tanh (x / 2) = 2 * sigmoid x - 1 := by
  rw [sigmoid_eq_tanh]; ring

/-- GELU (tanh approximation), exactly as in `_gelu` (puffernet.h:92). -/
noncomputable def gelu (x : ℝ) : ℝ :=
  0.5 * x * (1 + Real.tanh (0.7978845608028654 * (x + 0.044715 * x ^ 3)))

theorem gelu_zero : gelu 0 = 0 := by simp [gelu]

/-- **GELU preserves the sign of its input** (nonnegative side). `0 ≤ x → 0 ≤ gelu x` — the gate factor
    `1 + tanh(…)` is strictly positive (`tanh > −1`), so `gelu x = ½·x·(gate)` keeps `x`'s sign. -/
theorem gelu_nonneg_of_nonneg (x : ℝ) (h : 0 ≤ x) : 0 ≤ gelu x := by
  unfold gelu
  have ht : 0 < 1 + Real.tanh (0.7978845608028654 * (x + 0.044715 * x ^ 3)) := by
    have := Real.neg_one_lt_tanh (0.7978845608028654 * (x + 0.044715 * x ^ 3)); linarith
  nlinarith [ht, h]

/-- **GELU preserves the sign of its input** (nonpositive side). `x ≤ 0 → gelu x ≤ 0`. -/
theorem gelu_nonpos_of_nonpos (x : ℝ) (h : x ≤ 0) : gelu x ≤ 0 := by
  unfold gelu
  have ht : 0 < 1 + Real.tanh (0.7978845608028654 * (x + 0.044715 * x ^ 3)) := by
    have := Real.neg_one_lt_tanh (0.7978845608028654 * (x + 0.044715 * x ^ 3)); linarith
  nlinarith [ht, h]

/-- **GELU is sign-preserving.** `0 ≤ x · gelu x` — the output always has the same sign as the input
    (`x·gelu x = ½·x²·(1 + tanh(…)) ≥ 0`), the gated-identity structure of GELU. -/
theorem gelu_mul_self_nonneg (x : ℝ) : 0 ≤ x * gelu x := by
  unfold gelu
  have ht : 0 < 1 + Real.tanh (0.7978845608028654 * (x + 0.044715 * x ^ 3)) := by
    have := Real.neg_one_lt_tanh (0.7978845608028654 * (x + 0.044715 * x ^ 3)); linarith
  nlinarith [ht, sq_nonneg x]

/-- **GELU is magnitude-bounded by its input.** `|gelu x| ≤ |x|` — GELU never amplifies. Writing
    `gelu x = x · (½(1 + tanh(…)))`, the gate `½(1 + tanh(…)) ∈ (0,1)` (since `tanh ∈ (−1,1)`) is a contraction
    factor, so the output magnitude is at most the input's. Combined with `gelu_nonneg_of_nonneg` /
    `gelu_nonpos_of_nonpos`, GELU squeezes each input toward `0` without crossing it: `0 ≤ gelu x ≤ x` for `x ≥ 0`
    and `x ≤ gelu x ≤ 0` for `x ≤ 0`. -/
theorem gelu_abs_le (x : ℝ) : |gelu x| ≤ |x| := by
  unfold gelu
  set t := 0.7978845608028654 * (x + 0.044715 * x ^ 3) with ht
  have h1 : Real.tanh t < 1 := Real.tanh_lt_one t
  have h2 : -1 < Real.tanh t := Real.neg_one_lt_tanh t
  have hcoef : |0.5 * x * (1 + Real.tanh t)| = |x| * (0.5 * (1 + Real.tanh t)) := by
    rw [show 0.5 * x * (1 + Real.tanh t) = x * (0.5 * (1 + Real.tanh t)) by ring, abs_mul,
      abs_of_pos (by linarith : (0:ℝ) < 0.5 * (1 + Real.tanh t))]
  rw [hcoef]
  calc |x| * (0.5 * (1 + Real.tanh t)) ≤ |x| * 1 := by
        apply mul_le_mul_of_nonneg_left _ (abs_nonneg x); linarith
    _ = |x| := mul_one _

/-! ### Softmax (`_softmax_multidiscrete` probabilities) -/

/-- Softmax denominator `Σ e^{lⱼ}`. -/
noncomputable def softmaxDenom {ι : Type*} (s : Finset ι) (l : ι → ℝ) : ℝ :=
  ∑ j ∈ s, Real.exp (l j)

/-- Softmax probability `e^{lᵢ} / Σ e^{lⱼ}`. -/
noncomputable def softmax {ι : Type*} (s : Finset ι) (l : ι → ℝ) (i : ι) : ℝ :=
  Real.exp (l i) / softmaxDenom s l

theorem softmaxDenom_pos {ι : Type*} (s : Finset ι) (l : ι → ℝ) (hs : s.Nonempty) :
    0 < softmaxDenom s l :=
  Finset.sum_pos (fun _ _ => Real.exp_pos _) hs

theorem softmax_pos {ι : Type*} (s : Finset ι) (l : ι → ℝ) (i : ι) (hs : s.Nonempty) :
    0 < softmax s l i :=
  div_pos (Real.exp_pos _) (softmaxDenom_pos s l hs)

/-- **Softmax is a probability distribution**: the outputs sum to 1. -/
theorem softmax_sum_one {ι : Type*} (s : Finset ι) (l : ι → ℝ) (hs : s.Nonempty) :
    ∑ i ∈ s, softmax s l i = 1 := by
  unfold softmax
  rw [← Finset.sum_div]
  exact div_self (ne_of_gt (softmaxDenom_pos s l hs))

/-- **Softmax of constant logits is uniform.** If every logit equals `c`, then `softmax s (const c) i = 1/|s|`
    — equal logits give the uniform (maximum-entropy) policy, independent of `i` and of the constant `c`. -/
theorem softmax_const {ι : Type*} (s : Finset ι) (c : ℝ) (i : ι) (hs : s.Nonempty) :
    softmax s (fun _ => c) i = 1 / s.card := by
  unfold softmax softmaxDenom
  rw [Finset.sum_const, nsmul_eq_mul]
  have hcard : (s.card : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Finset.card_ne_zero.mpr hs)
  field_simp

/-- **The two-class softmax is the sigmoid of the logit gap.** `softmax {i,j} l i = sigmoid(lᵢ − lⱼ)` — over a
    two-element set the softmax collapses to the logistic function of the logit difference
    (`e^{lᵢ}/(e^{lᵢ}+e^{lⱼ}) = 1/(1+e^{-(lᵢ−lⱼ)})`). This is why binary policies/classifiers use a single sigmoid
    output: it IS the 2-class categorical softmax, bridging the discrete (`softmax`) and logistic (`sigmoid`) heads. -/
theorem softmax_pair {ι : Type*} [DecidableEq ι] (i j : ι) (hij : i ≠ j) (l : ι → ℝ) :
    softmax {i, j} l i = sigmoid (l i - l j) := by
  unfold softmax softmaxDenom sigmoid
  rw [Finset.sum_pair hij, neg_sub, Real.exp_sub]
  have hei : (0:ℝ) < Real.exp (l i) := Real.exp_pos _
  have hej : (0:ℝ) < Real.exp (l j) := Real.exp_pos _
  field_simp

/-- The other class of the two-element softmax: `softmax {i,j} l j = sigmoid(lⱼ − lᵢ)` (by `softmax_pair` on the
    swapped pair; consistent with `sigmoid_symm`, the two classes sum to 1). -/
theorem softmax_pair_right {ι : Type*} [DecidableEq ι] (i j : ι) (hij : i ≠ j) (l : ι → ℝ) :
    softmax {i, j} l j = sigmoid (l j - l i) := by
  rw [Finset.pair_comm i j]
  exact softmax_pair j i hij.symm l

/-- The denominator scales by `eᶜ` under a uniform logit shift (`exp(lⱼ + c) = eᶜ·exp lⱼ`). -/
theorem softmaxDenom_shift {ι : Type*} (s : Finset ι) (l : ι → ℝ) (c : ℝ) :
    softmaxDenom s (fun i => l i + c) = Real.exp c * softmaxDenom s l := by
  unfold softmaxDenom
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  rw [Real.exp_add]; ring

/-- **Softmax is shift-invariant.** Adding the same constant `c` to every logit leaves the softmax unchanged:
    `softmax s (l + c) i = softmax s l i` (the `eᶜ` factors cancel between numerator and denominator). This is
    exactly why subtracting the max logit — the numerically-stable softmax the trainer runs — computes the same
    probabilities as the naive definition. -/
theorem softmax_shift {ι : Type*} (s : Finset ι) (l : ι → ℝ) (c : ℝ) (i : ι) :
    softmax s (fun j => l j + c) i = softmax s l i := by
  unfold softmax
  rw [softmaxDenom_shift, Real.exp_add,
    mul_comm (Real.exp (l i)) (Real.exp c),
    mul_div_mul_left _ _ (Real.exp_ne_zero c)]

/-- **Softmax is monotone in the logit.** A higher logit yields a higher probability (same denominator, `exp`
    monotone): `lᵢ ≤ lⱼ → softmax s l i ≤ softmax s l j`. So the policy's action preference respects the logit
    order — the most probable action is the highest-logit one. -/
theorem softmax_le_softmax_of_le {ι : Type*} (s : Finset ι) (l : ι → ℝ) (i j : ι)
    (hs : s.Nonempty) (h : l i ≤ l j) : softmax s l i ≤ softmax s l j := by
  unfold softmax
  have hD : 0 < softmaxDenom s l := softmaxDenom_pos s l hs
  gcongr

/-- **Softmax is strictly monotone in the logit.** `lᵢ < lⱼ → softmax s l i < softmax s l j`. -/
theorem softmax_lt_softmax_of_lt {ι : Type*} (s : Finset ι) (l : ι → ℝ) (i j : ι)
    (hs : s.Nonempty) (h : l i < l j) : softmax s l i < softmax s l j := by
  unfold softmax
  have hD : 0 < softmaxDenom s l := softmaxDenom_pos s l hs
  gcongr

/-- **Softmax order = logit order.** `softmax s l i ≤ softmax s l j ↔ lᵢ ≤ lⱼ` — the probability ordering
    coincides exactly with the logit ordering (so `argmax` of the policy is `argmax` of the logits). -/
theorem softmax_le_softmax_iff {ι : Type*} (s : Finset ι) (l : ι → ℝ) (i j : ι) (hs : s.Nonempty) :
    softmax s l i ≤ softmax s l j ↔ l i ≤ l j := by
  constructor
  · intro h
    by_contra hlt
    push_neg at hlt
    exact absurd h (not_le.mpr (softmax_lt_softmax_of_lt s l j i hs hlt))
  · exact softmax_le_softmax_of_le s l i j hs

/-- **Softmax probability equality ⟺ logit equality.** `softmax s l i = softmax s l j ↔ lᵢ = lⱼ` — two actions
    have the same probability exactly when their logits are equal (the softmax is order-isomorphic to the
    logits; `le_antisymm` of `softmax_le_softmax_iff` both ways). -/
theorem softmax_eq_iff {ι : Type*} (s : Finset ι) (l : ι → ℝ) (i j : ι) (hs : s.Nonempty) :
    softmax s l i = softmax s l j ↔ l i = l j := by
  constructor
  · intro h
    exact le_antisymm ((softmax_le_softmax_iff s l i j hs).mp h.le)
      ((softmax_le_softmax_iff s l j i hs).mp h.ge)
  · intro h; unfold softmax; rw [h]

/-- **Log-softmax = logit − log-partition.** `log softmax(l)ᵢ = lᵢ − log(Σⱼ e^{lⱼ})` — the log-probability is
    the logit minus the log-partition function `log(softmaxDenom)` (the log-sum-exp of the logits). This is the
    identity behind the numerically-stable log-softmax and the log-policy the actor optimizes (`log πᵢ` is affine
    in the logits, offset by the shared normalizer). -/
theorem log_softmax_eq {ι : Type*} (s : Finset ι) (l : ι → ℝ) (i : ι) (hs : s.Nonempty) :
    Real.log (softmax s l i) = l i - Real.log (softmaxDenom s l) := by
  unfold softmax
  rw [Real.log_div (Real.exp_ne_zero _) (ne_of_gt (softmaxDenom_pos s l hs)), Real.log_exp]

/-- **Log-ratio of two softmax entries is the logit gap.** `log softmax(l)ᵢ − log softmax(l)ⱼ = lᵢ − lⱼ` — the
    shared log-partition cancels, so the log-probability *difference* is exactly the logit difference. This is why
    the PPO policy log-ratio `new_logp − old_logp` reduces to a logit gap (and is shift-invariant). -/
theorem log_softmax_sub {ι : Type*} (s : Finset ι) (l : ι → ℝ) (i j : ι) (hs : s.Nonempty) :
    Real.log (softmax s l i) - Real.log (softmax s l j) = l i - l j := by
  rw [log_softmax_eq s l i hs, log_softmax_eq s l j hs]; ring

/-- **Log-softmax (log-policy) is shift-invariant.** `log softmax(l + c) i = log softmax(l) i` — the
    log-probability inherits the softmax's shift-invariance (`softmax_shift`), so the numerically-stable
    max-subtraction leaves the log-policy the trainer optimizes unchanged. -/
theorem log_softmax_shift {ι : Type*} (s : Finset ι) (l : ι → ℝ) (c : ℝ) (i : ι) :
    Real.log (softmax s (fun j => l j + c) i) = Real.log (softmax s l i) := by
  rw [softmax_shift]

/-- **Log-softmax is monotone in the logit.** `lᵢ ≤ lⱼ → log softmax(l) i ≤ log softmax(l) j` — the
    log-probability respects the logit order (`softmax` monotone + `Real.log` monotone on positives). -/
theorem log_softmax_le_of_le {ι : Type*} (s : Finset ι) (l : ι → ℝ) (i j : ι)
    (hs : s.Nonempty) (h : l i ≤ l j) :
    Real.log (softmax s l i) ≤ Real.log (softmax s l j) :=
  Real.log_le_log (softmax_pos s l i hs) (softmax_le_softmax_of_le s l i j hs h)

/-- **Log-softmax is strictly monotone in the logit.** `lᵢ < lⱼ → log softmax(l) i < log softmax(l) j`. -/
theorem log_softmax_lt_of_lt {ι : Type*} (s : Finset ι) (l : ι → ℝ) (i j : ι)
    (hs : s.Nonempty) (h : l i < l j) :
    Real.log (softmax s l i) < Real.log (softmax s l j) :=
  Real.log_lt_log (softmax_pos s l i hs) (softmax_lt_softmax_of_lt s l i j hs h)

/-! ### LayerNorm (`_layernorm`) -/

/-- Batch feature mean. -/
noncomputable def lnMean {ι : Type*} (s : Finset ι) (x : ι → ℝ) : ℝ := (∑ i ∈ s, x i) / s.card

/-- Batch feature variance (population). -/
noncomputable def lnVar {ι : Type*} (s : Finset ι) (x : ι → ℝ) : ℝ :=
  (∑ i ∈ s, (x i - lnMean s x) ^ 2) / s.card

/-- Normalizing denominator `√(var + 1e-5)` (matches the `+1e-5` epsilon in the C). -/
noncomputable def lnDenom {ι : Type*} (s : Finset ι) (x : ι → ℝ) : ℝ :=
  Real.sqrt (lnVar s x + 1e-5)

/-- LayerNorm output `((xᵢ − mean)/denom)·wᵢ + bᵢ`. -/
noncomputable def layerNorm {ι : Type*} (s : Finset ι) (x w b : ι → ℝ) (i : ι) : ℝ :=
  (x i - lnMean s x) / lnDenom s x * w i + b i

theorem lnVar_nonneg {ι : Type*} (s : Finset ι) (x : ι → ℝ) : 0 ≤ lnVar s x :=
  div_nonneg (Finset.sum_nonneg fun _ _ => sq_nonneg _) (Nat.cast_nonneg _)

/-- **LayerNorm variance computational formula (König–Huygens).** `lnVar s x = (Σᵢ xᵢ²)/|s| − (mean)²` — the
    population variance equals the mean of squares minus the square of the mean. This is the classic
    "shortcut"/one-pass identity behind every streaming variance computation (the form the streaming
    LayerNorm/normalization kernels actually use), recasting `lnVar`'s two-pass centered-square definition
    `(Σ(xᵢ−mean)²)/|s|` as a single expression in the raw moments `Σxᵢ²` and `mean`. It is an exact algebraic
    identity, TOTAL in `s` (no nonemptiness needed: on the empty batch both sides collapse to `0`, since `Σ = 0`,
    `card = 0`, and division by zero yields `0` in Lean). Proof: expand `(xᵢ−mean)²`, distribute the sum, collapse
    `Σxᵢ = |s|·mean` (`lnMean`), and `field_simp` over the `card ≠ 0` field. -/
theorem lnVar_eq_mean_sq {ι : Type*} (s : Finset ι) (x : ι → ℝ) :
    lnVar s x = (∑ i ∈ s, (x i) ^ 2) / s.card - (lnMean s x) ^ 2 := by
  rcases s.eq_empty_or_nonempty with rfl | hs
  · simp [lnVar, lnMean]
  · have hc : (s.card : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Finset.card_ne_zero.mpr hs)
    have hS : ∑ i ∈ s, x i = s.card * lnMean s x := by
      rw [lnMean]; field_simp
    have key : ∑ i ∈ s, (x i - lnMean s x) ^ 2
        = (∑ i ∈ s, (x i) ^ 2) - s.card * (lnMean s x) ^ 2 := by
      have hpt : ∀ i ∈ s, (x i - lnMean s x) ^ 2
          = (x i) ^ 2 - 2 * lnMean s x * x i + (lnMean s x) ^ 2 := fun i _ => by ring
      rw [Finset.sum_congr rfl hpt, Finset.sum_add_distrib, Finset.sum_sub_distrib,
          ← Finset.mul_sum, Finset.sum_const, nsmul_eq_mul, hS]
      ring
    rw [lnVar, key]
    field_simp

/-- The `+1e-5` floor makes the LayerNorm denominator strictly positive (total). -/
theorem lnDenom_pos {ι : Type*} (s : Finset ι) (x : ι → ℝ) : 0 < lnDenom s x := by
  unfold lnDenom
  rw [Real.sqrt_pos]
  have h := lnVar_nonneg s x
  have : (0 : ℝ) < 1e-5 := by norm_num
  linarith

/-- **LayerNorm centers the batch**: pre-affine normalized features sum to zero. -/
theorem layerNorm_centered {ι : Type*} (s : Finset ι) (x : ι → ℝ) (hs : s.Nonempty) :
    ∑ i ∈ s, (x i - lnMean s x) / lnDenom s x = 0 := by
  rw [← Finset.sum_div]
  have hc : (s.card : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Finset.card_ne_zero.mpr hs)
  have hnum : ∑ i ∈ s, (x i - lnMean s x) = 0 := by
    unfold lnMean
    rw [Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul]
    have hcancel : (s.card : ℝ) * ((∑ i ∈ s, x i) / s.card) = ∑ i ∈ s, x i := by field_simp
    rw [hcancel, sub_self]
  rw [hnum, zero_div]

/-- **LayerNorm variance is `var/(var+ε)`.** The pre-affine normalized features have sum of squares
    `card · var/(var+1e-5)`: each term is `(xᵢ−mean)²/(var+ε)` (since `lnDenom² = var+ε`), and the sum of squared
    deviations is `card·var`. The scale counterpart to `layerNorm_centered` (zero mean). Unlike exact unit-variance
    normalization (cf. `advNorm_sum_sq_eq_card` with `ε=0`), the `+1e-5` denominator floor makes the realized
    variance `var/(var+ε)` — strictly below 1, approaching 1 as `var ≫ ε`. -/
theorem layerNorm_sum_sq {ι : Type*} (s : Finset ι) (x : ι → ℝ) (hs : s.Nonempty) :
    ∑ i ∈ s, ((x i - lnMean s x) / lnDenom s x) ^ 2
      = s.card * lnVar s x / (lnVar s x + 1e-5) := by
  have hv := lnVar_nonneg s x
  have hden : lnDenom s x ^ 2 = lnVar s x + 1e-5 := by
    unfold lnDenom; rw [Real.sq_sqrt (by linarith)]
  have hcard : (s.card : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Finset.card_ne_zero.mpr hs)
  have hsumsq : ∑ i ∈ s, (x i - lnMean s x) ^ 2 = s.card * lnVar s x := by
    unfold lnVar; field_simp
  calc ∑ i ∈ s, ((x i - lnMean s x) / lnDenom s x) ^ 2
      = ∑ i ∈ s, (x i - lnMean s x) ^ 2 / (lnVar s x + 1e-5) := by
        apply Finset.sum_congr rfl; intro i _; rw [div_pow, hden]
    _ = (∑ i ∈ s, (x i - lnMean s x) ^ 2) / (lnVar s x + 1e-5) := by rw [← Finset.sum_div]
    _ = s.card * lnVar s x / (lnVar s x + 1e-5) := by rw [hsumsq]

/-- **LayerNorm never over-normalizes.** The pre-affine normalized features have sum of squares `≤ card`, i.e.
    realized variance `≤ 1` — the `+1e-5` denominator floor means LayerNorm's output variance can only fall short
    of unit, never exceed it (`var/(var+ε) ≤ 1` for `var ≥ 0`, `ε > 0`). -/
theorem layerNorm_sum_sq_le_card {ι : Type*} (s : Finset ι) (x : ι → ℝ) (hs : s.Nonempty) :
    ∑ i ∈ s, ((x i - lnMean s x) / lnDenom s x) ^ 2 ≤ s.card := by
  rw [layerNorm_sum_sq s x hs]
  have hv := lnVar_nonneg s x
  have hpos : (0 : ℝ) < lnVar s x + 1e-5 := by linarith
  rw [div_le_iff₀ hpos]
  have hc : (0 : ℝ) ≤ s.card := Nat.cast_nonneg _
  nlinarith [hc, hv]

/-! ### LayerNorm input shift-invariance -/

/-- The batch mean shifts by `c` under a uniform input shift: `mean(x + c) = mean(x) + c`. -/
theorem lnMean_shift {ι : Type*} (s : Finset ι) (x : ι → ℝ) (c : ℝ) (hs : s.Nonempty) :
    lnMean s (fun i => x i + c) = lnMean s x + c := by
  unfold lnMean
  rw [Finset.sum_add_distrib, Finset.sum_const, nsmul_eq_mul]
  have hc : (s.card : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Finset.card_ne_zero.mpr hs)
  field_simp

/-- The batch variance is invariant to a uniform input shift (the deviations from the mean are unchanged). -/
theorem lnVar_shift {ι : Type*} (s : Finset ι) (x : ι → ℝ) (c : ℝ) (hs : s.Nonempty) :
    lnVar s (fun i => x i + c) = lnVar s x := by
  unfold lnVar
  rw [lnMean_shift s x c hs]
  congr 1
  apply Finset.sum_congr rfl
  intro i _
  congr 1
  ring

/-- The normalizing denominator is invariant to a uniform input shift (`lnVar_shift`). -/
theorem lnDenom_shift {ι : Type*} (s : Finset ι) (x : ι → ℝ) (c : ℝ) (hs : s.Nonempty) :
    lnDenom s (fun i => x i + c) = lnDenom s x := by
  unfold lnDenom; rw [lnVar_shift s x c hs]

/-- **LayerNorm is input-shift-invariant.** Adding the same constant `c` to every input feature leaves the
    LayerNorm output unchanged: `layerNorm s (x + c) w b i = layerNorm s x w b i` — the mean shifts by `c` (so
    `xᵢ − mean` is unchanged) and the denominator is invariant. A structural invariance of the normalization,
    independent of the affine `w`/`b`. -/
theorem layerNorm_shift {ι : Type*} (s : Finset ι) (x w b : ι → ℝ) (c : ℝ) (i : ι) (hs : s.Nonempty) :
    layerNorm s (fun k => x k + c) w b i = layerNorm s x w b i := by
  unfold layerNorm
  rw [lnMean_shift s x c hs, lnDenom_shift s x c hs]
  have : x i + c - (lnMean s x + c) = x i - lnMean s x := by ring
  rw [this]

end Puffer.Net
