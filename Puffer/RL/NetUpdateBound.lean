/-
The WHOLE-NET vector update bound — lifting the per-weight step bound to the actual parameter arrays.

`UpdateADBound.axpyStep_AD_error` bounds ONE weight's SGD step. The trainer updates a whole parameter
array at once via `NNTrain.vecAxpy`/`matAxpy` (`updateMLP`/`updateMLPAD`'s final line:
`matAxpy lr p.W1 gW1`, `vecAxpy lr p.b1 gb1`, …). Since

    vecAxpy s a b = (Array.range a.size).map (fun i => a[i]! + s · b[i]!)

every component of the update IS an `axpyStep` (`aᵢ + s·bᵢ`), so the per-weight bound lifts componentwise
to the full array with no new arithmetic:

  • `vecAxpy_entrywise_error` / `matAxpy_entrywise_error` — the GENERAL vector/matrix bound, parameterized
    by the ideal per-component gradient `Rg` and its error `εg` (whatever certifies them). Every entry of the
    updated array is within the per-weight bound of the ideal real ascent step.
  • `vecAxpy_AD_error` / `matAxpy_AD_error` — the AD instantiation: with a gradient array holding the
    forward-mode-AD gradients (`g[i]! = dF e σ i`, `hgrad`), every weight's update is within
    `u64·|…| + u64·|lr·g| + |lr|·derivErrBnd e σ i` of the ideal ascent against the TRUE per-weight derivative
    `∂(evalR e)/∂(var i)`. The matrix form threads an explicit flattening `vidx i j` (which objective variable
    the weight at `(i,j)` is).

This is the `∀`-over-all-weights (ℓ∞ / entrywise) form of the composed update bound: it bounds the LITERAL
`vecAxpy`/`matAxpy` output the trainer runs, not a single component. Axiom-clean beyond the Float base.
An aggregate scalar (uniform-`B` sup, or Frobenius `√·`) follows by bounding each component's magnitude.
-/
import Puffer.RL.NNTrain
import Puffer.RL.UpdateADBound

namespace Puffer.RL.NetUpdateBound

open Puffer.FloatR
open Puffer.FloatR.ADR (Expr evalR envR derivErrBnd dF WD PosR)
open Puffer.RL.NNTrain (vecAxpy matAxpy)

/-! ### Structural: `vecAxpy`/`matAxpy` read off as an `axpy` step per component -/

theorem vecAxpy_size (s : Float) (a b : Array Float) : (vecAxpy s a b).size = a.size := by
  unfold vecAxpy; rw [Array.size_map, Array.size_range]

/-- In range, `vecAxpy s a b` reads off the scalar `axpy` step `aᵢ + s·bᵢ`. -/
theorem vecAxpy_getElem (s : Float) (a b : Array Float) (i : Nat) (hi : i < a.size) :
    (vecAxpy s a b)[i]! = a[i]! + s * b[i]! := by
  unfold vecAxpy
  rw [Array.getElem!_eq_getD, Array.getD]
  simp [hi, Array.getElem_map, Array.getElem_range, Array.size_map, Array.size_range]

/-- **Per-step weight displacement bound (trust region of one SGD `vecAxpy` step).** The LITERAL update
    `vecAxpy lr p g` moves weight `i` away from its old value `p[i]!` by at most the exact gradient step
    `|lr·gᵢ|` (inflated by two unit-roundoff factors, one for the multiply and one for the add) plus a
    relative-roundoff floor `u64·|pᵢ|`:
      `|toReal (vecAxpy lr p g)ᵢ − toReal pᵢ| ≤ u64·|toReal pᵢ| + (1+u64)²·|toReal lr · toReal gᵢ|`.
    Unlike every other `NetUpdateBound` theorem (which bounds the distance from the *ideal ascent target*
    `pᵢ + lr·Rgᵢ`), this bounds the distance from the *input* `pᵢ`, so it needs no gradient-error data at all
    — it is the one-step parameter-space trust region of the executed optimizer: a weight with a tiny gradient
    barely moves (`gᵢ = 0 ⟹` moves ≤ `u64·|pᵢ|`), and no weight can jump more than essentially its own
    gradient step. -/
theorem vecAxpy_step_size (lr : Float) (p g : Array Float) (i : Nat) (hi : i < p.size) :
    |toReal ((vecAxpy lr p g)[i]!) - toReal (p[i]!)|
      ≤ u64 * |toReal (p[i]!)| + (1 + u64) ^ 2 * |toReal lr * toReal (g[i]!)| := by
  rw [vecAxpy_getElem lr p g i hi]
  obtain ⟨δ, hδ, he⟩ := add_model (p[i]!) (lr * g[i]!)
  have h1δ : |1 + δ| ≤ 1 + u64 :=
    (abs_add_le _ _).trans (by simpa using add_le_add_left hδ 1)
  have key : toReal (p[i]! + lr * g[i]!) - toReal (p[i]!)
      = toReal (p[i]!) * δ + toReal (lr * g[i]!) * (1 + δ) := by rw [he]; ring
  rw [key]
  calc |toReal (p[i]!) * δ + toReal (lr * g[i]!) * (1 + δ)|
      ≤ |toReal (p[i]!) * δ| + |toReal (lr * g[i]!) * (1 + δ)| := abs_add_le _ _
    _ = |toReal (p[i]!)| * |δ| + |toReal (lr * g[i]!)| * |1 + δ| := by rw [abs_mul, abs_mul]
    _ ≤ |toReal (p[i]!)| * u64
          + (1 + u64) * |toReal lr * toReal (g[i]!)| * (1 + u64) := by
        refine add_le_add (mul_le_mul_of_nonneg_left hδ (abs_nonneg _)) ?_
        exact mul_le_mul (mul_abs_le lr (g[i]!)) h1δ (abs_nonneg _)
          (mul_nonneg (by have := u64_pos.le; linarith) (abs_nonneg _))
    _ = u64 * |toReal (p[i]!)| + (1 + u64) ^ 2 * |toReal lr * toReal (g[i]!)| := by ring

theorem matAxpy_size (s : Float) (a b : Array (Array Float)) : (matAxpy s a b).size = a.size := by
  unfold matAxpy; rw [Array.size_map, Array.size_range]

/-- In range, `matAxpy s a b` reads off row `i` as `vecAxpy s aᵢ bᵢ`. -/
theorem matAxpy_getElem (s : Float) (a b : Array (Array Float)) (i : Nat) (hi : i < a.size) :
    (matAxpy s a b)[i]! = vecAxpy s (a[i]!) (b[i]!) := by
  unfold matAxpy
  rw [Array.getElem!_eq_getD, Array.getD]
  simp [hi, Array.getElem_map, Array.getElem_range, Array.size_map, Array.size_range]

/-! ### General whole-array update bound (parameterized by the ideal gradient `Rg` and error `εg`) -/

/-- **Whole-vector update bound.** Every entry of `vecAxpy lr p g` is within the per-weight `axpyStep`
    bound of the ideal real ascent step `toReal pᵢ + toReal lr · Rg i`, given a per-component gradient
    error `|toReal gᵢ − Rg i| ≤ εg i`. The ℓ∞/entrywise form of the composed update over the whole vector. -/
theorem vecAxpy_entrywise_error (lr : Float) (p g : Array Float) (Rg εg : Nat → ℝ)
    (hg : ∀ i, i < p.size → |toReal (g[i]!) - Rg i| ≤ εg i)
    (i : Nat) (hi : i < p.size) :
    |toReal ((vecAxpy lr p g)[i]!) - (toReal (p[i]!) + toReal lr * Rg i)|
      ≤ u64 * |toReal (p[i]!) + toReal (lr * g[i]!)|
          + (u64 * |toReal lr * toReal (g[i]!)| + |toReal lr| * εg i) := by
  rw [vecAxpy_getElem lr p g i hi]
  exact axpyStep_error (p[i]!) lr (g[i]!) (Rg i) (εg i) (hg i hi)

/-- **Whole-matrix update bound.** Every entry of `matAxpy lr p g` is within the per-weight `axpyStep`
    bound of the ideal real ascent step. Reduces row-wise to `vecAxpy_entrywise_error`. -/
theorem matAxpy_entrywise_error (lr : Float) (p g : Array (Array Float)) (Rg εg : Nat → Nat → ℝ)
    (hg : ∀ i j, i < p.size → j < (p[i]!).size → |toReal ((g[i]!)[j]!) - Rg i j| ≤ εg i j)
    (i j : Nat) (hi : i < p.size) (hj : j < (p[i]!).size) :
    |toReal (((matAxpy lr p g)[i]!)[j]!) - (toReal ((p[i]!)[j]!) + toReal lr * Rg i j)|
      ≤ u64 * |toReal ((p[i]!)[j]!) + toReal (lr * (g[i]!)[j]!)|
          + (u64 * |toReal lr * toReal ((g[i]!)[j]!)| + |toReal lr| * εg i j) := by
  rw [matAxpy_getElem lr p g i hi]
  exact vecAxpy_entrywise_error lr (p[i]!) (g[i]!) (Rg i) (εg i)
    (fun j hj => hg i j hi hj) j hj

/-! ### Forward-mode AD instantiation: the whole update against the TRUE per-weight derivatives -/

/-- **Whole-vector AD update bound.** If the gradient array holds the forward-mode-AD gradients
    (`g[i]! = dF e σ i` for the scalar objective `e` whose variables are the weight indices), then every
    weight's SGD step is within `u64·|…| + u64·|lr·g| + |lr|·derivErrBnd e σ i` of the ideal real ascent
    against the true derivative `∂(evalR e)/∂(var i)`. Discharges `εg` for the whole vector at once. -/
theorem vecAxpy_AD_error (e : Expr) (σ : Nat → Float) (lr : Float) (p g : Array Float)
    (hwd : WD e σ) (hp : PosR e (envR σ)) (hgrad : ∀ i, i < p.size → g[i]! = dF e σ i)
    (i : Nat) (hi : i < p.size) :
    |toReal ((vecAxpy lr p g)[i]!)
        - (toReal (p[i]!) + toReal lr
            * deriv (fun t => evalR e (Function.update (envR σ) i t)) (envR σ i))|
      ≤ u64 * |toReal (p[i]!) + toReal (lr * g[i]!)|
          + (u64 * |toReal lr * toReal (g[i]!)| + |toReal lr| * derivErrBnd e σ i) := by
  rw [vecAxpy_getElem lr p g i hi, hgrad i hi]
  exact Puffer.FloatR.ADR.axpyStep_AD_error e σ i (p[i]!) lr hwd hp

/-- **Whole-matrix AD update bound.** As `vecAxpy_AD_error`, with an explicit flattening `vidx i j` naming
    which objective variable the weight at row `i`, column `j` is; the gradient entry holds that variable's
    AD derivative (`hgrad`). Every matrix weight's step is within the proven bound of the ideal ascent
    against `∂(evalR e)/∂(var (vidx i j))`. -/
theorem matAxpy_AD_error (e : Expr) (σ : Nat → Float) (lr : Float) (p g : Array (Array Float))
    (vidx : Nat → Nat → Nat) (hwd : WD e σ) (hp : PosR e (envR σ))
    (hgrad : ∀ i j, i < p.size → j < (p[i]!).size → (g[i]!)[j]! = dF e σ (vidx i j))
    (i j : Nat) (hi : i < p.size) (hj : j < (p[i]!).size) :
    |toReal (((matAxpy lr p g)[i]!)[j]!)
        - (toReal ((p[i]!)[j]!) + toReal lr
            * deriv (fun t => evalR e (Function.update (envR σ) (vidx i j) t)) (envR σ (vidx i j)))|
      ≤ u64 * |toReal ((p[i]!)[j]!) + toReal (lr * (g[i]!)[j]!)|
          + (u64 * |toReal lr * toReal ((g[i]!)[j]!)| + |toReal lr| * derivErrBnd e σ (vidx i j)) := by
  rw [matAxpy_getElem lr p g i hi, vecAxpy_getElem lr (p[i]!) (g[i]!) j hj, hgrad i j hi hj]
  exact Puffer.FloatR.ADR.axpyStep_AD_error e σ (vidx i j) ((p[i]!)[j]!) lr hwd hp

/-! ### Aggregate scalar: from the entrywise `∀`-bound to one sup / Frobenius number

The entrywise theorems bound each component separately. With uniform handles on the three
component-dependent magnitudes (`Bmag`, `Bprod`, `Bε`) they collapse to a single scalar bound
`B = u64·Bmag + (u64·Bprod + |lr|·Bε)` — the ℓ∞ (sup) norm of the whole-vector update error — and, summing
the squares, the Frobenius bound `∑ (Δᵢ)² ≤ n·B²` (so `‖update − ideal‖₂ ≤ √n·B`). -/

/-- **Sup (ℓ∞) whole-vector bound.** Uniform handles `Bmag`/`Bprod`/`Bε` on the three per-component
    magnitudes collapse `vecAxpy_entrywise_error` to a single scalar `u64·Bmag + (u64·Bprod + |lr|·Bε)`
    bounding *every* component of the update error. -/
theorem vecAxpy_sup_error (lr : Float) (p g : Array Float) (Rg εg : Nat → ℝ)
    (hg : ∀ i, i < p.size → |toReal (g[i]!) - Rg i| ≤ εg i)
    (Bmag Bprod Bε : ℝ)
    (hmag : ∀ i, i < p.size → |toReal (p[i]!) + toReal (lr * g[i]!)| ≤ Bmag)
    (hprod : ∀ i, i < p.size → |toReal lr * toReal (g[i]!)| ≤ Bprod)
    (hεb : ∀ i, i < p.size → εg i ≤ Bε)
    (i : Nat) (hi : i < p.size) :
    |toReal ((vecAxpy lr p g)[i]!) - (toReal (p[i]!) + toReal lr * Rg i)|
      ≤ u64 * Bmag + (u64 * Bprod + |toReal lr| * Bε) := by
  refine (vecAxpy_entrywise_error lr p g Rg εg hg i hi).trans ?_
  have hu : (0:ℝ) ≤ u64 := le_of_lt u64_pos
  exact add_le_add (mul_le_mul_of_nonneg_left (hmag i hi) hu)
    (add_le_add (mul_le_mul_of_nonneg_left (hprod i hi) hu)
      (mul_le_mul_of_nonneg_left (hεb i hi) (abs_nonneg _)))

/-- Pure aggregation: entrywise `|f i| ≤ B` gives `∑ (f i)² ≤ n·B²` (so the ℓ₂ norm is `≤ √n·B`). -/
theorem frob_sum_le (f : Nat → ℝ) (n : Nat) (B : ℝ) (h : ∀ i, i < n → |f i| ≤ B) :
    ∑ i ∈ Finset.range n, (f i) ^ 2 ≤ (n : ℝ) * B ^ 2 := by
  have hle : ∑ i ∈ Finset.range n, (f i) ^ 2 ≤ ∑ i ∈ Finset.range n, B ^ 2 := by
    apply Finset.sum_le_sum
    intro i hi
    rw [Finset.mem_range] at hi
    nlinarith [h i hi, le_abs_self (f i), neg_abs_le (f i), abs_nonneg (f i)]
  rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul] at hle
  exact hle

/-- **Frobenius (ℓ₂) whole-vector bound.** The sum of squared component errors of the update is within
    `p.size · B²` for the uniform per-component bound `B = u64·Bmag + (u64·Bprod + |lr|·Bε)`; equivalently
    `‖update − ideal‖₂ ≤ √(p.size)·B`. -/
theorem vecAxpy_frob_error (lr : Float) (p g : Array Float) (Rg εg : Nat → ℝ)
    (hg : ∀ i, i < p.size → |toReal (g[i]!) - Rg i| ≤ εg i)
    (Bmag Bprod Bε : ℝ)
    (hmag : ∀ i, i < p.size → |toReal (p[i]!) + toReal (lr * g[i]!)| ≤ Bmag)
    (hprod : ∀ i, i < p.size → |toReal lr * toReal (g[i]!)| ≤ Bprod)
    (hεb : ∀ i, i < p.size → εg i ≤ Bε) :
    ∑ i ∈ Finset.range p.size,
        (toReal ((vecAxpy lr p g)[i]!) - (toReal (p[i]!) + toReal lr * Rg i)) ^ 2
      ≤ (p.size : ℝ) * (u64 * Bmag + (u64 * Bprod + |toReal lr| * Bε)) ^ 2 :=
  frob_sum_le _ p.size _ (fun i hi =>
    vecAxpy_sup_error lr p g Rg εg hg Bmag Bprod Bε hmag hprod hεb i hi)

/-- **Sup (ℓ∞) whole-vector AD bound.** The AD instantiation of `vecAxpy_sup_error`: with the gradient
    array holding the forward-mode AD gradients, every weight's update is within the single scalar
    `u64·Bmag + (u64·Bprod + |lr|·Bε)` of the ideal ascent against the true per-weight derivative. -/
theorem vecAxpy_AD_sup_error (e : Expr) (σ : Nat → Float) (lr : Float) (p g : Array Float)
    (hwd : WD e σ) (hp : PosR e (envR σ)) (hgrad : ∀ i, i < p.size → g[i]! = dF e σ i)
    (Bmag Bprod Bε : ℝ)
    (hmag : ∀ i, i < p.size → |toReal (p[i]!) + toReal (lr * g[i]!)| ≤ Bmag)
    (hprod : ∀ i, i < p.size → |toReal lr * toReal (g[i]!)| ≤ Bprod)
    (hεb : ∀ i, i < p.size → derivErrBnd e σ i ≤ Bε)
    (i : Nat) (hi : i < p.size) :
    |toReal ((vecAxpy lr p g)[i]!)
        - (toReal (p[i]!) + toReal lr
            * deriv (fun t => evalR e (Function.update (envR σ) i t)) (envR σ i))|
      ≤ u64 * Bmag + (u64 * Bprod + |toReal lr| * Bε) := by
  refine (vecAxpy_AD_error e σ lr p g hwd hp hgrad i hi).trans ?_
  have hu : (0:ℝ) ≤ u64 := le_of_lt u64_pos
  exact add_le_add (mul_le_mul_of_nonneg_left (hmag i hi) hu)
    (add_le_add (mul_le_mul_of_nonneg_left (hprod i hi) hu)
      (mul_le_mul_of_nonneg_left (hεb i hi) (abs_nonneg _)))

/-- **Sup (ℓ∞) whole-matrix bound.** As `vecAxpy_sup_error`, over every matrix entry. -/
theorem matAxpy_sup_error (lr : Float) (p g : Array (Array Float)) (Rg εg : Nat → Nat → ℝ)
    (hg : ∀ i j, i < p.size → j < (p[i]!).size → |toReal ((g[i]!)[j]!) - Rg i j| ≤ εg i j)
    (Bmag Bprod Bε : ℝ)
    (hmag : ∀ i j, i < p.size → j < (p[i]!).size →
      |toReal ((p[i]!)[j]!) + toReal (lr * (g[i]!)[j]!)| ≤ Bmag)
    (hprod : ∀ i j, i < p.size → j < (p[i]!).size → |toReal lr * toReal ((g[i]!)[j]!)| ≤ Bprod)
    (hεb : ∀ i j, i < p.size → j < (p[i]!).size → εg i j ≤ Bε)
    (i j : Nat) (hi : i < p.size) (hj : j < (p[i]!).size) :
    |toReal (((matAxpy lr p g)[i]!)[j]!) - (toReal ((p[i]!)[j]!) + toReal lr * Rg i j)|
      ≤ u64 * Bmag + (u64 * Bprod + |toReal lr| * Bε) := by
  refine (matAxpy_entrywise_error lr p g Rg εg hg i j hi hj).trans ?_
  have hu : (0:ℝ) ≤ u64 := le_of_lt u64_pos
  exact add_le_add (mul_le_mul_of_nonneg_left (hmag i j hi hj) hu)
    (add_le_add (mul_le_mul_of_nonneg_left (hprod i j hi hj) hu)
      (mul_le_mul_of_nonneg_left (hεb i j hi hj) (abs_nonneg _)))

end Puffer.RL.NetUpdateBound
