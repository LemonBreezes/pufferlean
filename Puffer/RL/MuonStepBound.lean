/-
Operator-norm bound for one full Muon optimizer step (`stepMat`: Nesterov momentum → Newton–Schulz
orthogonalize → decoupled weight decay). The step's new weight is

    W' = (1 − lr·wd)·W + (lr·scale)·ortho,   ortho = newtonSchulz update,   update = grad + μ·(μ·mom + grad)

(a `matLin` = 2-term linear combination). So its operator norm is a triangle inequality on top of the
already-closed Newton–Schulz bound: `‖W'‖ ≤ |1−lr·wd|·‖W‖ + |lr·scale|·‖ortho‖ + (matLin rounding)`, and
`‖ortho‖ ≤ √1.3131 + rounding` (`NewtonSchulzRunnable.newtonSchulz_opNorm`).

  • `matLin_getElem`   : `(matLin a X b Y)[i][j] = a·X[i][j] + b·Y[i][j]`.
  • `matLin_opNorm_le` : the reusable 2-term bound — `‖toMatrixF (matLin a X b Y)‖ ≤ |toReal a|·‖toMatrixF X‖
      + |toReal b|·‖toMatrixF Y‖ + √(r·c)·E`, `E` a uniform per-entry rounding bound (`matLinEntry_error` +
      `l2_opNorm_le_of_entrywise`, same shape as `fold_opNorm_le`).
  • `stepMat_opNorm_le`: the Muon weight-update norm-growth bound — `‖toMatrixF (stepMat …).1‖ ≤
      |toReal(1−lr·wd)|·‖toMatrixF W‖ + |toReal(lr·scale)|·B + √(r·c)·E` for any `B` bounding `‖toMatrixF
      ortho‖` (`B = √1.3131 + rounding` from the Newton–Schulz capstone). `stepMat.1` is DEFEQ to the `matLin`.

Axiom-clean modulo the trusted Float base. This closes the optimizer step on top of the orthogonalization:
one Muon update grows the weight operator norm by at most `|1−lr·wd|·‖W‖` (weight-decay contraction) plus an
O(1) orthogonalized term — the contraction/growth recurrence for the trained weights.
-/
import Mathlib
import Puffer.RL.NewtonSchulzFloat
import Puffer.RL.MuonMatrixRuntime

namespace Puffer.RL.MuonStepBound

open scoped Matrix Matrix.Norms.L2Operator BigOperators
open Matrix
open Puffer.FloatR (toReal u64)
open Puffer.FloatR.Muon (Mat matLin stepMat newtonSchulz stepVec)
open Puffer.FloatR (add_model mul_model u64_pos)
open Puffer.RL.MatrixEmbed (toMatrixF)
open Puffer.RL.NewtonSchulzFloat (l2_opNorm_le_of_entrywise)
open Puffer.RL.MuonMatrixRuntime (matLinEntryErrBnd matLinEntry_error)

/-- `(matLin a X b Y)[i][j] = a·X[i][j] + b·Y[i][j]`. -/
theorem matLin_getElem (a : Float) (X : Mat) (b : Float) (Y : Mat) (i j : Nat)
    (hi : i < X.size) (hj : j < (X[i]!).size) :
    ((matLin a X b Y)[i]!)[j]! = a * (X[i]!)[j]! + b * (Y[i]!)[j]! := by
  unfold matLin
  rw [getElem!_pos _ i (by simpa using hi), Array.getElem_map, Array.getElem_range,
    getElem!_pos _ j (by simpa using hj), Array.getElem_map, Array.getElem_range]

/-- **`matLin` operator-norm bound.** `‖toMatrixF (matLin a X b Y)‖ ≤ |toReal a|·‖toMatrixF X‖ +
    |toReal b|·‖toMatrixF Y‖ + √(r·c)·E`, given a uniform per-entry rounding bound `E`. Triangle inequality
    against the exact `toReal a • toMatrixF X + toReal b • toMatrixF Y`, difference bounded entrywise by
    `matLinEntry_error` and carried into the operator norm by `l2_opNorm_le_of_entrywise`. -/
theorem matLin_opNorm_le {r c : Nat} [Nonempty (Fin c)] (a : Float) (X : Mat) (b : Float) (Y : Mat) (E : ℝ)
    (hE : 0 ≤ E)
    (hXsz : X.size = r) (hXrow : ∀ i, i < r → (X[i]!).size = c)
    (hentry : ∀ (i : Fin r) (j : Fin c),
      matLinEntryErrBnd a ((X[i.1]!)[j.1]!) b ((Y[i.1]!)[j.1]!) 0 0 ≤ E) :
    ‖toMatrixF r c (matLin a X b Y)‖
      ≤ |toReal a| * ‖toMatrixF r c X‖ + |toReal b| * ‖toMatrixF r c Y‖ + Real.sqrt ((r : ℝ) * c) * E := by
  set N : Matrix (Fin r) (Fin c) ℝ := toReal a • toMatrixF r c X + toReal b • toMatrixF r c Y with hN
  have hdiff : ‖toMatrixF r c (matLin a X b Y) - N‖ ≤ Real.sqrt ((r : ℝ) * c) * E := by
    refine l2_opNorm_le_of_entrywise _ E hE (fun i j => ?_)
    rw [Matrix.sub_apply]
    have hi : (i : Nat) < X.size := by rw [hXsz]; exact i.2
    have hj : (j : Nat) < (X[i.1]!).size := by rw [hXrow i.1 (by rw [hXsz] at hi; exact hi)]; exact j.2
    have hMLval : toMatrixF r c (matLin a X b Y) i j
        = toReal (a * (X[i.1]!)[j.1]! + b * (Y[i.1]!)[j.1]!) := by
      simp only [toMatrixF, Matrix.of_apply]; rw [matLin_getElem a X b Y i.1 j.1 hi hj]
    have hNval : N i j = toReal a * toReal ((X[i.1]!)[j.1]!) + toReal b * toReal ((Y[i.1]!)[j.1]!) := by
      simp only [hN, Matrix.add_apply, Matrix.smul_apply, toMatrixF, Matrix.of_apply, smul_eq_mul]
    rw [hMLval, hNval]
    exact le_trans (matLinEntry_error a _ b _ _ _ 0 0 (by simp) (by simp)) (hentry i j)
  have hNbnd : ‖N‖ ≤ |toReal a| * ‖toMatrixF r c X‖ + |toReal b| * ‖toMatrixF r c Y‖ := by
    calc ‖N‖ ≤ ‖toReal a • toMatrixF r c X‖ + ‖toReal b • toMatrixF r c Y‖ := norm_add_le _ _
      _ = |toReal a| * ‖toMatrixF r c X‖ + |toReal b| * ‖toMatrixF r c Y‖ := by
          rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs]
  calc ‖toMatrixF r c (matLin a X b Y)‖
      = ‖N + (toMatrixF r c (matLin a X b Y) - N)‖ := by rw [add_sub_cancel]
    _ ≤ ‖N‖ + ‖toMatrixF r c (matLin a X b Y) - N‖ := norm_add_le _ _
    _ ≤ (|toReal a| * ‖toMatrixF r c X‖ + |toReal b| * ‖toMatrixF r c Y‖) + Real.sqrt ((r : ℝ) * c) * E :=
        add_le_add hNbnd hdiff
    _ = _ := by ring

/-- **Muon weight-update operator-norm bound.** One full `stepMat` (Nesterov + Newton–Schulz + weight decay)
    grows the weight operator norm by at most `|toReal(1−lr·wd)|·‖toMatrixF W‖ + |toReal(lr·scale)|·B +
    √(r·c)·E`, where `B` bounds the orthogonalized update's norm `‖toMatrixF ortho‖` (`B = √1.3131 + rounding`
    from `NewtonSchulzRunnable.newtonSchulz_opNorm`) and `E` is the final `matLin`'s per-entry rounding. The
    `|toReal(1−lr·wd)|` factor is the weight-decay contraction; the O(1) `B` term is the orthogonalized step.
    `stepMat.1` is defeq to `matLin (1−lr·wd) W (lr·scale) ortho`. -/
theorem stepMat_opNorm_le {r c : Nat} [Nonempty (Fin c)] (W grad mom : Mat) (lr wd mu eps : Float)
    (E B : ℝ) (hE : 0 ≤ E)
    (hWsz : W.size = r) (hWrow : ∀ i, i < r → (W[i]!).size = c)
    (hentry : ∀ (i : Fin r) (j : Fin c),
      matLinEntryErrBnd (1.0 - lr * wd) ((W[i.1]!)[j.1]!)
        (lr * Float.sqrt (max 1.0 (Float.ofNat W.size / Float.ofNat (W[0]!).size)))
        (((newtonSchulz (matLin 1.0 grad mu (matLin mu mom 1.0 grad)) eps)[i.1]!)[j.1]!) 0 0 ≤ E)
    (hortho : ‖toMatrixF r c (newtonSchulz (matLin 1.0 grad mu (matLin mu mom 1.0 grad)) eps)‖ ≤ B) :
    ‖toMatrixF r c (stepMat W grad mom lr wd mu eps).1‖
      ≤ |toReal (1.0 - lr * wd)| * ‖toMatrixF r c W‖
        + |toReal (lr * Float.sqrt (max 1.0 (Float.ofNat W.size / Float.ofNat (W[0]!).size)))| * B
        + Real.sqrt ((r : ℝ) * c) * E := by
  have h := matLin_opNorm_le (1.0 - lr * wd) W
    (lr * Float.sqrt (max 1.0 (Float.ofNat W.size / Float.ofNat (W[0]!).size)))
    (newtonSchulz (matLin 1.0 grad mu (matLin mu mom 1.0 grad)) eps) E hE hWsz hWrow hentry
  have hb := mul_le_mul_of_nonneg_left hortho
    (abs_nonneg (toReal (lr * Float.sqrt (max 1.0 (Float.ofNat W.size / Float.ofNat (W[0]!).size)))))
  calc ‖toMatrixF r c (stepMat W grad mom lr wd mu eps).1‖
      = ‖toMatrixF r c (matLin (1.0 - lr * wd) W
          (lr * Float.sqrt (max 1.0 (Float.ofNat W.size / Float.ofNat (W[0]!).size)))
          (newtonSchulz (matLin 1.0 grad mu (matLin mu mom 1.0 grad)) eps))‖ := rfl
    _ ≤ _ := h
    _ ≤ _ := by linarith [hb]

/-! ### The 1D bias update (`stepVec`): Nesterov + weight decay, no orthogonalization

`stepVec` produces `nb[i] = b[i]·(1−lr·wd) + lr·upd[i]` per component (`upd[i] = grad[i] + μ·(μ·mom[i] +
grad[i])`) — a plain affine step, no Newton–Schulz. Its per-entry magnitude growth is bounded directly. -/

/-- `((Array.range n).map f)[i]! = f i` for `i < n`. -/
theorem range_map_get {α} [Inhabited α] (n : Nat) (f : Nat → α) (i : Nat) (hi : i < n) :
    ((Array.range n).map f)[i]! = f i := by
  rw [getElem!_pos _ i (by simpa using hi), Array.getElem_map, Array.getElem_range]

/-- `|toReal (x·a + c·y)| ≤ (1+u64)²·(|toReal a|·|toReal x| + |toReal c|·|toReal y|)` — the abs bound for a
    Float 2-term affine combination (two `mul` roundings inside one `add` rounding). -/
theorem addmul_abs_le (x a c y : Float) :
    |toReal (x * a + c * y)| ≤ (1 + u64) ^ 2 * (|toReal a| * |toReal x| + |toReal c| * |toReal y|) := by
  obtain ⟨δ0, hδ0, he0⟩ := add_model (x * a) (c * y)
  obtain ⟨δ1, hδ1, he1⟩ := mul_model x a
  obtain ⟨δ2, hδ2, he2⟩ := mul_model c y
  have habs : ∀ δ : ℝ, |δ| ≤ u64 → |1 + δ| ≤ 1 + u64 := fun δ hδ => by
    calc |1 + δ| ≤ |(1 : ℝ)| + |δ| := abs_add_le _ _
      _ = 1 + |δ| := by rw [abs_one]
      _ ≤ 1 + u64 := by linarith
  have hxa : |toReal (x * a)| ≤ (1 + u64) * (|toReal a| * |toReal x|) := by
    rw [he1, abs_mul, abs_mul]
    calc |toReal x| * |toReal a| * |1 + δ1| ≤ |toReal x| * |toReal a| * (1 + u64) :=
          mul_le_mul_of_nonneg_left (habs δ1 hδ1) (by positivity)
      _ = (1 + u64) * (|toReal a| * |toReal x|) := by ring
  have hcy : |toReal (c * y)| ≤ (1 + u64) * (|toReal c| * |toReal y|) := by
    rw [he2, abs_mul, abs_mul]
    calc |toReal c| * |toReal y| * |1 + δ2| ≤ |toReal c| * |toReal y| * (1 + u64) :=
          mul_le_mul_of_nonneg_left (habs δ2 hδ2) (by positivity)
      _ = (1 + u64) * (|toReal c| * |toReal y|) := by ring
  have h1u : (0 : ℝ) ≤ 1 + u64 := by linarith [u64_pos.le]
  calc |toReal (x * a + c * y)|
      = |toReal (x * a) + toReal (c * y)| * |1 + δ0| := by rw [he0, abs_mul]
    _ ≤ (|toReal (x * a)| + |toReal (c * y)|) * (1 + u64) :=
        mul_le_mul (abs_add_le _ _) (habs δ0 hδ0) (abs_nonneg _) (by positivity)
    _ ≤ ((1 + u64) * (|toReal a| * |toReal x|) + (1 + u64) * (|toReal c| * |toReal y|)) * (1 + u64) :=
        mul_le_mul_of_nonneg_right (add_le_add hxa hcy) h1u
    _ = (1 + u64) ^ 2 * (|toReal a| * |toReal x| + |toReal c| * |toReal y|) := by ring

/-- **Muon bias-update per-entry bound.** Each new bias component `nb[i] = b[i]·(1−lr·wd) + lr·upd[i]`
    (Nesterov + weight decay, NO orthogonalization) satisfies `|toReal nb[i]| ≤ (1+u64)²·(|toReal(1−lr·wd)|·
    |toReal b[i]| + |toReal lr|·|toReal upd[i]|)`, `upd[i] = grad[i] + μ·(μ·mom[i] + grad[i])`. The 1D analogue
    of `stepMat_opNorm_le`: a weight-decay contraction on the old bias plus a scaled Nesterov update. -/
theorem stepVec_entry_le (b grad mom : Array Float) (lr wd mu : Float) (i : Nat) (hi : i < b.size) :
    |toReal ((stepVec b grad mom lr wd mu).1[i]!)|
      ≤ (1 + u64) ^ 2 * (|toReal (1.0 - lr * wd)| * |toReal (b[i]!)|
          + |toReal lr| * |toReal (grad[i]! + mu * (mu * mom[i]! + grad[i]!))|) := by
  have hget : (stepVec b grad mom lr wd mu).1[i]!
      = b[i]! * (1.0 - lr * wd) + lr * (grad[i]! + mu * (mu * mom[i]! + grad[i]!)) := by
    unfold stepVec
    rw [range_map_get b.size _ i hi, range_map_get b.size _ i hi, range_map_get b.size _ i hi]
  rw [hget]
  exact addmul_abs_le _ _ _ _

/-- **Nesterov update magnitude bound.** The Float value `g + mu·(mu·m + g)` (the Nesterov update, exact ℝ
    value `mu²·m + mu·g + g`) has `|toReal| ≤ (1+u64)⁴·(|toReal mu|²·|toReal m| + |toReal mu|·|toReal g| +
    |toReal g|)` — four rounding layers (`mul`, `add`, `mul`, `add`), each inflating by `≤ (1+u64)`. Bounds the
    per-step bias forcing in terms of the gradient (`g`) and momentum (`m`) entry magnitudes. -/
theorem nesterov_upd_abs_le (g mu m : Float) :
    |toReal (g + mu * (mu * m + g))|
      ≤ (1 + u64) ^ 4 * (|toReal mu| ^ 2 * |toReal m| + |toReal mu| * |toReal g| + |toReal g|) := by
  have hu : (0 : ℝ) ≤ u64 := u64_pos.le
  have habs : ∀ δ : ℝ, |δ| ≤ u64 → |1 + δ| ≤ 1 + u64 := fun δ hδ => by
    calc |1 + δ| ≤ |(1 : ℝ)| + |δ| := abs_add_le _ _
      _ = 1 + |δ| := by rw [abs_one]
      _ ≤ 1 + u64 := by linarith
  have hmm : (0 : ℝ) ≤ |toReal mu| := abs_nonneg _
  have hgg : (0 : ℝ) ≤ |toReal g| := abs_nonneg _
  have hmnn : (0 : ℝ) ≤ |toReal m| := abs_nonneg _
  obtain ⟨δ1, hd1, he1⟩ := mul_model mu m
  have b1 : |toReal (mu * m)| ≤ (1 + u64) * (|toReal mu| * |toReal m|) := by
    rw [he1, abs_mul, abs_mul]
    calc |toReal mu| * |toReal m| * |1 + δ1| ≤ |toReal mu| * |toReal m| * (1 + u64) :=
          mul_le_mul_of_nonneg_left (habs δ1 hd1) (by positivity)
      _ = (1 + u64) * (|toReal mu| * |toReal m|) := by ring
  obtain ⟨δ2, hd2, he2⟩ := add_model (mu * m) g
  have b2 : |toReal (mu * m + g)| ≤ (1 + u64) ^ 2 * (|toReal mu| * |toReal m|) + (1 + u64) * |toReal g| := by
    rw [he2, abs_mul]
    calc |toReal (mu * m) + toReal g| * |1 + δ2|
        ≤ (|toReal (mu * m)| + |toReal g|) * (1 + u64) :=
          mul_le_mul (abs_add_le _ _) (habs δ2 hd2) (abs_nonneg _) (by positivity)
      _ ≤ ((1 + u64) * (|toReal mu| * |toReal m|) + |toReal g|) * (1 + u64) :=
          mul_le_mul_of_nonneg_right (by linarith [b1]) (by positivity)
      _ = (1 + u64) ^ 2 * (|toReal mu| * |toReal m|) + (1 + u64) * |toReal g| := by ring
  obtain ⟨δ3, hd3, he3⟩ := mul_model mu (mu * m + g)
  have b3 : |toReal (mu * (mu * m + g))|
      ≤ (1 + u64) ^ 3 * (|toReal mu| ^ 2 * |toReal m|) + (1 + u64) ^ 2 * (|toReal mu| * |toReal g|) := by
    rw [he3, abs_mul, abs_mul]
    calc |toReal mu| * |toReal (mu * m + g)| * |1 + δ3|
        ≤ |toReal mu| * ((1 + u64) ^ 2 * (|toReal mu| * |toReal m|) + (1 + u64) * |toReal g|) * (1 + u64) := by
          apply mul_le_mul (mul_le_mul_of_nonneg_left b2 hmm) (habs δ3 hd3) (abs_nonneg _) (by positivity)
      _ = (1 + u64) ^ 3 * (|toReal mu| ^ 2 * |toReal m|) + (1 + u64) ^ 2 * (|toReal mu| * |toReal g|) := by ring
  obtain ⟨δ4, hd4, he4⟩ := add_model g (mu * (mu * m + g))
  rw [he4, abs_mul]
  have h1le : (1 : ℝ) ≤ 1 + u64 := by linarith
  have hp14 : (1 + u64) ≤ (1 + u64) ^ 4 := by
    calc (1 + u64) = (1 + u64) ^ 1 := (pow_one _).symm
      _ ≤ (1 + u64) ^ 4 := pow_le_pow_right₀ h1le (by norm_num)
  have hp34 : (1 + u64) ^ 3 ≤ (1 + u64) ^ 4 := pow_le_pow_right₀ h1le (by norm_num)
  calc |toReal g + toReal (mu * (mu * m + g))| * |1 + δ4|
      ≤ (|toReal g| + |toReal (mu * (mu * m + g))|) * (1 + u64) :=
        mul_le_mul (abs_add_le _ _) (habs δ4 hd4) (abs_nonneg _) (by positivity)
    _ ≤ (|toReal g| + ((1 + u64) ^ 3 * (|toReal mu| ^ 2 * |toReal m|)
          + (1 + u64) ^ 2 * (|toReal mu| * |toReal g|))) * (1 + u64) :=
        mul_le_mul_of_nonneg_right (by linarith [b3]) (by positivity)
    _ = (1 + u64) * |toReal g| + (1 + u64) ^ 4 * (|toReal mu| ^ 2 * |toReal m|)
          + (1 + u64) ^ 3 * (|toReal mu| * |toReal g|) := by ring
    _ ≤ (1 + u64) ^ 4 * (|toReal mu| ^ 2 * |toReal m| + |toReal mu| * |toReal g| + |toReal g|) := by
        nlinarith [hp14, hp34, hgg, mul_nonneg hmm hgg, mul_nonneg (mul_nonneg hmm hmm) hmnn]

/-- **Matrix Nesterov-update per-entry magnitude bound.** Each entry of the Newton–Schulz *input*
    `update = matLin 1.0 grad mu (matLin mu mom 1.0 grad)` (the orthogonalization pre-image inside the Muon
    `stepMat`) satisfies, at any in-range index `(i,j)`,
      `|toReal update[i][j]| ≤ (1+u64)⁴·(|toReal μ|²·|toReal mom[i][j]| + |toReal μ|·|toReal grad[i][j]|
        + |toReal grad[i][j]|)`.
    The exact real entry value is `1.0·gᵢⱼ + μ·(μ·mᵢⱼ + 1.0·gᵢⱼ)` (two nested `matLin` combines, each with its
    own `1.0` coefficient rounding), unfolded by `matLin_getElem`; two applications of `addmul_abs_le` (a
    `(1+u64)²` per 2-term combine) absorb into the same `(1+u64)⁴` prefactor as the 1D `nesterov_upd_abs_le`,
    using `(1+u64)² ≤ (1+u64)⁴`. This is the MATRIX analogue of `nesterov_upd_abs_le` (which covers the bias
    `stepVec`'s bare-`+` update `g + μ·(μ·m + g)`); here the update flows through the explicit-coefficient
    `matLin` kernels of the 2D weight path, so the coefficient roundings are genuine and the bound is the
    entrywise forcing that feeds the Newton–Schulz orthogonalization. -/
theorem nesterovMat_entry_abs_le (grad mom : Mat) (mu : Float) (i j : Nat)
    (hgi : i < grad.size) (hgj : j < (grad[i]!).size)
    (hmi : i < mom.size) (hmj : j < (mom[i]!).size) :
    |toReal (((matLin 1.0 grad mu (matLin mu mom 1.0 grad))[i]!)[j]!)|
      ≤ (1 + u64) ^ 4 * (|toReal mu| ^ 2 * |toReal ((mom[i]!)[j]!)|
          + |toReal mu| * |toReal ((grad[i]!)[j]!)| + |toReal ((grad[i]!)[j]!)|) := by
  rw [matLin_getElem 1.0 grad mu (matLin mu mom 1.0 grad) i j hgi hgj,
      matLin_getElem mu mom 1.0 grad i j hmi hmj]
  set g := (grad[i]!)[j]! with hg
  set m := (mom[i]!)[j]! with hm
  have hu : (0 : ℝ) ≤ u64 := u64_pos.le
  have hP1 : (1 : ℝ) ≤ 1 + u64 := by linarith
  have hG : (0 : ℝ) ≤ |toReal g| := abs_nonneg _
  have hU : (0 : ℝ) ≤ |toReal mu| := abs_nonneg _
  have h1 : |toReal (1.0 : Float)| = (1 : ℝ) := by rw [Puffer.FloatR.toReal_oneLit, abs_one]
  have hP2nn : (0 : ℝ) ≤ (1 + u64) ^ 2 := by positivity
  have hP42 : (1 + u64) ^ 2 ≤ (1 + u64) ^ 4 := pow_le_pow_right₀ hP1 (by norm_num)
  have hinner : |toReal (mu * m + 1.0 * g)|
      ≤ (1 + u64) ^ 2 * (|toReal mu| * |toReal m| + |toReal g|) := by
    have h := addmul_abs_le mu m 1.0 g
    rw [h1] at h
    calc |toReal (mu * m + 1.0 * g)|
        ≤ (1 + u64) ^ 2 * (|toReal m| * |toReal mu| + 1 * |toReal g|) := h
      _ = (1 + u64) ^ 2 * (|toReal mu| * |toReal m| + |toReal g|) := by ring
  have houter := addmul_abs_le 1.0 g mu (mu * m + 1.0 * g)
  rw [h1] at houter
  calc |toReal (1.0 * g + mu * (mu * m + 1.0 * g))|
      ≤ (1 + u64) ^ 2 * (|toReal g| * 1 + |toReal mu| * |toReal (mu * m + 1.0 * g)|) := houter
    _ ≤ (1 + u64) ^ 2 * (|toReal g| * 1
          + |toReal mu| * ((1 + u64) ^ 2 * (|toReal mu| * |toReal m| + |toReal g|))) := by
        apply mul_le_mul_of_nonneg_left _ hP2nn
        have h := mul_le_mul_of_nonneg_left hinner hU
        linarith
    _ = (1 + u64) ^ 2 * |toReal g| + (1 + u64) ^ 4 * (|toReal mu| ^ 2 * |toReal m|)
          + (1 + u64) ^ 4 * (|toReal mu| * |toReal g|) := by ring
    _ ≤ (1 + u64) ^ 4 * |toReal g| + (1 + u64) ^ 4 * (|toReal mu| ^ 2 * |toReal m|)
          + (1 + u64) ^ 4 * (|toReal mu| * |toReal g|) := by
        have := mul_le_mul_of_nonneg_right hP42 hG
        linarith
    _ = (1 + u64) ^ 4 * (|toReal mu| ^ 2 * |toReal m| + |toReal mu| * |toReal g| + |toReal g|) := by ring

/-- **`matLin` per-entry rounding reduced to entry magnitudes.** `matLinEntryErrBnd a x b y 0 0 ≤
    u64·(2+u64)·(|toReal a|·|toReal x| + |toReal b|·|toReal y|)` — the `matLin` combine's rounding on the
    entry `a·x + b·y` (exact inputs `εx = εy = 0`) reduces to the coefficient·entry magnitudes. The weight-side
    analogue of `nesterov_upd_abs_le`: it grinds the per-step weight-update `E` down to the fundamental entry
    magnitudes `|toReal x|` (weight) and `|toReal y|` (orthogonalized update). -/
theorem matLinEntryErrBnd_le (a x b y : Float) :
    matLinEntryErrBnd a x b y 0 0
      ≤ u64 * (2 + u64) * (|toReal a| * |toReal x| + |toReal b| * |toReal y|) := by
  unfold matLinEntryErrBnd
  simp only [mul_zero, add_zero]
  obtain ⟨δ1, hd1, he1⟩ := mul_model a x
  obtain ⟨δ2, hd2, he2⟩ := mul_model b y
  have hu : (0 : ℝ) ≤ u64 := u64_pos.le
  have habs : ∀ δ : ℝ, |δ| ≤ u64 → |1 + δ| ≤ 1 + u64 := fun δ hδ => by
    calc |1 + δ| ≤ |(1 : ℝ)| + |δ| := abs_add_le _ _
      _ = 1 + |δ| := by rw [abs_one]
      _ ≤ 1 + u64 := by linarith
  have hax : |toReal (a * x)| ≤ (1 + u64) * (|toReal a| * |toReal x|) := by
    rw [he1, abs_mul, abs_mul]
    calc |toReal a| * |toReal x| * |1 + δ1| ≤ |toReal a| * |toReal x| * (1 + u64) :=
          mul_le_mul_of_nonneg_left (habs δ1 hd1) (by positivity)
      _ = (1 + u64) * (|toReal a| * |toReal x|) := by ring
  have hby : |toReal (b * y)| ≤ (1 + u64) * (|toReal b| * |toReal y|) := by
    rw [he2, abs_mul, abs_mul]
    calc |toReal b| * |toReal y| * |1 + δ2| ≤ |toReal b| * |toReal y| * (1 + u64) :=
          mul_le_mul_of_nonneg_left (habs δ2 hd2) (by positivity)
      _ = (1 + u64) * (|toReal b| * |toReal y|) := by ring
  have hsum : |toReal (a * x) + toReal (b * y)| ≤ |toReal (a * x)| + |toReal (b * y)| := abs_add_le _ _
  rw [abs_mul, abs_mul]
  nlinarith [hax, hby, hsum, hu, mul_nonneg (abs_nonneg (toReal a)) (abs_nonneg (toReal x)),
    mul_nonneg (abs_nonneg (toReal b)) (abs_nonneg (toReal y))]

/-- `|toReal (mu·m + g)| ≤ (1+u64)²·|toReal mu|·|toReal m| + (1+u64)·|toReal g|` — one `mul` then one `add`
    rounding. Bounds one Muon momentum-update entry (`newMom[i] = mu·mom[i] + grad[i]`) by the previous
    momentum and the gradient magnitude. -/
theorem mulAdd_abs_le (mu m g : Float) :
    |toReal (mu * m + g)| ≤ (1 + u64) ^ 2 * (|toReal mu| * |toReal m|) + (1 + u64) * |toReal g| := by
  obtain ⟨δ0, hd0, he0⟩ := add_model (mu * m) g
  obtain ⟨δ1, hd1, he1⟩ := mul_model mu m
  have hu : (0 : ℝ) ≤ u64 := u64_pos.le
  have habs : ∀ δ : ℝ, |δ| ≤ u64 → |1 + δ| ≤ 1 + u64 := fun δ hδ => by
    calc |1 + δ| ≤ |(1 : ℝ)| + |δ| := abs_add_le _ _
      _ = 1 + |δ| := by rw [abs_one]
      _ ≤ 1 + u64 := by linarith
  have hmm : |toReal (mu * m)| ≤ (1 + u64) * (|toReal mu| * |toReal m|) := by
    rw [he1, abs_mul, abs_mul]
    calc |toReal mu| * |toReal m| * |1 + δ1| ≤ |toReal mu| * |toReal m| * (1 + u64) :=
          mul_le_mul_of_nonneg_left (habs δ1 hd1) (by positivity)
      _ = (1 + u64) * (|toReal mu| * |toReal m|) := by ring
  rw [he0, abs_mul]
  calc |toReal (mu * m) + toReal g| * |1 + δ0|
      ≤ (|toReal (mu * m)| + |toReal g|) * (1 + u64) :=
        mul_le_mul (abs_add_le _ _) (habs δ0 hd0) (abs_nonneg _) (by positivity)
    _ ≤ ((1 + u64) * (|toReal mu| * |toReal m|) + |toReal g|) * (1 + u64) :=
        mul_le_mul_of_nonneg_right (by linarith [hmm]) (by positivity)
    _ = (1 + u64) ^ 2 * (|toReal mu| * |toReal m|) + (1 + u64) * |toReal g| := by ring

/-- The Muon momentum-update entry: `(stepVec b grad mom lr wd mu).2[i]! = mu·mom[i]! + grad[i]!`
    (`newMom = μ·mom + grad`). -/
theorem stepVec_snd_get (b grad mom : Array Float) (lr wd mu : Float) (i : Nat) (hi : i < b.size) :
    (stepVec b grad mom lr wd mu).2[i]! = mu * mom[i]! + grad[i]! := by
  unfold stepVec; rw [range_map_get b.size _ i hi]

end Puffer.RL.MuonStepBound
