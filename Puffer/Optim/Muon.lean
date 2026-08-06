/-
Muon optimizer — Newton–Schulz orthogonalization, over ℝ.

`~/src/PufferLib/src/muon.cu` orthogonalizes each 2D update `X` by 5 Newton–Schulz
iterations with a tuned coefficient schedule (`ns_coeffs`, lines 78–84):

    X ↦ aᵢ·X + bᵢ·(XXᵀ)X + cᵢ·(XXᵀ)²X.

Writing `X = U Σ Vᵀ` (SVD), each iteration acts **diagonally on the singular
values** by the odd scalar polynomial `φ(σ) = a·σ + b·σ³ + c·σ⁵` (because
`(XXᵀ)ᵏ X = U Σ^{2k+1} Vᵀ`), so `X_next = U·φ(Σ)·Vᵀ`. Orthogonalization ⟺ `φ`
drives every singular value to 1. We formalize that scalar core, and prove the
convergence of the *classical* Newton–Schulz map (the `a=3/2, b=−1/2, c=0`
special case) with its exact quadratic-convergence identity.
-/
import Mathlib

namespace Puffer.Optim.Muon

/-- The scalar Newton–Schulz map `φ(σ) = a·σ + b·σ³ + c·σ⁵` — the action of one
    Muon iteration on a singular value. -/
def nsScalar (a b c σ : ℝ) : ℝ := a * σ + b * σ ^ 3 + c * σ ^ 5

/-- SVD-action factorization: `φ(σ) = σ · (a + b·σ² + c·σ⁴)`. The `σ` factor is why
    the iteration preserves the singular vectors and only reshapes the spectrum. -/
theorem nsScalar_factor (a b c σ : ℝ) :
    nsScalar a b c σ = σ * (a + b * σ ^ 2 + c * σ ^ 4) := by
  unfold nsScalar; ring

/-- `φ` is odd — consistent with orthogonalizing signed spectra. -/
theorem nsScalar_odd (a b c σ : ℝ) : nsScalar a b c (-σ) = - nsScalar a b c σ := by
  unfold nsScalar; ring

/-- One is a fixed point iff the coefficients sum to one. -/
theorem nsScalar_one (a b c : ℝ) : nsScalar a b c 1 = a + b + c := by
  unfold nsScalar; ring

/-- PufferLib's tuned 5-step coefficient schedule (`ns_coeffs`, muon.cu:78–84). Note
    the per-step sums differ from 1 — it is a schedule tuned so the *composition*
    maps `[σ_min, 1]` to ≈1 in few steps, not five copies of a fixed-point map. -/
noncomputable def muonCoeffs : List (ℝ × ℝ × ℝ) :=
  [(4.0848, -6.8946, 2.9270),
   (3.9505, -6.3029, 2.6377),
   (3.7418, -5.5913, 2.3037),
   (2.8769, -3.1427, 1.2046),
   (2.8366, -3.0525, 1.2012)]

/-! ### Classical Newton–Schulz map and its convergence

The textbook Newton–Schulz orthogonalization iterates `φ(σ) = 1.5σ − 0.5σ³`, the
`a=3/2, b=−1/2, c=0` special case of `nsScalar`. -/

/-- Classical Newton–Schulz scalar map `φ(σ) = 1.5σ − 0.5σ³`. -/
noncomputable def nsClassical (σ : ℝ) : ℝ := (3 / 2) * σ - (1 / 2) * σ ^ 3

theorem nsClassical_eq_nsScalar (σ : ℝ) : nsClassical σ = nsScalar (3 / 2) (-(1 / 2)) 0 σ := by
  unfold nsClassical nsScalar; ring

theorem nsClassical_fixed_one : nsClassical 1 = 1 := by unfold nsClassical; norm_num

/-- `φ(0) = 0` — zero is a fixed point (a singular value of 0 stays 0: the map can't manufacture rank). -/
theorem nsClassical_fixed_zero : nsClassical 0 = 0 := by unfold nsClassical; norm_num

/-- **The only fixed points on `[0,1]` are `0` and `1`.** `φ(σ) = σ ⟺ σ = 0 ∨ σ = 1`, since
    `φ(σ) − σ = ½·σ·(1−σ)·(1+σ)` and the factor `1+σ > 0` on the interval, so the product vanishes exactly at
    `σ = 0` or `σ = 1`. The iteration has NO spurious interior fixed point that could trap a singular value away
    from 0 or 1: combined with `nsClassical_ge_self` (strictly moves up off `0`), every `σ ∈ (0,1]` is driven
    toward the attracting fixed point `1`, while `0` is the sole repelling one. -/
theorem nsClassical_fixed_iff (σ : ℝ) (h0 : 0 ≤ σ) (h1 : σ ≤ 1) :
    nsClassical σ = σ ↔ σ = 0 ∨ σ = 1 := by
  constructor
  · intro h
    have hfact : nsClassical σ - σ = (1 / 2) * σ * (1 - σ) * (1 + σ) := by unfold nsClassical; ring
    have hne : (1 : ℝ) + σ ≠ 0 := by linarith
    have h2 : (1 / 2) * σ * (1 - σ) * (1 + σ) = 0 := by rw [← hfact]; linarith
    rw [mul_eq_zero] at h2
    rcases h2 with h2 | h2
    · rw [mul_eq_zero, mul_eq_zero] at h2
      rcases h2 with (h2 | h2) | h2
      · norm_num at h2
      · exact Or.inl h2
      · exact Or.inr (by linarith)
    · exact absurd h2 hne
  · intro h
    rcases h with h | h
    · rw [h]; exact nsClassical_fixed_zero
    · rw [h]; exact nsClassical_fixed_one

/-- **Exact quadratic-convergence identity**: the distance to 1 is a *square* in the
    current error — `1 − φ(σ) = ½·(σ+2)·(1−σ)²`. This is why Newton–Schulz doubles
    correct digits each step. -/
theorem nsClassical_error (σ : ℝ) :
    1 - nsClassical σ = (1 / 2) * (σ + 2) * (1 - σ) ^ 2 := by
  unfold nsClassical; ring

/-- On `[0,1]`, `φ(σ) ≤ 1` (never overshoots). -/
theorem nsClassical_le_one (σ : ℝ) (h0 : 0 ≤ σ) (_h1 : σ ≤ 1) : nsClassical σ ≤ 1 := by
  have he := nsClassical_error σ
  nlinarith [mul_nonneg (by linarith : (0 : ℝ) ≤ σ + 2) (sq_nonneg (1 - σ))]

/-- On `[0,1]`, `φ` moves the singular value *up* toward 1: `σ ≤ φ(σ)`. -/
theorem nsClassical_ge_self (σ : ℝ) (h0 : 0 ≤ σ) (h1 : σ ≤ 1) : σ ≤ nsClassical σ := by
  have : nsClassical σ - σ = (1 / 2) * σ * (1 - σ) * (1 + σ) := by unfold nsClassical; ring
  nlinarith [mul_nonneg (mul_nonneg (by linarith : (0:ℝ) ≤ (1/2) * σ) (by linarith : (0:ℝ) ≤ 1 - σ))
    (by linarith : (0:ℝ) ≤ 1 + σ)]

/-- **Monotonicity on `[0,1]`**: `φ` preserves the ordering of singular values — `σ₁ ≤ σ₂ ⟹ φ(σ₁) ≤ φ(σ₂)`.
    The increment factors as `φ(σ₂) − φ(σ₁) = ½·(σ₂−σ₁)·(3 − (σ₂²+σ₂σ₁+σ₁²)) ≥ 0`, since on `[0,1]` the
    quadratic form `σ₂²+σ₂σ₁+σ₁² ≤ 3`. So the Newton–Schulz map is order-preserving on the unit interval: the
    whole singular-value spectrum is pulled toward 1 without any crossings, which is what makes the iteration a
    faithful orthogonalization (largest and smallest singular values keep their relative order). -/
theorem nsClassical_mono (σ1 σ2 : ℝ) (h0 : 0 ≤ σ1) (h : σ1 ≤ σ2) (h1 : σ2 ≤ 1) :
    nsClassical σ1 ≤ nsClassical σ2 := by
  unfold nsClassical
  nlinarith [mul_nonneg (sub_nonneg.mpr h)
    (by nlinarith [sq_nonneg σ1, sq_nonneg σ2, mul_nonneg h0 (h0.trans h)] :
      (0:ℝ) ≤ 3 - (σ2^2 + σ2*σ1 + σ1^2)), sq_nonneg (σ2 - σ1)]

/-- **Strict monotonicity on `[0,1)`**: `φ` is *strictly* order-preserving — `0 ≤ σ₁ < σ₂ ≤ 1 ⟹ φ(σ₁) < φ(σ₂)`.
    The increment factor `3 − (σ₂²+σ₂σ₁+σ₁²)` is strictly positive here (the quadratic form is `< 3` unless
    `σ₁ = σ₂ = 1`, excluded by `σ₁ < σ₂`), so distinct singular values map to *distinct* images: the Newton–Schulz
    map is injective on `[0,1]`, never collapsing two singular values together. This no-collapse/rank-preservation
    guarantee strengthens `nsClassical_mono` — the orthogonalization keeps the spectrum strictly ordered. -/
theorem nsClassical_strictMono (σ1 σ2 : ℝ) (h0 : 0 ≤ σ1) (h : σ1 < σ2) (h1 : σ2 ≤ 1) :
    nsClassical σ1 < nsClassical σ2 := by
  unfold nsClassical
  have hσ1lt1 : σ1 < 1 := lt_of_lt_of_le h h1
  have hσ2nn : 0 ≤ σ2 := h0.trans h.le
  have hform : (0:ℝ) < 3 - (σ2^2 + σ2*σ1 + σ1^2) := by
    nlinarith [mul_nonneg h0 hσ2nn, mul_nonneg (sub_nonneg.mpr h1) hσ2nn,
      mul_nonneg (sub_nonneg.mpr h1) h0, mul_pos (sub_pos.mpr hσ1lt1) (sub_pos.mpr hσ1lt1),
      sq_nonneg σ2, sq_nonneg σ1, h1, hσ1lt1, h0, hσ2nn]
  nlinarith [mul_pos (sub_pos.mpr h) hform]

/-- **Quadratic convergence bound**: on `[0,1]`, `1 − φ(σ) ≤ 1.5·(1−σ)²`. -/
theorem nsClassical_quadratic (σ : ℝ) (_h0 : 0 ≤ σ) (h1 : σ ≤ 1) :
    1 - nsClassical σ ≤ (3 / 2) * (1 - σ) ^ 2 := by
  rw [nsClassical_error]
  nlinarith [mul_nonneg (sq_nonneg (1 - σ)) (by linarith : (0 : ℝ) ≤ 1 - σ)]

/-- The iterate stays in `[0,1]` — the iteration is well-defined and monotone up. -/
theorem nsClassical_iterate_mem (n : ℕ) (σ : ℝ) (h0 : 0 ≤ σ) (h1 : σ ≤ 1) :
    0 ≤ nsClassical^[n] σ ∧ nsClassical^[n] σ ≤ 1 := by
  induction n with
  | zero => exact ⟨h0, h1⟩
  | succ k ih =>
      obtain ⟨hk0, hk1⟩ := ih
      rw [Function.iterate_succ_apply']
      exact ⟨le_trans hk0 (nsClassical_ge_self _ hk0 hk1), nsClassical_le_one _ hk0 hk1⟩

/-- **Iteration converges quadratically**: the error after `n+1` steps is at most
    `1.5×` the square of the error after `n` steps. -/
theorem nsClassical_iterate_quadratic (n : ℕ) (σ : ℝ) (h0 : 0 ≤ σ) (h1 : σ ≤ 1) :
    1 - nsClassical^[n + 1] σ ≤ (3 / 2) * (1 - nsClassical^[n] σ) ^ 2 := by
  obtain ⟨hk0, hk1⟩ := nsClassical_iterate_mem n σ h0 h1
  rw [Function.iterate_succ_apply']
  exact nsClassical_quadratic _ hk0 hk1

/-! ### Other Muon step components (faithful specs) -/

/-- Fused weight update: `w ← w·(1 − lr·wd) − lr·scale·update` (muon.cu:65). -/
def weightUpdate (w update lr wd scale : ℝ) : ℝ := w * (1 - lr * wd) - lr * scale * update

/-- Nesterov momentum accumulator update `m ← μ·m + g` (muon.cu:52). -/
def nesterovMomentum (m g μ : ℝ) : ℝ := μ * m + g

/-- Weight-decay factor `(1 − lr·wd)` is a contraction in `[0,1)` for valid `lr,wd`. -/
theorem weightDecay_factor_lt_one (lr wd : ℝ) (hlr : 0 < lr) (hwd : 0 < wd) :
    1 - lr * wd < 1 := by nlinarith

/-- **Newton–Schulz scalar magnitude bound.** `|φ(σ)| = |a·σ + b·σ³ + c·σ⁵| ≤ |a|·|σ| + |b|·|σ|³ + |c|·|σ|⁵`
    (triangle inequality + `abs_mul`/`abs_pow`) — the quintic never exceeds the sum of its per-term magnitudes. -/
theorem nsScalar_bound (a b c σ : ℝ) :
    |nsScalar a b c σ| ≤ |a| * |σ| + |b| * |σ|^3 + |c| * |σ|^5 := by
  unfold nsScalar
  have h1 := abs_add_le (a * σ + b * σ^3) (c * σ^5)
  have h2 := abs_add_le (a * σ) (b * σ^3)
  have e1 : |a * σ| = |a| * |σ| := abs_mul a σ
  have e2 : |b * σ^3| = |b| * |σ|^3 := by rw [abs_mul, abs_pow]
  have e3 : |c * σ^5| = |c| * |σ|^5 := by rw [abs_mul, abs_pow]
  linarith [h1, h2, e1, e2, e3]

/-- The weight-decay factor `(1 − lr·wd)` is nonnegative when `lr·wd ≤ 1` (the valid hyperparameter range). -/
theorem weightDecay_factor_nonneg (lr wd : ℝ) (h : lr * wd ≤ 1) : 0 ≤ 1 - lr * wd := by linarith

/-- **Weight decay is non-expansive.** For a factor `f = 1 − lr·wd ∈ [0,1]`, scaling a weight by `f` never
    increases its magnitude: `|w·f| ≤ |w|`. This is why plain weight decay is a stable regularizer — it can
    only shrink weights. -/
theorem weightDecay_nonexpansive (w lr wd : ℝ) (h0 : 0 ≤ 1 - lr * wd) (h1 : 1 - lr * wd ≤ 1) :
    |w * (1 - lr * wd)| ≤ |w| := by
  rw [abs_mul, abs_of_nonneg h0]
  exact mul_le_of_le_one_right (abs_nonneg w) h1

/-- **The Muon weight update is magnitude-bounded.** With the weight-decay factor in `[0,1]`, one fused weight
    step moves `w` by at most the gradient term: `|weightUpdate| ≤ |w| + |lr·scale·update|` — the decayed weight
    is non-expansive (`weightDecay_nonexpansive`) and the update adds at most `|lr·scale·update|`. -/
theorem weightUpdate_bound (w update lr wd scale : ℝ) (h0 : 0 ≤ 1 - lr * wd) (h1 : 1 - lr * wd ≤ 1) :
    |weightUpdate w update lr wd scale| ≤ |w| + |lr * scale * update| := by
  rw [weightUpdate]
  calc |w * (1 - lr * wd) - lr * scale * update|
      ≤ |w * (1 - lr * wd)| + |lr * scale * update| := abs_sub _ _
    _ ≤ |w| + |lr * scale * update| := by
        linarith [weightDecay_nonexpansive w lr wd h0 h1]

/-- **Nesterov-momentum single-step bound.** `|μ·m + g| ≤ |μ|·|m| + |g|` (triangle + `abs_mul`). -/
theorem nesterovMomentum_bound (m g μ : ℝ) :
    |nesterovMomentum m g μ| ≤ |μ| * |m| + |g| := by
  rw [nesterovMomentum]
  calc |μ * m + g| ≤ |μ * m| + |g| := abs_add_le _ _
    _ = |μ| * |m| + |g| := by rw [abs_mul]

/-- **Momentum boundedness invariant.** If `|m| ≤ B`, `|g| ≤ G`, the momentum coefficient `μ ∈ [0,1]`, and `B`
    is large enough that `G ≤ (1−μ)·B`, then the updated momentum stays `≤ B`. So with a uniformly-bounded
    gradient the Nesterov accumulator never escapes the ball of radius `B = G/(1−μ)` — the standard
    momentum-stability invariant. -/
theorem nesterovMomentum_bound_step (m g μ B G : ℝ) (hm : |m| ≤ B) (hg : |g| ≤ G)
    (hμ0 : 0 ≤ μ) (_hμ1 : μ ≤ 1) (hinv : G ≤ (1 - μ) * B) :
    |nesterovMomentum m g μ| ≤ B := by
  have hb := nesterovMomentum_bound m g μ
  rw [abs_of_nonneg hμ0] at hb
  have h2 : μ * |m| ≤ μ * B := mul_le_mul_of_nonneg_left hm hμ0
  nlinarith [hb, h2, hg, hinv]

/-- **Closed form of the Nesterov-momentum accumulator under a constant gradient.** The Muon step iterates
    `m ← μ·m + g` (`nesterovMomentum`, muon.cu:52). When the gradient `g` is held fixed, the accumulator after
    `n` steps starting from `m₀` has the exact closed form
    `m_n = μⁿ·m₀ + g·(1 − μⁿ)/(1 − μ)` — the solution of the linear recurrence as a geometric series. This
    complements the file's single-step bounds (`nesterovMomentum_bound`, `nesterovMomentum_bound_step`) with the
    exact MULTI-step trajectory (nothing in the file previously gave a closed form for the iterate). The
    hypothesis `μ ≠ 1` is load-bearing: at `μ = 1` the recurrence is `m ← m + g` with true value `m₀ + n·g`
    (unbounded drift), whereas the right-hand side divides by `1 − μ = 0` (e.g. `μ=1, m₀=0, g=1, n=2` iterates to
    `2 = m₀ + n·g`, not the `/(1−μ)` form). Taking `n → ∞` for `μ ∈ [0,1)` this exposes the steady state
    `m∞ = g/(1−μ)` — momentum's effective-learning-rate amplification factor `1/(1−μ)`. Proof: induction on `n`,
    with the successor step clearing the `1 − μ` denominator via `field_simp; ring`. -/
theorem nesterovMomentum_iterate_closed (m0 g μ : ℝ) (hμ : μ ≠ 1) (n : ℕ) :
    (fun m => nesterovMomentum m g μ)^[n] m0
      = μ ^ n * m0 + g * (1 - μ ^ n) / (1 - μ) := by
  have hd : (1 : ℝ) - μ ≠ 0 := sub_ne_zero.mpr (Ne.symm hμ)
  induction n with
  | zero => simp
  | succ k ih =>
    rw [Function.iterate_succ_apply', ih]
    simp only [nesterovMomentum]
    field_simp
    ring

end Puffer.Optim.Muon
