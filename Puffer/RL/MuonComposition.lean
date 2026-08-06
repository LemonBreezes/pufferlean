/-
The 5-iteration Newton–Schulz COMPOSITION stays bounded — the analytic resolution of the
"composition-interval" caveat.

The per-step bound `t·q(t)² ≤ 1.63` (`MuonScalarBound`) is on `t ∈ [0,1]` (singular values `σ ≤ 1`).
But one step can push the operator norm to `√1.63 ≈ 1.277 > 1`, so its output feeds the NEXT step at
`σ ≤ 1.277`, i.e. `t = σ² ≤ 1.63` — OUTSIDE `[0,1]`. Naively chaining the `[0,1]` bound therefore fails;
the composition could in principle blow up. It does NOT: the tuned schedule is designed so the composition
converges, and the singular values stay bounded and decay back toward 1.

This file PROVES that, by discharging the per-step scalar bound on the actual GROWING intervals of the
composition. With the norm-squared sequence `B = [1, 1.63, 1.63, 1.57, 1.33, 1.3111]` (each `B_k` a valid
upper bound for `max_{t∈[0,B_{k-1}]} t·q_k(t)²`, verified numerically), each step `k` maps
`t ∈ [0, B_{k-1}]` into `[0, B_k]`:

  step 1 (coeffs 1): `t∈[0,1]`    ⟹ `t·q₁(t)² ≤ 1.63`   (max ≈ 1.62089)
  step 2 (coeffs 2): `t∈[0,1.63]` ⟹ `t·q₂(t)² ≤ 1.63`   (max ≈ 1.61267)
  step 3 (coeffs 3): `t∈[0,1.63]` ⟹ `t·q₃(t)² ≤ 1.57`   (max ≈ 1.55563)
  step 4 (coeffs 4): `t∈[0,1.57]` ⟹ `t·q₄(t)² ≤ 1.33`   (max ≈ 1.31510)
  step 5 (coeffs 5): `t∈[0,1.33]` ⟹ `t·q₅(t)² ≤ 1.3111` (max ≈ 1.31107)

Composing: the squared singular values stay `≤ 1.63` throughout (operator norm `≤ √1.63 ≈ 1.277`), and the
FINAL iterate has squared singular values `≤ 1.3111`, i.e. operator norm `≤ √1.3111 ≈ 1.1450`. So the full
5-step Newton–Schulz composition is DIMENSION-FREE O(1) — never exceeding `≈1.277`, ending `≈1.145`. The
composition is bounded; the tower's constant does not blow up over the 5 iterations.

Each bound is a degree-5 polynomial inequality on `[0, B_{k-1}]` closed by `nlinarith` with a
Positivstellensatz hint anchored at the interior maximizer (the wider intervals need higher-degree product
terms `t·(B−t)·(t−t₀)²` and `(t·(t−t₀))²`). Axiom-clean, no `sorry`.

REMAINING (matrix-level assembly): fold these per-step scalar bounds through the operator-norm step
(generalized `opNorm_muon_step_le` as a function of the input norm, using `eigenvalue ≤ ‖Gram‖ = ‖X‖²`)
to get `‖newtonSchulz-composition seed‖₂ ≤ √1.3111` on the abstract `Matrix` side; the per-step spectral
reduction and the runnable-step transport are already proved.
-/
import Mathlib

namespace Puffer.RL.MuonComposition

/-- Step 1 (coeffs `(4.0848,-6.8946,2.9270)`), input `t∈[0,1]` ⟹ `t·q(t)² ≤ 1.63`. -/
theorem muon_comp_step1 (t : ℝ) (h0 : 0 ≤ t) (h1 : t ≤ 1) :
    t * (4.0848 - 6.8946 * t + 2.9270 * t ^ 2) ^ 2 ≤ 1.63 := by
  nlinarith [mul_nonneg h0 (sq_nonneg (t - 0.2373)),
    mul_nonneg (sub_nonneg.2 h1) (sq_nonneg (t - 0.2373)),
    mul_nonneg (mul_nonneg h0 (sub_nonneg.2 h1)) (sq_nonneg (t - 0.2373)),
    sq_nonneg (t * (t - 0.2373)), sq_nonneg (t - 0.2373),
    mul_nonneg h0 (sub_nonneg.2 h1), h0, sub_nonneg.2 h1,
    mul_nonneg (mul_nonneg h0 h0) (sub_nonneg.2 h1)]

/-- Step 2 (coeffs `(3.9505,-6.3029,2.6377)`), input `t∈[0,1.63]` ⟹ `t·q(t)² ≤ 1.63`. -/
theorem muon_comp_step2 (t : ℝ) (h0 : 0 ≤ t) (h1 : t ≤ 1.63) :
    t * (3.9505 - 6.3029 * t + 2.6377 * t ^ 2) ^ 2 ≤ 1.63 := by
  nlinarith [mul_nonneg h0 (sq_nonneg (t - 0.2539)),
    mul_nonneg (sub_nonneg.2 h1) (sq_nonneg (t - 0.2539)),
    mul_nonneg (mul_nonneg h0 (sub_nonneg.2 h1)) (sq_nonneg (t - 0.2539)),
    sq_nonneg (t * (t - 0.2539)), sq_nonneg (t - 0.2539),
    mul_nonneg h0 (sub_nonneg.2 h1), h0, sub_nonneg.2 h1,
    mul_nonneg (mul_nonneg h0 h0) (sub_nonneg.2 h1)]

/-- Step 3 (coeffs `(3.7418,-5.5913,2.3037)`), input `t∈[0,1.63]` ⟹ `t·q(t)² ≤ 1.57`. -/
theorem muon_comp_step3 (t : ℝ) (h0 : 0 ≤ t) (h1 : t ≤ 1.63) :
    t * (3.7418 - 5.5913 * t + 2.3037 * t ^ 2) ^ 2 ≤ 1.57 := by
  nlinarith [mul_nonneg h0 (sq_nonneg (t - 0.2750)),
    mul_nonneg (sub_nonneg.2 h1) (sq_nonneg (t - 0.2750)),
    mul_nonneg (mul_nonneg h0 (sub_nonneg.2 h1)) (sq_nonneg (t - 0.2750)),
    sq_nonneg (t * (t - 0.2750)), sq_nonneg (t - 0.2750),
    mul_nonneg h0 (sub_nonneg.2 h1), h0, sub_nonneg.2 h1,
    mul_nonneg (mul_nonneg h0 h0) (sub_nonneg.2 h1)]

/-- Step 4 (coeffs `(2.8769,-3.1427,1.2046)`), input `t∈[0,1.57]` ⟹ `t·q(t)² ≤ 1.33`. -/
theorem muon_comp_step4 (t : ℝ) (h0 : 0 ≤ t) (h1 : t ≤ 1.57) :
    t * (2.8769 - 3.1427 * t + 1.2046 * t ^ 2) ^ 2 ≤ 1.33 := by
  nlinarith [mul_nonneg h0 (sq_nonneg (t - 0.4154)),
    mul_nonneg (sub_nonneg.2 h1) (sq_nonneg (t - 0.4154)),
    mul_nonneg (mul_nonneg h0 (sub_nonneg.2 h1)) (sq_nonneg (t - 0.4154)),
    sq_nonneg (t * (t - 0.4154)), sq_nonneg (t - 0.4154),
    mul_nonneg h0 (sub_nonneg.2 h1), h0, sub_nonneg.2 h1,
    mul_nonneg (mul_nonneg h0 h0) (sub_nonneg.2 h1)]

/-- Step 5 (coeffs `(2.8366,-3.0525,1.2012)`), input `t∈[0,1.33]` ⟹ `t·q(t)² ≤ 1.3111`
    (the true interior max is `≈ 1.31107`, so this is within `3e-5` of tight). -/
theorem muon_comp_step5 (t : ℝ) (h0 : 0 ≤ t) (h1 : t ≤ 1.33) :
    t * (2.8366 - 3.0525 * t + 1.2012 * t ^ 2) ^ 2 ≤ 1.3111 := by
  nlinarith [mul_nonneg h0 (sq_nonneg (t - 0.43236)),
    mul_nonneg (sub_nonneg.2 h1) (sq_nonneg (t - 0.43236)),
    mul_nonneg (mul_nonneg h0 (sub_nonneg.2 h1)) (sq_nonneg (t - 0.43236)),
    sq_nonneg (t * (t - 0.43236)), sq_nonneg (t - 0.43236),
    mul_nonneg h0 (sub_nonneg.2 h1), h0, sub_nonneg.2 h1,
    mul_nonneg (mul_nonneg h0 h0) (sub_nonneg.2 h1)]

/-- **The composition stays bounded.** Chaining the five steps: any squared singular value that starts in
    `[0,1]` stays in `[0,1.63]` through every step and lands in `[0,1.3111]` after all five. Stated as the
    propagation of the norm-squared bound `t₀ ≤ 1 ⟹ t₅ ≤ 1.3111` for any values with `t_{k+1} ≤` the step-`k`
    scalar image — the discrete form of "the singular values never blow up over the 5 iterations". -/
theorem muon_comp_bounded (t0 t1 t2 t3 t4 t5 : ℝ)
    (hstart : 0 ≤ t0 ∧ t0 ≤ 1)
    (h1 : 0 ≤ t1 ∧ t1 ≤ t0 * (4.0848 - 6.8946 * t0 + 2.9270 * t0 ^ 2) ^ 2)
    (h2 : 0 ≤ t2 ∧ t2 ≤ t1 * (3.9505 - 6.3029 * t1 + 2.6377 * t1 ^ 2) ^ 2)
    (h3 : 0 ≤ t3 ∧ t3 ≤ t2 * (3.7418 - 5.5913 * t2 + 2.3037 * t2 ^ 2) ^ 2)
    (h4 : 0 ≤ t4 ∧ t4 ≤ t3 * (2.8769 - 3.1427 * t3 + 1.2046 * t3 ^ 2) ^ 2)
    (h5 : 0 ≤ t5 ∧ t5 ≤ t4 * (2.8366 - 3.0525 * t4 + 1.2012 * t4 ^ 2) ^ 2) :
    t5 ≤ 1.3111 := by
  have b1 : t1 ≤ 1.63 := h1.2.trans (muon_comp_step1 t0 hstart.1 hstart.2)
  have b2 : t2 ≤ 1.63 := h2.2.trans (muon_comp_step2 t1 h1.1 b1)
  have b3 : t3 ≤ 1.57 := h3.2.trans (muon_comp_step3 t2 h2.1 b2)
  have b4 : t4 ≤ 1.33 := h4.2.trans (muon_comp_step4 t3 h3.1 b3)
  exact h5.2.trans (muon_comp_step5 t4 h4.1 b4)

end Puffer.RL.MuonComposition
