/-
The coefficient-robust 5-iteration composition bound — the scalar composition (`MuonComposition`) proven
for the ACTUAL `toReal` Float coefficients on the growing intervals, so it can transport to the runnable
mirror (whose per-step embedding `nsIterR_toMatrixR_pstep` carries `toReal (4.0848 : Float)`, not the exact
ℝ literal).

`MuonComposition.muon_comp_step1..5` are for the exact ℝ literals `4.0848…`. Here each step is re-proven for
coefficients within `10⁻⁶` of those literals (the `lit_close` bound for the Float coefficients), on the same
growing intervals `[0, B_{k-1}]`. The `≈10⁻¹⁶` coefficient perturbation is absorbed by proving the exact
bound to `B_k − 0.002` and adding the (loose) `≤ 0.002` perturbation margin:

  • `float_comp_step` : perturbation on `[0,U]` (`U ≤ 2`) — coeffs within `10⁻⁶` of exact `(a₀,b₀,c₀)`
      (each `≤ 7`) with exact bound `≤ C₀` give `≤ C₀ + 0.002` (the `float_coeff_bound` technique of
      `MuonCoeffFloat`, generalized off `[0,1]`).
  • `muon_comp_step1..5_float` : the five steps for the Float schedule `Puffer.FloatR.Muon.muonCoeffs`,
      `t·(toReal a + toReal b·t + toReal c·t²)² ≤ B_k` on `[0, B_{k-1}]` (schedule `[1,1.63,1.63,1.57,1.33,1.3131]`).
  • `muon_comp_bounded_float` : chaining — `t₀ ≤ 1 ⟹ t₅ ≤ 1.3131` for the Float-coefficient composition.

So the runnable Muon composition's squared singular values stay `≤ 1.63` throughout and end `≤ 1.3131` — the
coefficient-robust form of the composition boundedness. Axiom-clean beyond the trusted Float base
(`toReal`, `toReal_neg`, `toReal_ofScientific_close`). This removes the coefficient-rounding obstacle to the
full mirror-fold composition (remaining: the shape-tracked 5-fold nesting via `nsIterR_toMatrixR_pstep`).
-/
import Mathlib
import Puffer.RL.MuonCoeffFloat

namespace Puffer.RL.MuonCompositionFloat

open Puffer.FloatR (toReal)
open Puffer.RL.MuonCoeffFloat (lit_close)

/-- Perturbation on `[0,U]` (`U ≤ 2`): coeffs within `10⁻⁶` of exact `(a₀,b₀,c₀)` (each `≤ 7`) with exact
    bound `≤ C₀` give `≤ C₀ + 0.002`. Generalizes `MuonCoeffFloat.float_coeff_bound` off `[0,1]`. -/
theorem float_comp_step (a0 b0 c0 α β γ U C : ℝ)
    (ha0 : |a0| ≤ 7) (hb0 : |b0| ≤ 7) (hc0 : |c0| ≤ 7)
    (ha : |α - a0| ≤ 1e-6) (hb : |β - b0| ≤ 1e-6) (hc : |γ - c0| ≤ 1e-6)
    (hU : U ≤ 2) (t : ℝ) (h0 : 0 ≤ t) (h1 : t ≤ U)
    (hexact : t * (a0 + b0 * t + c0 * t ^ 2) ^ 2 ≤ C - 0.002) :
    t * (α + β * t + γ * t ^ 2) ^ 2 ≤ C := by
  set q0 := a0 + b0 * t + c0 * t ^ 2 with hq0def
  have ht2 : t ^ 2 ≤ 4 := by nlinarith [h0, h1, hU]
  have hq0abs : |q0| ≤ 49 := by
    rw [hq0def, abs_le] at *
    constructor <;> nlinarith [ha0.1, ha0.2, hb0.1, hb0.2, hc0.1, hc0.2, h0, h1, hU, ht2]
  have hd : |(α + β * t + γ * t ^ 2) - q0| ≤ 7e-6 := by
    rw [abs_le] at ha hb hc ⊢; rw [hq0def]
    constructor <;> nlinarith [ha.1, ha.2, hb.1, hb.2, hc.1, hc.2, h0, h1, hU, ht2]
  have hqsq : (α + β * t + γ * t ^ 2) ^ 2 ≤ q0 ^ 2 + 2 * 49 * 7e-6 + (7e-6) ^ 2 := by
    have hd' := abs_le.1 hd
    nlinarith [abs_le.1 hq0abs, hd'.1, hd'.2, sq_nonneg (α + β * t + γ * t ^ 2 - q0)]
  nlinarith [hexact, hqsq, h0, h1, hU, mul_nonneg h0 (sq_nonneg (α + β * t + γ * t ^ 2))]

theorem muon_comp_step1_float (t : ℝ) (h0 : 0 ≤ t) (h1 : t ≤ 1) :
    t * (toReal (4.0848 : Float) + toReal (-6.8946 : Float) * t + toReal (2.9270 : Float) * t ^ 2) ^ 2
      ≤ 1.63 := by
  have hbc : |toReal (-6.8946 : Float) - (-6.8946 : ℝ)| ≤ 1e-6 := by
    rw [show (-6.8946 : Float) = -(6.8946 : Float) from rfl, Puffer.FloatR.toReal_neg,
      show (-toReal (6.8946 : Float)) - (-6.8946 : ℝ)
        = -(toReal (6.8946 : Float) - (6.8946 : ℝ)) from by ring, abs_neg]
    exact lit_close 68946 4 (by norm_num)
  refine float_comp_step 4.0848 (-6.8946) 2.9270 _ _ _ 1 1.63 (by norm_num) (by norm_num) (by norm_num)
    (lit_close 40848 4 (by norm_num)) hbc (lit_close 29270 4 (by norm_num)) (by norm_num) t h0 h1 ?_
  nlinarith [mul_nonneg h0 (sq_nonneg (t - 0.2373)), mul_nonneg (sub_nonneg.2 h1) (sq_nonneg (t - 0.2373)),
    mul_nonneg (mul_nonneg h0 (sub_nonneg.2 h1)) (sq_nonneg (t - 0.2373)), sq_nonneg (t * (t - 0.2373)),
    sq_nonneg (t - 0.2373), mul_nonneg h0 (sub_nonneg.2 h1), h0, sub_nonneg.2 h1,
    mul_nonneg (mul_nonneg h0 h0) (sub_nonneg.2 h1)]

theorem muon_comp_step2_float (t : ℝ) (h0 : 0 ≤ t) (h1 : t ≤ 1.63) :
    t * (toReal (3.9505 : Float) + toReal (-6.3029 : Float) * t + toReal (2.6377 : Float) * t ^ 2) ^ 2
      ≤ 1.63 := by
  have hbc : |toReal (-6.3029 : Float) - (-6.3029 : ℝ)| ≤ 1e-6 := by
    rw [show (-6.3029 : Float) = -(6.3029 : Float) from rfl, Puffer.FloatR.toReal_neg,
      show (-toReal (6.3029 : Float)) - (-6.3029 : ℝ)
        = -(toReal (6.3029 : Float) - (6.3029 : ℝ)) from by ring, abs_neg]
    exact lit_close 63029 4 (by norm_num)
  refine float_comp_step 3.9505 (-6.3029) 2.6377 _ _ _ 1.63 1.63 (by norm_num) (by norm_num) (by norm_num)
    (lit_close 39505 4 (by norm_num)) hbc (lit_close 26377 4 (by norm_num)) (by norm_num) t h0 h1 ?_
  nlinarith [mul_nonneg h0 (sq_nonneg (t - 0.2539)), mul_nonneg (sub_nonneg.2 h1) (sq_nonneg (t - 0.2539)),
    mul_nonneg (mul_nonneg h0 (sub_nonneg.2 h1)) (sq_nonneg (t - 0.2539)), sq_nonneg (t * (t - 0.2539)),
    sq_nonneg (t - 0.2539), mul_nonneg h0 (sub_nonneg.2 h1), h0, sub_nonneg.2 h1,
    mul_nonneg (mul_nonneg h0 h0) (sub_nonneg.2 h1)]

theorem muon_comp_step3_float (t : ℝ) (h0 : 0 ≤ t) (h1 : t ≤ 1.63) :
    t * (toReal (3.7418 : Float) + toReal (-5.5913 : Float) * t + toReal (2.3037 : Float) * t ^ 2) ^ 2
      ≤ 1.57 := by
  have hbc : |toReal (-5.5913 : Float) - (-5.5913 : ℝ)| ≤ 1e-6 := by
    rw [show (-5.5913 : Float) = -(5.5913 : Float) from rfl, Puffer.FloatR.toReal_neg,
      show (-toReal (5.5913 : Float)) - (-5.5913 : ℝ)
        = -(toReal (5.5913 : Float) - (5.5913 : ℝ)) from by ring, abs_neg]
    exact lit_close 55913 4 (by norm_num)
  refine float_comp_step 3.7418 (-5.5913) 2.3037 _ _ _ 1.63 1.57 (by norm_num) (by norm_num) (by norm_num)
    (lit_close 37418 4 (by norm_num)) hbc (lit_close 23037 4 (by norm_num)) (by norm_num) t h0 h1 ?_
  nlinarith [mul_nonneg h0 (sq_nonneg (t - 0.2750)), mul_nonneg (sub_nonneg.2 h1) (sq_nonneg (t - 0.2750)),
    mul_nonneg (mul_nonneg h0 (sub_nonneg.2 h1)) (sq_nonneg (t - 0.2750)), sq_nonneg (t * (t - 0.2750)),
    sq_nonneg (t - 0.2750), mul_nonneg h0 (sub_nonneg.2 h1), h0, sub_nonneg.2 h1,
    mul_nonneg (mul_nonneg h0 h0) (sub_nonneg.2 h1)]

theorem muon_comp_step4_float (t : ℝ) (h0 : 0 ≤ t) (h1 : t ≤ 1.57) :
    t * (toReal (2.8769 : Float) + toReal (-3.1427 : Float) * t + toReal (1.2046 : Float) * t ^ 2) ^ 2
      ≤ 1.33 := by
  have hbc : |toReal (-3.1427 : Float) - (-3.1427 : ℝ)| ≤ 1e-6 := by
    rw [show (-3.1427 : Float) = -(3.1427 : Float) from rfl, Puffer.FloatR.toReal_neg,
      show (-toReal (3.1427 : Float)) - (-3.1427 : ℝ)
        = -(toReal (3.1427 : Float) - (3.1427 : ℝ)) from by ring, abs_neg]
    exact lit_close 31427 4 (by norm_num)
  refine float_comp_step 2.8769 (-3.1427) 1.2046 _ _ _ 1.57 1.33 (by norm_num) (by norm_num) (by norm_num)
    (lit_close 28769 4 (by norm_num)) hbc (lit_close 12046 4 (by norm_num)) (by norm_num) t h0 h1 ?_
  nlinarith [mul_nonneg h0 (sq_nonneg (t - 0.4154)), mul_nonneg (sub_nonneg.2 h1) (sq_nonneg (t - 0.4154)),
    mul_nonneg (mul_nonneg h0 (sub_nonneg.2 h1)) (sq_nonneg (t - 0.4154)), sq_nonneg (t * (t - 0.4154)),
    sq_nonneg (t - 0.4154), mul_nonneg h0 (sub_nonneg.2 h1), h0, sub_nonneg.2 h1,
    mul_nonneg (mul_nonneg h0 h0) (sub_nonneg.2 h1)]

theorem muon_comp_step5_float (t : ℝ) (h0 : 0 ≤ t) (h1 : t ≤ 1.33) :
    t * (toReal (2.8366 : Float) + toReal (-3.0525 : Float) * t + toReal (1.2012 : Float) * t ^ 2) ^ 2
      ≤ 1.3131 := by
  have hbc : |toReal (-3.0525 : Float) - (-3.0525 : ℝ)| ≤ 1e-6 := by
    rw [show (-3.0525 : Float) = -(3.0525 : Float) from rfl, Puffer.FloatR.toReal_neg,
      show (-toReal (3.0525 : Float)) - (-3.0525 : ℝ)
        = -(toReal (3.0525 : Float) - (3.0525 : ℝ)) from by ring, abs_neg]
    exact lit_close 30525 4 (by norm_num)
  refine float_comp_step 2.8366 (-3.0525) 1.2012 _ _ _ 1.33 1.3131 (by norm_num) (by norm_num) (by norm_num)
    (lit_close 28366 4 (by norm_num)) hbc (lit_close 12012 4 (by norm_num)) (by norm_num) t h0 h1 ?_
  nlinarith [mul_nonneg h0 (sq_nonneg (t - 0.43236)), mul_nonneg (sub_nonneg.2 h1) (sq_nonneg (t - 0.43236)),
    mul_nonneg (mul_nonneg h0 (sub_nonneg.2 h1)) (sq_nonneg (t - 0.43236)), sq_nonneg (t * (t - 0.43236)),
    sq_nonneg (t - 0.43236), mul_nonneg h0 (sub_nonneg.2 h1), h0, sub_nonneg.2 h1,
    mul_nonneg (mul_nonneg h0 h0) (sub_nonneg.2 h1)]

/-- **The Float-coefficient composition stays bounded.** Same as `MuonComposition.muon_comp_bounded` but
    for the `toReal` Float coefficients: `t₀ ≤ 1 ⟹ t₅ ≤ 1.3131`. -/
theorem muon_comp_bounded_float (t0 t1 t2 t3 t4 t5 : ℝ)
    (hstart : 0 ≤ t0 ∧ t0 ≤ 1)
    (h1 : 0 ≤ t1 ∧ t1 ≤ t0 * (toReal (4.0848 : Float) + toReal (-6.8946 : Float) * t0
      + toReal (2.9270 : Float) * t0 ^ 2) ^ 2)
    (h2 : 0 ≤ t2 ∧ t2 ≤ t1 * (toReal (3.9505 : Float) + toReal (-6.3029 : Float) * t1
      + toReal (2.6377 : Float) * t1 ^ 2) ^ 2)
    (h3 : 0 ≤ t3 ∧ t3 ≤ t2 * (toReal (3.7418 : Float) + toReal (-5.5913 : Float) * t2
      + toReal (2.3037 : Float) * t2 ^ 2) ^ 2)
    (h4 : 0 ≤ t4 ∧ t4 ≤ t3 * (toReal (2.8769 : Float) + toReal (-3.1427 : Float) * t3
      + toReal (1.2046 : Float) * t3 ^ 2) ^ 2)
    (h5 : 0 ≤ t5 ∧ t5 ≤ t4 * (toReal (2.8366 : Float) + toReal (-3.0525 : Float) * t4
      + toReal (1.2012 : Float) * t4 ^ 2) ^ 2) :
    t5 ≤ 1.3131 := by
  have b1 : t1 ≤ 1.63 := h1.2.trans (muon_comp_step1_float t0 hstart.1 hstart.2)
  have b2 : t2 ≤ 1.63 := h2.2.trans (muon_comp_step2_float t1 h1.1 b1)
  have b3 : t3 ≤ 1.57 := h3.2.trans (muon_comp_step3_float t2 h2.1 b2)
  have b4 : t4 ≤ 1.33 := h4.2.trans (muon_comp_step4_float t3 h3.1 b3)
  exact h5.2.trans (muon_comp_step5_float t4 h4.1 b4)

/-- **The tail Newton–Schulz map has a forward-invariant interval — the Float composition stays
    bounded under UNBOUNDED continued iteration.** The five tuned steps land the squared singular
    value at `≤ 1.3131` (`muon_comp_bounded_float`). This shows the schedule can then be run with its
    LAST coefficients forever without blowing up: the step-5 scalar map
    `f₅(s) = s·(a₅ + b₅·s + c₅·s²)²` (Float coefficients `2.8366, -3.0525, 1.2012`) maps `[0, 1.3131]`
    into itself, so for every iteration count `n`, `f₅^[n]` keeps any `t ∈ [0, 1.3131]` in `[0, 1.3131]`.
    The closure of the loop is that step 5's own bound `1.3131` is `≤` its admissible input radius `1.33`
    (`muon_comp_step5_float`), so `[0,1.3131]` is a genuine invariant SET of the tail map, not merely a
    one-step image. Both hypotheses are load-bearing: dropping `t ≤ 1.3131` fails (e.g. `t = 2` gives
    `f₅(2) ≈ 4.72 > 1.3131`), and dropping `0 ≤ t` fails the lower `0 ≤ f₅^[n] t` at `n = 0`. -/
theorem muon_step5_float_iterate_invariant (n : ℕ) (t : ℝ) (h0 : 0 ≤ t) (h1 : t ≤ 1.3131) :
    0 ≤ (fun s : ℝ => s * (toReal (2.8366 : Float) + toReal (-3.0525 : Float) * s
          + toReal (1.2012 : Float) * s ^ 2) ^ 2)^[n] t
      ∧ (fun s : ℝ => s * (toReal (2.8366 : Float) + toReal (-3.0525 : Float) * s
          + toReal (1.2012 : Float) * s ^ 2) ^ 2)^[n] t ≤ 1.3131 := by
  induction n with
  | zero => exact ⟨h0, h1⟩
  | succ k ih =>
    obtain ⟨hk0, hk1⟩ := ih
    rw [Function.iterate_succ_apply']
    exact ⟨mul_nonneg hk0 (sq_nonneg _),
      muon_comp_step5_float _ hk0 (hk1.trans (by norm_num))⟩

end Puffer.RL.MuonCompositionFloat
