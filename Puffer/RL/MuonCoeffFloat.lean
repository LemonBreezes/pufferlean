/-
Closing the coefficient-rounding gap: the scalar bound `t·q(t)² ≤ 1.64` on `[0,1]` for the ACTUAL
Float Muon coefficients (`toReal` of `Puffer.FloatR.Muon.muonCoeffs`), not just the exact ℝ literals.

`nsIterR_opNorm_le` (`NewtonSchulzAssembly.lean`) needs the scalar bound on `toReal a + toReal b·t +
toReal c·t²` for the Float coefficients `(a,b,c)`; `muon_scalar_bound` (`MuonScalarBound.lean`) proved
it only for the exact ℝ literals `(4.0848, …)`. The gap is the `≈10⁻¹⁶` rounding of decimal→binary64
literal parsing. It is closed here using the trusted-model axiom `toReal_ofScientific_close` (Float
decimal literals round to nearest, relative `u64` — the literal analogue of the `(1+δ)` op-models):

  • `lit_close` : `|toReal (lit : Float) − (lit : ℝ)| ≤ 10⁻⁶` for any `OfScientific` literal `≤ 7`
    in magnitude (the actual bound is `≈ 7·u64 ≈ 8·10⁻¹⁶`; `10⁻⁶` is a slack threshold);
  • `float_coeff_bound` : a PERTURBATION bound — if `(α,β,γ)` are within `10⁻⁶` of exact `(a₀,b₀,c₀)`
    (each `≤ 7`) and `t·(a₀+b₀t+c₀t²)² ≤ 1.63`, then `t·(α+βt+γt²)² ≤ 1.64` on `[0,1]`. The looser
    `1.64` (vs `1.63`) leaves margin for the coefficient perturbation; `√1.64 < 1.2807` is still O(1);
  • `muon_scalar_bound_float` : combines them over the real `Float` schedule — for every
    `(a,b,c) ∈ Puffer.FloatR.Muon.muonCoeffs` and `t ∈ [0,1]`,
    `t·(toReal a + toReal b·t + toReal c·t²)² ≤ 1.64`;
  • `nsIterR_opNorm_le_muon` : the payoff — for the runnable Float coefficients and an iterate whose
    Gram eigenvalues lie in `[0,1]`, `‖toMatrixR (nsIterR X (a,b,c))‖₂ ≤ √1.64 < 1.2807`.

Axiom-clean beyond the trusted Float axioms (`toReal`, `toReal_neg`, `toReal_ofScientific_close`).
This removes the coefficient-rounding residual gap flagged in `NewtonSchulzAssembly.lean`: the tight
Newton–Schulz per-step operator-norm bound now holds for the trainer's ACTUAL Float coefficients.
-/
import Mathlib
import Puffer.Float.Basic
import Puffer.Float.Muon
import Puffer.RL.NewtonSchulzAssembly

namespace Puffer.RL.MuonCoeffFloat

open scoped Matrix Matrix.Norms.L2Operator
open Matrix
open Puffer.FloatR (toReal u64 u64_pos toReal_neg toReal_ofScientific_close)
open Puffer.RL.NewtonSchulzError (MatR nsIterR)
open Puffer.RL.MatrixEmbed (toMatrixR)
open Puffer.RL.NewtonSchulzAssembly (nsIterR_opNorm_le)

/-- A positive `OfScientific` Float literal embeds within `10⁻⁶` of its exact ℝ value (magnitude `≤ 7`;
    the true bound is `≈ 7·u64 ≈ 8·10⁻¹⁶`). -/
theorem lit_close (m e : Nat) (hb : |(OfScientific.ofScientific m true e : ℝ)| ≤ 7) :
    |toReal (OfScientific.ofScientific m true e : Float)
      - (OfScientific.ofScientific m true e : ℝ)| ≤ 1e-6 := by
  have hax := toReal_ofScientific_close m true e
  have hle : u64 * |(OfScientific.ofScientific m true e : ℝ)| ≤ 1e-6 :=
    le_trans (mul_le_mul_of_nonneg_left hb u64_pos.le) (by unfold u64; norm_num)
  linarith [hax, hle]

/-- **Perturbation bound.** Coefficients within `10⁻⁶` of exact `(a₀,b₀,c₀)` (each `≤ 7` in magnitude),
    given the exact bound `≤ 1.63`, satisfy `t·(α+βt+γt²)² ≤ 1.64` on `[0,1]`. -/
theorem float_coeff_bound (a0 b0 c0 α β γ : ℝ)
    (ha0 : |a0| ≤ 7) (hb0 : |b0| ≤ 7) (hc0 : |c0| ≤ 7)
    (ha : |α - a0| ≤ 1e-6) (hb : |β - b0| ≤ 1e-6) (hc : |γ - c0| ≤ 1e-6)
    (t : ℝ) (h0 : 0 ≤ t) (h1 : t ≤ 1)
    (hexact : t * (a0 + b0 * t + c0 * t ^ 2) ^ 2 ≤ 1.63) :
    t * (α + β * t + γ * t ^ 2) ^ 2 ≤ 1.64 := by
  set q0 := a0 + b0 * t + c0 * t ^ 2 with hq0def
  have ht2 : t ^ 2 ≤ 1 := by nlinarith [h0, h1]
  have hq0abs : |q0| ≤ 21 := by
    rw [hq0def, abs_le] at *
    constructor <;> nlinarith [ha0.1, ha0.2, hb0.1, hb0.2, hc0.1, hc0.2, h0, h1, ht2]
  have hd : |(α + β * t + γ * t ^ 2) - q0| ≤ 3e-6 := by
    rw [abs_le] at ha hb hc ⊢; rw [hq0def]
    constructor <;> nlinarith [ha.1, ha.2, hb.1, hb.2, hc.1, hc.2, h0, h1, ht2]
  have hqsq : (α + β * t + γ * t ^ 2) ^ 2 ≤ q0 ^ 2 + 2 * 21 * 3e-6 + (3e-6) ^ 2 := by
    have hd' := abs_le.1 hd
    nlinarith [abs_le.1 hq0abs, hd'.1, hd'.2, sq_nonneg (α + β * t + γ * t ^ 2 - q0)]
  nlinarith [hexact, hqsq, h0, h1, mul_nonneg h0 (sq_nonneg (α + β * t + γ * t ^ 2))]

/-- Closeness of a NEGATIVE Float literal `-lit` to its exact ℝ value. -/
private theorem neg_lit_close (m e : Nat) (v : ℝ)
    (hv : (OfScientific.ofScientific m true e : ℝ) = v) (hb : |v| ≤ 7) :
    |toReal (-(OfScientific.ofScientific m true e : Float)) - (-v)| ≤ 1e-6 := by
  rw [toReal_neg, show (-toReal (OfScientific.ofScientific m true e : Float)) - (-v)
      = -(toReal (OfScientific.ofScientific m true e : Float)
          - (OfScientific.ofScientific m true e : ℝ)) from by rw [hv]; ring, abs_neg]
  exact lit_close m e (hv ▸ hb)

/-- **Uniform Float Muon scalar bound.** For every triple in the real `Float` schedule
    `Puffer.FloatR.Muon.muonCoeffs` and every `t ∈ [0,1]`,
    `t·(toReal a + toReal b·t + toReal c·t²)² ≤ 1.64`. -/
theorem muon_scalar_bound_float {a b c : Float}
    (hmem : (a, b, c) ∈ Puffer.FloatR.Muon.muonCoeffs.toList) (t : ℝ) (h0 : 0 ≤ t) (h1 : t ≤ 1) :
    t * (toReal a + toReal b * t + toReal c * t ^ 2) ^ 2 ≤ 1.64 := by
  fin_cases hmem
  · exact float_coeff_bound 4.0848 (-6.8946) 2.9270 _ _ _ (by norm_num) (by norm_num) (by norm_num)
      (lit_close 40848 4 (by norm_num)) (neg_lit_close 68946 4 _ rfl (by norm_num))
      (lit_close 29270 4 (by norm_num)) t h0 h1
      (by nlinarith [mul_nonneg h0 (sq_nonneg (t - 0.2373)),
        mul_nonneg (sub_nonneg.2 h1) (sq_nonneg (t - 0.2373)), sq_nonneg (t - 0.2373),
        mul_nonneg h0 (sub_nonneg.2 h1), h0, sub_nonneg.2 h1,
        mul_nonneg (mul_nonneg h0 h0) (sub_nonneg.2 h1)])
  · exact float_coeff_bound 3.9505 (-6.3029) 2.6377 _ _ _ (by norm_num) (by norm_num) (by norm_num)
      (lit_close 39505 4 (by norm_num)) (neg_lit_close 63029 4 _ rfl (by norm_num))
      (lit_close 26377 4 (by norm_num)) t h0 h1
      (by nlinarith [mul_nonneg h0 (sq_nonneg (t - 0.2539)),
        mul_nonneg (sub_nonneg.2 h1) (sq_nonneg (t - 0.2539)), sq_nonneg (t - 0.2539),
        mul_nonneg h0 (sub_nonneg.2 h1), h0, sub_nonneg.2 h1,
        mul_nonneg (mul_nonneg h0 h0) (sub_nonneg.2 h1)])
  · exact float_coeff_bound 3.7418 (-5.5913) 2.3037 _ _ _ (by norm_num) (by norm_num) (by norm_num)
      (lit_close 37418 4 (by norm_num)) (neg_lit_close 55913 4 _ rfl (by norm_num))
      (lit_close 23037 4 (by norm_num)) t h0 h1
      (by nlinarith [mul_nonneg h0 (sq_nonneg (t - 0.2750)),
        mul_nonneg (sub_nonneg.2 h1) (sq_nonneg (t - 0.2750)), sq_nonneg (t - 0.2750),
        mul_nonneg h0 (sub_nonneg.2 h1), h0, sub_nonneg.2 h1,
        mul_nonneg (mul_nonneg h0 h0) (sub_nonneg.2 h1)])
  · exact float_coeff_bound 2.8769 (-3.1427) 1.2046 _ _ _ (by norm_num) (by norm_num) (by norm_num)
      (lit_close 28769 4 (by norm_num)) (neg_lit_close 31427 4 _ rfl (by norm_num))
      (lit_close 12046 4 (by norm_num)) t h0 h1
      (by nlinarith [mul_nonneg h0 (sq_nonneg (t - 0.4153)),
        mul_nonneg (sub_nonneg.2 h1) (sq_nonneg (t - 0.4153)), sq_nonneg (t - 0.4153),
        mul_nonneg h0 (sub_nonneg.2 h1), h0, sub_nonneg.2 h1,
        mul_nonneg (mul_nonneg h0 h0) (sub_nonneg.2 h1)])
  · exact float_coeff_bound 2.8366 (-3.0525) 1.2012 _ _ _ (by norm_num) (by norm_num) (by norm_num)
      (lit_close 28366 4 (by norm_num)) (neg_lit_close 30525 4 _ rfl (by norm_num))
      (lit_close 12012 4 (by norm_num)) t h0 h1
      (by nlinarith [mul_nonneg h0 (sq_nonneg (t - 0.4324)),
        mul_nonneg (sub_nonneg.2 h1) (sq_nonneg (t - 0.4324)), sq_nonneg (t - 0.4324),
        mul_nonneg h0 (sub_nonneg.2 h1), h0, sub_nonneg.2 h1,
        mul_nonneg (mul_nonneg h0 h0) (sub_nonneg.2 h1)])

/-- **One runnable-Muon Newton–Schulz step is O(1), for the ACTUAL Float coefficients.** For every
    `(a,b,c) ∈ Puffer.FloatR.Muon.muonCoeffs` and any iterate `X` (`r ≤ c` branch) whose Gram matrix
    `MX·MXᴴ` has all eigenvalues in `[0,1]`, `‖toMatrixR (nsIterR X (a,b,c))‖₂ ≤ √1.64 < 1.2807`. The
    coefficient-rounding gap is discharged by `muon_scalar_bound_float`. -/
theorem nsIterR_opNorm_le_muon (X : MatR) {a b c : Float} (r cc : Nat)
    (hmem : (a, b, c) ∈ Puffer.FloatR.Muon.muonCoeffs.toList)
    (hXsz : X.size = r) (hXrow : ∀ i, i < r → (X[i]!).size = cc)
    (hr : 0 < r) (hrc : r ≤ cc)
    (hev : ∀ i, 0 ≤ (isHermitian_mul_conjTranspose_self (toMatrixR r cc X)).eigenvalues i
        ∧ (isHermitian_mul_conjTranspose_self (toMatrixR r cc X)).eigenvalues i ≤ 1) :
    ‖toMatrixR r cc (nsIterR X (a, b, c))‖ ≤ Real.sqrt 1.64 := by
  refine nsIterR_opNorm_le X a b c r cc 1.64 (by norm_num) hXsz hXrow hr hrc fun i => ?_
  obtain ⟨h0, h1⟩ := hev i
  rw [abs_of_nonneg (mul_nonneg h0 (sq_nonneg _))]
  exact muon_scalar_bound_float hmem _ h0 h1

end Puffer.RL.MuonCoeffFloat
