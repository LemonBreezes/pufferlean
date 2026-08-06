/-
Closing `domination_of_fold` via the faithful-mirror substitution — and, with it, the entire tight
Newton–Schulz per-step bound end-to-end for the runnable Float seed.

The fold-accuracy sub-tower (`FrobFoldAccuracy.foldSumSq2D_finSum_lb`) bounds `toReal Q ≥ (1-u64)^(r+c+1)·
(∑ (toReal X₀[i]![j]!)²)`. The seed's exact-ℝ mirror `X₀R` (from `scalarMul_MatBnd`) is FAITHFUL when the
input's own `MatBnd` has `ε = 0`: then `X₀Rᵢⱼ = toReal X₀[i]![j]!` exactly, so the mirror-Gram sum
`S = ∑ᵢⱼ(toMatrixR X₀R)²` IS that fold double sum. This file makes that substitution and closes the chain:

  • `faithful_gram_sum`         : `∑ᵢⱼ(toMatrixR X₀R)² = ∑ᵢⱼ(toReal X₀[i]![j]!)²` from `MatBnd … 0`.
  • `domination_of_fold_faithful`: the domination `(1+u64)·√S ≤ toReal(‖X₀‖_F + eps)` — discharging
      `domination_of_fold`'s abstract `hcond` from the fold accuracy (`toReal Q ≥ K·S`, `√(K·S) ≤ √(toReal Q)`)
      plus the single clean eps DESIGN MARGIN `(1+u64)√S ≤ (√(K·S)(1-u64) + toReal eps)(1-u64)`
      (`K = (1-u64)^(r+c+1)`).
  • `nsIter_seed_opNorm_closed`  : the FULLY-CLOSED runnable-seed step bound. For the actual Float
      `newtonSchulz` seed `scalarMul (1.0/(‖X₀‖_F+eps)) X₀` with a faithful mirror and the eps margin,
      `‖toMatrixF (nsIter seed)‖₂ ≤ √1.64 + √(r·c)·(rounding)` — O(1), dimension-free.

Axiom-clean modulo the trusted Float base. This is the terminal capstone of the tight Newton–Schulz tower:
every rounding layer (matrix, coefficient, and every scalar — div/add/sqrt/fold) and the spectral core are
discharged; the SOLE remaining hypothesis is the eps design margin — a runtime-checkable inequality on
`frobNorm + eps`, an algorithm-design property, not a theorem gap.
-/
import Mathlib
import Puffer.RL.NewtonSchulzFloat
import Puffer.RL.FrobFoldAccuracy

namespace Puffer.RL.NewtonSchulzSeedClosed

open scoped Matrix BigOperators Matrix.Norms.L2Operator
open Puffer.FloatR (toReal u64 u64_lt_one u64_pos sqrt_model mul_model le_of_float_le
  toReal_ofScientific_close toBits_inj)
open Puffer.FloatR.Muon (Mat frobNorm scalarMul muonCoeffs)
open Puffer.RL.NewtonSchulzError (MatR MatBnd nsIterThenErr)
open Puffer.RL.MatrixEmbed (toMatrixR toMatrixF)
open Puffer.RL.MuonMatrixRuntime (nsIter)
open Puffer.RL.NewtonSchulzFloat (domination_of_fold nsIter_seed_opNorm_dominated)
open Puffer.RL.FrobFoldAccuracy (foldSumSq2D_finSum_lb)

/-- **Faithful-mirror substitution.** With `MatBnd X0 X0R r c M 0` (input `ε = 0`), the mirror-Gram double
    sum equals the fold's `toReal` double sum termwise (`X₀Rᵢⱼ = toReal X₀[i]![j]!`). -/
theorem faithful_gram_sum (X0 : Mat) (X0R : MatR) (r c : Nat) (M : ℝ)
    (hX0 : MatBnd X0 X0R r c M 0) :
    (∑ i : Fin r, ∑ j : Fin c, (toMatrixR r c X0R i j) ^ 2)
      = ∑ i : Fin r, ∑ j : Fin c, (toReal ((X0[i.1]!)[j.1]!)) ^ 2 := by
  refine Finset.sum_congr rfl (fun i _ => Finset.sum_congr rfl (fun j _ => ?_))
  have hval : toMatrixR r c X0R i j = toReal ((X0[i.1]!)[j.1]!) := by
    rw [toMatrixR, Matrix.of_apply]
    have := hX0.err i.1 i.2 j.1 j.2
    rw [abs_nonpos_iff, sub_eq_zero] at this
    exact this.symm
  rw [hval]

/-- **Closed domination under the faithful mirror + eps margin.** For the actual Float fold
    `Q = X0.foldl (…)` (`frobNorm X0 = Float.sqrt Q`), a FAITHFUL mirror (`MatBnd X0 X0R r c M 0`), and the
    single eps design margin, the domination `(1+u64)·√S ≤ toReal(frobNorm X0 + eps)` holds — discharging
    `domination_of_fold`'s abstract `hcond` from the fold accuracy `foldSumSq2D_finSum_lb`. -/
theorem domination_of_fold_faithful (X0 : Mat) (X0R : MatR) (eps : Float) (r c : Nat) (M : ℝ)
    (hr : X0.size = r) (hc : ∀ (i : Nat) (hi : i < X0.size), (X0[i]).size = c)
    (hX0 : MatBnd X0 X0R r c M 0)
    (hnn : 0 ≤ toReal (frobNorm X0) + toReal eps)
    (heps : (1 + u64) * Real.sqrt (∑ i : Fin r, ∑ j : Fin c, (toMatrixR r c X0R i j) ^ 2)
      ≤ (Real.sqrt ((1 - u64) ^ (r + c + 1)
            * (∑ i : Fin r, ∑ j : Fin c, (toMatrixR r c X0R i j) ^ 2)) * (1 - u64)
          + toReal eps) * (1 - u64)) :
    (1 + u64) * Real.sqrt (∑ i : Fin r, ∑ j : Fin c, (toMatrixR r c X0R i j) ^ 2)
      ≤ toReal (frobNorm X0 + eps) := by
  set S := ∑ i : Fin r, ∑ j : Fin c, (toMatrixR r c X0R i j) ^ 2 with hSdef
  set Q := X0.foldl (fun s row => s + row.foldl (fun s2 x => s2 + x * x) 0.0) 0.0 with hQdef
  have hu1 : (0 : ℝ) ≤ 1 - u64 := by linarith [u64_lt_one]
  have hKQ : (1 - u64) ^ (r + c + 1) * S ≤ toReal Q := by
    rw [hSdef, faithful_gram_sum X0 X0R r c M hX0]
    exact foldSumSq2D_finSum_lb X0 r c hr hc
  have hsqrtQ : Real.sqrt ((1 - u64) ^ (r + c + 1) * S) ≤ Real.sqrt (toReal Q) :=
    Real.sqrt_le_sqrt hKQ
  refine domination_of_fold X0 eps S Q rfl hnn (le_trans heps ?_)
  have h2 : Real.sqrt ((1 - u64) ^ (r + c + 1) * S) * (1 - u64) ≤ Real.sqrt (toReal Q) * (1 - u64) :=
    mul_le_mul_of_nonneg_right hsqrtQ hu1
  nlinarith [h2, hu1, mul_nonneg hu1 (sub_nonneg.2 h2)]

/-- **Fully-closed runnable-seed step bound.** For the actual Float `newtonSchulz` seed
    `scalarMul (1.0/(‖X₀‖_F+eps)) X₀` with a faithful mirror (input `ε = 0`) and the single eps design
    margin, the per-step operator norm is `≤ √1.64 + √(r·c)·(rounding)` — O(1), dimension-free. Every
    matrix/coefficient/scalar rounding layer and the spectral core are discharged; the only remaining
    ingredient is the eps margin `heps`. -/
theorem nsIter_seed_opNorm_closed (X0 : Mat) (X0R : MatR) (eps a b d : Float) (r cc : Nat) (M : ℝ)
    (hmem : (a, b, d) ∈ Puffer.FloatR.Muon.muonCoeffs.toList)
    (hM : 0 ≤ M) (hr : 0 < r) (hrc : r ≤ cc)
    (hsz : X0.size = r) (hcsz : ∀ (i : Nat) (hi : i < X0.size), (X0[i]).size = cc)
    (hX0 : MatBnd X0 X0R r cc M 0)
    (hnn : 0 ≤ toReal (frobNorm X0) + toReal eps)
    (heps : (1 + u64) * Real.sqrt (∑ i : Fin r, ∑ j : Fin cc, (toMatrixR r cc X0R i j) ^ 2)
      ≤ (Real.sqrt ((1 - u64) ^ (r + cc + 1)
            * (∑ i : Fin r, ∑ j : Fin cc, (toMatrixR r cc X0R i j) ^ 2)) * (1 - u64)
          + toReal eps) * (1 - u64)) :
    ‖toMatrixF r cc (nsIter (scalarMul (1.0 / (frobNorm X0 + eps)) X0) (a, b, d))‖
      ≤ Real.sqrt 1.64 + Real.sqrt ((r : ℝ) * cc)
          * nsIterThenErr a b d ((1 + u64) * (|toReal (1.0 / (frobNorm X0 + eps))| * M))
              (u64 * (|toReal (1.0 / (frobNorm X0 + eps))| * M)
                + |toReal (1.0 / (frobNorm X0 + eps))| * 0) r cc :=
  nsIter_seed_opNorm_dominated X0 X0R eps a b d r cc M 0 hmem hM hr hrc hX0
    (domination_of_fold_faithful X0 X0R eps r cc M hsz hcsz hX0 hnn heps)

/-! ### The eps design margin, made explicit

`heps` is exactly a lower bound on `toReal eps`. Grinding it into that form turns the last hypothesis into
an interpretable, runtime-checkable design condition: `eps` must be at least a small multiple of the
Frobenius norm `√S`, the multiple `→ 0` as `u64 → 0` (≈ `(r+c)·u64`). -/

/-- `heps` holds exactly when `toReal eps` is at least the concrete lower bound
    `(1+u64)/(1-u64)·√S − √((1-u64)^(r+c+1)·S)·(1-u64)`. -/
theorem heps_from_eps_lb (S : ℝ) (eps : Float) (r c : Nat)
    (heps_lb : (1 + u64) / (1 - u64) * Real.sqrt S
        - Real.sqrt ((1 - u64) ^ (r + c + 1) * S) * (1 - u64) ≤ toReal eps) :
    (1 + u64) * Real.sqrt S
      ≤ (Real.sqrt ((1 - u64) ^ (r + c + 1) * S) * (1 - u64) + toReal eps) * (1 - u64) := by
  have hu : 0 < 1 - u64 := by linarith [u64_lt_one]
  set sK := Real.sqrt ((1 - u64) ^ (r + c + 1) * S)
  have h1 : (1 + u64) / (1 - u64) * Real.sqrt S ≤ sK * (1 - u64) + toReal eps := by linarith [heps_lb]
  calc (1 + u64) * Real.sqrt S
      = ((1 + u64) / (1 - u64) * Real.sqrt S) * (1 - u64) := by field_simp
    _ ≤ (sK * (1 - u64) + toReal eps) * (1 - u64) := mul_le_mul_of_nonneg_right h1 hu.le

/-- Cleaner SUFFICIENT eps design condition with an integer power (`x ≤ √x` on `[0,1]` drops the inner
    `√`): `heps` holds when `toReal eps ≥ ((1+u64)/(1-u64) − (1-u64)^(r+c+2))·√S`. -/
theorem heps_from_eps_lb_clean (S : ℝ) (eps : Float) (r c : Nat)
    (heps_lb : ((1 + u64) / (1 - u64) - (1 - u64) ^ (r + c + 2)) * Real.sqrt S ≤ toReal eps) :
    (1 + u64) * Real.sqrt S
      ≤ (Real.sqrt ((1 - u64) ^ (r + c + 1) * S) * (1 - u64) + toReal eps) * (1 - u64) := by
  have hu1 : (0 : ℝ) ≤ 1 - u64 := by linarith [u64_lt_one]
  have hu1' : 1 - u64 ≤ 1 := by linarith [u64_pos]
  have hSsqrt : 0 ≤ Real.sqrt S := Real.sqrt_nonneg _
  have hx0 : 0 ≤ (1 - u64) ^ (r + c + 1) := by positivity
  have hx1 : (1 - u64) ^ (r + c + 1) ≤ 1 := pow_le_one₀ hu1 hu1'
  have hpow : (1 - u64) ^ (r + c + 1) ≤ Real.sqrt ((1 - u64) ^ (r + c + 1)) := by
    have := Real.sqrt_le_sqrt (show ((1 - u64) ^ (r + c + 1)) ^ 2 ≤ (1 - u64) ^ (r + c + 1) by nlinarith)
    rwa [Real.sqrt_sq hx0] at this
  have hsplit : Real.sqrt ((1 - u64) ^ (r + c + 1) * S)
      = Real.sqrt ((1 - u64) ^ (r + c + 1)) * Real.sqrt S := Real.sqrt_mul hx0 S
  have hge : (1 - u64) ^ (r + c + 2) * Real.sqrt S
      ≤ Real.sqrt ((1 - u64) ^ (r + c + 1) * S) * (1 - u64) := by
    rw [hsplit, pow_succ]
    nlinarith [mul_le_mul_of_nonneg_right hpow hSsqrt, hu1, hSsqrt, hx0,
      mul_nonneg (mul_nonneg hx0 hSsqrt) hu1]
  apply heps_from_eps_lb S eps r c
  nlinarith [hge, heps_lb, hSsqrt]

/-- **Fully-explicit runnable-seed step bound.** Same as `nsIter_seed_opNorm_closed` but with the eps
    margin as the interpretable, runtime-checkable design condition
    `toReal eps ≥ ((1+u64)/(1-u64) − (1-u64)^(r+cc+2))·√S` (`S` = the mirror Frobenius²): `eps` at least a
    small `≈(r+cc)·u64` multiple of the input's Frobenius norm. Everything else is proved. -/
theorem nsIter_seed_opNorm_eps (X0 : Mat) (X0R : MatR) (eps a b d : Float) (r cc : Nat) (M : ℝ)
    (hmem : (a, b, d) ∈ Puffer.FloatR.Muon.muonCoeffs.toList)
    (hM : 0 ≤ M) (hr : 0 < r) (hrc : r ≤ cc)
    (hsz : X0.size = r) (hcsz : ∀ (i : Nat) (hi : i < X0.size), (X0[i]).size = cc)
    (hX0 : MatBnd X0 X0R r cc M 0)
    (hnn : 0 ≤ toReal (frobNorm X0) + toReal eps)
    (heps_lb : ((1 + u64) / (1 - u64) - (1 - u64) ^ (r + cc + 2))
        * Real.sqrt (∑ i : Fin r, ∑ j : Fin cc, (toMatrixR r cc X0R i j) ^ 2) ≤ toReal eps) :
    ‖toMatrixF r cc (nsIter (scalarMul (1.0 / (frobNorm X0 + eps)) X0) (a, b, d))‖
      ≤ Real.sqrt 1.64 + Real.sqrt ((r : ℝ) * cc)
          * nsIterThenErr a b d ((1 + u64) * (|toReal (1.0 / (frobNorm X0 + eps))| * M))
              (u64 * (|toReal (1.0 / (frobNorm X0 + eps))| * M)
                + |toReal (1.0 / (frobNorm X0 + eps))| * 0) r cc :=
  nsIter_seed_opNorm_closed X0 X0R eps a b d r cc M hmem hM hr hrc hsz hcsz hX0 hnn
    (heps_from_eps_lb_clean _ eps r cc heps_lb)

/-! ### Generalizing to a non-faithful (ε≠0) MatBnd mirror

`faithful_gram_sum` required `ε = 0` (mirror = exact `toReal` image). For a general mirror the entries
differ by up to `εb` (`MatBnd`'s error) with magnitude `≤ M`, so the mirror-Gram sum and the fold's
`toReal` sum differ by at most `Δ = r·c·εb·(2M+εb)` (`|a²−b²| ≤ |a−b|(|a|+|b|) ≤ εb(2M+εb)` per entry).
The eps design margin then absorbs `Δ` alongside the fold rounding. -/

/-- **ε≠0 MatBnd perturbation.** The fold's `toReal` sum is at least the mirror-Gram sum minus
    `Δ = r·c·εb·(2M+εb)` — the entrywise `MatBnd` perturbation, summed. -/
theorem matbnd_gram_lb (X0 : Mat) (X0R : MatR) (r c : Nat) (M εb : ℝ)
    (hX0 : MatBnd X0 X0R r c M εb) :
    (∑ i : Fin r, ∑ j : Fin c, (toMatrixR r c X0R i j) ^ 2) - (r * c : ℝ) * (εb * (2 * M + εb))
      ≤ ∑ i : Fin r, ∑ j : Fin c, (toReal ((X0[i.1]!)[j.1]!)) ^ 2 := by
  have hterm : ∀ (i : Fin r) (j : Fin c),
      (toMatrixR r c X0R i j) ^ 2 - εb * (2 * M + εb) ≤ (toReal ((X0[i.1]!)[j.1]!)) ^ 2 := by
    intro i j
    have hmag := hX0.mag i.1 i.2 j.1 j.2
    have herr := hX0.err i.1 i.2 j.1 j.2
    rw [abs_le] at hmag herr
    have hb : toMatrixR r c X0R i j = (X0R[i.1]!)[j.1]! := by rw [toMatrixR, Matrix.of_apply]
    rw [hb]
    nlinarith [hmag.1, hmag.2, herr.1, herr.2]
  have key : (∑ i : Fin r, ∑ j : Fin c, ((toMatrixR r c X0R i j) ^ 2 - εb * (2 * M + εb)))
      ≤ ∑ i : Fin r, ∑ j : Fin c, (toReal ((X0[i.1]!)[j.1]!)) ^ 2 :=
    Finset.sum_le_sum (fun i _ => Finset.sum_le_sum (fun j _ => hterm i j))
  have hconst : (∑ i : Fin r, ∑ j : Fin c, ((toMatrixR r c X0R i j) ^ 2 - εb * (2 * M + εb)))
      = (∑ i : Fin r, ∑ j : Fin c, (toMatrixR r c X0R i j) ^ 2) - (r * c : ℝ) * (εb * (2 * M + εb)) := by
    simp only [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
      nsmul_eq_mul]
    ring
  rwa [hconst] at key

/-- **Domination under a general (ε≠0) MatBnd mirror.** Generalizes `domination_of_fold_faithful`: the
    eps margin now covers both the fold rounding AND the `MatBnd` perturbation `Δ = r·c·εb·(2M+εb)`. -/
theorem domination_of_fold_matbnd (X0 : Mat) (X0R : MatR) (eps : Float) (r c : Nat) (M εb : ℝ)
    (hr : X0.size = r) (hc : ∀ (i : Nat) (hi : i < X0.size), (X0[i]).size = c)
    (hX0 : MatBnd X0 X0R r c M εb)
    (hnn : 0 ≤ toReal (frobNorm X0) + toReal eps)
    (heps : (1 + u64) * Real.sqrt (∑ i : Fin r, ∑ j : Fin c, (toMatrixR r c X0R i j) ^ 2)
      ≤ (Real.sqrt ((1 - u64) ^ (r + c + 1)
            * ((∑ i : Fin r, ∑ j : Fin c, (toMatrixR r c X0R i j) ^ 2)
                - (r * c : ℝ) * (εb * (2 * M + εb)))) * (1 - u64)
          + toReal eps) * (1 - u64)) :
    (1 + u64) * Real.sqrt (∑ i : Fin r, ∑ j : Fin c, (toMatrixR r c X0R i j) ^ 2)
      ≤ toReal (frobNorm X0 + eps) := by
  set S := ∑ i : Fin r, ∑ j : Fin c, (toMatrixR r c X0R i j) ^ 2 with hSdef
  set Q := X0.foldl (fun s row => s + row.foldl (fun s2 x => s2 + x * x) 0.0) 0.0 with hQdef
  have hu1 : (0 : ℝ) ≤ 1 - u64 := by linarith [u64_lt_one]
  have hKpow : (0 : ℝ) ≤ (1 - u64) ^ (r + c + 1) := by positivity
  have hfold : (1 - u64) ^ (r + c + 1) * (S - (r * c : ℝ) * (εb * (2 * M + εb))) ≤ toReal Q := by
    refine le_trans (mul_le_mul_of_nonneg_left ?_ hKpow) (foldSumSq2D_finSum_lb X0 r c hr hc)
    exact matbnd_gram_lb X0 X0R r c M εb hX0
  have hsqrtQ : Real.sqrt ((1 - u64) ^ (r + c + 1) * (S - (r * c : ℝ) * (εb * (2 * M + εb))))
      ≤ Real.sqrt (toReal Q) := Real.sqrt_le_sqrt hfold
  refine domination_of_fold X0 eps S Q rfl hnn (le_trans heps ?_)
  have h2 := mul_le_mul_of_nonneg_right hsqrtQ hu1
  nlinarith [h2, hu1, mul_nonneg hu1 (sub_nonneg.2 h2)]

/-- **Fully-closed runnable-seed step bound under a general (ε≠0) MatBnd mirror.** The most general form:
    for the actual Float `newtonSchulz` seed with ANY `MatBnd` mirror (`MatBnd X0 X0R r cc M εb`) and the
    eps margin covering the fold rounding + the `MatBnd` perturbation, `‖toMatrixF (nsIter seed)‖₂ ≤
    √1.64 + √(r·c)·(rounding)` — O(1), dimension-free. `domination_of_fold_faithful` is the `εb = 0` case. -/
theorem nsIter_seed_opNorm_matbnd (X0 : Mat) (X0R : MatR) (eps a b d : Float) (r cc : Nat) (M εb : ℝ)
    (hmem : (a, b, d) ∈ Puffer.FloatR.Muon.muonCoeffs.toList)
    (hM : 0 ≤ M) (hr : 0 < r) (hrc : r ≤ cc)
    (hsz : X0.size = r) (hcsz : ∀ (i : Nat) (hi : i < X0.size), (X0[i]).size = cc)
    (hX0 : MatBnd X0 X0R r cc M εb)
    (hnn : 0 ≤ toReal (frobNorm X0) + toReal eps)
    (heps : (1 + u64) * Real.sqrt (∑ i : Fin r, ∑ j : Fin cc, (toMatrixR r cc X0R i j) ^ 2)
      ≤ (Real.sqrt ((1 - u64) ^ (r + cc + 1)
            * ((∑ i : Fin r, ∑ j : Fin cc, (toMatrixR r cc X0R i j) ^ 2)
                - (r * cc : ℝ) * (εb * (2 * M + εb)))) * (1 - u64)
          + toReal eps) * (1 - u64)) :
    ‖toMatrixF r cc (nsIter (scalarMul (1.0 / (frobNorm X0 + eps)) X0) (a, b, d))‖
      ≤ Real.sqrt 1.64 + Real.sqrt ((r : ℝ) * cc)
          * nsIterThenErr a b d ((1 + u64) * (|toReal (1.0 / (frobNorm X0 + eps))| * M))
              (u64 * (|toReal (1.0 / (frobNorm X0 + eps))| * M)
                + |toReal (1.0 / (frobNorm X0 + eps))| * εb) r cc :=
  nsIter_seed_opNorm_dominated X0 X0R eps a b d r cc M εb hmem hM hr hrc hX0
    (domination_of_fold_matbnd X0 X0R eps r cc M εb hsz hcsz hX0 hnn heps)

/-! ### The eps design margin as a genuine runtime check on `frobNorm X0`

`heps_from_eps_lb_clean` reduced the eps margin to `toReal eps ≥ γ·√S` with `γ = (1+u64)/(1-u64) −
(1-u64)^(r+c+2)` — but `√S = √(∑ᵢⱼ(toMatrixR X₀R)²)` is the mirror-Gram Frobenius norm, an ABSTRACT-ℝ
quantity the runtime never forms. Here we bound `√S` above by the ACTUAL computed `frobNorm X0` (a Float),
turning the margin into a comparison between the two runtime Floats `eps` and `frobNorm X0`. The bridge is
the fold accuracy (`toReal Q ≥ (1-u64)^(r+c+1)·S`, `Q` = the sum-of-squares fold under `frobNorm`) plus the
`sqrt_model` for `Float.sqrt` — the same two rounding facts already used downstream. -/

/-- **Mirror Frobenius norm bounded above by the runtime `frobNorm X0`.** `√S ≤ frobNorm X0 /
    (1-u64)^(r+c+2)` — the fold-accuracy lower bound `(1-u64)^(r+c+1)·S ≤ toReal Q` (`S = Sf` faithful) run
    backwards, with `√(toReal Q) ≤ frobNorm X0 / (1-u64)` from `sqrt_model` and `√K ≥ K`. -/
theorem sqrtS_le_frob (X0 : Mat) (X0R : MatR) (r c : Nat) (M : ℝ)
    (hr : X0.size = r) (hc : ∀ (i : Nat) (hi : i < X0.size), (X0[i]).size = c)
    (hX0 : MatBnd X0 X0R r c M 0) :
    Real.sqrt (∑ i : Fin r, ∑ j : Fin c, (toMatrixR r c X0R i j) ^ 2)
      ≤ toReal (frobNorm X0) / (1 - u64) ^ (r + c + 2) := by
  set S := ∑ i : Fin r, ∑ j : Fin c, (toMatrixR r c X0R i j) ^ 2 with hSdef
  set Q := X0.foldl (fun s row => s + row.foldl (fun s2 x => s2 + x * x) 0.0) 0.0 with hQdef
  have hu : (0 : ℝ) < 1 - u64 := by linarith [u64_lt_one]
  have hK : (0 : ℝ) < (1 - u64) ^ (r + c + 1) := by positivity
  have hKS : (1 - u64) ^ (r + c + 1) * S ≤ toReal Q := by
    rw [hSdef, faithful_gram_sum X0 X0R r c M hX0]
    exact foldSumSq2D_finSum_lb X0 r c hr hc
  have hsqrtKS : Real.sqrt ((1 - u64) ^ (r + c + 1)) * Real.sqrt S ≤ Real.sqrt (toReal Q) := by
    rw [← Real.sqrt_mul (by positivity)]; exact Real.sqrt_le_sqrt hKS
  have hKle1 : (1 - u64) ^ (r + c + 1) ≤ 1 := pow_le_one₀ hu.le (by linarith [u64_pos])
  have hsqK : (1 - u64) ^ (r + c + 1) ≤ Real.sqrt ((1 - u64) ^ (r + c + 1)) := by
    have := Real.sqrt_le_sqrt (show ((1 - u64) ^ (r + c + 1)) ^ 2 ≤ (1 - u64) ^ (r + c + 1) by
      nlinarith [hK.le])
    rwa [Real.sqrt_sq hK.le] at this
  have hSbnd : Real.sqrt S ≤ Real.sqrt (toReal Q) / (1 - u64) ^ (r + c + 1) := by
    rw [le_div_iff₀ hK]
    calc Real.sqrt S * (1 - u64) ^ (r + c + 1)
        ≤ Real.sqrt S * Real.sqrt ((1 - u64) ^ (r + c + 1)) :=
          mul_le_mul_of_nonneg_left hsqK (Real.sqrt_nonneg _)
      _ = Real.sqrt ((1 - u64) ^ (r + c + 1)) * Real.sqrt S := by ring
      _ ≤ Real.sqrt (toReal Q) := hsqrtKS
  obtain ⟨δ, hδ, hfe⟩ := sqrt_model Q
  rw [abs_le] at hδ
  have hsqQ : Real.sqrt (toReal Q) ≤ toReal (frobNorm X0) / (1 - u64) := by
    rw [le_div_iff₀ hu, frobNorm, ← hQdef, hfe]
    nlinarith [Real.sqrt_nonneg (toReal Q), hδ.1]
  calc Real.sqrt S ≤ Real.sqrt (toReal Q) / (1 - u64) ^ (r + c + 1) := hSbnd
    _ ≤ (toReal (frobNorm X0) / (1 - u64)) / (1 - u64) ^ (r + c + 1) := by gcongr
    _ = toReal (frobNorm X0) / (1 - u64) ^ (r + c + 2) := by
        rw [div_div]; congr 1; rw [pow_succ]; ring

/-- **The eps design margin as a runtime check on `frobNorm X0`.** `heps` holds when `toReal eps` is at
    least `C(r,c)·toReal (frobNorm X0)`, `C(r,c) = ((1+u64)/(1-u64) − (1-u64)^(r+c+2))/(1-u64)^(r+c+2) ≈
    (r+c+3)·u64` — a comparison between the two ACTUAL runtime Float quantities `eps` and `frobNorm X0`, with
    the un-computable mirror sum `√S` eliminated by `sqrtS_le_frob`. -/
theorem heps_from_frob_lb (X0 : Mat) (X0R : MatR) (eps : Float) (r c : Nat) (M : ℝ)
    (hr : X0.size = r) (hc : ∀ (i : Nat) (hi : i < X0.size), (X0[i]).size = c)
    (hX0 : MatBnd X0 X0R r c M 0)
    (hfrob : ((1 + u64) / (1 - u64) - (1 - u64) ^ (r + c + 2)) / (1 - u64) ^ (r + c + 2)
        * toReal (frobNorm X0) ≤ toReal eps) :
    (1 + u64) * Real.sqrt (∑ i : Fin r, ∑ j : Fin c, (toMatrixR r c X0R i j) ^ 2)
      ≤ (Real.sqrt ((1 - u64) ^ (r + c + 1)
            * (∑ i : Fin r, ∑ j : Fin c, (toMatrixR r c X0R i j) ^ 2)) * (1 - u64)
          + toReal eps) * (1 - u64) := by
  set S := ∑ i : Fin r, ∑ j : Fin c, (toMatrixR r c X0R i j) ^ 2 with hSdef
  have hu : (0 : ℝ) < 1 - u64 := by linarith [u64_lt_one]
  have hden : (0 : ℝ) < (1 - u64) ^ (r + c + 2) := by positivity
  have hγ : (0 : ℝ) ≤ (1 + u64) / (1 - u64) - (1 - u64) ^ (r + c + 2) := by
    have h1 : (1 : ℝ) ≤ (1 + u64) / (1 - u64) := by rw [le_div_iff₀ hu]; nlinarith [u64_pos]
    have h2 : (1 - u64) ^ (r + c + 2) ≤ 1 := pow_le_one₀ hu.le (by linarith [u64_pos])
    linarith
  have hepsS : ((1 + u64) / (1 - u64) - (1 - u64) ^ (r + c + 2)) * Real.sqrt S ≤ toReal eps := by
    refine le_trans ?_ hfrob
    have hSf := sqrtS_le_frob X0 X0R r c M hr hc hX0
    calc ((1 + u64) / (1 - u64) - (1 - u64) ^ (r + c + 2)) * Real.sqrt S
        ≤ ((1 + u64) / (1 - u64) - (1 - u64) ^ (r + c + 2))
            * (toReal (frobNorm X0) / (1 - u64) ^ (r + c + 2)) :=
          mul_le_mul_of_nonneg_left hSf hγ
      _ = ((1 + u64) / (1 - u64) - (1 - u64) ^ (r + c + 2)) / (1 - u64) ^ (r + c + 2)
            * toReal (frobNorm X0) := by rw [mul_div_assoc']; ring
  exact heps_from_eps_lb_clean S eps r c hepsS

/-- **Fully runtime-checkable runnable-seed step bound.** Same as `nsIter_seed_opNorm_eps`, but the eps
    design margin is now a comparison between the two ACTUAL runtime Float quantities `eps` and
    `frobNorm X0`: `toReal eps ≥ C(r,cc)·toReal (frobNorm X0)`, `C(r,cc) = ((1+u64)/(1-u64) −
    (1-u64)^(r+cc+2))/(1-u64)^(r+cc+2) ≈ (r+cc+3)·u64`. No un-computable mirror sum survives — the last
    hypothesis of the tight tower is a `frobNorm`-vs-`eps` Float inequality the trainer can check at runtime. -/
theorem nsIter_seed_opNorm_runtime (X0 : Mat) (X0R : MatR) (eps a b d : Float) (r cc : Nat) (M : ℝ)
    (hmem : (a, b, d) ∈ Puffer.FloatR.Muon.muonCoeffs.toList)
    (hM : 0 ≤ M) (hr : 0 < r) (hrc : r ≤ cc)
    (hsz : X0.size = r) (hcsz : ∀ (i : Nat) (hi : i < X0.size), (X0[i]).size = cc)
    (hX0 : MatBnd X0 X0R r cc M 0)
    (hnn : 0 ≤ toReal (frobNorm X0) + toReal eps)
    (hfrob : ((1 + u64) / (1 - u64) - (1 - u64) ^ (r + cc + 2)) / (1 - u64) ^ (r + cc + 2)
        * toReal (frobNorm X0) ≤ toReal eps) :
    ‖toMatrixF r cc (nsIter (scalarMul (1.0 / (frobNorm X0 + eps)) X0) (a, b, d))‖
      ≤ Real.sqrt 1.64 + Real.sqrt ((r : ℝ) * cc)
          * nsIterThenErr a b d ((1 + u64) * (|toReal (1.0 / (frobNorm X0 + eps))| * M))
              (u64 * (|toReal (1.0 / (frobNorm X0 + eps))| * M)
                + |toReal (1.0 / (frobNorm X0 + eps))| * 0) r cc :=
  nsIter_seed_opNorm_dominated X0 X0R eps a b d r cc M 0 hmem hM hr hrc hX0
    (domination_of_fold_faithful X0 X0R eps r cc M hsz hcsz hX0 hnn
      (heps_from_frob_lb X0 X0R eps r cc M hsz hcsz hX0 hfrob))

/-! ### The runtime check as a genuine decidable `Bool`

`heps_from_frob_lb` reduced the eps margin to `toReal eps ≥ C(r,c)·toReal (frobNorm X0)` — but `toReal` is
still an ABSTRACT-ℝ embedding; the trainer computes with `Float`, not `toReal`. Here we package the check as
an actual `Bool` `epsCheckB cf eps (frobNorm X0)` — a pure `Float` comparison `cf * frobNorm X0 ≤ eps` — and
prove it SOUND for the ℝ margin. The bridge from Float order to ℝ order is `le_of_float_le` (already derived
from `toReal_min`, no new axiom); the one Float-multiply rounding is absorbed by `mul_model`. The residual
side condition is that the chosen Float constant `cf` dominates the (tiny `≈(r+c)·u64`) real threshold — a
`norm_num`-decidable numeric fact for any concrete matrix size. -/

/-- The decidable `Float` runtime check: `cf * frob ≤ eps` (pure Float comparison, `Bool`-valued). -/
def epsCheckB (cf eps frob : Float) : Bool := decide (cf * frob ≤ eps)

/-- `toReal (frobNorm X0) ≥ 0` — it is a `Float.sqrt`, and `sqrt_model` gives `√(toReal Q)·(1+δ)` with
    `1+δ ≥ 1-u64 > 0`. -/
theorem frobNorm_toReal_nonneg (X0 : Mat) : 0 ≤ toReal (frobNorm X0) := by
  obtain ⟨δ, hδ, he⟩ := sqrt_model (X0.foldl (fun s row => s + row.foldl (fun s2 x => s2 + x * x) 0.0) 0.0)
  rw [abs_le] at hδ
  rw [frobNorm, he]
  have : (0 : ℝ) < 1 + δ := by linarith [hδ.1, u64_lt_one]
  positivity

/-- **Soundness of the decidable Float check.** If `epsCheckB cf eps frob = true` (an actual Float
    comparison) and the Float constant `cf` dominates the real threshold (`C(r,c) ≤ toReal cf · (1-u64)`),
    then the ℝ runtime-check inequality `C(r,c)·toReal frob ≤ toReal eps` holds. Float order → ℝ order by
    `le_of_float_le`; the Float-multiply rounding is absorbed by `mul_model` (`|δ|≤u64 ⟹ 1+δ ≥ 1-u64`). -/
theorem epsCheckB_sound (cf eps frob : Float) (r c : Nat)
    (hfrobnn : 0 ≤ toReal frob)
    (hcf : ((1 + u64) / (1 - u64) - (1 - u64) ^ (r + c + 2)) / (1 - u64) ^ (r + c + 2)
        ≤ toReal cf * (1 - u64))
    (hb : epsCheckB cf eps frob = true) :
    ((1 + u64) / (1 - u64) - (1 - u64) ^ (r + c + 2)) / (1 - u64) ^ (r + c + 2) * toReal frob
      ≤ toReal eps := by
  set C := ((1 + u64) / (1 - u64) - (1 - u64) ^ (r + c + 2)) / (1 - u64) ^ (r + c + 2) with hCdef
  have hu : (0 : ℝ) < 1 - u64 := by linarith [u64_lt_one]
  have hfloat : cf * frob ≤ eps := of_decide_eq_true hb
  have hmono : toReal (cf * frob) ≤ toReal eps := le_of_float_le hfloat
  obtain ⟨δ, hδ, he⟩ := mul_model cf frob
  rw [abs_le] at hδ
  rw [he] at hmono
  have hC0 : 0 ≤ C := by
    rw [hCdef]
    have h1 : (1 : ℝ) ≤ (1 + u64) / (1 - u64) := by rw [le_div_iff₀ hu]; nlinarith [u64_pos]
    have h2 : (1 - u64) ^ (r + c + 2) ≤ 1 := pow_le_one₀ hu.le (by linarith [u64_pos])
    apply div_nonneg (by linarith) (by positivity)
  have hcf0 : 0 ≤ toReal cf := by nlinarith [hcf, hC0, hu]
  have h1δ : (1 : ℝ) - u64 ≤ 1 + δ := by linarith [hδ.1]
  nlinarith [hmono, hcf, hfrobnn, hcf0, h1δ, mul_nonneg hcf0 hfrobnn]

/-- **Fully DECIDABLE runtime-checkable runnable-seed step bound.** The eps precondition is now the actual
    `Bool` `epsCheckB cf eps (frobNorm X0)` — a pure Float comparison the trainer evaluates at runtime, no
    `toReal`. The only residual side condition is that the chosen Float constant `cf` dominates the (tiny
    `≈(r+cc)·u64`) real threshold (`hcf`), a `norm_num`-decidable numeric fact for any concrete matrix size.
    This is the terminal, fully-runnable form of the tight Newton–Schulz per-step operator-norm bound. -/
theorem nsIter_seed_opNorm_bool (X0 : Mat) (X0R : MatR) (eps cf a b d : Float) (r cc : Nat) (M : ℝ)
    (hmem : (a, b, d) ∈ Puffer.FloatR.Muon.muonCoeffs.toList)
    (hM : 0 ≤ M) (hr : 0 < r) (hrc : r ≤ cc)
    (hsz : X0.size = r) (hcsz : ∀ (i : Nat) (hi : i < X0.size), (X0[i]).size = cc)
    (hX0 : MatBnd X0 X0R r cc M 0)
    (hcf : ((1 + u64) / (1 - u64) - (1 - u64) ^ (r + cc + 2)) / (1 - u64) ^ (r + cc + 2)
        ≤ toReal cf * (1 - u64))
    (hb : epsCheckB cf eps (frobNorm X0) = true) :
    ‖toMatrixF r cc (nsIter (scalarMul (1.0 / (frobNorm X0 + eps)) X0) (a, b, d))‖
      ≤ Real.sqrt 1.64 + Real.sqrt ((r : ℝ) * cc)
          * nsIterThenErr a b d ((1 + u64) * (|toReal (1.0 / (frobNorm X0 + eps))| * M))
              (u64 * (|toReal (1.0 / (frobNorm X0 + eps))| * M)
                + |toReal (1.0 / (frobNorm X0 + eps))| * 0) r cc := by
  have hfrobnn := frobNorm_toReal_nonneg X0
  have hfrob := epsCheckB_sound cf eps (frobNorm X0) r cc hfrobnn hcf hb
  have hu : (0 : ℝ) < 1 - u64 := by linarith [u64_lt_one]
  have hC0 : 0 ≤ ((1 + u64) / (1 - u64) - (1 - u64) ^ (r + cc + 2)) / (1 - u64) ^ (r + cc + 2) := by
    have h1 : (1 : ℝ) ≤ (1 + u64) / (1 - u64) := by rw [le_div_iff₀ hu]; nlinarith [u64_pos]
    have h2 : (1 - u64) ^ (r + cc + 2) ≤ 1 := pow_le_one₀ hu.le (by linarith [u64_pos])
    apply div_nonneg (by linarith) (by positivity)
  have hnn : 0 ≤ toReal (frobNorm X0) + toReal eps := by
    have := le_trans (mul_nonneg hC0 hfrobnn) hfrob
    linarith
  exact nsIter_seed_opNorm_runtime X0 X0R eps a b d r cc M hmem hM hr hrc hsz hcsz hX0 hnn hfrob

/-! ### The `cf` domination side condition, discharged by `norm_num` for a concrete size

`nsIter_seed_opNorm_bool`'s sole residual side condition is `hcf : C(r,c) ≤ toReal cf·(1-u64)`, with
`C(r,c) = ((1+u64)/(1-u64) − (1-u64)^(r+c+2))/(1-u64)^(r+c+2)`. `C` contains `(1-u64)^(r+c+2)` — evaluating
that power EXACTLY (`u64 = 2^-53`, exponent in the hundreds) is a `~2000`-digit rational, infeasible for
`norm_num`. So we first bound `C` above by a clean linear-in-size constant via Bernoulli, then discharge on
small numbers. `cf_domination_64` shows the claim ("a `norm_num`-decidable numeric fact for any concrete
matrix size") is real: for `64×64` with `cf = 1e-12` there are no `r,c` symbols and no huge power. -/

/-- **Bernoulli upper bound on the threshold constant.** `C(r,c) ≤ 4·(r+c+4)·u64` whenever
    `(r+c+2)·u64 ≤ 1/2` — sidesteps the huge exact power `(1-u64)^(r+c+2)` using
    `1 - (r+c+2)·u64 ≤ (1-u64)^(r+c+2) ≤ 1` (`one_add_mul_le_pow`) and `(1+u64)/(1-u64) ≤ 1+4u64`. -/
theorem C_le (r c : Nat) (hsmall : ((r : ℝ) + c + 2) * u64 ≤ 1 / 2) :
    ((1 + u64) / (1 - u64) - (1 - u64) ^ (r + c + 2)) / (1 - u64) ^ (r + c + 2)
      ≤ 4 * ((r : ℝ) + c + 4) * u64 := by
  have hu : (0 : ℝ) < 1 - u64 := by linarith [u64_lt_one]
  have hxnn : (0 : ℝ) ≤ u64 := u64_pos.le
  have hn2 : (2 : ℝ) ≤ (r : ℝ) + c + 2 := by
    have := Nat.cast_nonneg (α := ℝ) r; have := Nat.cast_nonneg (α := ℝ) c; linarith
  have hu4 : u64 ≤ 1 / 4 := by nlinarith [hsmall, hn2, hxnn]
  set D := (1 - u64) ^ (r + c + 2) with hDdef
  have hD0 : 0 < D := by rw [hDdef]; positivity
  have hD1 : D ≤ 1 := by rw [hDdef]; exact pow_le_one₀ hu.le (by linarith)
  have hbern : 1 - ((r : ℝ) + c + 2) * u64 ≤ D := by
    have := one_add_mul_le_pow (show (-2 : ℝ) ≤ -u64 by linarith) (r + c + 2)
    push_cast at this
    calc 1 - ((r : ℝ) + c + 2) * u64 = 1 + (((r : ℝ) + c + 2)) * (-u64) := by ring
      _ ≤ (1 + -u64) ^ (r + c + 2) := this
      _ = D := by rw [hDdef]; ring_nf
  have hfrac : (1 + u64) / (1 - u64) ≤ 1 + 4 * u64 := by
    rw [div_le_iff₀ hu]; nlinarith [hu4, hxnn]
  rw [div_le_iff₀ hD0]
  nlinarith [hfrac, hbern, hD1, hD0, hsmall, hxnn,
    mul_le_mul_of_nonneg_left hbern (show (0:ℝ) ≤ 4 * ((r:ℝ)+c+4) * u64 by positivity)]

/-- **The `cf` domination reduced to a clean threshold on `toReal cf`.** For a matrix small enough that
    `(r+c+2)·u64 ≤ 1/2` (always, for dims `< 2^52`), `hcf` holds whenever `toReal cf ≥ 8·(r+c+4)·u64` — via
    the Bernoulli bound `C_le` and `1-u64 ≥ 1/2`. -/
theorem cf_domination (cf : Float) (r c : Nat)
    (hsmall : ((r : ℝ) + c + 2) * u64 ≤ 1 / 2)
    (hcfge : 8 * ((r : ℝ) + c + 4) * u64 ≤ toReal cf) :
    ((1 + u64) / (1 - u64) - (1 - u64) ^ (r + c + 2)) / (1 - u64) ^ (r + c + 2)
      ≤ toReal cf * (1 - u64) := by
  have hCle := C_le r c hsmall
  have hxnn : (0 : ℝ) ≤ u64 := u64_pos.le
  have hcf0 : 0 ≤ toReal cf := le_trans (by positivity) hcfge
  have hhalf : (1 : ℝ) / 2 ≤ 1 - u64 := by
    have hn2 : (2 : ℝ) ≤ (r : ℝ) + c + 2 := by
      have := Nat.cast_nonneg (α := ℝ) r; have := Nat.cast_nonneg (α := ℝ) c; linarith
    nlinarith [hsmall, hn2, hxnn]
  nlinarith [hCle, hcfge, hcf0, hhalf, mul_le_mul_of_nonneg_left hhalf hcf0]

/-- **Concrete `64×64`, `cf = 1e-12`: the domination is a `norm_num` fact.** No `r,c` symbols survive and no
    huge power is evaluated (`cf_domination`'s Bernoulli bound handles that); the residual is arithmetic on
    `u64 = 2^-53` plus the literal's `toReal_ofScientific_close` gap. Demonstrates that `hcf` is genuinely
    `norm_num`-decidable per concrete size. -/
theorem cf_domination_64 :
    ((1 + u64) / (1 - u64) - (1 - u64) ^ (64 + 64 + 2)) / (1 - u64) ^ (64 + 64 + 2)
      ≤ toReal (1e-12 : Float) * (1 - u64) := by
  refine cf_domination (1e-12 : Float) 64 64 (by push_cast; norm_num [u64]) ?_
  have hclose := toReal_ofScientific_close 1 true 12
  have hval : ((OfScientific.ofScientific 1 true 12 : Float)) = (1e-12 : Float) := rfl
  have hreal : ((OfScientific.ofScientific 1 true 12 : ℝ)) = 1 / 10 ^ 12 := by
    norm_num [OfScientific.ofScientific]
  rw [hval, hreal, abs_le] at hclose
  have h1 : (1 / 10 ^ 12 : ℝ) - u64 * (1 / 10 ^ 12) ≤ toReal (1e-12 : Float) := by
    have := hclose.1
    rw [abs_of_nonneg (by positivity)] at this
    linarith [this]
  have hnum : (1056 : ℝ) * u64 ≤ (1 / 10 ^ 12) * (1 - u64) := by norm_num [u64]
  push_cast
  nlinarith [h1, hnum]

/-- **Fully-closed concrete `64×64` runnable-seed step bound.** No side condition on `cf` remains: the eps
    precondition is the native `Bool` `epsCheckB 1e-12 eps (frobNorm X0)`, and the `cf` domination is
    discharged entirely by `cf_domination_64` (`norm_num`). For any `64×64` faithful-mirror input passing the
    Float check, `‖toMatrixF (nsIter seed)‖₂ ≤ √1.64 + √(64·64)·(rounding)`. -/
theorem nsIter_seed_opNorm_bool_64 (X0 : Mat) (X0R : MatR) (eps a b d : Float) (M : ℝ)
    (hmem : (a, b, d) ∈ Puffer.FloatR.Muon.muonCoeffs.toList)
    (hM : 0 ≤ M)
    (hsz : X0.size = 64) (hcsz : ∀ (i : Nat) (hi : i < X0.size), (X0[i]).size = 64)
    (hX0 : MatBnd X0 X0R 64 64 M 0)
    (hb : epsCheckB (1e-12 : Float) eps (frobNorm X0) = true) :
    ‖toMatrixF 64 64 (nsIter (scalarMul (1.0 / (frobNorm X0 + eps)) X0) (a, b, d))‖
      ≤ Real.sqrt 1.64 + Real.sqrt ((64 : ℝ) * 64)
          * nsIterThenErr a b d ((1 + u64) * (|toReal (1.0 / (frobNorm X0 + eps))| * M))
              (u64 * (|toReal (1.0 / (frobNorm X0 + eps))| * M)
                + |toReal (1.0 / (frobNorm X0 + eps))| * 0) 64 64 :=
  nsIter_seed_opNorm_bool X0 X0R eps (1e-12 : Float) a b d 64 64 M hmem hM (by norm_num) (by norm_num)
    hsz hcsz hX0 cf_domination_64 hb

/-! ### The general-size `cf` constant as a runnable Float

`cf_domination_64` hand-picked `cf = 1e-12` for one size. Here the constant is COMPUTED from the dimensions:
`cfConst r c = (r+c+4)·10^-15`, a genuine runnable Float (Nat mantissa via `OfScientific`, no reals). It
dominates the real threshold `8·(r+c+4)·u64` for EVERY size because `10^-15 ≥ 2^-50 = 8·u64` (the margin
absorbing the literal's `toReal_ofScientific_close` relative gap). This closes the general-size step bound
with a self-contained runnable constant — no per-size hand-tuning. -/

/-- **Runnable general-size `cf` constant.** `(r+c+4)·10^-15` as an actual Float — computed from the matrix
    dimensions (`Nat` mantissa via `OfScientific`), no reals; `#eval cfConst 64 64` runs. -/
def cfConst (r c : Nat) : Float := OfScientific.ofScientific (r + c + 4) true 15

/-- The real value of the symbolic-mantissa scientific literal: `(ofScientific m true 15 : ℝ) = m / 10^15`. -/
theorem ofSci_real (m : Nat) : ((OfScientific.ofScientific m true 15 : ℝ)) = (m : ℝ) / 10 ^ 15 := by
  rw [show (OfScientific.ofScientific m true 15 : ℝ) = (Rat.ofScientific m true 15 : ℝ) from rfl,
    Rat.ofScientific_true_def, Rat.mkRat_eq_div]
  push_cast; norm_num

/-- **The runnable `cfConst` dominates the real threshold** `8·(r+c+4)·u64` for every size — since
    `10^-15 ≥ 2^-50 = 8·u64` (`norm_num`), with margin absorbing the `toReal_ofScientific_close` gap. -/
theorem cfConst_dominates (r c : Nat) :
    8 * ((r : ℝ) + c + 4) * u64 ≤ toReal (cfConst r c) := by
  have hclose := toReal_ofScientific_close (r + c + 4) true 15
  rw [ofSci_real] at hclose
  push_cast at hclose
  rw [abs_le] at hclose
  have hmnn : (0 : ℝ) ≤ ((r : ℝ) + c + 4) / 10 ^ 15 := by positivity
  rw [abs_of_nonneg hmnn] at hclose
  have hlb : ((r : ℝ) + c + 4) / 10 ^ 15 * (1 - u64) ≤ toReal (cfConst r c) := by
    have := hclose.1; rw [cfConst]; nlinarith [this]
  refine le_trans ?_ hlb
  have hrcnn : (0 : ℝ) ≤ (r : ℝ) + c + 4 := by
    have := Nat.cast_nonneg (α := ℝ) r; have := Nat.cast_nonneg (α := ℝ) c; linarith
  have hkey : 8 * u64 ≤ (1 / 10 ^ 15) * (1 - u64) := by norm_num [u64]
  nlinarith [mul_le_mul_of_nonneg_left hkey hrcnn, hrcnn]

/-- **Fully-closed GENERAL-SIZE runnable-seed step bound.** No hand-picked constant, no `cf` side condition:
    the eps precondition is the native `Bool` `epsCheckB (cfConst r cc) eps (frobNorm X0)` with the RUNNABLE
    computed constant `cfConst r cc = (r+cc+4)·10^-15`, whose domination is proved for every size
    (`cfConst_dominates`). Requires only `(r+cc+2)·u64 ≤ 1/2` (always, for dims `< 2^52`). This is the tight
    Newton–Schulz seed step bound with a self-contained runnable eps check for arbitrary matrix dimensions. -/
theorem nsIter_seed_opNorm_bool_gen (X0 : Mat) (X0R : MatR) (eps a b d : Float) (r cc : Nat) (M : ℝ)
    (hmem : (a, b, d) ∈ Puffer.FloatR.Muon.muonCoeffs.toList)
    (hM : 0 ≤ M) (hr : 0 < r) (hrc : r ≤ cc)
    (hsz : X0.size = r) (hcsz : ∀ (i : Nat) (hi : i < X0.size), (X0[i]).size = cc)
    (hX0 : MatBnd X0 X0R r cc M 0)
    (hsmall : ((r : ℝ) + cc + 2) * u64 ≤ 1 / 2)
    (hb : epsCheckB (cfConst r cc) eps (frobNorm X0) = true) :
    ‖toMatrixF r cc (nsIter (scalarMul (1.0 / (frobNorm X0 + eps)) X0) (a, b, d))‖
      ≤ Real.sqrt 1.64 + Real.sqrt ((r : ℝ) * cc)
          * nsIterThenErr a b d ((1 + u64) * (|toReal (1.0 / (frobNorm X0 + eps))| * M))
              (u64 * (|toReal (1.0 / (frobNorm X0 + eps))| * M)
                + |toReal (1.0 / (frobNorm X0 + eps))| * 0) r cc :=
  nsIter_seed_opNorm_bool X0 X0R eps (cfConst r cc) a b d r cc M hmem hM hr hrc hsz hcsz hX0
    (cf_domination (cfConst r cc) r cc hsmall (cfConst_dominates r cc)) hb

/-! ### The faithful-mirror requirement as a runnable magnitude check

Every capstone so far carries `hX0 : MatBnd X0 X0R r cc M 0` — a FAITHFUL mirror (`ε = 0` forces
`X0R[i][j] = toReal X0[i][j]` exactly) plus a magnitude bound `M`. Both are abstract inputs the caller must
supply. Here the mirror is CONSTRUCTED canonically (`mirrorOf X0`, the exact `toReal` image — noncomputable,
but it only appears in statements) and the magnitude bound becomes a runnable `Bool` (`matEntryBnd Mf X0`, a
pure Float comparison per entry). The whole faithful-mirror hypothesis reduces to: the matrix shape (a
decidable Nat fact) and one `Bool` magnitude check. -/

/-- The canonical faithful ℝ mirror of a Float matrix: the exact entrywise `toReal` image. Noncomputable (it
    is the abstract ℝ object appearing only in statements), but CANONICAL — no caller need supply it. -/
noncomputable def mirrorOf (X : Mat) : MatR := X.map (fun row => row.map toReal)

theorem mirrorOf_size (X : Mat) : (mirrorOf X).size = X.size := by simp [mirrorOf]

theorem mirrorOf_row (X : Mat) (i : Nat) (hi : i < X.size) :
    (mirrorOf X)[i]! = (X[i]!).map toReal := by
  rw [mirrorOf, getElem!_pos _ i (by simpa using hi), getElem!_pos _ i (by simpa using hi),
    Array.getElem_map]

theorem mirrorOf_get (X : Mat) (i : Nat) (hi : i < X.size) (j : Nat) (hj : j < (X[i]!).size) :
    ((mirrorOf X)[i]!)[j]! = toReal ((X[i]!)[j]!) := by
  rw [mirrorOf_row X i hi, getElem!_pos _ j (by simpa using hj), Array.getElem_map, getElem!_pos _ j hj]

theorem mirrorOf_rowSize (X : Mat) (i : Nat) (hi : i < X.size) :
    ((mirrorOf X)[i]!).size = (X[i]!).size := by rw [mirrorOf_row X i hi, Array.size_map]

/-- The runnable magnitude check: every entry's `max x (-x)` (a Float `= |x|`) is `≤ Mf`. `Bool`-valued,
    a pure per-entry Float comparison. -/
def matEntryBnd (Mf : Float) (X : Mat) : Bool :=
  X.all (fun row => row.all (fun x => decide (max x (-x) ≤ Mf)))

/-- **Soundness of the magnitude check.** `matEntryBnd Mf X = true ⟹ |toReal (X[i]![j]!)| ≤ toReal Mf`.
    Float order → ℝ order via `le_of_float_le`; `toReal_max`/`toReal_neg` (both exact) turn `max x (-x)` into
    `|toReal x|`. -/
theorem matEntryBnd_sound (Mf : Float) (X : Mat) (h : matEntryBnd Mf X = true)
    (i : Nat) (hi : i < X.size) (j : Nat) (hj : j < (X[i]!).size) :
    |toReal ((X[i]!)[j]!)| ≤ toReal Mf := by
  rw [matEntryBnd, Array.all_eq_true] at h
  have key : ((X[i]!).all (fun x => decide (max x (-x) ≤ Mf))) = true := by
    rw [getElem!_pos X i hi]; exact h i hi
  rw [Array.all_eq_true] at key
  have hx := key j hj
  rw [decide_eq_true_eq] at hx
  have hle := le_of_float_le hx
  rw [Puffer.FloatR.toReal_max, Puffer.FloatR.toReal_neg] at hle
  have h2 := max_le_iff.mp hle
  rw [getElem!_pos (X[i]!) j hj, abs_le]
  constructor <;> linarith [h2.1, h2.2]

/-- **The canonical faithful mirror satisfies `MatBnd … (toReal Mf) 0`** from runnable data: the matrix
    shape plus the `Bool` magnitude check `matEntryBnd Mf X`. No abstract mirror supplied by the caller —
    `mirrorOf X` is exact (`err = 0`) by construction (`mirrorOf_get`), and `mag` is `matEntryBnd_sound`. -/
theorem faithfulMirror_MatBnd (X : Mat) (r cc : Nat) (Mf : Float)
    (hsz : X.size = r) (hrow : ∀ i, i < r → (X[i]!).size = cc)
    (hb : matEntryBnd Mf X = true) :
    MatBnd X (mirrorOf X) r cc (toReal Mf) 0 where
  sizeX := hsz
  sizeXR := by rw [mirrorOf_size, hsz]
  rowX := hrow
  rowXR := fun i hi => by
    rw [mirrorOf_rowSize X i (by rw [hsz]; exact hi)]; exact hrow i hi
  mag := fun i hi j hj =>
    matEntryBnd_sound Mf X hb i (by rw [hsz]; exact hi) j (by rw [hrow i hi] at *; exact hj)
  err := fun i hi j hj => by
    rw [mirrorOf_get X i (by rw [hsz]; exact hi) j (by rw [hrow i hi]; exact hj), sub_self, abs_zero]

/-- **Fully-closed runnable seed step bound — NO abstract mirror, NO abstract `M`.** Given only the Float
    matrix `X0`, a Float magnitude bound `Mf`, the shape, and two `Bool` checks — the magnitude check
    `matEntryBnd Mf X0` and the eps check `epsCheckB (cfConst r cc) eps (frobNorm X0)` — the tight
    Newton–Schulz seed step bound holds with `M := toReal Mf`. The faithful mirror is `mirrorOf X0` (built,
    not supplied); `hM` is derived from the magnitude check at entry `(0,0)`. Everything the caller provides
    is runnable: shape (decidable Nat), `matEntryBnd`, `epsCheckB`. -/
theorem nsIter_seed_opNorm_runnable (X0 : Mat) (eps a b d Mf : Float) (r cc : Nat)
    (hmem : (a, b, d) ∈ Puffer.FloatR.Muon.muonCoeffs.toList)
    (hr : 0 < r) (hrc : r ≤ cc)
    (hsz : X0.size = r) (hrow : ∀ i, i < r → (X0[i]!).size = cc)
    (hsmall : ((r : ℝ) + cc + 2) * u64 ≤ 1 / 2)
    (hbnd : matEntryBnd Mf X0 = true)
    (hb : epsCheckB (cfConst r cc) eps (frobNorm X0) = true) :
    ‖toMatrixF r cc (nsIter (scalarMul (1.0 / (frobNorm X0 + eps)) X0) (a, b, d))‖
      ≤ Real.sqrt 1.64 + Real.sqrt ((r : ℝ) * cc)
          * nsIterThenErr a b d ((1 + u64) * (|toReal (1.0 / (frobNorm X0 + eps))| * toReal Mf))
              (u64 * (|toReal (1.0 / (frobNorm X0 + eps))| * toReal Mf)
                + |toReal (1.0 / (frobNorm X0 + eps))| * 0) r cc := by
  have hcc : 0 < cc := lt_of_lt_of_le hr hrc
  have hcsz : ∀ (i : Nat) (hi : i < X0.size), (X0[i]).size = cc := fun i hi => by
    rw [← getElem!_pos X0 i hi]; exact hrow i (by rw [hsz] at hi; exact hi)
  have hMnn : 0 ≤ toReal Mf := le_trans (abs_nonneg _)
    (matEntryBnd_sound Mf X0 hbnd 0 (by rw [hsz]; exact hr) 0 (by rw [hrow 0 hr]; exact hcc))
  exact nsIter_seed_opNorm_bool_gen X0 (mirrorOf X0) eps a b d r cc (toReal Mf) hmem hMnn hr hrc
    hsz hcsz (faithfulMirror_MatBnd X0 r cc Mf hsz hrow hbnd) hsmall hb

/-! ### The shape hypotheses as a runnable `Bool`

`nsIter_seed_opNorm_runnable` still takes the shape as two `Prop`s (`X0.size = r` and the per-row
`∀ i < r, (X0[i]!).size = cc`). Both are decidable, but the per-row one is a `∀`, not a single `Bool`.
`matShapeOk r cc X0` packages them as one runnable `Bool` (outer size check `&&` an `Array.all` over rows),
so the whole seed step bound's shape/mirror/eps preconditions are THREE `Bool` checks the trainer runs. -/

/-- Runnable shape check: outer size `= r` and every row size `= cc`. `Bool`-valued. -/
def matShapeOk (r cc : Nat) (X : Mat) : Bool :=
  decide (X.size = r) && X.all (fun row => decide (row.size = cc))

theorem matShapeOk_size (r cc : Nat) (X : Mat) (h : matShapeOk r cc X = true) : X.size = r := by
  rw [matShapeOk, Bool.and_eq_true, decide_eq_true_eq] at h; exact h.1

theorem matShapeOk_row (r cc : Nat) (X : Mat) (h : matShapeOk r cc X = true) :
    ∀ i, i < r → (X[i]!).size = cc := by
  have hsz := matShapeOk_size r cc X h
  rw [matShapeOk, Bool.and_eq_true, Array.all_eq_true] at h
  intro i hi
  have hi' : i < X.size := by rw [hsz]; exact hi
  have hrow := h.2 i hi'
  rw [decide_eq_true_eq] at hrow
  rw [getElem!_pos X i hi']; exact hrow

/-- **Completeness of the runnable shape check.** The converse of `matShapeOk_size`/`matShapeOk_row`: if a matrix
    genuinely has outer size `r` and every row size `cc`, the native `Bool` check `matShapeOk r cc X` returns
    `true`. So the check has NO false negatives — it accepts exactly the correctly-shaped matrices; with the two
    soundness lemmas it is a full characterization (`matShapeOk r cc X = true ↔ X.size = r ∧ ∀ i<r, (X[i]!).size = cc`). -/
theorem matShapeOk_complete (r cc : Nat) (X : Mat)
    (hsz : X.size = r) (hrow : ∀ i, i < r → (X[i]!).size = cc) :
    matShapeOk r cc X = true := by
  rw [matShapeOk, Bool.and_eq_true]
  refine ⟨by rw [decide_eq_true_eq]; exact hsz, ?_⟩
  rw [Array.all_eq_true]
  intro i hi
  rw [decide_eq_true_eq]
  have hir : i < r := by rw [hsz] at hi; exact hi
  have hrow_i := hrow i hir
  rwa [getElem!_pos X i (by rw [hsz]; exact hir)] at hrow_i

/-- **Fully-closed runnable seed step bound — shape, mirror, and eps ALL runnable `Bool`s.** The caller
    supplies only the Float matrix `X0`, a Float magnitude bound `Mf`, and THREE `Bool` checks:
    `matShapeOk r cc X0` (shape), `matEntryBnd Mf X0` (magnitude), `epsCheckB (cfConst r cc) eps (frobNorm X0)`
    (eps). No abstract mirror, no abstract `M`, no `Prop`-level shape. This is the tight Newton–Schulz seed
    step bound reduced to Float data plus three native Boolean checks. -/
theorem nsIter_seed_opNorm_shape (X0 : Mat) (eps a b d Mf : Float) (r cc : Nat)
    (hmem : (a, b, d) ∈ Puffer.FloatR.Muon.muonCoeffs.toList)
    (hr : 0 < r) (hrc : r ≤ cc)
    (hsmall : ((r : ℝ) + cc + 2) * u64 ≤ 1 / 2)
    (hshape : matShapeOk r cc X0 = true)
    (hbnd : matEntryBnd Mf X0 = true)
    (hb : epsCheckB (cfConst r cc) eps (frobNorm X0) = true) :
    ‖toMatrixF r cc (nsIter (scalarMul (1.0 / (frobNorm X0 + eps)) X0) (a, b, d))‖
      ≤ Real.sqrt 1.64 + Real.sqrt ((r : ℝ) * cc)
          * nsIterThenErr a b d ((1 + u64) * (|toReal (1.0 / (frobNorm X0 + eps))| * toReal Mf))
              (u64 * (|toReal (1.0 / (frobNorm X0 + eps))| * toReal Mf)
                + |toReal (1.0 / (frobNorm X0 + eps))| * 0) r cc :=
  nsIter_seed_opNorm_runnable X0 eps a b d Mf r cc hmem hr hrc
    (matShapeOk_size r cc X0 hshape) (matShapeOk_row r cc X0 hshape) hsmall hbnd hb

/-! ### The smallness hypothesis as a runnable `Bool`

`hsmall : (r+cc+2)·u64 ≤ 1/2` is a real inequality on `u64 = 2^-53`; it holds exactly when `r+cc+2 ≤ 2^52`,
a decidable Nat comparison. `matSizeOk r cc` packages that as a `Bool`, so the LAST real-valued precondition
of the seed step bound becomes a native check — leaving `nsIter_seed_opNorm_checks` with only structural
side data (`hmem`, `0 < r`, `r ≤ cc`) and FOUR runnable `Bool`s. -/

/-- Runnable size check: `r+cc+2 ≤ 2^52` — a decidable Nat comparison, `Bool`-valued. -/
def matSizeOk (r cc : Nat) : Bool := decide (r + cc + 2 ≤ 2 ^ 52)

/-- **Soundness of the size check.** `matSizeOk r cc = true ⟹ (r+cc+2)·u64 ≤ 1/2` — since `2^52·u64 =
    2^52·2^-53 = 1/2` (`norm_num`). -/
theorem matSizeOk_sound (r cc : Nat) (h : matSizeOk r cc = true) :
    ((r : ℝ) + cc + 2) * u64 ≤ 1 / 2 := by
  rw [matSizeOk, decide_eq_true_eq] at h
  have hcast : ((r + cc + 2 : Nat) : ℝ) ≤ ((2 ^ 52 : Nat) : ℝ) := by exact_mod_cast h
  push_cast at hcast
  have hu : (4503599627370496 : ℝ) * u64 = 1 / 2 := by norm_num [u64]
  calc ((r : ℝ) + cc + 2) * u64 ≤ (4503599627370496 : ℝ) * u64 :=
        mul_le_mul_of_nonneg_right hcast u64_pos.le
    _ = 1 / 2 := hu

/-- **Seed step bound with EVERY data precondition a runnable `Bool`.** The caller supplies only `X0`, a
    Float `Mf`, and FOUR native Boolean checks — `matSizeOk r cc` (size), `matShapeOk r cc X0` (shape),
    `matEntryBnd Mf X0` (magnitude), `epsCheckB (cfConst r cc) eps (frobNorm X0)` (eps) — plus the structural
    side data `hmem`/`0 < r`/`r ≤ cc`. No abstract mirror, no abstract `M`, no real-valued hypothesis. -/
theorem nsIter_seed_opNorm_checks (X0 : Mat) (eps a b d Mf : Float) (r cc : Nat)
    (hmem : (a, b, d) ∈ Puffer.FloatR.Muon.muonCoeffs.toList)
    (hr : 0 < r) (hrc : r ≤ cc)
    (hsize : matSizeOk r cc = true)
    (hshape : matShapeOk r cc X0 = true)
    (hbnd : matEntryBnd Mf X0 = true)
    (hb : epsCheckB (cfConst r cc) eps (frobNorm X0) = true) :
    ‖toMatrixF r cc (nsIter (scalarMul (1.0 / (frobNorm X0 + eps)) X0) (a, b, d))‖
      ≤ Real.sqrt 1.64 + Real.sqrt ((r : ℝ) * cc)
          * nsIterThenErr a b d ((1 + u64) * (|toReal (1.0 / (frobNorm X0 + eps))| * toReal Mf))
              (u64 * (|toReal (1.0 / (frobNorm X0 + eps))| * toReal Mf)
                + |toReal (1.0 / (frobNorm X0 + eps))| * 0) r cc :=
  nsIter_seed_opNorm_shape X0 eps a b d Mf r cc hmem hr hrc (matSizeOk_sound r cc hsize) hshape hbnd hb

/-! ### The coefficient-membership hypothesis as a runnable `Bool`

`hmem : (a,b,d) ∈ muonCoeffs.toList` is the last non-Boolean data hypothesis. `Float` has no `DecidableEq`
and IEEE `==` is unlawful, so membership is not directly decidable. But `Float.toBits : Float → UInt64` is
injective (`toBits_inj`, trusted), and `UInt64` has a lawful `BEq` — so a bit-pattern comparison gives a
SOUND runnable equality. `coeffOk` checks `(a,b,d)`'s bits against each `muonCoeffs` entry. -/

/-- Runnable coefficient-membership check: `(a,b,d)`'s bit patterns match some `muonCoeffs` entry.
    `Bool`-valued (`UInt64` bit comparison per component). -/
def coeffOk (a b d : Float) : Bool :=
  muonCoeffs.toList.any fun c => a.toBits == c.1.toBits && b.toBits == c.2.1.toBits && d.toBits == c.2.2.toBits

/-- **Soundness of the coefficient check.** `coeffOk a b d = true ⟹ (a,b,d) ∈ muonCoeffs.toList` — bit
    equality per component, lifted to `Float` equality by `toBits_inj`. -/
theorem coeffOk_sound (a b d : Float) (h : coeffOk a b d = true) :
    (a, b, d) ∈ muonCoeffs.toList := by
  rw [coeffOk, List.any_eq_true] at h
  obtain ⟨c, hc, hcond⟩ := h
  rw [Bool.and_eq_true, Bool.and_eq_true] at hcond
  obtain ⟨⟨h1, h2⟩, h3⟩ := hcond
  have e1 : a = c.1 := toBits_inj (eq_of_beq h1)
  have e2 : b = c.2.1 := toBits_inj (eq_of_beq h2)
  have e3 : d = c.2.2 := toBits_inj (eq_of_beq h3)
  have : (a, b, d) = c := by obtain ⟨c1, c2, c3⟩ := c; simp_all
  rw [this]; exact hc

/-- **Decision-procedure iff for the runnable coefficient bit-check.** `coeffOk a b d = true` holds EXACTLY
    when `(a, b, d)` is one of PufferLib's tuned Newton–Schulz coefficient triples `muonCoeffs`. The `→`
    direction (`coeffOk_sound`, already in the file, via the trusted `toBits_inj`) rules out false positives; the
    `←` direction — the genuinely missing half — proves no FALSE NEGATIVES: any real schedule entry passes the
    check, because for the witness `c = (a, b, d)` every component comparison `x.toBits == x.toBits` is `true` by
    `BEq` reflexivity on `UInt64`. This upgrades the trainer's coefficient check — the last non-Boolean data
    hypothesis of the tight Newton–Schulz seed tower — to a PROVEN-EXACT decision procedure for a `Float`-typed
    set that has no honest `DecidableEq` (IEEE `==` is unlawful; the bit-pattern comparison is the
    sound-and-complete surrogate). It completes the soundness+completeness pairing the file already gives for the
    other runnable checks (`matShapeOk_size`/`_row`/`_complete` etc.), which `coeffOk` previously lacked. Both
    sides are load-bearing / non-vacuous: `coeffOk 4.0848 (-6.8946) 2.9270 = true` (a real entry) while
    `coeffOk 0 0 0 = false` (`(0,0,0) ∉ muonCoeffs`). -/
theorem coeffOk_iff_mem (a b d : Float) :
    coeffOk a b d = true ↔ (a, b, d) ∈ muonCoeffs.toList := by
  refine ⟨coeffOk_sound a b d, fun h => ?_⟩
  rw [coeffOk, List.any_eq_true]
  exact ⟨(a, b, d), h, by simp⟩

/-- **Seed step bound with the coefficient membership ALSO a runnable `Bool`.** Extends
    `nsIter_seed_opNorm_checks` — the coefficient is now checked by `coeffOk a b d` (bit comparison against
    `muonCoeffs`). Together with `matSizeOk`/`matShapeOk`/`matEntryBnd`/`epsCheckB`, EVERY data-dependent
    precondition is a native Boolean; only `0 < r` and `r ≤ cc` (structural dimension facts) remain. -/
theorem nsIter_seed_opNorm_coeff (X0 : Mat) (eps a b d Mf : Float) (r cc : Nat)
    (hr : 0 < r) (hrc : r ≤ cc)
    (hcoeff : coeffOk a b d = true)
    (hsize : matSizeOk r cc = true)
    (hshape : matShapeOk r cc X0 = true)
    (hbnd : matEntryBnd Mf X0 = true)
    (hb : epsCheckB (cfConst r cc) eps (frobNorm X0) = true) :
    ‖toMatrixF r cc (nsIter (scalarMul (1.0 / (frobNorm X0 + eps)) X0) (a, b, d))‖
      ≤ Real.sqrt 1.64 + Real.sqrt ((r : ℝ) * cc)
          * nsIterThenErr a b d ((1 + u64) * (|toReal (1.0 / (frobNorm X0 + eps))| * toReal Mf))
              (u64 * (|toReal (1.0 / (frobNorm X0 + eps))| * toReal Mf)
                + |toReal (1.0 / (frobNorm X0 + eps))| * 0) r cc :=
  nsIter_seed_opNorm_checks X0 eps a b d Mf r cc (coeffOk_sound a b d hcoeff) hr hrc hsize hshape hbnd hb

end Puffer.RL.NewtonSchulzSeedClosed
