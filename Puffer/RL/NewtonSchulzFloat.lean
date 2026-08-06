/-
Closing the last residual: the Float→mirror rounding gap. Transports the tight per-step Newton–Schulz
operator-norm bound from the exact-ℝ mirror (`nsIterR`) to the RUNNABLE Float step (`nsIter`).

`nsIterR_opNorm_le_muon` (`MuonCoeffFloat.lean`) bounds `‖toMatrixR (nsIterR XR (a,b,c))‖₂ ≤ √1.64` on
the exact-ℝ mirror. `nsIter_MatBnd_le` (`NewtonSchulzError.lean`) already proves the runnable Float step
is within a `MatBnd` (entrywise magnitude + error `nsIterThenErr`) of that mirror. The one missing piece
is ENTRYWISE-close ⟹ OPERATOR-NORM-close:

  • `l2_opNorm_le_of_entrywise` : `‖A‖₂ ≤ √(r·c)·ε` when every entry `|Aᵢⱼ| ≤ ε`. Proved by reusing the
    spectral bridge — `‖A‖² = ‖AᴴA‖ ≤ maxᵢ λᵢ(AᴴA) ≤ trace(AᴴA) = ∑ᵢⱼ Aᵢⱼ² ≤ r·c·ε²` (eigenvalues of
    the PSD Gram `AᴴA` are nonnegative, so each is `≤` their sum = trace), via
    `opNorm_le_of_eigenvalue_bound` + `trace_eq_sum_eigenvalues` + `Finset.single_le_sum`.

  • `nsIter_opNorm_le_muon` : the payoff — for the actual Float schedule and an input `X` (mirror `XR`,
    `r ≤ c` branch) with `MatBnd X XR r c M ε` and mirror-Gram eigenvalues in `[0,1]`,

        ‖toMatrixF (nsIter X (a,b,c))‖₂  ≤  √1.64  +  √(r·c)·nsIterThenErr(a,b,c,M,ε,r,c).

    The first term is the tight per-step spectral bound (`< 1.2807`, dimension-free); the second is the
    Float rounding, exactly the accumulated `MatBnd` error carried entrywise into the operator norm by
    `l2_opNorm_le_of_entrywise`. Triangle inequality on `toMatrixF (nsIter …) = toMatrixR (nsIterR …) +
    (toMatrixF − toMatrixR)`.

Axiom-clean modulo the trusted Float base (`toReal`, `add_model`, `mul_model`, `toReal_neg`,
`toReal_ofScientific_close`, `toReal_zeroLit`). This is the FINAL gap of the tight Newton–Schulz tower:
the O(1) per-step operator-norm bound now holds for the trainer's ACTUAL runnable Float iterate, with the
Float rounding made fully explicit (no `sorry`, no unquantified hand-wave). The only ingredient still
supplied as a hypothesis is the `[0,1]` mirror-Gram spectrum precondition — the effect of `newtonSchulz`'s
Frobenius-normalization seed, an input property rather than a step property.
-/
import Mathlib
import Puffer.RL.SpectralBridge
import Puffer.RL.MuonCoeffFloat

namespace Puffer.RL.NewtonSchulzFloat

open scoped Matrix Matrix.Norms.L2Operator
open Matrix
open Puffer.RL.SpectralBridge (opNorm_le_of_eigenvalue_bound)
open Puffer.RL.NewtonSchulzError
open Puffer.RL.MatrixEmbed (toMatrixR toMatrixF toMatrixF_sub_toMatrixR_entry)
open Puffer.RL.MuonMatrixRuntime (nsIter)
open Puffer.FloatR.Muon (Mat scalarMul frobNorm)
open Puffer.FloatR (u64 toReal div_model add_model sqrt_model toReal_oneLit u64_lt_one u64_pos)
open Puffer.RL.MuonCoeffFloat (nsIterR_opNorm_le_muon)

/-- **Entrywise ⟹ operator norm.** If every entry of `A` has magnitude `≤ ε`, its l2 operator norm is
    `≤ √(r·c)·ε`. Proof: `‖A‖² = ‖AᴴA‖ ≤ maxᵢ λᵢ(AᴴA) ≤ trace(AᴴA) = ∑ᵢⱼ Aᵢⱼ² ≤ r·c·ε²`, reusing the
    spectral bridge (eigenvalues of the PSD Gram `AᴴA` are `≥ 0`, hence each `≤` their sum = trace). -/
theorem l2_opNorm_le_of_entrywise {r cc : Nat} [Nonempty (Fin cc)]
    (A : Matrix (Fin r) (Fin cc) ℝ) (ε : ℝ) (hε : 0 ≤ ε)
    (h : ∀ i j, |A i j| ≤ ε) : ‖A‖ ≤ Real.sqrt ((r : ℝ) * cc) * ε := by
  have hHerm := isHermitian_conjTranspose_mul_self A
  have htrace : (Aᴴ * A).trace = ∑ j, ∑ i, (A i j) ^ 2 := by
    simp only [Matrix.trace, Matrix.diag_apply, Matrix.mul_apply, Matrix.conjTranspose_apply,
      star_trivial]
    congr 1; funext j; congr 1; funext i; rw [sq]
  have hsum : ∑ i, hHerm.eigenvalues i = (Aᴴ * A).trace := by
    rw [hHerm.trace_eq_sum_eigenvalues]; simp
  have htracebnd : (Aᴴ * A).trace ≤ (r : ℝ) * cc * ε ^ 2 := by
    rw [htrace]
    calc ∑ j : Fin cc, ∑ i : Fin r, (A i j) ^ 2
        ≤ ∑ _j : Fin cc, ∑ _i : Fin r, ε ^ 2 := by
          apply Finset.sum_le_sum; intro j _; apply Finset.sum_le_sum; intro i _
          nlinarith [h i j, abs_nonneg (A i j), sq_abs (A i j)]
      _ = (r : ℝ) * cc * ε ^ 2 := by
          simp [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]; ring
  have heval : ∀ i, |hHerm.eigenvalues i| ≤ (r : ℝ) * cc * ε ^ 2 := by
    intro i
    rw [abs_of_nonneg (eigenvalues_conjTranspose_mul_self_nonneg A i)]
    calc hHerm.eigenvalues i
        ≤ ∑ j, hHerm.eigenvalues j :=
          Finset.single_le_sum (fun j _ => eigenvalues_conjTranspose_mul_self_nonneg A j)
            (Finset.mem_univ i)
      _ = (Aᴴ * A).trace := hsum
      _ ≤ (r : ℝ) * cc * ε ^ 2 := htracebnd
  have hgram : ‖Aᴴ * A‖ ≤ (r : ℝ) * cc * ε ^ 2 :=
    opNorm_le_of_eigenvalue_bound hHerm (by positivity) heval
  have hAsq : ‖A‖ ^ 2 ≤ (r : ℝ) * cc * ε ^ 2 := by
    rw [sq, ← l2_opNorm_conjTranspose_mul_self]; exact hgram
  have := Real.sqrt_le_sqrt hAsq
  rwa [Real.sqrt_sq (norm_nonneg A), Real.sqrt_mul (by positivity), Real.sqrt_sq hε] at this

/-- **One runnable Float Newton–Schulz step's operator norm.** For the actual Float schedule and an
    input `X` (mirror `XR`, `r ≤ c` branch) with `MatBnd X XR r c M ε` and mirror-Gram eigenvalues in
    `[0,1]`, `‖toMatrixF (nsIter X (a,b,d))‖₂ ≤ √1.64 + √(r·c)·nsIterThenErr` — the tight per-step
    spectral bound (`< 1.2807`) plus the explicit accumulated Float rounding. -/
theorem nsIter_opNorm_le_muon (X : Mat) (XR : MatR) (a b d : Float) (r c : Nat) (M ε : ℝ)
    (hmem : (a, b, d) ∈ Puffer.FloatR.Muon.muonCoeffs.toList)
    (hM : 0 ≤ M) (hr : 0 < r) (hrc : r ≤ c) (hX : MatBnd X XR r c M ε)
    (hev : ∀ i, 0 ≤ (isHermitian_mul_conjTranspose_self (toMatrixR r c XR)).eigenvalues i
        ∧ (isHermitian_mul_conjTranspose_self (toMatrixR r c XR)).eigenvalues i ≤ 1) :
    ‖toMatrixF r c (nsIter X (a, b, d))‖
      ≤ Real.sqrt 1.64 + Real.sqrt ((r : ℝ) * c) * nsIterThenErr a b d M ε r c := by
  have hc : 0 < c := lt_of_lt_of_le hr hrc
  have : Nonempty (Fin c) := ⟨⟨0, hc⟩⟩
  have hstep := nsIter_MatBnd_le X XR r c M ε a b d hM hr hrc hX
  have hentry : ∀ i j, |toMatrixF r c (nsIter X (a, b, d)) i j
      - toMatrixR r c (nsIterR XR (a, b, d)) i j| ≤ nsIterThenErr a b d M ε r c :=
    fun i j => toMatrixF_sub_toMatrixR_entry _ _ r c _ _ hstep i j
  have hεnn : 0 ≤ nsIterThenErr a b d M ε r c :=
    le_trans (abs_nonneg _) (hentry ⟨0, hr⟩ ⟨0, hc⟩)
  have hdiff : ‖toMatrixF r c (nsIter X (a, b, d)) - toMatrixR r c (nsIterR XR (a, b, d))‖
      ≤ Real.sqrt ((r : ℝ) * c) * nsIterThenErr a b d M ε r c := by
    refine l2_opNorm_le_of_entrywise _ _ hεnn (fun i j => ?_)
    rw [Matrix.sub_apply]; exact hentry i j
  have hmirror : ‖toMatrixR r c (nsIterR XR (a, b, d))‖ ≤ Real.sqrt 1.64 :=
    nsIterR_opNorm_le_muon XR r c hmem hX.sizeXR hX.rowXR hr hrc hev
  calc ‖toMatrixF r c (nsIter X (a, b, d))‖
      = ‖toMatrixR r c (nsIterR XR (a, b, d))
          + (toMatrixF r c (nsIter X (a, b, d)) - toMatrixR r c (nsIterR XR (a, b, d)))‖ := by
        rw [add_sub_cancel]
    _ ≤ ‖toMatrixR r c (nsIterR XR (a, b, d))‖
          + ‖toMatrixF r c (nsIter X (a, b, d)) - toMatrixR r c (nsIterR XR (a, b, d))‖ :=
        norm_add_le _ _
    _ ≤ Real.sqrt 1.64 + Real.sqrt ((r : ℝ) * c) * nsIterThenErr a b d M ε r c :=
        add_le_add hmirror hdiff

/-! ### Discharging the `[0,1]` Gram-spectrum precondition from Frobenius normalization

The remaining hypothesis of `nsIter_opNorm_le_muon` is that the mirror-Gram `MX·MXᴴ` has all eigenvalues
in `[0,1]`. This is an INPUT property: it is exactly what `newtonSchulz`'s Frobenius-normalization seed
`X ← (‖X₀‖_F + eps)⁻¹ · X₀` guarantees. Below: each Gram eigenvalue is `≤` the squared Frobenius norm
(so `Frobenius² ≤ 1 ⟹` spectrum `⊆ [0,1]`), and the exact-ℝ normalization achieves `Frobenius² ≤ 1`. -/

/-- Each eigenvalue of the Gram matrix `A·Aᴴ` is `≤` the squared Frobenius norm `∑ᵢⱼ Aᵢⱼ²`
    (eigenvalues of the PSD Gram are `≥ 0`, so each is `≤` their sum = `trace(A·Aᴴ) = ∑ᵢⱼ Aᵢⱼ²`). -/
theorem gram_eigenvalue_le_frobenius_sq {r c : Nat} (A : Matrix (Fin r) (Fin c) ℝ) (i : Fin r) :
    (isHermitian_mul_conjTranspose_self A).eigenvalues i ≤ ∑ i, ∑ j, (A i j) ^ 2 := by
  have hHerm := isHermitian_mul_conjTranspose_self A
  have htrace : (A * Aᴴ).trace = ∑ i, ∑ j, (A i j) ^ 2 := by
    simp only [Matrix.trace, Matrix.diag_apply, Matrix.mul_apply, Matrix.conjTranspose_apply,
      star_trivial]
    congr 1; funext i; congr 1; funext j; rw [sq]
  calc hHerm.eigenvalues i
      ≤ ∑ j, hHerm.eigenvalues j :=
        Finset.single_le_sum (fun j _ => eigenvalues_self_mul_conjTranspose_nonneg A j)
          (Finset.mem_univ i)
    _ = (A * Aᴴ).trace := by rw [hHerm.trace_eq_sum_eigenvalues]; simp
    _ = ∑ i, ∑ j, (A i j) ^ 2 := htrace

/-- `Frobenius² ≤ 1 ⟹` the Gram matrix `A·Aᴴ` has all eigenvalues in `[0,1]` — exactly the precondition
    of `nsIter_opNorm_le_muon` / `nsIterR_opNorm_le_muon`. -/
theorem gram_spectrum_subset_unit {r c : Nat} (A : Matrix (Fin r) (Fin c) ℝ)
    (hF : ∑ i, ∑ j, (A i j) ^ 2 ≤ 1) :
    ∀ i, 0 ≤ (isHermitian_mul_conjTranspose_self A).eigenvalues i
       ∧ (isHermitian_mul_conjTranspose_self A).eigenvalues i ≤ 1 :=
  fun i => ⟨eigenvalues_self_mul_conjTranspose_nonneg A i,
            le_trans (gram_eigenvalue_le_frobenius_sq A i) hF⟩

/-- **Exact-ℝ Frobenius normalization.** Scaling a real matrix `M` by `(‖M‖_F + eps)⁻¹` (`eps ≥ 0`)
    yields `Frobenius² ≤ 1` — the seed transformation of `newtonSchulz`, on the exact-ℝ side. -/
theorem frobenius_normalized_le_one {r c : Nat} (M : Matrix (Fin r) (Fin c) ℝ) (eps : ℝ)
    (heps : 0 ≤ eps) :
    ∑ i, ∑ j, ((Real.sqrt (∑ i, ∑ j, (M i j) ^ 2) + eps)⁻¹ * M i j) ^ 2 ≤ 1 := by
  set S := ∑ i, ∑ j, (M i j) ^ 2 with hSdef
  have hSnn : 0 ≤ S := by
    rw [hSdef]; apply Finset.sum_nonneg; intro i _; apply Finset.sum_nonneg; intro j _; positivity
  set F := Real.sqrt S with hFdef
  have hFnn : 0 ≤ F := Real.sqrt_nonneg _
  have hFsq : F ^ 2 = S := by rw [hFdef, sq, Real.mul_self_sqrt hSnn]
  have hpull : ∑ i, ∑ j, ((F + eps)⁻¹ * M i j) ^ 2 = (F + eps)⁻¹ ^ 2 * S := by
    rw [hSdef, Finset.mul_sum]; congr 1; funext i; rw [Finset.mul_sum]; congr 1; funext j
    rw [mul_pow]
  rw [hpull]
  rcases eq_or_lt_of_le (add_nonneg hFnn heps) with hz | hpos
  · rw [← hz]; simp
  · rw [← hFsq, inv_pow, ← div_eq_inv_mul, div_le_one (by positivity)]
    nlinarith [hFnn, heps, mul_nonneg hFnn heps]

/-- **Runnable Float step bound from a Frobenius-normalized input.** Same as `nsIter_opNorm_le_muon`
    but with the `[0,1]` Gram-spectrum precondition replaced by the natural, directly-checkable Frobenius
    bound `∑ᵢⱼ (toMatrixR XR)ᵢⱼ² ≤ 1` — which `frobenius_normalized_le_one` supplies for a normalized
    seed. This closes the precondition as an INPUT property of `newtonSchulz`'s Frobenius-normalization. -/
theorem nsIter_opNorm_le_muon_frob (X : Mat) (XR : MatR) (a b d : Float) (r c : Nat) (M ε : ℝ)
    (hmem : (a, b, d) ∈ Puffer.FloatR.Muon.muonCoeffs.toList)
    (hM : 0 ≤ M) (hr : 0 < r) (hrc : r ≤ c) (hX : MatBnd X XR r c M ε)
    (hF : ∑ i, ∑ j, (toMatrixR r c XR i j) ^ 2 ≤ 1) :
    ‖toMatrixF r c (nsIter X (a, b, d))‖
      ≤ Real.sqrt 1.64 + Real.sqrt ((r : ℝ) * c) * nsIterThenErr a b d M ε r c :=
  nsIter_opNorm_le_muon X XR a b d r c M ε hmem hM hr hrc hX
    (gram_spectrum_subset_unit (toMatrixR r c XR) hF)

/-! ### The seed scale-rounding sliver

`newtonSchulz`'s seed is `scalarMul (1.0/(‖X₀‖_F + eps)) X₀` — a Float scalar multiply whose scale is a
ROUNDED reciprocal Frobenius norm. Its exact-ℝ mirror (from `scalarMul_MatBnd`) is `scalarMulR (toReal
(1.0/(‖X₀‖_F+eps))) X₀R` — EXACT real scaling (no per-entry rounding on the mirror side). So the seed's
mirror Frobenius² is exactly `s²·∑X₀R²` with `s = toReal (1.0/(‖X₀‖_F+eps))`, and the whole seed sliver
reduces to the single SCALAR condition `s²·∑X₀R² ≤ 1` on the rounded scale — no matrix rounding remains.
That scalar condition is the design property of the normalization (`eps` chosen so the rounded reciprocal
does not overshoot); its `toReal`-of-`frobNorm`/division rounding is a 1-D fact for `div_model`+`sqrt_model`,
outside the matrix mathematics. -/

/-- Exact Frobenius² scaling of the mirror seed: `∑ᵢⱼ (s·Xᵢⱼ)² = s²·∑ᵢⱼ Xᵢⱼ²` (mirror mult is exact). -/
theorem scalarMulR_frobenius_sq (s : ℝ) (X : MatR) (r c : Nat)
    (hX : X.size = r) (hXrow : ∀ i, i < r → (X[i]!).size = c) :
    ∑ i, ∑ j, (toMatrixR r c (scalarMulR s X) i j) ^ 2
      = s ^ 2 * ∑ i, ∑ j, (toMatrixR r c X i j) ^ 2 := by
  rw [Finset.mul_sum]; congr 1; funext i; rw [Finset.mul_sum]; congr 1; funext j
  have hi : (i : Nat) < X.size := by rw [hX]; exact i.2
  have hj : (j : Nat) < (X[i.1]!).size := by rw [hXrow i.1 (by rw [hX] at hi; exact hi)]; exact j.2
  show (toMatrixR r c (scalarMulR s X) i j) ^ 2 = s ^ 2 * (toMatrixR r c X i j) ^ 2
  simp only [toMatrixR, Matrix.of_apply]
  rw [scalarMulR_getElem s X i.1 j.1 hi hj]; ring

/-- The mirror seed's Gram spectrum `⊆ [0,1]` from the SCALAR scale condition `s²·∑X² ≤ 1`. -/
theorem seed_gram_spectrum (s : ℝ) (X : MatR) (r c : Nat)
    (hX : X.size = r) (hXrow : ∀ i, i < r → (X[i]!).size = c)
    (hscale : s ^ 2 * ∑ i, ∑ j, (toMatrixR r c X i j) ^ 2 ≤ 1) :
    ∀ i, 0 ≤ (isHermitian_mul_conjTranspose_self (toMatrixR r c (scalarMulR s X))).eigenvalues i
       ∧ (isHermitian_mul_conjTranspose_self (toMatrixR r c (scalarMulR s X))).eigenvalues i ≤ 1 :=
  gram_spectrum_subset_unit (toMatrixR r c (scalarMulR s X))
    (by rw [scalarMulR_frobenius_sq s X r c hX hXrow]; exact hscale)

/-- **Runnable Float Newton–Schulz step on the normalized seed.** The per-step operator-norm bound holds
    for the Float seed `scalarMul c X0`, with the `[0,1]` Gram precondition reduced to the single SCALAR
    scale condition `(toReal c)²·∑X0Rᵢⱼ² ≤ 1` (the design property of the Frobenius normalization). The
    error term carries the seed's `scalarMul_MatBnd` magnitude/error inflation. -/
theorem nsIter_seed_opNorm (X0 : Mat) (X0R : MatR) (c a b d : Float) (r cc : Nat) (M ε : ℝ)
    (hmem : (a, b, d) ∈ Puffer.FloatR.Muon.muonCoeffs.toList)
    (hM : 0 ≤ M) (hr : 0 < r) (hrc : r ≤ cc) (hX0 : MatBnd X0 X0R r cc M ε)
    (hscale : (toReal c) ^ 2 * ∑ i, ∑ j, (toMatrixR r cc X0R i j) ^ 2 ≤ 1) :
    ‖toMatrixF r cc (nsIter (scalarMul c X0) (a, b, d))‖
      ≤ Real.sqrt 1.64 + Real.sqrt ((r : ℝ) * cc)
          * nsIterThenErr a b d ((1 + u64) * (|toReal c| * M))
              (u64 * (|toReal c| * M) + |toReal c| * ε) r cc :=
  nsIter_opNorm_le_muon (scalarMul c X0) (scalarMulR (toReal c) X0R) a b d r cc _ _ hmem
    (mul_nonneg one_add_u64_nonneg (mul_nonneg (abs_nonneg _) hM)) hr hrc
    (scalarMul_MatBnd c X0 X0R r cc M ε hM hX0)
    (seed_gram_spectrum (toReal c) X0R r cc hX0.sizeXR hX0.rowXR hscale)

/-- **The scalar seed condition, discharged under denominator domination.** The rounded reciprocal scale
    `1.0/(‖X₀‖_F + eps)` keeps the seed inside the unit ball (`(toReal scale)²·S ≤ 1`) exactly when the
    real value of the computed denominator dominates the true mirror Frobenius norm `√S` by the
    division-rounding safety factor `(1+u64)`: `(1+u64)·√S ≤ toReal (‖X₀‖_F + eps)`. This is the design
    role of `eps` — it must cover the `frobNorm` round-down plus the reciprocal's `(1+u64)` factor. The
    residual is now a purely SCALAR `div_model` fact (proved) plus this one interpretable, runtime-checkable
    inequality on `frobNorm + eps` — no matrix content, no unquantified hand-wave. -/
theorem seed_scale_le_one (X0 : Mat) (eps : Float) (S : ℝ) (hS : 0 ≤ S)
    (hdom : (1 + u64) * Real.sqrt S ≤ toReal (frobNorm X0 + eps)) :
    (toReal (1.0 / (frobNorm X0 + eps))) ^ 2 * S ≤ 1 := by
  obtain ⟨δ, hδ, heq⟩ := div_model 1.0 (frobNorm X0 + eps)
  rw [heq, toReal_oneLit]
  set D := toReal (frobNorm X0 + eps) with hDdef
  have hFnn : 0 ≤ Real.sqrt S := Real.sqrt_nonneg _
  have hF : Real.sqrt S ^ 2 = S := Real.sq_sqrt hS
  rw [abs_le] at hδ
  have hDnn : (0 : ℝ) ≤ D := le_trans (mul_nonneg (by linarith [u64_pos]) hFnn) hdom
  have hkey : (1 + δ) * Real.sqrt S ≤ D :=
    le_trans (mul_le_mul_of_nonneg_right (by linarith [hδ.2]) hFnn) hdom
  have h1δ : 0 ≤ 1 + δ := by linarith [hδ.1, u64_lt_one]
  rcases eq_or_lt_of_le hDnn with hD0 | hDpos
  · rw [← hD0]; simp
  · rw [div_mul_eq_mul_div, div_pow, div_mul_eq_mul_div, div_le_one (by positivity), ← hF]
    nlinarith [hkey, h1δ, hFnn, mul_nonneg h1δ hFnn]

/-- **Fully-reduced runnable-seed step bound.** For the actual Float `newtonSchulz` seed
    `scalarMul (1.0/(‖X₀‖_F+eps)) X₀`, the per-step operator norm is `≤ √1.64 + √(r·c)·(rounding)`
    (O(1), dimension-free), with the ONLY remaining hypothesis the single interpretable domination
    inequality `(1+u64)·√(∑(toMatrixR X₀R)²) ≤ toReal (‖X₀‖_F + eps)` — the design condition on `eps`. -/
theorem nsIter_seed_opNorm_dominated (X0 : Mat) (X0R : MatR) (eps a b d : Float) (r cc : Nat) (M ε : ℝ)
    (hmem : (a, b, d) ∈ Puffer.FloatR.Muon.muonCoeffs.toList)
    (hM : 0 ≤ M) (hr : 0 < r) (hrc : r ≤ cc) (hX0 : MatBnd X0 X0R r cc M ε)
    (hdom : (1 + u64) * Real.sqrt (∑ i, ∑ j, (toMatrixR r cc X0R i j) ^ 2)
      ≤ toReal (frobNorm X0 + eps)) :
    ‖toMatrixF r cc (nsIter (scalarMul (1.0 / (frobNorm X0 + eps)) X0) (a, b, d))‖
      ≤ Real.sqrt 1.64 + Real.sqrt ((r : ℝ) * cc)
          * nsIterThenErr a b d ((1 + u64) * (|toReal (1.0 / (frobNorm X0 + eps))| * M))
              (u64 * (|toReal (1.0 / (frobNorm X0 + eps))| * M)
                + |toReal (1.0 / (frobNorm X0 + eps))| * ε) r cc :=
  nsIter_seed_opNorm X0 X0R (1.0 / (frobNorm X0 + eps)) a b d r cc M ε hmem hM hr hrc hX0
    (seed_scale_le_one X0 eps _
      (Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => sq_nonneg _) hdom)

/-! ### Grinding the eps-domination inequality into its primitive `frobNorm` layers

The domination hypothesis `(1+u64)·√S ≤ toReal (‖X₀‖_F + eps)` is peeled through the two rounding
layers of `frobNorm X₀ + eps` — the outer `add_model` and `frobNorm`'s `sqrt_model` — reducing it to a
condition on the Float FOLD sum-of-squares `Q` (`frobNorm X₀ = Float.sqrt Q`) and `toReal eps`. The
irreducible remainder is then the fold accuracy `toReal Q` vs the mirror `S` (a scalar sum-of-products
sub-tower, analogous to the matmul error tower) together with the `eps` design margin. -/

/-- Peel the outer `add_model`: domination reduces to a condition on the pre-final-rounding real
    denominator `toReal(frobNorm X₀) + toReal eps`. -/
theorem domination_of_denom_real (X0 : Mat) (eps : Float) (S : ℝ)
    (hnn : 0 ≤ toReal (frobNorm X0) + toReal eps)
    (hcond : (1 + u64) * Real.sqrt S ≤ (toReal (frobNorm X0) + toReal eps) * (1 - u64)) :
    (1 + u64) * Real.sqrt S ≤ toReal (frobNorm X0 + eps) := by
  obtain ⟨δ, hδ, heq⟩ := add_model (frobNorm X0) eps
  rw [heq]; rw [abs_le] at hδ
  calc (1 + u64) * Real.sqrt S
      ≤ (toReal (frobNorm X0) + toReal eps) * (1 - u64) := hcond
    _ ≤ (toReal (frobNorm X0) + toReal eps) * (1 + δ) :=
        mul_le_mul_of_nonneg_left (by linarith [hδ.1]) hnn

/-- `√(toReal a)·(1-u64) ≤ toReal (Float.sqrt a)` — the lower-bound half of `sqrt_model`. -/
theorem sqrt_lb (a : Float) : Real.sqrt (toReal a) * (1 - u64) ≤ toReal (Float.sqrt a) := by
  obtain ⟨δ, hδ, heq⟩ := sqrt_model a
  rw [heq]; rw [abs_le] at hδ
  exact mul_le_mul_of_nonneg_left (by linarith [hδ.1]) (Real.sqrt_nonneg _)

/-- Peel BOTH the outer add rounding and `frobNorm`'s sqrt rounding: domination reduces to a condition on
    the Float FOLD sum-of-squares `Q` (with `frobNorm X₀ = Float.sqrt Q`, provable by `rfl`). -/
theorem domination_of_fold (X0 : Mat) (eps : Float) (S : ℝ) (Q : Float)
    (hQdef : frobNorm X0 = Float.sqrt Q)
    (hnn : 0 ≤ toReal (frobNorm X0) + toReal eps)
    (hcond : (1 + u64) * Real.sqrt S
      ≤ (Real.sqrt (toReal Q) * (1 - u64) + toReal eps) * (1 - u64)) :
    (1 + u64) * Real.sqrt S ≤ toReal (frobNorm X0 + eps) := by
  refine domination_of_denom_real X0 eps S hnn (le_trans hcond ?_)
  have hsqrt : Real.sqrt (toReal Q) * (1 - u64) ≤ toReal (frobNorm X0) := by
    rw [hQdef]; exact sqrt_lb Q
  exact mul_le_mul_of_nonneg_right (by linarith [hsqrt]) (by linarith [u64_lt_one])

/-- **Runnable-seed step bound, domination reduced to the Float fold-sum.** The per-step operator norm
    of the runnable Float seed is `≤ √1.64 + √(r·c)·(rounding)` (O(1), dimension-free), with the seed
    condition now expressed on the Float fold sum-of-squares `Q` (`frobNorm X₀ = Float.sqrt Q`) and
    `toReal eps` — both rounding layers of `frobNorm + eps` peeled. -/
theorem nsIter_seed_opNorm_fold (X0 : Mat) (X0R : MatR) (eps a b d : Float) (r cc : Nat) (M ε : ℝ)
    (Q : Float) (hmem : (a, b, d) ∈ Puffer.FloatR.Muon.muonCoeffs.toList)
    (hM : 0 ≤ M) (hr : 0 < r) (hrc : r ≤ cc) (hX0 : MatBnd X0 X0R r cc M ε)
    (hQdef : frobNorm X0 = Float.sqrt Q)
    (hnn : 0 ≤ toReal (frobNorm X0) + toReal eps)
    (hcond : (1 + u64) * Real.sqrt (∑ i, ∑ j, (toMatrixR r cc X0R i j) ^ 2)
      ≤ (Real.sqrt (toReal Q) * (1 - u64) + toReal eps) * (1 - u64)) :
    ‖toMatrixF r cc (nsIter (scalarMul (1.0 / (frobNorm X0 + eps)) X0) (a, b, d))‖
      ≤ Real.sqrt 1.64 + Real.sqrt ((r : ℝ) * cc)
          * nsIterThenErr a b d ((1 + u64) * (|toReal (1.0 / (frobNorm X0 + eps))| * M))
              (u64 * (|toReal (1.0 / (frobNorm X0 + eps))| * M)
                + |toReal (1.0 / (frobNorm X0 + eps))| * ε) r cc :=
  nsIter_seed_opNorm_dominated X0 X0R eps a b d r cc M ε hmem hM hr hrc hX0
    (domination_of_fold X0 eps _ Q hQdef hnn hcond)

end Puffer.RL.NewtonSchulzFloat
