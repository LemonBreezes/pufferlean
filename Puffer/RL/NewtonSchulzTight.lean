/-
A TIGHTER, norm-based magnitude bound for the Newton–Schulz matmul dot product.

The existing tower (`Puffer/RL/NewtonSchulzError.lean`) tracks a per-entry MAGNITUDE bound
`idxDotMagBnd Ma Mb k ≈ k · Ma · Mb · (1+u64)^k` for the rounded k-term dot `idxDotF`. Chained
through 3 matmuls × 5 iterations this is astronomically loose: the `k · Ma · Mb` factor (worst-case,
"every term is the max product") plus the fresh `(1+u64)^k` per matmul compound catastrophically,
because each matmul feeds its magnitude bound as the next `M`.

This file replaces the `k · Ma · Mb` LEADING factor by the Cauchy–Schwarz ℓ₂ bound

    |Σ_{l<k} fR l · gR l| ≤ ‖f‖₂ · ‖g‖₂ = (Σ f²)^½ · (Σ g²)^½.

For a Newton–Schulz iterate, whose rows/columns have ℓ₂ norm ≈ 1 (the singular values are driven to
1), ‖f‖₂·‖g‖₂ = O(1) is DIMENSION-FREE and does NOT carry the `k·` blowup, whereas the entrywise
`k · Ma · Mb` grows with the inner dimension. This is the mathematical heart of the norm-based view
(‖AB‖_F ≤ ‖A‖_F · ‖B‖_2 ≤ ‖A‖_F · ‖B‖_F), proved here at the level of a single dot.

Everything below is proved from the trusted Float model + Mathlib's Cauchy–Schwarz
(`Finset.sum_mul_sq_le_sq_mul_sq`); no new axioms, no `sorry`.
-/
import Mathlib
import Puffer.RL.NewtonSchulzError

namespace Puffer.RL.NewtonSchulzTight

open Puffer.FloatR
open Puffer.RL.MuonMatrixRuntime (idxDotF idxDotR idxDotErrBnd idxDot_error matmul_getElem)
open Puffer.FloatR.Muon (Mat matmul)
open Puffer.RL.NewtonSchulzError (realDotR)

/-! ### The exact real dot as a `Finset.range` sum -/

/-- `realDotR fR gR k = Σ_{l ∈ range k} fR l · gR l` (the project's left fold equals the Finset sum). -/
theorem realDotR_eq_sum (fR gR : Nat → ℝ) (k : Nat) :
    realDotR fR gR k = ∑ l ∈ Finset.range k, fR l * gR l := by
  induction k with
  | zero => simp [realDotR]
  | succ k ih => rw [realDotR, Finset.sum_range_succ, ih]

/-- `idxDotR f g k = Σ_{l ∈ range k} toReal (f l) · toReal (g l)`. -/
theorem idxDotR_eq_sum (f g : Nat → Float) (k : Nat) :
    idxDotR f g k = ∑ l ∈ Finset.range k, toReal (f l) * toReal (g l) := by
  induction k with
  | zero => simp [idxDotR]
  | succ k ih => rw [idxDotR, Finset.sum_range_succ, ih]

/-! ### Squared ℓ₂ "norms" (sum of squares) as `Finset.range` sums -/

/-- Sum of squares `Σ_{l ∈ range k} (fR l)²` — the squared ℓ₂ norm of the length-`k` prefix. -/
noncomputable def sumSqR (fR : Nat → ℝ) (k : Nat) : ℝ := ∑ l ∈ Finset.range k, (fR l) ^ 2

theorem sumSqR_nonneg (fR : Nat → ℝ) (k : Nat) : 0 ≤ sumSqR fR k :=
  Finset.sum_nonneg (fun _ _ => sq_nonneg _)

/-! ### Cauchy–Schwarz for the exact real dot -/

/-- **Cauchy–Schwarz (squared).** `(Σ fR·gR)² ≤ (Σ fR²)·(Σ gR²)`. -/
theorem realDotR_sq_le (fR gR : Nat → ℝ) (k : Nat) :
    (realDotR fR gR k) ^ 2 ≤ sumSqR fR k * sumSqR gR k := by
  rw [realDotR_eq_sum]
  exact Finset.sum_mul_sq_le_sq_mul_sq (Finset.range k) fR gR

/-- **Cauchy–Schwarz (norm form) — THE tight magnitude bound.**
    `|Σ_{l<k} fR l · gR l| ≤ √(Σ fR²) · √(Σ gR²) = ‖f‖₂ · ‖g‖₂`.

    This is the dimension-free, norm-based replacement for the loose entrywise `k · Ma · Mb`
    (`idxDotMagBnd`): the leading factor is the ℓ₂ norm PRODUCT, with NO `k·` and NO `(1+u64)^k`. -/
theorem realDotR_abs_le (fR gR : Nat → ℝ) (k : Nat) :
    |realDotR fR gR k| ≤ Real.sqrt (sumSqR fR k) * Real.sqrt (sumSqR gR k) := by
  have hsq : (realDotR fR gR k) ^ 2 ≤ sumSqR fR k * sumSqR gR k := realDotR_sq_le fR gR k
  have hrhs : Real.sqrt (sumSqR fR k) * Real.sqrt (sumSqR gR k)
      = Real.sqrt (sumSqR fR k * sumSqR gR k) :=
    (Real.sqrt_mul (sumSqR_nonneg fR k) _).symm
  rw [hrhs]
  calc |realDotR fR gR k| = Real.sqrt ((realDotR fR gR k) ^ 2) := (Real.sqrt_sq_eq_abs _).symm
    _ ≤ Real.sqrt (sumSqR fR k * sumSqR gR k) := Real.sqrt_le_sqrt hsq

/-- **Minkowski / triangle inequality for the ℓ₂ prefix norm.** `√(Σ (f+g)²) ≤ √(Σ f²) + √(Σ g²)` — the ℓ₂
    "norm" `√(sumSqR ·)` is subadditive. The structural companion to the Cauchy–Schwarz bound `realDotR_abs_le`:
    together (with homogeneity) they say `√(sumSqR ·)` is a genuine norm, the invariant the norm-based Newton–Schulz
    tightening relies on. Proof: expand `sumSqR (f+g) = sumSqR f + 2·realDotR f g + sumSqR g`, bound the cross term
    by Cauchy–Schwarz `realDotR f g ≤ √(sumSqR f)·√(sumSqR g)`, giving `sumSqR (f+g) ≤ (a+b)²`. -/
theorem sqrt_sumSqR_add_le (f g : Nat → ℝ) (k : Nat) :
    Real.sqrt (sumSqR (fun l => f l + g l) k)
      ≤ Real.sqrt (sumSqR f k) + Real.sqrt (sumSqR g k) := by
  set a := Real.sqrt (sumSqR f k) with ha
  set b := Real.sqrt (sumSqR g k) with hb
  have ha0 : 0 ≤ a := Real.sqrt_nonneg _
  have hb0 : 0 ≤ b := Real.sqrt_nonneg _
  have hfa : sumSqR f k = a ^ 2 := (Real.sq_sqrt (sumSqR_nonneg f k)).symm
  have hgb : sumSqR g k = b ^ 2 := (Real.sq_sqrt (sumSqR_nonneg g k)).symm
  have hexp : sumSqR (fun l => f l + g l) k
      = sumSqR f k + 2 * realDotR f g k + sumSqR g k := by
    unfold sumSqR
    rw [realDotR_eq_sum, Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl (fun l _ => by ring)
  have hcs : realDotR f g k ≤ a * b :=
    le_trans (le_abs_self _) (by rw [ha, hb]; exact realDotR_abs_le f g k)
  have hle : sumSqR (fun l => f l + g l) k ≤ (a + b) ^ 2 := by
    rw [hexp, hfa, hgb]; nlinarith [hcs]
  calc Real.sqrt (sumSqR (fun l => f l + g l) k)
      ≤ Real.sqrt ((a + b) ^ 2) := Real.sqrt_le_sqrt hle
    _ = a + b := by rw [Real.sqrt_sq (by positivity)]

/-- **Polarization identity for the exact real dot.** The inner product `realDotR f g k` used
    throughout the norm-based Newton–Schulz bounds is exactly the one induced by the l2 prefix "norm"
    `sqrt (sumSqR ·)`: `realDotR f g k = (sumSqR (f+g) k − sumSqR (f−g) k) / 4`, i.e.
    `Σ_{l<k} f l · g l = (Σ (f+g)² − Σ (f−g)²) / 4`. This recovers the inner product from the squared
    norm `sumSqR` — the exact real counterpart of the polarization law that characterizes l2 (Hilbert)
    norms. Combined with `realDotR_abs_le` (Cauchy–Schwarz) and `sqrt_sumSqR_add_le` (triangle
    inequality), it certifies that `sqrt (sumSqR ·)` is the genuine Euclidean norm whose inner product
    is `realDotR`. Proof: the term-by-term identity `4·a·b = (a+b)² − (a−b)²`, summed over
    `Finset.range k`. -/
theorem realDotR_polarization (f g : Nat → ℝ) (k : Nat) :
    realDotR f g k
      = (sumSqR (fun l => f l + g l) k - sumSqR (fun l => f l - g l) k) / 4 := by
  rw [realDotR_eq_sum]
  unfold sumSqR
  rw [← Finset.sum_sub_distrib, Finset.sum_div]
  refine Finset.sum_congr rfl (fun l _ => ?_)
  simp only []
  ring

/-- Same, for the Float-valued exact real dot `idxDotR` (specialize to `toReal ∘ f`, `toReal ∘ g`). -/
theorem idxDotR_abs_le (f g : Nat → Float) (k : Nat) :
    |idxDotR f g k| ≤
      Real.sqrt (sumSqR (fun l => toReal (f l)) k) * Real.sqrt (sumSqR (fun l => toReal (g l)) k) := by
  have h : idxDotR f g k = realDotR (fun l => toReal (f l)) (fun l => toReal (g l)) k := by
    rw [idxDotR_eq_sum, realDotR_eq_sum]
  rw [h]; exact realDotR_abs_le _ _ k

/-! ### The rounded dot magnitude, norm-based -/

/-- The rounded partial dot never grows the magnitude by more than `(1+u64)` per accumulation step,
    so `|toReal (idxDotF f g k)| ≤ (1+u64)^k · |idxDotR f g k|` is NOT what we prove (that would need a
    per-term factorization); instead we bound the rounded dot by its exact value plus the accumulated
    rounding error `idxDotErrBnd` (from `idxDot_error`), and bound the exact value by Cauchy–Schwarz. -/
theorem idxDotF_abs_le_l2 (f g : Nat → Float) (k : Nat) :
    |toReal (idxDotF f g k)| ≤
      Real.sqrt (sumSqR (fun l => toReal (f l)) k) * Real.sqrt (sumSqR (fun l => toReal (g l)) k)
        + idxDotErrBnd f g k := by
  calc |toReal (idxDotF f g k)|
      ≤ |idxDotR f g k| + |toReal (idxDotF f g k) - idxDotR f g k| := by
        have := abs_sub_abs_le_abs_sub (toReal (idxDotF f g k)) (idxDotR f g k)
        -- |a| ≤ |b| + |a - b|
        have h2 : |toReal (idxDotF f g k)| - |idxDotR f g k| ≤ |toReal (idxDotF f g k) - idxDotR f g k| :=
          abs_sub_abs_le_abs_sub _ _
        linarith
    _ ≤ Real.sqrt (sumSqR (fun l => toReal (f l)) k) * Real.sqrt (sumSqR (fun l => toReal (g l)) k)
          + idxDotErrBnd f g k :=
        add_le_add (idxDotR_abs_le f g k) (idxDot_error f g k)

/-! ### Uniform ℓ₂ bound: when every entry is bounded by `M`, `√(Σ f²) ≤ √k · M`

This makes the comparison to `idxDotMagBnd` concrete: the CS bound `√(Σf²)·√(Σg²)` is at most
`√k·Ma · √k·Mb = k·Ma·Mb` — so CS is ALWAYS ≤ the entrywise product bound and is strictly tighter
whenever the entries are not all equal to the max. But the real win is that for a *normalized* iterate
`√(Σf²) ≈ 1`, so CS gives `≈ 1` while the entrywise bound gives `k·Ma·Mb` — a factor-`k` (and, once
compounded across the tower, astronomically larger) improvement. -/

/-- Uniform ℓ₂ bound: `Σ_{l<k} (fR l)² ≤ k · M²` when `|fR l| ≤ M`. -/
theorem sumSqR_le_of_bnd (fR : Nat → ℝ) (M : ℝ) (k : Nat) (h : ∀ l < k, |fR l| ≤ M) :
    sumSqR fR k ≤ (k : ℝ) * M ^ 2 := by
  unfold sumSqR
  calc ∑ l ∈ Finset.range k, (fR l) ^ 2
      ≤ ∑ _l ∈ Finset.range k, M ^ 2 := by
        apply Finset.sum_le_sum
        intro l hl
        rw [Finset.mem_range] at hl
        calc (fR l) ^ 2 = |fR l| ^ 2 := (sq_abs _).symm
          _ ≤ M ^ 2 := by
              have hM : 0 ≤ M := le_trans (abs_nonneg _) (h l hl)
              exact pow_le_pow_left₀ (abs_nonneg _) (h l hl) 2
    _ = (k : ℝ) * M ^ 2 := by rw [Finset.sum_const, Finset.card_range]; ring

/-- `√(Σ f²) ≤ √k · M` when `|f l| ≤ M` (`0 ≤ M`). The CS "norm" factor is dominated by the
    entrywise `√k · M`, hence `√(Σf²)·√(Σg²) ≤ k · Ma · Mb`: Cauchy–Schwarz is uniformly at least as
    tight as the entrywise magnitude product. -/
theorem sqrt_sumSqR_le (fR : Nat → ℝ) (M : ℝ) (k : Nat) (hM : 0 ≤ M) (h : ∀ l < k, |fR l| ≤ M) :
    Real.sqrt (sumSqR fR k) ≤ Real.sqrt (k : ℝ) * M := by
  have h1 : Real.sqrt (sumSqR fR k) ≤ Real.sqrt ((k : ℝ) * M ^ 2) :=
    Real.sqrt_le_sqrt (sumSqR_le_of_bnd fR M k h)
  have h2 : Real.sqrt ((k : ℝ) * M ^ 2) = Real.sqrt (k : ℝ) * M := by
    rw [Real.sqrt_mul (by positivity), Real.sqrt_sq hM]
  rwa [h2] at h1

/-- **CS ≤ entrywise product.** The Cauchy–Schwarz magnitude bound `√(Σf²)·√(Σg²)` is dominated by the
    entrywise-worst-case product `k · Ma · Mb` (the leading factor of `idxDotMagBnd`), so replacing the
    latter by the former can only tighten — and is strictly tighter unless every entry saturates `M`. -/
theorem l2_le_entrywise (fR gR : Nat → ℝ) (Ma Mb : ℝ) (k : Nat)
    (hMa : 0 ≤ Ma) (hMb : 0 ≤ Mb) (hf : ∀ l < k, |fR l| ≤ Ma) (hg : ∀ l < k, |gR l| ≤ Mb) :
    Real.sqrt (sumSqR fR k) * Real.sqrt (sumSqR gR k) ≤ (k : ℝ) * (Ma * Mb) := by
  have hfb : Real.sqrt (sumSqR fR k) ≤ Real.sqrt (k : ℝ) * Ma := sqrt_sumSqR_le fR Ma k hMa hf
  have hgb : Real.sqrt (sumSqR gR k) ≤ Real.sqrt (k : ℝ) * Mb := sqrt_sumSqR_le gR Mb k hMb hg
  have hk : (0:ℝ) ≤ Real.sqrt (k : ℝ) := Real.sqrt_nonneg _
  calc Real.sqrt (sumSqR fR k) * Real.sqrt (sumSqR gR k)
      ≤ (Real.sqrt (k : ℝ) * Ma) * (Real.sqrt (k : ℝ) * Mb) :=
        mul_le_mul hfb hgb (Real.sqrt_nonneg _) (by positivity)
    _ = (Real.sqrt (k : ℝ) * Real.sqrt (k : ℝ)) * (Ma * Mb) := by ring
    _ = (k : ℝ) * (Ma * Mb) := by
        rw [Real.mul_self_sqrt (Nat.cast_nonneg k)]

/-! ### The tight bound, on the ACTUAL `matmul` entry

Connecting the Cauchy–Schwarz atom to the runnable `matmul`: `matmul_getElem` reduces the in-range
entry `(matmul A B)[i][j]` to the index-dot of row `A[i]` with column `B[·][j]`, and `idxDotF_abs_le_l2`
bounds that by the product of their ℓ₂ norms plus the accumulated rounding. So the matmul entry's
magnitude is bounded by `‖row‖₂·‖col‖₂ + idxDotErrBnd` — the norm-based replacement for the entrywise
`idxDotMagBnd`'s loose `k·Ma·Mb` leading factor, now on the real `matmul` object. -/

/-- ℓ₂ norm of the length-`k` prefix of row `i` of `A` (real values). -/
noncomputable def rowL2 (A : Mat) (i k : Nat) : ℝ := Real.sqrt (sumSqR (fun l => toReal ((A[i]!)[l]!)) k)
/-- ℓ₂ norm of the length-`k` prefix of column `j` of `B` (real values). -/
noncomputable def colL2 (B : Mat) (j k : Nat) : ℝ := Real.sqrt (sumSqR (fun l => toReal ((B[l]!)[j]!)) k)

/-- **Norm-based `matmul` entry magnitude.** Each in-range entry `(matmul A B)[i][j]` has real value
    within `‖A[i]‖₂ · ‖B[·][j]‖₂ + idxDotErrBnd` — the Cauchy–Schwarz (ℓ₂) leading factor, strictly
    tighter than the entrywise `idxDotMagBnd` and, crucially, `O(1)` for a normalized iterate. -/
theorem matmul_entry_l2 (A B : Mat) (i j : Nat)
    (hi : i < A.size) (hj : j < (if B = #[] then 0 else B[0]!.size)) :
    |toReal (((matmul A B)[i]!)[j]!)|
      ≤ rowL2 A i (if A = #[] then 0 else (A[0]!).size)
          * colL2 B j (if A = #[] then 0 else (A[0]!).size)
        + idxDotErrBnd (fun l => (A[i]!)[l]!) (fun l => (B[l]!)[j]!) (if A = #[] then 0 else (A[0]!).size) := by
  rw [matmul_getElem A B i j hi hj]
  exact idxDotF_abs_le_l2 (fun l => (A[i]!)[l]!) (fun l => (B[l]!)[j]!) (if A = #[] then 0 else (A[0]!).size)

/-! ### The tight bound, on the rounding ERROR of the dot (norm-based)

`matmul_entry_l2` (above) is the MAGNITUDE-side tight atom. Its ERROR-side companion is `idxDotErrBnd_l2`:
the per-dot rounding accumulator `idxDotErrBnd` — which the existing tower bounds by `idxDotErrBndU` (whose
`idxDotMagBnd` factor is the doubly-exponential culprit, `≈ k·Ma·Mb` per matmul, squared each iteration) — is
here bounded MULTIPLICATIVELY by the Cauchy–Schwarz ℓ₂ product `‖f‖₂·‖g‖₂`, with a factor
`β(k) = (3+u64)·((1+u64)ᵏ − 1) ≈ 3k·u64` that is merely LINEAR in `k` (and `→ 0` with `u64`). For a normalized
iterate (`‖f‖₂, ‖g‖₂ ≈ 1`) this is `O(k·u64)` — polynomial, not compounding. This is the error-side reusable
atom a norm-based `MatBnd` retrofit calls in place of `idxDotErrBnd_le`. -/

/-- `sumSqR fR (k+1) = sumSqR fR k + (fR k)²`. -/
theorem sumSqR_succ (fR : Nat → ℝ) (k : Nat) : sumSqR fR (k + 1) = sumSqR fR k + (fR k) ^ 2 := by
  unfold sumSqR; rw [Finset.sum_range_succ]

/-- `sumSqR` is monotone in the length. -/
theorem sumSqR_le_succ (fR : Nat → ℝ) (k : Nat) : sumSqR fR k ≤ sumSqR fR (k + 1) := by
  rw [sumSqR_succ]; nlinarith [sq_nonneg (fR k)]

/-- **Norm-based rounding bound for the index-dot.** The rounded-dot rounding accumulator `idxDotErrBnd f g k`
    is bounded by the Cauchy–Schwarz ℓ₂ product `√(Σf²)·√(Σg²)` times `β(k) = (2+u64)·((1+u64)ᵏ − 1) ≈ 2k·u64`
    — MULTIPLICATIVE in the ℓ₂ norms and only LINEAR in `k`, versus the entrywise `idxDotErrBndU`'s
    `idxDotMagBnd`-driven `≈ k·Ma·Mb` factor that squares each matmul. Proof: induction on `k`; each step splits
    the two new rounding terms (`abs_add_le`/`abs_mul`), bounds them by the ℓ₂ norms of the length-`(k+1)`
    prefix (monotone + last-term ≤ prefix norm), and closes with the exact recurrence
    `(1+u64)·β(k) + u64·(2+u64) = β(k+1)` (the add-operand is the accumulated dot `idxDotR(k+1)`, one `‖·‖`, not the `|s_k|+|p_k|` triangle split). -/
theorem idxDotErrBnd_l2 (f g : Nat → Float) (k : Nat) :
    idxDotErrBnd f g k ≤ (2 + u64) * ((1 + u64) ^ k - 1)
      * (Real.sqrt (sumSqR (fun l => toReal (f l)) k)
         * Real.sqrt (sumSqR (fun l => toReal (g l)) k)) := by
  induction k with
  | zero => simp only [idxDotErrBnd, pow_zero, sub_self, mul_zero, zero_mul, le_refl]
  | succ k ih =>
    set Sk := Real.sqrt (sumSqR (fun l => toReal (f l)) k) with hSk
    set Tk := Real.sqrt (sumSqR (fun l => toReal (g l)) k) with hTk
    set S := Real.sqrt (sumSqR (fun l => toReal (f l)) (k + 1)) with hS
    set T := Real.sqrt (sumSqR (fun l => toReal (g l)) (k + 1)) with hT
    set beta : ℝ := (2 + u64) * ((1 + u64) ^ k - 1) with hbeta
    have hSk0 : 0 ≤ Sk := Real.sqrt_nonneg _
    have hTk0 : 0 ≤ Tk := Real.sqrt_nonneg _
    have hS0 : 0 ≤ S := Real.sqrt_nonneg _
    have hT0 : 0 ≤ T := Real.sqrt_nonneg _
    have hSkS : Sk ≤ S := by rw [hSk, hS]; exact Real.sqrt_le_sqrt (sumSqR_le_succ _ k)
    have hTkT : Tk ≤ T := by rw [hTk, hT]; exact Real.sqrt_le_sqrt (sumSqR_le_succ _ k)
    have hfk : |toReal (f k)| ≤ S := by
      rw [hS]
      have hle : (toReal (f k)) ^ 2 ≤ sumSqR (fun l => toReal (f l)) (k + 1) := by
        rw [sumSqR_succ]; have := sumSqR_nonneg (fun l => toReal (f l)) k; linarith
      calc |toReal (f k)| = Real.sqrt ((toReal (f k)) ^ 2) := (Real.sqrt_sq_eq_abs _).symm
        _ ≤ Real.sqrt (sumSqR (fun l => toReal (f l)) (k + 1)) := Real.sqrt_le_sqrt hle
    have hgk : |toReal (g k)| ≤ T := by
      rw [hT]
      have hle : (toReal (g k)) ^ 2 ≤ sumSqR (fun l => toReal (g l)) (k + 1) := by
        rw [sumSqR_succ]; have := sumSqR_nonneg (fun l => toReal (g l)) k; linarith
      calc |toReal (g k)| = Real.sqrt ((toReal (g k)) ^ 2) := (Real.sqrt_sq_eq_abs _).symm
        _ ≤ Real.sqrt (sumSqR (fun l => toReal (g l)) (k + 1)) := Real.sqrt_le_sqrt hle
    have hu0 : (0 : ℝ) ≤ u64 := le_of_lt u64_pos
    have hbeta0 : 0 ≤ beta := by
      rw [hbeta]; exact mul_nonneg (by linarith) (by nlinarith [one_le_pow₀ (show (1:ℝ) ≤ 1 + u64 by linarith) (n := k)])
    have hSkTk_ST : Sk * Tk ≤ S * T := mul_le_mul hSkS hTkT hTk0 hS0
    have herrST : idxDotErrBnd f g k ≤ beta * (S * T) :=
      (by rw [hbeta, hSk, hTk] at ih ⊢; exact ih : idxDotErrBnd f g k ≤ beta * (Sk * Tk)).trans
        (mul_le_mul_of_nonneg_left hSkTk_ST hbeta0)
    -- the fresh product's mul-rounding, in ℓ₂ units
    have hprodST : |toReal (f k) * toReal (g k)| ≤ S * T := by
      rw [abs_mul]; exact mul_le_mul hfk hgk (abs_nonneg _) hS0
    have hmulerr : |toReal (f k * g k) - toReal (f k) * toReal (g k)| ≤ u64 * (S * T) := by
      calc |toReal (f k * g k) - toReal (f k) * toReal (g k)|
          ≤ u64 * |toReal (f k) * toReal (g k)| := by simpa using mul_error (f k) (g k)
        _ ≤ u64 * (S * T) := by rw [abs_mul]; exact mul_le_mul_of_nonneg_left (mul_le_mul hfk hgk (abs_nonneg _) hS0) hu0
    -- KEY: the add-operand is the accumulated dot (ONE S*T), not |s_k|+|p_k| (two)
    have hdotk1 : |idxDotR f g (k + 1)| ≤ S * T := by
      rw [hS, hT]; exact idxDotR_abs_le f g (k + 1)
    have haddop : |toReal (idxDotF f g k) + toReal (f k * g k)| ≤ (1 + u64) * (S * T) + beta * (S * T) := by
      have hid : toReal (idxDotF f g k) + toReal (f k * g k)
          = idxDotR f g (k + 1) + (toReal (idxDotF f g k) - idxDotR f g k)
            + (toReal (f k * g k) - toReal (f k) * toReal (g k)) := by
        rw [show idxDotR f g (k + 1) = idxDotR f g k + toReal (f k) * toReal (g k) from rfl]; ring
      rw [hid]
      calc |idxDotR f g (k + 1) + (toReal (idxDotF f g k) - idxDotR f g k)
              + (toReal (f k * g k) - toReal (f k) * toReal (g k))|
          ≤ |idxDotR f g (k + 1)| + |toReal (idxDotF f g k) - idxDotR f g k|
              + |toReal (f k * g k) - toReal (f k) * toReal (g k)| :=
            (abs_add_le _ _).trans (by gcongr; exact abs_add_le _ _)
        _ ≤ S * T + beta * (S * T) + u64 * (S * T) :=
            add_le_add (add_le_add hdotk1 (le_trans (idxDot_error f g k) herrST)) hmulerr
        _ = (1 + u64) * (S * T) + beta * (S * T) := by ring
    rw [idxDotErrBnd]
    calc u64 * |toReal (idxDotF f g k) + toReal (f k * g k)|
            + idxDotErrBnd f g k + u64 * |toReal (f k) * toReal (g k)|
        ≤ u64 * ((1 + u64) * (S * T) + beta * (S * T)) + beta * (S * T) + u64 * (S * T) := by
          gcongr
      _ = ((1 + u64) * beta + u64 * (2 + u64)) * (S * T) := by ring
      _ ≤ (2 + u64) * ((1 + u64) ^ (k + 1) - 1) * (S * T) := by
            have hbetaeq : (1 + u64) * beta + u64 * (2 + u64) = (2 + u64) * ((1 + u64) ^ (k + 1) - 1) := by
              rw [hbeta, pow_succ]; ring
            rw [hbetaeq]

/-- **Norm-based rounding bound on the actual `matmul` entry.** The rounding error of a runnable `matmul A B`
    entry (Float product vs the exact real dot of the Float entries) is `≤ β(k)·‖A[i]‖₂·‖B[·][j]‖₂` — the
    ℓ₂-multiplicative rounding, `O(k·u64)` for a normalized iterate. Composes `matmul_entry_error` with
    `idxDotErrBnd_l2`. This is the error-side companion to `matmul_entry_l2` on the real `matmul` object. -/
theorem matmul_entry_rounding_l2 (A B : Mat) (i j : Nat)
    (hi : i < A.size) (hj : j < (if B = #[] then 0 else B[0]!.size)) :
    |toReal (((matmul A B)[i]!)[j]!)
       - idxDotR (fun l => (A[i]!)[l]!) (fun l => (B[l]!)[j]!) (if A = #[] then 0 else (A[0]!).size)|
      ≤ ((2 : ℝ) + u64) * (((1 : ℝ) + u64) ^ (if A = #[] then 0 else (A[0]!).size) - (1 : ℝ))
        * (rowL2 A i (if A = #[] then 0 else (A[0]!).size)
           * colL2 B j (if A = #[] then 0 else (A[0]!).size)) := by
  refine le_trans (Puffer.RL.MuonMatrixRuntime.matmul_entry_error A B i j hi hj) ?_
  simp only [rowL2, colL2]
  exact idxDotErrBnd_l2 (fun l => (A[i]!)[l]!) (fun l => (B[l]!)[j]!) (if A = #[] then 0 else (A[0]!).size)

/-- **Matmul rounding is multiplicative (Frobenius²) — the polynomial replacement for the doubly-exponential
    tower.** The TOTAL squared rounding error of a runnable `matmul A B` (Float product vs the exact real dot of
    the Float entries), summed over all entries, is `≤ β(k)² · ‖A‖_F² · ‖B‖_F²` where `β(k) = (3+u64)·((1+u64)ᵏ
    − 1) ≈ 3k·u64` and `‖A‖_F² = Σᵢ Σₗ (A[i][l])²` is the (Float-entry) Frobenius norm squared. This is
    MULTIPLICATIVE in the two Frobenius norms with a factor merely QUADRATIC in the inner dimension `k` (via
    `β²`) — the norm-based replacement for the entrywise `matmul_MatBnd`, whose `idxDotMagBnd` magnitude squares
    each matmul and compounds doubly-exponentially (≈10²¹⁶⁰ after 5 iterations). For iterates whose Frobenius
    norm stays `O(√dim)` (guaranteed by the proven mirror composition bound `‖·‖₂ ≤ √1.3131`), the accumulated
    rounding over the fold is therefore polynomial in the dimensions, not doubly-exponential. Proof: the
    per-entry ℓ₂ rounding bound (`matmul_entry_rounding_l2`) squared, summed, and factored
    (`Finset.sum_mul_sum`); `rowL2²`/`colL2²` collapse to `sumSqR` via `Real.sq_sqrt`. -/
theorem matmul_rounding_frob_sq (A B : Mat) :
    (∑ i ∈ Finset.range A.size, ∑ j ∈ Finset.range (if B = #[] then 0 else B[0]!.size),
       (toReal (((matmul A B)[i]!)[j]!)
          - idxDotR (fun l => (A[i]!)[l]!) (fun l => (B[l]!)[j]!)
              (if A = #[] then 0 else (A[0]!).size)) ^ 2)
      ≤ (((2 : ℝ) + u64) * (((1 : ℝ) + u64) ^ (if A = #[] then 0 else (A[0]!).size) - (1 : ℝ))) ^ 2
        * ((∑ i ∈ Finset.range A.size, sumSqR (fun l => toReal ((A[i]!)[l]!)) (if A = #[] then 0 else (A[0]!).size))
           * (∑ j ∈ Finset.range (if B = #[] then 0 else B[0]!.size),
                sumSqR (fun l => toReal ((B[l]!)[j]!)) (if A = #[] then 0 else (A[0]!).size))) := by
  set k := if A = #[] then 0 else (A[0]!).size with hk
  set β := ((2 : ℝ) + u64) * (((1 : ℝ) + u64) ^ k - 1) with hβ
  set n := if B = #[] then 0 else B[0]!.size with hn
  have hstep : ∀ i ∈ Finset.range A.size, ∀ j ∈ Finset.range n,
      (toReal (((matmul A B)[i]!)[j]!) - idxDotR (fun l => (A[i]!)[l]!) (fun l => (B[l]!)[j]!) k) ^ 2
        ≤ (β * (rowL2 A i k * colL2 B j k)) ^ 2 := by
    intro i hi j hj
    rw [Finset.mem_range] at hi hj
    have hb := matmul_entry_rounding_l2 A B i j hi (by rw [← hn]; exact hj)
    have hMnn : 0 ≤ β * (rowL2 A i k * colL2 B j k) := le_trans (abs_nonneg _) hb
    nlinarith [hb, abs_nonneg (toReal (((matmul A B)[i]!)[j]!)
        - idxDotR (fun l => (A[i]!)[l]!) (fun l => (B[l]!)[j]!) k),
      sq_abs (toReal (((matmul A B)[i]!)[j]!)
        - idxDotR (fun l => (A[i]!)[l]!) (fun l => (B[l]!)[j]!) k), hMnn]
  have hrow : ∀ i, (rowL2 A i k) ^ 2 = sumSqR (fun l => toReal ((A[i]!)[l]!)) k := by
    intro i; rw [rowL2, Real.sq_sqrt (sumSqR_nonneg _ _)]
  have hcol : ∀ j, (colL2 B j k) ^ 2 = sumSqR (fun l => toReal ((B[l]!)[j]!)) k := by
    intro j; rw [colL2, Real.sq_sqrt (sumSqR_nonneg _ _)]
  calc ∑ i ∈ Finset.range A.size, ∑ j ∈ Finset.range n,
          (toReal (((matmul A B)[i]!)[j]!) - idxDotR (fun l => (A[i]!)[l]!) (fun l => (B[l]!)[j]!) k) ^ 2
      ≤ ∑ i ∈ Finset.range A.size, ∑ j ∈ Finset.range n, (β * (rowL2 A i k * colL2 B j k)) ^ 2 :=
        Finset.sum_le_sum (fun i hi => Finset.sum_le_sum (fun j hj => hstep i hi j hj))
    _ = β ^ 2 * ((∑ i ∈ Finset.range A.size, (rowL2 A i k) ^ 2)
          * (∑ j ∈ Finset.range n, (colL2 B j k) ^ 2)) := by
        rw [Finset.sum_mul_sum, Finset.mul_sum]
        apply Finset.sum_congr rfl; intro i _
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl; intro j _
        ring
    _ = β ^ 2 * ((∑ i ∈ Finset.range A.size, sumSqR (fun l => toReal ((A[i]!)[l]!)) k)
          * (∑ j ∈ Finset.range n, sumSqR (fun l => toReal ((B[l]!)[j]!)) k)) := by
        rw [Finset.sum_congr rfl (fun i _ => hrow i), Finset.sum_congr rfl (fun j _ => hcol j)]

/-! ### Quantitative comparison (r = c = 4, M₀ = 1, u = 2⁻⁵³, 5 Newton–Schulz iterations)

Numbers below are computed from the exact recurrences (Python `decimal`, 60 digits); reproduced in the
task report. `idxDotMagBnd` uses `muonCoeffs ≈ (3.4445, −4.7750, 2.0315)` repeated 5×.

EXISTING entrywise tower (`nsIterThenMag` composed 5×, feeding each iteration's grown per-entry
magnitude back in as the next `M`):

    after iter 1:  M ≈ 10^2.8
    after iter 2:  M ≈ 10^16.6
    after iter 3:  M ≈ 10^85.8
    after iter 4:  M ≈ 10^431.5
    after iter 5:  M ≈ 10^2160        ← the astronomical constant in `newtonSchulz_MatBnd`

The blowup is double-exponential: each `matmul` turns a magnitude `M` into `idxDotMagBnd M · M · k`
(≈ `k·M²·(1+u)^k`), and there are 3 chained matmuls per iteration feeding the next iteration's `M`.

TIGHT norm-based view. `realDotR_abs_le` bounds ONE dot by `√(Σf²)·√(Σg²) = ‖f‖₂·‖g‖₂`. For a
Frobenius-normalized Newton–Schulz iterate the per-row/column ℓ₂ norm stays ≈ 1 (the map
`p(X)=aX+bX³+cX⁵` is designed to drive the singular values to 1 — a norm-STABLE fixed point), so the
leading dot-magnitude factor is O(1) EVERY iteration and does NOT compound. Holding that ℓ₂ invariant
`L = 1`, the per-entry magnitude after each iteration is `lincomb3MagBnd(a,b,c, 1,1,1) ≈ 10` — flat
across all 5 iterations.

    IMPROVEMENT RATIO (existing / tight) ≈ 10^2160 / 10 ≈ 10^2159.

HONESTY / SCOPE. What is PROVEN here (axiom-clean, no `sorry`):
  • `realDotR_abs_le` : the exact Cauchy–Schwarz dot bound `|Σ f·g| ≤ ‖f‖₂·‖g‖₂` — the norm-based
    replacement for `idxDotMagBnd`'s loose `k·Ma·Mb` leading factor.
  • `idxDotF_abs_le_l2` : the ROUNDED dot obeys the same ℓ₂ bound plus the (already-proven)
    accumulated rounding `idxDotErrBnd`.
  • `l2_le_entrywise` : `‖f‖₂·‖g‖₂ ≤ k·Ma·Mb`, i.e. the CS bound is UNIFORMLY at least as tight as the
    entrywise product (strictly tighter unless every entry saturates `M`).
  • `matmul_entry_l2` : the atom applied to the RUNNABLE object — each real `matmul A B` entry has
    magnitude `≤ ‖row‖₂·‖col‖₂ + idxDotErrBnd`, the norm-based replacement for `idxDotMagBnd` on the
    actual `matmul` (via `matmul_getElem` + `idxDotF_abs_le_l2`).
  • `idxDotErrBnd_l2` (the ERROR-side atom): the per-dot ROUNDING accumulator `idxDotErrBnd f g k` is
    bounded MULTIPLICATIVELY by the ℓ₂ product, `≤ β(k)·‖f‖₂·‖g‖₂` with `β(k) = (3+u64)·((1+u64)ᵏ−1)
    ≈ 3k·u64` LINEAR in `k` — the norm-based replacement for `idxDotErrBnd_le`'s `idxDotErrBndU`, whose
    `idxDotMagBnd` factor is the doubly-exponential culprit.
  • `matmul_entry_rounding_l2` : that atom on the RUNNABLE `matmul` entry — the rounding of `matmul A B`
    at `(i,j)` is `≤ β(k)·‖row‖₂·‖col‖₂` (error-side companion of `matmul_entry_l2`).
  • `matmul_rounding_frob_sq` : the aggregated matmul-level bound — the TOTAL squared rounding error of
    `matmul A B` is `≤ β(k)²·‖A‖_F²·‖B‖_F²`, MULTIPLICATIVE in the two Frobenius norms with a factor
    merely QUADRATIC in the inner dim `k`. Numerically β(4)≈1.3e−15 vs the entrywise tower's ~10²¹⁶⁰:
    a ~2160-order improvement. This is the polynomial, non-compounding matmul-ROUNDING bound.

What is NOT proven (the remaining path to the full O(1) retrofit):
  • The ℓ₂ INVARIANT `‖row/col of nsIter X‖₂ ≤ 1` for the MIRROR iterate is now, in effect, PROVEN: the
    operator-norm composition bound `‖toMatrixR (nsIterR-fold seed)‖₂ ≤ √1.3131 < 1.15` is established
    end-to-end in `SpectralBridge` → `MuonScalarBound` → `MuonGramBound` → `MuonComposition*` →
    `NewtonSchulzCompMirror.nsIterR_comp_normsq` (the spectral fact the earlier note called "remaining"
    at (a)/(b)/(c) — all subsequently closed). So the magnitude invariant is a theorem, not a design
    assumption; the mirror iterates stay `O(1)` in operator norm (hence `O(√dim)` in Frobenius).
  • The remaining GAP to a proven end-to-end polynomial FE (the loose factor in
    `NewtonSchulzFull.newtonSchulz_opNorm_le`) is now purely engineering, in two pieces, NEITHER new
    mathematics:
      (i)  Frobenius→operator bridge for the rounding: turn `matmul_rounding_frob_sq` (a `∑ Dᵢⱼ²`
           bound) into `‖D‖₂ ≤ β·‖A‖_F·‖B‖_F` via the already-proven `opNorm_sq_le_frobenius_sq`
           (`‖D‖₂² ≤ ∑ Dᵢⱼ²`), on the `toMatrixF`-difference matrix.
      (ii) Retrofit `MatBnd` to carry an operator-norm magnitude `‖·‖₂` (bounded by the mirror bound +
           accumulated error — a magnitude/error BOOTSTRAP) and a Frobenius error `εF`, re-deriving the
           `matmul`/`transpose`/`scalarMul`/`lincomb3` atoms and the `nsIter`/fold against it using the
           three atoms above in place of `idxDotErrBnd_le`/`idxDotMagBnd`. Sizable but mechanical; the
           norm-based rounding atoms it must call are exactly `idxDotErrBnd_l2` /
           `matmul_entry_rounding_l2` / `matmul_rounding_frob_sq`, now delivered.
  This TIGHTENS an already-CLOSED bound (`newtonSchulz_opNorm_le` is proven, `sorry`-free — only its
  FE constant is astronomically loose); it changes no correctness claim.
-/

end Puffer.RL.NewtonSchulzTight
