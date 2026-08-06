/-
Grinding the numeric core of the Newton–Schulz matrix bound.

`newtonSchulz_eq_foldl` (in `MuonMatrixRuntime`) already reduces the 5-iteration loop to
`muonCoeffs.foldl nsIter seed`, and `foldl_rel` lifts a per-iteration Float↔ℝ bound to the whole
fold. What remains is the concrete per-iteration bound. Each iteration is 3 chained `matmul`s plus a
`lincomb3`, so the atom is a UNIFORM (max-over-entries) bound on one `matmul`, which in turn rests on
uniform bounds for the per-entry index-dot `idxDotF` that `matmul_entry_error` reduces to.

This file builds those uniform `idxDot` bounds (Stage 1): given per-index magnitude/error bounds on
the two factor functions, bound (a) the MAGNITUDE of the rounded dot `idxDotF` — which grows by a
`(1+u64)` factor per term, hence the recursive `idxDotMagBnd`; (b) the rounding error accumulator
`idxDotErrBnd` uniformly; and (c) the PERTURBATION between the Float-valued real dot `idxDotR` and the
exact real-real dot `realDotR` of the ℝ mirror. These are the numeric heart; later stages lift them
through the matrix `getElem` plumbing and the iteration/fold.
-/
import Mathlib
import Puffer.RL.MuonMatrixRuntime

namespace Puffer.RL.NewtonSchulzError

open Puffer.FloatR
open Puffer.RL.MuonMatrixRuntime (idxDotF idxDotR idxDotErrBnd)

/-- `0 ≤ 1 + u64` (the rounded-growth factor is nonnegative). -/
theorem one_add_u64_nonneg : (0 : ℝ) ≤ 1 + u64 := by linarith [u64_pos]

/-! ### Exact real–real dot of the ℝ mirror -/

/-- The exact real dot `Σ_{l<k} fR l · gR l` of two real-valued index functions (the ℝ mirror of
    `matmul`'s per-entry accumulator; no rounding, exact real arithmetic). -/
noncomputable def realDotR (fR gR : Nat → ℝ) : Nat → ℝ
  | 0 => 0
  | k + 1 => realDotR fR gR k + fR k * gR k

/-! ### (a) Magnitude of the rounded index-dot -/

/-- Recursive magnitude bound for `idxDotF`: each term contributes a rounded product
    (`≤ (1+u64)·Mf·Mg`) and the running sum is re-rounded (another `(1+u64)`). -/
noncomputable def idxDotMagBnd (Mf Mg : ℝ) : Nat → ℝ
  | 0 => 0
  | k + 1 => (idxDotMagBnd Mf Mg k + (1 + u64) * (Mf * Mg)) * (1 + u64)

/-- **Uniform magnitude bound.** If every factor entry is bounded (`|toReal (f l)| ≤ Mf`,
    `|toReal (g l)| ≤ Mg` for `l < k`), the rounded dot `idxDotF f g k` is within `idxDotMagBnd`. -/
theorem idxDotF_abs_le (f g : Nat → Float) (Mf Mg : ℝ) (hMf : 0 ≤ Mf) (_hMg : 0 ≤ Mg) :
    ∀ k : Nat, (∀ l < k, |toReal (f l)| ≤ Mf) → (∀ l < k, |toReal (g l)| ≤ Mg) →
      |toReal (idxDotF f g k)| ≤ idxDotMagBnd Mf Mg k := by
  intro k
  induction k with
  | zero => intro _ _; simp [idxDotF, idxDotMagBnd, toReal_zeroLit]
  | succ k ih =>
      intro hf hg
      have hfk : |toReal (f k)| ≤ Mf := hf k (Nat.lt_succ_self k)
      have hgk : |toReal (g k)| ≤ Mg := hg k (Nat.lt_succ_self k)
      have ihk := ih (fun l hl => hf l (Nat.lt_succ_of_lt hl)) (fun l hl => hg l (Nat.lt_succ_of_lt hl))
      have hprod : |toReal (f k * g k)| ≤ (1 + u64) * (Mf * Mg) := by
        calc |toReal (f k * g k)| ≤ (1 + u64) * |toReal (f k) * toReal (g k)| := mul_abs_le (f k) (g k)
          _ = (1 + u64) * (|toReal (f k)| * |toReal (g k)|) := by rw [abs_mul]
          _ ≤ (1 + u64) * (Mf * Mg) :=
              mul_le_mul_of_nonneg_left (mul_le_mul hfk hgk (abs_nonneg _) hMf) one_add_u64_nonneg
      simp only [idxDotF, idxDotMagBnd]
      calc |toReal (idxDotF f g k + f k * g k)|
          ≤ (1 + u64) * |toReal (idxDotF f g k) + toReal (f k * g k)| := add_abs_le _ _
        _ ≤ (1 + u64) * (|toReal (idxDotF f g k)| + |toReal (f k * g k)|) :=
            mul_le_mul_of_nonneg_left (abs_add_le _ _) one_add_u64_nonneg
        _ ≤ (1 + u64) * (idxDotMagBnd Mf Mg k + (1 + u64) * (Mf * Mg)) :=
            mul_le_mul_of_nonneg_left (add_le_add ihk hprod) one_add_u64_nonneg
        _ = (idxDotMagBnd Mf Mg k + (1 + u64) * (Mf * Mg)) * (1 + u64) := by ring

/-! ### (b) Uniform bound on the rounding-error accumulator -/

/-- Uniform bound for `idxDotErrBnd`: at each term add `u64·(partial-sum + product magnitude)` and
    `u64·(product magnitude)`, using the uniform magnitude bounds. -/
noncomputable def idxDotErrBndU (Mf Mg : ℝ) : Nat → ℝ
  | 0 => 0
  | k + 1 => u64 * (idxDotMagBnd Mf Mg k + (1 + u64) * (Mf * Mg))
      + idxDotErrBndU Mf Mg k + u64 * (Mf * Mg)

/-- **Uniform error-bound.** `idxDotErrBnd f g k ≤ idxDotErrBndU Mf Mg k` under the magnitude
    hypotheses (the `|·|` terms in `idxDotErrBnd` are dominated by the uniform magnitude bounds). -/
theorem idxDotErrBnd_le (f g : Nat → Float) (Mf Mg : ℝ) (hMf : 0 ≤ Mf) (hMg : 0 ≤ Mg) :
    ∀ k : Nat, (∀ l < k, |toReal (f l)| ≤ Mf) → (∀ l < k, |toReal (g l)| ≤ Mg) →
      idxDotErrBnd f g k ≤ idxDotErrBndU Mf Mg k := by
  intro k
  induction k with
  | zero => intro _ _; simp [idxDotErrBnd, idxDotErrBndU]
  | succ k ih =>
      intro hf hg
      have hfk : |toReal (f k)| ≤ Mf := hf k (Nat.lt_succ_self k)
      have hgk : |toReal (g k)| ≤ Mg := hg k (Nat.lt_succ_self k)
      have hfk' := fun l hl => hf l (Nat.lt_succ_of_lt hl)
      have hgk' := fun l hl => hg l (Nat.lt_succ_of_lt hl)
      have ihk := ih hfk' hgk'
      have hmag : |toReal (idxDotF f g k)| ≤ idxDotMagBnd Mf Mg k :=
        idxDotF_abs_le f g Mf Mg hMf hMg k hfk' hgk'
      have hprodR : |toReal (f k) * toReal (g k)| ≤ Mf * Mg := by
        rw [abs_mul]; exact mul_le_mul hfk hgk (abs_nonneg _) hMf
      have hprod : |toReal (f k * g k)| ≤ (1 + u64) * (Mf * Mg) := by
        calc |toReal (f k * g k)| ≤ (1 + u64) * |toReal (f k) * toReal (g k)| := mul_abs_le (f k) (g k)
          _ ≤ (1 + u64) * (Mf * Mg) := mul_le_mul_of_nonneg_left hprodR one_add_u64_nonneg
      have hsum : |toReal (idxDotF f g k) + toReal (f k * g k)|
          ≤ idxDotMagBnd Mf Mg k + (1 + u64) * (Mf * Mg) :=
        (abs_add_le _ _).trans (add_le_add hmag hprod)
      simp only [idxDotErrBnd, idxDotErrBndU]
      exact add_le_add (add_le_add
        (mul_le_mul_of_nonneg_left hsum u64_pos.le) ihk)
        (mul_le_mul_of_nonneg_left hprodR u64_pos.le)

/-! ### (c) Perturbation: Float-real dot vs the exact real-real dot -/

/-- Recursive perturbation bound: each term contributes `Mf·εg + (Mg+εg)·εf` (real exact addition,
    so — unlike the magnitude — errors accumulate ADDITIVELY, no `(1+u64)` compounding). -/
noncomputable def realDotPerturbBnd (Mf Mg εf εg : ℝ) : Nat → ℝ
  | 0 => 0
  | k + 1 => realDotPerturbBnd Mf Mg εf εg k + (Mf * εg + (Mg + εg) * εf)

/-- **Closed form of the input-perturbation bound.** The recursive `realDotPerturbBnd Mf Mg εf εg` — which the
    `matmul` error bound uses for the "inputs already inexact" contribution — is exactly the LINEAR function
    `realDotPerturbBnd Mf Mg εf εg k = k · (Mf·εg + (Mg + εg)·εf)` of the contraction length `k`: every one of the
    `k` terms adds the SAME constant, because — unlike the rounding accumulators — the real mirror's dot is exact
    addition with no `(1 + u64)` compounding. This pins the input-perturbation error as growing exactly LINEARLY in
    the inner matrix dimension `k`, in sharp contrast to the GEOMETRIC growth of the rounding bounds
    `idxDotMagBnd`/`idxDotErrBndU`. Proved by induction on `k` (`Nat.cast_succ` + `ring` at the successor). -/
theorem realDotPerturbBnd_closed (Mf Mg εf εg : ℝ) (k : Nat) :
    realDotPerturbBnd Mf Mg εf εg k = (k : ℝ) * (Mf * εg + (Mg + εg) * εf) := by
  induction k with
  | zero => simp [realDotPerturbBnd]
  | succ k ih =>
      rw [realDotPerturbBnd, ih, Nat.cast_succ]
      ring

/-- **Perturbation bound.** The Float-valued real dot `idxDotR f g k` (= `Σ toReal(f)·toReal(g)`)
    is within `realDotPerturbBnd` of the exact real-real dot `realDotR fR gR k`, given per-index
    magnitude bounds on `f,g` and error bounds `|toReal (f l) − fR l| ≤ εf`, likewise for `g`. -/
theorem idxDotR_perturb (f g : Nat → Float) (fR gR : Nat → ℝ) (Mf Mg εf εg : ℝ) :
    ∀ k : Nat, (∀ l < k, |toReal (f l)| ≤ Mf) → (∀ l < k, |toReal (g l)| ≤ Mg) →
      (∀ l < k, |toReal (f l) - fR l| ≤ εf) → (∀ l < k, |toReal (g l) - gR l| ≤ εg) →
      |idxDotR f g k - realDotR fR gR k| ≤ realDotPerturbBnd Mf Mg εf εg k := by
  intro k
  induction k with
  | zero => intro _ _ _ _; simp [idxDotR, realDotR, realDotPerturbBnd]
  | succ k ih =>
      intro hf hg hef heg
      have ihk := ih (fun l hl => hf l (Nat.lt_succ_of_lt hl)) (fun l hl => hg l (Nat.lt_succ_of_lt hl))
        (fun l hl => hef l (Nat.lt_succ_of_lt hl)) (fun l hl => heg l (Nat.lt_succ_of_lt hl))
      have hfk : |toReal (f k)| ≤ Mf := hf k (Nat.lt_succ_self k)
      have hgk : |toReal (g k)| ≤ Mg := hg k (Nat.lt_succ_self k)
      have hefk : |toReal (f k) - fR k| ≤ εf := hef k (Nat.lt_succ_self k)
      have hegk : |toReal (g k) - gR k| ≤ εg := heg k (Nat.lt_succ_self k)
      -- |gR k| ≤ Mg + εg
      have hgRk : |gR k| ≤ Mg + εg := by
        calc |gR k| ≤ |toReal (g k)| + |gR k - toReal (g k)| := by
              have := abs_add_le (toReal (g k)) (gR k - toReal (g k)); simpa using this
          _ ≤ Mg + εg := by rw [abs_sub_comm]; exact add_le_add hgk hegk
      -- per-term product perturbation
      have hterm : |toReal (f k) * toReal (g k) - fR k * gR k| ≤ Mf * εg + (Mg + εg) * εf := by
        have hsplit : toReal (f k) * toReal (g k) - fR k * gR k
            = toReal (f k) * (toReal (g k) - gR k) + gR k * (toReal (f k) - fR k) := by ring
        calc |toReal (f k) * toReal (g k) - fR k * gR k|
            ≤ |toReal (f k) * (toReal (g k) - gR k)| + |gR k * (toReal (f k) - fR k)| := by
              rw [hsplit]; exact abs_add_le _ _
          _ = |toReal (f k)| * |toReal (g k) - gR k| + |gR k| * |toReal (f k) - fR k| := by
              rw [abs_mul, abs_mul]
          _ ≤ Mf * εg + (Mg + εg) * εf :=
              add_le_add (mul_le_mul hfk hegk (abs_nonneg _) (le_trans (abs_nonneg _) hfk))
                (mul_le_mul hgRk hefk (abs_nonneg _) (le_trans (abs_nonneg _) hgRk))
      simp only [idxDotR, realDotR, realDotPerturbBnd]
      calc |idxDotR f g k + toReal (f k) * toReal (g k) - (realDotR fR gR k + fR k * gR k)|
          = |(idxDotR f g k - realDotR fR gR k) + (toReal (f k) * toReal (g k) - fR k * gR k)| := by
              ring_nf
        _ ≤ |idxDotR f g k - realDotR fR gR k| + |toReal (f k) * toReal (g k) - fR k * gR k| :=
            abs_add_le _ _
        _ ≤ realDotPerturbBnd Mf Mg εf εg k + (Mf * εg + (Mg + εg) * εf) := add_le_add ihk hterm

/-! ### Stage 2: the uniform matrix `matmul` bound

We package a Float matrix `X` with its ℝ mirror `XR` as `MatBnd X XR r c M ε`: both are `r×c`
rectangular, every Float entry has `|toReal (X[i][j])| ≤ M`, and every entry is within `ε` of the
mirror `XR[i][j]`. `matmul_MatBnd` then shows one `matmul` maps `(r×k, M_a, ε_a)·(k×n, M_b, ε_b)` to a
`r×n` product bounded by the composed magnitude `idxDotMagBnd` and error
`idxDotErrBndU + realDotPerturbBnd` — combining `matmul_entry_error` (rounding), `idxDotR_perturb`
(input perturbation), and the Stage-1 uniform bounds. This is the reusable matrix atom the chained
matmuls of one Newton–Schulz iteration compose. -/

open Puffer.FloatR.Muon (Mat matmul transpose)
open Puffer.RL.MuonMatrixRuntime (matmul_entry_error matmul_size matmul_rowSize
  transpose_size transpose_rowSize transpose_getElem)

/-- ℝ matrix. -/
abbrev MatR := Array (Array ℝ)

/-- ℝ mirror of `matmul`: entry `(i,j)` is the exact real dot `realDotR (AR[i]) (BR[·][j])` over the
    inner dimension `kR = AR[0].size`, laid out as `range`-maps for clean entry extraction. -/
noncomputable def matmulR (A B : MatR) : MatR :=
  (Array.range A.size).map (fun i =>
    (Array.range (if B.size = 0 then 0 else (B[0]!).size)).map (fun j =>
      realDotR (fun l => (A[i]!)[l]!) (fun l => (B[l]!)[j]!) (if A.size = 0 then 0 else (A[0]!).size)))

theorem matmulR_size (A B : MatR) : (matmulR A B).size = A.size := by
  simp [matmulR, Array.size_map, Array.size_range]

/-- The `i`-th row of `matmulR` is the `range`-map of column dots. -/
theorem matmulR_row (A B : MatR) (i : Nat) (hi : i < A.size) :
    (matmulR A B)[i]! =
      (Array.range (if B.size = 0 then 0 else (B[0]!).size)).map (fun j =>
        realDotR (fun l => (A[i]!)[l]!) (fun l => (B[l]!)[j]!)
          (if A.size = 0 then 0 else (A[0]!).size)) := by
  rw [matmulR, Array.getElem!_eq_getD]
  simp [Array.getD, hi]

theorem matmulR_rowSize (A B : MatR) (i : Nat) (hi : i < A.size) :
    ((matmulR A B)[i]!).size = (if B.size = 0 then 0 else (B[0]!).size) := by
  rw [matmulR_row A B i hi]; simp [Array.size_map, Array.size_range]

theorem matmulR_getElem (A B : MatR) (i j : Nat) (hi : i < A.size)
    (hj : j < (if B.size = 0 then 0 else (B[0]!).size)) :
    ((matmulR A B)[i]!)[j]! =
      realDotR (fun l => (A[i]!)[l]!) (fun l => (B[l]!)[j]!) (if A.size = 0 then 0 else (A[0]!).size) := by
  rw [matmulR_row A B i hi]
  set n' := if B.size = 0 then 0 else (B[0]!).size with hn'
  rw [Array.getElem!_eq_getD]
  simp [Array.getD, hj, Array.getElem_map, Array.getElem_range]

/-- Float matrix `X` (with ℝ mirror `XR`) is `r×c` rectangular, entrywise bounded by `M` in
    magnitude and within `ε` of the mirror. -/
structure MatBnd (X : Mat) (XR : MatR) (r c : Nat) (M ε : ℝ) : Prop where
  sizeX : X.size = r
  sizeXR : XR.size = r
  rowX : ∀ i < r, (X[i]!).size = c
  rowXR : ∀ i < r, (XR[i]!).size = c
  mag : ∀ i < r, ∀ j < c, |toReal ((X[i]!)[j]!)| ≤ M
  err : ∀ i < r, ∀ j < c, |toReal ((X[i]!)[j]!) - (XR[i]!)[j]!| ≤ ε

/-- **`MatBnd` monotonicity (bound weakening).** A Float matrix `X` bounded by `(M, ε)` against its ℝ mirror `XR`
    is also bounded by any looser pair `(M', ε')` with `M ≤ M'` and `ε ≤ ε'`. The shape fields (`size`/`row`) are
    inherited unchanged; the magnitude and error bounds relax by transitivity. This is the reusable weakening step
    that lets a chain of `matmul`/`transpose`/`lincomb3` bounds be reconciled against a single dominating bound. -/
theorem MatBnd_mono (X : Mat) (XR : MatR) (r c : Nat) (M ε M' ε' : ℝ)
    (hM : M ≤ M') (hε : ε ≤ ε') (h : MatBnd X XR r c M ε) :
    MatBnd X XR r c M' ε' where
  sizeX := h.sizeX
  sizeXR := h.sizeXR
  rowX := h.rowX
  rowXR := h.rowXR
  mag := fun i hi j hj => le_trans (h.mag i hi j hj) hM
  err := fun i hi j hj => le_trans (h.err i hi j hj) hε

/-- **Uniform matrix `matmul` bound.** `matmul` of an `r×k` (bound `Ma,εa`) and a `k×n` (bound
    `Mb,εb`) Float matrix yields an `r×n` product entrywise bounded by the composed magnitude
    `idxDotMagBnd Ma Mb k` and error `idxDotErrBndU Ma Mb k + realDotPerturbBnd Ma Mb εa εb k`
    (against the mirror `matmulR AR BR`). Requires `0 < k` (nondegenerate inner dimension). -/
theorem matmul_MatBnd (A B : Mat) (AR BR : MatR) (r k n : Nat) (Ma εa Mb εb : ℝ)
    (hMa : 0 ≤ Ma) (hMb : 0 ≤ Mb) (hk : 0 < k)
    (hA : MatBnd A AR r k Ma εa) (hB : MatBnd B BR k n Mb εb) :
    MatBnd (matmul A B) (matmulR AR BR) r n
      (idxDotMagBnd Ma Mb k) (idxDotErrBndU Ma Mb k + realDotPerturbBnd Ma Mb εa εb k) where
  sizeX := by rw [matmul_size]; exact hA.sizeX
  sizeXR := by rw [matmulR_size]; exact hA.sizeXR
  rowX := by
    intro i hi
    have hr : 0 < r := Nat.lt_of_le_of_lt (Nat.zero_le i) hi
    have hi' : i < A.size := by rw [hA.sizeX]; exact hi
    have hBne : B ≠ #[] := by
      intro h; have := hB.sizeX; rw [h] at this; simp at this; omega
    rw [matmul_rowSize A B i hi', if_neg hBne, hB.rowX 0 hk]
  rowXR := by
    intro i hi
    have hi' : i < AR.size := by rw [hA.sizeXR]; exact hi
    have hnpos : ¬ (BR.size = 0) := by rw [hB.sizeXR]; omega
    rw [matmulR_rowSize AR BR i hi', if_neg hnpos, hB.rowXR 0 hk]
  mag := by
    intro i hi j hj
    have hr : 0 < r := Nat.lt_of_le_of_lt (Nat.zero_le i) hi
    have hi' : i < A.size := by rw [hA.sizeX]; exact hi
    have hAne : A ≠ #[] := by
      intro h; have := hA.sizeX; rw [h] at this; simp at this; omega
    have hBne : B ≠ #[] := by
      intro h; have := hB.sizeX; rw [h] at this; simp at this; omega
    have hkeq : (if A = #[] then 0 else (A[0]!).size) = k := by rw [if_neg hAne, hA.rowX 0 hr]
    have hjB : j < (if B = #[] then 0 else (B[0]!).size) := by rw [if_neg hBne, hB.rowX 0 hk]; exact hj
    rw [Puffer.RL.MuonMatrixRuntime.matmul_getElem A B i j hi' hjB, hkeq]
    -- per-index magnitude hypotheses
    refine idxDotF_abs_le _ _ Ma Mb hMa hMb k ?_ ?_
    · intro l hl; exact hA.mag i hi l hl
    · intro l hl; exact hB.mag l hl j hj
  err := by
    intro i hi j hj
    have hr : 0 < r := Nat.lt_of_le_of_lt (Nat.zero_le i) hi
    have hi' : i < A.size := by rw [hA.sizeX]; exact hi
    have hiR : i < AR.size := by rw [hA.sizeXR]; exact hi
    have hAne : A ≠ #[] := by
      intro h; have := hA.sizeX; rw [h] at this; simp at this; omega
    have hBne : B ≠ #[] := by
      intro h; have := hB.sizeX; rw [h] at this; simp at this; omega
    have hkeq : (if A = #[] then 0 else (A[0]!).size) = k := by rw [if_neg hAne, hA.rowX 0 hr]
    have hjB : j < (if B = #[] then 0 else (B[0]!).size) := by rw [if_neg hBne, hB.rowX 0 hk]; exact hj
    have hjR : j < (if BR.size = 0 then 0 else (BR[0]!).size) := by
      have : ¬ (BR.size = 0) := by rw [hB.sizeXR]; omega
      rw [if_neg this, hB.rowXR 0 hk]; exact hj
    have hkR : (if AR.size = 0 then 0 else (AR[0]!).size) = k := by
      have : ¬ (AR.size = 0) := by rw [hA.sizeXR]; omega
      rw [if_neg this, hA.rowXR 0 hr]
    -- split: matmul entry vs idxDotR (rounding) + idxDotR vs realDotR (perturbation)
    rw [matmulR_getElem AR BR i j hiR hjR, hkR]
    have hround := matmul_entry_error A B i j hi' hjB
    rw [hkeq] at hround
    have hpert := idxDotR_perturb (fun l => (A[i]!)[l]!) (fun l => (B[l]!)[j]!)
      (fun l => (AR[i]!)[l]!) (fun l => (BR[l]!)[j]!) Ma Mb εa εb k
      (fun l hl => hA.mag i hi l hl) (fun l hl => hB.mag l hl j hj)
      (fun l hl => hA.err i hi l hl) (fun l hl => hB.err l hl j hj)
    have hEB := idxDotErrBnd_le (fun l => (A[i]!)[l]!) (fun l => (B[l]!)[j]!) Ma Mb hMa hMb k
      (fun l hl => hA.mag i hi l hl) (fun l hl => hB.mag l hl j hj)
    calc |toReal (((matmul A B)[i]!)[j]!) - realDotR (fun l => (AR[i]!)[l]!) (fun l => (BR[l]!)[j]!) k|
        ≤ |toReal (((matmul A B)[i]!)[j]!) - idxDotR (fun l => (A[i]!)[l]!) (fun l => (B[l]!)[j]!) k|
            + |idxDotR (fun l => (A[i]!)[l]!) (fun l => (B[l]!)[j]!) k
              - realDotR (fun l => (AR[i]!)[l]!) (fun l => (BR[l]!)[j]!) k| := abs_sub_le _ _ _
      _ ≤ idxDotErrBndU Ma Mb k + realDotPerturbBnd Ma Mb εa εb k :=
          add_le_add (le_trans hround hEB) hpert

/-! ### Stage 3: the map-based matrix atoms (`scalarMul`, `lincomb3`)

Unlike `matmul`/`transpose`, these are plain `Array.map`/`range`-map operations (no `Id.run`), so
their entry extraction is direct. `scalarMul c X` scales every entry; `lincomb3 a X b Y c Z` is the
per-entry Newton–Schulz combine, whose bound reuses `lincomb3Entry_error`. -/

open Puffer.FloatR.Muon (scalarMul lincomb3)
open Puffer.RL.MuonMatrixRuntime (lincomb3EntryErrBnd matLinEntryErrBnd lincomb3Entry_error)

/-- `getElem!` through a plain `Array.map` (in range). -/
theorem arrMap_getElem! {α β} [Inhabited α] [Inhabited β] (X : Array α) (g : α → β) (i : Nat)
    (hi : i < X.size) : (X.map g)[i]! = g (X[i]!) := by
  rw [Array.getElem!_eq_getD, Array.getElem!_eq_getD, Array.getD, Array.getD]
  simp [hi, Array.getElem_map]

/-- `getElem!` through a `range`-map (in range). -/
theorem rangeMap_getElem! {β} [Inhabited β] (m : Nat) (g : Nat → β) (i : Nat) (hi : i < m) :
    ((Array.range m).map g)[i]! = g i := by
  rw [Array.getElem!_eq_getD, Array.getD]; simp [hi, Array.getElem_map, Array.getElem_range]

/-- ℝ mirror of `scalarMul`. -/
noncomputable def scalarMulR (c : ℝ) (X : MatR) : MatR := X.map (·.map (c * ·))

theorem scalarMul_getElem (c : Float) (X : Mat) (i j : Nat) (hi : i < X.size)
    (hj : j < (X[i]!).size) : ((scalarMul c X)[i]!)[j]! = c * ((X[i]!)[j]!) := by
  simp only [scalarMul]
  rw [arrMap_getElem! X (·.map (c * ·)) i hi, arrMap_getElem! (X[i]!) (c * ·) j hj]

theorem scalarMulR_getElem (c : ℝ) (X : MatR) (i j : Nat) (hi : i < X.size)
    (hj : j < (X[i]!).size) : ((scalarMulR c X)[i]!)[j]! = c * ((X[i]!)[j]!) := by
  simp only [scalarMulR]
  rw [arrMap_getElem! X (·.map (c * ·)) i hi, arrMap_getElem! (X[i]!) (c * ·) j hj]

theorem scalarMul_size (c : Float) (X : Mat) : (scalarMul c X).size = X.size := by
  simp [scalarMul, Array.size_map]

theorem scalarMul_rowSize (c : Float) (X : Mat) (i : Nat) (hi : i < X.size) :
    ((scalarMul c X)[i]!).size = (X[i]!).size := by
  simp only [scalarMul]; rw [arrMap_getElem! X (·.map (c * ·)) i hi]; simp [Array.size_map]

theorem scalarMulR_size (c : ℝ) (X : MatR) : (scalarMulR c X).size = X.size := by
  simp [scalarMulR, Array.size_map]

theorem scalarMulR_rowSize (c : ℝ) (X : MatR) (i : Nat) (hi : i < X.size) :
    ((scalarMulR c X)[i]!).size = (X[i]!).size := by
  simp only [scalarMulR]; rw [arrMap_getElem! X (·.map (c * ·)) i hi]; simp [Array.size_map]

/-- **`scalarMul` matrix bound.** Scaling by `c` scales the magnitude by `(1+u64)·|toReal c|` and
    the error by `|toReal c|` (plus the `u64` rounding of the scale). -/
theorem scalarMul_MatBnd (c : Float) (X : Mat) (XR : MatR) (r cc : Nat) (M ε : ℝ)
    (_hM : 0 ≤ M) (hX : MatBnd X XR r cc M ε) :
    MatBnd (scalarMul c X) (scalarMulR (toReal c) XR) r cc
      ((1 + u64) * (|toReal c| * M)) (u64 * (|toReal c| * M) + |toReal c| * ε) where
  sizeX := by rw [scalarMul_size]; exact hX.sizeX
  sizeXR := by rw [scalarMulR_size]; exact hX.sizeXR
  rowX := by intro i hi; rw [scalarMul_rowSize c X i (by rw [hX.sizeX]; exact hi)]; exact hX.rowX i hi
  rowXR := by
    intro i hi; rw [scalarMulR_rowSize (toReal c) XR i (by rw [hX.sizeXR]; exact hi)]
    exact hX.rowXR i hi
  mag := by
    intro i hi j hj
    rw [scalarMul_getElem c X i j (by rw [hX.sizeX]; exact hi) (by rw [hX.rowX i hi]; exact hj)]
    calc |toReal (c * (X[i]!)[j]!)| ≤ (1 + u64) * |toReal c * toReal ((X[i]!)[j]!)| := mul_abs_le _ _
      _ = (1 + u64) * (|toReal c| * |toReal ((X[i]!)[j]!)|) := by rw [abs_mul]
      _ ≤ (1 + u64) * (|toReal c| * M) :=
          mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left (hX.mag i hi j hj) (abs_nonneg _))
            one_add_u64_nonneg
  err := by
    intro i hi j hj
    rw [scalarMul_getElem c X i j (by rw [hX.sizeX]; exact hi) (by rw [hX.rowX i hi]; exact hj),
      scalarMulR_getElem (toReal c) XR i j (by rw [hX.sizeXR]; exact hi) (by rw [hX.rowXR i hi]; exact hj)]
    have h := mulApprox_error c ((X[i]!)[j]!) (toReal c) ((XR[i]!)[j]!) 0 ε (by simp) (hX.err i hi j hj)
    have hmag : |toReal c * toReal ((X[i]!)[j]!)| ≤ |toReal c| * M := by
      rw [abs_mul]; exact mul_le_mul_of_nonneg_left (hX.mag i hi j hj) (abs_nonneg _)
    calc |toReal (c * (X[i]!)[j]!) - toReal c * (XR[i]!)[j]!|
        ≤ u64 * |toReal c * toReal ((X[i]!)[j]!)| + |toReal c| * ε + |(XR[i]!)[j]!| * 0 := h
      _ = u64 * |toReal c * toReal ((X[i]!)[j]!)| + |toReal c| * ε := by ring
      _ ≤ u64 * (|toReal c| * M) + |toReal c| * ε :=
          add_le_add (mul_le_mul_of_nonneg_left hmag u64_pos.le) le_rfl

/-- ℝ mirror of `lincomb3`. -/
noncomputable def lincomb3R (a : ℝ) (X : MatR) (b : ℝ) (Y : MatR) (c : ℝ) (Z : MatR) : MatR :=
  (Array.range X.size).map (fun i => (Array.range (X[i]!).size).map (fun j =>
    a * (X[i]!)[j]! + b * (Y[i]!)[j]! + c * (Z[i]!)[j]!))

theorem lincomb3_getElem (a : Float) (X : Mat) (b : Float) (Y : Mat) (c : Float) (Z : Mat)
    (i j : Nat) (hi : i < X.size) (hj : j < (X[i]!).size) :
    ((lincomb3 a X b Y c Z)[i]!)[j]! =
      a * (X[i]!)[j]! + b * (Y[i]!)[j]! + c * (Z[i]!)[j]! := by
  rw [lincomb3, rangeMap_getElem! X.size _ i hi, rangeMap_getElem! (X[i]!).size _ j hj]

theorem lincomb3R_getElem (a : ℝ) (X : MatR) (b : ℝ) (Y : MatR) (c : ℝ) (Z : MatR)
    (i j : Nat) (hi : i < X.size) (hj : j < (X[i]!).size) :
    ((lincomb3R a X b Y c Z)[i]!)[j]! =
      a * (X[i]!)[j]! + b * (Y[i]!)[j]! + c * (Z[i]!)[j]! := by
  rw [lincomb3R, rangeMap_getElem! X.size _ i hi, rangeMap_getElem! (X[i]!).size _ j hj]

theorem lincomb3_size (a : Float) (X : Mat) (b : Float) (Y : Mat) (c : Float) (Z : Mat) :
    (lincomb3 a X b Y c Z).size = X.size := by simp [lincomb3, Array.size_map, Array.size_range]

theorem lincomb3_rowSize (a : Float) (X : Mat) (b : Float) (Y : Mat) (c : Float) (Z : Mat)
    (i : Nat) (hi : i < X.size) : ((lincomb3 a X b Y c Z)[i]!).size = (X[i]!).size := by
  rw [lincomb3, rangeMap_getElem! X.size _ i hi]; simp [Array.size_map, Array.size_range]

theorem lincomb3R_size (a : ℝ) (X : MatR) (b : ℝ) (Y : MatR) (c : ℝ) (Z : MatR) :
    (lincomb3R a X b Y c Z).size = X.size := by simp [lincomb3R, Array.size_map, Array.size_range]

theorem lincomb3R_rowSize (a : ℝ) (X : MatR) (b : ℝ) (Y : MatR) (c : ℝ) (Z : MatR)
    (i : Nat) (hi : i < X.size) : ((lincomb3R a X b Y c Z)[i]!).size = (X[i]!).size := by
  rw [lincomb3R, rangeMap_getElem! X.size _ i hi]; simp [Array.size_map, Array.size_range]

/-! Generic magnitude helpers for rounded scale/sum (used to make `lincomb3`'s bounds uniform). -/

/-- `|fl(a·x)| ≤ (1+u64)·(|a|·Mx)` when `|toReal x| ≤ Mx`. -/
theorem absmul_le (a x : Float) (Mx : ℝ) (hx : |toReal x| ≤ Mx) :
    |toReal (a * x)| ≤ (1 + u64) * (|toReal a| * Mx) := by
  calc |toReal (a * x)| ≤ (1 + u64) * |toReal a * toReal x| := mul_abs_le a x
    _ = (1 + u64) * (|toReal a| * |toReal x|) := by rw [abs_mul]
    _ ≤ (1 + u64) * (|toReal a| * Mx) :=
        mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hx (abs_nonneg _)) one_add_u64_nonneg

/-- `|fl(u+v)| ≤ (1+u64)·(Bu+Bv)` when `|toReal u| ≤ Bu`, `|toReal v| ≤ Bv`. -/
theorem absadd_le (u v : Float) (Bu Bv : ℝ) (hu : |toReal u| ≤ Bu) (hv : |toReal v| ≤ Bv) :
    |toReal (u + v)| ≤ (1 + u64) * (Bu + Bv) := by
  calc |toReal (u + v)| ≤ (1 + u64) * |toReal u + toReal v| := add_abs_le u v
    _ ≤ (1 + u64) * (|toReal u| + |toReal v|) :=
        mul_le_mul_of_nonneg_left (abs_add_le _ _) one_add_u64_nonneg
    _ ≤ (1 + u64) * (Bu + Bv) := mul_le_mul_of_nonneg_left (add_le_add hu hv) one_add_u64_nonneg

/-- Uniform magnitude bound for one `lincomb3` entry `a·x + b·y + c·z` (left-assoc, each op rounded). -/
noncomputable def lincomb3MagBnd (a b c : Float) (Mx My Mz : ℝ) : ℝ :=
  (1 + u64) * ((1 + u64) * ((1 + u64) * (|toReal a| * Mx) + (1 + u64) * (|toReal b| * My))
    + (1 + u64) * (|toReal c| * Mz))

theorem lincomb3_mag_le (a b c x y z : Float) (Mx My Mz : ℝ)
    (hx : |toReal x| ≤ Mx) (hy : |toReal y| ≤ My) (hz : |toReal z| ≤ Mz) :
    |toReal (a * x + b * y + c * z)| ≤ lincomb3MagBnd a b c Mx My Mz := by
  have hxy : |toReal (a * x + b * y)|
      ≤ (1 + u64) * ((1 + u64) * (|toReal a| * Mx) + (1 + u64) * (|toReal b| * My)) :=
    absadd_le _ _ _ _ (absmul_le a x Mx hx) (absmul_le b y My hy)
  exact absadd_le _ _ _ _ hxy (absmul_le c z Mz hz)

/-- Uniform error bound for one `lincomb3` entry, dominating the value-dependent
    `lincomb3EntryErrBnd` via the magnitude bounds. -/
noncomputable def lincomb3EntryErrBndU (a b c : Float) (Mx εx My εy Mz εz : ℝ) : ℝ :=
  u64 * ((1 + u64) * ((1 + u64) * (|toReal a| * Mx) + (1 + u64) * (|toReal b| * My))
      + (1 + u64) * (|toReal c| * Mz))
  + (u64 * ((1 + u64) * (|toReal a| * Mx) + (1 + u64) * (|toReal b| * My))
      + (u64 * (|toReal a| * Mx) + |toReal a| * εx)
      + (u64 * (|toReal b| * My) + |toReal b| * εy))
  + (u64 * (|toReal c| * Mz) + |toReal c| * εz)

theorem lincomb3_errbnd_le (a x b y c z : Float) (Mx εx My εy Mz εz : ℝ)
    (hx : |toReal x| ≤ Mx) (hy : |toReal y| ≤ My) (hz : |toReal z| ≤ Mz) :
    lincomb3EntryErrBnd a x b y c z εx εy εz ≤ lincomb3EntryErrBndU a b c Mx εx My εy Mz εz := by
  have hax : |toReal (a * x)| ≤ (1 + u64) * (|toReal a| * Mx) := absmul_le a x Mx hx
  have hby : |toReal (b * y)| ≤ (1 + u64) * (|toReal b| * My) := absmul_le b y My hy
  have hprodA : |toReal a * toReal x| ≤ |toReal a| * Mx := by
    rw [abs_mul]; exact mul_le_mul_of_nonneg_left hx (abs_nonneg _)
  have hprodB : |toReal b * toReal y| ≤ |toReal b| * My := by
    rw [abs_mul]; exact mul_le_mul_of_nonneg_left hy (abs_nonneg _)
  have hprodC : |toReal c * toReal z| ≤ |toReal c| * Mz := by
    rw [abs_mul]; exact mul_le_mul_of_nonneg_left hz (abs_nonneg _)
  have hsumAB : |toReal (a * x) + toReal (b * y)|
      ≤ (1 + u64) * (|toReal a| * Mx) + (1 + u64) * (|toReal b| * My) :=
    (abs_add_le _ _).trans (add_le_add hax hby)
  have hxyMag : |toReal (a * x + b * y)|
      ≤ (1 + u64) * ((1 + u64) * (|toReal a| * Mx) + (1 + u64) * (|toReal b| * My)) :=
    absadd_le _ _ _ _ hax hby
  have hczMag : |toReal (c * z)| ≤ (1 + u64) * (|toReal c| * Mz) := absmul_le c z Mz hz
  have hsumABC : |toReal (a * x + b * y) + toReal (c * z)|
      ≤ (1 + u64) * ((1 + u64) * (|toReal a| * Mx) + (1 + u64) * (|toReal b| * My))
        + (1 + u64) * (|toReal c| * Mz) :=
    (abs_add_le _ _).trans (add_le_add hxyMag hczMag)
  unfold lincomb3EntryErrBnd matLinEntryErrBnd lincomb3EntryErrBndU
  gcongr <;> first
    | exact hsumABC | exact hsumAB | exact hprodA | exact hprodB | exact hprodC
    | exact u64_pos.le

/-- **`lincomb3` matrix bound.** The per-entry Newton–Schulz combine of three `r×c` matrices (all
    same shape) is bounded uniformly: `lincomb3_mag_le` for magnitude and
    `lincomb3Entry_error` composed with `lincomb3_errbnd_le` for the error. -/
theorem lincomb3_MatBnd (a : Float) (X : Mat) (b : Float) (Y : Mat) (c : Float) (Z : Mat)
    (XR YR ZR : MatR) (r cc : Nat) (Mx εx My εy Mz εz : ℝ)
    (hX : MatBnd X XR r cc Mx εx) (hY : MatBnd Y YR r cc My εy) (hZ : MatBnd Z ZR r cc Mz εz) :
    MatBnd (lincomb3 a X b Y c Z) (lincomb3R (toReal a) XR (toReal b) YR (toReal c) ZR) r cc
      (lincomb3MagBnd a b c Mx My Mz)
      (lincomb3EntryErrBndU a b c Mx εx My εy Mz εz) where
  sizeX := by rw [lincomb3_size]; exact hX.sizeX
  sizeXR := by rw [lincomb3R_size]; exact hX.sizeXR
  rowX := by
    intro i hi; rw [lincomb3_rowSize a X b Y c Z i (by rw [hX.sizeX]; exact hi)]; exact hX.rowX i hi
  rowXR := by
    intro i hi
    rw [lincomb3R_rowSize (toReal a) XR (toReal b) YR (toReal c) ZR i (by rw [hX.sizeXR]; exact hi)]
    exact hX.rowXR i hi
  mag := by
    intro i hi j hj
    have hix : i < X.size := by rw [hX.sizeX]; exact hi
    have hjx : j < (X[i]!).size := by rw [hX.rowX i hi]; exact hj
    rw [lincomb3_getElem a X b Y c Z i j hix hjx]
    exact lincomb3_mag_le a b c ((X[i]!)[j]!) ((Y[i]!)[j]!) ((Z[i]!)[j]!) Mx My Mz
      (hX.mag i hi j hj) (hY.mag i hi j hj) (hZ.mag i hi j hj)
  err := by
    intro i hi j hj
    have hix : i < X.size := by rw [hX.sizeX]; exact hi
    have hjx : j < (X[i]!).size := by rw [hX.rowX i hi]; exact hj
    rw [lincomb3_getElem a X b Y c Z i j hix hjx,
      lincomb3R_getElem (toReal a) XR (toReal b) YR (toReal c) ZR i j
        (by rw [hX.sizeXR]; exact hi) (by rw [hX.rowXR i hi]; exact hj)]
    refine le_trans (lincomb3Entry_error a ((X[i]!)[j]!) b ((Y[i]!)[j]!) c ((Z[i]!)[j]!)
      ((XR[i]!)[j]!) ((YR[i]!)[j]!) ((ZR[i]!)[j]!) εx εy εz
      (hX.err i hi j hj) (hY.err i hi j hj) (hZ.err i hi j hj)) ?_
    exact lincomb3_errbnd_le a ((X[i]!)[j]!) b ((Y[i]!)[j]!) c ((Z[i]!)[j]!) Mx εx My εy Mz εz
      (hX.mag i hi j hj) (hY.mag i hi j hj) (hZ.mag i hi j hj)

/-! ### `transpose` matrix bound (pure reindexing)

`transpose` moves entry `X[i][j]` to position `[j][i]` with no arithmetic, so it preserves the
magnitude and error bounds exactly and swaps the dimensions `r×c → c×r`. -/

/-- ℝ mirror of `transpose`. -/
noncomputable def transposeR (X : MatR) : MatR :=
  (Array.range (if X.size = 0 then 0 else (X[0]!).size)).map (fun j =>
    (Array.range X.size).map (fun i => (X[i]!)[j]!))

theorem transposeR_size (X : MatR) :
    (transposeR X).size = (if X.size = 0 then 0 else (X[0]!).size) := by
  simp [transposeR, Array.size_map, Array.size_range]

theorem transposeR_rowSize (X : MatR) (j : Nat) (hj : j < (if X.size = 0 then 0 else (X[0]!).size)) :
    ((transposeR X)[j]!).size = X.size := by
  rw [transposeR, rangeMap_getElem! _ _ j hj]; simp [Array.size_map, Array.size_range]

theorem transposeR_getElem (X : MatR) (i j : Nat)
    (hj : j < (if X.size = 0 then 0 else (X[0]!).size)) (hi : i < X.size) :
    ((transposeR X)[j]!)[i]! = (X[i]!)[j]! := by
  rw [transposeR, rangeMap_getElem! _ _ j hj, rangeMap_getElem! _ _ i hi]

/-- **`transpose` matrix bound.** `transpose` of an `r×c` matrix (bound `M,ε`) is a `c×r` matrix with
    the SAME bounds (pure reindexing). Requires `0 < r` (nonempty, so the `c` dimension is `X[0].size`). -/
theorem transpose_MatBnd (X : Mat) (XR : MatR) (r cc : Nat) (M ε : ℝ) (hr : 0 < r)
    (hX : MatBnd X XR r cc M ε) :
    MatBnd (transpose X) (transposeR XR) cc r M ε where
  sizeX := by
    have hXne : X ≠ #[] := by intro h; have := hX.sizeX; rw [h] at this; simp at this; omega
    rw [transpose_size, if_neg hXne, hX.rowX 0 hr]
  sizeXR := by
    have : ¬ (XR.size = 0) := by rw [hX.sizeXR]; omega
    rw [transposeR_size, if_neg this, hX.rowXR 0 hr]
  rowX := by
    intro j hj
    have hXne : X ≠ #[] := by intro h; have := hX.sizeX; rw [h] at this; simp at this; omega
    have hj' : j < (if X = #[] then 0 else (X[0]!).size) := by rw [if_neg hXne, hX.rowX 0 hr]; exact hj
    rw [transpose_rowSize X j hj', hX.sizeX]
  rowXR := by
    intro j hj
    have hne : ¬ (XR.size = 0) := by rw [hX.sizeXR]; omega
    have hj' : j < (if XR.size = 0 then 0 else (XR[0]!).size) := by
      rw [if_neg hne, hX.rowXR 0 hr]; exact hj
    rw [transposeR_rowSize XR j hj', hX.sizeXR]
  mag := by
    intro j hj i hi
    have hXne : X ≠ #[] := by intro h; have := hX.sizeX; rw [h] at this; simp at this; omega
    have hj' : j < (if X = #[] then 0 else (X[0]!).size) := by rw [if_neg hXne, hX.rowX 0 hr]; exact hj
    rw [transpose_getElem X i j hj' (by rw [hX.sizeX]; exact hi)]
    exact hX.mag i hi j hj
  err := by
    intro j hj i hi
    have hXne : X ≠ #[] := by intro h; have := hX.sizeX; rw [h] at this; simp at this; omega
    have hj' : j < (if X = #[] then 0 else (X[0]!).size) := by rw [if_neg hXne, hX.rowX 0 hr]; exact hj
    have hjR : j < (if XR.size = 0 then 0 else (XR[0]!).size) := by
      have hne : ¬ (XR.size = 0) := by rw [hX.sizeXR]; omega
      rw [if_neg hne, hX.rowXR 0 hr]; exact hj
    rw [transpose_getElem X i j hj' (by rw [hX.sizeX]; exact hi),
      transposeR_getElem XR i j hjR (by rw [hX.sizeXR]; exact hi)]
    exact hX.err i hi j hj

/-! ### Stage 4: assembling one Newton–Schulz iteration (`nsIter`)

`nsIter` preserves the `r×c` shape and its shape-branch is fixed by `(r,c)`, so within one
`newtonSchulz` run the same branch is taken every iteration. We give the ℝ mirror `nsIterR` and prove
the per-iteration `MatBnd` for the `r ≤ c` branch by chaining `transpose_MatBnd`, three
`matmul_MatBnd`s, and `lincomb3_MatBnd` — the full composition of the atoms. -/

open Puffer.RL.MuonMatrixRuntime (nsIter)

/-- `idxDotMagBnd` is nonnegative (needed to feed successive `matmul_MatBnd`s). -/
theorem idxDotMagBnd_nonneg (Mf Mg : ℝ) (hMf : 0 ≤ Mf) (hMg : 0 ≤ Mg) :
    ∀ k, 0 ≤ idxDotMagBnd Mf Mg k := by
  intro k
  induction k with
  | zero => simp [idxDotMagBnd]
  | succ k ih =>
      simp only [idxDotMagBnd]
      have : (0:ℝ) ≤ idxDotMagBnd Mf Mg k + (1 + u64) * (Mf * Mg) :=
        add_nonneg ih (mul_nonneg one_add_u64_nonneg (mul_nonneg hMf hMg))
      exact mul_nonneg this one_add_u64_nonneg

/-- ℝ mirror of one Newton–Schulz iteration (`nsIterR`), branching on the real matrix's own shape. -/
noncomputable def nsIterR (X : MatR) (coef : Float × Float × Float) : MatR :=
  let (a, b, c) := coef
  if X.size ≤ (X[0]!).size then
    let A := matmulR X (transposeR X)
    let AX := matmulR A X
    lincomb3R (toReal a) X (toReal b) AX (toReal c) (matmulR A AX)
  else
    let A := matmulR (transposeR X) X
    let XA := matmulR X A
    lincomb3R (toReal a) X (toReal b) XA (toReal c) (matmulR XA A)

/-- Composed magnitude bound for one iteration (`r ≤ c` branch). -/
noncomputable def nsIterThenMag (a b cc : Float) (M : ℝ) (r c : Nat) : ℝ :=
  let MA := idxDotMagBnd M M c
  let MAX := idxDotMagBnd MA M r
  let MAAX := idxDotMagBnd MA MAX r
  lincomb3MagBnd a b cc M MAX MAAX

/-- Composed error bound for one iteration (`r ≤ c` branch). -/
noncomputable def nsIterThenErr (a b cc : Float) (M ε : ℝ) (r c : Nat) : ℝ :=
  let MA := idxDotMagBnd M M c
  let εA := idxDotErrBndU M M c + realDotPerturbBnd M M ε ε c
  let MAX := idxDotMagBnd MA M r
  let εAX := idxDotErrBndU MA M r + realDotPerturbBnd MA M εA ε r
  let MAAX := idxDotMagBnd MA MAX r
  let εAAX := idxDotErrBndU MA MAX r + realDotPerturbBnd MA MAX εA εAX r
  lincomb3EntryErrBndU a b cc M ε MAX εAX MAAX εAAX

/-- **One Newton–Schulz iteration bound (`r ≤ c` branch).** For a nonempty `r×c` matrix with `r ≤ c`,
    `nsIter X (a,b,c)` is within the composed `MatBnd` of the real mirror `nsIterR XR (a,b,c)`. Chains
    `transpose_MatBnd` + 3 `matmul_MatBnd`s + `lincomb3_MatBnd`. -/
theorem nsIter_MatBnd_le (X : Mat) (XR : MatR) (r c : Nat) (M ε : ℝ) (a b cc : Float)
    (hM : 0 ≤ M) (hr : 0 < r) (hrc : r ≤ c) (hX : MatBnd X XR r c M ε) :
    MatBnd (nsIter X (a, b, cc)) (nsIterR XR (a, b, cc)) r c
      (nsIterThenMag a b cc M r c) (nsIterThenErr a b cc M ε r c) := by
  have hc : 0 < c := lt_of_lt_of_le hr hrc
  have hMA : 0 ≤ idxDotMagBnd M M c := idxDotMagBnd_nonneg M M hM hM c
  have hMAX : 0 ≤ idxDotMagBnd (idxDotMagBnd M M c) M r :=
    idxDotMagBnd_nonneg _ M hMA hM r
  -- shape-branch conditions
  have hcond : X.size ≤ (X[0]!).size := by rw [hX.sizeX, hX.rowX 0 hr]; exact hrc
  have hcondR : XR.size ≤ (XR[0]!).size := by rw [hX.sizeXR, hX.rowXR 0 hr]; exact hrc
  -- step 1: transpose (c×r)
  have htX : MatBnd (transpose X) (transposeR XR) c r M ε := transpose_MatBnd X XR r c M ε hr hX
  -- step 2: A = X·Xᵀ (r×r), inner dim c
  have hA := matmul_MatBnd X (transpose X) XR (transposeR XR) r c r M ε M ε hM hM hc hX htX
  -- step 3: AX = A·X (r×c), inner dim r
  have hAX := matmul_MatBnd (matmul X (transpose X)) X (matmulR XR (transposeR XR)) XR r r c
    (idxDotMagBnd M M c) (idxDotErrBndU M M c + realDotPerturbBnd M M ε ε c) M ε hMA hM hr hA hX
  -- step 4: AAX = A·AX (r×c), inner dim r
  have hAAX := matmul_MatBnd (matmul X (transpose X)) (matmul (matmul X (transpose X)) X)
    (matmulR XR (transposeR XR)) (matmulR (matmulR XR (transposeR XR)) XR) r r c
    (idxDotMagBnd M M c) (idxDotErrBndU M M c + realDotPerturbBnd M M ε ε c)
    (idxDotMagBnd (idxDotMagBnd M M c) M r)
    (idxDotErrBndU (idxDotMagBnd M M c) M r
      + realDotPerturbBnd (idxDotMagBnd M M c) M (idxDotErrBndU M M c + realDotPerturbBnd M M ε ε c) ε r)
    hMA hMAX hr hA hAX
  -- step 5: lincomb3 a·X + b·AX + c·AAX (r×c)
  have hout := lincomb3_MatBnd a X b (matmul (matmul X (transpose X)) X) cc
    (matmul (matmul X (transpose X)) (matmul (matmul X (transpose X)) X))
    XR (matmulR (matmulR XR (transposeR XR)) XR)
    (matmulR (matmulR XR (transposeR XR)) (matmulR (matmulR XR (transposeR XR)) XR)) r c
    M ε
    (idxDotMagBnd (idxDotMagBnd M M c) M r)
    (idxDotErrBndU (idxDotMagBnd M M c) M r
      + realDotPerturbBnd (idxDotMagBnd M M c) M (idxDotErrBndU M M c + realDotPerturbBnd M M ε ε c) ε r)
    (idxDotMagBnd (idxDotMagBnd M M c) (idxDotMagBnd (idxDotMagBnd M M c) M r) r)
    (idxDotErrBndU (idxDotMagBnd M M c) (idxDotMagBnd (idxDotMagBnd M M c) M r) r
      + realDotPerturbBnd (idxDotMagBnd M M c) (idxDotMagBnd (idxDotMagBnd M M c) M r)
        (idxDotErrBndU M M c + realDotPerturbBnd M M ε ε c)
        (idxDotErrBndU (idxDotMagBnd M M c) M r
          + realDotPerturbBnd (idxDotMagBnd M M c) M (idxDotErrBndU M M c + realDotPerturbBnd M M ε ε c) ε r)
        r)
    hX hAX hAAX
  -- unfold the branch on both sides and match
  simp only [nsIter, nsIterR, hcond, hcondR, if_true, nsIterThenMag, nsIterThenErr]
  exact hout

/-- Composed magnitude bound for one iteration (`r > c` branch). -/
noncomputable def nsIterElseMag (a b cc : Float) (M : ℝ) (r c : Nat) : ℝ :=
  let MA := idxDotMagBnd M M r
  let MXA := idxDotMagBnd M MA c
  let MXAA := idxDotMagBnd MXA MA c
  lincomb3MagBnd a b cc M MXA MXAA

/-- Composed error bound for one iteration (`r > c` branch). -/
noncomputable def nsIterElseErr (a b cc : Float) (M ε : ℝ) (r c : Nat) : ℝ :=
  let MA := idxDotMagBnd M M r
  let εA := idxDotErrBndU M M r + realDotPerturbBnd M M ε ε r
  let MXA := idxDotMagBnd M MA c
  let εXA := idxDotErrBndU M MA c + realDotPerturbBnd M MA ε εA c
  let MXAA := idxDotMagBnd MXA MA c
  let εXAA := idxDotErrBndU MXA MA c + realDotPerturbBnd MXA MA εXA εA c
  lincomb3EntryErrBndU a b cc M ε MXA εXA MXAA εXAA

/-- **One Newton–Schulz iteration bound (`r > c` branch).** The tall-matrix case: `A = XᵀX` (c×c),
    `XA = X·A`, `XAA = XA·A`, then the combine. Symmetric mirror of `nsIter_MatBnd_le`. -/
theorem nsIter_MatBnd_gt (X : Mat) (XR : MatR) (r c : Nat) (M ε : ℝ) (a b cc : Float)
    (hM : 0 ≤ M) (hc : 0 < c) (hcr : c < r) (hX : MatBnd X XR r c M ε) :
    MatBnd (nsIter X (a, b, cc)) (nsIterR XR (a, b, cc)) r c
      (nsIterElseMag a b cc M r c) (nsIterElseErr a b cc M ε r c) := by
  have hr : 0 < r := lt_trans hc hcr
  have hMA : 0 ≤ idxDotMagBnd M M r := idxDotMagBnd_nonneg M M hM hM r
  have hMXA : 0 ≤ idxDotMagBnd M (idxDotMagBnd M M r) c := idxDotMagBnd_nonneg M _ hM hMA c
  have hncond : ¬ (X.size ≤ (X[0]!).size) := by rw [hX.sizeX, hX.rowX 0 hr]; omega
  have hncondR : ¬ (XR.size ≤ (XR[0]!).size) := by rw [hX.sizeXR, hX.rowXR 0 hr]; omega
  -- step 1: transpose (c×r)
  have htX : MatBnd (transpose X) (transposeR XR) c r M ε := transpose_MatBnd X XR r c M ε hr hX
  -- step 2: A = Xᵀ·X (c×c), inner dim r
  have hA := matmul_MatBnd (transpose X) X (transposeR XR) XR c r c M ε M ε hM hM hr htX hX
  -- step 3: XA = X·A (r×c), inner dim c
  have hXA := matmul_MatBnd X (matmul (transpose X) X) XR (matmulR (transposeR XR) XR) r c c
    M ε (idxDotMagBnd M M r) (idxDotErrBndU M M r + realDotPerturbBnd M M ε ε r) hM hMA hc hX hA
  -- step 4: XAA = XA·A (r×c), inner dim c
  have hXAA := matmul_MatBnd (matmul X (matmul (transpose X) X)) (matmul (transpose X) X)
    (matmulR XR (matmulR (transposeR XR) XR)) (matmulR (transposeR XR) XR) r c c
    (idxDotMagBnd M (idxDotMagBnd M M r) c)
    (idxDotErrBndU M (idxDotMagBnd M M r) c
      + realDotPerturbBnd M (idxDotMagBnd M M r) ε (idxDotErrBndU M M r + realDotPerturbBnd M M ε ε r) c)
    (idxDotMagBnd M M r) (idxDotErrBndU M M r + realDotPerturbBnd M M ε ε r)
    hMXA hMA hc hXA hA
  -- step 5: lincomb3 a·X + b·XA + c·XAA (r×c)
  have hout := lincomb3_MatBnd a X b (matmul X (matmul (transpose X) X)) cc
    (matmul (matmul X (matmul (transpose X) X)) (matmul (transpose X) X))
    XR (matmulR XR (matmulR (transposeR XR) XR))
    (matmulR (matmulR XR (matmulR (transposeR XR) XR)) (matmulR (transposeR XR) XR)) r c
    M ε
    (idxDotMagBnd M (idxDotMagBnd M M r) c)
    (idxDotErrBndU M (idxDotMagBnd M M r) c
      + realDotPerturbBnd M (idxDotMagBnd M M r) ε (idxDotErrBndU M M r + realDotPerturbBnd M M ε ε r) c)
    (idxDotMagBnd (idxDotMagBnd M (idxDotMagBnd M M r) c) (idxDotMagBnd M M r) c)
    (idxDotErrBndU (idxDotMagBnd M (idxDotMagBnd M M r) c) (idxDotMagBnd M M r) c
      + realDotPerturbBnd (idxDotMagBnd M (idxDotMagBnd M M r) c) (idxDotMagBnd M M r)
        (idxDotErrBndU M (idxDotMagBnd M M r) c
          + realDotPerturbBnd M (idxDotMagBnd M M r) ε (idxDotErrBndU M M r + realDotPerturbBnd M M ε ε r) c)
        (idxDotErrBndU M M r + realDotPerturbBnd M M ε ε r) c)
    hX hXA hXAA
  simp only [nsIter, nsIterR, if_neg hncond, if_neg hncondR, nsIterElseMag, nsIterElseErr]
  exact hout

/-! ### Stage 5: the 5-iteration composition

`nsIter` preserves the `r×c` shape, so `newtonSchulz` runs the SAME branch every iteration. We package
the per-iteration bound map `nsIterBnd` (a `ℝ×ℝ → ℝ×ℝ` carrying the magnitude AND error bounds), prove
it's preserved by one step (`nsIter_MatBnd_le`/`_gt` + magnitude nonnegativity), and apply the
pair-accumulator `foldl_rel_gen` over `muonCoeffs` to bound the whole 5-iteration fold. -/

open Puffer.RL.MuonMatrixRuntime (foldl_rel_gen)

theorem lincomb3MagBnd_nonneg (a b c : Float) (Mx My Mz : ℝ) (hx : 0 ≤ Mx) (hy : 0 ≤ My) (hz : 0 ≤ Mz) :
    0 ≤ lincomb3MagBnd a b c Mx My Mz := by
  unfold lincomb3MagBnd
  refine mul_nonneg one_add_u64_nonneg (add_nonneg (mul_nonneg one_add_u64_nonneg (add_nonneg ?_ ?_)) ?_)
  · exact mul_nonneg one_add_u64_nonneg (mul_nonneg (abs_nonneg _) hx)
  · exact mul_nonneg one_add_u64_nonneg (mul_nonneg (abs_nonneg _) hy)
  · exact mul_nonneg one_add_u64_nonneg (mul_nonneg (abs_nonneg _) hz)

theorem nsIterThenMag_nonneg (a b cc : Float) (M : ℝ) (r c : Nat) (hM : 0 ≤ M) :
    0 ≤ nsIterThenMag a b cc M r c := by
  unfold nsIterThenMag
  have hMA := idxDotMagBnd_nonneg M M hM hM c
  have hMAX := idxDotMagBnd_nonneg _ M hMA hM r
  have hMAAX := idxDotMagBnd_nonneg _ _ hMA hMAX r
  exact lincomb3MagBnd_nonneg a b cc M _ _ hM hMAX hMAAX

theorem nsIterElseMag_nonneg (a b cc : Float) (M : ℝ) (r c : Nat) (hM : 0 ≤ M) :
    0 ≤ nsIterElseMag a b cc M r c := by
  unfold nsIterElseMag
  have hMA := idxDotMagBnd_nonneg M M hM hM r
  have hMXA := idxDotMagBnd_nonneg M _ hM hMA c
  have hMXAA := idxDotMagBnd_nonneg _ _ hMXA hMA c
  exact lincomb3MagBnd_nonneg a b cc M _ _ hM hMXA hMXAA

/-- Per-iteration bound map (magnitude, error), branching on the fixed shape `r ≤ c`. -/
noncomputable def nsIterBnd (r c : Nat) (coef : Float × Float × Float) (Me : ℝ × ℝ) : ℝ × ℝ :=
  if r ≤ c then
    (nsIterThenMag coef.1 coef.2.1 coef.2.2 Me.1 r c, nsIterThenErr coef.1 coef.2.1 coef.2.2 Me.1 Me.2 r c)
  else
    (nsIterElseMag coef.1 coef.2.1 coef.2.2 Me.1 r c, nsIterElseErr coef.1 coef.2.1 coef.2.2 Me.1 Me.2 r c)

/-- **The 5-iteration fold bound.** For a nonempty `r×c` seed within `MatBnd`, the whole
    `muonCoeffs.foldl nsIter seed` is within the folded `MatBnd` of the ℝ mirror
    `muonCoeffs.toList.foldl nsIterR seedR`, with bounds folded by `nsIterBnd`. Via `foldl_rel_gen`. -/
theorem nsFold_MatBnd (seed : Mat) (seedR : MatR) (r c : Nat) (M0 ε0 : ℝ)
    (hr : 0 < r) (hc : 0 < c) (hM0 : 0 ≤ M0) (hseed : MatBnd seed seedR r c M0 ε0) :
    0 ≤ (Puffer.FloatR.Muon.muonCoeffs.toList.foldl (fun e a => nsIterBnd r c a e) (M0, ε0)).1
    ∧ MatBnd (Puffer.FloatR.Muon.muonCoeffs.foldl nsIter seed)
        (Puffer.FloatR.Muon.muonCoeffs.toList.foldl nsIterR seedR) r c
        (Puffer.FloatR.Muon.muonCoeffs.toList.foldl (fun e a => nsIterBnd r c a e) (M0, ε0)).1
        (Puffer.FloatR.Muon.muonCoeffs.toList.foldl (fun e a => nsIterBnd r c a e) (M0, ε0)).2 := by
  have hstep : ∀ (b : Mat) (c' : MatR) (d : ℝ × ℝ) (a : Float × Float × Float),
      (0 ≤ d.1 ∧ MatBnd b c' r c d.1 d.2) →
      (0 ≤ (nsIterBnd r c a d).1
        ∧ MatBnd (nsIter b a) (nsIterR c' a) r c (nsIterBnd r c a d).1 (nsIterBnd r c a d).2) := by
    rintro b c' ⟨M, ε⟩ ⟨a1, a2, a3⟩ ⟨hd, hMB⟩
    by_cases hrc : r ≤ c
    · simp only [nsIterBnd, if_pos hrc]
      exact ⟨nsIterThenMag_nonneg a1 a2 a3 M r c hd,
        nsIter_MatBnd_le b c' r c M ε a1 a2 a3 hd hr hrc hMB⟩
    · simp only [nsIterBnd, if_neg hrc]
      exact ⟨nsIterElseMag_nonneg a1 a2 a3 M r c hd,
        nsIter_MatBnd_gt b c' r c M ε a1 a2 a3 hd hc (not_le.mp hrc) hMB⟩
  have key := foldl_rel_gen
    (fun (b : Mat) (c' : MatR) (d : ℝ × ℝ) => 0 ≤ d.1 ∧ MatBnd b c' r c d.1 d.2)
    nsIter nsIterR (nsIterBnd r c) hstep Puffer.FloatR.Muon.muonCoeffs.toList seed seedR (M0, ε0)
    ⟨hM0, hseed⟩
  rw [← Array.foldl_toList]
  exact key

/-! ### The closed end-to-end `newtonSchulz` bound

Combine the frobNorm-normalized seed (`scalarMul_MatBnd`) with `nsFold_MatBnd` and
`newtonSchulz_eq_foldl`. The ℝ mirror is `newtonSchulz` run in exact real arithmetic, seeded by the
(real value of the) rounded normalization scale `toReal (1/(‖X₀‖_F + eps))` — so the theorem bounds the
runnable `newtonSchulz X0 eps` against a faithful real-arithmetic computation on the same data. -/

/-- **Closed `newtonSchulz` error bound.** For a nonempty `r×c` input `X0` within `MatBnd` of its
    mirror `X0R`, every entry of the runnable `newtonSchulz X0 eps` is within the fully-composed
    `MatBnd` (5 iterations of magnitude/error growth, seeded by the normalization scale) of the
    corresponding entry of the real mirror. This closes the Muon-matrix trifecta on `newtonSchulz`. -/
theorem newtonSchulz_MatBnd (X0 : Mat) (X0R : MatR) (eps : Float) (r c : Nat) (M ε : ℝ)
    (hr : 0 < r) (hc : 0 < c) (hM : 0 ≤ M) (hX0 : MatBnd X0 X0R r c M ε) :
    MatBnd (Puffer.FloatR.Muon.newtonSchulz X0 eps)
      (Puffer.FloatR.Muon.muonCoeffs.toList.foldl nsIterR
        (scalarMulR (toReal (1.0 / (Puffer.FloatR.Muon.frobNorm X0 + eps))) X0R)) r c
      (Puffer.FloatR.Muon.muonCoeffs.toList.foldl (fun e a => nsIterBnd r c a e)
        ((1 + u64) * (|toReal (1.0 / (Puffer.FloatR.Muon.frobNorm X0 + eps))| * M),
         u64 * (|toReal (1.0 / (Puffer.FloatR.Muon.frobNorm X0 + eps))| * M)
           + |toReal (1.0 / (Puffer.FloatR.Muon.frobNorm X0 + eps))| * ε)).1
      (Puffer.FloatR.Muon.muonCoeffs.toList.foldl (fun e a => nsIterBnd r c a e)
        ((1 + u64) * (|toReal (1.0 / (Puffer.FloatR.Muon.frobNorm X0 + eps))| * M),
         u64 * (|toReal (1.0 / (Puffer.FloatR.Muon.frobNorm X0 + eps))| * M)
           + |toReal (1.0 / (Puffer.FloatR.Muon.frobNorm X0 + eps))| * ε)).2 := by
  rw [Puffer.RL.MuonMatrixRuntime.newtonSchulz_eq_foldl]
  exact (nsFold_MatBnd
    (Puffer.FloatR.Muon.scalarMul (1.0 / (Puffer.FloatR.Muon.frobNorm X0 + eps)) X0)
    (scalarMulR (toReal (1.0 / (Puffer.FloatR.Muon.frobNorm X0 + eps))) X0R) r c _ _ hr hc
    (mul_nonneg one_add_u64_nonneg (mul_nonneg (abs_nonneg _) hM))
    (scalarMul_MatBnd (1.0 / (Puffer.FloatR.Muon.frobNorm X0 + eps)) X0 X0R r c M ε hM hX0)).2

end Puffer.RL.NewtonSchulzError
