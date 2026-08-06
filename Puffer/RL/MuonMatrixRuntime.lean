/-
Error bounds for Muon's matrix/vector arithmetic — extending the trifecta from the SCALAR
Newton–Schulz map (`nsScalarF_error`) toward the full matrix `stepMat`.

`Puffer/Float/Muon.lean` builds the Muon update from a few pointwise arithmetic kernels —
`matLin` (`a·X + b·Y`), `lincomb3` (`a·X + b·Y + c·Z`, the Newton–Schulz combine), and the
1D `stepVec` (Nesterov momentum + decoupled weight decay for biases). Here we bound each of
those per-entry/per-element against its exact-ℝ value, by composing the `mulApprox_error` /
`addApprox_error` / `subApprox_error` propagation helpers. These are the linear-combination
and vector-update building blocks; `stepVec` is a COMPLETE Muon path (the bias update), so this
closes the trifecta on it. We also bound the FULL `matmul` entry: `matmul_entry_error` shows each
in-range `(matmul A B)[i][j]` is within a proven error bound of its exact real value (`matmul`
reduces to a nested `List.map`/`List.foldl`, whose per-entry `getElem` is the accumulator fold
`idxDotF`). The remaining matrix pieces — `frobNorm` (`sqrt` of a sum-of-squares fold) and the
5-iteration `newtonSchulz` composition — are scoped as follow-up.

NOTE: `matmul` seeds its accumulator with the `OfScientific` literal `0.0`, which is not defeq to
`(0 : Float)` (Float has no `DecidableEq`), so we match it exactly and lean on the trusted axiom
`toReal_zeroLit : toReal (0.0 : Float) = 0` (the `0.0`-literal analog of `toReal_zero`).
-/
import Mathlib
import Puffer.Float.Basic
import Puffer.Float.Muon

namespace Puffer.RL.MuonMatrixRuntime

open Puffer.FloatR
open Puffer.FloatR.Muon (Mat matmul)

/-! ### `matLin` entry: `a·x + b·y` (one entry of `a·X + b·Y`) -/

/-- ℝ error bound for one `matLin` entry `a·x + b·y`, given input errors `εx,εy` on `x,y`
    (coefficients `a,b` are exact Floats). -/
noncomputable def matLinEntryErrBnd (a x b y : Float) (εx εy : ℝ) : ℝ :=
  u64 * |toReal (a * x) + toReal (b * y)|
  + (u64 * |toReal a * toReal x| + |toReal a| * εx)
  + (u64 * |toReal b * toReal y| + |toReal b| * εy)

/-- **`matLin` entry error.** `a·x + b·y` (Float) is within `matLinEntryErrBnd` of
    `toReal a · xR + toReal b · yR`. -/
theorem matLinEntry_error (a x b y : Float) (xR yR εx εy : ℝ)
    (hx : |toReal x - xR| ≤ εx) (hy : |toReal y - yR| ≤ εy) :
    |toReal (a * x + b * y) - (toReal a * xR + toReal b * yR)| ≤ matLinEntryErrBnd a x b y εx εy := by
  have hax := mulApprox_error a x (toReal a) xR 0 εx (by simp) hx
  have hby := mulApprox_error b y (toReal b) yR 0 εy (by simp) hy
  have h := addApprox_error (a * x) (b * y) (toReal a * xR) (toReal b * yR) _ _ hax hby
  simpa [matLinEntryErrBnd, mul_zero, add_zero] using h

/-! ### `lincomb3` entry: `a·x + b·y + c·z` (the Newton–Schulz combine, per entry) -/

/-- ℝ error bound for one `lincomb3` entry `(a·x + b·y) + c·z`. -/
noncomputable def lincomb3EntryErrBnd (a x b y c z : Float) (εx εy εz : ℝ) : ℝ :=
  u64 * |toReal (a * x + b * y) + toReal (c * z)|
  + matLinEntryErrBnd a x b y εx εy
  + (u64 * |toReal c * toReal z| + |toReal c| * εz)

/-- **`lincomb3` entry error.** `a·x + b·y + c·z` (Float, left-assoc) is within
    `lincomb3EntryErrBnd` of `toReal a · xR + toReal b · yR + toReal c · zR`. -/
theorem lincomb3Entry_error (a x b y c z : Float) (xR yR zR εx εy εz : ℝ)
    (hx : |toReal x - xR| ≤ εx) (hy : |toReal y - yR| ≤ εy) (hz : |toReal z - zR| ≤ εz) :
    |toReal (a * x + b * y + c * z) - (toReal a * xR + toReal b * yR + toReal c * zR)|
      ≤ lincomb3EntryErrBnd a x b y c z εx εy εz := by
  have hxy := matLinEntry_error a x b y xR yR εx εy hx hy
  have hcz := mulApprox_error c z (toReal c) zR 0 εz (by simp) hz
  have h := addApprox_error (a * x + b * y) (c * z)
    (toReal a * xR + toReal b * yR) (toReal c * zR) _ _ hxy hcz
  simpa [lincomb3EntryErrBnd, mul_zero, add_zero] using h

/-! ### `matmul` accumulator: a LEFT-fold dot (the core of matrix multiply)

Muon's `matmul` computes each entry with a mutable accumulator `s := s + A[i][l]·B[l][j]`,
i.e. a LEFT-associated dot `((0 + a₀b₀) + a₁b₁) + …` (distinct from `dotF`'s right-nested
fold). We bound this fold against its exact real value. This is the per-entry error of the
matrix multiply; connecting it to the imperative `matmul` loop (a `forIn`↔`foldl` rewrite)
and composing through `newtonSchulz` remain follow-up. -/

/-- Left-fold dot with a running accumulator `acc`. -/
def dotLGo (acc : Float) : List Float → List Float → Float
  | [], _ => acc
  | _, [] => acc
  | a :: as, b :: bs => dotLGo (acc + a * b) as bs

/-- Exact-ℝ left-fold dot. -/
noncomputable def dotLGoR (acc : ℝ) : List Float → List Float → ℝ
  | [], _ => acc
  | _, [] => acc
  | a :: as, b :: bs => dotLGoR (acc + toReal a * toReal b) as bs

/-- Certified error bound for the left-fold dot, threading the accumulator error `εacc`. -/
noncomputable def dotLGoErrBnd : Float → List Float → List Float → ℝ → ℝ
  | _, [], _, εacc => εacc
  | _, _, [], εacc => εacc
  | acc, a :: as, b :: bs, εacc =>
      dotLGoErrBnd (acc + a * b) as bs (u64 * |toReal acc + toReal (a * b)| + εacc + u64 * |toReal a * toReal b|)

/-- **Left-fold dot error (accumulator form).** -/
theorem dotLGo_error (as : List Float) : ∀ (acc : Float) (bs : List Float) (accR εacc : ℝ),
    |toReal acc - accR| ≤ εacc →
    |toReal (dotLGo acc as bs) - dotLGoR accR as bs| ≤ dotLGoErrBnd acc as bs εacc := by
  induction as with
  | nil => intro acc bs accR εacc hacc; simpa [dotLGo, dotLGoR, dotLGoErrBnd] using hacc
  | cons a as ih =>
      intro acc bs accR εacc hacc
      cases bs with
      | nil => simpa [dotLGo, dotLGoR, dotLGoErrBnd] using hacc
      | cons b bs =>
          simp only [dotLGo, dotLGoR, dotLGoErrBnd]
          have hstep : |toReal (acc + a * b) - (accR + toReal a * toReal b)|
              ≤ u64 * |toReal acc + toReal (a * b)| + εacc + u64 * |toReal a * toReal b| := by
            have h := addApprox_error acc (a * b) accR (toReal a * toReal b) εacc
              (u64 * |toReal a * toReal b|) hacc (by simpa using mul_error a b)
            simpa using h
          exact ih (acc + a * b) bs (accR + toReal a * toReal b) _ hstep

/-- The matmul-entry left-fold dot `Σₗ aₗ·bₗ` (accumulate from 0). -/
def dotL (as bs : List Float) : Float := dotLGo 0 as bs
noncomputable def dotLR (as bs : List Float) : ℝ := dotLGoR 0 as bs
noncomputable def dotLErrBnd (as bs : List Float) : ℝ := dotLGoErrBnd 0 as bs 0

/-- **`matmul` entry error.** The left-fold dot `dotL as bs` deviates from the exact real
    dot `dotLR as bs` by at most `dotLErrBnd as bs`. -/
theorem dotL_error (as bs : List Float) : |toReal (dotL as bs) - dotLR as bs| ≤ dotLErrBnd as bs :=
  dotLGo_error as 0 bs 0 0 (by simp)

/-! ### `frobNorm` of a vector: `√(Σ xᵢ²)` (the L2/Frobenius norm)

A sum of squares `Σ xᵢ²` IS `dotL xs xs` (the vector dotted with itself), so `dotL_error`
bounds it; `frobNormL_error` composes that with `sqrtApprox_error` for the norm `√(dotL xs xs)`.
(Muon's `frobNorm` over a full matrix is the nested-fold extension — scoped follow-up.) -/

/-- `toReal (x·x) ≥ 0`: a squared Float is nonnegative (the (1+δ) factor stays positive). -/
theorem toReal_mulSelf_nonneg (x : Float) : 0 ≤ toReal (x * x) := by
  obtain ⟨δ, hδ, he⟩ := mul_model x x
  rw [he]
  exact mul_nonneg (mul_self_nonneg _) (by linarith [u64_lt_one, (abs_le.mp hδ).1])

/-- The `Float` sum of squares `dotLGo` stays nonnegative. -/
theorem toReal_dotLGo_sq_nonneg (xs : List Float) :
    ∀ acc : Float, 0 ≤ toReal acc → 0 ≤ toReal (dotLGo acc xs xs) := by
  induction xs with
  | nil => intro acc h; simpa [dotLGo] using h
  | cons x xs ih =>
      intro acc h
      simp only [dotLGo]
      refine ih (acc + x * x) ?_
      obtain ⟨δ, hδ, he⟩ := add_model acc (x * x)
      rw [he]
      exact mul_nonneg (add_nonneg h (toReal_mulSelf_nonneg x))
        (by linarith [u64_lt_one, (abs_le.mp hδ).1])

/-- The exact-ℝ sum of squares stays nonnegative. -/
theorem dotLGoR_sq_nonneg (xs : List Float) : ∀ acc : ℝ, 0 ≤ acc → 0 ≤ dotLGoR acc xs xs := by
  induction xs with
  | nil => intro acc h; simpa [dotLGoR] using h
  | cons x xs ih => intro acc h; simp only [dotLGoR]; exact ih _ (add_nonneg h (mul_self_nonneg _))

/-- **Vector Frobenius/L2-norm error.** `√(dotL xs xs)` (Float) is within
    `u64·√(toReal(dotL xs xs)) + √(dotLErrBnd xs xs)` of the exact real norm `√(dotLR xs xs)`. -/
theorem frobNormL_error (xs : List Float) :
    |toReal (Float.sqrt (dotL xs xs)) - Real.sqrt (dotLR xs xs)|
      ≤ u64 * Real.sqrt (toReal (dotL xs xs)) + Real.sqrt (dotLErrBnd xs xs) :=
  sqrtApprox_error (dotL xs xs) (dotLR xs xs) (dotLErrBnd xs xs) (dotL_error xs xs)
    (toReal_dotLGo_sq_nonneg xs 0 (by simp)) (dotLGoR_sq_nonneg xs 0 le_rfl)

/-! ### `frobNorm` of a full matrix: `√(Σᵢⱼ Aᵢⱼ²)`

Muon's `frobNorm A = √(A.foldl (fun s row => s + row.foldl (·+·²) 0.0) 0.0)`: an OUTER row
fold whose per-row contribution is the INNER sum-of-squares `dotLGo 0.0 row row`. We give the
Float sum `sumSqMat` matching that nested fold exactly, its exact-ℝ mirror `sumSqMatR`, and an
error thread `sumSqMatErrBnd`; `sumSqMat_eq` proves the fold *equals* `sumSqMat`, and
`frobNormMat_error` composes `sumSqMat_error` with `sqrtApprox_error`.  Each inner fold seeds the
`0.0` literal, so we reuse `toReal_zeroLit` (as in the `matmul` accumulator). -/

/-- `Σ x²` written as the code's left fold equals the self-dot `dotLGo` (both peel one entry,
    adding `x·x`). -/
theorem foldl_sq_eq_dotLGo (L : List Float) :
    ∀ acc : Float, List.foldl (fun s x => s + x * x) acc L = dotLGo acc L L := by
  induction L with
  | nil => intro acc; rfl
  | cons x xs ih => intro acc; simp only [List.foldl_cons, dotLGo]; exact ih (acc + x * x)

/-- Float sum of squared entries, matching `frobNorm`'s outer fold: each row contributes its
    own sum of squares `dotLGo 0.0 row row`, added into the running accumulator. -/
def sumSqMatGo (acc : Float) : List (Array Float) → Float
  | [] => acc
  | row :: rows => sumSqMatGo (acc + dotLGo 0.0 row.toList row.toList) rows

/-- Exact-ℝ mirror of `sumSqMatGo`. -/
noncomputable def sumSqMatGoR (acc : ℝ) : List (Array Float) → ℝ
  | [] => acc
  | row :: rows => sumSqMatGoR (acc + dotLGoR 0 row.toList row.toList) rows

/-- Certified error bound for `sumSqMatGo`, threading the accumulator error and each row's
    sum-of-squares error `dotLGoErrBnd 0.0 row row 0`. -/
noncomputable def sumSqMatGoErrBnd : Float → List (Array Float) → ℝ → ℝ
  | _, [], εacc => εacc
  | acc, row :: rows, εacc =>
      sumSqMatGoErrBnd (acc + dotLGo 0.0 row.toList row.toList) rows
        (u64 * |toReal acc + toReal (dotLGo 0.0 row.toList row.toList)| + εacc
          + dotLGoErrBnd 0.0 row.toList row.toList 0)

/-- **Matrix sum-of-squares error (accumulator form).** -/
theorem sumSqMatGo_error (rows : List (Array Float)) :
    ∀ (acc : Float) (accR εacc : ℝ), |toReal acc - accR| ≤ εacc →
    |toReal (sumSqMatGo acc rows) - sumSqMatGoR accR rows| ≤ sumSqMatGoErrBnd acc rows εacc := by
  induction rows with
  | nil => intro acc accR εacc h; simpa [sumSqMatGo, sumSqMatGoR, sumSqMatGoErrBnd] using h
  | cons row rows ih =>
      intro acc accR εacc hacc
      simp only [sumSqMatGo, sumSqMatGoR, sumSqMatGoErrBnd]
      have hrow : |toReal (dotLGo 0.0 row.toList row.toList) - dotLGoR 0 row.toList row.toList|
          ≤ dotLGoErrBnd 0.0 row.toList row.toList 0 :=
        dotLGo_error row.toList 0.0 row.toList 0 0 (by rw [toReal_zeroLit]; simp)
      have hstep : |toReal (acc + dotLGo 0.0 row.toList row.toList)
            - (accR + dotLGoR 0 row.toList row.toList)|
          ≤ u64 * |toReal acc + toReal (dotLGo 0.0 row.toList row.toList)| + εacc
            + dotLGoErrBnd 0.0 row.toList row.toList 0 := by
        have h := addApprox_error acc (dotLGo 0.0 row.toList row.toList) accR
          (dotLGoR 0 row.toList row.toList) εacc (dotLGoErrBnd 0.0 row.toList row.toList 0)
          hacc hrow
        simpa using h
      exact ih (acc + dotLGo 0.0 row.toList row.toList) (accR + dotLGoR 0 row.toList row.toList)
        _ hstep

/-- The Float sum of squared entries (seeded `0.0`, matching the code). -/
def sumSqMat (A : Mat) : Float := sumSqMatGo 0.0 A.toList
/-- Exact-ℝ sum of squared entries. -/
noncomputable def sumSqMatR (A : Mat) : ℝ := sumSqMatGoR 0 A.toList
/-- Certified error bound for `sumSqMat`. -/
noncomputable def sumSqMatErrBnd (A : Mat) : ℝ := sumSqMatGoErrBnd 0.0 A.toList 0

/-- The nested fold inside `frobNorm` equals `sumSqMat` (peel the `Array.foldl`s to `List.foldl`
    and rewrite each row's inner fold via `foldl_sq_eq_dotLGo`). -/
theorem sumSqMat_eq (A : Mat) :
    (A.foldl (fun s row => s + row.foldl (fun s2 x => s2 + x * x) 0.0) 0.0) = sumSqMat A := by
  suffices h : ∀ (rows : List (Array Float)) (acc : Float),
      List.foldl (fun s row => s + Array.foldl (fun s2 x => s2 + x * x) 0.0 row) acc rows
        = sumSqMatGo acc rows by
    rw [sumSqMat, ← Array.foldl_toList]; exact h A.toList 0.0
  intro rows
  induction rows with
  | nil => intro acc; rfl
  | cons row rows ih =>
      intro acc
      simp only [List.foldl_cons, sumSqMatGo]
      rw [← Array.foldl_toList, foldl_sq_eq_dotLGo]
      exact ih (acc + dotLGo 0.0 row.toList row.toList)

/-- The Float matrix sum-of-squares is nonnegative. -/
theorem toReal_sumSqMatGo_nonneg (rows : List (Array Float)) :
    ∀ acc : Float, 0 ≤ toReal acc → 0 ≤ toReal (sumSqMatGo acc rows) := by
  induction rows with
  | nil => intro acc h; simpa [sumSqMatGo] using h
  | cons row rows ih =>
      intro acc h
      simp only [sumSqMatGo]
      refine ih (acc + dotLGo 0.0 row.toList row.toList) ?_
      obtain ⟨δ, hδ, he⟩ := add_model acc (dotLGo 0.0 row.toList row.toList)
      rw [he]
      exact mul_nonneg
        (add_nonneg h (toReal_dotLGo_sq_nonneg row.toList 0.0 (le_of_eq toReal_zeroLit.symm)))
        (by linarith [u64_lt_one, (abs_le.mp hδ).1])

/-- The exact-ℝ matrix sum-of-squares is nonnegative. -/
theorem sumSqMatGoR_nonneg (rows : List (Array Float)) :
    ∀ acc : ℝ, 0 ≤ acc → 0 ≤ sumSqMatGoR acc rows := by
  induction rows with
  | nil => intro acc h; simpa [sumSqMatGoR] using h
  | cons row rows ih =>
      intro acc h; simp only [sumSqMatGoR]
      exact ih _ (add_nonneg h (dotLGoR_sq_nonneg row.toList 0 le_rfl))

theorem sumSqMat_nonneg (A : Mat) : 0 ≤ toReal (sumSqMat A) :=
  toReal_sumSqMatGo_nonneg A.toList 0.0 (le_of_eq toReal_zeroLit.symm)
theorem sumSqMatR_nonneg (A : Mat) : 0 ≤ sumSqMatR A :=
  sumSqMatGoR_nonneg A.toList 0 le_rfl

/-- **Matrix `frobNorm` error.** `frobNorm A` (Float) is within
    `u64·√(toReal (sumSqMat A)) + √(sumSqMatErrBnd A)` of the exact real Frobenius norm
    `√(sumSqMatR A)`. -/
theorem frobNormMat_error (A : Mat) :
    |toReal (Puffer.FloatR.Muon.frobNorm A) - Real.sqrt (sumSqMatR A)|
      ≤ u64 * Real.sqrt (toReal (sumSqMat A)) + Real.sqrt (sumSqMatErrBnd A) := by
  have hsum : Puffer.FloatR.Muon.frobNorm A = Float.sqrt (sumSqMat A) := by
    rw [Puffer.FloatR.Muon.frobNorm, sumSqMat_eq]
  rw [hsum]
  exact sqrtApprox_error (sumSqMat A) (sumSqMatR A) (sumSqMatErrBnd A)
    (sumSqMatGo_error A.toList 0.0 0 0 (by rw [toReal_zeroLit]; simp))
    (sumSqMat_nonneg A) (sumSqMatR_nonneg A)

/-! ### Connecting the bound to `matmul`'s imperative accumulator loop

`matmul`'s inner loop is `let mut s := 0.0; for l in [0:k] do s := s + A[i]![l]! * B[l]![j]!`.
We give an INDEX-based left fold `idxDotF f g k` (with `f l = A[i]![l]!`, `g l = B[l]![j]!`),
prove its error bound (`idxDot_error`), prove the imperative loop equals it (`accLoop_eq`, by
reducing the `Std.Range` `for` to a `List.foldl` over `List.range`), and combine the two
(`matmulAcc_error`). The loop here is seeded with `0.0` — EXACTLY `matmul`'s accumulator init —
so this genuinely bounds `matmul`'s inner loop (with `f := A[i]![·]!`, `g := B[·]![j]!`). This
lifts to the full entry via `matmul_getElem`/`matmul_entry_error` below. -/

/-- Index-based left-fold `((0 + f0·g0) + f1·g1) + …` — the value `matmul`'s accumulator holds. -/
def idxDotF (f g : Nat → Float) : Nat → Float
  | 0 => 0.0
  | k + 1 => idxDotF f g k + f k * g k

/-- Its exact-ℝ value. -/
noncomputable def idxDotR (f g : Nat → Float) : Nat → ℝ
  | 0 => 0
  | k + 1 => idxDotR f g k + toReal (f k) * toReal (g k)

/-- Certified error bound (accumulate one add-rounding + one mul-rounding per index). -/
noncomputable def idxDotErrBnd (f g : Nat → Float) : Nat → ℝ
  | 0 => 0
  | k + 1 => u64 * |toReal (idxDotF f g k) + toReal (f k * g k)|
      + idxDotErrBnd f g k + u64 * |toReal (f k) * toReal (g k)|

theorem idxDot_error (f g : Nat → Float) (k : Nat) :
    |toReal (idxDotF f g k) - idxDotR f g k| ≤ idxDotErrBnd f g k := by
  induction k with
  | zero => simp [idxDotF, idxDotR, idxDotErrBnd]
  | succ k ih =>
      simp only [idxDotF, idxDotR, idxDotErrBnd]
      have h := addApprox_error (idxDotF f g k) (f k * g k) (idxDotR f g k)
        (toReal (f k) * toReal (g k)) (idxDotErrBnd f g k) (u64 * |toReal (f k) * toReal (g k)|)
        ih (by simpa using mul_error (f k) (g k))
      simpa using h

/-- `idxDotF` is the `List.foldl` over `List.range k` (built up index by index). -/
theorem idxDotF_eq_foldl (f g : Nat → Float) (k : Nat) :
    idxDotF f g k = (List.range k).foldl (fun s l => s + f l * g l) 0.0 := by
  induction k with
  | zero => simp [idxDotF]
  | succ k ih => rw [idxDotF, ih, List.range_succ, List.foldl_append]; rfl

/-- **The imperative accumulator loop equals `idxDotF`.** (`matmul`'s inner loop, with
    `f l = A[i]![l]!`, `g l = B[l]![j]!`.) -/
theorem accLoop_eq (f g : Nat → Float) (k : Nat) :
    (Id.run do
      let mut s : Float := 0.0
      for l in [0:k] do
        s := s + f l * g l
      return s) = idxDotF f g k := by
  rw [idxDotF_eq_foldl]
  -- the `Std.Range` `for` reduces to `List.foldl` over `List.range' 0 k`, leaving `pure x = x`
  simp [Id.run, List.range_eq_range']
  rfl

/-- **`matmul` accumulator error.** `matmul`'s per-entry accumulator loop is within
    `idxDotErrBnd` of the exact real sum `Σₗ f l · g l`. -/
theorem matmulAcc_error (f g : Nat → Float) (k : Nat) :
    |toReal (Id.run do
      let mut s : Float := 0.0
      for l in [0:k] do
        s := s + f l * g l
      return s) - idxDotR f g k| ≤ idxDotErrBnd f g k := by
  rw [accLoop_eq]; exact idxDot_error f g k

/-- **Full `matmul` entry = the accumulator fold.** For an in-range entry, `(matmul A B)[i][j]`
    equals `idxDotF` over row `A[i]` and column `B[·][j]` — lifting the accumulator through the
    outer `Array.push` assembly loops (`matmul` reduces to a nested `List.map`/`List.foldl`, whose
    per-entry `getElem` is exactly the accumulator fold). Pure structural equality (no float axioms). -/
theorem matmul_getElem (A B : Mat) (i j : Nat)
    (hi : i < A.size) (hj : j < (if B = #[] then 0 else B[0]!.size)) :
    ((matmul A B)[i]!)[j]! = idxDotF (fun l => (A[i]!)[l]!) (fun l => (B[l]!)[j]!)
      (if A = #[] then 0 else (A[0]!).size) := by
  have hM : matmul A B = (List.map (fun i' => (List.map (fun j' =>
      idxDotF (fun l => (A[i']!)[l]!) (fun l => (B[l]!)[j']!) (if A = #[] then 0 else (A[0]!).size))
      (List.range' 0 (if B = #[] then 0 else (B[0]!).size))).toArray)
    (List.range' 0 A.size)).toArray := by
    unfold matmul; simp only [idxDotF_eq_foldl, List.range_eq_range']; simp [Id.run]; rfl
  rw [hM, Array.getElem!_eq_getD]
  simp [Array.getD, hi, List.getElem_range']
  rw [Array.getElem!_eq_getD]
  simp only [Array.getD]
  intro h
  exact absurd hj (not_lt.mpr h)

/-- **Full `matmul` entry error.** Each in-range `(matmul A B)[i][j]` is within `idxDotErrBnd` of
    the exact real dot of row `A[i]` with column `B[·][j]` — the accumulator bound, lifted to the
    runnable matrix output. -/
theorem matmul_entry_error (A B : Mat) (i j : Nat)
    (hi : i < A.size) (hj : j < (if B = #[] then 0 else B[0]!.size)) :
    |toReal (((matmul A B)[i]!)[j]!)
       - idxDotR (fun l => (A[i]!)[l]!) (fun l => (B[l]!)[j]!) (if A = #[] then 0 else (A[0]!).size)|
      ≤ idxDotErrBnd (fun l => (A[i]!)[l]!) (fun l => (B[l]!)[j]!) (if A = #[] then 0 else (A[0]!).size) := by
  rw [matmul_getElem A B i j hi hj]
  exact idxDot_error _ _ _

/-- `matmul` preserves the outer dimension: `(matmul A B).size = A.size`. -/
theorem matmul_size (A B : Mat) : (matmul A B).size = A.size := by
  have hM : matmul A B = (List.map (fun i' => (List.map (fun j' =>
      idxDotF (fun l => (A[i']!)[l]!) (fun l => (B[l]!)[j']!) (if A = #[] then 0 else (A[0]!).size))
      (List.range' 0 (if B = #[] then 0 else (B[0]!).size))).toArray)
    (List.range' 0 A.size)).toArray := by
    unfold matmul; simp only [idxDotF_eq_foldl, List.range_eq_range']; simp [Id.run]; rfl
  rw [hM]; simp [List.size_toArray, List.length_map, List.length_range']

/-- `matmul` row length is the inner dimension `n` (columns of `B`). -/
theorem matmul_rowSize (A B : Mat) (i : Nat) (hi : i < A.size) :
    ((matmul A B)[i]!).size = (if B = #[] then 0 else (B[0]!).size) := by
  have hM : matmul A B = (List.map (fun i' => (List.map (fun j' =>
      idxDotF (fun l => (A[i']!)[l]!) (fun l => (B[l]!)[j']!) (if A = #[] then 0 else (A[0]!).size))
      (List.range' 0 (if B = #[] then 0 else (B[0]!).size))).toArray)
    (List.range' 0 A.size)).toArray := by
    unfold matmul; simp only [idxDotF_eq_foldl, List.range_eq_range']; simp [Id.run]; rfl
  rw [hM, Array.getElem!_eq_getD]
  simp [Array.getD, hi, List.getElem_range', List.size_toArray, List.length_map, List.length_range']

/-! ### `transpose` dimensions and entries

`transpose A` is a nested `Array.push` loop (`for j; for i; row.push A[i][j]`). It reduces to a nested
`List.map`/`toArray` form (push loops from `#[]` become `(List.map … range').toArray`), giving clean
size/entry lemmas: `(transpose A)[j][i] = A[i][j]`, dimensions swapped to `n×m`. -/

theorem transpose_hM (A : Mat) : Puffer.FloatR.Muon.transpose A =
    (List.map (fun j' => (List.map (fun i' => (A[i']!)[j']!)
      (List.range' 0 A.size)).toArray)
      (List.range' 0 (if A = #[] then 0 else (A[0]!).size))).toArray := by
  unfold Puffer.FloatR.Muon.transpose; simp [Id.run]; rfl

/-- `transpose` outer dimension is the inner dimension `n` of `A`. -/
theorem transpose_size (A : Mat) :
    (Puffer.FloatR.Muon.transpose A).size = (if A = #[] then 0 else (A[0]!).size) := by
  rw [transpose_hM]; simp [List.size_toArray, List.length_map, List.length_range']

/-- The `j`-th row of `transpose A` collects column `j` of `A`. -/
theorem transpose_row (A : Mat) (j : Nat) (hj : j < (if A = #[] then 0 else (A[0]!).size)) :
    (Puffer.FloatR.Muon.transpose A)[j]! =
      (List.map (fun i' => (A[i']!)[j]!) (List.range' 0 A.size)).toArray := by
  rw [transpose_hM, Array.getElem!_eq_getD]
  simp [Array.getD, hj, List.getElem_range', List.size_toArray, List.length_map, List.length_range',
    List.getElem_toArray, List.getElem_map]

/-- `transpose` row length is the outer dimension `m = A.size`. -/
theorem transpose_rowSize (A : Mat) (j : Nat) (hj : j < (if A = #[] then 0 else (A[0]!).size)) :
    ((Puffer.FloatR.Muon.transpose A)[j]!).size = A.size := by
  rw [transpose_row A j hj]; simp [List.size_toArray, List.length_map, List.length_range']

/-- **`transpose` entry.** `(transpose A)[j][i] = A[i][j]` (pure reindexing, no arithmetic). -/
theorem transpose_getElem (A : Mat) (i j : Nat)
    (hj : j < (if A = #[] then 0 else (A[0]!).size)) (hi : i < A.size) :
    ((Puffer.FloatR.Muon.transpose A)[j]!)[i]! = (A[i]!)[j]! := by
  rw [transpose_row A j hj, Array.getElem!_eq_getD]
  simp [Array.getD, hi, List.getElem_toArray, List.getElem_map, List.getElem_range',
    List.size_toArray, List.length_map, List.length_range']

/-! ### `stepVec` element: the complete 1D Muon update (Nesterov + weight decay)

`stepVec` computes, per bias index, `newMom = μ·mom + grad`, `update = grad + μ·newMom`,
`out = b·(1 − lr·wd) + lr·update`. We bound `out` against its exact-ℝ value. -/

/-- Exact-ℝ value of one `stepVec` output element. -/
noncomputable def stepVecElemR (b grad mom lr wd mu : Float) : ℝ :=
  toReal b * (1 - toReal lr * toReal wd)
    + toReal lr * (toReal grad + toReal mu * (toReal mu * toReal mom + toReal grad))

/-- ℝ error bound for one `stepVec` output element (built by composing the momentum,
    Nesterov, weight-decay-scale, and final `matLin` steps). -/
noncomputable def stepVecElemErrBnd (b grad mom lr wd mu : Float) : ℝ :=
  -- newMom = μ·mom + grad
  let εnm := u64 * |toReal (mu * mom) + toReal grad| + u64 * |toReal mu * toReal mom|
  -- update = grad + μ·newMom
  let εmn := u64 * |toReal mu * toReal (mu * mom + grad)| + |toReal mu| * εnm
  let εup := u64 * |toReal grad + toReal (mu * (mu * mom + grad))| + εmn
  -- c = 1 − lr·wd
  let εc := u64 * |toReal 1 - toReal (lr * wd)| + u64 * |toReal lr * toReal wd|
  matLinEntryErrBnd b (1 - lr * wd) lr (grad + mu * (mu * mom + grad)) εc εup

/-- **`stepVec` element error (complete 1D Muon path).** One `stepVec` output element is within
    `stepVecElemErrBnd` of its exact real value `stepVecElemR`. -/
theorem stepVecElem_error (b grad mom lr wd mu : Float) :
    |toReal (b * (1 - lr * wd) + lr * (grad + mu * (mu * mom + grad))) - stepVecElemR b grad mom lr wd mu|
      ≤ stepVecElemErrBnd b grad mom lr wd mu := by
  -- newMom = μ·mom + grad
  have hnm : |toReal (mu * mom + grad) - (toReal mu * toReal mom + toReal grad)|
      ≤ u64 * |toReal (mu * mom) + toReal grad| + u64 * |toReal mu * toReal mom| := by
    have := addApprox_error (mu * mom) grad (toReal mu * toReal mom) (toReal grad)
      (u64 * |toReal mu * toReal mom|) 0 (by simpa using mul_error mu mom) (by simp)
    simpa [add_zero] using this
  -- μ·newMom
  have hmn : |toReal (mu * (mu * mom + grad)) - toReal mu * (toReal mu * toReal mom + toReal grad)|
      ≤ u64 * |toReal mu * toReal (mu * mom + grad)| + |toReal mu| * (u64 * |toReal (mu * mom) + toReal grad| + u64 * |toReal mu * toReal mom|) := by
    have := mulApprox_error mu (mu * mom + grad) (toReal mu) (toReal mu * toReal mom + toReal grad)
      0 _ (by simp) hnm
    simpa [mul_zero, add_zero] using this
  -- update = grad + μ·newMom
  have hup : |toReal (grad + mu * (mu * mom + grad)) - (toReal grad + toReal mu * (toReal mu * toReal mom + toReal grad))|
      ≤ u64 * |toReal grad + toReal (mu * (mu * mom + grad))|
        + (u64 * |toReal mu * toReal (mu * mom + grad)| + |toReal mu| * (u64 * |toReal (mu * mom) + toReal grad| + u64 * |toReal mu * toReal mom|)) := by
    have := addApprox_error grad (mu * (mu * mom + grad)) (toReal grad)
      (toReal mu * (toReal mu * toReal mom + toReal grad)) 0 _ (by simp) hmn
    simpa [add_zero] using this
  -- c = 1 − lr·wd
  have hc : |toReal (1 - lr * wd) - (1 - toReal lr * toReal wd)|
      ≤ u64 * |toReal 1 - toReal (lr * wd)| + u64 * |toReal lr * toReal wd| := by
    have := subApprox_error 1 (lr * wd) 1 (toReal lr * toReal wd)
      0 (u64 * |toReal lr * toReal wd|) (by simp) (by simpa using mul_error lr wd)
    simpa [add_zero] using this
  -- out = b·c + lr·update  (matLin entry)
  have h := matLinEntry_error b (1 - lr * wd) lr (grad + mu * (mu * mom + grad))
    (1 - toReal lr * toReal wd)
    (toReal grad + toReal mu * (toReal mu * toReal mom + toReal grad)) _ _ hc hup
  simpa [stepVecElemR, stepVecElemErrBnd] using h

/-! ### `newtonSchulz` as an inductive fold over the coefficient schedule

`newtonSchulz` is an imperative `Id.run` loop: seed `X ← (1/(‖X₀‖_F + eps))·X₀`, then for each of
the 5 tuned coefficient triples run one quintic step `X ← a·X + b·(XXᵀ)X + c·(XXᵀ)²X` (choosing the
smaller Gram side). To reason about it inductively (error growth, magnitude growth, fixed points) we
first strip the imperative shell: `nsIter` is one iteration as a pure function of `(X, coef)`, and
`newtonSchulz_eq_foldl` proves the loop equals `muonCoeffs.foldl nsIter seed`. This is a PURE
structural equality (no float axioms) — the gating scaffolding for every later bound, which now
recurses over the fold instead of the opaque `for`. -/

/-- One Newton–Schulz iteration as a pure function of the current iterate and coefficient triple
    (`X ← a·X + b·(XXᵀ)X + c·(XXᵀ)²X`, smaller-Gram-side branch), matching `newtonSchulz`'s loop body. -/
def nsIter (X : Mat) (coef : Float × Float × Float) : Mat :=
  let (a, b, c) := coef
  if X.size ≤ X[0]!.size then
    let A := matmul X (Puffer.FloatR.Muon.transpose X)
    let AX := matmul A X
    Puffer.FloatR.Muon.lincomb3 a X b AX c (matmul A AX)
  else
    let A := matmul (Puffer.FloatR.Muon.transpose X) X
    let XA := matmul X A
    Puffer.FloatR.Muon.lincomb3 a X b XA c (matmul XA A)

/-- **`newtonSchulz` = fold of `nsIter` over the coefficient schedule.** The imperative 5-iteration
    loop equals a `foldl` over `muonCoeffs`, seeded by the Frobenius-normalized input. Pure structural
    equality (only `propext`/`Classical.choice`/`Quot.sound`) — the scaffolding for inductive bounds. -/
theorem newtonSchulz_eq_foldl (X0 : Mat) (eps : Float) :
    Puffer.FloatR.Muon.newtonSchulz X0 eps
      = Puffer.FloatR.Muon.muonCoeffs.foldl nsIter
          (Puffer.FloatR.Muon.scalarMul (1.0 / (Puffer.FloatR.Muon.frobNorm X0 + eps)) X0) := by
  unfold Puffer.FloatR.Muon.newtonSchulz Puffer.FloatR.Muon.newtonSchulzWith
  simp only [Id.run, bind_pure_comp, map_pure, bind_pure, ← apply_ite,
    Array.forIn_pure_yield_eq_foldl]
  rfl

/-- **Growth-tracking fold composition.** If a relation `P b c ε` ("Float state `b` is within
    `ε` of real state `c`") is preserved by each step — the Float step `stepF`, the real step
    `stepR`, and a per-step error map `bnd` satisfying `P b c ε → P (stepF b a) (stepR c a) (bnd a ε)`
    — then it is preserved by folding the whole list, with the error folded by the same `bnd`.

    This is the reusable backbone of the `newtonSchulz` bound: with `newtonSchulz_eq_foldl`, bounding
    the 5-iteration loop reduces to the SINGLE per-iteration obligation `P (nsIter X c) (nsIterR XR c)
    (bnd c ε)` (instantiate `P` = "entrywise within ε", `stepF` = `nsIter`, `l` = `muonCoeffs.toList`).
    The error accumulates as `muonCoeffs.foldl bnd ε₀` — the composed magnitude/error growth. -/
theorem foldl_rel {α β γ : Type} (P : β → γ → ℝ → Prop)
    (stepF : β → α → β) (stepR : γ → α → γ) (bnd : α → ℝ → ℝ)
    (step : ∀ b c ε a, P b c ε → P (stepF b a) (stepR c a) (bnd a ε))
    (l : List α) :
    ∀ b c ε, P b c ε → P (l.foldl stepF b) (l.foldl stepR c) (l.foldl (fun e a => bnd a e) ε) := by
  induction l with
  | nil => intro b c ε h; simpa using h
  | cons a as ih =>
      intro b c ε h
      simp only [List.foldl_cons]
      exact ih _ _ _ (step b c ε a h)

/-- **Growth-tracking fold, generalized accumulator.** Same as `foldl_rel` but the tracked bound
    lives in an ARBITRARY type `δ` (not just `ℝ`), so a per-step map can carry several quantities at
    once — e.g. `δ = ℝ × ℝ` to thread a magnitude bound alongside the error bound through the fold.
    This is what the `newtonSchulz` composition needs, since each `nsIter` grows both. -/
theorem foldl_rel_gen {α β γ δ : Type} (P : β → γ → δ → Prop)
    (stepF : β → α → β) (stepR : γ → α → γ) (bnd : α → δ → δ)
    (step : ∀ b c d a, P b c d → P (stepF b a) (stepR c a) (bnd a d))
    (l : List α) :
    ∀ b c d, P b c d → P (l.foldl stepF b) (l.foldl stepR c) (l.foldl (fun e a => bnd a e) d) := by
  induction l with
  | nil => intro b c d h; simpa using h
  | cons a as ih =>
      intro b c d h
      simp only [List.foldl_cons]
      exact ih _ _ _ (step b c d a h)

end Puffer.RL.MuonMatrixRuntime
