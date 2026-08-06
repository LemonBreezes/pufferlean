/-
The WHOLE runnable `newtonSchulz X0 eps` (all 5 Muon iterations) O(1) operator-norm bound `√1.3131 + rounding`
— with EVERY data-dependent precondition a native runnable `Bool`. This unifies the two halves of the tight
Newton–Schulz tower:

  • `NewtonSchulzFull.newtonSchulz_opNorm_le` (the fold bound `‖toMatrixF (newtonSchulz X0 eps)‖ ≤
      √1.3131 + √(r·c)·rounding`) — its remaining precondition being the seed normalization
      `‖toMatrixR (mirror seed)‖² ≤ 1`;
  • `NewtonSchulzSeedClosed`'s runnable `Bool` checks (`matSizeOk`/`matShapeOk`/`matEntryBnd`/`epsCheckB`
      + the canonical mirror `mirrorOf`) which discharge the seed's shape/magnitude/eps-margin data.

The bridge is the missing spectral fact **spectral norm ≤ Frobenius norm**:
  • `gram'_eig_le_frob` : each eigenvalue of `Aᴴ·A` is `≤ ∑ᵢⱼ Aᵢⱼ²` (PSD eigenvalues `≤` their sum = trace);
  • `opNorm_sq_le_frobenius_sq` : `‖A‖₂² ≤ ∑ᵢⱼ Aᵢⱼ²` (via `opNorm_le_of_gram_eigenvalue_bound` at
      `c = √(∑Aᵢⱼ²)`);
  • `seed_opNorm_sq_le_one` : the normalized seed's operator-norm² `≤ 1` — `opNorm² ≤ Frobenius² =
      scale²·∑X₀R²` (`scalarMulR_frobenius_sq`) `≤ 1` (`seed_scale_le_one` from the denominator domination).

`newtonSchulz_opNorm_runnable` then discharges the domination from the runnable `Bool`s (the same
`epsCheckB`/`cfConst` machinery that closes the seed step) and hands the seed bound to
`newtonSchulz_opNorm_le`. Axiom-clean modulo the trusted Float base (+ `toBits_inj`, unused here). No abstract
mirror, no abstract `M`, no `Prop`-level shape, no real-valued hypothesis — the whole 5-iteration algorithm's
dimension-free O(1) spectral bound holds given only the Float matrix `X0`, a Float magnitude `Mf`, and four
native Boolean checks (plus the structural `0 < r`, `0 < c`, `r ≤ c`).
-/
import Mathlib
import Puffer.RL.NewtonSchulzFull
import Puffer.RL.NewtonSchulzSeedClosed
import Puffer.RL.SpectralBridge

namespace Puffer.RL.NewtonSchulzRunnable

open scoped Matrix BigOperators Matrix.Norms.L2Operator
open Matrix
open Puffer.FloatR (toReal u64 u64_lt_one u64_pos toBits_inj)
open Puffer.FloatR.Muon (Mat frobNorm scalarMul muonCoeffs newtonSchulz newtonSchulzWith matMaxAbs)
open Puffer.RL.NewtonSchulzError (MatR MatBnd scalarMulR nsIterBnd)
open Puffer.RL.MatrixEmbed (toMatrixR toMatrixF)
open Puffer.RL.NewtonSchulzFloat (scalarMulR_frobenius_sq seed_scale_le_one)
open Puffer.RL.SpectralBridge (opNorm_le_of_gram_eigenvalue_bound)
open Puffer.RL.NewtonSchulzSeedClosed
open Puffer.RL.NewtonSchulzFull (newtonSchulz_opNorm_le)

/-- Each eigenvalue of the Gram matrix `Aᴴ·A` is `≤` the squared Frobenius norm `∑ᵢⱼ Aᵢⱼ²` (its PSD
    eigenvalues are `≥ 0`, so each is `≤` their sum = `trace(Aᴴ·A) = ∑ᵢⱼ Aᵢⱼ²`). The `Aᴴ·A` analogue of
    `NewtonSchulzFloat.gram_eigenvalue_le_frobenius_sq` (which is for `A·Aᴴ`). -/
theorem gram'_eig_le_frob {r c : Nat} (A : Matrix (Fin r) (Fin c) ℝ) (i : Fin c) :
    (isHermitian_conjTranspose_mul_self A).eigenvalues i ≤ ∑ i, ∑ j, (A i j) ^ 2 := by
  have hHerm := isHermitian_conjTranspose_mul_self A
  have htrace : (Aᴴ * A).trace = ∑ i, ∑ j, (A i j) ^ 2 := by
    simp only [Matrix.trace, Matrix.diag_apply, Matrix.mul_apply, Matrix.conjTranspose_apply,
      star_trivial]
    rw [Finset.sum_comm]; congr 1; funext i; congr 1; funext j; rw [sq]
  calc hHerm.eigenvalues i
      ≤ ∑ j, hHerm.eigenvalues j :=
        Finset.single_le_sum (fun j _ => eigenvalues_conjTranspose_mul_self_nonneg A j)
          (Finset.mem_univ i)
    _ = (Aᴴ * A).trace := by rw [hHerm.trace_eq_sum_eigenvalues]; simp
    _ = ∑ i, ∑ j, (A i j) ^ 2 := htrace

/-- **Spectral norm ≤ Frobenius norm** (squared): `‖A‖₂² ≤ ∑ᵢⱼ Aᵢⱼ²`. Routes `‖A‖ ≤ √(∑Aᵢⱼ²)` through
    `opNorm_le_of_gram_eigenvalue_bound` with the eigenvalue bound `gram'_eig_le_frob`. -/
theorem opNorm_sq_le_frobenius_sq {r c : Nat} [Nonempty (Fin c)] (A : Matrix (Fin r) (Fin c) ℝ) :
    ‖A‖ ^ 2 ≤ ∑ i, ∑ j, (A i j) ^ 2 := by
  set S := ∑ i, ∑ j, (A i j) ^ 2 with hSdef
  have hSnn : 0 ≤ S := by
    rw [hSdef]; apply Finset.sum_nonneg; intro i _; apply Finset.sum_nonneg; intro j _; positivity
  have hle : ‖A‖ ≤ Real.sqrt S :=
    opNorm_le_of_gram_eigenvalue_bound (Real.sqrt_nonneg S)
      (fun i => by rw [Real.sq_sqrt hSnn]; exact gram'_eig_le_frob A i)
  nlinarith [hle, norm_nonneg A, Real.sq_sqrt hSnn, Real.sqrt_nonneg S]

/-- **The normalized seed's operator-norm² is `≤ 1`** from the denominator domination `(1+u64)·√S ≤
    toReal(‖X₀‖_F+eps)`. `opNorm² ≤ Frobenius² = scale²·S` (`opNorm_sq_le_frobenius_sq` +
    `scalarMulR_frobenius_sq`) `≤ 1` (`seed_scale_le_one`). This supplies exactly `newtonSchulz_opNorm_le`'s
    `hseed`. -/
theorem seed_opNorm_sq_le_one (X0 : Mat) (X0R : MatR) (eps : Float) (r c : Nat) [Nonempty (Fin c)]
    (hsz : X0R.size = r) (hrow : ∀ i, i < r → (X0R[i]!).size = c)
    (hdom : (1 + u64) * Real.sqrt (∑ i : Fin r, ∑ j : Fin c, (toMatrixR r c X0R i j) ^ 2)
        ≤ toReal (frobNorm X0 + eps)) :
    ‖toMatrixR r c (scalarMulR (toReal (1.0 / (frobNorm X0 + eps))) X0R)‖ ^ 2 ≤ 1 := by
  set S := ∑ i : Fin r, ∑ j : Fin c, (toMatrixR r c X0R i j) ^ 2 with hSdef
  have hSnn : 0 ≤ S := by
    rw [hSdef]; apply Finset.sum_nonneg; intro i _; apply Finset.sum_nonneg; intro j _; positivity
  calc ‖toMatrixR r c (scalarMulR (toReal (1.0 / (frobNorm X0 + eps))) X0R)‖ ^ 2
      ≤ ∑ i, ∑ j, (toMatrixR r c (scalarMulR (toReal (1.0 / (frobNorm X0 + eps))) X0R) i j) ^ 2 :=
        opNorm_sq_le_frobenius_sq _
    _ = (toReal (1.0 / (frobNorm X0 + eps))) ^ 2 * S :=
        scalarMulR_frobenius_sq (toReal (1.0 / (frobNorm X0 + eps))) X0R r c hsz hrow
    _ ≤ 1 := seed_scale_le_one X0 eps S hSnn hdom

/-- **The WHOLE runnable `newtonSchulz` O(1) bound from runnable `Bool` checks.** The complete 5-iteration
    Muon orthogonalization — actual Float coefficients, real IEEE arithmetic — has
    `‖toMatrixF (newtonSchulz X0 eps)‖₂ ≤ √1.3131 + √(r·c)·(accumulated rounding)` (DIMENSION-FREE O(1) in the
    tight spectral constant `√1.3131 < 1.15`), given ONLY the Float matrix `X0`, a Float magnitude `Mf`, and
    FOUR native Boolean checks — `matSizeOk r c` (size), `matShapeOk r c X0` (shape), `matEntryBnd Mf X0`
    (magnitude/faithful mirror), `epsCheckB (cfConst r c) eps (frobNorm X0)` (eps design margin) — plus the
    structural `0 < r`, `0 < c`, `r ≤ c`. The faithful mirror is `mirrorOf X0` (built), the seed
    normalization `‖·‖² ≤ 1` is discharged via `seed_opNorm_sq_le_one`, and the eps domination via the
    runnable `epsCheckB`/`cfConst` chain. The terminal fully-runnable statement of the tight tower. -/
theorem newtonSchulz_opNorm_runnable (X0 : Mat) (eps Mf : Float) (r c : Nat)
    (hr : 0 < r) (hc : 0 < c) (hrc : r ≤ c)
    (hsize : matSizeOk r c = true)
    (hshape : matShapeOk r c X0 = true)
    (hbnd : matEntryBnd Mf X0 = true)
    (hb : epsCheckB (cfConst r c) eps (frobNorm X0) = true) :
    ‖toMatrixF r c (newtonSchulz X0 eps)‖
      ≤ Real.sqrt 1.3131 + Real.sqrt ((r : ℝ) * c)
          * (muonCoeffs.toList.foldl (fun e a => nsIterBnd r c a e)
              ((1 + u64) * (|toReal (1.0 / (frobNorm X0 + eps))| * toReal Mf),
               u64 * (|toReal (1.0 / (frobNorm X0 + eps))| * toReal Mf)
                 + |toReal (1.0 / (frobNorm X0 + eps))| * 0)).2 := by
  have hsz := matShapeOk_size r c X0 hshape
  have hrow := matShapeOk_row r c X0 hshape
  have hX0 := faithfulMirror_MatBnd X0 r c Mf hsz hrow hbnd
  have hfrobnn := frobNorm_toReal_nonneg X0
  have hcf := cf_domination (cfConst r c) r c (matSizeOk_sound r c hsize) (cfConst_dominates r c)
  have hfrob := epsCheckB_sound (cfConst r c) eps (frobNorm X0) r c hfrobnn hcf hb
  have hcsz : ∀ (i : Nat) (hi : i < X0.size), (X0[i]).size = c := fun i hi => by
    rw [← getElem!_pos X0 i hi]; exact hrow i (by rw [hsz] at hi; exact hi)
  have heps := heps_from_frob_lb X0 (mirrorOf X0) eps r c (toReal Mf) hsz hcsz hX0 hfrob
  have hu : (0 : ℝ) < 1 - u64 := by linarith [u64_lt_one]
  have hC0 : 0 ≤ ((1 + u64) / (1 - u64) - (1 - u64) ^ (r + c + 2)) / (1 - u64) ^ (r + c + 2) := by
    have h1 : (1 : ℝ) ≤ (1 + u64) / (1 - u64) := by rw [le_div_iff₀ hu]; nlinarith [u64_pos]
    have h2 : (1 - u64) ^ (r + c + 2) ≤ 1 := pow_le_one₀ hu.le (by linarith [u64_pos])
    apply div_nonneg (by linarith) (by positivity)
  have hnn : 0 ≤ toReal (frobNorm X0) + toReal eps := by
    have := le_trans (mul_nonneg hC0 hfrobnn) hfrob; linarith
  have hMnn : 0 ≤ toReal Mf := le_trans (abs_nonneg _)
    (matEntryBnd_sound Mf X0 hbnd 0 (by rw [hsz]; exact hr) 0
      (by rw [hrow 0 hr]; exact lt_of_lt_of_le hr hrc))
  have hdom := domination_of_fold_faithful X0 (mirrorOf X0) eps r c (toReal Mf) hsz hcsz hX0 hnn heps
  have : Nonempty (Fin c) := ⟨⟨0, hc⟩⟩
  have hseed := seed_opNorm_sq_le_one X0 (mirrorOf X0) eps r c
    (by rw [mirrorOf_size, hsz])
    (fun i hi => by rw [mirrorOf_rowSize X0 i (by rw [hsz]; exact hi)]; exact hrow i hi) hdom
  exact newtonSchulz_opNorm_le X0 (mirrorOf X0) eps r c (toReal Mf) 0 hr hc hMnn hrc hX0 hseed

/-! ### Threading the coefficient check into a parameterized fold

`newtonSchulz` hard-codes iterating over `muonCoeffs`, so its whole-fold bound has no arbitrary coefficient
to check. `newtonSchulzWith coeffs` runs the SAME body over an arbitrary schedule `coeffs`
(`newtonSchulz = newtonSchulzWith muonCoeffs`, `rfl`). The runnable `coeffListOk coeffs` verifies `coeffs`
bit-matches `muonCoeffs` positionally (per-component `coeffOk`-style `toBits` comparison), so the proven O(1)
bound transports to ANY coefficient table that passes the check — e.g. one loaded from a config. -/

/-- Positional bit-equality of coefficient triples (`toBits` per component). -/
def coeffEq (p q : Float × Float × Float) : Bool :=
  p.1.toBits == q.1.toBits && p.2.1.toBits == q.2.1.toBits && p.2.2.toBits == q.2.2.toBits

/-- `coeffEq p q = true ⟹ p = q` — bit equality per component lifted to `Float` (and `Prod`) equality by
    `toBits_inj`. -/
theorem coeffEq_sound {p q : Float × Float × Float} (h : coeffEq p q = true) : p = q := by
  rw [coeffEq, Bool.and_eq_true, Bool.and_eq_true] at h
  obtain ⟨⟨h1, h2⟩, h3⟩ := h
  obtain ⟨p1, p2, p3⟩ := p; obtain ⟨q1, q2, q3⟩ := q
  simp only [Prod.mk.injEq]
  exact ⟨toBits_inj (eq_of_beq h1), toBits_inj (eq_of_beq h2), toBits_inj (eq_of_beq h3)⟩

/-- Runnable coefficient-schedule check: `coeffs` has the same size as `muonCoeffs` and bit-matches it
    positionally. `Bool`-valued. -/
def coeffListOk (coeffs : Array (Float × Float × Float)) : Bool :=
  coeffs.size == muonCoeffs.size &&
    (List.range muonCoeffs.size).all (fun i => coeffEq (coeffs[i]!) (muonCoeffs[i]!))

/-- **Soundness of the schedule check.** `coeffListOk coeffs = true ⟹ coeffs = muonCoeffs` — size equality
    + positional `coeffEq` (`toBits_inj`) via `Array.ext`. -/
theorem coeffListOk_sound (coeffs : Array (Float × Float × Float)) (h : coeffListOk coeffs = true) :
    coeffs = muonCoeffs := by
  rw [coeffListOk, Bool.and_eq_true, beq_iff_eq, List.all_eq_true] at h
  obtain ⟨hsize, hall⟩ := h
  apply Array.ext hsize
  intro i hcoeff hmuon
  have := coeffEq_sound (hall i (List.mem_range.mpr hmuon))
  rwa [getElem!_pos coeffs i hcoeff, getElem!_pos muonCoeffs i hmuon] at this

/-- **The parameterized runnable fold bound.** For an arbitrary coefficient schedule `coeffs` that passes
    the runnable `coeffListOk` check (bit-matches `muonCoeffs`), the whole `newtonSchulzWith coeffs X0 eps`
    satisfies the same dimension-free O(1) bound `√1.3131 + √(r·c)·(rounding)` — given the four data `Bool`
    checks. This threads the coefficient check into the fold: the O(1) guarantee holds for any schedule the
    trainer supplies AS LONG AS it passes `coeffListOk` at runtime. -/
theorem newtonSchulzWith_opNorm_runnable (coeffs : Array (Float × Float × Float))
    (X0 : Mat) (eps Mf : Float) (r c : Nat)
    (hr : 0 < r) (hc : 0 < c) (hrc : r ≤ c)
    (hcoeffs : coeffListOk coeffs = true)
    (hsize : matSizeOk r c = true)
    (hshape : matShapeOk r c X0 = true)
    (hbnd : matEntryBnd Mf X0 = true)
    (hb : epsCheckB (cfConst r c) eps (frobNorm X0) = true) :
    ‖toMatrixF r c (newtonSchulzWith coeffs X0 eps)‖
      ≤ Real.sqrt 1.3131 + Real.sqrt ((r : ℝ) * c)
          * (muonCoeffs.toList.foldl (fun e a => nsIterBnd r c a e)
              ((1 + u64) * (|toReal (1.0 / (frobNorm X0 + eps))| * toReal Mf),
               u64 * (|toReal (1.0 / (frobNorm X0 + eps))| * toReal Mf)
                 + |toReal (1.0 / (frobNorm X0 + eps))| * 0)).2 := by
  rw [coeffListOk_sound coeffs hcoeffs]
  exact newtonSchulz_opNorm_runnable X0 eps Mf r c hr hc hrc hsize hshape hbnd hb

/-! ### The structural dimension facts as a runnable `Bool`

The fold capstone still takes `0 < r`, `0 < c`, `r ≤ c` as `Prop`s. They are decidable Nat comparisons;
`matDimOk r c` bundles them (`0 < r`, `r ≤ c`; `0 < c` follows). With it, `newtonSchulzWith_opNorm_all_bool`
takes EVERY precondition as a native runnable `Bool` — nothing but Float data and Boolean checks. -/

/-- Runnable dimension check: `0 < r` and `r ≤ c` (whence `0 < c`). `Bool`-valued. -/
def matDimOk (r c : Nat) : Bool := decide (0 < r) && decide (r ≤ c)

theorem matDimOk_pos {r c : Nat} (h : matDimOk r c = true) : 0 < r := by
  rw [matDimOk, Bool.and_eq_true, decide_eq_true_eq, decide_eq_true_eq] at h; exact h.1

theorem matDimOk_le {r c : Nat} (h : matDimOk r c = true) : r ≤ c := by
  rw [matDimOk, Bool.and_eq_true, decide_eq_true_eq, decide_eq_true_eq] at h; exact h.2

/-- **The tight Newton–Schulz O(1) bound with EVERY precondition a runnable `Bool`.** The whole
    `newtonSchulzWith coeffs X0 eps` (5 iterations, real IEEE arithmetic) satisfies `‖·‖₂ ≤
    √1.3131 + √(r·c)·(rounding)` given ONLY the Float data (`coeffs`, `X0`, `eps`, `Mf`) and SIX native
    Boolean checks — `matDimOk r c` (dimensions), `coeffListOk coeffs` (schedule), `matSizeOk r c` (size),
    `matShapeOk r c X0` (shape), `matEntryBnd Mf X0` (magnitude), `epsCheckB (cfConst r c) eps (frobNorm X0)`
    (eps margin). No `Prop`-level hypothesis remains: the entire precondition set is runtime-decidable. -/
theorem newtonSchulzWith_opNorm_all_bool (coeffs : Array (Float × Float × Float))
    (X0 : Mat) (eps Mf : Float) (r c : Nat)
    (hdim : matDimOk r c = true)
    (hcoeffs : coeffListOk coeffs = true)
    (hsize : matSizeOk r c = true)
    (hshape : matShapeOk r c X0 = true)
    (hbnd : matEntryBnd Mf X0 = true)
    (hb : epsCheckB (cfConst r c) eps (frobNorm X0) = true) :
    ‖toMatrixF r c (newtonSchulzWith coeffs X0 eps)‖
      ≤ Real.sqrt 1.3131 + Real.sqrt ((r : ℝ) * c)
          * (muonCoeffs.toList.foldl (fun e a => nsIterBnd r c a e)
              ((1 + u64) * (|toReal (1.0 / (frobNorm X0 + eps))| * toReal Mf),
               u64 * (|toReal (1.0 / (frobNorm X0 + eps))| * toReal Mf)
                 + |toReal (1.0 / (frobNorm X0 + eps))| * 0)).2 :=
  newtonSchulzWith_opNorm_runnable coeffs X0 eps Mf r c (matDimOk_pos hdim)
    (lt_of_lt_of_le (matDimOk_pos hdim) (matDimOk_le hdim)) (matDimOk_le hdim)
    hcoeffs hsize hshape hbnd hb

/-! ### The magnitude bound computed from the matrix (no caller-supplied `Mf`)

`Mf` was a free Float parameter the caller had to choose. `matMaxAbs X0` (the max `|entry|`, folded) computes
it from the matrix itself, so the capstone has no magnitude parameter — the bound is `toReal (matMaxAbs X0)`,
and the runnable `matEntryBnd (matMaxAbs X0) X0` check gates it (`Float` `≤` is not even a `Preorder` —
`a ≤ a` fails for NaN — so this check is a genuine runtime NaN/finiteness guard, not provable away). -/

/-- **`matMaxAbs X` dominates every entry in ℝ.** The runnable folded max-abs `matMaxAbs X`
    (`X.foldl (fun acc row => row.foldl (fun a x => max a (max x (-x))) acc) 0.0`, used as the self-computed
    magnitude in `newtonSchulzWith_opNorm_selfM` and downstream) really is an upper bound on the magnitude of
    every matrix entry: for any in-range indices `i < X.size`, `j < (X[i]!).size`,
    `|toReal ((X[i]!)[j]!)| ≤ toReal (matMaxAbs X)`. This is the ℝ-level content behind the runnable finiteness
    guard `matEntryBnd (matMaxAbs X) X` — whose Float `≤` is ONLY a NaN guard (unprovable at the Float level,
    since Float `≤` is not even a `Preorder`): in the trusted-model image, `matMaxAbs X` genuinely dominates each
    `|toReal entry|` UNCONDITIONALLY, no `Bool` check assumed (unlike `matEntryBnd_sound`), because `matMaxAbs X`
    *is* that max by construction. Both index hypotheses are load-bearing — out of range, `X[i]!`/`(X[i]!)[j]!`
    is the panic default whose `toReal` is unconstrained (e.g. `X = #[#[]]` has `matMaxAbs X` embedding to `0`,
    yet `(X[0]!)[0]!` is the default). Proved from the exact `toReal_max`/`toReal_neg` model by a nested
    `List.foldl` max-domination induction (each fold step `max a (max x (-x))` becomes `max (toReal a) |toReal x|`
    in ℝ, and monotonicity carries every entry's `|toReal x|` up to the final accumulator). -/
theorem matMaxAbs_entry_le (X : Mat) (i : Nat) (hi : i < X.size)
    (j : Nat) (hj : j < (X[i]!).size) :
    |toReal ((X[i]!)[j]!)| ≤ toReal (matMaxAbs X) := by
  -- `|t| = max t (-t)` in ℝ
  have habs : ∀ t : ℝ, |t| = max t (-t) := by
    intro t; rcases le_total 0 t with h | h
    · rw [abs_of_nonneg h, max_eq_left (by linarith)]
    · rw [abs_of_nonpos h, max_eq_right (by linarith)]
  -- inner (per-row) fold: the running max only grows (lower bound by the accumulator)
  have inner_lb : ∀ (l : List Float) (acc : Float),
      toReal acc ≤ toReal (l.foldl (fun a x => max a (max x (-x))) acc) := by
    intro l
    induction l with
    | nil => intro acc; simp
    | cons y ys ih =>
      intro acc
      simp only [List.foldl_cons]
      refine le_trans ?_ (ih (max acc (max y (-y))))
      rw [Puffer.FloatR.toReal_max]; exact le_max_left _ _
  -- inner (per-row) fold: every element of the row is dominated
  have inner_entry : ∀ (l : List Float) (acc : Float) (x : Float), x ∈ l →
      |toReal x| ≤ toReal (l.foldl (fun a x => max a (max x (-x))) acc) := by
    intro l
    induction l with
    | nil => intro acc x hx; simp at hx
    | cons y ys ih =>
      intro acc x hx
      simp only [List.foldl_cons]
      rcases List.mem_cons.mp hx with rfl | h
      · refine le_trans ?_ (inner_lb ys _)
        rw [Puffer.FloatR.toReal_max, Puffer.FloatR.toReal_max, Puffer.FloatR.toReal_neg, habs]
        exact le_max_right _ _
      · exact ih (max acc (max y (-y))) x h
  -- array wrappers (bridge `Array.foldl` to `List.foldl`)
  have arr_lb : ∀ (r : Array Float) (acc : Float),
      toReal acc ≤ toReal (r.foldl (fun a x => max a (max x (-x))) acc) := by
    intro r acc; rw [← Array.foldl_toList]; exact inner_lb r.toList acc
  have arr_entry : ∀ (r : Array Float) (acc : Float) (x : Float), x ∈ r.toList →
      |toReal x| ≤ toReal (r.foldl (fun a x => max a (max x (-x))) acc) := by
    intro r acc x hx; rw [← Array.foldl_toList]; exact inner_entry r.toList acc x hx
  -- outer (row-of-rows) fold: monotone lower bound
  have outer_lb : ∀ (rows : List (Array Float)) (acc : Float),
      toReal acc ≤ toReal (rows.foldl (fun a r => r.foldl (fun a x => max a (max x (-x))) a) acc) := by
    intro rows
    induction rows with
    | nil => intro acc; simp
    | cons r rs ih =>
      intro acc
      simp only [List.foldl_cons]
      exact le_trans (arr_lb r acc) (ih _)
  -- outer fold: any entry of any row is dominated
  have outer_entry : ∀ (rows : List (Array Float)) (acc : Float) (r : Array Float), r ∈ rows →
      ∀ (x : Float), x ∈ r.toList →
      |toReal x| ≤ toReal (rows.foldl (fun a r => r.foldl (fun a x => max a (max x (-x))) a) acc) := by
    intro rows
    induction rows with
    | nil => intro acc r hr; simp at hr
    | cons r0 rs ih =>
      intro acc r hr x hx
      simp only [List.foldl_cons]
      rcases List.mem_cons.mp hr with rfl | h
      · exact le_trans (arr_entry _ acc x hx) (outer_lb rs _)
      · exact ih (r0.foldl (fun a x => max a (max x (-x))) acc) r h x hx
  -- assemble
  rw [matMaxAbs, ← Array.foldl_toList]
  have hrowmem : (X[i]!) ∈ X.toList := by
    rw [getElem!_pos X i hi]; exact Array.mem_def.mp (Array.getElem_mem hi)
  have hxmem : ((X[i]!)[j]!) ∈ (X[i]!).toList := by
    rw [getElem!_pos (X[i]!) j hj]; exact Array.mem_def.mp (Array.getElem_mem hj)
  exact outer_entry X.toList 0.0 (X[i]!) hrowmem ((X[i]!)[j]!) hxmem

/-- **Self-contained magnitude.** The tight Newton–Schulz O(1) bound with the magnitude bound COMPUTED from
    `X0` (`matMaxAbs X0`, the folded max `|entry|`) — no caller-supplied `Mf`. Guarded by the same six
    runnable `Bool`s, with `matEntryBnd` now checking the computed max (a runtime finiteness guard). The bound
    is `toReal (matMaxAbs X0)` throughout the error term. -/
theorem newtonSchulzWith_opNorm_selfM (coeffs : Array (Float × Float × Float))
    (X0 : Mat) (eps : Float) (r c : Nat)
    (hdim : matDimOk r c = true)
    (hcoeffs : coeffListOk coeffs = true)
    (hsize : matSizeOk r c = true)
    (hshape : matShapeOk r c X0 = true)
    (hbnd : matEntryBnd (matMaxAbs X0) X0 = true)
    (hb : epsCheckB (cfConst r c) eps (frobNorm X0) = true) :
    ‖toMatrixF r c (newtonSchulzWith coeffs X0 eps)‖
      ≤ Real.sqrt 1.3131 + Real.sqrt ((r : ℝ) * c)
          * (muonCoeffs.toList.foldl (fun e a => nsIterBnd r c a e)
              ((1 + u64) * (|toReal (1.0 / (frobNorm X0 + eps))| * toReal (matMaxAbs X0)),
               u64 * (|toReal (1.0 / (frobNorm X0 + eps))| * toReal (matMaxAbs X0))
                 + |toReal (1.0 / (frobNorm X0 + eps))| * 0)).2 :=
  newtonSchulzWith_opNorm_all_bool coeffs X0 eps (matMaxAbs X0) r c hdim hcoeffs hsize hshape hbnd hb

/-! ### The `eps` parameter as a computed default

`eps` was a free Float the caller had to supply large enough to pass the design margin. `epsDefault X0 r c :=
cfConst r c * frobNorm X0` computes it — exactly the product `epsCheckB` compares against, so the check
reduces to the reflexive `v ≤ v` (true for finite `v`, false for NaN — again a runtime finiteness guard, not
provable away since `Float` `≤` is not a `Preorder`). With both `Mf` (`matMaxAbs`) and `eps` (`epsDefault`)
computed, the whole-algorithm bound takes only the schedule, the matrix, and the dimensions. -/

/-- The computed default `eps = cfConst r c · ‖X₀‖_F` — exactly the eps-margin threshold, so `epsDefault`
    is the smallest reflexive choice that passes `epsCheckB`. Runnable. -/
def epsDefault (X0 : Mat) (r c : Nat) : Float := cfConst r c * frobNorm X0

/-- **Both magnitude and eps computed — the bound takes only `coeffs`, `X0`, `r`, `c`.** The tight
    Newton–Schulz O(1) bound with `Mf = matMaxAbs X0` and `eps = epsDefault X0 r c` both computed from the
    matrix; guarded by the same six runnable `Bool`s (`matEntryBnd`/`epsCheckB` now checking the computed
    values — runtime finiteness guards). No free magnitude or eps parameter remains. -/
theorem newtonSchulzWith_opNorm_autoEps (coeffs : Array (Float × Float × Float))
    (X0 : Mat) (r c : Nat)
    (hdim : matDimOk r c = true)
    (hcoeffs : coeffListOk coeffs = true)
    (hsize : matSizeOk r c = true)
    (hshape : matShapeOk r c X0 = true)
    (hbnd : matEntryBnd (matMaxAbs X0) X0 = true)
    (hb : epsCheckB (cfConst r c) (epsDefault X0 r c) (frobNorm X0) = true) :
    ‖toMatrixF r c (newtonSchulzWith coeffs X0 (epsDefault X0 r c))‖
      ≤ Real.sqrt 1.3131 + Real.sqrt ((r : ℝ) * c)
          * (muonCoeffs.toList.foldl (fun e a => nsIterBnd r c a e)
              ((1 + u64) * (|toReal (1.0 / (frobNorm X0 + epsDefault X0 r c))| * toReal (matMaxAbs X0)),
               u64 * (|toReal (1.0 / (frobNorm X0 + epsDefault X0 r c))| * toReal (matMaxAbs X0))
                 + |toReal (1.0 / (frobNorm X0 + epsDefault X0 r c))| * 0)).2 :=
  newtonSchulzWith_opNorm_selfM coeffs X0 (epsDefault X0 r c) r c hdim hcoeffs hsize hshape hbnd hb

/-! ### The coefficient schedule defaulted to `muonCoeffs`

Unlike the magnitude/eps defaults (whose checks are NaN-guards), the coefficient default is FULLY
dischargeable: `coeffListOk` compares `toBits` (`UInt64`), whose `==` is lawfully reflexive (no NaN in
`UInt64`), so `coeffListOk muonCoeffs = true` is provable — no runtime check survives. `newtonSchulzWith
muonCoeffs = newtonSchulz` (`rfl`), so this yields the bound for the ACTUAL `newtonSchulz X0 (epsDefault …)`
with the coefficient schedule and its check both eliminated. -/

/-- `coeffEq p p = true` — bit-equality is reflexive (`UInt64` `==` is lawful). -/
theorem coeffEq_self (p : Float × Float × Float) : coeffEq p p = true := by simp [coeffEq]

/-- `coeffListOk muonCoeffs = true` — the schedule trivially bit-matches itself (reflexive, provable, no
    `native_decide` and no `toBits_inj`). -/
theorem coeffListOk_muonCoeffs : coeffListOk muonCoeffs = true := by
  rw [coeffListOk, Bool.and_eq_true]
  refine ⟨by rw [beq_iff_eq], ?_⟩
  rw [List.all_eq_true]; intro i _; exact coeffEq_self _

/-- **The actual `newtonSchulz`, no coefficient/magnitude/eps parameters.** The tight O(1) bound for the
    real `newtonSchulz X0 (epsDefault X0 r c)` (default schedule `muonCoeffs`, computed eps and magnitude) —
    taking ONLY the Float matrix `X0` and its dimensions `r`, `c`, plus five runnable `Bool` checks
    (`matDimOk`/`matSizeOk`/`matShapeOk`/`matEntryBnd`/`epsCheckB`). The coefficient check is discharged
    (`coeffListOk_muonCoeffs`); `newtonSchulzWith muonCoeffs = newtonSchulz` (`rfl`). -/
theorem newtonSchulz_opNorm_auto (X0 : Mat) (r c : Nat)
    (hdim : matDimOk r c = true)
    (hsize : matSizeOk r c = true)
    (hshape : matShapeOk r c X0 = true)
    (hbnd : matEntryBnd (matMaxAbs X0) X0 = true)
    (hb : epsCheckB (cfConst r c) (epsDefault X0 r c) (frobNorm X0) = true) :
    ‖toMatrixF r c (newtonSchulz X0 (epsDefault X0 r c))‖
      ≤ Real.sqrt 1.3131 + Real.sqrt ((r : ℝ) * c)
          * (muonCoeffs.toList.foldl (fun e a => nsIterBnd r c a e)
              ((1 + u64) * (|toReal (1.0 / (frobNorm X0 + epsDefault X0 r c))| * toReal (matMaxAbs X0)),
               u64 * (|toReal (1.0 / (frobNorm X0 + epsDefault X0 r c))| * toReal (matMaxAbs X0))
                 + |toReal (1.0 / (frobNorm X0 + epsDefault X0 r c))| * 0)).2 :=
  newtonSchulzWith_opNorm_autoEps muonCoeffs X0 r c hdim coeffListOk_muonCoeffs hsize hshape hbnd hb

/-! ### The dimensions computed from `X0`

`r`, `c` were free `Nat`s. They ARE the matrix's own dimensions: `r = X0.size`, `c = (X0[0]!).size`.
Instantiating there removes them as parameters — the tight O(1) bound then takes ONLY the Float matrix `X0`,
with the five `Bool` checks read at the computed dimensions. (The checks stay: `matShapeOk` verifies
rectangularity — every row the same width as row 0 — and `matDimOk`/`matSizeOk` verify nonemptiness, the
`r ≤ c` orientation, and the `< 2^52` size limit; all genuine input properties.) -/

set_option maxHeartbeats 1000000 in
/-- **The tight O(1) bound taking only the matrix `X0`.** `newtonSchulz X0 (epsDefault …)` at the matrix's
    own dimensions `r = X0.size`, `c = (X0[0]!).size` — no `r`, `c`, `coeffs`, `Mf`, or `eps` parameters, just
    `X0` and the five runnable `Bool` checks (read at the computed dimensions). The terminal fully-computed
    runnable statement of the tight Newton–Schulz operator-norm bound. -/
theorem newtonSchulz_opNorm_dims (X0 : Mat)
    (hdim : matDimOk X0.size (X0[0]!).size = true)
    (hsize : matSizeOk X0.size (X0[0]!).size = true)
    (hshape : matShapeOk X0.size (X0[0]!).size X0 = true)
    (hbnd : matEntryBnd (matMaxAbs X0) X0 = true)
    (hb : epsCheckB (cfConst X0.size (X0[0]!).size)
        (epsDefault X0 X0.size (X0[0]!).size) (frobNorm X0) = true) :
    ‖toMatrixF X0.size (X0[0]!).size (newtonSchulz X0 (epsDefault X0 X0.size (X0[0]!).size))‖
      ≤ Real.sqrt 1.3131 + Real.sqrt ((X0.size : ℝ) * (X0[0]!).size)
          * (muonCoeffs.toList.foldl (fun e a => nsIterBnd X0.size (X0[0]!).size a e)
              (((1 : ℝ) + u64) * (|toReal (1.0 / (frobNorm X0 + epsDefault X0 X0.size (X0[0]!).size))|
                  * toReal (matMaxAbs X0)),
               u64 * (|toReal (1.0 / (frobNorm X0 + epsDefault X0 X0.size (X0[0]!).size))|
                   * toReal (matMaxAbs X0))
                 + |toReal (1.0 / (frobNorm X0 + epsDefault X0 X0.size (X0[0]!).size))| * 0)).2 :=
  newtonSchulz_opNorm_auto X0 X0.size (X0[0]!).size hdim hsize hshape hbnd hb

/-! ### The shape check reduced to pure rectangularity

At the matrix's own dimensions the size-equality half of `matShapeOk` (`X0.size = X0.size`) is trivially
true, so the check collapses to pure rectangularity — every row the width of row 0. `matRectOk X0` isolates
that, and `newtonSchulz_opNorm_rect` takes it in place of the full `matShapeOk`. -/

/-- Rectangularity check: every row has the width of row 0. `Bool`-valued. -/
def matRectOk (X : Mat) : Bool := X.all (fun row => decide (row.size = (X[0]!).size))

/-- At self dimensions the shape check IS the rectangularity check (`X.size = X.size` is free). -/
theorem matShapeOk_size_self (X : Mat) :
    matShapeOk X.size (X[0]!).size X = matRectOk X := by
  unfold matShapeOk matRectOk
  rw [show decide (X.size = X.size) = true from by simp, Bool.true_and]

/-- **The tight O(1) bound with the shape check as pure rectangularity.** Same as `newtonSchulz_opNorm_dims`
    but the shape hypothesis is `matRectOk X0` (every row the width of row 0) — the size-vs-`r` half is
    discharged at the matrix's own dimensions. -/
theorem newtonSchulz_opNorm_rect (X0 : Mat)
    (hdim : matDimOk X0.size (X0[0]!).size = true)
    (hsize : matSizeOk X0.size (X0[0]!).size = true)
    (hrect : matRectOk X0 = true)
    (hbnd : matEntryBnd (matMaxAbs X0) X0 = true)
    (hb : epsCheckB (cfConst X0.size (X0[0]!).size)
        (epsDefault X0 X0.size (X0[0]!).size) (frobNorm X0) = true) :
    ‖toMatrixF X0.size (X0[0]!).size (newtonSchulz X0 (epsDefault X0 X0.size (X0[0]!).size))‖
      ≤ Real.sqrt 1.3131 + Real.sqrt ((X0.size : ℝ) * (X0[0]!).size)
          * (muonCoeffs.toList.foldl (fun e a => nsIterBnd X0.size (X0[0]!).size a e)
              (((1 : ℝ) + u64) * (|toReal (1.0 / (frobNorm X0 + epsDefault X0 X0.size (X0[0]!).size))|
                  * toReal (matMaxAbs X0)),
               u64 * (|toReal (1.0 / (frobNorm X0 + epsDefault X0 X0.size (X0[0]!).size))|
                   * toReal (matMaxAbs X0))
                 + |toReal (1.0 / (frobNorm X0 + epsDefault X0 X0.size (X0[0]!).size))| * 0)).2 :=
  newtonSchulz_opNorm_dims X0 hdim hsize ((matShapeOk_size_self X0).trans hrect) hbnd hb

/-! ### Merging the two dimension checks

`matDimOk` (nonempty + `r ≤ c` orientation) and `matSizeOk` (`r+c+2 ≤ 2^52` size limit) are both Nat checks
on the dimensions. `matDimsOk` combines them into one, so the capstone takes FOUR Boolean checks. -/

/-- Combined dimension check: nonemptiness, `r ≤ c` orientation, and the `< 2^52` size limit. `Bool`. -/
def matDimsOk (r c : Nat) : Bool := matDimOk r c && matSizeOk r c

theorem matDimsOk_dim {r c : Nat} (h : matDimsOk r c = true) : matDimOk r c = true :=
  (Bool.and_eq_true _ _ ▸ h).1

theorem matDimsOk_size {r c : Nat} (h : matDimsOk r c = true) : matSizeOk r c = true :=
  (Bool.and_eq_true _ _ ▸ h).2

/-- **The tight O(1) bound with the two dimension checks merged.** `newtonSchulz_opNorm_rect` with `matDimOk`
    and `matSizeOk` combined into the single `matDimsOk X0.size (X0[0]!).size` — leaving FOUR runnable `Bool`
    checks: `matDimsOk`, `matRectOk`, `matEntryBnd`, `epsCheckB`. -/
theorem newtonSchulz_opNorm_final (X0 : Mat)
    (hdims : matDimsOk X0.size (X0[0]!).size = true)
    (hrect : matRectOk X0 = true)
    (hbnd : matEntryBnd (matMaxAbs X0) X0 = true)
    (hb : epsCheckB (cfConst X0.size (X0[0]!).size)
        (epsDefault X0 X0.size (X0[0]!).size) (frobNorm X0) = true) :
    ‖toMatrixF X0.size (X0[0]!).size (newtonSchulz X0 (epsDefault X0 X0.size (X0[0]!).size))‖
      ≤ Real.sqrt 1.3131 + Real.sqrt ((X0.size : ℝ) * (X0[0]!).size)
          * (muonCoeffs.toList.foldl (fun e a => nsIterBnd X0.size (X0[0]!).size a e)
              (((1 : ℝ) + u64) * (|toReal (1.0 / (frobNorm X0 + epsDefault X0 X0.size (X0[0]!).size))|
                  * toReal (matMaxAbs X0)),
               u64 * (|toReal (1.0 / (frobNorm X0 + epsDefault X0 X0.size (X0[0]!).size))|
                   * toReal (matMaxAbs X0))
                 + |toReal (1.0 / (frobNorm X0 + epsDefault X0 X0.size (X0[0]!).size))| * 0)).2 :=
  newtonSchulz_opNorm_rect X0 (matDimsOk_dim hdims) (matDimsOk_size hdims) hrect hbnd hb

/-! ### Merging dimensions and rectangularity into one structural check

`matDimsOk` (a dimension check on `X0.size`, `(X0[0]!).size`) and `matRectOk X0` (rectangularity) are both
structural facts about `X0`. `matStructOk X0` bundles them into a single `X0`-level `Bool`, so the capstone
takes THREE checks: structure, finiteness, eps. -/

/-- Combined structural check on `X0`: dimensions (nonempty, `r ≤ c`, `< 2^52`) AND rectangularity. `Bool`. -/
def matStructOk (X : Mat) : Bool := matDimsOk X.size (X[0]!).size && matRectOk X

theorem matStructOk_dims {X : Mat} (h : matStructOk X = true) : matDimsOk X.size (X[0]!).size = true :=
  (Bool.and_eq_true _ _ ▸ h).1

theorem matStructOk_rect {X : Mat} (h : matStructOk X = true) : matRectOk X = true :=
  (Bool.and_eq_true _ _ ▸ h).2

/-- **The tight O(1) bound with all structure in one check.** `newtonSchulz_opNorm_final` with `matDimsOk`
    and `matRectOk` bundled into `matStructOk X0`, leaving THREE runnable `Bool` checks: `matStructOk`
    (dimensions + rectangularity), `matEntryBnd` (finiteness), `epsCheckB` (eps margin). -/
theorem newtonSchulz_opNorm_struct (X0 : Mat)
    (hstruct : matStructOk X0 = true)
    (hbnd : matEntryBnd (matMaxAbs X0) X0 = true)
    (hb : epsCheckB (cfConst X0.size (X0[0]!).size)
        (epsDefault X0 X0.size (X0[0]!).size) (frobNorm X0) = true) :
    ‖toMatrixF X0.size (X0[0]!).size (newtonSchulz X0 (epsDefault X0 X0.size (X0[0]!).size))‖
      ≤ Real.sqrt 1.3131 + Real.sqrt ((X0.size : ℝ) * (X0[0]!).size)
          * (muonCoeffs.toList.foldl (fun e a => nsIterBnd X0.size (X0[0]!).size a e)
              (((1 : ℝ) + u64) * (|toReal (1.0 / (frobNorm X0 + epsDefault X0 X0.size (X0[0]!).size))|
                  * toReal (matMaxAbs X0)),
               u64 * (|toReal (1.0 / (frobNorm X0 + epsDefault X0 X0.size (X0[0]!).size))|
                   * toReal (matMaxAbs X0))
                 + |toReal (1.0 / (frobNorm X0 + epsDefault X0 X0.size (X0[0]!).size))| * 0)).2 :=
  newtonSchulz_opNorm_final X0 (matStructOk_dims hstruct) (matStructOk_rect hstruct) hbnd hb

/-! ### Merging the two Float-comparison guards into one finiteness check

`matEntryBnd (matMaxAbs X0) X0` (entries `≤` their computed max) and `epsCheckB … (frobNorm X0)` (the
computed eps clears the margin) are both the Float-comparison NaN/finiteness guards (`Float` `≤` is not
reflexive under NaN, so each fails on non-finite data). `matFiniteOk X0` bundles them, leaving TWO checks:
well-formedness and finiteness. -/

/-- Combined finiteness guard on `X0`: the magnitude check (entries `≤ matMaxAbs X0`) AND the eps check
    (`epsCheckB` at the computed eps) — both pass exactly when `X0` is finite. `Bool`. -/
def matFiniteOk (X : Mat) : Bool :=
  matEntryBnd (matMaxAbs X) X
    && epsCheckB (cfConst X.size (X[0]!).size) (epsDefault X X.size (X[0]!).size) (frobNorm X)

theorem matFiniteOk_bnd {X : Mat} (h : matFiniteOk X = true) : matEntryBnd (matMaxAbs X) X = true :=
  (Bool.and_eq_true _ _ ▸ h).1

theorem matFiniteOk_eps {X : Mat} (h : matFiniteOk X = true) :
    epsCheckB (cfConst X.size (X[0]!).size) (epsDefault X X.size (X[0]!).size) (frobNorm X) = true :=
  (Bool.and_eq_true _ _ ▸ h).2

/-- **The tight O(1) bound gated by just two checks: well-formed and finite.** `newtonSchulz_opNorm_struct`
    with the two Float-comparison guards bundled into `matFiniteOk X0`. The dimension-free O(1) operator-norm
    bound for the actual runnable `newtonSchulz X0 (epsDefault …)` from the Float matrix `X0` and exactly TWO
    decidable checks — `matStructOk X0` (well-formed) and `matFiniteOk X0` (finite). -/
theorem newtonSchulz_opNorm_wf (X0 : Mat)
    (hstruct : matStructOk X0 = true)
    (hfin : matFiniteOk X0 = true) :
    ‖toMatrixF X0.size (X0[0]!).size (newtonSchulz X0 (epsDefault X0 X0.size (X0[0]!).size))‖
      ≤ Real.sqrt 1.3131 + Real.sqrt ((X0.size : ℝ) * (X0[0]!).size)
          * (muonCoeffs.toList.foldl (fun e a => nsIterBnd X0.size (X0[0]!).size a e)
              (((1 : ℝ) + u64) * (|toReal (1.0 / (frobNorm X0 + epsDefault X0 X0.size (X0[0]!).size))|
                  * toReal (matMaxAbs X0)),
               u64 * (|toReal (1.0 / (frobNorm X0 + epsDefault X0 X0.size (X0[0]!).size))|
                   * toReal (matMaxAbs X0))
                 + |toReal (1.0 / (frobNorm X0 + epsDefault X0 X0.size (X0[0]!).size))| * 0)).2 :=
  newtonSchulz_opNorm_struct X0 hstruct (matFiniteOk_bnd hfin) (matFiniteOk_eps hfin)

/-! ### One input-validity check

`matStructOk X0` (well-formed) and `matFiniteOk X0` (finite) are the two irreducible input facts. `matOk X0`
merges them into a single `Bool` — the whole precondition of the tight O(1) bound is then ONE decidable check
on the input matrix. -/

/-- The complete input-validity check: `X0` is a well-formed rectangular matrix (`matStructOk`) AND finite
    (`matFiniteOk`). `Bool`-valued — the sole precondition of the tight Newton–Schulz O(1) bound. -/
def matOk (X : Mat) : Bool := matStructOk X && matFiniteOk X

theorem matOk_struct {X : Mat} (h : matOk X = true) : matStructOk X = true :=
  (Bool.and_eq_true _ _ ▸ h).1

theorem matOk_fin {X : Mat} (h : matOk X = true) : matFiniteOk X = true :=
  (Bool.and_eq_true _ _ ▸ h).2

/-- **The tight dimension-free O(1) Newton–Schulz bound gated by a SINGLE input check.** For any Float
    matrix `X0` passing the one decidable validity check `matOk X0` (well-formed + finite), the actual
    runnable `newtonSchulz X0 (epsDefault …)` — 5 Muon iterations, real IEEE arithmetic — has operator norm
    `≤ √1.3131 + √(r·c)·(rounding)`, DIMENSION-FREE O(1) in the tight spectral constant `√1.3131 < 1.15`. Every
    other ingredient (coefficient schedule, magnitude, eps, dimensions) is built in or computed from `X0`;
    the spectral bound is proved down to the trusted Float model. The terminal statement of the tight
    Newton–Schulz tower. -/
theorem newtonSchulz_opNorm (X0 : Mat) (hok : matOk X0 = true) :
    ‖toMatrixF X0.size (X0[0]!).size (newtonSchulz X0 (epsDefault X0 X0.size (X0[0]!).size))‖
      ≤ Real.sqrt 1.3131 + Real.sqrt ((X0.size : ℝ) * (X0[0]!).size)
          * (muonCoeffs.toList.foldl (fun e a => nsIterBnd X0.size (X0[0]!).size a e)
              (((1 : ℝ) + u64) * (|toReal (1.0 / (frobNorm X0 + epsDefault X0 X0.size (X0[0]!).size))|
                  * toReal (matMaxAbs X0)),
               u64 * (|toReal (1.0 / (frobNorm X0 + epsDefault X0 X0.size (X0[0]!).size))|
                   * toReal (matMaxAbs X0))
                 + |toReal (1.0 / (frobNorm X0 + epsDefault X0 X0.size (X0[0]!).size))| * 0)).2 :=
  newtonSchulz_opNorm_wf X0 (matOk_struct hok) (matOk_fin hok)

end Puffer.RL.NewtonSchulzRunnable
