/-
# The Newton–Schulz unit ball on REAL matrices: the complexification wiring

C59 (`NewtonSchulzBall`) proved the invariant operator-norm unit ball and the `3^k` k-iteration Lipschitz for the
classical Newton–Schulz step in an abstract C*-order setting `[CStarAlgebra A][PartialOrder A][StarOrderedRing A]`,
and C62 (`MuonBallWholeRun`) built the Muon whole-run interval on it. Both disclosed one remaining wiring: the
repo's trainer matrices are REAL, while the C*-order machinery is complex. This module closes that gap by
COMPLEXIFICATION:

* a local `CStarAlgebra (Matrix (Fin d) (Fin d) ℂ)` instance assembled from Mathlib's scoped L2-operator-norm
  pieces (`Matrix.instL2OpNormedRing`/`instL2OpNormedAlgebra`/`instCStarRing`, scoped `Matrix.Norms.L2Operator`) —
  with the Loewner order (`MatrixOrder` scoped `Matrix.instPartialOrder`/`instStarOrderedRing`), C59's theorems
  apply DIRECTLY to `Matrix (Fin d) (Fin d) ℂ` with the L2 operator norm (the instance is `scoped` to this
  namespace to avoid leaking the L2 norm choice into other files);
* **`complexify X := X.map Complex.ofReal`** — the entrywise real→complex embedding, a ⋆-ring homomorphism
  (`complexify_mul`/`complexify_star`/`complexify_smul`/`complexify_add`/`complexify_sub`), so the Newton–Schulz
  step COMMUTES with it (`complexify_nsStarStep`, `complexify_nsIter` — the step is a ⋆-polynomial);
* **`norm_complexify : ‖complexify X‖ = ‖X‖`** — the KEY analytic step, proved two-sided with no SVD:
  `‖X‖ ≤ ‖complexify X‖` because real unit vectors are complex unit vectors of the same norm
  (`norm_toLp_ofReal`, `mulVec_complexify_real`), and `‖complexify X‖ ≤ ‖X‖` because a complex vector splits into
  real and imaginary parts that a REAL matrix does not mix — `‖X_ℂ(u+iv)‖² = ‖Xu‖² + ‖Xv‖²` (entrywise
  `mulVec_complexify_decomp` + `Complex.norm_add_mul_I`), so the complex Rayleigh quotient never exceeds the real
  one;
* the pulled-back capstones on REAL matrices: **`nsStarStep_ball_invariant_real`** (`‖X‖ ≤ 1 ⟹
  ‖nsStarStep (3/2)(−1/2) 0 X‖ ≤ 1`), **`nsIter_ball_invariant_real`**, and **`nsIter_lipschitz_ball_real`**
  (`‖nsIter … k X − nsIter … k Y‖ ≤ 3^k·‖X − Y‖` on the real unit ball) — C59's full payoff on the repo's actual
  trainer shape, making C62's Muon whole-run interval applicable to real matrices.

**Scope (honestly disclosed).** The norm equality is full (two-sided), so C59/C62's unit-ball results now hold for
REAL square matrices with the L2 operator norm — the complexification residual both modules disclosed is CLOSED.
The `CStarAlgebra` instance on `Matrix (Fin d) (Fin d) ℂ` is assembled from Mathlib's own scoped pieces (no new
structure, no axiom) and kept `scoped` to this namespace. What remains beyond this module is only what C62 already
named: the gradient's operator-norm Lipschitz `G` (C60 sup-metric + C63 transfer + the vector↔matrix reshaping)
and the per-step Float error `B` — plus the concrete-run data-dependent conditions.
-/
import Puffer.RL.NewtonSchulzBall

namespace Puffer.RL.RealMatrixBall

open Matrix WithLp
open Puffer.RL.NewtonSchulzMatrixLipschitz (nsStarStep)
open Puffer.RL.NewtonSchulzIterate (nsIter nsIter_zero nsIter_succ)
open scoped Matrix.Norms.L2Operator MatrixOrder ComplexOrder

variable {d : ℕ}

/-- The L2-operator-norm C*-algebra structure on complex square matrices, assembled from Mathlib's scoped
    `Matrix.Norms.L2Operator` pieces (`instL2OpNormedRing`/`instL2OpNormedAlgebra`/`instCStarRing`) plus the global
    star/star-module/complete-space instances. `scoped` so the L2 norm choice does not leak into files that expect a
    different matrix norm. With the scoped `MatrixOrder` (Loewner) instances this makes `Matrix (Fin d) (Fin d) ℂ`
    an instance of C59's `[CStarAlgebra][PartialOrder][StarOrderedRing]` setting. -/
noncomputable scoped instance : CStarAlgebra (Matrix (Fin d) (Fin d) ℂ) :=
  { Matrix.instL2OpNormedRing, Matrix.instL2OpNormedAlgebra, Matrix.instCStarRing,
    (inferInstance : StarRing (Matrix (Fin d) (Fin d) ℂ)),
    (inferInstance : StarModule ℂ (Matrix (Fin d) (Fin d) ℂ)),
    (inferInstance : CompleteSpace (Matrix (Fin d) (Fin d) ℂ)) with }

/-- **The entrywise real→complex embedding** of a square matrix. A ⋆-ring homomorphism (below), norm-preserving
    for the L2 operator norms (`norm_complexify`). -/
def complexify (X : Matrix (Fin d) (Fin d) ℝ) : Matrix (Fin d) (Fin d) ℂ :=
  X.map Complex.ofReal

theorem complexify_add (X Y : Matrix (Fin d) (Fin d) ℝ) :
    complexify (X + Y) = complexify X + complexify Y := by
  ext i j
  simp [complexify, Matrix.map_apply]

theorem complexify_sub (X Y : Matrix (Fin d) (Fin d) ℝ) :
    complexify (X - Y) = complexify X - complexify Y := by
  ext i j
  simp [complexify, Matrix.map_apply]

theorem complexify_smul (r : ℝ) (X : Matrix (Fin d) (Fin d) ℝ) :
    complexify (r • X) = r • complexify X := by
  ext i j
  simp [complexify, Matrix.map_apply, Complex.real_smul]

theorem complexify_mul (X Y : Matrix (Fin d) (Fin d) ℝ) :
    complexify (X * Y) = complexify X * complexify Y := by
  ext i j
  simp only [complexify, Matrix.map_apply, Matrix.mul_apply]
  push_cast
  rfl

/-- The embedding intertwines the stars: real `star` (= transpose, trivial conjugation) maps to complex `star`
    (= conjugate transpose; `Complex.ofReal` is conjugation-fixed). -/
theorem complexify_star (X : Matrix (Fin d) (Fin d) ℝ) :
    complexify (star X) = star (complexify X) := by
  ext i j
  simp [complexify, Matrix.map_apply, Matrix.star_eq_conjTranspose]

/-- The Newton–Schulz step is a ⋆-polynomial, so it commutes with the complexification. -/
theorem complexify_nsStarStep (a b c : ℝ) (X : Matrix (Fin d) (Fin d) ℝ) :
    complexify (nsStarStep a b c X) = nsStarStep a b c (complexify X) := by
  simp only [nsStarStep, complexify_add, complexify_smul, complexify_mul, complexify_star]

/-- The k-fold Newton–Schulz iterate commutes with the complexification (induction over `k`). -/
theorem complexify_nsIter (a b c : ℝ) (k : ℕ) (X : Matrix (Fin d) (Fin d) ℝ) :
    complexify (nsIter a b c k X) = nsIter a b c k (complexify X) := by
  induction k with
  | zero => simp
  | succ n ih => rw [nsIter_succ, nsIter_succ, ← ih, complexify_nsStarStep]

/-! ## The operator-norm equality `‖complexify X‖ = ‖X‖` -/

/-- Complexifying a real vector preserves the Euclidean norm (componentwise `‖(r : ℂ)‖ = ‖r‖`). -/
theorem norm_toLp_ofReal (v : Fin d → ℝ) :
    ‖(toLp 2 (fun i => (v i : ℂ)) : EuclideanSpace ℂ (Fin d))‖
      = ‖(toLp 2 v : EuclideanSpace ℝ (Fin d))‖ := by
  rw [EuclideanSpace.norm_eq, EuclideanSpace.norm_eq]
  congr 1
  exact Finset.sum_congr rfl fun i _ => by simp [Complex.norm_real]

/-- The complexified matrix acting on a complexified real vector is the complexified image. -/
theorem mulVec_complexify_real (X : Matrix (Fin d) (Fin d) ℝ) (v : Fin d → ℝ) :
    (complexify X) *ᵥ (fun i => (v i : ℂ)) = fun i => ((X *ᵥ v) i : ℂ) := by
  funext i
  simp only [Matrix.mulVec, dotProduct, complexify, Matrix.map_apply]
  push_cast
  rfl

/-- The L2-operator-norm bound for `mulVec`, in `toLp` form (via the defining CLM's `le_opNorm`). -/
theorem toLp_mulVec_le {𝕜 : Type*} [RCLike 𝕜] (A : Matrix (Fin d) (Fin d) 𝕜) (x : Fin d → 𝕜) :
    ‖(toLp 2 (A *ᵥ x) : EuclideanSpace 𝕜 (Fin d))‖
      ≤ ‖A‖ * ‖(toLp 2 x : EuclideanSpace 𝕜 (Fin d))‖ :=
  ((Matrix.toEuclideanLin.trans LinearMap.toContinuousLinearMap) A).le_opNorm (toLp 2 x)

/-- **Real and imaginary parts do not mix under a real matrix**: the complexified action decomposes entrywise as
    `(X_ℂ z)ᵢ = (X·Re z)ᵢ + (X·Im z)ᵢ·I`. -/
theorem mulVec_complexify_decomp (X : Matrix (Fin d) (Fin d) ℝ) (z : Fin d → ℂ) (i : Fin d) :
    ((complexify X) *ᵥ z) i
      = (((X *ᵥ fun j => (z j).re) i : ℝ) : ℂ)
        + (((X *ᵥ fun j => (z j).im) i : ℝ) : ℂ) * Complex.I := by
  simp only [Matrix.mulVec, dotProduct, complexify, Matrix.map_apply]
  push_cast
  rw [Finset.sum_mul, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun j _ => ?_
  conv_lhs => rw [← Complex.re_add_im (z j)]
  ring

/-- Easy direction: real unit vectors are complex unit vectors, so the real operator norm is dominated. -/
theorem norm_le_norm_complexify (X : Matrix (Fin d) (Fin d) ℝ) : ‖X‖ ≤ ‖complexify X‖ := by
  conv_lhs => rw [Matrix.l2_opNorm_def]
  refine ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg _) fun v => ?_
  show ‖(toLp 2 (X *ᵥ ofLp v) : EuclideanSpace ℝ (Fin d))‖ ≤ ‖complexify X‖ * ‖v‖
  calc ‖(toLp 2 (X *ᵥ ofLp v) : EuclideanSpace ℝ (Fin d))‖
      = ‖(toLp 2 (fun i => ((X *ᵥ ofLp v) i : ℂ)) : EuclideanSpace ℂ (Fin d))‖ :=
        (norm_toLp_ofReal _).symm
    _ = ‖(toLp 2 ((complexify X) *ᵥ (fun i => (ofLp v i : ℂ))) : EuclideanSpace ℂ (Fin d))‖ := by
        rw [mulVec_complexify_real]
    _ ≤ ‖complexify X‖ * ‖(toLp 2 (fun i => (ofLp v i : ℂ)) : EuclideanSpace ℂ (Fin d))‖ :=
        toLp_mulVec_le _ _
    _ = ‖complexify X‖ * ‖v‖ := by rw [norm_toLp_ofReal]

/-- Hard direction: a complex vector splits into real and imaginary parts a real matrix does not mix, so
    `‖X_ℂ z‖² = ‖X·Re z‖² + ‖X·Im z‖² ≤ ‖X‖²·(‖Re z‖² + ‖Im z‖²) = ‖X‖²·‖z‖²`. -/
theorem norm_complexify_le (X : Matrix (Fin d) (Fin d) ℝ) : ‖complexify X‖ ≤ ‖X‖ := by
  conv_lhs => rw [Matrix.l2_opNorm_def]
  refine ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg _) fun z => ?_
  show ‖(toLp 2 ((complexify X) *ᵥ ofLp z) : EuclideanSpace ℂ (Fin d))‖ ≤ ‖X‖ * ‖z‖
  have himg : ‖(toLp 2 ((complexify X) *ᵥ ofLp z) : EuclideanSpace ℂ (Fin d))‖ ^ 2
      = ‖(toLp 2 (X *ᵥ fun j => (ofLp z j).re) : EuclideanSpace ℝ (Fin d))‖ ^ 2
        + ‖(toLp 2 (X *ᵥ fun j => (ofLp z j).im) : EuclideanSpace ℝ (Fin d))‖ ^ 2 := by
    rw [EuclideanSpace.norm_sq_eq, EuclideanSpace.norm_sq_eq, EuclideanSpace.norm_sq_eq,
      ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    have happ : (toLp 2 ((complexify X) *ᵥ ofLp z) : EuclideanSpace ℂ (Fin d)) i
        = ((complexify X) *ᵥ ofLp z) i := rfl
    rw [happ, mulVec_complexify_decomp X (ofLp z) i, Complex.norm_add_mul_I,
      Real.sq_sqrt (by positivity)]
    simp [Real.norm_eq_abs, sq_abs]
  have hz : ‖z‖ ^ 2 = ‖(toLp 2 (fun j => (ofLp z j).re) : EuclideanSpace ℝ (Fin d))‖ ^ 2
      + ‖(toLp 2 (fun j => (ofLp z j).im) : EuclideanSpace ℝ (Fin d))‖ ^ 2 := by
    rw [EuclideanSpace.norm_sq_eq, EuclideanSpace.norm_sq_eq, EuclideanSpace.norm_sq_eq,
      ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    have hzi : (z : EuclideanSpace ℂ (Fin d)) i = ofLp z i := rfl
    rw [hzi, Complex.sq_norm, Complex.normSq_apply]
    simp [Real.norm_eq_abs, sq_abs]
    ring
  have hsq : ‖(toLp 2 ((complexify X) *ᵥ ofLp z) : EuclideanSpace ℂ (Fin d))‖ ^ 2
      ≤ (‖X‖ * ‖z‖) ^ 2 := by
    rw [himg]
    have h1 := pow_le_pow_left₀ (norm_nonneg _) (toLp_mulVec_le X (fun j => (ofLp z j).re)) 2
    have h2 := pow_le_pow_left₀ (norm_nonneg _) (toLp_mulVec_le X (fun j => (ofLp z j).im)) 2
    calc ‖(toLp 2 (X *ᵥ fun j => (ofLp z j).re) : EuclideanSpace ℝ (Fin d))‖ ^ 2
          + ‖(toLp 2 (X *ᵥ fun j => (ofLp z j).im) : EuclideanSpace ℝ (Fin d))‖ ^ 2
        ≤ (‖X‖ * ‖(toLp 2 (fun j => (ofLp z j).re) : EuclideanSpace ℝ (Fin d))‖) ^ 2
          + (‖X‖ * ‖(toLp 2 (fun j => (ofLp z j).im) : EuclideanSpace ℝ (Fin d))‖) ^ 2 :=
          add_le_add h1 h2
      _ = (‖X‖ * ‖z‖) ^ 2 := by rw [mul_pow, mul_pow, ← mul_add, ← hz, mul_pow]
  calc ‖(toLp 2 ((complexify X) *ᵥ ofLp z) : EuclideanSpace ℂ (Fin d))‖
      = Real.sqrt (‖(toLp 2 ((complexify X) *ᵥ ofLp z) : EuclideanSpace ℂ (Fin d))‖ ^ 2) :=
        (Real.sqrt_sq (norm_nonneg _)).symm
    _ ≤ Real.sqrt ((‖X‖ * ‖z‖) ^ 2) := Real.sqrt_le_sqrt hsq
    _ = ‖X‖ * ‖z‖ := Real.sqrt_sq (by positivity)

/-- **The complexification preserves the L2 operator norm** (two-sided; no SVD — real unit vectors embed, and a
    real matrix does not mix real and imaginary parts). -/
theorem norm_complexify (X : Matrix (Fin d) (Fin d) ℝ) : ‖complexify X‖ = ‖X‖ :=
  le_antisymm (norm_complexify_le X) (norm_le_norm_complexify X)

/-! ## The pulled-back capstones on real matrices -/

/-- **The invariant unit ball on REAL matrices**: `‖X‖ ≤ 1 ⟹ ‖nsStarStep (3/2)(−1/2) 0 X‖ ≤ 1` for the classical
    Newton–Schulz step on `Matrix (Fin d) (Fin d) ℝ` with the L2 operator norm — C59's ball invariance pulled back
    through the norm-preserving complexification. -/
theorem nsStarStep_ball_invariant_real (X : Matrix (Fin d) (Fin d) ℝ) (hX : ‖X‖ ≤ 1) :
    ‖nsStarStep (3 / 2) (-(1 / 2)) 0 X‖ ≤ 1 := by
  rw [← norm_complexify, complexify_nsStarStep]
  exact Puffer.RL.NewtonSchulzBall.nsStarStep_ball_invariant _ (by rw [norm_complexify]; exact hX)

/-- The k-fold classical Newton–Schulz iterate keeps REAL matrices in the unit ball. -/
theorem nsIter_ball_invariant_real (k : ℕ) (X : Matrix (Fin d) (Fin d) ℝ) (hX : ‖X‖ ≤ 1) :
    ‖nsIter (3 / 2) (-(1 / 2)) 0 k X‖ ≤ 1 := by
  rw [← norm_complexify, complexify_nsIter]
  exact Puffer.RL.NewtonSchulzBall.nsIter_ball_invariant k _ (by rw [norm_complexify]; exact hX)

/-- **The `3^k` k-iteration Lipschitz bound on the REAL unit ball** — C59's `nsIter_lipschitz_ball` pulled back:
    for `‖X‖, ‖Y‖ ≤ 1`, `‖nsIter (3/2)(−1/2) 0 k X − nsIter … k Y‖ ≤ 3^k·‖X − Y‖`. With this, C62's Muon
    whole-run interval applies to the repo's REAL trainer matrices. -/
theorem nsIter_lipschitz_ball_real (k : ℕ) (X Y : Matrix (Fin d) (Fin d) ℝ)
    (hX : ‖X‖ ≤ 1) (hY : ‖Y‖ ≤ 1) :
    ‖nsIter (3 / 2) (-(1 / 2)) 0 k X - nsIter (3 / 2) (-(1 / 2)) 0 k Y‖ ≤ 3 ^ k * ‖X - Y‖ := by
  have key := Puffer.RL.NewtonSchulzBall.nsIter_lipschitz_ball k (complexify X) (complexify Y)
    (by rw [norm_complexify]; exact hX) (by rw [norm_complexify]; exact hY)
  rw [← complexify_sub, norm_complexify] at key
  calc ‖nsIter (3 / 2) (-(1 / 2)) 0 k X - nsIter (3 / 2) (-(1 / 2)) 0 k Y‖
      = ‖complexify (nsIter (3 / 2) (-(1 / 2)) 0 k X - nsIter (3 / 2) (-(1 / 2)) 0 k Y)‖ :=
        (norm_complexify _).symm
    _ = ‖nsIter (3 / 2) (-(1 / 2)) 0 k (complexify X) - nsIter (3 / 2) (-(1 / 2)) 0 k (complexify Y)‖ := by
        rw [complexify_sub, complexify_nsIter, complexify_nsIter]
    _ ≤ 3 ^ k * ‖X - Y‖ := key

end Puffer.RL.RealMatrixBall
