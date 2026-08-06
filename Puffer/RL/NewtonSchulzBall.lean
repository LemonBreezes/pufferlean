/-
# The Newton–Schulz invariant operator-norm ball — via the C*-identity, no SVD

C56 (`NewtonSchulzStable`) proved the scalar singular-value core of Newton–Schulz norm-stability (classical
coefficients `(3/2, −1/2, 0)`: the interval `[0,1]` is invariant, the basin `[0,√3]` maps into it) and the
operator-norm Lipschitz constant `Lu = 3` ON the unit ball, but disclosed the INVARIANT BALL itself —
`‖X‖ ≤ 1 ⟹ ‖nsStarStep (3/2) (−1/2) 0 X‖ ≤ 1` — as blocked on a singular-value characterization
(`‖nsStarStep X‖ = maxᵢ nsScalar(σᵢ)`) for which Mathlib has no SVD API. This module CLOSES that gap with an
SVD-FREE route through the C*-identity and the order structure of a C*-algebra:

* **The Gram identity** (`star_nsStarStep_mul_self`): writing `F := nsStarStep (3/2) (−1/2) 0 x` and
  `g := x⋆x`, a purely algebraic expansion gives `F⋆F = (9/4)•g − (3/2)•g² + (1/4)•g³ = q(g)` with
  `q(t) = t·(3/2 − t/2)² = nsScalar(√t)²` — the whole NS Gram is a polynomial in the positive element `g`.
* **The factored complement** (`one_sub_star_nsStarStep_mul_self`): `1 − F⋆F = c·((4)•1 − g)·c` with
  `c = ½(1 − g)` selfadjoint — the operator counterpart of the scalar factoring
  `1 − q(t) = ¼(1−t)²(4−t)` (`q_le_one`).
* **Ball invariance** (`nsStarStep_ball_invariant`): `0 ≤ F⋆F` is free (`star_mul_self_nonneg`), and
  `F⋆F ≤ 1` follows from the factored complement by conjugation-positivity
  (`star_left_conjugate_nonneg`: `c⋆·(4•1−g)·c ≥ 0`, with `4•1−g ≥ 0` from `‖g‖ = ‖x‖² ≤ 1 ≤ 4` via
  `CStarAlgebra.norm_le_iff_le_algebraMap`). Then `0 ≤ F⋆F ≤ 1` gives `‖F⋆F‖ ≤ 1`
  (`CStarAlgebra.norm_le_one_iff_of_nonneg`), and the C*-identity `‖F⋆F‖ = ‖F‖²` yields `‖F‖ ≤ 1`.
  NO SVD, no cfc: only the C*-identity, conjugation-positivity, and the norm↔order dictionary.
* **The payoff** (`nsIter_ball_invariant`, `nsIter_lipschitz_ball`): the unit ball is invariant under EVERY
  NS iterate, so the `k`-fold classical NS map is `3^k`-Lipschitz on the unit ball — C50's uniform bound with
  `Lu = 3` (C56's on-ball constant) realized WITHOUT C50's growing crude radii (`nsMagBound 1 = 2`): the
  TRUE radii stay at `1`. This discharges C53's `hLu`-shaped hypothesis on the unit ball.

**Setting (honestly disclosed).** The ball-invariance section works in Mathlib's standard unital-C*-algebra
order setting `[CStarAlgebra A] [PartialOrder A] [StarOrderedRing A]` — every unital C*-algebra carries this
order canonically (Mathlib keeps it as mixin instances); `ℂ` instantiates it out of the box (demonstrated
below via `ComplexOrder`). This is STRONGER than the bare `[NormedRing R] [NormedAlgebra ℝ R] [StarRing R]
[NormedStarGroup R]` setting of C47/C50/C56 — necessarily so: the sharp bound is spectral (the crude triangle
bound on `q(g)` gives `9/4 + 3/2 + 1/4 = 4`, not `1`; positivity of `g` is genuinely load-bearing). For
concrete matrices, Mathlib provides the ingredients (`Matrix.instCStarRing` under `Matrix.Norms.L2Operator`,
and the Loewner `Matrix.instStarOrderedRing` in `Mathlib.Analysis.Matrix.Order`) but no pre-bundled
`CStarAlgebra (Matrix n n ℂ)` instance — assembling that bundle (and the ℝ-matrix complexification) is the
remaining wiring, disclosed, not done here. Composing the `3^k` ball Lipschitz into C53's `muonStep`
(gradient clipped to the unit ball) is the immediate next composition.
-/
import Puffer.RL.NewtonSchulzStable
import Mathlib.Analysis.CStarAlgebra.ContinuousFunctionalCalculus.Order

open Puffer.RL.NewtonSchulzMatrixLipschitz (nsStarStep)
open Puffer.RL.NewtonSchulzIterate (nsIter nsIter_zero nsIter_succ)
open Puffer.RL.NewtonSchulzStable (nsStarStep_lipschitz_unit)

namespace Puffer.RL.NewtonSchulzBall

/-! ### The scalar Gram polynomial `q(t) = t·(3/2 − t/2)²` -/

/-- `q(t) = t·(3/2 − t/2)² ≥ 0` for `t ≥ 0` — the scalar shadow of `F⋆F ≥ 0`. -/
theorem q_nonneg (t : ℝ) (h0 : 0 ≤ t) : 0 ≤ t * (3 / 2 - t / 2) ^ 2 :=
  mul_nonneg h0 (sq_nonneg _)

/-- **The scalar complement factoring.** `1 − q(t) = ¼(1−t)²(4−t) ≥ 0`, so `q(t) ≤ 1` on the whole basin
    `0 ≤ t ≤ 4` (not just `[0,1]`) — the scalar shadow of the operator factoring
    `1 − F⋆F = c·(4•1−g)·c`. -/
theorem q_le_one (t : ℝ) (_h0 : 0 ≤ t) (h4 : t ≤ 4) : t * (3 / 2 - t / 2) ^ 2 ≤ 1 := by
  nlinarith [mul_nonneg (sq_nonneg (1 - t)) (sub_nonneg.mpr h4)]

section CStar

variable {A : Type*} [CStarAlgebra A]

/-- Real scalars pass through `star` in a C*-algebra: `star (r • a) = r • star a`. (The ℝ-action is the
    restriction of the ℂ-action — `Complex.coe_smul` is `rfl` — and `star` conjugates the complex scalar,
    which is real.) This substitutes for the `StarModule ℝ A` instance Mathlib does not derive globally. -/
theorem star_real_smul (r : ℝ) (a : A) : star (r • a) = r • star a := by
  rw [← Complex.coe_smul, star_smul, Complex.star_def, Complex.conj_ofReal, Complex.coe_smul]

/-- **The Gram identity (the `q(g)` shape).** For the classical NS step `F = (3/2)•x − (1/2)•(x·x⋆·x)`,
    `F⋆F = (9/4)•g − (3/2)•g² + (1/4)•g³` with `g = x⋆x` — the NS Gram is the polynomial
    `q(t) = t·(3/2 − t/2)²` evaluated at the positive element `g`. Purely algebraic (expansion +
    `module`); no norm, order, or spectrum involved. -/
theorem star_nsStarStep_mul_self (x : A) :
    star (nsStarStep (3 / 2) (-(1 / 2)) 0 x) * nsStarStep (3 / 2) (-(1 / 2)) 0 x
      = (9 / 4 : ℝ) • (star x * x) - (3 / 2 : ℝ) • (star x * x * (star x * x))
        + (1 / 4 : ℝ) • (star x * x * (star x * x) * (star x * x)) := by
  simp only [nsStarStep, zero_smul, add_zero]
  simp only [star_add, star_real_smul, star_mul, star_star]
  simp only [add_mul, mul_add, smul_mul_assoc, mul_smul_comm, mul_assoc]
  module

/-- **The factored complement.** `1 − F⋆F = c·((4:ℝ)•1 − g)·c` with `c = ½(1 − g)` — the operator
    counterpart of `1 − q(t) = ¼(1−t)²(4−t)`. Since `c` is selfadjoint and `(4:ℝ)•1 − g ≥ 0` on the unit
    ball, conjugation-positivity turns this identity into `F⋆F ≤ 1`. Purely algebraic. -/
theorem one_sub_star_nsStarStep_mul_self (x : A) :
    (1 : A) - star (nsStarStep (3 / 2) (-(1 / 2)) 0 x) * nsStarStep (3 / 2) (-(1 / 2)) 0 x
      = ((1 / 2 : ℝ) • ((1 : A) - star x * x)) * ((4 : ℝ) • (1 : A) - star x * x)
          * ((1 / 2 : ℝ) • ((1 : A) - star x * x)) := by
  simp only [nsStarStep, zero_smul, add_zero]
  simp only [star_add, star_real_smul, star_mul, star_star]
  simp only [add_mul, mul_add, sub_mul, mul_sub, smul_mul_assoc, mul_smul_comm, smul_smul,
    one_mul, mul_one, mul_assoc]
  module

section Order

variable [PartialOrder A] [StarOrderedRing A]

/-- **THE INVARIANT BALL (C56's disclosed gap, closed — no SVD).** In a unital C*-algebra with its canonical
    order, the classical Newton–Schulz step maps the closed unit ball into itself:
    `‖x‖ ≤ 1 → ‖nsStarStep (3/2) (−1/2) 0 x‖ ≤ 1`. Proof: `g = x⋆x ≥ 0` with `‖g‖ = ‖x‖² ≤ 1`, so
    `g ≤ (4:ℝ)•1` (norm↔order dictionary); `0 ≤ F⋆F` free; `F⋆F ≤ 1` by the factored complement +
    conjugation-positivity; `‖F⋆F‖ ≤ 1` by the order↔norm dictionary; `‖F‖² = ‖F⋆F‖` by the C*-identity. -/
theorem nsStarStep_ball_invariant (x : A) (hx : ‖x‖ ≤ 1) :
    ‖nsStarStep (3 / 2) (-(1 / 2)) 0 x‖ ≤ 1 := by
  have hg0 : (0 : A) ≤ star x * x := star_mul_self_nonneg x
  have hgn : ‖star x * x‖ ≤ 1 := by
    rw [CStarRing.norm_star_mul_self]
    exact mul_le_one₀ hx (norm_nonneg x) hx
  have hg4 : star x * x ≤ (4 : ℝ) • (1 : A) := by
    have h := (CStarAlgebra.norm_le_iff_le_algebraMap (star x * x)
      (by norm_num : (0 : ℝ) ≤ 4) hg0).mp (hgn.trans (by norm_num))
    rwa [Algebra.algebraMap_eq_smul_one] at h
  have hc_sa : star ((1 / 2 : ℝ) • ((1 : A) - star x * x))
      = (1 / 2 : ℝ) • ((1 : A) - star x * x) := by
    rw [star_real_smul, star_sub, star_one, (IsSelfAdjoint.star_mul_self x).star_eq]
  have hpos : (0 : A) ≤ (1 : A)
      - star (nsStarStep (3 / 2) (-(1 / 2)) 0 x) * nsStarStep (3 / 2) (-(1 / 2)) 0 x := by
    rw [one_sub_star_nsStarStep_mul_self x]
    have h := star_left_conjugate_nonneg (sub_nonneg.mpr hg4)
      ((1 / 2 : ℝ) • ((1 : A) - star x * x))
    rwa [hc_sa] at h
  have hFF0 : (0 : A)
      ≤ star (nsStarStep (3 / 2) (-(1 / 2)) 0 x) * nsStarStep (3 / 2) (-(1 / 2)) 0 x :=
    star_mul_self_nonneg _
  have hn : ‖star (nsStarStep (3 / 2) (-(1 / 2)) 0 x) * nsStarStep (3 / 2) (-(1 / 2)) 0 x‖ ≤ 1 :=
    (CStarAlgebra.norm_le_one_iff_of_nonneg _ hFF0).mpr (sub_nonneg.mp hpos)
  rw [CStarRing.norm_star_mul_self] at hn
  nlinarith [norm_nonneg (nsStarStep (3 / 2) (-(1 / 2)) 0 x)]

/-- The unit ball is invariant under EVERY iterate of the classical NS map — the TRUE radii stay at `1`
    (C50's crude `nsMagBound` recursion grows: `nsMagBound 1 = 2`; the spectral route shows the crude
    growth is not real). -/
theorem nsIter_ball_invariant (k : ℕ) (x : A) (hx : ‖x‖ ≤ 1) :
    ‖nsIter (3 / 2) (-(1 / 2)) 0 k x‖ ≤ 1 := by
  induction k with
  | zero => simpa using hx
  | succ n ih => rw [nsIter_succ]; exact nsStarStep_ball_invariant _ ih

/-- **THE PAYOFF: the `k`-fold classical NS map is `3^k`-Lipschitz on the unit ball.** Both iterates stay
    in the invariant ball (`nsIter_ball_invariant`), and each step is `3`-Lipschitz there (C56's
    `nsStarStep_lipschitz_unit`). This realizes C50's uniform bound with `Lu = 3` WITHOUT the growing crude
    radii — discharging C53's `hLu`-shaped hypothesis on the unit ball. -/
theorem nsIter_lipschitz_ball (k : ℕ) (x y : A) (hx : ‖x‖ ≤ 1) (hy : ‖y‖ ≤ 1) :
    ‖nsIter (3 / 2) (-(1 / 2)) 0 k x - nsIter (3 / 2) (-(1 / 2)) 0 k y‖
      ≤ 3 ^ k * ‖x - y‖ := by
  induction k with
  | zero => simp
  | succ n ih =>
      rw [nsIter_succ, nsIter_succ]
      calc ‖nsStarStep (3 / 2) (-(1 / 2)) 0 (nsIter (3 / 2) (-(1 / 2)) 0 n x)
              - nsStarStep (3 / 2) (-(1 / 2)) 0 (nsIter (3 / 2) (-(1 / 2)) 0 n y)‖
          ≤ 3 * ‖nsIter (3 / 2) (-(1 / 2)) 0 n x - nsIter (3 / 2) (-(1 / 2)) 0 n y‖ :=
            nsStarStep_lipschitz_unit _ _ (nsIter_ball_invariant n x hx)
              (nsIter_ball_invariant n y hy)
        _ ≤ 3 * (3 ^ n * ‖x - y‖) :=
            mul_le_mul_of_nonneg_left ih (by norm_num)
        _ = 3 ^ (n + 1) * ‖x - y‖ := by ring

end Order

end CStar

-- Non-vacuity: `ℂ` (a unital C*-algebra with its canonical order, via `ComplexOrder`) instantiates the
-- ball invariance out of the box.
open scoped ComplexOrder in
example (x : ℂ) (hx : ‖x‖ ≤ 1) : ‖nsStarStep (3 / 2) (-(1 / 2)) 0 x‖ ≤ 1 :=
  nsStarStep_ball_invariant x hx

end Puffer.RL.NewtonSchulzBall
