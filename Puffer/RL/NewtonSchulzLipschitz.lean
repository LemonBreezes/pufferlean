/-
# The scalar Newton–Schulz map is Lipschitz: a concrete constant for C42's open `L`

C42 (`MuonAscentBridge`) proved the whole-run geometric error interval holds for ANY `L`-Lipschitz step map, but
left Muon's OWN step-Lipschitz `L` unbounded — the genuinely-open ingredient of a real Muon whole-run interval.
Muon's step orthogonalizes the momentum via Newton–Schulz, which acts on the singular values through the SCALAR
quintic `nsScalar a b c σ = a·σ + b·σ³ + c·σ⁵` (`Puffer/Optim/Muon.lean`). This module bounds that scalar map's
Lipschitz constant — the tractable core of Muon's step sensitivity.

* `nsScalar_lipschitz` — on the bounded singular-value range `|σ|, |σ'| ≤ M`,
  `|nsScalar a b c σ − nsScalar a b c σ'| ≤ (|a| + 3·|b|·M² + 5·|c|·M⁴)·|σ − σ'|`. Proved by ALGEBRAIC FACTORING
  (no mean-value theorem): `nsScalar σ − nsScalar σ' = (σ − σ')·(a + b·(σ²+σσ'+σ'²) + c·(σ⁴+σ³σ'+σ²σ'²+σσ'³+σ'⁴))`,
  then the bracket is bounded by `|a| + 3|b|M² + 5|c|M⁴` via the triangle inequality on its 3- and 5-term factors
  (each monomial `≤ M²` resp. `M⁴` on `|σ|,|σ'| ≤ M`). The constant `L_ns = |a| + 3|b|M² + 5|c|M⁴` is an upper bound
  on the sup of `|nsScalar'(σ)| = |a + 3bσ² + 5cσ⁴|` over `|σ| ≤ M` (the triangle-majorized derivative bound —
  tight when the terms align in sign, e.g. loosely `3/2 + (3/2)M² ≥ 3/2` in the classical case) — obtained without
  differentiating.
* `nsScalar_lipschitz_delta` — the same in C42's `hlip` shape: `|σ − σ'| ≤ δ ⟹ |nsScalar σ − nsScalar σ'| ≤ L_ns·δ`
  (using `L_ns ≥ 0`). This is precisely the per-coordinate Lipschitz hypothesis C42's accumulation consumes.

**Scope (honestly disclosed).** This bounds the SCALAR Newton–Schulz map's Lipschitz constant
`L_ns = |a| + 3|b|M² + 5|c|M⁴` on the singular-value range `|σ| ≤ M` — a genuine, concrete ingredient of Muon's
step-Lipschitz `L` (C42's open constant). It does NOT establish the full MATRIX Muon-step operator-norm Lipschitz:
Newton–Schulz on a matrix is a nonlinear matrix polynomial `X ↦ X·(a·I + b·XᵀX + c·(XᵀX)²)` (or its symmetric form),
and the singular-value map's Lipschitz constant does NOT lift trivially to the matrix map — the operator-norm
sensitivity of the matrix polynomial (the interaction of the singular-VALUE map with the rotation of singular
VECTORS under perturbation) is the remaining hard piece, requiring SVD-continuity / matrix-perturbation analysis.
That matrix lift is left open here; this file supplies the scalar core exactly. `a, b, c` are the Newton–Schulz
quintic coefficients (`nsClassical = nsScalar (3/2) (−1/2) 0` per `Puffer/Optim/Muon.lean`), and `M` is a bound on
the singular-value range (from the Muon Frobenius normalization, `‖X‖_F ≤ 1`-style, giving `M` near 1).
-/
import Puffer.Optim.Muon
open Puffer.Optim.Muon (nsScalar)

namespace Puffer.RL.NewtonSchulzLipschitz

/-- **The scalar Newton–Schulz map is `L_ns`-Lipschitz on `|σ| ≤ M`.** With `L_ns = |a| + 3·|b|·M² + 5·|c|·M⁴`,
    `|nsScalar a b c σ − nsScalar a b c σ'| ≤ L_ns·|σ − σ'|`. Proof: factor `nsScalar σ − nsScalar σ' = (σ − σ')·B`
    with `B = a + b·(σ²+σσ'+σ'²) + c·(σ⁴+σ³σ'+σ²σ'²+σσ'³+σ'⁴)` (`ring`), and bound `|B| ≤ L_ns` by the triangle
    inequality on the 3-term (`≤ 3M²`) and 5-term (`≤ 5M⁴`) symmetric factors, each monomial bounded by `M²`/`M⁴`
    on `|σ|,|σ'| ≤ M`. -/
theorem nsScalar_lipschitz (a b c M σ σ' : ℝ) (hM : 0 ≤ M)
    (hσ : |σ| ≤ M) (hσ' : |σ'| ≤ M) :
    |nsScalar a b c σ - nsScalar a b c σ'|
      ≤ (|a| + 3 * |b| * M ^ 2 + 5 * |c| * M ^ 4) * |σ - σ'| := by
  -- algebraic factoring of the difference
  have hfac : nsScalar a b c σ - nsScalar a b c σ'
      = (σ - σ') * (a + b * (σ ^ 2 + σ * σ' + σ' ^ 2)
          + c * (σ ^ 4 + σ ^ 3 * σ' + σ ^ 2 * σ' ^ 2 + σ * σ' ^ 3 + σ' ^ 4)) := by
    simp only [nsScalar]; ring
  -- the 3-term symmetric factor is `≤ 3M²`
  have n1 : |σ ^ 2| ≤ M ^ 2 := by rw [abs_pow]; gcongr
  have n2 : |σ * σ'| ≤ M ^ 2 := by
    rw [abs_mul]
    calc |σ| * |σ'| ≤ M * M := by gcongr
      _ = M ^ 2 := by ring
  have n3 : |σ' ^ 2| ≤ M ^ 2 := by rw [abs_pow]; gcongr
  have hQ3 : |σ ^ 2 + σ * σ' + σ' ^ 2| ≤ 3 * M ^ 2 := by
    have t1 : |σ ^ 2 + σ * σ' + σ' ^ 2| ≤ |σ ^ 2 + σ * σ'| + |σ' ^ 2| := abs_add_le _ _
    have t2 : |σ ^ 2 + σ * σ'| ≤ |σ ^ 2| + |σ * σ'| := abs_add_le _ _
    linarith [t1, t2, n1, n2, n3]
  -- the 5-term symmetric factor is `≤ 5M⁴`
  have m1 : |σ ^ 4| ≤ M ^ 4 := by rw [abs_pow]; gcongr
  have m2 : |σ ^ 3 * σ'| ≤ M ^ 4 := by
    rw [abs_mul, abs_pow]
    calc |σ| ^ 3 * |σ'| ≤ M ^ 3 * M := by gcongr
      _ = M ^ 4 := by ring
  have m3 : |σ ^ 2 * σ' ^ 2| ≤ M ^ 4 := by
    rw [abs_mul, abs_pow, abs_pow]
    calc |σ| ^ 2 * |σ'| ^ 2 ≤ M ^ 2 * M ^ 2 := by gcongr
      _ = M ^ 4 := by ring
  have m4 : |σ * σ' ^ 3| ≤ M ^ 4 := by
    rw [abs_mul, abs_pow]
    calc |σ| * |σ'| ^ 3 ≤ M * M ^ 3 := by gcongr
      _ = M ^ 4 := by ring
  have m5 : |σ' ^ 4| ≤ M ^ 4 := by rw [abs_pow]; gcongr
  have hQ5 : |σ ^ 4 + σ ^ 3 * σ' + σ ^ 2 * σ' ^ 2 + σ * σ' ^ 3 + σ' ^ 4| ≤ 5 * M ^ 4 := by
    have t1 : |σ ^ 4 + σ ^ 3 * σ' + σ ^ 2 * σ' ^ 2 + σ * σ' ^ 3 + σ' ^ 4|
        ≤ |σ ^ 4 + σ ^ 3 * σ' + σ ^ 2 * σ' ^ 2 + σ * σ' ^ 3| + |σ' ^ 4| := abs_add_le _ _
    have t2 : |σ ^ 4 + σ ^ 3 * σ' + σ ^ 2 * σ' ^ 2 + σ * σ' ^ 3|
        ≤ |σ ^ 4 + σ ^ 3 * σ' + σ ^ 2 * σ' ^ 2| + |σ * σ' ^ 3| := abs_add_le _ _
    have t3 : |σ ^ 4 + σ ^ 3 * σ' + σ ^ 2 * σ' ^ 2|
        ≤ |σ ^ 4 + σ ^ 3 * σ'| + |σ ^ 2 * σ' ^ 2| := abs_add_le _ _
    have t4 : |σ ^ 4 + σ ^ 3 * σ'| ≤ |σ ^ 4| + |σ ^ 3 * σ'| := abs_add_le _ _
    linarith [t1, t2, t3, t4, m1, m2, m3, m4, m5]
  -- the bracket `B` is bounded by `L_ns`
  have hbr : |a + b * (σ ^ 2 + σ * σ' + σ' ^ 2)
        + c * (σ ^ 4 + σ ^ 3 * σ' + σ ^ 2 * σ' ^ 2 + σ * σ' ^ 3 + σ' ^ 4)|
      ≤ |a| + 3 * |b| * M ^ 2 + 5 * |c| * M ^ 4 := by
    have t1 : |a + b * (σ ^ 2 + σ * σ' + σ' ^ 2)
          + c * (σ ^ 4 + σ ^ 3 * σ' + σ ^ 2 * σ' ^ 2 + σ * σ' ^ 3 + σ' ^ 4)|
        ≤ |a + b * (σ ^ 2 + σ * σ' + σ' ^ 2)|
          + |c * (σ ^ 4 + σ ^ 3 * σ' + σ ^ 2 * σ' ^ 2 + σ * σ' ^ 3 + σ' ^ 4)| := abs_add_le _ _
    have t2 : |a + b * (σ ^ 2 + σ * σ' + σ' ^ 2)|
        ≤ |a| + |b * (σ ^ 2 + σ * σ' + σ' ^ 2)| := abs_add_le _ _
    have eb : |b * (σ ^ 2 + σ * σ' + σ' ^ 2)| = |b| * |σ ^ 2 + σ * σ' + σ' ^ 2| := abs_mul _ _
    have ec : |c * (σ ^ 4 + σ ^ 3 * σ' + σ ^ 2 * σ' ^ 2 + σ * σ' ^ 3 + σ' ^ 4)|
        = |c| * |σ ^ 4 + σ ^ 3 * σ' + σ ^ 2 * σ' ^ 2 + σ * σ' ^ 3 + σ' ^ 4| := abs_mul _ _
    have hb1 : |b| * |σ ^ 2 + σ * σ' + σ' ^ 2| ≤ 3 * |b| * M ^ 2 := by
      calc |b| * |σ ^ 2 + σ * σ' + σ' ^ 2| ≤ |b| * (3 * M ^ 2) :=
            mul_le_mul_of_nonneg_left hQ3 (abs_nonneg b)
        _ = 3 * |b| * M ^ 2 := by ring
    have hc1 : |c| * |σ ^ 4 + σ ^ 3 * σ' + σ ^ 2 * σ' ^ 2 + σ * σ' ^ 3 + σ' ^ 4| ≤ 5 * |c| * M ^ 4 := by
      calc |c| * |σ ^ 4 + σ ^ 3 * σ' + σ ^ 2 * σ' ^ 2 + σ * σ' ^ 3 + σ' ^ 4| ≤ |c| * (5 * M ^ 4) :=
            mul_le_mul_of_nonneg_left hQ5 (abs_nonneg c)
        _ = 5 * |c| * M ^ 4 := by ring
    linarith [t1, t2, eb.le, eb.ge, ec.le, ec.ge, hb1, hc1]
  -- combine: `|(σ−σ')·B| = |σ−σ'|·|B| ≤ |σ−σ'|·L_ns = L_ns·|σ−σ'|`
  rw [hfac, abs_mul]
  calc |σ - σ'| * |a + b * (σ ^ 2 + σ * σ' + σ' ^ 2)
          + c * (σ ^ 4 + σ ^ 3 * σ' + σ ^ 2 * σ' ^ 2 + σ * σ' ^ 3 + σ' ^ 4)|
      ≤ |σ - σ'| * (|a| + 3 * |b| * M ^ 2 + 5 * |c| * M ^ 4) :=
        mul_le_mul_of_nonneg_left hbr (abs_nonneg _)
    _ = (|a| + 3 * |b| * M ^ 2 + 5 * |c| * M ^ 4) * |σ - σ'| := by ring

/-- **The scalar Newton–Schulz Lipschitz bound in C42's `hlip` shape.** From `|σ − σ'| ≤ δ`,
    `|nsScalar a b c σ − nsScalar a b c σ'| ≤ (|a| + 3·|b|·M² + 5·|c|·M⁴)·δ` — the per-coordinate Lipschitz
    hypothesis `whole_run_sup_interval_of_step`/`muon_whole_run_sup_interval` (C42) consumes, discharged for the
    scalar singular-value map with the concrete constant `L_ns`. Uses `nsScalar_lipschitz` and `L_ns ≥ 0`. -/
theorem nsScalar_lipschitz_delta (a b c M σ σ' δ : ℝ) (hM : 0 ≤ M)
    (hσ : |σ| ≤ M) (hσ' : |σ'| ≤ M) (hδ : |σ - σ'| ≤ δ) :
    |nsScalar a b c σ - nsScalar a b c σ'|
      ≤ (|a| + 3 * |b| * M ^ 2 + 5 * |c| * M ^ 4) * δ :=
  (nsScalar_lipschitz a b c M σ σ' hM hσ hσ').trans
    (mul_le_mul_of_nonneg_left hδ (by positivity))

end Puffer.RL.NewtonSchulzLipschitz
