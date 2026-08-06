/-
# Newton–Schulz norm-stability: the invariant unit ball and the uniform Lipschitz constant `Lu = 3`

C50 (`NewtonSchulzIterate`) / C53 (`MuonStepLipschitz`) bound the `k`-iteration NS Lipschitz by `Lu^k` under a
UNIFORM per-ball hypothesis `hLu : ∀ i, nsLconst a b c (nsRadius … i) ≤ Lu` — but left `Lu` (the NS iteration's
norm-stability) as an unestablished hypothesis. This module establishes it for the CLASSICAL Newton–Schulz map
`nsScalar (3/2) (−1/2) 0 σ = 1.5σ − 0.5σ³` (`Puffer.Optim.Muon.nsClassical`).

**The subtlety.** C50's `nsMagBound` (the crude SUBMULTIPLICATIVE operator-norm bound) GROWS: at `‖X‖ = 1` it gives
`|3/2|·1 + |1/2|·1³ = 2`, so C50's `nsRadius` is not `≤ 1`-stable. But the crude bound ignores the SINGULAR-VALUE
structure: one NS iteration acts on `X = UΣVᴴ` as `nsStarStep X = U·(nsScalar applied to Σ)·Vᴴ`, so the TRUE operator
norm is `‖nsStarStep X‖ = maxᵢ nsScalar(σᵢ)`, and `nsScalar` has an INVARIANT INTERVAL `[0,1]` that keeps the operator
norm `≤ 1`. On that invariant ball the per-ball Lipschitz constant is the UNIFORM `Lu = nsLconst (3/2)(−1/2) 0 1 = 3`.

**Scalar core (the singular-value picture — proved):**
* `nsScalar_classical_invariant` — `[0,1]` is invariant: `0 ≤ σ ≤ 1 ⟹ nsScalar (3/2)(−1/2) 0 σ ∈ [0,1]`.
* `nsScalar_classical_into_unit` — the basin `[0,√3]` maps INTO `[0,1]`: `0 ≤ σ`, `σ² ≤ 3 ⟹ nsScalar … σ ∈ [0,1]`
  (spectral-normalized gradients — Muon normalizes before NS — land in `[0,1]` after one step and stay). The upper
  bound `≤ 1` holds for ALL `σ ≥ 0` (`1 − nsScalar … σ = ½(σ−1)²(σ+2) ≥ 0`); the lower bound needs `σ² ≤ 3`.
* `nsScalar_classical_iterate_stable` — the whole scalar orbit stays in `[0,1]` (via `Muon.nsClassical_iterate_mem`).

**The uniform constant `Lu = 3` (the bridge to C50/C53):**
* `nsLconst_classical_le_three` — on any radius `r ∈ [0,1]`, C50's per-ball constant `nsLconst (3/2)(−1/2) 0 r ≤ 3`. So
  IF the NS radius stays `≤ 1` (the operator-norm invariance below), C50's `hLu` holds with `Lu = 3`.
* `nsStarStep_lipschitz_unit` — the OPERATOR-NORM Lipschitz on the unit ball is `3`: `‖x‖,‖y‖ ≤ 1 ⟹ ‖nsStarStep
  (3/2)(−1/2) 0 x − nsStarStep … y‖ ≤ 3·‖x−y‖` (C47's `nsStarStep_lipschitz` at `M = 1`, in the abstract normed ∗-ring
  matrices instantiate) — the concrete `Lu = 3` for C53's Muon-step Lipschitz on the invariant ball.

**Scope (honestly disclosed).** The SINGULAR-VALUE norm-stability (scalar invariant `[0,1]`, basin `[0,√3]`, orbit
stability) is fully proved, and the operator-norm Lipschitz constant `Lu = 3` on the unit ball is concrete (C47). What
remains for a self-contained abstract-`nsStarStep` proof of the INVARIANT OPERATOR-NORM BALL (`‖X‖ ≤ 1 ⟹ ‖nsStarStep
X‖ ≤ 1`, i.e. the NS radius genuinely stays `≤ 1`) is the singular-value characterization `‖nsStarStep X‖ = maxᵢ
nsScalar(σᵢ)` — the eigenvalue identity `λᵢ((nsStarStep X)ᴴ(nsStarStep X)) = nsScalar(σᵢ)²`, liftable via the repo's
`SpectralBridge.opNorm_le_of_gram_eigenvalue_bound` (Mathlib v4.28.0 has no SVD/polar-decomposition API, so this
polynomial-of-a-Hermitian-matrix eigenvalue step is intricate; `NewtonSchulzFull.newtonSchulz_opNorm_le` already
establishes the analogous stability for the concrete `Mat` representation). So `Lu = 3` is established in the
singular-value picture and as the operator-norm Lipschitz constant; the abstract-matrix invariant-ball lift is the
disclosed remaining step. Classical coefficients `(3/2, −1/2, 0)`; the `‖X‖ ≤ √3` initial condition is Muon's
pre-NS spectral normalization.
-/
import Puffer.RL.NewtonSchulzMatrixLipschitz
import Puffer.RL.NewtonSchulzIterate
import Puffer.RL.NewtonSchulzLipschitz
import Puffer.Optim.Muon
open Puffer.Optim.Muon (nsScalar nsClassical nsClassical_eq_nsScalar nsClassical_le_one nsClassical_ge_self
  nsClassical_iterate_mem)
open Puffer.RL.NewtonSchulzMatrixLipschitz (nsStarStep nsStarStep_lipschitz)
open Puffer.RL.NewtonSchulzIterate (nsLconst)

namespace Puffer.RL.NewtonSchulzStable

/-- **`[0,1]` is invariant under the classical Newton–Schulz map.** `0 ≤ σ ≤ 1 ⟹ nsScalar (3/2)(−1/2) 0 σ ∈ [0,1]`.
    Upper bound from `Muon.nsClassical_le_one`; lower bound from `Muon.nsClassical_ge_self` (`σ ≤ nsClassical σ`) and
    `σ ≥ 0`. This is the per-singular-value invariance that keeps the operator norm `≤ 1`. -/
theorem nsScalar_classical_invariant (σ : ℝ) (h0 : 0 ≤ σ) (h1 : σ ≤ 1) :
    0 ≤ nsScalar (3 / 2) (-(1 / 2)) 0 σ ∧ nsScalar (3 / 2) (-(1 / 2)) 0 σ ≤ 1 := by
  rw [← nsClassical_eq_nsScalar]
  exact ⟨le_trans h0 (nsClassical_ge_self σ h0 h1), nsClassical_le_one σ h0 h1⟩

/-- **The basin `[0,√3]` maps into `[0,1]`.** `0 ≤ σ`, `σ² ≤ 3 ⟹ nsScalar (3/2)(−1/2) 0 σ ∈ [0,1]`. The upper bound
    `≤ 1` holds for ALL `σ ≥ 0` (`1 − nsClassical σ = ½(σ−1)²(σ+2) ≥ 0`); the lower bound `≥ 0` uses `σ² ≤ 3` (via the
    factorization `nsClassical σ = σ·(3 − σ²)/2`). So a spectral-normalized gradient (singular values `≤ √3`) is driven
    into the invariant unit interval after one step. -/
theorem nsScalar_classical_into_unit (σ : ℝ) (h0 : 0 ≤ σ) (h3 : σ ^ 2 ≤ 3) :
    0 ≤ nsScalar (3 / 2) (-(1 / 2)) 0 σ ∧ nsScalar (3 / 2) (-(1 / 2)) 0 σ ≤ 1 := by
  rw [← nsClassical_eq_nsScalar]
  refine ⟨?_, ?_⟩
  · have hfac : nsClassical σ = σ * (3 - σ ^ 2) / 2 := by unfold nsClassical; ring
    rw [hfac]
    exact div_nonneg (mul_nonneg h0 (by linarith)) (by norm_num)
  · have he : 1 - nsClassical σ = (σ - 1) ^ 2 * (σ + 2) / 2 := by unfold nsClassical; ring
    nlinarith [mul_nonneg (sq_nonneg (σ - 1)) (by linarith : (0 : ℝ) ≤ σ + 2)]

/-- **The scalar orbit stays in `[0,1]`.** Every classical NS iterate of a value in `[0,1]` stays in `[0,1]`
    (`Muon.nsClassical_iterate_mem`) — so on the invariant unit interval the per-singular-value Lipschitz constant is
    uniformly `≤ 3` across ALL iterations (see `nsLconst_classical_le_three`). -/
theorem nsScalar_classical_iterate_stable (n : ℕ) (σ : ℝ) (h0 : 0 ≤ σ) (h1 : σ ≤ 1) :
    0 ≤ nsClassical^[n] σ ∧ nsClassical^[n] σ ≤ 1 :=
  nsClassical_iterate_mem n σ h0 h1

/-- **The uniform Lipschitz constant `Lu = 3` on the unit ball.** On any radius `r ∈ [0,1]`, C50's per-ball Lipschitz
    constant `nsLconst (3/2)(−1/2) 0 r = 3/2 + (3/2)r² ≤ 3`. So if the NS radius stays `≤ 1` (the operator-norm
    invariance the scalar core justifies), C50/C53's uniform hypothesis `hLu : ∀ i, nsLconst … (radius i) ≤ Lu` holds
    with the concrete `Lu = 3`. -/
theorem nsLconst_classical_le_three (r : ℝ) (hr0 : 0 ≤ r) (hr1 : r ≤ 1) :
    nsLconst (3 / 2) (-(1 / 2)) 0 r ≤ 3 := by
  unfold nsLconst
  rw [abs_of_pos (by norm_num : (0:ℝ) < 3 / 2), abs_of_neg (by norm_num : (-(1/2):ℝ) < 0), abs_zero]
  nlinarith [mul_nonneg hr0 hr0, hr1, hr0]

/-- **`nsLconst (3/2)(−1/2) 0 1 = 3`** — the exact uniform constant on the unit ball (C44's `nsScalar_lipschitz`
    constant at `M = 1`). -/
theorem nsLconst_classical_one : nsLconst (3 / 2) (-(1 / 2)) 0 1 = 3 := by
  unfold nsLconst; norm_num

section OpNorm
variable {R : Type*} [NormedRing R] [NormedAlgebra ℝ R] [StarRing R] [NormedStarGroup R]

/-- **The operator-norm Lipschitz constant `Lu = 3` on the unit ball.** On `‖x‖,‖y‖ ≤ 1`, one classical Newton–Schulz
    step is `3`-Lipschitz in the operator norm: `‖nsStarStep (3/2)(−1/2) 0 x − nsStarStep … y‖ ≤ 3·‖x−y‖`. Directly
    C47's `nsStarStep_lipschitz` at `M = 1` (constant `|3/2| + 3·|−1/2|·1 + 5·|0|·1 = 3`), in the abstract normed ∗-ring
    real matrices instantiate. This is the concrete `Lu` C53's Muon-step Lipschitz needs on the invariant unit ball. -/
theorem nsStarStep_lipschitz_unit (x y : R) (hx : ‖x‖ ≤ 1) (hy : ‖y‖ ≤ 1) :
    ‖nsStarStep (3 / 2) (-(1 / 2)) 0 x - nsStarStep (3 / 2) (-(1 / 2)) 0 y‖ ≤ 3 * ‖x - y‖ := by
  have h := nsStarStep_lipschitz (3 / 2) (-(1 / 2)) 0 1 x y (by norm_num) hx hy
  calc ‖nsStarStep (3 / 2) (-(1 / 2)) 0 x - nsStarStep (3 / 2) (-(1 / 2)) 0 y‖
      ≤ (|(3 : ℝ) / 2| + 3 * |(-(1 / 2) : ℝ)| * 1 ^ 2 + 5 * |(0 : ℝ)| * 1 ^ 4) * ‖x - y‖ := h
    _ = 3 * ‖x - y‖ := by norm_num

end OpNorm

end Puffer.RL.NewtonSchulzStable
