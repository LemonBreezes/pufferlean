/-
# The sup ↔ operator-norm bridge: dimension-factor comparison and the Lipschitz transfer

C60 (`GradOpnormLip`) delivered the gradient-map Lipschitz `G` in the SUP-METRIC picture (per-coordinate /
max-entry bounds, concrete via C26/C28's `Gtot`) and named as its residual the wiring to C53
(`MuonStepLipschitz`)'s OPERATOR-NORM `hgradLip : ∀ x y, ‖grad x − grad y‖ ≤ G·‖x−y‖`. This module supplies that
wiring for matrix-valued maps: the two norms compare with explicit DIMENSION factors, and a sup-entrywise
Lipschitz map is therefore operator-norm Lipschitz with a `√(m'·n')`-inflated constant.

The two comparisons (both directions REUSED from the repo's existing spectral machinery — not re-proved):

* `entry_le_opNorm` — `|M i j| ≤ ‖M‖` (each entry is `⟨eᵢ, M eⱼ⟩`, bounded by the operator norm; the repo's
  `NewtonSchulzNormTower.abs_matrix_entry_le_l2_opNorm`).
* `opNorm_le_sqrt_dim_mul` — `(∀ i j, |M i j| ≤ B) → ‖M‖ ≤ √(m·n)·B` (via the Frobenius/trace route; the repo's
  `NewtonSchulzFloat.l2_opNorm_le_of_entrywise`).

The NEW content is the transfer:

* `sup_lipschitz_to_opnorm_lipschitz` — a map `F` between (rectangular, real) matrix spaces that is
  ENTRYWISE-SUP Lipschitz (`(∀ i j, |X i j − Y i j| ≤ δ) → ∀ i' j', |F X i' j' − F Y i' j'| ≤ G·δ`) is
  OPERATOR-NORM Lipschitz with constant `√(m'·n')·G`: chain `entry ≤ opnorm` on the INPUT difference (feeding
  `δ := ‖X − Y‖`), then `opnorm ≤ √dim·entry` on the OUTPUT difference. The conclusion
  `‖F X − F Y‖ ≤ (√(m'·n')·G)·‖X − Y‖` is EXACTLY the `hlip` shape C53's `muonStep_lipschitz` / C42's
  `muon_whole_run_opnorm_interval` consume.
* `entrywise_close_of_opnorm_close` — the input-side direction alone (`‖X − Y‖ ≤ δ ⟹ entries δ-close`), the form
  that feeds a sup-Lipschitz hypothesis from an operator-norm bound.

**Scope (honestly disclosed).** The dimension factor `√(m'·n')` is REAL and unavoidable in general — the max-entry
and operator norms are equivalent only up to dimension (e.g. the all-ones matrix has max-entry `1` and operator
norm `√(m·n)`), so a sup-metric `G`-Lipschitz map is opnorm-Lipschitz only with the inflated constant. The bridge
is stated for MATRIX-valued maps; connecting C60's concrete `Gtot` (a statement about the `Nat → ℝ` parameter
vector) through this bridge to C53's matrix-shaped `hgradLip` additionally requires the parameter-vector ↔ matrix
RESHAPING (`Fin d → ℝ` ↔ `Matrix (Fin m') (Fin n') ℝ` with `d = m'·n'`, carrying the sup metric across the
reshape) — mechanical glue, not done here. The output column dimension must be nonempty (`[Nonempty (Fin n')]`,
inherited from the Frobenius/eigenvalue route of `l2_opNorm_le_of_entrywise`).
-/
import Puffer.RL.NewtonSchulzNormTower
open scoped Matrix Matrix.Norms.L2Operator
open Matrix
open Puffer.RL.NewtonSchulzNormTower (abs_matrix_entry_le_l2_opNorm)
open Puffer.RL.NewtonSchulzFloat (l2_opNorm_le_of_entrywise)

namespace Puffer.RL.SupOpnormBridge

/-- **Entry ≤ operator norm.** Every entry of a real matrix is bounded by its L2 operator norm — the max-entry
    (sup) norm is dominated by the operator norm, with NO dimension factor in this direction. Re-export of the
    repo's `abs_matrix_entry_le_l2_opNorm` (proved via `A *ᵥ eⱼ` and `PiLp.norm_apply_le`). -/
theorem entry_le_opNorm {m n : Nat} (M : Matrix (Fin m) (Fin n) ℝ) (i : Fin m) (j : Fin n) :
    |M i j| ≤ ‖M‖ :=
  abs_matrix_entry_le_l2_opNorm M i j

/-- **Operator norm ≤ √(dimension) · max-entry.** A uniform entry bound `B` gives `‖M‖ ≤ √(m·n)·B` — the reverse
    comparison, carrying the unavoidable `√(m·n)` dimension factor (tight up to constants: the all-ones matrix has
    max-entry `1` and operator norm `√(m·n)`). Re-export of the repo's `l2_opNorm_le_of_entrywise` (proved via the
    trace/eigenvalue Frobenius route). -/
theorem opNorm_le_sqrt_dim_mul {m n : Nat} [Nonempty (Fin n)] (M : Matrix (Fin m) (Fin n) ℝ)
    (B : ℝ) (hB : 0 ≤ B) (h : ∀ i j, |M i j| ≤ B) :
    ‖M‖ ≤ Real.sqrt ((m : ℝ) * n) * B :=
  l2_opNorm_le_of_entrywise M B hB h

/-- **Operator-norm closeness gives entrywise closeness** (no dimension factor this direction): if
    `‖X − Y‖ ≤ δ` then every entry pair is `δ`-close — the form that feeds a sup-metric Lipschitz hypothesis
    (C60's shape) from an operator-norm bound. -/
theorem entrywise_close_of_opnorm_close {m n : Nat} (X Y : Matrix (Fin m) (Fin n) ℝ) (δ : ℝ)
    (h : ‖X - Y‖ ≤ δ) : ∀ i j, |X i j - Y i j| ≤ δ := by
  intro i j
  have hentry := entry_le_opNorm (X - Y) i j
  rw [Matrix.sub_apply] at hentry
  exact hentry.trans h

/-- **THE BRIDGE: sup-entrywise Lipschitz ⟹ operator-norm Lipschitz, with the dimension factor.** For a map `F`
    between real matrix spaces that is entrywise-sup `G`-Lipschitz — the shape C60's `gradMap` machinery
    delivers — `F` is operator-norm Lipschitz with the `√(m'·n')`-inflated constant:

        ‖F X − F Y‖ ≤ (√(m'·n')·G)·‖X − Y‖.

    Chain: the INPUT difference's entries are `‖X − Y‖`-close (`entry_le_opNorm`, no factor), so the sup-Lipschitz
    hypothesis applies at `δ := ‖X − Y‖`, bounding every OUTPUT entry by `G·‖X − Y‖`; then `opNorm_le_sqrt_dim_mul`
    on the output difference pays the `√(m'·n')` factor once. The conclusion is exactly the `hlip` shape C53's
    `muonStep_lipschitz` and C42's `muon_whole_run_opnorm_interval` consume, with `L = √(m'·n')·G`. -/
theorem sup_lipschitz_to_opnorm_lipschitz {m n m' n' : Nat} [Nonempty (Fin n')]
    (F : Matrix (Fin m) (Fin n) ℝ → Matrix (Fin m') (Fin n') ℝ) (G : ℝ) (hG : 0 ≤ G)
    (hF : ∀ (X Y : Matrix (Fin m) (Fin n) ℝ) (δ : ℝ), (∀ i j, |X i j - Y i j| ≤ δ) →
      ∀ i' j', |F X i' j' - F Y i' j'| ≤ G * δ)
    (X Y : Matrix (Fin m) (Fin n) ℝ) :
    ‖F X - F Y‖ ≤ Real.sqrt ((m' : ℝ) * n') * G * ‖X - Y‖ := by
  -- input entries are ‖X − Y‖-close (entry ≤ opnorm, applied to the difference):
  have hin : ∀ i j, |X i j - Y i j| ≤ ‖X - Y‖ :=
    entrywise_close_of_opnorm_close X Y ‖X - Y‖ le_rfl
  -- every output entry is G·‖X − Y‖-bounded:
  have hout : ∀ i' j', |(F X - F Y) i' j'| ≤ G * ‖X - Y‖ := by
    intro i' j'
    rw [Matrix.sub_apply]
    exact hF X Y ‖X - Y‖ hin i' j'
  -- pay the output dimension factor once:
  have hop := opNorm_le_sqrt_dim_mul (F X - F Y) (G * ‖X - Y‖)
    (mul_nonneg hG (norm_nonneg _)) hout
  calc ‖F X - F Y‖ ≤ Real.sqrt ((m' : ℝ) * n') * (G * ‖X - Y‖) := hop
    _ = Real.sqrt ((m' : ℝ) * n') * G * ‖X - Y‖ := by ring

/-- **The C53/C42 `hlip`-shaped corollary.** The bridge restated with the Lipschitz constant
    `L := √(m'·n')·G` explicit, quantified over all pairs — verbatim the `hlip : ∀ x y, ‖F x − F y‖ ≤ L·‖x−y‖`
    hypothesis of C42's `muon_whole_run_opnorm_interval` (and the shape C53's gradient hypothesis takes). What
    remains to feed C60's concrete `Gtot` here is only the parameter-vector ↔ matrix reshaping (see the module
    scope note). -/
theorem opnorm_hlip_of_sup_lipschitz {m n m' n' : Nat} [Nonempty (Fin n')]
    (F : Matrix (Fin m) (Fin n) ℝ → Matrix (Fin m') (Fin n') ℝ) (G : ℝ) (hG : 0 ≤ G)
    (hF : ∀ (X Y : Matrix (Fin m) (Fin n) ℝ) (δ : ℝ), (∀ i j, |X i j - Y i j| ≤ δ) →
      ∀ i' j', |F X i' j' - F Y i' j'| ≤ G * δ) :
    ∀ X Y, ‖F X - F Y‖ ≤ (Real.sqrt ((m' : ℝ) * n') * G) * ‖X - Y‖ :=
  fun X Y => sup_lipschitz_to_opnorm_lipschitz F G hG hF X Y

end Puffer.RL.SupOpnormBridge
