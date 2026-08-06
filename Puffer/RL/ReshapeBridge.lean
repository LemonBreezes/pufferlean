/-
# The parameter-vector ↔ matrix reshaping: closing C63's glue

C63 (`SupOpnormBridge`) delivered the sup↔operator-norm Lipschitz transfer for MATRIX-valued maps and disclosed
one remaining piece: the parameter-vector ↔ matrix RESHAPING (`Fin d → ℝ` ↔ `Matrix (Fin m) (Fin n) ℝ` with
`d = m·n`), needed so that C60 (`GradOpnormLip`)'s sup-metric gradient Lipschitz — which lives on parameter
VECTORS — can flow through C63's bridge into the operator-norm `hgradLip` shape C53 (`MuonStepLipschitz`)
consumes. This module supplies that glue:

* `vecToMatrix`/`matrixToVec` — the reshaping along Mathlib's index bijection
  `finProdFinEquiv : Fin m × Fin n ≃ Fin (m * n)`, with both round-trips proved.
* `reshape_sup_close` — entrywise-sup closeness is INVARIANT under the reshape (both directions; pure
  re-indexing along the bijection, no constants).
* `matrixized F := vecToMatrix ∘ F ∘ matrixToVec` — the matrix-valued map induced by a vector map.
* `matrixized_sup_lipschitz` — the induced map inherits the entrywise-sup Lipschitz constant UNCHANGED.
* `matrixized_opnorm_lipschitz` — composed with C63's bridge: a sup-metric `G`-Lipschitz VECTOR map induces an
  operator-norm `(√(m'·n')·G)`-Lipschitz MATRIX map — verbatim the `hlip`/`hgradLip` shape C42's
  `muon_whole_run_opnorm_interval` and C53's `muonStep_lipschitz` consume (square `m'=m, n'=n` gives the
  endomorphism case C53 needs).
* `gradMapFin_matrixized_opnorm_lipschitz` — the named end-to-end instance: C60's finite gradient map
  `gradMapFin e (m·n)`, matrixized, is operator-norm Lipschitz with constant `√(m·n)·G`.

**Scope (honestly disclosed).** This closes the C60→C63→C53 chain AT THE SHAPE LEVEL: the repo's sup-metric
gradient Lipschitz now reaches the operator-norm `hgradLip` form, with the dimension factor `√(m'·n')` (C63's,
real and unavoidable) paid once. The REGIONAL caveat persists exactly as C60 disclosed: the `gradMapFin`
instance below takes C60's GLOBAL per-coordinate hypothesis `hG` (the shape C58's Banach chain uses); for the
concrete `ppoObjE` the honest discharge is REGIONAL (C28's region/clip-interior/entropy slate — a global opnorm
`hgradLip` would need those conditions everywhere, which they do not hold). The reshape itself is exact — sup
closeness transports with NO constant — so the only inflation in the chain is C63's `√(m'·n')`.
-/
import Puffer.RL.SupOpnormBridge
import Puffer.RL.GradOpnormLip
open scoped Matrix Matrix.Norms.L2Operator
open Matrix
open Puffer.FloatR.ADR
open Puffer.RL.SupOpnormBridge (opnorm_hlip_of_sup_lipschitz)
open Puffer.RL.GradOpnormLip (gradMapFin gradMapFin_sup_lipschitz)

namespace Puffer.RL.ReshapeBridge

/-- Reshape a `Fin (m*n)`-indexed parameter vector into an `m × n` real matrix, along Mathlib's index
    bijection `finProdFinEquiv : Fin m × Fin n ≃ Fin (m * n)`. -/
def vecToMatrix {m n : Nat} (v : Fin (m * n) → ℝ) : Matrix (Fin m) (Fin n) ℝ :=
  fun i j => v (finProdFinEquiv (i, j))

/-- Flatten an `m × n` real matrix into a `Fin (m*n)`-indexed parameter vector (the inverse reshape). -/
def matrixToVec {m n : Nat} (M : Matrix (Fin m) (Fin n) ℝ) : Fin (m * n) → ℝ :=
  fun k => M (finProdFinEquiv.symm k).1 (finProdFinEquiv.symm k).2

/-- Round-trip: reshaping a flattened matrix recovers the matrix. -/
theorem vecToMatrix_matrixToVec {m n : Nat} (M : Matrix (Fin m) (Fin n) ℝ) :
    vecToMatrix (matrixToVec M) = M := by
  funext i j
  simp only [vecToMatrix, matrixToVec, Equiv.symm_apply_apply]

/-- Round-trip: flattening a reshaped vector recovers the vector. -/
theorem matrixToVec_vecToMatrix {m n : Nat} (v : Fin (m * n) → ℝ) :
    matrixToVec (vecToMatrix v) = v := by
  funext k
  simp only [matrixToVec, vecToMatrix, Prod.mk.eta, Equiv.apply_symm_apply]

/-- **Sup-closeness is invariant under the reshape** (pure re-indexing along the bijection — no constants):
    two parameter vectors are entrywise `δ`-close iff their reshaped matrices are. -/
theorem reshape_sup_close {m n : Nat} (v v' : Fin (m * n) → ℝ) (δ : ℝ) :
    (∀ k, |v k - v' k| ≤ δ) ↔ (∀ i j, |vecToMatrix v i j - vecToMatrix v' i j| ≤ δ) := by
  constructor
  · intro h i j
    exact h (finProdFinEquiv (i, j))
  · intro h k
    have hk := h (finProdFinEquiv.symm k).1 (finProdFinEquiv.symm k).2
    simpa only [vecToMatrix, Prod.mk.eta, Equiv.apply_symm_apply] using hk

/-- **The matrix-valued map induced by a vector map**: conjugate `F` by the reshape,
    `matrixized F := vecToMatrix ∘ F ∘ matrixToVec`. For C60's gradient map (`d = m·n`, an endomorphism) this
    is the gradient as a map on weight MATRICES. -/
def matrixized {m n m' n' : Nat} (F : (Fin (m * n) → ℝ) → (Fin (m' * n') → ℝ))
    (X : Matrix (Fin m) (Fin n) ℝ) : Matrix (Fin m') (Fin n') ℝ :=
  vecToMatrix (F (matrixToVec X))

/-- **The induced map inherits the entrywise-sup Lipschitz constant unchanged**: if `F` is sup-`G`-Lipschitz on
    vectors, `matrixized F` is entrywise-sup-`G`-Lipschitz on matrices (the reshape transports closeness exactly —
    input side by flattening, output side by reshaping). -/
theorem matrixized_sup_lipschitz {m n m' n' : Nat}
    (F : (Fin (m * n) → ℝ) → (Fin (m' * n') → ℝ)) (G : ℝ)
    (hF : ∀ (v v' : Fin (m * n) → ℝ) (δ : ℝ), (∀ k, |v k - v' k| ≤ δ) →
      ∀ k', |F v k' - F v' k'| ≤ G * δ)
    (X Y : Matrix (Fin m) (Fin n) ℝ) (δ : ℝ) (hδ : ∀ i j, |X i j - Y i j| ≤ δ) :
    ∀ i' j', |matrixized F X i' j' - matrixized F Y i' j'| ≤ G * δ := by
  have hin : ∀ k, |matrixToVec X k - matrixToVec Y k| ≤ δ := by
    intro k
    simp only [matrixToVec]
    exact hδ _ _
  have hout := hF (matrixToVec X) (matrixToVec Y) δ hin
  intro i' j'
  simpa only [matrixized, vecToMatrix] using hout (finProdFinEquiv (i', j'))

/-- **CAPSTONE: a sup-Lipschitz vector map induces an operator-norm Lipschitz matrix map.** Composing the
    reshape transport with C63's `opnorm_hlip_of_sup_lipschitz`: if `F` is sup-`G`-Lipschitz on parameter
    vectors, `matrixized F` is operator-norm `(√(m'·n')·G)`-Lipschitz —

        ∀ X Y, ‖matrixized F X − matrixized F Y‖ ≤ (√(m'·n')·G)·‖X − Y‖,

    verbatim the `hlip` shape of C42's `muon_whole_run_opnorm_interval` and (for square `m'=m, n'=n`, the
    endomorphism case) C53's `hgradLip`. The only inflation in the whole chain is C63's `√(m'·n')` (the reshape
    is exact). -/
theorem matrixized_opnorm_lipschitz {m n m' n' : Nat} [Nonempty (Fin n')]
    (F : (Fin (m * n) → ℝ) → (Fin (m' * n') → ℝ)) (G : ℝ) (hG : 0 ≤ G)
    (hF : ∀ (v v' : Fin (m * n) → ℝ) (δ : ℝ), (∀ k, |v k - v' k| ≤ δ) →
      ∀ k', |F v k' - F v' k'| ≤ G * δ) :
    ∀ X Y : Matrix (Fin m) (Fin n) ℝ,
      ‖matrixized F X - matrixized F Y‖ ≤ (Real.sqrt ((m' : ℝ) * n') * G) * ‖X - Y‖ :=
  opnorm_hlip_of_sup_lipschitz (matrixized F) G hG
    (fun X Y δ hδ => matrixized_sup_lipschitz F G hF X Y δ hδ)

/-- **The named end-to-end instance: C60's finite gradient map, matrixized, is opnorm-Lipschitz.** For an
    objective `e` whose per-coordinate gradient is globally `G`-Lipschitz in the sup metric (C60's `hG` shape —
    dischargeable REGIONALLY for the concrete `ppoObjE` by C28, as C60 disclosed), the induced map on `m × n`
    weight matrices satisfies

        ‖matrixized (gradMapFin e (m·n)) X − matrixized (gradMapFin e (m·n)) Y‖ ≤ (√(m·n)·G)·‖X − Y‖

    — the operator-norm `hgradLip` shape (an endomorphism, `m'=m, n'=n`) that C53's `muonStep_lipschitz`
    consumes, with the dimension factor explicit. This closes the C60→C63→C53 chain at the shape level. -/
theorem gradMapFin_matrixized_opnorm_lipschitz (e : Expr) {m n : Nat} [Nonempty (Fin n)]
    (G : ℝ) (hG0 : 0 ≤ G)
    (hG : ∀ (σ σ' : Nat → ℝ) (δ : ℝ), (∀ i, |σ i - σ' i| ≤ δ) →
        ∀ k, |derivR e σ k - derivR e σ' k| ≤ G * δ) :
    ∀ X Y : Matrix (Fin m) (Fin n) ℝ,
      ‖matrixized (gradMapFin e (m * n)) X - matrixized (gradMapFin e (m * n)) Y‖
        ≤ (Real.sqrt ((m : ℝ) * n) * G) * ‖X - Y‖ :=
  matrixized_opnorm_lipschitz (gradMapFin e (m * n)) G hG0
    (fun v v' δ hδ => gradMapFin_sup_lipschitz e (m * n) G hG v v' δ hδ)

end Puffer.RL.ReshapeBridge
