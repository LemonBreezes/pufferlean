/-
# The gradient-map Lipschitz `G`: from per-coordinate budgets to a Lipschitz gradient MAP (sup-metric)

C53 (`MuonStepLipschitz`) takes the gradient's Lipschitz as the abstract hypothesis `hgradLip : ∀ x y,
‖grad x − grad y‖ ≤ G·‖x−y‖` (operator norm, on a normed ∗-ring). The repo ALREADY has the per-coordinate
gradient-Lipschitz for the concrete PPO objective — C26/C28's `ppoObjE_gradlip_Gdelta` gives
`|derivR (ppoObjE …) σ k − derivR (ppoObjE …) σ' k| ≤ Gtot·δ` whenever `∀ i, |σ i − σ' i| ≤ δ` (per coordinate
`k`, sup-metric, on the region). This module packages that into the statement the step-map machinery consumes:
the gradient as a MAP `σ ↦ ∇e(σ)` is `G`-Lipschitz in the SUP metric.

* `gradMap e σ := fun k => derivR e σ k` — the full gradient vector at `σ`.
* `gradMap_sup_lipschitz` — the (near-definitional, but load-bearing) repackaging: a per-coordinate regional
  gradient-Lipschitz bound makes `gradMap e` a `G`-Lipschitz MAP in the sup metric on the region. This is exactly
  the `hG` shape C27/C32's `gradAscentE_sup_lipschitz`/`wdAscentE_sup_lipschitz` and C58's `hgLip` consume.
* `gradMapFin`/`gradMapFin_sup_lipschitz`/`gradMapFin_lipschitzWith` — the finite-dimensional restriction
  `finRestrict ∘ gradMap e ∘ natExtend : (Fin d → ℝ) → (Fin d → ℝ)` is sup-Lipschitz, hence a Mathlib
  `LipschitzWith ⟨G,_⟩` via C55's bridge — under a GLOBAL per-coordinate hypothesis `hG` (the shape C58's
  weight-decay/Banach chain takes globally; C28 discharges it REGIONALLY — see the scope note).
* `ppoObjE_gradMap_lipschitz` — the CONCRETE capstone: for the actual PPO objective `ppoObjE`, under C28's
  regional slate (params in `[−R,R]`, clip-interior, entropy budgets at BOTH points — the same honest conditions
  as C28/C38), the gradient map is `Gtot`-sup-Lipschitz with C26/C28's CONCRETE constant `Gtot`. This supplies
  the `G` ingredient of every sup-metric step map built from this gradient (plain ascent `1+|lr|·G` (C27),
  weight decay `|1−wd|+|lr|·G` (C32/C58), and the sup-metric Muon shape), feeding C42's
  `whole_run_sup_interval_of_step` accumulation at trajectory point pairs.

**Scope (honestly disclosed).** This establishes the gradient-map Lipschitz `G` in the SUP-METRIC picture —
concrete for the PPO objective (`G = Gtot`, regional: the region/clip-interior/entropy conditions are C28's honest
slate, the same one C35/C38 carry). It feeds the sup-metric whole-run machinery directly (C42's abstract
accumulation, C27/C32's step maps, C58's `hgLip`). The OPERATOR-NORM version that C53's matrix-Muon `hgradLip`
uses is NOT delivered here: the repo's `derivR` gradient lives on the `Nat → ℝ` parameter vector (sup metric),
while C53's `grad : R → R` lives on the matrix ∗-ring (L2 operator norm); identifying the two needs the
parameter-vector ↔ matrix reshaping together with a sup↔operator-norm comparison (dimension factors, e.g.
`‖·‖_op ≤ √(r·c)·max-entry`). `MatrixEmbed` provides the entrywise `Mat`↔`Matrix` embedding but not that norm
comparison — the named residual. The `gradMapFin` `LipschitzWith` form takes a GLOBAL `hG` (as C58 does); for the
concrete `ppoObjE` the honest statement is the REGIONAL capstone (a global `LipschitzWith` would require the
region conditions to hold everywhere, which they do not).
-/
import Puffer.RL.WholeRunFromC26
import Puffer.RL.BanachConcrete
import Puffer.RL.WdNatFinBridge
open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)
open Puffer.RL.WholeRunFromC26 (ppoObjE Gtot ppoObjE_gradlip_Gdelta)
open Puffer.RL.SoftmaxExpr (logSoftmaxE)
open Puffer.RL.SurrogateExpr (ratioE)
open Puffer.RL.BanachConcrete (sup_lipschitz_to_lipschitzWith)
open Puffer.RL.WdNatFinBridge (finRestrict natExtend)
open scoped NNReal

namespace Puffer.RL.GradOpnormLip

/-- **The gradient as a map**: `σ ↦ ∇e(σ)`, the full gradient vector `fun k => derivR e σ k` at the parameter
    point `σ`. The object whose Lipschitz constant the step-map machinery (C27/C32/C42/C53/C58) consumes. -/
noncomputable def gradMap (e : Expr) (σ : Nat → ℝ) : Nat → ℝ := fun k => derivR e σ k

/-- **Per-coordinate ⟹ map (sup-metric, regional).** If every coordinate of the gradient difference is bounded
    by `G·δ` on a region (C26/C28's per-coordinate regional shape, with the region an arbitrary predicate), then
    the gradient MAP is `G`-Lipschitz in the sup metric on that region — exactly the `hG` shape
    `gradAscentE_sup_lipschitz` (C27), `wdAscentE_sup_lipschitz` (C32), and C58's `hgLip` consume. Near-definitional;
    the value is stating the MAP property once and wiring it. -/
theorem gradMap_sup_lipschitz (e : Expr) (region : (Nat → ℝ) → Prop) (G : ℝ)
    (hG : ∀ (σ σ' : Nat → ℝ) (δ : ℝ) (k : Nat), region σ → region σ' →
      (∀ i, |σ i - σ' i| ≤ δ) → |derivR e σ k - derivR e σ' k| ≤ G * δ) :
    ∀ (σ σ' : Nat → ℝ) (δ : ℝ), region σ → region σ' → (∀ i, |σ i - σ' i| ≤ δ) →
      ∀ k, |gradMap e σ k - gradMap e σ' k| ≤ G * δ :=
  fun σ σ' δ hσ hσ' hδ k => hG σ σ' δ k hσ hσ' hδ

/-- The finite-dimensional gradient map: restrict the `Nat`-indexed gradient to the first `d` parameters
    (`finRestrict ∘ gradMap e ∘ natExtend`), a self-map of `Fin d → ℝ` — the space where the concrete Banach
    machinery (C55/C58) lives. -/
noncomputable def gradMapFin (e : Expr) (d : ℕ) (v : Fin d → ℝ) : Fin d → ℝ :=
  finRestrict (gradMap e (natExtend v))

/-- **The finite gradient map is sup-Lipschitz** under a GLOBAL per-coordinate gradient-Lipschitz `hG` (the shape
    C58's `hgLip` takes; C28 discharges it regionally for `ppoObjE`). Mirrors C58's `wdFin_lipschitz`: the
    zero-padded extensions are `δ`-close where the originals are and `0`-close beyond `d`. -/
theorem gradMapFin_sup_lipschitz (e : Expr) (d : ℕ) (G : ℝ)
    (hG : ∀ (σ σ' : Nat → ℝ) (δ : ℝ), (∀ i, |σ i - σ' i| ≤ δ) →
        ∀ k, |derivR e σ k - derivR e σ' k| ≤ G * δ)
    (v v' : Fin d → ℝ) (δ : ℝ) (hδ : ∀ i, |v i - v' i| ≤ δ) :
    ∀ i, |gradMapFin e d v i - gradMapFin e d v' i| ≤ G * δ := by
  rcases Nat.eq_zero_or_pos d with hd0 | hdpos
  · subst hd0; exact fun i => i.elim0
  · have hδ0 : 0 ≤ δ := le_trans (abs_nonneg _) (hδ ⟨0, hdpos⟩)
    have hclose : ∀ j, |natExtend v j - natExtend v' j| ≤ δ := by
      intro j
      by_cases h : j < d
      · simp only [natExtend, dif_pos h]; exact hδ ⟨j, h⟩
      · simp only [natExtend, dif_neg h]; simpa using hδ0
    have key := hG (natExtend v) (natExtend v') δ hclose
    intro i
    simpa only [gradMapFin, gradMap, finRestrict] using key i.val

/-- **The finite gradient map is Mathlib-`LipschitzWith`** on `Fin d → ℝ` (sup/Pi metric) — C55's
    `sup_lipschitz_to_lipschitzWith` bridge applied to `gradMapFin_sup_lipschitz`. With `G < 1` this would make
    the gradient map itself `ContractingWith` (not needed here; recorded for the Banach-side consumers). -/
theorem gradMapFin_lipschitzWith (e : Expr) (d : ℕ) (G : ℝ) (hG0 : 0 ≤ G)
    (hG : ∀ (σ σ' : Nat → ℝ) (δ : ℝ), (∀ i, |σ i - σ' i| ≤ δ) →
        ∀ k, |derivR e σ k - derivR e σ' k| ≤ G * δ) :
    LipschitzWith ⟨G, hG0⟩ (gradMapFin e d) :=
  sup_lipschitz_to_lipschitzWith (gradMapFin e d) G hG0 (gradMapFin_sup_lipschitz e d G hG)

/-- **CAPSTONE: the concrete PPO objective's gradient map is `Gtot`-sup-Lipschitz on the region.** For the actual
    `ppoObjE` under C28's regional slate — params in `[−R,R]`, ratio in the clip interior, and the entropy
    value/derivative budgets at BOTH points (the derivative budgets now `∀ k`, since the map statement covers every
    coordinate) — the gradient map moves by at most `Gtot·δ` in the sup metric when the parameters move by `δ`:
    `∀ k, |gradMap (ppoObjE …) σ k − gradMap (ppoObjE …) σ' k| ≤ Gtot·δ`, with C26/C28's CONCRETE constant `Gtot`.
    Discharged coordinate-wise by C28's `ppoObjE_gradlip_Gdelta`. This is the concrete `G` for every sup-metric
    step map built from this gradient (C27 ascent, C32/C58 weight decay), in exactly the `hG`/`hgLip` shape those
    consume at trajectory point pairs. -/
theorem ppoObjE_gradMap_lipschitz (chosen e V : Expr) (es logps : List Expr)
    (oldLogp g lo hi cv ce ret : Float)
    (hch : Smooth chosen) (he : Smooth e) (hes : ∀ lp ∈ es, Smooth lp) (hV : Smooth V)
    (σ σ' : Nat → ℝ) (R δ : ℝ)
    (hσ : ∀ i, |σ i| ≤ R) (hσ' : ∀ i, |σ' i| ≤ R) (hδ : ∀ i, |σ i - σ' i| ≤ δ) (hR : 0 ≤ R)
    (hloσ : toReal lo < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ)
    (hhiσ : evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ < toReal hi)
    (hloσ' : toReal lo < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ')
    (hhiσ' : evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ' < toReal hi)
    (Ment Lvent Dment Dlent : ℝ)
    (hMσ : ∀ lp ∈ logps, |evalR lp σ| ≤ Ment) (hMσ' : ∀ lp ∈ logps, |evalR lp σ'| ≤ Ment)
    (hLvE : ∀ lp ∈ logps, |evalR lp σ - evalR lp σ'| ≤ Lvent * δ)
    (hDmEσ : ∀ k, ∀ lp ∈ logps, |derivR lp σ k| ≤ Dment)
    (hDmEσ' : ∀ k, ∀ lp ∈ logps, |derivR lp σ' k| ≤ Dment)
    (hDlE : ∀ k, ∀ lp ∈ logps, |derivR lp σ k - derivR lp σ' k| ≤ Dlent * δ)
    (hDmEn : 0 ≤ Dment) :
    ∀ k, |gradMap (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) σ k
        - gradMap (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) σ' k|
      ≤ Gtot chosen e V es logps oldLogp g cv ce ret R Ment Lvent Dment Dlent * δ :=
  fun k => ppoObjE_gradlip_Gdelta chosen e V es logps oldLogp g lo hi cv ce ret
    hch he hes hV σ σ' R δ k hσ hσ' hδ hR hloσ hhiσ hloσ' hhiσ'
    Ment Lvent Dment Dlent hMσ hMσ' hLvE (hDmEσ k) (hDmEσ' k) (hDlE k) hDmEn

end Puffer.RL.GradOpnormLip
