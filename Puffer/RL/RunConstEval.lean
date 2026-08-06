/-
# Runnable run-constant evaluators — completing C78's constant plumbing

C78 (`BudgetEval`) made the whole-run budget RECURSIONS runnable (`runBoundF`/`wdUniformBoundF`
with proven domination) but took the step CONSTANTS as `C ≤ toReal Cf`-style hypotheses — "the
caller's constant plumbing". This module supplies those dominating Float representatives:
computable mirrors of C64's `sgdStepC`/`muonStepC` (the latter threading C61's `nsScalarFBound`)
and C67's `wdStepRho`/`wdStepC`, each with a proven DOMINATION lemma, composed into ALL-BOOL
whole-run finiteness certificates.

**The unit design (the accounting, proved once per shape).** Every constant is a tree of
`(1+u64) * (x ⊙ y)` nodes (⊙ ∈ {·, +}) plus bare `(1+u64) * x` scalings. Each node's Float
mirror `slackF * (xF ⊙F yF)` performs exactly TWO roundings (the op and the slack multiply), and
C78's `slackF_key : (1+u64) ≤ toReal slackF · (1−u64)²` absorbs the node's ℝ-side `(1+u64)` plus
both — so ONE `slackF_key` per node, compositionally:

* `unit_mul_dominates` / `unit_add_dominates` — `(1+u64)·(x ⊙ y) ≤ toReal (slackF * (xF ⊙F yF))`
  from the arguments' domination (2 roundings + 1 ℝ factor per unit).
* `unit_scale_dominates` — `(1+u64)·x ≤ toReal (slackF * xF)` (1 rounding + 1 ℝ factor; the
  key's second `(1−u64)` is spare slack).
* `sgdStepCF` (2 units), `nsScalarFBoundF` (7 units — C61's op tree, the shared `σ²` sub-budget
  bounded once and reused at its three occurrences), `muonStepCF` (9 units), `wdStepRhoF`
  (2 scale units), `wdStepCF` (= `slackF * sgdStepCF`, 3 units) — each with `*_dominates`.
* `contractThresholdF = 0.999` with the OFFLINE `contractThresholdF_lt_one : toReal
  contractThresholdF < 1` — C70's one-time-gap pattern for C67's STRICT contraction
  `wdStepRho Bd < 1`: the runnable check is the non-strict `checkLe (wdStepRhoF BdF)
  contractThresholdF`, strictness supplied by the proven gap below `1`.

**The all-Bool capstones** (every hypothesis a runtime `Bool`; the horizon `n` is a parameter):

* `trainRun_all_finite_runnable` — C64's SGD whole-run: region checks on `x`/`w`, `checkAbsLe`
  on seed/step-size, two `checkLe 0.0` nonnegativity checks, and ONE budget check
  `checkLe (runBoundF (sgdStepCF …) n B0F) capF` ⟹ every weight at every step `m ≤ n` finite.
* `muonTrainRun_all_finite_runnable` — the Muon-shaped run (adds `checkAbsLe` on the NS
  coefficients; the budget threads `muonStepCF`).
* `wdTrainRun_all_finite_uniform_runnable` — **the horizon-free crown**: an ARBITRARILY LONG
  weight-decay run certified by FINITELY MANY Bools — the data checks, the contraction check
  against `contractThresholdF`, and ONE n-independent budget check on `wdUniformBoundF`.

**Scope (honestly disclosed).** The mirrors inherit C78's upward slack (sound-not-complete,
≈0.1% per unit — conservatism compounds through the deeper trees, e.g. `muonStepCF`'s 9 units,
still negligible against `capF = 1e300`). The bounds fed to the ℝ theorems are `toReal` of the
CHECKED Float thresholds (`Bx := toReal BxF`, …) — the harness picks the thresholds, the Bools
bind the data to them. The strict contraction rides C70's one-time-gap pattern (the offline
`contractThresholdF_lt_one`, proved once). NO new axiom: everything routes through C78's
`slackF_key`/`capF_le`, C70's checkers, and the pre-existing `(1+δ)` models.
-/
import Puffer.RL.BudgetEval
open Puffer.FloatR
open Puffer.RL.FiniteBound (overflowBound)
open Puffer.RL.WholeRunFinite (sgdStepC sgdStepC_nonneg muonStepC muonStepC_nonneg
  nsScalarFBound_nonneg runBound trainRun muonTrainRun trainRun_all_finite
  muonTrainRun_all_finite)
open Puffer.RL.MuonUpdateFinite (nsScalarFBound)
open Puffer.RL.WdRunFinite (wdStepRho wdStepC wdStepRho_nonneg wdStepC_nonneg wdUniformBound
  wdTrainRun wdTrainRun_all_finite_uniform)
open Puffer.RL.MarginCheck (checkLe checkLe_sound checkAbsLe checkAbsLe_sound checkRegion
  checkRegion_sound)
open Puffer.RL.BudgetEval (slackF slackF_key slackF_nonneg capF capF_le runBoundF
  runBound_le_overflow_of_check wdUniformBoundF wdUniformBound_le_overflow_of_check)

namespace Puffer.RL.RunConstEval

/-- `1 − u64 ≥ 0` (local convenience; C78's twin is private there). -/
private theorem one_sub_u64_nonneg : (0 : ℝ) ≤ 1 - u64 := by linarith [u64_lt_one]

private theorem one_add_u64_nonneg : (0 : ℝ) ≤ 1 + u64 := by linarith [u64_pos]

/-! ### The three unit lemmas (one `slackF_key` each) -/

/-- **Multiplicative unit**: `(1+u64)·(x·y) ≤ toReal (slackF * (xF * yF))` from the arguments'
    domination. One ℝ factor + two roundings (the product and the slack multiply), absorbed by
    one `slackF_key`. -/
theorem unit_mul_dominates (xF yF : Float) (x y : ℝ)
    (hx0 : 0 ≤ x) (hxf : x ≤ toReal xF) (hy0 : 0 ≤ y) (hyf : y ≤ toReal yF) :
    (1 + u64) * (x * y) ≤ toReal (slackF * (xF * yF)) := by
  obtain ⟨δ₁, hδ₁, e₁⟩ := mul_model xF yF
  obtain ⟨δ₂, hδ₂, e₂⟩ := mul_model slackF (xF * yF)
  have hd₁ : 1 - u64 ≤ 1 + δ₁ := by have := (abs_le.mp hδ₁).1; linarith
  have hd₂ : 1 - u64 ≤ 1 + δ₂ := by have := (abs_le.mp hδ₂).1; linarith
  have hs := slackF_nonneg
  have hu1 := one_sub_u64_nonneg
  have hxy : x * y ≤ toReal xF * toReal yF := mul_le_mul hxf hyf hy0 (hx0.trans hxf)
  have hxy0 : 0 ≤ toReal xF * toReal yF := mul_nonneg (hx0.trans hxf) (hy0.trans hyf)
  have h₁ : (x * y) * (1 - u64) ≤ toReal (xF * yF) := by
    rw [e₁]
    calc (x * y) * (1 - u64) ≤ (toReal xF * toReal yF) * (1 - u64) :=
          mul_le_mul_of_nonneg_right hxy hu1
      _ ≤ (toReal xF * toReal yF) * (1 + δ₁) := mul_le_mul_of_nonneg_left hd₁ hxy0
  have h₁0 : 0 ≤ toReal (xF * yF) := by
    rw [e₁]; exact mul_nonneg hxy0 (by linarith)
  rw [e₂]
  calc (1 + u64) * (x * y)
      ≤ (toReal slackF * ((1 - u64) * (1 - u64))) * (x * y) :=
        mul_le_mul_of_nonneg_right slackF_key (mul_nonneg hx0 hy0)
    _ = (toReal slackF * ((x * y) * (1 - u64))) * (1 - u64) := by ring
    _ ≤ (toReal slackF * toReal (xF * yF)) * (1 - u64) :=
        mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left h₁ hs) hu1
    _ ≤ (toReal slackF * toReal (xF * yF)) * (1 + δ₂) :=
        mul_le_mul_of_nonneg_left hd₂ (mul_nonneg hs h₁0)

/-- **Additive unit**: `(1+u64)·(x+y) ≤ toReal (slackF * (xF + yF))` — the same two-rounding
    spend with the sum in place of the product. -/
theorem unit_add_dominates (xF yF : Float) (x y : ℝ)
    (hx0 : 0 ≤ x) (hxf : x ≤ toReal xF) (hy0 : 0 ≤ y) (hyf : y ≤ toReal yF) :
    (1 + u64) * (x + y) ≤ toReal (slackF * (xF + yF)) := by
  obtain ⟨δ₁, hδ₁, e₁⟩ := add_model xF yF
  obtain ⟨δ₂, hδ₂, e₂⟩ := mul_model slackF (xF + yF)
  have hd₁ : 1 - u64 ≤ 1 + δ₁ := by have := (abs_le.mp hδ₁).1; linarith
  have hd₂ : 1 - u64 ≤ 1 + δ₂ := by have := (abs_le.mp hδ₂).1; linarith
  have hs := slackF_nonneg
  have hu1 := one_sub_u64_nonneg
  have hxy : x + y ≤ toReal xF + toReal yF := add_le_add hxf hyf
  have hxy0 : 0 ≤ toReal xF + toReal yF := add_nonneg (hx0.trans hxf) (hy0.trans hyf)
  have h₁ : (x + y) * (1 - u64) ≤ toReal (xF + yF) := by
    rw [e₁]
    calc (x + y) * (1 - u64) ≤ (toReal xF + toReal yF) * (1 - u64) :=
          mul_le_mul_of_nonneg_right hxy hu1
      _ ≤ (toReal xF + toReal yF) * (1 + δ₁) := mul_le_mul_of_nonneg_left hd₁ hxy0
  have h₁0 : 0 ≤ toReal (xF + yF) := by
    rw [e₁]; exact mul_nonneg hxy0 (by linarith)
  rw [e₂]
  calc (1 + u64) * (x + y)
      ≤ (toReal slackF * ((1 - u64) * (1 - u64))) * (x + y) :=
        mul_le_mul_of_nonneg_right slackF_key (add_nonneg hx0 hy0)
    _ = (toReal slackF * ((x + y) * (1 - u64))) * (1 - u64) := by ring
    _ ≤ (toReal slackF * toReal (xF + yF)) * (1 - u64) :=
        mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left h₁ hs) hu1
    _ ≤ (toReal slackF * toReal (xF + yF)) * (1 + δ₂) :=
        mul_le_mul_of_nonneg_left hd₂ (mul_nonneg hs h₁0)

/-- **Scaling unit**: `(1+u64)·x ≤ toReal (slackF * xF)` — one rounding + one ℝ factor; the
    key's second `(1−u64)` is spare slack. -/
theorem unit_scale_dominates (xF : Float) (x : ℝ) (hx0 : 0 ≤ x) (hxf : x ≤ toReal xF) :
    (1 + u64) * x ≤ toReal (slackF * xF) := by
  obtain ⟨δ, hδ, e⟩ := mul_model slackF xF
  have hd : 1 - u64 ≤ 1 + δ := by have := (abs_le.mp hδ).1; linarith
  have hu1 := one_sub_u64_nonneg
  have hs := slackF_nonneg
  have hxf0 : 0 ≤ toReal xF := hx0.trans hxf
  rw [e]
  calc (1 + u64) * x
      ≤ (toReal slackF * ((1 - u64) * (1 - u64))) * x :=
        mul_le_mul_of_nonneg_right slackF_key hx0
    _ = (toReal slackF * (x * (1 - u64))) * (1 - u64) := by ring
    _ ≤ (toReal slackF * (toReal xF * (1 - u64))) * (1 - u64) :=
        mul_le_mul_of_nonneg_right
          (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_right hxf hu1) hs) hu1
    _ ≤ (toReal slackF * (toReal xF * 1)) * (1 - u64) := by
        have h1 : (1 : ℝ) - u64 ≤ 1 := by linarith [u64_pos]
        exact mul_le_mul_of_nonneg_right
          (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left h1 hxf0) hs) hu1
    _ = (toReal slackF * toReal xF) * (1 - u64) := by ring
    _ ≤ (toReal slackF * toReal xF) * (1 + δ) :=
        mul_le_mul_of_nonneg_left hd (mul_nonneg hs hxf0)

/-! ### The SGD step constant (C64's `sgdStepC`) -/

/-- Computable mirror of C64's `sgdStepC = (1+u64)·(Blr·((1+u64)·(Bgs·Bx)))` — two units. -/
def sgdStepCF (Bx Bgs Blr : Float) : Float := slackF * (Blr * (slackF * (Bgs * Bx)))

/-- **DOMINATION**: `sgdStepC Bx Bgs Blr ≤ toReal (sgdStepCF BxF BgsF BlrF)` from the bounds'
    domination (inner unit: the gradient product; outer unit: the step-size product). -/
theorem sgdStepCF_dominates (BxF BgsF BlrF : Float) (Bx Bgs Blr : ℝ)
    (hx0 : 0 ≤ Bx) (hxf : Bx ≤ toReal BxF)
    (hgs0 : 0 ≤ Bgs) (hgsf : Bgs ≤ toReal BgsF)
    (hlr0 : 0 ≤ Blr) (hlrf : Blr ≤ toReal BlrF) :
    sgdStepC Bx Bgs Blr ≤ toReal (sgdStepCF BxF BgsF BlrF) := by
  have hinner : (1 + u64) * (Bgs * Bx) ≤ toReal (slackF * (BgsF * BxF)) :=
    unit_mul_dominates BgsF BxF Bgs Bx hgs0 hgsf hx0 hxf
  have hinner0 : 0 ≤ (1 + u64) * (Bgs * Bx) :=
    mul_nonneg one_add_u64_nonneg (mul_nonneg hgs0 hx0)
  show (1 + u64) * (Blr * ((1 + u64) * (Bgs * Bx))) ≤ toReal (sgdStepCF BxF BgsF BlrF)
  exact unit_mul_dominates BlrF (slackF * (BgsF * BxF)) Blr ((1 + u64) * (Bgs * Bx))
    hlr0 hlrf hinner0 hinner

/-! ### The NS circuit budget (C61's `nsScalarFBound`) -/

/-- Computable mirror of C61's `nsScalarFBound` — seven units over its op tree, the shared
    `σ²` sub-budget `slackF * (Bσ * Bσ)` written at its three occurrences (same value, bounded
    once). -/
def nsScalarFBoundF (Ba Bb Bc Bσ : Float) : Float :=
  slackF * (Bσ * (slackF *
    (slackF * (Ba + slackF * (Bb * (slackF * (Bσ * Bσ))))
      + slackF * (Bc * (slackF * ((slackF * (Bσ * Bσ)) * (slackF * (Bσ * Bσ))))))))

/-- **DOMINATION** for the NS circuit budget: compositional over the seven units (S = the σ²
    budget, U = the `b`-term, L = the left summand, T = the σ⁴ budget, V = the `c`-term,
    M = the bracket, and the outer σ-product). -/
theorem nsScalarFBoundF_dominates (BaF BbF BcF BσF : Float) (Ba Bb Bc Bσ : ℝ)
    (ha0 : 0 ≤ Ba) (haf : Ba ≤ toReal BaF) (hb0 : 0 ≤ Bb) (hbf : Bb ≤ toReal BbF)
    (hc0 : 0 ≤ Bc) (hcf : Bc ≤ toReal BcF) (hσ0 : 0 ≤ Bσ) (hσf : Bσ ≤ toReal BσF) :
    nsScalarFBound Ba Bb Bc Bσ ≤ toReal (nsScalarFBoundF BaF BbF BcF BσF) := by
  have h1u := one_add_u64_nonneg
  -- S: the σ² budget
  have hS : (1 + u64) * (Bσ * Bσ) ≤ toReal (slackF * (BσF * BσF)) :=
    unit_mul_dominates BσF BσF Bσ Bσ hσ0 hσf hσ0 hσf
  have hS0 : 0 ≤ (1 + u64) * (Bσ * Bσ) := mul_nonneg h1u (mul_nonneg hσ0 hσ0)
  -- U: the b-term (1+u64)·(Bb·S)
  have hU : (1 + u64) * (Bb * ((1 + u64) * (Bσ * Bσ)))
      ≤ toReal (slackF * (BbF * (slackF * (BσF * BσF)))) :=
    unit_mul_dominates BbF (slackF * (BσF * BσF)) Bb ((1 + u64) * (Bσ * Bσ)) hb0 hbf hS0 hS
  have hU0 : 0 ≤ (1 + u64) * (Bb * ((1 + u64) * (Bσ * Bσ))) :=
    mul_nonneg h1u (mul_nonneg hb0 hS0)
  -- L: the left summand (1+u64)·(Ba + U)
  have hL : (1 + u64) * (Ba + (1 + u64) * (Bb * ((1 + u64) * (Bσ * Bσ))))
      ≤ toReal (slackF * (BaF + slackF * (BbF * (slackF * (BσF * BσF))))) :=
    unit_add_dominates BaF (slackF * (BbF * (slackF * (BσF * BσF)))) Ba
      ((1 + u64) * (Bb * ((1 + u64) * (Bσ * Bσ)))) ha0 haf hU0 hU
  have hL0 : 0 ≤ (1 + u64) * (Ba + (1 + u64) * (Bb * ((1 + u64) * (Bσ * Bσ)))) :=
    mul_nonneg h1u (add_nonneg ha0 hU0)
  -- T: the σ⁴ budget (1+u64)·(S·S)
  have hT : (1 + u64) * (((1 + u64) * (Bσ * Bσ)) * ((1 + u64) * (Bσ * Bσ)))
      ≤ toReal (slackF * ((slackF * (BσF * BσF)) * (slackF * (BσF * BσF)))) :=
    unit_mul_dominates (slackF * (BσF * BσF)) (slackF * (BσF * BσF))
      ((1 + u64) * (Bσ * Bσ)) ((1 + u64) * (Bσ * Bσ)) hS0 hS hS0 hS
  have hT0 : 0 ≤ (1 + u64) * (((1 + u64) * (Bσ * Bσ)) * ((1 + u64) * (Bσ * Bσ))) :=
    mul_nonneg h1u (mul_nonneg hS0 hS0)
  -- V: the c-term (1+u64)·(Bc·T)
  have hV : (1 + u64) * (Bc * ((1 + u64) * (((1 + u64) * (Bσ * Bσ)) * ((1 + u64) * (Bσ * Bσ)))))
      ≤ toReal (slackF * (BcF * (slackF * ((slackF * (BσF * BσF)) * (slackF * (BσF * BσF)))))) :=
    unit_mul_dominates BcF (slackF * ((slackF * (BσF * BσF)) * (slackF * (BσF * BσF)))) Bc
      ((1 + u64) * (((1 + u64) * (Bσ * Bσ)) * ((1 + u64) * (Bσ * Bσ)))) hc0 hcf hT0 hT
  have hV0 : 0 ≤ (1 + u64) * (Bc * ((1 + u64) *
      (((1 + u64) * (Bσ * Bσ)) * ((1 + u64) * (Bσ * Bσ))))) :=
    mul_nonneg h1u (mul_nonneg hc0 hT0)
  -- M: the bracket (1+u64)·(L + V)
  have hM : (1 + u64) * ((1 + u64) * (Ba + (1 + u64) * (Bb * ((1 + u64) * (Bσ * Bσ))))
        + (1 + u64) * (Bc * ((1 + u64) * (((1 + u64) * (Bσ * Bσ)) * ((1 + u64) * (Bσ * Bσ))))))
      ≤ toReal (slackF * (slackF * (BaF + slackF * (BbF * (slackF * (BσF * BσF))))
        + slackF * (BcF * (slackF * ((slackF * (BσF * BσF)) * (slackF * (BσF * BσF))))))) :=
    unit_add_dominates
      (slackF * (BaF + slackF * (BbF * (slackF * (BσF * BσF)))))
      (slackF * (BcF * (slackF * ((slackF * (BσF * BσF)) * (slackF * (BσF * BσF))))))
      ((1 + u64) * (Ba + (1 + u64) * (Bb * ((1 + u64) * (Bσ * Bσ)))))
      ((1 + u64) * (Bc * ((1 + u64) * (((1 + u64) * (Bσ * Bσ)) * ((1 + u64) * (Bσ * Bσ))))))
      hL0 hL hV0 hV
  have hM0 : 0 ≤ (1 + u64) * ((1 + u64) * (Ba + (1 + u64) * (Bb * ((1 + u64) * (Bσ * Bσ))))
        + (1 + u64) * (Bc * ((1 + u64) *
          (((1 + u64) * (Bσ * Bσ)) * ((1 + u64) * (Bσ * Bσ)))))) :=
    mul_nonneg h1u (add_nonneg hL0 hV0)
  -- the outer σ-product
  show (1 + u64) * (Bσ * ((1 + u64) *
      ((1 + u64) * (Ba + (1 + u64) * (Bb * ((1 + u64) * (Bσ * Bσ))))
        + (1 + u64) * (Bc * ((1 + u64) *
          (((1 + u64) * (Bσ * Bσ)) * ((1 + u64) * (Bσ * Bσ))))))))
    ≤ toReal (nsScalarFBoundF BaF BbF BcF BσF)
  exact unit_mul_dominates BσF
    (slackF * (slackF * (BaF + slackF * (BbF * (slackF * (BσF * BσF))))
      + slackF * (BcF * (slackF * ((slackF * (BσF * BσF)) * (slackF * (BσF * BσF)))))))
    Bσ
    ((1 + u64) * ((1 + u64) * (Ba + (1 + u64) * (Bb * ((1 + u64) * (Bσ * Bσ))))
      + (1 + u64) * (Bc * ((1 + u64) * (((1 + u64) * (Bσ * Bσ)) * ((1 + u64) * (Bσ * Bσ)))))))
    hσ0 hσf hM0 hM

/-! ### The Muon step constant (C64's `muonStepC`) -/

/-- Computable mirror of C64's `muonStepC = (1+u64)·(Blr·nsScalarFBound Ba Bb Bc
    ((1+u64)·(Bgs·Bx)))` — the gradient unit, the NS budget mirror, and the outer unit. -/
def muonStepCF (Bx Bgs Blr Ba Bb Bc : Float) : Float :=
  slackF * (Blr * nsScalarFBoundF Ba Bb Bc (slackF * (Bgs * Bx)))

/-- **DOMINATION** for the Muon step constant (nine units in total). -/
theorem muonStepCF_dominates (BxF BgsF BlrF BaF BbF BcF : Float) (Bx Bgs Blr Ba Bb Bc : ℝ)
    (hx0 : 0 ≤ Bx) (hxf : Bx ≤ toReal BxF)
    (hgs0 : 0 ≤ Bgs) (hgsf : Bgs ≤ toReal BgsF)
    (hlr0 : 0 ≤ Blr) (hlrf : Blr ≤ toReal BlrF)
    (ha0 : 0 ≤ Ba) (haf : Ba ≤ toReal BaF)
    (hb0 : 0 ≤ Bb) (hbf : Bb ≤ toReal BbF)
    (hc0 : 0 ≤ Bc) (hcf : Bc ≤ toReal BcF) :
    muonStepC Bx Bgs Blr Ba Bb Bc ≤ toReal (muonStepCF BxF BgsF BlrF BaF BbF BcF) := by
  have hG : (1 + u64) * (Bgs * Bx) ≤ toReal (slackF * (BgsF * BxF)) :=
    unit_mul_dominates BgsF BxF Bgs Bx hgs0 hgsf hx0 hxf
  have hG0 : 0 ≤ (1 + u64) * (Bgs * Bx) :=
    mul_nonneg one_add_u64_nonneg (mul_nonneg hgs0 hx0)
  have hNS : nsScalarFBound Ba Bb Bc ((1 + u64) * (Bgs * Bx))
      ≤ toReal (nsScalarFBoundF BaF BbF BcF (slackF * (BgsF * BxF))) :=
    nsScalarFBoundF_dominates BaF BbF BcF (slackF * (BgsF * BxF)) Ba Bb Bc
      ((1 + u64) * (Bgs * Bx)) ha0 haf hb0 hbf hc0 hcf hG0 hG
  have hNS0 : 0 ≤ nsScalarFBound Ba Bb Bc ((1 + u64) * (Bgs * Bx)) :=
    nsScalarFBound_nonneg Ba Bb Bc _ ha0 hb0 hc0 hG0
  show (1 + u64) * (Blr * nsScalarFBound Ba Bb Bc ((1 + u64) * (Bgs * Bx)))
    ≤ toReal (muonStepCF BxF BgsF BlrF BaF BbF BcF)
  exact unit_mul_dominates BlrF (nsScalarFBoundF BaF BbF BcF (slackF * (BgsF * BxF))) Blr
    (nsScalarFBound Ba Bb Bc ((1 + u64) * (Bgs * Bx))) hlr0 hlrf hNS0 hNS

/-! ### The weight-decay slope and constant (C67's `wdStepRho`/`wdStepC`) -/

/-- Computable mirror of C67's `wdStepRho Bd = (1+u64)²·Bd` — two scaling units. -/
def wdStepRhoF (Bd : Float) : Float := slackF * (slackF * Bd)

/-- **DOMINATION** for the weight-decay slope (feeds both the contraction check and the
    uniform-budget evaluator). -/
theorem wdStepRhoF_dominates (BdF : Float) (Bd : ℝ) (hd0 : 0 ≤ Bd) (hdf : Bd ≤ toReal BdF) :
    wdStepRho Bd ≤ toReal (wdStepRhoF BdF) := by
  have hinner : (1 + u64) * Bd ≤ toReal (slackF * BdF) := unit_scale_dominates BdF Bd hd0 hdf
  have hinner0 : 0 ≤ (1 + u64) * Bd := mul_nonneg one_add_u64_nonneg hd0
  have heq : wdStepRho Bd = (1 + u64) * ((1 + u64) * Bd) := by
    unfold Puffer.RL.WdRunFinite.wdStepRho; ring
  rw [heq]
  exact unit_scale_dominates (slackF * BdF) ((1 + u64) * Bd) hinner0 hinner

/-- Computable mirror of C67's `wdStepC = (1+u64)·sgdStepC` — one scaling unit over the SGD
    constant's mirror. -/
def wdStepCF (Bx Bgs Blr : Float) : Float := slackF * sgdStepCF Bx Bgs Blr

/-- **DOMINATION** for the weight-decay step constant. -/
theorem wdStepCF_dominates (BxF BgsF BlrF : Float) (Bx Bgs Blr : ℝ)
    (hx0 : 0 ≤ Bx) (hxf : Bx ≤ toReal BxF)
    (hgs0 : 0 ≤ Bgs) (hgsf : Bgs ≤ toReal BgsF)
    (hlr0 : 0 ≤ Blr) (hlrf : Blr ≤ toReal BlrF) :
    wdStepC Bx Bgs Blr ≤ toReal (wdStepCF BxF BgsF BlrF) := by
  have hsgd := sgdStepCF_dominates BxF BgsF BlrF Bx Bgs Blr hx0 hxf hgs0 hgsf hlr0 hlrf
  have hsgd0 : 0 ≤ sgdStepC Bx Bgs Blr := sgdStepC_nonneg Bx Bgs Blr hx0 hgs0 hlr0
  have heq : wdStepC Bx Bgs Blr = (1 + u64) * sgdStepC Bx Bgs Blr := by
    unfold Puffer.RL.WdRunFinite.wdStepC Puffer.RL.WholeRunFinite.sgdStepC; ring
  rw [heq]
  exact unit_scale_dominates (sgdStepCF BxF BgsF BlrF) (sgdStepC Bx Bgs Blr) hsgd0 hsgd

/-! ### The offline contraction threshold (C70's one-time-gap pattern) -/

/-- A Float threshold provably below `1`, for the STRICT contraction check: the runnable check
    is the non-strict `checkLe (wdStepRhoF BdF) contractThresholdF`; strictness comes from the
    offline `contractThresholdF_lt_one` — C70's one-time-gap pattern.

    **The quantified rejection band** (the sound-not-complete conservatism, measured): the check
    accepts exactly when `wdStepRhoF BdF ≈ 1.001²·BdF ≤ 0.999`, i.e. roughly `BdF ≲ 0.9970`
    (`0.999/1.001² ≈ 0.997005`; measured: `wdStepRhoF 0.997 = 0.998995` accepted,
    `wdStepRhoF 0.9971 = 0.999095` rejected). Decay factors within ≈0.3% of the contraction
    boundary — the two `slackF` units plus the `0.999`-vs-`1` gap — are soundly REJECTED though
    truly contractive: e.g. `wdStepRhoF 0.998 = 0.999997 > 0.999` fails the check even though
    `wdStepRho 0.998 = (1+u64)²·0.998 < 1` genuinely holds in ℝ. -/
def contractThresholdF : Float := 0.999

/-- **The offline gap** (proved once): `toReal contractThresholdF < 1` — the literal `0.999`'s
    `toReal` is within relative `u64` of `0.999`, comfortably below `1`. -/
theorem contractThresholdF_lt_one : toReal contractThresholdF < 1 := by
  have h := toReal_ofScientific_close 999 true 3
  have habs : |(0.999 : ℝ)| = (0.999 : ℝ) := abs_of_pos (by norm_num)
  rw [habs] at h
  have h2 := (abs_le.mp h).2
  have hu : u64 ≤ (1 : ℝ) / 1000000 := by unfold u64; norm_num
  unfold contractThresholdF
  nlinarith [h2, hu, u64_pos]

/-! ### The all-Bool whole-run capstones -/

/-- **THE RUNNABLE SGD WHOLE-RUN CERTIFICATE**: every hypothesis a runtime `Bool` (the horizon
    `n` is a parameter). Region checks bind the data to the Float thresholds; the two
    `checkLe 0.0` checks supply the thresholds' nonnegativity; ONE budget check on the evaluated
    run budget (C78's `runBoundF` at this module's `sgdStepCF`) certifies the ℝ-side
    `runBound (sgdStepC …) n B₀ ≤ overflowBound` that C64's `trainRun_all_finite` consumes. -/
theorem trainRun_all_finite_runnable (x w : List Float) (gseed lr : Float) (n : Nat)
    (BxF B0F BgsF BlrF : Float)
    (hx : checkRegion x BxF = true) (hw : checkRegion w B0F = true)
    (hgs : checkAbsLe gseed BgsF = true) (hlr : checkAbsLe lr BlrF = true)
    (hBx0 : checkLe 0.0 BxF = true) (hB00 : checkLe 0.0 B0F = true)
    (hbudget : checkLe (runBoundF (sgdStepCF BxF BgsF BlrF) n B0F) capF = true) :
    ∀ m, m ≤ n → ∀ u ∈ trainRun x gseed lr m w, u.isFinite = true := by
  have hBx0' : (0 : ℝ) ≤ toReal BxF := by
    have := checkLe_sound hBx0; rwa [toReal_zeroLit] at this
  have hB00' : (0 : ℝ) ≤ toReal B0F := by
    have := checkLe_sound hB00; rwa [toReal_zeroLit] at this
  have hgs' := checkAbsLe_sound hgs
  have hlr' := checkAbsLe_sound hlr
  have hgs0 : (0 : ℝ) ≤ toReal BgsF := (abs_nonneg _).trans hgs'
  have hlr0 : (0 : ℝ) ≤ toReal BlrF := (abs_nonneg _).trans hlr'
  have hC0 : 0 ≤ sgdStepC (toReal BxF) (toReal BgsF) (toReal BlrF) :=
    sgdStepC_nonneg _ _ _ hBx0' hgs0 hlr0
  have hCc : sgdStepC (toReal BxF) (toReal BgsF) (toReal BlrF)
      ≤ toReal (sgdStepCF BxF BgsF BlrF) :=
    sgdStepCF_dominates BxF BgsF BlrF _ _ _ hBx0' le_rfl hgs0 le_rfl hlr0 le_rfl
  exact trainRun_all_finite x w gseed lr n (toReal BxF) (toReal B0F) (toReal BgsF)
    (toReal BlrF) hBx0' hB00' (checkRegion_sound hx) (checkRegion_sound hw) hgs' hlr'
    (runBound_le_overflow_of_check (sgdStepCF BxF BgsF BlrF) B0F _ _ n hC0 hCc hB00' le_rfl
      hbudget)

/-- **THE RUNNABLE MUON WHOLE-RUN CERTIFICATE**: as the SGD one, with `checkAbsLe` on the NS
    coefficients and the budget threading `muonStepCF` (nine units of slack). -/
theorem muonTrainRun_all_finite_runnable (x w : List Float) (gseed lr a b c : Float) (n : Nat)
    (BxF B0F BgsF BlrF BaF BbF BcF : Float)
    (hx : checkRegion x BxF = true) (hw : checkRegion w B0F = true)
    (hgs : checkAbsLe gseed BgsF = true) (hlr : checkAbsLe lr BlrF = true)
    (ha : checkAbsLe a BaF = true) (hb : checkAbsLe b BbF = true)
    (hc : checkAbsLe c BcF = true)
    (hBx0 : checkLe 0.0 BxF = true) (hB00 : checkLe 0.0 B0F = true)
    (hbudget : checkLe (runBoundF (muonStepCF BxF BgsF BlrF BaF BbF BcF) n B0F) capF = true) :
    ∀ m, m ≤ n → ∀ u ∈ muonTrainRun x gseed lr a b c m w, u.isFinite = true := by
  have hBx0' : (0 : ℝ) ≤ toReal BxF := by
    have := checkLe_sound hBx0; rwa [toReal_zeroLit] at this
  have hB00' : (0 : ℝ) ≤ toReal B0F := by
    have := checkLe_sound hB00; rwa [toReal_zeroLit] at this
  have hgs' := checkAbsLe_sound hgs
  have hlr' := checkAbsLe_sound hlr
  have ha' := checkAbsLe_sound ha
  have hb' := checkAbsLe_sound hb
  have hc' := checkAbsLe_sound hc
  have hgs0 : (0 : ℝ) ≤ toReal BgsF := (abs_nonneg _).trans hgs'
  have hlr0 : (0 : ℝ) ≤ toReal BlrF := (abs_nonneg _).trans hlr'
  have ha0 : (0 : ℝ) ≤ toReal BaF := (abs_nonneg _).trans ha'
  have hb0 : (0 : ℝ) ≤ toReal BbF := (abs_nonneg _).trans hb'
  have hc0 : (0 : ℝ) ≤ toReal BcF := (abs_nonneg _).trans hc'
  have hC0 : 0 ≤ muonStepC (toReal BxF) (toReal BgsF) (toReal BlrF) (toReal BaF)
      (toReal BbF) (toReal BcF) :=
    muonStepC_nonneg _ _ _ _ _ _ hBx0' hgs0 hlr0 ha0 hb0 hc0
  have hCc : muonStepC (toReal BxF) (toReal BgsF) (toReal BlrF) (toReal BaF) (toReal BbF)
      (toReal BcF) ≤ toReal (muonStepCF BxF BgsF BlrF BaF BbF BcF) :=
    muonStepCF_dominates BxF BgsF BlrF BaF BbF BcF _ _ _ _ _ _
      hBx0' le_rfl hgs0 le_rfl hlr0 le_rfl ha0 le_rfl hb0 le_rfl hc0 le_rfl
  exact muonTrainRun_all_finite x w gseed lr a b c n (toReal BxF) (toReal B0F) (toReal BgsF)
    (toReal BlrF) (toReal BaF) (toReal BbF) (toReal BcF) hBx0' hB00'
    (checkRegion_sound hx) (checkRegion_sound hw) hgs' hlr' ha' hb' hc'
    (runBound_le_overflow_of_check (muonStepCF BxF BgsF BlrF BaF BbF BcF) B0F _ _ n hC0 hCc
      hB00' le_rfl hbudget)

/-- **THE RUNNABLE HORIZON-FREE CERTIFICATE (the crown)**: an ARBITRARILY LONG weight-decay run
    certified overflow-free by FINITELY MANY runtime Bools — the data checks, the contraction
    check `checkLe (wdStepRhoF BdF) contractThresholdF` (strictness from the offline
    `contractThresholdF_lt_one`), and ONE n-independent budget check on C78's
    `wdUniformBoundF` at this module's mirrored slope/constant. No hypothesis mentions the
    horizon: the conclusion is `∀ m`. -/
theorem wdTrainRun_all_finite_uniform_runnable (x w : List Float) (gseed lr d : Float)
    (BxF B0F BgsF BlrF BdF : Float)
    (hx : checkRegion x BxF = true) (hw : checkRegion w B0F = true)
    (hgs : checkAbsLe gseed BgsF = true) (hlr : checkAbsLe lr BlrF = true)
    (hd : checkAbsLe d BdF = true)
    (hBx0 : checkLe 0.0 BxF = true) (hB00 : checkLe 0.0 B0F = true)
    (hcontr : checkLe (wdStepRhoF BdF) contractThresholdF = true)
    (hbudget : checkLe (wdUniformBoundF (wdStepRhoF BdF) (wdStepCF BxF BgsF BlrF) B0F) capF
      = true) :
    ∀ m, ∀ u ∈ wdTrainRun x gseed lr d m w, u.isFinite = true := by
  have hBx0' : (0 : ℝ) ≤ toReal BxF := by
    have := checkLe_sound hBx0; rwa [toReal_zeroLit] at this
  have hB00' : (0 : ℝ) ≤ toReal B0F := by
    have := checkLe_sound hB00; rwa [toReal_zeroLit] at this
  have hgs' := checkAbsLe_sound hgs
  have hlr' := checkAbsLe_sound hlr
  have hd' := checkAbsLe_sound hd
  have hgs0 : (0 : ℝ) ≤ toReal BgsF := (abs_nonneg _).trans hgs'
  have hlr0 : (0 : ℝ) ≤ toReal BlrF := (abs_nonneg _).trans hlr'
  have hd0 : (0 : ℝ) ≤ toReal BdF := (abs_nonneg _).trans hd'
  have hρdom : wdStepRho (toReal BdF) ≤ toReal (wdStepRhoF BdF) :=
    wdStepRhoF_dominates BdF _ hd0 le_rfl
  have hρlt : toReal (wdStepRhoF BdF) < 1 :=
    lt_of_le_of_lt (checkLe_sound hcontr) contractThresholdF_lt_one
  have hcontract : wdStepRho (toReal BdF) < 1 := lt_of_le_of_lt hρdom hρlt
  have hC0 : 0 ≤ wdStepC (toReal BxF) (toReal BgsF) (toReal BlrF) :=
    wdStepC_nonneg _ _ _ hBx0' hgs0 hlr0
  have hCc : wdStepC (toReal BxF) (toReal BgsF) (toReal BlrF)
      ≤ toReal (wdStepCF BxF BgsF BlrF) :=
    wdStepCF_dominates BxF BgsF BlrF _ _ _ hBx0' le_rfl hgs0 le_rfl hlr0 le_rfl
  exact wdTrainRun_all_finite_uniform x w gseed lr d (toReal BxF) (toReal B0F) (toReal BgsF)
    (toReal BlrF) (toReal BdF) hBx0' hB00' (checkRegion_sound hx) (checkRegion_sound hw)
    hgs' hlr' hd' hcontract
    (wdUniformBound_le_overflow_of_check (wdStepRhoF BdF) (wdStepCF BxF BgsF BlrF) B0F
      (wdStepRho (toReal BdF)) (wdStepC (toReal BxF) (toReal BgsF) (toReal BlrF))
      (toReal B0F) hρdom hρlt hC0 hCc hB00' le_rfl hbudget)

/-! ### Non-vacuity: the capstone shape composes (Bool hypotheses in hypothesis form — Float
    comparison is kernel-opaque, the C70/C79 split; the harness's native evaluation supplies
    the witnesses). -/

example (x w : List Float) (gseed lr d : Float) (BxF B0F BgsF BlrF BdF : Float)
    (hx : checkRegion x BxF = true) (hw : checkRegion w B0F = true)
    (hgs : checkAbsLe gseed BgsF = true) (hlr : checkAbsLe lr BlrF = true)
    (hd : checkAbsLe d BdF = true)
    (hBx0 : checkLe 0.0 BxF = true) (hB00 : checkLe 0.0 B0F = true)
    (hcontr : checkLe (wdStepRhoF BdF) contractThresholdF = true)
    (hbudget : checkLe (wdUniformBoundF (wdStepRhoF BdF) (wdStepCF BxF BgsF BlrF) B0F) capF
      = true) :
    ∀ u ∈ wdTrainRun x gseed lr d 1000000 w, u.isFinite = true :=
  wdTrainRun_all_finite_uniform_runnable x w gseed lr d BxF B0F BgsF BlrF BdF
    hx hw hgs hlr hd hBx0 hB00 hcontr hbudget 1000000

end Puffer.RL.RunConstEval
