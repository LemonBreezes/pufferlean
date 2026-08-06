/-
# Concrete Banach: the weight-decay fixed point exists on the finite parameter space `Fin d → ℝ`

C52 (`WeightDecayFixedPoint`) proved `banach_fixedPoint_exists` ABSTRACTLY — a `K`-Lipschitz self-map with `K < 1`
on any nonempty COMPLETE metric space has a fixed point — but did NOT instantiate it for the weight-decay update,
because the codebase's `Nat → ℝ` with the sup metric is not a standard complete-metric Mathlib instance (infinite
index). This module closes that gap at the CONCRETE finite parameter space `Fin d → ℝ`, which IS a complete metric
space (`Pi.completeSpace`, sup metric) — a real network has finitely many parameters, so `Fin d → ℝ` is the right
space.

* `sup_lipschitz_to_lipschitzWith` — the metric-space bridge: the codebase's sup-bound Lipschitz form (`∀ σ σ' δ,
  (∀ i, |σ i − σ' i| ≤ δ) → ∀ i, |f σ i − f σ' i| ≤ K·δ`) implies Mathlib's `LipschitzWith ⟨K,hK⟩ f` on `Fin d → ℝ`
  (via `dist_pi_le_iff` / `dist_le_pi_dist`: the `Fin d → ℝ` distance is the coordinate sup, and `dist = |·−·|` in ℝ).
* `contractingWith_of_sup_lipschitz` — with `K < 1` added, `ContractingWith ⟨K,hK⟩ f`.
* `fin_fixedPoint_exists` — the CONCRETE Banach existence: a sup-Lipschitz self-map on `Fin d → ℝ` with `K < 1` has a
  fixed point `∃ θ*, f θ* = θ*` (composing the bridge with C52's `banach_fixedPoint_exists` on the complete space
  `Fin d → ℝ`).
* `wdShaped_fin_fixedPoint` — the capstone: a finite-dim WEIGHT-DECAY-shaped map (sup-Lipschitz with C32's constant
  `K = |1 − wd| + |lr|·G`) that CONTRACTS (`K < 1`, weight decay dominating — C32's `wd_L_lt_one`) has a fixed point.
  This discharges the fixed-point EXISTENCE that C52/C49 assumed — concretely, for finitely many parameters.

**Scope (honestly disclosed).** This instantiates C52's abstract Banach existence at the concrete finite parameter
space `Fin d → ℝ` (complete via `Pi.completeSpace`), closing C52's un-wired instantiation glue: a sup-Lipschitz
weight-decay-shaped map with `K < 1` on finitely many parameters HAS a fixed point. Remaining: the codebase's
`wdAscentE` is stated on `Nat → ℝ`; bridging the `Nat`-indexed weight-decay step to the `Fin d`-indexed one (a
finite-support / restriction argument) is not done here — `wdShaped_fin_fixedPoint` is stated for a `Fin d → ℝ` map
carrying C32's Lipschitz constant, which is the shape `wdAscentE` restricted to `d` parameters has. The fixed point's
LOCATION is data-dependent (a stationary point of the weight-decay-regularized objective); its EXISTENCE is now
concrete Banach (`K < 1` ⟹ a fixed point exists, for finite parameters).
-/
import Puffer.RL.WeightDecayFixedPoint
import Mathlib.Topology.MetricSpace.Contracting
open Puffer.FloatR (toReal)
open Puffer.RL.WeightDecayFixedPoint (banach_fixedPoint_exists)
open scoped NNReal

namespace Puffer.RL.BanachConcrete

/-- **Metric-space bridge.** The codebase's sup-bound Lipschitz form implies Mathlib's `LipschitzWith` on the finite
    parameter space `Fin d → ℝ`. The `Fin d → ℝ` distance is the coordinate supremum (`dist_pi_le_iff` /
    `dist_le_pi_dist`), and `dist = |·−·|` in ℝ (`Real.dist_eq`), so a uniform per-coordinate bound by `K·(sup dist)`
    is exactly the metric Lipschitz condition. -/
theorem sup_lipschitz_to_lipschitzWith {d : ℕ} (f : (Fin d → ℝ) → (Fin d → ℝ)) (K : ℝ) (hK : 0 ≤ K)
    (hf : ∀ (σ σ' : Fin d → ℝ) (δ : ℝ), (∀ i, |σ i - σ' i| ≤ δ) → ∀ i, |f σ i - f σ' i| ≤ K * δ) :
    LipschitzWith ⟨K, hK⟩ f := by
  apply LipschitzWith.of_dist_le_mul
  intro σ σ'
  rw [dist_pi_le_iff (by positivity)]
  intro i
  rw [Real.dist_eq, NNReal.coe_mk]
  exact hf σ σ' (dist σ σ') (fun j => by rw [← Real.dist_eq]; exact dist_le_pi_dist σ σ' j) i

/-- With `K < 1` added, the sup-Lipschitz map is a Mathlib `ContractingWith ⟨K,hK⟩` on `Fin d → ℝ`. -/
theorem contractingWith_of_sup_lipschitz {d : ℕ} (f : (Fin d → ℝ) → (Fin d → ℝ)) (K : ℝ) (hK : 0 ≤ K) (hK1 : K < 1)
    (hf : ∀ (σ σ' : Fin d → ℝ) (δ : ℝ), (∀ i, |σ i - σ' i| ≤ δ) → ∀ i, |f σ i - f σ' i| ≤ K * δ) :
    ContractingWith ⟨K, hK⟩ f :=
  ⟨by exact_mod_cast hK1, sup_lipschitz_to_lipschitzWith f K hK hf⟩

/-- **Concrete Banach existence.** A sup-Lipschitz self-map on the finite parameter space `Fin d → ℝ` with constant
    `K < 1` has a fixed point `∃ θ*, f θ* = θ*` — composing the metric-space bridge with C52's `banach_fixedPoint_exists`
    on the complete space `Fin d → ℝ` (`Pi.completeSpace`; `Fin d → ℝ` is nonempty for every `d`). This is C52's
    abstract Banach existence made concrete for finitely many parameters. -/
theorem fin_fixedPoint_exists {d : ℕ} (f : (Fin d → ℝ) → (Fin d → ℝ)) (K : ℝ) (hK : 0 ≤ K) (hK1 : K < 1)
    (hf : ∀ (σ σ' : Fin d → ℝ) (δ : ℝ), (∀ i, |σ i - σ' i| ≤ δ) → ∀ i, |f σ i - f σ' i| ≤ K * δ) :
    ∃ θstar : Fin d → ℝ, f θstar = θstar :=
  banach_fixedPoint_exists (contractingWith_of_sup_lipschitz f K hK hK1 hf)

/-- **CAPSTONE: the weight-decay fixed point exists (concretely, on finite parameters).** A finite-dim
    weight-decay-shaped map `f : (Fin d → ℝ) → (Fin d → ℝ)` — sup-Lipschitz with C32's constant
    `K = |1 − wd| + |lr|·G` — that CONTRACTS (`K < 1`, i.e. weight decay dominates the ascent, C32's `wd_L_lt_one`
    condition `wd > |lr|·G`) has a fixed point `θ*`. This supplies the fixed-point EXISTENCE that C52's
    `wdAscentE_fixedPoint_contracts` and C49 take as given — for the `Fin d → ℝ`-shaped weight-decay map (the shape the
    literal `Nat`-indexed `wdAscentE` has once restricted to `d` parameters); bridging to the literal `wdAscentE`
    (a finite-support/restriction argument) is the one remaining step. Now a concrete Banach consequence of the
    contraction, for finitely many parameters. (The gradient-Lipschitz `G ≥ 0` makes the constant nonnegative.) -/
theorem wdShaped_fin_fixedPoint {d : ℕ} (lr wd : Float) (G : ℝ) (hG0 : 0 ≤ G)
    (f : (Fin d → ℝ) → (Fin d → ℝ))
    (hK1 : |1 - toReal wd| + |toReal lr| * G < 1)
    (hf : ∀ (σ σ' : Fin d → ℝ) (δ : ℝ), (∀ i, |σ i - σ' i| ≤ δ) →
        ∀ i, |f σ i - f σ' i| ≤ (|1 - toReal wd| + |toReal lr| * G) * δ) :
    ∃ θstar : Fin d → ℝ, f θstar = θstar :=
  fin_fixedPoint_exists f (|1 - toReal wd| + |toReal lr| * G) (by positivity) hK1 hf

end Puffer.RL.BanachConcrete
