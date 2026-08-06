/-
# The `Nat`↔`Fin d` bridge: the LITERAL `wdAscentE` has a fixed point (finite-parameter network)

C55 (`BanachConcrete`) proved a `Fin d → ℝ`-shaped weight-decay map has a fixed point (concrete Banach on the
complete finite-dimensional space), but left the bridge to the LITERAL weight-decay update `wdAscentE :
(Nat → ℝ) → (Nat → ℝ)` (C32, defined on the codebase's `Nat`-indexed parameter space) as remaining glue. This module
closes it with the standard FINITE-SUPPORT / RESTRICTION argument, giving the literal `wdAscentE` a fixed point for a
finite-`d`-parameter objective (the real case: a network has finitely many parameters, and the objective's gradient
vanishes for non-existent parameter indices `≥ d`).

* `finRestrict`/`natExtend` — the restriction `(Nat → ℝ) → (Fin d → ℝ)` and zero-padding extension `(Fin d → ℝ) →
  (Nat → ℝ)`, with the round-trips `finRestrict_natExtend` (always) and `natExtend_finRestrict` (for `[0,d)`-supported
  functions).
* `wdAscentE_preserves_support` — `wdAscentE e lr wd σ` is `[0,d)`-supported when `σ` is and the gradient vanishes
  beyond `d` (`(1−wd)·0 + lr·0 = 0`).
* `wdFin`/`wdFin_lipschitz` — the `Fin d → ℝ` restriction `finRestrict ∘ wdAscentE ∘ natExtend`, sup-Lipschitz with
  C32's constant `K = |1−wd| + |lr|·G` (from `wdAscentE_sup_lipschitz` on the extended points).
* `wdAscentE_fixedPoint_exists` — THE BRIDGE: for a finite-`d`-parameter objective under weight decay (`K < 1`), the
  LITERAL `wdAscentE` has a fixed point `θ* : Nat → ℝ`. Take C55's `Fin d` fixed point `v*`, extend it; `natExtend v*`
  is `[0,d)`-supported, so `wdAscentE (natExtend v*)` is too, hence equals `natExtend (finRestrict (wdAscentE
  (natExtend v*))) = natExtend v*` — a fixed point. Discharges the fixed-point EXISTENCE C52/C49 assumed.
* `clip_invariant_of_weight_decay_concrete` — the capstone: composing the constructed fixed point with C52's
  `clip_invariant_of_weight_decay_fixedPoint`, the clip-interior is `∀n`-invariant for the literal weight-decay
  optimizer on a finite-parameter network (modulo the fixed-point ratio margin, a checkable one-point condition).

**Scope (honestly disclosed).** The honest hypothesis is `∀ σ k, d ≤ k → derivR e σ k = 0` — the objective is a
finite-`d`-parameter expression (true for any concrete network: parameter indices `≥ d` do not occur in `e`, so their
partials are `0`). The fixed point's EXISTENCE is now concrete for the literal `wdAscentE`; its LOCATION stays
data-dependent (a stationary point of the weight-decay-regularized objective), and the fixed-point ratio margin
(`ratio(θ*) ∈ (lo+Lr·d0, hi−Lr·d0)`) is the checkable one-point condition (C49), quantified here over the constructed
fixed point. The `K < 1` weight-decay condition (`wd > |lr|·G`) is C32's; `G` (gradient Lipschitz) and `Lr` (the
log-softmax+exp ratio budget) are the network's structural constants.
-/
import Puffer.RL.BanachConcrete
import Puffer.RL.WeightDecayInterval
open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)
open Puffer.RL.WeightDecayInterval (wdAscentE wdAscentE_sup_lipschitz)
open Puffer.RL.BanachConcrete (wdShaped_fin_fixedPoint)
open Puffer.RL.WeightDecayFixedPoint (clip_invariant_of_weight_decay_fixedPoint)
open Puffer.RL.SoftmaxExpr (logSoftmaxE)
open Puffer.RL.SurrogateExpr (ratioE)

namespace Puffer.RL.WdNatFinBridge

/-- Restrict a `Nat`-indexed parameter vector to its first `d` coordinates. -/
def finRestrict {d : ℕ} (σ : Nat → ℝ) : Fin d → ℝ := fun i => σ i.val

/-- Extend a `Fin d`-indexed vector to `Nat → ℝ` by padding with `0` beyond `d`. -/
noncomputable def natExtend {d : ℕ} (v : Fin d → ℝ) : Nat → ℝ := fun k => if h : k < d then v ⟨k, h⟩ else 0

/-- `finRestrict (natExtend v) = v` (the `Fin d` round-trip, always). -/
theorem finRestrict_natExtend {d : ℕ} (v : Fin d → ℝ) : finRestrict (natExtend v) = v := by
  funext i
  simp only [finRestrict, natExtend, dif_pos i.isLt]

/-- `natExtend (finRestrict σ) = σ` for a `[0,d)`-supported `σ` (`∀ k ≥ d, σ k = 0`). -/
theorem natExtend_finRestrict {d : ℕ} (σ : Nat → ℝ) (hσ : ∀ k, d ≤ k → σ k = 0) :
    natExtend (finRestrict (d := d) σ) = σ := by
  funext k
  by_cases h : k < d
  · simp only [natExtend, finRestrict, dif_pos h]
  · simp only [natExtend, dif_neg h]; exact (hσ k (Nat.not_lt.mp h)).symm

/-- **Weight-decay preserves finite support.** If the objective's gradient vanishes beyond `d` and `σ` is
    `[0,d)`-supported, then `wdAscentE e lr wd σ` is too: `(1−wd)·0 + lr·0 = 0` for `k ≥ d`. -/
theorem wdAscentE_preserves_support (e : Expr) (lr wd : Float) (σ : Nat → ℝ) (d : ℕ)
    (hgs : ∀ k, d ≤ k → derivR e σ k = 0) (hσs : ∀ k, d ≤ k → σ k = 0) :
    ∀ k, d ≤ k → wdAscentE e lr wd σ k = 0 := by
  intro k hk
  simp only [wdAscentE, hσs k hk, hgs k hk, mul_zero, add_zero]

/-- The `Fin d → ℝ` restriction of `wdAscentE`: `finRestrict ∘ wdAscentE ∘ natExtend`. -/
noncomputable def wdFin (e : Expr) (lr wd : Float) (d : ℕ) (v : Fin d → ℝ) : Fin d → ℝ :=
  finRestrict (wdAscentE e lr wd (natExtend v))

/-- **`wdFin` is sup-Lipschitz with C32's constant `K = |1−wd| + |lr|·G`** — `wdAscentE_sup_lipschitz` applied to the
    extended points (the extension is `δ`-close where the originals are, and `0`-close beyond `d`). -/
theorem wdFin_lipschitz (e : Expr) (lr wd : Float) (d : ℕ) (G : ℝ)
    (hgLip : ∀ (σ σ' : Nat → ℝ) (δ : ℝ), (∀ i, |σ i - σ' i| ≤ δ) →
        ∀ k, |derivR e σ k - derivR e σ' k| ≤ G * δ)
    (v v' : Fin d → ℝ) (δ : ℝ) (hδ : ∀ i, |v i - v' i| ≤ δ) :
    ∀ i, |wdFin e lr wd d v i - wdFin e lr wd d v' i| ≤ (|1 - toReal wd| + |toReal lr| * G) * δ := by
  rcases Nat.eq_zero_or_pos d with hd0 | hdpos
  · subst hd0; exact fun i => i.elim0
  · have hδ0 : 0 ≤ δ := le_trans (abs_nonneg _) (hδ ⟨0, hdpos⟩)
    have hclose : ∀ j, |natExtend v j - natExtend v' j| ≤ δ := by
      intro j
      by_cases h : j < d
      · simp only [natExtend, dif_pos h]; exact hδ ⟨j, h⟩
      · simp only [natExtend, dif_neg h]; simpa using hδ0
    have key := wdAscentE_sup_lipschitz e lr wd G δ (natExtend v) (natExtend v') hclose
      (hgLip (natExtend v) (natExtend v') δ hclose)
    intro i
    simpa only [wdFin, finRestrict] using key i.val

/-- **THE `Nat`↔`Fin d` BRIDGE.** For a finite-`d`-parameter objective (gradient vanishing beyond `d`) under weight
    decay (`K = |1−wd| + |lr|·G < 1`), the LITERAL `wdAscentE : (Nat → ℝ) → (Nat → ℝ)` has a fixed point `θ*`. Extend
    C55's `Fin d` Banach fixed point; support-preservation makes the extension a genuine fixed point of the literal
    update. Discharges the fixed-point EXISTENCE C52/C49 assumed — now concrete for the literal `Nat`-indexed step. -/
theorem wdAscentE_fixedPoint_exists (e : Expr) (lr wd : Float) (d : ℕ) (G : ℝ) (hG0 : 0 ≤ G)
    (hK1 : |1 - toReal wd| + |toReal lr| * G < 1)
    (hgLip : ∀ (σ σ' : Nat → ℝ) (δ : ℝ), (∀ i, |σ i - σ' i| ≤ δ) →
        ∀ k, |derivR e σ k - derivR e σ' k| ≤ G * δ)
    (hgsupp : ∀ (σ : Nat → ℝ) (k : ℕ), d ≤ k → derivR e σ k = 0) :
    ∃ θstar : Nat → ℝ, wdAscentE e lr wd θstar = θstar := by
  obtain ⟨v, hv⟩ := wdShaped_fin_fixedPoint lr wd G hG0 (wdFin e lr wd d) hK1
    (wdFin_lipschitz e lr wd d G hgLip)
  refine ⟨natExtend v, ?_⟩
  have hσsupp : ∀ k, d ≤ k → natExtend v k = 0 := by
    intro k hk; simp only [natExtend, dif_neg (Nat.not_lt.mpr hk)]
  have hwsupp : ∀ k, d ≤ k → wdAscentE e lr wd (natExtend v) k = 0 :=
    wdAscentE_preserves_support e lr wd (natExtend v) d (fun k hk => hgsupp (natExtend v) k hk) hσsupp
  have hrestr : finRestrict (wdAscentE e lr wd (natExtend v)) = v := hv
  calc wdAscentE e lr wd (natExtend v)
      = natExtend (finRestrict (wdAscentE e lr wd (natExtend v))) :=
        (natExtend_finRestrict (wdAscentE e lr wd (natExtend v)) hwsupp).symm
    _ = natExtend v := by rw [hrestr]

/-- **CAPSTONE: clip-interior invariance for the literal weight-decay optimizer on a finite-parameter network.**
    Composes the constructed fixed point (`wdAscentE_fixedPoint_exists`) with C52's
    `clip_invariant_of_weight_decay_fixedPoint`: for a finite-`d`-parameter objective under weight decay (`K < 1`),
    the literal `wdAscentE` trajectory keeps the PPO ratio in `(lo,hi)` for ALL `n`. The fixed-point-dependent
    conditions (initial distance, ratio Lipschitz, and the ratio margin `ratio(θ*) ∈ (lo+Lr·d0, hi−Lr·d0)`) are
    supplied for the constructed fixed point (quantified over any fixed point, since its location is data-dependent).
    The fixed-point EXISTENCE is now discharged — the clip-interior reachability closed for the weight-decay optimizer
    on a concrete finite-parameter network. -/
theorem clip_invariant_of_weight_decay_concrete
    (chosen e : Expr) (es : List Expr) (oldLogp lo hi : Float)
    (eObj : Expr) (lr wd : Float) (d : ℕ) (G Lr d0 : ℝ)
    (θ : Nat → (Nat → ℝ))
    (hG : ∀ (σ σ' : Nat → ℝ) (δ : ℝ), (∀ i, |σ i - σ' i| ≤ δ) →
        ∀ k, |derivR eObj σ k - derivR eObj σ' k| ≤ G * δ)
    (hGn : 0 ≤ G) (hd0 : 0 ≤ d0) (hLr : 0 ≤ Lr)
    (hK1 : |1 - toReal wd| + |toReal lr| * G < 1)
    (hgsupp : ∀ (σ : Nat → ℝ) (k : ℕ), d ≤ k → derivR eObj σ k = 0)
    (htraj : ∀ n, θ (n + 1) = wdAscentE eObj lr wd (θ n))
    (hinit : ∀ θstar : Nat → ℝ, wdAscentE eObj lr wd θstar = θstar → ∀ k, |θ 0 k - θstar k| ≤ d0)
    (hRLip : ∀ θstar : Nat → ℝ, wdAscentE eObj lr wd θstar = θstar →
        ∀ n, (∀ k, |θ n k - θstar k| ≤ (|1 - toReal wd| + |toReal lr| * G) ^ n * d0) →
          |evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ n)
            - evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) θstar|
            ≤ Lr * ((|1 - toReal wd| + |toReal lr| * G) ^ n * d0))
    (hlo : ∀ θstar : Nat → ℝ, wdAscentE eObj lr wd θstar = θstar →
        toReal lo < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) θstar - Lr * d0)
    (hhi : ∀ θstar : Nat → ℝ, wdAscentE eObj lr wd θstar = θstar →
        evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) θstar + Lr * d0 < toReal hi)
    (n : Nat) :
    toReal lo < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ n)
      ∧ evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ n) < toReal hi := by
  obtain ⟨θstar, hstar⟩ := wdAscentE_fixedPoint_exists eObj lr wd d G hGn hK1 hG hgsupp
  exact clip_invariant_of_weight_decay_fixedPoint chosen e es oldLogp lo hi eObj lr wd G Lr d0 θ θstar
    hG hGn hd0 hLr (le_of_lt hK1) (fun k => congrFun hstar k) htraj (hinit θstar hstar)
    (hRLip θstar hstar) (hlo θstar hstar) (hhi θstar hstar) n

end Puffer.RL.WdNatFinBridge
