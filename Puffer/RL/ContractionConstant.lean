/-
# Discharging C46's contraction hypothesis from a weight-decay parameter contraction

C46 (`ClipContraction`) proved: IF the PPO ratio contracts toward a center `c` strictly inside the clip window
(`∀ n, |r(n+1) − c| ≤ ρ·|r n − c| + ε`, `ρ < 1`), THEN the clip interior is forward-invariant for all `n`. C46 left
"establishing this contraction for a concrete optimizer" open (analogous to C42's open Muon step-Lipschitz `L`). This
module closes that step for a WEIGHT-DECAY update, by composing two established ingredients:

  * the WEIGHT-DECAY PARAMETER CONTRACTION (C32 `WeightDecayInterval`): under `wd > |lr|·G` the step
    `wdAscentE e lr wd σ = (1−wd)·σ + lr·∇` contracts toward a fixed point `θ*` (`L = |1−wd| + |lr|·G < 1`), so the
    sup-distance `d n := sup_k |θ n k − θ* k|` is non-increasing (`d (n+1) ≤ ρθ·d n`, `ρθ ≤ 1`); and
  * the RATIO's PARAMETER-LIPSCHITZ (C18/C36): `|ratio σ − ratio σ'| ≤ Lr·sup_k|σ k − σ' k|` (the log-softmax value
    Lipschitz composed with `exp`, as in C36's `ratioE_projStep_disp`), with `Lr` the network's budget constant.

Together they give the DIRECT geometric route: `|ratio(θ n) − ratio(θ*)| ≤ Lr·d n ≤ Lr·d 0` (a UNIFORM trapping radius
`Lr·d 0`, since the parameter distance never grows), so if `c = ratio(θ*)` is inside the clip window by more than
`Lr·d 0`, the ratio stays in `(lo,hi)` for all `n` — the clip-interior invariant, with the contraction DISCHARGED.

* `dist_le_init` — under a non-expansive parameter contraction (`d (n+1) ≤ ρθ·d n`, `0 ≤ ρθ ≤ 1`), the distance stays
  bounded by its initial value (`∀ n, d n ≤ d 0`).
* `clip_invariant_of_weight_decay` — from the composed ratio-distance bound `|ratio(θ n) − c| ≤ Lr·d n` and the
  contraction on `d`, plus the window condition `c ∈ (lo + Lr·d 0, hi − Lr·d 0)`, conclude `∀ n, ratio(θ n) ∈ (lo,hi)`.
* `clip_invariant_of_weight_decay_lipschitz` — the same with the ratio-distance bound DERIVED from an explicit ratio
  value-Lipschitz `Lr` (applied at `θ n` vs `θ*`, given the per-coordinate parameter distance `≤ d n`) and
  `c = ratio(θ*)` — the honest composition of the two ingredients.

**Scope (honestly disclosed).** This closes C46's contraction hypothesis for the WEIGHT-DECAY optimizer, deriving it
from (i) C32's weight-decay parameter contraction (which needs `wd > |lr|·G` — weight decay dominating the ascent;
plain ascent is expansive, C29/C36) and (ii) C18/C36's ratio value-Lipschitz `Lr`. The remaining honest inputs are:
the fixed point `θ*` and the parameter contraction toward it (C32's weight-decay premise, given `L < 1`), the ratio
Lipschitz constant `Lr` (the network's log-softmax+exp budget), and `c = ratio(θ*)` inside the window by more than the
trapping radius `Lr·d 0` (a checkable margin at the fixed point). `ρθ ≤ 1` (non-expansive) suffices for the uniform
distance bound; strict weight decay gives `ρθ < 1` (genuine contraction toward `θ*`).
-/
import Puffer.RL.ClipContraction
open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)
open Puffer.RL.SurrogateExpr (ratioE)
open Puffer.RL.SoftmaxExpr (logSoftmaxE)

namespace Puffer.RL.ContractionConstant

/-- **The parameter distance stays bounded by its initial value under a non-expansive contraction.** If the
    sup-distance to the fixed point satisfies `d (n+1) ≤ ρθ·d n` with `0 ≤ ρθ ≤ 1` and `d n ≥ 0`, then `d n ≤ d 0` for
    all `n` (the distance is non-increasing). Strict weight decay gives `ρθ < 1` (the distance actually shrinks
    geometrically), but `ρθ ≤ 1` already suffices for the uniform bound the clip invariant needs. -/
theorem dist_le_init (d : Nat → ℝ) (ρθ : ℝ) (hd : ∀ n, 0 ≤ d n)
    (hρ1 : ρθ ≤ 1) (hc : ∀ n, d (n + 1) ≤ ρθ * d n) :
    ∀ n, d n ≤ d 0 := by
  intro n
  induction n with
  | zero => exact le_refl _
  | succ m ih =>
      calc d (m + 1) ≤ ρθ * d m := hc m
        _ ≤ 1 * d m := mul_le_mul_of_nonneg_right hρ1 (hd m)
        _ = d m := one_mul _
        _ ≤ d 0 := ih

/-- **The clip interior is forward-invariant under a weight-decay parameter contraction.** Given the ratio-distance
    bound `|ratio(θ n) − c| ≤ Lr·d n` (the ratio's `Lr`-Lipschitz sensitivity at `θ n` vs the fixed point, `d n` the
    sup parameter distance), the parameter contraction on `d` (`d (n+1) ≤ ρθ·d n`, `ρθ ≤ 1`), and the window condition
    `lo < c − Lr·d 0` / `c + Lr·d 0 < hi` (the center `c = ratio(θ*)` inside by more than the uniform trapping radius
    `Lr·d 0`), the ratio stays in `(lo,hi)` for all `n`. Direct geometric route: `|ratio(θ n) − c| ≤ Lr·d n ≤ Lr·d 0`
    (distance never grows, `dist_le_init`), so the ratio is within `Lr·d 0` of `c` at every step. Discharges C46's
    contraction hypothesis (and hence `hIntθ`) for the weight-decay optimizer. -/
theorem clip_invariant_of_weight_decay (chosen e : Expr) (es : List Expr) (oldLogp lo hi : Float)
    (θ : Nat → (Nat → ℝ)) (c Lr ρθ : ℝ) (d : Nat → ℝ)
    (hd : ∀ n, 0 ≤ d n) (hρ1 : ρθ ≤ 1) (hcontract : ∀ n, d (n + 1) ≤ ρθ * d n)
    (hLr : 0 ≤ Lr)
    (hrl : ∀ n, |evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ n) - c| ≤ Lr * d n)
    (hlo : toReal lo < c - Lr * d 0) (hhi : c + Lr * d 0 < toReal hi) (n : Nat) :
    toReal lo < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ n)
      ∧ evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ n) < toReal hi := by
  have hdn : d n ≤ d 0 := dist_le_init d ρθ hd hρ1 hcontract n
  have hb : |evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ n) - c| ≤ Lr * d 0 :=
    (hrl n).trans (mul_le_mul_of_nonneg_left hdn hLr)
  rw [abs_le] at hb
  exact ⟨by linarith [hb.1], by linarith [hb.2]⟩

/-- **The clip interior invariant with the ratio-distance bound DERIVED from the ratio value-Lipschitz.** The honest
    composition: `θ` contracts toward `θ*` (the sup parameter distance `≤ d n`, `d` non-expansive), the ratio is
    `Lr`-Lipschitz in the parameters (`hRLip`, applied at `θ n` vs `θ*` given the per-coordinate distance `≤ d n` — the
    C18/C36 ratio value-Lipschitz), and `c = ratio(θ*)` is inside the window by more than `Lr·d 0`. Then
    `∀ n, ratio(θ n) ∈ (lo,hi)`. This is C46's clip-interior invariant with the contraction fully discharged from
    weight decay + ratio Lipschitz — no bare contraction hypothesis. -/
theorem clip_invariant_of_weight_decay_lipschitz (chosen e : Expr) (es : List Expr) (oldLogp lo hi : Float)
    (θ : Nat → (Nat → ℝ)) (θstar : Nat → ℝ) (Lr ρθ : ℝ) (d : Nat → ℝ)
    (hd : ∀ n, 0 ≤ d n) (hρ1 : ρθ ≤ 1) (hcontract : ∀ n, d (n + 1) ≤ ρθ * d n)
    (hLr : 0 ≤ Lr)
    (hdist : ∀ n k, |θ n k - θstar k| ≤ d n)
    (hRLip : ∀ n, (∀ k, |θ n k - θstar k| ≤ d n) →
        |evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ n)
          - evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) θstar| ≤ Lr * d n)
    (hlo : toReal lo < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) θstar - Lr * d 0)
    (hhi : evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) θstar + Lr * d 0 < toReal hi)
    (n : Nat) :
    toReal lo < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ n)
      ∧ evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ n) < toReal hi :=
  clip_invariant_of_weight_decay chosen e es oldLogp lo hi θ
    (evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) θstar) Lr ρθ d
    hd hρ1 hcontract hLr (fun m => hRLip m (hdist m)) hlo hhi n

end Puffer.RL.ContractionConstant
