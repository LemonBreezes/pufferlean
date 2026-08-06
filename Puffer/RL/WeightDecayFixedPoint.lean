/-
# Weight-decay fixed point and geometric convergence: discharging C49's contraction premise

C49 (`ContractionConstant`) closed C46's clip-interior invariant for the weight-decay optimizer, but TOOK as
hypotheses (i) a fixed point `θ*` of the weight-decay step and (ii) the parameter trajectory contracting toward it
(`d (n+1) ≤ ρθ·d n`). This module PROVIDES those two ingredients:

* **Geometric contraction toward a GIVEN fixed point** (`traj_contracts_to_fixedPoint`) — no completeness needed.
  For ANY step map that is `K`-Lipschitz in the sup-metric (C32's `wdAscentE_sup_lipschitz` shape) with a fixed
  point `θ*` (`step θ* = θ*`), the trajectory `θ (n+1) = step (θ n)` contracts geometrically:
  `∀ n k, |θ n k − θ* k| ≤ K^n · d0` (`d0` the initial distance bound). So `d n := K^n·d0` satisfies C49's
  `d (n+1) = K·d n` and `hdist : |θ n k − θ* k| ≤ d n` DIRECTLY.
* **Weight-decay specialization** (`wdAscentE_fixedPoint_contracts`) — the same for C32's `wdAscentE` with
  `K = |1 − wd| + |lr|·G` (via `wdAscentE_sup_lipschitz`), given a fixed point and a global gradient-Lipschitz `G`.
* **Fixed-point EXISTENCE via Banach** (`banach_fixedPoint_exists`) — Mathlib's `ContractingWith` gives that a
  `K`-Lipschitz self-map with `K < 1` on a nonempty COMPLETE (E)metric space has a fixed point `∃ x, f x = x`.
* **Capstone** (`clip_invariant_of_weight_decay_fixedPoint`) — composes the geometric contraction with C49's
  `clip_invariant_of_weight_decay_lipschitz`: for the weight-decay trajectory with a fixed point `θ*`, the ratio
  Lipschitz `Lr`, and `ratio(θ*)` inside the clip window by more than `Lr·d0`, `∀ n, ratio(θ n) ∈ (lo,hi)` — C49's
  clip invariant with the CONTRACTION now fully discharged from `wdAscentE`'s structure (no bare contraction
  hypothesis).

**Scope (honestly disclosed).** The geometric contraction toward a GIVEN fixed point (points 1/2/4) is
unconditional (only the `K`-Lipschitz step + the fixed point + the initial-distance bound), and it discharges
C49's `hcontract`/`hdist` for the weight-decay optimizer. The fixed-point EXISTENCE (`banach_fixedPoint_exists`) is
proved ABSTRACTLY for any nonempty complete (E)metric space via Banach; WIRING it to the concrete weight-decay step
requires a complete metric-space instance on the parameters — the codebase's `Nat → ℝ` with the sup metric is not a
standard complete instance (one would instantiate at the FINITE parameter space `Fin d → ℝ`, complete via
`Pi.completeSpace`, and cast `wdAscentE` to a `ContractingWith` map, using `wd_L_lt_one` for `K < 1`). So the
existence is available abstractly and the contraction-to-a-fixed-point is concrete; connecting them at the finite
parameter type is the remaining glue. The `K < 1` weight-decay condition (`wd > |lr|·G`) is C32's; `Lr` is the
network's log-softmax+exp ratio budget; `ratio(θ*) ∈ (lo+Lr·d0, hi−Lr·d0)` is the one-point checkable margin (C49).
The fixed point is a stationary point of the weight-decay-regularized objective (`wd·θ* = lr·∇(θ*)`): its LOCATION
is data-dependent, its EXISTENCE is Banach.
-/
import Puffer.RL.ContractionConstant
import Puffer.RL.WeightDecayInterval
import Mathlib.Topology.MetricSpace.Contracting
open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)
open Puffer.RL.SurrogateExpr (ratioE)
open Puffer.RL.SoftmaxExpr (logSoftmaxE)
open Puffer.RL.WeightDecayInterval (wdAscentE wdAscentE_sup_lipschitz)
open Puffer.RL.ContractionConstant (clip_invariant_of_weight_decay_lipschitz)

namespace Puffer.RL.WeightDecayFixedPoint

/-- **Geometric contraction toward a given fixed point** (no completeness needed). For a step map that is
    `K`-Lipschitz in the sup-metric (`∀ σ σ' δ, (∀ i, |σ i − σ' i| ≤ δ) → ∀ k, |step σ k − step σ' k| ≤ K·δ`) with a
    fixed point `θ*` (`step θ* = θ*`), the trajectory `θ (n+1) = step (θ n)` from a start within `d0` of `θ*`
    contracts geometrically: `∀ n k, |θ n k − θ* k| ≤ K^n · d0`. Induction: one step multiplies the distance bound by
    `K` (Lipschitz + the fixed point). Directly supplies C49's `d n := K^n·d0` (with `d (n+1) = K·d n`, `hdist`). -/
theorem traj_contracts_to_fixedPoint
    (step : (Nat → ℝ) → (Nat → ℝ)) (θ : Nat → (Nat → ℝ)) (θstar : Nat → ℝ) (K d0 : ℝ)
    (hfix : ∀ k, step θstar k = θstar k)
    (htraj : ∀ n, θ (n + 1) = step (θ n))
    (hLip : ∀ (σ σ' : Nat → ℝ) (δ : ℝ), (∀ i, |σ i - σ' i| ≤ δ) → ∀ k, |step σ k - step σ' k| ≤ K * δ)
    (hinit : ∀ k, |θ 0 k - θstar k| ≤ d0) :
    ∀ n k, |θ n k - θstar k| ≤ K ^ n * d0 := by
  intro n
  induction n with
  | zero => intro k; simpa using hinit k
  | succ m ih =>
      intro k
      rw [htraj m, ← hfix k]
      calc |step (θ m) k - step θstar k|
          ≤ K * (K ^ m * d0) := hLip (θ m) θstar (K ^ m * d0) ih k
        _ = K ^ (m + 1) * d0 := by ring

/-- **Weight-decay specialization.** For C32's `wdAscentE e lr wd` with a global gradient-Lipschitz `G` and a fixed
    point `θ*`, the trajectory contracts toward `θ*` with `K = |1 − wd| + |lr|·G`: `∀ n k, |θ n k − θ* k| ≤
    (|1 − wd| + |lr|·G)^n · d0`. Uses `wdAscentE_sup_lipschitz` for the per-step Lipschitz. Discharges C49's
    `hcontract`/`hdist` for weight decay. -/
theorem wdAscentE_fixedPoint_contracts (eObj : Expr) (lr wd : Float) (G d0 : ℝ)
    (θ : Nat → (Nat → ℝ)) (θstar : Nat → ℝ)
    (hG : ∀ (σ σ' : Nat → ℝ) (δ : ℝ), (∀ i, |σ i - σ' i| ≤ δ) →
        ∀ k, |derivR eObj σ k - derivR eObj σ' k| ≤ G * δ)
    (hfix : ∀ k, wdAscentE eObj lr wd θstar k = θstar k)
    (htraj : ∀ n, θ (n + 1) = wdAscentE eObj lr wd (θ n))
    (hinit : ∀ k, |θ 0 k - θstar k| ≤ d0) :
    ∀ n k, |θ n k - θstar k| ≤ (|1 - toReal wd| + |toReal lr| * G) ^ n * d0 :=
  traj_contracts_to_fixedPoint (wdAscentE eObj lr wd) θ θstar (|1 - toReal wd| + |toReal lr| * G) d0
    hfix htraj
    (fun σ σ' δ hclose => wdAscentE_sup_lipschitz eObj lr wd G δ σ σ' hclose (fun k => hG σ σ' δ hclose k))
    hinit

/-- **Fixed-point existence via Banach.** A `K`-Lipschitz self-map with `K < 1` (`ContractingWith K f`) on a nonempty
    COMPLETE (E)metric space has a fixed point: `∃ x, f x = x` (Mathlib's `ContractingWith.fixedPoint`). This supplies
    the EXISTENCE of the fixed point `θ*` that `traj_contracts_to_fixedPoint` and C49 take as given — abstractly, for
    any complete space (wiring to the concrete weight-decay parameters needs a complete instance, e.g. `Fin d → ℝ`). -/
theorem banach_fixedPoint_exists {α : Type*} [MetricSpace α] [CompleteSpace α] [Nonempty α]
    {K : NNReal} {f : α → α} (hf : ContractingWith K f) : ∃ x, f x = x :=
  ⟨ContractingWith.fixedPoint f hf, hf.fixedPoint_isFixedPt⟩

/-- **CAPSTONE: clip-interior invariant for weight decay with the contraction discharged from a fixed point.**
    Composes `wdAscentE_fixedPoint_contracts` (the geometric contraction) with C49's
    `clip_invariant_of_weight_decay_lipschitz`. For the weight-decay trajectory `θ` toward a fixed point `θ*`
    (`K = |1 − wd| + |lr|·G ≤ 1`), with the ratio's parameter-Lipschitz `Lr` and `ratio(θ*)` inside the clip window by
    more than the trapping radius `Lr·d0`, the ratio stays in `(lo,hi)` for ALL `n` — C46/C49's clip-interior
    invariant (`hIntθ`), now with the contraction fully discharged from `wdAscentE`'s structure + the fixed point (no
    bare contraction hypothesis; the fixed point's existence is Banach, `banach_fixedPoint_exists`). -/
theorem clip_invariant_of_weight_decay_fixedPoint
    (chosen e : Expr) (es : List Expr) (oldLogp lo hi : Float)
    (eObj : Expr) (lr wd : Float) (G Lr d0 : ℝ)
    (θ : Nat → (Nat → ℝ)) (θstar : Nat → ℝ)
    (hG : ∀ (σ σ' : Nat → ℝ) (δ : ℝ), (∀ i, |σ i - σ' i| ≤ δ) →
        ∀ k, |derivR eObj σ k - derivR eObj σ' k| ≤ G * δ)
    (hGn : 0 ≤ G) (hd0 : 0 ≤ d0) (hLr : 0 ≤ Lr)
    (hK1 : |1 - toReal wd| + |toReal lr| * G ≤ 1)
    (hfix : ∀ k, wdAscentE eObj lr wd θstar k = θstar k)
    (htraj : ∀ n, θ (n + 1) = wdAscentE eObj lr wd (θ n))
    (hinit : ∀ k, |θ 0 k - θstar k| ≤ d0)
    (hRLip : ∀ n, (∀ k, |θ n k - θstar k| ≤ (|1 - toReal wd| + |toReal lr| * G) ^ n * d0) →
        |evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ n)
          - evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) θstar|
          ≤ Lr * ((|1 - toReal wd| + |toReal lr| * G) ^ n * d0))
    (hlo : toReal lo < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) θstar - Lr * d0)
    (hhi : evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) θstar + Lr * d0 < toReal hi)
    (n : Nat) :
    toReal lo < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ n)
      ∧ evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ n) < toReal hi := by
  have hKn : 0 ≤ |1 - toReal wd| + |toReal lr| * G :=
    add_nonneg (abs_nonneg _) (mul_nonneg (abs_nonneg _) hGn)
  have hcontr := wdAscentE_fixedPoint_contracts eObj lr wd G d0 θ θstar hG hfix htraj hinit
  refine clip_invariant_of_weight_decay_lipschitz chosen e es oldLogp lo hi θ θstar Lr
    (|1 - toReal wd| + |toReal lr| * G) (fun m => (|1 - toReal wd| + |toReal lr| * G) ^ m * d0)
    (fun m => mul_nonneg (pow_nonneg hKn m) hd0) hK1
    (fun m => le_of_eq (by ring)) hLr hcontr hRLip ?_ ?_ n
  · simpa using hlo
  · simpa using hhi

end Puffer.RL.WeightDecayFixedPoint
