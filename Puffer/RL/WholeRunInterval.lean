/-
# THE WHOLE-RUN ERROR INTERVAL: `puffer` trajectory vs infinite-precision ℝ, over N training steps

This wires the concrete whole-objective gradient-Lipschitz constant `G` (C26,
`PPOTotalGradConcrete.ppoTotalObj_softmax_gradient_lipschitz`) into the N-step accumulation machinery (C2/C3,
`MuonTrainBound`), yielding the literal goal of the whole development: a bound on how far the RUNNABLE (`Float`)
training trajectory can drift from the ideal EXACT-ℝ trajectory after any number `n` of gradient-ascent steps.

The bridge `gradAscentE_sup_lipschitz` turns the objective's gradient-Lipschitz `G` (C26's per-coordinate bound
`∀ k, |derivR e σ k − derivR e σ' k| ≤ G·δ` under `∀ i, |σ i − σ' i| ≤ δ`) into the ascent-step Lipschitz
`L = 1 + |lr|·G` in the sup-metric — one exact-ℝ gradient-ascent step `σ ↦ σ + lr·∇e(σ)` moves any two δ-close
parameter vectors to `(1+|lr|·G)·δ`-close vectors. Then `ppo_whole_run_sup_interval` accumulates over `n` steps
via C2's scalar affine recurrence (`affine_recur_le`):

    |θ n k − θ' n k|  ≤  L^n · d₀  +  B · Σ_{j<n} L^j       (per coordinate k, all n)

for the runnable trajectory `θ` and the ideal trajectory `θ'` (which follows the exact-ℝ ascent `θ'(n+1) =
gradAscentE e lr (θ' n)`), given the per-step rounding bound `B` (`|θ(n+1) k − gradAscentE e lr (θ n) k| ≤ B` —
the C1 single-step composition interval) and the initial divergence `d₀`. This is exactly `nstep_trajectory_error`
(C2) specialized to the sup-metric with `L = 1 + |lr|·G` — the whole-run error interval with `G` now the concrete
constant C26 computes from the softmax-MLP's `Smooth` logit budgets.

**Gradient ascent is EXPANSIVE** (`L = 1 + |lr|·G ≥ 1`, since `|lr|·G ≥ 0`), so the honest whole-run bound is the
GEOMETRIC form, not the horizon-free `d₀ + B/(1−L)` (which needs a CONTRACTION `L < 1`, e.g. from strong weight
decay). For `L = 1` (an affine objective, `G = 0`) it is the linear drift `d₀ + n·B`; for `L > 1` it grows like
`L^n·d₀ + B·(L^n − 1)/(L − 1)` — the initial error and the accumulated per-step rounding, both amplified
geometrically by the ascent's expansion. This is the fundamental accuracy statement: the `puffer` binary's output
stays within a computable interval of the infinite-precision computation, the interval being `L^n·d₀ + B·Σ L^j`
with every constant (`L` via C26's `G`, `B` via C1) concrete.

**Scope (honestly disclosed):** the gradient-Lipschitz `hG` is a hypothesis, discharged by C26 — but C26's bound
holds on the network's valid region (params bounded by `R`, clip interior, hidden units active, entropy budgets
uniform), so applying this whole-run bound requires the trajectory to STAY in that region for all `n` (a
reachability/invariance condition not established here). The per-step bound `B` is the C1 reverse-mode composition
interval (its concrete value is that development's domain). The distance is the sup-metric (per coordinate). The
ascent map is exact-ℝ gradient ascent; matching it to the actual `puffer` optimizer (Muon/Adam) is a further step.
So this is the accumulation SKELETON with the ascent-Lipschitz `L = 1 + |lr|·G` discharged concretely — the final
interval is `L^n·d₀ + B·Σ_{j<n} L^j` once `hG` (C26) and `B` (C1) are supplied along the trajectory.
-/
import Puffer.Float.AutoDiffR
import Puffer.RL.MuonTrainBound

open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)
open Puffer.RL.MuonTrainBound (affine_recur_le)

namespace Puffer.RL.WholeRunInterval

/-- One exact-ℝ gradient-ascent step on an `Expr` objective: `σ k ↦ σ k + lr · ∂e/∂(param k)`. -/
noncomputable def gradAscentE (e : Expr) (lr : Float) (σ : Nat → ℝ) : Nat → ℝ :=
  fun k => σ k + toReal lr * derivR e σ k

/-- **THE BRIDGE: gradient-Lipschitz `G` ⟹ ascent-step Lipschitz `L = 1 + |lr|·G` (sup-metric).** If the
    objective's gradient is `G`-Lipschitz — for δ-close inputs (`∀ i, |σ i − σ' i| ≤ δ`) the gradient
    coordinates are `G·δ`-close (`∀ k, |derivR e σ k − derivR e σ' k| ≤ G·δ`, exactly C26's form) — then one
    gradient-ascent step maps δ-close vectors to `(1 + |lr|·G)·δ`-close ones:
    `∀ k, |gradAscentE e lr σ k − gradAscentE e lr σ' k| ≤ (1 + |toReal lr|·G)·δ`. This is C3's
    `ascent_map_lipschitz` in the per-coordinate sup-metric (matching C26), discharging the `hlip` hypothesis of
    the N-step accumulation. Proof: `F σ k − F σ' k = (σ k − σ' k) + lr·(∇e σ k − ∇e σ' k)`, triangle + `abs_mul`. -/
theorem gradAscentE_sup_lipschitz (e : Expr) (lr : Float) (G δ : ℝ) (σ σ' : Nat → ℝ)
    (hclose : ∀ i, |σ i - σ' i| ≤ δ)
    (hG : ∀ k, |derivR e σ k - derivR e σ' k| ≤ G * δ) :
    ∀ k, |gradAscentE e lr σ k - gradAscentE e lr σ' k| ≤ (1 + |toReal lr| * G) * δ := by
  intro k
  simp only [gradAscentE]
  calc |σ k + toReal lr * derivR e σ k - (σ' k + toReal lr * derivR e σ' k)|
      = |(σ k - σ' k) + toReal lr * (derivR e σ k - derivR e σ' k)| := by ring_nf
    _ ≤ |σ k - σ' k| + |toReal lr * (derivR e σ k - derivR e σ' k)| := abs_add_le _ _
    _ = |σ k - σ' k| + |toReal lr| * |derivR e σ k - derivR e σ' k| := by rw [abs_mul]
    _ ≤ δ + |toReal lr| * (G * δ) :=
        add_le_add (hclose k) (mul_le_mul_of_nonneg_left (hG k) (abs_nonneg _))
    _ = (1 + |toReal lr| * G) * δ := by ring

/-- The error-bound sequence: `errBound L B d0 0 = d0`, `errBound L B d0 (n+1) = L·errBound L B d0 n + B` — the
    per-step-amplified accumulation of the initial divergence `d0` and the per-step rounding `B`. -/
noncomputable def errBound (L B d0 : ℝ) : Nat → ℝ
  | 0 => d0
  | n + 1 => L * errBound L B d0 n + B

/-- **THE WHOLE-RUN ERROR INTERVAL.** For the runnable (`Float`) training trajectory `θ` and the ideal (exact-ℝ)
    trajectory `θ'` — which follows the exact-ℝ gradient ascent `θ'(n+1) = gradAscentE e lr (θ' n)` — with the
    objective's gradient `G`-Lipschitz (`hG`, discharged by C26), the per-step rounding bound `B`
    (`|θ(n+1) k − gradAscentE e lr (θ n) k| ≤ B`, the C1 single-step interval), and the initial divergence `d0`,
    after ANY number `n` of steps the two trajectories stay within the geometric interval PER COORDINATE:
    `|θ n k − θ' n k| ≤ L^n · d0 + B · Σ_{j<n} L^j` with `L = 1 + |toReal lr|·G`. This is the whole-run accuracy
    of the runnable trainer against infinite-precision ℝ, with `L` (via C26's concrete `G`) and `B` (via C1)
    computable constants. Gradient ascent is expansive (`L ≥ 1`) so the bound grows geometrically — for `L = 1`
    it is the linear drift `d0 + n·B`. Proof: `gradAscentE_sup_lipschitz` gives the ascent-Lipschitz per step, a
    triangle-inequality induction bounds `|θ n k − θ' n k|` by `errBound L B d0 n`, and C2's `affine_recur_le`
    closes the geometric sum. -/
theorem ppo_whole_run_sup_interval (e : Expr) (lr : Float) (G B d0 L : ℝ)
    (θ θ' : Nat → (Nat → ℝ))
    (hL : L = 1 + |toReal lr| * G)
    (hG : ∀ σ σ' δ, (∀ i, |σ i - σ' i| ≤ δ) → ∀ k, |derivR e σ k - derivR e σ' k| ≤ G * δ)
    (hideal : ∀ n, θ' (n + 1) = gradAscentE e lr (θ' n))
    (hstep : ∀ n k, |θ (n + 1) k - gradAscentE e lr (θ n) k| ≤ B)
    (hd0 : ∀ k, |θ 0 k - θ' 0 k| ≤ d0)
    (hL0 : 0 ≤ L) (n k : Nat) :
    |θ n k - θ' n k| ≤ L ^ n * d0 + B * ∑ j ∈ Finset.range n, L ^ j := by
  have hbound : ∀ m j, |θ m j - θ' m j| ≤ errBound L B d0 m := by
    intro m
    induction m with
    | zero => intro j; exact hd0 j
    | succ p ih =>
        intro j
        have hasc : ∀ i, |gradAscentE e lr (θ p) i - gradAscentE e lr (θ' p) i|
            ≤ L * errBound L B d0 p :=
          hL.symm ▸ gradAscentE_sup_lipschitz e lr G (errBound L B d0 p) (θ p) (θ' p) ih
            (hG (θ p) (θ' p) (errBound L B d0 p) ih)
        calc |θ (p + 1) j - θ' (p + 1) j|
            = |θ (p + 1) j - gradAscentE e lr (θ' p) j| := by rw [hideal p]
          _ ≤ |θ (p + 1) j - gradAscentE e lr (θ p) j|
              + |gradAscentE e lr (θ p) j - gradAscentE e lr (θ' p) j| := abs_sub_le _ _ _
          _ ≤ B + L * errBound L B d0 p := add_le_add (hstep p j) (hasc j)
          _ = errBound L B d0 (p + 1) := by simp only [errBound]; ring
  have hrec : ∀ j, errBound L B d0 (j + 1) ≤ L * errBound L B d0 j + B :=
    fun j => le_of_eq (by simp only [errBound])
  have hkey := affine_recur_le (errBound L B d0) L B hL0 hrec n
  have h0 : errBound L B d0 0 = d0 := rfl
  rw [h0] at hkey
  exact (hbound n k).trans hkey

end Puffer.RL.WholeRunInterval
