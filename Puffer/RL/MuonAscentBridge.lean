/-
# Bridging the whole-run error interval toward the actual Muon optimizer

The whole-run error interval (C27 `WholeRunInterval.ppo_whole_run_sup_interval`) models the ideal trajectory as
EXACT-ℝ gradient ascent (`gradAscentE e lr σ = σ + lr·∇e(σ)`) and accumulates the runnable step's deviation `B` into
`L^n·d0 + B·Σ_{j<n} L^j`, with `L = 1 + |lr|·G` the ascent-step Lipschitz. The real `puffer` trainer uses MUON
(Newton–Schulz orthogonalization of the momentum), NOT plain gradient ascent. This module makes the bridge honest.

**The key structural fact: the accumulation is OPTIMIZER-AGNOSTIC.** The geometric interval `L^n·d0 + B·Σ L^j` never
used any property of `gradAscentE` beyond (i) the ideal step map is `L`-Lipschitz and (ii) the runnable step stays
within `B` of the ideal step per iteration. So the SAME machinery yields a whole-run interval for Muon — once Muon's
own step-Lipschitz `L` and Float-vs-ℝ per-step error `B` are supplied. This is precisely what C2's generic
`nstep_trajectory_error` (`MuonTrainBound`, over any step map `F` and distance `d`) already provides; this file
exposes it in the two metrics the trainer uses.

* `whole_run_sup_interval_of_step` — the PER-COORDINATE whole-run interval for an ARBITRARY step map `F` with a
  sup-metric Lipschitz hypothesis `hlip` (`(∀ i, |σ i − σ' i| ≤ δ) → ∀ k, |F σ k − F σ' k| ≤ L·δ`). This is exactly
  C27's `ppo_whole_run_sup_interval` proof with `gradAscentE e lr` abstracted to `F` and `gradAscentE_sup_lipschitz`
  abstracted to `hlip` — so it uniformly covers gradient ascent (C27, `L = 1 + |lr|·G`), projected ascent (C29,
  `projAscentE`), AND a Muon step map `Fμ` (with `L` Muon's own sup-Lipschitz). Reuses C27's `errBound` and C2's
  `affine_recur_le`.
* `muon_whole_run_opnorm_interval` — the NORM-metric whole-run interval for an arbitrary `L`-Lipschitz step map `F`
  on a normed space (the natural setting for Muon, whose per-step bounds live in the matrix OPERATOR norm). A direct
  instantiation of C2's `nstep_trajectory_error` with `d x y = ‖x − y‖`, discharging its triangle hypothesis via
  `norm_add_le`. Instantiate `F` with the exact-ℝ Muon step, `L` with its operator-norm Lipschitz, `B` with the
  Float-vs-ℝ Muon per-step error to obtain the Muon whole-run interval.

**Scope (honestly disclosed).** These package the whole-run accumulation for a Muon (or any) step map; they do NOT
prove Muon ≈ gradient ascent — it is NOT. Muon changes the update DIRECTION (it orthogonalizes the momentum via
Newton–Schulz), so it is not a small perturbation of `gradAscentE`, and C27's `L = 1 + |lr|·G` does NOT transfer:
the Muon interval needs Muon's OWN step-Lipschitz `L` (a bound on the Newton–Schulz map's sensitivity, which is
nonlinear — the genuinely remaining ingredient, not supplied here). The per-step error `B` (how far the Float Muon
step strays from the exact-ℝ Muon step) is dischargeable by the existing operator-norm Muon Float-error machinery
(`MuonStepBound.stepMat_opNorm_le`, `MuonRuntime`'s `nsScalarF_error`) — modulo the operator-norm ↔ sup-metric
conversion for the per-coordinate form. So this is the honest bridge: the whole-run interval holds for Muon with
`L`/`B` its two ingredients, reducing the Muon accuracy statement to (Muon step Lipschitz) + (Muon Float per-step
error) — exactly as the gradient-ascent interval reduced to (`1 + |lr|·G`) + (`B` from C1).
-/
import Puffer.RL.WholeRunInterval
import Puffer.RL.MuonTrainBound

open Puffer.FloatR (toReal)
open Puffer.RL.WholeRunInterval (errBound)
open Puffer.RL.MuonTrainBound (affine_recur_le nstep_trajectory_error)

namespace Puffer.RL.MuonAscentBridge

/-- **Whole-run interval for an ARBITRARY step map (per-coordinate / sup-metric).** For the runnable trajectory `θ`
    and the ideal trajectory `θ'` following an arbitrary step map `F` exactly (`θ'(n+1) = F (θ' n)`), given `F` is
    `L`-Lipschitz in the sup-metric (`hlip`: δ-close inputs map to `L·δ`-close outputs, per coordinate), the runnable
    step stays within `B` of `F` per iteration (`hstep`), and the initial divergence is `d0`, then after any `n`
    steps `|θ n k − θ' n k| ≤ L^n·d0 + B·Σ_{j<n} L^j` per coordinate. This is C27's `ppo_whole_run_sup_interval` with
    the concrete `gradAscentE e lr` / `gradAscentE_sup_lipschitz` abstracted to `F` / `hlip` — so it covers gradient
    ascent (C27), projected ascent (C29), and a Muon step map alike, the choice of optimizer entering ONLY through
    `L` and `B`. Reuses C27's `errBound` and C2's `affine_recur_le`. -/
theorem whole_run_sup_interval_of_step (F : (Nat → ℝ) → (Nat → ℝ)) (B d0 L : ℝ)
    (θ θ' : Nat → (Nat → ℝ))
    (hlip : ∀ (σ σ' : Nat → ℝ) (δ : ℝ), (∀ i, |σ i - σ' i| ≤ δ) → ∀ k, |F σ k - F σ' k| ≤ L * δ)
    (hideal : ∀ n, θ' (n + 1) = F (θ' n))
    (hstep : ∀ n k, |θ (n + 1) k - F (θ n) k| ≤ B)
    (hd0 : ∀ k, |θ 0 k - θ' 0 k| ≤ d0)
    (hL0 : 0 ≤ L) (n k : Nat) :
    |θ n k - θ' n k| ≤ L ^ n * d0 + B * ∑ j ∈ Finset.range n, L ^ j := by
  have hbound : ∀ m j, |θ m j - θ' m j| ≤ errBound L B d0 m := by
    intro m
    induction m with
    | zero => intro j; exact hd0 j
    | succ p ih =>
        intro j
        have hlp : ∀ i, |F (θ p) i - F (θ' p) i| ≤ L * errBound L B d0 p :=
          hlip (θ p) (θ' p) (errBound L B d0 p) ih
        calc |θ (p + 1) j - θ' (p + 1) j|
            = |θ (p + 1) j - F (θ' p) j| := by rw [hideal p]
          _ ≤ |θ (p + 1) j - F (θ p) j| + |F (θ p) j - F (θ' p) j| := abs_sub_le _ _ _
          _ ≤ B + L * errBound L B d0 p := add_le_add (hstep p j) (hlp j)
          _ = errBound L B d0 (p + 1) := by simp only [errBound]; ring
  have hrec : ∀ j, errBound L B d0 (j + 1) ≤ L * errBound L B d0 j + B :=
    fun j => le_of_eq (by simp only [errBound])
  have hkey := affine_recur_le (errBound L B d0) L B hL0 hrec n
  have h0 : errBound L B d0 0 = d0 := rfl
  rw [h0] at hkey
  exact (hbound n k).trans hkey

/-- **The Muon whole-run interval, per-coordinate.** Specialization of `whole_run_sup_interval_of_step` to the
    exact-ℝ Muon step map `Fμ`: given `Fμ` is `L`-Lipschitz in the sup-metric (`L` = Muon's OWN step-Lipschitz — the
    sensitivity of the Newton–Schulz-orthogonalized update, NOT `1 + |lr|·G`) and the runnable Float Muon step stays
    within `B` of `Fμ` per iteration (`B` = the Float-vs-ℝ Muon per-step error), the runnable Muon trajectory stays
    within `L^n·d0 + B·Σ_{j<n} L^j` of the exact-ℝ Muon trajectory. The Muon accuracy statement reduced to its two
    ingredients — structurally identical to the gradient-ascent interval, with the optimizer entering only through
    `L`/`B`. -/
theorem muon_whole_run_sup_interval (Fμ : (Nat → ℝ) → (Nat → ℝ)) (B d0 L : ℝ)
    (θ θ' : Nat → (Nat → ℝ))
    (hlip : ∀ (σ σ' : Nat → ℝ) (δ : ℝ), (∀ i, |σ i - σ' i| ≤ δ) → ∀ k, |Fμ σ k - Fμ σ' k| ≤ L * δ)
    (hideal : ∀ n, θ' (n + 1) = Fμ (θ' n))
    (hstep : ∀ n k, |θ (n + 1) k - Fμ (θ n) k| ≤ B)
    (hd0 : ∀ k, |θ 0 k - θ' 0 k| ≤ d0)
    (hL0 : 0 ≤ L) (n k : Nat) :
    |θ n k - θ' n k| ≤ L ^ n * d0 + B * ∑ j ∈ Finset.range n, L ^ j :=
  whole_run_sup_interval_of_step Fμ B d0 L θ θ' hlip hideal hstep hd0 hL0 n k

/-- **The Muon whole-run interval, norm-metric.** For an arbitrary `L`-Lipschitz step map `F` on a normed space `E`
    (the natural setting for Muon, whose per-step bounds live in the matrix OPERATOR norm), with the runnable
    trajectory `θ` within `B` of `F` per step and the ideal `θ'` following `F` exactly, the divergence after `n`
    steps is `‖θ n − θ' n‖ ≤ L^n·‖θ 0 − θ' 0‖ + B·Σ_{j<n} L^j`. A direct instantiation of C2's generic
    `nstep_trajectory_error` with the norm distance `d x y = ‖x − y‖` (its triangle hypothesis discharged via
    `norm_add_le` on `x − z = (x − y) + (y − z)`). Instantiate `F` with the exact-ℝ Muon step, `L` with its
    operator-norm Lipschitz, and `B` with the Float-vs-ℝ Muon per-step operator-norm error. -/
theorem muon_whole_run_opnorm_interval {E : Type*} [NormedAddCommGroup E]
    (F : E → E) (θ θ' : Nat → E) (L B : ℝ) (hL : 0 ≤ L)
    (hlip : ∀ x y, ‖F x - F y‖ ≤ L * ‖x - y‖)
    (hstep : ∀ k, ‖θ (k + 1) - F (θ k)‖ ≤ B)
    (hideal : ∀ k, θ' (k + 1) = F (θ' k)) (n : Nat) :
    ‖θ n - θ' n‖ ≤ L ^ n * ‖θ 0 - θ' 0‖ + B * ∑ k ∈ Finset.range n, L ^ k :=
  nstep_trajectory_error (fun x y => ‖x - y‖) F θ θ' L B hL
    (fun x y z => by
      show ‖x - z‖ ≤ ‖x - y‖ + ‖y - z‖
      rw [show x - z = (x - y) + (y - z) by abel]; exact norm_add_le _ _)
    hlip hstep hideal n

end Puffer.RL.MuonAscentBridge
