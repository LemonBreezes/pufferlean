/-
# Trajectory reachability: forward-invariance (trapping-region) discharge of the whole-run region conditions

C28 (`WholeRunFromC26`) proved the whole-run error interval GIVEN the trajectory stays in C26's valid region at
every step — the `∀ p` invariance slate `hRegσ`/`hRegσ'`/`hIntσ`/`hIntσ'`/`hMσ`/… was ASSUMED wholesale (a global
"stays in region forever" hypothesis). C29 (`RegionInvariance`) observed the param-bound part is preserved by
PROJECTED ascent. This module supplies the general reduction that makes those assumptions honest: reachability is
never proved "from nothing" (data-dependent, and false in general for expansive dynamics) — it is proved as
FORWARD-INVARIANCE of a TRAPPING REGION (`P(θ 0)` + one-step invariance ⟹ `∀ n, P (θ n)`).

* `traj_invariant` — the backbone: for ANY predicate `P` and update `step`, from `P (θ 0)`, the update law
  `θ (n+1) = step (θ n)`, and one-step invariance `∀ σ, P σ → P (step σ)`, conclude `∀ n, P (θ n)`. This is the
  standard reduction of `∀ n`-reachability to a LOCAL trapping condition (plain `Nat` induction).

* `InRegVal` / `InReg1` — the value-level region predicate at a point (ratio in the clip interior, entropy value
  and derivative budgets) and its bundle with param-boundedness.

* `ppo_whole_run_reachable` — the projected whole-run interval (C29) for the concrete softmax-MLP PPO objective
  (C28's `Gtot`), where the IDEAL trajectory's region membership is no longer assumed:
    - **param-boundedness DISCHARGED** — `projAscentE_mem` makes `|projAscentE … σ k| ≤ R` hold UNCONDITIONALLY, so
      the param clause of the ideal's one-step invariance is FREE (projection), not a hypothesis;
    - **value-level membership REDUCED** — the ideal's clip-interior + entropy conditions `∀ p` come from
      `traj_invariant` applied to `InReg1`, i.e. from `InReg1 (θ' 0)` (start in region) + a LOCAL one-step trapping
      premise `hTrap` (the region is trapping for the projected update), replacing C28's global ∀n assumption.
  The per-point gradient-Lipschitz `hGtraj` that C29's interval needs is discharged at each step by C28's
  `ppoObjE_gradlip_Gdelta`.

**Scope (honestly disclosed).** This does NOT prove unconditional reachability (a concrete trajectory need not stay
in any fixed region — plain ascent is expansive, C29). It provides the correct forward-invariance FORM: the ideal
reference trajectory's param-boundedness is a genuine THEOREM (projection), and its value-level membership is
reduced from "assume it holds at all `n`" to "start in the region + the region is one-step-trapping under the
projected update" (`hTrap`) — a LOCAL, per-step-checkable condition rather than a global orbit assumption. The
runnable (Float) trajectory `θ` is only known within `B`/step of the projected update (non-deterministic in ℝ), so
its region membership stays as per-step premises (`hRegθ`/`hIntθ`/`hMθ`/`hDmEθ`) — exactly the quantities the
`puffer verify` harness checks numerically each step. The coupling budgets `hLvE`/`hDlE` (value/derivative
Lipschitz between the two trajectories) are the network's `Smooth` constants over the `R`-region, not reachability
conditions, and remain premises. `B` (per-step interval, C1), weight-clamping-vs-gradient-clipping, and the
exact-ℝ-ascent-vs-Muon/Adam gap are as in C27–C29.
-/
import Puffer.RL.RegionInvariance
import Puffer.RL.WholeRunFromC26
open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)
open Puffer.RL.RegionInvariance (projAscentE projAscentE_mem projAscentE_traj_bounded proj_whole_run_sup_interval)
open Puffer.RL.WholeRunFromC26 (ppoObjE Gtot ppoObjE_gradlip_Gdelta)
open Puffer.RL.SoftmaxExpr (logSoftmaxE)
open Puffer.RL.SurrogateExpr (ratioE)

namespace Puffer.RL.TrajReachability

/-- **The forward-invariance backbone.** For any predicate `P` on parameter vectors and any update map `step`, a
    trajectory `θ` following `step` (`θ (n+1) = step (θ n)`) that starts in `P` and whose `P` is one-step-invariant
    (`∀ σ, P σ → P (step σ)`) satisfies `P` at every step. Plain `Nat` induction — the standard reduction of
    `∀ n`-reachability to a LOCAL trapping condition. Everything downstream (param-boundedness, clip-interior,
    entropy budgets) is an instance. -/
theorem traj_invariant {P : (Nat → ℝ) → Prop} {step : (Nat → ℝ) → (Nat → ℝ)}
    (θ : Nat → (Nat → ℝ)) (h0 : P (θ 0))
    (hstep : ∀ n, θ (n + 1) = step (θ n)) (hinv : ∀ σ, P σ → P (step σ)) :
    ∀ n, P (θ n) := by
  intro n
  induction n with
  | zero => exact h0
  | succ p ih => rw [hstep p]; exact hinv (θ p) ih

/-- **The value-level region predicate at a point.** The PPO ratio lies in the clip interior `(lo, hi)`, and every
    log-prob's value `|evalR lp σ| ≤ Ment` and derivative `|derivR lp σ k| ≤ Dment` (all `k`) meet the entropy
    budgets. These are the value-level parts of C26's applicability conditions — the ones NOT closed by projection
    (they depend on the objective's value, not param-boundedness alone). -/
def InRegVal (chosen e : Expr) (es logps : List Expr) (oldLogp lo hi : Float) (Ment Dment : ℝ)
    (σ : Nat → ℝ) : Prop :=
  (toReal lo < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ)
  ∧ (evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ < toReal hi)
  ∧ (∀ lp ∈ logps, |evalR lp σ| ≤ Ment)
  ∧ (∀ k, ∀ lp ∈ logps, |derivR lp σ k| ≤ Dment)

/-- **The full region predicate**: param-boundedness `|σ i| ≤ R` conjoined with the value-level `InRegVal`. This is
    the `P` fed to `traj_invariant`; its param clause is discharged by projection (`projAscentE_mem`), so only the
    value clauses need a trapping premise from the caller. -/
def InReg1 (chosen e : Expr) (es logps : List Expr) (oldLogp lo hi : Float) (R Ment Dment : ℝ)
    (σ : Nat → ℝ) : Prop :=
  (∀ i, |σ i| ≤ R) ∧ InRegVal chosen e es logps oldLogp lo hi Ment Dment σ

/-- **THE REACHABLE WHOLE-RUN INTERVAL.** The projected whole-run interval (C29) for the concrete PPO objective
    `ppoObjE` (gradient-Lipschitz constant `Gtot`, C28), with the IDEAL trajectory's region membership no longer
    assumed wholesale:

    * **param-boundedness of the ideal is a THEOREM** — the param clause of the ideal's one-step invariance is
      supplied UNCONDITIONALLY by `projAscentE_mem` (projection lands in `[−R, R]` by construction), not taken as a
      hypothesis;
    * **value-level membership of the ideal is REDUCED to a local trapping premise** — `hVal0` (the ideal starts in
      the region) plus `hTrap` (the region is one-step-invariant under the projected update) give `∀ p`
      clip-interior + entropy via `traj_invariant`, replacing C28's global "stays in region for all `n`" assumption.

    The runnable `θ` stays within `B` of the projected update each step; its region membership
    (`hRegθ`/`hIntθ`/`hMθ`/`hDmEθ`) and the coupling budgets (`hLvE`/`hDlE`) remain per-step premises (see the module
    scope note). The conclusion is C27's geometric interval `L^n·d0 + B·Σ_{j<n} L^j`, `L = 1 + |lr|·Gtot`. -/
theorem ppo_whole_run_reachable (chosen e V : Expr) (es logps : List Expr)
    (oldLogp g lo hi cv ce ret lr : Float) (B d0 L R Ment Lvent Dment Dlent : ℝ)
    (hch : Smooth chosen) (he : Smooth e) (hes : ∀ lp ∈ es, Smooth lp) (hV : Smooth V) (hR : 0 ≤ R)
    (θ θ' : Nat → (Nat → ℝ))
    (hL : L = 1 + |toReal lr| * Gtot chosen e V es logps oldLogp g cv ce ret R Ment Lvent Dment Dlent)
    (hL0 : 0 ≤ L)
    (hideal : ∀ n, θ' (n + 1)
        = projAscentE (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr R (θ' n))
    (hstep : ∀ n k, |θ (n + 1) k
        - projAscentE (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr R (θ n) k| ≤ B)
    (hd0 : ∀ k, |θ 0 k - θ' 0 k| ≤ d0)
    -- the ideal starts in the region (param-bound + value-level):
    (h0θ'bound : ∀ i, |θ' 0 i| ≤ R)
    (hVal0 : InRegVal chosen e es logps oldLogp lo hi Ment Dment (θ' 0))
    -- the region is one-step-trapping for the projected update (the LOCAL reachability premise):
    (hTrap : ∀ σ, InReg1 chosen e es logps oldLogp lo hi R Ment Dment σ →
        InRegVal chosen e es logps oldLogp lo hi Ment Dment
          (projAscentE (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr R σ))
    -- the runnable trajectory's per-step (runtime-checkable) region membership:
    (hRegθ : ∀ p i, |θ p i| ≤ R)
    (hIntθ : ∀ p, toReal lo < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ p)
        ∧ evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ p) < toReal hi)
    (hMθ : ∀ p, ∀ lp ∈ logps, |evalR lp (θ p)| ≤ Ment)
    (hDmEθ : ∀ p k, ∀ lp ∈ logps, |derivR lp (θ p) k| ≤ Dment)
    -- coupling budgets (network `Smooth` constants over the region), as in C28:
    (hLvE : ∀ p (δ : ℝ), (∀ i, |θ p i - θ' p i| ≤ δ) →
        ∀ lp ∈ logps, |evalR lp (θ p) - evalR lp (θ' p)| ≤ Lvent * δ)
    (hDlE : ∀ p (δ : ℝ) k, (∀ i, |θ p i - θ' p i| ≤ δ) →
        ∀ lp ∈ logps, |derivR lp (θ p) k - derivR lp (θ' p) k| ≤ Dlent * δ)
    (hDmEn : 0 ≤ Dment) (n k : Nat) :
    |θ n k - θ' n k| ≤ L ^ n * d0 + B * ∑ j ∈ Finset.range n, L ^ j := by
  -- one-step invariance of the FULL bundle: param via projection (FREE), value via the caller's trapping premise
  have hstepinv : ∀ σ, InReg1 chosen e es logps oldLogp lo hi R Ment Dment σ →
      InReg1 chosen e es logps oldLogp lo hi R Ment Dment
        (projAscentE (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr R σ) :=
    fun σ hσ => ⟨projAscentE_mem _ lr R σ hR, hTrap σ hσ⟩
  -- the ideal stays in the region for ALL p (forward-invariance from start + trapping):
  have hInv : ∀ p, InReg1 chosen e es logps oldLogp lo hi R Ment Dment (θ' p) :=
    traj_invariant θ'
      (⟨h0θ'bound, hVal0⟩ : InReg1 chosen e es logps oldLogp lo hi R Ment Dment (θ' 0)) hideal hstepinv
  -- feed C29's projected interval; discharge its per-point gradient-Lipschitz by C28's Gtot bound
  refine proj_whole_run_sup_interval (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr R
    (Gtot chosen e V es logps oldLogp g cv ce ret R Ment Lvent Dment Dlent) B d0 L θ θ' hL
    ?_ hideal hstep hd0 hL0 n k
  intro p δ hclose k'
  exact ppoObjE_gradlip_Gdelta chosen e V es logps oldLogp g lo hi cv ce ret hch he hes hV
    (θ p) (θ' p) R δ k'
    (hRegθ p) (hInv p).1 hclose hR
    (hIntθ p).1 (hIntθ p).2 (hInv p).2.1 (hInv p).2.2.1
    Ment Lvent Dment Dlent
    (hMθ p) (hInv p).2.2.2.1 (hLvE p δ hclose)
    (hDmEθ p k') ((hInv p).2.2.2.2 k') (hDlE p δ k' hclose) hDmEn

end Puffer.RL.TrajReachability
