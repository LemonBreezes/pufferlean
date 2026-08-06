/-
# The whole-run interval with the gradient-Lipschitz `G` DISCHARGED from C26

C27 (`WholeRunInterval`) gave the whole-run error interval `L^n·d0 + B·Σ_{j<n} L^j` taking the objective's
gradient-Lipschitz `hG` as an abstract (global) hypothesis. This module discharges `hG` directly from C26
(`PPOTotalGradConcrete.ppoTotalObj_softmax_gradient_lipschitz`) for the concrete softmax-MLP PPO objective — so
the whole-run interval holds with `G` the concrete constant C26 computes, with NO free gradient-Lipschitz
hypothesis, only the explicit condition that the trajectory stays in C26's valid region.

Two observations bridge C27 and C26:
* **δ-factoring** (`ppoObjE_gradlip_Gdelta`): C26's bound has a `·δ` in every term, so it is exactly
  `≤ Gtot·δ` with `Gtot` a single fixed constant (the objective's gradient-Lipschitz, δ factored out) — precisely
  the `≤ G·δ` shape `hG` needs.
* **Trajectory-point `hG`** (`ppo_whole_run_sup_interval_traj`): C27's proof only ever uses `hG` at the trajectory
  point pairs `(θ p, θ' p)`, so a per-trajectory-point gradient-Lipschitz suffices — and THAT is dischargeable by
  C26 (which is regional, not global).

`ppo_whole_run_from_C26` is the full instantiation: for the concrete objective `ppoObjE`, given the trajectory
stays in C26's region at every step (params bounded by `R`, ratio in the clip interior, entropy budgets holding —
the explicit INVARIANCE hypotheses), the whole-run interval holds with `G = Gtot`:

    |θ n k − θ' n k|  ≤  L^n · d0 + B · Σ_{j<n} L^j,   L = 1 + |lr|·Gtot.

**Scope (honestly disclosed):** the invariance hypotheses (`hRegσ`/`hRegσ'`/`hIntσ`/`hIntσ'`/`hMσ`/…/`hDlE`) are
exactly C26's applicability conditions, quantified over every trajectory step `p` — the region/clip-interior/
entropy-budget conditions the C27 docstring flagged as an unestablished invariance are now EXPLICIT hypotheses
here. This closes the "instantiate `hG` from C26" gap: the abstract gradient-Lipschitz is replaced by C26's
concrete constant `Gtot`, conditioned on the trajectory-stays-in-region invariance (which itself — proving a
concrete trajectory satisfies it for all `n` — remains a separate reachability question). `B` (the per-step
interval, C1) and the exact-ℝ-ascent-vs-Muon/Adam gap are as in C27.
-/
import Puffer.RL.PPOTotalGradConcrete
import Puffer.RL.WholeRunInterval
open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)
open Puffer.RL.PPOObjectiveExpr (ppoTotalObjE)
open Puffer.RL.ValueEntropyExpr (valueSqErrE entropyCatE)
open Puffer.RL.SurrogateExpr (ppoSurrogateE ratioE)
open Puffer.RL.SoftmaxExpr (logSoftmaxE)
open Puffer.RL.SurrogateConcreteExpr (sfM sfLv sfDm sfDl)
open Puffer.RL.PPOTotalGradConcrete (ppoTotalObj_softmax_gradient_lipschitz)
open Puffer.RL.WholeRunInterval (gradAscentE gradAscentE_sup_lipschitz errBound)
open Puffer.RL.MuonTrainBound (affine_recur_le)

namespace Puffer.RL.WholeRunFromC26

/-- The concrete PPO total objective as an `Expr`: clipped surrogate over a softmax-log-prob ratio, minus the
    value squared error, plus the entropy bonus (the C26 loss). -/
noncomputable def ppoObjE (chosen e V : Expr) (es logps : List Expr)
    (oldLogp g lo hi cv ce ret : Float) : Expr :=
  ppoTotalObjE (ppoSurrogateE (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) g lo hi)
    (valueSqErrE V ret) (entropyCatE logps) cv ce

/-- **C26's whole-objective gradient-Lipschitz constant `Gtot`** — C26's bound with `δ` factored out: the sum of
    the surrogate constant `|g|·exp(sfM−oldLogp)·(sfDl+sfDm·sfLv)`, the value `|cv|·dLip R (valueSqErrE V ret)`,
    and the entropy `|ce|·(length logps)·exp(Ment)·((Ment+1)·Dlent+(Ment+2)·Dment·Lvent)`. -/
noncomputable def Gtot (chosen e V : Expr) (es logps : List Expr)
    (oldLogp g cv ce ret : Float) (R Ment Lvent Dment Dlent : ℝ) : ℝ :=
  |toReal g| * (Real.exp (sfM R chosen e (e :: es) - toReal oldLogp)
      * (sfDl R chosen e (e :: es) + sfDm R chosen e (e :: es) * sfLv R chosen e (e :: es)))
    + |toReal cv| * dLip R (valueSqErrE V ret)
    + |toReal ce| * ((logps.length : ℝ)
        * (Real.exp Ment * ((Ment + 1) * Dlent + (Ment + 2) * Dment * Lvent)))

/-- **C26 restated as `≤ Gtot·δ`** (the regional per-pair gradient-Lipschitz, `δ` factored out) — exactly the
    `≤ G·δ` shape that C27's `hG` consumes. Trivial from C26 by `ring` (every term of C26's bound carries `·δ`). -/
theorem ppoObjE_gradlip_Gdelta (chosen e V : Expr) (es logps : List Expr)
    (oldLogp g lo hi cv ce ret : Float)
    (hch : Smooth chosen) (he : Smooth e) (hes : ∀ lp ∈ es, Smooth lp) (hV : Smooth V)
    (σ σ' : Nat → ℝ) (R δ : ℝ) (k : Nat)
    (hσ : ∀ i, |σ i| ≤ R) (hσ' : ∀ i, |σ' i| ≤ R) (hδ : ∀ i, |σ i - σ' i| ≤ δ) (hR : 0 ≤ R)
    (hloσ : toReal lo < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ)
    (hhiσ : evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ < toReal hi)
    (hloσ' : toReal lo < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ')
    (hhiσ' : evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ' < toReal hi)
    (Ment Lvent Dment Dlent : ℝ)
    (hMσ : ∀ lp ∈ logps, |evalR lp σ| ≤ Ment) (hMσ' : ∀ lp ∈ logps, |evalR lp σ'| ≤ Ment)
    (hLvE : ∀ lp ∈ logps, |evalR lp σ - evalR lp σ'| ≤ Lvent * δ)
    (hDmEσ : ∀ lp ∈ logps, |derivR lp σ k| ≤ Dment) (hDmEσ' : ∀ lp ∈ logps, |derivR lp σ' k| ≤ Dment)
    (hDlE : ∀ lp ∈ logps, |derivR lp σ k - derivR lp σ' k| ≤ Dlent * δ) (hDmEn : 0 ≤ Dment) :
    |derivR (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) σ k
        - derivR (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) σ' k|
      ≤ Gtot chosen e V es logps oldLogp g cv ce ret R Ment Lvent Dment Dlent * δ := by
  have h := ppoTotalObj_softmax_gradient_lipschitz chosen e V es logps oldLogp g lo hi cv ce ret
    hch he hes hV σ σ' R δ k hσ hσ' hδ hR hloσ hhiσ hloσ' hhiσ'
    Ment Lvent Dment Dlent hMσ hMσ' hLvE hDmEσ hDmEσ' hDlE hDmEn
  refine h.trans (le_of_eq ?_)
  simp only [Gtot]; ring

/-- **C27's whole-run, generalized to a per-trajectory-point `hGtraj`.** Identical to C27's
    `ppo_whole_run_sup_interval` except the gradient-Lipschitz is required only at the trajectory point pairs
    `(θ p, θ' p)` (which is all C27's proof uses) — making it dischargeable by the REGIONAL C26. -/
theorem ppo_whole_run_sup_interval_traj (e : Expr) (lr : Float) (G B d0 L : ℝ)
    (θ θ' : Nat → (Nat → ℝ))
    (hL : L = 1 + |toReal lr| * G)
    (hGtraj : ∀ p (δ : ℝ), (∀ i, |θ p i - θ' p i| ≤ δ) →
      ∀ k, |derivR e (θ p) k - derivR e (θ' p) k| ≤ G * δ)
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
            (hGtraj p (errBound L B d0 p) ih)
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

/-- **THE WHOLE-RUN INTERVAL WITH `G` FROM C26.** For the concrete PPO objective `ppoObjE`, given the trajectory
    stays in C26's valid region at every step — the explicit invariance hypotheses `hRegσ`/`hRegσ'` (params
    bounded by `R`), `hIntσ`/`hIntσ'` (ratio in the clip interior), `hMσ`/…/`hDlE` (entropy budgets) — the
    runnable trajectory stays within `L^n·d0 + B·Σ_{j<n} L^j` of the ideal, with `L = 1 + |lr|·Gtot` and `Gtot`
    C26's concrete gradient-Lipschitz constant. NO free gradient-Lipschitz hypothesis: `hGtraj` is discharged at
    each step by `ppoObjE_gradlip_Gdelta` (C26) using the invariance. This is the whole-run error interval fully
    reduced to (i) the per-step interval `B` (C1), (ii) the trajectory-stays-in-region invariance, and (iii) the
    exact-ℝ-ascent model — the abstract `G` of C27 now the concrete `Gtot` of the actual softmax-MLP PPO loss. -/
theorem ppo_whole_run_from_C26 (chosen e V : Expr) (es logps : List Expr)
    (oldLogp g lo hi cv ce ret lr : Float) (B d0 L R Ment Lvent Dment Dlent : ℝ)
    (hch : Smooth chosen) (he : Smooth e) (hes : ∀ lp ∈ es, Smooth lp) (hV : Smooth V) (hR : 0 ≤ R)
    (θ θ' : Nat → (Nat → ℝ))
    (hL : L = 1 + |toReal lr| * Gtot chosen e V es logps oldLogp g cv ce ret R Ment Lvent Dment Dlent)
    (hL0 : 0 ≤ L)
    (hideal : ∀ n, θ' (n + 1) = gradAscentE (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr (θ' n))
    (hstep : ∀ n k, |θ (n + 1) k
        - gradAscentE (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr (θ n) k| ≤ B)
    (hd0 : ∀ k, |θ 0 k - θ' 0 k| ≤ d0)
    -- the invariance: the trajectory satisfies C26's region/interior/entropy conditions at every step:
    (hRegσ : ∀ p i, |θ p i| ≤ R) (hRegσ' : ∀ p i, |θ' p i| ≤ R)
    (hIntσ : ∀ p, toReal lo < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ p)
        ∧ evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ p) < toReal hi)
    (hIntσ' : ∀ p, toReal lo < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ' p)
        ∧ evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ' p) < toReal hi)
    (hMσ : ∀ p, ∀ lp ∈ logps, |evalR lp (θ p)| ≤ Ment) (hMσ' : ∀ p, ∀ lp ∈ logps, |evalR lp (θ' p)| ≤ Ment)
    (hLvE : ∀ p (δ : ℝ), (∀ i, |θ p i - θ' p i| ≤ δ) →
        ∀ lp ∈ logps, |evalR lp (θ p) - evalR lp (θ' p)| ≤ Lvent * δ)
    (hDmEσ : ∀ p k, ∀ lp ∈ logps, |derivR lp (θ p) k| ≤ Dment)
    (hDmEσ' : ∀ p k, ∀ lp ∈ logps, |derivR lp (θ' p) k| ≤ Dment)
    (hDlE : ∀ p (δ : ℝ) k, (∀ i, |θ p i - θ' p i| ≤ δ) →
        ∀ lp ∈ logps, |derivR lp (θ p) k - derivR lp (θ' p) k| ≤ Dlent * δ)
    (hDmEn : 0 ≤ Dment) (n k : Nat) :
    |θ n k - θ' n k|
      ≤ L ^ n * d0 + B * ∑ j ∈ Finset.range n, L ^ j :=
  ppo_whole_run_sup_interval_traj (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr
    (Gtot chosen e V es logps oldLogp g cv ce ret R Ment Lvent Dment Dlent) B d0 L θ θ' hL
    (fun p δ hclose k =>
      ppoObjE_gradlip_Gdelta chosen e V es logps oldLogp g lo hi cv ce ret hch he hes hV
        (θ p) (θ' p) R δ k (hRegσ p) (hRegσ' p) hclose hR
        (hIntσ p).1 (hIntσ p).2 (hIntσ' p).1 (hIntσ' p).2
        Ment Lvent Dment Dlent (hMσ p) (hMσ' p) (hLvE p δ hclose)
        (hDmEσ p k) (hDmEσ' p k) (hDlE p δ k hclose) hDmEn)
    hideal hstep hd0 hL0 n k

end Puffer.RL.WholeRunFromC26
