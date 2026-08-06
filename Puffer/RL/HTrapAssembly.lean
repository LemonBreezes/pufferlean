/-
# Assembling the barriers: C36 (clip) + C37 (entropy) → `InRegVal` at the projected step, and the reachability closure

C36 (`ClipBarrier`) discharged the clip-interior clause of C35's `hTrap` from a per-step margin; C37
(`EntropyBarrier`) discharged the entropy clauses unconditionally over the region. This module glues the two into
C35's `InRegVal` predicate — the exact conclusion `hTrap` requires — and closes the ideal-trajectory reachability
from per-step barrier conditions.

* `clipMargin` — the concrete per-step clip move `d = exp(Mlog − oldLogp)·(vLip R chosen + vLip R (expSumE (e::es))/c)
  ·(|lr|·Gmag)` (C36's `ratioE_projStep_disp` bound), abbreviated.
* `hTrap_step` — the glue: from `σ` in the region with the C36 budgets (`Gmag`/`Mlog`/`c`), the clip MARGIN
  `ratio(σ) ∈ (lo + clipMargin, hi − clipMargin)`, and the C37 uniform entropy budgets, one projected step lands with
  `InRegVal chosen e es logps oldLogp lo hi Ment Dment (projAscentE (ppoObjE …) lr R σ)` — BOTH value-level clauses
  (clip-interior via C36, entropy via C37) satisfied at the next point. This is exactly the shape `hTrap` concludes.
* `ideal_region_forall` — the reachability closure: from `InRegVal (θ' 0)` and the per-step barrier preconditions
  holding along the ideal trajectory (`∀ p`: gradient/floor/log-prob-magnitude budgets and the clip margin at `θ' p`),
  the ideal stays in the full region for ALL `p` (`∀ p, InReg1 … (θ' p)`) — param-boundedness by projection
  (`projAscentE_mem`), value-level by induction with `hTrap_step`. This supplies the `∀ p` region conditions the
  whole-run interval (C28/C35) consumes.

**Scope (honestly disclosed).** The reachability closure is NOT unconditional: the clip MARGIN
`ratio(θ' p) ∈ (lo + clipMargin, hi − clipMargin)` is an explicit per-step hypothesis (`hmargin`), because — as
established in C36 — a fixed clip band is not forward-invariant under bounded displacement without a contraction, so
the margin is not self-reproducing. The entropy clauses ARE closed unconditionally (C37). Thus `ideal_region_forall`
converts C35's abstract "stays in region ∀n" into a concrete per-step slate: param-bound (discharged by projection),
entropy budgets (discharged unconditionally over the region), and the clip margin (a per-step runtime condition — the
`puffer verify` harness checks it each step). `Gmag`/`Mlog`/`c`/`Ment`/`Dment` are the network's budget constants
(C17/C18/C25/C26), supplied as hypotheses. The runnable-`θ` side and the coupling budgets are as in C28/C35.
-/
import Puffer.RL.TrajReachability
import Puffer.RL.ClipBarrier
import Puffer.RL.EntropyBarrier
open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)
open Puffer.RL.RegionInvariance (projAscentE projAscentE_mem)
open Puffer.RL.WholeRunFromC26 (ppoObjE)
open Puffer.RL.SoftmaxExpr (logSoftmaxE expSumE)
open Puffer.RL.SurrogateExpr (ratioE)
open Puffer.RL.TrajReachability (InRegVal InReg1)
open Puffer.RL.ClipBarrier (clip_barrier_concrete)
open Puffer.RL.EntropyBarrier (entropy_barrier)
open Puffer.RL.RegionInvariance (proj_whole_run_sup_interval)
open Puffer.RL.WholeRunFromC26 (Gtot ppoObjE_gradlip_Gdelta)

namespace Puffer.RL.HTrapAssembly

/-- The concrete per-step clip move `d` (C36's `ratioE_projStep_disp` bound): the most the PPO ratio can change over
    one projected step, `exp(Mlog − oldLogp)·(vLip R chosen + vLip R (expSumE (e::es))/c)·(|lr|·Gmag)`. -/
noncomputable def clipMargin (chosen e : Expr) (es : List Expr) (oldLogp lr : Float) (R Gmag Mlog c : ℝ) : ℝ :=
  Real.exp (Mlog - toReal oldLogp) * (vLip R chosen + vLip R (expSumE (e :: es)) / c) * (|toReal lr| * Gmag)

/-- **The barrier glue.** One projected step from `σ` — in the region, with the C36 budgets (`Gmag`/`Mlog`/`c`), the
    clip MARGIN `ratio(σ) ∈ (lo + clipMargin, hi − clipMargin)`, and the C37 uniform entropy budgets — lands with
    `InRegVal` at the next point: the clip-interior clause via C36's `clip_barrier_concrete`, the entropy clauses via
    C37's `entropy_barrier` (unconditional). This is exactly the conclusion C35's `hTrap` requires, discharged AT `σ`
    from a per-step precondition. -/
theorem hTrap_step (chosen e V : Expr) (es logps : List Expr)
    (oldLogp g lo hi cv ce ret lr : Float) (R Gmag Mlog c Ment Dment : ℝ) (σ : Nat → ℝ)
    (hch : Smooth chosen) (he : Smooth e) (hes : ∀ lp ∈ es, Smooth lp)
    (hσ : ∀ k, |σ k| ≤ R) (hR : 0 ≤ R)
    (hGmag : ∀ k, |derivR (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) σ k| ≤ Gmag)
    (hc : 0 < c)
    (hfloorσ : c ≤ evalR (expSumE (e :: es)) σ)
    (hfloorτ : c ≤ evalR (expSumE (e :: es))
        (projAscentE (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr R σ))
    (hMσ : evalR (logSoftmaxE chosen (e :: es)) σ ≤ Mlog)
    (hMτ : evalR (logSoftmaxE chosen (e :: es))
        (projAscentE (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr R σ) ≤ Mlog)
    (hlo : toReal lo + clipMargin chosen e es oldLogp lr R Gmag Mlog c
          < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ)
    (hhi : evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) σ
          < toReal hi - clipMargin chosen e es oldLogp lr R Gmag Mlog c)
    (hMent : ∀ ρ : Nat → ℝ, (∀ k, |ρ k| ≤ R) → ∀ lp ∈ logps, |evalR lp ρ| ≤ Ment)
    (hDment : ∀ ρ : Nat → ℝ, (∀ k, |ρ k| ≤ R) → ∀ k, ∀ lp ∈ logps, |derivR lp ρ k| ≤ Dment) :
    InRegVal chosen e es logps oldLogp lo hi Ment Dment
      (projAscentE (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr R σ) := by
  obtain ⟨hcliplo, hcliphi⟩ := clip_barrier_concrete chosen e V es logps oldLogp g lo hi cv ce ret lr R
    Gmag Mlog c σ hch he hes hσ hR hGmag hc hfloorσ hfloorτ hMσ hMτ hlo hhi
  obtain ⟨hev, hde⟩ := entropy_barrier (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) logps lr R
    Ment Dment σ hR hMent hDment
  exact ⟨hcliplo, hcliphi, hev, hde⟩

/-- **The ideal-trajectory reachability closure.** From `InRegVal (θ' 0)` (start in the value-level region) and the
    per-step barrier preconditions holding along the ideal trajectory `θ'` — the gradient/floor/log-prob-magnitude
    budgets and the clip MARGIN at every `θ' p` — the ideal stays in the full region for ALL `p`:
    `∀ p, InReg1 … (θ' p)`. Param-boundedness is discharged by projection (`projAscentE_mem`, so no margin), the
    value-level `InRegVal` by induction with `hTrap_step` (the clip via the per-step margin, the entropy
    unconditionally). This is exactly the `∀ p` region slate the whole-run interval (C28/C35) consumes for the ideal
    trajectory — now reduced to a concrete per-step barrier slate. -/
theorem ideal_region_forall (chosen e V : Expr) (es logps : List Expr)
    (oldLogp g lo hi cv ce ret lr : Float) (R Gmag Mlog c Ment Dment : ℝ)
    (θ' : Nat → (Nat → ℝ))
    (hch : Smooth chosen) (he : Smooth e) (hes : ∀ lp ∈ es, Smooth lp)
    (hR : 0 ≤ R) (hc : 0 < c)
    (hideal : ∀ n, θ' (n + 1)
        = projAscentE (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr R (θ' n))
    (h0reg : ∀ k, |θ' 0 k| ≤ R)
    (hVal0 : InRegVal chosen e es logps oldLogp lo hi Ment Dment (θ' 0))
    (hMent : ∀ ρ : Nat → ℝ, (∀ k, |ρ k| ≤ R) → ∀ lp ∈ logps, |evalR lp ρ| ≤ Ment)
    (hDment : ∀ ρ : Nat → ℝ, (∀ k, |ρ k| ≤ R) → ∀ k, ∀ lp ∈ logps, |derivR lp ρ k| ≤ Dment)
    (hGmag : ∀ p k, |derivR (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) (θ' p) k| ≤ Gmag)
    (hfloor : ∀ p, c ≤ evalR (expSumE (e :: es)) (θ' p))
    (hM : ∀ p, evalR (logSoftmaxE chosen (e :: es)) (θ' p) ≤ Mlog)
    (hmargin : ∀ p, toReal lo + clipMargin chosen e es oldLogp lr R Gmag Mlog c
          < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ' p)
        ∧ evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ' p)
          < toReal hi - clipMargin chosen e es oldLogp lr R Gmag Mlog c) :
    ∀ p, InReg1 chosen e es logps oldLogp lo hi R Ment Dment (θ' p) := by
  -- param-boundedness for all p, by projection
  have hreg : ∀ p k, |θ' p k| ≤ R := by
    intro p
    induction p with
    | zero => exact h0reg
    | succ q _ => intro k; rw [hideal q]; exact projAscentE_mem _ lr R (θ' q) hR k
  -- value-level membership for all p, by induction with hTrap_step
  have hval : ∀ p, InRegVal chosen e es logps oldLogp lo hi Ment Dment (θ' p) := by
    intro p
    induction p with
    | zero => exact hVal0
    | succ q _ =>
        rw [hideal q]
        exact hTrap_step chosen e V es logps oldLogp g lo hi cv ce ret lr R Gmag Mlog c Ment Dment (θ' q)
          hch he hes (hreg q) hR (hGmag q) hc (hfloor q) (hideal q ▸ hfloor (q + 1))
          (hM q) (hideal q ▸ hM (q + 1)) (hmargin q).1 (hmargin q).2 hMent hDment
  exact fun p => ⟨hreg p, hval p⟩

/-- **THE WHOLE-RUN INTERVAL FROM PER-STEP BARRIERS.** The projected whole-run error interval (C29) for the concrete
    softmax-MLP PPO objective, with the IDEAL trajectory's region conditions supplied by `ideal_region_forall` — so
    the ideal's clip-interior + entropy membership `∀ p` is REDUCED to the concrete per-step barrier slate
    (`hGmagθ'`/`hfloorθ'`/`hMθ'` budgets and the clip margin `hmarginθ'` at each step), with param-boundedness
    discharged by projection. The per-point gradient-Lipschitz that C29's interval needs is discharged at each step by
    C28's `ppoObjE_gradlip_Gdelta` using those `∀ p` conditions. Conclusion: `L^n·d0 + B·Σ_{j<n} L^j`,
    `L = 1 + |lr|·Gtot`. This is C35's `ppo_whole_run_reachable` with the abstract `hTrap` premise ELIMINATED —
    replaced by the C36/C37 barriers plus the per-step margin. The runnable `θ` conditions (`hRegθ`/…) and coupling
    budgets (`hLvE`/`hDlE`) remain per-step premises exactly as in C35 (see the module scope note). -/
theorem ppo_whole_run_from_barriers (chosen e V : Expr) (es logps : List Expr)
    (oldLogp g lo hi cv ce ret lr : Float)
    (B d0 L R Gmag Mlog c Ment Lvent Dment Dlent : ℝ)
    (hch : Smooth chosen) (he : Smooth e) (hes : ∀ lp ∈ es, Smooth lp) (hV : Smooth V)
    (hR : 0 ≤ R) (hc : 0 < c)
    (θ θ' : Nat → (Nat → ℝ))
    (hL : L = 1 + |toReal lr| * Gtot chosen e V es logps oldLogp g cv ce ret R Ment Lvent Dment Dlent)
    (hL0 : 0 ≤ L)
    (hideal : ∀ n, θ' (n + 1)
        = projAscentE (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr R (θ' n))
    (hstep : ∀ n k, |θ (n + 1) k
        - projAscentE (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr R (θ n) k| ≤ B)
    (hd0 : ∀ k, |θ 0 k - θ' 0 k| ≤ d0)
    (h0θ'reg : ∀ k, |θ' 0 k| ≤ R)
    (hVal0 : InRegVal chosen e es logps oldLogp lo hi Ment Dment (θ' 0))
    (hMent : ∀ ρ : Nat → ℝ, (∀ k, |ρ k| ≤ R) → ∀ lp ∈ logps, |evalR lp ρ| ≤ Ment)
    (hDment : ∀ ρ : Nat → ℝ, (∀ k, |ρ k| ≤ R) → ∀ k, ∀ lp ∈ logps, |derivR lp ρ k| ≤ Dment)
    -- the ideal's per-step barrier preconditions (reachability closure):
    (hGmagθ' : ∀ p k, |derivR (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) (θ' p) k| ≤ Gmag)
    (hfloorθ' : ∀ p, c ≤ evalR (expSumE (e :: es)) (θ' p))
    (hMθ' : ∀ p, evalR (logSoftmaxE chosen (e :: es)) (θ' p) ≤ Mlog)
    (hmarginθ' : ∀ p, toReal lo + clipMargin chosen e es oldLogp lr R Gmag Mlog c
          < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ' p)
        ∧ evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ' p)
          < toReal hi - clipMargin chosen e es oldLogp lr R Gmag Mlog c)
    -- the runnable trajectory's per-step (runtime-checkable) region membership, as in C35:
    (hRegθ : ∀ p i, |θ p i| ≤ R)
    (hIntθ : ∀ p, toReal lo < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ p)
        ∧ evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ p) < toReal hi)
    (hMθ : ∀ p, ∀ lp ∈ logps, |evalR lp (θ p)| ≤ Ment)
    (hDmEθ : ∀ p k, ∀ lp ∈ logps, |derivR lp (θ p) k| ≤ Dment)
    -- coupling budgets (network `Smooth` constants over the region), as in C28/C35:
    (hLvE : ∀ p (δ : ℝ), (∀ i, |θ p i - θ' p i| ≤ δ) →
        ∀ lp ∈ logps, |evalR lp (θ p) - evalR lp (θ' p)| ≤ Lvent * δ)
    (hDlE : ∀ p (δ : ℝ) k, (∀ i, |θ p i - θ' p i| ≤ δ) →
        ∀ lp ∈ logps, |derivR lp (θ p) k - derivR lp (θ' p) k| ≤ Dlent * δ)
    (hDmEn : 0 ≤ Dment) (n k : Nat) :
    |θ n k - θ' n k| ≤ L ^ n * d0 + B * ∑ j ∈ Finset.range n, L ^ j := by
  have hInv : ∀ p, InReg1 chosen e es logps oldLogp lo hi R Ment Dment (θ' p) :=
    ideal_region_forall chosen e V es logps oldLogp g lo hi cv ce ret lr R Gmag Mlog c Ment Dment θ'
      hch he hes hR hc hideal h0θ'reg hVal0 hMent hDment hGmagθ' hfloorθ' hMθ' hmarginθ'
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

end Puffer.RL.HTrapAssembly
