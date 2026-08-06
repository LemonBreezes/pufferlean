/-
# Restricted-horizon whole-run theorems: consuming a finite trace's premise slate

C73 (`TraceCheck`) certifies the runnable trajectory's region/clip-interior premises at every
RECORDED step — a slate quantified `∀ p < tr.length`. The existing whole-run theorems (C42's
`whole_run_sup_interval_of_step`, C29's `proj_whole_run_sup_interval`, C38's
`ppo_whole_run_from_barriers`) STATE their trajectory premises as unbounded `∀ p`, although their
accumulation proofs only ever consume the premises at steps `p < n` below the queried step (C73's
verifier confirmed this). This module performs the named MECHANICAL RESTATEMENT: the same theorems
with every trajectory premise quantified `∀ p < N` and the conclusion at steps `n ≤ N` — so C73's
finite-trace slate feeds them DIRECTLY.

* `whole_run_sup_interval_of_step_fin` — C42's optimizer-agnostic accumulation, restricted horizon.
  The step-map Lipschitz `hlip` is unchanged (it is a property of the MAP, not the trajectory);
  `hideal`/`hstep` are consumed only at `n < N`.
* `proj_whole_run_sup_interval_fin` — C29's projected-ascent interval, restricted horizon: the
  per-trajectory-pair gradient-Lipschitz `hGtraj` and the step premises quantified `∀ p < N`.
* `ideal_region_forall_fin` — C38's reachability closure, restricted horizon: the ideal-side
  per-step barrier preconditions (`hGmag`/`hfloor`/`hM`/`hmargin`) quantified `∀ p < N` yield
  `InReg1` at every `p < N` (the same induction — the step at `q + 1 < N` consumes the
  preconditions at `q < N` and `q + 1 < N` only, so the uniform `∀ p < N` slate suffices).
* `ppo_whole_run_from_barriers_fin` — C38's barrier capstone, restricted horizon on BOTH sides:
  the runnable slate (`hRegθ`/`hIntθ`/`hMθ`/`hDmEθ`) — of which the region/interior halves are
  exactly the shapes C73's `trace_region_slate`/`trace_interior_slate` produce at `N := tr.length`
  (at `toReal Rf`, bridged to `R` in the connector; the entropy halves are discharged there via
  C41) — the ideal-side barrier slate, the per-step error `hstep`, and the coupling budgets all
  `∀ p < N`, conclusion at `n ≤ N`.
* `trace_feeds_whole_run` — the END-TO-END connector: a passing `checkTrace` (ONE runtime `Bool`)
  + C73's representation/forward-error plumbing + C70's one-time gap constants + the theory-side
  ideal/coupling premises ⟹ the whole-run error interval `L^n·d0 + B·Σ_{j<n} L^j` at every
  `n ≤ tr.length`. The runnable region/interior slates come from the trace (C73); the runnable
  entropy budgets are DISCHARGED from the trace's region slate via C41's `region_entropy_smooth`
  (`Smooth` log-probs + `vMag`/`dMag` caps) — no per-step entropy premises remain.

**Scope (honestly disclosed).** This is a mechanical restatement — the proofs are C42/C29/C38's
verbatim with the horizon threaded through the inductions; no new mathematics. The conclusion
reaches `n ≤ N` (the recorded horizon — the honest reach of a finite trace; the unbounded-`∀ p`
originals remain for theory-side use). In the connector, the ideal-side premises
(`hGmagθ'`/`hfloorθ'`/`hMθ'`/`hmarginθ'`), the per-step Float error `B`, the coupling budgets
(`hLvE`/`hDlE`), and the initial conditions are the same theory-side slate as C38 (restricted to
`p < tr.length`); the representation (`hrep` — `θ p` is the zero-padded recorded row) and the
ratio forward error (`herr`, budget `e`) are C73's caller plumbing; the gap hypotheses
(`hLoGap`/`hHiGap`) are C70's offline constants-only obligations; `toReal Rf ≤ R` bridges the
runtime region radius to the theory radius.
-/
import Puffer.RL.HTrapAssembly
import Puffer.RL.TraceCheck
import Puffer.RL.RunnableRegion

open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)
open Puffer.RL.WholeRunInterval (errBound)
open Puffer.RL.MuonTrainBound (affine_recur_le)
open Puffer.RL.RegionInvariance (projAscentE projAscentE_sup_lipschitz projAscentE_mem)
open Puffer.RL.WholeRunFromC26 (ppoObjE Gtot ppoObjE_gradlip_Gdelta)
open Puffer.RL.SoftmaxExpr (logSoftmaxE expSumE)
open Puffer.RL.SurrogateExpr (ratioE)
open Puffer.RL.TrajReachability (InRegVal InReg1)
open Puffer.RL.HTrapAssembly (clipMargin hTrap_step)
open Puffer.RL.RunnableRegion (region_entropy_smooth)
open Puffer.RL.TraceCheck (Trace checkTrace padRow trace_to_traj_premises)

namespace Puffer.RL.FiniteHorizonRun

/-- **C42's optimizer-agnostic whole-run interval, restricted horizon.** Identical to
    `MuonAscentBridge.whole_run_sup_interval_of_step` except the ideal-step law `hideal` and the
    per-step error `hstep` are required only at steps `n < N`, and the conclusion holds at `n ≤ N`.
    The step-map Lipschitz `hlip` is unchanged — it is a property of the map `F`, not of the
    trajectory. The proof is the same `errBound` induction; it only ever consumed the premises at
    `p < n ≤ N`, which the restatement makes explicit. -/
theorem whole_run_sup_interval_of_step_fin (F : (Nat → ℝ) → (Nat → ℝ)) (B d0 L : ℝ)
    (θ θ' : Nat → (Nat → ℝ)) (N : Nat)
    (hlip : ∀ (σ σ' : Nat → ℝ) (δ : ℝ), (∀ i, |σ i - σ' i| ≤ δ) → ∀ k, |F σ k - F σ' k| ≤ L * δ)
    (hideal : ∀ n, n < N → θ' (n + 1) = F (θ' n))
    (hstep : ∀ n, n < N → ∀ k, |θ (n + 1) k - F (θ n) k| ≤ B)
    (hd0 : ∀ k, |θ 0 k - θ' 0 k| ≤ d0)
    (hL0 : 0 ≤ L) (n : Nat) (hn : n ≤ N) (k : Nat) :
    |θ n k - θ' n k| ≤ L ^ n * d0 + B * ∑ j ∈ Finset.range n, L ^ j := by
  have hbound : ∀ m, m ≤ N → ∀ j, |θ m j - θ' m j| ≤ errBound L B d0 m := by
    intro m
    induction m with
    | zero => intro _ j; exact hd0 j
    | succ p ih =>
        intro hpN j
        have hpN' : p < N := Nat.lt_of_succ_le hpN
        have ihp := ih (Nat.le_of_lt hpN')
        have hlp : ∀ i, |F (θ p) i - F (θ' p) i| ≤ L * errBound L B d0 p :=
          hlip (θ p) (θ' p) (errBound L B d0 p) ihp
        calc |θ (p + 1) j - θ' (p + 1) j|
            = |θ (p + 1) j - F (θ' p) j| := by rw [hideal p hpN']
          _ ≤ |θ (p + 1) j - F (θ p) j| + |F (θ p) j - F (θ' p) j| := abs_sub_le _ _ _
          _ ≤ B + L * errBound L B d0 p := add_le_add (hstep p hpN' j) (hlp j)
          _ = errBound L B d0 (p + 1) := by simp only [errBound]; ring
  have hrec : ∀ j, errBound L B d0 (j + 1) ≤ L * errBound L B d0 j + B :=
    fun j => le_of_eq (by simp only [errBound])
  have hkey := affine_recur_le (errBound L B d0) L B hL0 hrec n
  have h0 : errBound L B d0 0 = d0 := rfl
  rw [h0] at hkey
  exact (hbound n hn k).trans hkey

/-- **C29's projected whole-run interval, restricted horizon.** Identical to
    `RegionInvariance.proj_whole_run_sup_interval` except the per-trajectory-pair
    gradient-Lipschitz `hGtraj`, the ideal-step law, and the per-step error are required only at
    steps `p < N`, and the conclusion holds at `n ≤ N`. Same `errBound` induction with the horizon
    threaded (each step of the induction consumes the premises at its own index only). -/
theorem proj_whole_run_sup_interval_fin (e : Expr) (lr : Float) (R G B d0 L : ℝ)
    (θ θ' : Nat → (Nat → ℝ)) (N : Nat)
    (hL : L = 1 + |toReal lr| * G)
    (hGtraj : ∀ p, p < N → ∀ (δ : ℝ), (∀ i, |θ p i - θ' p i| ≤ δ) →
      ∀ k, |derivR e (θ p) k - derivR e (θ' p) k| ≤ G * δ)
    (hidealθ' : ∀ n, n < N → θ' (n + 1) = projAscentE e lr R (θ' n))
    (hstep : ∀ n, n < N → ∀ k, |θ (n + 1) k - projAscentE e lr R (θ n) k| ≤ B)
    (hd0 : ∀ k, |θ 0 k - θ' 0 k| ≤ d0)
    (hL0 : 0 ≤ L) (n : Nat) (hn : n ≤ N) (k : Nat) :
    |θ n k - θ' n k| ≤ L ^ n * d0 + B * ∑ j ∈ Finset.range n, L ^ j := by
  have hbound : ∀ m, m ≤ N → ∀ j, |θ m j - θ' m j| ≤ errBound L B d0 m := by
    intro m
    induction m with
    | zero => intro _ j; exact hd0 j
    | succ p ih =>
        intro hpN j
        have hpN' : p < N := Nat.lt_of_succ_le hpN
        have ihp := ih (Nat.le_of_lt hpN')
        have hasc : ∀ i, |projAscentE e lr R (θ p) i - projAscentE e lr R (θ' p) i|
            ≤ L * errBound L B d0 p :=
          hL.symm ▸ projAscentE_sup_lipschitz e lr R G (errBound L B d0 p) (θ p) (θ' p) ihp
            (hGtraj p hpN' (errBound L B d0 p) ihp)
        calc |θ (p + 1) j - θ' (p + 1) j|
            = |θ (p + 1) j - projAscentE e lr R (θ' p) j| := by rw [hidealθ' p hpN']
          _ ≤ |θ (p + 1) j - projAscentE e lr R (θ p) j|
              + |projAscentE e lr R (θ p) j - projAscentE e lr R (θ' p) j| := abs_sub_le _ _ _
          _ ≤ B + L * errBound L B d0 p := add_le_add (hstep p hpN' j) (hasc j)
          _ = errBound L B d0 (p + 1) := by simp only [errBound]; ring
  have hrec : ∀ j, errBound L B d0 (j + 1) ≤ L * errBound L B d0 j + B :=
    fun j => le_of_eq (by simp only [errBound])
  have hkey := affine_recur_le (errBound L B d0) L B hL0 hrec n
  have h0 : errBound L B d0 0 = d0 := rfl
  rw [h0] at hkey
  exact (hbound n hn k).trans hkey

/-- **C38's reachability closure, restricted horizon.** From `InRegVal (θ' 0)` and the ideal-side
    per-step barrier preconditions quantified `∀ p < N` only, the ideal stays in the full region at
    every `p < N`. The same two inductions as `HTrapAssembly.ideal_region_forall`: the step at
    `q + 1 < N` consumes the preconditions at `q < N` and `q + 1 < N` only, so the uniform
    `∀ p < N` slate suffices — the restatement makes the consumption pattern explicit. -/
theorem ideal_region_forall_fin (chosen e V : Expr) (es logps : List Expr)
    (oldLogp g lo hi cv ce ret lr : Float) (R Gmag Mlog c Ment Dment : ℝ)
    (θ' : Nat → (Nat → ℝ)) (N : Nat)
    (hch : Smooth chosen) (he : Smooth e) (hes : ∀ lp ∈ es, Smooth lp)
    (hR : 0 ≤ R) (hc : 0 < c)
    (hideal : ∀ n, n < N → θ' (n + 1)
        = projAscentE (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr R (θ' n))
    (h0reg : ∀ k, |θ' 0 k| ≤ R)
    (hVal0 : InRegVal chosen e es logps oldLogp lo hi Ment Dment (θ' 0))
    (hMent : ∀ ρ : Nat → ℝ, (∀ k, |ρ k| ≤ R) → ∀ lp ∈ logps, |evalR lp ρ| ≤ Ment)
    (hDment : ∀ ρ : Nat → ℝ, (∀ k, |ρ k| ≤ R) → ∀ k, ∀ lp ∈ logps, |derivR lp ρ k| ≤ Dment)
    (hGmag : ∀ p, p < N →
        ∀ k, |derivR (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) (θ' p) k| ≤ Gmag)
    (hfloor : ∀ p, p < N → c ≤ evalR (expSumE (e :: es)) (θ' p))
    (hM : ∀ p, p < N → evalR (logSoftmaxE chosen (e :: es)) (θ' p) ≤ Mlog)
    (hmargin : ∀ p, p < N → toReal lo + clipMargin chosen e es oldLogp lr R Gmag Mlog c
          < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ' p)
        ∧ evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ' p)
          < toReal hi - clipMargin chosen e es oldLogp lr R Gmag Mlog c) :
    ∀ p, p < N → InReg1 chosen e es logps oldLogp lo hi R Ment Dment (θ' p) := by
  -- param-boundedness at every p < N, by projection
  have hreg : ∀ p, p < N → ∀ k, |θ' p k| ≤ R := by
    intro p
    induction p with
    | zero => intro _ k; exact h0reg k
    | succ q _ =>
        intro hq1 k
        rw [hideal q (Nat.lt_of_succ_lt hq1)]
        exact projAscentE_mem _ lr R (θ' q) hR k
  -- value-level membership at every p < N, by induction with hTrap_step
  have hval : ∀ p, p < N → InRegVal chosen e es logps oldLogp lo hi Ment Dment (θ' p) := by
    intro p
    induction p with
    | zero => intro _; exact hVal0
    | succ q _ =>
        intro hq1
        have hqN : q < N := Nat.lt_of_succ_lt hq1
        rw [hideal q hqN]
        exact hTrap_step chosen e V es logps oldLogp g lo hi cv ce ret lr R Gmag Mlog c Ment Dment
          (θ' q) hch he hes (hreg q hqN) hR (hGmag q hqN) hc (hfloor q hqN)
          ((hideal q hqN) ▸ hfloor (q + 1) hq1)
          (hM q hqN) ((hideal q hqN) ▸ hM (q + 1) hq1)
          (hmargin q hqN).1 (hmargin q hqN).2 hMent hDment
  exact fun p hp => ⟨hreg p hp, hval p hp⟩

/-- **C38's barrier capstone, restricted horizon.** `HTrapAssembly.ppo_whole_run_from_barriers`
    with EVERY trajectory premise — the ideal-side barrier slate, the runnable-side slate
    (`hRegθ`/`hIntθ`/`hMθ`/`hDmEθ` — exactly the shapes C73's `trace_region_slate`/
    `trace_interior_slate` produce at `N := tr.length`), the per-step error `hstep`, and the
    coupling budgets — quantified `∀ p < N`, and the conclusion at `n ≤ N`. The proof is C38's
    verbatim: `ideal_region_forall_fin` supplies `InReg1` at `p < N`, and the restricted
    accumulation `proj_whole_run_sup_interval_fin` consumes the per-pair gradient-Lipschitz
    (discharged stepwise by C28's `ppoObjE_gradlip_Gdelta`) at `p < N` only. -/
theorem ppo_whole_run_from_barriers_fin (chosen e V : Expr) (es logps : List Expr)
    (oldLogp g lo hi cv ce ret lr : Float)
    (B d0 L R Gmag Mlog c Ment Lvent Dment Dlent : ℝ)
    (hch : Smooth chosen) (he : Smooth e) (hes : ∀ lp ∈ es, Smooth lp) (hV : Smooth V)
    (hR : 0 ≤ R) (hc : 0 < c)
    (θ θ' : Nat → (Nat → ℝ)) (N : Nat)
    (hL : L = 1 + |toReal lr| * Gtot chosen e V es logps oldLogp g cv ce ret R Ment Lvent Dment Dlent)
    (hL0 : 0 ≤ L)
    (hideal : ∀ n, n < N → θ' (n + 1)
        = projAscentE (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr R (θ' n))
    (hstep : ∀ n, n < N → ∀ k, |θ (n + 1) k
        - projAscentE (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) lr R (θ n) k| ≤ B)
    (hd0 : ∀ k, |θ 0 k - θ' 0 k| ≤ d0)
    (h0θ'reg : ∀ k, |θ' 0 k| ≤ R)
    (hVal0 : InRegVal chosen e es logps oldLogp lo hi Ment Dment (θ' 0))
    (hMent : ∀ ρ : Nat → ℝ, (∀ k, |ρ k| ≤ R) → ∀ lp ∈ logps, |evalR lp ρ| ≤ Ment)
    (hDment : ∀ ρ : Nat → ℝ, (∀ k, |ρ k| ≤ R) → ∀ k, ∀ lp ∈ logps, |derivR lp ρ k| ≤ Dment)
    -- the ideal's per-step barrier preconditions, restricted to the horizon:
    (hGmagθ' : ∀ p, p < N →
        ∀ k, |derivR (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret) (θ' p) k| ≤ Gmag)
    (hfloorθ' : ∀ p, p < N → c ≤ evalR (expSumE (e :: es)) (θ' p))
    (hMθ' : ∀ p, p < N → evalR (logSoftmaxE chosen (e :: es)) (θ' p) ≤ Mlog)
    (hmarginθ' : ∀ p, p < N → toReal lo + clipMargin chosen e es oldLogp lr R Gmag Mlog c
          < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ' p)
        ∧ evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ' p)
          < toReal hi - clipMargin chosen e es oldLogp lr R Gmag Mlog c)
    -- the runnable trajectory's per-step slate, restricted — C73's trace shapes:
    (hRegθ : ∀ p, p < N → ∀ i, |θ p i| ≤ R)
    (hIntθ : ∀ p, p < N → toReal lo < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ p)
        ∧ evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ p) < toReal hi)
    (hMθ : ∀ p, p < N → ∀ lp ∈ logps, |evalR lp (θ p)| ≤ Ment)
    (hDmEθ : ∀ p, p < N → ∀ k, ∀ lp ∈ logps, |derivR lp (θ p) k| ≤ Dment)
    -- coupling budgets, restricted:
    (hLvE : ∀ p, p < N → ∀ (δ : ℝ), (∀ i, |θ p i - θ' p i| ≤ δ) →
        ∀ lp ∈ logps, |evalR lp (θ p) - evalR lp (θ' p)| ≤ Lvent * δ)
    (hDlE : ∀ p, p < N → ∀ (δ : ℝ) (k : Nat), (∀ i, |θ p i - θ' p i| ≤ δ) →
        ∀ lp ∈ logps, |derivR lp (θ p) k - derivR lp (θ' p) k| ≤ Dlent * δ)
    (hDmEn : 0 ≤ Dment) (n : Nat) (hn : n ≤ N) (k : Nat) :
    |θ n k - θ' n k| ≤ L ^ n * d0 + B * ∑ j ∈ Finset.range n, L ^ j := by
  have hInv : ∀ p, p < N → InReg1 chosen e es logps oldLogp lo hi R Ment Dment (θ' p) :=
    ideal_region_forall_fin chosen e V es logps oldLogp g lo hi cv ce ret lr R Gmag Mlog c Ment
      Dment θ' N hch he hes hR hc hideal h0θ'reg hVal0 hMent hDment hGmagθ' hfloorθ' hMθ' hmarginθ'
  refine proj_whole_run_sup_interval_fin (ppoObjE chosen e V es logps oldLogp g lo hi cv ce ret)
    lr R (Gtot chosen e V es logps oldLogp g cv ce ret R Ment Lvent Dment Dlent) B d0 L θ θ' N hL
    ?_ hideal hstep hd0 hL0 n hn k
  intro p hp δ hclose k'
  exact ppoObjE_gradlip_Gdelta chosen e V es logps oldLogp g lo hi cv ce ret hch he hes hV
    (θ p) (θ' p) R δ k'
    (hRegθ p hp) (hInv p hp).1 hclose hR
    (hIntθ p hp).1 (hIntθ p hp).2 (hInv p hp).2.1 (hInv p hp).2.2.1
    Ment Lvent Dment Dlent
    (hMθ p hp) (hInv p hp).2.2.2.1 (hLvE p hp δ hclose)
    (hDmEθ p hp k') ((hInv p hp).2.2.2.2 k') (hDlE p hp δ k' hclose) hDmEn

/-- **THE END-TO-END CONNECTOR: one runtime `Bool` ⟹ the whole-run error interval over the
    recorded horizon.** A passing `checkTrace` (C73), the trace representation (`hrep` — each
    `θ p` the zero-padded recorded row) and ratio forward error (`herr`, budget `e`), C70's
    one-time gap constants (`hLoGap`/`hHiGap`), and the radius bridge `toReal Rf ≤ R` supply the
    runnable REGION and CLIP-INTERIOR slates at every recorded step; the runnable ENTROPY budgets
    are then DISCHARGED from that region slate via C41's `region_entropy_smooth` (`Smooth`
    log-probs with `vMag`/`dMag` caps — no per-step entropy premises remain). Together with the
    theory-side ideal/coupling premises (restricted to the horizon), the restricted barrier
    capstone yields `|θ n k − θ' n k| ≤ L^n·d0 + B·Σ_{j<n} L^j` at every `n ≤ tr.length` — the
    machine-checked whole-run interval standing behind a single passing runtime check. -/
theorem trace_feeds_whole_run (chosen eE V : Expr) (es logps : List Expr)
    (oldLogp g lo hi cv ce ret lr : Float)
    (B d0 L R Gmag Mlog c Ment Lvent Dment Dlent : ℝ)
    (hch : Smooth chosen) (he : Smooth eE) (hes : ∀ lp ∈ es, Smooth lp) (hV : Smooth V)
    (hR : 0 ≤ R) (hc : 0 < c)
    (θ θ' : Nat → (Nat → ℝ)) (tr : Trace) (Rf tLo tHi : Float) (e : ℝ)
    (hL : L = 1 + |toReal lr| * Gtot chosen eE V es logps oldLogp g cv ce ret R Ment Lvent Dment Dlent)
    (hL0 : 0 ≤ L)
    -- the runtime check + C73's trace plumbing + C70's gap constants:
    (hcheck : checkTrace tr Rf tLo tHi = true)
    (hR0 : 0 ≤ toReal Rf) (hRfR : toReal Rf ≤ R)
    (hrep : ∀ p (hp : p < tr.length), ∀ i, θ p i = toReal (padRow (tr[p].1) i))
    (herr : ∀ p (hp : p < tr.length),
      |evalR (ratioE (logSoftmaxE chosen (eE :: es)) oldLogp) (θ p) - toReal (tr[p].2)| ≤ e)
    (hLoGap : toReal lo + e < toReal tLo) (hHiGap : toReal tHi + e < toReal hi)
    -- the runnable entropy budgets' structural inputs (C41 discharge from the region):
    (hlp : ∀ lp ∈ logps, Smooth lp)
    (hVm : ∀ lp ∈ logps, vMag R lp ≤ Ment) (hDm : ∀ lp ∈ logps, dMag R lp ≤ Dment)
    -- the theory-side ideal/step/coupling premises, restricted to the recorded horizon:
    (hideal : ∀ n, n < tr.length → θ' (n + 1)
        = projAscentE (ppoObjE chosen eE V es logps oldLogp g lo hi cv ce ret) lr R (θ' n))
    (hstep : ∀ n, n < tr.length → ∀ k, |θ (n + 1) k
        - projAscentE (ppoObjE chosen eE V es logps oldLogp g lo hi cv ce ret) lr R (θ n) k| ≤ B)
    (hd0 : ∀ k, |θ 0 k - θ' 0 k| ≤ d0)
    (h0θ'reg : ∀ k, |θ' 0 k| ≤ R)
    (hVal0 : InRegVal chosen eE es logps oldLogp lo hi Ment Dment (θ' 0))
    (hMent : ∀ ρ : Nat → ℝ, (∀ k, |ρ k| ≤ R) → ∀ lp ∈ logps, |evalR lp ρ| ≤ Ment)
    (hDment : ∀ ρ : Nat → ℝ, (∀ k, |ρ k| ≤ R) → ∀ k, ∀ lp ∈ logps, |derivR lp ρ k| ≤ Dment)
    (hGmagθ' : ∀ p, p < tr.length →
        ∀ k, |derivR (ppoObjE chosen eE V es logps oldLogp g lo hi cv ce ret) (θ' p) k| ≤ Gmag)
    (hfloorθ' : ∀ p, p < tr.length → c ≤ evalR (expSumE (eE :: es)) (θ' p))
    (hMθ' : ∀ p, p < tr.length → evalR (logSoftmaxE chosen (eE :: es)) (θ' p) ≤ Mlog)
    (hmarginθ' : ∀ p, p < tr.length →
        toReal lo + clipMargin chosen eE es oldLogp lr R Gmag Mlog c
          < evalR (ratioE (logSoftmaxE chosen (eE :: es)) oldLogp) (θ' p)
        ∧ evalR (ratioE (logSoftmaxE chosen (eE :: es)) oldLogp) (θ' p)
          < toReal hi - clipMargin chosen eE es oldLogp lr R Gmag Mlog c)
    (hLvE : ∀ p, p < tr.length → ∀ (δ : ℝ), (∀ i, |θ p i - θ' p i| ≤ δ) →
        ∀ lp ∈ logps, |evalR lp (θ p) - evalR lp (θ' p)| ≤ Lvent * δ)
    (hDlE : ∀ p, p < tr.length → ∀ (δ : ℝ) (k : Nat), (∀ i, |θ p i - θ' p i| ≤ δ) →
        ∀ lp ∈ logps, |derivR lp (θ p) k - derivR lp (θ' p) k| ≤ Dlent * δ)
    (hDmEn : 0 ≤ Dment) (n : Nat) (hn : n ≤ tr.length) (k : Nat) :
    |θ n k - θ' n k| ≤ L ^ n * d0 + B * ∑ j ∈ Finset.range n, L ^ j := by
  -- the trace-derived runnable region + clip-interior slates (C73)
  have hslate := trace_to_traj_premises chosen eE es oldLogp lo hi θ tr Rf tLo tHi e
    hcheck hR0 hrep herr hLoGap hHiGap
  have hRegθ : ∀ p, p < tr.length → ∀ i, |θ p i| ≤ R :=
    fun p hp i => ((hslate p hp).1 i).trans hRfR
  have hIntθ : ∀ p, p < tr.length →
      toReal lo < evalR (ratioE (logSoftmaxE chosen (eE :: es)) oldLogp) (θ p)
        ∧ evalR (ratioE (logSoftmaxE chosen (eE :: es)) oldLogp) (θ p) < toReal hi :=
    fun p hp => (hslate p hp).2
  -- the runnable entropy budgets from the region slate (C41)
  have hMθ : ∀ p, p < tr.length → ∀ lp ∈ logps, |evalR lp (θ p)| ≤ Ment :=
    fun p hp => (region_entropy_smooth logps R Ment Dment (θ p) (hRegθ p hp) hR hlp hVm hDm).1
  have hDmEθ : ∀ p, p < tr.length → ∀ k, ∀ lp ∈ logps, |derivR lp (θ p) k| ≤ Dment :=
    fun p hp => (region_entropy_smooth logps R Ment Dment (θ p) (hRegθ p hp) hR hlp hVm hDm).2
  exact ppo_whole_run_from_barriers_fin chosen eE V es logps oldLogp g lo hi cv ce ret lr
    B d0 L R Gmag Mlog c Ment Lvent Dment Dlent hch he hes hV hR hc θ θ' tr.length hL hL0
    hideal hstep hd0 h0θ'reg hVal0 hMent hDment hGmagθ' hfloorθ' hMθ' hmarginθ'
    hRegθ hIntθ hMθ hDmEθ hLvE hDlE hDmEn n hn k

end Puffer.RL.FiniteHorizonRun
