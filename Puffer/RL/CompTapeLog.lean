/-
# The FULL-grammar comp-tape certificate: C71's hypothesis discharge composed with C72's floored log

C71 (`CompTapeWeights`) discharged all three of C68's tape hypotheses (`hDw`/`hwf`/`hE`) for
tapes the expression compiler `comp` actually produces — but only over the `LogFree` sub-grammar,
because the log node's edge weight `1.0 / x` and value need a domain floor. C72 (`LogTapeFinite`)
supplied exactly those floored bounds (`log_edge_weight_mag`, `evalFL_mag` over the full grammar
under `LogFloored`). This module is their COMPOSITION — the complete tape certificate for the
FULL 11-op expression grammar:

* `compWeightBoundL B c e` — the log-inclusive edge-weight budget: C71's `compWeightBound` with
  the log case `max ((1/c)·(1+u64)) (subtree)` (C72's floored edge-weight bound) and the
  `mul`/`exp` sibling-value budgets upgraded from C68's `fwdBound` to C72's log-inclusive
  `fwdBoundL` (subtrees may now contain `log`). `compWeightBoundL_eq_compWeightBound` shows the
  extension is CONSERVATIVE (the budgets coincide on `LogFree` terms).
* `comp_deps_bounded_log` — C71's compilation-state induction extended to the full grammar under
  `LogFloored σ c e` (+ `0 < c`): every NEW node of the compiled tape has `≤ 2` edges (the log
  node has 1), each with weight `≤ compWeightBoundL B c e`. The log case identifies the recorded
  weight `1.0 / ra.2.val[ra.1]!` as `1.0 / evalF a σ` via `comp_root_val` (the same per-node
  value readout as C71's `mul` case) and bounds it by C72's `log_edge_weight_mag` under the
  floor; the `mul`/`exp` sibling values are bounded by C72's `evalFL_mag`.
* `comp_hDw_log`/`comp_hE_log` (+ C71's grammar-independent `comp_hwf`, reused) and
  **`adGrad_isFinite_comp_log`** (capstone): for a FULL-grammar expression with floored log
  arguments compiled by the ACTUAL compiler from `Tape.empty` on inputs bounded by `B`, EVERY
  adjoint the ACTUAL imperative `grads` engine produces is overflow-free — NO free tape
  hypotheses, ONE checkable budget
  `sweepBound 2 (compWeightBoundL B c e) n 1 ≤ overflowBound`.

**Scope (honestly disclosed).** The certificate is for `comp`-from-`Tape.empty` tapes of
full-grammar expressions under C72's per-input floor `LogFloored σ c e` — the honest semantic
hypothesis, dischargeable for the actual PPO loss's only `log` (the partition) via the structural
floor (C9's `expSumE_floor` / C51's `sumExpF_pos`); the quantitative ℝ→Float floor transfer
through the forward error remains the caller's glue, exactly as C72/C51 disclose. Hand-built
tapes keep C68's three-hypothesis interface. NO new axiom: everything reduces to C71's induction
apparatus, C72's floored log bounds, and the pre-existing comp layout lemmas.
-/
import Puffer.RL.CompTapeWeights
import Puffer.RL.LogTapeFinite

namespace Puffer.RL.CompTapeLog

open Puffer.FloatR (toReal u64 expEps toReal_oneLit)
open Puffer.FloatR.AD (Tape V grads)
open Puffer.FloatR.ADR (Expr evalF)
open Puffer.FloatR.ADReverse (comp pushT comp_root_val comp_preserve comp_root_lt
  comp_deps_size comp_preserve_deps push_getElem!_size push_getElem!_lt)
open Puffer.RL.ADTapeFinite (LogFree fwdBound sweepBound adGrad_isFinite)
open Puffer.RL.FiniteBound (overflowBound)
open Puffer.RL.LossFinite (expF_mag_le)
open Puffer.RL.CompTapeWeights (compWeightBound mem_singleton_arr mem_pair_arr oneLit_mag
  negOneLit_mag ite_zero_one_mag ite_one_zero_mag comp_hwf)
open Puffer.RL.LogTapeFinite (LogFloored fwdBoundL fwdBoundL_eq_fwdBound evalFL_mag
  log_edge_weight_mag logFree_logFloored)

/-! ### The log-inclusive edge-weight budget -/

/-- **The full-grammar comp-tape edge-weight budget.** C71's `compWeightBound` with (i) the log
    case `max ((1/c)·(1+u64)) (subtree)` — C72's floored bound on the recorded `1.0 / x` edge
    weight — and (ii) the `mul`/`exp` sibling-value budgets taken from C72's log-inclusive
    `fwdBoundL` (the sibling subtrees may themselves contain `log`). All other cases mirror
    C71 verbatim. -/
noncomputable def compWeightBoundL (B c : ℝ) : Expr → ℝ
  | .var _ => 1
  | .const _ => 1
  | .add a b => max 1 (max (compWeightBoundL B c a) (compWeightBoundL B c b))
  | .sub a b => max 1 (max (compWeightBoundL B c a) (compWeightBoundL B c b))
  | .mul a b => max (max (fwdBoundL B c a) (fwdBoundL B c b))
      (max (compWeightBoundL B c a) (compWeightBoundL B c b))
  | .scale cc a => max |toReal cc| (compWeightBoundL B c a)
  | .exp a => max (Real.exp (fwdBoundL B c a) * (1 + expEps)) (compWeightBoundL B c a)
  | .log a => max ((1 / c) * (1 + u64)) (compWeightBoundL B c a)
  | .relu a => max 1 (compWeightBoundL B c a)
  | .max a b => max 1 (max (compWeightBoundL B c a) (compWeightBoundL B c b))
  | .min a b => max 1 (max (compWeightBoundL B c a) (compWeightBoundL B c b))

/-- The log-inclusive budget is nonnegative (no floor-positivity needed — every case routes
    through a subtree or the constant `1`). -/
theorem compWeightBoundL_nonneg (B c : ℝ) : ∀ e : Expr, 0 ≤ compWeightBoundL B c e := by
  intro e
  induction e with
  | var i => simp only [compWeightBoundL]; norm_num
  | const cc => simp only [compWeightBoundL]; norm_num
  | add a b _ _ => simp only [compWeightBoundL]
                   exact le_trans zero_le_one (le_max_left _ _)
  | sub a b _ _ => simp only [compWeightBoundL]
                   exact le_trans zero_le_one (le_max_left _ _)
  | mul a b iha _ => simp only [compWeightBoundL]
                     exact iha.trans ((le_max_left _ _).trans (le_max_right _ _))
  | scale cc a _ => simp only [compWeightBoundL]
                    exact (abs_nonneg _).trans (le_max_left _ _)
  | exp a iha => simp only [compWeightBoundL]
                 exact iha.trans (le_max_right _ _)
  | log a iha => simp only [compWeightBoundL]
                 exact iha.trans (le_max_right _ _)
  | relu a _ => simp only [compWeightBoundL]
                exact le_trans zero_le_one (le_max_left _ _)
  | max a b _ _ => simp only [compWeightBoundL]
                   exact le_trans zero_le_one (le_max_left _ _)
  | min a b _ _ => simp only [compWeightBoundL]
                   exact le_trans zero_le_one (le_max_left _ _)

/-- On the `LogFree` sub-grammar the log-inclusive budget coincides with C71's (conservative
    extension; the `mul`/`exp` cases via C72's `fwdBoundL_eq_fwdBound`). -/
theorem compWeightBoundL_eq_compWeightBound (B c : ℝ) :
    ∀ e, LogFree e → compWeightBoundL B c e = compWeightBound B e := by
  intro e
  induction e with
  | var i => intro _; rfl
  | const cc => intro _; rfl
  | add a b iha ihb =>
      intro h; simp only [compWeightBoundL, compWeightBound, iha h.1, ihb h.2]
  | sub a b iha ihb =>
      intro h; simp only [compWeightBoundL, compWeightBound, iha h.1, ihb h.2]
  | mul a b iha ihb =>
      intro h
      simp only [compWeightBoundL, compWeightBound, iha h.1, ihb h.2,
        fwdBoundL_eq_fwdBound B c a h.1, fwdBoundL_eq_fwdBound B c b h.2]
  | scale cc a iha => intro h; simp only [compWeightBoundL, compWeightBound, iha h]
  | exp a iha =>
      intro h
      simp only [compWeightBoundL, compWeightBound, iha h, fwdBoundL_eq_fwdBound B c a h]
  | log a _ => intro h; exact h.elim
  | relu a iha => intro h; simp only [compWeightBoundL, compWeightBound, iha h]
  | max a b iha ihb =>
      intro h; simp only [compWeightBoundL, compWeightBound, iha h.1, ihb h.2]
  | min a b iha ihb =>
      intro h; simp only [compWeightBoundL, compWeightBound, iha h.1, ihb h.2]

/-! ### The traversal, full grammar: every new node is 2-edge, weight-bounded -/

/-- **The full-grammar comp traversal (the crux).** C71's `comp_deps_bounded` induction extended
    to the FULL grammar under `LogFloored σ c e` (+ `0 < c`): for a floored expression compiled
    onto ANY base tape, every NEW node has `≤ 2` edges, each with weight
    `≤ compWeightBoundL B c e`. The only new case is `log`: the recorded weight
    `1.0 / ra.2.val[ra.1]!` is identified as `1.0 / evalF a σ` via `comp_root_val` (C71's mul-case
    value readout) and bounded by C72's `log_edge_weight_mag` under the floor. The `mul`/`exp`
    sibling values are bounded by C72's log-inclusive `evalFL_mag`; all other cases mirror C71. -/
theorem comp_deps_bounded_log (σ : Nat → Float) (B c : ℝ) (hc : 0 < c)
    (hσ : ∀ i, |toReal (σ i)| ≤ B) :
    ∀ (e : Expr), LogFloored σ c e → ∀ (t : Tape) (idx : Nat),
      t.deps.size ≤ idx → idx < (comp σ e t).2.deps.size →
      ((comp σ e t).2.deps[idx]!).size ≤ 2 ∧
        ∀ p ∈ (comp σ e t).2.deps[idx]!, |toReal p.2| ≤ compWeightBoundL B c e := by
  intro e
  induction e with
  | var i =>
      intro _ t idx hlo hhi
      simp only [comp, pushT, Array.size_push] at hhi ⊢
      have heq : idx = t.deps.size := by omega
      subst heq
      rw [push_getElem!_size]
      exact ⟨by simp, fun p hp => absurd hp (by simp)⟩
  | const cc =>
      intro _ t idx hlo hhi
      simp only [comp, pushT, Array.size_push] at hhi ⊢
      have heq : idx = t.deps.size := by omega
      subst heq
      rw [push_getElem!_size]
      exact ⟨by simp, fun p hp => absurd hp (by simp)⟩
  | add a b iha ihb =>
      intro hlf t idx hlo hhi
      simp only [comp, pushT, Array.size_push] at hhi ⊢
      simp only [compWeightBoundL]
      rcases Nat.lt_succ_iff_lt_or_eq.mp hhi with hlt | heq
      · rw [push_getElem!_lt _ _ idx hlt]
        by_cases hlta : idx < (comp σ a t).2.deps.size
        · rw [comp_preserve_deps σ b _ idx hlta]
          obtain ⟨hsz, hw⟩ := iha hlf.1 t idx hlo hlta
          exact ⟨hsz, fun p hp =>
            (hw p hp).trans ((le_max_left _ _).trans (le_max_right _ _))⟩
        · obtain ⟨hsz, hw⟩ := ihb hlf.2 (comp σ a t).2 idx (Nat.le_of_not_lt hlta) hlt
          exact ⟨hsz, fun p hp =>
            (hw p hp).trans ((le_max_right _ _).trans (le_max_right _ _))⟩
      · subst heq
        rw [push_getElem!_size]
        refine ⟨by simp, ?_⟩
        intro p hp
        rcases mem_pair_arr hp with rfl | rfl
        · exact oneLit_mag.trans (le_max_left _ _)
        · exact oneLit_mag.trans (le_max_left _ _)
  | sub a b iha ihb =>
      intro hlf t idx hlo hhi
      simp only [comp, pushT, Array.size_push] at hhi ⊢
      simp only [compWeightBoundL]
      rcases Nat.lt_succ_iff_lt_or_eq.mp hhi with hlt | heq
      · rw [push_getElem!_lt _ _ idx hlt]
        by_cases hlta : idx < (comp σ a t).2.deps.size
        · rw [comp_preserve_deps σ b _ idx hlta]
          obtain ⟨hsz, hw⟩ := iha hlf.1 t idx hlo hlta
          exact ⟨hsz, fun p hp =>
            (hw p hp).trans ((le_max_left _ _).trans (le_max_right _ _))⟩
        · obtain ⟨hsz, hw⟩ := ihb hlf.2 (comp σ a t).2 idx (Nat.le_of_not_lt hlta) hlt
          exact ⟨hsz, fun p hp =>
            (hw p hp).trans ((le_max_right _ _).trans (le_max_right _ _))⟩
      · subst heq
        rw [push_getElem!_size]
        refine ⟨by simp, ?_⟩
        intro p hp
        rcases mem_pair_arr hp with rfl | rfl
        · exact oneLit_mag.trans (le_max_left _ _)
        · exact negOneLit_mag.trans (le_max_left _ _)
  | mul a b iha ihb =>
      intro hlf t idx hlo hhi
      have hb_val : (comp σ b (comp σ a t).2).2.val[(comp σ b (comp σ a t).2).1]! = evalF b σ :=
        comp_root_val σ b (comp σ a t).2
      have ha_val : (comp σ b (comp σ a t).2).2.val[(comp σ a t).1]! = evalF a σ := by
        rw [comp_preserve σ b (comp σ a t).2 (comp σ a t).1 (comp_root_lt σ a t)]
        exact comp_root_val σ a t
      simp only [comp, pushT, Array.size_push] at hhi ⊢
      simp only [compWeightBoundL]
      rcases Nat.lt_succ_iff_lt_or_eq.mp hhi with hlt | heq
      · rw [push_getElem!_lt _ _ idx hlt]
        by_cases hlta : idx < (comp σ a t).2.deps.size
        · rw [comp_preserve_deps σ b _ idx hlta]
          obtain ⟨hsz, hw⟩ := iha hlf.1 t idx hlo hlta
          exact ⟨hsz, fun p hp =>
            (hw p hp).trans ((le_max_left _ _).trans (le_max_right _ _))⟩
        · obtain ⟨hsz, hw⟩ := ihb hlf.2 (comp σ a t).2 idx (Nat.le_of_not_lt hlta) hlt
          exact ⟨hsz, fun p hp =>
            (hw p hp).trans ((le_max_right _ _).trans (le_max_right _ _))⟩
      · subst heq
        rw [push_getElem!_size]
        refine ⟨by simp, ?_⟩
        intro p hp
        rcases mem_pair_arr hp with rfl | rfl
        · have h : |toReal ((comp σ b (comp σ a t).2).2.val[(comp σ b (comp σ a t).2).1]!)|
              ≤ fwdBoundL B c b := by
            rw [hb_val]; exact evalFL_mag σ B c hc hσ b hlf.2
          exact h.trans ((le_max_right _ _).trans (le_max_left _ _))
        · have h : |toReal ((comp σ b (comp σ a t).2).2.val[(comp σ a t).1]!)|
              ≤ fwdBoundL B c a := by
            rw [ha_val]; exact evalFL_mag σ B c hc hσ a hlf.1
          exact h.trans ((le_max_left _ _).trans (le_max_left _ _))
  | scale cc a iha =>
      intro hlf t idx hlo hhi
      simp only [comp, pushT, Array.size_push] at hhi ⊢
      simp only [compWeightBoundL]
      rcases Nat.lt_succ_iff_lt_or_eq.mp hhi with hlt | heq
      · rw [push_getElem!_lt _ _ idx hlt]
        obtain ⟨hsz, hw⟩ := iha hlf t idx hlo hlt
        exact ⟨hsz, fun p hp => (hw p hp).trans (le_max_right _ _)⟩
      · subst heq
        rw [push_getElem!_size]
        refine ⟨by simp, ?_⟩
        intro p hp
        obtain rfl := mem_singleton_arr hp
        exact le_max_left _ _
  | exp a iha =>
      intro hlf t idx hlo hhi
      simp only [comp, pushT, Array.size_push] at hhi ⊢
      simp only [compWeightBoundL]
      rcases Nat.lt_succ_iff_lt_or_eq.mp hhi with hlt | heq
      · rw [push_getElem!_lt _ _ idx hlt]
        obtain ⟨hsz, hw⟩ := iha hlf t idx hlo hlt
        exact ⟨hsz, fun p hp => (hw p hp).trans (le_max_right _ _)⟩
      · subst heq
        rw [push_getElem!_size]
        refine ⟨by simp, ?_⟩
        intro p hp
        obtain rfl := mem_singleton_arr hp
        have h : |toReal (Float.exp ((comp σ a t).2.val[(comp σ a t).1]!))|
            ≤ Real.exp (fwdBoundL B c a) * (1 + expEps) := by
          rw [comp_root_val σ a t]
          exact expF_mag_le _ _ ((le_abs_self _).trans (evalFL_mag σ B c hc hσ a hlf))
        exact h.trans (le_max_left _ _)
  | log a iha =>
      intro hlf t idx hlo hhi
      have ha_val : (comp σ a t).2.val[(comp σ a t).1]! = evalF a σ := comp_root_val σ a t
      simp only [comp, pushT, Array.size_push] at hhi ⊢
      simp only [compWeightBoundL]
      rcases Nat.lt_succ_iff_lt_or_eq.mp hhi with hlt | heq
      · rw [push_getElem!_lt _ _ idx hlt]
        obtain ⟨hsz, hw⟩ := iha hlf.1 t idx hlo hlt
        exact ⟨hsz, fun p hp => (hw p hp).trans (le_max_right _ _)⟩
      · subst heq
        rw [push_getElem!_size]
        refine ⟨by simp, ?_⟩
        intro p hp
        obtain rfl := mem_singleton_arr hp
        have h : |toReal ((1.0 : Float) / (comp σ a t).2.val[(comp σ a t).1]!)|
            ≤ (1 / c) * (1 + u64) := by
          rw [ha_val]
          exact log_edge_weight_mag _ c hc hlf.2
        exact h.trans (le_max_left _ _)
  | relu a iha =>
      intro hlf t idx hlo hhi
      simp only [comp, pushT, Array.size_push] at hhi ⊢
      simp only [compWeightBoundL]
      rcases Nat.lt_succ_iff_lt_or_eq.mp hhi with hlt | heq
      · rw [push_getElem!_lt _ _ idx hlt]
        obtain ⟨hsz, hw⟩ := iha hlf t idx hlo hlt
        exact ⟨hsz, fun p hp => (hw p hp).trans (le_max_right _ _)⟩
      · subst heq
        rw [push_getElem!_size]
        refine ⟨by simp, ?_⟩
        intro p hp
        obtain rfl := mem_singleton_arr hp
        exact (ite_zero_one_mag _).trans (le_max_left _ _)
  | max a b iha ihb =>
      intro hlf t idx hlo hhi
      simp only [comp, pushT, Array.size_push] at hhi ⊢
      simp only [compWeightBoundL]
      rcases Nat.lt_succ_iff_lt_or_eq.mp hhi with hlt | heq
      · rw [push_getElem!_lt _ _ idx hlt]
        by_cases hlta : idx < (comp σ a t).2.deps.size
        · rw [comp_preserve_deps σ b _ idx hlta]
          obtain ⟨hsz, hw⟩ := iha hlf.1 t idx hlo hlta
          exact ⟨hsz, fun p hp =>
            (hw p hp).trans ((le_max_left _ _).trans (le_max_right _ _))⟩
        · obtain ⟨hsz, hw⟩ := ihb hlf.2 (comp σ a t).2 idx (Nat.le_of_not_lt hlta) hlt
          exact ⟨hsz, fun p hp =>
            (hw p hp).trans ((le_max_right _ _).trans (le_max_right _ _))⟩
      · subst heq
        rw [push_getElem!_size]
        refine ⟨by simp, ?_⟩
        intro p hp
        rcases mem_pair_arr hp with rfl | rfl
        · exact (ite_zero_one_mag _).trans (le_max_left _ _)
        · exact (ite_one_zero_mag _).trans (le_max_left _ _)
  | min a b iha ihb =>
      intro hlf t idx hlo hhi
      simp only [comp, pushT, Array.size_push] at hhi ⊢
      simp only [compWeightBoundL]
      rcases Nat.lt_succ_iff_lt_or_eq.mp hhi with hlt | heq
      · rw [push_getElem!_lt _ _ idx hlt]
        by_cases hlta : idx < (comp σ a t).2.deps.size
        · rw [comp_preserve_deps σ b _ idx hlta]
          obtain ⟨hsz, hw⟩ := iha hlf.1 t idx hlo hlta
          exact ⟨hsz, fun p hp =>
            (hw p hp).trans ((le_max_left _ _).trans (le_max_right _ _))⟩
        · obtain ⟨hsz, hw⟩ := ihb hlf.2 (comp σ a t).2 idx (Nat.le_of_not_lt hlta) hlt
          exact ⟨hsz, fun p hp =>
            (hw p hp).trans ((le_max_right _ _).trans (le_max_right _ _))⟩
      · subst heq
        rw [push_getElem!_size]
        refine ⟨by simp, ?_⟩
        intro p hp
        rcases mem_pair_arr hp with rfl | rfl
        · exact (ite_one_zero_mag _).trans (le_max_left _ _)
        · exact (ite_zero_one_mag _).trans (le_max_left _ _)

/-! ### C68's tape hypotheses, derived for the full grammar -/

/-- **`hDw` derived, full grammar**: every edge weight of the compiled floored tape is
    `≤ compWeightBoundL B c e`. -/
theorem comp_hDw_log (σ : Nat → Float) (B c : ℝ) (hc : 0 < c)
    (hσ : ∀ i, |toReal (σ i)| ≤ B) (e : Expr) (hlf : LogFloored σ c e) :
    ∀ idx, idx < (comp σ e Tape.empty).2.val.size →
      ∀ p ∈ (comp σ e Tape.empty).2.deps[idx]!, |toReal p.2| ≤ compWeightBoundL B c e := by
  intro idx hidx p hp
  have hdsz : (comp σ e Tape.empty).2.deps.size = (comp σ e Tape.empty).2.val.size :=
    comp_deps_size σ e Tape.empty rfl
  exact (comp_deps_bounded_log σ B c hc hσ e hlf Tape.empty idx (Nat.zero_le _)
    (by rw [hdsz]; exact hidx)).2 p hp

/-- **`hE` derived, full grammar**: every node of the compiled floored tape has `≤ 2` edges. -/
theorem comp_hE_log (σ : Nat → Float) (B c : ℝ) (hc : 0 < c)
    (hσ : ∀ i, |toReal (σ i)| ≤ B) (e : Expr) (hlf : LogFloored σ c e) :
    ∀ idx, idx < (comp σ e Tape.empty).2.val.size →
      ((comp σ e Tape.empty).2.deps[idx]!).size ≤ 2 := by
  intro idx hidx
  have hdsz : (comp σ e Tape.empty).2.deps.size = (comp σ e Tape.empty).2.val.size :=
    comp_deps_size σ e Tape.empty rfl
  exact (comp_deps_bounded_log σ B c hc hσ e hlf Tape.empty idx (Nat.zero_le _)
    (by rw [hdsz]; exact hidx)).1

/-- **CAPSTONE: the FULL-grammar comp-tape certificate.** For a full-grammar expression with
    floored log arguments (`LogFloored σ c e`, `0 < c`) compiled by the ACTUAL compiler `comp`
    from `Tape.empty` on inputs bounded by `B`, every adjoint the ACTUAL imperative `grads`
    engine produces is overflow-free — NO free tape hypotheses (`hwf` from C71's
    grammar-independent `comp_hwf`; `hDw`/`hE` from the full-grammar traversal), given ONLY the
    single checkable budget `sweepBound 2 (compWeightBoundL B c e) n 1 ≤ overflowBound`. This
    composes C71 (the comp-tape discharge) with C72 (the floored log bounds), completing the
    tape certificate for the entire expression grammar. -/
theorem adGrad_isFinite_comp_log (σ : Nat → Float) (B c : ℝ) (hc : 0 < c)
    (hσ : ∀ i, |toReal (σ i)| ≤ B) (e : Expr) (hlf : LogFloored σ c e) (root : V)
    (hbound : sweepBound 2 (compWeightBoundL B c e) (comp σ e Tape.empty).2.val.size 1
      ≤ overflowBound) :
    ∀ j, j < (comp σ e Tape.empty).2.val.size →
      ((grads (comp σ e Tape.empty).2 root)[j]!).isFinite = true :=
  adGrad_isFinite (comp σ e Tape.empty).2 root 2 (compWeightBoundL B c e)
    (compWeightBoundL_nonneg B c e) (comp_hwf σ e) (comp_hDw_log σ B c hc hσ e hlf)
    (comp_hE_log σ B c hc hσ e hlf) hbound

/-! ### Non-vacuity: the capstone on a genuinely log-containing expression -/

/-- A concrete floored instance: `log (const 1.0)` is `LogFloored` at floor `1` (the argument's
    value is exactly `1.0`, `toReal 1.0 = 1` by the exact literal axiom). -/
example (σ : Nat → Float) : LogFloored σ 1 (.log (.const 1.0)) :=
  ⟨trivial, le_of_eq toReal_oneLit.symm⟩

/-- The full-grammar capstone instantiates on the genuinely log-containing `log (const 1.0)`
    (budget check as hypothesis, as in C71's demonstration). -/
example (σ : Nat → Float) (B : ℝ) (hσ : ∀ i, |toReal (σ i)| ≤ B) (root : V)
    (hbound : sweepBound 2 (compWeightBoundL B 1 (.log (.const 1.0)))
      (comp σ (.log (.const 1.0)) Tape.empty).2.val.size 1 ≤ overflowBound) :
    ∀ j, j < (comp σ (.log (.const 1.0)) Tape.empty).2.val.size →
      ((grads (comp σ (.log (.const 1.0)) Tape.empty).2 root)[j]!).isFinite = true :=
  adGrad_isFinite_comp_log σ B 1 one_pos hσ (.log (.const 1.0))
    ⟨trivial, le_of_eq toReal_oneLit.symm⟩ root hbound

end Puffer.RL.CompTapeLog
