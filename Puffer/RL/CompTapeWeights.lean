/-
# Discharging the AD-tape hypotheses for comp-built tapes: `D`, `hwf`, `hE` all derived

C68 (`ADTapeFinite`) certified the ACTUAL reverse-mode engine (`grads`) overflow-free for ANY
tape under three hypotheses — in-bounds edges (`hwf`), edge weights `≤ D` (`hDw`), and `≤ E`
edges per node (`hE`) — and disclosed the discharge of `D` for `comp`-built tapes as the
mechanical remaining glue ("it needs the per-node value-readout of `comp`, a `comp_root_val`-style
traversal"). This module performs that traversal, discharging ALL THREE hypotheses for tapes the
repo's expression compiler `comp` actually produces:

* `compWeightBound B e` — the explicit edge-weight budget, recursive over the `Expr`: the
  compiler's weights are the constants `1.0`/`-1.0` (add/sub), `0/1` gates (relu/max/min), the
  `scale` coefficient `|toReal c|`, the SIBLING forward values for `mul` (bounded by C68's
  `fwdBound` via `comp_root_val`/`comp_preserve` — the per-node value readout), and the node's own
  value for `exp` (bounded by `expF_mag_le` at the argument's forward budget).
* `comp_deps_bounded` — the traversal (induction over `comp`'s compilation): every NEW node of a
  compiled `LogFree` expression has `≤ 2` edges, each with weight `≤ compWeightBound B e`. The
  invariant is purely deps-indexed (old nodes are untouched by `comp_preserve_deps`; the new root
  node's literal edge array is read off by `push_getElem!_size`).
* `comp_hwf`/`comp_hDw`/`comp_hE` — the three C68 hypotheses, DERIVED for a tape compiled from
  `Tape.empty` (the compiler's standard entry point): `hwf` from the pre-existing tree-locality
  `Central.comp_edges_range`, `hDw`/`hE` from `comp_deps_bounded`.
* **`adGrad_isFinite_comp`** (capstone) — C68's `adGrad_isFinite` with NO free tape hypotheses:
  for a `LogFree` expression compiled by the ACTUAL compiler on inputs bounded by `B`, EVERY
  adjoint (hence every leaf gradient) the ACTUAL imperative engine produces is overflow-free,
  given ONLY the single budget check
  `sweepBound 2 (compWeightBound B e) n 1 ≤ overflowBound` (`n` = compiled tape size).

**Scope (honestly disclosed).** The discharge is for tapes compiled by `comp` from `Tape.empty`
(the standard compile-from-scratch entry point) out of `LogFree` expressions — `log` remains
excluded exactly as in C68 (its edge weight `1/x` and value need a domain floor, C51's honest
pattern; `Puffer.RL.LogTapeFinite` supplies the floored log value/edge-weight bounds, so the
log-inclusive discharge is their composition with this induction — the remaining glue).
Hand-built tapes (the trainer's direct `AutoDiff.lean` programs) keep C68's
three-hypothesis interface. `compWeightBound` is conservative (a max over the per-op weight
budgets; the `mul`/`exp` weights use the forward budget `fwdBound`, which itself inflates per op).
NO new axiom: everything reduces to C68's lemmas, the pre-existing `comp` layout lemmas
(`comp_root_val`/`comp_preserve`/`comp_preserve_deps`/`Central.comp_edges_range`), and the
trusted-base literals (`toReal_zeroLit`/`toReal_oneLit`/`toReal_neg`).
-/
import Puffer.RL.ADTapeFinite

namespace Puffer.RL.CompTapeWeights

open Puffer.FloatR (toReal expEps toReal_zeroLit toReal_oneLit toReal_neg)
open Puffer.FloatR.AD (Tape V grads)
open Puffer.FloatR.ADR (Expr evalF)
open Puffer.FloatR.ADReverse (comp pushT comp_root_val comp_preserve comp_root_lt
  comp_deps_size comp_preserve_deps push_getElem!_size push_getElem!_lt)
open Puffer.RL.ADTapeFinite (LogFree fwdBound evalF_mag sweepBound adGrad_isFinite)
open Puffer.RL.FiniteBound (overflowBound)
open Puffer.RL.LossFinite (expF_mag_le)

/-! ### The explicit edge-weight budget -/

/-- **The comp-built tape's edge-weight budget**, recursive over the expression. Each case is
    `max (this node's own edge-weight budget) (the subtrees' budgets)`: constants `1` for the
    `add`/`sub`/`relu`/`max`/`min` weights (`1.0`/`-1.0`/`0/1` gates), `|toReal c|` for `scale`,
    the SIBLING forward budgets for `mul` (the compiler stores the sibling's VALUE as the weight),
    and the exp-node value budget `exp(fwdBound B a)·(1+expEps)` for `exp` (the compiler stores
    the node's own value). (`log` gets a junk `0`; the theorems require `LogFree`.) -/
noncomputable def compWeightBound (B : ℝ) : Expr → ℝ
  | .var _ => 1
  | .const _ => 1
  | .add a b => max 1 (max (compWeightBound B a) (compWeightBound B b))
  | .sub a b => max 1 (max (compWeightBound B a) (compWeightBound B b))
  | .mul a b => max (max (fwdBound B a) (fwdBound B b))
      (max (compWeightBound B a) (compWeightBound B b))
  | .scale c a => max |toReal c| (compWeightBound B a)
  | .exp a => max (Real.exp (fwdBound B a) * (1 + expEps)) (compWeightBound B a)
  | .log _ => 0
  | .relu a => max 1 (compWeightBound B a)
  | .max a b => max 1 (max (compWeightBound B a) (compWeightBound B b))
  | .min a b => max 1 (max (compWeightBound B a) (compWeightBound B b))

/-- The budget is nonnegative (for every expression — the `log` junk value is `0`). -/
theorem compWeightBound_nonneg (B : ℝ) : ∀ e : Expr, 0 ≤ compWeightBound B e := by
  intro e
  induction e with
  | var i => simp only [compWeightBound]; norm_num
  | const c => simp only [compWeightBound]; norm_num
  | add a b _ _ => simp only [compWeightBound]
                   exact le_trans zero_le_one (le_max_left _ _)
  | sub a b _ _ => simp only [compWeightBound]
                   exact le_trans zero_le_one (le_max_left _ _)
  | mul a b iha _ => simp only [compWeightBound]
                     exact iha.trans ((le_max_left _ _).trans (le_max_right _ _))
  | scale c a _ => simp only [compWeightBound]
                   exact (abs_nonneg _).trans (le_max_left _ _)
  | exp a iha => simp only [compWeightBound]
                 exact iha.trans (le_max_right _ _)
  | log a _ => simp only [compWeightBound]; exact le_refl 0
  | relu a _ => simp only [compWeightBound]
                exact le_trans zero_le_one (le_max_left _ _)
  | max a b _ _ => simp only [compWeightBound]
                   exact le_trans zero_le_one (le_max_left _ _)
  | min a b _ _ => simp only [compWeightBound]
                   exact le_trans zero_le_one (le_max_left _ _)

/-! ### Literal-weight helpers -/

theorem mem_singleton_arr {α : Type _} {p x : α} (hp : p ∈ #[x]) : p = x := by
  simpa using hp

theorem mem_pair_arr {α : Type _} {p x y : α} (hp : p ∈ #[x, y]) : p = x ∨ p = y := by
  simpa using hp

theorem oneLit_mag : |toReal (1.0 : Float)| ≤ 1 := by rw [toReal_oneLit]; simp

theorem negOneLit_mag : |toReal (-1.0 : Float)| ≤ 1 := by
  rw [toReal_neg, toReal_oneLit]; simp

theorem ite_zero_one_mag (c : Prop) [Decidable c] :
    |toReal (if c then (0.0 : Float) else 1.0)| ≤ 1 := by
  split_ifs
  · rw [toReal_zeroLit]; simp
  · rw [toReal_oneLit]; simp

theorem ite_one_zero_mag (c : Prop) [Decidable c] :
    |toReal (if c then (1.0 : Float) else 0.0)| ≤ 1 := by
  split_ifs
  · rw [toReal_oneLit]; simp
  · rw [toReal_zeroLit]; simp

/-! ### The traversal: every new node of a compiled expression is 2-edge, weight-bounded -/

/-- **The comp traversal (the crux).** For a `LogFree` expression compiled onto ANY base tape,
    every NEW node (deps index `≥ t.deps.size`) has `≤ 2` edges, each with weight
    `≤ compWeightBound B e`. Induction over the compilation: old nodes are preserved
    (`comp_preserve_deps`), the subtrees' nodes are the IHs (lifted along the `max` budget), and
    the root node's literal edge array is read off by `push_getElem!_size` — with the `mul`/`exp`
    weights identified as forward VALUES via `comp_root_val`/`comp_preserve` and bounded by C68's
    `evalF_mag`. The invariant is purely deps-indexed (no well-formedness hypothesis needed). -/
theorem comp_deps_bounded (σ : Nat → Float) (B : ℝ) (hσ : ∀ i, |toReal (σ i)| ≤ B) :
    ∀ (e : Expr), LogFree e → ∀ (t : Tape) (idx : Nat),
      t.deps.size ≤ idx → idx < (comp σ e t).2.deps.size →
      ((comp σ e t).2.deps[idx]!).size ≤ 2 ∧
        ∀ p ∈ (comp σ e t).2.deps[idx]!, |toReal p.2| ≤ compWeightBound B e := by
  intro e
  induction e with
  | var i =>
      intro _ t idx hlo hhi
      simp only [comp, pushT, Array.size_push] at hhi ⊢
      have heq : idx = t.deps.size := by omega
      subst heq
      rw [push_getElem!_size]
      exact ⟨by simp, fun p hp => absurd hp (by simp)⟩
  | const c =>
      intro _ t idx hlo hhi
      simp only [comp, pushT, Array.size_push] at hhi ⊢
      have heq : idx = t.deps.size := by omega
      subst heq
      rw [push_getElem!_size]
      exact ⟨by simp, fun p hp => absurd hp (by simp)⟩
  | add a b iha ihb =>
      intro hlf t idx hlo hhi
      simp only [comp, pushT, Array.size_push] at hhi ⊢
      simp only [compWeightBound]
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
      simp only [compWeightBound]
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
      simp only [compWeightBound]
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
              ≤ fwdBound B b := by rw [hb_val]; exact evalF_mag σ B hσ b hlf.2
          exact h.trans ((le_max_right _ _).trans (le_max_left _ _))
        · have h : |toReal ((comp σ b (comp σ a t).2).2.val[(comp σ a t).1]!)|
              ≤ fwdBound B a := by rw [ha_val]; exact evalF_mag σ B hσ a hlf.1
          exact h.trans ((le_max_left _ _).trans (le_max_left _ _))
  | scale c a iha =>
      intro hlf t idx hlo hhi
      simp only [comp, pushT, Array.size_push] at hhi ⊢
      simp only [compWeightBound]
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
      simp only [compWeightBound]
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
            ≤ Real.exp (fwdBound B a) * (1 + expEps) := by
          rw [comp_root_val σ a t]
          exact expF_mag_le _ _ ((le_abs_self _).trans (evalF_mag σ B hσ a hlf))
        exact h.trans (le_max_left _ _)
  | log a _ =>
      intro hlf
      exact hlf.elim
  | relu a iha =>
      intro hlf t idx hlo hhi
      simp only [comp, pushT, Array.size_push] at hhi ⊢
      simp only [compWeightBound]
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
      simp only [compWeightBound]
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
      simp only [compWeightBound]
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

/-! ### C68's three tape hypotheses, derived for a compile-from-scratch tape -/

/-- **`hwf` derived**: every edge of the compiled tape is in bounds — from the pre-existing
    tree-locality `Central.comp_edges_range` at the empty base tape. -/
theorem comp_hwf (σ : Nat → Float) (e : Expr) :
    ∀ idx, idx < (comp σ e Tape.empty).2.val.size →
      ∀ ed ∈ (comp σ e Tape.empty).2.deps[idx]!, ed.1 < (comp σ e Tape.empty).2.val.size :=
  fun idx hidx ed hed =>
    ((Puffer.FloatR.ADReverse.Central.comp_edges_range σ e Tape.empty rfl idx
      (Nat.zero_le _) hidx ed hed).2).trans hidx

/-- **`hDw` derived**: every edge weight of the compiled tape is `≤ compWeightBound B e`. -/
theorem comp_hDw (σ : Nat → Float) (B : ℝ) (hσ : ∀ i, |toReal (σ i)| ≤ B)
    (e : Expr) (hlf : LogFree e) :
    ∀ idx, idx < (comp σ e Tape.empty).2.val.size →
      ∀ p ∈ (comp σ e Tape.empty).2.deps[idx]!, |toReal p.2| ≤ compWeightBound B e := by
  intro idx hidx p hp
  have hdsz : (comp σ e Tape.empty).2.deps.size = (comp σ e Tape.empty).2.val.size :=
    comp_deps_size σ e Tape.empty rfl
  exact (comp_deps_bounded σ B hσ e hlf Tape.empty idx (Nat.zero_le _)
    (by rw [hdsz]; exact hidx)).2 p hp

/-- **`hE` derived**: every node of the compiled tape has `≤ 2` edges. -/
theorem comp_hE (σ : Nat → Float) (B : ℝ) (hσ : ∀ i, |toReal (σ i)| ≤ B)
    (e : Expr) (hlf : LogFree e) :
    ∀ idx, idx < (comp σ e Tape.empty).2.val.size →
      ((comp σ e Tape.empty).2.deps[idx]!).size ≤ 2 := by
  intro idx hidx
  have hdsz : (comp σ e Tape.empty).2.deps.size = (comp σ e Tape.empty).2.val.size :=
    comp_deps_size σ e Tape.empty rfl
  exact (comp_deps_bounded σ B hσ e hlf Tape.empty idx (Nat.zero_le _)
    (by rw [hdsz]; exact hidx)).1

/-- **CAPSTONE: the ACTUAL engine's gradients of a comp-built tape are overflow-free with NO free
    tape hypotheses.** For a `LogFree` expression compiled by the ACTUAL compiler `comp` from
    `Tape.empty` on inputs bounded by `B`, every adjoint the ACTUAL imperative `grads` sweep
    produces is finite, given ONLY the single checkable budget
    `sweepBound 2 (compWeightBound B e) n 1 ≤ overflowBound` — C68's `adGrad_isFinite` with `D`,
    `hwf`, and `hE` all DERIVED (`D := compWeightBound B e`, `E := 2`). -/
theorem adGrad_isFinite_comp (σ : Nat → Float) (B : ℝ) (hσ : ∀ i, |toReal (σ i)| ≤ B)
    (e : Expr) (hlf : LogFree e) (root : V)
    (hbound : sweepBound 2 (compWeightBound B e) (comp σ e Tape.empty).2.val.size 1
      ≤ overflowBound) :
    ∀ j, j < (comp σ e Tape.empty).2.val.size →
      ((grads (comp σ e Tape.empty).2 root)[j]!).isFinite = true :=
  adGrad_isFinite (comp σ e Tape.empty).2 root 2 (compWeightBound B e)
    (compWeightBound_nonneg B e) (comp_hwf σ e) (comp_hDw σ B hσ e hlf)
    (comp_hE σ B hσ e hlf) hbound

end Puffer.RL.CompTapeWeights
