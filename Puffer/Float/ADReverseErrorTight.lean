/-
Tight (per-node) reverse-sweep error bound — replacing the uniform-`M` `sweepBnd`.

`ADReverseError.sweepBnd` grows a SINGLE magnitude `M` (and error `ε`) by `MstepF ~ (1+u64)(1+W)` for EVERY
edge of the whole tape, so it compounds to `~(1+W)^(edge count)` — astronomically loose. But the `comp` tape
is a TREE (each node's adjoint is written exactly once, by its unique consumer), so a leaf adjoint is a
product of edge weights along ONE root→leaf path, `~W^depth`.

This file realizes that: the bound is a PER-INDEX pair of functions `(Mf, εf : Nat → ℝ)` instead of a global
`(M, ε)`. One edge `adj[p] += a·w` (source `a = adj[idx]`, `idx > p`) updates ONLY index `p`
(`Function.update`), using the SOURCE index's own bound `Mf idx`/`εf idx` — no global growth. Since each
node is updated once by its unique consumer, `εf[leaf]` accumulates only along the path, so it is the tight
`~depth·W^depth` bound (for contractive `|∂|≤1` nets it is `O(depth)` — linear, `= 2·depth·u64` at `W=1` —
instead of the uniform bound's super-exponential `~(1+W)^(edges)`). Same trusted base.

Layer 1 (this file, so far): the generalized per-index primitives `update_mag'`/`update_err'` (separate
target/source bounds) + `AdjBndF` + `update_AdjBndF` (one edge preserves the per-index invariant).
-/
import Puffer.Float.ADReverseError

namespace Puffer.FloatR.ADReverseErrorTight

open Puffer.FloatR
open Puffer.FloatR.ADReverse (stepNode stepNodeR bsweep bsweepR gradsF gradsF_eq_bsweep
  set!_getElem!_self set!_getElem!_ne_gen)
open Puffer.FloatR.ADReverseError (one_add_u64_nonneg stepNode_size MstepF epsStepF sweepBnd)
open Puffer.FloatR.AD (Tape V)

/-- Per-index magnitude step for one edge `adj[p] += a·w`: target bound `Mp`, source bound `Ma`. -/
noncomputable def MstepP (Mp Ma W : ℝ) : ℝ := (1 + u64) * (Mp + (1 + u64) * (Ma * W))

/-- Per-index error step: target `(Mp, εp)`, source `(Ma, εa)`. -/
noncomputable def epsStepP (Mp Ma εp εa W : ℝ) : ℝ :=
  u64 * (Mp + (1 + u64) * (Ma * W)) + εp + (u64 * (Ma * W) + W * εa)

/-- **Update magnitude, split bounds.** `|fl(vp + a·w)| ≤ MstepP Mvp Ma W`, `|vp| ≤ Mvp`, `|a| ≤ Ma`. -/
theorem update_mag' (vp a w : Float) (Mvp Ma W : ℝ)
    (hvp : |toReal vp| ≤ Mvp) (ha : |toReal a| ≤ Ma) (hw : |toReal w| ≤ W) (hMa : 0 ≤ Ma) :
    |toReal (vp + a * w)| ≤ MstepP Mvp Ma W := by
  have hmul : |toReal (a * w)| ≤ (1 + u64) * (Ma * W) := by
    calc |toReal (a * w)| ≤ (1 + u64) * |toReal a * toReal w| := mul_abs_le a w
      _ = (1 + u64) * (|toReal a| * |toReal w|) := by rw [abs_mul]
      _ ≤ (1 + u64) * (Ma * W) :=
          mul_le_mul_of_nonneg_left (mul_le_mul ha hw (abs_nonneg _) hMa) one_add_u64_nonneg
  unfold MstepP
  calc |toReal (vp + a * w)| ≤ (1 + u64) * |toReal vp + toReal (a * w)| := add_abs_le vp (a * w)
    _ ≤ (1 + u64) * (|toReal vp| + |toReal (a * w)|) :=
        mul_le_mul_of_nonneg_left (abs_add_le _ _) one_add_u64_nonneg
    _ ≤ (1 + u64) * (Mvp + (1 + u64) * (Ma * W)) :=
        mul_le_mul_of_nonneg_left (add_le_add hvp hmul) one_add_u64_nonneg

/-- **Update error, split bounds.** `|fl(vp + a·w) − (vpR + aR·toReal w)| ≤ epsStepP Mvp Ma εvp εa W`. -/
theorem update_err' (vp a w : Float) (vpR aR Mvp Ma εvp εa W : ℝ)
    (hvpE : |toReal vp - vpR| ≤ εvp) (haE : |toReal a - aR| ≤ εa)
    (hvp : |toReal vp| ≤ Mvp) (ha : |toReal a| ≤ Ma) (hw : |toReal w| ≤ W)
    (hMa : 0 ≤ Ma) (hεa : 0 ≤ εa) :
    |toReal (vp + a * w) - (vpR + aR * toReal w)| ≤ epsStepP Mvp Ma εvp εa W := by
  have hprodMW : |toReal a * toReal w| ≤ Ma * W := by
    rw [abs_mul]; exact mul_le_mul ha hw (abs_nonneg _) hMa
  have hmulE : |toReal (a * w) - aR * toReal w| ≤ u64 * (Ma * W) + W * εa := by
    have h := mulApprox_error a w aR (toReal w) εa 0 haE (by simp)
    rw [mul_zero, add_zero] at h
    exact le_trans h (add_le_add (mul_le_mul_of_nonneg_left hprodMW u64_pos.le)
      (mul_le_mul_of_nonneg_right hw hεa))
  have hmul_mag : |toReal (a * w)| ≤ (1 + u64) * (Ma * W) :=
    (mul_abs_le a w).trans (mul_le_mul_of_nonneg_left hprodMW one_add_u64_nonneg)
  have hsum : |toReal vp + toReal (a * w)| ≤ Mvp + (1 + u64) * (Ma * W) :=
    (abs_add_le _ _).trans (add_le_add hvp hmul_mag)
  have h := addApprox_error vp (a * w) vpR (aR * toReal w) εvp (u64 * (Ma * W) + W * εa) hvpE hmulE
  refine le_trans h ?_
  unfold epsStepP
  exact add_le_add (add_le_add (mul_le_mul_of_nonneg_left hsum u64_pos.le) (le_refl εvp)) (le_refl _)

/-- Per-index reverse-sweep invariant: `adjF` entrywise within magnitude `Mf j` and error `εf j`. -/
def AdjBndF (adjF : Array Float) (adjR : Array ℝ) (Mf εf : Nat → ℝ) : Prop :=
  adjF.size = adjR.size ∧
  ∀ j, j < adjF.size → |toReal (adjF[j]!)| ≤ Mf j ∧ |toReal (adjF[j]!) - adjR[j]!| ≤ εf j

/-- **One edge preserves the per-index invariant.** The edge `adj[p] += a·w` (source `a`, FIXED bounds
    `Ma`/`εa`) grows ONLY index `p`'s bound via `Function.update` — untouched indices keep theirs. -/
theorem update_AdjBndF (adjF : Array Float) (adjR : Array ℝ) (Mf εf : Nat → ℝ) (Ma εa W : ℝ)
    (p : Nat) (w a : Float) (aR : ℝ)
    (hrel : AdjBndF adjF adjR Mf εf) (hp : p < adjF.size) (hw : |toReal w| ≤ W)
    (haM : |toReal a| ≤ Ma) (haE : |toReal a - aR| ≤ εa)
    (hMa : 0 ≤ Ma) (hW : 0 ≤ W) (hεa : 0 ≤ εa) :
    AdjBndF (adjF.set! p (adjF[p]! + a * w)) (adjR.set! p (adjR[p]! + aR * toReal w))
      (Function.update Mf p (MstepP (Mf p) Ma W))
      (Function.update εf p (epsStepP (Mf p) Ma (εf p) εa W)) := by
  obtain ⟨hsz, hb⟩ := hrel
  refine ⟨by rw [Array.size_set!, Array.size_set!, hsz], ?_⟩
  intro j hj
  rw [Array.size_set!] at hj
  by_cases hjp : j = p
  · subst hjp
    rw [set!_getElem!_self adjF j _ hp, set!_getElem!_self adjR j _ (by rw [← hsz]; exact hp),
      Function.update_self, Function.update_self]
    obtain ⟨hmagp, herrp⟩ := hb j hj
    exact ⟨update_mag' (adjF[j]!) a w (Mf j) Ma W hmagp haM hw hMa,
           update_err' (adjF[j]!) a w (adjR[j]!) aR (Mf j) Ma (εf j) εa W
             herrp haE hmagp haM hw hMa hεa⟩
  · rw [set!_getElem!_ne_gen adjF p _ j (Ne.symm hjp), set!_getElem!_ne_gen adjR p _ j (Ne.symm hjp),
      Function.update_of_ne hjp, Function.update_of_ne hjp]
    exact hb j hj

theorem MstepP_nonneg (Mp Ma W : ℝ) (hMp : 0 ≤ Mp) (hMa : 0 ≤ Ma) (hW : 0 ≤ W) :
    0 ≤ MstepP Mp Ma W :=
  mul_nonneg one_add_u64_nonneg (add_nonneg hMp (mul_nonneg one_add_u64_nonneg (mul_nonneg hMa hW)))

theorem epsStepP_nonneg (Mp Ma εp εa W : ℝ) (hMp : 0 ≤ Mp) (hMa : 0 ≤ Ma) (hεp : 0 ≤ εp)
    (hεa : 0 ≤ εa) (hW : 0 ≤ W) : 0 ≤ epsStepP Mp Ma εp εa W :=
  add_nonneg (add_nonneg (mul_nonneg u64_pos.le
      (add_nonneg hMp (mul_nonneg one_add_u64_nonneg (mul_nonneg hMa hW)))) hεp)
    (add_nonneg (mul_nonneg u64_pos.le (mul_nonneg hMa hW)) (mul_nonneg hW hεa))

/-- Per-index bound accumulator over ONE node's edge list (source bounds `Ma`/`εa` FIXED). -/
noncomputable def edgeBndF (Ma εa W : ℝ) :
    List (Nat × Float) → (Nat → ℝ) × (Nat → ℝ) → (Nat → ℝ) × (Nat → ℝ)
  | [], Mε => Mε
  | e :: es, Mε => edgeBndF Ma εa W es
      (Function.update Mε.1 e.1 (MstepP (Mε.1 e.1) Ma W),
       Function.update Mε.2 e.1 (epsStepP (Mε.1 e.1) Ma (Mε.2 e.1) εa W))

/-- **One node's edge fold preserves the per-index invariant** (source `a` with FIXED bounds `Ma`/`εa`). -/
theorem foldEdgesF_error (a : Float) (aR : ℝ) (Ma εa W : ℝ) (hW : 0 ≤ W) (hMa : 0 ≤ Ma) (hεa : 0 ≤ εa)
    (haM : |toReal a| ≤ Ma) (haE : |toReal a - aR| ≤ εa) :
    ∀ (es : List (Nat × Float)) (adjF : Array Float) (adjR : Array ℝ) (Mf εf : Nat → ℝ),
      AdjBndF adjF adjR Mf εf → (∀ e ∈ es, e.1 < adjF.size ∧ |toReal e.2| ≤ W) →
      AdjBndF (es.foldl (fun adj e => adj.set! e.1 (adj[e.1]! + a * e.2)) adjF)
              (es.foldl (fun adj e => adj.set! e.1 (adj[e.1]! + aR * toReal e.2)) adjR)
              (edgeBndF Ma εa W es (Mf, εf)).1 (edgeBndF Ma εa W es (Mf, εf)).2 := by
  intro es
  induction es with
  | nil => intro adjF adjR Mf εf hrel _; simpa [edgeBndF] using hrel
  | cons e es ih =>
      intro adjF adjR Mf εf hrel hedges
      simp only [List.foldl_cons, edgeBndF]
      have he1 : e.1 < adjF.size := (hedges e List.mem_cons_self).1
      have hew : |toReal e.2| ≤ W := (hedges e List.mem_cons_self).2
      have hstep := update_AdjBndF adjF adjR Mf εf Ma εa W e.1 e.2 a aR hrel he1 hew haM haE hMa hW hεa
      have hedges' : ∀ e' ∈ es,
          e'.1 < (adjF.set! e.1 (adjF[e.1]! + a * e.2)).size ∧ |toReal e'.2| ≤ W := by
        intro e' he'; rw [Array.size_set!]; exact hedges e' (List.mem_cons_of_mem _ he')
      exact ih _ _ _ _ hstep hedges'

/-- **One reverse-sweep node step preserves the per-index invariant.** The source node `idx`'s own bounds
    `Mf idx`/`εf idx` (nonneg from the invariant) fix the edge source; only its parents' entries grow. -/
theorem stepNodeF_error (t : Tape) (adjF : Array Float) (adjR : Array ℝ) (Mf εf : Nat → ℝ)
    (idx : Nat) (W : ℝ) (hrel : AdjBndF adjF adjR Mf εf) (hidx : idx < adjF.size)
    (hedges : ∀ e ∈ t.deps[idx]!, e.1 < adjF.size ∧ |toReal e.2| ≤ W) (hW : 0 ≤ W) :
    AdjBndF (stepNode t adjF idx) (stepNodeR t adjR idx)
      (edgeBndF (Mf idx) (εf idx) W (t.deps[idx]!).toList (Mf, εf)).1
      (edgeBndF (Mf idx) (εf idx) W (t.deps[idx]!).toList (Mf, εf)).2 := by
  simp only [stepNode, stepNodeR, ← Array.foldl_toList]
  obtain ⟨hsz, hb⟩ := hrel
  have haM : |toReal (adjF[idx]!)| ≤ Mf idx := (hb idx hidx).1
  have haE : |toReal (adjF[idx]!) - adjR[idx]!| ≤ εf idx := (hb idx hidx).2
  have hMa : 0 ≤ Mf idx := le_trans (abs_nonneg _) haM
  have hεa : 0 ≤ εf idx := le_trans (abs_nonneg _) haE
  have hedges' : ∀ e ∈ (t.deps[idx]!).toList, e.1 < adjF.size ∧ |toReal e.2| ≤ W :=
    fun e he => hedges e (Array.mem_def.mpr he)
  exact foldEdgesF_error (adjF[idx]!) (adjR[idx]!) (Mf idx) (εf idx) W hW hMa hεa haM haE
    (t.deps[idx]!).toList adjF adjR Mf εf ⟨hsz, hb⟩ hedges'

/-- **The whole index-list sweep preserves the per-index invariant** (bound threaded as a pair `Mε`). -/
theorem foldIdxsF_error (t : Tape) (W : ℝ) (size : Nat) (hW : 0 ≤ W) :
    ∀ (is : List Nat) (adjF : Array Float) (adjR : Array ℝ) (Mε : (Nat → ℝ) × (Nat → ℝ)),
      AdjBndF adjF adjR Mε.1 Mε.2 → adjF.size = size →
      (∀ idx ∈ is, idx < size ∧ ∀ e ∈ t.deps[idx]!, e.1 < size ∧ |toReal e.2| ≤ W) →
      AdjBndF (is.foldl (fun adj idx => stepNode t adj idx) adjF)
              (is.foldl (fun adj idx => stepNodeR t adj idx) adjR)
              (is.foldl (fun Mε idx =>
                edgeBndF (Mε.1 idx) (Mε.2 idx) W (t.deps[idx]!).toList Mε) Mε).1
              (is.foldl (fun Mε idx =>
                edgeBndF (Mε.1 idx) (Mε.2 idx) W (t.deps[idx]!).toList Mε) Mε).2 := by
  intro is
  induction is with
  | nil => intro adjF adjR Mε hrel _ _; simpa using hrel
  | cons idx is ih =>
      intro adjF adjR Mε hrel hszeq hidxs
      simp only [List.foldl_cons]
      have hidx : idx < adjF.size := by rw [hszeq]; exact (hidxs idx List.mem_cons_self).1
      have hedges : ∀ e ∈ t.deps[idx]!, e.1 < adjF.size ∧ |toReal e.2| ≤ W := by
        intro e he; obtain ⟨h1, h2⟩ := (hidxs idx List.mem_cons_self).2 e he
        exact ⟨by rw [hszeq]; exact h1, h2⟩
      have hstep := stepNodeF_error t adjF adjR Mε.1 Mε.2 idx W hrel hidx hedges hW
      have hszeq' : (stepNode t adjF idx).size = size := by rw [stepNode_size]; exact hszeq
      have hidxs' : ∀ idx' ∈ is, idx' < size ∧ ∀ e ∈ t.deps[idx']!, e.1 < size ∧ |toReal e.2| ≤ W :=
        fun idx' hidx' => hidxs idx' (List.mem_cons_of_mem _ hidx')
      exact ih (stepNode t adjF idx) (stepNodeR t adjR idx)
        (edgeBndF (Mε.1 idx) (Mε.2 idx) W (t.deps[idx]!).toList Mε) hstep hszeq' hidxs'

/-- Per-index bound accumulator over the whole sweep window `[lo, hi)` (high→low). -/
noncomputable def sweepBndF (t : Tape) (W : ℝ) (lo hi : Nat) (Mε : (Nat → ℝ) × (Nat → ℝ)) :
    (Nat → ℝ) × (Nat → ℝ) :=
  ((List.range (hi - lo)).map (fun i => hi - 1 - i)).foldl
    (fun Mε idx => edgeBndF (Mε.1 idx) (Mε.2 idx) W (t.deps[idx]!).toList Mε) Mε

/-- **Sweep error (per-index).** `bsweep` within the per-index `sweepBndF` of `bsweepR`, entrywise. -/
theorem bsweepF_error (t : Tape) (W : ℝ) (size : Nat) (hW : 0 ≤ W)
    (lo hi : Nat) (adjF : Array Float) (adjR : Array ℝ) (Mε : (Nat → ℝ) × (Nat → ℝ))
    (hrel : AdjBndF adjF adjR Mε.1 Mε.2) (hszeq : adjF.size = size) (hhi : hi ≤ size)
    (hedges : ∀ idx, lo ≤ idx → idx < hi → ∀ e ∈ t.deps[idx]!, e.1 < size ∧ |toReal e.2| ≤ W) :
    AdjBndF (bsweep t adjF lo hi) (bsweepR t adjR lo hi)
      (sweepBndF t W lo hi Mε).1 (sweepBndF t W lo hi Mε).2 := by
  have key := foldIdxsF_error t W size hW ((List.range (hi - lo)).map (fun i => hi - 1 - i))
    adjF adjR Mε hrel hszeq ?_
  · simp only [bsweep, bsweepR, sweepBndF, List.foldl_map] at key ⊢
    exact key
  · intro idx hidx
    rw [List.mem_map] at hidx
    obtain ⟨i, hi_mem, rfl⟩ := hidx
    rw [List.mem_range] at hi_mem
    exact ⟨by omega, hedges (hi - 1 - i) (by omega) (by omega)⟩

/-- One node's edge fold preserves per-index nonnegativity of both bound functions. -/
theorem edgeBndF_nonneg (Ma εa W : ℝ) (hMa : 0 ≤ Ma) (hεa : 0 ≤ εa) (hW : 0 ≤ W) :
    ∀ (es : List (Nat × Float)) (Mε : (Nat → ℝ) × (Nat → ℝ)),
      (∀ j, 0 ≤ Mε.1 j) → (∀ j, 0 ≤ Mε.2 j) →
      (∀ j, 0 ≤ (edgeBndF Ma εa W es Mε).1 j) ∧ (∀ j, 0 ≤ (edgeBndF Ma εa W es Mε).2 j) := by
  intro es
  induction es with
  | nil => intro Mε hM hε; exact ⟨hM, hε⟩
  | cons e es ih =>
      intro Mε hM hε
      simp only [edgeBndF]
      refine ih _ ?_ ?_
      · intro j; dsimp only
        by_cases hje : j = e.1
        · subst hje; rw [Function.update_self]; exact MstepP_nonneg _ _ _ (hM _) hMa hW
        · rw [Function.update_of_ne hje]; exact hM j
      · intro j; dsimp only
        by_cases hje : j = e.1
        · subst hje; rw [Function.update_self]; exact epsStepP_nonneg _ _ _ _ _ (hM _) hMa (hε _) hεa hW
        · rw [Function.update_of_ne hje]; exact hε j

/-- The whole per-index sweep preserves nonnegativity of the error-bound function `(sweepBndF …).2`. -/
theorem sweepBndF_snd_nonneg (t : Tape) (W : ℝ) (lo hi : Nat) (hW : 0 ≤ W)
    (Mε : (Nat → ℝ) × (Nat → ℝ)) (hM : ∀ j, 0 ≤ Mε.1 j) (hε : ∀ j, 0 ≤ Mε.2 j) :
    ∀ j, 0 ≤ (sweepBndF t W lo hi Mε).2 j := by
  unfold sweepBndF
  generalize (((List.range (hi - lo)).map fun i => hi - 1 - i)) = idxs
  suffices H : ∀ (Mε : (Nat → ℝ) × (Nat → ℝ)), (∀ j, 0 ≤ Mε.1 j) → (∀ j, 0 ≤ Mε.2 j) →
      (∀ j, 0 ≤ (idxs.foldl (fun Mε idx =>
        edgeBndF (Mε.1 idx) (Mε.2 idx) W (t.deps[idx]!).toList Mε) Mε).1 j)
      ∧ (∀ j, 0 ≤ (idxs.foldl (fun Mε idx =>
        edgeBndF (Mε.1 idx) (Mε.2 idx) W (t.deps[idx]!).toList Mε) Mε).2 j) by
    exact (H Mε hM hε).2
  intro Mε
  induction idxs generalizing Mε with
  | nil => intro hM hε; exact ⟨hM, hε⟩
  | cons idx is ih =>
      intro hM hε
      simp only [List.foldl_cons]
      obtain ⟨hM', hε'⟩ := edgeBndF_nonneg (Mε.1 idx) (Mε.2 idx) W (hM idx) (hε idx) hW
        (t.deps[idx]!).toList Mε hM hε
      exact ih _ hM' hε'

/-- **Locality (frame law) of the tight per-index reverse-sweep bound.** If an index `j` is never the
    target of any dependency edge of any node processed in the window `[lo, hi)`, then the per-index bound
    `sweepBndF` leaves BOTH `j`'s magnitude entry and `j`'s error entry exactly at their seed values
    `Mε.1 j` / `Mε.2 j`. This is the formal statement of the design thesis in the file header — each edge
    `adj[p] += a·w` grows ONLY index `p`'s bound — lifted from a single edge (`update_AdjBndF`) to the WHOLE
    reverse sweep: an adjoint slot that is never written keeps its seed bound (e.g. the reverse-sweep root,
    seeded to magnitude 1 / error 0, whose adjoint no edge ever overwrites). The hypothesis is load-bearing:
    a single self-targeting edge replaces `j`'s error entry by `epsStepP ≥ εa·W + u64·(…) > 0`, breaking
    equality. Two structural inductions — over one node's edge list (`edgeFrame`, via `Function.update_of_ne`)
    then over the swept index list `(range (hi−lo)).map (hi−1−·)`. -/
theorem sweepBndF_frame (t : Tape) (W : ℝ) (lo hi : Nat)
    (Mε : (Nat → ℝ) × (Nat → ℝ)) (j : Nat)
    (hj : ∀ idx, lo ≤ idx → idx < hi → ∀ e ∈ t.deps[idx]!, e.1 ≠ j) :
    (sweepBndF t W lo hi Mε).1 j = Mε.1 j ∧ (sweepBndF t W lo hi Mε).2 j = Mε.2 j := by
  -- Edge-level frame: one node's edge fold leaves a non-target index untouched.
  have edgeFrame : ∀ (Ma εa : ℝ) (es : List (Nat × Float)) (P : (Nat → ℝ) × (Nat → ℝ)),
      (∀ e ∈ es, e.1 ≠ j) →
      (edgeBndF Ma εa W es P).1 j = P.1 j ∧ (edgeBndF Ma εa W es P).2 j = P.2 j := by
    intro Ma εa es
    induction es with
    | nil => intro P _; exact ⟨rfl, rfl⟩
    | cons e es ih =>
        intro P hes
        have hne : j ≠ e.1 := (hes e List.mem_cons_self).symm
        have htail : ∀ e' ∈ es, e'.1 ≠ j := fun e' he' => hes e' (List.mem_cons_of_mem _ he')
        simp only [edgeBndF]
        obtain ⟨h1, h2⟩ := ih (Function.update P.1 e.1 (MstepP (P.1 e.1) Ma W),
          Function.update P.2 e.1 (epsStepP (P.1 e.1) Ma (P.2 e.1) εa W)) htail
        refine ⟨?_, ?_⟩
        · rw [h1]; simp only [Function.update_of_ne hne]
        · rw [h2]; simp only [Function.update_of_ne hne]
  -- Fold-level frame over the swept index list.
  unfold sweepBndF
  suffices H : ∀ (idxs : List Nat),
      (∀ idx ∈ idxs, ∀ e ∈ t.deps[idx]!, e.1 ≠ j) →
      ∀ (P : (Nat → ℝ) × (Nat → ℝ)),
        (idxs.foldl (fun P idx =>
            edgeBndF (P.1 idx) (P.2 idx) W (t.deps[idx]!).toList P) P).1 j = P.1 j ∧
        (idxs.foldl (fun P idx =>
            edgeBndF (P.1 idx) (P.2 idx) W (t.deps[idx]!).toList P) P).2 j = P.2 j by
    refine H _ ?_ Mε
    intro idx hidx e he
    rw [List.mem_map] at hidx
    obtain ⟨i, hi_mem, rfl⟩ := hidx
    rw [List.mem_range] at hi_mem
    exact hj (hi - 1 - i) (by omega) (by omega) e he
  intro idxs
  induction idxs with
  | nil => intro _ P; exact ⟨rfl, rfl⟩
  | cons idx idxs ih =>
      intro hidxs P
      simp only [List.foldl_cons]
      have htail : ∀ idx' ∈ idxs, ∀ e ∈ t.deps[idx']!, e.1 ≠ j :=
        fun idx' hidx' => hidxs idx' (List.mem_cons_of_mem _ hidx')
      have hnode : ∀ e ∈ (t.deps[idx]!).toList, e.1 ≠ j :=
        fun e he => hidxs idx List.mem_cons_self e (Array.mem_def.mpr he)
      obtain ⟨he1, he2⟩ := edgeFrame (P.1 idx) (P.2 idx) (t.deps[idx]!).toList P hnode
      obtain ⟨hr1, hr2⟩ := ih htail (edgeBndF (P.1 idx) (P.2 idx) W (t.deps[idx]!).toList P)
      exact ⟨by rw [hr1, he1], by rw [hr2, he2]⟩

/-- The per-index seed bound: magnitude `1` at the root (seeded to `1.0`), `0` elsewhere; error `0`. -/
theorem seed_AdjBndF (n root : Nat) (hroot : root < n) :
    AdjBndF ((Array.replicate n (0.0:Float)).set! root 1.0)
      ((Array.replicate n (0:ℝ)).set! root 1)
      (fun j => if j = root then 1 else 0) (fun _ => 0) := by
  refine ⟨by rw [Array.size_set!, Array.size_set!, Array.size_replicate, Array.size_replicate], ?_⟩
  intro j hj
  rw [Array.size_set!, Array.size_replicate] at hj
  by_cases hjr : j = root
  · subst hjr
    rw [set!_getElem!_self _ j _ (by rw [Array.size_replicate]; exact hj),
      set!_getElem!_self _ j _ (by rw [Array.size_replicate]; exact hj)]
    simp [toReal_oneLit]
  · rw [set!_getElem!_ne_gen _ root _ j (Ne.symm hjr), set!_getElem!_ne_gen _ root _ j (Ne.symm hjr),
      getElem!_pos _ j (by rw [Array.size_replicate]; exact hj),
      getElem!_pos _ j (by rw [Array.size_replicate]; exact hj),
      Array.getElem_replicate, Array.getElem_replicate]
    simp [hjr, toReal_zeroLit]

/-- **The tight per-index reverse-sweep error.** Every entry of `gradsF t root` is within the PER-INDEX
    `sweepBndF` of the exact-ℝ sweep `bsweepR` — the tight (tree/path) replacement for the uniform
    `ADReverseError.gradsF_error`. The bound at each leaf accumulates only along that leaf's root-path. -/
theorem gradsF_errorF (t : Tape) (root : V) (W : ℝ) (hW : 0 ≤ W) (hroot : root < t.val.size)
    (hedges : ∀ idx, idx < t.val.size → ∀ e ∈ t.deps[idx]!, e.1 < t.val.size ∧ |toReal e.2| ≤ W) :
    AdjBndF (gradsF t root)
      (bsweepR t ((Array.replicate t.val.size (0:ℝ)).set! root 1) 0 t.val.size)
      (sweepBndF t W 0 t.val.size ((fun j => if j = root then 1 else 0), (fun _ => 0))).1
      (sweepBndF t W 0 t.val.size ((fun j => if j = root then 1 else 0), (fun _ => 0))).2 := by
  rw [gradsF_eq_bsweep]
  exact bsweepF_error t W t.val.size hW 0 t.val.size _ _
    ((fun j => if j = root then 1 else 0), (fun _ => 0))
    (seed_AdjBndF t.val.size root hroot) (by rw [Array.size_set!, Array.size_replicate]) (le_refl _)
    (fun idx _ hidx => hedges idx hidx)

/-! ### Concrete demonstration: the tight bound is STRICTLY smaller than the uniform one -/

/-- A concrete depth-2 chain tape (`node2 → node1 → node0`, the shape `comp` builds for `scale c (scale c
    (var 0))`): three nodes, `deps[2]=[(1,w)]`, `deps[1]=[(0,w)]`, `deps[0]=[]`, root = node 2. (The `val`
    entries are irrelevant — both bounds fold the edge-weight bound `W` over the deps STRUCTURE, not the
    primal.) -/
def chainTape (w : Float) : Tape := { val := #[0.0, 0.0, 0.0], deps := #[#[], #[(0, w)], #[(1, w)]] }

/-- **The tight per-index bound is STRICTLY smaller than the uniform bound — machine-checked on a concrete
    tape.** For the depth-2 chain, the LEAF's per-index sweep error `(sweepBndF …).2 0` is strictly less than
    the uniform `(sweepBnd …).2`, for EVERY `W ≥ 0`. Both sides reduce (`rfl`) to their closed forms; the
    difference is `2·u64 + u64² + (6·u64 + 6·u64² + 2·u64³)·W ≥ 2·u64 > 0`. A permanent, in-repo witness that
    the per-node refactor genuinely tightens the bound (not merely restructures it). -/
theorem sweepBndF_lt_sweepBnd_chain2 (w : Float) (W : ℝ) (hW : 0 ≤ W) :
    (sweepBndF (chainTape w) W 0 3 ((fun j => if j = 2 then (1:ℝ) else 0), (fun _ => 0))).2 0
      < (sweepBnd (chainTape w) W 0 3 (1, 0)).2 := by
  have hlhs :
      (sweepBndF (chainTape w) W 0 3 ((fun j => if j = 2 then (1:ℝ) else 0), (fun _ => 0))).2 0
        = epsStepP 0 (MstepP 0 1 W) 0 (epsStepP 0 1 0 0 W) W := rfl
  have hrhs : (sweepBnd (chainTape w) W 0 3 (1, 0)).2 = epsStepF (MstepF 1 W) (epsStepF 1 0 W) W := rfl
  rw [hlhs, hrhs]
  simp only [epsStepP, MstepP, epsStepF, MstepF]
  nlinarith [u64_pos, hW, mul_nonneg u64_pos.le hW,
    mul_nonneg (mul_nonneg u64_pos.le u64_pos.le) hW,
    mul_nonneg (mul_nonneg (mul_nonneg u64_pos.le u64_pos.le) u64_pos.le) hW,
    mul_pos u64_pos u64_pos]

end Puffer.FloatR.ADReverseErrorTight
