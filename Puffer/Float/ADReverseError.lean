/-
Layer 2 of the reverse-mode-tape equivalence: the FLOAT-ROUNDING bound.

`ADReverse.lean` proves the flat reverse sweep correct at the exact-ℝ STRUCTURE layer — the ℝ sweep
`bsweepR` (the ℝ image of the actual Float `gradsF`/`bsweep` fold) reads off the reverse-mode gradient
(`bsweepR_reads_revE_RF`). What it does NOT do is bound the Float sweep's own rounding: `stepNode` uses
rounded `+`/`*` (`adj[e.1] + a * e.2`), whereas `stepNodeR` does the same fold in exact ℝ. This file
attacks that gap — `|toReal (bsweep …) − bsweepR …|` — the εg tail bracketed there.

This module proves the ARITHMETIC CORE: the per-operation rounding bounds for one reverse-mode update
`vp + a·w`, tracking both a magnitude bound (`update_mag`) and the Float↔ℝ error (`update_err`) via the
`(1+δ)` model. These are the atoms any full sweep bound composes.

The core is then LIFTED through the mutable-array `stepNode`/`bsweep` fold to a CLOSED per-entry bound
on the whole sweep (`gradsF_error`: the literal Float `gradsF` output is entrywise within `sweepBnd` of
the exact-ℝ sweep `bsweepR`). As warned, the constant `sweepBnd` is ASTRONOMICALLY loose — each edge
multiplies the magnitude bound by `~(1+u64)(1+W)` and reverse-mode adjoints are products of edge weights
(geometric growth), exactly like the `newtonSchulz` `matmul` constant — but it is a genuine, closed,
machine-checked bound. Composed with `ADReverse.bsweepR_reads_revE_RF` (which reads `bsweepR` at the
leaves), this connects the literal Float engine `gradsF` to the reverse-mode gradient with a proven
error, completing the Float↔ℝ story the ℝ-structure layer deferred.
-/
import Mathlib
import Puffer.Float.ADReverse

namespace Puffer.FloatR.ADReverseError

open Puffer.FloatR
open Puffer.FloatR.ADReverse (stepNode stepNodeR bsweep bsweepR gradsF gradsF_eq_bsweep set!_getElem!_self set!_getElem!_ne_gen)
open Puffer.FloatR.AD (Tape V)

/-- `0 ≤ 1 + u64`. -/
theorem one_add_u64_nonneg : (0 : ℝ) ≤ 1 + u64 := by linarith [u64_pos]

/-- Magnitude bound after one rounded reverse-mode update `vp + a·w`: each of the two ops grows the
    bound by a `(1+u64)` factor. -/
noncomputable def MstepF (M W : ℝ) : ℝ := (1 + u64) * (M + (1 + u64) * (M * W))

/-- Float↔ℝ error bound after one rounded reverse-mode update: the add- and mul-rounding of this step
    plus the propagated input errors. -/
noncomputable def epsStepF (M ε W : ℝ) : ℝ :=
  u64 * (M + (1 + u64) * (M * W)) + ε + (u64 * (M * W) + W * ε)

/-- **Update magnitude.** `|fl(vp + a·w)| ≤ MstepF M W`, given `|vp|,|a| ≤ M` and `|w| ≤ W`. -/
theorem update_mag (vp a w : Float) (M W : ℝ)
    (hvp : |toReal vp| ≤ M) (ha : |toReal a| ≤ M) (hw : |toReal w| ≤ W) (hM : 0 ≤ M) :
    |toReal (vp + a * w)| ≤ MstepF M W := by
  have hmul : |toReal (a * w)| ≤ (1 + u64) * (M * W) := by
    calc |toReal (a * w)| ≤ (1 + u64) * |toReal a * toReal w| := mul_abs_le a w
      _ = (1 + u64) * (|toReal a| * |toReal w|) := by rw [abs_mul]
      _ ≤ (1 + u64) * (M * W) :=
          mul_le_mul_of_nonneg_left (mul_le_mul ha hw (abs_nonneg _) hM) one_add_u64_nonneg
  calc |toReal (vp + a * w)| ≤ (1 + u64) * |toReal vp + toReal (a * w)| := add_abs_le vp (a * w)
    _ ≤ (1 + u64) * (|toReal vp| + |toReal (a * w)|) :=
        mul_le_mul_of_nonneg_left (abs_add_le _ _) one_add_u64_nonneg
    _ ≤ (1 + u64) * (M + (1 + u64) * (M * W)) :=
        mul_le_mul_of_nonneg_left (add_le_add hvp hmul) one_add_u64_nonneg

/-- **Update error.** `|fl(vp + a·w) − (vpR + aR·toReal w)| ≤ epsStepF M ε W`, given the input
    magnitude bounds (`|vp|,|a| ≤ M`, `|w| ≤ W`) and error bounds (`|vp−vpR|,|a−aR| ≤ ε`). The ℝ side
    uses the EXACT weight `toReal w` (the tape's Float weight has no error against its own real value).-/
theorem update_err (vp a w : Float) (vpR aR M ε W : ℝ)
    (hvpE : |toReal vp - vpR| ≤ ε) (haE : |toReal a - aR| ≤ ε)
    (hvp : |toReal vp| ≤ M) (ha : |toReal a| ≤ M) (hw : |toReal w| ≤ W)
    (hM : 0 ≤ M) (hε : 0 ≤ ε) :
    |toReal (vp + a * w) - (vpR + aR * toReal w)| ≤ epsStepF M ε W := by
  have hprodMW : |toReal a * toReal w| ≤ M * W := by
    rw [abs_mul]; exact mul_le_mul ha hw (abs_nonneg _) hM
  have hmulE : |toReal (a * w) - aR * toReal w| ≤ u64 * (M * W) + W * ε := by
    have h := mulApprox_error a w aR (toReal w) ε 0 haE (by simp)
    rw [mul_zero, add_zero] at h
    exact le_trans h (add_le_add (mul_le_mul_of_nonneg_left hprodMW u64_pos.le)
      (mul_le_mul_of_nonneg_right hw hε))
  have hmul_mag : |toReal (a * w)| ≤ (1 + u64) * (M * W) :=
    (mul_abs_le a w).trans (mul_le_mul_of_nonneg_left hprodMW one_add_u64_nonneg)
  have hsum : |toReal vp + toReal (a * w)| ≤ M + (1 + u64) * (M * W) :=
    (abs_add_le _ _).trans (add_le_add hvp hmul_mag)
  have h := addApprox_error vp (a * w) vpR (aR * toReal w) ε (u64 * (M * W) + W * ε) hvpE hmulE
  refine le_trans h ?_
  unfold epsStepF
  exact add_le_add (add_le_add (mul_le_mul_of_nonneg_left hsum u64_pos.le) (le_refl ε)) (le_refl _)

/-- `MstepF` is nonnegative (given `M,W ≥ 0`), and `M ≤ MstepF M W` — the bound only grows. -/
theorem le_MstepF (M W : ℝ) (hM : 0 ≤ M) (hW : 0 ≤ W) : M ≤ MstepF M W := by
  unfold MstepF
  have h1 : (0:ℝ) ≤ (1 + u64) * (M * W) := mul_nonneg one_add_u64_nonneg (mul_nonneg hM hW)
  calc M = 1 * M := (one_mul M).symm
    _ ≤ (1 + u64) * M := by
        apply mul_le_mul_of_nonneg_right _ hM; linarith [u64_pos]
    _ ≤ (1 + u64) * (M + (1 + u64) * (M * W)) :=
        mul_le_mul_of_nonneg_left (by linarith) one_add_u64_nonneg

theorem MstepF_nonneg (M W : ℝ) (hM : 0 ≤ M) (hW : 0 ≤ W) : 0 ≤ MstepF M W :=
  le_trans hM (le_MstepF M W hM hW)

/-- `ε ≤ epsStepF M ε W` — the error bound only grows. -/
theorem le_epsStepF (M ε W : ℝ) (hM : 0 ≤ M) (hε : 0 ≤ ε) (hW : 0 ≤ W) : ε ≤ epsStepF M ε W := by
  unfold epsStepF
  have h1 : (0:ℝ) ≤ u64 * (M + (1 + u64) * (M * W)) :=
    mul_nonneg u64_pos.le (by nlinarith [one_add_u64_nonneg, mul_nonneg hM hW])
  have h2 : (0:ℝ) ≤ u64 * (M * W) + W * ε :=
    add_nonneg (mul_nonneg u64_pos.le (mul_nonneg hM hW)) (mul_nonneg hW hε)
  linarith

/-! ### The full sweep bound: lifting the per-op core through the mutable-array fold

`AdjBnd adjF adjR M ε` = the Float array `adjF` is entrywise within magnitude `M` and error `ε` of the
ℝ array `adjR`. `update_AdjBnd` (one edge) → `foldEdges_error` (one node's edges) → `stepNode_error`
(one node) → `foldIdxs_error`/`bsweep_error` (the whole sweep) → `gradsF_error` (the literal `gradsF`).
The bound `sweepBnd` is a fold of `edgeBnd` over the swept nodes — a valid but ASTRONOMICALLY loose
constant (each edge multiplies the magnitude bound by `~(1+u64)(1+W)`, and reverse-mode adjoints are
products of edge weights). This closes the Float↔ℝ Layer-2 gap for the reverse sweep. -/

/-- Float array `adjF` is within (magnitude `M`, error `ε`) of ℝ array `adjR`, entrywise (in bounds). -/
def AdjBnd (adjF : Array Float) (adjR : Array ℝ) (M ε : ℝ) : Prop :=
  adjF.size = adjR.size ∧
  ∀ j, j < adjF.size → |toReal (adjF[j]!)| ≤ M ∧ |toReal (adjF[j]!) - adjR[j]!| ≤ ε

/-- One reverse-mode edge update (at index `p`, weight `w`, with fixed adjoints `a`/`aR`) preserves
    `AdjBnd`, growing the bounds by one `MstepF`/`epsStepF` step. -/
theorem update_AdjBnd (adjF : Array Float) (adjR : Array ℝ) (M ε W : ℝ) (p : Nat) (w a : Float) (aR : ℝ)
    (hrel : AdjBnd adjF adjR M ε) (hp : p < adjF.size) (hw : |toReal w| ≤ W)
    (haM : |toReal a| ≤ M) (haE : |toReal a - aR| ≤ ε)
    (hM : 0 ≤ M) (hW : 0 ≤ W) (hε : 0 ≤ ε) :
    AdjBnd (adjF.set! p (adjF[p]! + a * w)) (adjR.set! p (adjR[p]! + aR * toReal w))
      (MstepF M W) (epsStepF M ε W) := by
  obtain ⟨hsz, hb⟩ := hrel
  refine ⟨by rw [Array.size_set!, Array.size_set!, hsz], ?_⟩
  intro j hj
  rw [Array.size_set!] at hj
  by_cases hjp : j = p
  · subst hjp
    rw [set!_getElem!_self adjF j _ hp, set!_getElem!_self adjR j _ (by rw [← hsz]; exact hp)]
    obtain ⟨hmag, herr⟩ := hb j hp
    exact ⟨update_mag (adjF[j]!) a w M W hmag haM hw hM,
           update_err (adjF[j]!) a w (adjR[j]!) aR M ε W herr haE hmag haM hw hM hε⟩
  · rw [set!_getElem!_ne_gen adjF p _ j (Ne.symm hjp), set!_getElem!_ne_gen adjR p _ j (Ne.symm hjp)]
    obtain ⟨hmag, herr⟩ := hb j hj
    exact ⟨le_trans hmag (le_MstepF M W hM hW), le_trans herr (le_epsStepF M ε W hM hε hW)⟩


/-- The per-edge bound accumulator: each edge grows `(M, ε)` by one `MstepF`/`epsStepF`. -/
noncomputable def edgeBnd (W : ℝ) : List (Nat × Float) → ℝ × ℝ → ℝ × ℝ
  | [], p => p
  | _ :: es, p => edgeBnd W es (MstepF p.1 W, epsStepF p.1 p.2 W)

/-- **Geometric growth of the reverse-mode magnitude bound (exact closed form).** Processing a list `es`
    of reverse-mode edges multiplies the magnitude-bound component of the `edgeBnd` accumulator by EXACTLY
    the geometric factor `((1+u64)(1+(1+u64)·W))^(es.length)`:
      `(edgeBnd W es (M, ε)).1 = ((1+u64)·(1+(1+u64)·W))^(es.length) · M`.
    The per-edge magnitude update `MstepF` is affine-homogeneous in `M` (`MstepF M W = c(W)·M` with
    `c(W) = (1+u64)(1+(1+u64)·W)`), so the fold is a pure geometric progression in the number of edges. This
    is the exact form of the "geometric growth" the module docstring flags as the source of the
    astronomically loose sweep constant, and it is unconditional (holds for every `W`, `M`, `ε`). Note the
    first component is independent of the initial error `ε` — only the edge COUNT and the initial magnitude
    `M` matter. -/
theorem edgeBnd_fst_pow (W : ℝ) :
    ∀ (es : List (Nat × Float)) (M ε : ℝ),
      (edgeBnd W es (M, ε)).1 = ((1 + u64) * (1 + (1 + u64) * W)) ^ es.length * M := by
  intro es
  induction es with
  | nil => intro M ε; simp [edgeBnd]
  | cons e es ih =>
      intro M ε
      simp only [edgeBnd]
      rw [ih (MstepF M W) (epsStepF M ε W)]
      have hM : MstepF M W = (1 + u64) * (1 + (1 + u64) * W) * M := by unfold MstepF; ring
      rw [hM, List.length_cons, pow_succ]
      ring

theorem foldEdges_error (a : Float) (aR : ℝ) (W : ℝ) (hW : 0 ≤ W) :
    ∀ (es : List (Nat × Float)) (adjF : Array Float) (adjR : Array ℝ) (M ε : ℝ),
      AdjBnd adjF adjR M ε → (∀ e ∈ es, e.1 < adjF.size ∧ |toReal e.2| ≤ W) →
      |toReal a| ≤ M → |toReal a - aR| ≤ ε → 0 ≤ M → 0 ≤ ε →
      AdjBnd (es.foldl (fun adj e => adj.set! e.1 (adj[e.1]! + a * e.2)) adjF)
             (es.foldl (fun adj e => adj.set! e.1 (adj[e.1]! + aR * toReal e.2)) adjR)
             (edgeBnd W es (M, ε)).1 (edgeBnd W es (M, ε)).2 := by
  intro es
  induction es with
  | nil => intro adjF adjR M ε hrel _ _ _ _ _; simpa [edgeBnd] using hrel
  | cons e es ih =>
      intro adjF adjR M ε hrel hedges haM haE hM hε
      simp only [List.foldl_cons, edgeBnd]
      have he1 : e.1 < adjF.size := (hedges e (List.mem_cons_self)).1
      have hew : |toReal e.2| ≤ W := (hedges e (List.mem_cons_self)).2
      have hstep := update_AdjBnd adjF adjR M ε W e.1 e.2 a aR hrel he1 hew haM haE hM hW hε
      have haM' : |toReal a| ≤ MstepF M W := le_trans haM (le_MstepF M W hM hW)
      have haE' : |toReal a - aR| ≤ epsStepF M ε W := le_trans haE (le_epsStepF M ε W hM hε hW)
      have hM' : 0 ≤ MstepF M W := MstepF_nonneg M W hM hW
      have hε' : 0 ≤ epsStepF M ε W := le_trans hε (le_epsStepF M ε W hM hε hW)
      have hedges' : ∀ e' ∈ es, e'.1 < (adjF.set! e.1 (adjF[e.1]! + a * e.2)).size ∧ |toReal e'.2| ≤ W := by
        intro e' he'; rw [Array.size_set!]; exact hedges e' (List.mem_cons_of_mem _ he')
      exact ih _ _ (MstepF M W) (epsStepF M ε W) hstep hedges' haM' haE' hM' hε'


theorem stepNode_error (t : Tape) (adjF : Array Float) (adjR : Array ℝ) (idx : Nat) (M ε W : ℝ)
    (hrel : AdjBnd adjF adjR M ε) (hidx : idx < adjF.size)
    (hedges : ∀ e ∈ t.deps[idx]!, e.1 < adjF.size ∧ |toReal e.2| ≤ W)
    (hM : 0 ≤ M) (hW : 0 ≤ W) (hε : 0 ≤ ε) :
    AdjBnd (stepNode t adjF idx) (stepNodeR t adjR idx)
      (edgeBnd W (t.deps[idx]!).toList (M, ε)).1 (edgeBnd W (t.deps[idx]!).toList (M, ε)).2 := by
  simp only [stepNode, stepNodeR, ← Array.foldl_toList]
  obtain ⟨hsz, hb⟩ := hrel
  have haM : |toReal (adjF[idx]!)| ≤ M := (hb idx hidx).1
  have haE : |toReal (adjF[idx]!) - adjR[idx]!| ≤ ε := (hb idx hidx).2
  have hedges' : ∀ e ∈ (t.deps[idx]!).toList, e.1 < adjF.size ∧ |toReal e.2| ≤ W := by
    intro e he; exact hedges e (Array.mem_def.mpr he)
  exact foldEdges_error (adjF[idx]!) (adjR[idx]!) W hW (t.deps[idx]!).toList adjF adjR M ε
    ⟨hsz, hb⟩ hedges' haM haE hM hε


theorem stepNode_size (t : Tape) (adj : Array Float) (idx : Nat) :
    (stepNode t adj idx).size = adj.size := by
  show (Array.foldl _ adj (t.deps[idx]!)).size = adj.size
  refine Array.foldl_induction (motive := fun _ (acc : Array Float) => acc.size = adj.size) rfl (fun i acc h => ?_)
  simp only []; rw [Array.size_set!]; exact h

theorem edgeBnd_nonneg (W : ℝ) (hW : 0 ≤ W) : ∀ (es : List (Nat × Float)) (M ε : ℝ),
    0 ≤ M → 0 ≤ ε → 0 ≤ (edgeBnd W es (M, ε)).1 ∧ 0 ≤ (edgeBnd W es (M, ε)).2 := by
  intro es
  induction es with
  | nil => intro M ε hM hε; exact ⟨hM, hε⟩
  | cons e es ih =>
      intro M ε hM hε
      simp only [edgeBnd]
      exact ih (MstepF M W) (epsStepF M ε W) (MstepF_nonneg M W hM hW)
        (le_trans hε (le_epsStepF M ε W hM hε hW))

theorem foldIdxs_error (t : Tape) (W : ℝ) (size : Nat) (hW : 0 ≤ W) :
    ∀ (is : List Nat) (adjF : Array Float) (adjR : Array ℝ) (M ε : ℝ),
      AdjBnd adjF adjR M ε → adjF.size = size →
      (∀ idx ∈ is, idx < size ∧ ∀ e ∈ t.deps[idx]!, e.1 < size ∧ |toReal e.2| ≤ W) →
      0 ≤ M → 0 ≤ ε →
      AdjBnd (is.foldl (fun adj idx => stepNode t adj idx) adjF)
             (is.foldl (fun adj idx => stepNodeR t adj idx) adjR)
             (is.foldl (fun p idx => edgeBnd W (t.deps[idx]!).toList p) (M, ε)).1
             (is.foldl (fun p idx => edgeBnd W (t.deps[idx]!).toList p) (M, ε)).2 := by
  intro is
  induction is with
  | nil => intro adjF adjR M ε hrel _ _ _ _; simpa using hrel
  | cons idx is ih =>
      intro adjF adjR M ε hrel hszeq hidxs hM hε
      simp only [List.foldl_cons]
      have hidx : idx < adjF.size := by rw [hszeq]; exact (hidxs idx List.mem_cons_self).1
      have hedges : ∀ e ∈ t.deps[idx]!, e.1 < adjF.size ∧ |toReal e.2| ≤ W := by
        intro e he; obtain ⟨h1, h2⟩ := (hidxs idx List.mem_cons_self).2 e he
        exact ⟨by rw [hszeq]; exact h1, h2⟩
      have hstep := stepNode_error t adjF adjR idx M ε W hrel hidx hedges hM hW hε
      obtain ⟨hM', hε'⟩ := edgeBnd_nonneg W hW (t.deps[idx]!).toList M ε hM hε
      have hszeq' : (stepNode t adjF idx).size = size := by rw [stepNode_size]; exact hszeq
      have hidxs' : ∀ idx' ∈ is, idx' < size ∧ ∀ e ∈ t.deps[idx']!, e.1 < size ∧ |toReal e.2| ≤ W :=
        fun idx' hidx' => hidxs idx' (List.mem_cons_of_mem _ hidx')
      exact ih (stepNode t adjF idx) (stepNodeR t adjR idx)
        (edgeBnd W (t.deps[idx]!).toList (M, ε)).1 (edgeBnd W (t.deps[idx]!).toList (M, ε)).2
        hstep hszeq' hidxs' hM' hε'


/-- The bound accumulator over the whole sweep (folds `edgeBnd` over the swept node indices). -/
noncomputable def sweepBnd (t : Tape) (W : ℝ) (lo hi : Nat) (p : ℝ × ℝ) : ℝ × ℝ :=
  ((List.range (hi - lo)).map (fun i => hi - 1 - i)).foldl
    (fun p idx => edgeBnd W (t.deps[idx]!).toList p) p

/-- **Sweep error.** The Float sweep `bsweep` is within `sweepBnd` of the exact-ℝ sweep `bsweepR`,
    entrywise, given the seed relation and tape-wide edge bounds. -/
theorem bsweep_error (t : Tape) (W : ℝ) (size : Nat) (hW : 0 ≤ W)
    (lo hi : Nat) (adjF : Array Float) (adjR : Array ℝ) (M ε : ℝ)
    (hrel : AdjBnd adjF adjR M ε) (hszeq : adjF.size = size) (hhi : hi ≤ size)
    (hedges : ∀ idx, lo ≤ idx → idx < hi → ∀ e ∈ t.deps[idx]!, e.1 < size ∧ |toReal e.2| ≤ W)
    (hM : 0 ≤ M) (hε : 0 ≤ ε) :
    AdjBnd (bsweep t adjF lo hi) (bsweepR t adjR lo hi)
      (sweepBnd t W lo hi (M, ε)).1 (sweepBnd t W lo hi (M, ε)).2 := by
  have key := foldIdxs_error t W size hW ((List.range (hi - lo)).map (fun i => hi - 1 - i))
    adjF adjR M ε hrel hszeq ?_ hM hε
  · rw [List.foldl_map, List.foldl_map] at key
    exact key
  · intro idx hidx
    rw [List.mem_map] at hidx
    obtain ⟨i, hi_mem, rfl⟩ := hidx
    rw [List.mem_range] at hi_mem
    exact ⟨by omega, hedges (hi - 1 - i) (by omega) (by omega)⟩


theorem seed_AdjBnd (n root : Nat) (hroot : root < n) :
    AdjBnd ((Array.replicate n (0.0:Float)).set! root 1.0)
      ((Array.replicate n (0:ℝ)).set! root 1) 1 0 := by
  refine ⟨by rw [Array.size_set!, Array.size_set!, Array.size_replicate, Array.size_replicate], ?_⟩
  intro j hj
  rw [Array.size_set!, Array.size_replicate] at hj
  by_cases hjr : j = root
  · subst hjr
    rw [set!_getElem!_self _ j 1.0 (by rw [Array.size_replicate]; exact hj),
        set!_getElem!_self _ j (1:ℝ) (by rw [Array.size_replicate]; exact hj), toReal_oneLit]
    constructor <;> norm_num
  · rw [set!_getElem!_ne_gen _ root 1.0 j (Ne.symm hjr),
        set!_getElem!_ne_gen _ root (1:ℝ) j (Ne.symm hjr),
        getElem!_pos _ j (by rw [Array.size_replicate]; exact hj), Array.getElem_replicate,
        getElem!_pos _ j (by rw [Array.size_replicate]; exact hj), Array.getElem_replicate, toReal_zeroLit]
    constructor <;> norm_num

/-- **The actual Float reverse sweep vs the exact-ℝ sweep.** Every entry of `gradsF t root` (the
    literal Float engine output) is within `sweepBnd` of the corresponding entry of the exact-ℝ sweep
    `bsweepR` — closing the Float↔ℝ (Layer-2) gap for the reverse-mode sweep. -/
theorem gradsF_error (t : Tape) (root : V) (W : ℝ) (hW : 0 ≤ W) (hroot : root < t.val.size)
    (hedges : ∀ idx, idx < t.val.size → ∀ e ∈ t.deps[idx]!, e.1 < t.val.size ∧ |toReal e.2| ≤ W) :
    AdjBnd (gradsF t root)
      (bsweepR t ((Array.replicate t.val.size (0:ℝ)).set! root 1) 0 t.val.size)
      (sweepBnd t W 0 t.val.size (1, 0)).1 (sweepBnd t W 0 t.val.size (1, 0)).2 := by
  rw [gradsF_eq_bsweep]
  exact bsweep_error t W t.val.size hW 0 t.val.size _ _ 1 0
    (seed_AdjBnd t.val.size root hroot) (by rw [Array.size_set!, Array.size_replicate]) (le_refl _)
    (fun idx _ hidx => hedges idx hidx) (by norm_num) (le_refl 0)

end Puffer.FloatR.ADReverseError
