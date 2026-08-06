/-
# Finiteness of the ACTUAL reverse-mode AD tape: the `grads` engine's adjoints are overflow-free

C57 (`BackwardFinite`)/C61/C64/C67 certified backward+optimizer finiteness for a SELF-CONTAINED
linear-layer backward (`gradW`), each disclosing the repo's actual reverse-mode engine — the
Wengert tape `Puffer.FloatR.AD.Tape` with its imperative reverse sweep `grads`
(`Puffer/Float/AutoDiff.lean`) — as the remaining AD structure. This module certifies THAT
engine, in two halves:

* **Forward (the actual `evalF`)**: `fwdBound`, a recursive magnitude budget over the ACTUAL AD
  grammar `Puffer.FloatR.ADR.Expr`, and `evalF_mag`/`evalF_isFinite`: on the LOG-FREE sub-grammar
  (`LogFree` — `log`'s magnitude needs a domain floor, see scope note) with inputs bounded by `B`,
  every forward value is magnitude-bounded and overflow-free. Composes the pre-existing `(1+δ)`
  op bounds: `add_bound`/`sub_bound`/`mul_bound` (C51/C54), `expF_mag_le` (C48 — the exp argument
  bounded above via the recursive bound), `toReal_reluF` (relu never increases magnitude), and
  the EXACT `min_mag_le`/`max_mag_le` (C51).

* **Backward (the ACTUAL sweep)**: the magnitude invariant over the real reverse sweep. The
  imperative `grads` equals the functional fold `gradsF` (`grads_eq_gradsF`, ADReverse Stage A),
  a fold of `stepNode` (per node: push `adjoint · edgeWeight` into each parent's adjoint slot).
  We prove the per-edge budget `edgeBound` (each edge write is one rounded add of one rounded
  multiply: `C ↦ (1+u64)·(C + (1+u64)·(C·D))`), the per-node invariant `stepNode_mag` (via
  `Array.foldl_induction` over the node's edge array), and the per-sweep invariant
  `foldl_stepNode_mag`/`sweepBound` — for an ARBITRARY tape with in-bounds edges (`hwf`), edge
  weights `≤ D`, and `≤ E` edges per node — seeded at the engine's actual initial adjoint array
  (`0.0`s plus a `1.0` at the root: entries `≤ 1` by `toReal_zeroLit`/`toReal_oneLit`).
  Capstones: `gradsF_entry_mag`, `gradsF_entry_isFinite`, and **`adGrad_isFinite`** — every
  adjoint (hence every gradient read at a leaf) that the ACTUAL imperative engine `grads`
  produces is overflow-free, given the single budget check `sweepBound E D n 1 ≤ overflowBound`.

**Scope (honestly disclosed).** The backward certificate is about the ACTUAL tape engine
(`grads`, via `grads_eq_gradsF`) — unlike C57's self-contained `gradW` — and holds for ANY tape
(comp-built from an `Expr`, or hand-built as the trainer's `AutoDiff.lean` programs are), under
three checkable tape hypotheses: every edge targets an in-bounds node (`hwf` — true for
`comp`-built tapes by `comp_edges_range`, and for any well-formed Wengert program), every edge
weight is `≤ D`, and every node has `≤ E` edges (`E = 2` for all tapes the op combinators build).
The edge-weight budget `D` is the honest interface: for `comp`-built log-free tapes the weights
are `0.0`/`1.0`/`-1.0`/`scale` coefficients/sibling forward values/`exp` values — all bounded by
the FORWARD budget (`evalF_mag` + `toReal_neg`, since `comp_root_val` identifies tape values with
`evalF`); discharging `D` from `fwdBound` per-tape is the mechanical remaining glue (it needs the
per-node value-readout of `comp`, a `comp_root_val`-style traversal). `log` is excluded from the
forward budget (its value `log x → −∞` as `x → 0⁺` and its edge weight `1/x` need a partition-
floor side-condition, as in C51's `logPart_isFinite` — the same honest domain condition). The
sweep budget `sweepBound E D n 1` GROWS with the tape size `n` (each node's step inflates by the
`(1+u64)` factors — realistic: adjoints genuinely accumulate); `≤ overflowBound` is the checkable
condition. NO new axiom: C43's `isFinite_of_bounded` plus the pre-existing `(1+δ)` model base.
This slots into C57/C64/C67's training-step/whole-run structure, which accepts any
bounded-gradient source.
-/
import Puffer.Float.ADReverse
import Puffer.RL.FiniteBound
import Puffer.RL.LossFinite
import Puffer.RL.LossForwardFinite
import Puffer.RL.PPOLossScalar

namespace Puffer.RL.ADTapeFinite

open Puffer.FloatR (toReal u64 u64_pos expEps toReal_zeroLit toReal_oneLit toReal_reluF reluF)
open Puffer.FloatR.AD (Tape V grads)
open Puffer.FloatR.ADR (Expr evalF)
open Puffer.FloatR.ADReverse (gradsF stepNode grads_eq_gradsF set!_getElem!_in)
open Puffer.RL.FiniteBound (isFinite_of_bounded overflowBound)
open Puffer.RL.LossFinite (expF_mag_le)
open Puffer.RL.LossForwardFinite (mul_bound sub_bound min_mag_le max_mag_le)
open Puffer.RL.PPOLossScalar (add_bound)

/-! ### Forward half: magnitude over the ACTUAL `evalF`, log-free sub-grammar -/

/-- The log-free sub-grammar of the AD IR: every op except `log` (whose magnitude needs a
    domain floor — the honest side-condition, cf. C51's `logPart_isFinite`). -/
def LogFree : Expr → Prop
  | .var _ => True
  | .const _ => True
  | .add a b => LogFree a ∧ LogFree b
  | .sub a b => LogFree a ∧ LogFree b
  | .mul a b => LogFree a ∧ LogFree b
  | .scale _ a => LogFree a
  | .exp a => LogFree a
  | .log _ => False
  | .relu a => LogFree a
  | .max a b => LogFree a ∧ LogFree b
  | .min a b => LogFree a ∧ LogFree b

/-- Recursive magnitude budget over the ACTUAL AD grammar: one `(1+u64)` per rounded op,
    `exp(·)·(1+expEps)` for the exp node, exact (no inflation) for `relu`/`max`/`min` (exact
    embeddings), inputs budgeted `B`. (`log` gets a junk `0`; the theorems require `LogFree`.) -/
noncomputable def fwdBound (B : ℝ) : Expr → ℝ
  | .var _ => B
  | .const c => |toReal c|
  | .add a b => (1 + u64) * (fwdBound B a + fwdBound B b)
  | .sub a b => (1 + u64) * (fwdBound B a + fwdBound B b)
  | .mul a b => (1 + u64) * (fwdBound B a * fwdBound B b)
  | .scale c a => (1 + u64) * (|toReal c| * fwdBound B a)
  | .exp a => Real.exp (fwdBound B a) * (1 + expEps)
  | .log _ => 0
  | .relu a => fwdBound B a
  | .max a b => max (fwdBound B a) (fwdBound B b)
  | .min a b => max (fwdBound B a) (fwdBound B b)

/-- **Forward magnitude over the actual `evalF`.** On the log-free sub-grammar, with every input
    `|toReal (σ i)| ≤ B`, the Float forward value is bounded by the recursive budget. -/
theorem evalF_mag (σ : Nat → Float) (B : ℝ) (hσ : ∀ i, |toReal (σ i)| ≤ B) :
    ∀ (e : Expr), LogFree e → |toReal (evalF e σ)| ≤ fwdBound B e := by
  intro e
  induction e with
  | var i => intro _; simpa only [evalF, fwdBound] using hσ i
  | const c => intro _; simp only [evalF, fwdBound]; exact le_refl _
  | add a b iha ihb =>
      intro hlf
      simp only [evalF, fwdBound]
      exact add_bound _ _ _ _ (iha hlf.1) (ihb hlf.2)
  | sub a b iha ihb =>
      intro hlf
      simp only [evalF, fwdBound]
      exact sub_bound _ _ _ _ (iha hlf.1) (ihb hlf.2)
  | mul a b iha ihb =>
      intro hlf
      simp only [evalF, fwdBound]
      exact mul_bound _ _ _ _ (iha hlf.1) (ihb hlf.2)
  | scale c a iha =>
      intro hlf
      simp only [evalF, fwdBound]
      exact mul_bound _ _ _ _ (le_refl |toReal c|) (iha hlf)
  | exp a iha =>
      intro hlf
      simp only [evalF, fwdBound]
      exact expF_mag_le _ _ ((le_abs_self _).trans (iha hlf))
  | log a _ => intro hlf; exact hlf.elim
  | relu a iha =>
      intro hlf
      simp only [evalF, fwdBound]
      rw [toReal_reluF]
      have h : |max (toReal (evalF a σ)) 0| ≤ |toReal (evalF a σ)| := by
        rcases le_total (toReal (evalF a σ)) 0 with h | h
        · rw [max_eq_right h, abs_zero]; exact abs_nonneg _
        · rw [max_eq_left h]
      exact h.trans (iha hlf)
  | max a b iha ihb =>
      intro hlf
      simp only [evalF, fwdBound]
      exact (max_mag_le _ _).trans (max_le_max (iha hlf.1) (ihb hlf.2))
  | min a b iha ihb =>
      intro hlf
      simp only [evalF, fwdBound]
      exact (min_mag_le _ _).trans (max_le_max (iha hlf.1) (ihb hlf.2))

/-- **Forward finiteness over the actual `evalF`**: bounded inputs + the budget under
    `overflowBound` ⟹ the Float forward value is overflow-free. -/
theorem evalF_isFinite (σ : Nat → Float) (B : ℝ) (hσ : ∀ i, |toReal (σ i)| ≤ B)
    (e : Expr) (hlf : LogFree e) (hbound : fwdBound B e ≤ overflowBound) :
    (evalF e σ).isFinite = true :=
  isFinite_of_bounded _ ((evalF_mag σ B hσ e hlf).trans hbound)

/-! ### Backward half: magnitude invariant over the ACTUAL reverse sweep -/

/-- Per-edge budget: each edge write inside a node replaces one adjoint slot by
    `slot + adjoint·edgeWeight` — one rounded multiply (`(1+u64)·(A·D)`) inside one rounded add.
    `edgeBound A D C k` bounds every slot after `k` edge writes from a uniform start `C`, with
    the node's (fixed, pre-read) adjoint `≤ A` and edge weights `≤ D`. -/
noncomputable def edgeBound (A D C : ℝ) : Nat → ℝ
  | 0 => C
  | k + 1 => (1 + u64) * (edgeBound A D C k + (1 + u64) * (A * D))

@[simp] theorem edgeBound_zero (A D C : ℝ) : edgeBound A D C 0 = C := rfl

theorem edgeBound_succ (A D C : ℝ) (k : Nat) :
    edgeBound A D C (k + 1) = (1 + u64) * (edgeBound A D C k + (1 + u64) * (A * D)) := rfl

theorem edgeBound_nonneg (A D C : ℝ) (hA : 0 ≤ A) (hD : 0 ≤ D) (hC : 0 ≤ C) :
    ∀ k, 0 ≤ edgeBound A D C k
  | 0 => hC
  | k + 1 => by
      have h := edgeBound_nonneg A D C hA hD hC k
      have hu := u64_pos
      have h1 : (0 : ℝ) ≤ 1 + u64 := by linarith
      have h2 : (0 : ℝ) ≤ (1 + u64) * (A * D) := mul_nonneg h1 (mul_nonneg hA hD)
      rw [edgeBound_succ]
      exact mul_nonneg h1 (by linarith)

theorem edgeBound_le_succ (A D C : ℝ) (hA : 0 ≤ A) (hD : 0 ≤ D) (hC : 0 ≤ C) (k : Nat) :
    edgeBound A D C k ≤ edgeBound A D C (k + 1) := by
  have h := edgeBound_nonneg A D C hA hD hC k
  have hu := u64_pos
  have h1 : (1 : ℝ) ≤ 1 + u64 := by linarith
  have h2 : (0 : ℝ) ≤ (1 + u64) * (A * D) :=
    mul_nonneg (by linarith) (mul_nonneg hA hD)
  rw [edgeBound_succ]
  calc edgeBound A D C k
      ≤ edgeBound A D C k + (1 + u64) * (A * D) := by linarith
    _ = 1 * (edgeBound A D C k + (1 + u64) * (A * D)) := by ring
    _ ≤ (1 + u64) * (edgeBound A D C k + (1 + u64) * (A * D)) :=
        mul_le_mul_of_nonneg_right h1 (by linarith)

theorem edgeBound_mono (A D C : ℝ) (hA : 0 ≤ A) (hD : 0 ≤ D) (hC : 0 ≤ C)
    {k k' : Nat} (h : k ≤ k') : edgeBound A D C k ≤ edgeBound A D C k' := by
  induction h with
  | refl => exact le_refl _
  | step _ ih => exact ih.trans (edgeBound_le_succ A D C hA hD hC _)

/-- Per-node budget: a node with `≤ E` edges, all weights `≤ D`, processed from a uniform slot
    bound `C` (which also bounds the node's own adjoint), leaves every slot `≤ nodeBound E D C`. -/
noncomputable def nodeBound (E : Nat) (D C : ℝ) : ℝ := edgeBound C D C E

theorem le_nodeBound (E : Nat) (D C : ℝ) (hD : 0 ≤ D) (hC : 0 ≤ C) : C ≤ nodeBound E D C :=
  edgeBound_mono C D C hC hD hC (Nat.zero_le E)

/-- **Per-node magnitude invariant over the ACTUAL `stepNode`.** If all adjoint slots are `≤ C`,
    the node's edges are in bounds with weights `≤ D` and count `≤ E`, then after `stepNode`
    (the real engine's per-node reverse step) the size is unchanged and every slot is
    `≤ nodeBound E D C`. Proved by `Array.foldl_induction` over the node's edge array. -/
theorem stepNode_mag (t : Tape) (adj : Array Float) (idx : Nat) (E : Nat) (D C : ℝ)
    (hC0 : 0 ≤ C) (hD0 : 0 ≤ D) (hidx : idx < adj.size)
    (hwf : ∀ e ∈ t.deps[idx]!, e.1 < adj.size)
    (hDw : ∀ e ∈ t.deps[idx]!, |toReal e.2| ≤ D)
    (hE : (t.deps[idx]!).size ≤ E)
    (hadj : ∀ j, j < adj.size → |toReal adj[j]!| ≤ C) :
    (stepNode t adj idx).size = adj.size ∧
      ∀ j, j < adj.size → |toReal (stepNode t adj idx)[j]!| ≤ nodeBound E D C := by
  have ha : |toReal adj[idx]!| ≤ C := hadj idx hidx
  have key : ((t.deps[idx]!).foldl (fun acc e => acc.set! e.1 (acc[e.1]! + adj[idx]! * e.2)) adj).size
        = adj.size ∧
      ∀ j, j < adj.size →
        |toReal ((t.deps[idx]!).foldl
          (fun acc e => acc.set! e.1 (acc[e.1]! + adj[idx]! * e.2)) adj)[j]!|
          ≤ edgeBound C D C (t.deps[idx]!).size := by
    refine Array.foldl_induction
      (motive := fun k (acc : Array Float) =>
        acc.size = adj.size ∧ ∀ j, j < adj.size → |toReal acc[j]!| ≤ edgeBound C D C k)
      ⟨rfl, fun j hj => hadj j hj⟩ ?_
    intro i acc hacc
    obtain ⟨hsz, hbd⟩ := hacc
    have hmem : (t.deps[idx]!)[i] ∈ t.deps[idx]! := Array.mem_of_getElem rfl
    have he1 : ((t.deps[idx]!)[i]).1 < adj.size := hwf _ hmem
    have he2 : |toReal ((t.deps[idx]!)[i]).2| ≤ D := hDw _ hmem
    refine ⟨by rw [Array.size_set!]; exact hsz, ?_⟩
    intro j hj
    have hj' : j < acc.size := by rw [hsz]; exact hj
    rw [set!_getElem!_in acc _ _ j hj']
    by_cases hcase : ((t.deps[idx]!)[i]).1 = j
    · rw [if_pos hcase]
      have h1 : |toReal acc[((t.deps[idx]!)[i]).1]!| ≤ edgeBound C D C i := hbd _ he1
      have h2 : |toReal (adj[idx]! * ((t.deps[idx]!)[i]).2)| ≤ (1 + u64) * (C * D) :=
        mul_bound _ _ C D ha he2
      rw [edgeBound_succ]
      exact add_bound _ _ _ _ h1 h2
    · rw [if_neg hcase]
      exact (hbd j hj).trans (edgeBound_le_succ C D C hC0 hD0 hC0 _)
  have hrw : stepNode t adj idx
      = (t.deps[idx]!).foldl (fun acc e => acc.set! e.1 (acc[e.1]! + adj[idx]! * e.2)) adj := rfl
  rw [hrw]
  exact ⟨key.1, fun j hj => (key.2 j hj).trans (edgeBound_mono C D C hC0 hD0 hC0 hE)⟩

/-- Per-sweep budget: iterate the per-node budget over `m` processed nodes (inside-first,
    matching the sweep's `foldl` direction). -/
noncomputable def sweepBound (E : Nat) (D : ℝ) : Nat → ℝ → ℝ
  | 0, C => C
  | m + 1, C => sweepBound E D m (nodeBound E D C)

theorem sweepBound_succ (E : Nat) (D : ℝ) (m : Nat) (C : ℝ) :
    sweepBound E D (m + 1) C = sweepBound E D m (nodeBound E D C) := rfl

/-- **Per-sweep magnitude invariant over the ACTUAL sweep's fold.** Folding `stepNode` over any
    list of in-bounds node indices, from adjoints uniformly `≤ C`, keeps every slot bounded by
    the iterated budget (and never resizes). -/
theorem foldl_stepNode_mag (t : Tape) (n E : Nat) (D : ℝ) (hD0 : 0 ≤ D)
    (hwf : ∀ idx, idx < n → ∀ e ∈ t.deps[idx]!, e.1 < n)
    (hDw : ∀ idx, idx < n → ∀ e ∈ t.deps[idx]!, |toReal e.2| ≤ D)
    (hE : ∀ idx, idx < n → (t.deps[idx]!).size ≤ E) :
    ∀ (l : List Nat) (adj : Array Float) (C : ℝ), 0 ≤ C → adj.size = n →
      (∀ j, j < n → |toReal adj[j]!| ≤ C) → (∀ i ∈ l, i < n) →
      (List.foldl (fun acc i => stepNode t acc i) adj l).size = n ∧
        ∀ j, j < n →
          |toReal (List.foldl (fun acc i => stepNode t acc i) adj l)[j]!|
            ≤ sweepBound E D l.length C := by
  intro l
  induction l with
  | nil =>
      intro adj C _ hsz hadj _
      simp only [List.foldl_nil, List.length_nil]
      exact ⟨hsz, hadj⟩
  | cons i rest ih =>
      intro adj C hC hsz hadj hmem
      have hi : i < n := hmem i (List.mem_cons.mpr (Or.inl rfl))
      have hstep := stepNode_mag t adj i E D C hC hD0
        (by rw [hsz]; exact hi)
        (fun e he => by rw [hsz]; exact hwf i hi e he)
        (hDw i hi) (hE i hi)
        (fun j hj => hadj j (by rw [← hsz]; exact hj))
      have hsz' : (stepNode t adj i).size = n := hstep.1.trans hsz
      have hadj' : ∀ j, j < n → |toReal (stepNode t adj i)[j]!| ≤ nodeBound E D C :=
        fun j hj => hstep.2 j (by rw [hsz]; exact hj)
      have hC' : 0 ≤ nodeBound E D C := hC.trans (le_nodeBound E D C hD0 hC)
      have hmem' : ∀ i' ∈ rest, i' < n := fun i' hi' => hmem i' (List.mem_cons.mpr (Or.inr hi'))
      have h := ih (stepNode t adj i) (nodeBound E D C) hC' hsz' hadj' hmem'
      simp only [List.foldl_cons, List.length_cons]
      rw [sweepBound_succ]
      exact h

/-! ### The engine's actual initial adjoint array: entries `≤ 1` -/

theorem adj0_size (n root : Nat) :
    ((Array.replicate n (0.0 : Float)).set! root 1.0).size = n := by
  rw [Array.size_set!, Array.size_replicate]

theorem adj0_mag (n root : Nat) :
    ∀ j, j < n → |toReal (((Array.replicate n (0.0 : Float)).set! root 1.0))[j]!| ≤ 1 := by
  intro j hj
  have hj' : j < (Array.replicate n (0.0 : Float)).size := by
    rw [Array.size_replicate]; exact hj
  rw [set!_getElem!_in _ root _ j hj']
  by_cases h : root = j
  · rw [if_pos h, toReal_oneLit]; norm_num
  · rw [if_neg h]
    have hz : (Array.replicate n (0.0 : Float))[j]! = 0.0 := by
      rw [getElem!_pos _ j hj']
      simp
    rw [hz, toReal_zeroLit]; norm_num

/-! ### Capstones: the ACTUAL engine's adjoints are bounded and overflow-free -/

/-- `gradsF`'s reverse-order fold, rewritten as a plain `stepNode` fold over the mapped
    (reversed) index list — the shape the sweep invariant consumes. -/
theorem gradsF_eq_mapped_fold (t : Tape) (root : V) :
    gradsF t root = List.foldl (fun acc i => stepNode t acc i)
      ((Array.replicate t.val.size 0.0).set! root 1.0)
      ((List.range t.val.size).map (fun i => t.val.size - 1 - i)) := by
  unfold gradsF
  exact (List.foldl_map (f := fun i => t.val.size - 1 - i)
    (g := fun (acc : Array Float) i => stepNode t acc i)
    (l := List.range t.val.size)).symm

/-- **The actual functional sweep's adjoints are magnitude-bounded**: for ANY tape whose edges
    are in bounds, with edge weights `≤ D` and `≤ E` edges per node, every adjoint after the
    full sweep is `≤ sweepBound E D n 1` (`n` = tape size; the seed array is `≤ 1`). -/
theorem gradsF_entry_mag (t : Tape) (root : V) (E : Nat) (D : ℝ) (hD0 : 0 ≤ D)
    (hwf : ∀ idx, idx < t.val.size → ∀ e ∈ t.deps[idx]!, e.1 < t.val.size)
    (hDw : ∀ idx, idx < t.val.size → ∀ e ∈ t.deps[idx]!, |toReal e.2| ≤ D)
    (hE : ∀ idx, idx < t.val.size → (t.deps[idx]!).size ≤ E) :
    ∀ j, j < t.val.size →
      |toReal ((gradsF t root)[j]!)| ≤ sweepBound E D t.val.size 1 := by
  intro j hj
  rw [gradsF_eq_mapped_fold]
  have hmem : ∀ i ∈ (List.range t.val.size).map (fun i => t.val.size - 1 - i),
      i < t.val.size := by
    intro i' hi'
    obtain ⟨i, hi, rfl⟩ := List.mem_map.mp hi'
    rw [List.mem_range] at hi
    omega
  have h := foldl_stepNode_mag t t.val.size E D hD0 hwf hDw hE
    ((List.range t.val.size).map (fun i => t.val.size - 1 - i))
    ((Array.replicate t.val.size 0.0).set! root 1.0) 1 (by norm_num)
    (adj0_size _ _) (adj0_mag _ _) hmem
  have hlen : ((List.range t.val.size).map (fun i => t.val.size - 1 - i)).length
      = t.val.size := by rw [List.length_map, List.length_range]
  rw [hlen] at h
  exact h.2 j hj

/-- **The functional sweep's adjoints are overflow-free** under the single budget check. -/
theorem gradsF_entry_isFinite (t : Tape) (root : V) (E : Nat) (D : ℝ) (hD0 : 0 ≤ D)
    (hwf : ∀ idx, idx < t.val.size → ∀ e ∈ t.deps[idx]!, e.1 < t.val.size)
    (hDw : ∀ idx, idx < t.val.size → ∀ e ∈ t.deps[idx]!, |toReal e.2| ≤ D)
    (hE : ∀ idx, idx < t.val.size → (t.deps[idx]!).size ≤ E)
    (hbound : sweepBound E D t.val.size 1 ≤ overflowBound) :
    ∀ j, j < t.val.size → ((gradsF t root)[j]!).isFinite = true :=
  fun j hj => isFinite_of_bounded _
    ((gradsF_entry_mag t root E D hD0 hwf hDw hE j hj).trans hbound)

/-- **CAPSTONE: the ACTUAL imperative reverse-mode engine's gradient is overflow-free.** Every
    adjoint the trainer's real `grads` sweep (`Puffer/Float/AutoDiff.lean`, via
    `grads_eq_gradsF`) produces — in particular every gradient read at a leaf — is finite, for
    any tape with in-bounds edges, edge weights `≤ D`, and `≤ E` edges per node, given the one
    checkable budget `sweepBound E D n 1 ≤ overflowBound`. -/
theorem adGrad_isFinite (t : Tape) (root : V) (E : Nat) (D : ℝ) (hD0 : 0 ≤ D)
    (hwf : ∀ idx, idx < t.val.size → ∀ e ∈ t.deps[idx]!, e.1 < t.val.size)
    (hDw : ∀ idx, idx < t.val.size → ∀ e ∈ t.deps[idx]!, |toReal e.2| ≤ D)
    (hE : ∀ idx, idx < t.val.size → (t.deps[idx]!).size ≤ E)
    (hbound : sweepBound E D t.val.size 1 ≤ overflowBound) :
    ∀ j, j < t.val.size → ((grads t root)[j]!).isFinite = true := by
  intro j hj
  rw [grads_eq_gradsF]
  exact gradsF_entry_isFinite t root E D hD0 hwf hDw hE hbound j hj

end Puffer.RL.ADTapeFinite
