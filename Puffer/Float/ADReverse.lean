/-
Reverse-mode tape ↔ forward-mode AD equivalence (attacking PLAN.md's "reverse-mode-tape
equivalence" — the last AD gap).

The trainer's reverse-mode engine (`Puffer/Float/AutoDiff.lean`) is an imperative `Id.run`
double-loop over a mutable adjoint array, so it "cannot be inducted on directly" (the reason the
verified forward-mode AD in `AutoDiffR.lean` was built on an explicit inductive `Expr` instead).
Here we bridge the two:

  * **Stage A — `grads_eq_gradsF`**: the imperative reverse sweep `grads` EQUALS a pure functional
    fold `gradsF` (a `List.foldl` of a per-node edge-fold). Pure structural equality (no axioms),
    via the `forIn`→`foldl` reductions. This makes the actual engine amenable to equational reasoning.

  * **Stage B — the compiler `comp`**: a pure state-passing `Expr → Tape → V × Tape` that builds a
    tape with exactly the `AutoDiff.lean` op conventions (same primal values, same local-derivative
    edges). We prove it PRIMAL-correct (`comp_root_val`: the compiled root's tape value = `evalF e σ`
    — so the tape computes the same function as the `Expr`), plus the structural facts the reverse
    sweep will need: monotone growth (`comp_size_le`), root-in-bounds (`comp_root_lt`), and prefix
    preservation (`comp_preserve`).

Stage C — the GRADIENT equivalence — is proven at the ℝ-STRUCTURE layer. The reverse-sweep adjoint
invariant (`central`) holds over the topologically-ordered `comp`-built tape (every edge points to a
strictly-earlier node — `comp_edges_range` — so each node's adjoint is written exactly once by its
unique parent). Its payoff, `bsweepR_reads_revE_RF`, shows the flat sweep read at the variable leaves
equals the faithful reverse pass `revE_RF`; and `bsweepR_reads_derivR` reaches the true `derivR`. TWO
CAVEATS, disclosed in the theorem names/docstrings (Float is not a ring, `(1.0:Float) ≠ 1`): (i) the
endpoints are over the exact-ℝ sweep `bsweepR` — the ℝ image of `gradsF`'s fold, connected to the Float
engine only by `gradsF_eq_bsweep` (Float `gradsF = bsweep`); the sweep's own float rounding,
`toReal (gradsF …) ≈ bsweepR …`, is the deferred Layer-2 εg bound. (ii) `bsweepR_reads_derivR` is
CONDITIONAL on the primal bridge `revE_RF = revE_R` (rounded vs ideal weights), generally false under
rounding — an explicit hypothesis, the same Layer-2 obligation. So: the reverse ALGORITHM is proven
correct on the flat tape; quantifying its rounding is future work. All results here are pure/structural
(standard logic axioms + the sanctioned `toReal`; no other float-model axioms, no `sorry`).
-/
import Mathlib
import Puffer.Float.AutoDiff
import Puffer.Float.AutoDiffR

namespace Puffer.FloatR.ADReverse

open Puffer.FloatR.AD (Tape V grads)
open Puffer.FloatR.ADR (Expr evalF evalR derivR envR occurs)
open Puffer.FloatR (toReal)

/-! ### Array `push` index helpers -/

/-- The pushed element sits at the old size. -/
theorem push_getElem!_size {α} [Inhabited α] (a : Array α) (v : α) : (a.push v)[a.size]! = v := by
  have h1 : a.size < (a.push v).size := by rw [Array.size_push]; omega
  rw [getElem!_pos (a.push v) a.size h1, Array.getElem_push, dif_neg (Nat.lt_irrefl _)]

/-- Pushing preserves the existing prefix. -/
theorem push_getElem!_lt {α} [Inhabited α] (a : Array α) (v : α) (j : Nat) (h : j < a.size) :
    (a.push v)[j]! = a[j]! := by
  have h1 : j < (a.push v).size := by rw [Array.size_push]; omega
  rw [getElem!_pos (a.push v) j h1, getElem!_pos a j h, Array.getElem_push, dif_pos h]

/-! ### Stage A: the imperative reverse sweep equals a pure functional fold -/

/-- One node's reverse step: read its adjoint `a`, then push `a·edgeDeriv` to each parent. -/
def stepNode (t : Tape) (adj : Array Float) (idx : Nat) : Array Float :=
  let a := adj[idx]!
  (t.deps[idx]!).foldl (fun adj e => adj.set! e.1 (adj[e.1]! + a * e.2)) adj

/-- Pure functional reverse sweep: seed the root adjoint to 1, then fold `stepNode` over the nodes
    in reverse order (`n-1 … 0`). -/
def gradsF (t : Tape) (root : V) : Array Float :=
  let n := t.val.size
  let adj0 := (Array.replicate n 0.0).set! root 1.0
  (List.range n).foldl (fun adj i => stepNode t adj (n - 1 - i)) adj0

/-- **The imperative reverse sweep = the functional fold.** Pure structural equality (only the
    standard logic axioms) — reduces the nested `Id.run`/`forIn` loops of `grads` to `gradsF`. -/
theorem grads_eq_gradsF (t : Tape) (root : V) : grads t root = gradsF t root := by
  unfold grads gradsF stepNode
  simp [Id.run, List.range_eq_range', Array.forIn_pure_yield_eq_foldl]; rfl

/-! ### Stage B: the pure compiler `Expr → Tape`, primal-correct and topologically well-formed -/

/-- Append a node with value `v` and local-derivative edges `ds`. -/
def pushT (t : Tape) (v : Float) (ds : Array (Nat × Float)) : Tape :=
  Tape.mk (t.val.push v) (t.deps.push ds)

/-- Compile an `Expr` into the tape, mirroring `AutoDiff.lean`'s op conventions (same primal, same
    local-derivative edges). Returns the root handle and the grown tape. Plain `let`s (single
    evaluation of each subtree, `.1`/`.2` projections) keep it zeta-reducible for proofs. -/
def comp (σ : Nat → Float) : Expr → Tape → V × Tape
  | .var i, t => (t.val.size, pushT t (σ i) #[])
  | .const c, t => (t.val.size, pushT t c #[])
  | .add a b, t =>
      let ra := comp σ a t; let rb := comp σ b ra.2
      (rb.2.val.size, pushT rb.2 (rb.2.val[ra.1]! + rb.2.val[rb.1]!) #[(ra.1, 1.0), (rb.1, 1.0)])
  | .sub a b, t =>
      let ra := comp σ a t; let rb := comp σ b ra.2
      (rb.2.val.size, pushT rb.2 (rb.2.val[ra.1]! - rb.2.val[rb.1]!) #[(ra.1, 1.0), (rb.1, -1.0)])
  | .mul a b, t =>
      let ra := comp σ a t; let rb := comp σ b ra.2
      (rb.2.val.size, pushT rb.2 (rb.2.val[ra.1]! * rb.2.val[rb.1]!)
        #[(ra.1, rb.2.val[rb.1]!), (rb.1, rb.2.val[ra.1]!)])
  | .scale c a, t =>
      let ra := comp σ a t
      (ra.2.val.size, pushT ra.2 (c * ra.2.val[ra.1]!) #[(ra.1, c)])
  | .exp a, t =>
      let ra := comp σ a t; let e := Float.exp (ra.2.val[ra.1]!)
      (ra.2.val.size, pushT ra.2 e #[(ra.1, e)])
  | .log a, t =>
      let ra := comp σ a t
      (ra.2.val.size, pushT ra.2 (Float.log (ra.2.val[ra.1]!)) #[(ra.1, 1.0 / ra.2.val[ra.1]!)])
  | .relu a, t =>
      let ra := comp σ a t; let va := ra.2.val[ra.1]!
      (ra.2.val.size, pushT ra.2 (if va < 0.0 then 0.0 else va) #[(ra.1, if va < 0.0 then 0.0 else 1.0)])
  | .max a b, t =>
      let ra := comp σ a t; let rb := comp σ b ra.2
      let va := rb.2.val[ra.1]!; let vb := rb.2.val[rb.1]!
      (rb.2.val.size, pushT rb.2 (if va ≤ vb then vb else va)
        #[(ra.1, if va ≤ vb then 0.0 else 1.0), (rb.1, if va ≤ vb then 1.0 else 0.0)])
  | .min a b, t =>
      let ra := comp σ a t; let rb := comp σ b ra.2
      let va := rb.2.val[ra.1]!; let vb := rb.2.val[rb.1]!
      (rb.2.val.size, pushT rb.2 (if va ≤ vb then va else vb)
        #[(ra.1, if va ≤ vb then 1.0 else 0.0), (rb.1, if va ≤ vb then 0.0 else 1.0)])

/-- The compiled tape only grows (its size never shrinks). -/
theorem comp_size_le (σ : Nat → Float) (e : Expr) (t : Tape) : t.val.size ≤ (comp σ e t).2.val.size := by
  induction e generalizing t with
  | var i => simp [comp, pushT, Array.size_push]
  | const c => simp [comp, pushT, Array.size_push]
  | scale c a iha => simp only [comp, pushT, Array.size_push]; exact le_trans (iha t) (Nat.le_succ _)
  | exp a iha => simp only [comp, pushT, Array.size_push]; exact le_trans (iha t) (Nat.le_succ _)
  | log a iha => simp only [comp, pushT, Array.size_push]; exact le_trans (iha t) (Nat.le_succ _)
  | relu a iha => simp only [comp, pushT, Array.size_push]; exact le_trans (iha t) (Nat.le_succ _)
  | add a b iha ihb => simp only [comp, pushT, Array.size_push]
                       exact le_trans (le_trans (iha t) (ihb _)) (Nat.le_succ _)
  | sub a b iha ihb => simp only [comp, pushT, Array.size_push]
                       exact le_trans (le_trans (iha t) (ihb _)) (Nat.le_succ _)
  | mul a b iha ihb => simp only [comp, pushT, Array.size_push]
                       exact le_trans (le_trans (iha t) (ihb _)) (Nat.le_succ _)
  | max a b iha ihb => simp only [comp, pushT, Array.size_push]
                       exact le_trans (le_trans (iha t) (ihb _)) (Nat.le_succ _)
  | min a b iha ihb => simp only [comp, pushT, Array.size_push]
                       exact le_trans (le_trans (iha t) (ihb _)) (Nat.le_succ _)

/-- The compiled root handle is a valid index (in bounds). The root is always the "size just before
    this node's own final `push`", and that push grows the size by one — so it reduces to `X < X+1`. -/
theorem comp_root_lt (σ : Nat → Float) (e : Expr) (t : Tape) :
    (comp σ e t).1 < (comp σ e t).2.val.size := by
  cases e <;> simp only [comp, pushT, Array.size_push] <;> exact Nat.lt_succ_self _

/-- Compiling only appends nodes, so it preserves the values of all pre-existing nodes. -/
theorem comp_preserve (σ : Nat → Float) (e : Expr) :
    ∀ (t : Tape) (j : Nat), j < t.val.size → (comp σ e t).2.val[j]! = t.val[j]! := by
  induction e with
  | var i => intro t j hj; simpa only [comp, pushT] using push_getElem!_lt _ _ j hj
  | const c => intro t j hj; simpa only [comp, pushT] using push_getElem!_lt _ _ j hj
  | scale c a iha => intro t j hj; simp only [comp, pushT]
                     rw [push_getElem!_lt _ _ j (lt_of_lt_of_le hj (comp_size_le σ a t))]; exact iha t j hj
  | exp a iha => intro t j hj; simp only [comp, pushT]
                 rw [push_getElem!_lt _ _ j (lt_of_lt_of_le hj (comp_size_le σ a t))]; exact iha t j hj
  | log a iha => intro t j hj; simp only [comp, pushT]
                 rw [push_getElem!_lt _ _ j (lt_of_lt_of_le hj (comp_size_le σ a t))]; exact iha t j hj
  | relu a iha => intro t j hj; simp only [comp, pushT]
                  rw [push_getElem!_lt _ _ j (lt_of_lt_of_le hj (comp_size_le σ a t))]; exact iha t j hj
  | add a b iha ihb => intro t j hj; simp only [comp, pushT]
                       have h1 : j < (comp σ a t).2.val.size := lt_of_lt_of_le hj (comp_size_le σ a t)
                       have h2 : j < (comp σ b (comp σ a t).2).2.val.size := lt_of_lt_of_le h1 (comp_size_le σ b _)
                       rw [push_getElem!_lt _ _ j h2, ihb _ j h1, iha t j hj]
  | sub a b iha ihb => intro t j hj; simp only [comp, pushT]
                       have h1 : j < (comp σ a t).2.val.size := lt_of_lt_of_le hj (comp_size_le σ a t)
                       have h2 : j < (comp σ b (comp σ a t).2).2.val.size := lt_of_lt_of_le h1 (comp_size_le σ b _)
                       rw [push_getElem!_lt _ _ j h2, ihb _ j h1, iha t j hj]
  | mul a b iha ihb => intro t j hj; simp only [comp, pushT]
                       have h1 : j < (comp σ a t).2.val.size := lt_of_lt_of_le hj (comp_size_le σ a t)
                       have h2 : j < (comp σ b (comp σ a t).2).2.val.size := lt_of_lt_of_le h1 (comp_size_le σ b _)
                       rw [push_getElem!_lt _ _ j h2, ihb _ j h1, iha t j hj]
  | max a b iha ihb => intro t j hj; simp only [comp, pushT]
                       have h1 : j < (comp σ a t).2.val.size := lt_of_lt_of_le hj (comp_size_le σ a t)
                       have h2 : j < (comp σ b (comp σ a t).2).2.val.size := lt_of_lt_of_le h1 (comp_size_le σ b _)
                       rw [push_getElem!_lt _ _ j h2, ihb _ j h1, iha t j hj]
  | min a b iha ihb => intro t j hj; simp only [comp, pushT]
                       have h1 : j < (comp σ a t).2.val.size := lt_of_lt_of_le hj (comp_size_le σ a t)
                       have h2 : j < (comp σ b (comp σ a t).2).2.val.size := lt_of_lt_of_le h1 (comp_size_le σ b _)
                       rw [push_getElem!_lt _ _ j h2, ihb _ j h1, iha t j hj]

/-- **Primal correctness.** The compiled root's tape value equals the forward-mode value `evalF e σ`
    — i.e. `comp` builds a tape that computes the same function as the `Expr`. -/
theorem comp_root_val (σ : Nat → Float) (e : Expr) :
    ∀ (t : Tape), (comp σ e t).2.val[(comp σ e t).1]! = evalF e σ := by
  induction e with
  | var i => intro t; simpa only [comp, pushT, evalF] using push_getElem!_size _ _
  | const c => intro t; simpa only [comp, pushT, evalF] using push_getElem!_size _ _
  | scale c a iha => intro t; simp only [comp, pushT, evalF]; rw [push_getElem!_size, iha t]
  | exp a iha => intro t; simp only [comp, pushT, evalF]; rw [push_getElem!_size, iha t]
  | log a iha => intro t; simp only [comp, pushT, evalF]; rw [push_getElem!_size, iha t]
  | relu a iha => intro t; simp only [comp, pushT, evalF, reluF]; rw [push_getElem!_size, iha t]
  | add a b iha ihb =>
      intro t; simp only [comp, pushT, evalF]
      rw [push_getElem!_size, comp_preserve σ b (comp σ a t).2 (comp σ a t).1 (comp_root_lt σ a t),
        iha t, ihb (comp σ a t).2]
  | sub a b iha ihb =>
      intro t; simp only [comp, pushT, evalF]
      rw [push_getElem!_size, comp_preserve σ b (comp σ a t).2 (comp σ a t).1 (comp_root_lt σ a t),
        iha t, ihb (comp σ a t).2]
  | mul a b iha ihb =>
      intro t; simp only [comp, pushT, evalF]
      rw [push_getElem!_size, comp_preserve σ b (comp σ a t).2 (comp σ a t).1 (comp_root_lt σ a t),
        iha t, ihb (comp σ a t).2]
  | max a b iha ihb =>
      intro t; simp only [comp, pushT, evalF]
      rw [push_getElem!_size, comp_preserve σ b (comp σ a t).2 (comp σ a t).1 (comp_root_lt σ a t),
        iha t, ihb (comp σ a t).2]
      rfl
  | min a b iha ihb =>
      intro t; simp only [comp, pushT, evalF]
      rw [push_getElem!_size, comp_preserve σ b (comp σ a t).2 (comp σ a t).1 (comp_root_lt σ a t),
        iha t, ihb (comp σ a t).2]
      rfl

/-! ### Stage C (core): reverse-mode accumulation of the tape's edges computes the true derivative

The remaining flat-array reverse-sweep invariant is intricate, but its mathematical CONTENT — that
accumulating the tape's local-derivative edges IN REVERSE (adjoint pushed from output to inputs) yields
the same gradient as the forward-mode `derivR` — is captured by `revE_R`, a recursive reverse pass that
uses EXACTLY the edge multipliers the tape records (`add`↦1, `mul a b`↦(`b`,`a`), `exp`↦`eˣ`, `log`↦
`1/x`, `relu`/`max`/`min`↦the selected-branch indicator — cf. `comp`/`AutoDiff.lean`). We prove it
computes `derivR` over ℝ, so reverse-mode ≡ forward-mode ≡ the true derivative at the algorithm level.
(The remaining Stage-C step is purely structural: that the flat `gradsF` fold over a `comp`-built tape
realizes this recursive `revE_R` — the reverse-DFS-order bookkeeping.) -/

/-- Reverse pass over ℝ: given the adjoint `ā` flowing into `e`'s output, accumulate the gradient
    contribution to each variable, pushing `ā` through the tape's local-derivative edges. -/
noncomputable def revE_R : Expr → (Nat → ℝ) → ℝ → (Nat → ℝ)
  | .var i, _, ā => fun k => if k = i then ā else 0
  | .const _, _, _ => fun _ => 0
  | .add a b, σ, ā => fun k => revE_R a σ ā k + revE_R b σ ā k
  | .sub a b, σ, ā => fun k => revE_R a σ ā k + revE_R b σ (-ā) k
  | .mul a b, σ, ā => fun k => revE_R a σ (ā * evalR b σ) k + revE_R b σ (ā * evalR a σ) k
  | .scale c a, σ, ā => fun k => revE_R a σ (ā * toReal c) k
  | .exp a, σ, ā => fun k => revE_R a σ (ā * Real.exp (evalR a σ)) k
  | .log a, σ, ā => fun k => revE_R a σ (ā * (1 / evalR a σ)) k
  | .relu a, σ, ā => fun k => revE_R a σ (ā * (if 0 < evalR a σ then 1 else 0)) k
  | .max a b, σ, ā => fun k =>
      revE_R a σ (ā * (if evalR a σ ≤ evalR b σ then 0 else 1)) k
      + revE_R b σ (ā * (if evalR a σ ≤ evalR b σ then 1 else 0)) k
  | .min a b, σ, ā => fun k =>
      revE_R a σ (ā * (if evalR a σ ≤ evalR b σ then 1 else 0)) k
      + revE_R b σ (ā * (if evalR a σ ≤ evalR b σ then 0 else 1)) k

/-- **Reverse pass is linear in the adjoint, with coefficient `derivR`.** The key lemma: pushing an
    adjoint `ā` through the tape edges scales the forward-mode derivative by `ā`. -/
theorem revE_R_eq (e : Expr) (σ : Nat → ℝ) : ∀ (ā : ℝ) (k : Nat), revE_R e σ ā k = ā * derivR e σ k := by
  induction e with
  | var i => intro ā k; simp only [revE_R, derivR]
             rcases eq_or_ne k i with h | h
             · subst h; simp
             · rw [if_neg h, if_neg (Ne.symm h), mul_zero]
  | const c => intro ā k; simp [revE_R, derivR]
  | add a b iha ihb => intro ā k; simp only [revE_R, derivR, iha, ihb]; ring
  | sub a b iha ihb => intro ā k; simp only [revE_R, derivR, iha, ihb]; ring
  | mul a b iha ihb => intro ā k; simp only [revE_R, derivR, iha, ihb]; ring
  | scale c a iha => intro ā k; simp only [revE_R, derivR, iha]; ring
  | exp a iha => intro ā k; simp only [revE_R, derivR, iha]; ring
  | log a iha => intro ā k; simp only [revE_R, derivR, iha]; ring
  | relu a iha => intro ā k; simp only [revE_R, derivR, iha]
                  by_cases h : 0 < evalR a σ <;> simp [h]
  | max a b iha ihb => intro ā k; simp only [revE_R, derivR, iha, ihb]
                       by_cases h : evalR a σ ≤ evalR b σ <;> simp [h]
  | min a b iha ihb => intro ā k; simp only [revE_R, derivR, iha, ihb]
                       by_cases h : evalR a σ ≤ evalR b σ <;> simp [h]

/-- **Reverse-mode = forward-mode = the true derivative.** Seeding the output adjoint to `1`, the
    reverse pass over the tape's edge conventions yields exactly `derivR` (which `derivR_hasDerivAt`
    proves is the real derivative). So the tape's local-derivative bookkeeping is correct. -/
theorem revE_R_eq_derivR (e : Expr) (σ : Nat → ℝ) (k : Nat) : revE_R e σ 1 k = derivR e σ k := by
  rw [revE_R_eq]; ring

/-! ### Stage C: the FLAT-ARRAY reverse sweep realizes the reverse pass at the leaves (ℝ layer)

We prove that the flat reverse sweep over a `comp`-built tape realizes the reverse-mode gradient at the
variable leaves — but at the exact-ℝ STRUCTURE layer, not over the literal Float `gradsF`. The reason:
the executable Float sweep uses *rounded* `+`/`*`, so `toReal (gradsF …)` cannot equal an exact-ℝ
quantity on the nose (Float is not a ring — `(1.0:Float) ≠ 1`). So, exactly as the file splits `dF`
(Float) from `derivR` (ℝ), we work with `bsweepR`: the exact-ℝ image of `gradsF`'s *identical* fold
(`gradsF = bsweep`, its Float twin, by `gradsF_eq_bsweep` — the accumulator is the only difference). The
theorem `bsweepR_reads_revE_RF` shows this ℝ sweep, read at the leaves, equals the faithful reverse pass
`revE_RF` (edge weights = `toReal` of the *actual* Float tape weights); `bsweepR_reads_derivR` reaches
`derivR` via `revE_R_eq_derivR`, conditional on the primal bridge. The residual `toReal (gradsF …) ≈
bsweepR …` rounding bound — the actual Float↔ℝ gap of the sweep — is the deferred Layer-2 εg work.

The tape is a TREE (each `comp` subexpression fills a disjoint contiguous index range, each `var`
occurrence gets a fresh leaf), so each node's adjoint is written exactly once by its unique parent
(processed earlier in the high→low sweep). The whole proof is a single structural induction (`central`)
threading the incoming root adjoint through `comp`'s nested regions. -/

/-- ℝ-valued reverse step: the `stepNode` fold with exact ℝ arithmetic and `toReal`'d edge weights. -/
noncomputable def stepNodeR (t : Tape) (adj : Array ℝ) (idx : Nat) : Array ℝ :=
  let a := adj[idx]!
  (t.deps[idx]!).foldl (fun adj e => adj.set! e.1 (adj[e.1]! + a * toReal e.2)) adj

/-- Bounded reverse sweep on the index window `[lo, hi)`, high→low (as `gradsF` sweeps `[0, n)`). -/
noncomputable def bsweepR (t : Tape) (adj : Array ℝ) (lo hi : Nat) : Array ℝ :=
  (List.range (hi - lo)).foldl (fun adj i => stepNodeR t adj (hi - 1 - i)) adj

/-- The `Float` bounded sweep (same shape as `gradsF`, parameterized `lo/hi`). -/
def bsweep (t : Tape) (adj : Array Float) (lo hi : Nat) : Array Float :=
  (List.range (hi - lo)).foldl (fun adj i => stepNode t adj (hi - 1 - i)) adj

/-- `gradsF` is the `Float` `bsweep` over the whole `[0, n)` range (pure structural equality). -/
theorem gradsF_eq_bsweep (t : Tape) (root : V) :
    gradsF t root = bsweep t ((Array.replicate t.val.size 0.0).set! root 1.0) 0 t.val.size := by
  unfold gradsF bsweep; simp only [Nat.sub_zero]

/-- Faithful ℝ reverse pass whose edge weights are `toReal` of the ACTUAL Float tape weights (the
    tape stores `evalF b σ`, not the ideal `evalR b σ`). This is the exact structural target. -/
noncomputable def revE_RF : Expr → (Nat → Float) → ℝ → (Nat → ℝ)
  | .var i, _, ā => fun k => if k = i then ā else 0
  | .const _, _, _ => fun _ => 0
  | .add a b, σ, ā => fun k => revE_RF a σ (ā * toReal (1.0:Float)) k + revE_RF b σ (ā * toReal (1.0:Float)) k
  | .sub a b, σ, ā => fun k => revE_RF a σ (ā * toReal (1.0:Float)) k + revE_RF b σ (ā * toReal (-1.0:Float)) k
  | .mul a b, σ, ā => fun k => revE_RF a σ (ā * toReal (evalF b σ)) k + revE_RF b σ (ā * toReal (evalF a σ)) k
  | .scale c a, σ, ā => fun k => revE_RF a σ (ā * toReal c) k
  | .exp a, σ, ā => fun k => revE_RF a σ (ā * toReal (Float.exp (evalF a σ))) k
  | .log a, σ, ā => fun k => revE_RF a σ (ā * toReal (1.0 / evalF a σ)) k
  | .relu a, σ, ā => fun k => revE_RF a σ (ā * toReal (if evalF a σ < 0.0 then 0.0 else 1.0)) k
  | .max a b, σ, ā => fun k =>
      revE_RF a σ (ā * toReal (if evalF a σ ≤ evalF b σ then 0.0 else 1.0)) k
      + revE_RF b σ (ā * toReal (if evalF a σ ≤ evalF b σ then 1.0 else 0.0)) k
  | .min a b, σ, ā => fun k =>
      revE_RF a σ (ā * toReal (if evalF a σ ≤ evalF b σ then 1.0 else 0.0)) k
      + revE_RF b σ (ā * toReal (if evalF a σ ≤ evalF b σ then 0.0 else 1.0)) k

/-- **The faithful reverse pass is homogeneous (linear) in the seed adjoint.** Pushing a seed adjoint `ā` into
    `e`'s output and accumulating through the tape's *actual* Float edge weights scales every leaf contribution by
    `ā`: `revE_RF e σ ā k = ā · revE_RF e σ 1 k`. This is the structural core of reverse-mode AD — the reverse sweep
    computes a linear vector-Jacobian product in the seed — and, unlike the ideal-weight pass, is stated purely over
    `revE_RF`, whose Float edge weights have no closed-form derivative to factor through. Proved by structural
    induction over all 11 `Expr` nodes, instantiating each child's IH at its concrete Float edge weight. -/
theorem revE_RF_smul_seed (e : Expr) (σ : Nat → Float) (k : Nat) :
    ∀ (ā : ℝ), revE_RF e σ ā k = ā * revE_RF e σ 1 k := by
  induction e with
  | var i => intro ā; simp only [revE_RF]; by_cases h : k = i <;> simp [h]
  | const c => intro ā; simp only [revE_RF]; ring
  | add a b iha ihb =>
      intro ā; simp only [revE_RF]
      rw [iha (ā * toReal (1.0:Float)), ihb (ā * toReal (1.0:Float)),
          iha (1 * toReal (1.0:Float)), ihb (1 * toReal (1.0:Float))]; ring
  | sub a b iha ihb =>
      intro ā; simp only [revE_RF]
      rw [iha (ā * toReal (1.0:Float)), ihb (ā * toReal (-1.0:Float)),
          iha (1 * toReal (1.0:Float)), ihb (1 * toReal (-1.0:Float))]; ring
  | mul a b iha ihb =>
      intro ā; simp only [revE_RF]
      rw [iha (ā * toReal (evalF b σ)), ihb (ā * toReal (evalF a σ)),
          iha (1 * toReal (evalF b σ)), ihb (1 * toReal (evalF a σ))]; ring
  | scale c a iha =>
      intro ā; simp only [revE_RF]
      rw [iha (ā * toReal c), iha (1 * toReal c)]; ring
  | exp a iha =>
      intro ā; simp only [revE_RF]
      rw [iha (ā * toReal (Float.exp (evalF a σ))), iha (1 * toReal (Float.exp (evalF a σ)))]; ring
  | log a iha =>
      intro ā; simp only [revE_RF]
      rw [iha (ā * toReal (1.0 / evalF a σ)), iha (1 * toReal (1.0 / evalF a σ))]; ring
  | relu a iha =>
      intro ā; simp only [revE_RF]
      rw [iha (ā * toReal (if evalF a σ < 0.0 then 0.0 else 1.0)),
          iha (1 * toReal (if evalF a σ < 0.0 then 0.0 else 1.0))]; ring
  | max a b iha ihb =>
      intro ā; simp only [revE_RF]
      rw [iha (ā * toReal (if evalF a σ ≤ evalF b σ then 0.0 else 1.0)),
          ihb (ā * toReal (if evalF a σ ≤ evalF b σ then 1.0 else 0.0)),
          iha (1 * toReal (if evalF a σ ≤ evalF b σ then 0.0 else 1.0)),
          ihb (1 * toReal (if evalF a σ ≤ evalF b σ then 1.0 else 0.0))]; ring
  | min a b iha ihb =>
      intro ā; simp only [revE_RF]
      rw [iha (ā * toReal (if evalF a σ ≤ evalF b σ then 1.0 else 0.0)),
          ihb (ā * toReal (if evalF a σ ≤ evalF b σ then 0.0 else 1.0)),
          iha (1 * toReal (if evalF a σ ≤ evalF b σ then 1.0 else 0.0)),
          ihb (1 * toReal (if evalF a σ ≤ evalF b σ then 0.0 else 1.0))]; ring

/-- Per-variable leaf readout: re-walk `comp`'s layout; each `.var i` leaf sits at index `t.val.size`
    for the sub-tape `t` it is compiled into. Reads `adjR` there, gated by `i = k`, summed. -/
noncomputable def leafAdjSumR (σ : Nat → Float) (adjR : Array ℝ) : Expr → Tape → Nat → ℝ
  | .var i, t, k => if i = k then adjR[t.val.size]! else 0
  | .const _, _, _ => 0
  | .add a b, t, k => leafAdjSumR σ adjR a t k + leafAdjSumR σ adjR b (comp σ a t).2 k
  | .sub a b, t, k => leafAdjSumR σ adjR a t k + leafAdjSumR σ adjR b (comp σ a t).2 k
  | .mul a b, t, k => leafAdjSumR σ adjR a t k + leafAdjSumR σ adjR b (comp σ a t).2 k
  | .scale _ a, t, k => leafAdjSumR σ adjR a t k
  | .exp a, t, k => leafAdjSumR σ adjR a t k
  | .log a, t, k => leafAdjSumR σ adjR a t k
  | .relu a, t, k => leafAdjSumR σ adjR a t k
  | .max a b, t, k => leafAdjSumR σ adjR a t k + leafAdjSumR σ adjR b (comp σ a t).2 k
  | .min a b, t, k => leafAdjSumR σ adjR a t k + leafAdjSumR σ adjR b (comp σ a t).2 k

/-! #### Array `set!` read lemmas -/

theorem set!_getElem!_in {α} [Inhabited α] (a : Array α) (i : Nat) (v : α) (j : Nat) (hj : j < a.size) :
    (a.set! i v)[j]! = if i = j then v else a[j]! := by
  have hj' : j < (a.set! i v).size := by rw [Array.size_set!]; exact hj
  rw [getElem!_pos (a.set! i v) j hj', getElem!_pos a j hj]; exact Array.getElem_setIfInBounds hj

theorem set!_getElem!_self {α} [Inhabited α] (a : Array α) (i : Nat) (v : α) (hi : i < a.size) :
    (a.set! i v)[i]! = v := by rw [set!_getElem!_in a i v i hi, if_pos rfl]

theorem set!_getElem!_ne_gen {α} [Inhabited α] (a : Array α) (i : Nat) (v : α) (j : Nat)
    (hne : i ≠ j) : (a.set! i v)[j]! = a[j]! := by
  by_cases hj : j < a.size
  · rw [set!_getElem!_in a i v j hj, if_neg hne]
  · have hj' : ¬ j < (a.set! i v).size := by rw [Array.size_set!]; exact hj
    rw [getElem!_neg (a.set! i v) j hj', getElem!_neg a j hj]

/-! #### Sweep algebra -/

theorem bsweepR_succ (t : Tape) (adj : Array ℝ) (lo hi : Nat) (h : lo ≤ hi) :
    bsweepR t adj lo (hi + 1) = bsweepR t (stepNodeR t adj hi) lo hi := by
  unfold bsweepR
  rw [show hi + 1 - lo = (hi - lo) + 1 by omega, List.range_succ_eq_map,
      List.foldl_cons, List.foldl_map]
  simp only [Nat.sub_zero, Nat.add_sub_cancel]
  apply List.foldl_ext; intro b i hi'; rw [List.mem_range] at hi'; congr 1; omega

theorem bsweepR_self (t : Tape) (adj : Array ℝ) (lo : Nat) : bsweepR t adj lo lo = adj := by
  unfold bsweepR; simp

theorem bsweepR_split (t : Tape) (adj : Array ℝ) (lo mid hi : Nat) (h1 : lo ≤ mid) (h2 : mid ≤ hi) :
    bsweepR t adj lo hi = bsweepR t (bsweepR t adj mid hi) lo mid := by
  unfold bsweepR
  rw [show hi - lo = (hi - mid) + (mid - lo) by omega, List.range_add, List.foldl_append,
      List.foldl_map]
  rw [List.foldl_ext (fun x y => stepNodeR t x (hi - 1 - (hi - mid + y)))
      (fun adj i => stepNodeR t adj (mid - 1 - i)) _
      (by intro b i hi'; rw [List.mem_range] at hi'
          show stepNodeR t b (hi - 1 - (hi - mid + i)) = stepNodeR t b (mid - 1 - i)
          have key : hi - 1 - (hi - mid + i) = mid - 1 - i := by omega
          rw [key])]

/-! #### `comp` layout: sizes and root index -/

theorem comp_deps_size (σ : Nat → Float) (e : Expr) :
    ∀ t : Tape, t.deps.size = t.val.size → (comp σ e t).2.deps.size = (comp σ e t).2.val.size := by
  induction e with
  | var i => intro t h; simp [comp, pushT, Array.size_push, h]
  | const c => intro t h; simp [comp, pushT, Array.size_push, h]
  | scale c a iha => intro t h; simp only [comp, pushT, Array.size_push]; rw [iha t h]
  | exp a iha => intro t h; simp only [comp, pushT, Array.size_push]; rw [iha t h]
  | log a iha => intro t h; simp only [comp, pushT, Array.size_push]; rw [iha t h]
  | relu a iha => intro t h; simp only [comp, pushT, Array.size_push]; rw [iha t h]
  | add a b iha ihb => intro t h; simp only [comp, pushT, Array.size_push]; rw [ihb _ (iha t h)]
  | sub a b iha ihb => intro t h; simp only [comp, pushT, Array.size_push]; rw [ihb _ (iha t h)]
  | mul a b iha ihb => intro t h; simp only [comp, pushT, Array.size_push]; rw [ihb _ (iha t h)]
  | max a b iha ihb => intro t h; simp only [comp, pushT, Array.size_push]; rw [ihb _ (iha t h)]
  | min a b iha ihb => intro t h; simp only [comp, pushT, Array.size_push]; rw [ihb _ (iha t h)]

theorem comp_root_succ (σ : Nat → Float) (e : Expr) (t : Tape) :
    (comp σ e t).1 + 1 = (comp σ e t).2.val.size := by
  cases e <;> simp only [comp, pushT, Array.size_push]

theorem comp_root_ge (σ : Nat → Float) (e : Expr) (t : Tape) : t.val.size ≤ (comp σ e t).1 := by
  cases e <;> simp only [comp, pushT] <;>
    first
    | exact le_refl _
    | (exact le_trans (comp_size_le σ _ t) (le_trans (comp_size_le σ _ _) (le_refl _)))
    | exact le_trans (comp_size_le σ _ t) (le_refl _)

/-- `comp` only appends `deps`, so `deps.size` is monotone. -/
theorem comp_deps_le (σ : Nat → Float) (e : Expr) (t : Tape) : t.deps.size ≤ (comp σ e t).2.deps.size := by
  induction e generalizing t with
  | var i => simp [comp, pushT, Array.size_push]
  | const c => simp [comp, pushT, Array.size_push]
  | scale c a iha => simp only [comp, pushT, Array.size_push]; exact le_trans (iha t) (Nat.le_succ _)
  | exp a iha => simp only [comp, pushT, Array.size_push]; exact le_trans (iha t) (Nat.le_succ _)
  | log a iha => simp only [comp, pushT, Array.size_push]; exact le_trans (iha t) (Nat.le_succ _)
  | relu a iha => simp only [comp, pushT, Array.size_push]; exact le_trans (iha t) (Nat.le_succ _)
  | add a b iha ihb => simp only [comp, pushT, Array.size_push]; exact le_trans (le_trans (iha t) (ihb _)) (Nat.le_succ _)
  | sub a b iha ihb => simp only [comp, pushT, Array.size_push]; exact le_trans (le_trans (iha t) (ihb _)) (Nat.le_succ _)
  | mul a b iha ihb => simp only [comp, pushT, Array.size_push]; exact le_trans (le_trans (iha t) (ihb _)) (Nat.le_succ _)
  | max a b iha ihb => simp only [comp, pushT, Array.size_push]; exact le_trans (le_trans (iha t) (ihb _)) (Nat.le_succ _)
  | min a b iha ihb => simp only [comp, pushT, Array.size_push]; exact le_trans (le_trans (iha t) (ihb _)) (Nat.le_succ _)

/-- Compiling preserves the `deps` of all pre-existing nodes (mirror of `comp_preserve`). -/
theorem comp_preserve_deps (σ : Nat → Float) (e : Expr) :
    ∀ (t : Tape) (j : Nat), j < t.deps.size → (comp σ e t).2.deps[j]! = t.deps[j]! := by
  induction e with
  | var i => intro t j hj; simpa only [comp, pushT] using push_getElem!_lt _ _ j hj
  | const c => intro t j hj; simpa only [comp, pushT] using push_getElem!_lt _ _ j hj
  | scale c a iha => intro t j hj; simp only [comp, pushT]
                     rw [push_getElem!_lt _ _ j (lt_of_lt_of_le hj (comp_deps_le σ a t))]; exact iha t j hj
  | exp a iha => intro t j hj; simp only [comp, pushT]
                 rw [push_getElem!_lt _ _ j (lt_of_lt_of_le hj (comp_deps_le σ a t))]; exact iha t j hj
  | log a iha => intro t j hj; simp only [comp, pushT]
                 rw [push_getElem!_lt _ _ j (lt_of_lt_of_le hj (comp_deps_le σ a t))]; exact iha t j hj
  | relu a iha => intro t j hj; simp only [comp, pushT]
                  rw [push_getElem!_lt _ _ j (lt_of_lt_of_le hj (comp_deps_le σ a t))]; exact iha t j hj
  | add a b iha ihb => intro t j hj; simp only [comp, pushT]
                       have h1 : j < (comp σ a t).2.deps.size := lt_of_lt_of_le hj (comp_deps_le σ a t)
                       have h2 : j < (comp σ b (comp σ a t).2).2.deps.size := lt_of_lt_of_le h1 (comp_deps_le σ b _)
                       rw [push_getElem!_lt _ _ j h2, ihb _ j h1, iha t j hj]
  | sub a b iha ihb => intro t j hj; simp only [comp, pushT]
                       have h1 : j < (comp σ a t).2.deps.size := lt_of_lt_of_le hj (comp_deps_le σ a t)
                       have h2 : j < (comp σ b (comp σ a t).2).2.deps.size := lt_of_lt_of_le h1 (comp_deps_le σ b _)
                       rw [push_getElem!_lt _ _ j h2, ihb _ j h1, iha t j hj]
  | mul a b iha ihb => intro t j hj; simp only [comp, pushT]
                       have h1 : j < (comp σ a t).2.deps.size := lt_of_lt_of_le hj (comp_deps_le σ a t)
                       have h2 : j < (comp σ b (comp σ a t).2).2.deps.size := lt_of_lt_of_le h1 (comp_deps_le σ b _)
                       rw [push_getElem!_lt _ _ j h2, ihb _ j h1, iha t j hj]
  | max a b iha ihb => intro t j hj; simp only [comp, pushT]
                       have h1 : j < (comp σ a t).2.deps.size := lt_of_lt_of_le hj (comp_deps_le σ a t)
                       have h2 : j < (comp σ b (comp σ a t).2).2.deps.size := lt_of_lt_of_le h1 (comp_deps_le σ b _)
                       rw [push_getElem!_lt _ _ j h2, ihb _ j h1, iha t j hj]
  | min a b iha ihb => intro t j hj; simp only [comp, pushT]
                       have h1 : j < (comp σ a t).2.deps.size := lt_of_lt_of_le hj (comp_deps_le σ a t)
                       have h2 : j < (comp σ b (comp σ a t).2).2.deps.size := lt_of_lt_of_le h1 (comp_deps_le σ b _)
                       rw [push_getElem!_lt _ _ j h2, ihb _ j h1, iha t j hj]

/-! #### Step/sweep frame lemmas: reads, size, and untouched-index preservation -/

theorem stepNodeR_size (t : Tape) (adj : Array ℝ) (idx : Nat) :
    (stepNodeR t adj idx).size = adj.size := by
  show (Array.foldl _ adj (t.deps[idx]!)).size = adj.size
  refine Array.foldl_induction (motive := fun _ (acc : Array ℝ) => acc.size = adj.size) rfl (fun i acc h => ?_)
  simp only []; rw [Array.size_set!]; exact h

theorem stepNodeR_empty (t : Tape) (adj : Array ℝ) (idx : Nat) (h : t.deps[idx]! = #[]) :
    stepNodeR t adj idx = adj := by unfold stepNodeR; rw [h]; rfl

/-- One-edge node: reads the single child, adding the scaled edge weight. -/
theorem stepNodeR_single_at (t : Tape) (adj : Array ℝ) (idx p : Nat) (d : Float)
    (hdeps : t.deps[idx]! = #[(p, d)]) (hp : p < adj.size) :
    (stepNodeR t adj idx)[p]! = adj[p]! + adj[idx]! * toReal d := by
  unfold stepNodeR; rw [hdeps]
  show ((adj.set! p (adj[p]! + adj[idx]! * toReal d)))[p]! = _
  exact set!_getElem!_self adj p _ hp

/-- Two-edge node: read at the FIRST child `p` (through the later `set!` at `q`, `p ≠ q`). -/
theorem stepNodeR_two_at_p (t : Tape) (adj : Array ℝ) (idx p q : Nat) (dp dq : Float)
    (hdeps : t.deps[idx]! = #[(p, dp), (q, dq)]) (hpq : p ≠ q) (hp : p < adj.size) :
    (stepNodeR t adj idx)[p]! = adj[p]! + adj[idx]! * toReal dp := by
  unfold stepNodeR; rw [hdeps]
  show (((adj.set! p (adj[p]! + adj[idx]! * toReal dp)).set! q
        ((adj.set! p (adj[p]! + adj[idx]! * toReal dp))[q]! + adj[idx]! * toReal dq)))[p]! = _
  rw [set!_getElem!_ne_gen _ q _ p hpq.symm]; exact set!_getElem!_self adj p _ hp

/-- Two-edge node: read at the SECOND child `q` (the last `set!`). -/
theorem stepNodeR_two_at_q (t : Tape) (adj : Array ℝ) (idx p q : Nat) (dp dq : Float)
    (hdeps : t.deps[idx]! = #[(p, dp), (q, dq)]) (hq : q < adj.size) :
    (stepNodeR t adj idx)[q]! = (adj.set! p (adj[p]! + adj[idx]! * toReal dp))[q]! + adj[idx]! * toReal dq := by
  unfold stepNodeR; rw [hdeps]
  show (((adj.set! p (adj[p]! + adj[idx]! * toReal dp)).set! q
        ((adj.set! p (adj[p]! + adj[idx]! * toReal dp))[q]! + adj[idx]! * toReal dq)))[q]! = _
  refine set!_getElem!_self _ q _ ?_; rw [Array.size_set!]; exact hq

/-- A node whose edges never target `j` leaves `adj[j]` untouched. -/
theorem stepNodeR_preserve (t : Tape) (adj : Array ℝ) (idx j : Nat)
    (hj : ∀ e ∈ t.deps[idx]!, e.1 ≠ j) : (stepNodeR t adj idx)[j]! = adj[j]! := by
  show (Array.foldl _ adj (t.deps[idx]!))[j]! = adj[j]!
  refine Array.foldl_induction (motive := fun _ (acc : Array ℝ) => acc[j]! = adj[j]!) rfl (fun i acc h => ?_)
  simp only []
  have hne : (t.deps[idx]!)[i].1 ≠ j := hj _ (Array.mem_of_getElem rfl)
  exact (set!_getElem!_ne_gen acc _ _ j hne).trans h

theorem bsweepR_size (t : Tape) (adj : Array ℝ) (lo hi : Nat) :
    (bsweepR t adj lo hi).size = adj.size := by
  unfold bsweepR
  refine List.foldlRecOn (List.range (hi-lo)) (fun adj i => stepNodeR t adj (hi-1-i))
    (motive := fun (acc : Array ℝ) => acc.size = adj.size) rfl (fun acc h i _ => ?_)
  simp only []; rw [stepNodeR_size]; exact h

/-- Sweeping a window whose nodes never target `j` preserves `adj[j]`. -/
theorem bsweepR_preserve (t : Tape) (adj : Array ℝ) (lo hi j : Nat)
    (hj : ∀ idx, lo ≤ idx → idx < hi → ∀ e ∈ t.deps[idx]!, e.1 ≠ j) :
    (bsweepR t adj lo hi)[j]! = adj[j]! := by
  unfold bsweepR
  refine List.foldlRecOn (List.range (hi-lo)) (fun adj i => stepNodeR t adj (hi-1-i))
    (motive := fun (acc : Array ℝ) => acc[j]! = adj[j]!) rfl (fun acc h i hi_mem => ?_)
  rw [List.mem_range] at hi_mem
  rw [stepNodeR_preserve t acc (hi - 1 - i) j (hj (hi-1-i) (by omega) (by omega))]; exact h

/-- The sweep depends only on the tape's `deps` over the swept window. -/
theorem bsweepR_congr (t t' : Tape) (adj : Array ℝ) (lo hi : Nat)
    (h : ∀ idx, lo ≤ idx → idx < hi → t.deps[idx]! = t'.deps[idx]!) :
    bsweepR t adj lo hi = bsweepR t' adj lo hi := by
  unfold bsweepR
  apply List.foldl_ext
  intro b i hi_mem; rw [List.mem_range] at hi_mem
  show stepNodeR t b (hi-1-i) = stepNodeR t' b (hi-1-i)
  unfold stepNodeR; rw [h (hi-1-i) (by omega) (by omega)]


/-! #### Stage C: tree-locality, readout-congruence, the central invariant, and the payoff

With the frame lemmas in place, `comp_edges_range` (every edge of an in-range node points strictly
earlier — the tree-locality that makes each adjoint written exactly once) and `leafAdjSumR_congr`
(the readout depends only on the block's adjoints) feed the `central` induction, which threads the
incoming root adjoint through `comp`'s nested regions. `bsweepR_reads_revE_RF` instantiates it at the
empty tape (the flat ℝ sweep reads off `revE_RF`); `bsweepR_reads_derivR` reaches the true derivative
via `revE_R_eq_derivR`, conditional on the per-fragment primal bridge. Both are over the exact-ℝ sweep
`bsweepR` (= the ℝ image of `gradsF`'s fold); the sweep's own Float rounding is the Layer-2 tail. -/

theorem comp_edges_range (σ : Nat → Float) (e : Expr) :
    ∀ (t : Tape) (idx : Nat), t.deps.size = t.val.size → t.val.size ≤ idx → idx ≤ (comp σ e t).1 →
      ∀ ed ∈ (comp σ e t).2.deps[idx]!, t.val.size ≤ ed.1 ∧ ed.1 < idx := by
  induction e with
  | var i =>
      intro t idx hwf hlo hhi ed hed
      have hroot : (comp σ (Expr.var i) t).1 = t.val.size := by simp only [comp]
      have hidx : idx = t.val.size := le_antisymm (hhi.trans_eq hroot) hlo
      subst hidx
      simp only [comp, pushT] at hed
      rw [← hwf, push_getElem!_size] at hed
      simp at hed
  | const c =>
      intro t idx hwf hlo hhi ed hed
      have hroot : (comp σ (Expr.const c) t).1 = t.val.size := by simp only [comp]
      have hidx : idx = t.val.size := le_antisymm (hhi.trans_eq hroot) hlo
      subst hidx
      simp only [comp, pushT] at hed
      rw [← hwf, push_getElem!_size] at hed
      simp at hed
  | scale c a iha =>
      intro t idx hwf hlo hhi ed hed
      have hroot : (comp σ (Expr.scale c a) t).1 = (comp σ a t).2.val.size := by simp only [comp]
      have htape : (comp σ (Expr.scale c a) t).2 =
          pushT (comp σ a t).2 (c * (comp σ a t).2.val[(comp σ a t).1]!) #[((comp σ a t).1, c)] := by
        simp only [comp]
      have hdsz : (comp σ a t).2.deps.size = (comp σ a t).2.val.size := comp_deps_size σ a t hwf
      rw [hroot] at hhi
      rcases eq_or_lt_of_le hhi with hidxeq | hidxlt
      · subst hidxeq
        rw [htape] at hed; simp only [pushT] at hed
        rw [← hdsz, push_getElem!_size] at hed
        simp only [Array.mem_singleton] at hed
        subst hed
        exact ⟨comp_root_ge σ a t, comp_root_lt σ a t⟩
      · have hidx_lt_dsz : idx < (comp σ a t).2.deps.size := by rw [hdsz]; exact hidxlt
        rw [htape] at hed; simp only [pushT] at hed
        rw [push_getElem!_lt _ _ idx hidx_lt_dsz] at hed
        have hidx_le : idx ≤ (comp σ a t).1 := by
          have hs := comp_root_succ σ a t; rw [← hs] at hidxlt; omega
        exact iha t idx hwf hlo hidx_le ed hed
  | exp a iha =>
      intro t idx hwf hlo hhi ed hed
      have hroot : (comp σ (Expr.exp a) t).1 = (comp σ a t).2.val.size := by simp only [comp]
      have htape : (comp σ (Expr.exp a) t).2 =
          pushT (comp σ a t).2 (Float.exp (comp σ a t).2.val[(comp σ a t).1]!)
            #[((comp σ a t).1, Float.exp (comp σ a t).2.val[(comp σ a t).1]!)] := by simp only [comp]
      have hdsz : (comp σ a t).2.deps.size = (comp σ a t).2.val.size := comp_deps_size σ a t hwf
      rw [hroot] at hhi
      rcases eq_or_lt_of_le hhi with hidxeq | hidxlt
      · subst hidxeq
        rw [htape] at hed; simp only [pushT] at hed
        rw [← hdsz, push_getElem!_size] at hed
        simp only [Array.mem_singleton] at hed
        subst hed
        exact ⟨comp_root_ge σ a t, comp_root_lt σ a t⟩
      · have hidx_lt_dsz : idx < (comp σ a t).2.deps.size := by rw [hdsz]; exact hidxlt
        rw [htape] at hed; simp only [pushT] at hed
        rw [push_getElem!_lt _ _ idx hidx_lt_dsz] at hed
        have hidx_le : idx ≤ (comp σ a t).1 := by
          have hs := comp_root_succ σ a t; rw [← hs] at hidxlt; omega
        exact iha t idx hwf hlo hidx_le ed hed
  | log a iha =>
      intro t idx hwf hlo hhi ed hed
      have hroot : (comp σ (Expr.log a) t).1 = (comp σ a t).2.val.size := by simp only [comp]
      have htape : (comp σ (Expr.log a) t).2 =
          pushT (comp σ a t).2 (Float.log (comp σ a t).2.val[(comp σ a t).1]!)
            #[((comp σ a t).1, 1.0 / (comp σ a t).2.val[(comp σ a t).1]!)] := by simp only [comp]
      have hdsz : (comp σ a t).2.deps.size = (comp σ a t).2.val.size := comp_deps_size σ a t hwf
      rw [hroot] at hhi
      rcases eq_or_lt_of_le hhi with hidxeq | hidxlt
      · subst hidxeq
        rw [htape] at hed; simp only [pushT] at hed
        rw [← hdsz, push_getElem!_size] at hed
        simp only [Array.mem_singleton] at hed
        subst hed
        exact ⟨comp_root_ge σ a t, comp_root_lt σ a t⟩
      · have hidx_lt_dsz : idx < (comp σ a t).2.deps.size := by rw [hdsz]; exact hidxlt
        rw [htape] at hed; simp only [pushT] at hed
        rw [push_getElem!_lt _ _ idx hidx_lt_dsz] at hed
        have hidx_le : idx ≤ (comp σ a t).1 := by
          have hs := comp_root_succ σ a t; rw [← hs] at hidxlt; omega
        exact iha t idx hwf hlo hidx_le ed hed
  | relu a iha =>
      intro t idx hwf hlo hhi ed hed
      have hroot : (comp σ (Expr.relu a) t).1 = (comp σ a t).2.val.size := by simp only [comp]
      have htape : (comp σ (Expr.relu a) t).2 =
          pushT (comp σ a t).2 (if (comp σ a t).2.val[(comp σ a t).1]! < 0.0 then 0.0 else (comp σ a t).2.val[(comp σ a t).1]!)
            #[((comp σ a t).1, if (comp σ a t).2.val[(comp σ a t).1]! < 0.0 then 0.0 else 1.0)] := by simp only [comp]
      have hdsz : (comp σ a t).2.deps.size = (comp σ a t).2.val.size := comp_deps_size σ a t hwf
      rw [hroot] at hhi
      rcases eq_or_lt_of_le hhi with hidxeq | hidxlt
      · subst hidxeq
        rw [htape] at hed; simp only [pushT] at hed
        rw [← hdsz, push_getElem!_size] at hed
        simp only [Array.mem_singleton] at hed
        subst hed
        exact ⟨comp_root_ge σ a t, comp_root_lt σ a t⟩
      · have hidx_lt_dsz : idx < (comp σ a t).2.deps.size := by rw [hdsz]; exact hidxlt
        rw [htape] at hed; simp only [pushT] at hed
        rw [push_getElem!_lt _ _ idx hidx_lt_dsz] at hed
        have hidx_le : idx ≤ (comp σ a t).1 := by
          have hs := comp_root_succ σ a t; rw [← hs] at hidxlt; omega
        exact iha t idx hwf hlo hidx_le ed hed
  | add a b iha ihb =>
      intro t idx hwf hlo hhi ed hed
      have hroot : (comp σ (Expr.add a b) t).1 = (comp σ b (comp σ a t).2).2.val.size := by simp only [comp]
      have htape : (comp σ (Expr.add a b) t).2 =
          pushT (comp σ b (comp σ a t).2).2
            ((comp σ b (comp σ a t).2).2.val[(comp σ a t).1]! + (comp σ b (comp σ a t).2).2.val[(comp σ b (comp σ a t).2).1]!)
            #[((comp σ a t).1, (1.0:Float)), ((comp σ b (comp σ a t).2).1, (1.0:Float))] := by simp only [comp]
      have hwf_ra : (comp σ a t).2.deps.size = (comp σ a t).2.val.size := comp_deps_size σ a t hwf
      have hdsz : (comp σ b (comp σ a t).2).2.deps.size = (comp σ b (comp σ a t).2).2.val.size :=
        comp_deps_size σ b (comp σ a t).2 hwf_ra
      rw [hroot] at hhi
      rcases eq_or_lt_of_le hhi with hidxeq | hidxlt
      · subst hidxeq
        rw [htape] at hed; simp only [pushT] at hed
        rw [← hdsz, push_getElem!_size] at hed
        simp only [Array.mem_def, List.mem_cons, List.not_mem_nil, or_false] at hed
        have hra_ge : t.val.size ≤ (comp σ a t).1 := comp_root_ge σ a t
        have hra_lt : (comp σ a t).1 < (comp σ a t).2.val.size := comp_root_lt σ a t
        have hra2_le : (comp σ a t).2.val.size ≤ (comp σ b (comp σ a t).2).2.val.size := comp_size_le σ b _
        have hrb_ge : (comp σ a t).2.val.size ≤ (comp σ b (comp σ a t).2).1 := comp_root_ge σ b _
        have hrb_lt : (comp σ b (comp σ a t).2).1 < (comp σ b (comp σ a t).2).2.val.size := comp_root_lt σ b _
        rcases hed with h1 | h2
        · rw [h1]; exact ⟨hra_ge, lt_of_lt_of_le hra_lt hra2_le⟩
        · rw [h2]; exact ⟨le_trans hra_ge (le_trans (le_of_lt hra_lt) hrb_ge), hrb_lt⟩
      · rw [htape] at hed; simp only [pushT] at hed
        have hidx_lt_dsz : idx < (comp σ b (comp σ a t).2).2.deps.size := by rw [hdsz]; exact hidxlt
        rw [push_getElem!_lt _ _ idx hidx_lt_dsz] at hed
        by_cases hreg : idx < (comp σ a t).2.val.size
        · have hidx_lt_ra_dsz : idx < (comp σ a t).2.deps.size := by rw [hwf_ra]; exact hreg
          rw [comp_preserve_deps σ b (comp σ a t).2 idx hidx_lt_ra_dsz] at hed
          have hidx_le : idx ≤ (comp σ a t).1 := by
            have hs := comp_root_succ σ a t; rw [← hs] at hreg; omega
          exact iha t idx hwf hlo hidx_le ed hed
        · push_neg at hreg
          have hidx_le : idx ≤ (comp σ b (comp σ a t).2).1 := by
            have hs := comp_root_succ σ b (comp σ a t).2; rw [← hs] at hidxlt; omega
          have hres := ihb (comp σ a t).2 idx hwf_ra hreg hidx_le ed hed
          exact ⟨le_trans (comp_size_le σ a t) hres.1, hres.2⟩
  | sub a b iha ihb =>
      intro t idx hwf hlo hhi ed hed
      have hroot : (comp σ (Expr.sub a b) t).1 = (comp σ b (comp σ a t).2).2.val.size := by simp only [comp]
      have htape : (comp σ (Expr.sub a b) t).2 =
          pushT (comp σ b (comp σ a t).2).2
            ((comp σ b (comp σ a t).2).2.val[(comp σ a t).1]! - (comp σ b (comp σ a t).2).2.val[(comp σ b (comp σ a t).2).1]!)
            #[((comp σ a t).1, (1.0:Float)), ((comp σ b (comp σ a t).2).1, (-1.0:Float))] := by simp only [comp]
      have hwf_ra : (comp σ a t).2.deps.size = (comp σ a t).2.val.size := comp_deps_size σ a t hwf
      have hdsz : (comp σ b (comp σ a t).2).2.deps.size = (comp σ b (comp σ a t).2).2.val.size :=
        comp_deps_size σ b (comp σ a t).2 hwf_ra
      rw [hroot] at hhi
      rcases eq_or_lt_of_le hhi with hidxeq | hidxlt
      · subst hidxeq
        rw [htape] at hed; simp only [pushT] at hed
        rw [← hdsz, push_getElem!_size] at hed
        simp only [Array.mem_def, List.mem_cons, List.not_mem_nil, or_false] at hed
        have hra_ge : t.val.size ≤ (comp σ a t).1 := comp_root_ge σ a t
        have hra_lt : (comp σ a t).1 < (comp σ a t).2.val.size := comp_root_lt σ a t
        have hra2_le : (comp σ a t).2.val.size ≤ (comp σ b (comp σ a t).2).2.val.size := comp_size_le σ b _
        have hrb_ge : (comp σ a t).2.val.size ≤ (comp σ b (comp σ a t).2).1 := comp_root_ge σ b _
        have hrb_lt : (comp σ b (comp σ a t).2).1 < (comp σ b (comp σ a t).2).2.val.size := comp_root_lt σ b _
        rcases hed with h1 | h2
        · rw [h1]; exact ⟨hra_ge, lt_of_lt_of_le hra_lt hra2_le⟩
        · rw [h2]; exact ⟨le_trans hra_ge (le_trans (le_of_lt hra_lt) hrb_ge), hrb_lt⟩
      · rw [htape] at hed; simp only [pushT] at hed
        have hidx_lt_dsz : idx < (comp σ b (comp σ a t).2).2.deps.size := by rw [hdsz]; exact hidxlt
        rw [push_getElem!_lt _ _ idx hidx_lt_dsz] at hed
        by_cases hreg : idx < (comp σ a t).2.val.size
        · have hidx_lt_ra_dsz : idx < (comp σ a t).2.deps.size := by rw [hwf_ra]; exact hreg
          rw [comp_preserve_deps σ b (comp σ a t).2 idx hidx_lt_ra_dsz] at hed
          have hidx_le : idx ≤ (comp σ a t).1 := by
            have hs := comp_root_succ σ a t; rw [← hs] at hreg; omega
          exact iha t idx hwf hlo hidx_le ed hed
        · push_neg at hreg
          have hidx_le : idx ≤ (comp σ b (comp σ a t).2).1 := by
            have hs := comp_root_succ σ b (comp σ a t).2; rw [← hs] at hidxlt; omega
          have hres := ihb (comp σ a t).2 idx hwf_ra hreg hidx_le ed hed
          exact ⟨le_trans (comp_size_le σ a t) hres.1, hres.2⟩
  | mul a b iha ihb =>
      intro t idx hwf hlo hhi ed hed
      have hroot : (comp σ (Expr.mul a b) t).1 = (comp σ b (comp σ a t).2).2.val.size := by simp only [comp]
      have htape : (comp σ (Expr.mul a b) t).2 =
          pushT (comp σ b (comp σ a t).2).2
            ((comp σ b (comp σ a t).2).2.val[(comp σ a t).1]! * (comp σ b (comp σ a t).2).2.val[(comp σ b (comp σ a t).2).1]!)
            #[((comp σ a t).1, (comp σ b (comp σ a t).2).2.val[(comp σ b (comp σ a t).2).1]!),
              ((comp σ b (comp σ a t).2).1, (comp σ b (comp σ a t).2).2.val[(comp σ a t).1]!)] := by simp only [comp]
      have hwf_ra : (comp σ a t).2.deps.size = (comp σ a t).2.val.size := comp_deps_size σ a t hwf
      have hdsz : (comp σ b (comp σ a t).2).2.deps.size = (comp σ b (comp σ a t).2).2.val.size :=
        comp_deps_size σ b (comp σ a t).2 hwf_ra
      rw [hroot] at hhi
      rcases eq_or_lt_of_le hhi with hidxeq | hidxlt
      · subst hidxeq
        rw [htape] at hed; simp only [pushT] at hed
        rw [← hdsz, push_getElem!_size] at hed
        simp only [Array.mem_def, List.mem_cons, List.not_mem_nil, or_false] at hed
        have hra_ge : t.val.size ≤ (comp σ a t).1 := comp_root_ge σ a t
        have hra_lt : (comp σ a t).1 < (comp σ a t).2.val.size := comp_root_lt σ a t
        have hra2_le : (comp σ a t).2.val.size ≤ (comp σ b (comp σ a t).2).2.val.size := comp_size_le σ b _
        have hrb_ge : (comp σ a t).2.val.size ≤ (comp σ b (comp σ a t).2).1 := comp_root_ge σ b _
        have hrb_lt : (comp σ b (comp σ a t).2).1 < (comp σ b (comp σ a t).2).2.val.size := comp_root_lt σ b _
        rcases hed with h1 | h2
        · rw [h1]; exact ⟨hra_ge, lt_of_lt_of_le hra_lt hra2_le⟩
        · rw [h2]; exact ⟨le_trans hra_ge (le_trans (le_of_lt hra_lt) hrb_ge), hrb_lt⟩
      · rw [htape] at hed; simp only [pushT] at hed
        have hidx_lt_dsz : idx < (comp σ b (comp σ a t).2).2.deps.size := by rw [hdsz]; exact hidxlt
        rw [push_getElem!_lt _ _ idx hidx_lt_dsz] at hed
        by_cases hreg : idx < (comp σ a t).2.val.size
        · have hidx_lt_ra_dsz : idx < (comp σ a t).2.deps.size := by rw [hwf_ra]; exact hreg
          rw [comp_preserve_deps σ b (comp σ a t).2 idx hidx_lt_ra_dsz] at hed
          have hidx_le : idx ≤ (comp σ a t).1 := by
            have hs := comp_root_succ σ a t; rw [← hs] at hreg; omega
          exact iha t idx hwf hlo hidx_le ed hed
        · push_neg at hreg
          have hidx_le : idx ≤ (comp σ b (comp σ a t).2).1 := by
            have hs := comp_root_succ σ b (comp σ a t).2; rw [← hs] at hidxlt; omega
          have hres := ihb (comp σ a t).2 idx hwf_ra hreg hidx_le ed hed
          exact ⟨le_trans (comp_size_le σ a t) hres.1, hres.2⟩
  | max a b iha ihb =>
      intro t idx hwf hlo hhi ed hed
      have hroot : (comp σ (Expr.max a b) t).1 = (comp σ b (comp σ a t).2).2.val.size := by simp only [comp]
      have htape : (comp σ (Expr.max a b) t).2 =
          pushT (comp σ b (comp σ a t).2).2
            (if (comp σ b (comp σ a t).2).2.val[(comp σ a t).1]! ≤ (comp σ b (comp σ a t).2).2.val[(comp σ b (comp σ a t).2).1]!
              then (comp σ b (comp σ a t).2).2.val[(comp σ b (comp σ a t).2).1]! else (comp σ b (comp σ a t).2).2.val[(comp σ a t).1]!)
            #[((comp σ a t).1, if (comp σ b (comp σ a t).2).2.val[(comp σ a t).1]! ≤ (comp σ b (comp σ a t).2).2.val[(comp σ b (comp σ a t).2).1]! then (0.0:Float) else 1.0),
              ((comp σ b (comp σ a t).2).1, if (comp σ b (comp σ a t).2).2.val[(comp σ a t).1]! ≤ (comp σ b (comp σ a t).2).2.val[(comp σ b (comp σ a t).2).1]! then (1.0:Float) else 0.0)] := by simp only [comp]
      have hwf_ra : (comp σ a t).2.deps.size = (comp σ a t).2.val.size := comp_deps_size σ a t hwf
      have hdsz : (comp σ b (comp σ a t).2).2.deps.size = (comp σ b (comp σ a t).2).2.val.size :=
        comp_deps_size σ b (comp σ a t).2 hwf_ra
      rw [hroot] at hhi
      rcases eq_or_lt_of_le hhi with hidxeq | hidxlt
      · subst hidxeq
        rw [htape] at hed; simp only [pushT] at hed
        rw [← hdsz, push_getElem!_size] at hed
        simp only [Array.mem_def, List.mem_cons, List.not_mem_nil, or_false] at hed
        have hra_ge : t.val.size ≤ (comp σ a t).1 := comp_root_ge σ a t
        have hra_lt : (comp σ a t).1 < (comp σ a t).2.val.size := comp_root_lt σ a t
        have hra2_le : (comp σ a t).2.val.size ≤ (comp σ b (comp σ a t).2).2.val.size := comp_size_le σ b _
        have hrb_ge : (comp σ a t).2.val.size ≤ (comp σ b (comp σ a t).2).1 := comp_root_ge σ b _
        have hrb_lt : (comp σ b (comp σ a t).2).1 < (comp σ b (comp σ a t).2).2.val.size := comp_root_lt σ b _
        rcases hed with h1 | h2
        · rw [h1]; exact ⟨hra_ge, lt_of_lt_of_le hra_lt hra2_le⟩
        · rw [h2]; exact ⟨le_trans hra_ge (le_trans (le_of_lt hra_lt) hrb_ge), hrb_lt⟩
      · rw [htape] at hed; simp only [pushT] at hed
        have hidx_lt_dsz : idx < (comp σ b (comp σ a t).2).2.deps.size := by rw [hdsz]; exact hidxlt
        rw [push_getElem!_lt _ _ idx hidx_lt_dsz] at hed
        by_cases hreg : idx < (comp σ a t).2.val.size
        · have hidx_lt_ra_dsz : idx < (comp σ a t).2.deps.size := by rw [hwf_ra]; exact hreg
          rw [comp_preserve_deps σ b (comp σ a t).2 idx hidx_lt_ra_dsz] at hed
          have hidx_le : idx ≤ (comp σ a t).1 := by
            have hs := comp_root_succ σ a t; rw [← hs] at hreg; omega
          exact iha t idx hwf hlo hidx_le ed hed
        · push_neg at hreg
          have hidx_le : idx ≤ (comp σ b (comp σ a t).2).1 := by
            have hs := comp_root_succ σ b (comp σ a t).2; rw [← hs] at hidxlt; omega
          have hres := ihb (comp σ a t).2 idx hwf_ra hreg hidx_le ed hed
          exact ⟨le_trans (comp_size_le σ a t) hres.1, hres.2⟩
  | min a b iha ihb =>
      intro t idx hwf hlo hhi ed hed
      have hroot : (comp σ (Expr.min a b) t).1 = (comp σ b (comp σ a t).2).2.val.size := by simp only [comp]
      have htape : (comp σ (Expr.min a b) t).2 =
          pushT (comp σ b (comp σ a t).2).2
            (if (comp σ b (comp σ a t).2).2.val[(comp σ a t).1]! ≤ (comp σ b (comp σ a t).2).2.val[(comp σ b (comp σ a t).2).1]!
              then (comp σ b (comp σ a t).2).2.val[(comp σ a t).1]! else (comp σ b (comp σ a t).2).2.val[(comp σ b (comp σ a t).2).1]!)
            #[((comp σ a t).1, if (comp σ b (comp σ a t).2).2.val[(comp σ a t).1]! ≤ (comp σ b (comp σ a t).2).2.val[(comp σ b (comp σ a t).2).1]! then (1.0:Float) else 0.0),
              ((comp σ b (comp σ a t).2).1, if (comp σ b (comp σ a t).2).2.val[(comp σ a t).1]! ≤ (comp σ b (comp σ a t).2).2.val[(comp σ b (comp σ a t).2).1]! then (0.0:Float) else 1.0)] := by simp only [comp]
      have hwf_ra : (comp σ a t).2.deps.size = (comp σ a t).2.val.size := comp_deps_size σ a t hwf
      have hdsz : (comp σ b (comp σ a t).2).2.deps.size = (comp σ b (comp σ a t).2).2.val.size :=
        comp_deps_size σ b (comp σ a t).2 hwf_ra
      rw [hroot] at hhi
      rcases eq_or_lt_of_le hhi with hidxeq | hidxlt
      · subst hidxeq
        rw [htape] at hed; simp only [pushT] at hed
        rw [← hdsz, push_getElem!_size] at hed
        simp only [Array.mem_def, List.mem_cons, List.not_mem_nil, or_false] at hed
        have hra_ge : t.val.size ≤ (comp σ a t).1 := comp_root_ge σ a t
        have hra_lt : (comp σ a t).1 < (comp σ a t).2.val.size := comp_root_lt σ a t
        have hra2_le : (comp σ a t).2.val.size ≤ (comp σ b (comp σ a t).2).2.val.size := comp_size_le σ b _
        have hrb_ge : (comp σ a t).2.val.size ≤ (comp σ b (comp σ a t).2).1 := comp_root_ge σ b _
        have hrb_lt : (comp σ b (comp σ a t).2).1 < (comp σ b (comp σ a t).2).2.val.size := comp_root_lt σ b _
        rcases hed with h1 | h2
        · rw [h1]; exact ⟨hra_ge, lt_of_lt_of_le hra_lt hra2_le⟩
        · rw [h2]; exact ⟨le_trans hra_ge (le_trans (le_of_lt hra_lt) hrb_ge), hrb_lt⟩
      · rw [htape] at hed; simp only [pushT] at hed
        have hidx_lt_dsz : idx < (comp σ b (comp σ a t).2).2.deps.size := by rw [hdsz]; exact hidxlt
        rw [push_getElem!_lt _ _ idx hidx_lt_dsz] at hed
        by_cases hreg : idx < (comp σ a t).2.val.size
        · have hidx_lt_ra_dsz : idx < (comp σ a t).2.deps.size := by rw [hwf_ra]; exact hreg
          rw [comp_preserve_deps σ b (comp σ a t).2 idx hidx_lt_ra_dsz] at hed
          have hidx_le : idx ≤ (comp σ a t).1 := by
            have hs := comp_root_succ σ a t; rw [← hs] at hreg; omega
          exact iha t idx hwf hlo hidx_le ed hed
        · push_neg at hreg
          have hidx_le : idx ≤ (comp σ b (comp σ a t).2).1 := by
            have hs := comp_root_succ σ b (comp σ a t).2; rw [← hs] at hidxlt; omega
          have hres := ihb (comp σ a t).2 idx hwf_ra hreg hidx_le ed hed
          exact ⟨le_trans (comp_size_le σ a t) hres.1, hres.2⟩


theorem leafAdjSumR_congr (σ : Nat → Float) (adj adj2 : Array ℝ) (e : Expr) :
    ∀ (t : Tape) (k : Nat), (∀ j, t.val.size ≤ j → j < (comp σ e t).2.val.size → adj[j]! = adj2[j]!) →
      leafAdjSumR σ adj e t k = leafAdjSumR σ adj2 e t k := by
  induction e with
  | var i =>
      intro t k h
      simp only [leafAdjSumR]
      have hbound : t.val.size < (comp σ (.var i) t).2.val.size := by
        simp only [comp, pushT, Array.size_push]; exact Nat.lt_succ_self _
      by_cases hik : i = k
      · rw [if_pos hik, if_pos hik, h t.val.size (le_refl _) hbound]
      · rw [if_neg hik, if_neg hik]
  | const c => intro t k h; simp only [leafAdjSumR]
  | scale c a iha =>
      intro t k h
      simp only [leafAdjSumR]
      apply iha t k
      intro j hj1 hj2
      apply h j hj1
      calc j < (comp σ a t).2.val.size := hj2
        _ ≤ (comp σ (.scale c a) t).2.val.size := by
              simp only [comp, pushT, Array.size_push]; exact Nat.le_succ _
  | exp a iha =>
      intro t k h
      simp only [leafAdjSumR]
      apply iha t k
      intro j hj1 hj2
      apply h j hj1
      calc j < (comp σ a t).2.val.size := hj2
        _ ≤ (comp σ (.exp a) t).2.val.size := by
              simp only [comp, pushT, Array.size_push]; exact Nat.le_succ _
  | log a iha =>
      intro t k h
      simp only [leafAdjSumR]
      apply iha t k
      intro j hj1 hj2
      apply h j hj1
      calc j < (comp σ a t).2.val.size := hj2
        _ ≤ (comp σ (.log a) t).2.val.size := by
              simp only [comp, pushT, Array.size_push]; exact Nat.le_succ _
  | relu a iha =>
      intro t k h
      simp only [leafAdjSumR]
      apply iha t k
      intro j hj1 hj2
      apply h j hj1
      calc j < (comp σ a t).2.val.size := hj2
        _ ≤ (comp σ (.relu a) t).2.val.size := by
              simp only [comp, pushT, Array.size_push]; exact Nat.le_succ _
  | add a b iha ihb =>
      intro t k h
      simp only [leafAdjSumR]
      have hab : (comp σ a t).2.val.size ≤ (comp σ b (comp σ a t).2).2.val.size := comp_size_le σ b _
      have hbtop : (comp σ b (comp σ a t).2).2.val.size ≤ (comp σ (.add a b) t).2.val.size := by
        simp only [comp, pushT, Array.size_push]; exact Nat.le_succ _
      have hta : t.val.size ≤ (comp σ a t).2.val.size := comp_size_le σ a t
      rw [iha t k (fun j hj1 hj2 => h j hj1 (lt_of_lt_of_le (lt_of_lt_of_le hj2 hab) hbtop))]
      rw [ihb (comp σ a t).2 k (fun j hj1 hj2 => h j (le_trans hta hj1) (lt_of_lt_of_le hj2 hbtop))]
  | sub a b iha ihb =>
      intro t k h
      simp only [leafAdjSumR]
      have hab : (comp σ a t).2.val.size ≤ (comp σ b (comp σ a t).2).2.val.size := comp_size_le σ b _
      have hbtop : (comp σ b (comp σ a t).2).2.val.size ≤ (comp σ (.sub a b) t).2.val.size := by
        simp only [comp, pushT, Array.size_push]; exact Nat.le_succ _
      have hta : t.val.size ≤ (comp σ a t).2.val.size := comp_size_le σ a t
      rw [iha t k (fun j hj1 hj2 => h j hj1 (lt_of_lt_of_le (lt_of_lt_of_le hj2 hab) hbtop))]
      rw [ihb (comp σ a t).2 k (fun j hj1 hj2 => h j (le_trans hta hj1) (lt_of_lt_of_le hj2 hbtop))]
  | mul a b iha ihb =>
      intro t k h
      simp only [leafAdjSumR]
      have hab : (comp σ a t).2.val.size ≤ (comp σ b (comp σ a t).2).2.val.size := comp_size_le σ b _
      have hbtop : (comp σ b (comp σ a t).2).2.val.size ≤ (comp σ (.mul a b) t).2.val.size := by
        simp only [comp, pushT, Array.size_push]; exact Nat.le_succ _
      have hta : t.val.size ≤ (comp σ a t).2.val.size := comp_size_le σ a t
      rw [iha t k (fun j hj1 hj2 => h j hj1 (lt_of_lt_of_le (lt_of_lt_of_le hj2 hab) hbtop))]
      rw [ihb (comp σ a t).2 k (fun j hj1 hj2 => h j (le_trans hta hj1) (lt_of_lt_of_le hj2 hbtop))]
  | max a b iha ihb =>
      intro t k h
      simp only [leafAdjSumR]
      have hab : (comp σ a t).2.val.size ≤ (comp σ b (comp σ a t).2).2.val.size := comp_size_le σ b _
      have hbtop : (comp σ b (comp σ a t).2).2.val.size ≤ (comp σ (.max a b) t).2.val.size := by
        simp only [comp, pushT, Array.size_push]; exact Nat.le_succ _
      have hta : t.val.size ≤ (comp σ a t).2.val.size := comp_size_le σ a t
      rw [iha t k (fun j hj1 hj2 => h j hj1 (lt_of_lt_of_le (lt_of_lt_of_le hj2 hab) hbtop))]
      rw [ihb (comp σ a t).2 k (fun j hj1 hj2 => h j (le_trans hta hj1) (lt_of_lt_of_le hj2 hbtop))]
  | min a b iha ihb =>
      intro t k h
      simp only [leafAdjSumR]
      have hab : (comp σ a t).2.val.size ≤ (comp σ b (comp σ a t).2).2.val.size := comp_size_le σ b _
      have hbtop : (comp σ b (comp σ a t).2).2.val.size ≤ (comp σ (.min a b) t).2.val.size := by
        simp only [comp, pushT, Array.size_push]; exact Nat.le_succ _
      have hta : t.val.size ≤ (comp σ a t).2.val.size := comp_size_le σ a t
      rw [iha t k (fun j hj1 hj2 => h j hj1 (lt_of_lt_of_le (lt_of_lt_of_le hj2 hab) hbtop))]
      rw [ihb (comp σ a t).2 k (fun j hj1 hj2 => h j (le_trans hta hj1) (lt_of_lt_of_le hj2 hbtop))]

namespace Central

theorem comp_edges_range (σ : Nat → Float) (e : Expr) (t : Tape) (hd : t.deps.size = t.val.size) :
    ∀ idx, t.val.size ≤ idx → idx < (comp σ e t).2.val.size →
      ∀ ed ∈ (comp σ e t).2.deps[idx]!, t.val.size ≤ ed.1 ∧ ed.1 < idx := by
  intro idx hlo hhi ed hed
  have hle : idx ≤ (comp σ e t).1 :=
    Nat.lt_succ_iff.mp (show idx < (comp σ e t).1 + 1 by rw [comp_root_succ]; exact hhi)
  exact _root_.Puffer.FloatR.ADReverse.comp_edges_range σ e t idx hd hlo hle ed hed

theorem leafAdjSumR_congr (σ : Nat → Float) (e : Expr) (t : Tape) (k : Nat)
    (adj adj' : Array ℝ)
    (h : ∀ j, t.val.size ≤ j → j < (comp σ e t).2.val.size → adj[j]! = adj'[j]!) :
    leafAdjSumR σ adj e t k = leafAdjSumR σ adj' e t k :=
  _root_.Puffer.FloatR.ADReverse.leafAdjSumR_congr σ adj adj' e t k h

abbrev CentralP (σ : Nat → Float) (e : Expr) (k : Nat) : Prop :=
  ∀ (t : Tape) (adj : Array ℝ) (ā : ℝ),
    t.deps.size = t.val.size →
    (comp σ e t).2.val.size ≤ adj.size →
    adj[(comp σ e t).1]! = ā →
    (∀ j, t.val.size ≤ j → j < (comp σ e t).1 → adj[j]! = 0) →
    (leafAdjSumR σ (bsweepR (comp σ e t).2 adj t.val.size ((comp σ e t).1 + 1)) e t k = revE_RF e σ ā k)
    ∧ (∀ j, j < t.val.size → (bsweepR (comp σ e t).2 adj t.val.size ((comp σ e t).1 + 1))[j]! = adj[j]!)

theorem unary_case (σ : Nat → Float) (a : Expr) (op : Expr) (w : Float) (k : Nat)
    (iha : CentralP σ a k)
    (t : Tape) (adj : Array ℝ) (ā : ℝ)
    (hdeps : t.deps.size = t.val.size)
    (hsz : (comp σ op t).2.val.size ≤ adj.size)
    (hroot : adj[(comp σ op t).1]! = ā)
    (hclean : ∀ j, t.val.size ≤ j → j < (comp σ op t).1 → adj[j]! = 0)
    (hop_root : (comp σ op t).1 = (comp σ a t).2.val.size)
    (hop_deps : (comp σ op t).2.deps[(comp σ a t).2.val.size]! = #[((comp σ a t).1, w)])
    (hop_sz : (comp σ op t).2.val.size = (comp σ a t).2.val.size + 1)
    (hop_congr : ∀ idx, idx < (comp σ a t).2.val.size →
      (comp σ op t).2.deps[idx]! = (comp σ a t).2.deps[idx]!)
    (hop_leaf : ∀ X : Array ℝ, leafAdjSumR σ X op t k = leafAdjSumR σ X a t k)
    (hop_rev : revE_RF op σ ā k = revE_RF a σ (ā * toReal w) k) :
    (leafAdjSumR σ (bsweepR (comp σ op t).2 adj t.val.size ((comp σ op t).1 + 1)) op t k = revE_RF op σ ā k)
    ∧ (∀ j, j < t.val.size → (bsweepR (comp σ op t).2 adj t.val.size ((comp σ op t).1 + 1))[j]! = adj[j]!) := by
  have hle : t.val.size ≤ (comp σ a t).2.val.size := comp_size_le σ a t
  have hrasucc : (comp σ a t).1 + 1 = (comp σ a t).2.val.size := comp_root_succ σ a t
  have hra_lt : (comp σ a t).1 < (comp σ a t).2.val.size := comp_root_lt σ a t
  have hra_ge : t.val.size ≤ (comp σ a t).1 := comp_root_ge σ a t
  have hrasz : (comp σ a t).2.val.size < adj.size := by rw [hop_sz] at hsz; omega
  have hra1_sz : (comp σ a t).1 < adj.size := lt_trans hra_lt hrasz
  have hclean_ra1 : adj[(comp σ a t).1]! = 0 := hclean (comp σ a t).1 hra_ge (by rw [hop_root]; exact hra_lt)
  have hroot' : adj[(comp σ a t).2.val.size]! = ā := by rw [← hop_root]; exact hroot
  rw [hop_root, bsweepR_succ _ _ _ _ hle]
  set adj1 := stepNodeR (comp σ op t).2 adj (comp σ a t).2.val.size with hadj1
  have hadj1_ra1 : adj1[(comp σ a t).1]! = ā * toReal w := by
    rw [hadj1, stepNodeR_single_at _ _ _ _ _ hop_deps hra1_sz, hclean_ra1, hroot', zero_add]
  have hadj1_ne : ∀ j, j ≠ (comp σ a t).1 → adj1[j]! = adj[j]! := by
    intro j hj
    rw [hadj1]; apply stepNodeR_preserve
    intro ed hed; rw [hop_deps] at hed; simp at hed; subst hed; exact hj.symm
  have hadj1_sz : adj1.size = adj.size := by rw [hadj1]; exact stepNodeR_size _ _ _
  rw [bsweepR_congr (comp σ op t).2 (comp σ a t).2 adj1 t.val.size (comp σ a t).2.val.size
      (fun idx _ hhi => hop_congr idx hhi), ← hrasucc]
  have hIH := iha t adj1 (ā * toReal w) hdeps (by rw [hadj1_sz]; exact le_of_lt hrasz)
    (by rw [hadj1_ra1]) (by intro j hj1 hj2
                            rw [hadj1_ne j (ne_of_lt hj2)]
                            exact hclean j hj1 (by rw [hop_root]; exact lt_trans hj2 hra_lt))
  obtain ⟨hIH1, hIH2⟩ := hIH
  refine ⟨?_, fun j hj => ?_⟩
  · rw [hop_leaf, hIH1, hop_rev]
  · rw [hIH2 j hj, hadj1_ne j (ne_of_lt (lt_of_lt_of_le hj hra_ge))]

theorem binary_case (σ : Nat → Float) (a b : Expr) (op : Expr) (wa wb : Float) (k : Nat)
    (iha : CentralP σ a k) (ihb : CentralP σ b k)
    (t : Tape) (adj : Array ℝ) (ā : ℝ)
    (hdeps : t.deps.size = t.val.size)
    (hsz : (comp σ op t).2.val.size ≤ adj.size)
    (hroot : adj[(comp σ op t).1]! = ā)
    (hclean : ∀ j, t.val.size ≤ j → j < (comp σ op t).1 → adj[j]! = 0)
    (hop_root : (comp σ op t).1 = (comp σ b (comp σ a t).2).2.val.size)
    (hop_deps : (comp σ op t).2.deps[(comp σ b (comp σ a t).2).2.val.size]! =
      #[((comp σ a t).1, wa), ((comp σ b (comp σ a t).2).1, wb)])
    (hop_sz : (comp σ op t).2.val.size = (comp σ b (comp σ a t).2).2.val.size + 1)
    (hop_congr : ∀ idx, idx < (comp σ b (comp σ a t).2).2.val.size →
      (comp σ op t).2.deps[idx]! = (comp σ b (comp σ a t).2).2.deps[idx]!)
    (hop_leaf : ∀ X : Array ℝ, leafAdjSumR σ X op t k =
      leafAdjSumR σ X a t k + leafAdjSumR σ X b (comp σ a t).2 k)
    (hop_rev : revE_RF op σ ā k =
      revE_RF a σ (ā * toReal wa) k + revE_RF b σ (ā * toReal wb) k) :
    (leafAdjSumR σ (bsweepR (comp σ op t).2 adj t.val.size ((comp σ op t).1 + 1)) op t k = revE_RF op σ ā k)
    ∧ (∀ j, j < t.val.size → (bsweepR (comp σ op t).2 adj t.val.size ((comp σ op t).1 + 1))[j]! = adj[j]!) := by
  have hdepA : (comp σ a t).2.deps.size = (comp σ a t).2.val.size := comp_deps_size σ a t hdeps
  have hdepB : (comp σ b (comp σ a t).2).2.deps.size = (comp σ b (comp σ a t).2).2.val.size :=
    comp_deps_size σ b (comp σ a t).2 hdepA
  have hAlt : (comp σ a t).1 < (comp σ a t).2.val.size := comp_root_lt σ a t
  have hAge : t.val.size ≤ (comp σ a t).1 := comp_root_ge σ a t
  have hAsucc : (comp σ a t).1 + 1 = (comp σ a t).2.val.size := comp_root_succ σ a t
  have hBlt : (comp σ b (comp σ a t).2).1 < (comp σ b (comp σ a t).2).2.val.size := comp_root_lt σ b (comp σ a t).2
  have hBge : (comp σ a t).2.val.size ≤ (comp σ b (comp σ a t).2).1 := comp_root_ge σ b (comp σ a t).2
  have hBsucc : (comp σ b (comp σ a t).2).1 + 1 = (comp σ b (comp σ a t).2).2.val.size := comp_root_succ σ b (comp σ a t).2
  have hAmid : t.val.size ≤ (comp σ a t).2.val.size := comp_size_le σ a t
  have hBmid : (comp σ a t).2.val.size ≤ (comp σ b (comp σ a t).2).2.val.size := comp_size_le σ b (comp σ a t).2
  have hpq : (comp σ a t).1 ≠ (comp σ b (comp σ a t).2).1 := ne_of_lt (lt_of_lt_of_le hAlt hBge)
  have hBsz : (comp σ b (comp σ a t).2).2.val.size < adj.size := by rw [hop_sz] at hsz; omega
  have hp_sz : (comp σ a t).1 < adj.size := lt_of_lt_of_le hAlt (le_trans hBmid (le_of_lt hBsz))
  have hq_sz : (comp σ b (comp σ a t).2).1 < adj.size := lt_trans hBlt hBsz
  have hp_clean : adj[(comp σ a t).1]! = 0 :=
    hclean _ hAge (by rw [hop_root]; exact lt_of_lt_of_le hAlt hBmid)
  have hq_clean : adj[(comp σ b (comp σ a t).2).1]! = 0 :=
    hclean _ (le_trans hAge (le_trans (le_of_lt hAlt) hBge)) (by rw [hop_root]; exact hBlt)
  have hroot' : adj[(comp σ b (comp σ a t).2).2.val.size]! = ā := by rw [← hop_root]; exact hroot
  rw [hop_root, bsweepR_succ _ _ _ _ (le_trans hAmid hBmid)]
  set adj1 := stepNodeR (comp σ op t).2 adj (comp σ b (comp σ a t).2).2.val.size with hadj1
  have hadj1_p : adj1[(comp σ a t).1]! = ā * toReal wa := by
    rw [hadj1, stepNodeR_two_at_p _ _ _ _ _ _ _ hop_deps hpq hp_sz, hp_clean, hroot', zero_add]
  have hadj1_q : adj1[(comp σ b (comp σ a t).2).1]! = ā * toReal wb := by
    rw [hadj1, stepNodeR_two_at_q _ _ _ _ _ _ _ hop_deps hq_sz,
        set!_getElem!_ne_gen _ _ _ _ hpq, hq_clean, hroot', zero_add]
  have hadj1_ne : ∀ j, j ≠ (comp σ a t).1 → j ≠ (comp σ b (comp σ a t).2).1 → adj1[j]! = adj[j]! := by
    intro j hjp hjq
    rw [hadj1]; apply stepNodeR_preserve
    intro ed hed; rw [hop_deps] at hed
    simp only [Array.mem_def, List.mem_cons] at hed
    rcases hed with h | h | h
    · rw [h]; exact fun hh => hjp hh.symm
    · rw [h]; exact fun hh => hjq hh.symm
    · exact absurd h (List.not_mem_nil)
  have hadj1_sz : adj1.size = adj.size := by rw [hadj1]; exact stepNodeR_size _ _ _
  rw [bsweepR_split (comp σ op t).2 adj1 t.val.size (comp σ a t).2.val.size
      (comp σ b (comp σ a t).2).2.val.size hAmid hBmid]
  rw [bsweepR_congr (comp σ op t).2 (comp σ b (comp σ a t).2).2 adj1
      (comp σ a t).2.val.size (comp σ b (comp σ a t).2).2.val.size
      (fun idx _ hhi => hop_congr idx hhi)]
  rw [show (comp σ b (comp σ a t).2).2.val.size = (comp σ b (comp σ a t).2).1 + 1 from hBsucc.symm]
  have hIHb := ihb (comp σ a t).2 adj1 (ā * toReal wb) hdepA
    (by rw [hadj1_sz]; exact le_of_lt hBsz) hadj1_q
    (by intro j hj1 hj2
        rw [hadj1_ne j (ne_of_gt (lt_of_lt_of_le hAlt hj1)) (ne_of_lt hj2)]
        exact hclean j (le_trans hAge (le_trans (le_of_lt hAlt) hj1))
          (by rw [hop_root]; exact lt_trans hj2 hBlt))
  obtain ⟨hIHb1, hIHb2⟩ := hIHb
  set adj2 := bsweepR (comp σ b (comp σ a t).2).2 adj1 (comp σ a t).2.val.size ((comp σ b (comp σ a t).2).1 + 1) with hadj2
  rw [bsweepR_congr (comp σ op t).2 (comp σ a t).2 adj2 t.val.size (comp σ a t).2.val.size
      (by intro idx _ hhi
          have hstep1 : (comp σ op t).2.deps[idx]! = (comp σ b (comp σ a t).2).2.deps[idx]! :=
            hop_congr idx (lt_of_lt_of_le hhi hBmid)
          rw [hstep1]
          exact comp_preserve_deps σ b (comp σ a t).2 idx (by rw [hdepA]; exact hhi))]
  rw [show (comp σ a t).2.val.size = (comp σ a t).1 + 1 from hAsucc.symm]
  have hadj2_p : adj2[(comp σ a t).1]! = ā * toReal wa := by
    rw [hIHb2 (comp σ a t).1 hAlt, hadj1_p]
  have hadj2_clean : ∀ j, t.val.size ≤ j → j < (comp σ a t).1 → adj2[j]! = 0 := by
    intro j hj1 hj2
    rw [hIHb2 j (lt_trans hj2 hAlt),
        hadj1_ne j (ne_of_lt hj2) (ne_of_lt (lt_of_lt_of_le hj2 (le_trans (le_of_lt hAlt) hBge)))]
    exact hclean j hj1 (by rw [hop_root]; exact lt_of_lt_of_le hj2 (le_trans (le_of_lt hAlt) (le_trans hBge (le_of_lt hBlt))))
  have hadj2_sz : adj2.size = adj.size := by rw [hadj2, bsweepR_size]; exact hadj1_sz
  have hIHa := iha t adj2 (ā * toReal wa) hdeps (by rw [hadj2_sz]; exact le_of_lt (lt_of_le_of_lt hBmid hBsz)) hadj2_p hadj2_clean
  obtain ⟨hIHa1, hIHa2⟩ := hIHa
  set adj3 := bsweepR (comp σ a t).2 adj2 t.val.size ((comp σ a t).1 + 1) with hadj3
  refine ⟨?_, fun j hj => ?_⟩
  · rw [hop_leaf, hop_rev, hIHa1]
    congr 1
    rw [← hIHb1]
    apply leafAdjSumR_congr
    intro j hj1 hj2
    rw [hadj3]; apply bsweepR_preserve
    intro idx hidxlo hidxhi ed hed
    have hidxhi' : idx < (comp σ a t).2.val.size := by rw [← hAsucc]; exact hidxhi
    have hedge := comp_edges_range σ a t hdeps idx hidxlo hidxhi' ed hed
    exact ne_of_lt (lt_of_lt_of_le (lt_of_lt_of_le hedge.2 (le_of_lt hidxhi')) hj1)
  · rw [hIHa2 j hj, hIHb2 j (lt_of_lt_of_le hj hAmid),
        hadj1_ne j (ne_of_lt (lt_of_lt_of_le hj hAge))
          (ne_of_lt (lt_of_lt_of_le hj (le_trans hAmid hBge)))]

end Central


open Central in
theorem central (σ : Nat → Float) (e : Expr) (k : Nat) :
    ∀ (t : Tape) (adj : Array ℝ) (ā : ℝ),
      t.deps.size = t.val.size →
      (comp σ e t).2.val.size ≤ adj.size →
      adj[(comp σ e t).1]! = ā →
      (∀ j, t.val.size ≤ j → j < (comp σ e t).1 → adj[j]! = 0) →
      (leafAdjSumR σ (bsweepR (comp σ e t).2 adj t.val.size ((comp σ e t).1 + 1)) e t k = revE_RF e σ ā k)
      ∧ (∀ j, j < t.val.size → (bsweepR (comp σ e t).2 adj t.val.size ((comp σ e t).1 + 1))[j]! = adj[j]!) := by
  induction e with
  | var i =>
    intro t adj ā hdeps hsz hroot hclean
    have hroot_idx : (comp σ (Expr.var i) t).1 = t.val.size := by simp [comp]
    rw [hroot_idx, bsweepR_succ _ _ _ _ (le_refl t.val.size), bsweepR_self]
    have hdroot : (comp σ (Expr.var i) t).2.deps[t.val.size]! = #[] := by
      show (t.deps.push #[])[t.val.size]! = #[]
      rw [← hdeps]; exact push_getElem!_size _ _
    rw [stepNodeR_empty _ _ _ hdroot]
    refine ⟨?_, fun j hj => rfl⟩
    simp only [leafAdjSumR, revE_RF]
    rw [hroot_idx] at hroot
    rcases eq_or_ne i k with h | h
    · subst h; simp [hroot]
    · rw [if_neg h, if_neg (Ne.symm h)]
  | const c =>
    intro t adj ā hdeps hsz hroot hclean
    have hroot_idx : (comp σ (Expr.const c) t).1 = t.val.size := by simp [comp]
    rw [hroot_idx, bsweepR_succ _ _ _ _ (le_refl t.val.size), bsweepR_self]
    have hdroot : (comp σ (Expr.const c) t).2.deps[t.val.size]! = #[] := by
      show (t.deps.push #[])[t.val.size]! = #[]
      rw [← hdeps]; exact push_getElem!_size _ _
    rw [stepNodeR_empty _ _ _ hdroot]
    exact ⟨rfl, fun j hj => rfl⟩
  | scale c a iha =>
    intro t adj ā hdeps hsz hroot hclean
    have hdepA : (comp σ a t).2.deps.size = (comp σ a t).2.val.size := comp_deps_size σ a t hdeps
    refine unary_case σ a (Expr.scale c a) c k iha t adj ā hdeps hsz hroot hclean
      (by simp [comp]) ?_ (by simp [comp, pushT, Array.size_push]) ?_ (fun X => rfl) rfl
    · show ((comp σ a t).2.deps.push #[((comp σ a t).1, c)])[(comp σ a t).2.val.size]! = _
      rw [← hdepA]; exact push_getElem!_size _ _
    · intro idx hhi
      show ((comp σ a t).2.deps.push _)[idx]! = _
      exact push_getElem!_lt _ _ idx (by rw [hdepA]; exact hhi)
  | exp a iha =>
    intro t adj ā hdeps hsz hroot hclean
    have hdepA : (comp σ a t).2.deps.size = (comp σ a t).2.val.size := comp_deps_size σ a t hdeps
    refine unary_case σ a (Expr.exp a) (Float.exp (evalF a σ)) k iha t adj ā hdeps hsz hroot hclean
      (by simp [comp]) ?_ (by simp [comp, pushT, Array.size_push]) ?_ (fun X => rfl) rfl
    · show ((comp σ a t).2.deps.push #[((comp σ a t).1, Float.exp ((comp σ a t).2.val[(comp σ a t).1]!))])[(comp σ a t).2.val.size]! = _
      rw [← hdepA, push_getElem!_size, comp_root_val]
    · intro idx hhi
      show ((comp σ a t).2.deps.push _)[idx]! = _
      exact push_getElem!_lt _ _ idx (by rw [hdepA]; exact hhi)
  | log a iha =>
    intro t adj ā hdeps hsz hroot hclean
    have hdepA : (comp σ a t).2.deps.size = (comp σ a t).2.val.size := comp_deps_size σ a t hdeps
    refine unary_case σ a (Expr.log a) (1.0 / evalF a σ) k iha t adj ā hdeps hsz hroot hclean
      (by simp [comp]) ?_ (by simp [comp, pushT, Array.size_push]) ?_ (fun X => rfl) rfl
    · show ((comp σ a t).2.deps.push #[((comp σ a t).1, 1.0 / ((comp σ a t).2.val[(comp σ a t).1]!))])[(comp σ a t).2.val.size]! = _
      rw [← hdepA, push_getElem!_size, comp_root_val]
    · intro idx hhi
      show ((comp σ a t).2.deps.push _)[idx]! = _
      exact push_getElem!_lt _ _ idx (by rw [hdepA]; exact hhi)
  | relu a iha =>
    intro t adj ā hdeps hsz hroot hclean
    have hdepA : (comp σ a t).2.deps.size = (comp σ a t).2.val.size := comp_deps_size σ a t hdeps
    refine unary_case σ a (Expr.relu a) (if evalF a σ < 0.0 then 0.0 else 1.0) k iha t adj ā hdeps hsz hroot hclean
      (by simp [comp]) ?_ (by simp [comp, pushT, Array.size_push]) ?_ (fun X => rfl) rfl
    · show ((comp σ a t).2.deps.push #[((comp σ a t).1, if (comp σ a t).2.val[(comp σ a t).1]! < 0.0 then 0.0 else 1.0)])[(comp σ a t).2.val.size]! = _
      rw [← hdepA, push_getElem!_size, comp_root_val]
    · intro idx hhi
      show ((comp σ a t).2.deps.push _)[idx]! = _
      exact push_getElem!_lt _ _ idx (by rw [hdepA]; exact hhi)
  | add a b iha ihb =>
    intro t adj ā hdeps hsz hroot hclean
    have hdepA : (comp σ a t).2.deps.size = (comp σ a t).2.val.size := comp_deps_size σ a t hdeps
    have hdepB : (comp σ b (comp σ a t).2).2.deps.size = (comp σ b (comp σ a t).2).2.val.size :=
      comp_deps_size σ b (comp σ a t).2 hdepA
    refine binary_case σ a b (Expr.add a b) 1.0 1.0 k iha ihb t adj ā hdeps hsz hroot hclean
      (by simp [comp]) ?_ (by simp [comp, pushT, Array.size_push]) ?_ (fun X => rfl) rfl
    · show ((comp σ b (comp σ a t).2).2.deps.push _)[(comp σ b (comp σ a t).2).2.val.size]! = _
      rw [← hdepB]; exact push_getElem!_size _ _
    · intro idx hhi
      show ((comp σ b (comp σ a t).2).2.deps.push _)[idx]! = _
      exact push_getElem!_lt _ _ idx (by rw [hdepB]; exact hhi)
  | sub a b iha ihb =>
    intro t adj ā hdeps hsz hroot hclean
    have hdepA : (comp σ a t).2.deps.size = (comp σ a t).2.val.size := comp_deps_size σ a t hdeps
    have hdepB : (comp σ b (comp σ a t).2).2.deps.size = (comp σ b (comp σ a t).2).2.val.size :=
      comp_deps_size σ b (comp σ a t).2 hdepA
    refine binary_case σ a b (Expr.sub a b) 1.0 (-1.0) k iha ihb t adj ā hdeps hsz hroot hclean
      (by simp [comp]) ?_ (by simp [comp, pushT, Array.size_push]) ?_ (fun X => rfl) rfl
    · show ((comp σ b (comp σ a t).2).2.deps.push _)[(comp σ b (comp σ a t).2).2.val.size]! = _
      rw [← hdepB]; exact push_getElem!_size _ _
    · intro idx hhi
      show ((comp σ b (comp σ a t).2).2.deps.push _)[idx]! = _
      exact push_getElem!_lt _ _ idx (by rw [hdepB]; exact hhi)
  | mul a b iha ihb =>
    intro t adj ā hdeps hsz hroot hclean
    have hdepA : (comp σ a t).2.deps.size = (comp σ a t).2.val.size := comp_deps_size σ a t hdeps
    have hdepB : (comp σ b (comp σ a t).2).2.deps.size = (comp σ b (comp σ a t).2).2.val.size :=
      comp_deps_size σ b (comp σ a t).2 hdepA
    refine binary_case σ a b (Expr.mul a b) (evalF b σ) (evalF a σ) k iha ihb t adj ā hdeps hsz hroot hclean
      (by simp [comp]) ?_ (by simp [comp, pushT, Array.size_push]) ?_ (fun X => rfl) rfl
    · show ((comp σ b (comp σ a t).2).2.deps.push
        #[((comp σ a t).1, (comp σ b (comp σ a t).2).2.val[(comp σ b (comp σ a t).2).1]!),
          ((comp σ b (comp σ a t).2).1, (comp σ b (comp σ a t).2).2.val[(comp σ a t).1]!)])[(comp σ b (comp σ a t).2).2.val.size]! = _
      rw [← hdepB, push_getElem!_size, comp_root_val σ b (comp σ a t).2,
        comp_preserve σ b (comp σ a t).2 (comp σ a t).1 (comp_root_lt σ a t), comp_root_val σ a t]
    · intro idx hhi
      show ((comp σ b (comp σ a t).2).2.deps.push _)[idx]! = _
      exact push_getElem!_lt _ _ idx (by rw [hdepB]; exact hhi)
  | max a b iha ihb =>
    intro t adj ā hdeps hsz hroot hclean
    have hdepA : (comp σ a t).2.deps.size = (comp σ a t).2.val.size := comp_deps_size σ a t hdeps
    have hdepB : (comp σ b (comp σ a t).2).2.deps.size = (comp σ b (comp σ a t).2).2.val.size :=
      comp_deps_size σ b (comp σ a t).2 hdepA
    have hva : (comp σ b (comp σ a t).2).2.val[(comp σ a t).1]! = evalF a σ := by
      rw [comp_preserve σ b (comp σ a t).2 (comp σ a t).1 (comp_root_lt σ a t), comp_root_val σ a t]
    have hvb : (comp σ b (comp σ a t).2).2.val[(comp σ b (comp σ a t).2).1]! = evalF b σ :=
      comp_root_val σ b (comp σ a t).2
    refine binary_case σ a b (Expr.max a b)
      (if evalF a σ ≤ evalF b σ then 0.0 else 1.0) (if evalF a σ ≤ evalF b σ then 1.0 else 0.0)
      k iha ihb t adj ā hdeps hsz hroot hclean
      (by simp [comp]) ?_ (by simp [comp, pushT, Array.size_push]) ?_ (fun X => rfl) rfl
    · show ((comp σ b (comp σ a t).2).2.deps.push _)[(comp σ b (comp σ a t).2).2.val.size]! = _
      rw [← hdepB, push_getElem!_size, hva, hvb]
    · intro idx hhi
      show ((comp σ b (comp σ a t).2).2.deps.push _)[idx]! = _
      exact push_getElem!_lt _ _ idx (by rw [hdepB]; exact hhi)
  | min a b iha ihb =>
    intro t adj ā hdeps hsz hroot hclean
    have hdepA : (comp σ a t).2.deps.size = (comp σ a t).2.val.size := comp_deps_size σ a t hdeps
    have hdepB : (comp σ b (comp σ a t).2).2.deps.size = (comp σ b (comp σ a t).2).2.val.size :=
      comp_deps_size σ b (comp σ a t).2 hdepA
    have hva : (comp σ b (comp σ a t).2).2.val[(comp σ a t).1]! = evalF a σ := by
      rw [comp_preserve σ b (comp σ a t).2 (comp σ a t).1 (comp_root_lt σ a t), comp_root_val σ a t]
    have hvb : (comp σ b (comp σ a t).2).2.val[(comp σ b (comp σ a t).2).1]! = evalF b σ :=
      comp_root_val σ b (comp σ a t).2
    refine binary_case σ a b (Expr.min a b)
      (if evalF a σ ≤ evalF b σ then 1.0 else 0.0) (if evalF a σ ≤ evalF b σ then 0.0 else 1.0)
      k iha ihb t adj ā hdeps hsz hroot hclean
      (by simp [comp]) ?_ (by simp [comp, pushT, Array.size_push]) ?_ (fun X => rfl) rfl
    · show ((comp σ b (comp σ a t).2).2.deps.push _)[(comp σ b (comp σ a t).2).2.val.size]! = _
      rw [← hdepB, push_getElem!_size, hva, hvb]
    · intro idx hhi
      show ((comp σ b (comp σ a t).2).2.deps.push _)[idx]! = _
      exact push_getElem!_lt _ _ idx (by rw [hdepB]; exact hhi)

/-- **Flat-array reverse-mode readout (ℝ layer).** The exact-ℝ reverse sweep `bsweepR` — the ℝ image
    of `gradsF`'s identical fold (`gradsF = bsweep`, its Float twin, by `gradsF_eq_bsweep`; the only
    difference is exact-ℝ vs rounded arithmetic in the accumulator, the deferred Layer-2 εg bound) —
    run on the `comp`-built tape and read at the variable leaves (`leafAdjSumR`), equals the faithful
    reverse pass `revE_RF` seeded with output-adjoint `1`. So the flat sweep's STRUCTURE realizes the
    reverse pass. (This is over `bsweepR`, not the literal Float `gradsF`: the Float→ℝ rounding of the
    sweep is Layer 2, unproved here.) -/
theorem bsweepR_reads_revE_RF (σ : Nat → Float) (e : Expr) (k : Nat) :
    leafAdjSumR σ
      (bsweepR (comp σ e Tape.empty).2
        ((Array.replicate (comp σ e Tape.empty).2.val.size (0:ℝ)).set! (comp σ e Tape.empty).1 1)
        0 (comp σ e Tape.empty).2.val.size) e Tape.empty k
      = revE_RF e σ 1 k := by
  have hrs := comp_root_succ σ e Tape.empty
  have hrl := comp_root_lt σ e Tape.empty
  have hemp : (Tape.empty : Tape).val.size = 0 := rfl
  have hadj0sz :
      ((Array.replicate (comp σ e Tape.empty).2.val.size (0:ℝ)).set! (comp σ e Tape.empty).1 1).size
        = (comp σ e Tape.empty).2.val.size := by rw [Array.size_set!, Array.size_replicate]
  have hroot0 :
      ((Array.replicate (comp σ e Tape.empty).2.val.size (0:ℝ)).set! (comp σ e Tape.empty).1 1)[(comp σ e Tape.empty).1]!
        = (1:ℝ) := set!_getElem!_self _ _ 1 (by rw [Array.size_replicate]; exact hrl)
  have hclean0 : ∀ j, (0:Nat) ≤ j → j < (comp σ e Tape.empty).1 →
      ((Array.replicate (comp σ e Tape.empty).2.val.size (0:ℝ)).set! (comp σ e Tape.empty).1 1)[j]! = 0 := by
    intro j _ hj
    rw [set!_getElem!_ne_gen _ _ 1 j (Ne.symm (ne_of_lt hj))]
    rw [getElem!_pos _ j (by rw [Array.size_replicate]; exact lt_trans hj hrl), Array.getElem_replicate]
  have hc := (central σ e k Tape.empty
    ((Array.replicate (comp σ e Tape.empty).2.val.size (0:ℝ)).set! (comp σ e Tape.empty).1 1)
    1 rfl hadj0sz.ge hroot0 (by rw [hemp]; exact hclean0)).1
  rw [hemp, hrs] at hc
  exact hc

/-- **Reverse-sweep gradient sparsity / locality.** If the variable `k` does not syntactically occur in the
    expression `e`, then the exact-ℝ reverse sweep `bsweepR` — run on the `comp`-built tape (seeded with the output
    adjoint `1` at the root) and read back at the variable leaves via `leafAdjSumR` — assigns `k` exactly `0`. This
    is the reverse-mode (flat-sweep) analogue of the forward-mode `derivR_eq_zero_of_not_occurs`: the reverse-mode
    AD engine never manufactures a spurious gradient for a variable absent from the expression, no matter how the
    adjoints are accumulated through the tape's edges. The `¬ occurs k e` hypothesis is load-bearing — for
    `e = .var k` the leaf readout is the seed `1 ≠ 0`. Proved by composing the file's payoff
    `bsweepR_reads_revE_RF` (the flat sweep reads off the faithful reverse pass `revE_RF`) with an inlined
    structural induction showing `revE_RF` itself vanishes on absent variables at every seed `ā` (the `add`/`sub`/
    `mul`/`max`/`min` nodes split the non-occurrence via `not_or` and sum the two `0` sub-adjoints; the unary nodes
    thread the single IH; the `var i` leaf reads the seed only when `k = i`, excluded by `¬ occurs k (.var k)`; the
    `const` leaf is `0`). Note: this cannot reuse the downstream `revE_RF_eq_zero_of_not_occurs`
    (in `ADReverseBridge`, which imports THIS file), so the `revE_RF` vanishing is re-derived inline here. -/
theorem bsweepR_reads_eq_zero_of_not_occurs (σ : Nat → Float) (e : Expr) (k : Nat)
    (h : ¬ occurs k e) :
    leafAdjSumR σ
      (bsweepR (comp σ e Tape.empty).2
        ((Array.replicate (comp σ e Tape.empty).2.val.size (0:ℝ)).set! (comp σ e Tape.empty).1 1)
        0 (comp σ e Tape.empty).2.val.size) e Tape.empty k
      = 0 := by
  rw [bsweepR_reads_revE_RF]
  suffices H : ∀ (e' : Expr), ¬ occurs k e' → ∀ ā : ℝ, revE_RF e' σ ā k = 0 by
    exact H e h 1
  intro e'
  induction e' with
  | var i =>
      intro h' ā
      simp only [occurs] at h'
      simp only [revE_RF, if_neg (fun hh : k = i => h' hh.symm)]
  | const c => intro h' ā; simp only [revE_RF]
  | add a b iha ihb =>
      intro h' ā; simp only [occurs, not_or] at h'
      simp only [revE_RF, iha h'.1, ihb h'.2, add_zero]
  | sub a b iha ihb =>
      intro h' ā; simp only [occurs, not_or] at h'
      simp only [revE_RF, iha h'.1, ihb h'.2, add_zero]
  | mul a b iha ihb =>
      intro h' ā; simp only [occurs, not_or] at h'
      simp only [revE_RF, iha h'.1, ihb h'.2, add_zero]
  | scale c a iha =>
      intro h' ā; simp only [occurs] at h'
      simp only [revE_RF, iha h']
  | exp a iha =>
      intro h' ā; simp only [occurs] at h'
      simp only [revE_RF, iha h']
  | log a iha =>
      intro h' ā; simp only [occurs] at h'
      simp only [revE_RF, iha h']
  | relu a iha =>
      intro h' ā; simp only [occurs] at h'
      simp only [revE_RF, iha h']
  | max a b iha ihb =>
      intro h' ā; simp only [occurs, not_or] at h'
      simp only [revE_RF, iha h'.1, ihb h'.2, add_zero]
  | min a b iha ihb =>
      intro h' ā; simp only [occurs, not_or] at h'
      simp only [revE_RF, iha h'.1, ihb h'.2, add_zero]

/-- **The ℝ sweep reads off the true derivative — CONDITIONAL on the primal bridge.** Composing
    `bsweepR_reads_revE_RF` with `revE_R_eq_derivR` gives that the flat ℝ sweep, read at the leaves,
    equals `derivR` — the genuine gradient — provided `hbridge : revE_RF e σ 1 k = revE_R e (envR σ) 1 k`.
    That bridge equates the reverse pass using the tape's ROUNDED Float weights (`toReal (evalF b σ)`,
    `Float.exp`, the Float kink signs) with the one using the IDEAL real weights (`evalR b σ`,
    `Real.exp`, real signs); it holds exactly when the primal has no rounding and is otherwise the
    Layer-2 εg obligation (NOT proved here — it is generally false under rounding, hence an explicit
    hypothesis, not a theorem). So this states the reverse ALGORITHM is correct on the flat tape;
    quantifying the rounding is future work. -/
theorem bsweepR_reads_derivR (σ : Nat → Float) (e : Expr) (k : Nat)
    (hbridge : revE_RF e σ 1 k = revE_R e (envR σ) 1 k) :
    leafAdjSumR σ
      (bsweepR (comp σ e Tape.empty).2
        ((Array.replicate (comp σ e Tape.empty).2.val.size (0:ℝ)).set! (comp σ e Tape.empty).1 1)
        0 (comp σ e Tape.empty).2.val.size) e Tape.empty k
      = derivR e (envR σ) k := by
  rw [bsweepR_reads_revE_RF, hbridge, revE_R_eq_derivR]

/-! ### Reverse-sweep length invariant: `grads` produces one adjoint per tape node -/

/-- An `Id`-monad `forIn` over a list, whose body only `yield`s a size-preserving update of the accumulator
    array, preserves the array's size. -/
theorem list_forIn_id_size {α} (l : List α) (init : Array Float)
    (g : α → Array Float → Array Float) (hg : ∀ x a, (g x a).size = a.size) :
    (forIn (m := Id) l init (fun x a => pure (ForInStep.yield (g x a)))).size = init.size := by
  induction l generalizing init with
  | nil => rfl
  | cons x xs ih =>
    rw [List.forIn_cons]; simp only [pure_bind]; rw [ih (g x init)]; exact hg x init

/-- Array version of `list_forIn_id_size`. -/
theorem array_forIn_id_size {α} (xs : Array α) (init : Array Float)
    (g : α → Array Float → Array Float) (hg : ∀ x a, (g x a).size = a.size) :
    (forIn (m := Id) xs init (fun x a => pure (ForInStep.yield (g x a)))).size = init.size := by
  rw [← Array.forIn_toList]; exact list_forIn_id_size xs.toList init g hg

/-- Range version of `list_forIn_id_size`. -/
theorem range_forIn_id_size (rg : Std.Legacy.Range) (init : Array Float)
    (g : Nat → Array Float → Array Float) (hg : ∀ x a, (g x a).size = a.size) :
    (forIn (m := Id) rg init (fun x a => pure (ForInStep.yield (g x a)))).size = init.size := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  exact list_forIn_id_size _ init g hg

/-- **Reverse-sweep alignment.** `grads` returns exactly one gradient per tape node: the adjoint array it
    produces has the same length as the tape's value array. Its nested imperative `forIn` sweep only ever
    `Array.set!`s into the adjoint accumulator (initialized to `Array.replicate t.val.size 0.0`), and `set!`
    preserves size — so no reverse pass can grow or shrink the gradient vector. This is the structural
    precondition the per-leaf gradient reads `g[i]!` of `grads`/`adGrad` consumers rely on. -/
theorem grads_size (t : Tape) (root : V) : (grads t root).size = t.val.size := by
  unfold grads
  simp only [Id.run, bind_pure_comp, map_pure, bind_pure]
  refine (range_forIn_id_size _ _
    (fun i r => forIn (m := Id) t.deps[t.val.size - 1 - i]! r
      (fun e r_1 => pure (ForInStep.yield (r_1.set! e.1 (r_1[e.1]! + r[t.val.size - 1 - i]! * e.2))))) ?_).trans ?_
  · intro i a
    exact array_forIn_id_size _ a _ (fun e r_1 => by rw [Array.size_set!])
  · rw [Array.size_set!, Array.size_replicate]

end Puffer.FloatR.ADReverse
