import Puffer.Check.Core

/-!
# C86: tail-recursive checker core (stack-safe at n ≈ 10⁶)

C78 and C81 disclosed one runtime caveat about the checker/evaluator layer:
the recursive evaluators are *not* tail-recursive, so at extreme input sizes
(a 10⁶-step trace, a 10⁶-term dot budget) the recursion depth equals the
input length.  This module closes that caveat: for every `Puffer.Check.Core`
function whose recursion depth grows with an **unbounded input axis** (trace
length, list length, or a `Nat` iteration count), it provides a `T`-suffixed
variant whose recursive call is in syntactic tail position — which the Lean
compiler turns into a loop — together with a **proved equality** to the
original, so every bridge-transferred soundness theorem applies to the
`T`-variant verbatim after one rewrite.

**Float discipline.** Float `+`/`*`/`max` satisfy no provable algebraic laws,
so the `T`-variants are *not* reassociated folds: each accumulator carries the
fully-evaluated prefix and performs **exactly the original's operations, in
the original order and association**.  Every equality below is therefore pure
structural induction (plus `Nat` index arithmetic via `omega`) with **zero
Float algebra** — and, operationally, a `T`-variant computes the bit-identical
Float, not merely a provably-equal one.

Covered (variant + equality theorem):
* `allT` / `allT_eq` — generic tail-recursive `List.all`, the engine for the
  three list-shaped checkers;
* `checkRegionT`, `checkTraceT`, `runTraceChecksT` (+ the wholesale Bool
  corollary `runTraceChecksT_allOk`); `runTraceChecksT` returns the *same*
  `VerifyReport` type, so `formatReport`/callers switch with no other change;
* `dotBoundFT` — the C78 dot budget (original recursion is genuinely
  non-tail: the call sits inside the slacked arithmetic);
* `edgeBoundFT`, `nodeBoundFT`, `sweepBoundFT` — the C80 sweep family
  (`edgeBoundF` is the non-tail culprit; `sweepBoundF`'s outer loop is
  already a tail call but performs a depth-`E` `edgeBoundF` per iteration,
  so `sweepBoundFT` swaps in the loop variant);
* `runBoundFT` — the original is *already* syntactically tail-recursive;
  the variant exists only so the `T` family is surface-complete.

Deliberately NOT covered (disclosed, not forced):
* `fwdBoundF` / `compWeightBoundF` recurse on **`Expr` tree depth**, bounded
  by the fixed expression grammar of the differentiated program — a different
  axis than trace length, and not unbounded for any fixed training setup;
* the arity-fixed evaluators (`stepBoundF`, `wdUniformBoundF`, the C81
  run-constant mirrors, the C70 scalar checkers, `absF`) do not recurse.

Honest operational note: the Lean compiler emits loops for self tail calls,
so the `T`-variants run at n ≈ 10⁶ in the compiled `puffer` binary.  The
originals' `List.all`/`&&` shape is *typically* also loop-compiled after
`Bool.and` inlining — the point of the `T` family is that stack safety
becomes a *syntactic* property of the code rather than a fact about the
inliner.  Import discipline: this file imports only `Puffer.Check.Core`
(Mathlib-free closure `Core → Expr → ErrBnd → Exec → ∅`), so it may be linked
into the exe.  Zero axioms, zero `sorry`, zero `native_decide`.
-/

namespace Puffer.Check

/-! ### Generic tail-recursive conjunction -/

/-- Tail-recursive `List.all`: the recursive call is in syntactic tail
    position (a `bif` branch), so the compiler emits a loop. -/
def allT {α : Type _} (p : α → Bool) : List α → Bool
  | [] => true
  | a :: l => bif p a then allT p l else false

/-- `allT` agrees with `List.all` — a Bool-only induction, no Float facts. -/
theorem allT_eq {α : Type _} (p : α → Bool) : (l : List α) → allT p l = l.all p
  | [] => rfl
  | a :: l => by
    rw [List.all_cons]
    cases h : p a with
    | true => simpa [allT, h] using allT_eq p l
    | false => simp [allT, h]

/-! ### C70/C73 checkers, tail form -/

/-- Tail-recursive `checkRegion`. -/
def checkRegionT (θrow : List Float) (R : Float) : Bool :=
  allT (fun x => checkAbsLe x R) θrow

theorem checkRegionT_eq (θrow : List Float) (R : Float) :
    checkRegionT θrow R = checkRegion θrow R :=
  allT_eq _ θrow

/-- Tail-recursive `checkTrace`; the per-step region check is also the loop
    variant, so the whole check runs in constant stack. -/
def checkTraceT (tr : Trace) (R tLo tHi : Float) : Bool :=
  allT (fun s => checkRegionT s.1 R && checkClipMargin s.2 tLo tHi) tr

theorem checkTraceT_eq (tr : Trace) (R tLo tHi : Float) :
    checkTraceT tr R tLo tHi = checkTrace tr R tLo tHi := by
  unfold checkTraceT checkTrace
  rw [allT_eq]
  congr 1
  funext s
  rw [checkRegionT_eq]

/-! ### C78 dot budget, tail form

`dotBoundF B (n+1) = slackF * (slackF * (B * B) + dotBoundF B n)` iterates one
fixed Float map from `0.0`, so the accumulator variant applies the *same* map
to the running value — same operations, same order, same association. -/

/-- Accumulator loop for `dotBoundFT`. -/
def dotBoundFT.go (B : Float) : Nat → Float → Float
  | 0, acc => acc
  | n + 1, acc => dotBoundFT.go B n (slackF * (slackF * (B * B) + acc))

/-- Tail-recursive `dotBoundF`. -/
def dotBoundFT (B : Float) (n : Nat) : Float := dotBoundFT.go B n 0.0

/-- Accumulator invariant: running the loop `n` more times from the depth-`m`
    value lands on the depth-`m + n` value. -/
theorem dotBoundFT.go_spec (B : Float) :
    (n m : Nat) → dotBoundFT.go B n (dotBoundF B m) = dotBoundF B (m + n)
  | 0, _ => rfl
  | n + 1, m => by
    show dotBoundFT.go B n (slackF * (slackF * (B * B) + dotBoundF B m)) = _
    have h : slackF * (slackF * (B * B) + dotBoundF B m) = dotBoundF B (m + 1) := rfl
    rw [h, dotBoundFT.go_spec B n (m + 1)]
    congr 1
    omega

theorem dotBoundFT_eq (B : Float) (n : Nat) : dotBoundFT B n = dotBoundF B n := by
  have h0 : (0.0 : Float) = dotBoundF B 0 := rfl
  show dotBoundFT.go B n 0.0 = dotBoundF B n
  rw [h0, dotBoundFT.go_spec B n 0, Nat.zero_add]

/-! ### C80 sweep family, tail form -/

/-- Accumulator loop for `edgeBoundFT` (the seed is the accumulator's initial
    value, so only `A` and `D` are loop-invariant). -/
def edgeBoundFT.go (A D : Float) : Nat → Float → Float
  | 0, acc => acc
  | k + 1, acc => edgeBoundFT.go A D k (slackF * (acc + slackF * (A * D)))

/-- Tail-recursive `edgeBoundF`. -/
def edgeBoundFT (A D C : Float) (k : Nat) : Float := edgeBoundFT.go A D k C

theorem edgeBoundFT.go_spec (A D C : Float) :
    (n m : Nat) → edgeBoundFT.go A D n (edgeBoundF A D C m) = edgeBoundF A D C (m + n)
  | 0, _ => rfl
  | n + 1, m => by
    show edgeBoundFT.go A D n
      (slackF * (edgeBoundF A D C m + slackF * (A * D))) = _
    have h : slackF * (edgeBoundF A D C m + slackF * (A * D))
        = edgeBoundF A D C (m + 1) := rfl
    rw [h, edgeBoundFT.go_spec A D C n (m + 1)]
    congr 1
    omega

theorem edgeBoundFT_eq (A D C : Float) (k : Nat) :
    edgeBoundFT A D C k = edgeBoundF A D C k := by
  have h := edgeBoundFT.go_spec A D C k 0
  rw [Nat.zero_add] at h
  exact h

/-- Tail-form `nodeBoundF`: one constant-stack `edgeBoundFT` loop. -/
def nodeBoundFT (E : Nat) (D C : Float) : Float := edgeBoundFT C D C E

theorem nodeBoundFT_eq (E : Nat) (D C : Float) :
    nodeBoundFT E D C = nodeBoundF E D C :=
  edgeBoundFT_eq C D C E

/-- Tail-form `sweepBoundF`: the outer node loop was already a tail call; the
    depth risk was the per-node `edgeBoundF`, now the loop variant. -/
def sweepBoundFT (E : Nat) (D : Float) : Nat → Float → Float
  | 0, C => C
  | m + 1, C => sweepBoundFT E D m (nodeBoundFT E D C)

theorem sweepBoundFT_eq (E : Nat) (D : Float) :
    (m : Nat) → (C : Float) → sweepBoundFT E D m C = sweepBoundF E D m C
  | 0, _ => rfl
  | m + 1, C => by
    show sweepBoundFT E D m (nodeBoundFT E D C) = _
    rw [nodeBoundFT_eq, sweepBoundFT_eq E D m (nodeBoundF E D C)]
    rfl

/-! ### C78 run budget, tail form (surface uniformity only) -/

/-- `runBoundF` is already syntactically tail-recursive; this variant exists
    so callers can switch to the `T` family wholesale. -/
def runBoundFT (C : Float) : Nat → Float → Float
  | 0, B => B
  | n + 1, B => runBoundFT C n (stepBoundF C B)

theorem runBoundFT_eq (C : Float) :
    (n : Nat) → (B : Float) → runBoundFT C n B = runBoundF C n B
  | 0, _ => rfl
  | n + 1, B => by
    show runBoundFT C n (stepBoundF C B) = _
    rw [runBoundFT_eq C n (stepBoundF C B)]
    rfl

/-! ### C82 report surface, tail form -/

/-- Tail-recursive `runTraceChecks`.  Returns the *same* `VerifyReport`
    structure, so `formatReport` and the exe wiring consume it unchanged. -/
def runTraceChecksT (tr : Trace) (R tLo tHi : Float)
    (budgets : List (Float × Float)) : VerifyReport :=
  { regionMarginOk := checkTraceT tr R tLo tHi
    budgetsOk := allT (fun bc => checkLe bc.1 bc.2) budgets }

theorem runTraceChecksT_eq (tr : Trace) (R tLo tHi : Float)
    (budgets : List (Float × Float)) :
    runTraceChecksT tr R tLo tHi budgets = runTraceChecks tr R tLo tHi budgets := by
  unfold runTraceChecksT runTraceChecks
  rw [checkTraceT_eq, allT_eq]

/-- The wholesale switch: the tail-form aggregate Bool *is* the verified
    aggregate Bool, so every bridge-transferred theorem consuming
    `(runTraceChecks …).allOk = true` (in particular C82's
    `allOk_feeds_whole_run`) applies to `runTraceChecksT` after this rewrite. -/
theorem runTraceChecksT_allOk (tr : Trace) (R tLo tHi : Float)
    (budgets : List (Float × Float)) :
    (runTraceChecksT tr R tLo tHi budgets).allOk
      = (runTraceChecks tr R tLo tHi budgets).allOk := by
  rw [runTraceChecksT_eq]

/-! ### Build-asserted demos: T-variant = original on concrete inputs

The equalities are theorems; these `#eval`s additionally confirm the compiled
code paths produce bit-identical Floats (`==` on non-NaN values). -/

-- Deep dot budget: loop variant matches the recursive original at depth 512.
/--
info: true
-/
#guard_msgs in
#eval dotBoundFT 1.0 512 == dotBoundF 1.0 512

-- Sweep family: 32 nodes × 8 edges, loop variant matches the original.
/--
info: true
-/
#guard_msgs in
#eval sweepBoundFT 8 1.5 32 1.0 == sweepBoundF 8 1.5 32 1.0

-- Run budget: 256 slacked steps, loop variant matches the original.
/--
info: true
-/
#guard_msgs in
#eval runBoundFT 0.5 256 1.0 == runBoundF 0.5 256 1.0

-- The C82 demo aggregate, recomputed through the tail-form report: same
-- passing verdict as `Core.lean`'s build-asserted demo.
/--
info: true
-/
#guard_msgs in
#eval (runTraceChecksT [([0.5, -0.25], 0.75), ([0.9, 0.1], 0.5)]
  1.0 0.25 0.8 [(1.0, capF), (2048.0, capF)]).allOk

-- And the failing case still fails (ratio 0.9 breaches the [0.25, 0.8] margin).
/--
info: false
-/
#guard_msgs in
#eval (runTraceChecksT [([0.5, -0.25], 0.9)] 1.0 0.25 0.8 []).allOk

end Puffer.Check
