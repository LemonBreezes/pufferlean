/-
The REVERSE-mode composed per-weight update bound — closing the `εg` gap for the DEFAULT trainer.

`UpdateADBound` closed the SGD-step `εg` gap for the *forward-mode* AD (`dF`). The trainer, however,
runs the *reverse-mode* tape engine (`AutoDiff.grads`, an imperative adjoint sweep). This file closes
the same gap for that engine, composing the two proven halves of the reverse-mode equivalence:

  • **Float→ℝ (Layer 2)** — `ADReverseError.gradsF_error`: the literal Float sweep `gradsF t root` is
    entrywise within `sweepBnd` of the exact-ℝ sweep `bsweepR` (`AdjBnd`).
  • **ℝ-structure (Stage C)** — `ADReverse.bsweepR_reads_derivR`: that ℝ sweep, read at the variable-`k`
    leaves (`leafAdjSumR`), equals the symbolic derivative `derivR e (envR σ) k` (given the primal
    bridge `hbridge : revE_RF = revE_R`).

The missing joint was a `leafAdjSumR`-level error lemma bridging the *entrywise* array bound to the
*leaf-sum* readout. `leafAdjSumR_error` supplies it: two arrays entrywise within `ε` give leaf-sums
within `(leafCount e k)·ε` (the number of `var k` leaves, by structural induction). Composing:

  • `reverseGrad_error`    — `|revGrad − derivR e (envR σ) k| ≤ (leafCount e k)·sweepBnd`
  • `reverseGrad_deriv_error` — same, against the TRUE derivative `∂(evalR e)/∂(var k)` (`derivR_eq_deriv`)
  • `reverseAxpyStep_error` — feed that gradient error into `axpyStep_error`: one SGD step on any Float
    gradient readout `g` realizing the reverse leaf-sum is within a proven bound of the ideal real ascent.

`revGrad` is the ℝ image of the reverse-mode Float gradient read at the `var k` leaves. The bound is
conditional on the two disclosed Layer-2 hypotheses (`hedges`: tape edge weights bounded by `W`;
`hbridge`: rounded primal = ideal primal — exact absent primal rounding) and, for the true-derivative
form, `PosR` (every `log` argument `> 0`). `sweepBnd` is astronomically loose (reverse adjoints are
products of edge weights) but a genuine, closed, machine-checked bound. Axiom-clean beyond the Float base.
-/
import Puffer.Float.ADReverseError
import Puffer.Float.ADReverseErrorTight
import Puffer.Float.ADReverseBridge
import Puffer.RL.UpdateRuntime

namespace Puffer.FloatR.ADReverse

open Puffer.FloatR
open Puffer.FloatR.ADR (Expr evalR derivR envR derivR_eq_deriv PosR WD occurs)
open Puffer.FloatR.AD (Tape V)
open Puffer.FloatR.ADReverseError (AdjBnd sweepBnd gradsF_error)
open Puffer.FloatR.ADReverseErrorTight (AdjBndF sweepBndF gradsF_errorF sweepBndF_snd_nonneg)

/-- Number of `var k` leaves in an expression — the count of array reads `leafAdjSumR` performs for
    variable `k`. Mirrors `leafAdjSumR`'s recursion (structure only; independent of the tape/assignment). -/
def leafCount : Expr → Nat → Nat
  | .var i, k => if i = k then 1 else 0
  | .const _, _ => 0
  | .add a b, k => leafCount a k + leafCount b k
  | .sub a b, k => leafCount a k + leafCount b k
  | .mul a b, k => leafCount a k + leafCount b k
  | .scale _ a, k => leafCount a k
  | .exp a, k => leafCount a k
  | .log a, k => leafCount a k
  | .relu a, k => leafCount a k
  | .max a b, k => leafCount a k + leafCount b k
  | .min a b, k => leafCount a k + leafCount b k

/-- **Leaf-sum error from entrywise error.** If two adjoint arrays are entrywise within `ε` (at every
    index), their `leafAdjSumR` readouts for variable `k` differ by at most `(leafCount e k)·ε`. Pure
    structural induction on `e` (binary nodes add via the triangle inequality; unary nodes pass through;
    a `var k` leaf contributes one `ε`, other leaves zero). -/
theorem leafAdjSumR_error (σ : Nat → Float) (adj1 adj2 : Array ℝ) (ε : ℝ) (hε : 0 ≤ ε)
    (h : ∀ (j : Nat), |adj1[j]! - adj2[j]!| ≤ ε) :
    ∀ (e : Expr) (t : Tape) (k : Nat),
      |leafAdjSumR σ adj1 e t k - leafAdjSumR σ adj2 e t k| ≤ (leafCount e k : ℝ) * ε := by
  -- the binary-node step, factored out (add/sub/mul/max/min all read the same recursion shape)
  have binstep : ∀ (la1 la2 lb1 lb2 : ℝ) (ca cb : Nat),
      |la1 - la2| ≤ (ca : ℝ) * ε → |lb1 - lb2| ≤ (cb : ℝ) * ε →
      |(la1 + lb1) - (la2 + lb2)| ≤ ((ca + cb : Nat) : ℝ) * ε := by
    intro la1 la2 lb1 lb2 ca cb h1 h2
    calc |(la1 + lb1) - (la2 + lb2)|
        = |(la1 - la2) + (lb1 - lb2)| := by rw [show (la1 + lb1) - (la2 + lb2)
            = (la1 - la2) + (lb1 - lb2) from by ring]
      _ ≤ |la1 - la2| + |lb1 - lb2| := abs_add_le _ _
      _ ≤ (ca : ℝ) * ε + (cb : ℝ) * ε := add_le_add h1 h2
      _ = ((ca + cb : Nat) : ℝ) * ε := by push_cast; ring
  intro e
  induction e with
  | var i =>
      intro t k
      by_cases hik : i = k
      · simp only [leafAdjSumR, leafCount, hik, if_true, Nat.cast_one, one_mul]
        exact h t.val.size
      · simp only [leafAdjSumR, leafCount, hik, if_false, Nat.cast_zero, zero_mul,
          sub_self, abs_zero, le_refl]
  | const c => intro t k; simp only [leafAdjSumR, leafCount, sub_self, abs_zero, Nat.cast_zero,
      zero_mul, le_refl]
  | add a b iha ihb => intro t k
                       simpa only [leafAdjSumR, leafCount] using
                         binstep _ _ _ _ _ _ (iha t k) (ihb (comp σ a t).2 k)
  | sub a b iha ihb => intro t k
                       simpa only [leafAdjSumR, leafCount] using
                         binstep _ _ _ _ _ _ (iha t k) (ihb (comp σ a t).2 k)
  | mul a b iha ihb => intro t k
                       simpa only [leafAdjSumR, leafCount] using
                         binstep _ _ _ _ _ _ (iha t k) (ihb (comp σ a t).2 k)
  | scale c a iha => intro t k; simpa only [leafAdjSumR, leafCount] using iha t k
  | exp a iha => intro t k; simpa only [leafAdjSumR, leafCount] using iha t k
  | log a iha => intro t k; simpa only [leafAdjSumR, leafCount] using iha t k
  | relu a iha => intro t k; simpa only [leafAdjSumR, leafCount] using iha t k
  | max a b iha ihb => intro t k
                       simpa only [leafAdjSumR, leafCount] using
                         binstep _ _ _ _ _ _ (iha t k) (ihb (comp σ a t).2 k)
  | min a b iha ihb => intro t k
                       simpa only [leafAdjSumR, leafCount] using
                         binstep _ _ _ _ _ _ (iha t k) (ihb (comp σ a t).2 k)

/-- `sweepBnd`'s error component is nonnegative (seeded at `(1,0)`, each `edgeBnd` step preserves
    nonnegativity via `edgeBnd_nonneg`). -/
theorem sweepBnd_snd_nonneg (t : Tape) (W : ℝ) (hW : 0 ≤ W) (lo hi : Nat) :
    0 ≤ (sweepBnd t W lo hi (1, 0)).2 := by
  unfold sweepBnd
  generalize (((List.range (hi - lo)).map fun i => hi - 1 - i)) = idxs
  suffices H : ∀ (p : ℝ × ℝ), 0 ≤ p.1 → 0 ≤ p.2 →
      0 ≤ (idxs.foldl (fun p idx =>
        Puffer.FloatR.ADReverseError.edgeBnd W (t.deps[idx]!).toList p) p).2 by
    exact H (1, 0) (by norm_num) (le_refl 0)
  intro p
  induction idxs generalizing p with
  | nil => intro _ h2; exact h2
  | cons i is ih =>
      intro h1 h2
      obtain ⟨hM', hε'⟩ := Puffer.FloatR.ADReverseError.edgeBnd_nonneg W hW
        (t.deps[i]!).toList p.1 p.2 h1 h2
      exact ih _ hM' hε'

/-- The reverse-mode Float gradient read at the `var k` leaves (ℝ image): sum over the `var k` leaves of
    `toReal` of the reverse sweep's adjoint. This is what the trainer's reverse engine yields for weight `k`. -/
noncomputable def revGrad (σ : Nat → Float) (e : Expr) (k : Nat) : ℝ :=
  leafAdjSumR σ ((gradsF (comp σ e Tape.empty).2 (comp σ e Tape.empty).1).map toReal) e Tape.empty k

/-- **Reverse-mode gradient sparsity/locality (Float readout).** If variable `k` does not occur
    syntactically in `e`, then the reverse-mode Float gradient read at the `var k` leaves is EXACTLY zero
    — unconditionally (no edge-weight bound `hedges`, no primal bridge, no `PosR`): for an absent variable
    the reverse engine returns `0` on the nose, not merely within `revGradBnd`. This is the Float-readout
    analog of `derivR_eq_zero_of_not_occurs` (ideal ℝ derivative, a203) and `revE_RF_eq_zero_of_not_occurs`
    (tape reverse pass, a225), completing that locality family down to the actual gradient the trainer emits.
    Proved by a structural induction whose core — `leafAdjSumR σ adj e t k = 0` for ANY adjoint array `adj`
    — holds because `¬ occurs k e` means there is no `var k` leaf to read, so every summand is gated to `0`.
    The hypothesis is load-bearing: `revGrad σ (var k) k` reads adjoint `1` at the seed leaf, so it is
    `≠ 0` when `k` DOES occur. -/
theorem revGrad_eq_zero_of_not_occurs (σ : Nat → Float) (e : Expr) (k : Nat)
    (h : ¬ occurs k e) : revGrad σ e k = 0 := by
  -- The general fact: the leaf-sum readout of any adjoint array vanishes off the support of `k`.
  have key : ∀ (adj : Array ℝ) (e : Expr), ¬ occurs k e →
      ∀ (t : Tape), leafAdjSumR σ adj e t k = 0 := by
    intro adj e
    induction e with
    | var i => intro h t; simp only [occurs] at h; simp only [leafAdjSumR, if_neg h]
    | const c => intro _ t; simp only [leafAdjSumR]
    | add a b iha ihb =>
        intro h t; simp only [occurs, not_or] at h
        simp only [leafAdjSumR, iha h.1 t, ihb h.2 (comp σ a t).2, add_zero]
    | sub a b iha ihb =>
        intro h t; simp only [occurs, not_or] at h
        simp only [leafAdjSumR, iha h.1 t, ihb h.2 (comp σ a t).2, add_zero]
    | mul a b iha ihb =>
        intro h t; simp only [occurs, not_or] at h
        simp only [leafAdjSumR, iha h.1 t, ihb h.2 (comp σ a t).2, add_zero]
    | scale c a iha => intro h t; simp only [occurs] at h; simp only [leafAdjSumR, iha h t]
    | exp a iha => intro h t; simp only [occurs] at h; simp only [leafAdjSumR, iha h t]
    | log a iha => intro h t; simp only [occurs] at h; simp only [leafAdjSumR, iha h t]
    | relu a iha => intro h t; simp only [occurs] at h; simp only [leafAdjSumR, iha h t]
    | max a b iha ihb =>
        intro h t; simp only [occurs, not_or] at h
        simp only [leafAdjSumR, iha h.1 t, ihb h.2 (comp σ a t).2, add_zero]
    | min a b iha ihb =>
        intro h t; simp only [occurs, not_or] at h
        simp only [leafAdjSumR, iha h.1 t, ihb h.2 (comp σ a t).2, add_zero]
  simp only [revGrad]
  exact key _ e h Tape.empty

/-- The certified reverse-mode gradient error: `(leafCount e k)·sweepBnd`. -/
noncomputable def revGradBnd (σ : Nat → Float) (e : Expr) (k : Nat) (W : ℝ) : ℝ :=
  (leafCount e k : ℝ)
    * (sweepBnd (comp σ e Tape.empty).2 W 0 (comp σ e Tape.empty).2.val.size (1, 0)).2

/-- **The reverse-mode gradient is within a proven bound of `derivR`.** Composing `gradsF_error`
    (Float sweep entrywise within `sweepBnd` of `bsweepR`), `leafAdjSumR_error` (entrywise → leaf-sum),
    and `bsweepR_reads_derivR` (the ℝ sweep reads `derivR`). `hedges`/`hbridge` are the disclosed Layer-2
    hypotheses. -/
theorem reverseGrad_error (σ : Nat → Float) (e : Expr) (k : Nat) (W : ℝ) (hW : 0 ≤ W)
    (hedges : ∀ idx, idx < (comp σ e Tape.empty).2.val.size →
      ∀ ed ∈ (comp σ e Tape.empty).2.deps[idx]!,
        ed.1 < (comp σ e Tape.empty).2.val.size ∧ |toReal ed.2| ≤ W)
    (hbridge : revE_RF e σ 1 k = revE_R e (envR σ) 1 k) :
    |revGrad σ e k - derivR e (envR σ) k| ≤ revGradBnd σ e k W := by
  simp only [revGrad, revGradBnd]
  set T := (comp σ e Tape.empty).2 with hT
  set R := (comp σ e Tape.empty).1 with hR
  set ε := (sweepBnd T W 0 T.val.size (1, 0)).2 with hεdef
  have hroot : R < T.val.size := comp_root_lt σ e Tape.empty
  have hAdj : AdjBnd (gradsF T R)
      (bsweepR T ((Array.replicate T.val.size (0:ℝ)).set! R 1) 0 T.val.size)
      (sweepBnd T W 0 T.val.size (1, 0)).1 ε :=
    gradsF_error T R W hW hroot hedges
  have hε : 0 ≤ ε := sweepBnd_snd_nonneg T W hW 0 T.val.size
  set arrR := bsweepR T ((Array.replicate T.val.size (0:ℝ)).set! R 1) 0 T.val.size with harrR
  have hsize : (gradsF T R).size = arrR.size := hAdj.1
  have hentry : ∀ (j : Nat), |((gradsF T R).map toReal)[j]! - arrR[j]!| ≤ ε := by
    intro j
    by_cases hj : j < (gradsF T R).size
    · have hmap : ((gradsF T R).map toReal)[j]! = toReal ((gradsF T R)[j]!) := by
        rw [getElem!_pos ((gradsF T R).map toReal) j (by rw [Array.size_map]; exact hj),
          Array.getElem_map, getElem!_pos (gradsF T R) j hj]
      rw [hmap]; exact (hAdj.2 j hj).2
    · rw [getElem!_neg ((gradsF T R).map toReal) j (by rw [Array.size_map]; exact hj),
        getElem!_neg arrR j (by rw [← hsize]; exact hj), sub_self, abs_zero]
      exact hε
  have hleaf := leafAdjSumR_error σ ((gradsF T R).map toReal) arrR ε hε hentry e Tape.empty k
  have hreads : leafAdjSumR σ arrR e Tape.empty k = derivR e (envR σ) k := by
    rw [harrR, hT, hR]; exact bsweepR_reads_derivR σ e k hbridge
  rw [hreads] at hleaf
  exact hleaf

/-- **The reverse-mode gradient is within a proven bound of the TRUE derivative** `∂(evalR e)/∂(var k)`.
    `reverseGrad_error` + `derivR_eq_deriv` (needs `PosR`). -/
theorem reverseGrad_deriv_error (σ : Nat → Float) (e : Expr) (k : Nat) (W : ℝ) (hW : 0 ≤ W)
    (hedges : ∀ idx, idx < (comp σ e Tape.empty).2.val.size →
      ∀ ed ∈ (comp σ e Tape.empty).2.deps[idx]!,
        ed.1 < (comp σ e Tape.empty).2.val.size ∧ |toReal ed.2| ≤ W)
    (hbridge : revE_RF e σ 1 k = revE_R e (envR σ) 1 k) (hp : PosR e (envR σ)) :
    |revGrad σ e k - deriv (fun t => evalR e (Function.update (envR σ) k t)) (envR σ k)|
      ≤ revGradBnd σ e k W := by
  have h := reverseGrad_error σ e k W hW hedges hbridge
  rwa [derivR_eq_deriv e (envR σ) k hp] at h

/-- **The reverse-mode composed SGD-step bound.** For any Float gradient readout `g` realizing the reverse
    leaf-sum (`hgeq`; exact when weight `k` occurs once), one gradient-ascent update `p + lr·g` is within
    `u64·|…| + u64·|lr·g| + |lr|·revGradBnd` of the ideal real ascent step against the true derivative —
    the reverse-mode analog of `UpdateADBound.axpyStep_AD_error`, discharging `axpyStep_error`'s `εg`. -/
theorem reverseAxpyStep_error (σ : Nat → Float) (e : Expr) (k : Nat) (W : ℝ) (hW : 0 ≤ W)
    (hedges : ∀ idx, idx < (comp σ e Tape.empty).2.val.size →
      ∀ ed ∈ (comp σ e Tape.empty).2.deps[idx]!,
        ed.1 < (comp σ e Tape.empty).2.val.size ∧ |toReal ed.2| ≤ W)
    (hbridge : revE_RF e σ 1 k = revE_R e (envR σ) 1 k) (hp : PosR e (envR σ))
    (p lr g : Float) (hgeq : toReal g = revGrad σ e k) :
    |toReal (p + lr * g)
        - (toReal p + toReal lr
            * deriv (fun t => evalR e (Function.update (envR σ) k t)) (envR σ k))|
      ≤ u64 * |toReal p + toReal (lr * g)|
          + (u64 * |toReal lr * toReal g| + |toReal lr| * revGradBnd σ e k W) := by
  have hg : |toReal g - deriv (fun t => evalR e (Function.update (envR σ) k t)) (envR σ k)|
      ≤ revGradBnd σ e k W := by
    rw [hgeq]; exact reverseGrad_deriv_error σ e k W hW hedges hbridge hp
  exact axpyStep_error p lr g
    (deriv (fun t => evalR e (Function.update (envR σ) k t)) (envR σ k)) (revGradBnd σ e k W) hg

/-! ### The TIGHT (per-node) reverse-mode gradient bound

`reverseGrad_error` above uses the uniform `sweepBnd` (astronomically loose). Here the same result is proved
against the per-node `ADReverseErrorTight.sweepBndF` — the tight tree/path bound. `leafErrSum` sums the
per-index error `εf` only over variable `k`'s leaves (mirroring `leafAdjSumR`), so `revGradBndF` accumulates
only along each leaf's root-path rather than compounding over the whole tape. -/

/-- Per-index leaf error accumulator: sums the per-index error `εf` at the `var k` leaves (mirrors
    `leafAdjSumR`'s recursion). -/
noncomputable def leafErrSum (σ : Nat → Float) (εf : Nat → ℝ) : Expr → Tape → Nat → ℝ
  | .var i, t, k => if i = k then εf t.val.size else 0
  | .const _, _, _ => 0
  | .add a b, t, k => leafErrSum σ εf a t k + leafErrSum σ εf b (comp σ a t).2 k
  | .sub a b, t, k => leafErrSum σ εf a t k + leafErrSum σ εf b (comp σ a t).2 k
  | .mul a b, t, k => leafErrSum σ εf a t k + leafErrSum σ εf b (comp σ a t).2 k
  | .scale _ a, t, k => leafErrSum σ εf a t k
  | .exp a, t, k => leafErrSum σ εf a t k
  | .log a, t, k => leafErrSum σ εf a t k
  | .relu a, t, k => leafErrSum σ εf a t k
  | .max a b, t, k => leafErrSum σ εf a t k + leafErrSum σ εf b (comp σ a t).2 k
  | .min a b, t, k => leafErrSum σ εf a t k + leafErrSum σ εf b (comp σ a t).2 k

/-- **Per-index leaf-sum error.** Two arrays within a PER-INDEX error `εf j` (at every index) give
    `leafAdjSumR` readouts within `leafErrSum σ εf e t k` (the sum of `εf` at the `var k` leaves). -/
theorem leafAdjSumR_errorF (σ : Nat → Float) (adj1 adj2 : Array ℝ) (εf : Nat → ℝ)
    (h : ∀ (j : Nat), |adj1[j]! - adj2[j]!| ≤ εf j) :
    ∀ (e : Expr) (t : Tape) (k : Nat),
      |leafAdjSumR σ adj1 e t k - leafAdjSumR σ adj2 e t k| ≤ leafErrSum σ εf e t k := by
  have binstep : ∀ (la1 la2 lb1 lb2 Ba Bb : ℝ),
      |la1 - la2| ≤ Ba → |lb1 - lb2| ≤ Bb → |(la1 + lb1) - (la2 + lb2)| ≤ Ba + Bb := by
    intro la1 la2 lb1 lb2 Ba Bb h1 h2
    calc |(la1 + lb1) - (la2 + lb2)|
        = |(la1 - la2) + (lb1 - lb2)| := by rw [show (la1 + lb1) - (la2 + lb2)
            = (la1 - la2) + (lb1 - lb2) from by ring]
      _ ≤ |la1 - la2| + |lb1 - lb2| := abs_add_le _ _
      _ ≤ Ba + Bb := add_le_add h1 h2
  intro e
  induction e with
  | var i => intro t k
             by_cases hik : i = k
             · simp only [leafAdjSumR, leafErrSum, hik, if_true]; exact h t.val.size
             · simp only [leafAdjSumR, leafErrSum, hik, if_false, sub_self, abs_zero, le_refl]
  | const c => intro t k; simp only [leafAdjSumR, leafErrSum, sub_self, abs_zero, le_refl]
  | add a b iha ihb => intro t k; simp only [leafAdjSumR, leafErrSum]
                       exact binstep _ _ _ _ _ _ (iha t k) (ihb (comp σ a t).2 k)
  | sub a b iha ihb => intro t k; simp only [leafAdjSumR, leafErrSum]
                       exact binstep _ _ _ _ _ _ (iha t k) (ihb (comp σ a t).2 k)
  | mul a b iha ihb => intro t k; simp only [leafAdjSumR, leafErrSum]
                       exact binstep _ _ _ _ _ _ (iha t k) (ihb (comp σ a t).2 k)
  | scale c a iha => intro t k; simpa only [leafAdjSumR, leafErrSum] using iha t k
  | exp a iha => intro t k; simpa only [leafAdjSumR, leafErrSum] using iha t k
  | log a iha => intro t k; simpa only [leafAdjSumR, leafErrSum] using iha t k
  | relu a iha => intro t k; simpa only [leafAdjSumR, leafErrSum] using iha t k
  | max a b iha ihb => intro t k; simp only [leafAdjSumR, leafErrSum]
                       exact binstep _ _ _ _ _ _ (iha t k) (ihb (comp σ a t).2 k)
  | min a b iha ihb => intro t k; simp only [leafAdjSumR, leafErrSum]
                       exact binstep _ _ _ _ _ _ (iha t k) (ihb (comp σ a t).2 k)

/-- The TIGHT reverse-mode gradient error: the per-index `sweepBndF` error summed over `var k`'s leaves. -/
noncomputable def revGradBndF (σ : Nat → Float) (e : Expr) (k : Nat) (W : ℝ) : ℝ :=
  leafErrSum σ (sweepBndF (comp σ e Tape.empty).2 W 0 (comp σ e Tape.empty).2.val.size
    ((fun j => if j = (comp σ e Tape.empty).1 then 1 else 0), (fun _ => 0))).2 e Tape.empty k

/-- **The Float reverse gradient vs the exact-ℝ reverse pass `revE_RF`.** The pure Float↔ℝ SWEEP-rounding
    error, bounded by the tight per-index `revGradBndF` — UNCONDITIONAL (no `hbridge`, no `WD`): it does not
    yet reach `derivR`, only the tape-weighted reverse pass `revE_RF e σ 1 k`. The shared core of the exact
    and bounded gradient bounds. -/
theorem reverseGrad_vs_revERF (σ : Nat → Float) (e : Expr) (k : Nat) (W : ℝ) (hW : 0 ≤ W)
    (hedges : ∀ idx, idx < (comp σ e Tape.empty).2.val.size →
      ∀ ed ∈ (comp σ e Tape.empty).2.deps[idx]!,
        ed.1 < (comp σ e Tape.empty).2.val.size ∧ |toReal ed.2| ≤ W) :
    |revGrad σ e k - revE_RF e σ 1 k| ≤ revGradBndF σ e k W := by
  simp only [revGrad, revGradBndF]
  set T := (comp σ e Tape.empty).2 with hT
  set R := (comp σ e Tape.empty).1 with hR
  set εf := (sweepBndF T W 0 T.val.size ((fun j => if j = R then (1:ℝ) else 0), (fun _ => 0))).2 with hεf
  have hroot : R < T.val.size := comp_root_lt σ e Tape.empty
  have hAdj : AdjBndF (gradsF T R)
      (bsweepR T ((Array.replicate T.val.size (0:ℝ)).set! R 1) 0 T.val.size)
      (sweepBndF T W 0 T.val.size ((fun j => if j = R then (1:ℝ) else 0), (fun _ => 0))).1 εf :=
    gradsF_errorF T R W hW hroot hedges
  have hεnn : ∀ j, 0 ≤ εf j :=
    sweepBndF_snd_nonneg T W 0 T.val.size hW ((fun j => if j = R then (1:ℝ) else 0), (fun _ => 0))
      (fun j => by dsimp only; split <;> norm_num) (fun _ => le_refl 0)
  set arrR := bsweepR T ((Array.replicate T.val.size (0:ℝ)).set! R 1) 0 T.val.size with harrR
  have hsize : (gradsF T R).size = arrR.size := hAdj.1
  have hentry : ∀ (j : Nat), |((gradsF T R).map toReal)[j]! - arrR[j]!| ≤ εf j := by
    intro j
    by_cases hj : j < (gradsF T R).size
    · have hmap : ((gradsF T R).map toReal)[j]! = toReal ((gradsF T R)[j]!) := by
        rw [getElem!_pos ((gradsF T R).map toReal) j (by rw [Array.size_map]; exact hj),
          Array.getElem_map, getElem!_pos (gradsF T R) j hj]
      rw [hmap]; exact (hAdj.2 j hj).2
    · rw [getElem!_neg ((gradsF T R).map toReal) j (by rw [Array.size_map]; exact hj),
        getElem!_neg arrR j (by rw [← hsize]; exact hj), sub_self, abs_zero]
      exact hεnn j
  have hleaf := leafAdjSumR_errorF σ ((gradsF T R).map toReal) arrR εf hentry e Tape.empty k
  have hreads : leafAdjSumR σ arrR e Tape.empty k = revE_RF e σ 1 k := by
    rw [harrR, hT, hR]; exact bsweepR_reads_revE_RF σ e k
  rw [hreads] at hleaf
  exact hleaf

/-- **The reverse-mode gradient is within the TIGHT per-node bound of `derivR`.** Same as
    `reverseGrad_error` but against `revGradBndF` (per-index `sweepBndF`); uses the exact primal bridge. -/
theorem reverseGrad_errorF (σ : Nat → Float) (e : Expr) (k : Nat) (W : ℝ) (hW : 0 ≤ W)
    (hedges : ∀ idx, idx < (comp σ e Tape.empty).2.val.size →
      ∀ ed ∈ (comp σ e Tape.empty).2.deps[idx]!,
        ed.1 < (comp σ e Tape.empty).2.val.size ∧ |toReal ed.2| ≤ W)
    (hbridge : revE_RF e σ 1 k = revE_R e (envR σ) 1 k) :
    |revGrad σ e k - derivR e (envR σ) k| ≤ revGradBndF σ e k W := by
  have h := reverseGrad_vs_revERF σ e k W hW hedges
  rwa [hbridge, revE_R_eq_derivR] at h

/-- **The FULLY-QUANTIFIED reverse-mode gradient bound — NO opaque `hbridge`.** Combining the tight sweep
    error (`reverseGrad_vs_revERF`) with the BOUNDED primal bridge (`bridge_gap_le`, item a57), the Float
    reverse gradient is within `revGradBndF + bridgeBnd` of the TRUE `derivR` under `WD` alone — the sweep
    rounding PLUS the primal rounding, both quantified, with the last opaque assumption gone. -/
theorem reverseGrad_errorF_bounded (σ : Nat → Float) (e : Expr) (k : Nat) (W : ℝ) (hW : 0 ≤ W)
    (hedges : ∀ idx, idx < (comp σ e Tape.empty).2.val.size →
      ∀ ed ∈ (comp σ e Tape.empty).2.deps[idx]!,
        ed.1 < (comp σ e Tape.empty).2.val.size ∧ |toReal ed.2| ≤ W)
    (hwd : WD e σ) :
    |revGrad σ e k - derivR e (envR σ) k| ≤ revGradBndF σ e k W + bridgeBnd e σ k := by
  calc |revGrad σ e k - derivR e (envR σ) k|
      ≤ |revGrad σ e k - revE_RF e σ 1 k| + |revE_RF e σ 1 k - derivR e (envR σ) k| :=
        abs_sub_le _ _ _
    _ ≤ revGradBndF σ e k W + bridgeBnd e σ k :=
        add_le_add (reverseGrad_vs_revERF σ e k W hW hedges) (bridge_gap_le e σ k hwd)

/-! ### `hbridge` discharged: reverse-mode gradient bound with NO opaque bridge hypothesis

`ADReverseBridge` discharges `hbridge` into concrete conditions — unconditional on the linear fragment
(`IsLinear`), or the per-node primal-exactness/kink-agreement `BridgeExact` for all ops. These corollaries
carry that in, so the reverse-mode gradient bound no longer takes the opaque `revE_RF = revE_R` equality. -/

/-- **Reverse-mode gradient bound, linear objectives — no bridge hypothesis.** For a linear objective
    (`IsLinear e`: linear layers/readouts) `hbridge` holds unconditionally, so the only remaining
    side-conditions are the edge-weight bound `hedges` and its well-formedness `hW : 0 ≤ W` — no opaque
    reverse-pass assumption survives. -/
theorem reverseGrad_error_of_linear (σ : Nat → Float) (e : Expr) (k : Nat) (W : ℝ) (hW : 0 ≤ W)
    (hedges : ∀ idx, idx < (comp σ e Tape.empty).2.val.size →
      ∀ ed ∈ (comp σ e Tape.empty).2.deps[idx]!,
        ed.1 < (comp σ e Tape.empty).2.val.size ∧ |toReal ed.2| ≤ W)
    (hlin : IsLinear e) :
    |revGrad σ e k - derivR e (envR σ) k| ≤ revGradBnd σ e k W :=
  reverseGrad_error σ e k W hW hedges (bridge_of_linear e σ k hlin)

/-- **Reverse-mode gradient bound from the per-node `BridgeExact` conditions** — replaces the opaque
    `hbridge` with concrete primal-exactness/kink-agreement facts (all ops). -/
theorem reverseGrad_error_of_bridgeExact (σ : Nat → Float) (e : Expr) (k : Nat) (W : ℝ) (hW : 0 ≤ W)
    (hedges : ∀ idx, idx < (comp σ e Tape.empty).2.val.size →
      ∀ ed ∈ (comp σ e Tape.empty).2.deps[idx]!,
        ed.1 < (comp σ e Tape.empty).2.val.size ∧ |toReal ed.2| ≤ W)
    (hBE : BridgeExact e σ) :
    |revGrad σ e k - derivR e (envR σ) k| ≤ revGradBnd σ e k W :=
  reverseGrad_error σ e k W hW hedges (bridge_of_bridgeExact e σ k hBE)

/-- **Reverse-mode gradient vs the TRUE derivative, linear objectives — no bridge hypothesis.** -/
theorem reverseGrad_deriv_error_of_linear (σ : Nat → Float) (e : Expr) (k : Nat) (W : ℝ) (hW : 0 ≤ W)
    (hedges : ∀ idx, idx < (comp σ e Tape.empty).2.val.size →
      ∀ ed ∈ (comp σ e Tape.empty).2.deps[idx]!,
        ed.1 < (comp σ e Tape.empty).2.val.size ∧ |toReal ed.2| ≤ W)
    (hlin : IsLinear e) (hp : PosR e (envR σ)) :
    |revGrad σ e k - deriv (fun t => evalR e (Function.update (envR σ) k t)) (envR σ k)|
      ≤ revGradBnd σ e k W :=
  reverseGrad_deriv_error σ e k W hW hedges (bridge_of_linear e σ k hlin) hp

end Puffer.FloatR.ADReverse
