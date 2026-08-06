/-
Connecting the RUNNABLE softmax to its bounded functional model.

`SoftmaxBound.softmaxFval` is the faithful functional model of a numerically-stable softmax, and
`softmaxFval_error_spec` bounds it against the canonical ℝ softmax `Puffer.Net.softmax`. But the trainer
actually calls `Puffer.RL.Train.softmax` — an `Id.run do` block that folds/maps over honest `Array Float`s
(`m = maxⱼ lⱼ`, `exps = map (exp (·−m))`, `z = foldl (+) 0`, output `map (·/z)`). This file closes that last
gap: it proves the executable's `i`-th output *equals* the functional model, so the whole softmax error
bound transfers verbatim to the code that runs.

The bridge is pure structural plumbing (no new floating-point math):

  • `arrFoldl_add_eq_sumIdxFGo` — an `Array.foldl (·+·) 0` equals `SoftmaxBound.sumIdxFGo` over `List.range`
    (via `Array.foldl_toList`, a generic `listFoldl_eq_rangeFoldl` index-reindexing, and `toList_getElem!`).
  • `sDenom_eq` — the runnable denominator `z` is exactly the model's `sumIdxFGo` sum of shifted exponentials.
  • `train_softmax_getElem!` — `(Train.softmax logits)[i]! = softmaxFval (logits[·]!) (max) logits.size i`
    for every in-range `i` (`Id.run`/`Array.map` unfolding + `arrMapGetElem!`).
  • `train_softmax_error` — the composed Float↔ℝ bound, now on the ACTUAL runnable output, against
    `Puffer.Net.softmax`. Denominator floor `dmin ≤ |toReal (sDenom logits)|` (the sum of positive
    exponentials, `≳ 1`) supplied as a hypothesis, as in `softmaxFval_error_spec`.

Axiom-clean beyond the trusted Float (1+δ) base — no `native_decide`, no `sorry`.
-/
import Puffer.RL.SoftmaxBound
import Puffer.RL.Train

namespace Puffer.RL.SoftmaxExec

open Puffer.FloatR
open Puffer.RL.SoftmaxBound

/-! ### Generic `Array.foldl`/`List.foldl` reindexing plumbing -/

/-- `sumIdxFGo` is literally the `List.foldl` of rounded adds. -/
theorem sumIdxFGo_eq_foldl (e : Nat → Float) (acc : Float) (is : List Nat) :
    sumIdxFGo e acc is = is.foldl (fun acc i => acc + e i) acc := by
  induction is generalizing acc with
  | nil => rfl
  | cons i is ih => simp only [sumIdxFGo]; rw [ih]; rfl

/-- A `List.foldl` over `l` equals the same fold reindexed over `List.range l.length` with `l[k]!`. -/
theorem listFoldl_eq_rangeFoldl {α β} [Inhabited α] (F : β → α → β) :
    ∀ (l : List α) (init : β),
      l.foldl F init = (List.range l.length).foldl (fun acc k => F acc l[k]!) init := by
  intro l
  induction l with
  | nil => intro init; simp
  | cons x xs ih =>
      intro init
      simp only [List.length_cons, List.range_succ_eq_map, List.foldl_cons, List.foldl_map,
        List.getElem!_cons_succ, List.getElem!_cons_zero]
      exact ih (F init x)

/-- `getElem!` commutes with `Array.toList` (both `default` out of range). -/
theorem toList_getElem! (a : Array Float) (k : Nat) : a.toList[k]! = a[k]! := by
  by_cases hk : k < a.size
  · rw [getElem!_pos a k hk, getElem!_pos a.toList k (by simpa using hk), Array.getElem_toList]
  · rw [getElem!_neg a k hk, getElem!_neg a.toList k (by simpa using hk)]

/-- `sumIdxFGo` depends on the summand function only at the list's indices. -/
theorem sumIdxFGo_congr (e1 e2 : Nat → Float) (acc : Float) (is : List Nat)
    (h : ∀ k ∈ is, e1 k = e2 k) : sumIdxFGo e1 acc is = sumIdxFGo e2 acc is := by
  induction is generalizing acc with
  | nil => rfl
  | cons i is ih =>
      simp only [sumIdxFGo]
      rw [h i (List.mem_cons_self)]
      exact ih (acc + e2 i) (fun k hk => h k (List.mem_cons_of_mem _ hk))

/-- **An honest `Array.foldl (·+·) 0` is a `sumIdxFGo` over `List.range`.** The bridge from the runnable
    array fold to the functional summation model. -/
theorem arrFoldl_add_eq_sumIdxFGo (a : Array Float) :
    a.foldl (·+·) 0.0 = sumIdxFGo (fun k => a[k]!) 0.0 (List.range a.size) := by
  rw [sumIdxFGo_eq_foldl, ← Array.foldl_toList,
    listFoldl_eq_rangeFoldl (fun acc x => acc + x) a.toList 0.0, Array.length_toList]
  simp only [toList_getElem!]

/-- `getElem!` through a plain `Array.map` (in range). -/
theorem arrMapGetElem! {α β} [Inhabited α] [Inhabited β] (X : Array α) (g : α → β) (i : Nat)
    (hi : i < X.size) : (X.map g)[i]! = g (X[i]!) := by
  rw [Array.getElem!_eq_getD, Array.getElem!_eq_getD, Array.getD, Array.getD]
  simp [hi, Array.getElem_map]

/-! ### The runnable softmax's max-shift and denominator -/

/-- The max-shift `m = maxⱼ lⱼ` computed by `Train.softmax` (the exact `foldl`, seed `−1e30`). -/
def sMax (logits : Array Float) : Float :=
  logits.foldl (fun a x => if x > a then x else a) (-1.0e30)

/-- The denominator `z = Σⱼ exp(lⱼ − m)` computed by `Train.softmax` (the exact `Array.foldl (+)`). -/
def sDenom (logits : Array Float) : Float :=
  (logits.map (fun x => Float.exp (x - sMax logits))).foldl (· + ·) 0.0

/-- The runnable denominator `z` is exactly the functional model's `sumIdxFGo` sum. -/
theorem sDenom_eq (logits : Array Float) :
    sDenom logits
      = sumIdxFGo (fun k => Float.exp (logits[k]! - sMax logits)) 0.0 (List.range logits.size) := by
  rw [sDenom, arrFoldl_add_eq_sumIdxFGo, Array.size_map]
  apply sumIdxFGo_congr
  intro k hk; simp only [List.mem_range] at hk
  exact arrMapGetElem! logits (fun x => Float.exp (x - sMax logits)) k hk

/-! ### The exec ↔ functional-model identity, and the composed error bound -/

/-- **The runnable softmax equals the functional model, componentwise.** For every in-range `i`,
    `(Train.softmax logits)[i]!` is exactly `softmaxFval (logits[·]!) (max) logits.size i`. Pure
    `Id.run`/`Array.map`/`foldl` unfolding — the numerically-stable executable and the bounded model
    compute the same `Float`. -/
theorem train_softmax_getElem! (logits : Array Float) (i : Nat) (hi : i < logits.size) :
    (Puffer.RL.Train.softmax logits)[i]!
      = softmaxFval (fun k => logits[k]!) (sMax logits) logits.size i := by
  have hunfold : Puffer.RL.Train.softmax logits
      = (logits.map (fun x => Float.exp (x - sMax logits))).map (fun e => e / sDenom logits) := rfl
  rw [hunfold,
    arrMapGetElem! _ (fun e => e / sDenom logits) i (by rw [Array.size_map]; exact hi),
    arrMapGetElem! logits (fun x => Float.exp (x - sMax logits)) i hi,
    softmaxFval, sDenom_eq]

/-- **Runnable softmax Float↔ℝ accuracy.** The trainer's actual `Train.softmax` output `[i]!` is within
    the composed `div/sum/exp` bound of the canonical ℝ softmax `Puffer.Net.softmax`. The whole
    `SoftmaxBound` error analysis, transported to the code that runs. `dmin > 0`, `dmin ≤ |toReal z|` is
    the denominator floor (sum of positive exponentials, `≳ 1`), supplied as a hypothesis. -/
theorem train_softmax_error (logits : Array Float) (i : Nat) (hi : i < logits.size) (dmin : ℝ)
    (hdmin : 0 < dmin) (hden : dmin ≤ |toReal (sDenom logits)|)
    (hyR : (∑ j ∈ Finset.range logits.size,
              Real.exp (toReal logits[j]! - toReal (sMax logits))) ≠ 0) :
    |toReal ((Puffer.RL.Train.softmax logits)[i]!)
        - Puffer.Net.softmax (Finset.range logits.size) (fun j => toReal logits[j]!) i|
      ≤ u64 * |toReal (Float.exp (logits[i]! - sMax logits))
              / toReal (sumIdxFGo (fun k => Float.exp (logits[k]! - sMax logits)) 0.0
                          (List.range logits.size))|
        + (expShiftBnd logits[i]! (sMax logits)
            + |Puffer.Net.softmax (Finset.range logits.size) (fun j => toReal logits[j]!) i|
              * sumIdxErrBnd (fun k => Float.exp (logits[k]! - sMax logits))
                  (fun k => expShiftBnd logits[k]! (sMax logits)) 0.0 (List.range logits.size) 0)
          / dmin := by
  rw [train_softmax_getElem! logits i hi]
  rw [sDenom_eq] at hden
  exact softmaxFval_error_spec (fun k => logits[k]!) (sMax logits) logits.size i dmin hdmin hden hyR

/-! ### Runtime soundness: the executed policy is a genuine (positive) distribution -/

/-- A `Float.exp` output is strictly positive at the `toReal` level (`Real.exp > 0`, the `(1+δ)` rounding
    factor stays `> 0` since `|δ| ≤ expEps < 1`). -/
theorem exp_toReal_pos (a : Float) : 0 < toReal (Float.exp a) := by
  obtain ⟨δ, hδ, he⟩ := exp_model a
  rw [he]
  have hu : (expEps:ℝ) < 1 := by unfold expEps; norm_num
  have hd : (0:ℝ) < 1 + δ := by have := (abs_le.mp hδ).1; linarith
  exact mul_pos (Real.exp_pos _) hd

/-- **A fold of strictly-positive summands is strictly positive** (at `toReal`). The running `sumIdxFGo`
    accumulator stays `> 0` once nonempty: each hardware add of a positive term to a `≥ 0` accumulator keeps
    the `toReal` positive (the `add_model` `(1+δ)` factor is `> 0` since `|δ| ≤ u64 < 1`). -/
theorem sumIdxFGo_pos (e : Nat → Float) (is : List Nat) :
    ∀ (acc : Float), 0 ≤ toReal acc → (∀ k ∈ is, 0 < toReal (e k)) → is ≠ [] →
    0 < toReal (sumIdxFGo e acc is) := by
  induction is with
  | nil => intro acc _ _ hne; exact absurd rfl hne
  | cons i is' ih =>
    intro acc hacc hpos _
    simp only [sumIdxFGo]
    have hei : 0 < toReal (e i) := hpos i (List.mem_cons_self ..)
    obtain ⟨δ, hδ, he⟩ := add_model acc (e i)
    have hu : (u64:ℝ) < 1 := by unfold u64; norm_num
    have hd : (0:ℝ) < 1 + δ := by have := (abs_le.mp hδ).1; linarith
    have hacc' : 0 < toReal (acc + e i) := by
      rw [he]
      have hs : 0 < toReal acc + toReal (e i) := by linarith
      exact mul_pos hs hd
    cases is' with
    | nil => simpa [sumIdxFGo] using hacc'
    | cons j js =>
        exact ih (acc + e i) hacc'.le (fun k hk => hpos k (List.mem_cons_of_mem _ hk)) (by simp)

/-- **The functional-model softmax value is strictly positive** (`n > 0`). Numerator `exp > 0`, denominator
    a nonempty fold of `exp`s `> 0` (`sumIdxFGo_pos`), and the `div_model` `(1+δ)` factor `> 0`. -/
theorem softmaxFval_pos (l : Nat → Float) (m : Float) (n i : Nat) (hn : 0 < n) :
    0 < toReal (softmaxFval l m n i) := by
  rw [softmaxFval]
  obtain ⟨δ, hδ, he⟩ := div_model (Float.exp (l i - m))
    (sumIdxFGo (fun k => Float.exp (l k - m)) 0.0 (List.range n))
  rw [he]
  have hnum : 0 < toReal (Float.exp (l i - m)) := exp_toReal_pos _
  have hden : 0 < toReal (sumIdxFGo (fun k => Float.exp (l k - m)) 0.0 (List.range n)) :=
    sumIdxFGo_pos _ _ 0.0 (by rw [toReal_zeroLit]) (fun k _ => exp_toReal_pos _)
      (List.ne_nil_of_length_pos (by rw [List.length_range]; exact hn))
  have hu : (u64:ℝ) < 1 := by unfold u64; norm_num
  have hd : (0:ℝ) < 1 + δ := by have := (abs_le.mp hδ).1; linarith
  exact mul_pos (div_pos hnum hden) hd

/-- **Runtime soundness: the softmax denominator computed by the trainer is strictly positive.** For a nonempty
    `logits`, the exact `Array.foldl (+)` accumulator `z = Σⱼ exp(lⱼ − m)` (`sDenom`) is `> 0` at the `toReal`
    level — it is a nonempty fold of strictly-positive shifted exponentials (`sumIdxFGo_pos` + `exp_toReal_pos`,
    via `sDenom_eq`). This certifies the trainer's division `e / z` is safe and sign-preserving (never `/0`, never
    negative): the sign of `z` is known, so `|toReal (sDenom logits)| = toReal (sDenom logits)`, realizing the
    denominator-floor hypothesis of `train_softmax_error`. -/
theorem sDenom_pos (logits : Array Float) (hn : 0 < logits.size) :
    0 < toReal (sDenom logits) := by
  rw [sDenom_eq]
  exact sumIdxFGo_pos _ _ 0.0 (by rw [toReal_zeroLit]) (fun k _ => exp_toReal_pos _)
    (List.ne_nil_of_length_pos (by rw [List.length_range]; exact hn))

/-- **Runtime soundness: the executed softmax assigns strictly positive probability to every action.** For
    every in-range `i`, the trainer's actual `Train.softmax logits`[i]! is `> 0` — the runtime counterpart of
    the ℝ `Puffer.Net.softmax_pos`. The trained categorical policy never assigns an exactly-zero (or negative)
    probability, so its `log`-probabilities and entropy stay well-defined. -/
theorem train_softmax_pos (logits : Array Float) (i : Nat) (hi : i < logits.size) :
    0 < toReal ((Puffer.RL.Train.softmax logits)[i]!) := by
  rw [train_softmax_getElem! logits i hi]
  exact softmaxFval_pos (fun k => logits[k]!) (sMax logits) logits.size i (by omega)

/-- **Runtime soundness: the executed policy sums to ≈ 1.** The actual `Train.softmax logits` probabilities
    sum to within the summed per-component error of `1` — the ℝ-softmax sums to exactly `1`, so the runtime
    deviation is `Σᵢ softmaxFvalErrBnd` (via the exec=model bridge + `softmaxFval_sum_error`). With
    `train_softmax_pos`, the trained categorical policy is an approximate probability distribution: every prob
    `> 0`, and they sum to `≈ 1`. -/
theorem train_softmax_sum_error (logits : Array Float) (dmin : ℝ) (hn : 0 < logits.size)
    (hdmin : 0 < dmin) (hden : dmin ≤ |toReal (sDenom logits)|)
    (hyR : (∑ j ∈ Finset.range logits.size,
              Real.exp (toReal logits[j]! - toReal (sMax logits))) ≠ 0) :
    |(∑ i ∈ Finset.range logits.size, toReal ((Puffer.RL.Train.softmax logits)[i]!)) - 1|
      ≤ ∑ i ∈ Finset.range logits.size,
          softmaxFvalErrBnd (fun k => logits[k]!) (sMax logits) logits.size i dmin := by
  rw [sDenom_eq] at hden
  have hcongr : (∑ i ∈ Finset.range logits.size, toReal ((Puffer.RL.Train.softmax logits)[i]!))
      = ∑ i ∈ Finset.range logits.size,
          toReal (softmaxFval (fun k => logits[k]!) (sMax logits) logits.size i) := by
    apply Finset.sum_congr rfl
    intro i hi
    rw [train_softmax_getElem! logits i (Finset.mem_range.mp hi)]
  rw [hcongr]
  exact softmaxFval_sum_error (fun k => logits[k]!) (sMax logits) logits.size dmin hn hdmin hden hyR

/-! ### Inverse-CDF correctness of the categorical sampler -/

/-- Fold-append (split) law for the runnable running sum `sumIdxFGo`:
    `sumIdxFGo e acc (is ++ js) = sumIdxFGo e (sumIdxFGo e acc is) js`. The left fold over a concatenated
    index list is the fold over `js` seeded by the fold over `is` — the runnable-sum counterpart of
    `List.foldl_append`, and the recurrence backbone for splitting a cumulative sum at its last index. -/
theorem sumIdxFGo_append (e : Nat → Float) (acc : Float) (is js : List Nat) :
    sumIdxFGo e acc (is ++ js) = sumIdxFGo e (sumIdxFGo e acc is) js := by
  induction is generalizing acc with
  | nil => rfl
  | cons i is ih => simp only [List.cons_append, sumIdxFGo]; exact ih (acc + e i)

/-- **The categorical sampler respects the uniform draw `u`: the sampled index's Float cumulative sum
    exceeds `u`.** If `sampleCat probs u = i` lands on a NON-final index (`i + 1 < probs.size`), then `u` is
    strictly below the running Float cumulative sum `Σ_{k≤i} probs[k]!` computed by the loop, expressed as
    the codebase's canonical running-sum model `sumIdxFGo (probs[·]!) 0.0 (List.range (i+1))` — the same
    object as the softmax denominator `sDenom`. This is the inverse-CDF *correctness* of `Train.sampleCat`,
    the runtime companion to `sampleCat_lt_size` (which only proves the index is in-bounds): a non-final index
    is only ever returned by the early-`return` branch, whose guard is exactly `u < acc`, and the loop's
    accumulator there is precisely the model's `sumIdxFGo` partial sum. Proved by a loop-invariant argument
    over the desugared `forIn` (done/yield branch analysis + the `cum_succ` recurrence via `sumIdxFGo_append`).
    Both hypotheses are load-bearing: `hres` fixes the index under discussion, and `hi : i + 1 < probs.size`
    is essential — dropping it admits the fall-through index `probs.size - 1` (fired precisely when `u`
    exceeds the whole probability mass), for which the conclusion is false
    (`sampleCat #[0.3,0.3] 0.9 = 1 = size−1` yet `0.9 < cumsum(2) = 0.6` is false). Non-vacuous:
    `sampleCat #[0.3,0.3,0.4] 0.2 = 0` with `0.2 < cumsum(1) = 0.3` true. Axiom-clean beyond `Prop` — the
    statement lives entirely at the level of the loop's own Float `<` guard and the Float running sum, so no
    `toReal`/ℝ reasoning and no Float base axioms are needed. -/
theorem sampleCat_cumsum (probs : Array Float) (u : Float) (i : Nat)
    (hi : i + 1 < probs.size) (hres : Puffer.RL.Train.sampleCat probs u = i) :
    u < sumIdxFGo (fun k => probs[k]!) 0.0 (List.range (i + 1)) := by
  -- recurrence: the cumulative sum through `j` extends by `probs[j]!`
  have cum_succ : ∀ j : Nat,
      sumIdxFGo (fun k => probs[k]!) 0.0 (List.range (j + 1))
        = sumIdxFGo (fun k => probs[k]!) 0.0 (List.range j) + probs[j]! := by
    intro j
    rw [List.range_succ, sumIdxFGo_append]
    rfl
  -- loop invariant: the returned `some a` state carries `u < cumsum(a+1)`
  have key : ∀ (len s a : Nat),
      (forIn (m := Id) (List.range' s len)
          (⟨none, sumIdxFGo (fun k => probs[k]!) 0.0 (List.range s)⟩ : MProd (Option Nat) Float)
          (fun (i : Nat) (r : MProd (Option Nat) Float) =>
            if u < r.snd + probs[i]!
            then pure (ForInStep.done ⟨some i, r.snd + probs[i]!⟩)
            else pure (ForInStep.yield ⟨none, r.snd + probs[i]!⟩))).fst = some a →
      u < sumIdxFGo (fun k => probs[k]!) 0.0 (List.range (a + 1)) := by
    intro len
    induction len with
    | zero =>
        intro s a ha
        simp only [List.range'_zero, List.forIn_nil] at ha
        simp at ha
    | succ len ih =>
        intro s a ha
        rw [List.range'_succ] at ha
        by_cases hb : u < sumIdxFGo (fun k => probs[k]!) 0.0 (List.range s) + probs[s]!
        · -- early-return (done) branch
          simp only [List.forIn_cons, hb, if_true, pure_bind] at ha
          have hax : (some s : Option Nat) = some a := ha
          have has : a = s := (Option.some.inj hax).symm
          rw [has, cum_succ s]
          exact hb
        · -- continue (yield) branch
          simp only [List.forIn_cons, hb, if_false, pure_bind] at ha
          rw [← cum_succ s] at ha
          exact ih (s + 1) a ha
  -- characterize sampleCat via the loop result
  have hchar_some : ∀ a,
      (forIn (m := Id) (List.range' 0 probs.size) (⟨none, 0.0⟩ : MProd (Option Nat) Float)
          (fun (i : Nat) (r : MProd (Option Nat) Float) =>
            if u < r.snd + probs[i]!
            then pure (ForInStep.done ⟨some i, r.snd + probs[i]!⟩)
            else pure (ForInStep.yield ⟨none, r.snd + probs[i]!⟩))).fst = some a →
      Puffer.RL.Train.sampleCat probs u = a := by
    intro a hfa
    unfold Puffer.RL.Train.sampleCat
    simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, Nat.sub_zero,
      Nat.add_one_sub_one, Nat.div_one]
    show (match (forIn (m := Id) (List.range' 0 probs.size) (⟨none, 0.0⟩ : MProd (Option Nat) Float)
            (fun (i : Nat) (r : MProd (Option Nat) Float) =>
              if u < r.snd + probs[i]!
              then pure (ForInStep.done ⟨some i, r.snd + probs[i]!⟩)
              else pure (ForInStep.yield ⟨none, r.snd + probs[i]!⟩))).fst with
          | none => probs.size - 1
          | some a => a) = a
    rw [hfa]
  have hchar_none :
      (forIn (m := Id) (List.range' 0 probs.size) (⟨none, 0.0⟩ : MProd (Option Nat) Float)
          (fun (i : Nat) (r : MProd (Option Nat) Float) =>
            if u < r.snd + probs[i]!
            then pure (ForInStep.done ⟨some i, r.snd + probs[i]!⟩)
            else pure (ForInStep.yield ⟨none, r.snd + probs[i]!⟩))).fst = none →
      Puffer.RL.Train.sampleCat probs u = probs.size - 1 := by
    intro hfn
    unfold Puffer.RL.Train.sampleCat
    simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, Nat.sub_zero,
      Nat.add_one_sub_one, Nat.div_one]
    show (match (forIn (m := Id) (List.range' 0 probs.size) (⟨none, 0.0⟩ : MProd (Option Nat) Float)
            (fun (i : Nat) (r : MProd (Option Nat) Float) =>
              if u < r.snd + probs[i]!
              then pure (ForInStep.done ⟨some i, r.snd + probs[i]!⟩)
              else pure (ForInStep.yield ⟨none, r.snd + probs[i]!⟩))).fst with
          | none => probs.size - 1
          | some a => a) = probs.size - 1
    rw [hfn]
  cases hv : (forIn (m := Id) (List.range' 0 probs.size) (⟨none, 0.0⟩ : MProd (Option Nat) Float)
          (fun (i : Nat) (r : MProd (Option Nat) Float) =>
            if u < r.snd + probs[i]!
            then pure (ForInStep.done ⟨some i, r.snd + probs[i]!⟩)
            else pure (ForInStep.yield ⟨none, r.snd + probs[i]!⟩))).fst with
  | none =>
      have := hchar_none hv
      omega
  | some a =>
      have hsa := hchar_some a hv
      rw [hsa] at hres
      have hk := key probs.size 0 a hv
      rw [hres] at hk
      exact hk

end Puffer.RL.SoftmaxExec
