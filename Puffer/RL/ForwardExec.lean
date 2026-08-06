/-
Connecting the RUNNABLE MLP forward pass to the proven per-neuron / per-logit bounds.

`ForwardRuntime` proves the forward-pass accuracy bounds — `linZ_error` (one linear unit), `neuron_error`
(dense-ReLU neuron), `logit_error` (output logit with the hidden perturbation folded in) — but STATED on
the List-based kernel `dotF w x`. The trainer, however, runs `NNTrain.forwardAll` — an `Array Float`
computation using `dotFA w x = dotF w.toList x.toList` through `Array.range`-maps:

    z1ⱼ = b1ⱼ + dotFA W1ⱼ x        hⱼ = reluF z1ⱼ        logitₖ = b2ₖ + dotFA W2ₖ h

This file closes that gap: each component of `forwardAll`, indexed `[·]!`, reduces (pure `Array.range`/`map`
unfolding + `dotFA`'s `toList` defn) to exactly the List expression the `ForwardRuntime` bound targets, so
the proven bounds land verbatim on the code the trainer runs — the same exec-connection pattern as
`SoftmaxExec`/`LayerNormExec`.

  • `forwardAll_z1_getElem!`/`forwardAll_z1_error` — the pre-activation `z1ⱼ` vs its ideal real value
    (`linZ_error`).
  • `forwardAll_h_getElem!`/`forwardAll_neuron_error` — the hidden ReLU output `hⱼ` vs `max(ideal, 0)`
    (`neuron_error`), with NO error growth through the ReLU.
  • `forwardAll_logit_error` — the output logit `logitₖ` vs the ideal real logit `toReal b2ₖ + dotRm W2ₖ hR`,
    where `hR = hRList p x` is the exact real hidden vector; the layer-2 rounding PLUS the hidden
    perturbation folded through `dotDiffBnd` (`logit_error`).

Per-index, for in-range indices (`j < |b1|`, `k < |b2|`). Axiom-clean beyond the trusted Float base — the
`getElem!` reductions are pure logic; the bounds inherit `ForwardRuntime`'s footprint.
-/
import Puffer.RL.ForwardRuntime
import Puffer.RL.NNTrain

namespace Puffer.RL.ForwardExec

open Puffer.FloatR
open Puffer.RL.NNTrain

/-! ### `getElem!` plumbing through the `Array.range`/`map` forward pass -/

/-- `getElem!` through `(Array.range m).map g` (in range). -/
theorem rangeMapGetElem! {β} [Inhabited β] (m : Nat) (g : Nat → β) (i : Nat) (hi : i < m) :
    ((Array.range m).map g)[i]! = g i := by
  rw [Array.getElem!_eq_getD, Array.getD]; simp [hi, Array.getElem_map, Array.getElem_range]

/-- `getElem!` through a plain `Array.map` (in range). -/
theorem arrMapGetElem! {α β} [Inhabited α] [Inhabited β] (X : Array α) (g : α → β) (i : Nat)
    (hi : i < X.size) : (X.map g)[i]! = g (X[i]!) := by
  rw [Array.getElem!_eq_getD, Array.getElem!_eq_getD, Array.getD, Array.getD]
  simp [hi, Array.getElem_map]

/-! ### One-hot observation encoding -/

/-- The one-hot observation `oneHot size i` has exactly `size` entries. -/
theorem oneHot_size (size i : Nat) : (Puffer.RL.NNTrain.oneHot size i).size = size := by
  unfold Puffer.RL.NNTrain.oneHot; rw [Array.size_map, Array.size_range]

/-- Slot `j` of the one-hot observation is `1.0` at the hot index `i` and `0.0` elsewhere. -/
theorem oneHot_getElem! (size i j : Nat) (hj : j < size) :
    (Puffer.RL.NNTrain.oneHot size i)[j]! = if j == i then 1.0 else 0.0 := by
  unfold Puffer.RL.NNTrain.oneHot
  rw [rangeMapGetElem! _ _ j hj]

/-- **The one-hot observation embeds to the exact real indicator.** `toReal (oneHot size i)[j]! = if j = i
    then 1 else 0` — the input encoding carries no rounding (the `1.0`/`0.0` literals are exact via
    `toReal_oneLit`/`toReal_zeroLit`). -/
theorem oneHot_toReal_getElem! (size i j : Nat) (hj : j < size) :
    toReal ((Puffer.RL.NNTrain.oneHot size i)[j]!) = if j = i then 1 else 0 := by
  rw [oneHot_getElem! size i j hj]
  by_cases h : j = i
  · simp [h, toReal_oneLit]
  · simp [h, toReal_zeroLit, beq_eq_false_iff_ne.mpr h]

/-- Each one-hot entry is nonnegative (it is `0` or `1`). -/
theorem oneHot_toReal_nonneg (size i j : Nat) (hj : j < size) :
    0 ≤ toReal ((Puffer.RL.NNTrain.oneHot size i)[j]!) := by
  rw [oneHot_toReal_getElem! size i j hj]; split_ifs <;> norm_num

/-- Each one-hot entry is at most `1`. -/
theorem oneHot_toReal_le_one (size i j : Nat) (hj : j < size) :
    toReal ((Puffer.RL.NNTrain.oneHot size i)[j]!) ≤ 1 := by
  rw [oneHot_toReal_getElem! size i j hj]; split_ifs <;> norm_num

/-- **The one-hot observation sums to `1`.** `∑ⱼ toReal (oneHot size i)[j]! = 1` (for `i < size`) — together
    with `oneHot_toReal_nonneg`/`_le_one`, the observation is a genuine probability vector (a point mass at
    the hot index). -/
theorem oneHot_toReal_sum (size i : Nat) (hi : i < size) :
    ∑ j ∈ Finset.range size, toReal ((Puffer.RL.NNTrain.oneHot size i)[j]!) = 1 := by
  rw [Finset.sum_congr rfl (fun j hj => oneHot_toReal_getElem! size i j (Finset.mem_range.mp hj))]
  rw [Finset.sum_ite_eq' (Finset.range size) i (fun _ => (1:ℝ))]
  simp [Finset.mem_range.mpr hi]

/-! ### Layer 1: pre-activation and hidden ReLU -/

/-- The runnable pre-activation `z1ⱼ = b1ⱼ + dotF W1ⱼ x` (in range). -/
theorem forwardAll_z1_getElem! (p : MLP) (x : Array Float) (j : Nat) (hj : j < p.b1.size) :
    (forwardAll p x).1[j]! = p.b1[j]! + dotF (p.W1[j]!).toList x.toList := by
  show ((Array.range p.b1.size).map (fun j => p.b1[j]! + dotFA p.W1[j]! x))[j]! = _
  rw [rangeMapGetElem! _ _ j hj]; rfl

/-- **Runnable pre-activation error.** `forwardAll`'s `z1ⱼ` is within `z1ErrBnd` of its exact real value —
    `linZ_error` on the actual net. -/
theorem forwardAll_z1_error (p : MLP) (x : Array Float) (j : Nat) (hj : j < p.b1.size) :
    |toReal ((forwardAll p x).1[j]!)
        - (toReal p.b1[j]! + dotR (p.W1[j]!).toList x.toList)|
      ≤ z1ErrBnd (p.W1[j]!).toList p.b1[j]! x.toList := by
  rw [forwardAll_z1_getElem! p x j hj]
  exact linZ_error (p.W1[j]!).toList p.b1[j]! x.toList

/-! #### One-hot pre-activation collapse (tabular / discrete observations)

For a one-hot observation (maze, boxoban feed one-hot state encodings), the
first-layer dot `∑ᵢ W1ⱼ[i]·x[i]` collapses to a SINGLE weight
lookup `W1ⱼ[hot]` — so the ideal pre-activation is just `b1ⱼ + W1ⱼ[hot]`, with no sum
over the input dimension. -/

/-- `a.toList[k]! = a[k]!` — the List/Array `getElem!` bridge. -/
private theorem toList_getElem! (a : Array Float) (k : Nat) : a.toList[k]! = a[k]! := by
  by_cases hk : k < a.size
  · rw [getElem!_pos a k hk, getElem!_pos a.toList k (by simpa using hk), Array.getElem_toList]
  · rw [getElem!_neg a k hk, getElem!_neg a.toList k (by simpa using hk)]

/-- If the second vector is real-valued zero across its length, the real dot vanishes. -/
private theorem dotR_snd_zero (ws xs : List Float)
    (hz : ∀ k, k < xs.length → toReal (xs[k]!) = 0) : dotR ws xs = 0 := by
  induction ws generalizing xs with
  | nil => rfl
  | cons w ws ih =>
    cases xs with
    | nil => rfl
    | cons x xs =>
      simp only [dotR]
      have hx : toReal x = 0 := by have := hz 0 (by simp); simpa using this
      have hz' : ∀ k, k < xs.length → toReal (xs[k]!) = 0 := fun k hk => by
        have := hz (k + 1) (by simpa using hk); simpa using this
      rw [hx, ih xs hz']; ring

/-- **Real dot against a one-hot second vector collapses to the indexed weight.** If `xs`
    is `1` at position `i` and `0` elsewhere (and matches `ws` in length), then
    `dotR ws xs = toReal ws[i]`. -/
private theorem dotR_indicator (ws xs : List Float) (i : Nat)
    (hlen : xs.length = ws.length) (hi : i < ws.length)
    (hind : ∀ k, k < ws.length → toReal (xs[k]!) = if k = i then 1 else 0) :
    dotR ws xs = toReal (ws[i]!) := by
  induction ws generalizing xs i with
  | nil => exact absurd hi (by simp)
  | cons w ws ih =>
    cases xs with
    | nil => simp at hlen
    | cons x xs =>
      have hlen' : xs.length = ws.length := by simpa using hlen
      simp only [dotR]
      rcases Nat.eq_zero_or_pos i with hi0 | hipos
      · subst hi0
        have hx : toReal x = 1 := by have := hind 0 (by simp); simpa using this
        have hz : ∀ k, k < xs.length → toReal (xs[k]!) = 0 := by
          intro k hk
          have := hind (k + 1) (by simp only [List.length_cons]; omega); simpa using this
        rw [hx, dotR_snd_zero ws xs hz]; simp
      · obtain ⟨i', rfl⟩ := Nat.exists_eq_succ_of_ne_zero hipos.ne'
        have hx : toReal x = 0 := by have := hind 0 (by simp); simpa using this
        have hi' : i' < ws.length := by simp only [List.length_cons] at hi; omega
        have hind' : ∀ k, k < ws.length → toReal (xs[k]!) = if k = i' then 1 else 0 := by
          intro k hk
          have := hind (k + 1) (by simp only [List.length_cons]; omega); simpa using this
        rw [hx, ih xs i' hlen' hi' hind']; simp

/-- **One-hot pre-activation error (tabular observations).** For a ONE-HOT input
    `oneHot size i` with `size = |W1ⱼ|` (a discrete/tabular state encoding), the first-layer
    dot collapses to a single weight: the runnable pre-activation `z1ⱼ` is within `z1ErrBnd`
    of `toReal b1ⱼ + toReal W1ⱼ[i]` — bias plus the `i`-th weight of neuron `j`, no sum over
    the input dimension. Specializes `forwardAll_z1_error` at the one-hot input. -/
theorem forwardAll_z1_oneHot_error (p : MLP) (size i j : Nat)
    (hj : j < p.b1.size) (hsz : (p.W1[j]!).size = size) (hi : i < size) :
    |toReal ((forwardAll p (oneHot size i)).1[j]!)
        - (toReal p.b1[j]! + toReal ((p.W1[j]!)[i]!))|
      ≤ z1ErrBnd (p.W1[j]!).toList p.b1[j]! (oneHot size i).toList := by
  have h := forwardAll_z1_error p (oneHot size i) j hj
  have hcollapse :
      dotR (p.W1[j]!).toList (oneHot size i).toList = toReal ((p.W1[j]!)[i]!) := by
    have hlen : (oneHot size i).toList.length = (p.W1[j]!).toList.length := by
      rw [Array.length_toList, Array.length_toList, oneHot_size, hsz]
    have hiw : i < (p.W1[j]!).toList.length := by rw [Array.length_toList, hsz]; exact hi
    have hind : ∀ k, k < (p.W1[j]!).toList.length →
        toReal ((oneHot size i).toList[k]!) = if k = i then 1 else 0 := by
      intro k hk
      rw [Array.length_toList, hsz] at hk
      rw [toList_getElem!, oneHot_toReal_getElem! size i k hk]
    rw [dotR_indicator (p.W1[j]!).toList (oneHot size i).toList i hlen hiw hind, toList_getElem!]
  rwa [hcollapse] at h

/-- The runnable hidden output `hⱼ = reluF (b1ⱼ + dotF W1ⱼ x)` (in range). -/
theorem forwardAll_h_getElem! (p : MLP) (x : Array Float) (j : Nat) (hj : j < p.b1.size) :
    (forwardAll p x).2.1[j]! = reluF (p.b1[j]! + dotF (p.W1[j]!).toList x.toList) := by
  show (((Array.range p.b1.size).map (fun j => p.b1[j]! + dotFA p.W1[j]! x)).map reluF)[j]! = _
  rw [arrMapGetElem! _ reluF j (by rw [Array.size_map, Array.size_range]; exact hj),
    rangeMapGetElem! _ _ j hj]; rfl

/-- **Runnable dense-ReLU neuron error.** `forwardAll`'s hidden `hⱼ` is within `z1ErrBnd` of its ideal
    `max(real, 0)` — `neuron_error` on the actual net (ReLU passes the error through with no growth). -/
theorem forwardAll_neuron_error (p : MLP) (x : Array Float) (j : Nat) (hj : j < p.b1.size) :
    |toReal ((forwardAll p x).2.1[j]!)
        - max (toReal p.b1[j]! + dotR (p.W1[j]!).toList x.toList) 0|
      ≤ z1ErrBnd (p.W1[j]!).toList p.b1[j]! x.toList := by
  rw [forwardAll_h_getElem! p x j hj]
  exact neuron_error (p.W1[j]!).toList p.b1[j]! x.toList

/-! ### Runtime soundness: the ReLU hidden layer lies in the nonnegative orthant -/

/-- The runnable hidden activation is EXACTLY `max(preactivation, 0)` under `toReal` — the ReLU is exact
    (`toReal_reluF`), so no rounding is introduced by the activation itself. -/
theorem forwardAll_h_toReal (p : MLP) (x : Array Float) (j : Nat) (hj : j < p.b1.size) :
    toReal ((forwardAll p x).2.1[j]!)
      = max (toReal (p.b1[j]! + dotF (p.W1[j]!).toList x.toList)) 0 := by
  rw [forwardAll_h_getElem! p x j hj, toReal_reluF]

/-- **Runtime soundness: the hidden activations are nonnegative.** Every in-range entry of `forwardAll`'s
    ReLU'd hidden vector `h` is `≥ 0` at the `toReal` level — the network's hidden representation lives in the
    nonnegative orthant, a structural invariant of the ReLU layer (`max(·, 0) ≥ 0`, exact via `toReal_reluF`). -/
theorem forwardAll_h_nonneg (p : MLP) (x : Array Float) (j : Nat) (hj : j < p.b1.size) :
    0 ≤ toReal ((forwardAll p x).2.1[j]!) := by
  rw [forwardAll_h_getElem! p x j hj, toReal_reluF]
  exact le_max_right _ _

/-! ### Layer 2: output logits (with the hidden perturbation folded in) -/

/-- The ideal REAL hidden vector — each neuron at its exact real value `max(toReal b1ⱼ + dotR W1ⱼ x, 0)`. -/
noncomputable def hRList (p : MLP) (x : Array Float) : List ℝ :=
  (List.range p.b1.size).map (fun j => max (toReal p.b1[j]! + dotR (p.W1[j]!).toList x.toList) 0)

/-- The IDEAL real hidden vector `hRList` is entrywise nonnegative (each `max(real, 0) ≥ 0`) — the ℝ
    reference lives in the same nonnegative orthant as the runnable hidden layer (`forwardAll_h_nonneg`). -/
theorem hRList_nonneg (p : MLP) (x : Array Float) : ∀ v ∈ hRList p x, 0 ≤ v := by
  intro v hv
  rw [hRList, List.mem_map] at hv
  obtain ⟨j, _, rfl⟩ := hv
  exact le_max_right _ _

/-- `forwardAll`'s hidden array has `|b1|` entries. -/
theorem forwardAll_h_size (p : MLP) (x : Array Float) : (forwardAll p x).2.1.size = p.b1.size := by
  show (((Array.range p.b1.size).map _).map reluF).size = _
  rw [Array.size_map, Array.size_map, Array.size_range]

/-- The runnable output logit `logitₖ = b2ₖ + dotF W2ₖ h` (in range). -/
theorem forwardAll_logit_getElem! (p : MLP) (x : Array Float) (k : Nat) (hk : k < p.b2.size) :
    (forwardAll p x).2.2[k]! = p.b2[k]! + dotF (p.W2[k]!).toList ((forwardAll p x).2.1).toList := by
  show ((Array.range p.b2.size).map (fun k => p.b2[k]! + dotFA p.W2[k]! (forwardAll p x).2.1))[k]! = _
  rw [rangeMapGetElem! _ _ k hk]; rfl

/-- **Runnable output-logit error.** `forwardAll`'s `logitₖ` is within `z1ErrBnd + dotDiffBnd` of the ideal
    real logit `toReal b2ₖ + dotRm W2ₖ hR` (`hR = hRList p x` the exact real hidden) — `logit_error` on the
    actual net: the layer-2 linear-unit rounding PLUS the hidden error folded through the dot. -/
theorem forwardAll_logit_error (p : MLP) (x : Array Float) (k : Nat) (hk : k < p.b2.size) :
    |toReal ((forwardAll p x).2.2[k]!)
        - (toReal p.b2[k]! + dotRm (p.W2[k]!).toList (hRList p x))|
      ≤ z1ErrBnd (p.W2[k]!).toList p.b2[k]! ((forwardAll p x).2.1).toList
        + dotDiffBnd (p.W2[k]!).toList ((forwardAll p x).2.1).toList (hRList p x) := by
  rw [forwardAll_logit_getElem! p x k hk]
  have hlen : ((forwardAll p x).2.1).toList.length = (hRList p x).length := by
    rw [Array.length_toList, forwardAll_h_size, hRList, List.length_map, List.length_range]
  exact logit_error (p.W2[k]!).toList p.b2[k]! ((forwardAll p x).2.1).toList (hRList p x) hlen

end Puffer.RL.ForwardExec
