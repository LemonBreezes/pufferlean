/-
The policy entropy `H(π) = −Σ πᵢ log πᵢ` (PPO's entropy bonus) under forward-pass perturbation — the last
objective term, composing the softmax (a72) and KL (a74) bounds with no probability floor.

PPO adds an entropy bonus `+ ent_coef · H(π)` to encourage exploration; `H` depends on the policy, which
drifts when the forward pass rounds. The clean bound falls out of the decomposition
`H(π_a) − H(π_b) = −KL(π_a ‖ π_b) − Σ (πᵢ^a − πᵢ^b) log πᵢ^b`:

  • `entropy` / `entropy_nonneg` — `H(π) = −Σ πᵢ log πᵢ ≥ 0`.
  • `entropy_perturb` — `|H(π_a) − H(π_b)| ≤ 2ε + (e^{2ε} − 1)·H(π_b)`: the KL term is `≤ 2ε` (a74's
    `klDiv_le`/`klDiv_nonneg`), and the softmax-drift term is bounded using a72's tight `softmax_input_perturb`
    weighted by `|log πᵢ^b|`, collapsed via the identity `Σ πᵢ^b |log πᵢ^b| = H(π_b)` (each `πᵢ ≤ 1`, so
    `|log πᵢ| = −log πᵢ`). NO probability floor — the potentially-huge `|log πᵢ^b|` is absorbed by `πᵢ^b`.
  • `entropy_run_perturb` — on the RUNNABLE net: the entropy of the running actor policy (first-`A` softmax of
    the actual forward logits) is within `2ε + (e^{2ε} − 1)·H_ideal` of the ideal, `ε` the forward-pass
    logit error.

All ℝ (Mathlib) — `entropy_perturb`/`entropy_nonneg` are axiom-clean; the capstone inherits the trusted Float
base via the forward-pass logit bound.
-/
import Puffer.RL.PolicyKL
import Puffer.RL.ActorCriticBound

namespace Puffer.RL.EntropyPerturb

open Puffer.FloatR
open Puffer.Net
open Puffer.RL.PolicyBound
open Puffer.RL.PolicyKL

/-! ### The ℝ entropy perturbation -/

variable {ι : Type*}

/-- Policy (Shannon) entropy `H(π) = −Σᵢ πᵢ log πᵢ`. -/
noncomputable def entropy (s : Finset ι) (l : ι → ℝ) : ℝ :=
  -∑ i ∈ s, softmax s l i * Real.log (softmax s l i)

/-- Entropy is nonnegative (each `πᵢ log πᵢ ≤ 0`). -/
theorem entropy_nonneg (s : Finset ι) (l : ι → ℝ) (hs : s.Nonempty) : 0 ≤ entropy s l := by
  rw [entropy, neg_nonneg]
  apply Finset.sum_nonpos
  intro i hi
  exact mul_nonpos_of_nonneg_of_nonpos (softmax_pos s l i hs).le
    (Real.log_nonpos (softmax_pos s l i hs).le (softmax_le_one s l i hs hi))

/-- **Entropy is strictly positive for a genuine choice.** When the action set has `≥ 2` elements, the softmax
    policy is never deterministic (every `πᵢ ∈ (0,1)`), so `0 < entropy s l`. Each `−πᵢ log πᵢ ≥ 0` and at least
    one is strictly positive (some `πᵢ < 1`, via `softmax_lt_one`). This is why the entropy bonus keeps a softmax
    policy exploring: it can only reach `0` in the degenerate single-action limit, never at a reachable policy. -/
theorem entropy_pos (s : Finset ι) (l : ι → ℝ) (hs : s.Nonempty) (hcard : 2 ≤ s.card) :
    0 < entropy s l := by
  obtain ⟨a, ha, b, hb, hab⟩ := Finset.one_lt_card.mp (by omega : 1 < s.card)
  rw [entropy, ← Finset.sum_neg_distrib]
  apply Finset.sum_pos'
  · intro i hi
    have : softmax s l i * Real.log (softmax s l i) ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos (softmax_pos s l i hs).le
        (Real.log_nonpos (softmax_pos s l i hs).le (softmax_le_one s l i hs hi))
    linarith
  · refine ⟨a, ha, ?_⟩
    have hpi : 0 < softmax s l a := softmax_pos s l a hs
    have hlt : softmax s l a < 1 := softmax_lt_one s l a b hs ha hb hab.symm
    have hlog : Real.log (softmax s l a) < 0 := Real.log_neg hpi hlt
    have : softmax s l a * Real.log (softmax s l a) < 0 := mul_neg_of_pos_of_neg hpi hlog
    linarith

/-- **Entropy is shift-invariant.** Adding a constant `c` to every logit leaves the categorical entropy
    unchanged (`entropy s (l + c) = entropy s l`) — since the softmax probabilities are shift-invariant
    (`softmax_shift`), so is the entropy bonus. The numerically-stable max-subtraction the trainer runs does not
    change the entropy term it optimizes. -/
theorem entropy_shift (s : Finset ι) (l : ι → ℝ) (c : ℝ) :
    entropy s (fun i => l i + c) = entropy s l := by
  unfold entropy
  congr 1
  apply Finset.sum_congr rfl
  intro i _
  rw [softmax_shift]

/-- **The uniform policy achieves entropy `log |s|`.** For constant logits (which give the uniform policy,
    `softmax_const`), `entropy s (const c) = log(s.card)` — the maximum-entropy value for `|s|` outcomes. This
    is the largest entropy any categorical policy over `s` can have. -/
theorem entropy_const (s : Finset ι) (c : ℝ) (hs : s.Nonempty) :
    entropy s (fun _ => c) = Real.log s.card := by
  unfold entropy
  have h : ∀ i ∈ s, softmax s (fun _ => c) i * Real.log (softmax s (fun _ => c) i)
      = (1/(s.card:ℝ)) * Real.log (1/(s.card:ℝ)) := by
    intro i _; rw [softmax_const s c i hs]
  rw [Finset.sum_congr rfl h, Finset.sum_const, nsmul_eq_mul]
  have hcard : (s.card:ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Finset.card_ne_zero.mpr hs)
  rw [Real.log_div one_ne_zero hcard, Real.log_one]
  field_simp
  ring

/-- **KL from the uniform policy = `log|s| − H`.** `KL(π_a ‖ uniform) = log(s.card) − entropy s a` — the
    divergence of any policy from the uniform reference is the entropy gap to the maximum `log|s|`. Since
    `KL ≥ 0` (`klDiv_nonneg`), this re-derives `entropy ≤ log|s|` for policies of the form `softmax a`. -/
theorem klDiv_const_right (s : Finset ι) (a : ι → ℝ) (c : ℝ) (hs : s.Nonempty) :
    klDiv s a (fun _ => c) = Real.log s.card - entropy s a := by
  unfold klDiv
  have h1 : ∀ i ∈ s, softmax s a i * (Real.log (softmax s a i) - Real.log (softmax s (fun _ => c) i))
      = softmax s a i * Real.log (softmax s a i) - Real.log (1/(s.card:ℝ)) * softmax s a i := by
    intro i _; rw [softmax_const s c i hs]; ring
  rw [Finset.sum_congr rfl h1, Finset.sum_sub_distrib, ← Finset.mul_sum,
    softmax_sum_one s a hs, mul_one]
  unfold entropy
  rw [Real.log_div one_ne_zero (Nat.cast_ne_zero.mpr (Finset.card_ne_zero.mpr hs)), Real.log_one]
  ring

/-- **Entropy is at most `log|s|`** (for `softmax`-policies). `entropy s a ≤ log(s.card)` — the uniform policy
    maximizes entropy. Immediate from `klDiv_const_right` and `klDiv_nonneg` (`KL(π_a ‖ uniform) ≥ 0`). -/
theorem entropy_le_log_card (s : Finset ι) (a : ι → ℝ) (hs : s.Nonempty) :
    entropy s a ≤ Real.log s.card := by
  have hk := klDiv_nonneg s a (fun _ => (0:ℝ)) hs
  rw [klDiv_const_right s a 0 hs] at hk
  linarith

/-- **Non-uniform policies have strictly sub-maximal entropy (uniqueness of the max-entropy policy).** If two
    actions `i, j ∈ s` carry different logits (`a i ≠ a j`), then `entropy s a < log|s|` — the uniform policy is
    the *unique* maximizer of the categorical entropy. This is the strict sharpening of `entropy_le_log_card`:
    that bound is tight *only* at the uniform policy, so any policy the trainer's entropy bonus is still pushing
    toward uniform sits strictly below `log|s|`. Proof: `entropy s a = log|s| − KL(π_a ‖ uniform)`
    (`klDiv_const_right`), and the divergence is strictly positive — `≥ 0` (`klDiv_nonneg`) and vanishing only
    when `π_a` matches the uniform policy pointwise (`klDiv_eq_zero_iff`), which by `softmax_eq_iff` would force
    `a i = a j`, contradicting the hypothesis. The hypothesis `a i ≠ a j` is load-bearing: with constant logits
    `entropy_const` gives equality `entropy = log|s|`, so the strict inequality is false without it. -/
theorem entropy_lt_log_card (s : Finset ι) (a : ι → ℝ) (i j : ι)
    (hi : i ∈ s) (hj : j ∈ s) (hij : a i ≠ a j) :
    entropy s a < Real.log s.card := by
  have hs : s.Nonempty := ⟨i, hi⟩
  have hkl : klDiv s a (fun _ => (0 : ℝ)) = Real.log s.card - entropy s a :=
    klDiv_const_right s a 0 hs
  have hpos : 0 < klDiv s a (fun _ => (0 : ℝ)) := by
    rcases (klDiv_nonneg s a (fun _ => (0 : ℝ)) hs).lt_or_eq with h | h
    · exact h
    · exfalso
      have hzero := (klDiv_eq_zero_iff s a (fun _ => (0 : ℝ)) hs).mp h.symm
      have hi' := hzero i hi
      have hj' := hzero j hj
      rw [softmax_const s 0 i hs] at hi'
      rw [softmax_const s 0 j hs] at hj'
      exact hij ((softmax_eq_iff s a i j hs).mp (by rw [hi', hj']))
  linarith

/-! ### Cross-entropy and its decomposition -/

/-- Cross-entropy `H(π_a, π_b) = −Σᵢ π_a(i)·log π_b(i)` — the expected surprisal of `π_b` under `π_a`. -/
noncomputable def crossEntropy (s : Finset ι) (a b : ι → ℝ) : ℝ :=
  -∑ i ∈ s, softmax s a i * Real.log (softmax s b i)

/-- **Cross-entropy decomposition.** `H(π_a, π_b) = H(π_a) + KL(π_a ‖ π_b)` — the fundamental identity: cross
    entropy is the policy's own entropy plus its divergence from the target. -/
theorem crossEntropy_eq (s : Finset ι) (a b : ι → ℝ) :
    crossEntropy s a b = entropy s a + klDiv s a b := by
  unfold crossEntropy entropy klDiv
  rw [← Finset.sum_neg_distrib, neg_add_eq_sub, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl; intro i _; ring

/-- **Cross-entropy is minimized by matching the target.** `H(π_a) ≤ H(π_a, π_b)` — since `KL ≥ 0`, the cross
    entropy is never below the entropy, with equality iff the policies agree. This is why minimizing
    cross-entropy pushes `π_b` toward `π_a`. -/
theorem crossEntropy_ge_entropy (s : Finset ι) (a b : ι → ℝ) (hs : s.Nonempty) :
    entropy s a ≤ crossEntropy s a b := by
  rw [crossEntropy_eq]; linarith [klDiv_nonneg s a b hs]

/-- **Self cross-entropy is entropy.** `H(π_a, π_a) = H(π_a)` (the divergence term vanishes). -/
theorem crossEntropy_self (s : Finset ι) (a : ι → ℝ) :
    crossEntropy s a a = entropy s a := by
  rw [crossEntropy_eq, klDiv_self, add_zero]

/-- Cross-entropy is invariant to shifting the first policy's logits (`softmax_shift`). -/
theorem crossEntropy_shift_left (s : Finset ι) (a b : ι → ℝ) (c : ℝ) :
    crossEntropy s (fun i => a i + c) b = crossEntropy s a b := by
  unfold crossEntropy
  congr 1; apply Finset.sum_congr rfl; intro i _; rw [softmax_shift]

/-- Cross-entropy is invariant to shifting the second policy's logits (`softmax_shift`). -/
theorem crossEntropy_shift_right (s : Finset ι) (a b : ι → ℝ) (c : ℝ) :
    crossEntropy s a (fun i => b i + c) = crossEntropy s a b := by
  unfold crossEntropy
  congr 1; apply Finset.sum_congr rfl; intro i _; rw [softmax_shift]

/-- **Entropy perturbation.** `|H(π_a) − H(π_b)| ≤ 2ε + (e^{2ε} − 1)·H(π_b)` under an `ε`-perturbation of the
    logits — the KL term (a74, `≤ 2ε`) plus the softmax drift (a72) weighted by `|log π_b|`, collapsed via
    `Σ πᵢ|log πᵢ| = H(π)`. No probability floor. -/
theorem entropy_perturb (s : Finset ι) (a b : ι → ℝ) (ε : ℝ) (hs : s.Nonempty)
    (hab : ∀ k ∈ s, |a k - b k| ≤ ε) :
    |entropy s a - entropy s b| ≤ 2*ε + (Real.exp (2*ε) - 1) * entropy s b := by
  have hentropy : ∑ i ∈ s, softmax s b i * |Real.log (softmax s b i)| = entropy s b := by
    rw [entropy, ← Finset.sum_neg_distrib]
    apply Finset.sum_congr rfl
    intro i hi
    rw [abs_of_nonpos (Real.log_nonpos (softmax_pos s b i hs).le (softmax_le_one s b i hs hi))]; ring
  have hsum : |∑ i ∈ s, (softmax s a i - softmax s b i) * Real.log (softmax s b i)|
      ≤ (Real.exp (2*ε) - 1) * entropy s b := by
    calc |∑ i ∈ s, (softmax s a i - softmax s b i) * Real.log (softmax s b i)|
        ≤ ∑ i ∈ s, |(softmax s a i - softmax s b i) * Real.log (softmax s b i)| :=
          Finset.abs_sum_le_sum_abs _ _
      _ = ∑ i ∈ s, |softmax s a i - softmax s b i| * |Real.log (softmax s b i)| := by simp_rw [abs_mul]
      _ ≤ ∑ i ∈ s, (softmax s b i * (Real.exp (2*ε) - 1)) * |Real.log (softmax s b i)| := by
          apply Finset.sum_le_sum
          intro i hi
          exact mul_le_mul_of_nonneg_right (softmax_input_perturb s a b i ε hs hi hab) (abs_nonneg _)
      _ = (Real.exp (2*ε) - 1) * ∑ i ∈ s, softmax s b i * |Real.log (softmax s b i)| := by
          rw [Finset.mul_sum]; apply Finset.sum_congr rfl; intro i _; ring
      _ = (Real.exp (2*ε) - 1) * entropy s b := by rw [hentropy]
  have hid : entropy s a - entropy s b
      = -klDiv s a b - ∑ i ∈ s, (softmax s a i - softmax s b i) * Real.log (softmax s b i) := by
    have lhs_eq : entropy s a - entropy s b
        = ∑ i ∈ s, (softmax s b i * Real.log (softmax s b i)
            - softmax s a i * Real.log (softmax s a i)) := by
      rw [entropy, entropy, neg_sub_neg, ← Finset.sum_sub_distrib]
    have rhs_eq : -klDiv s a b - ∑ i ∈ s, (softmax s a i - softmax s b i) * Real.log (softmax s b i)
        = ∑ i ∈ s, (softmax s b i * Real.log (softmax s b i)
            - softmax s a i * Real.log (softmax s a i)) := by
      rw [klDiv, ← Finset.sum_neg_distrib, ← Finset.sum_sub_distrib]
      apply Finset.sum_congr rfl; intro i _; ring
    rw [lhs_eq, rhs_eq]
  rw [hid]
  refine (abs_sub _ _).trans ?_
  rw [abs_neg, abs_of_nonneg (klDiv_nonneg s a b hs)]
  exact add_le_add (klDiv_le s a b ε hs hab) hsum

/-! ### Capstone: the running actor policy's entropy under forward-pass error -/

open Puffer.RL.NNTrain
open Puffer.RL.ActorCriticBound (idealLogit acLogits acLogits_getElem!)
open Puffer.RL.ForwardExec (forwardAll_logit_error hRList)

/-- **Running-actor entropy perturbation.** The entropy of the running actor policy (first-`A` softmax of the
    actual forward logits) is within `2ε + (e^{2ε} − 1)·H_ideal` of the ideal-logit entropy, `ε` a uniform
    forward-pass logit-error bound (from `forwardAll_logit_error`). -/
theorem entropy_run_perturb (p : MLP) (obs : Array Float) (ε : ℝ) (hpos1 : 0 < p.b2.size - 1)
    (hεbnd : ∀ k, k < p.b2.size - 1 →
      z1ErrBnd (p.W2[k]!).toList p.b2[k]! ((forwardAll p obs).2.1).toList
        + dotDiffBnd (p.W2[k]!).toList ((forwardAll p obs).2.1).toList (hRList p obs) ≤ ε) :
    |entropy (Finset.range (p.b2.size - 1)) (fun j => toReal (acLogits p obs)[j]!)
        - entropy (Finset.range (p.b2.size - 1)) (idealLogit p obs)|
      ≤ 2*ε + (Real.exp (2*ε) - 1) * entropy (Finset.range (p.b2.size - 1)) (idealLogit p obs) := by
  apply entropy_perturb (Finset.range (p.b2.size - 1)) (fun j => toReal (acLogits p obs)[j]!)
    (idealLogit p obs) ε (Finset.nonempty_range_iff.mpr (by omega))
  intro k hk
  have hk' : k < p.b2.size - 1 := Finset.mem_range.mp hk
  rw [acLogits_getElem! p obs k hk']
  exact (forwardAll_logit_error p obs k (by omega)).trans (hεbnd k hk')

end Puffer.RL.EntropyPerturb
