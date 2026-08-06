/-
The KL divergence between policies — PPO's trust-region / early-stopping metric — within a proven bound of
the logit perturbation, completing the policy-perturbation story.

PPO monitors an approximate KL `KL(π_old ‖ π)` per epoch to decide when to stop (the trust region) — at
runtime via single-sample Schulman-k3 estimators (`PPO.approxKLOld`/`approxKLNew`); `klDiv` here is the exact
full-distribution ℝ-KL those estimate. The KL between the running policy and its ideal is likewise the
natural "how far did floating-point drift the policy" measure. Both fall out of a73's log-softmax
2-Lipschitz bound with one line:

  • `klDiv` — `KL(π_a ‖ π_b) = Σᵢ πᵢ^a (log πᵢ^a − log πᵢ^b)`.
  • `klDiv_le` — the **trust-region bound**: under `|aₖ − bₖ| ≤ ε` on the logits, `KL(π_a ‖ π_b) ≤ 2ε`.
    Since each `log πᵢ^a − log πᵢ^b ≤ 2ε` (`logSoftmax_perturb`) and `πᵢ^a ≥ 0`, `Σ πᵢ^a·(·) ≤ 2ε·Σ πᵢ^a = 2ε`.
  • `klDiv_nonneg` — `0 ≤ KL` (Gibbs' inequality, elementary via `log x ≤ x − 1` and `Σ π = 1`).
  • `policyKL_le` — the capstone on the runnable net: the KL from the ℝ-softmax of the ACTUAL (rounded)
    forward logits to the ideal-logit policy is `≤ 2ε`, `ε` a uniform forward-pass logit-error bound (from
    `ForwardExec.forwardAll_logit_error`) — a trust-region certificate on how far floating-point drifts the
    running policy from its ideal.

All ℝ (Mathlib) — axiom-clean, no Float axioms in the `klDiv` lemmas; `policyKL_le` inherits `ForwardExec`'s
footprint.
-/
import Puffer.RL.LogPolicyBound

namespace Puffer.RL.PolicyKL

open Puffer.FloatR
open Puffer.Net
open Puffer.RL.PolicyBound
open Puffer.RL.LogPolicyBound

/-! ### KL divergence between softmax policies over ℝ -/

variable {ι : Type*}

/-- KL divergence `KL(π_a ‖ π_b) = Σᵢ πᵢ^a (log πᵢ^a − log πᵢ^b)`. -/
noncomputable def klDiv (s : Finset ι) (a b : ι → ℝ) : ℝ :=
  ∑ i ∈ s, softmax s a i * (Real.log (softmax s a i) - Real.log (softmax s b i))

/-- **KL trust-region bound.** Under an `ε`-perturbation of the logits, `KL(π_a ‖ π_b) ≤ 2ε` — each
    log-ratio is `≤ 2ε` (`logSoftmax_perturb`), weighted by `πᵢ^a ≥ 0` summing to `1`. -/
theorem klDiv_le (s : Finset ι) (a b : ι → ℝ) (ε : ℝ) (hs : s.Nonempty)
    (hab : ∀ k ∈ s, |a k - b k| ≤ ε) :
    klDiv s a b ≤ 2*ε := by
  calc klDiv s a b
      ≤ ∑ i ∈ s, softmax s a i * (2*ε) := by
        apply Finset.sum_le_sum
        intro i hi
        apply mul_le_mul_of_nonneg_left _ (softmax_pos s a i hs).le
        exact (le_abs_self _).trans (logSoftmax_perturb s a b i ε hs hi hab)
    _ = (∑ i ∈ s, softmax s a i) * (2*ε) := by rw [← Finset.sum_mul]
    _ = 2*ε := by rw [softmax_sum_one s a hs, one_mul]

/-- **KL is nonnegative** (Gibbs' inequality, via `log x ≤ x − 1` and `Σ π = 1`). -/
theorem klDiv_nonneg (s : Finset ι) (a b : ι → ℝ) (hs : s.Nonempty) :
    0 ≤ klDiv s a b := by
  have heq : klDiv s a b = -∑ i ∈ s, softmax s a i * Real.log (softmax s b i / softmax s a i) := by
    rw [← Finset.sum_neg_distrib, klDiv]
    apply Finset.sum_congr rfl
    intro i _
    rw [Real.log_div (softmax_pos s b i hs).ne' (softmax_pos s a i hs).ne']; ring
  have hle : ∑ i ∈ s, softmax s a i * Real.log (softmax s b i / softmax s a i) ≤ 0 := by
    calc ∑ i ∈ s, softmax s a i * Real.log (softmax s b i / softmax s a i)
        ≤ ∑ i ∈ s, softmax s a i * (softmax s b i / softmax s a i - 1) := by
          apply Finset.sum_le_sum
          intro i _
          apply mul_le_mul_of_nonneg_left _ (softmax_pos s a i hs).le
          exact Real.log_le_sub_one_of_pos (div_pos (softmax_pos s b i hs) (softmax_pos s a i hs))
      _ = ∑ i ∈ s, (softmax s b i - softmax s a i) := by
          apply Finset.sum_congr rfl
          intro i _
          rw [mul_sub, mul_one, mul_div_cancel₀ _ (softmax_pos s a i hs).ne']
      _ = (∑ i ∈ s, softmax s b i) - (∑ i ∈ s, softmax s a i) := by rw [Finset.sum_sub_distrib]
      _ = 0 := by rw [softmax_sum_one s b hs, softmax_sum_one s a hs, sub_self]
  rw [heq]; linarith

/-- **Self-divergence is zero.** `KL(π_a ‖ π_a) = 0` — a policy has zero divergence from itself (each term's
    log-ratio vanishes). Together with `klDiv_nonneg`, `klDiv` behaves as a genuine divergence. -/
theorem klDiv_self (s : Finset ι) (a : ι → ℝ) : klDiv s a a = 0 := by
  unfold klDiv; simp

/-- **The `log x ≤ x − 1` inequality is tight only at `x = 1`.** For `x > 0`, `log x = x − 1 → x = 1` — the
    equality case of the fundamental bound (Gibbs' strict slack). Proof: writing `y = log x`, `x = e^y` and the
    hypothesis says `e^y = y + 1`, which forces `y = 0` since `y + 1 < e^y` strictly for `y ≠ 0`
    (`Real.add_one_lt_exp`). -/
theorem log_eq_sub_one (x : ℝ) (hx : 0 < x) (h : Real.log x = x - 1) : x = 1 := by
  by_contra hne
  have hy : Real.log x ≠ 0 := fun hc => hne (by rw [← Real.exp_log hx, hc, Real.exp_zero])
  have hlt := Real.add_one_lt_exp hy
  rw [Real.exp_log hx] at hlt
  linarith

/-- **KL definiteness (the equality case of Gibbs' inequality).** `KL(π_a ‖ π_b) = 0 ↔ the two softmax policies
    agree pointwise on `s`. Together with `klDiv_nonneg` and `klDiv_self`, this makes `klDiv` a genuine
    divergence: nonnegative, and zero *exactly* when the distributions coincide. The forward direction rewrites
    `klDiv` as `Σ πᵢ^a·[(qᵢ/pᵢ − 1) − log(qᵢ/pᵢ)]` (a sum of nonnegative slacks summing to the same total, since
    `Σ(qᵢ − pᵢ) = 0`); a zero sum of nonnegatives forces each slack to vanish, and `log_eq_sub_one` then pins
    `qᵢ/pᵢ = 1`, i.e. `πᵢ^b = πᵢ^a`. -/
theorem klDiv_eq_zero_iff (s : Finset ι) (a b : ι → ℝ) (hs : s.Nonempty) :
    klDiv s a b = 0 ↔ ∀ i ∈ s, softmax s a i = softmax s b i := by
  constructor
  · intro h0 i hi
    have hsum0 : ∑ j ∈ s, (softmax s b j - softmax s a j) = 0 := by
      rw [Finset.sum_sub_distrib, softmax_sum_one s b hs, softmax_sum_one s a hs, sub_self]
    have key : ∀ j ∈ s, softmax s a j * (Real.log (softmax s a j) - Real.log (softmax s b j))
        = softmax s a j * ((softmax s b j / softmax s a j - 1)
            - Real.log (softmax s b j / softmax s a j)) - (softmax s b j - softmax s a j) := by
      intro j _
      have hp := softmax_pos s a j hs
      have hq := softmax_pos s b j hs
      rw [Real.log_div hq.ne' hp.ne']
      field_simp
      ring
    have hklslack : klDiv s a b = ∑ j ∈ s, softmax s a j
        * ((softmax s b j / softmax s a j - 1) - Real.log (softmax s b j / softmax s a j)) := by
      calc klDiv s a b
          = ∑ j ∈ s, (softmax s a j * ((softmax s b j / softmax s a j - 1)
              - Real.log (softmax s b j / softmax s a j)) - (softmax s b j - softmax s a j)) := by
            rw [klDiv]; exact Finset.sum_congr rfl key
        _ = (∑ j ∈ s, softmax s a j * ((softmax s b j / softmax s a j - 1)
              - Real.log (softmax s b j / softmax s a j))) - ∑ j ∈ s, (softmax s b j - softmax s a j) := by
            rw [Finset.sum_sub_distrib]
        _ = _ := by rw [hsum0, sub_zero]
    have hnn : ∀ j ∈ s, 0 ≤ softmax s a j * ((softmax s b j / softmax s a j - 1)
        - Real.log (softmax s b j / softmax s a j)) := by
      intro j _
      apply mul_nonneg (softmax_pos s a j hs).le
      have := Real.log_le_sub_one_of_pos (div_pos (softmax_pos s b j hs) (softmax_pos s a j hs))
      linarith
    have hzero := (Finset.sum_eq_zero_iff_of_nonneg hnn).mp (hklslack ▸ h0) i hi
    have hp := softmax_pos s a i hs
    have hbr : (softmax s b i / softmax s a i - 1)
        - Real.log (softmax s b i / softmax s a i) = 0 := by
      rcases mul_eq_zero.mp hzero with h | h
      · exact absurd h hp.ne'
      · exact h
    have hx : softmax s b i / softmax s a i = 1 :=
      log_eq_sub_one _ (div_pos (softmax_pos s b i hs) hp) (by linarith)
    exact ((div_eq_one_iff_eq hp.ne').mp hx).symm
  · intro h
    unfold klDiv
    apply Finset.sum_eq_zero
    intro i hi
    rw [h i hi]; ring

/-- **KL is shift-invariant in the first policy's logits.** `KL(π_{a+c} ‖ π_b) = KL(π_a ‖ π_b)` — adding a
    constant to `a`'s logits leaves the softmax `π_a` (hence the divergence) unchanged (`softmax_shift`). -/
theorem klDiv_shift_left (s : Finset ι) (a b : ι → ℝ) (c : ℝ) :
    klDiv s (fun i => a i + c) b = klDiv s a b := by
  unfold klDiv
  apply Finset.sum_congr rfl
  intro i _
  rw [softmax_shift]

/-- **KL is shift-invariant in the second policy's logits.** `KL(π_a ‖ π_{b+c}) = KL(π_a ‖ π_b)`. -/
theorem klDiv_shift_right (s : Finset ι) (a b : ι → ℝ) (c : ℝ) :
    klDiv s a (fun i => b i + c) = klDiv s a b := by
  unfold klDiv
  apply Finset.sum_congr rfl
  intro i _
  rw [softmax_shift]

/-- **Symmetric KL (Jeffreys divergence) is the logit–probability inner product.**
    `KL(π_a ‖ π_b) + KL(π_b ‖ π_a) = Σᵢ (π_a(i) − π_b(i))·(aᵢ − bᵢ)` — the two-way (Jeffreys) divergence between
    two softmax policies equals the inner product of the logit shift `a − b` with the induced probability shift
    `π_a − π_b`. This is the categorical/exponential-family instance of the "symmetric Bregman divergence =
    ⟨Δ natural-params, Δ mean-params⟩" identity: the logits are the natural parameters and the softmax
    probabilities the dual (mean) parameters. Proof: the log-softmax identity `log πᵢ = lᵢ − log Z` per
    coordinate — the shared log-partition drift `log Z_a − log Z_b` factors out and is annihilated by
    `Σ(π_a − π_b) = 0` (`softmax_sum_one` twice). Holds unconditionally (empty support gives `0 = 0`). -/
theorem symmKL_eq_inner (s : Finset ι) (a b : ι → ℝ) :
    klDiv s a b + klDiv s b a = ∑ i ∈ s, (softmax s a i - softmax s b i) * (a i - b i) := by
  rcases s.eq_empty_or_nonempty with hemp | hs
  · subst hemp; simp [klDiv]
  · have hz : ∑ i ∈ s, (softmax s a i - softmax s b i) = 0 := by
      rw [Finset.sum_sub_distrib, softmax_sum_one s a hs, softmax_sum_one s b hs, sub_self]
    calc klDiv s a b + klDiv s b a
        = ∑ i ∈ s, (softmax s a i - softmax s b i)
            * (Real.log (softmax s a i) - Real.log (softmax s b i)) := by
          rw [klDiv, klDiv, ← Finset.sum_add_distrib]
          exact Finset.sum_congr rfl (fun i _ => by ring)
      _ = ∑ i ∈ s, ((softmax s a i - softmax s b i) * (a i - b i)
            - (softmax s a i - softmax s b i)
              * (Real.log (softmaxDenom s a) - Real.log (softmaxDenom s b))) := by
          apply Finset.sum_congr rfl
          intro i _
          rw [Puffer.Net.log_softmax_eq s a i hs, Puffer.Net.log_softmax_eq s b i hs]; ring
      _ = (∑ i ∈ s, (softmax s a i - softmax s b i) * (a i - b i))
            - ∑ i ∈ s, (softmax s a i - softmax s b i)
              * (Real.log (softmaxDenom s a) - Real.log (softmaxDenom s b)) := by
          rw [Finset.sum_sub_distrib]
      _ = (∑ i ∈ s, (softmax s a i - softmax s b i) * (a i - b i))
            - (∑ i ∈ s, (softmax s a i - softmax s b i))
              * (Real.log (softmaxDenom s a) - Real.log (softmaxDenom s b)) := by
          rw [Finset.sum_mul]
      _ = ∑ i ∈ s, (softmax s a i - softmax s b i) * (a i - b i) := by
          rw [hz, zero_mul, sub_zero]

/-- **KL is the Bregman divergence of the log-partition function.** `KL(π_a ‖ π_b) = A(b) − A(a) − ⟨∇A(a), b − a⟩`,
    spelled out as `klDiv s a b = log Z_b − log Z_a − Σᵢ π_a(i)·(bᵢ − aᵢ)`, where `A(l) = log(softmaxDenom s l)` is
    the convex log-partition (cumulant) function of the categorical exponential family and `∇A(a)ᵢ = π_a(i)` is the
    softmax (its dual/mean parameter). Equivalently, KL is the (expected logit difference) minus the (free-energy /
    log-partition difference): `Σᵢ π_a(i)(aᵢ − bᵢ) − (log Z_a − log Z_b)`. This is the fundamental one-sided closed
    form UNDERLYING `symmKL_eq_inner` (summing this for `(a,b)` and `(b,a)` cancels the log-partition terms via
    `Σ(π_a − π_b) = 0`) and `klDiv_const_right` (taking `b` constant collapses it to `log|s| − H`). The RHS exposes
    the log-partition `Z_l = softmaxDenom s l` — which does NOT appear anywhere in `klDiv`'s definition
    (`Σ π_a(log π_a − log π_b)`), so this is a genuine non-definitional identity, not a restatement. Proof: the
    per-coordinate log-softmax identity `log π_l(i) = lᵢ − log Z_l` (`log_softmax_eq`) factors the shared `log Z`
    terms out of the sum, then `Σ π_a = 1` (`softmax_sum_one`) pins the free-energy coefficient. Holds
    unconditionally (empty support gives `0 = 0`). -/
theorem klDiv_eq_logPartition_bregman (s : Finset ι) (a b : ι → ℝ) :
    klDiv s a b
      = Real.log (softmaxDenom s b) - Real.log (softmaxDenom s a)
        - ∑ i ∈ s, softmax s a i * (b i - a i) := by
  rcases s.eq_empty_or_nonempty with hemp | hs
  · subst hemp; simp [klDiv, softmaxDenom]
  · have hneg : ∑ i ∈ s, softmax s a i * (a i - b i)
        = -∑ i ∈ s, softmax s a i * (b i - a i) := by
      rw [← Finset.sum_neg_distrib]
      apply Finset.sum_congr rfl; intro i _; ring
    calc klDiv s a b
        = ∑ i ∈ s, (softmax s a i * (a i - b i)
            + softmax s a i * (Real.log (softmaxDenom s b) - Real.log (softmaxDenom s a))) := by
          rw [klDiv]
          apply Finset.sum_congr rfl
          intro i _
          rw [Puffer.Net.log_softmax_eq s a i hs, Puffer.Net.log_softmax_eq s b i hs]; ring
      _ = (∑ i ∈ s, softmax s a i * (a i - b i))
            + (∑ i ∈ s, softmax s a i)
              * (Real.log (softmaxDenom s b) - Real.log (softmaxDenom s a)) := by
          rw [Finset.sum_add_distrib, Finset.sum_mul]
      _ = (∑ i ∈ s, softmax s a i * (a i - b i))
            + (Real.log (softmaxDenom s b) - Real.log (softmaxDenom s a)) := by
          rw [softmax_sum_one s a hs, one_mul]
      _ = Real.log (softmaxDenom s b) - Real.log (softmaxDenom s a)
            - ∑ i ∈ s, softmax s a i * (b i - a i) := by
          rw [hneg]; ring

/-- **Logit and probability shifts are positively aligned.** `0 ≤ Σᵢ (π_a(i) − π_b(i))·(aᵢ − bᵢ)` — moving a
    logit up never on-net decreases the corresponding softmax probability relative to the reference. Immediate
    from `symmKL_eq_inner` and the nonnegativity of both one-sided KLs (`klDiv_nonneg`): the symmetric KL is
    `≥ 0`, hence so is the inner product it equals. The monotone/correlation law behind softmax's
    order-preservation in the natural parameters. -/
theorem inner_shift_nonneg (s : Finset ι) (a b : ι → ℝ) (hs : s.Nonempty) :
    0 ≤ ∑ i ∈ s, (softmax s a i - softmax s b i) * (a i - b i) := by
  rw [← symmKL_eq_inner]
  exact add_nonneg (klDiv_nonneg s a b hs) (klDiv_nonneg s b a hs)

/-! ### Capstone: the running-policy KL trust-region certificate -/

open Puffer.RL.NNTrain
open Puffer.RL.ForwardExec (forwardAll_logit_error hRList)

/-- **Running-policy KL trust-region bound.** The KL from the ℝ-softmax of the actual (rounded) forward
    logits to the ideal-logit policy is `≤ 2ε`, `ε` a uniform forward-pass logit-error bound (from
    `forwardAll_logit_error`). A trust-region certificate: floating-point drifts the running policy from its
    ideal by at most `2ε` in KL. -/
theorem policyKL_le (p : MLP) (size s : Nat) (ε : ℝ) (hpos : 0 < p.b2.size)
    (hεbnd : ∀ k, k < p.b2.size →
      z1ErrBnd (p.W2[k]!).toList p.b2[k]! ((forwardAll p (oneHot size s)).2.1).toList
        + dotDiffBnd (p.W2[k]!).toList ((forwardAll p (oneHot size s)).2.1).toList (hRList p (oneHot size s))
        ≤ ε) :
    klDiv (Finset.range p.b2.size)
        (fun j => toReal (forwardAll p (oneHot size s)).2.2[j]!) (idealLogitR p size s)
      ≤ 2*ε := by
  apply klDiv_le _ _ _ ε (Finset.nonempty_range_iff.mpr (by omega))
  intro k hk
  have hk' : k < p.b2.size := Finset.mem_range.mp hk
  exact (forwardAll_logit_error p (oneHot size s) k hk').trans (hεbnd k hk')

end Puffer.RL.PolicyKL
