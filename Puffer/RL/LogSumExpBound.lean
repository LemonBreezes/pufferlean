/-
The log-sum-exp `logSumExp(l) = log Σⱼ exp lⱼ` — the numerically-stable log-normalizer the AD-tape log-policy
uses (`logπ(a) = lₐ − logSumExp(l)`) — with a proven Float↔ℝ accuracy bound.

The policy-perturbation bounds (a72–a81) hold the softmax's OWN Float rounding out of scope (a72's `Bsm`);
this file supplies the corresponding piece for the LOG-policy: how accurately the running Float
`logSumExp` (`Float.log (Σ Float.exp)`) tracks its ℝ meaning. It reuses the softmax machinery:

  • `lseF` — the functional model `Float.log (sumIdxFGo (exp ·) 0 (range n))` (running sum of rounded exps,
    then the outer `Float.log`).
  • `lseF_error` — within `logEps·|log Dꜰ| + Bₛᵤₘ/d` of `log Σⱼ exp(toReal lⱼ)`: the outer `Float.log`
    rounding (`log_error`) plus log's Lipschitzness (`abs_log_sub_le`, a73) applied to the sum-of-exps error
    `Bₛᵤₘ` (`sumIdxFGo_error` over `expApprox_error` summands, a64's core). `d ≤ Dꜰ, Σexp` is the denominator
    floor (the sum of positive exponentials, `≳ 1`), supplied as a hypothesis.
  • `lseF_error_spec` — the same, stated against the library's `Real.log (Net.softmaxDenom …)` (the softmax
    denominator's log — exactly the log-normalizer in `log π = lₐ − log Z`).
  • `logpF` / `logpF_error` — the action log-prob `logπ(a) = lₐ − logSumExp(l)` (the AD tape's own `newLogp`):
    the outer subtraction rounding (`sub_error`) composed with the `logSumExp` error (`lseF_error`), landing
    on the ℝ log-softmax of the actual logits `toReal lₐ − log Σⱼ exp(toReal lⱼ)`.
  • `logpF_ideal_error` — the actor-side end-to-end: `logpF` within its budget `+ 2ε` of `log softmax(b)ₐ`
    for any reference logits `b` within `ε` of the actual logits (`log_softmax_eq` bridges to `log softmax`,
    then a73's `logSoftmax_perturb` — the log-policy is 2-Lipschitz). Instantiate `b := idealLogit`,
    `ε :=` the forward-pass logit error, for the full running-policy log-prob vs the ideal.
  • `logpF_forward_ideal_error` — that instantiation made concrete: `logpF` over the ACTUAL forward logits
    `(forwardAll p obs).2.2` vs the ideal log-policy `log softmax(idealLogit)ₐ`, discharging `ε` from
    `forwardAll_logit_error` (a71). The actor mirror of `policyProbs_error` (a72), on the log scale.

Axiom-clean beyond the trusted Float (1+δ) base (`exp/log/add/mul` models via `expApprox_error`/`log_error`).
-/
import Puffer.RL.SoftmaxBound
import Puffer.RL.LogPolicyBound
import Puffer.RL.ActorCriticBound

namespace Puffer.RL.LogSumExpBound

open Puffer.FloatR
open Puffer.RL.SoftmaxBound (sumIdxFGo sumIdxErrBnd sumIdxFGo_error sumIdxRGo_eq_sum)
open Puffer.RL.LogPolicyBound (abs_log_sub_le log_softmax_eq logSoftmax_perturb)
open Puffer.Net (softmax softmaxDenom softmaxDenom_pos softmaxDenom_shift)
open Puffer.RL.NNTrain
open Puffer.RL.ActorCriticBound (idealLogit)
open Puffer.RL.ForwardExec (forwardAll_logit_error hRList)

/-- The Float log-normalizer `logSumExp(l) = Float.log (Σⱼ Float.exp lⱼ)` (running sum of rounded exps). -/
def lseF (l : Nat → Float) (n : Nat) : Float :=
  Float.log (sumIdxFGo (fun k => Float.exp (l k)) 0.0 (List.range n))

/-- **Log-sum-exp dominates each element.** For any logit index `i ∈ s`, the logit `l i` is at most the
    log-normalizer `log Σⱼ exp lⱼ = log(softmaxDenom s l)` — the log-sum-exp (`lseF`'s ℝ meaning) is an upper
    bound for every individual logit. Equivalently the log-softmax `log π(i) = lᵢ − log Z ≤ 0`, i.e. every softmax
    probability is `≤ 1`. Immediate from `exp lᵢ ≤ Σⱼ exp lⱼ` (`Finset.single_le_sum`) and `log`'s monotonicity. -/
theorem logit_le_log_softmaxDenom {ι : Type*} (s : Finset ι) (l : ι → ℝ) (i : ι) (hi : i ∈ s) :
    l i ≤ Real.log (softmaxDenom s l) := by
  have hle : Real.exp (l i) ≤ softmaxDenom s l :=
    Finset.single_le_sum (f := fun j => Real.exp (l j)) (fun j _ => (Real.exp_pos _).le) hi
  calc l i = Real.log (Real.exp (l i)) := (Real.log_exp _).symm
    _ ≤ Real.log (softmaxDenom s l) := Real.log_le_log (Real.exp_pos _) hle

/-- **Log-sum-exp shift-equivariance.** Adding a constant `c` to every logit shifts the log-normalizer
    by exactly `c`: `log Σⱼ exp(lⱼ + c) = c + log Σⱼ exp lⱼ`, i.e.
    `Real.log (softmaxDenom s (l + c)) = c + Real.log (softmaxDenom s l)`. This is the log-sum-exp
    companion of `softmaxDenom_shift` (which scales the denominator by `eᶜ`): the log-normalizer is
    shift-EQUIVARIANT, absorbing the constant additively. It is exactly why the numerically-stable
    `logSumExp` may subtract the max logit `m` before exponentiating and add it back afterwards —
    `log Σⱼ exp lⱼ = m + log Σⱼ exp(lⱼ − m)` — and, since the log-softmax `log π(a) = lₐ − log Z` sees
    the same `+c` on both terms, it is why the log-policy (like the softmax) is shift-invariant. The
    nonemptiness hypothesis is load-bearing: on `s = ∅` both denominators are `0`, so the LHS is
    `log 0 = 0` while the RHS is `c + log 0 = c`, false for `c ≠ 0`. -/
theorem log_softmaxDenom_shift {ι : Type*} (s : Finset ι) (l : ι → ℝ) (c : ℝ)
    (hne : s.Nonempty) :
    Real.log (softmaxDenom s (fun j => l j + c)) = c + Real.log (softmaxDenom s l) := by
  rw [softmaxDenom_shift, Real.log_mul (Real.exp_ne_zero c)
      (ne_of_gt (softmaxDenom_pos s l hne)), Real.log_exp]

/-- **Log-sum-exp accuracy.** The Float `logSumExp` is within `logEps·|log Dꜰ| + Bₛᵤₘ/d` of the ℝ
    `log Σⱼ exp(toReal lⱼ)` — `log_error` (the outer `Float.log` rounding) + `abs_log_sub_le` (log's
    Lipschitzness on the sum-of-exps error `Bₛᵤₘ`, floored by `d`). `d ≤ Dꜰ, Σexp` is the denominator floor
    (the sum of positive exponentials, `≳ 1`), supplied as a hypothesis. -/
theorem lseF_error (l : Nat → Float) (n : Nat) (d : ℝ) (hd : 0 < d)
    (hden : d ≤ toReal (sumIdxFGo (fun k => Float.exp (l k)) 0.0 (List.range n)))
    (hRden : d ≤ ∑ j ∈ Finset.range n, Real.exp (toReal (l j))) :
    |toReal (lseF l n) - Real.log (∑ j ∈ Finset.range n, Real.exp (toReal (l j)))|
      ≤ logEps * |Real.log (toReal (sumIdxFGo (fun k => Float.exp (l k)) 0.0 (List.range n)))|
        + (sumIdxErrBnd (fun k => Float.exp (l k)) (fun k => expEps * Real.exp (toReal (l k))) 0.0
            (List.range n) 0) / d := by
  set DF := sumIdxFGo (fun k => Float.exp (l k)) 0.0 (List.range n) with hDF
  set DR := ∑ j ∈ Finset.range n, Real.exp (toReal (l j)) with hDR
  have hDFpos : 0 < toReal DF := lt_of_lt_of_le hd hden
  have h0acc : |toReal (0.0 : Float) - (0 : ℝ)| ≤ 0 := by rw [toReal_zeroLit]; simp
  have hsum : |toReal DF - DR|
      ≤ sumIdxErrBnd (fun k => Float.exp (l k)) (fun k => expEps * Real.exp (toReal (l k))) 0.0
          (List.range n) 0 := by
    have h := sumIdxFGo_error (fun k => Float.exp (l k)) (fun k => Real.exp (toReal (l k)))
      (fun k => expEps * Real.exp (toReal (l k))) (List.range n) 0.0 0 0 h0acc
      (fun k _ => by simpa using expApprox_error (l k) (toReal (l k)) 0 (by simp))
    rw [hDF, hDR]; rwa [sumIdxRGo_eq_sum] at h
  have hlog1 : |toReal (Float.log DF) - Real.log (toReal DF)| ≤ logEps * |Real.log (toReal DF)| :=
    log_error DF hDFpos
  have hlog2 : |Real.log (toReal DF) - Real.log DR| ≤ |toReal DF - DR| / d :=
    abs_log_sub_le (toReal DF) DR d hd hden hRden
  have hdiv : |toReal DF - DR| / d
      ≤ (sumIdxErrBnd (fun k => Float.exp (l k)) (fun k => expEps * Real.exp (toReal (l k))) 0.0
          (List.range n) 0) / d := by gcongr
  calc |toReal (lseF l n) - Real.log DR|
      ≤ |toReal (Float.log DF) - Real.log (toReal DF)| + |Real.log (toReal DF) - Real.log DR| :=
        abs_sub_le _ _ _
    _ ≤ logEps * |Real.log (toReal DF)|
        + (sumIdxErrBnd (fun k => Float.exp (l k)) (fun k => expEps * Real.exp (toReal (l k))) 0.0
            (List.range n) 0) / d := by
        have := hlog2.trans hdiv; linarith

/-- **Log-sum-exp accuracy, against `Net.softmaxDenom`.** The Float `logSumExp` is within the bound of
    `Real.log (Puffer.Net.softmaxDenom (Finset.range n) (toReal ∘ l))` — the log of the softmax denominator,
    i.e. exactly the log-normalizer `log Z` in `log π(a) = lₐ − log Z`. -/
theorem lseF_error_spec (l : Nat → Float) (n : Nat) (d : ℝ) (hd : 0 < d)
    (hden : d ≤ toReal (sumIdxFGo (fun k => Float.exp (l k)) 0.0 (List.range n)))
    (hRden : d ≤ Puffer.Net.softmaxDenom (Finset.range n) (fun j => toReal (l j))) :
    |toReal (lseF l n) - Real.log (Puffer.Net.softmaxDenom (Finset.range n) (fun j => toReal (l j)))|
      ≤ logEps * |Real.log (toReal (sumIdxFGo (fun k => Float.exp (l k)) 0.0 (List.range n)))|
        + (sumIdxErrBnd (fun k => Float.exp (l k)) (fun k => expEps * Real.exp (toReal (l k))) 0.0
            (List.range n) 0) / d := by
  have hspec : Puffer.Net.softmaxDenom (Finset.range n) (fun j => toReal (l j))
      = ∑ j ∈ Finset.range n, Real.exp (toReal (l j)) := rfl
  rw [hspec]
  exact lseF_error l n d hd hden (hspec ▸ hRden)

/-! ### The action log-prob `log π(a) = lₐ − logSumExp(l)` -/

/-- Executable log-policy `log π(a) = lₐ − logSumExp(l)` — the AD-tape's action log-prob. -/
def logpF (l : Nat → Float) (a n : Nat) : Float := l a - lseF l n

/-- **Log-policy runtime error.** The running `lₐ − logSumExp(l)` deviates from the ℝ log-softmax of the
    ACTUAL logits `toReal lₐ − log Σⱼ exp(toReal lⱼ)` by at most the outer subtraction rounding plus the
    `logSumExp` error (a82). (The logits are exact `Float` inputs; the forward-pass logit error composes
    separately, like a82.) -/
theorem logpF_error (l : Nat → Float) (a n : Nat) (d : ℝ) (hd : 0 < d)
    (hden : d ≤ toReal (sumIdxFGo (fun k => Float.exp (l k)) 0.0 (List.range n)))
    (hRden : d ≤ ∑ j ∈ Finset.range n, Real.exp (toReal (l j))) :
    |toReal (logpF l a n) - (toReal (l a) - Real.log (∑ j ∈ Finset.range n, Real.exp (toReal (l j))))|
      ≤ u64 * |toReal (l a) - toReal (lseF l n)|
        + (logEps * |Real.log (toReal (sumIdxFGo (fun k => Float.exp (l k)) 0.0 (List.range n)))|
            + (sumIdxErrBnd (fun k => Float.exp (l k)) (fun k => expEps * Real.exp (toReal (l k))) 0.0
                (List.range n) 0) / d) := by
  have hsub : |toReal (l a - lseF l n) - (toReal (l a) - toReal (lseF l n))|
      ≤ u64 * |toReal (l a) - toReal (lseF l n)| := sub_error (l a) (lseF l n)
  have hlse := lseF_error l n d hd hden hRden
  calc |toReal (logpF l a n)
          - (toReal (l a) - Real.log (∑ j ∈ Finset.range n, Real.exp (toReal (l j))))|
      ≤ |toReal (l a - lseF l n) - (toReal (l a) - toReal (lseF l n))|
        + |Real.log (∑ j ∈ Finset.range n, Real.exp (toReal (l j))) - toReal (lseF l n)| := by
        rw [logpF]
        have hsplit : toReal (l a - lseF l n)
              - (toReal (l a) - Real.log (∑ j ∈ Finset.range n, Real.exp (toReal (l j))))
            = (toReal (l a - lseF l n) - (toReal (l a) - toReal (lseF l n)))
              + (Real.log (∑ j ∈ Finset.range n, Real.exp (toReal (l j))) - toReal (lseF l n)) := by ring
        rw [hsplit]; exact abs_add_le _ _
    _ ≤ _ := add_le_add hsub (by rw [abs_sub_comm]; exact hlse)

/-! ### Action log-prob vs the ideal log-policy (forward-logit perturbation) -/

/-- **Action log-prob vs the ideal log-policy.** The runnable `logpF l a n` (`lₐ − logSumExp`) is within its
    `logSumExp`-rounding budget (a108) `+ 2ε` of `log softmax(b)ₐ` for any reference logits `b` within `ε` of
    the actual logits' `toReal` — via `log_softmax_eq` (bridge to `log softmax`) + `logSoftmax_perturb`
    (log-policy 2-Lipschitz, a73). Instantiate `b := idealLogit`, `ε :=` the forward-pass logit error. -/
theorem logpF_ideal_error (l : Nat → Float) (a n : Nat) (b : Nat → ℝ) (ε : ℝ)
    (d : ℝ) (hd : 0 < d)
    (hden : d ≤ toReal (sumIdxFGo (fun k => Float.exp (l k)) 0.0 (List.range n)))
    (hRden : d ≤ ∑ j ∈ Finset.range n, Real.exp (toReal (l j)))
    (hne : (Finset.range n).Nonempty) (ha : a < n)
    (hab : ∀ k, k < n → |toReal (l k) - b k| ≤ ε) :
    |toReal (logpF l a n) - Real.log (softmax (Finset.range n) b a)|
      ≤ (u64 * |toReal (l a) - toReal (lseF l n)|
          + (logEps * |Real.log (toReal (sumIdxFGo (fun k => Float.exp (l k)) 0.0 (List.range n)))|
              + (sumIdxErrBnd (fun k => Float.exp (l k)) (fun k => expEps * Real.exp (toReal (l k))) 0.0
                  (List.range n) 0) / d)) + 2 * ε := by
  -- a108: logpF vs the log-softmax of the ACTUAL logits (rewritten via log_softmax_eq)
  have hden_eq : (∑ j ∈ Finset.range n, Real.exp (toReal (l j)))
      = softmaxDenom (Finset.range n) (fun k => toReal (l k)) := rfl
  have h108 := logpF_error l a n d hd hden hRden
  rw [hden_eq, ← log_softmax_eq (Finset.range n) (fun k => toReal (l k)) a hne] at h108
  -- a73: log-softmax perturbation from the actual to the reference logits
  have hpert : |Real.log (softmax (Finset.range n) (fun k => toReal (l k)) a)
        - Real.log (softmax (Finset.range n) b a)| ≤ 2 * ε :=
    logSoftmax_perturb (Finset.range n) (fun k => toReal (l k)) b a ε hne (Finset.mem_range.mpr ha)
      (fun k hk => hab k (Finset.mem_range.mp hk))
  calc |toReal (logpF l a n) - Real.log (softmax (Finset.range n) b a)|
      ≤ |toReal (logpF l a n) - Real.log (softmax (Finset.range n) (fun k => toReal (l k)) a)|
        + |Real.log (softmax (Finset.range n) (fun k => toReal (l k)) a)
            - Real.log (softmax (Finset.range n) b a)| := abs_sub_le _ _ _
    _ ≤ _ := add_le_add h108 hpert

/-! ### Concrete: the running policy log-prob on the actual forward net -/

/-- **Running-policy log-prob on the actual net vs the ideal.** The AD tape's `newLogp` computed over the
    ACTUAL forward logits — `logpF (fun k => (forwardAll p obs).2.2[k]!) a n` — is within its logSumExp
    rounding budget `+ 2ε` of the ideal log-policy `log softmax(idealLogit)ₐ`, `ε` a uniform forward-pass
    logit-error bound. The actor mirror of `policyProbs_error` (a72), on the log scale. -/
theorem logpF_forward_ideal_error (p : MLP) (obs : Array Float) (a n : Nat) (ε : ℝ)
    (d : ℝ) (hd : 0 < d)
    (hden : d ≤ toReal (sumIdxFGo (fun k => Float.exp ((forwardAll p obs).2.2[k]!)) 0.0 (List.range n)))
    (hRden : d ≤ ∑ j ∈ Finset.range n, Real.exp (toReal ((forwardAll p obs).2.2[j]!)))
    (hne : (Finset.range n).Nonempty) (ha : a < n) (hn : n ≤ p.b2.size)
    (hεbnd : ∀ k, k < n →
      z1ErrBnd (p.W2[k]!).toList p.b2[k]! ((forwardAll p obs).2.1).toList
        + dotDiffBnd (p.W2[k]!).toList ((forwardAll p obs).2.1).toList (hRList p obs) ≤ ε) :
    |toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n)
        - Real.log (softmax (Finset.range n) (idealLogit p obs) a)|
      ≤ (u64 * |toReal ((forwardAll p obs).2.2[a]!)
              - toReal (lseF (fun k => (forwardAll p obs).2.2[k]!) n)|
          + (logEps * |Real.log (toReal (sumIdxFGo (fun k => Float.exp ((forwardAll p obs).2.2[k]!)) 0.0
                (List.range n)))|
              + (sumIdxErrBnd (fun k => Float.exp ((forwardAll p obs).2.2[k]!))
                  (fun k => expEps * Real.exp (toReal ((forwardAll p obs).2.2[k]!))) 0.0
                  (List.range n) 0) / d)) + 2 * ε := by
  refine logpF_ideal_error (fun k => (forwardAll p obs).2.2[k]!) a n (idealLogit p obs) ε
    d hd hden hRden hne ha ?_
  intro k hk
  exact (forwardAll_logit_error p obs k (by omega)).trans (hεbnd k hk)

end Puffer.RL.LogSumExpBound
