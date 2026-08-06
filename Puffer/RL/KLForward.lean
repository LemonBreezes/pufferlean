/-
The Schulman approximate-KL early-stopping estimators on the ACTUAL forward net vs the ideal — the metric
side of the per-minibatch computation, made concrete (a123 left the log-ratio error `ε` abstract; this
discharges it from the forward pass, exactly as a119 made a115's ratio bound concrete).

The trainer's log-ratio is `lr = newLogp − oldLogp` (a Float sub) where `newLogp = logpF (forwardAll…)` — the
AD tape's log-prob over the real forward logits. Its distance from the ideal log-ratio
`log softmax(idealLogit)ₐ − oldLogp` splits into the `newLogp − oldLogp` sub-rounding (`sub_error`, the
`oldLogp` term cancels in the reference) plus the forward log-prob budget `logpFwdBnd` (a110):

  • `logRatioFwdBnd` / `logRatio_forward_ideal_error` — the forward log-ratio total error and the bound it
    satisfies (`sub_error` ⊕ `logpF_forward_ideal_error`).
  • `approxKLOldF_forward_ideal_error` — the old-KL estimator `−lr` over the forward log-ratio is within just
    `logRatioFwdBnd` of the ideal (the estimator is exact, so no rounding added; discharges a123's
    `approxKLOldF_ideal_error`).
  • `approxKLNewF_forward_ideal_error` — the k3 estimator `(exp lr − 1) − lr` over the forward log-ratio is
    within its rounding budget `+ exp(idealLogRatio)·(e^{logRatioFwdBnd} − 1) + logRatioFwdBnd` of the ideal
    (discharges a123's `approxKLNewF_ideal_error`).

Axiom-clean (a123 ⊕ a110 footprint: the Float base + exp/log/add/sub/mul models + the `toReal_*` projections).
-/
import Puffer.RL.RatioForward

namespace Puffer.RL.KLForward

open Puffer.FloatR
open Puffer.Net (softmax)
open Puffer.RL.NNTrain (MLP forwardAll)
open Puffer.RL.ActorCriticBound (idealLogit)
open Puffer.RL.LogSumExpBound (logpF logpF_forward_ideal_error)
open Puffer.RL.PPORuntime (approxKLOldF approxKLNewF approxKLNewErrBnd
  approxKLOldF_ideal_error approxKLNewF_ideal_error
  approxKLOldF_error approxKLNewF_error abs_exp_sub_exp_le)
open Puffer.RL.PPO (approxKLOld approxKLNew)
open Puffer.RL.RatioForward (logpFwdBnd)
open Puffer.FloatR (z1ErrBnd dotDiffBnd)
open Puffer.RL.ForwardExec (hRList)

/-- The forward-net log-ratio total error: the `newLogp − oldLogp` Float-sub rounding + the forward log-prob
    budget `logpFwdBnd` (a110). -/
noncomputable def logRatioFwdBnd (p : MLP) (obs : Array Float) (oldLogp : Float) (a n : Nat) (ε d : ℝ) : ℝ :=
  u64 * |toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n) - toReal oldLogp|
    + logpFwdBnd p obs a n d ε

/-- **Forward log-ratio vs the ideal.** The trainer's log-ratio `logpF (forwardAll…) − oldLogp` is within
    `logRatioFwdBnd` of the ideal `log softmax(idealLogit)ₐ − oldLogp` — the `newLogp − oldLogp` sub-rounding
    (`sub_error`; `oldLogp` cancels in the reference) plus the forward log-prob budget `logpFwdBnd` (a110). -/
theorem logRatio_forward_ideal_error (p : MLP) (obs : Array Float) (oldLogp : Float) (a n : Nat)
    (ε d : ℝ) (hd : 0 < d)
    (hden : d ≤ toReal (Puffer.RL.SoftmaxBound.sumIdxFGo
      (fun k => Float.exp ((forwardAll p obs).2.2[k]!)) 0.0 (List.range n)))
    (hRden : d ≤ ∑ j ∈ Finset.range n, Real.exp (toReal ((forwardAll p obs).2.2[j]!)))
    (hne : (Finset.range n).Nonempty) (ha : a < n) (hn : n ≤ p.b2.size)
    (hεbnd : ∀ k, k < n →
      z1ErrBnd (p.W2[k]!).toList p.b2[k]! ((forwardAll p obs).2.1).toList
        + dotDiffBnd (p.W2[k]!).toList ((forwardAll p obs).2.1).toList (hRList p obs) ≤ ε) :
    |toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp)
        - (Real.log (softmax (Finset.range n) (idealLogit p obs) a) - toReal oldLogp)|
      ≤ logRatioFwdBnd p obs oldLogp a n ε d := by
  have hsub := sub_error (logpF (fun k => (forwardAll p obs).2.2[k]!) a n) oldLogp
  have hlp : |toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n)
        - Real.log (softmax (Finset.range n) (idealLogit p obs) a)| ≤ logpFwdBnd p obs a n d ε :=
    logpF_forward_ideal_error p obs a n ε d hd hden hRden hne ha hn hεbnd
  rw [logRatioFwdBnd]
  calc |toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp)
          - (Real.log (softmax (Finset.range n) (idealLogit p obs) a) - toReal oldLogp)|
      ≤ |toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp)
            - (toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n) - toReal oldLogp)|
        + |(toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n) - toReal oldLogp)
            - (Real.log (softmax (Finset.range n) (idealLogit p obs) a) - toReal oldLogp)| := abs_sub_le _ _ _
    _ ≤ _ := by
        refine add_le_add hsub ?_
        rw [show (toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n) - toReal oldLogp)
              - (Real.log (softmax (Finset.range n) (idealLogit p obs) a) - toReal oldLogp)
            = toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n)
              - Real.log (softmax (Finset.range n) (idealLogit p obs) a) by ring]
        exact hlp

/-- **Old-KL estimator on the forward net vs the ideal.** The old-KL estimator `−lr` over the forward
    log-ratio is within just `logRatioFwdBnd` of `approxKLOld(idealLogRatio)` — the estimator is exact, so no
    rounding is added. Discharges a123's `approxKLOldF_ideal_error` with the forward log-ratio bound. -/
theorem approxKLOldF_forward_ideal_error (p : MLP) (obs : Array Float) (oldLogp : Float) (a n : Nat)
    (ε d : ℝ) (hd : 0 < d)
    (hden : d ≤ toReal (Puffer.RL.SoftmaxBound.sumIdxFGo
      (fun k => Float.exp ((forwardAll p obs).2.2[k]!)) 0.0 (List.range n)))
    (hRden : d ≤ ∑ j ∈ Finset.range n, Real.exp (toReal ((forwardAll p obs).2.2[j]!)))
    (hne : (Finset.range n).Nonempty) (ha : a < n) (hn : n ≤ p.b2.size)
    (hεbnd : ∀ k, k < n →
      z1ErrBnd (p.W2[k]!).toList p.b2[k]! ((forwardAll p obs).2.1).toList
        + dotDiffBnd (p.W2[k]!).toList ((forwardAll p obs).2.1).toList (hRList p obs) ≤ ε) :
    |toReal (approxKLOldF (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp))
        - approxKLOld (Real.log (softmax (Finset.range n) (idealLogit p obs) a) - toReal oldLogp)|
      ≤ logRatioFwdBnd p obs oldLogp a n ε d :=
  approxKLOldF_ideal_error (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp)
    (Real.log (softmax (Finset.range n) (idealLogit p obs) a) - toReal oldLogp) _
    (logRatio_forward_ideal_error p obs oldLogp a n ε d hd hden hRden hne ha hn hεbnd)

/-- **k3 KL estimator on the forward net vs the ideal.** The Schulman k3 estimator `(exp lr − 1) − lr` over
    the forward log-ratio is within its own rounding budget `approxKLNewErrBnd` plus the `exp`-Lipschitz
    perturbation `exp(idealLogRatio)·(e^{logRatioFwdBnd} − 1) + logRatioFwdBnd` of `approxKLNew(idealLogRatio)`.
    Discharges a123's `approxKLNewF_ideal_error` with the forward log-ratio bound. -/
theorem approxKLNewF_forward_ideal_error (p : MLP) (obs : Array Float) (oldLogp : Float) (a n : Nat)
    (ε d : ℝ) (hd : 0 < d)
    (hden : d ≤ toReal (Puffer.RL.SoftmaxBound.sumIdxFGo
      (fun k => Float.exp ((forwardAll p obs).2.2[k]!)) 0.0 (List.range n)))
    (hRden : d ≤ ∑ j ∈ Finset.range n, Real.exp (toReal ((forwardAll p obs).2.2[j]!)))
    (hne : (Finset.range n).Nonempty) (ha : a < n) (hn : n ≤ p.b2.size)
    (hεbnd : ∀ k, k < n →
      z1ErrBnd (p.W2[k]!).toList p.b2[k]! ((forwardAll p obs).2.1).toList
        + dotDiffBnd (p.W2[k]!).toList ((forwardAll p obs).2.1).toList (hRList p obs) ≤ ε) :
    |toReal (approxKLNewF (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp))
        - approxKLNew (Real.log (softmax (Finset.range n) (idealLogit p obs) a) - toReal oldLogp)|
      ≤ approxKLNewErrBnd (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp)
        + (Real.exp (Real.log (softmax (Finset.range n) (idealLogit p obs) a) - toReal oldLogp)
              * (Real.exp (logRatioFwdBnd p obs oldLogp a n ε d) - 1)
            + logRatioFwdBnd p obs oldLogp a n ε d) :=
  approxKLNewF_ideal_error (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp)
    (Real.log (softmax (Finset.range n) (idealLogit p obs) a) - toReal oldLogp) _
    (logRatio_forward_ideal_error p obs oldLogp a n ε d hd hden hRden hne ha hn hεbnd)

/-- **Differencing the two Schulman KL estimators on the forward net recovers the ideal importance
    ratio minus one.** PufferLib runs both single-sample KL estimators over the AD tape's forward
    log-ratio `lr = logpF(forwardAll…) − oldLogp`: the old `approxKLOld = −lr` and the k3
    `approxKLNew = (exp lr − 1) − lr`. Their difference is algebraically `exp lr − 1`, i.e. the
    importance ratio minus one. This says the Float-computed difference
    `toReal(approxKLNewF lr) − toReal(approxKLOldF lr)` recovers the *ideal* ratio-minus-one
    `exp(log softmax(idealLogit)ₐ − oldLogp) − 1 = π_ideal(a)/e^{oldLogp} − 1` up to the k3 rounding
    budget `approxKLNewErrBnd` plus the `exp`-Lipschitz forward budget
    `exp(idealLR)·(e^{logRatioFwdBnd} − 1)` carrying the concrete forward log-ratio error
    `logRatioFwdBnd`. Composes `approxKLNewF_error`/`approxKLOldF_error` (the estimator rounding) with
    `logRatio_forward_ideal_error` through `abs_exp_sub_exp_le`. The forward hypotheses are
    load-bearing: they gate `logRatio_forward_ideal_error`, without which the `exp`-difference term is
    unbounded. -/
theorem approxKL_diff_forward_ideal_error (p : MLP) (obs : Array Float) (oldLogp : Float) (a n : Nat)
    (ε d : ℝ) (hd : 0 < d)
    (hden : d ≤ toReal (Puffer.RL.SoftmaxBound.sumIdxFGo
      (fun k => Float.exp ((forwardAll p obs).2.2[k]!)) 0.0 (List.range n)))
    (hRden : d ≤ ∑ j ∈ Finset.range n, Real.exp (toReal ((forwardAll p obs).2.2[j]!)))
    (hne : (Finset.range n).Nonempty) (ha : a < n) (hn : n ≤ p.b2.size)
    (hεbnd : ∀ k, k < n →
      z1ErrBnd (p.W2[k]!).toList p.b2[k]! ((forwardAll p obs).2.1).toList
        + dotDiffBnd (p.W2[k]!).toList ((forwardAll p obs).2.1).toList (hRList p obs) ≤ ε) :
    |(toReal (approxKLNewF (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp))
          - toReal (approxKLOldF (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp)))
        - (Real.exp (Real.log (softmax (Finset.range n) (idealLogit p obs) a) - toReal oldLogp) - 1)|
      ≤ approxKLNewErrBnd (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp)
        + Real.exp (Real.log (softmax (Finset.range n) (idealLogit p obs) a) - toReal oldLogp)
            * (Real.exp (logRatioFwdBnd p obs oldLogp a n ε d) - 1) := by
  -- the old estimator is exactly `−lr` (no rounding); the k3 estimator rounds against `approxKLNew`
  have hold : toReal (approxKLOldF (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp))
      = -(toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp)) := by
    rw [approxKLOldF_error, approxKLOld]
  have hnew := approxKLNewF_error (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp)
  -- Step A: the Float difference recovers `exp(toReal lr) − 1` up to the k3 rounding budget
  have hA : |(toReal (approxKLNewF (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp))
          - toReal (approxKLOldF (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp)))
        - (Real.exp (toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp)) - 1)|
      ≤ approxKLNewErrBnd (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp) := by
    have hrw : (toReal (approxKLNewF (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp))
            - toReal (approxKLOldF (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp)))
          - (Real.exp (toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp)) - 1)
        = toReal (approxKLNewF (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp))
          - approxKLNew (toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp)) := by
      rw [hold, approxKLNew]; ring
    rw [hrw]; exact hnew
  -- Step B: the concrete forward log-ratio error, gating the `exp` perturbation
  have hlr : |toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp)
        - (Real.log (softmax (Finset.range n) (idealLogit p obs) a) - toReal oldLogp)|
      ≤ logRatioFwdBnd p obs oldLogp a n ε d :=
    logRatio_forward_ideal_error p obs oldLogp a n ε d hd hden hRden hne ha hn hεbnd
  have hexp : |Real.exp (toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp))
        - Real.exp (Real.log (softmax (Finset.range n) (idealLogit p obs) a) - toReal oldLogp)|
      ≤ Real.exp (Real.log (softmax (Finset.range n) (idealLogit p obs) a) - toReal oldLogp)
          * (Real.exp (logRatioFwdBnd p obs oldLogp a n ε d) - 1) :=
    abs_exp_sub_exp_le _ _ _ hlr
  -- triangle inequality through the actual-net ratio-minus-one
  calc |(toReal (approxKLNewF (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp))
            - toReal (approxKLOldF (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp)))
          - (Real.exp (Real.log (softmax (Finset.range n) (idealLogit p obs) a) - toReal oldLogp) - 1)|
      ≤ |(toReal (approxKLNewF (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp))
              - toReal (approxKLOldF (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp)))
            - (Real.exp (toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp)) - 1)|
        + |(Real.exp (toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp)) - 1)
            - (Real.exp (Real.log (softmax (Finset.range n) (idealLogit p obs) a) - toReal oldLogp) - 1)| :=
        abs_sub_le _ _ _
    _ ≤ approxKLNewErrBnd (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp)
        + Real.exp (Real.log (softmax (Finset.range n) (idealLogit p obs) a) - toReal oldLogp)
            * (Real.exp (logRatioFwdBnd p obs oldLogp a n ε d) - 1) := by
        refine add_le_add hA ?_
        have hcancel : (Real.exp (toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp)) - 1)
              - (Real.exp (Real.log (softmax (Finset.range n) (idealLogit p obs) a) - toReal oldLogp) - 1)
            = Real.exp (toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp))
              - Real.exp (Real.log (softmax (Finset.range n) (idealLogit p obs) a) - toReal oldLogp) := by
          ring
        rw [hcancel]; exact hexp

end Puffer.RL.KLForward
