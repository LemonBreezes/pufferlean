/-
The PPO importance ratio on the ACTUAL forward net vs the ideal — the actor's ratio story closed
end-to-end on the runnable network.

`ratioFullF_ideal_error` (a115, `PPORuntime`) bounds the trainer's ratio `exp(newLogp − oldLogp)` against
the ideal ratio `exp(logpIdeal − oldLogp)` given `newLogp` is within `ε` of the ideal log-policy; it left `ε`
abstract. `logpF_forward_ideal_error` (a110, `LogSumExpBound`) discharges exactly that `ε` for the concrete
`newLogp = logpF (fun k => (forwardAll p obs).2.2[k]!) a n` — the AD tape's log-prob over the real forward
logits — landing on `log softmax(idealLogit)ₐ` with the explicit budget `logSumExp-rounding + 2ε_logit`.

  • `logpFwdBnd` — that forward-net log-prob total error budget (logSumExp rounding + `2·ε` forward-logit
    error), named so the ratio bound stays readable.
  • `ratioFullF_forward_ideal_error` — the composition: the running ratio over the ACTUAL forward net's
    log-prob is within a115's `exp`-rounding + `exp`-perturbation budget of the ideal ratio
    `exp(log softmax(idealLogit)ₐ − oldLogp)`, the `exp`-Lipschitz factor now carrying the concrete
    `logpFwdBnd`. The ratio-scale mirror of `policyProbs_error` (a72) on the actual net.

Together with the value-side `valueSqLossF_forward_error` (a106), both PPO heads — actor ratio and critic
value loss — now have a runnable-net-vs-ideal accuracy bound. Axiom-clean (a110 ⊕ a115 footprint: the Float
base + `exp`/`log`/`add`/`sub`/`mul` models).
-/
import Puffer.RL.PPORuntime
import Puffer.RL.LogSumExpBound

namespace Puffer.RL.RatioForward

open Puffer.FloatR
open Puffer.RL.SoftmaxBound (sumIdxFGo sumIdxErrBnd)
open Puffer.Net (softmax)
open Puffer.RL.NNTrain (MLP forwardAll)
open Puffer.RL.ActorCriticBound (idealLogit)
open Puffer.RL.LogSumExpBound (logpF lseF logpF_forward_ideal_error)
open Puffer.RL.PPORuntime (ratioFullF ratioFullF_ideal_error)
open Puffer.FloatR (z1ErrBnd dotDiffBnd)
open Puffer.RL.ForwardExec (hRList)

/-- The forward-net log-prob total error budget: the logSumExp rounding budget (a108/a110) plus `2·ε`
    for the forward-pass logit error. -/
noncomputable def logpFwdBnd (p : MLP) (obs : Array Float) (a n : Nat) (d ε : ℝ) : ℝ :=
  u64 * |toReal ((forwardAll p obs).2.2[a]!)
          - toReal (lseF (fun k => (forwardAll p obs).2.2[k]!) n)|
    + (logEps * |Real.log (toReal (sumIdxFGo (fun k => Float.exp ((forwardAll p obs).2.2[k]!)) 0.0
            (List.range n)))|
        + (sumIdxErrBnd (fun k => Float.exp ((forwardAll p obs).2.2[k]!))
            (fun k => expEps * Real.exp (toReal ((forwardAll p obs).2.2[k]!))) 0.0
            (List.range n) 0) / d) + 2 * ε

/-- **Running PPO ratio on the actual forward net vs the ideal ratio.** The ratio the trainer computes from
    the AD tape's log-prob over the REAL forward logits — `ratioFullF (logpF (forwardAll p obs).2.2 a n)
    oldLogp` — is within a115's `exp`-rounding + `exp`-perturbation budget of the ideal ratio
    `exp(log softmax(idealLogit)ₐ − oldLogp)`, the `exp`-Lipschitz factor carrying the concrete forward
    log-prob budget `logpFwdBnd` (a110's logSumExp rounding + `2ε_logit`). Composes `logpF_forward_ideal_error`
    (a110) into `ratioFullF_ideal_error` (a115). -/
theorem ratioFullF_forward_ideal_error (p : MLP) (obs : Array Float) (oldLogp : Float) (a n : Nat)
    (ε d : ℝ) (hd : 0 < d)
    (hden : d ≤ toReal (sumIdxFGo (fun k => Float.exp ((forwardAll p obs).2.2[k]!)) 0.0 (List.range n)))
    (hRden : d ≤ ∑ j ∈ Finset.range n, Real.exp (toReal ((forwardAll p obs).2.2[j]!)))
    (hne : (Finset.range n).Nonempty) (ha : a < n) (hn : n ≤ p.b2.size)
    (hεbnd : ∀ k, k < n →
      z1ErrBnd (p.W2[k]!).toList p.b2[k]! ((forwardAll p obs).2.1).toList
        + dotDiffBnd (p.W2[k]!).toList ((forwardAll p obs).2.1).toList (hRList p obs) ≤ ε) :
    |toReal (ratioFullF (logpF (fun k => (forwardAll p obs).2.2[k]!) a n) oldLogp)
        - Real.exp (Real.log (softmax (Finset.range n) (idealLogit p obs) a) - toReal oldLogp)|
      ≤ (expEps * Real.exp (toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp))
          + Real.exp (toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp))
              * (Real.exp (u64 * |toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n)
                    - toReal oldLogp|) - 1))
        + Real.exp (Real.log (softmax (Finset.range n) (idealLogit p obs) a) - toReal oldLogp)
            * (Real.exp (logpFwdBnd p obs a n d ε) - 1) := by
  have hb : |toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n)
        - Real.log (softmax (Finset.range n) (idealLogit p obs) a)| ≤ logpFwdBnd p obs a n d ε :=
    logpF_forward_ideal_error p obs a n ε d hd hden hRden hne ha hn hεbnd
  exact ratioFullF_ideal_error (logpF (fun k => (forwardAll p obs).2.2[k]!) a n) oldLogp
    (Real.log (softmax (Finset.range n) (idealLogit p obs) a)) (logpFwdBnd p obs a n d ε) hb

end Puffer.RL.RatioForward
