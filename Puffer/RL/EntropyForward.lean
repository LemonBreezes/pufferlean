/-
The discrete-policy entropy bonus on the ACTUAL forward net vs the ideal — the entropy head's runnable
accuracy bound, completing the trio (actor ratio, critic value loss, entropy) on the concrete network.

Two error layers compose (the softmax's own execution rounding stays SCOPED OUT, exactly as in
`entropyF_softmax_error` — the reference probabilities are the ideal softmax of the forward logits):

  • `entropyF_softmax_error` (a104, `EntropyRuntime`) — the running `entropyF` REDUCTION `−Σ pᵢ log pᵢ` over a
    prob function whose `toReal` is the ideal softmax of the forward logits, within its per-term `sumIdxErrBnd`
    budget of the categorical entropy `entropy (range) (toReal ∘ acLogits)`.
  • `entropy_run_perturb` (a80, `EntropyPerturb`) — that forward-logit categorical entropy is within
    `2ε + (e^{2ε}−1)·H_ideal` of the ideal-logit entropy `entropy (range) (idealLogit)`, `ε` a uniform
    forward-pass logit-error bound (from `forwardAll_logit_error`).

`entropyF_forward_ideal_error` triangles the two: the runnable entropy reduction over the forward-net policy
is within (reduction budget) + (forward perturbation) of the ideal-logit categorical entropy. The entropy
counterpart of `ratioFullF_forward_ideal_error` (a119, actor) and `valueSqLossF_forward_error` (a106, critic);
all three PPO-objective heads now carry a runnable-net-vs-ideal bound.

Axiom-clean (a104 ⊕ a80 footprint: the Float base + `add`/`mul`/`log` models + `toReal_neg`).
-/
import Puffer.RL.EntropyRuntime
import Puffer.RL.EntropyPerturb

namespace Puffer.RL.EntropyForward

open Puffer.FloatR
open Puffer.Net (softmax)
open Puffer.RL.NNTrain (MLP forwardAll)
open Puffer.RL.ActorCriticBound (idealLogit acLogits)
open Puffer.RL.SoftmaxBound (sumIdxErrBnd)
open Puffer.RL.EntropyRuntime (entropyF entTermBnd entropyF_softmax_error)
open Puffer.RL.EntropyPerturb (entropy entropy_run_perturb entropy_le_log_card entropy_nonneg)
open Puffer.FloatR (z1ErrBnd dotDiffBnd)
open Puffer.RL.ForwardExec (hRList)

/-- **Running discrete-policy entropy on the actual forward net vs the ideal.** The trainer's entropy bonus —
    the runnable reduction `entropyF pf A` over a policy `pf` whose `toReal` is the ideal softmax of the ACTUAL
    forward logits `acLogits p obs` (the softmax's own rounding scoped out, as in a104) — is within a104's
    per-term reduction budget plus `entropy_run_perturb`'s forward-logit perturbation `2ε + (e^{2ε}−1)·H_ideal`
    of the ideal-logit categorical entropy `entropy (range A) (idealLogit p obs)`. Composes a104 ⊕ a80 by a
    triangle through the forward-logit entropy. -/
theorem entropyF_forward_ideal_error (p : MLP) (obs : Array Float) (pf : Nat → Float)
    (ε : ℝ) (hpos1 : 0 < p.b2.size - 1)
    (hpsm : ∀ i, i < p.b2.size - 1 →
      toReal (pf i) = softmax (Finset.range (p.b2.size - 1))
        (fun j => toReal (acLogits p obs)[j]!) i)
    (hεbnd : ∀ k, k < p.b2.size - 1 →
      z1ErrBnd (p.W2[k]!).toList p.b2[k]! ((forwardAll p obs).2.1).toList
        + dotDiffBnd (p.W2[k]!).toList ((forwardAll p obs).2.1).toList (hRList p obs) ≤ ε) :
    |toReal (entropyF pf (p.b2.size - 1)) - entropy (Finset.range (p.b2.size - 1)) (idealLogit p obs)|
      ≤ sumIdxErrBnd (fun i => pf i * Float.log (pf i)) (entTermBnd pf) 0.0
          (List.range (p.b2.size - 1)) 0
        + (2*ε + (Real.exp (2*ε) - 1)
            * entropy (Finset.range (p.b2.size - 1)) (idealLogit p obs)) := by
  have hne : (Finset.range (p.b2.size - 1)).Nonempty := Finset.nonempty_range_iff.mpr (by omega)
  have h104 := entropyF_softmax_error pf (p.b2.size - 1)
    (fun j => toReal (acLogits p obs)[j]!) hpsm hne
  have hpert := entropy_run_perturb p obs ε hpos1 hεbnd
  calc |toReal (entropyF pf (p.b2.size - 1)) - entropy (Finset.range (p.b2.size - 1)) (idealLogit p obs)|
      ≤ |toReal (entropyF pf (p.b2.size - 1))
            - entropy (Finset.range (p.b2.size - 1)) (fun j => toReal (acLogits p obs)[j]!)|
        + |entropy (Finset.range (p.b2.size - 1)) (fun j => toReal (acLogits p obs)[j]!)
            - entropy (Finset.range (p.b2.size - 1)) (idealLogit p obs)| := abs_sub_le _ _ _
    _ ≤ _ := add_le_add h104 hpert

/-- **Forward-net entropy bonus is bounded by `log(#actions)`.** The running `entropyF` over the ACTUAL
    forward-net policy is `≤ log(A) + forward-budget` (`A = #actions = b2.size − 1`) — the max-entropy bound
    (a160) transported to the concrete network, up to a120's forward error. So the trainer's entropy term over
    the real net can never exceed `log(#actions)` beyond the forward + reduction error. -/
theorem entropyF_forward_le_log_card (p : MLP) (obs : Array Float) (pf : Nat → Float)
    (ε : ℝ) (hpos1 : 0 < p.b2.size - 1)
    (hpsm : ∀ i, i < p.b2.size - 1 →
      toReal (pf i) = softmax (Finset.range (p.b2.size - 1)) (fun j => toReal (acLogits p obs)[j]!) i)
    (hεbnd : ∀ k, k < p.b2.size - 1 →
      z1ErrBnd (p.W2[k]!).toList p.b2[k]! ((forwardAll p obs).2.1).toList
        + dotDiffBnd (p.W2[k]!).toList ((forwardAll p obs).2.1).toList (hRList p obs) ≤ ε) :
    toReal (entropyF pf (p.b2.size - 1))
      ≤ Real.log (p.b2.size - 1)
        + (sumIdxErrBnd (fun i => pf i * Float.log (pf i)) (entTermBnd pf) 0.0
            (List.range (p.b2.size - 1)) 0
          + (2*ε + (Real.exp (2*ε) - 1)
              * entropy (Finset.range (p.b2.size - 1)) (idealLogit p obs))) := by
  have hne : (Finset.range (p.b2.size - 1)).Nonempty := Finset.nonempty_range_iff.mpr (by omega)
  have herr := entropyF_forward_ideal_error p obs pf ε hpos1 hpsm hεbnd
  set B := sumIdxErrBnd (fun i => pf i * Float.log (pf i)) (entTermBnd pf) 0.0
            (List.range (p.b2.size - 1)) 0
          + (2*ε + (Real.exp (2*ε) - 1)
              * entropy (Finset.range (p.b2.size - 1)) (idealLogit p obs)) with hB
  have hmax := entropy_le_log_card (Finset.range (p.b2.size - 1)) (idealLogit p obs) hne
  rw [Finset.card_range, Nat.cast_sub (by omega : 1 ≤ p.b2.size), Nat.cast_one] at hmax
  have hb := (abs_le.mp herr).2
  linarith

/-- **Forward-net entropy bonus lower bracket: `−(forward budget) ≤ entropyF`.** The running `entropyF`
    over the ACTUAL forward-net policy `pf` can dip at most the composed forward-vs-ideal error `B` below
    `0` — i.e. `−B ≤ toReal (entropyF pf A)` (`A = #actions = b2.size − 1`). Since the ideal-logit
    categorical entropy is `≥ 0` (`entropy_nonneg`, ultimately KL/`log_le_sub_one` nonnegativity), the
    triangle bound `entropyF_forward_ideal_error` forces `entropyF ≥ H_ideal − B ≥ −B`, independent of the
    unknown `H_ideal` (which could be as large as `log A`). This is the LOWER half of the bracket whose UPPER
    half is `entropyF_forward_le_log_card` (`≤ log A + B`); together the trainer's runnable entropy term over
    the real net lives in `[−B, log A + B]`, so `entCoef·H` is essentially nonnegative on the actual forward
    pass — the forward-net counterpart of `EntropyRuntime.neg_budget_le_entropyF` (the runtime-only floor).
    All three hypotheses are load-bearing: `hpos1` (range nonemptiness, needed by both
    `entropyF_forward_ideal_error` and `entropy_nonneg`), `hpsm` (the softmax spec-bridge tying `pf` to the
    ideal forward softmax), and `hεbnd` (certifying `ε` as a genuine forward-pass logit-error bound). -/
theorem neg_bnd_le_entropyF_forward (p : MLP) (obs : Array Float) (pf : Nat → Float)
    (ε : ℝ) (hpos1 : 0 < p.b2.size - 1)
    (hpsm : ∀ i, i < p.b2.size - 1 →
      toReal (pf i) = softmax (Finset.range (p.b2.size - 1))
        (fun j => toReal (acLogits p obs)[j]!) i)
    (hεbnd : ∀ k, k < p.b2.size - 1 →
      z1ErrBnd (p.W2[k]!).toList p.b2[k]! ((forwardAll p obs).2.1).toList
        + dotDiffBnd (p.W2[k]!).toList ((forwardAll p obs).2.1).toList (hRList p obs) ≤ ε) :
    -(sumIdxErrBnd (fun i => pf i * Float.log (pf i)) (entTermBnd pf) 0.0
          (List.range (p.b2.size - 1)) 0
        + (2*ε + (Real.exp (2*ε) - 1)
            * entropy (Finset.range (p.b2.size - 1)) (idealLogit p obs)))
      ≤ toReal (entropyF pf (p.b2.size - 1)) := by
  have hne : (Finset.range (p.b2.size - 1)).Nonempty := Finset.nonempty_range_iff.mpr (by omega)
  have herr := entropyF_forward_ideal_error p obs pf ε hpos1 hpsm hεbnd
  have hnn := entropy_nonneg (Finset.range (p.b2.size - 1)) (idealLogit p obs) hne
  have hlo := (abs_le.mp herr).1
  linarith

end Puffer.RL.EntropyForward
