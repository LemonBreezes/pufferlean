/-
The COMPLETE PPO objective the trainer computes — surrogate − vfCoef·(unclipped value loss) + entCoef·entropy
— on the ACTUAL forward net, within a certified bound of the IDEAL-net objective. The capstone that assembles
the three forward term-bounds (surrogate a121, value loss a106, entropy a120) into one end-to-end statement.

Two layers:

  • `ppoTotalSqObjF` / `ppoTotalSqObjF_ideal_error` — the MODULAR core. `ppoTotalSqObjF` is the objective with
    the UNCLIPPED value loss `½(V−ret)²` the AD tape actually uses (`mlpGradPPO`'s `scale (vfCoef·0.5) diff²`),
    unlike a103's `ppoTotalObjF` which used the clipped `valueLossF`. `ppoTotalSqObjF_ideal_error` bounds it
    against ANY three ℝ references `Ssurr`/`Svl`/`Sent` (not just the runtime reals of a103): given the three
    component bounds, the total is within the two coefficient-multiply + sub/add roundings (`mulApprox`/
    `subApprox`/`addApprox`) composed with the three budgets. Because the references are arbitrary, the SAME
    theorem accepts either the rounding-only bounds (like a103) OR the forward-net-vs-ideal bounds below.

  • `ppoTotalSqObjF_forward_error` — the CONCRETE forward instance. Discharges the three references and budgets
    with the actual forward bounds: `ppoSurrF_forward_error` (a121, `Ssurr = ppoObjLoHi` at the ideal ratio),
    `valueSqLossF_forward_error` (a106, `Svl = ½(V*−ret)²` at the ideal value), `entropyF_forward_ideal_error`
    (a120, `Sent = entropy(idealLogit)`). The result: the whole objective over the real forward pass — the
    forward-net ratio in the surrogate, the forward-net value in the loss, the forward-net policy in the entropy
    — is within the summed forward budgets of the ideal-logit objective. `BsurrFwd`/`BvlFwd`/`BentFwd` name the
    three forward budgets (matching a121/a106/a120's RHS def-eq) to keep the statement readable.

Axiom-clean (a121 ⊕ a106 ⊕ a120 footprint: the Float base + exp/log/add/sub/mul models + the `toReal_*`
projections).
-/
import Puffer.RL.SurrogateForward
import Puffer.RL.ValueLossForward
import Puffer.RL.EntropyForward

namespace Puffer.RL.PPOTotalForward

open Puffer.FloatR
open Puffer.Net (softmax)
open Puffer.RL.NNTrain (MLP forwardAll policyAndValue)
open Puffer.RL.ActorCriticBound (idealLogit acLogits)
open Puffer.RL.LogSumExpBound (logpF)
open Puffer.RL.PPORuntime (ppoSurrF valueSqLossF sqBnd ratioFullF)
open Puffer.RL.EntropyRuntime (entropyF entTermBnd)
open Puffer.RL.EntropyPerturb (entropy)
open Puffer.RL.SoftmaxBound (sumIdxErrBnd)
open Puffer.RL.RatioForward (logpFwdBnd)
open Puffer.RL.SurrogateForward (ppoSurrF_forward_error)
open Puffer.RL.ValueLossForward (valueSqLossF_forward_error)
open Puffer.RL.EntropyForward (entropyF_forward_ideal_error)

/-- The full PPO objective the trainer computes: `surrogate − vfCoef·(½(val−ret)²) + entCoef·entropy`, with
    the UNCLIPPED value loss (`valueSqLossF`) the AD tape uses — unlike a103's clipped `ppoTotalObjF`. -/
def ppoTotalSqObjF (g r lo hi val ret vfCoef entCoef : Float) (p : Nat → Float) (A : Nat) : Float :=
  ppoSurrF g r lo hi - vfCoef * valueSqLossF val ret + entCoef * entropyF p A

/-- **Total PPO objective (unclipped) vs an arbitrary ideal reference.** Given the surrogate, value loss, and
    entropy each within `Bsurr`/`Bvl`/`Bent` of ANY ℝ references `Ssurr`/`Svl`/`Sent`, the running
    `ppoTotalSqObjF` is within the two coefficient-multiply + sub/add roundings of the combined reference
    `Ssurr − vfCoef·Svl + entCoef·Sent`. Modular in the references, so it accepts either the rounding-only
    component bounds (a93/a97, like a103) or the forward-net-vs-ideal bounds (a121/a106/a120). -/
theorem ppoTotalSqObjF_ideal_error (g r lo hi val ret vfCoef entCoef : Float)
    (p : Nat → Float) (A : Nat) (Ssurr Svl Sent Bsurr Bvl Bent : ℝ)
    (hsurr : |toReal (ppoSurrF g r lo hi) - Ssurr| ≤ Bsurr)
    (hvl : |toReal (valueSqLossF val ret) - Svl| ≤ Bvl)
    (hent : |toReal (entropyF p A) - Sent| ≤ Bent) :
    |toReal (ppoTotalSqObjF g r lo hi val ret vfCoef entCoef p A)
        - (Ssurr - toReal vfCoef * Svl + toReal entCoef * Sent)|
      ≤ u64 * |toReal (ppoSurrF g r lo hi - vfCoef * valueSqLossF val ret)
              + toReal (entCoef * entropyF p A)|
        + (u64 * |toReal (ppoSurrF g r lo hi) - toReal (vfCoef * valueSqLossF val ret)|
            + Bsurr
            + (u64 * |toReal vfCoef * toReal (valueSqLossF val ret)|
                + |toReal vfCoef| * Bvl))
          + (u64 * |toReal entCoef * toReal (entropyF p A)|
              + |toReal entCoef| * Bent) := by
  have h0 : ∀ z : Float, |toReal z - toReal z| ≤ 0 := fun z => by simp
  have hq1 : |toReal (vfCoef * valueSqLossF val ret) - toReal vfCoef * Svl|
      ≤ u64 * |toReal vfCoef * toReal (valueSqLossF val ret)| + |toReal vfCoef| * Bvl := by
    have := mulApprox_error vfCoef (valueSqLossF val ret) (toReal vfCoef) Svl 0 Bvl (h0 vfCoef) hvl
    simpa using this
  have hq2 : |toReal (entCoef * entropyF p A) - toReal entCoef * Sent|
      ≤ u64 * |toReal entCoef * toReal (entropyF p A)| + |toReal entCoef| * Bent := by
    have := mulApprox_error entCoef (entropyF p A) (toReal entCoef) Sent 0 Bent (h0 entCoef) hent
    simpa using this
  have hs := subApprox_error (ppoSurrF g r lo hi) (vfCoef * valueSqLossF val ret)
    Ssurr (toReal vfCoef * Svl) Bsurr _ hsurr hq1
  have hres := addApprox_error (ppoSurrF g r lo hi - vfCoef * valueSqLossF val ret)
    (entCoef * entropyF p A) (Ssurr - toReal vfCoef * Svl) (toReal entCoef * Sent) _ _ hs hq2
  rw [ppoTotalSqObjF]
  simpa using hres

/-- **Unclipped total-objective runtime accuracy.** The running unclipped PPO objective
    `ppoTotalSqObjF = ppoSurrF − vfCoef·½(val−ret)² + entCoef·entropy` (the exact scalar the AD tape
    ascends) computes its own ℝ-arithmetic meaning
    `toReal(ppoSurrF) − toReal vfCoef · toReal(valueSqLossF) + toReal entCoef · toReal(entropyF)`
    within the four hardware roundings of its composition — the outer add, the sub, the
    value-coefficient multiply, and the entropy-coefficient multiply — each contributing one
    `u64·|·|` term at the magnitude of its own exact operand. This is the concrete "computed value ≈
    its real meaning" guarantee for the UNCLIPPED total (the ½(val−ret)² value term the AD tape
    actually uses), the rounding-only sibling of the clipped `ppoTotalObjF_error`. It specialises the
    reference-parameterised core `ppoTotalSqObjF_ideal_error` to the exact real evaluations of the
    three Float components with zero component budgets (`Bsurr = Bvl = Bent = 0`, each `|x − x| = 0`),
    so the `Bsurr`/`|coef|·B` terms vanish and the bound is a pure 4-term rounding budget. -/
theorem ppoTotalSqObjF_error (g r lo hi val ret vfCoef entCoef : Float) (p : Nat → Float) (A : Nat) :
    |toReal (ppoTotalSqObjF g r lo hi val ret vfCoef entCoef p A)
        - (toReal (ppoSurrF g r lo hi) - toReal vfCoef * toReal (valueSqLossF val ret)
            + toReal entCoef * toReal (entropyF p A))|
      ≤ u64 * |toReal (ppoSurrF g r lo hi - vfCoef * valueSqLossF val ret)
              + toReal (entCoef * entropyF p A)|
        + u64 * |toReal (ppoSurrF g r lo hi) - toReal (vfCoef * valueSqLossF val ret)|
        + u64 * |toReal vfCoef * toReal (valueSqLossF val ret)|
        + u64 * |toReal entCoef * toReal (entropyF p A)| := by
  have h := ppoTotalSqObjF_ideal_error g r lo hi val ret vfCoef entCoef p A
    (toReal (ppoSurrF g r lo hi)) (toReal (valueSqLossF val ret)) (toReal (entropyF p A))
    0 0 0 (by simp) (by simp) (by simp)
  refine le_trans h (le_of_eq ?_)
  ring

/-- The a121 surrogate-forward budget (`ppoSurrF_error` rounding + `|g|·(a119 ratio budget)`). -/
noncomputable def BsurrFwd (p : MLP) (obs : Array Float) (g oldLogp lo hi : Float) (a n : Nat) (ε d : ℝ) : ℝ :=
  u64 * max |toReal g * toReal (ratioFullF (logpF (fun k => (forwardAll p obs).2.2[k]!) a n) oldLogp)|
        |toReal g * Puffer.RL.PPO.clampR
          (toReal (ratioFullF (logpF (fun k => (forwardAll p obs).2.2[k]!) a n) oldLogp))
          (toReal lo) (toReal hi)|
    + |toReal g| *
        ((expEps * Real.exp (toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp))
            + Real.exp (toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n - oldLogp))
                * (Real.exp (u64 * |toReal (logpF (fun k => (forwardAll p obs).2.2[k]!) a n)
                      - toReal oldLogp|) - 1))
          + Real.exp (Real.log (softmax (Finset.range n) (idealLogit p obs) a) - toReal oldLogp)
              * (Real.exp (logpFwdBnd p obs a n d ε) - 1))

/-- The a106 value-loss-forward budget (`½·square` Float rounding + forward-value perturbation). -/
noncomputable def BvlFwd (p : MLP) (obs : Array Float) (ret : Float) : ℝ :=
  (u64 * |toReal (0.5 : Float) * toReal (((policyAndValue p obs).2 - ret) * ((policyAndValue p obs).2 - ret))|
      + |toReal (0.5 : Float)| * sqBnd ((policyAndValue p obs).2 - ret)
          (toReal ((policyAndValue p obs).2) - toReal ret)
          (u64 * |toReal ((policyAndValue p obs).2) - toReal ret|)
      + |(toReal ((policyAndValue p obs).2) - toReal ret) ^ 2| * (u64 * |(0.5 : ℝ)|))
    + (z1ErrBnd (p.W2[p.b2.size - 1]!).toList p.b2[p.b2.size - 1]! ((forwardAll p obs).2.1).toList
        + dotDiffBnd (p.W2[p.b2.size - 1]!).toList ((forwardAll p obs).2.1).toList
            (Puffer.RL.ForwardExec.hRList p obs))
      * (|toReal ((policyAndValue p obs).2) - toReal ret|
          + |idealLogit p obs (p.b2.size - 1) - toReal ret|) / 2

/-- The a120 entropy-forward budget (reduction rounding + forward-logit perturbation). -/
noncomputable def BentFwd (p : MLP) (obs : Array Float) (pf : Nat → Float) (εe : ℝ) : ℝ :=
  sumIdxErrBnd (fun i => pf i * Float.log (pf i)) (entTermBnd pf) 0.0
      (List.range (p.b2.size - 1)) 0
    + (2*εe + (Real.exp (2*εe) - 1)
        * entropy (Finset.range (p.b2.size - 1)) (idealLogit p obs))

/-- **The complete PPO objective on the actual forward net vs the ideal-net objective.** The whole loss the
    trainer computes — the forward-net ratio in the clipped surrogate, the forward-net value in `½(V−ret)²`,
    the forward-net policy in the entropy — is within the summed forward budgets `BsurrFwd`/`BvlFwd`/`BentFwd`
    (a121/a106/a120) plus the coefficient roundings of the ideal-logit objective `ppoObjLoHi(g, ideal ratio) −
    vfCoef·½(V*−ret)² + entCoef·entropy(idealLogit)`. Discharges `ppoTotalSqObjF_ideal_error` with the three
    forward term-bounds. The end-to-end actor-critic accuracy statement. -/
theorem ppoTotalSqObjF_forward_error (p : MLP) (obs : Array Float)
    (g oldLogp lo hi ret vfCoef entCoef : Float) (pf : Nat → Float) (a n : Nat)
    (ε d εe : ℝ) (hpos : 0 < p.b2.size) (hd : 0 < d)
    (hden : d ≤ toReal (Puffer.RL.SoftmaxBound.sumIdxFGo
      (fun k => Float.exp ((forwardAll p obs).2.2[k]!)) 0.0 (List.range n)))
    (hRden : d ≤ ∑ j ∈ Finset.range n, Real.exp (toReal ((forwardAll p obs).2.2[j]!)))
    (hne : (Finset.range n).Nonempty) (ha : a < n) (hn : n ≤ p.b2.size)
    (hεbnd : ∀ k, k < n →
      z1ErrBnd (p.W2[k]!).toList p.b2[k]! ((forwardAll p obs).2.1).toList
        + dotDiffBnd (p.W2[k]!).toList ((forwardAll p obs).2.1).toList
            (Puffer.RL.ForwardExec.hRList p obs) ≤ ε)
    (hpos1 : 0 < p.b2.size - 1)
    (hpsm : ∀ i, i < p.b2.size - 1 →
      toReal (pf i) = softmax (Finset.range (p.b2.size - 1))
        (fun j => toReal (acLogits p obs)[j]!) i)
    (hεbnde : ∀ k, k < p.b2.size - 1 →
      z1ErrBnd (p.W2[k]!).toList p.b2[k]! ((forwardAll p obs).2.1).toList
        + dotDiffBnd (p.W2[k]!).toList ((forwardAll p obs).2.1).toList
            (Puffer.RL.ForwardExec.hRList p obs) ≤ εe) :
    |toReal (ppoTotalSqObjF g (ratioFullF (logpF (fun k => (forwardAll p obs).2.2[k]!) a n) oldLogp)
            lo hi (policyAndValue p obs).2 ret vfCoef entCoef pf (p.b2.size - 1))
        - (Puffer.RL.PPO.ppoObjLoHi (toReal g)
              (Real.exp (Real.log (softmax (Finset.range n) (idealLogit p obs) a) - toReal oldLogp))
              (toReal lo) (toReal hi)
            - toReal vfCoef * ((idealLogit p obs (p.b2.size - 1) - toReal ret) ^ 2 / 2)
            + toReal entCoef * entropy (Finset.range (p.b2.size - 1)) (idealLogit p obs))|
      ≤ u64 * |toReal (ppoSurrF g (ratioFullF (logpF (fun k => (forwardAll p obs).2.2[k]!) a n) oldLogp) lo hi
              - vfCoef * valueSqLossF (policyAndValue p obs).2 ret)
              + toReal (entCoef * entropyF pf (p.b2.size - 1))|
        + (u64 * |toReal (ppoSurrF g (ratioFullF (logpF (fun k => (forwardAll p obs).2.2[k]!) a n) oldLogp) lo hi)
                - toReal (vfCoef * valueSqLossF (policyAndValue p obs).2 ret)|
            + BsurrFwd p obs g oldLogp lo hi a n ε d
            + (u64 * |toReal vfCoef * toReal (valueSqLossF (policyAndValue p obs).2 ret)|
                + |toReal vfCoef| * BvlFwd p obs ret))
          + (u64 * |toReal entCoef * toReal (entropyF pf (p.b2.size - 1))|
              + |toReal entCoef| * BentFwd p obs pf εe) := by
  exact ppoTotalSqObjF_ideal_error g
    (ratioFullF (logpF (fun k => (forwardAll p obs).2.2[k]!) a n) oldLogp) lo hi
    (policyAndValue p obs).2 ret vfCoef entCoef pf (p.b2.size - 1)
    _ _ _ (BsurrFwd p obs g oldLogp lo hi a n ε d) (BvlFwd p obs ret) (BentFwd p obs pf εe)
    (ppoSurrF_forward_error p obs g oldLogp lo hi a n ε d hd hden hRden hne ha hn hεbnd)
    (valueSqLossF_forward_error p obs ret hpos)
    (entropyF_forward_ideal_error p obs pf εe hpos1 hpsm hεbnde)

end Puffer.RL.PPOTotalForward
