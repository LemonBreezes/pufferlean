/-
Runnable demo of the Muon optimizer-STEP validity checks.

The whole-optimizer-state training-stability certificate (`MuonTrainLoop.muonTraj_optimizer_bounded_full`)
rests, on the weight side, on a per-step orthogonalization bound `‖ortho‖₂ ≤ √1.3131 + √(r·c)·rounding`
(dimension-free O(1)). That bound is supplied by `newtonSchulz_opNorm` applied to the matrix the Muon step
actually orthogonalizes — the Nesterov update `update = matLin 1.0 grad μ (matLin μ mom 1.0 grad)` — and its
sole hypothesis is the single decidable validity check `matOk update`. So the optimizer-step guarantee is
gated by a genuine `Bool` the trainer evaluates on the update matrix it already forms each step.

The `#guard_msgs`-checked `#eval`s below RUN those checks on concrete optimizer data (verified at build time,
so they cannot rot):

  • the Nesterov update matrix that Muon orthogonalizes passes `matOk` (`true`) — the per-step O(1) ortho
    bound applies to this step;
  • its orthogonalized output `newtonSchulz update (epsDefault …)` is itself valid (`true`);
  • the actual `stepMat` weight update is well-formed (`true`) — shape and finiteness preserved by the real
    optimizer step;
  • a gradient carrying a NaN poisons the update, and the finiteness guard correctly REJECTS it (`false`) —
    the trainer learns the O(1) bound does not apply that step, rather than silently trusting it;
  • one full `applyMuon` step keeps BOTH weight matrices well-formed (`true`) — the multi-tensor update the
    `muonTraj` iterate runs.

Same `matOk` as `NewtonSchulzDemo`, here on the matrices the optimizer produces rather than a standalone input.
-/
import Puffer.RL.NewtonSchulzRunnable
import Puffer.RL.NNTrain

namespace Puffer.RL.MuonStepDemo

open Puffer.RL.NewtonSchulzRunnable (matOk epsDefault)
open Puffer.FloatR.Muon (matLin newtonSchulz stepMat)
open Puffer.RL.NNTrain (MLP MuonState applyMuon)

-- The Nesterov update `g + μ·(μ·m + g)` that the Muon step orthogonalizes — passes the validity check,
-- so the per-step dimension-free O(1) ortho bound (`√1.3131 < 1.15` + rounding) applies to this step.
/-- info: true -/
#guard_msgs in #eval
  let grad := #[#[0.1, 0.2, -0.3], #[0.4, -0.5, 0.6]];
  let mom := #[#[0.0, 0.1, 0.2], #[-0.1, 0.0, 0.3]];
  let mu := (0.95 : Float);
  matOk (matLin 1.0 grad mu (matLin mu mom 1.0 grad))

-- The orthogonalized output of that update (computed eps `epsDefault`, as the stability theorem uses) is
-- itself a valid matrix.
/-- info: true -/
#guard_msgs in #eval
  let grad := #[#[0.1, 0.2, -0.3], #[0.4, -0.5, 0.6]];
  let mom := #[#[0.0, 0.1, 0.2], #[-0.1, 0.0, 0.3]];
  let mu := (0.95 : Float);
  let update := matLin 1.0 grad mu (matLin mu mom 1.0 grad);
  matOk (newtonSchulz update (epsDefault update update.size (update[0]!).size))

-- The actual `stepMat` weight update `(1−lr·wd)·W + (lr·scale)·ortho` is well-formed — shape and finiteness
-- preserved by the real optimizer step.
/-- info: true -/
#guard_msgs in #eval
  let W := #[#[0.5, -0.2, 0.1], #[0.3, 0.4, -0.6]];
  let grad := #[#[0.1, 0.2, -0.3], #[0.4, -0.5, 0.6]];
  let mom := #[#[0.0, 0.1, 0.2], #[-0.1, 0.0, 0.3]];
  matOk (stepMat W grad mom 0.01 0.1 0.95 1e-7).1

-- A gradient with a NaN entry (0/0) poisons the update matrix, and the Float-comparison finiteness guard
-- REJECTS it — the trainer knows the O(1) bound does not hold this step.
/-- info: false -/
#guard_msgs in #eval
  let grad := #[#[0.1, 0.2, 0.0 / 0.0], #[0.4, -0.5, 0.6]];
  let mom := #[#[0.0, 0.1, 0.2], #[-0.1, 0.0, 0.3]];
  let mu := (0.95 : Float);
  matOk (matLin 1.0 grad mu (matLin mu mom 1.0 grad))

-- A concrete MLP + zero momentum + gradient, for a full `applyMuon` step.
def demoMLP : MLP :=
  { W1 := #[#[0.5, -0.2, 0.1], #[0.3, 0.4, -0.6]], b1 := #[0.1, -0.2],
    W2 := #[#[0.2, 0.1, -0.3], #[-0.4, 0.5, 0.6]], b2 := #[0.0, 0.1] }

def demoGrad : Array (Array Float) × Array Float × Array (Array Float) × Array Float :=
  (#[#[0.1, 0.2, -0.3], #[0.4, -0.5, 0.6]], #[0.05, -0.05],
   #[#[-0.1, 0.2, 0.3], #[0.4, -0.2, 0.1]], #[0.02, -0.03])

-- One full `applyMuon` step (the update the `muonTraj` iterate runs) keeps BOTH weight matrices well-formed.
/-- info: true -/
#guard_msgs in #eval
  let out := applyMuon demoMLP (MuonState.zeros demoMLP) demoGrad 0.01 0.1 0.95 1e-7;
  matOk out.1.W1 && matOk out.1.W2

end Puffer.RL.MuonStepDemo
