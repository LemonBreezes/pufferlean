/-
# Adam exec bridge — the Mathlib-free binary runs the SAME defs the proofs certify

`Puffer/Float/Exec.lean` (Mathlib-free, linked by the `puffer` binary) carries executable
Adam kernels; `Puffer/RL/AdamStep.lean` (Mathlib-side) carries the identically-shaped defs
plus the machine-checked error bounds `adamStepF_error`/`adamStepBcF_error`.  This module
proves the two are **definitionally equal** (`rfl`), so:

* the binary's `verify-adam` runs `Puffer.FloatR.adamStepF`, and
* `Puffer.RL.AdamStep.adamStepF_error` certifies `Puffer.RL.AdamStep.adamStepF`,

and these are the same function.  This is the C84 checker-core tripwire pattern: the module
is registered in `Puffer.lean`, so any drift between the Mathlib-free exec copy and the
proven original turns a plain `lake build` red.  All proofs are `rfl` — the bodies are
token-for-token copies — so no soundness claim is added; the equalities only guard the copy.
-/
import Puffer.Float.Exec
import Puffer.RL.AdamStep

namespace Puffer.RL.AdamExecBridge

theorem adamM1F_eq : @Puffer.FloatR.adamM1F = @Puffer.RL.AdamStep.adamM1F := rfl
theorem adamM2F_eq : @Puffer.FloatR.adamM2F = @Puffer.RL.AdamStep.adamM2F := rfl
theorem adamDirF_eq : @Puffer.FloatR.adamDirF = @Puffer.RL.AdamStep.adamDirF := rfl
theorem adamStepF_eq : @Puffer.FloatR.adamStepF = @Puffer.RL.AdamStep.adamStepF := rfl
theorem adamM1HatF_eq : @Puffer.FloatR.adamM1HatF = @Puffer.RL.AdamStep.adamM1HatF := rfl
theorem adamM2HatF_eq : @Puffer.FloatR.adamM2HatF = @Puffer.RL.AdamStep.adamM2HatF := rfl
theorem adamStepBcF_eq : @Puffer.FloatR.adamStepBcF = @Puffer.RL.AdamStep.adamStepBcF := rfl

end Puffer.RL.AdamExecBridge
