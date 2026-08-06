/-
The PER-POSITION accuracy of `computeGAE`'s TD error `deltaAt` and recursion weight `wAt` — the δ/w internal
rounding that a85 (`GAEInvariant`) held atomic. Together with a89's fold input-sensitivity
(`gadvListR_perturb`), these are the two halves needed to upgrade a85's reference (the `toReal` of the
COMPUTED Float deltas/weights) to the TRUE-arithmetic GAE.

`deltaAt = rₜ + γ·V_{t+1}·(1−doneₜ) − Vₜ` and `wAt = γ·λ·(1−doneₜ)` are computed by Float arithmetic over the
raw stored trajectory values (rewards/values are exact `toReal`, but the `γ·V·mask` products and the reward/
value add/subtract round). This file bounds each against its true real value:

  • `nntR`/`vNextR`/`deltaR`/`wR` — the real references (`1−doneₜ`, bootstrap `V_{t+1}`, the true TD error and
    weight), with `toReal_nnt`/`toReal_vNext` reducing the Float `if`-masks (`apply_ite` + `toReal_zeroLit`/
    `toReal_oneLit`).
  • `deltaExpr_error`/`wExpr_error` — the IF-FREE op-chain cores over abstract Floats (composed
    `mulApprox_error`/`addApprox_error`/`subApprox_error`), so the `if`-masks never enter the arithmetic proof.
  • `deltaAt_error`/`wAt_error` — the runnable per-position bounds, one-line instantiations of the cores at the
    actual `if`-mask Floats (discharging the reals via `toReal_nnt`/`toReal_vNext`).

Axiom-clean beyond the trusted Float (1+δ) base (`add/sub/mul_model` + `toReal` + `toReal_zeroLit`/
`toReal_oneLit`). Composing `deltaAt_error`/`wAt_error` (per-position budgets) into `gadvListR_perturb` (a89)
to land the true-arithmetic GAE bound on `computeGAE` is the disclosed next step.
-/
import Puffer.RL.GAEInvariant

namespace Puffer.RL.GAEDeltaError

open Puffer.FloatR
open Puffer.RL.NNTrain
open Puffer.RL.GAEInvariant (deltaAt wAt)

/-! ### Real references for the per-position TD error and weight -/

/-- The real terminal mask `1 − doneₜ`. -/
noncomputable def nntR (traj : Array Transition) (t : Nat) : ℝ :=
  if traj[t]!.terminal then 0 else 1

/-- The real bootstrap value `V_{t+1}` (0 past the horizon). -/
noncomputable def vNextR (traj : Array Transition) (n t : Nat) : ℝ :=
  if t + 1 < n then toReal (traj[t+1]!.value) else 0

/-- The true real TD error `δR_t = rₜ + γ·V_{t+1}·(1−doneₜ) − Vₜ`. -/
noncomputable def deltaR (traj : Array Transition) (gamma : Float) (n t : Nat) : ℝ :=
  toReal (traj[t]!.reward) + toReal gamma * vNextR traj n t * nntR traj t - toReal (traj[t]!.value)

/-- The true real GAE weight `wR_t = γ·λ·(1−doneₜ)`. -/
noncomputable def wR (traj : Array Transition) (gamma lam : Float) (t : Nat) : ℝ :=
  toReal gamma * toReal lam * nntR traj t

theorem toReal_nnt (traj : Array Transition) (t : Nat) :
    toReal (if traj[t]!.terminal then (0.0 : Float) else 1.0) = nntR traj t := by
  rw [nntR, apply_ite toReal, toReal_zeroLit, toReal_oneLit]

theorem toReal_vNext (traj : Array Transition) (n t : Nat) :
    toReal (if t + 1 < n then traj[t+1]!.value else (0.0 : Float)) = vNextR traj n t := by
  rw [vNextR, apply_ite toReal, toReal_zeroLit]

/-! ### If-free op-chain cores -/

/-- If-free core for the TD error `r + γ·vN·nn − v`. -/
theorem deltaExpr_error (r v vN nn gamma : Float) (vNr nnr : ℝ)
    (hvN : toReal vN = vNr) (hnn : toReal nn = nnr) :
    |toReal (r + gamma * vN * nn - v) - (toReal r + toReal gamma * vNr * nnr - toReal v)|
      ≤ u64 * |toReal (r + gamma * vN * nn) - toReal v|
        + (u64 * |toReal r + toReal (gamma * vN * nn)|
          + (u64 * |toReal (gamma * vN) * toReal nn| + |nnr| * (u64 * |toReal gamma * toReal vN|))) := by
  have h1 : |toReal (gamma * vN) - toReal gamma * vNr| ≤ u64 * |toReal gamma * toReal vN| := by
    have := mulApprox_error gamma vN (toReal gamma) vNr 0 0 (by simp) (by rw [hvN]; simp)
    simpa using this
  have h2 : |toReal (gamma * vN * nn) - toReal gamma * vNr * nnr|
      ≤ u64 * |toReal (gamma * vN) * toReal nn| + |nnr| * (u64 * |toReal gamma * toReal vN|) := by
    have := mulApprox_error (gamma * vN) nn (toReal gamma * vNr) nnr
      (u64 * |toReal gamma * toReal vN|) 0 h1 (by rw [hnn]; simp)
    simpa using this
  have h3 : |toReal (r + gamma * vN * nn) - (toReal r + toReal gamma * vNr * nnr)|
      ≤ u64 * |toReal r + toReal (gamma * vN * nn)|
        + (u64 * |toReal (gamma * vN) * toReal nn| + |nnr| * (u64 * |toReal gamma * toReal vN|)) := by
    have := addApprox_error r (gamma * vN * nn) (toReal r) (toReal gamma * vNr * nnr) 0 _ (by simp) h2
    simpa using this
  have h4 := subApprox_error (r + gamma * vN * nn) v
    (toReal r + toReal gamma * vNr * nnr) (toReal v) _ 0 h3 (by simp)
  simpa using h4

/-- If-free core for the weight `γ·λ·nn`. -/
theorem wExpr_error (gamma lam nn : Float) (nnr : ℝ) (hnn : toReal nn = nnr) :
    |toReal (gamma * lam * nn) - toReal gamma * toReal lam * nnr|
      ≤ u64 * |toReal (gamma * lam) * toReal nn| + |nnr| * (u64 * |toReal gamma * toReal lam|) := by
  have h1 : |toReal (gamma * lam) - toReal gamma * toReal lam| ≤ u64 * |toReal gamma * toReal lam| := by
    simpa using mul_error gamma lam
  have := mulApprox_error (gamma * lam) nn (toReal gamma * toReal lam) nnr
    (u64 * |toReal gamma * toReal lam|) 0 h1 (by rw [hnn]; simp)
  simpa using this

/-! ### Per-position error bounds on the runnable `deltaAt` / `wAt` -/

/-- Computable rounding budget for `deltaAt`. -/
noncomputable def deltaErrBnd (traj : Array Transition) (gamma : Float) (n t : Nat) : ℝ :=
  u64 * |toReal (traj[t]!.reward + gamma * (if t + 1 < n then traj[t+1]!.value else 0.0)
            * (if traj[t]!.terminal then 0.0 else 1.0)) - toReal (traj[t]!.value)|
  + (u64 * |toReal (traj[t]!.reward) + toReal (gamma * (if t + 1 < n then traj[t+1]!.value else 0.0)
            * (if traj[t]!.terminal then 0.0 else 1.0))|
    + (u64 * |toReal (gamma * (if t + 1 < n then traj[t+1]!.value else 0.0))
              * toReal (if traj[t]!.terminal then (0.0 : Float) else 1.0)|
      + |nntR traj t| * (u64 * |toReal gamma
              * toReal (if t + 1 < n then traj[t+1]!.value else (0.0 : Float))|)))

/-- **`deltaAt` accuracy.** The loop's TD error `deltaAt` is within `deltaErrBnd` of the true real
    `δR_t = rₜ + γ·V_{t+1}·(1−doneₜ) − Vₜ`. -/
theorem deltaAt_error (traj : Array Transition) (gamma : Float) (n t : Nat) :
    |toReal (deltaAt traj gamma n t) - deltaR traj gamma n t| ≤ deltaErrBnd traj gamma n t := by
  exact deltaExpr_error (traj[t]!.reward) (traj[t]!.value)
    (if t + 1 < n then traj[t+1]!.value else 0.0) (if traj[t]!.terminal then 0.0 else 1.0) gamma
    (vNextR traj n t) (nntR traj t) (toReal_vNext traj n t) (toReal_nnt traj t)

/-- **Terminal-step TD-error rounding is bootstrap-free.** At an episode boundary
    (`traj[t]!.terminal = true`) the GAE recursion resets, so the loop's computed TD error `deltaAt`
    matches its true real reference `deltaR = rₜ − Vₜ` to within `u·(2+u)·(|rₜ| + |Vₜ|)` — a clean
    relative bound involving NEITHER the discount `γ` NOR the bootstrap value `V_{t+1}` (both drop out
    through the zeroed terminal mask, so the only surviving rounding is the two add/sub rounds of `r`
    and `v`, plus the exact `·0.0` product). This sharpens the file's general per-step `deltaAt_error`
    (`≤ deltaErrBnd`, which carries the `u·|γ·V_{t+1}|` product-rounding term) at terminal positions.
    The hypothesis is load-bearing: without it the error picks up that `u·|γ·V_{t+1}|` term, which this
    reward/value bound does not cover. -/
theorem deltaAt_error_terminal (traj : Array Transition) (gamma : Float) (n t : Nat)
    (h : traj[t]!.terminal = true) :
    |toReal (deltaAt traj gamma n t) - deltaR traj gamma n t|
      ≤ u64 * (2 + u64) * (|toReal (traj[t]!.reward)| + |toReal (traj[t]!.value)|) := by
  have hmask : (if traj[t]!.terminal then (0.0 : Float) else 1.0) = 0.0 := if_pos h
  have hnnt : nntR traj t = 0 := by simp [nntR, h]
  have hdR : deltaR traj gamma n t = toReal (traj[t]!.reward) - toReal (traj[t]!.value) := by
    simp [deltaR, hnnt]
  have hdA : deltaAt traj gamma n t
      = traj[t]!.reward + gamma * (if t + 1 < n then traj[t+1]!.value else 0.0) * 0.0
        - traj[t]!.value := by
    simp only [deltaAt, hmask]
  rw [hdA, hdR]
  obtain ⟨δm, hδm, hem⟩ :=
    mul_model (gamma * (if t + 1 < n then traj[t+1]!.value else 0.0)) (0.0 : Float)
  have hz : toReal (gamma * (if t + 1 < n then traj[t+1]!.value else 0.0) * 0.0) = 0 := by
    rw [hem, toReal_zeroLit]; ring
  obtain ⟨δa, hδa, hea⟩ :=
    add_model (traj[t]!.reward) (gamma * (if t + 1 < n then traj[t+1]!.value else 0.0) * 0.0)
  obtain ⟨δs, hδs, hes⟩ :=
    sub_model (traj[t]!.reward + gamma * (if t + 1 < n then traj[t+1]!.value else 0.0) * 0.0)
      (traj[t]!.value)
  rw [hes, hea, hz]
  set R := toReal (traj[t]!.reward)
  set V := toReal (traj[t]!.value)
  have key : ((R + 0) * (1 + δa) - V) * (1 + δs) - (R - V)
      = R * (δa + δs + δa * δs) - V * δs := by ring
  rw [key]
  have hu : (0 : ℝ) ≤ u64 := u64_pos.le
  have hb1 : |R * (δa + δs + δa * δs)| ≤ |R| * (2 * u64 + u64 * u64) := by
    rw [abs_mul]
    apply mul_le_mul_of_nonneg_left _ (abs_nonneg R)
    calc |δa + δs + δa * δs|
        ≤ |δa| + |δs| + |δa * δs| :=
          (abs_add_le _ _).trans (add_le_add (abs_add_le _ _) le_rfl)
      _ = |δa| + |δs| + |δa| * |δs| := by rw [abs_mul]
      _ ≤ u64 + u64 + u64 * u64 :=
          add_le_add (add_le_add hδa hδs) (mul_le_mul hδa hδs (abs_nonneg _) hu)
      _ = 2 * u64 + u64 * u64 := by ring
  have hb2 : |V * δs| ≤ |V| * u64 := by
    rw [abs_mul]; exact mul_le_mul_of_nonneg_left hδs (abs_nonneg V)
  calc |R * (δa + δs + δa * δs) - V * δs|
      ≤ |R * (δa + δs + δa * δs)| + |V * δs| := abs_sub _ _
    _ ≤ |R| * (2 * u64 + u64 * u64) + |V| * u64 := add_le_add hb1 hb2
    _ ≤ u64 * (2 + u64) * (|R| + |V|) := by
        nlinarith [abs_nonneg R, abs_nonneg V, u64_pos.le,
          mul_nonneg u64_pos.le (abs_nonneg V),
          mul_nonneg (mul_nonneg u64_pos.le u64_pos.le) (abs_nonneg V)]

/-- Computable rounding budget for `wAt`. -/
noncomputable def wErrBnd (traj : Array Transition) (gamma lam : Float) (t : Nat) : ℝ :=
  u64 * |toReal (gamma * lam) * toReal (if traj[t]!.terminal then (0.0 : Float) else 1.0)|
  + |nntR traj t| * (u64 * |toReal gamma * toReal lam|)

/-- **`wAt` accuracy.** The loop's GAE weight `wAt` is within `wErrBnd` of the true real
    `wR_t = γ·λ·(1−doneₜ)`. -/
theorem wAt_error (traj : Array Transition) (gamma lam : Float) (t : Nat) :
    |toReal (wAt traj gamma lam t) - wR traj gamma lam t| ≤ wErrBnd traj gamma lam t := by
  exact wExpr_error gamma lam (if traj[t]!.terminal then 0.0 else 1.0) (nntR traj t) (toReal_nnt traj t)


end Puffer.RL.GAEDeltaError
