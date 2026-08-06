/-
# The full Muon step's operator-norm Lipschitz `L`, and the Muon whole-run interval

C42 (`MuonAscentBridge`) proved the whole-run geometric interval holds for ANY `L`-Lipschitz step map but left
Muon's OWN step-Lipschitz `L` open. C44/C47 bounded ONE Newton–Schulz step (`L_ns`); C50 (`NewtonSchulzIterate`)
composed the `k` NS iterations (`Lu^k` under a uniform per-ball bound). This module composes that `k`-iteration NS
Lipschitz with the outer MOMENTUM/PARAMETER update to form the FULL Muon step's operator-norm Lipschitz
`L = 1 + |lr|·Lu^k·G`, and wires it into C42's `muon_whole_run_opnorm_interval` — closing C42's open `L` end-to-end.

Working in C50's abstract normed ∗-ring `R` (which real matrices with the L2 operator norm instantiate, and which is
also a `NormedAddCommGroup`, so it feeds C42 directly):

* `muonStep grad lr a b c k x := x − lr • nsIter a b c k (grad x)` — the Muon update `θ ↦ θ − lr·NS(∇θ)`: the
  gradient `grad x` is orthogonalized by `k` Newton–Schulz iterations, then subtracted from the parameter.
* `muonStep_lipschitz` — the crux: if `grad` is `G`-Lipschitz and GLOBALLY magnitude-bounded (`∀ x, ‖grad x‖ ≤ M`
  — gradient clipping, so the NS ball of radius `M` applies at every point) with C50's uniform NS bound `Lu`, then
  `muonStep` is GLOBALLY `(1 + |lr|·Lu^k·G)`-Lipschitz: `‖(x − lr•NS(grad x)) − (y − lr•NS(grad y))‖ ≤ ‖x−y‖ +
  |lr|·‖NS(grad x) − NS(grad y)‖ ≤ ‖x−y‖ + |lr|·Lu^k·‖grad x − grad y‖ ≤ (1 + |lr|·Lu^k·G)·‖x−y‖`. This is the exact
  matrix analogue of gradient ascent's `1 + |lr|·G` (C3/C27), with the NS orthogonalization factor `Lu^k`.
* `muon_whole_run` — the capstone: `muonStep` (Lipschitz `L = 1 + |lr|·Lu^k·G`) fed into C42's whole-run interval,
  giving `‖θ n − θ' n‖ ≤ L^n·‖θ 0 − θ' 0‖ + B·Σ_{j<n} L^j` for the ACTUAL Muon optimizer.

**Scope (honestly disclosed).** This closes C42's open Muon step-Lipschitz `L = 1 + |lr|·Lu^k·G` by composing C50's
`k`-iteration NS Lipschitz (`Lu^k`) with the gradient's Lipschitz `G` and the affine parameter update. Honest inputs
(as hypotheses): `Lu` — the NS iteration's uniform per-ball bound (Muon's coefficients keeping the iterate norm ≲ 1,
C50's `hLu`, not established here); `G` — the objective gradient's operator-norm Lipschitz (the matrix analogue of
C4/C26's `dLip`); the GLOBAL gradient magnitude bound `∀ x, ‖grad x‖ ≤ M` (gradient clipping — the honest condition
that makes the NS ball global, so `muonStep` is globally Lipschitz for C42's global `hlip`); `B` — the Float-vs-ℝ
Muon per-step error (C1/`nsScalarF_error`). `L = 1 + |lr|·Lu^k·G ≥ 1` is EXPANSIVE (like plain ascent, C27), so the
geometric interval — a horizon-free Muon bound would need weight decay (cf. C32). `grad : R → R` abstracts the
objective's gradient in the matrix (operator-norm) setting.
-/
import Puffer.RL.NewtonSchulzIterate
import Puffer.RL.MuonAscentBridge
open Puffer.RL.NewtonSchulzIterate (nsIter nsLconst nsRadius nsLconst_nonneg nsIter_lipschitz_uniform)
open Puffer.RL.MuonAscentBridge (muon_whole_run_opnorm_interval)

namespace Puffer.RL.MuonStepLipschitz

variable {R : Type*} [NormedRing R] [NormedAlgebra ℝ R] [StarRing R] [NormedStarGroup R]

/-- **The full Muon update** as a self-map on the parameter space: `θ ↦ θ − lr·NS(∇θ)`, where `grad x` is the
    momentum/gradient at `x`, `nsIter a b c k (grad x)` orthogonalizes it via `k` Newton–Schulz iterations, and the
    result is subtracted (scaled by the step size `lr`) from the parameter. -/
def muonStep (grad : R → R) (lr : ℝ) (a b c : ℝ) (k : ℕ) (x : R) : R :=
  x - lr • nsIter a b c k (grad x)

/-- **The full Muon step is `(1 + |lr|·Lu^k·G)`-Lipschitz (globally).** Given `grad` is `G`-Lipschitz and GLOBALLY
    magnitude-bounded (`∀ x, ‖grad x‖ ≤ M` — gradient clipping, so the NS ball of radius `M` applies everywhere and
    C50's uniform NS Lipschitz `Lu` holds at every gradient pair), the Muon update is globally
    `(1 + |lr|·Lu^k·G)`-Lipschitz. Triangle on the subtraction + `norm_smul` + C50's `nsIter_lipschitz_uniform` + the
    gradient's Lipschitz. The matrix analogue of gradient ascent's `1 + |lr|·G`, with the NS factor `Lu^k`. -/
theorem muonStep_lipschitz (grad : R → R) (lr : ℝ) (a b c : ℝ) (k : ℕ) (M Lu G : ℝ)
    (hgradLip : ∀ x y, ‖grad x - grad y‖ ≤ G * ‖x - y‖)
    (hgradM : ∀ x, ‖grad x‖ ≤ M) (hM : 0 ≤ M)
    (hLu : ∀ i, nsLconst a b c (nsRadius a b c M i) ≤ Lu) :
    ∀ x y, ‖muonStep grad lr a b c k x - muonStep grad lr a b c k y‖
      ≤ (1 + |lr| * Lu ^ k * G) * ‖x - y‖ := by
  have hLu0 : (0 : ℝ) ≤ Lu := (nsLconst_nonneg a b c _).trans (hLu 0)
  have hLuk : (0 : ℝ) ≤ Lu ^ k := pow_nonneg hLu0 k
  intro x y
  have hrw : muonStep grad lr a b c k x - muonStep grad lr a b c k y
      = (x - y) - lr • (nsIter a b c k (grad x) - nsIter a b c k (grad y)) := by
    simp only [muonStep, smul_sub]; abel
  rw [hrw]
  calc ‖(x - y) - lr • (nsIter a b c k (grad x) - nsIter a b c k (grad y))‖
      ≤ ‖x - y‖ + ‖lr • (nsIter a b c k (grad x) - nsIter a b c k (grad y))‖ := norm_sub_le _ _
    _ = ‖x - y‖ + |lr| * ‖nsIter a b c k (grad x) - nsIter a b c k (grad y)‖ := by
        rw [norm_smul, Real.norm_eq_abs]
    _ ≤ ‖x - y‖ + |lr| * (Lu ^ k * ‖grad x - grad y‖) := by
        gcongr
        exact nsIter_lipschitz_uniform a b c M Lu hM (grad x) (grad y) (hgradM x) (hgradM y) k hLu
    _ ≤ ‖x - y‖ + |lr| * (Lu ^ k * (G * ‖x - y‖)) := by
        gcongr
        exact hgradLip x y
    _ = (1 + |lr| * Lu ^ k * G) * ‖x - y‖ := by ring

/-- **THE MUON WHOLE-RUN ERROR INTERVAL.** For the runnable trajectory `θ` within `B` per step of the ideal `θ'`
    (which follows the exact-ℝ Muon step `muonStep`), the geometric interval `L^n·‖θ 0 − θ' 0‖ + B·Σ_{j<n} L^j`
    holds with `L = 1 + |lr|·Lu^k·G` — the concrete full Muon step-Lipschitz. This closes C42's open `L`: the whole-run
    interval now stands for the ACTUAL Muon optimizer (NS orthogonalization + parameter update), with `L` composed from
    C50's NS Lipschitz `Lu^k`, the gradient's Lipschitz `G`, and the affine update. `muonStep_lipschitz` discharges
    C42's `hlip`; `hstep` is the per-step (C1/Float-error) interval `B`. -/
theorem muon_whole_run (grad : R → R) (lr : ℝ) (a b c : ℝ) (k : ℕ) (M Lu G B : ℝ)
    (θ θ' : Nat → R)
    (hgradLip : ∀ x y, ‖grad x - grad y‖ ≤ G * ‖x - y‖)
    (hgradM : ∀ x, ‖grad x‖ ≤ M) (hM : 0 ≤ M)
    (hLu : ∀ i, nsLconst a b c (nsRadius a b c M i) ≤ Lu) (hG : 0 ≤ G)
    (hstep : ∀ n, ‖θ (n + 1) - muonStep grad lr a b c k (θ n)‖ ≤ B)
    (hideal : ∀ n, θ' (n + 1) = muonStep grad lr a b c k (θ' n)) (n : Nat) :
    ‖θ n - θ' n‖ ≤ (1 + |lr| * Lu ^ k * G) ^ n * ‖θ 0 - θ' 0‖
      + B * ∑ j ∈ Finset.range n, (1 + |lr| * Lu ^ k * G) ^ j := by
  have hLu0 : (0 : ℝ) ≤ Lu := (nsLconst_nonneg a b c _).trans (hLu 0)
  have hL0 : (0 : ℝ) ≤ 1 + |lr| * Lu ^ k * G := by
    have : (0 : ℝ) ≤ |lr| * Lu ^ k * G :=
      mul_nonneg (mul_nonneg (abs_nonneg _) (pow_nonneg hLu0 k)) hG
    linarith
  exact muon_whole_run_opnorm_interval (muonStep grad lr a b c k) θ θ' (1 + |lr| * Lu ^ k * G) B hL0
    (muonStep_lipschitz grad lr a b c k M Lu G hgradLip hgradM hM hLu) hstep hideal n

end Puffer.RL.MuonStepLipschitz
