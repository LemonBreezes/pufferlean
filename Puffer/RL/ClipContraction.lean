/-
# Clip-interior forward-invariance UNDER A CONTRACTION: the honest closure of the clip-interior loop

C36 (`ClipBarrier`) established that a fixed clip band `(lo, hi)` is NOT forward-invariant under bounded displacement
without a CONTRACTION — one projected step maps `[lo, hi]` to `[lo − d, hi + d]`, which grows, so a constant band is
never trapping for an expansive (or merely non-contractive) update. Consequently C35 (`TrajReachability`), C38
(`HTrapAssembly`), and C41 (`RunnableRegion`) all left the clip-interior condition `hIntθ`/`hIntσ'`
(`∀ n, ratio(θ n) ∈ (lo, hi)`) as a per-step runtime check rather than a proven `∀ n` invariant.

This module supplies the missing sufficient condition — exactly the one C36 identified as necessary. WITH a
CONTRACTION of the ratio toward a center `c` strictly inside the clip window, the clip interior IS forward-invariant
for ALL `n`. This mirrors C32 (`WeightDecayInterval`), which obtained the horizon-free error bound under a
weight-decay contraction `L < 1` via the same `affine_recur_uniform`.

* `dist_to_center_bounded` — the trapping radius: a ratio sequence `r` obeying a CONTRACTIVE recurrence
  `|r (n+1) − c| ≤ ρ·|r n − c| + ε` with `0 ≤ ρ < 1`, `0 ≤ ε`, stays within `|r 0 − c| + ε/(1 − ρ)` of the center for
  all `n` (directly `affine_recur_uniform` on `a n := |r n − c|`).
* `clip_invariant_under_contraction` — the capstone (abstract sequence): if additionally the center is strictly
  inside the window by more than the trapping radius (`lo < c − radius`, `c + radius < hi`), the sequence stays in the
  OPEN interval `(lo, hi)` for all `n`.
* `ratio_clip_invariant_under_contraction` — instantiated at the PPO ratio `evalR (ratioE (logSoftmaxE chosen (e::es))
  oldLogp) (θ n)`: the clip-interior condition C35/C38/C41 left as a per-step premise is now a THEOREM for all `n`,
  under the ratio's contraction toward an interior center.

**Scope (honestly disclosed).** This proves clip-interior forward-invariance UNDER A CONTRACTION of the ratio toward
a center strictly inside the window — the exact condition C36 flagged as necessary (a fixed band is not trapping
without a contraction). The contractive recurrence `|r (n+1) − c| ≤ ρ·|r n − c| + ε` (`ρ < 1`) is the honest
hypothesis: it is what a CONTRACTIVE / weight-decay update produces (cf. C32), NOT plain projected gradient ascent
(which is expansive — C29/C36). So this closes the clip-interior loop CONDITIONALLY on the contraction, converting
the per-step runtime clip check into a genuine `∀ n` invariant whenever the update contracts the ratio. Establishing
the contractive recurrence for a concrete optimizer — deriving the ratio's contraction constant `ρ` from the update's
contraction (weight decay) composed with the ratio's Lipschitz sensitivity — is the remaining step, analogous to
C42's open Muon step-Lipschitz `L`. `ε` absorbs the per-step non-contractive drift (gradient/rounding); `ρ < 1` is the
net contraction after weight decay.
-/
import Puffer.RL.MuonTrainBound
import Puffer.RL.SurrogateExpr
import Puffer.RL.SoftmaxExpr
open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)
open Puffer.RL.SurrogateExpr (ratioE)
open Puffer.RL.SoftmaxExpr (logSoftmaxE)
open Puffer.RL.MuonTrainBound (affine_recur_uniform)

namespace Puffer.RL.ClipContraction

/-- **The trapping radius under a contraction.** A sequence `r` whose distance to a center `c` obeys the contractive
    affine recurrence `|r (n+1) − c| ≤ ρ·|r n − c| + ε` with `0 ≤ ρ < 1` and `0 ≤ ε` stays within
    `|r 0 − c| + ε/(1 − ρ)` of `c` for ALL `n`. Directly `affine_recur_uniform` (C2/C32's contractive bound) on the
    nonneg sequence `a n := |r n − c|`. -/
theorem dist_to_center_bounded (r : Nat → ℝ) (c ρ ε : ℝ)
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) (hε : 0 ≤ ε)
    (hrec : ∀ n, |r (n + 1) - c| ≤ ρ * |r n - c| + ε) (n : Nat) :
    |r n - c| ≤ |r 0 - c| + ε / (1 - ρ) :=
  affine_recur_uniform (fun k => |r k - c|) ρ ε hρ0 hρ1 hε (abs_nonneg _) hrec n

/-- **Clip-interior forward-invariance under a contraction (abstract).** If a sequence `r` contracts toward a center
    `c` (`|r (n+1) − c| ≤ ρ·|r n − c| + ε`, `ρ < 1`) and `c` sits strictly inside the open window `(lo, hi)` by more
    than the trapping radius (`lo < c − (|r 0 − c| + ε/(1 − ρ))` and `c + (…) < hi`), then `r n ∈ (lo, hi)` for ALL
    `n`. Proof: the trapping radius (`dist_to_center_bounded`) confines `r n` to `[c − radius, c + radius] ⊆ (lo, hi)`.
    This is a fixed band being genuinely forward-invariant — possible precisely BECAUSE the update contracts. -/
theorem clip_invariant_under_contraction (r : Nat → ℝ) (lo hi c ρ ε : ℝ)
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) (hε : 0 ≤ ε)
    (hrec : ∀ n, |r (n + 1) - c| ≤ ρ * |r n - c| + ε)
    (hlo : lo < c - (|r 0 - c| + ε / (1 - ρ)))
    (hhi : c + (|r 0 - c| + ε / (1 - ρ)) < hi)
    (n : Nat) : lo < r n ∧ r n < hi := by
  have hb : |r n - c| ≤ |r 0 - c| + ε / (1 - ρ) := dist_to_center_bounded r c ρ ε hρ0 hρ1 hε hrec n
  rw [abs_le] at hb
  exact ⟨by linarith [hb.1], by linarith [hb.2]⟩

/-- **The PPO ratio stays in the clip window for all `n`, under a contraction.** Instantiates
    `clip_invariant_under_contraction` at the compiled ratio `evalR (ratioE (logSoftmaxE chosen (e::es)) oldLogp)
    (θ n)`: given the ratio contracts toward an interior center `c` along the trajectory, the clip-interior condition
    `∀ n, toReal lo < ratio(θ n) < toReal hi` — which C35/C38/C41 took as a per-step premise (`hIntθ`/`hIntσ'`) — is
    a THEOREM. This discharges the last per-step runtime clip check as a genuine invariant on the contractive regime. -/
theorem ratio_clip_invariant_under_contraction (chosen e : Expr) (es : List Expr) (oldLogp lo hi : Float)
    (θ : Nat → (Nat → ℝ)) (c ρ ε : ℝ)
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) (hε : 0 ≤ ε)
    (hrec : ∀ n, |evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ (n + 1)) - c|
        ≤ ρ * |evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ n) - c| + ε)
    (hlo : toReal lo < c
        - (|evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ 0) - c| + ε / (1 - ρ)))
    (hhi : c + (|evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ 0) - c| + ε / (1 - ρ))
        < toReal hi)
    (n : Nat) :
    toReal lo < evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ n)
      ∧ evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ n) < toReal hi :=
  clip_invariant_under_contraction
    (fun m => evalR (ratioE (logSoftmaxE chosen (e :: es)) oldLogp) (θ m))
    (toReal lo) (toReal hi) c ρ ε hρ0 hρ1 hε hrec hlo hhi n

end Puffer.RL.ClipContraction
