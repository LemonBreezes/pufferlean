/-
Continuous-action (Gaussian) policy terms — the runnable `Float` differential entropy within a certified
bound of its ℝ spec (`Puffer.RL.PPO.gaussianEntropy`). Completes the runtime coverage of the PPO ℝ specs
for the continuous-action head (PufferLib's Gaussian policies), alongside the discrete-action PPO runtime
(`PPORuntime`).

`gaussianEntropy logStd = ½·(1 + log 2π) + logStd`. The `log 2π` term is a precomputed constant in the C, so
the runnable kernel takes it as an explicit `Float` parameter `log2pi` and we bound against the
constant-parameterized ℝ reference `gaussianEntropyR` (`gaussianEntropyR_eq`: at the exact constant it IS
`gaussianEntropy`; the `toReal log2pi` vs `log 2π` gap is a separate constant-accuracy piece).

  • `gaussianEntropyR` / `gaussianEntropyR_eq` — the parameterized ℝ reference and its coincidence with the
    true spec at `log2pi = log 2π`.
  • `gaussianEntropyF` / `gaussianEntropyF_error` — the runnable `0.5·(1 + log2pi) + logStd` within
    `gaussianEntropyErrBnd` (`addApprox`/`mulApprox` op-chain; the `0.5` literal gap via
    `toReal_ofScientific_close`, the `1` via `toReal_one`).
  • `gaussianLogpR` / `gaussianLogpR_eq` / `gaussianLogpF` / `gaussianLogpF_error` — the log-prob
    `−0.5·((a−μ)/exp logStd)² − 0.5·log2pi − logStd`: the full circuit `exp σ` (`exp_error`) → sub → a FLOORED
    div `(a−μ)/σ` (`divApprox_error`, denominator floor `dmin` supplied as a hypothesis, `σ = exp > 0`) →
    square (`mulApprox`) → the `−0.5·` scale (`negHalf_gap` for the `−0.5` literal via `toReal_neg`) →
    two subs. `negHalf_gap` handles the `−0.5` ofScientific gap.

Axiom-clean beyond the trusted Float base (`add/sub/mul/div/exp_model` + `toReal` + `toReal_one`/`toReal_neg`
+ `toReal_ofScientific_close`).
-/
import Mathlib
import Puffer.Float.Basic
import Puffer.RL.PPO

namespace Puffer.RL.GaussianRuntime

open Puffer.FloatR
open Puffer.RL.PPO (gaussianEntropy gaussianLogp)

/-- ℝ reference with the `log 2π` constant as a parameter (the C uses a precomputed literal). -/
noncomputable def gaussianEntropyR (logStd log2pi : ℝ) : ℝ := 0.5 * (1 + log2pi) + logStd

/-- When the constant is exactly `log 2π`, the parameterized reference IS `gaussianEntropy`. -/
theorem gaussianEntropyR_eq (logStd : ℝ) :
    gaussianEntropyR logStd (Real.log (2 * Real.pi)) = gaussianEntropy logStd := rfl

/-- Executable Gaussian entropy `0.5·(1 + log2pi) + logStd`. -/
def gaussianEntropyF (logStd log2pi : Float) : Float := 0.5 * (1 + log2pi) + logStd

/-- Certified rounding budget for `gaussianEntropyF` (the `1+·`, `0.5·`, `+logStd` circuit). -/
noncomputable def gaussianEntropyErrBnd (logStd log2pi : Float) : ℝ :=
  u64 * |toReal (0.5 * (1 + log2pi)) + toReal logStd|
    + (u64 * (|toReal (0.5 : Float)| * |toReal (1 + log2pi)|)
        + |toReal (0.5 : Float)| * (u64 * |(1 : ℝ) + toReal log2pi|)
        + |(1 : ℝ) + toReal log2pi| * (u64 * |(0.5 : ℝ)|))

/-- **Gaussian-entropy runtime error.** The running `gaussianEntropyF logStd log2pi` deviates from the
    constant-parameterized ℝ reference `gaussianEntropyR (toReal logStd) (toReal log2pi)` by at most
    `gaussianEntropyErrBnd`. -/
theorem gaussianEntropyF_error (logStd log2pi : Float) :
    |toReal (gaussianEntropyF logStd log2pi)
        - gaussianEntropyR (toReal logStd) (toReal log2pi)|
      ≤ gaussianEntropyErrBnd logStd log2pi := by
  have h0 : ∀ z : Float, |toReal z - toReal z| ≤ 0 := fun z => by simp
  have hinner : |toReal (1 + log2pi) - (1 + toReal log2pi)|
      ≤ u64 * |toReal (1 : Float) + toReal log2pi| := by
    have := addApprox_error (1 : Float) log2pi 1 (toReal log2pi) 0 0 (by simp) (h0 log2pi)
    simpa using this
  have h05 : |toReal (0.5 : Float) - (0.5 : ℝ)| ≤ u64 * |(0.5 : ℝ)| :=
    toReal_ofScientific_close 5 true 1
  have hscaled : |toReal (0.5 * (1 + log2pi)) - (0.5 : ℝ) * (1 + toReal log2pi)|
      ≤ u64 * (|toReal (0.5 : Float)| * |toReal (1 + log2pi)|)
        + |toReal (0.5 : Float)| * (u64 * |(1 : ℝ) + toReal log2pi|)
        + |(1 : ℝ) + toReal log2pi| * (u64 * |(0.5 : ℝ)|) := by
    have := mulApprox_error (0.5 : Float) (1 + log2pi) (0.5 : ℝ) (1 + toReal log2pi)
      (u64 * |(0.5 : ℝ)|) (u64 * |toReal (1 : Float) + toReal log2pi|) h05 hinner
    simpa using this
  have hres := addApprox_error (0.5 * (1 + log2pi)) logStd ((0.5 : ℝ) * (1 + toReal log2pi))
    (toReal logStd)
    (u64 * (|toReal (0.5 : Float)| * |toReal (1 + log2pi)|)
      + |toReal (0.5 : Float)| * (u64 * |(1 : ℝ) + toReal log2pi|)
      + |(1 : ℝ) + toReal log2pi| * (u64 * |(0.5 : ℝ)|)) 0 hscaled (h0 logStd)
  rw [gaussianEntropyF, gaussianEntropyR]
  simpa [gaussianEntropyErrBnd] using hres

/-! ### Gaussian log-prob (exp + floored div + square) -/

/-- The `−0.5` literal's rounding gap (via `toReal_neg` + `toReal_ofScientific_close`). -/
theorem negHalf_gap : |toReal (-0.5 : Float) - (-0.5 : ℝ)| ≤ u64 * |(-0.5 : ℝ)| := by
  have h := toReal_ofScientific_close 5 true 1
  have e1 : toReal (-0.5 : Float) - (-0.5 : ℝ) = -(toReal (0.5 : Float) - (0.5 : ℝ)) := by
    rw [toReal_neg]; ring
  rw [e1, abs_neg]
  calc |toReal (0.5 : Float) - (0.5 : ℝ)| ≤ u64 * |(0.5 : ℝ)| := h
    _ = u64 * |(-0.5 : ℝ)| := by norm_num

/-- ℝ Gaussian log-prob with the `log 2π` constant as a parameter. -/
noncomputable def gaussianLogpR (action mean logStd log2pi : ℝ) : ℝ :=
  -0.5 * ((action - mean) / Real.exp logStd) ^ 2 - 0.5 * log2pi - logStd

theorem gaussianLogpR_eq (action mean logStd : ℝ) :
    gaussianLogpR action mean logStd (Real.log (2 * Real.pi)) = gaussianLogp action mean logStd := rfl

/-- **The Gaussian log-density is maximized in the width `logStd` at the single-sample MLE
    `logStd = log|action − mean|`.** For a fixed observed `action ≠ mean`, viewing
    `gaussianLogpR action mean · log2pi` as a function of the log-standard-deviation, its unique global maximizer
    is `logStd = Real.log |action − mean|` (i.e. `σ = e^{logStd} = |action − mean|`): for every `logStd`,
    `gaussianLogpR action mean logStd log2pi ≤ gaussianLogpR action mean (log|action−mean|) log2pi`. This is the
    maximum-likelihood estimate of the width from ONE observation. The maximizer is independent of the `log 2π`
    constant `log2pi` the runtime carries (it only shifts the level, not the argmax), so a mildly inaccurate
    precomputed `log2pi` cannot move the MLE width. This complements the file/`PPO`'s action-side maximization
    (`gaussianLogp_le_at_mean`, the mode at `μ`) with maximization over the WIDTH. The hypothesis `action ≠ mean`
    is load-bearing: at `action = mean` the density `−0.5·log2pi − logStd` has NO maximum in `logStd` (it grows
    without bound as `logStd → −∞`), so the inequality fails (e.g. `logStd = −1` beats `logStd = log 0 = 0`).
    Proof: at the MLE width the standardized deviation is exactly `1`, and the general-width squared standardized
    deviation equals `e^{−2(logStd − log|action−mean|)}`; the claim then reduces to `1 − 2t ≤ e^{−2t}`
    (`Real.add_one_le_exp`) with `t = logStd − log|action−mean|`. -/
theorem gaussianLogpR_le_logStd_mle (action mean logStd log2pi : ℝ) (hd : action ≠ mean) :
    gaussianLogpR action mean logStd log2pi
      ≤ gaussianLogpR action mean (Real.log |action - mean|) log2pi := by
  have hdne : action - mean ≠ 0 := sub_ne_zero.mpr hd
  have habs : (0 : ℝ) < |action - mean| := abs_pos.mpr hdne
  have hexplog : Real.exp (Real.log |action - mean|) = |action - mean| := Real.exp_log habs
  unfold gaussianLogpR
  -- The standardized deviation at the MLE width is exactly 1.
  have hsq1 : ((action - mean) / Real.exp (Real.log |action - mean|)) ^ 2 = 1 := by
    rw [hexplog, div_pow, sq_abs, div_self (pow_ne_zero 2 hdne)]
  rw [hsq1]
  -- exp(2 log|d|) = d²
  have e2 : Real.exp (2 * Real.log |action - mean|) = (action - mean) ^ 2 := by
    rw [two_mul, Real.exp_add, hexplog, ← pow_two, sq_abs]
  -- exp(−2(logStd − log|d|)) = d² / (exp logStd)²
  have hexp2 : Real.exp (-2 * (logStd - Real.log |action - mean|))
      = (action - mean) ^ 2 / (Real.exp logStd) ^ 2 := by
    rw [show (-2 * (logStd - Real.log |action - mean|))
          = 2 * Real.log |action - mean| - 2 * logStd from by ring]
    rw [Real.exp_sub, e2]
    congr 1
    rw [two_mul, Real.exp_add, ← pow_two]
  -- The squared standardized deviation at width logStd equals exp(−2(logStd − log|d|)).
  have hq2 : ((action - mean) / Real.exp logStd) ^ 2
      = Real.exp (-2 * (logStd - Real.log |action - mean|)) := by
    rw [hexp2, div_pow]
  rw [hq2]
  -- Core scalar inequality: 1 − 2t ≤ e^{−2t}.
  have hkey := Real.add_one_le_exp (-2 * (logStd - Real.log |action - mean|))
  linarith [hkey]

/-- **Entropy plus log-density = ½ − ½·(standardized deviation)².** For the constant-parameterized ℝ references
    the continuous head emits, `gaussianEntropyR logStd log2pi + gaussianLogpR action mean logStd log2pi
    = ½ − ½·q²` with `q = (action − mean)/e^{logStd}` — both the `log 2π` constant and the width `logStd` cancel,
    leaving only the halved squared standardized deviation. The differential entropy of the policy and the height
    of its own log-density are thus linked pointwise, independent of the policy's scale. -/
theorem gaussianEntropyR_add_gaussianLogpR (action mean logStd log2pi : ℝ) :
    gaussianEntropyR logStd log2pi + gaussianLogpR action mean logStd log2pi
      = 0.5 - 0.5 * ((action - mean) / Real.exp logStd) ^ 2 := by
  unfold gaussianEntropyR gaussianLogpR
  ring

/-- **Entropy plus peak log-density is exactly ½.** At the mode (`action = mean`) the squared deviation vanishes,
    so `gaussianEntropyR logStd log2pi + gaussianLogpR mean mean logStd log2pi = ½` — the classic Gaussian identity
    `H = ½ − log f(mode)`, independent of the width `logStd` and the `log 2π` constant. Links the two exact
    quantities the continuous-action head emits (entropy bonus and log-prob). Corollary of the general form. -/
theorem gaussianEntropyR_add_gaussianLogpR_at_mean (mean logStd log2pi : ℝ) :
    gaussianEntropyR logStd log2pi + gaussianLogpR mean mean logStd log2pi = 0.5 := by
  rw [gaussianEntropyR_add_gaussianLogpR]; simp

/-- Executable Gaussian log-prob `−0.5·q² − 0.5·log2pi − logStd`, `q = (a−μ)/exp(logStd)`. -/
def gaussianLogpF (action mean logStd log2pi : Float) : Float :=
  -0.5 * (((action - mean) / Float.exp logStd) * ((action - mean) / Float.exp logStd))
    - 0.5 * log2pi - logStd

/-- Intermediate real values and error budgets (denominator floor `dmin`). -/
noncomputable def gaussianLogpErrBnd (action mean logStd log2pi : Float) (dmin : ℝ) : ℝ :=
  let σR := Real.exp (toReal logStd)
  let εσ := expEps * Real.exp (toReal logStd)
  let dR := toReal action - toReal mean
  let εd := u64 * |toReal action - toReal mean|
  let q := (action - mean) / Float.exp logStd
  let qR := dR / σR
  let εq := u64 * |toReal (action - mean) / toReal (Float.exp logStd)| + (εd + |qR| * εσ) / dmin
  let εq2 := u64 * |toReal q * toReal q| + |toReal q| * εq + |qR| * εq
  let εt1 := u64 * |toReal (-0.5 : Float) * toReal (q * q)|
    + |toReal (-0.5 : Float)| * εq2 + |qR ^ 2| * (u64 * |(-0.5 : ℝ)|)
  let εt2 := u64 * |toReal (0.5 : Float) * toReal log2pi| + |toReal log2pi| * (u64 * |(0.5 : ℝ)|)
  let εs1 := u64 * |toReal (-0.5 * (q * q)) - toReal (0.5 * log2pi)| + εt1 + εt2
  u64 * |toReal (-0.5 * (q * q) - 0.5 * log2pi) - toReal logStd| + εs1

theorem gaussianLogpF_error (action mean logStd log2pi : Float) (dmin : ℝ)
    (hdmin : 0 < dmin) (hdσ : dmin ≤ |toReal (Float.exp logStd)|) :
    |toReal (gaussianLogpF action mean logStd log2pi)
        - gaussianLogpR (toReal action) (toReal mean) (toReal logStd) (toReal log2pi)|
      ≤ gaussianLogpErrBnd action mean logStd log2pi dmin := by
  have h0 : ∀ z : Float, |toReal z - toReal z| ≤ 0 := fun z => by simp
  set σR := Real.exp (toReal logStd) with hσR
  have hσRne : σR ≠ 0 := Real.exp_ne_zero _
  -- d = action - mean
  have hd : |toReal (action - mean) - (toReal action - toReal mean)|
      ≤ u64 * |toReal action - toReal mean| := sub_error action mean
  -- σ = exp logStd
  have hσ : |toReal (Float.exp logStd) - σR| ≤ expEps * Real.exp (toReal logStd) := exp_error logStd
  -- q = d / σ
  have hq : |toReal ((action - mean) / Float.exp logStd)
        - (toReal action - toReal mean) / σR|
      ≤ u64 * |toReal (action - mean) / toReal (Float.exp logStd)|
        + (u64 * |toReal action - toReal mean|
            + |(toReal action - toReal mean) / σR| * (expEps * Real.exp (toReal logStd))) / dmin :=
    divApprox_error (action - mean) (Float.exp logStd) (toReal action - toReal mean) σR
      (u64 * |toReal action - toReal mean|) (expEps * Real.exp (toReal logStd)) dmin hd hσ hdmin hdσ hσRne
  set q := (action - mean) / Float.exp logStd with hqdef
  set qR := (toReal action - toReal mean) / σR with hqRdef
  set εq := u64 * |toReal (action - mean) / toReal (Float.exp logStd)|
    + (u64 * |toReal action - toReal mean| + |qR| * (expEps * Real.exp (toReal logStd))) / dmin with hεq
  -- q² = q * q
  have hq2 : |toReal (q * q) - qR * qR|
      ≤ u64 * |toReal q * toReal q| + |toReal q| * εq + |qR| * εq :=
    mulApprox_error q q qR qR εq εq hq hq
  set εq2 := u64 * |toReal q * toReal q| + |toReal q| * εq + |qR| * εq with hεq2
  -- t1 = -0.5 * q²
  have ht1 : |toReal (-0.5 * (q * q)) - (-0.5 : ℝ) * (qR * qR)|
      ≤ u64 * |toReal (-0.5 : Float) * toReal (q * q)|
        + |toReal (-0.5 : Float)| * εq2 + |qR * qR| * (u64 * |(-0.5 : ℝ)|) :=
    mulApprox_error (-0.5 : Float) (q * q) (-0.5 : ℝ) (qR * qR)
      (u64 * |(-0.5 : ℝ)|) εq2 negHalf_gap hq2
  -- t2 = 0.5 * log2pi
  have h05 : |toReal (0.5 : Float) - (0.5 : ℝ)| ≤ u64 * |(0.5 : ℝ)| :=
    toReal_ofScientific_close 5 true 1
  have ht2 : |toReal (0.5 * log2pi) - (0.5 : ℝ) * toReal log2pi|
      ≤ u64 * |toReal (0.5 : Float) * toReal log2pi| + |toReal log2pi| * (u64 * |(0.5 : ℝ)|) := by
    have := mulApprox_error (0.5 : Float) log2pi (0.5 : ℝ) (toReal log2pi)
      (u64 * |(0.5 : ℝ)|) 0 h05 (h0 log2pi)
    simpa using this
  -- s1 = t1 - t2
  have hs1 := subApprox_error (-0.5 * (q * q)) (0.5 * log2pi)
    ((-0.5 : ℝ) * (qR * qR)) ((0.5 : ℝ) * toReal log2pi)
    (u64 * |toReal (-0.5 : Float) * toReal (q * q)|
      + |toReal (-0.5 : Float)| * εq2 + |qR * qR| * (u64 * |(-0.5 : ℝ)|))
    (u64 * |toReal (0.5 : Float) * toReal log2pi| + |toReal log2pi| * (u64 * |(0.5 : ℝ)|))
    ht1 ht2
  -- res = s1 - logStd
  have hres := subApprox_error (-0.5 * (q * q) - 0.5 * log2pi) logStd
    ((-0.5 : ℝ) * (qR * qR) - (0.5 : ℝ) * toReal log2pi) (toReal logStd) _ 0 hs1 (h0 logStd)
  rw [gaussianLogpF, gaussianLogpR]
  simpa [gaussianLogpErrBnd, hσR, hqdef, hqRdef, hεq, hεq2, sq] using hres

end Puffer.RL.GaussianRuntime
