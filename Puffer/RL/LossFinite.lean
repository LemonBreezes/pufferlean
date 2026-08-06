/-
# Head-stage finiteness: lifting C43/C45's no-overflow certificate to the readout / policy head

C43 (`FiniteBound`) certified a single bounded linear kernel (`dotF`) overflow-free (from the one trusted
no-overflow axiom `isFinite_of_bounded` + the `(1+δ)` base); C45 (`ForwardFinite`) lifted that to a full MLP forward
pass (`mlpF`). This module carries the certificate ONE STAGE FURTHER — to the READOUT / policy-head operations
computed from the (bounded, finite) forward-pass output: an affine readout, a subtraction, and the `Float.exp` used
in the softmax numerator, plus the end-to-end `mlpF → readout` composition. Each is a magnitude bound propagated from
the `(1+δ)` base, then `isFinite_of_bounded` for overflow-freedom — exactly C43/C45's pattern. NO new axiom.

* `sub_mag_le` / `sub_isFinite` — a Float subtraction (`sub_model`): `|toReal (a − b)| ≤ (1+u)·(|a|+|b|)`, overflow-free
  when that is within threshold (the missing sibling of C43's `add`/`mul` magnitude lemmas).
* `outF` / `outF_mag_le` / `outF_isFinite` — an affine readout head `dotF h w + b`: magnitude `≤ (1+u)·(dotBound d B +
  Bb)` and overflow-free when that is within threshold (composes C43's `dotF_mag_le` with `add_mag_le`).
* `expF_mag_le` / `expF_isFinite` — `Float.exp` on a magnitude-bounded logit (`exp_model`): `|toReal (Float.exp a)| ≤
  exp(M)·(1+expEps)` when `toReal a ≤ M`, overflow-free when that is within threshold. The softmax numerator's
  no-overflow certificate — needing the logit bounded ABOVE (exp grows fast; unbounded logits genuinely overflow).
* `mlpMagBound_nonneg` / `mlpF_length_le` — the forward-pass output is nonneg-bounded and width-bounded (helpers for
  the composition).
* `forwardHead_isFinite` — **the payoff**: the whole `mlpF` forward pass FOLLOWED BY the affine readout is overflow-free
  when the composed bound `(1+u)·(dotBound d (max (mlpMagBound d Bw B0 Ws) Bw) + Bb) ≤ overflowBound`. Chains C45's
  `mlpF_mag_le` (forward output bounded) with `outF_isFinite` (readout on the bounded output).

**Scope (honestly disclosed).** This extends C43/C45's finiteness certificate from the forward pass up to the READOUT
HEAD (affine readout + the softmax `exp`), reusing C43's `isFinite_of_bounded` (the single trusted no-overflow fact) +
the `(1+δ)` base — NO new axiom. It covers: the MLP forward pass (C45) → an affine readout (`outF`) → optionally the
softmax `exp` on bounded logits (`expF`). It does NOT cover the FULL loss (the clip/ratio/log-partition/entropy
assembly), the BACKWARD / AD pass, or the OPTIMIZER — a full-trainer finiteness would compose those next. The magnitude
bound GROWS through the readout (the `(1+u)·(dotBound d B + Bb)` and, for `exp`, the `exp(M)` factor), realistically:
unbounded logits/weights genuinely can overflow, and `… ≤ overflowBound` is exactly the checkable condition under which
they provably do not. The `exp` certificate needs the logit bounded ABOVE (`toReal a ≤ M`) — the honest hypothesis for
a fast-growing op.
-/
import Puffer.RL.ForwardFinite
open Puffer.FloatR
open Puffer.RL.FiniteBound
open Puffer.RL.ForwardFinite

namespace Puffer.RL.LossFinite

/-- `0 ≤ 1 + u64` (roundoff factor positive). -/
private theorem one_add_u64_nonneg : (0 : ℝ) ≤ 1 + u64 := by have := u64_pos; linarith

/-- **Subtraction magnitude propagation** (`(1+u)` growth, PROVED from `sub_model`; no new axiom):
    `|toReal (a − b)| ≤ (1+u)·(|toReal a| + |toReal b|)` — the missing sibling of C43's `add_mag_le`/`mul_mag_le`. -/
theorem sub_mag_le (a b : Float) : |toReal (a - b)| ≤ (1 + u64) * (|toReal a| + |toReal b|) := by
  obtain ⟨δ, hδ, he⟩ := sub_model a b
  rw [he, abs_mul]
  have h1 : |1 + δ| ≤ 1 + u64 := (abs_add_le _ _).trans (by simpa using add_le_add_left hδ 1)
  calc |toReal a - toReal b| * |1 + δ|
      ≤ (|toReal a| + |toReal b|) * (1 + u64) :=
        mul_le_mul (abs_sub _ _) h1 (abs_nonneg _) (add_nonneg (abs_nonneg _) (abs_nonneg _))
    _ = (1 + u64) * (|toReal a| + |toReal b|) := by ring

/-- **A single Float subtraction is overflow-free** when the `(1+u)`-inflated operand-magnitude sum is within the
    overflow threshold. Proved from `isFinite_of_bounded` + `sub_mag_le`. -/
theorem sub_isFinite (a b : Float) (h : (1 + u64) * (|toReal a| + |toReal b|) ≤ overflowBound) :
    (a - b).isFinite = true :=
  isFinite_of_bounded _ ((sub_mag_le a b).trans h)

/-- **An affine readout head**: a `dotF` of the (bounded) activation `h` with the readout weights `w`, plus a bias `b`. -/
def outF (h : List Float) (w : List Float) (b : Float) : Float := FiniteBound.dotF h w + b

/-- **Readout magnitude propagation.** With activation and weight magnitudes `≤ B` (activation length `≤ d`) and bias
    `≤ Bb`, `|toReal (outF h w b)| ≤ (1+u)·(dotBound d B + Bb)` — the `dotF` bounded by C43's `dotF_mag_le` (lifted to
    the uniform width `d` by `dotBound_mono`), then the bias added via `add_mag_le`. -/
theorem outF_mag_le (B Bb : ℝ) (hB : 0 ≤ B) (d : ℕ) (h w : List Float) (b : Float)
    (hhd : h.length ≤ d) (hh : ∀ v ∈ h, |toReal v| ≤ B) (hw : ∀ v ∈ w, |toReal v| ≤ B)
    (hb : |toReal b| ≤ Bb) :
    |toReal (outF h w b)| ≤ (1 + u64) * (dotBound d B + Bb) := by
  have hdot : |toReal (FiniteBound.dotF h w)| ≤ dotBound d B :=
    (dotF_mag_le B hB h w hh hw).trans (dotBound_mono B ((min_le_left _ _).trans hhd))
  show |toReal (FiniteBound.dotF h w + b)| ≤ (1 + u64) * (dotBound d B + Bb)
  calc |toReal (FiniteBound.dotF h w + b)|
      ≤ (1 + u64) * (|toReal (FiniteBound.dotF h w)| + |toReal b|) := add_mag_le (FiniteBound.dotF h w) b
    _ ≤ (1 + u64) * (dotBound d B + Bb) :=
        mul_le_mul_of_nonneg_left (add_le_add hdot hb) one_add_u64_nonneg

/-- **An affine readout head is overflow-free** when its bound is within threshold. -/
theorem outF_isFinite (B Bb : ℝ) (hB : 0 ≤ B) (d : ℕ) (h w : List Float) (b : Float)
    (hhd : h.length ≤ d) (hh : ∀ v ∈ h, |toReal v| ≤ B) (hw : ∀ v ∈ w, |toReal v| ≤ B)
    (hb : |toReal b| ≤ Bb) (hbound : (1 + u64) * (dotBound d B + Bb) ≤ overflowBound) :
    (outF h w b).isFinite = true :=
  isFinite_of_bounded _ ((outF_mag_le B Bb hB d h w b hhd hh hw hb).trans hbound)

/-- **`Float.exp` magnitude propagation on a bounded logit** (`exp_model`): with the logit bounded ABOVE
    (`toReal a ≤ M`), `|toReal (Float.exp a)| ≤ exp(M)·(1+expEps)`. The upper bound on the logit is essential — `exp`
    grows fast, so an unbounded logit genuinely overflows. -/
theorem expF_mag_le (a : Float) (M : ℝ) (hM : toReal a ≤ M) :
    |toReal (Float.exp a)| ≤ Real.exp M * (1 + expEps) := by
  obtain ⟨δ, hδ, he⟩ := exp_model a
  rw [he, abs_mul, abs_of_pos (Real.exp_pos _)]
  have h1 : |1 + δ| ≤ 1 + expEps := (abs_add_le _ _).trans (by simpa using add_le_add_left hδ 1)
  exact mul_le_mul (Real.exp_le_exp.mpr hM) h1 (abs_nonneg _) (Real.exp_pos _).le

/-- **`Float.exp` on a bounded logit is overflow-free** when `exp(M)·(1+expEps) ≤ overflowBound`. The softmax
    numerator's no-overflow certificate (given the logit bounded above). -/
theorem expF_isFinite (a : Float) (M : ℝ) (hM : toReal a ≤ M)
    (hbound : Real.exp M * (1 + expEps) ≤ overflowBound) :
    (Float.exp a).isFinite = true :=
  isFinite_of_bounded _ ((expF_mag_le a M hM).trans hbound)

/-- The forward-pass magnitude bound is nonnegative (each layer maps to a nonneg `dotBound`; the base is the input
    bound). By induction over the layers. -/
theorem mlpMagBound_nonneg (d : ℕ) (Bw : ℝ) :
    ∀ (Ws : List (List (List Float))) (B : ℝ), 0 ≤ B → 0 ≤ mlpMagBound d Bw B Ws := by
  intro Ws
  induction Ws with
  | nil => intro B hB; simpa [mlpMagBound] using hB
  | cons _ rest ih =>
      intro B _
      simp only [mlpMagBound]
      exact ih (dotBound d (max B Bw)) (dotBound_nonneg (max B Bw) d)

/-- The forward-pass output width stays within the uniform bound `d` (each layer's output length = its neuron count
    `≤ d`, and the input length `≤ d`). By induction over the layers. -/
theorem mlpF_length_le (d : ℕ) :
    ∀ (Ws : List (List (List Float))) (x : List Float),
      (∀ W ∈ Ws, W.length ≤ d) → x.length ≤ d → (mlpF x Ws).length ≤ d := by
  intro Ws
  induction Ws with
  | nil => intro x _ hxd; simpa [mlpF] using hxd
  | cons W rest ih =>
      intro x hwidth hxd
      have hstep : (reluLayerF x W).length ≤ d := by
        rw [reluLayerF_length]; exact hwidth W (List.mem_cons.mpr (Or.inl rfl))
      show (mlpF (reluLayerF x W) rest).length ≤ d
      exact ih (reluLayerF x W) (fun W' hW' => hwidth W' (List.mem_cons.mpr (Or.inr hW'))) hstep

/-- **THE FORWARD-PASS + READOUT NO-OVERFLOW CERTIFICATE.** With magnitude-bounded weights (`≤ Bw`) and input (`≤ B0`),
    a uniform width bound `d`, readout weights `≤ Bw` and bias `≤ Bb`, and the composed bound within threshold, the
    whole MLP forward pass FOLLOWED BY the affine readout `outF (mlpF x Ws) wout bout` is overflow-free. Chains C45's
    `mlpF_mag_le` (the forward output is magnitude-bounded by `mlpMagBound d Bw B0 Ws` and width-bounded by `d`) with
    `outF_isFinite` (the readout on that bounded output), using the common bound `max (mlpMagBound …) Bw`. NO new axiom
    (C43's `isFinite_of_bounded` + the `(1+δ)` base). -/
theorem forwardHead_isFinite (d : ℕ) (Bw B0 Bb : ℝ) (hBw : 0 ≤ Bw) (hB0 : 0 ≤ B0)
    (Ws : List (List (List Float))) (hwidth : ∀ W ∈ Ws, W.length ≤ d)
    (hWs : ∀ W ∈ Ws, ∀ row ∈ W, ∀ w ∈ row, |toReal w| ≤ Bw)
    (x : List Float) (hxd : x.length ≤ d) (hx : ∀ v ∈ x, |toReal v| ≤ B0)
    (wout : List Float) (bout : Float)
    (hwout : ∀ v ∈ wout, |toReal v| ≤ Bw) (hbout : |toReal bout| ≤ Bb)
    (hbound : (1 + u64) * (dotBound d (max (mlpMagBound d Bw B0 Ws) Bw) + Bb) ≤ overflowBound) :
    (outF (mlpF x Ws) wout bout).isFinite = true := by
  have hhmem : ∀ v ∈ mlpF x Ws, |toReal v| ≤ mlpMagBound d Bw B0 Ws :=
    mlpF_mag_le d Bw hBw Ws hwidth hWs x B0 hB0 hxd hx
  have hhlen : (mlpF x Ws).length ≤ d := mlpF_length_le d Ws x hwidth hxd
  have hBoutn : 0 ≤ mlpMagBound d Bw B0 Ws := mlpMagBound_nonneg d Bw Ws B0 hB0
  have hBmax : (0 : ℝ) ≤ max (mlpMagBound d Bw B0 Ws) Bw := le_max_of_le_right hBw
  have hh' : ∀ v ∈ mlpF x Ws, |toReal v| ≤ max (mlpMagBound d Bw B0 Ws) Bw :=
    fun v hv => (hhmem v hv).trans (le_max_left _ _)
  have hw' : ∀ v ∈ wout, |toReal v| ≤ max (mlpMagBound d Bw B0 Ws) Bw :=
    fun v hv => (hwout v hv).trans (le_max_right _ _)
  exact outF_isFinite (max (mlpMagBound d Bw B0 Ws) Bw) Bb hBmax d (mlpF x Ws) wout bout
    hhlen hh' hw' hbout hbound

end Puffer.RL.LossFinite
