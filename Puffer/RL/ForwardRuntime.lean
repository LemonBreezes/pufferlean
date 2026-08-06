/-
Composed error bound for the MLP FORWARD PASS — the computation every training step is
built from (forward → loss → backprop → optimizer). NB: this is the forward pass, NOT the
weight-update arithmetic itself (that is the optimizer axpy, bounded separately in
`Puffer/RL/UpdateRuntime.lean`); the two are the FRONT and BACK ends of a training step and
are not yet chained — the connecting gradient error `εg` is the open middle (see PLAN.md M8).

The forward pass is, per neuron, a linear unit `z = b + dotF w x`, a ReLU, then a second
linear unit for the output logits:

    z1ⱼ = b1ⱼ + dotF W1ⱼ x        hⱼ = reluF z1ⱼ        logitₖ = b2ₖ + dotF W2ₖ h

Each piece already has (or is one step from) a proven Float-vs-ℝ bound: `dotF_error`
bounds the dot, `addApprox_error` the bias add, and `reluF` is EXACT (`toReal_reluF`)
and 1-Lipschitz. This module COMPOSES them into a per-neuron / per-logit bound:

  * `linZ_error`   — one linear unit `b + dotF w x` vs `toReal b + dotR w x`.
  * `reluF_error`  — ReLU propagates its input error with NO growth (coefficient 1).
  * `neuron_error` — a full dense-ReLU neuron vs its ideal real value.
  * `dotR_perturb` — how the second layer's dot moves when its input (the hidden)
                     moves; the cross term that folds the hidden error into the logit.
  * `logit_error`  — a full output logit vs the ideal real logit `toReal b2 + dotRm W2 hR`.

The runnable `Float` twins of these bounds live in `Puffer/Float/ErrBnd.lean`
(`z1ErrF`, `sumAbsMulF`, `logitErrF`, `fwdHidden`, `fwdLogits`, …); `puffer verify-fwd`
emits them and `tools/verify_fwd_ref.py` checks the actual error stays inside, against
an independent exact-ℝ reference.
-/
import Puffer.Float.Net
import Puffer.Net.Forward

namespace Puffer.FloatR

/-- Real dot of Float weights with a REAL vector (the ℝ meaning of a layer-2 dot when
    the hidden vector is taken at its ideal real values `hR`, not its rounded Floats). -/
noncomputable def dotRm : List Float → List ℝ → ℝ
  | [], _ => 0
  | _, [] => 0
  | w :: ws, r :: rs => toReal w * r + dotRm ws rs

/-- ℝ error bound of one pre-activation `z = b + dotF w x`: the dot bound plus one
    bias-add rounding. (The Float twin is `z1ErrF` in `ErrBnd.lean`.) -/
noncomputable def z1ErrBnd (w : List Float) (b : Float) (x : List Float) : ℝ :=
  u64 * |toReal b + toReal (dotF w x)| + dotErrBnd w x

theorem z1ErrBnd_nonneg (w : List Float) (b : Float) (x : List Float) : 0 ≤ z1ErrBnd w b x :=
  add_nonneg (mul_nonneg u64_pos.le (abs_nonneg _)) (dotErrBnd_nonneg _ _)

/-- **Linear-unit error.** `z = b + dotF w x` (bias-first, as in `forwardAll`) is within
    `z1ErrBnd` of its exact real value `toReal b + dotR w x`. -/
theorem linZ_error (w : List Float) (b : Float) (x : List Float) :
    |toReal (b + dotF w x) - (toReal b + dotR w x)| ≤ z1ErrBnd w b x := by
  have h := addApprox_error b (dotF w x) (toReal b) (dotR w x) 0 (dotErrBnd w x)
    (by simp) (dotF_error w x)
  simpa [z1ErrBnd] using h

/-- **ReLU error propagation.** `reluF` is exact and 1-Lipschitz, so it passes its input
    error through with no amplification and adds no new rounding. -/
theorem reluF_error (z : Float) (zR ε : ℝ) (h : |toReal z - zR| ≤ ε) :
    |toReal (reluF z) - max zR 0| ≤ ε := by
  rw [toReal_reluF]
  exact (abs_max_sub_max_le_abs (toReal z) zR 0).trans h

/-- **Dense-ReLU neuron error.** The composed `reluF (b + dotF w x)` is within
    `z1ErrBnd` of its ideal real value `max (toReal b + dotR w x) 0`. -/
theorem neuron_error (w : List Float) (b : Float) (x : List Float) :
    |toReal (reluF (b + dotF w x)) - max (toReal b + dotR w x) 0| ≤ z1ErrBnd w b x :=
  reluF_error (b + dotF w x) (toReal b + dotR w x) (z1ErrBnd w b x) (linZ_error w b x)

/-- `Σᵢ |toReal wᵢ| · |toReal hᵢ − hRᵢ|` — the exact per-entry contribution of a moving
    input vector to a dot with weights `w`. (Its Float twin is `sumAbsMulF`, evaluated at
    an upper bound `εh` of each `|toReal hᵢ − hRᵢ|`.) -/
noncomputable def dotDiffBnd : List Float → List Float → List ℝ → ℝ
  | [], _, _ => 0
  | _, [], _ => 0
  | _, _, [] => 0
  | w :: ws, h :: hs, r :: rs => |toReal w| * |toReal h - r| + dotDiffBnd ws hs rs

theorem dotDiffBnd_nonneg (w h : List Float) (hR : List ℝ) : 0 ≤ dotDiffBnd w h hR := by
  induction w generalizing h hR with
  | nil => simp [dotDiffBnd]
  | cons a as ih =>
    cases h with
    | nil => simp [dotDiffBnd]
    | cons h0 hs =>
      cases hR with
      | nil => simp [dotDiffBnd]
      | cons r rs =>
        simp only [dotDiffBnd]
        exact add_nonneg (mul_nonneg (abs_nonneg _) (abs_nonneg _)) (ih hs rs)

/-- **Dot input-perturbation.** Replacing the ideal real hidden `hR` by the rounded Float
    hidden `h` moves the real dot by at most `dotDiffBnd`. -/
theorem dotR_perturb (w h : List Float) (hR : List ℝ) (hlen : h.length = hR.length) :
    |dotR w h - dotRm w hR| ≤ dotDiffBnd w h hR := by
  induction h generalizing w hR with
  | nil =>
    cases hR with
    | nil => cases w <;> simp [dotR, dotRm, dotDiffBnd]
    | cons => simp at hlen
  | cons h0 hs ih =>
    cases hR with
    | nil => simp at hlen
    | cons r rs =>
      cases w with
      | nil => simp [dotR, dotRm, dotDiffBnd]
      | cons w0 ws =>
        simp only [dotR, dotRm, dotDiffBnd]
        have hlen' : hs.length = rs.length := by simpa using hlen
        have e : toReal w0 * toReal h0 + dotR ws hs - (toReal w0 * r + dotRm ws rs)
            = toReal w0 * (toReal h0 - r) + (dotR ws hs - dotRm ws rs) := by ring
        rw [e]
        calc |toReal w0 * (toReal h0 - r) + (dotR ws hs - dotRm ws rs)|
            ≤ |toReal w0 * (toReal h0 - r)| + |dotR ws hs - dotRm ws rs| := abs_add_le _ _
          _ ≤ |toReal w0| * |toReal h0 - r| + dotDiffBnd ws hs rs := by
                rw [abs_mul]; exact add_le_add le_rfl (ih ws rs hlen')

/-- **Output-logit error (end-to-end).** A full logit `b2 + dotF W2 h` is within
    `z1ErrBnd W2 b2 h + dotDiffBnd W2 h hR` of the ideal real logit
    `toReal b2 + dotRm W2 hR`: the layer-2 linear-unit rounding plus the hidden error
    folded through the dot. -/
theorem logit_error (w2 : List Float) (b2 : Float) (h : List Float) (hR : List ℝ)
    (hlen : h.length = hR.length) :
    |toReal (b2 + dotF w2 h) - (toReal b2 + dotRm w2 hR)|
      ≤ z1ErrBnd w2 b2 h + dotDiffBnd w2 h hR := by
  have e : toReal (b2 + dotF w2 h) - (toReal b2 + dotRm w2 hR)
      = (toReal (b2 + dotF w2 h) - (toReal b2 + dotR w2 h)) + (dotR w2 h - dotRm w2 hR) := by ring
  rw [e]
  exact (abs_add_le _ _).trans (add_le_add (linZ_error w2 b2 h) (dotR_perturb w2 h hR hlen))

/-- **The executable ReLU is EXACTLY the ℝ ReLU under `toReal`.** `toReal (reluF x) = relu (toReal x)` — the
    activation introduces no rounding (`toReal_reluF` is `max (toReal x) 0 = relu (toReal x)`), so the runnable
    `reluF` inherits every ℝ `relu` property directly. -/
theorem reluF_toReal_eq_relu (x : Float) : toReal (reluF x) = Puffer.Net.relu (toReal x) := by
  unfold Puffer.Net.relu; exact toReal_reluF x

/-- **The executable ReLU is monotone** (at `toReal`): `toReal x ≤ toReal y → toReal (reluF x) ≤ toReal
    (reluF y)` — via the exact bridge and ℝ `relu_mono`. -/
theorem reluF_toReal_mono (x y : Float) (h : toReal x ≤ toReal y) :
    toReal (reluF x) ≤ toReal (reluF y) := by
  rw [reluF_toReal_eq_relu, reluF_toReal_eq_relu]; exact Puffer.Net.relu_mono _ _ h

end Puffer.FloatR
