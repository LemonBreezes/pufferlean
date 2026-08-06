/-
Executable Float NN kernels, each carrying a PROVEN error bound vs its ℝ meaning.

`dotF`/`reluF`/`linearF` compute natively in `Float` (they run — see `Exe/Puffer.lean`).
`dotF_error` bounds the running result against the exact real dot product `dotR` by a
*computable* certified bound `dotErrBnd` — the accumulated per-operation rounding
error, obtained by composing the (1+δ) axioms in `Puffer/Float/Basic.lean`. This is
the "runnable + bounded-error" contract: hardware-float speed with a machine-checked
deviation from the ideal real value.
-/
import Mathlib
import Puffer.Float.Basic
import Puffer.Float.Exec

namespace Puffer.FloatR

/-- The exact real dot product of the inputs' real values (the ℝ meaning of `dotF`). -/
noncomputable def dotR : List Float → List Float → ℝ
  | [], _ => 0
  | _, [] => 0
  | x :: xs, w :: ws => toReal x * toReal w + dotR xs ws

theorem dotR_nil_left (ws : List Float) : dotR [] ws = 0 := rfl

theorem dotR_nil_right : ∀ (xs : List Float), dotR xs [] = 0
  | [] => rfl
  | _ :: _ => rfl

/-- **The real dot product is commutative** (`dotR xs ws = dotR ws xs`) — swapping inputs and weights leaves
    the linear combination unchanged (`mul_comm` at each element). -/
theorem dotR_comm (xs ws : List Float) : dotR xs ws = dotR ws xs := by
  induction xs generalizing ws with
  | nil => cases ws <;> rfl
  | cons x xs ih =>
    cases ws with
    | nil => rfl
    | cons w ws => simp only [dotR]; rw [mul_comm (toReal x) (toReal w), ih ws]

/-- The elementwise sum of product magnitudes `Σ |xᵢ|·|wᵢ|` — an upper envelope for the dot product. -/
noncomputable def dotAbsSum : List Float → List Float → ℝ
  | [], _ => 0
  | _, [] => 0
  | x :: xs, w :: ws => |toReal x| * |toReal w| + dotAbsSum xs ws

/-- **Dot-product magnitude bound.** `|dotR xs ws| ≤ Σ |xᵢ|·|wᵢ|` — the triangle inequality over the
    accumulated products bounds the linear-layer output by the sum of per-term magnitudes. -/
theorem dotR_abs_le (xs ws : List Float) : |dotR xs ws| ≤ dotAbsSum xs ws := by
  induction xs generalizing ws with
  | nil => simp [dotR, dotAbsSum]
  | cons x xs ih =>
    cases ws with
    | nil => simp [dotR, dotAbsSum]
    | cons w ws =>
      simp only [dotR, dotAbsSum]
      calc |toReal x * toReal w + dotR xs ws|
          ≤ |toReal x * toReal w| + |dotR xs ws| := abs_add_le _ _
        _ ≤ |toReal x| * |toReal w| + dotAbsSum xs ws := by
            rw [abs_mul]; exact add_le_add le_rfl (ih ws)

/-- A computable certified upper bound on the dot-product rounding error: at each
    element, one multiply rounding (`u·|xᵢwᵢ|`) plus one add rounding
    (`u·|prod + partial|`), accumulated down the list. -/
noncomputable def dotErrBnd : List Float → List Float → ℝ
  | [], _ => 0
  | _, [] => 0
  | x :: xs, w :: ws =>
      u64 * |toReal (x * w) + toReal (dotF xs ws)|
      + u64 * |toReal x * toReal w|
      + dotErrBnd xs ws

theorem dotErrBnd_nonneg (x w : List Float) : 0 ≤ dotErrBnd x w := by
  induction x generalizing w with
  | nil => simp [dotErrBnd]
  | cons a as ih =>
      cases w with
      | nil => simp [dotErrBnd]
      | cons b bs =>
          simp only [dotErrBnd]
          exact add_nonneg (add_nonneg (mul_nonneg u64_pos.le (abs_nonneg _))
            (mul_nonneg u64_pos.le (abs_nonneg _))) (ih bs)

/-- **Dot-product error bound (runnable ↔ real).** The hardware `dotF` deviates from
    the exact real dot product `dotR` by at most the certified bound `dotErrBnd`. -/
theorem dotF_error (x w : List Float) :
    |toReal (dotF x w) - dotR x w| ≤ dotErrBnd x w := by
  induction x generalizing w with
  | nil => simp [dotF, dotR, dotErrBnd]
  | cons a as ih =>
      cases w with
      | nil => simp [dotF, dotR, dotErrBnd]
      | cons b bs =>
          simp only [dotF, dotR, dotErrBnd]
          have split : toReal (a * b + dotF as bs) - (toReal a * toReal b + dotR as bs)
              = (toReal (a * b + dotF as bs) - (toReal (a * b) + toReal (dotF as bs)))
                + (toReal (a * b) - toReal a * toReal b)
                + (toReal (dotF as bs) - dotR as bs) := by ring
          rw [split]
          calc |(toReal (a * b + dotF as bs) - (toReal (a * b) + toReal (dotF as bs)))
                  + (toReal (a * b) - toReal a * toReal b)
                  + (toReal (dotF as bs) - dotR as bs)|
              ≤ (|toReal (a * b + dotF as bs) - (toReal (a * b) + toReal (dotF as bs))|
                  + |toReal (a * b) - toReal a * toReal b|)
                  + |toReal (dotF as bs) - dotR as bs| :=
                (abs_add_le _ _).trans (add_le_add (abs_add_le _ _) le_rfl)
            _ ≤ (u64 * |toReal (a * b) + toReal (dotF as bs)| + u64 * |toReal a * toReal b|)
                  + dotErrBnd as bs :=
                add_le_add (add_le_add (add_error (a * b) (dotF as bs)) (mul_error a b)) (ih bs)

/-- **Dense-ReLU layer outputs are nonnegative.** Every neuron output of a ReLU dense layer `denseRelu x w b`
    has nonnegative real value: `∀ y ∈ denseRelu x w b, 0 ≤ toReal y`. Each output is `reluF (linearF x wᵢ bᵢ)`,
    whose real value is `max (·) 0 ≥ 0` (`toReal_reluF`). This is the defining structural invariant of a ReLU
    layer — post-activation values are `≥ 0`, so any downstream op that needs nonnegative inputs is safe. -/
theorem denseRelu_toReal_nonneg (x : List Float) (w : List (List Float)) (b : List Float) :
    ∀ y ∈ denseRelu x w b, 0 ≤ toReal y := by
  intro y hy
  obtain ⟨wb, _, hwb⟩ := List.mem_map.mp hy
  rw [← hwb, toReal_reluF]
  exact le_max_right _ _

end Puffer.FloatR
