/-
The GAE fold's INPUT-SENSITIVITY — how the advantage `Aₜ = δₜ + wₜ·Aₜ₊₁` responds when its per-step TD-error
`δ` and weight `w` inputs are perturbed. The ℝ-level companion to the closed-form/accuracy story
(`GAEInvariant`, a85), in the mould of the softmax/log-policy/entropy input-perturbation bounds (a72–a81).

a85 bounds the runnable Float fold against `gadvListR` over the `toReal` of the ACTUAL computed deltas/weights
(the δ/w internal rounding held out of scope). This file supplies the reusable ℝ core needed to CLOSE that
gap: a Lipschitz-type sensitivity of `gadvListR` in its `(δ,w)` inputs. Composed with a per-position
`|toReal δ_float − δ_true| , |toReal w_float − w_true|` budget (a disclosed follow-up), it upgrades a85's
reference to the true-arithmetic GAE.

  • `gadvPerturbBnd` — the per-step budget: the δ-shift `|δa−δb|`, the `|wa|`-scaled tail sensitivity, and the
    weight-shift `|wa−wb|` times the perturbed-tail magnitude `|gadvListR rb|` (from the exact decomposition
    `wa·Ra − wb·Rb = wa·(Ra−Rb) + (wa−wb)·Rb`). `gadvPerturbBnd_nonneg`.
  • `gadvListR_perturb` — for two equal-length `(δ,w)` suffixes, `|gadvListR aR − gadvListR bR| ≤
    gadvPerturbBnd aR bR`. Induction on the suffix with the per-step split + triangle.

Pure ℝ (Mathlib) — axiom-clean (`propext`/`Classical.choice`/`Quot.sound` only, no Float base).
-/
import Puffer.RL.GAEInvariant

namespace Puffer.RL.GAEInputPerturb

open Puffer.RL.GAEInvariant (gadvListR)

/-- Input-sensitivity budget for the GAE fold `Aₜ = δₜ + wₜ·Aₜ₊₁`: at each step the δ-shift `|δa−δb|`, the
    `|wa|`-scaled tail sensitivity, and the weight-shift `|wa−wb|` times the perturbed tail magnitude
    `|gadvListR rb|`. -/
noncomputable def gadvPerturbBnd : List (ℝ × ℝ) → List (ℝ × ℝ) → ℝ
  | [], _ => 0
  | _ :: _, [] => 0
  | (δa, wa) :: ra, (δb, wb) :: rb =>
      |δa - δb| + |wa| * gadvPerturbBnd ra rb + |wa - wb| * |gadvListR rb|

theorem gadvPerturbBnd_nonneg : ∀ (aR bR : List (ℝ × ℝ)), 0 ≤ gadvPerturbBnd aR bR
  | [], _ => le_refl 0
  | _ :: _, [] => le_refl 0
  | (δa, wa) :: ra, (δb, wb) :: rb => by
      simp only [gadvPerturbBnd]
      have := gadvPerturbBnd_nonneg ra rb
      positivity

/-- **GAE fold input-sensitivity.** For two equal-length `(δ,w)` suffixes, the GAE advantages differ by at
    most `gadvPerturbBnd` — via the per-step decomposition `wa·Ra − wb·Rb = wa·(Ra−Rb) + (wa−wb)·Rb`, the
    triangle inequality, and the tail induction hypothesis. -/
theorem gadvListR_perturb : ∀ (aR bR : List (ℝ × ℝ)), aR.length = bR.length →
    |gadvListR aR - gadvListR bR| ≤ gadvPerturbBnd aR bR := by
  intro aR
  induction aR with
  | nil =>
    intro bR hlen
    have : bR = [] := List.length_eq_zero_iff.mp hlen.symm
    subst this
    simp [gadvListR, gadvPerturbBnd]
  | cons pa ra ih =>
    intro bR hlen
    obtain ⟨δa, wa⟩ := pa
    cases bR with
    | nil => simp at hlen
    | cons pb rb =>
      obtain ⟨δb, wb⟩ := pb
      have hlen' : ra.length = rb.length := by simpa using hlen
      have ihr := ih rb hlen'
      simp only [gadvListR, gadvPerturbBnd]
      have hsplit : δa + wa * gadvListR ra - (δb + wb * gadvListR rb)
          = (δa - δb) + (wa * (gadvListR ra - gadvListR rb) + (wa - wb) * gadvListR rb) := by ring
      calc |δa + wa * gadvListR ra - (δb + wb * gadvListR rb)|
          = |(δa - δb) + (wa * (gadvListR ra - gadvListR rb) + (wa - wb) * gadvListR rb)| := by rw [hsplit]
        _ ≤ |δa - δb| + |wa * (gadvListR ra - gadvListR rb) + (wa - wb) * gadvListR rb| := abs_add_le _ _
        _ ≤ |δa - δb| + (|wa * (gadvListR ra - gadvListR rb)| + |(wa - wb) * gadvListR rb|) :=
            add_le_add le_rfl (abs_add_le _ _)
        _ ≤ |δa - δb| + (|wa| * gadvPerturbBnd ra rb + |wa - wb| * |gadvListR rb|) := by
            rw [abs_mul, abs_mul]
            exact add_le_add le_rfl (add_le_add
              (mul_le_mul_of_nonneg_left ihr (abs_nonneg _)) le_rfl)
        _ = |δa - δb| + |wa| * gadvPerturbBnd ra rb + |wa - wb| * |gadvListR rb| := by ring

end Puffer.RL.GAEInputPerturb
