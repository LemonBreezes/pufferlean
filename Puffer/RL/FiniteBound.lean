/-
# Finiteness / no-overflow: the one place we ADD to the trusted base

**This module adds ONE new trusted axiom** (`isFinite_of_bounded`), unlike C1–C40, which used only the existing
`toReal` + `(1+δ)` relative-error base in `Puffer/Float/Basic.lean`. The addition is unavoidable and is the honest
price of a no-overflow guarantee, for a concrete reason:

* The `(1+δ)` op-models (`add_model`/`mul_model`/…) bound RELATIVE accuracy — `fl(a op b) = (a op b)(1+δ)`,
  `|δ| ≤ u`. They say NOTHING about overflow. Worse, they are only VALID in the absence of overflow (as
  `Basic.lean` states: "These hold for IEEE binary64 round-to-nearest in the absence of overflow/underflow"): an
  overflowed result is `±inf`/`nan`, for which no `(1+δ)` relation holds. So the relative-error base SILENTLY
  ASSUMES no overflow, and cannot by itself certify it — overflow is invisible to a relative-error axiom.
* Bridging the ℝ-magnitude world (where all of C1–C40 reason) to the executable no-overflow flag
  `Float.isFinite : Float → Bool` therefore requires a new trusted fact. We take the weakest honest one:

    `axiom isFinite_of_bounded (a : Float) : |toReal a| ≤ overflowBound → a.isFinite = true`

  i.e. a Float whose embedded magnitude is within the binary64 overflow threshold is finite. This is a statement
  about the `toReal` embedding vs the IEEE representation (like `toReal`/`toBits_inj` themselves — `Float` is an
  opaque extern type, so it is not provable in Lean); it is consistent (interpret `toReal` of an infinity as
  `> overflowBound`) and as weak as possible — it only lets a MAGNITUDE bound discharge the `isFinite` obligation.

Everything else here is PROVED from that axiom plus the existing `(1+δ)` base:

* `mul_mag_le` / `add_mag_le` — the `(1+u)` magnitude-propagation bounds (restated from `Basic.lean`'s
  `mul_abs_le`/`add_abs_le`; NO new axiom).
* `mul_isFinite` / `add_isFinite` — a single Float `×`/`+` whose ideal-operand magnitudes keep the `(1+u)`-inflated
  result within `overflowBound` is finite (from `isFinite_of_bounded` + the magnitude bounds).
* `dotF` (a self-contained Float dot product), `dotBound`, `dotF_mag_le` — the result magnitude propagates through
  the fold to a concrete closed bound `dotBound n B` (PROVED, no axiom, from the `(1+δ)` base).
* `dotF_isFinite` — **the payoff**: a dot product over inputs magnitude-bounded by `B`, whose length keeps
  `dotBound n B ≤ overflowBound`, is overflow-free (`isFinite = true`). A runnable no-overflow certificate for a
  bounded linear kernel.

**Scope (honestly disclosed).** This adds `isFinite_of_bounded` to the trusted base. The magnitude propagation and
the finiteness certificate are PROVED from it + the `(1+δ)` axioms. The result is conditional on the `(1+δ)` model
(itself a no-overflow-assuming trusted base): under that model, a bounded-input, bounded-length linear kernel is
certified overflow-free. Underflow/subnormal effects are folded into the `(1+δ)` model as elsewhere; `overflowBound`
is the binary64 max magnitude `(2 − 2⁻⁵²)·2¹⁰²³`.
-/
import Puffer.Float.Basic
open Puffer.FloatR

namespace Puffer.RL.FiniteBound

/-- The largest finite magnitude of IEEE-754 binary64: `(2 − 2⁻⁵²)·2¹⁰²³ ≈ 1.7977·10³⁰⁸`. The overflow threshold:
    a real magnitude at or below this is representable (finite), above it rounds to `±inf`. -/
noncomputable def overflowBound : ℝ := (2 - (2 : ℝ) ^ (-52 : ℤ)) * (2 : ℝ) ^ (1023 : ℕ)

/-- **The one new trusted axiom (no-overflow bridge).** A Float whose embedded magnitude is within the binary64
    overflow threshold is finite. NOT derivable from the `(1+δ)` relative-error base — those axioms are silent about
    (and only valid under) no-overflow; `Float.isFinite` lives outside the ℝ-relative world. As weak as possible: it
    only converts a magnitude bound into the executable `isFinite` flag. Consistent with the opaque `toReal`
    embedding (an infinity embeds `> overflowBound`), in the spirit of `toReal`/`toBits_inj`. -/
axiom isFinite_of_bounded (a : Float) : |toReal a| ≤ overflowBound → a.isFinite = true

/-- `0 ≤ 1 + u64` (roundoff factor is positive). -/
private theorem one_add_u64_nonneg : (0 : ℝ) ≤ 1 + u64 := by have := u64_pos; linarith

/-- **Product magnitude propagation** (`(1+u)` growth, PROVED from `mul_abs_le`; no new axiom):
    `|fl(a·b)| ≤ (1+u)·|a|·|b|`. -/
theorem mul_mag_le (a b : Float) : |toReal (a * b)| ≤ (1 + u64) * (|toReal a| * |toReal b|) := by
  rw [← abs_mul]; exact mul_abs_le a b

/-- **Sum magnitude propagation** (`(1+u)` growth, PROVED from `add_abs_le`; no new axiom):
    `|fl(a+b)| ≤ (1+u)·(|a|+|b|)`. -/
theorem add_mag_le (a b : Float) : |toReal (a + b)| ≤ (1 + u64) * (|toReal a| + |toReal b|) :=
  (add_abs_le a b).trans (mul_le_mul_of_nonneg_left (abs_add_le _ _) one_add_u64_nonneg)

/-- **A single Float product is overflow-free** when the `(1+u)`-inflated operand-magnitude product is within the
    overflow threshold. Proved from `isFinite_of_bounded` + `mul_mag_le`. -/
theorem mul_isFinite (a b : Float) (h : (1 + u64) * (|toReal a| * |toReal b|) ≤ overflowBound) :
    (a * b).isFinite = true :=
  isFinite_of_bounded _ ((mul_mag_le a b).trans h)

/-- **A single Float sum is overflow-free** when the `(1+u)`-inflated operand-magnitude sum is within the overflow
    threshold. Proved from `isFinite_of_bounded` + `add_mag_le`. -/
theorem add_isFinite (a b : Float) (h : (1 + u64) * (|toReal a| + |toReal b|) ≤ overflowBound) :
    (a + b).isFinite = true :=
  isFinite_of_bounded _ ((add_mag_le a b).trans h)

/-- A self-contained Float dot product: `Σᵢ xᵢ·wᵢ` accumulated as `x·w + rest` (stops at the shorter list). -/
def dotF : List Float → List Float → Float
  | [], _ => 0
  | _, [] => 0
  | x :: xs, w :: ws => x * w + dotF xs ws

/-- The propagated magnitude bound for an `n`-term `dotF` over inputs of magnitude `≤ B`: each accumulation step
    inflates by `(1+u)` and adds a `(1+u)·B²` term — `dotBound (n+1) B = (1+u)·((1+u)·B² + dotBound n B)`. -/
noncomputable def dotBound : ℕ → ℝ → ℝ
  | 0, _ => 0
  | n + 1, B => (1 + u64) * ((1 + u64) * (B * B) + dotBound n B)

/-- **Dot-product magnitude propagation** (PROVED from the `(1+δ)` base; no new axiom). Over inputs magnitude-bounded
    by `B`, `|toReal (dotF xs ws)| ≤ dotBound (min xs.length ws.length) B` — the running `(1+u)`-inflated bound. -/
theorem dotF_mag_le (B : ℝ) (hB : 0 ≤ B) :
    ∀ (xs ws : List Float), (∀ x ∈ xs, |toReal x| ≤ B) → (∀ w ∈ ws, |toReal w| ≤ B) →
      |toReal (dotF xs ws)| ≤ dotBound (min xs.length ws.length) B
  | [], _, _, _ => by simp [dotF, dotBound]
  | _ :: _, [], _, _ => by simp [dotF, dotBound]
  | x :: xs, w :: ws, hxs, hws => by
      have hx : |toReal x| ≤ B := hxs x (List.mem_cons.mpr (Or.inl rfl))
      have hw : |toReal w| ≤ B := hws w (List.mem_cons.mpr (Or.inl rfl))
      have htail := dotF_mag_le B hB xs ws
        (fun q hq => hxs q (List.mem_cons.mpr (Or.inr hq)))
        (fun q hq => hws q (List.mem_cons.mpr (Or.inr hq)))
      have hmul : |toReal (x * w)| ≤ (1 + u64) * (B * B) :=
        (mul_mag_le x w).trans
          (mul_le_mul_of_nonneg_left (mul_le_mul hx hw (abs_nonneg _) hB) one_add_u64_nonneg)
      have hlen : min (x :: xs).length (w :: ws).length = min xs.length ws.length + 1 := by
        simp only [List.length_cons]; omega
      show |toReal (x * w + dotF xs ws)| ≤ dotBound (min (x :: xs).length (w :: ws).length) B
      rw [hlen]
      calc |toReal (x * w + dotF xs ws)|
          ≤ (1 + u64) * (|toReal (x * w)| + |toReal (dotF xs ws)|) := add_mag_le _ _
        _ ≤ (1 + u64) * ((1 + u64) * (B * B) + dotBound (min xs.length ws.length) B) :=
            mul_le_mul_of_nonneg_left (add_le_add hmul htail) one_add_u64_nonneg
        _ = dotBound (min xs.length ws.length + 1) B := by rw [dotBound]

/-- **THE NO-OVERFLOW CERTIFICATE.** A Float dot product over inputs magnitude-bounded by `B` whose length keeps the
    propagated bound within the overflow threshold (`dotBound (min xs.length ws.length) B ≤ overflowBound`) is
    overflow-free: `(dotF xs ws).isFinite = true`. The magnitude bound is PROVED (`dotF_mag_le`); the finiteness step
    is the one new axiom (`isFinite_of_bounded`). A runnable overflow-freedom guarantee for a bounded linear kernel. -/
theorem dotF_isFinite (B : ℝ) (hB : 0 ≤ B) (xs ws : List Float)
    (hxs : ∀ x ∈ xs, |toReal x| ≤ B) (hws : ∀ w ∈ ws, |toReal w| ≤ B)
    (hbound : dotBound (min xs.length ws.length) B ≤ overflowBound) :
    (dotF xs ws).isFinite = true :=
  isFinite_of_bounded _ ((dotF_mag_le B hB xs ws hxs hws).trans hbound)

end Puffer.RL.FiniteBound
