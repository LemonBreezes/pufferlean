/-
Executable core of the verified forward-mode AD (`Puffer/Float/AutoDiffR.lean`) — Mathlib-FREE
so the `puffer` binary can RUN it (the "error-bound mode"). The `Expr` IR, its `Float` value
(`evalF`) and forward-derivative (`dF`), and the COMPUTABLE `Float` evaluators of the proven
error bounds (`evalErrBndF`/`derivErrBndF`) live here; the ℝ specs, correctness, and the bound
theorems that certify these live in `AutoDiffR.lean` (which imports this module).

`evalErrBndF`/`derivErrBndF` mirror the noncomputable ℝ bounds `evalErrBnd`/`derivErrBnd`
op-for-op with `toReal x ↦ x` (and `Real.exp/log ↦ Float.exp/log`), exactly as `dotErrBndF`
mirrors `dotErrBnd`. `puffer grad` emits `dF` + `derivErrBndF`; `tools/grad_ref.py` checks the
actual error against them versus an independent high-precision ℝ reference.
-/
import Puffer.Float.ErrBnd

namespace Puffer.FloatR.ADR

/-- The AD tape fragment, over indexed variables: the smooth ops plus the `relu` kink (whose
    derivative is certified only away from `0`; see `PosR`/`WD` in `AutoDiffR.lean`). -/
inductive Expr where
  | var (i : Nat)
  | const (c : Float)
  | add (a b : Expr)
  | sub (a b : Expr)
  | mul (a b : Expr)
  | scale (c : Float) (a : Expr)
  | exp (a : Expr)
  | log (a : Expr)
  | relu (a : Expr)
  | max (a b : Expr)
  | min (a b : Expr)
  deriving Inhabited

/-- `clamp a lo hi = min (max a lo) hi` — the two-sided clip (e.g. PPO's ratio clip),
    built as sugar from `min`/`max`/`const` so it needs no separate proofs. -/
def clampE (a : Expr) (lo hi : Float) : Expr := .min (.max a (.const lo)) (.const hi)

/-- Executable `Float` evaluation. -/
def evalF : Expr → (Nat → Float) → Float
  | .var i, σ => σ i
  | .const c, _ => c
  | .add a b, σ => evalF a σ + evalF b σ
  | .sub a b, σ => evalF a σ - evalF b σ
  | .mul a b, σ => evalF a σ * evalF b σ
  | .scale c a, σ => c * evalF a σ
  | .exp a, σ => Float.exp (evalF a σ)
  | .log a, σ => Float.log (evalF a σ)
  | .relu a, σ => reluF (evalF a σ)
  | .max a b, σ => max (evalF a σ) (evalF b σ)
  | .min a b, σ => min (evalF a σ) (evalF b σ)

/-- Executable `Float` forward-mode derivative w.r.t. variable `k`. -/
def dF : Expr → (Nat → Float) → Nat → Float
  | .var i, _, k => if i = k then 1 else 0
  | .const _, _, _ => 0
  | .add a b, σ, k => dF a σ k + dF b σ k
  | .sub a b, σ, k => dF a σ k - dF b σ k
  | .mul a b, σ, k => dF a σ k * evalF b σ + evalF a σ * dF b σ k
  | .scale c a, σ, k => c * dF a σ k
  | .exp a, σ, k => Float.exp (evalF a σ) * dF a σ k
  | .log a, σ, k => dF a σ k / evalF a σ
  -- Subgradient of `relu`: mirrors `reluF`'s branch (`x < 0.0 ↦ 0`, else pass through `dF a`).
  -- The `evalErrBndF (.relu a)` bound is certified only away from the kink (`WD`), where the
  -- `Float` sign and the real sign agree, so this exact `0`/`dF a` choice is the true subgradient.
  | .relu a, σ, k => if evalF a σ < 0.0 then 0 else dF a σ k
  -- Subgradient of `max`/`min`: mirror the `if a ≤ b` branch of Float `max`/`min` and pass through
  -- the selected argument's derivative. `max a b = if a ≤ b then b else a` (so `≤` ⇒ `b` selected);
  -- `min a b = if a ≤ b then a else b` (so `≤` ⇒ `a` selected). Certified only away from the kink
  -- (`a ≠ b`; see `PosR`/`WD`), where the Float branch and the real branch agree.
  | .max a b, σ, k => if evalF a σ ≤ evalF b σ then dF b σ k else dF a σ k
  | .min a b, σ, k => if evalF a σ ≤ evalF b σ then dF a σ k else dF b σ k

/-- `expEps`/`logEps` (`= 2⁻⁵²`) as `Float`. -/
def expEpsF : Float := 1.0 / 4503599627370496.0
def logEpsF : Float := 1.0 / 4503599627370496.0

/-- Computable `Float` mirror of the ℝ value-error bound `evalErrBnd` (`toReal x ↦ x`). -/
def evalErrBndF : Expr → (Nat → Float) → Float
  | .var _, _ => 0
  | .const _, _ => 0
  | .add a b, σ => u64F * Float.abs (evalF a σ + evalF b σ) + evalErrBndF a σ + evalErrBndF b σ
  | .sub a b, σ => u64F * Float.abs (evalF a σ - evalF b σ) + evalErrBndF a σ + evalErrBndF b σ
  | .mul a b, σ => u64F * Float.abs (evalF a σ * evalF b σ)
      + Float.abs (evalF a σ) * evalErrBndF b σ + Float.abs (evalF b σ) * evalErrBndF a σ
  | .scale c a, σ => u64F * Float.abs (c * evalF a σ) + Float.abs c * evalErrBndF a σ
  | .exp a, σ => expEpsF * Float.exp (evalF a σ)
      + Float.exp (evalF a σ) * (Float.exp (evalErrBndF a σ) - 1)
  | .log a, σ => logEpsF * Float.abs (Float.log (evalF a σ))
      + evalErrBndF a σ / (evalF a σ - evalErrBndF a σ)
  -- `reluF` is exact and 1-Lipschitz: it adds no rounding, so the value error is inherited
  -- verbatim from the argument (`|max x 0 − max y 0| ≤ |x − y|`).
  | .relu a, σ => evalErrBndF a σ
  -- `max`/`min` are exact (no rounding); the value error is bounded by the sum of the arguments'
  -- errors (`|max x y − max x' y'| ≤ max |x−x'| |y−y'| ≤ εx + εy`).
  | .max a b, σ => evalErrBndF a σ + evalErrBndF b σ
  | .min a b, σ => evalErrBndF a σ + evalErrBndF b σ

/-- Computable `Float` mirror of the ℝ derivative-error bound `derivErrBnd` (`toReal x ↦ x`). -/
def derivErrBndF : Expr → (Nat → Float) → Nat → Float
  | .var _, _, _ => 0
  | .const _, _, _ => 0
  | .add a b, σ, k => u64F * Float.abs (dF a σ k + dF b σ k) + derivErrBndF a σ k + derivErrBndF b σ k
  | .sub a b, σ, k => u64F * Float.abs (dF a σ k - dF b σ k) + derivErrBndF a σ k + derivErrBndF b σ k
  | .mul a b, σ, k =>
      u64F * Float.abs (dF a σ k * evalF b σ + evalF a σ * dF b σ k)
      + (u64F * Float.abs (dF a σ k * evalF b σ)
          + Float.abs (dF a σ k) * evalErrBndF b σ + Float.abs (evalF b σ) * derivErrBndF a σ k)
      + (u64F * Float.abs (evalF a σ * dF b σ k)
          + Float.abs (evalF a σ) * derivErrBndF b σ k + Float.abs (dF b σ k) * evalErrBndF a σ)
  | .scale c a, σ, k => u64F * Float.abs (c * dF a σ k) + Float.abs c * derivErrBndF a σ k
  | .exp a, σ, k =>
      u64F * Float.abs (Float.exp (evalF a σ) * dF a σ k)
      + Float.abs (Float.exp (evalF a σ)) * derivErrBndF a σ k
      + Float.abs (dF a σ k) * evalErrBndF (.exp a) σ
  | .log a, σ, k =>
      u64F * Float.abs (dF a σ k / evalF a σ)
      + (derivErrBndF a σ k + Float.abs (dF a σ k / evalF a σ) * evalErrBndF a σ)
          / (evalF a σ - evalErrBndF a σ)
  -- Away from the kink the subgradient is exactly `0` (below) or `dF a` (above), each exact
  -- in `Float`, so the derivative error is inherited verbatim from the argument.
  | .relu a, σ, k => derivErrBndF a σ k
  -- Away from the kink the subgradient is exactly the selected argument's (mirroring `dF`'s branch),
  -- so the derivative error is inherited verbatim from that argument.
  | .max a b, σ, k => if evalF a σ ≤ evalF b σ then derivErrBndF b σ k else derivErrBndF a σ k
  | .min a b, σ, k => if evalF a σ ≤ evalF b σ then derivErrBndF a σ k else derivErrBndF b σ k

end Puffer.FloatR.ADR
