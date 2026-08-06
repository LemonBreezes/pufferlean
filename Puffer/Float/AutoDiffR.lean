/-
Verified forward-mode automatic differentiation over a differentiable expression IR —
a machine-checked attack on the gradient-error gap `εg` (PLAN.md M8).

The trainer's reverse-mode tape (`Puffer/Float/AutoDiff.lean`) is an imperative, op-erased
array loop, so it cannot be inducted on directly. Here we take the SMOOTH fragment
(`var/const/add/sub/mul/scale/exp/log`) PLUS the `relu` kink as an explicit inductive `Expr`
and give it a functional forward-mode derivative. That buys two machine-checked facts
by clean structural induction:

  * `derivR_hasDerivAt` / `derivR_eq_deriv` — the symbolic derivative `derivR` really IS the
    mathematical derivative `∂(evalR)/∂(var k)` (AD *correctness* over ℝ), so bounding against
    it is not vacuous.
  * `evalF_error` / `dF_error` — the `Float` value and the `Float` forward-derivative track
    their exact real values within computable bounds (AD *error* — the `εg` content), built by
    pushing the `*Approx_error` lemmas through the circuit.

Together: `|toReal (dF e σ k) − ∂(evalR e)/∂(var k)| ≤ derivErrBnd e σ k` — a proven per-
component gradient error bound. For a scalar objective the forward-mode derivative w.r.t. each
input equals the same MATHEMATICAL gradient the reverse-mode trainer targets, so this is a
proven-correct gradient for expressions in this IR (wiring the trainer's tape to emit such
`Expr` values is the remaining step). NOTE the scope: this proves
agreement of *real derivatives of the same real function*; it does NOT prove the reverse-mode
`Tape`/`grads` (a different, op-erased representation) computes `dF`, nor does it bound that
reverse tape's own rounding — both are future work.

`log` (and its derivative `a'/a`, a DIVISION) requires positivity: the theorems for `log`-
containing expressions are conditioned on `PosR` (real argument > 0, for correctness) and `WD`
(the `Float` log-argument exceeds its own error bound, hence stays positive with a denominator
floor). For the `log`-free fragment these predicates are trivially `True`. CAVEAT: the `log`
VALUE bound inherits `log_model`'s near-1 optimism (a pure-relative model whose rounding term
`logEps·|log a|` vanishes as the argument → 1, where correctly-rounded `log` actually has a small
absolute floor); tightening to a mixed `logEps·|log a| + logAbsEps` model is deferred with the
rest of the `log_model` hardening.

`relu` (the kink most used in PPO/NN) is handled AWAY FROM ITS KINK: `PosR` requires the real
argument `≠ 0` (so `relu` is locally `id` or `const 0`, giving a genuine `HasDerivAt`), and `WD`
requires the `Float` error bound below the `Float` magnitude (`evalErrBnd a < |evalF a|`), which —
via `evalF_error` and `toReal_reluF` — forces the `Float` sign branch and the real sign to agree.
`reluF` is exact (`toReal_reluF`), so the value error is inherited verbatim (`relu` is 1-Lipschitz)
and the subgradient (`0` or `1`) is exact, so the derivative error is inherited too. The binary kink
ops `max`/`min` (and `clampE = min (max · lo) hi` as sugar) get the SAME treatment: `PosR` requires
the two real arguments unequal, `WD` requires their `Float` gap to exceed the sum of their error
bounds — which, via `le_of_float_le`/`le_of_not_float_le` (Float `≤` respects the real order, DERIVED
from `toReal_min`/`toReal_max`, no new axiom), forces the `Float` `if a ≤ b` branch and the real branch
to select the same argument. Remaining for full `εg`: equating this forward-mode AD to the reverse-mode
tape.
-/
import Puffer.Float.Basic
import Puffer.Float.Expr

namespace Puffer.FloatR.ADR

open Puffer.FloatR

/-- The real environment induced by a `Float` environment. -/
noncomputable def envR (σ : Nat → Float) : Nat → ℝ := fun i => toReal (σ i)

/-- Exact real evaluation. -/
noncomputable def evalR : Expr → (Nat → ℝ) → ℝ
  | .var i, σ => σ i
  | .const c, _ => toReal c
  | .add a b, σ => evalR a σ + evalR b σ
  | .sub a b, σ => evalR a σ - evalR b σ
  | .mul a b, σ => evalR a σ * evalR b σ
  | .scale c a, σ => toReal c * evalR a σ
  | .exp a, σ => Real.exp (evalR a σ)
  | .log a, σ => Real.log (evalR a σ)
  | .relu a, σ => max (evalR a σ) 0
  | .max a b, σ => max (evalR a σ) (evalR b σ)
  | .min a b, σ => min (evalR a σ) (evalR b σ)

/-- Symbolic derivative w.r.t. variable `k` (standard rules; matches `HasDerivAt` order). -/
noncomputable def derivR : Expr → (Nat → ℝ) → Nat → ℝ
  | .var i, _, k => if i = k then 1 else 0
  | .const _, _, _ => 0
  | .add a b, σ, k => derivR a σ k + derivR b σ k
  | .sub a b, σ, k => derivR a σ k - derivR b σ k
  | .mul a b, σ, k => derivR a σ k * evalR b σ + evalR a σ * derivR b σ k
  | .scale c a, σ, k => toReal c * derivR a σ k
  | .exp a, σ, k => Real.exp (evalR a σ) * derivR a σ k
  | .log a, σ, k => derivR a σ k / evalR a σ
  -- Away from the kink (`evalR a σ ≠ 0`, imposed by `PosR`) `relu` is locally `id` (if `> 0`)
  -- or `const 0` (if `< 0`), so the derivative is `derivR a` or `0` respectively.
  | .relu a, σ, k => if 0 < evalR a σ then derivR a σ k else 0
  -- Away from the kink (`evalR a σ ≠ evalR b σ`, imposed by `PosR`), the selected argument's
  -- derivative. `max a b = b` when `a ≤ b`; `min a b = a` when `a ≤ b`.
  | .max a b, σ, k => if evalR a σ ≤ evalR b σ then derivR b σ k else derivR a σ k
  | .min a b, σ, k => if evalR a σ ≤ evalR b σ then derivR a σ k else derivR b σ k

/-- Real well-definedness: every `log`-argument evaluates strictly positive (so `derivR` is
    the true derivative there). Trivially `True` for `log`-free expressions. -/
def PosR : Expr → (Nat → ℝ) → Prop
  | .var _, _ => True
  | .const _, _ => True
  | .add a b, σ => PosR a σ ∧ PosR b σ
  | .sub a b, σ => PosR a σ ∧ PosR b σ
  | .mul a b, σ => PosR a σ ∧ PosR b σ
  | .scale _ a, σ => PosR a σ
  | .exp a, σ => PosR a σ
  | .log a, σ => PosR a σ ∧ 0 < evalR a σ
  -- `relu`'s kink: require the real argument off `0` so `relu` is locally differentiable there.
  | .relu a, σ => PosR a σ ∧ evalR a σ ≠ 0
  -- `max`/`min`'s kink: require the two real arguments unequal so the winner is locally constant.
  | .max a b, σ => PosR a σ ∧ PosR b σ ∧ evalR a σ ≠ evalR b σ
  | .min a b, σ => PosR a σ ∧ PosR b σ ∧ evalR a σ ≠ evalR b σ

/-- Syntactic occurrence of variable `k` in an expression. -/
def occurs (k : Nat) : Expr → Prop
  | .var i => i = k
  | .const _ => False
  | .add a b => occurs k a ∨ occurs k b
  | .sub a b => occurs k a ∨ occurs k b
  | .mul a b => occurs k a ∨ occurs k b
  | .scale _ a => occurs k a
  | .exp a => occurs k a
  | .log a => occurs k a
  | .relu a => occurs k a
  | .max a b => occurs k a ∨ occurs k b
  | .min a b => occurs k a ∨ occurs k b

/-- **Gradient-zero condition.** If the variable `k` does not occur syntactically in `e`, then the real
    derivative of `e` w.r.t. `k` is identically zero — the exact-ℝ symbolic gradient has no spurious dependence
    on variables absent from the expression. Needs no positivity (`PosR`) hypothesis: it is a purely structural
    fact about `derivR`, holding across every kink branch (`relu`/`max`/`min` collapse via `ite_self` once the
    inner derivative is `0`, `log` via `zero_div`). -/
theorem derivR_eq_zero_of_not_occurs (e : Expr) (σ : Nat → ℝ) (k : Nat) :
    ¬ occurs k e → derivR e σ k = 0 := by
  induction e with
  | var i => intro h; simp only [occurs] at h; simp only [derivR, if_neg h]
  | const c => intro _; simp only [derivR]
  | add a b iha ihb =>
      intro h; simp only [occurs, not_or] at h
      simp only [derivR, iha h.1, ihb h.2, add_zero]
  | sub a b iha ihb =>
      intro h; simp only [occurs, not_or] at h
      simp only [derivR, iha h.1, ihb h.2, sub_zero]
  | mul a b iha ihb =>
      intro h; simp only [occurs, not_or] at h
      simp only [derivR, iha h.1, ihb h.2, zero_mul, mul_zero, add_zero]
  | scale c a iha =>
      intro h; simp only [occurs] at h
      simp only [derivR, iha h, mul_zero]
  | exp a iha =>
      intro h; simp only [occurs] at h
      simp only [derivR, iha h, mul_zero]
  | log a iha =>
      intro h; simp only [occurs] at h
      simp only [derivR, iha h, zero_div]
  | relu a iha =>
      intro h; simp only [occurs] at h
      simp only [derivR, iha h, ite_self]
  | max a b iha ihb =>
      intro h; simp only [occurs, not_or] at h
      simp only [derivR, iha h.1, ihb h.2, ite_self]
  | min a b iha ihb =>
      intro h; simp only [occurs, not_or] at h
      simp only [derivR, iha h.1, ihb h.2, ite_self]

/-- Evaluation locality (value-level companion): `evalR` depends on the environment only through the variables
    that occur in the expression. -/
theorem evalR_congr (e : Expr) (σ σ' : Nat → ℝ)
    (h : ∀ i, occurs i e → σ i = σ' i) : evalR e σ = evalR e σ' := by
  induction e with
  | var i => simp only [evalR]; exact h i (by simp [occurs])
  | const c => rfl
  | add a b iha ihb =>
      simp only [occurs] at h
      simp only [evalR, iha (fun i hi => h i (Or.inl hi)), ihb (fun i hi => h i (Or.inr hi))]
  | sub a b iha ihb =>
      simp only [occurs] at h
      simp only [evalR, iha (fun i hi => h i (Or.inl hi)), ihb (fun i hi => h i (Or.inr hi))]
  | mul a b iha ihb =>
      simp only [occurs] at h
      simp only [evalR, iha (fun i hi => h i (Or.inl hi)), ihb (fun i hi => h i (Or.inr hi))]
  | scale c a iha => simp only [occurs] at h; simp only [evalR, iha h]
  | exp a iha => simp only [occurs] at h; simp only [evalR, iha h]
  | log a iha => simp only [occurs] at h; simp only [evalR, iha h]
  | relu a iha => simp only [occurs] at h; simp only [evalR, iha h]
  | max a b iha ihb =>
      simp only [occurs] at h
      simp only [evalR, iha (fun i hi => h i (Or.inl hi)), ihb (fun i hi => h i (Or.inr hi))]
  | min a b iha ihb =>
      simp only [occurs] at h
      simp only [evalR, iha (fun i hi => h i (Or.inl hi)), ihb (fun i hi => h i (Or.inr hi))]

/-- **Gradient locality.** The exact-ℝ symbolic derivative `derivR e σ k` depends on the environment `σ` only
    through the variables that syntactically occur in `e`: if two environments agree on every occurring variable,
    they induce the same gradient component. This is the "moving-environment" strengthening of
    `derivR_eq_zero_of_not_occurs` (which is the special case where the differentiation variable `k` is absent,
    forcing the value `0`); here the *whole* environment may be perturbed on absent variables with no effect on any
    component. Needs no positivity (`PosR`): a purely structural fact, using the value-level companion `evalR_congr`
    at the `mul`/`exp`/`log`/`relu`/`max`/`min` nodes where `derivR` reads `evalR`. -/
theorem derivR_congr (e : Expr) (σ σ' : Nat → ℝ) (k : Nat)
    (h : ∀ i, occurs i e → σ i = σ' i) : derivR e σ k = derivR e σ' k := by
  induction e with
  | var i => rfl
  | const c => rfl
  | add a b iha ihb =>
      simp only [occurs] at h
      simp only [derivR, iha (fun i hi => h i (Or.inl hi)), ihb (fun i hi => h i (Or.inr hi))]
  | sub a b iha ihb =>
      simp only [occurs] at h
      simp only [derivR, iha (fun i hi => h i (Or.inl hi)), ihb (fun i hi => h i (Or.inr hi))]
  | mul a b iha ihb =>
      simp only [occurs] at h
      simp only [derivR, iha (fun i hi => h i (Or.inl hi)), ihb (fun i hi => h i (Or.inr hi)),
        evalR_congr a σ σ' (fun i hi => h i (Or.inl hi)),
        evalR_congr b σ σ' (fun i hi => h i (Or.inr hi))]
  | scale c a iha => simp only [occurs] at h; simp only [derivR, iha h]
  | exp a iha =>
      simp only [occurs] at h
      simp only [derivR, iha h, evalR_congr a σ σ' h]
  | log a iha =>
      simp only [occurs] at h
      simp only [derivR, iha h, evalR_congr a σ σ' h]
  | relu a iha =>
      simp only [occurs] at h
      simp only [derivR, iha h, evalR_congr a σ σ' h]
  | max a b iha ihb =>
      simp only [occurs] at h
      simp only [derivR, iha (fun i hi => h i (Or.inl hi)), ihb (fun i hi => h i (Or.inr hi)),
        evalR_congr a σ σ' (fun i hi => h i (Or.inl hi)),
        evalR_congr b σ σ' (fun i hi => h i (Or.inr hi))]
  | min a b iha ihb =>
      simp only [occurs] at h
      simp only [derivR, iha (fun i hi => h i (Or.inl hi)), ihb (fun i hi => h i (Or.inr hi)),
        evalR_congr a σ σ' (fun i hi => h i (Or.inl hi)),
        evalR_congr b σ σ' (fun i hi => h i (Or.inr hi))]

/-! ### Gradient-Lipschitz fragments (toward the ideal-step Lipschitz constant `L`)

The N-step accumulation `MuonTrainBound.nstep_trajectory_error` needs an `L`-Lipschitz ideal step map, which
(via `MuonTrainBound.ascent_map_lipschitz`) reduces to a Lipschitz bound `G` on the gradient map
`σ ↦ derivR e σ` — a smoothness/Hessian-type bound. In general `derivR e` is only LOCALLY Lipschitz (the `exp`
node grows without bound; the `log` node's derivative `1/evalR a` blows up near 0), so a full bound needs the
region-bounded simultaneous `evalR`/`derivR` induction. Two fully-closed fragments toward it: -/

/-- The AFFINE fragment of the expression grammar: variables, constants, and closure under `add`/`sub`/`scale`
    (multiply by a CONSTANT). No `mul`/`exp`/`log`/`relu`/`max`/`min` — i.e. genuinely affine functions of the
    variables. -/
inductive Affine : Expr → Prop
  | var (i : Nat) : Affine (.var i)
  | const (c : Float) : Affine (.const c)
  | add {a b : Expr} : Affine a → Affine b → Affine (.add a b)
  | sub {a b : Expr} : Affine a → Affine b → Affine (.sub a b)
  | scale (c : Float) {a : Expr} : Affine a → Affine (.scale c a)

/-- **The gradient of an affine loss is CONSTANT in the parameters** (`G = 0`). On the affine fragment the
    symbolic derivative `derivR e σ k` does not depend on the environment `σ` at all, so the gradient map is
    `0`-Lipschitz. Via `MuonTrainBound.ascent_map_lipschitz` this makes the ideal gradient-ascent step EXACTLY
    `1`-Lipschitz (`L = 1 + |lr|·0 = 1`) for an affine loss, discharging the `hlip` hypothesis of the N-step
    accumulation with `L = 1` (which yields the linear-drift bound `d₀ + n·B`). The nonlinear parts of a real
    network (`mul` for learned weights, `exp` for softmax) are only LOCALLY Lipschitz and need the region-bounded
    simultaneous `evalR`/`derivR` Lipschitz induction; this affine case is the globally-constant sub-fragment.
    Proved by induction on `Affine e` (each affine constructor's `derivR` rule is manifestly `σ`-free). -/
theorem derivR_const_of_affine {e : Expr} (ha : Affine e) (σ σ' : Nat → ℝ) (k : Nat) :
    derivR e σ k = derivR e σ' k := by
  induction ha with
  | var i => rfl
  | const c => rfl
  | add _ _ iha ihb => simp only [derivR, iha, ihb]
  | sub _ _ iha ihb => simp only [derivR, iha, ihb]
  | scale c _ iha => simp only [derivR, iha]

/-- **Gradient-Lipschitz of the bilinear atom `x_i · x_j`** (`G = 2`). The single product of two variables — the
    atom of a linear layer's weight·input term — has gradient
    `∂/∂x_k (x_i·x_j) = [i=k]·x_j + [j=k]·x_i`, which (unlike the affine fragment) DOES depend on the parameters.
    It is `2`-Lipschitz in the sup metric: if `|σ_m − σ'_m| ≤ δ` for every `m`, then the gradient changes by
    `≤ 2·δ` (the two Kronecker-selector terms each contribute `≤ δ`; the `i=j=k` diagonal case `∂/∂x(x²)=2x`
    saturates the `2`). This is the first σ-DEPENDENT gradient-Lipschitz case (the `mul` node), showing the
    technique extends past the constant affine fragment; the region-bounded general `mul` case scales `δ` by the
    operand magnitudes. -/
theorem derivR_mul_var_lipschitz (i j k : Nat) (σ σ' : Nat → ℝ) (δ : ℝ)
    (hd : ∀ m, |σ m - σ' m| ≤ δ) :
    |derivR (.mul (.var i) (.var j)) σ k - derivR (.mul (.var i) (.var j)) σ' k| ≤ 2 * δ := by
  have hδ : 0 ≤ δ := le_trans (abs_nonneg _) (hd 0)
  simp only [derivR, evalR]
  by_cases hik : i = k <;> by_cases hjk : j = k
  · rw [if_pos hik, if_pos hjk]
    calc |1 * σ j + σ i * 1 - (1 * σ' j + σ' i * 1)|
        = |(σ j - σ' j) + (σ i - σ' i)| := by ring_nf
      _ ≤ |σ j - σ' j| + |σ i - σ' i| := abs_add_le _ _
      _ ≤ δ + δ := add_le_add (hd j) (hd i)
      _ = 2 * δ := by ring
  · rw [if_pos hik, if_neg hjk]
    calc |1 * σ j + σ i * 0 - (1 * σ' j + σ' i * 0)|
        = |σ j - σ' j| := by ring_nf
      _ ≤ δ := hd j
      _ ≤ 2 * δ := by linarith
  · rw [if_neg hik, if_pos hjk]
    calc |0 * σ j + σ i * 1 - (0 * σ' j + σ' i * 1)|
        = |σ i - σ' i| := by ring_nf
      _ ≤ δ := hd i
      _ ≤ 2 * δ := by linarith
  · rw [if_neg hik, if_neg hjk]
    simp only [mul_zero, zero_mul, add_zero, sub_self, abs_zero]
    linarith

/-! #### Region-bounded gradient-Lipschitz for the smooth fragment (polynomial + `exp`)

The full `G` (gradient-Lipschitz constant) for the whole grammar is blocked by `relu`/`max`/`min` (whose
`derivR` JUMPS at the kink — genuinely NOT Lipschitz there) and by `log` (whose derivative `1/evalR a` needs a
positive denominator floor). The SMOOTH, kink-free `Smooth` fragment — `var`/`const`/`add`/`sub`/`scale`/`mul`/
`exp` (linear layers `W·x + b`, their products, AND `exp` nodes — e.g. softmax's log-partition numerator) —
admits a genuine region-bounded gradient-Lipschitz: over the input region `|σ i| ≤ R`, the gradient
`derivR e σ k` is `dLip R e`-Lipschitz in `σ` (sup metric). The bound is a COUPLED chain — value magnitude
(`evalR_mag`) → value Lipschitz (`evalR_lip`) → derivative magnitude (`derivR_mag`) → derivative Lipschitz
(`derivR_lip`) — because the `mul`/`exp` derivative rules mix values and derivatives; the budgets
`vMag`/`vLip`/`dMag`/`dLip` are the corresponding recursively-computed constants (the `exp` node's constants
grow like `exp (vMag a)`, via the local-Lipschitz lemma `exp_abs_sub_le`). This supplies the concrete
`G = dLip R e` that `MuonTrainBound.ascent_map_lipschitz` turns into the ideal-step Lipschitz `L`, discharging
the accumulation hypothesis on the smooth fragment. (`log` — the last softmax piece — remains: it needs a
positive-floor region hypothesis and quotient-difference bounds.) -/

/-- Product-difference bound: `|x·y − x'·y'| ≤ |x|·|y − y'| + |y'|·|x − x'|`. -/
theorem abs_mul_sub_mul_le (x y x' y' : ℝ) :
    |x * y - x' * y'| ≤ |x| * |y - y'| + |y'| * |x - x'| := by
  calc |x * y - x' * y'| = |x * (y - y') + y' * (x - x')| := by ring_nf
    _ ≤ |x * (y - y')| + |y' * (x - x')| := abs_add_le _ _
    _ = |x| * |y - y'| + |y'| * |x - x'| := by rw [abs_mul, abs_mul]

/-- **`exp` is `exp M`-Lipschitz on the ray `(−∞, M]`**: `|exp x − exp y| ≤ exp M · |x − y|` for `x, y ≤ M`.
    The local Lipschitz constant of `exp` on a region capped at `M`; the prerequisite for the `exp` node of the
    gradient-Lipschitz chain (its Lipschitz constant grows like `exp (vMag a)`). Proved from
    `Real.add_one_le_exp`: for `x ≤ y`, `exp y − exp x ≤ exp y·(y − x) ≤ exp M·(y − x)`. -/
theorem exp_abs_sub_le (x y M : ℝ) (hx : x ≤ M) (hy : y ≤ M) :
    |Real.exp x - Real.exp y| ≤ Real.exp M * |x - y| := by
  rcases le_total x y with hxy | hxy
  · rw [abs_of_nonpos (by simp [Real.exp_le_exp.mpr hxy]), abs_of_nonpos (by linarith), neg_sub, neg_sub]
    have key : Real.exp y - Real.exp x ≤ Real.exp y * (y - x) := by
      have h := Real.add_one_le_exp (x - y); have hpos := Real.exp_pos y
      have : Real.exp y * ((x - y) + 1) ≤ Real.exp y * Real.exp (x - y) := mul_le_mul_of_nonneg_left h hpos.le
      rw [← Real.exp_add, show y + (x - y) = x from by ring] at this; nlinarith [this]
    calc Real.exp y - Real.exp x ≤ Real.exp y * (y - x) := key
      _ ≤ Real.exp M * (y - x) := mul_le_mul_of_nonneg_right (Real.exp_le_exp.mpr hy) (by linarith)
  · rw [abs_of_nonneg (by simp [Real.exp_le_exp.mpr hxy]), abs_of_nonneg (by linarith)]
    have key : Real.exp x - Real.exp y ≤ Real.exp x * (x - y) := by
      have h := Real.add_one_le_exp (y - x); have hpos := Real.exp_pos x
      have : Real.exp x * ((y - x) + 1) ≤ Real.exp x * Real.exp (y - x) := mul_le_mul_of_nonneg_left h hpos.le
      rw [← Real.exp_add, show x + (y - x) = y from by ring] at this; nlinarith [this]
    calc Real.exp x - Real.exp y ≤ Real.exp x * (x - y) := key
      _ ≤ Real.exp M * (x - y) := mul_le_mul_of_nonneg_right (Real.exp_le_exp.mpr hx) (by linarith)

/-- The SMOOTH fragment: `var`/`const`/`add`/`sub`/`scale`/`mul`/`exp` — smooth and kink-free (no
    `log`/`relu`/`max`/`min`). Covers linear layers `W·x + b`, their products, and `exp` nodes. -/
inductive Smooth : Expr → Prop
  | var (i : Nat) : Smooth (.var i)
  | const (c : Float) : Smooth (.const c)
  | add {a b} : Smooth a → Smooth b → Smooth (.add a b)
  | sub {a b} : Smooth a → Smooth b → Smooth (.sub a b)
  | scale (c : Float) {a} : Smooth a → Smooth (.scale c a)
  | mul {a b} : Smooth a → Smooth b → Smooth (.mul a b)
  | exp {a} : Smooth a → Smooth (.exp a)

/-- Value-magnitude budget over the input region `|σ i| ≤ R`: `|evalR e σ| ≤ vMag R e`. -/
noncomputable def vMag (R : ℝ) : Expr → ℝ
  | .var _ => R
  | .const c => |toReal c|
  | .add a b => vMag R a + vMag R b
  | .sub a b => vMag R a + vMag R b
  | .scale c a => |toReal c| * vMag R a
  | .mul a b => vMag R a * vMag R b
  | .exp a => Real.exp (vMag R a)
  | _ => 0

/-- Derivative-magnitude budget: `|derivR e σ k| ≤ dMag R e`. -/
noncomputable def dMag (R : ℝ) : Expr → ℝ
  | .var _ => 1
  | .const _ => 0
  | .add a b => dMag R a + dMag R b
  | .sub a b => dMag R a + dMag R b
  | .scale c a => |toReal c| * dMag R a
  | .mul a b => dMag R a * vMag R b + vMag R a * dMag R b
  | .exp a => Real.exp (vMag R a) * dMag R a
  | _ => 0

/-- Value-Lipschitz budget: `|evalR e σ − evalR e σ'| ≤ vLip R e · δ`. -/
noncomputable def vLip (R : ℝ) : Expr → ℝ
  | .var _ => 1
  | .const _ => 0
  | .add a b => vLip R a + vLip R b
  | .sub a b => vLip R a + vLip R b
  | .scale c a => |toReal c| * vLip R a
  | .mul a b => vMag R a * vLip R b + vMag R b * vLip R a
  | .exp a => Real.exp (vMag R a) * vLip R a
  | _ => 0

/-- Derivative-Lipschitz budget (the gradient-Lipschitz constant `G`): `|derivR e σ k − derivR e σ' k| ≤ dLip R e · δ`. -/
noncomputable def dLip (R : ℝ) : Expr → ℝ
  | .var _ => 0
  | .const _ => 0
  | .add a b => dLip R a + dLip R b
  | .sub a b => dLip R a + dLip R b
  | .scale c a => |toReal c| * dLip R a
  | .mul a b => dMag R a * vLip R b + vMag R b * dLip R a + vMag R a * dLip R b + dMag R b * vLip R a
  | .exp a => Real.exp (vMag R a) * dLip R a + dMag R a * (Real.exp (vMag R a) * vLip R a)
  | _ => 0

theorem vMag_nonneg {e : Expr} (he : Smooth e) (R : ℝ) (hR : 0 ≤ R) : 0 ≤ vMag R e := by
  induction he with
  | var i => exact hR
  | const c => exact abs_nonneg _
  | add _ _ iha ihb => exact add_nonneg iha ihb
  | sub _ _ iha ihb => exact add_nonneg iha ihb
  | scale c _ iha => exact mul_nonneg (abs_nonneg _) iha
  | mul _ _ iha ihb => exact mul_nonneg iha ihb
  | exp _ iha => exact (Real.exp_pos _).le

theorem dMag_nonneg {e : Expr} (he : Smooth e) (R : ℝ) (hR : 0 ≤ R) : 0 ≤ dMag R e := by
  induction he with
  | var i => exact zero_le_one
  | const c => exact le_refl 0
  | add _ _ iha ihb => exact add_nonneg iha ihb
  | sub _ _ iha ihb => exact add_nonneg iha ihb
  | scale c _ iha => exact mul_nonneg (abs_nonneg _) iha
  | @mul a b ha hb iha ihb =>
      exact add_nonneg (mul_nonneg iha (vMag_nonneg hb R hR)) (mul_nonneg (vMag_nonneg ha R hR) ihb)
  | exp _ iha => exact mul_nonneg (Real.exp_pos _).le iha

/-- **Value magnitude over the region.** For a smooth expression, `|evalR e σ| ≤ vMag R e` whenever every
    input satisfies `|σ i| ≤ R`. -/
theorem evalR_mag {e : Expr} (he : Smooth e) (σ : Nat → ℝ) (R : ℝ)
    (hσ : ∀ i, |σ i| ≤ R) : |evalR e σ| ≤ vMag R e := by
  induction he with
  | var i => exact hσ i
  | const c => simp only [evalR, vMag]; exact le_refl _
  | add _ _ iha ihb => exact (abs_add_le _ _).trans (add_le_add iha ihb)
  | sub _ _ iha ihb => simp only [evalR, vMag]; exact (abs_sub _ _).trans (add_le_add iha ihb)
  | scale c _ iha => simp only [evalR, vMag, abs_mul]; exact mul_le_mul_of_nonneg_left iha (abs_nonneg _)
  | @mul a b _ _ iha ihb =>
      simp only [evalR, vMag, abs_mul]
      exact mul_le_mul iha ihb (abs_nonneg _) (le_trans (abs_nonneg _) iha)
  | @exp a _ iha =>
      simp only [evalR, vMag, abs_of_pos (Real.exp_pos _)]
      exact Real.exp_le_exp.mpr ((le_abs_self _).trans iha)

/-- **Derivative magnitude over the region.** For a smooth expression, `|derivR e σ k| ≤ dMag R e` on the
    region `|σ i| ≤ R`. -/
theorem derivR_mag {e : Expr} (he : Smooth e) (σ : Nat → ℝ) (R : ℝ) (k : Nat)
    (hσ : ∀ i, |σ i| ≤ R) (hR : 0 ≤ R) : |derivR e σ k| ≤ dMag R e := by
  induction he with
  | var i => simp only [derivR, dMag]; split_ifs <;> simp
  | const c => simp [derivR, dMag]
  | add _ _ iha ihb => exact (abs_add_le _ _).trans (add_le_add iha ihb)
  | sub _ _ iha ihb => simp only [derivR, dMag]; exact (abs_sub _ _).trans (add_le_add iha ihb)
  | scale c _ iha => simp only [derivR, dMag, abs_mul]; exact mul_le_mul_of_nonneg_left iha (abs_nonneg _)
  | @mul a b ha hb iha ihb =>
      simp only [derivR, dMag]
      have hmb : |evalR b σ| ≤ vMag R b := evalR_mag hb σ R hσ
      have hma : |evalR a σ| ≤ vMag R a := evalR_mag ha σ R hσ
      calc |derivR a σ k * evalR b σ + evalR a σ * derivR b σ k|
          ≤ |derivR a σ k * evalR b σ| + |evalR a σ * derivR b σ k| := abs_add_le _ _
        _ = |derivR a σ k| * |evalR b σ| + |evalR a σ| * |derivR b σ k| := by rw [abs_mul, abs_mul]
        _ ≤ dMag R a * vMag R b + vMag R a * dMag R b :=
            add_le_add (mul_le_mul iha hmb (abs_nonneg _) (dMag_nonneg ha R hR))
              (mul_le_mul hma ihb (abs_nonneg _) (vMag_nonneg ha R hR))
  | @exp a ha iha =>
      simp only [derivR, dMag, abs_mul, abs_of_pos (Real.exp_pos _)]
      have hExpLe : Real.exp (evalR a σ) ≤ Real.exp (vMag R a) :=
        Real.exp_le_exp.mpr ((le_abs_self _).trans (evalR_mag ha σ R hσ))
      exact mul_le_mul hExpLe iha (abs_nonneg _) (Real.exp_pos _).le

/-- **Value Lipschitz over the region.** For a smooth expression, `|evalR e σ − evalR e σ'| ≤ vLip R e · δ`
    whenever both inputs lie in the region `|σ i|, |σ' i| ≤ R` and differ by at most `δ` (sup metric). -/
theorem evalR_lip {e : Expr} (he : Smooth e) (σ σ' : Nat → ℝ) (R δ : ℝ)
    (hσ : ∀ i, |σ i| ≤ R) (hσ' : ∀ i, |σ' i| ≤ R) (hδ : ∀ i, |σ i - σ' i| ≤ δ) (hR : 0 ≤ R) :
    |evalR e σ - evalR e σ'| ≤ vLip R e * δ := by
  induction he with
  | var i => simp only [evalR, vLip, one_mul]; exact hδ i
  | const c => simp [evalR, vLip]
  | @add a b _ _ iha ihb =>
      simp only [evalR, vLip]
      calc |evalR a σ + evalR b σ - (evalR a σ' + evalR b σ')|
          = |(evalR a σ - evalR a σ') + (evalR b σ - evalR b σ')| := by ring_nf
        _ ≤ |evalR a σ - evalR a σ'| + |evalR b σ - evalR b σ'| := abs_add_le _ _
        _ ≤ vLip R a * δ + vLip R b * δ := add_le_add iha ihb
        _ = (vLip R a + vLip R b) * δ := by ring
  | @sub a b _ _ iha ihb =>
      simp only [evalR, vLip]
      calc |evalR a σ - evalR b σ - (evalR a σ' - evalR b σ')|
          = |(evalR a σ - evalR a σ') - (evalR b σ - evalR b σ')| := by ring_nf
        _ ≤ |evalR a σ - evalR a σ'| + |evalR b σ - evalR b σ'| := abs_sub _ _
        _ ≤ vLip R a * δ + vLip R b * δ := add_le_add iha ihb
        _ = (vLip R a + vLip R b) * δ := by ring
  | @scale c a _ iha =>
      simp only [evalR, vLip]
      calc |toReal c * evalR a σ - toReal c * evalR a σ'|
          = |toReal c| * |evalR a σ - evalR a σ'| := by rw [← mul_sub, abs_mul]
        _ ≤ |toReal c| * (vLip R a * δ) := mul_le_mul_of_nonneg_left iha (abs_nonneg _)
        _ = |toReal c| * vLip R a * δ := by ring
  | @mul a b ha hb iha ihb =>
      simp only [evalR, vLip]
      have hma : |evalR a σ| ≤ vMag R a := evalR_mag ha σ R hσ
      have hmb' : |evalR b σ'| ≤ vMag R b := evalR_mag hb σ' R hσ'
      calc |evalR a σ * evalR b σ - evalR a σ' * evalR b σ'|
          ≤ |evalR a σ| * |evalR b σ - evalR b σ'| + |evalR b σ'| * |evalR a σ - evalR a σ'| :=
            abs_mul_sub_mul_le _ _ _ _
        _ ≤ vMag R a * (vLip R b * δ) + vMag R b * (vLip R a * δ) :=
            add_le_add (mul_le_mul hma ihb (abs_nonneg _) (vMag_nonneg ha R hR))
              (mul_le_mul hmb' iha (abs_nonneg _) (vMag_nonneg hb R hR))
        _ = (vMag R a * vLip R b + vMag R b * vLip R a) * δ := by ring
  | @exp a ha iha =>
      simp only [evalR, vLip]
      have hMa : evalR a σ ≤ vMag R a := (le_abs_self _).trans (evalR_mag ha σ R hσ)
      have hMa' : evalR a σ' ≤ vMag R a := (le_abs_self _).trans (evalR_mag ha σ' R hσ')
      calc |Real.exp (evalR a σ) - Real.exp (evalR a σ')|
          ≤ Real.exp (vMag R a) * |evalR a σ - evalR a σ'| := exp_abs_sub_le _ _ _ hMa hMa'
        _ ≤ Real.exp (vMag R a) * (vLip R a * δ) := mul_le_mul_of_nonneg_left iha (Real.exp_pos _).le
        _ = Real.exp (vMag R a) * vLip R a * δ := by ring

/-- **Region-bounded gradient-Lipschitz for the smooth fragment.** For a smooth expression (linear layers,
    their products, and `exp` nodes), over the input region `|σ i|, |σ' i| ≤ R` with `|σ i − σ' i| ≤ δ`, the
    gradient is `dLip R e`-Lipschitz in the parameters: `|derivR e σ k − derivR e σ' k| ≤ dLip R e · δ`. This is
    the concrete gradient-Lipschitz constant `G = dLip R e` that `MuonTrainBound.ascent_map_lipschitz` converts
    into the ideal-step Lipschitz `L = 1 + |lr|·G`, discharging the `hlip` hypothesis of the N-step accumulation
    on the smooth fragment. The `mul`/`exp` cases are where the coupling bites: `derivR (a·b) = da·b + a·db` and
    `derivR (exp a) = exp(a)·da` mix values and derivatives, so bounding their variation needs the value
    magnitudes (`evalR_mag`), value Lipschitz (`evalR_lip`), derivative magnitudes (`derivR_mag`), and the
    derivative-Lipschitz IHs — assembled via `abs_mul_sub_mul_le` (and, for `exp`, `exp_abs_sub_le`) on each
    product term. (Scope: `relu`/`max`/`min` are excluded — their `derivR` jumps at kinks, so the gradient is
    genuinely NOT Lipschitz there; `log` — the last softmax piece — needs a positive-floor region hypothesis and
    quotient-difference bounds.) -/
theorem derivR_lip {e : Expr} (he : Smooth e) (σ σ' : Nat → ℝ) (R δ : ℝ) (k : Nat)
    (hσ : ∀ i, |σ i| ≤ R) (hσ' : ∀ i, |σ' i| ≤ R) (hδ : ∀ i, |σ i - σ' i| ≤ δ) (hR : 0 ≤ R) :
    |derivR e σ k - derivR e σ' k| ≤ dLip R e * δ := by
  induction he with
  | var i => simp only [derivR, dLip, zero_mul]; split_ifs <;> simp
  | const c => simp [derivR, dLip]
  | @add a b _ _ iha ihb =>
      simp only [derivR, dLip]
      calc |derivR a σ k + derivR b σ k - (derivR a σ' k + derivR b σ' k)|
          = |(derivR a σ k - derivR a σ' k) + (derivR b σ k - derivR b σ' k)| := by ring_nf
        _ ≤ |derivR a σ k - derivR a σ' k| + |derivR b σ k - derivR b σ' k| := abs_add_le _ _
        _ ≤ dLip R a * δ + dLip R b * δ := add_le_add iha ihb
        _ = (dLip R a + dLip R b) * δ := by ring
  | @sub a b _ _ iha ihb =>
      simp only [derivR, dLip]
      calc |derivR a σ k - derivR b σ k - (derivR a σ' k - derivR b σ' k)|
          = |(derivR a σ k - derivR a σ' k) - (derivR b σ k - derivR b σ' k)| := by ring_nf
        _ ≤ |derivR a σ k - derivR a σ' k| + |derivR b σ k - derivR b σ' k| := abs_sub _ _
        _ ≤ dLip R a * δ + dLip R b * δ := add_le_add iha ihb
        _ = (dLip R a + dLip R b) * δ := by ring
  | @scale c a _ iha =>
      simp only [derivR, dLip]
      calc |toReal c * derivR a σ k - toReal c * derivR a σ' k|
          = |toReal c| * |derivR a σ k - derivR a σ' k| := by rw [← mul_sub, abs_mul]
        _ ≤ |toReal c| * (dLip R a * δ) := mul_le_mul_of_nonneg_left iha (abs_nonneg _)
        _ = |toReal c| * dLip R a * δ := by ring
  | @mul a b ha hb iha ihb =>
      simp only [derivR, dLip]
      have hDma : |derivR a σ k| ≤ dMag R a := derivR_mag ha σ R k hσ hR
      have hVmb' : |evalR b σ'| ≤ vMag R b := evalR_mag hb σ' R hσ'
      have hVma : |evalR a σ| ≤ vMag R a := evalR_mag ha σ R hσ
      have hDmb' : |derivR b σ' k| ≤ dMag R b := derivR_mag hb σ' R k hσ' hR
      have hLeb : |evalR b σ - evalR b σ'| ≤ vLip R b * δ := evalR_lip hb σ σ' R δ hσ hσ' hδ hR
      have hLea : |evalR a σ - evalR a σ'| ≤ vLip R a * δ := evalR_lip ha σ σ' R δ hσ hσ' hδ hR
      calc |derivR a σ k * evalR b σ + evalR a σ * derivR b σ k
              - (derivR a σ' k * evalR b σ' + evalR a σ' * derivR b σ' k)|
          = |(derivR a σ k * evalR b σ - derivR a σ' k * evalR b σ')
              + (evalR a σ * derivR b σ k - evalR a σ' * derivR b σ' k)| := by ring_nf
        _ ≤ |derivR a σ k * evalR b σ - derivR a σ' k * evalR b σ'|
              + |evalR a σ * derivR b σ k - evalR a σ' * derivR b σ' k| := abs_add_le _ _
        _ ≤ (|derivR a σ k| * |evalR b σ - evalR b σ'| + |evalR b σ'| * |derivR a σ k - derivR a σ' k|)
              + (|evalR a σ| * |derivR b σ k - derivR b σ' k| + |derivR b σ' k| * |evalR a σ - evalR a σ'|) :=
            add_le_add (abs_mul_sub_mul_le _ _ _ _) (abs_mul_sub_mul_le _ _ _ _)
        _ ≤ (dMag R a * (vLip R b * δ) + vMag R b * (dLip R a * δ))
              + (vMag R a * (dLip R b * δ) + dMag R b * (vLip R a * δ)) :=
            add_le_add
              (add_le_add (mul_le_mul hDma hLeb (abs_nonneg _) (dMag_nonneg ha R hR))
                (mul_le_mul hVmb' iha (abs_nonneg _) (vMag_nonneg hb R hR)))
              (add_le_add (mul_le_mul hVma ihb (abs_nonneg _) (vMag_nonneg ha R hR))
                (mul_le_mul hDmb' hLea (abs_nonneg _) (dMag_nonneg hb R hR)))
        _ = (dMag R a * vLip R b + vMag R b * dLip R a + vMag R a * dLip R b + dMag R b * vLip R a) * δ := by
            ring
  | @exp a ha iha =>
      simp only [derivR, dLip]
      have hMa : evalR a σ ≤ vMag R a := (le_abs_self _).trans (evalR_mag ha σ R hσ)
      have hMa' : evalR a σ' ≤ vMag R a := (le_abs_self _).trans (evalR_mag ha σ' R hσ')
      have hExpLe : Real.exp (evalR a σ) ≤ Real.exp (vMag R a) := Real.exp_le_exp.mpr hMa
      have hDma' : |derivR a σ' k| ≤ dMag R a := derivR_mag ha σ' R k hσ' hR
      have hLea : |evalR a σ - evalR a σ'| ≤ vLip R a * δ := evalR_lip ha σ σ' R δ hσ hσ' hδ hR
      have hExpDiff : |Real.exp (evalR a σ) - Real.exp (evalR a σ')| ≤ Real.exp (vMag R a) * vLip R a * δ := by
        calc |Real.exp (evalR a σ) - Real.exp (evalR a σ')|
            ≤ Real.exp (vMag R a) * |evalR a σ - evalR a σ'| := exp_abs_sub_le _ _ _ hMa hMa'
          _ ≤ Real.exp (vMag R a) * (vLip R a * δ) := mul_le_mul_of_nonneg_left hLea (Real.exp_pos _).le
          _ = Real.exp (vMag R a) * vLip R a * δ := by ring
      calc |Real.exp (evalR a σ) * derivR a σ k - Real.exp (evalR a σ') * derivR a σ' k|
          ≤ |Real.exp (evalR a σ)| * |derivR a σ k - derivR a σ' k|
              + |derivR a σ' k| * |Real.exp (evalR a σ) - Real.exp (evalR a σ')| := abs_mul_sub_mul_le _ _ _ _
        _ ≤ Real.exp (vMag R a) * (dLip R a * δ) + dMag R a * (Real.exp (vMag R a) * vLip R a * δ) := by
            apply add_le_add
            · rw [abs_of_pos (Real.exp_pos _)]; exact mul_le_mul hExpLe iha (abs_nonneg _) (Real.exp_pos _).le
            · exact mul_le_mul hDma' hExpDiff (abs_nonneg _) (dMag_nonneg ha R hR)
        _ = (Real.exp (vMag R a) * dLip R a + dMag R a * (Real.exp (vMag R a) * vLip R a)) * δ := by ring

/-! #### The `log` node: gradient-Lipschitz with a positive denominator floor

`log` is the last softmax piece. Its symbolic derivative `derivR (log a) = derivR a / evalR a` is a DIVISION,
so — unlike `exp` — it cannot be folded into the recursive `Smooth` budgets: its Lipschitz constant blows up as
`evalR a → 0`, and no bound derivable from the input region `R` alone can rule that out. `log` needs a genuinely
SEMANTIC side-condition: a strictly positive floor `c` on the argument value. So rather than distort the whole
budget system with a global floor, we expose `log` as a COMPOSABLE NODE: given the argument `a` is `Smooth` and
`evalR a ≥ c > 0` on both points of the region, the `log`-node gradient is `(dLip a / c + dMag a · vLip a / c²)`-
Lipschitz. The constant reuses the same four `Smooth` budgets — this is exactly the C4/C5 machinery composed with
the quotient-difference bound `abs_div_sub_div_le` (`1/x` is `1/c²`-Lipschitz on `[c, ∞)`). -/

/-- **Quotient-difference bound with a positive floor.** For denominators `p, q ≥ c > 0`:
    `|x/p − y/q| ≤ |x − y|/c + |y|·|p − q|/c²`. The `1/x`-Lipschitz estimate underlying the `log`-node gradient
    bound, split as `x/p − y/q = (x − y)/p + y·(q − p)/(p·q)` then floored (`1/p ≤ 1/c`, `1/(p·q) ≤ 1/c²`). -/
theorem abs_div_sub_div_le (x p y q c : ℝ) (hc : 0 < c) (hp : c ≤ p) (hq : c ≤ q) :
    |x / p - y / q| ≤ |x - y| / c + |y| * |p - q| / c ^ 2 := by
  have hp0 : 0 < p := lt_of_lt_of_le hc hp
  have hq0 : 0 < q := lt_of_lt_of_le hc hq
  have key : x / p - y / q = (x - y) / p + y * (q - p) / (p * q) := by
    field_simp
    ring
  calc |x / p - y / q| = |(x - y) / p + y * (q - p) / (p * q)| := by rw [key]
    _ ≤ |(x - y) / p| + |y * (q - p) / (p * q)| := abs_add_le _ _
    _ = |x - y| / p + |y| * |q - p| / (p * q) := by
        rw [abs_div, abs_of_pos hp0, abs_div, abs_mul, abs_of_pos (mul_pos hp0 hq0)]
    _ = |x - y| / p + |y| * |p - q| / (p * q) := by rw [abs_sub_comm q p]
    _ ≤ |x - y| / c + |y| * |p - q| / c ^ 2 := by
        have hc2 : c ^ 2 ≤ p * q := by
          rw [sq]; exact mul_le_mul hp hq hc.le hp0.le
        gcongr

/-- **`log`-node region-bounded gradient-Lipschitz (with a positive floor).** For a `Smooth` argument `a` whose
    value stays `≥ c > 0` on both points of the region (`evalR a σ, evalR a σ' ≥ c`), the gradient of `log a`
    (`derivR (log a) = derivR a / evalR a`) is Lipschitz in the parameters:
    `|derivR (log a) σ k − derivR (log a) σ' k| ≤ (dLip R a / c + dMag R a · vLip R a / c²) · δ`. This is the
    LAST softmax piece assembled as a composable node — it consumes the same four `Smooth` budgets (`dLip R a`,
    `dMag R a`, `vLip R a` via `derivR_lip`/`derivR_mag`/`evalR_lip`) and the quotient bound `abs_div_sub_div_le`.
    The positive floor `c` is a genuinely SEMANTIC hypothesis (not derivable from the input region `R`), which is
    why `log` is NOT folded into the recursive budgets — its gradient's Lipschitz constant `∝ 1/c²` diverges as
    the argument → 0. With this node, softmax's `log`-partition is covered given a partition-value floor; `relu`
    remains the only intrinsically-excluded piece (its `derivR` jumps at the kink). -/
theorem derivR_log_lip {a : Expr} (ha : Smooth a) (σ σ' : Nat → ℝ) (k : Nat)
    (R δ c : ℝ) (hσ : ∀ i, |σ i| ≤ R) (hσ' : ∀ i, |σ' i| ≤ R)
    (hδ : ∀ i, |σ i - σ' i| ≤ δ) (hR : 0 ≤ R) (hc : 0 < c)
    (hfloor : c ≤ evalR a σ) (hfloor' : c ≤ evalR a σ') :
    |derivR (.log a) σ k - derivR (.log a) σ' k|
      ≤ (dLip R a / c + dMag R a * vLip R a / c ^ 2) * δ := by
  simp only [derivR]
  have hDl : |derivR a σ k - derivR a σ' k| ≤ dLip R a * δ :=
    derivR_lip ha σ σ' R δ k hσ hσ' hδ hR
  have hDm : |derivR a σ' k| ≤ dMag R a := derivR_mag ha σ' R k hσ' hR
  have hLv : |evalR a σ - evalR a σ'| ≤ vLip R a * δ := evalR_lip ha σ σ' R δ hσ hσ' hδ hR
  calc |derivR a σ k / evalR a σ - derivR a σ' k / evalR a σ'|
      ≤ |derivR a σ k - derivR a σ' k| / c
          + |derivR a σ' k| * |evalR a σ - evalR a σ'| / c ^ 2 :=
        abs_div_sub_div_le _ _ _ _ c hc hfloor hfloor'
    _ ≤ dLip R a * δ / c + dMag R a * (vLip R a * δ) / c ^ 2 := by
        gcongr
        exact le_trans (abs_nonneg _) hDm
    _ = (dLip R a / c + dMag R a * vLip R a / c ^ 2) * δ := by ring

/-! #### The `relu` node: gradient-Lipschitz AWAY FROM the kink

`relu` is the last piece of the grammar, and its gradient is genuinely NOT Lipschitz across the kink at
`evalR a = 0`: `derivR (relu a) = if 0 < evalR a then derivR a else 0` JUMPS from `derivR a` to `0` there, an
irreducible discontinuity (no finite constant bounds `|Δgrad|/δ` as a point pair straddles 0). But `relu` is
PIECEWISE LINEAR, so on each side of the kink it IS Lipschitz — and, unlike `log`, with NO amplification (no
floor constant), because `relu` is locally an isometry-of-gradient (identity or zero), not a nonlinear reshaping:

* **Active side** (`evalR a σ, evalR a σ' > 0`): `relu` is locally `id`, so `derivR (relu a) = derivR a` at both
  points and the gradient passes through with the argument's OWN constant — `derivR_relu_active_lip`:
  `|derivR (relu a) σ k − derivR (relu a) σ' k| ≤ dLip R a · δ` (same `dLip R a` as the smooth argument, no `1/c`).
* **Inactive side** (`evalR a σ ≤ 0`): `derivR (relu a) σ k = 0` pointwise — `derivR_relu_inactive_zero` — so the
  gradient is identically zero there (trivially `0`-Lipschitz).

So `relu`'s non-Lipschitzness is PURELY a kink-crossing artifact: given a same-side (off-kink) region hypothesis,
the `relu` node composes into the gradient-Lipschitz story with the exact smooth-argument constant. This is the
honest completion — every grammar node now has a region-bounded gradient-Lipschitz bound under an explicit,
node-appropriate off-singularity hypothesis (`exp`: none; `log`: positive floor; `relu`: same side of the kink). -/

/-- **`relu`-node gradient-Lipschitz on the active side.** When the `Smooth` argument `a` is strictly positive at
    both points (`evalR a σ, evalR a σ' > 0` — the region stays on the ACTIVE side of the kink), `relu` is locally
    the identity, so its gradient passes through unchanged and is Lipschitz with the argument's OWN constant:
    `|derivR (relu a) σ k − derivR (relu a) σ' k| ≤ dLip R a · δ`. No floor/amplification (contrast `log`'s `1/c²`):
    `relu` on the active side is exactly `id`, so `derivR (relu a) = derivR a` and the bound is just `derivR_lip`
    on `a`. The strict positivity at BOTH points is the load-bearing off-kink hypothesis — without it a straddling
    pair jumps by `|derivR a|` regardless of `δ`, which no finite constant bounds. -/
theorem derivR_relu_active_lip {a : Expr} (ha : Smooth a) (σ σ' : Nat → ℝ) (k : Nat)
    (R δ : ℝ) (hσ : ∀ i, |σ i| ≤ R) (hσ' : ∀ i, |σ' i| ≤ R)
    (hδ : ∀ i, |σ i - σ' i| ≤ δ) (hR : 0 ≤ R)
    (hpos : 0 < evalR a σ) (hpos' : 0 < evalR a σ') :
    |derivR (.relu a) σ k - derivR (.relu a) σ' k| ≤ dLip R a * δ := by
  simp only [derivR, if_pos hpos, if_pos hpos']
  exact derivR_lip ha σ σ' R δ k hσ hσ' hδ hR

/-- **`relu`-node gradient is zero on the inactive side.** When the argument is `≤ 0` (the INACTIVE side of the
    kink, including the kink point itself), `relu`'s symbolic gradient is identically zero: `derivR (relu a) σ k = 0`.
    Hence on any region staying `≤ 0` the `relu`-node gradient is trivially `0`-Lipschitz (`|Δgrad| = 0`). Together
    with `derivR_relu_active_lip`, this shows `relu` is gradient-Lipschitz on EITHER side of the kink; only
    kink-crossing (a straddling pair) is excluded. -/
theorem derivR_relu_inactive_zero {a : Expr} (σ : Nat → ℝ) (k : Nat)
    (hle : evalR a σ ≤ 0) :
    derivR (.relu a) σ k = 0 := by
  simp only [derivR, if_neg (not_lt.mpr hle)]

/-- **Gradient log-exp cancellation.** The exact-ℝ symbolic derivative respects the algebraic identity
    `log (exp x) = x` at the gradient level: `derivR (log (exp a)) = derivR a`. The chain rule produces
    `exp(evalR a) · derivR a / exp(evalR a)`, and the `exp` factor cancels (it is never zero), leaving the inner
    gradient. Unconditional — needs no `PosR`, since `exp` is strictly positive everywhere so the `log` argument
    is automatically well-defined at the kink-free point. -/
theorem derivR_log_exp (a : Expr) (σ : Nat → ℝ) (k : Nat) :
    derivR (.log (.exp a)) σ k = derivR a σ k := by
  simp only [derivR, evalR]
  rw [mul_div_cancel_left₀ _ (Real.exp_ne_zero _)]

/-- **Clip-interior identity (value AND gradient pass-through).** On the strict interior of its clip window
    (`toReal lo < evalR a σ < toReal hi`), the two-sided clip `clampE a lo hi`
    (`= .min (.max a (.const lo)) (.const hi)`, PPO's ratio clip) is the identity on both the exact-ℝ value and
    the exact-ℝ forward-mode gradient: `evalR (clampE a lo hi) σ = evalR a σ` and
    `derivR (clampE a lo hi) σ k = derivR a σ k`. This is exactly the "unclipped region" fact behind PPO's
    clipped surrogate — inside the trust region the clip vanishes and the gradient is the raw (unclipped)
    gradient of `a`. Both bounds are load-bearing: if `evalR a σ ≤ toReal lo` the clip saturates low (value
    `= toReal lo`, gradient `0`), and if `evalR a σ ≥ toReal hi` it saturates high (value `= toReal hi`,
    gradient `0`) — in either saturated case the gradient is `0`, not `derivR a σ k`. Purely structural (no
    `PosR` needed): the strict inequalities keep the argument off both min/max kinks, collapsing the nested
    `if`s of `evalR`/`derivR` via `max_eq_left`/`min_eq_left`/`if_pos`/`if_neg`. Complements the file's other
    structural laws (`derivR_eq_zero_of_not_occurs`, `derivR_congr`, `derivR_log_exp`) with the clip-specific
    one; `clampE` previously had ZERO theorems in this file. -/
theorem clampE_interior_id (a : Expr) (lo hi : Float) (σ : Nat → ℝ) (k : Nat)
    (hlo : toReal lo < evalR a σ) (hhi : evalR a σ < toReal hi) :
    evalR (clampE a lo hi) σ = evalR a σ ∧ derivR (clampE a lo hi) σ k = derivR a σ k := by
  refine ⟨?_, ?_⟩
  · unfold clampE
    simp only [evalR]
    rw [max_eq_left hlo.le, min_eq_left hhi.le]
  · unfold clampE
    simp only [derivR, evalR]
    rw [max_eq_left hlo.le, if_pos hhi.le, if_neg (not_le.mpr hlo)]

/-! ### AD correctness over ℝ: `derivR` is the real derivative -/

/-- `derivR e σ k` is genuinely `∂/∂t evalR e (σ with var k ↦ t)` at `t = σ k` (given `PosR`
    so every `log` argument is nonzero). -/
theorem derivR_hasDerivAt (e : Expr) (σ : Nat → ℝ) (k : Nat) :
    PosR e σ → HasDerivAt (fun t => evalR e (Function.update σ k t)) (derivR e σ k) (σ k) := by
  induction e with
  | var i =>
      intro _
      simp only [evalR, derivR, Function.update_apply]
      by_cases h : i = k
      · subst h; simpa using hasDerivAt_id (σ i)
      · simp only [if_neg h]; exact hasDerivAt_const _ _
  | const c => intro _; simpa [evalR] using hasDerivAt_const (σ k) (toReal c)
  | add a b iha ihb =>
      intro hp; simp only [PosR] at hp
      simpa only [evalR, derivR, Function.update_eq_self] using (iha hp.1).add (ihb hp.2)
  | sub a b iha ihb =>
      intro hp; simp only [PosR] at hp
      simpa only [evalR, derivR, Function.update_eq_self] using (iha hp.1).sub (ihb hp.2)
  | mul a b iha ihb =>
      intro hp; simp only [PosR] at hp
      have h := (iha hp.1).mul (ihb hp.2)
      simp only [evalR, derivR]; rw [Function.update_eq_self] at h; exact h
  | scale c a iha =>
      intro hp; simp only [PosR] at hp
      simpa only [evalR, derivR] using (iha hp).const_mul (toReal c)
  | exp a iha =>
      intro hp; simp only [PosR] at hp
      have h := (iha hp).exp
      simp only [evalR, derivR]; rw [Function.update_eq_self] at h; exact h
  | log a iha =>
      intro hp; simp only [PosR] at hp
      obtain ⟨ha, hpos⟩ := hp
      have hne : evalR a (Function.update σ k (σ k)) ≠ 0 := by
        rw [Function.update_eq_self]; exact hpos.ne'
      have h := (iha ha).log hne
      simp only [evalR, derivR]; rw [Function.update_eq_self] at h; exact h
  | relu a iha =>
      intro hp; simp only [PosR] at hp
      obtain ⟨ha, hne⟩ := hp
      -- `g t := evalR a (update σ k t)` has derivative `derivR a σ k` at `σ k` and `g (σ k) ≠ 0`.
      have hg : HasDerivAt (fun t => evalR a (Function.update σ k t)) (derivR a σ k) (σ k) := iha ha
      -- `g := fun t => evalR a (update σ k t)` tends to `evalR a σ` (from `HasDerivAt`).
      have htend : Filter.Tendsto (fun t => evalR a (Function.update σ k t))
          (nhds (σ k)) (nhds (evalR a σ)) := by
        have := hg.continuousAt
        rw [ContinuousAt, Function.update_eq_self] at this; exact this
      simp only [evalR, derivR]
      rcases lt_or_gt_of_ne hne with hlt | hgt
      · -- argument `< 0`: `relu` is locally the constant `0`.
        rw [if_neg (not_lt.mpr hlt.le)]
        have hev : (fun t => max (evalR a (Function.update σ k t)) 0)
            =ᶠ[nhds (σ k)] fun _ => (0 : ℝ) := by
          filter_upwards [htend.eventually_lt_const hlt] with t ht
          exact max_eq_right ht.le
        exact (hasDerivAt_const (σ k) (0 : ℝ)).congr_of_eventuallyEq hev
      · -- argument `> 0`: `relu` is locally the identity `g`.
        rw [if_pos hgt]
        have hev : (fun t => max (evalR a (Function.update σ k t)) 0)
            =ᶠ[nhds (σ k)] fun t => evalR a (Function.update σ k t) := by
          filter_upwards [htend.eventually_const_lt hgt] with t ht
          exact max_eq_left ht.le
        exact hg.congr_of_eventuallyEq hev
  | max a b iha ihb =>
      intro hp; simp only [PosR] at hp
      obtain ⟨ha, hb, hne⟩ := hp
      have hga := iha ha; have hgb := ihb hb
      have hta : Filter.Tendsto (fun t => evalR a (Function.update σ k t))
          (nhds (σ k)) (nhds (evalR a σ)) := by
        have := hga.continuousAt; rw [ContinuousAt, Function.update_eq_self] at this; exact this
      have htb : Filter.Tendsto (fun t => evalR b (Function.update σ k t))
          (nhds (σ k)) (nhds (evalR b σ)) := by
        have := hgb.continuousAt; rw [ContinuousAt, Function.update_eq_self] at this; exact this
      simp only [evalR, derivR]
      rcases lt_or_gt_of_ne hne with hlt | hgt
      · -- `evalR a < evalR b`: `max` is locally `b`.
        rw [if_pos hlt.le]
        have hev : (fun t => max (evalR a (Function.update σ k t)) (evalR b (Function.update σ k t)))
            =ᶠ[nhds (σ k)] fun t => evalR b (Function.update σ k t) := by
          filter_upwards [(hta.sub htb).eventually_lt_const (by linarith : evalR a σ - evalR b σ < 0)]
            with t ht
          exact max_eq_right (by linarith [ht])
        exact hgb.congr_of_eventuallyEq hev
      · -- `evalR a > evalR b`: `max` is locally `a`.
        rw [if_neg (not_le.mpr hgt)]
        have hev : (fun t => max (evalR a (Function.update σ k t)) (evalR b (Function.update σ k t)))
            =ᶠ[nhds (σ k)] fun t => evalR a (Function.update σ k t) := by
          filter_upwards [(hta.sub htb).eventually_const_lt (by linarith : (0:ℝ) < evalR a σ - evalR b σ)]
            with t ht
          exact max_eq_left (by linarith [ht])
        exact hga.congr_of_eventuallyEq hev
  | min a b iha ihb =>
      intro hp; simp only [PosR] at hp
      obtain ⟨ha, hb, hne⟩ := hp
      have hga := iha ha; have hgb := ihb hb
      have hta : Filter.Tendsto (fun t => evalR a (Function.update σ k t))
          (nhds (σ k)) (nhds (evalR a σ)) := by
        have := hga.continuousAt; rw [ContinuousAt, Function.update_eq_self] at this; exact this
      have htb : Filter.Tendsto (fun t => evalR b (Function.update σ k t))
          (nhds (σ k)) (nhds (evalR b σ)) := by
        have := hgb.continuousAt; rw [ContinuousAt, Function.update_eq_self] at this; exact this
      simp only [evalR, derivR]
      rcases lt_or_gt_of_ne hne with hlt | hgt
      · -- `evalR a < evalR b`: `min` is locally `a`.
        rw [if_pos hlt.le]
        have hev : (fun t => min (evalR a (Function.update σ k t)) (evalR b (Function.update σ k t)))
            =ᶠ[nhds (σ k)] fun t => evalR a (Function.update σ k t) := by
          filter_upwards [(hta.sub htb).eventually_lt_const (by linarith : evalR a σ - evalR b σ < 0)]
            with t ht
          exact min_eq_left (by linarith [ht])
        exact hga.congr_of_eventuallyEq hev
      · -- `evalR a > evalR b`: `min` is locally `b`.
        rw [if_neg (not_le.mpr hgt)]
        have hev : (fun t => min (evalR a (Function.update σ k t)) (evalR b (Function.update σ k t)))
            =ᶠ[nhds (σ k)] fun t => evalR b (Function.update σ k t) := by
          filter_upwards [(hta.sub htb).eventually_const_lt (by linarith : (0:ℝ) < evalR a σ - evalR b σ)]
            with t ht
          exact min_eq_right (by linarith [ht])
        exact hgb.congr_of_eventuallyEq hev

/-- `derivR` equals the Mathlib derivative (AD correctness, as a `deriv` equation). -/
theorem derivR_eq_deriv (e : Expr) (σ : Nat → ℝ) (k : Nat) (hp : PosR e σ) :
    derivR e σ k = deriv (fun t => evalR e (Function.update σ k t)) (σ k) :=
  (derivR_hasDerivAt e σ k hp).deriv.symm

/-! ### AD error bounds: the `Float` value and derivative track their real values -/

/-- Computable-shape (ℝ) error bound on the `Float` value `evalF e σ`. -/
noncomputable def evalErrBnd : Expr → (Nat → Float) → ℝ
  | .var _, _ => 0
  | .const _, _ => 0
  | .add a b, σ => u64 * |toReal (evalF a σ) + toReal (evalF b σ)| + evalErrBnd a σ + evalErrBnd b σ
  | .sub a b, σ => u64 * |toReal (evalF a σ) - toReal (evalF b σ)| + evalErrBnd a σ + evalErrBnd b σ
  | .mul a b, σ => u64 * |toReal (evalF a σ) * toReal (evalF b σ)|
      + |toReal (evalF a σ)| * evalErrBnd b σ + |evalR b (envR σ)| * evalErrBnd a σ
  | .scale c a, σ => u64 * |toReal c * toReal (evalF a σ)| + |toReal c| * evalErrBnd a σ
  | .exp a, σ => expEps * Real.exp (toReal (evalF a σ))
      + Real.exp (toReal (evalF a σ)) * (Real.exp (evalErrBnd a σ) - 1)
  | .log a, σ => logEps * |Real.log (toReal (evalF a σ))|
      + evalErrBnd a σ / (toReal (evalF a σ) - evalErrBnd a σ)
  | .relu a, σ => evalErrBnd a σ
  | .max a b, σ => evalErrBnd a σ + evalErrBnd b σ
  | .min a b, σ => evalErrBnd a σ + evalErrBnd b σ

/-- Float well-definedness: every `log`-argument's `Float` value strictly exceeds its own error
    bound (hence stays positive, with denominator floor `toReal(evalF a) − evalErrBnd a > 0`).
    Trivially `True` for `log`-free expressions. -/
def WD : Expr → (Nat → Float) → Prop
  | .var _, _ => True
  | .const _, _ => True
  | .add a b, σ => WD a σ ∧ WD b σ
  | .sub a b, σ => WD a σ ∧ WD b σ
  | .mul a b, σ => WD a σ ∧ WD b σ
  | .scale _ a, σ => WD a σ
  | .exp a, σ => WD a σ
  | .log a, σ => WD a σ ∧ evalErrBnd a σ < toReal (evalF a σ)
  -- `relu`'s away-from-kink condition: the argument's error bound is smaller than its `Float`
  -- magnitude, so the `Float` value is nonzero AND (via `evalF_error`) the real value shares its
  -- sign — the sign-branch of `dF` then agrees with `derivR` (no rounding is introduced).
  | .relu a, σ => WD a σ ∧ evalErrBnd a σ < |toReal (evalF a σ)|
  -- `max`/`min`'s away-from-kink condition: the two `Float` values are separated by more than the
  -- SUM of their error bounds, so the `Float` order of the arguments matches their real order — the
  -- `if a ≤ b` branch of `dF` then agrees with `derivR`'s (no rounding is introduced).
  | .max a b, σ => WD a σ ∧ WD b σ ∧ evalErrBnd a σ + evalErrBnd b σ < |toReal (evalF a σ) - toReal (evalF b σ)|
  | .min a b, σ => WD a σ ∧ WD b σ ∧ evalErrBnd a σ + evalErrBnd b σ < |toReal (evalF a σ) - toReal (evalF b σ)|

theorem evalErrBnd_nonneg (e : Expr) (σ : Nat → Float) : WD e σ → 0 ≤ evalErrBnd e σ := by
  induction e with
  | var i => intro _; simp [evalErrBnd]
  | const c => intro _; simp [evalErrBnd]
  | add a b iha ihb => intro hwd; simp only [WD] at hwd; simp only [evalErrBnd]
                       exact add_nonneg (add_nonneg (mul_nonneg u64_pos.le (abs_nonneg _)) (iha hwd.1)) (ihb hwd.2)
  | sub a b iha ihb => intro hwd; simp only [WD] at hwd; simp only [evalErrBnd]
                       exact add_nonneg (add_nonneg (mul_nonneg u64_pos.le (abs_nonneg _)) (iha hwd.1)) (ihb hwd.2)
  | mul a b iha ihb => intro hwd; simp only [WD] at hwd; simp only [evalErrBnd]
                       exact add_nonneg (add_nonneg (mul_nonneg u64_pos.le (abs_nonneg _))
                         (mul_nonneg (abs_nonneg _) (ihb hwd.2))) (mul_nonneg (abs_nonneg _) (iha hwd.1))
  | scale c a iha => intro hwd; simp only [evalErrBnd]
                     exact add_nonneg (mul_nonneg u64_pos.le (abs_nonneg _)) (mul_nonneg (abs_nonneg _) (iha hwd))
  | exp a iha => intro hwd; simp only [evalErrBnd]
                 have ha := iha hwd
                 have h1 : (1 : ℝ) ≤ Real.exp (evalErrBnd a σ) := by
                   rw [← Real.exp_zero]; exact Real.exp_le_exp.mpr ha
                 exact add_nonneg (mul_nonneg expEps_pos.le (Real.exp_pos _).le)
                   (mul_nonneg (Real.exp_pos _).le (by linarith))
  | log a iha => intro hwd; simp only [WD] at hwd; simp only [evalErrBnd]
                 have ha := iha hwd.1
                 have hden : 0 < toReal (evalF a σ) - evalErrBnd a σ := by linarith [hwd.2]
                 exact add_nonneg (mul_nonneg logEps_pos.le (abs_nonneg _)) (div_nonneg ha hden.le)
  | relu a iha => intro hwd; simp only [WD] at hwd; simp only [evalErrBnd]; exact iha hwd.1
  | max a b iha ihb => intro hwd; simp only [WD] at hwd; simp only [evalErrBnd]
                       exact add_nonneg (iha hwd.1) (ihb hwd.2.1)
  | min a b iha ihb => intro hwd; simp only [WD] at hwd; simp only [evalErrBnd]
                       exact add_nonneg (iha hwd.1) (ihb hwd.2.1)

/-- **Value error.** The `Float` evaluation tracks the exact real evaluation (given `WD`). -/
theorem evalF_error (e : Expr) (σ : Nat → Float) :
    WD e σ → |toReal (evalF e σ) - evalR e (envR σ)| ≤ evalErrBnd e σ := by
  induction e with
  | var i => intro _; simp [evalF, evalR, envR, evalErrBnd]
  | const c => intro _; simp [evalF, evalR, evalErrBnd]
  | add a b iha ihb => intro hwd; simp only [WD] at hwd
                       simpa only [evalF, evalR, evalErrBnd] using
                         addApprox_error (evalF a σ) (evalF b σ) _ _ _ _ (iha hwd.1) (ihb hwd.2)
  | sub a b iha ihb => intro hwd; simp only [WD] at hwd
                       simpa only [evalF, evalR, evalErrBnd] using
                         subApprox_error (evalF a σ) (evalF b σ) _ _ _ _ (iha hwd.1) (ihb hwd.2)
  | mul a b iha ihb => intro hwd; simp only [WD] at hwd
                       simpa only [evalF, evalR, evalErrBnd] using
                         mulApprox_error (evalF a σ) (evalF b σ) _ _ _ _ (iha hwd.1) (ihb hwd.2)
  | scale c a iha => intro hwd
                     have h := mulApprox_error c (evalF a σ) (toReal c) (evalR a (envR σ)) 0 (evalErrBnd a σ)
                       (by simp) (iha hwd)
                     simpa only [evalF, evalR, evalErrBnd, mul_zero, add_zero] using h
  | exp a iha => intro hwd
                 simpa only [evalF, evalR, evalErrBnd] using
                   expApprox_error (evalF a σ) (evalR a (envR σ)) (evalErrBnd a σ) (iha hwd)
  | log a iha => intro hwd; simp only [WD] at hwd
                 obtain ⟨ha, hlo⟩ := hwd
                 have hpos : 0 < toReal (evalF a σ) := lt_of_le_of_lt (evalErrBnd_nonneg a σ ha) hlo
                 simpa only [evalF, evalR, evalErrBnd] using
                   logApprox_error (evalF a σ) (evalR a (envR σ)) (evalErrBnd a σ) (iha ha) hpos hlo
  | relu a iha => intro hwd; simp only [WD] at hwd
                  simp only [evalF, evalR, evalErrBnd, toReal_reluF]
                  -- `|max x 0 − max y 0| ≤ max |x−y| |0| = |x−y| ≤ evalErrBnd a`.
                  calc |max (toReal (evalF a σ)) 0 - max (evalR a (envR σ)) 0|
                      ≤ max |toReal (evalF a σ) - evalR a (envR σ)| |(0 : ℝ) - 0| :=
                        abs_max_sub_max_le_max _ _ _ _
                    _ = |toReal (evalF a σ) - evalR a (envR σ)| := by simp
                    _ ≤ evalErrBnd a σ := iha hwd.1
  | max a b iha ihb =>
      intro hwd; simp only [WD] at hwd
      obtain ⟨ha, hb, _⟩ := hwd
      simp only [evalF, evalR, evalErrBnd, toReal_max]
      calc |max (toReal (evalF a σ)) (toReal (evalF b σ)) - max (evalR a (envR σ)) (evalR b (envR σ))|
          ≤ max |toReal (evalF a σ) - evalR a (envR σ)| |toReal (evalF b σ) - evalR b (envR σ)| :=
            abs_max_sub_max_le_max _ _ _ _
        _ ≤ evalErrBnd a σ + evalErrBnd b σ :=
            max_le (le_trans (iha ha) (le_add_of_nonneg_right (evalErrBnd_nonneg b σ hb)))
              (le_trans (ihb hb) (le_add_of_nonneg_left (evalErrBnd_nonneg a σ ha)))
  | min a b iha ihb =>
      intro hwd; simp only [WD] at hwd
      obtain ⟨ha, hb, _⟩ := hwd
      simp only [evalF, evalR, evalErrBnd, toReal_min]
      calc |min (toReal (evalF a σ)) (toReal (evalF b σ)) - min (evalR a (envR σ)) (evalR b (envR σ))|
          ≤ max |toReal (evalF a σ) - evalR a (envR σ)| |toReal (evalF b σ) - evalR b (envR σ)| :=
            abs_min_sub_min_le_max _ _ _ _
        _ ≤ evalErrBnd a σ + evalErrBnd b σ :=
            max_le (le_trans (iha ha) (le_add_of_nonneg_right (evalErrBnd_nonneg b σ hb)))
              (le_trans (ihb hb) (le_add_of_nonneg_left (evalErrBnd_nonneg a σ ha)))

/-- Computable-shape (ℝ) error bound on the `Float` forward-derivative `dF e σ k`. -/
noncomputable def derivErrBnd : Expr → (Nat → Float) → Nat → ℝ
  | .var _, _, _ => 0
  | .const _, _, _ => 0
  | .add a b, σ, k => u64 * |toReal (dF a σ k) + toReal (dF b σ k)| + derivErrBnd a σ k + derivErrBnd b σ k
  | .sub a b, σ, k => u64 * |toReal (dF a σ k) - toReal (dF b σ k)| + derivErrBnd a σ k + derivErrBnd b σ k
  | .mul a b, σ, k =>
      u64 * |toReal (dF a σ k * evalF b σ) + toReal (evalF a σ * dF b σ k)|
      + (u64 * |toReal (dF a σ k) * toReal (evalF b σ)|
          + |toReal (dF a σ k)| * evalErrBnd b σ + |evalR b (envR σ)| * derivErrBnd a σ k)
      + (u64 * |toReal (evalF a σ) * toReal (dF b σ k)|
          + |toReal (evalF a σ)| * derivErrBnd b σ k + |derivR b (envR σ) k| * evalErrBnd a σ)
  | .scale c a, σ, k => u64 * |toReal c * toReal (dF a σ k)| + |toReal c| * derivErrBnd a σ k
  | .exp a, σ, k =>
      u64 * |toReal (Float.exp (evalF a σ)) * toReal (dF a σ k)|
      + |toReal (Float.exp (evalF a σ))| * derivErrBnd a σ k
      + |derivR a (envR σ) k| * evalErrBnd (.exp a) σ
  | .log a, σ, k =>
      u64 * |toReal (dF a σ k) / toReal (evalF a σ)|
      + (derivErrBnd a σ k + |derivR a (envR σ) k / evalR a (envR σ)| * evalErrBnd a σ)
          / (toReal (evalF a σ) - evalErrBnd a σ)
  | .relu a, σ, k => derivErrBnd a σ k
  | .max a b, σ, k => if evalF a σ ≤ evalF b σ then derivErrBnd b σ k else derivErrBnd a σ k
  | .min a b, σ, k => if evalF a σ ≤ evalF b σ then derivErrBnd a σ k else derivErrBnd b σ k

theorem derivErrBnd_nonneg (e : Expr) (σ : Nat → Float) (k : Nat) :
    WD e σ → 0 ≤ derivErrBnd e σ k := by
  induction e with
  | var i => intro _; simp [derivErrBnd]
  | const c => intro _; simp [derivErrBnd]
  | add a b iha ihb => intro hwd; simp only [WD] at hwd; simp only [derivErrBnd]
                       exact add_nonneg (add_nonneg (mul_nonneg u64_pos.le (abs_nonneg _)) (iha hwd.1)) (ihb hwd.2)
  | sub a b iha ihb => intro hwd; simp only [WD] at hwd; simp only [derivErrBnd]
                       exact add_nonneg (add_nonneg (mul_nonneg u64_pos.le (abs_nonneg _)) (iha hwd.1)) (ihb hwd.2)
  | mul a b iha ihb => intro hwd; simp only [WD] at hwd; simp only [derivErrBnd]
                       refine add_nonneg (add_nonneg (mul_nonneg u64_pos.le (abs_nonneg _)) ?_) ?_
                       · exact add_nonneg (add_nonneg (mul_nonneg u64_pos.le (abs_nonneg _))
                           (mul_nonneg (abs_nonneg _) (evalErrBnd_nonneg b σ hwd.2))) (mul_nonneg (abs_nonneg _) (iha hwd.1))
                       · exact add_nonneg (add_nonneg (mul_nonneg u64_pos.le (abs_nonneg _))
                           (mul_nonneg (abs_nonneg _) (ihb hwd.2))) (mul_nonneg (abs_nonneg _) (evalErrBnd_nonneg a σ hwd.1))
  | scale c a iha => intro hwd; simp only [derivErrBnd]
                     exact add_nonneg (mul_nonneg u64_pos.le (abs_nonneg _)) (mul_nonneg (abs_nonneg _) (iha hwd))
  | exp a iha => intro hwd; simp only [derivErrBnd]
                 refine add_nonneg (add_nonneg (mul_nonneg u64_pos.le (abs_nonneg _))
                   (mul_nonneg (abs_nonneg _) (iha hwd))) (mul_nonneg (abs_nonneg _) ?_)
                 exact evalErrBnd_nonneg (.exp a) σ hwd
  | log a iha => intro hwd; simp only [WD] at hwd; simp only [derivErrBnd]
                 have hden : 0 < toReal (evalF a σ) - evalErrBnd a σ := by linarith [hwd.2]
                 refine add_nonneg (mul_nonneg u64_pos.le (abs_nonneg _)) (div_nonneg ?_ hden.le)
                 exact add_nonneg (iha hwd.1) (mul_nonneg (abs_nonneg _) (evalErrBnd_nonneg a σ hwd.1))
  | relu a iha => intro hwd; simp only [WD] at hwd; simp only [derivErrBnd]; exact iha hwd.1
  | max a b iha ihb => intro hwd; simp only [WD] at hwd; obtain ⟨ha, hb, _⟩ := hwd
                       simp only [derivErrBnd]; split_ifs
                       · exact ihb hb
                       · exact iha ha
  | min a b iha ihb => intro hwd; simp only [WD] at hwd; obtain ⟨ha, hb, _⟩ := hwd
                       simp only [derivErrBnd]; split_ifs
                       · exact iha ha
                       · exact ihb hb

/-- **Gradient error (per component).** The `Float` forward-derivative `dF e σ k` tracks the
    exact real derivative `derivR e (envR σ) k = ∂(evalR e)/∂(var k)` within `derivErrBnd`
    (given `WD`). -/
theorem dF_error (e : Expr) (σ : Nat → Float) (k : Nat) :
    WD e σ → |toReal (dF e σ k) - derivR e (envR σ) k| ≤ derivErrBnd e σ k := by
  induction e with
  | var i => intro _; by_cases h : i = k <;> simp [dF, derivR, derivErrBnd, h]
  | const c => intro _; simp [dF, derivR, derivErrBnd]
  | add a b iha ihb => intro hwd; simp only [WD] at hwd
                       simpa only [dF, derivR, derivErrBnd] using
                         addApprox_error (dF a σ k) (dF b σ k) _ _ _ _ (iha hwd.1) (ihb hwd.2)
  | sub a b iha ihb => intro hwd; simp only [WD] at hwd
                       simpa only [dF, derivR, derivErrBnd] using
                         subApprox_error (dF a σ k) (dF b σ k) _ _ _ _ (iha hwd.1) (ihb hwd.2)
  | mul a b iha ihb =>
      intro hwd; simp only [WD] at hwd
      have t1 := mulApprox_error (dF a σ k) (evalF b σ) (derivR a (envR σ) k) (evalR b (envR σ))
        (derivErrBnd a σ k) (evalErrBnd b σ) (iha hwd.1) (evalF_error b σ hwd.2)
      have t2 := mulApprox_error (evalF a σ) (dF b σ k) (evalR a (envR σ)) (derivR b (envR σ) k)
        (evalErrBnd a σ) (derivErrBnd b σ k) (evalF_error a σ hwd.1) (ihb hwd.2)
      have h := addApprox_error (dF a σ k * evalF b σ) (evalF a σ * dF b σ k)
        (derivR a (envR σ) k * evalR b (envR σ)) (evalR a (envR σ) * derivR b (envR σ) k) _ _ t1 t2
      simpa only [dF, derivR, derivErrBnd] using h
  | scale c a iha => intro hwd
                     have h := mulApprox_error c (dF a σ k) (toReal c) (derivR a (envR σ) k) 0 (derivErrBnd a σ k)
                       (by simp) (iha hwd)
                     simpa only [dF, derivR, derivErrBnd, mul_zero, add_zero] using h
  | exp a iha =>
      intro hwd
      have hval := evalF_error (Expr.exp a) σ hwd
      simp only [evalF, evalR] at hval
      have h := mulApprox_error (Float.exp (evalF a σ)) (dF a σ k)
        (Real.exp (evalR a (envR σ))) (derivR a (envR σ) k)
        (evalErrBnd (Expr.exp a) σ) (derivErrBnd a σ k) hval (iha hwd)
      simpa only [dF, derivR, derivErrBnd] using h
  | log a iha =>
      intro hwd; simp only [WD] at hwd
      obtain ⟨ha, hlo⟩ := hwd
      have hea : (0 : ℝ) ≤ evalErrBnd a σ := evalErrBnd_nonneg a σ ha
      have hpos : 0 < toReal (evalF a σ) := lt_of_le_of_lt hea hlo
      have hval := evalF_error a σ ha
      have hdmin : 0 < toReal (evalF a σ) - evalErrBnd a σ := by linarith
      have hdy : toReal (evalF a σ) - evalErrBnd a σ ≤ |toReal (evalF a σ)| := by
        rw [abs_of_pos hpos]; linarith
      have hyR : evalR a (envR σ) ≠ 0 := by
        have hle : toReal (evalF a σ) - evalErrBnd a σ ≤ evalR a (envR σ) := by
          have := (abs_le.mp hval).2; linarith
        exact ne_of_gt (lt_of_lt_of_le hdmin hle)
      have h := divApprox_error (dF a σ k) (evalF a σ) (derivR a (envR σ) k) (evalR a (envR σ))
        (derivErrBnd a σ k) (evalErrBnd a σ) (toReal (evalF a σ) - evalErrBnd a σ)
        (iha ha) hval hdmin hdy hyR
      simpa only [dF, derivR, derivErrBnd] using h
  | relu a iha =>
      intro hwd; simp only [WD] at hwd
      obtain ⟨ha, hlo⟩ := hwd
      have hea : (0 : ℝ) ≤ evalErrBnd a σ := evalErrBnd_nonneg a σ ha
      have hval := evalF_error a σ ha        -- |toReal(evalF a) − evalR a| ≤ evalErrBnd a
      simp only [dF, derivR, derivErrBnd]
      -- Split on `reluF`'s executable branch; `toReal_reluF` ties it to the real sign of the
      -- argument, and `hlo : evalErrBnd a < |toReal(evalF a)|` keeps the two sides in agreement.
      by_cases h : evalF a σ < 0.0
      · -- Below the kink: `Float` subgradient is `0`; the real argument is also `< 0`.
        rw [if_pos h]
        -- From `toReal_reluF`: `x < 0.0 → toReal x ≤ 0`.
        have hxle : toReal (evalF a σ) ≤ 0 := by
          have hr := toReal_reluF (evalF a σ); unfold reluF at hr
          rw [if_pos h, toReal_zeroLit] at hr
          exact le_of_max_le_left hr.symm.le
        have hxne : toReal (evalF a σ) ≠ 0 := by
          intro hz; rw [hz, abs_zero] at hlo; exact absurd hlo (not_lt.mpr hea)
        have hxlt : toReal (evalF a σ) < 0 := lt_of_le_of_ne hxle hxne
        have habs : |toReal (evalF a σ)| = -toReal (evalF a σ) := abs_of_neg hxlt
        have hRlt : evalR a (envR σ) < 0 := by
          have h1 := (abs_le.mp hval).1  -- -(evalErrBnd a) ≤ toReal(evalF a) − evalR a
          rw [habs] at hlo; linarith only [h1, hlo]
        rw [if_neg (not_lt.mpr hRlt.le), toReal_zero]
        simpa using derivErrBnd_nonneg a σ k ha
      · -- Above the kink: `Float` subgradient is `dF a`; the real argument is also `> 0`.
        rw [if_neg h]
        -- From `toReal_reluF`: `¬ (x < 0.0) → 0 ≤ toReal x`.
        have hxge : 0 ≤ toReal (evalF a σ) := by
          have hr := toReal_reluF (evalF a σ); unfold reluF at hr
          rw [if_neg h] at hr
          calc (0:ℝ) ≤ max (toReal (evalF a σ)) 0 := le_max_right _ _
            _ = toReal (evalF a σ) := hr.symm
        have hxne : toReal (evalF a σ) ≠ 0 := by
          intro hz; rw [hz, abs_zero] at hlo; exact absurd hlo (not_lt.mpr hea)
        have hxgt : 0 < toReal (evalF a σ) := lt_of_le_of_ne hxge (Ne.symm hxne)
        have habs : |toReal (evalF a σ)| = toReal (evalF a σ) := abs_of_pos hxgt
        have hRgt : 0 < evalR a (envR σ) := by
          have h2 := (abs_le.mp hval).2  -- toReal(evalF a) − evalR a ≤ evalErrBnd a
          rw [habs] at hlo; linarith only [h2, hlo]
        rw [if_pos hRgt]
        exact iha ha
  | max a b iha ihb =>
      intro hwd; simp only [WD] at hwd
      obtain ⟨ha, hb, hgap⟩ := hwd
      obtain ⟨hva1, hva2⟩ := abs_le.mp (evalF_error a σ ha)   -- -εa ≤ da−ra, da−ra ≤ εa
      obtain ⟨hvb1, hvb2⟩ := abs_le.mp (evalF_error b σ hb)
      simp only [dF, derivR, derivErrBnd]
      by_cases h : evalF a σ ≤ evalF b σ
      · -- Float selects `b`; the gap forces the real order `evalR a ≤ evalR b` (also selects `b`).
        have hdab := le_of_float_le h
        have habs : |toReal (evalF a σ) - toReal (evalF b σ)| = toReal (evalF b σ) - toReal (evalF a σ) := by
          rw [abs_of_nonpos (by linarith only [hdab])]; ring
        rw [habs] at hgap
        have hR : evalR a (envR σ) ≤ evalR b (envR σ) := by linarith only [hva1, hva2, hvb1, hvb2, hgap]
        simp only [if_pos h, if_pos hR]; exact ihb hb
      · -- Float selects `a`; the gap forces `¬ evalR a ≤ evalR b` (also selects `a`).
        have hdba := le_of_not_float_le h
        have habs : |toReal (evalF a σ) - toReal (evalF b σ)| = toReal (evalF a σ) - toReal (evalF b σ) := by
          rw [abs_of_nonneg (by linarith only [hdba])]
        rw [habs] at hgap
        have hR : ¬ evalR a (envR σ) ≤ evalR b (envR σ) := by
          rw [not_le]; linarith only [hva1, hva2, hvb1, hvb2, hgap]
        simp only [if_neg h, if_neg hR]; exact iha ha
  | min a b iha ihb =>
      intro hwd; simp only [WD] at hwd
      obtain ⟨ha, hb, hgap⟩ := hwd
      obtain ⟨hva1, hva2⟩ := abs_le.mp (evalF_error a σ ha)
      obtain ⟨hvb1, hvb2⟩ := abs_le.mp (evalF_error b σ hb)
      simp only [dF, derivR, derivErrBnd]
      by_cases h : evalF a σ ≤ evalF b σ
      · -- Float selects `a`; the gap forces `evalR a ≤ evalR b` (also selects `a`).
        have hdab := le_of_float_le h
        have habs : |toReal (evalF a σ) - toReal (evalF b σ)| = toReal (evalF b σ) - toReal (evalF a σ) := by
          rw [abs_of_nonpos (by linarith only [hdab])]; ring
        rw [habs] at hgap
        have hR : evalR a (envR σ) ≤ evalR b (envR σ) := by linarith only [hva1, hva2, hvb1, hvb2, hgap]
        simp only [if_pos h, if_pos hR]; exact iha ha
      · -- Float selects `b`; the gap forces `¬ evalR a ≤ evalR b` (also selects `b`).
        have hdba := le_of_not_float_le h
        have habs : |toReal (evalF a σ) - toReal (evalF b σ)| = toReal (evalF a σ) - toReal (evalF b σ) := by
          rw [abs_of_nonneg (by linarith only [hdba])]
        rw [habs] at hgap
        have hR : ¬ evalR a (envR σ) ≤ evalR b (envR σ) := by
          rw [not_le]; linarith only [hva1, hva2, hvb1, hvb2, hgap]
        simp only [if_neg h, if_neg hR]; exact ihb hb

end Puffer.FloatR.ADR
