/-
The Float ↔ ℝ bridge — the trusted numerical foundation.

To make a *runnable* Lean trainer whose accuracy is *proven*, we compute natively
in Lean's `Float` (IEEE-754 binary64, executed by hardware) and reason about the
results in ℝ. Lean/Mathlib has no formal model of `Float`, so we take the standard
verified-numerics stance: an embedding `toReal : Float → ℝ` together with the
**(1+δ) round-to-nearest model** as AXIOMS — the small, explicit trusted base.

    fl(a op b) = (a op b)·(1 + δ),   |δ| ≤ u = 2⁻⁵³      (op ∈ {+, −, ×})

These hold for IEEE binary64 round-to-nearest in the absence of overflow/underflow
(subnormals carry an extra absolute term we currently fold into this assumption;
documented, and replaceable by a full Flocq-style model later). Everything else in
`Puffer/Float/*` is *proved* from these axioms; nothing else is trusted.
-/
import Mathlib
import Puffer.Float.Exec
import Puffer.Float.Fma

namespace Puffer.FloatR

/-- Unit roundoff of IEEE-754 binary64 (round-to-nearest): `2⁻⁵³`. -/
noncomputable def u64 : ℝ := (2 : ℝ) ^ (-53 : ℤ)

theorem u64_pos : 0 < u64 := by unfold u64; positivity
theorem u64_lt_one : u64 < 1 := by unfold u64; norm_num

/-- The trusted embedding of the executable `Float` into ℝ. -/
axiom toReal : Float → ℝ

@[simp] axiom toReal_zero : toReal 0 = 0
@[simp] axiom toReal_one : toReal 1 = 1
/-- The `OfScientific` zero literal `0.0` embeds to `0`, like `toReal_zero` for the `OfNat`
    numeral. (Needed because `(0.0 : Float)` is not defeq to `(0 : Float)` — Float has no
    `DecidableEq` — yet code like `matmul`'s accumulator seeds with the `0.0` literal.) -/
@[simp] axiom toReal_zeroLit : toReal (0.0 : Float) = 0
/-- The `OfScientific` one literal `1.0` embeds to `1`, like `toReal_one` for the `OfNat` numeral
    (`1.0` is exactly representable in binary64, so this is exact — same trusted-model spirit as
    `toReal_zeroLit`). Needed e.g. by the reverse-mode adjoint seed, which uses the `1.0` literal. -/
@[simp] axiom toReal_oneLit : toReal (1.0 : Float) = 1

/-- (1+δ) model for hardware addition. -/
axiom add_model (a b : Float) :
    ∃ δ : ℝ, |δ| ≤ u64 ∧ toReal (a + b) = (toReal a + toReal b) * (1 + δ)

/-- (1+δ) model for hardware subtraction. -/
axiom sub_model (a b : Float) :
    ∃ δ : ℝ, |δ| ≤ u64 ∧ toReal (a - b) = (toReal a - toReal b) * (1 + δ)

/-- (1+δ) model for hardware multiplication. -/
axiom mul_model (a b : Float) :
    ∃ δ : ℝ, |δ| ≤ u64 ∧ toReal (a * b) = (toReal a * toReal b) * (1 + δ)

/-- (1+δ) model for a correctly-rounded **fused multiply-add** — a SINGLE rounding
    of the exact `a*b + c` (contrast `mul_model` then `add_model`, which rounds
    twice). `Puffer.Float.fma` (`Puffer/Float/Fma.lean`) realizes this in pure Lean
    via error-free transforms, validated bit-for-bit against the hardware `fma`
    over 2M triples. This is the FMA-aware extension of the trusted base: matching
    glibc's `-mfma` FMA contraction is exactly what makes `Puffer.Numeric.SinCosF`
    bit-exact vs the system `sinf`/`cosf` over `|x| ≤ 120`. -/
axiom fma_model (a b c : Float) :
    ∃ δ : ℝ, |δ| ≤ u64 ∧
      toReal (Puffer.Float.fma a b c) = (toReal a * toReal b + toReal c) * (1 + δ)

/-- Assumed relative accuracy of `Float.exp` (≈ 2 ulp — a realistic libm bound). -/
noncomputable def expEps : ℝ := (2 : ℝ) ^ (-52 : ℤ)

theorem expEps_pos : 0 < expEps := by unfold expEps; positivity

/-- (1+δ) model for `Float.exp`: it returns `e^{toReal a}` up to relative `expEps`. -/
axiom exp_model (a : Float) :
    ∃ δ : ℝ, |δ| ≤ expEps ∧ toReal (Float.exp a) = Real.exp (toReal a) * (1 + δ)

/-- Relative-error bound for `Float.log` (≈2 ulp, `= 2·u64`; same magnitude as `expEps`). -/
noncomputable def logEps : ℝ := (2 : ℝ) ^ (-52 : ℤ)

theorem logEps_pos : 0 < logEps := by unfold logEps; positivity

/-- (1+δ) model for `Float.log` (argument assumed `> 0`; `log` of `≤ 0` — NaN — is outside
    this trusted model). Needed for every policy objective (log-probabilities, `logSumExp`,
    entropy), which were previously unstatable in the trusted base. CAVEAT: a *pure relative*
    model is optimistic near `toReal a = 1`, where `Real.log → 0` and this bound forces the
    absolute error to vanish (correctly-rounded `log` has a small absolute floor there); unlike
    `exp`/`sqrt`, whose in-domain outputs never cross 0. When `log` is actually consumed, switch
    to a mixed `logEps·|log a| + logAbsEps` model to cover the near-1 region. -/
axiom log_model (a : Float) (ha : 0 < toReal a) :
    ∃ δ : ℝ, |δ| ≤ logEps ∧ toReal (Float.log a) = Real.log (toReal a) * (1 + δ)

/-- (1+δ) model for hardware division (no division by zero, in the trusted model). -/
axiom div_model (a b : Float) :
    ∃ δ : ℝ, |δ| ≤ u64 ∧ toReal (a / b) = (toReal a / toReal b) * (1 + δ)

/-- (1+δ) model for `Float.sqrt` (correctly rounded; argument assumed `≥ 0`). -/
axiom sqrt_model (a : Float) :
    ∃ δ : ℝ, |δ| ≤ u64 ∧ toReal (Float.sqrt a) = Real.sqrt (toReal a) * (1 + δ)

/-- Float `min`/`max` are exact (order-preserving embedding; NaN/±0 edge cases,
    like overflow for `±×`, are outside this trusted model). -/
axiom toReal_min (a b : Float) : toReal (min a b) = min (toReal a) (toReal b)
axiom toReal_max (a b : Float) : toReal (max a b) = max (toReal a) (toReal b)

/-- IEEE-754 negation is exact (sign-bit flip, no rounding) — same exact-embedding spirit as
    `toReal_min`/`toReal_max`. -/
@[simp] axiom toReal_neg (a : Float) : toReal (-a) = - toReal a

/-- **Decimal literals round to nearest.** A Float `OfScientific` literal (e.g. `4.0848`) embeds
    within relative `u64` of the exact real number the SAME decimal literal denotes in ℝ. This is
    the literal analogue of the `(1+δ)` op-models (`add_model` etc.): IEEE-754 decimal→binary64
    literal conversion is correctly rounded. (Overflow/subnormal literals are outside this trusted
    model, as elsewhere.) Lets the `toReal` of the Float Muon coefficients be pinned to their exact
    ℝ values up to rounding, closing the coefficient-rounding gap in the Newton–Schulz norm bound. -/
axiom toReal_ofScientific_close (m : Nat) (s : Bool) (e : Nat) :
    |toReal (OfScientific.ofScientific m s e : Float) - (OfScientific.ofScientific m s e : ℝ)|
      ≤ u64 * |(OfScientific.ofScientific m s e : ℝ)|

/-- `reluF` (the executable `if x < 0.0 then 0.0 else x`) is EXACT: it introduces no
    rounding (it returns either the `0.0` literal or its input verbatim), so its real
    value is `max (toReal x) 0`. Same spirit as `toReal_min`/`toReal_max` (an exact order
    fact, strictly weaker than the (1+δ) models — no quantitative claim), and it inherits
    the same caveat: NaN/±0/overflow are outside this trusted model. NOTE: this is an
    INDEPENDENT axiom, not a corollary of `toReal_zero` — `reluF`'s `0.0` is the
    `OfScientific` literal, not defeq to the `OfNat` numeral `0`, and `reluF` branches on
    Float `<`, not on `max`; so it also postulates `toReal 0.0 = 0` and the branch's
    agreement with the real order. -/
axiom toReal_reluF (x : Float) : toReal (reluF x) = max (toReal x) 0

/-- Float bit-extraction is injective: bit-identical `Float`s are equal. Trusted (a fact about the IEEE
    representation the FFI exposes, like `toReal`; `Float` is an opaque extern type, so this is not
    provable in Lean). SOUND — distinct bit patterns are distinct `Float`s (`+0.0`/`-0.0`, NaN payloads all
    differ in bits, and this axiom never equates them); equal bits genuinely mean the same `Float`. Enables a
    decidable, sound `Float` equality (via `UInt64`'s `DecidableEq`) where propositional `=` is otherwise
    undecidable and IEEE `==` is unlawful. -/
axiom toBits_inj : Function.Injective Float.toBits

/-- Float `≤` respects the real order (DERIVED from `toReal_min`, no new axiom): the `min`
    instance is `if a ≤ b then a else b`, so `a ≤ b` makes `min a b = a`, and `toReal_min`
    turns that into `toReal a = min (toReal a) (toReal b) ≤ toReal b`. -/
theorem le_of_float_le {a b : Float} (h : a ≤ b) : toReal a ≤ toReal b := by
  have hmin : min a b = a := by rw [show min a b = if a ≤ b then a else b from rfl, if_pos h]
  have := toReal_min a b
  rw [hmin] at this
  calc toReal a = min (toReal a) (toReal b) := this
    _ ≤ toReal b := min_le_right _ _

/-- The other direction, via `toReal_max` (DERIVED, no new axiom): `¬ (a ≤ b)` makes
    `max a b = a`, so `toReal a = max (toReal a) (toReal b) ≥ toReal b`. Together with
    `le_of_float_le` this pins the Float branch's real order in either case. -/
theorem le_of_not_float_le {a b : Float} (h : ¬ a ≤ b) : toReal b ≤ toReal a := by
  have hmax : max a b = a := by rw [show max a b = if a ≤ b then b else a from rfl, if_neg h]
  have := toReal_max a b
  rw [hmax] at this
  calc toReal b ≤ max (toReal a) (toReal b) := le_max_right _ _
    _ = toReal a := this.symm

/-! ### Derived rounding-error bounds (proved from the axioms) -/

/-- Absolute error of a rounded operation given its (1+δ) form. -/
private theorem err_of_model {r : ℝ} {fr : ℝ} (δ : ℝ) (hδ : |δ| ≤ u64)
    (he : fr = r * (1 + δ)) : |fr - r| ≤ u64 * |r| := by
  rw [he]
  have hr : r * (1 + δ) - r = r * δ := by ring
  rw [hr, abs_mul]
  calc |r| * |δ| ≤ |r| * u64 := mul_le_mul_of_nonneg_left hδ (abs_nonneg r)
    _ = u64 * |r| := mul_comm _ _

/-- **Addition error**: `|fl(a+b) − (a+b)| ≤ u·|a+b|`. -/
theorem add_error (a b : Float) :
    |toReal (a + b) - (toReal a + toReal b)| ≤ u64 * |toReal a + toReal b| := by
  obtain ⟨δ, hδ, he⟩ := add_model a b; exact err_of_model δ hδ he

/-- **Subtraction error**: `|fl(a−b) − (a−b)| ≤ u·|a−b|`. -/
theorem sub_error (a b : Float) :
    |toReal (a - b) - (toReal a - toReal b)| ≤ u64 * |toReal a - toReal b| := by
  obtain ⟨δ, hδ, he⟩ := sub_model a b; exact err_of_model δ hδ he

/-- **Multiplication error**: `|fl(a·b) − a·b| ≤ u·|a·b|`. -/
theorem mul_error (a b : Float) :
    |toReal (a * b) - toReal a * toReal b| ≤ u64 * |toReal a * toReal b| := by
  obtain ⟨δ, hδ, he⟩ := mul_model a b; exact err_of_model δ hδ he

/-- **Exp error**: `|fl(exp a) − e^{toReal a}| ≤ expEps · e^{toReal a}`. -/
theorem exp_error (a : Float) :
    |toReal (Float.exp a) - Real.exp (toReal a)| ≤ expEps * Real.exp (toReal a) := by
  obtain ⟨δ, hδ, he⟩ := exp_model a
  rw [he]
  have hr : Real.exp (toReal a) * (1 + δ) - Real.exp (toReal a) = Real.exp (toReal a) * δ := by ring
  rw [hr, abs_mul, abs_of_pos (Real.exp_pos _)]
  calc Real.exp (toReal a) * |δ|
      ≤ Real.exp (toReal a) * expEps := mul_le_mul_of_nonneg_left hδ (Real.exp_pos _).le
    _ = expEps * Real.exp (toReal a) := mul_comm _ _

/-- **Log error**: `|fl(log a) − log(toReal a)| ≤ logEps · |log(toReal a)|` (for `toReal a > 0`). -/
theorem log_error (a : Float) (ha : 0 < toReal a) :
    |toReal (Float.log a) - Real.log (toReal a)| ≤ logEps * |Real.log (toReal a)| := by
  obtain ⟨δ, hδ, he⟩ := log_model a ha
  rw [he]
  have hr : Real.log (toReal a) * (1 + δ) - Real.log (toReal a) = Real.log (toReal a) * δ := by ring
  rw [hr, abs_mul]
  calc |Real.log (toReal a)| * |δ|
      ≤ |Real.log (toReal a)| * logEps := mul_le_mul_of_nonneg_left hδ (abs_nonneg _)
    _ = logEps * |Real.log (toReal a)| := mul_comm _ _

/-- **Division error**: `|fl(a/b) − a/b| ≤ u·|a/b|`. -/
theorem div_error (a b : Float) :
    |toReal (a / b) - toReal a / toReal b| ≤ u64 * |toReal a / toReal b| := by
  obtain ⟨δ, hδ, he⟩ := div_model a b; exact err_of_model δ hδ he

/-- **Sqrt error**: `|fl(√a) − √(toReal a)| ≤ u·√(toReal a)`. -/
theorem sqrt_error (a : Float) :
    |toReal (Float.sqrt a) - Real.sqrt (toReal a)| ≤ u64 * Real.sqrt (toReal a) := by
  obtain ⟨δ, hδ, he⟩ := sqrt_model a
  rw [he]
  have hr : Real.sqrt (toReal a) * (1 + δ) - Real.sqrt (toReal a) = Real.sqrt (toReal a) * δ := by ring
  rw [hr, abs_mul, abs_of_nonneg (Real.sqrt_nonneg _)]
  calc Real.sqrt (toReal a) * |δ|
      ≤ Real.sqrt (toReal a) * u64 := mul_le_mul_of_nonneg_left hδ (Real.sqrt_nonneg _)
    _ = u64 * Real.sqrt (toReal a) := mul_comm _ _

/-- **Product propagation.** `fl(x·y)` vs exact `xR·yR`, given input errors `εx,εy`:
    `|fl(x·y) − xR·yR| ≤ u·|x·y| + |x|·εy + |yR|·εx`. -/
theorem mulApprox_error (x y : Float) (xR yR εx εy : ℝ)
    (hx : |toReal x - xR| ≤ εx) (hy : |toReal y - yR| ≤ εy) :
    |toReal (x * y) - xR * yR| ≤ u64 * |toReal x * toReal y| + |toReal x| * εy + |yR| * εx := by
  have hsplit : toReal (x * y) - xR * yR
      = (toReal (x * y) - toReal x * toReal y)
        + (toReal x * (toReal y - yR) + yR * (toReal x - xR)) := by ring
  calc |toReal (x * y) - xR * yR|
      ≤ |toReal (x * y) - toReal x * toReal y|
          + |toReal x * (toReal y - yR) + yR * (toReal x - xR)| := by rw [hsplit]; exact abs_add_le _ _
    _ ≤ u64 * |toReal x * toReal y| + (|toReal x| * εy + |yR| * εx) := by
        refine add_le_add (mul_error x y) ((abs_add_le _ _).trans ?_)
        rw [abs_mul, abs_mul]
        exact add_le_add (mul_le_mul_of_nonneg_left hy (abs_nonneg _))
          (mul_le_mul_of_nonneg_left hx (abs_nonneg _))
    _ = u64 * |toReal x * toReal y| + |toReal x| * εy + |yR| * εx := by ring

/-- **Sum propagation.** `fl(x+y)` vs exact `xR+yR`, given input errors `εx,εy`:
    `|fl(x+y) − (xR+yR)| ≤ u·|x+y| + εx + εy`. -/
theorem addApprox_error (x y : Float) (xR yR εx εy : ℝ)
    (hx : |toReal x - xR| ≤ εx) (hy : |toReal y - yR| ≤ εy) :
    |toReal (x + y) - (xR + yR)| ≤ u64 * |toReal x + toReal y| + εx + εy := by
  have hsplit : toReal (x + y) - (xR + yR)
      = (toReal (x + y) - (toReal x + toReal y)) + (toReal x - xR) + (toReal y - yR) := by ring
  calc |toReal (x + y) - (xR + yR)|
      ≤ (|toReal (x + y) - (toReal x + toReal y)| + |toReal x - xR|) + |toReal y - yR| := by
          rw [hsplit]; exact (abs_add_le _ _).trans (add_le_add (abs_add_le _ _) le_rfl)
    _ ≤ u64 * |toReal x + toReal y| + εx + εy := add_le_add (add_le_add (add_error x y) hx) hy

/-- **Subtraction propagation.** `fl(x−y)` vs exact `xR−yR`, given input errors `εx,εy`:
    `|fl(x−y) − (xR−yR)| ≤ u·|x−y| + εx + εy`. -/
theorem subApprox_error (x y : Float) (xR yR εx εy : ℝ)
    (hx : |toReal x - xR| ≤ εx) (hy : |toReal y - yR| ≤ εy) :
    |toReal (x - y) - (xR - yR)| ≤ u64 * |toReal x - toReal y| + εx + εy := by
  have hsplit : toReal (x - y) - (xR - yR)
      = (toReal (x - y) - (toReal x - toReal y)) + (toReal x - xR) + (yR - toReal y) := by ring
  calc |toReal (x - y) - (xR - yR)|
      ≤ (|toReal (x - y) - (toReal x - toReal y)| + |toReal x - xR|) + |yR - toReal y| := by
          rw [hsplit]; exact (abs_add_le _ _).trans (add_le_add (abs_add_le _ _) le_rfl)
    _ ≤ u64 * |toReal x - toReal y| + εx + εy := by
        refine add_le_add (add_le_add (sub_error x y) hx) ?_
        rw [abs_sub_comm]; exact hy

/-- `|1 − e^d| ≤ e^ε − 1` when `|d| ≤ ε` (the local Lipschitz factor of `exp`). -/
theorem abs_one_sub_exp_le {d ε : ℝ} (h : |d| ≤ ε) : |1 - Real.exp d| ≤ Real.exp ε - 1 := by
  obtain ⟨hd2, hd1⟩ := abs_le.mp h
  rcases le_total 0 d with hd | hd
  · have h1 : (1 : ℝ) ≤ Real.exp d := by rw [← Real.exp_zero]; exact Real.exp_le_exp.mpr hd
    rw [abs_of_nonpos (by linarith)]
    have : Real.exp d ≤ Real.exp ε := Real.exp_le_exp.mpr hd1
    linarith
  · have h1 : Real.exp d ≤ 1 := by rw [← Real.exp_zero]; exact Real.exp_le_exp.mpr hd
    rw [abs_of_nonneg (by linarith)]
    have hge : Real.exp (-ε) ≤ Real.exp d := Real.exp_le_exp.mpr hd2
    have hcosh : 2 ≤ Real.exp ε + Real.exp (-ε) := by
      have := Real.one_le_cosh ε; rw [Real.cosh_eq] at this; linarith
    linarith

/-- **Exp propagation.** `fl(exp x)` vs exact `e^{xR}`, given input error `εx`:
    rounding `expEps·e^{toReal x}` plus the exp Lipschitz factor `e^{toReal x}·(e^{εx}−1)`. -/
theorem expApprox_error (x : Float) (xR εx : ℝ) (hx : |toReal x - xR| ≤ εx) :
    |toReal (Float.exp x) - Real.exp xR|
      ≤ expEps * Real.exp (toReal x) + Real.exp (toReal x) * (Real.exp εx - 1) := by
  have hprop : |Real.exp (toReal x) - Real.exp xR| ≤ Real.exp (toReal x) * (Real.exp εx - 1) := by
    have hd : |xR - toReal x| ≤ εx := by rw [abs_sub_comm]; exact hx
    have he : Real.exp xR = Real.exp (toReal x) * Real.exp (xR - toReal x) := by
      rw [← Real.exp_add]; ring_nf
    have heq : Real.exp (toReal x) - Real.exp xR
        = Real.exp (toReal x) * (1 - Real.exp (xR - toReal x)) := by rw [he]; ring
    rw [heq, abs_mul, abs_of_pos (Real.exp_pos _)]
    exact mul_le_mul_of_nonneg_left (abs_one_sub_exp_le hd) (Real.exp_pos _).le
  calc |toReal (Float.exp x) - Real.exp xR|
      ≤ |toReal (Float.exp x) - Real.exp (toReal x)| + |Real.exp (toReal x) - Real.exp xR| :=
        abs_sub_le _ _ _
    _ ≤ expEps * Real.exp (toReal x) + Real.exp (toReal x) * (Real.exp εx - 1) :=
        add_le_add (exp_error x) hprop

/-- Log is locally Lipschitz: `|log u − log v| ≤ |u − v| / min u v` for `u,v > 0`. -/
theorem abs_log_sub_le {u v : ℝ} (hu : 0 < u) (hv : 0 < v) :
    |Real.log u - Real.log v| ≤ |u - v| / min u v := by
  have hm : 0 < min u v := lt_min hu hv
  have key : ∀ p q : ℝ, 0 < p → 0 < q → Real.log p - Real.log q ≤ (p - q) / q := by
    intro p q hp hq
    have h := Real.log_le_sub_one_of_pos (div_pos hp hq)
    rw [Real.log_div hp.ne' hq.ne'] at h
    have e : p / q - 1 = (p - q) / q := by field_simp
    rw [e] at h; exact h
  have hup : Real.log u - Real.log v ≤ (u - v) / v := key u v hu hv
  have hdn : Real.log v - Real.log u ≤ (v - u) / u := key v u hv hu
  have bup : (u - v) / v ≤ |u - v| / min u v := by
    have h1 : (u - v) / v ≤ |u - v| / v := by gcongr; exact le_abs_self _
    have h2 : |u - v| / v ≤ |u - v| / min u v := by gcongr; exact min_le_right u v
    exact h1.trans h2
  have bdn : (v - u) / u ≤ |u - v| / min u v := by
    rw [abs_sub_comm]
    have h1 : (v - u) / u ≤ |v - u| / u := by gcongr; exact le_abs_self _
    have h2 : |v - u| / u ≤ |v - u| / min u v := by gcongr; exact min_le_left u v
    exact h1.trans h2
  rw [abs_le]; exact ⟨by linarith, by linarith⟩

/-- **Log propagation.** `fl(log x)` vs exact `log xR`, given input error `εx` with the
    argument bounded away from 0 (`εx < toReal x`): rounding `logEps·|log(toReal x)|` plus the
    log Lipschitz factor `εx / (toReal x − εx)`. -/
theorem logApprox_error (x : Float) (xR εx : ℝ) (hx : |toReal x - xR| ≤ εx)
    (hpos : 0 < toReal x) (hlo : εx < toReal x) :
    |toReal (Float.log x) - Real.log xR|
      ≤ logEps * |Real.log (toReal x)| + εx / (toReal x - εx) := by
  have hε : 0 ≤ εx := le_trans (abs_nonneg _) hx
  have hxR : 0 < xR := by have := (abs_le.mp hx).2; linarith
  have hprop : |Real.log (toReal x) - Real.log xR| ≤ εx / (toReal x - εx) := by
    refine (abs_log_sub_le hpos hxR).trans ?_
    have hden : 0 < toReal x - εx := by linarith
    have hminpos : 0 < min (toReal x) xR := lt_min hpos hxR
    have hmin : toReal x - εx ≤ min (toReal x) xR := by
      refine le_min (by linarith) ?_
      have := (abs_le.mp hx).2; linarith
    calc |toReal x - xR| / min (toReal x) xR
        ≤ εx / min (toReal x) xR := by gcongr
      _ ≤ εx / (toReal x - εx) := by gcongr
  calc |toReal (Float.log x) - Real.log xR|
      ≤ |toReal (Float.log x) - Real.log (toReal x)| + |Real.log (toReal x) - Real.log xR| :=
        abs_sub_le _ _ _
    _ ≤ logEps * |Real.log (toReal x)| + εx / (toReal x - εx) :=
        add_le_add (log_error x hpos) hprop

/-- **Division propagation.** `fl(x/y)` vs exact `xR/yR`, given input errors `εx,εy` and a
    denominator floor `dmin ≤ |toReal y|` (`yR ≠ 0`): the division rounding plus the quotient
    perturbation `(εx + |xR/yR|·εy)/dmin`. -/
theorem divApprox_error (x y : Float) (xR yR εx εy dmin : ℝ)
    (hx : |toReal x - xR| ≤ εx) (hy : |toReal y - yR| ≤ εy)
    (hdmin : 0 < dmin) (hdy : dmin ≤ |toReal y|) (hyR : yR ≠ 0) :
    |toReal (x / y) - xR / yR| ≤ u64 * |toReal x / toReal y| + (εx + |xR / yR| * εy) / dmin := by
  have hb0 : toReal y ≠ 0 := fun h => by rw [h, abs_zero] at hdy; linarith
  have hperturb : |toReal x / toReal y - xR / yR| ≤ (εx + |xR / yR| * εy) / dmin := by
    have hsplit : toReal x / toReal y - xR / yR
        = (toReal x - xR) / toReal y + xR / yR * ((yR - toReal y) / toReal y) := by
      field_simp; ring
    have hyb : |yR - toReal y| ≤ εy := by rw [abs_sub_comm]; exact hy
    have hby : 0 < |toReal y| := abs_pos.mpr hb0
    have hεy : 0 ≤ εy := le_trans (abs_nonneg _) hy
    have hinner : |(yR - toReal y) / toReal y| ≤ εy / dmin := by
      rw [abs_div, div_le_div_iff₀ hby hdmin]
      nlinarith [mul_le_mul_of_nonneg_right hyb hdmin.le, mul_le_mul_of_nonneg_left hdy hεy]
    have t1 : |(toReal x - xR) / toReal y| ≤ εx / dmin := by
      rw [abs_div]; gcongr; exact le_trans (abs_nonneg _) hx
    have t2 : |xR / yR * ((yR - toReal y) / toReal y)| ≤ |xR / yR| * εy / dmin := by
      rw [abs_mul]
      calc |xR / yR| * |(yR - toReal y) / toReal y|
          ≤ |xR / yR| * (εy / dmin) := mul_le_mul_of_nonneg_left hinner (abs_nonneg _)
        _ = |xR / yR| * εy / dmin := by ring
    calc |toReal x / toReal y - xR / yR|
        ≤ |(toReal x - xR) / toReal y| + |xR / yR * ((yR - toReal y) / toReal y)| := by
          rw [hsplit]; exact abs_add_le _ _
      _ ≤ εx / dmin + |xR / yR| * εy / dmin := add_le_add t1 t2
      _ = (εx + |xR / yR| * εy) / dmin := by ring
  calc |toReal (x / y) - xR / yR|
      ≤ |toReal (x / y) - toReal x / toReal y| + |toReal x / toReal y - xR / yR| := abs_sub_le _ _ _
    _ ≤ u64 * |toReal x / toReal y| + (εx + |xR / yR| * εy) / dmin := add_le_add (div_error x y) hperturb

/-- `|√a − √b| ≤ √|a − b|` for `a,b ≥ 0` (the local Hölder-½ bound for `Real.sqrt`). -/
theorem abs_sqrt_sub_sqrt_le {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) :
    |Real.sqrt a - Real.sqrt b| ≤ Real.sqrt |a - b| := by
  have key : ∀ x y : ℝ, 0 ≤ y → y ≤ x → Real.sqrt x - Real.sqrt y ≤ Real.sqrt (x - y) := by
    intro x y hy hyx
    have hxy : 0 ≤ x - y := by linarith
    have hsq : x ≤ (Real.sqrt y + Real.sqrt (x - y)) ^ 2 := by
      have he : (Real.sqrt y + Real.sqrt (x - y)) ^ 2
          = y + (x - y) + 2 * (Real.sqrt y * Real.sqrt (x - y)) := by
        rw [add_sq, Real.sq_sqrt hy, Real.sq_sqrt hxy]; ring
      have hnn : 0 ≤ 2 * (Real.sqrt y * Real.sqrt (x - y)) := by positivity
      rw [he]; linarith
    have := Real.sqrt_le_sqrt hsq
    rw [Real.sqrt_sq (by positivity)] at this
    linarith
  rcases le_total b a with h | h
  · have hge : Real.sqrt b ≤ Real.sqrt a := Real.sqrt_le_sqrt h
    rw [abs_of_nonneg (by linarith : (0:ℝ) ≤ a - b), abs_of_nonneg (by linarith)]
    exact key a b hb h
  · have hge : Real.sqrt a ≤ Real.sqrt b := Real.sqrt_le_sqrt h
    rw [abs_of_nonpos (by linarith : a - b ≤ 0), abs_of_nonpos (by linarith), neg_sub, neg_sub]
    exact key b a ha h

/-- **Sqrt propagation.** `fl(√x)` vs exact `√xR`, given input error `εx` (both `toReal x` and
    `xR` nonnegative): the sqrt rounding `u64·√(toReal x)` plus the Hölder-½ propagation `√εx`. -/
theorem sqrtApprox_error (x : Float) (xR εx : ℝ) (hx : |toReal x - xR| ≤ εx)
    (hxpos : 0 ≤ toReal x) (hxR : 0 ≤ xR) :
    |toReal (Float.sqrt x) - Real.sqrt xR| ≤ u64 * Real.sqrt (toReal x) + Real.sqrt εx := by
  have hprop : |Real.sqrt (toReal x) - Real.sqrt xR| ≤ Real.sqrt εx :=
    (abs_sqrt_sub_sqrt_le hxpos hxR).trans (Real.sqrt_le_sqrt hx)
  calc |toReal (Float.sqrt x) - Real.sqrt xR|
      ≤ |toReal (Float.sqrt x) - Real.sqrt (toReal x)| + |Real.sqrt (toReal x) - Real.sqrt xR| :=
        abs_sub_le _ _ _
    _ ≤ u64 * Real.sqrt (toReal x) + Real.sqrt εx := add_le_add (sqrt_error x) hprop

/-- Rounded values don't grow by more than `(1+u)`: `|fl(a·b)| ≤ (1+u)·|a·b|`. -/
theorem mul_abs_le (a b : Float) :
    |toReal (a * b)| ≤ (1 + u64) * |toReal a * toReal b| := by
  obtain ⟨δ, hδ, he⟩ := mul_model a b
  rw [he, abs_mul]
  have h1 : |1 + δ| ≤ 1 + u64 := (abs_add_le _ _).trans (by simpa using add_le_add_left hδ 1)
  calc |toReal a * toReal b| * |1 + δ|
      ≤ |toReal a * toReal b| * (1 + u64) := mul_le_mul_of_nonneg_left h1 (abs_nonneg _)
    _ = (1 + u64) * |toReal a * toReal b| := by ring

/-- Rounded sums don't grow by more than `(1+u)`: `|fl(a+b)| ≤ (1+u)·|a+b|`. -/
theorem add_abs_le (a b : Float) :
    |toReal (a + b)| ≤ (1 + u64) * |toReal a + toReal b| := by
  obtain ⟨δ, hδ, he⟩ := add_model a b
  rw [he, abs_mul]
  have h1 : |1 + δ| ≤ 1 + u64 := (abs_add_le _ _).trans (by simpa using add_le_add_left hδ 1)
  calc |toReal a + toReal b| * |1 + δ|
      ≤ |toReal a + toReal b| * (1 + u64) := mul_le_mul_of_nonneg_left h1 (abs_nonneg _)
    _ = (1 + u64) * |toReal a + toReal b| := by ring

end Puffer.FloatR
