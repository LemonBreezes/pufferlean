/-
bf16 (bfloat16) rounding, over ℝ — the numeric foundation for PufferLib's
error-bound theorems.

PufferLib's `f32_to_bf16` (`~/src/PufferLib/src/bf16.h`) is `(uint16_t)(bits>>16)`:
it drops the low 16 bits of the IEEE-754 f32 pattern, keeping the sign, all 8
exponent bits, and the top 7 mantissa bits. That is **truncation toward zero to 7
fractional significand bits** — NOT round-to-nearest-even.

We model the *net* real→bf16 truncation as `bf16 : ℝ → ℝ`: for `x ≠ 0` with
`e = ⌊log₂|x|⌋` (so `2^e ≤ |x| < 2^{e+1}`), round `|x|` down to a multiple of the
bf16 step `2^{e-7}`, then restore the sign. The intermediate f32 rounding is a
finer step (≤ 2^{-24} relative) folded into the assumption that inputs are in the
normal range (no subnormal/overflow/NaN); the dominant bf16 term is `2^{-7}`.

Main results:
* `bf16_error_bound` : `|bf16 x - x| ≤ 2^{-7} · |x|`   (absolute, one-sided)
* `bf16_relative`    : `∃ δ, |δ| ≤ 2^{-7} ∧ bf16 x = x·(1+δ)`   ((1+δ) model)
* `bf16_abs_le`      : `|bf16 x| ≤ |x|`                (truncation shrinks magnitude)
-/
import Mathlib

namespace Puffer.Numeric

/-- The unit in the last place of bf16 at the scale of `x`: `2^{⌊log₂|x|⌋ - 7}`. -/
noncomputable def bf16Ulp (x : ℝ) : ℝ := (2 : ℝ) ^ (Int.log 2 |x| - 7)

/-- Magnitude of `bf16 x`: `|x|` truncated down to a multiple of `bf16Ulp x`. -/
noncomputable def bf16Mag (x : ℝ) : ℝ := (⌊|x| / bf16Ulp x⌋ : ℝ) * bf16Ulp x

/-- bf16 rounding of a real, truncating toward zero to 7 significand bits. -/
noncomputable def bf16 (x : ℝ) : ℝ := if x < 0 then -bf16Mag x else bf16Mag x

/-- **bf16 is odd**: truncation-toward-zero commutes with negation, `bf16 (−x) = − bf16 x`. Both `bf16Ulp` and
    `bf16Mag` depend on `x` only through `|x|` (so are invariant under negation), and the outer sign-restore flips
    with the input. This makes the sign a free parameter, so every error/magnitude fact proved for `x ≥ 0`
    transfers verbatim to `x ≤ 0`. -/
theorem bf16_neg (x : ℝ) : bf16 (-x) = - bf16 x := by
  have hulp : bf16Ulp (-x) = bf16Ulp x := by unfold bf16Ulp; rw [abs_neg]
  have hmag : bf16Mag (-x) = bf16Mag x := by unfold bf16Mag; rw [abs_neg, hulp]
  unfold bf16
  rcases lt_trichotomy x 0 with hx | hx | hx
  · rw [if_pos hx, if_neg (not_lt.mpr (by linarith : (0:ℝ) ≤ -x)), hmag, neg_neg]
  · subst hx; simp [bf16Mag, bf16Ulp]
  · rw [if_neg (not_lt.mpr hx.le), if_pos (by linarith : -x < 0), hmag]

theorem bf16Ulp_pos (x : ℝ) : 0 < bf16Ulp x := by unfold bf16Ulp; positivity

theorem bf16Mag_nonneg (x : ℝ) : 0 ≤ bf16Mag x := by
  have hs := bf16Ulp_pos x
  unfold bf16Mag
  have hfloor : (0 : ℝ) ≤ (⌊|x| / bf16Ulp x⌋ : ℝ) := by
    have hnn : (0 : ℝ) ≤ |x| / bf16Ulp x := div_nonneg (abs_nonneg x) hs.le
    exact_mod_cast Int.floor_nonneg.mpr hnn
  exact mul_nonneg hfloor hs.le

/-- Truncation never increases magnitude: `bf16Mag x ≤ |x|`. -/
theorem bf16Mag_le_abs (x : ℝ) : bf16Mag x ≤ |x| := by
  have hs := bf16Ulp_pos x
  have hsne : bf16Ulp x ≠ 0 := ne_of_gt hs
  unfold bf16Mag
  calc (⌊|x| / bf16Ulp x⌋ : ℝ) * bf16Ulp x
      ≤ (|x| / bf16Ulp x) * bf16Ulp x :=
        mul_le_mul_of_nonneg_right (Int.floor_le _) hs.le
    _ = |x| := by field_simp

/-- The truncation remainder is strictly below one ulp. -/
theorem abs_sub_bf16Mag_lt_ulp (x : ℝ) : |x| - bf16Mag x < bf16Ulp x := by
  have hs := bf16Ulp_pos x
  have hsne : bf16Ulp x ≠ 0 := ne_of_gt hs
  unfold bf16Mag
  have hfu := Int.lt_floor_add_one (|x| / bf16Ulp x)
  have hcancel : |x| / bf16Ulp x * bf16Ulp x = |x| := by field_simp
  nlinarith [hfu, hs, hcancel]

/-- One ulp at scale `x` is at most `2^{-7}·|x|` (since `2^{⌊log₂|x|⌋} ≤ |x|`). -/
theorem bf16Ulp_le (x : ℝ) (hx : x ≠ 0) : bf16Ulp x ≤ (2 : ℝ) ^ (-7 : ℤ) * |x| := by
  have hxpos : 0 < |x| := abs_pos.mpr hx
  have hlog : (2 : ℝ) ^ (Int.log 2 |x|) ≤ |x| := by
    exact_mod_cast Int.zpow_log_le_self (b := 2) (by norm_num) hxpos
  have hsplit : bf16Ulp x = (2 : ℝ) ^ (Int.log 2 |x|) * (2 : ℝ) ^ (-7 : ℤ) := by
    unfold bf16Ulp
    rw [sub_eq_add_neg, zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
  rw [hsplit, mul_comm]
  exact mul_le_mul_of_nonneg_left hlog (by positivity)

/-- The signed error equals `|x| - bf16Mag x` in every sign case. -/
theorem abs_bf16_sub (x : ℝ) : |bf16 x - x| = |x| - bf16Mag x := by
  have hle := bf16Mag_le_abs x
  have hnn := bf16Mag_nonneg x
  unfold bf16
  rcases lt_trichotomy x 0 with hx | hx | hx
  · rw [if_pos hx]
    have hax : |x| = -x := abs_of_neg hx
    rw [abs_of_nonneg (by linarith), hax]; ring
  · subst hx; simp [bf16Mag, bf16Ulp]
  · rw [if_neg (not_lt.mpr hx.le)]
    have hax : |x| = x := abs_of_pos hx
    rw [abs_of_nonpos (by linarith), hax]; ring

/-- **Absolute error bound**: `|bf16 x - x| ≤ 2^{-7}·|x|`. -/
theorem bf16_error_bound (x : ℝ) : |bf16 x - x| ≤ (2 : ℝ) ^ (-7 : ℤ) * |x| := by
  rw [abs_bf16_sub]
  by_cases hx : x = 0
  · simp [hx, bf16Mag, bf16Ulp]
  · have h1 : |x| - bf16Mag x < bf16Ulp x := abs_sub_bf16Mag_lt_ulp x
    have h2 : bf16Ulp x ≤ (2 : ℝ) ^ (-7 : ℤ) * |x| := bf16Ulp_le x hx
    linarith

/-- Truncation shrinks magnitude: `|bf16 x| ≤ |x|`. -/
theorem bf16_abs_le (x : ℝ) : |bf16 x| ≤ |x| := by
  unfold bf16
  rcases lt_trichotomy x 0 with hx | hx | hx
  · rw [if_pos hx, abs_neg, abs_of_nonneg (bf16Mag_nonneg x)]; exact bf16Mag_le_abs x
  · subst hx; simp [bf16Mag, bf16Ulp]
  · rw [if_neg (not_lt.mpr hx.le), abs_of_nonneg (bf16Mag_nonneg x)]; exact bf16Mag_le_abs x

/-- **(1+δ) relative model**: `bf16 x = x·(1+δ)` with `|δ| ≤ 2^{-7}`. This is the
    form used to propagate rounding error through arithmetic. -/
theorem bf16_relative (x : ℝ) :
    ∃ δ : ℝ, |δ| ≤ (2 : ℝ) ^ (-7 : ℤ) ∧ bf16 x = x * (1 + δ) := by
  by_cases hx : x = 0
  · exact ⟨0, by rw [abs_zero]; positivity, by simp [hx, bf16, bf16Mag, bf16Ulp]⟩
  · refine ⟨(bf16 x - x) / x, ?_, by field_simp; ring⟩
    rw [abs_div, div_le_iff₀ (abs_pos.mpr hx)]
    exact bf16_error_bound x

/-- **Propagation primitive.** Rounding an already-approximate value composes
    additively: if `approx` is within `ε` of `exact`, then `bf16 approx` is within
    `2^{-7}·|approx| + ε`. This is the atomic step for accumulating rounding error
    through a computation. -/
theorem bf16_accum_error (exact approx ε : ℝ) (h : |approx - exact| ≤ ε) :
    |bf16 approx - exact| ≤ (2 : ℝ) ^ (-7 : ℤ) * |approx| + ε := by
  calc |bf16 approx - exact|
      = |(bf16 approx - approx) + (approx - exact)| := by congr 1; ring
    _ ≤ |bf16 approx - approx| + |approx - exact| := abs_add_le _ _
    _ ≤ (2 : ℝ) ^ (-7 : ℤ) * |approx| + ε := add_le_add (bf16_error_bound approx) h

/-- The truncated magnitude `bf16Mag x` is already a fixed point of `bf16Mag` — the load-bearing content
    behind idempotence. Since the ulp `bf16Ulp x = 2^{⌊log₂|x|⌋−7}` is SCALE-dependent, this is not
    automatic: the proof establishes binade invariance, `2^e ≤ bf16Mag x < 2^{e+1}` (`e = ⌊log₂|x|⌋`), via
    `⌊|x|/s⌋ ≥ 128` (as `|x| ≥ 2^e = 128·s`) and `bf16Mag x ≤ |x|`, forcing `Int.log 2 (bf16Mag x) = e` and
    hence `bf16Ulp (bf16Mag x) = bf16Ulp x`; only then is the second floor exact (`⌊(k:ℝ)⌋ = k`). -/
theorem bf16Mag_idem (x : ℝ) : bf16Mag (bf16Mag x) = bf16Mag x := by
  by_cases hx : x = 0
  · subst hx; simp [bf16Mag, bf16Ulp]
  · have hxpos : 0 < |x| := abs_pos.mpr hx
    set e : ℤ := Int.log 2 |x| with he
    set s : ℝ := bf16Ulp x with hs
    have hspos : 0 < s := bf16Ulp_pos x
    have hsne : s ≠ 0 := ne_of_gt hspos
    have hsval : s = (2 : ℝ) ^ (e - 7) := by rw [hs]; rfl
    set k : ℤ := ⌊|x| / s⌋ with hk
    have hmval : bf16Mag x = (k : ℝ) * s := by rw [hk, hs]; rfl
    have hlog_le : (2 : ℝ) ^ e ≤ |x| := by
      rw [he]; exact_mod_cast Int.zpow_log_le_self (b := 2) (by norm_num) hxpos
    have hlog_lt : |x| < (2 : ℝ) ^ (e + 1) := by
      rw [he]; exact_mod_cast Int.lt_zpow_succ_log_self (b := 2) (by norm_num) |x|
    have h2e : (2 : ℝ) ^ e = 128 * s := by
      rw [hsval, show (128 : ℝ) = (2 : ℝ) ^ (7 : ℤ) by norm_num,
          ← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
      ring_nf
    have hk_ge : (128 : ℝ) ≤ (k : ℝ) := by
      have h1 : (128 : ℝ) ≤ |x| / s := by
        rw [le_div_iff₀ hspos, ← h2e]; exact hlog_le
      have : (128 : ℤ) ≤ k := by
        rw [hk]; exact Int.le_floor.mpr (by exact_mod_cast h1)
      exact_mod_cast this
    set m : ℝ := bf16Mag x with hm
    have hm_eq : m = (k : ℝ) * s := hmval
    have hm_lo : (2 : ℝ) ^ e ≤ m := by
      rw [hm_eq, h2e]; exact mul_le_mul_of_nonneg_right hk_ge hspos.le
    have hm_le_abs : m ≤ |x| := by rw [hm]; exact bf16Mag_le_abs x
    have hm_hi : m < (2 : ℝ) ^ (e + 1) := lt_of_le_of_lt hm_le_abs hlog_lt
    have hm_pos : 0 < m := lt_of_lt_of_le (by positivity) hm_lo
    have hm_abs : |m| = m := abs_of_pos hm_pos
    have hlog_m : Int.log 2 |m| = e := by
      rw [hm_abs]
      have h1 : e ≤ Int.log 2 m := by
        apply (Int.zpow_le_iff_le_log (b := 2) (by norm_num) hm_pos).mp
        exact_mod_cast hm_lo
      have h2 : Int.log 2 m < e + 1 := by
        apply (Int.lt_zpow_iff_log_lt (b := 2) (by norm_num) hm_pos).mp
        exact_mod_cast hm_hi
      omega
    have hulp_m : bf16Ulp m = s := by unfold bf16Ulp; rw [hlog_m, hsval]
    show bf16Mag m = m
    unfold bf16Mag
    rw [hulp_m, hm_abs, hm_eq, mul_div_assoc, div_self hsne, mul_one, Int.floor_intCast]

/-- For a nonnegative input, `bf16` is just `bf16Mag` (the sign branch is not taken). -/
theorem bf16_of_nonneg {y : ℝ} (hy : 0 ≤ y) : bf16 y = bf16Mag y := by
  unfold bf16; rw [if_neg (not_lt.mpr hy)]

/-- **bf16 truncation is monotone**: `x ≤ y → bf16 x ≤ bf16 y`. Round-toward-zero to the scale-dependent bf16
    grid preserves order. This is genuinely non-trivial — it does NOT follow from the interval error bound
    `bf16_error_bound` (`|bf16 x − x| ≤ 2^{-7}|x|`): for `x ≤ y` the two error intervals overlap, so that bound
    alone cannot order the outputs. The real content is grid-structural, since the ulp `bf16Ulp = 2^{⌊log₂|·|⌋−7}`
    jumps at every power of two. On nonnegatives (`hmono`): within a binade (`⌊log₂ a⌋ = ⌊log₂ b⌋`) the ulp is
    shared, so it reduces to floor monotonicity (`Int.floor_mono`); across binades (`⌊log₂ a⌋ < ⌊log₂ b⌋`) one
    shows `bf16Mag a ≤ |a| < 2^{⌊log₂a⌋+1} ≤ 2^{⌊log₂b⌋} ≤ bf16Mag b` — the last step being the binade-floor lower
    bound `hlow` (`⌊|b|/ulp⌋ ≥ 128`). Sign is handled uniformly (`bf16Mag` reads only `|·|`): both-nonneg reduces to
    `hmono`, mixed sign gives `−bf16Mag x ≤ bf16Mag y` from nonnegativity, and both-negative flips to `hmono` on
    `−y ≤ −x`. -/
theorem bf16_mono {x y : ℝ} (hxy : x ≤ y) : bf16 x ≤ bf16 y := by
  -- binade lower bound: for `a ≠ 0`, the truncated magnitude stays at or above `2^⌊log₂|a|⌋`.
  have hlow : ∀ a : ℝ, a ≠ 0 → (2 : ℝ) ^ (Int.log 2 |a|) ≤ bf16Mag a := by
    intro a ha
    have hapos : 0 < |a| := abs_pos.mpr ha
    have hspos : 0 < bf16Ulp a := bf16Ulp_pos a
    have hlog_le : (2 : ℝ) ^ (Int.log 2 |a|) ≤ |a| := by
      exact_mod_cast Int.zpow_log_le_self (b := 2) (by norm_num) hapos
    have h2e : (2 : ℝ) ^ (Int.log 2 |a|) = 128 * bf16Ulp a := by
      have hsval : bf16Ulp a = (2 : ℝ) ^ (Int.log 2 |a| - 7) := rfl
      rw [hsval, show (128 : ℝ) = (2 : ℝ) ^ (7 : ℤ) by norm_num,
          ← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
      ring_nf
    have hk_ge : (128 : ℝ) ≤ (⌊|a| / bf16Ulp a⌋ : ℝ) := by
      have h1 : (128 : ℝ) ≤ |a| / bf16Ulp a := by
        rw [le_div_iff₀ hspos, ← h2e]; exact hlog_le
      have hz : (128 : ℤ) ≤ ⌊|a| / bf16Ulp a⌋ := Int.le_floor.mpr (by exact_mod_cast h1)
      exact_mod_cast hz
    show (2 : ℝ) ^ (Int.log 2 |a|) ≤ (⌊|a| / bf16Ulp a⌋ : ℝ) * bf16Ulp a
    rw [h2e]
    exact mul_le_mul_of_nonneg_right hk_ge hspos.le
  -- monotonicity of `bf16Mag` on nonnegatives.
  have hmono : ∀ a b : ℝ, 0 ≤ a → a ≤ b → bf16Mag a ≤ bf16Mag b := by
    intro a b ha hab
    rcases eq_or_lt_of_le ha with ha0 | hapos
    · have hz : bf16Mag a = 0 := by rw [← ha0]; simp [bf16Mag, bf16Ulp]
      rw [hz]; exact bf16Mag_nonneg b
    · have hbpos : 0 < b := lt_of_lt_of_le hapos hab
      have haabs : |a| = a := abs_of_pos hapos
      have hbabs : |b| = b := abs_of_pos hbpos
      have hab_abs : |a| ≤ |b| := by rw [haabs, hbabs]; exact hab
      have hs : 0 < bf16Ulp a := bf16Ulp_pos a
      -- the binade of `a` is at most that of `b`.
      have h2a : (2 : ℝ) ^ (Int.log 2 |a|) ≤ |a| := by
        exact_mod_cast Int.zpow_log_le_self (b := 2) (by norm_num) (abs_pos.mpr hapos.ne')
      have h2ab : (2 : ℝ) ^ (Int.log 2 |a|) ≤ |b| := le_trans h2a hab_abs
      have hea_le_eb : Int.log 2 |a| ≤ Int.log 2 |b| :=
        (Int.zpow_le_iff_le_log (b := 2) (by norm_num) (abs_pos.mpr hbpos.ne')).mp
          (by exact_mod_cast h2ab)
      rcases eq_or_lt_of_le hea_le_eb with heq | hlt
      · -- same binade: equal ulp, reduce to floor monotonicity.
        have hulp_eq : bf16Ulp a = bf16Ulp b := by unfold bf16Ulp; rw [heq]
        have hinv : 0 ≤ (bf16Ulp a)⁻¹ := inv_nonneg.mpr hs.le
        have hdiv : |a| / bf16Ulp a ≤ |b| / bf16Ulp a := by
          rw [div_eq_mul_inv, div_eq_mul_inv]
          exact mul_le_mul_of_nonneg_right hab_abs hinv
        have hfloorR : (⌊|a| / bf16Ulp a⌋ : ℝ) ≤ (⌊|b| / bf16Ulp a⌋ : ℝ) := by
          exact_mod_cast Int.floor_mono hdiv
        unfold bf16Mag
        rw [← hulp_eq]
        exact mul_le_mul_of_nonneg_right hfloorR hs.le
      · -- strictly larger binade of `b`: cross-binade gap.
        have h1 : bf16Mag a ≤ |a| := bf16Mag_le_abs a
        have h2 : |a| < (2 : ℝ) ^ (Int.log 2 |a| + 1) := by
          exact_mod_cast Int.lt_zpow_succ_log_self (b := 2) (by norm_num) |a|
        have h3 : (2 : ℝ) ^ (Int.log 2 |a| + 1) ≤ (2 : ℝ) ^ (Int.log 2 |b|) := by
          apply zpow_le_zpow_right₀ (by norm_num : (1 : ℝ) ≤ 2)
          omega
        have h4 : (2 : ℝ) ^ (Int.log 2 |b|) ≤ bf16Mag b := hlow b hbpos.ne'
        linarith
  -- sign handling.
  rcases le_or_gt 0 x with hx | hx
  · have hy : 0 ≤ y := le_trans hx hxy
    rw [bf16_of_nonneg hx, bf16_of_nonneg hy]
    exact hmono x y hx hxy
  · rcases le_or_gt 0 y with hy | hy
    · have hbx : bf16 x = - bf16Mag x := by unfold bf16; rw [if_pos hx]
      rw [hbx, bf16_of_nonneg hy]
      linarith [bf16Mag_nonneg x, bf16Mag_nonneg y]
    · have hbx : bf16 x = - bf16Mag x := by unfold bf16; rw [if_pos hx]
      have hby : bf16 y = - bf16Mag y := by unfold bf16; rw [if_pos hy]
      rw [hbx, hby]
      have hmagneg : ∀ t : ℝ, bf16Mag (-t) = bf16Mag t := by
        intro t; unfold bf16Mag bf16Ulp; simp only [abs_neg]
      have h0y : 0 ≤ -y := by linarith
      have hyx : -y ≤ -x := by linarith
      have hm := hmono (-y) (-x) h0y hyx
      rw [hmagneg, hmagneg] at hm
      linarith

/-- **bf16 truncation is idempotent**: rounding a value that is already a bf16 truncation leaves it
    unchanged, `bf16 (bf16 x) = bf16 x`. This is the defining fixed-point property of round-toward-zero:
    `bf16 x` lands exactly on the bf16 grid at its own scale, so a second truncation is a no-op. It is
    genuinely non-trivial because the ulp is scale-dependent (`bf16Ulp` reads `⌊log₂|·|⌋`): one must show
    the truncated magnitude stays inside the same binade `[2^e, 2^{e+1})` as the input (`bf16Mag_idem`),
    hence has the same ulp, so the second floor is exact. Sign is handled uniformly via `bf16_neg`. -/
theorem bf16_idem (x : ℝ) : bf16 (bf16 x) = bf16 x := by
  have hB : bf16 (bf16Mag x) = bf16Mag x := by
    rw [bf16_of_nonneg (bf16Mag_nonneg x), bf16Mag_idem x]
  by_cases hx : x < 0
  · have hbx : bf16 x = - bf16Mag x := by unfold bf16; rw [if_pos hx]
    rw [hbx, bf16_neg, hB]
  · have hbx : bf16 x = bf16Mag x := by unfold bf16; rw [if_neg hx]
    rw [hbx]; exact hB

end Puffer.Numeric
