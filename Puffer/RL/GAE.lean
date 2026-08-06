/-
Generalized Advantage Estimation (GAE) + V-Trace, over ℝ.

Faithful to `puff_advantage_row_scalar` / the vectorized path in
`~/src/PufferLib/src/pufferlib.cu` (lines ~1271–1360). The per-row recurrence is

    A_t = δ_t + γλ · c_t · nnt_t · A_{t+1},     A_{last} = 0

where `nnt_t = 1 - done_{t+1}` and `c_t = min(importance_t, c_clip)`. Because each
CUDA thread owns one full row and walks it sequentially, this is deterministic and
NOT reduction-order-dependent — so it admits an exact real-number characterization.

Headline theorem (`gaeHead_eq_geoSum`): the recursion, with the per-step factor
`w = γλ` and no intermediate terminals/clipping (`c_t = nnt_t = 1`), equals the
closed form `A_t = Σ_k w^k · δ_{t+k}` — exactly the discounted sum of TD errors
that the kernel is meant to compute.
-/
import Mathlib

namespace Puffer.RL.GAE

open Finset

/-- The GAE advantage at the head of a δ-sequence, as the backward recurrence
    `A_0 = δ_0 + w · A_1` unrolled over the list (`w = γλ`; `A = 0` past the end). -/
def gaeHead (w : ℝ) : List ℝ → ℝ
  | [] => 0
  | δ :: rest => δ + w * gaeHead w rest

@[simp] theorem gaeHead_nil (w : ℝ) : gaeHead w [] = 0 := rfl

@[simp] theorem gaeHead_cons (w : ℝ) (δ : ℝ) (rest : List ℝ) :
    gaeHead w (δ :: rest) = δ + w * gaeHead w rest := rfl

/-- **GAE recursive = closed form.** The unrolled recurrence equals the
    geometric-weighted sum of the TD errors, `Σ_{i<n} w^i · δ_i`. -/
theorem gaeHead_eq_geoSum (w : ℝ) (ds : List ℝ) :
    gaeHead w ds = ∑ i ∈ range ds.length, w ^ i * ds.getD i 0 := by
  induction ds with
  | nil => simp
  | cons δ rest ih =>
    rw [gaeHead_cons, ih, List.length_cons, sum_range_succ']
    simp only [List.getD_cons_succ, List.getD_cons_zero, pow_zero, one_mul, pow_succ]
    rw [mul_sum]
    -- goal: δ + w * ∑ i, w^i * rest.getD i 0 = (∑ i, w^i * w * rest.getD i 0) + δ
    rw [add_comm]
    congr 1
    apply sum_congr rfl
    intro i _
    ring

/-! ### The two divergent TD-error kernels

PufferLib ships two GAE/V-Trace implementations that compute *different* TD errors
(`src/pufferlib.cu`): the **scalar** path (`puff_advantage_row_scalar`, and the CPU
`puff_advantage_cpu`) multiplies the importance ratio `ρ` onto the reward term only, while
the **vectorized** path (`puff_advantage_row_vec`, dispatched whenever the horizon is a
multiple of 8 — i.e. every real GPU `puffer train`) multiplies `ρ` onto the whole TD residual
(the textbook V-trace form). Our runnable trainers use `deltaVec` — the GPU path that actually
trains — verified bit-for-bit against the compiled `pufferlib._C.puff_advantage`. We formalize
both and pin down exactly when they agree (`deltaVec_eq_deltaScalar_iff`: iff `ρ = 1`). -/

/-- Scalar-kernel TD error: `δ = ρ·r + γ·V'·nnt − V` (ρ on the reward only). -/
def deltaScalar (γ ρ r V Vnext nnt : ℝ) : ℝ := ρ * r + γ * Vnext * nnt - V

/-- Vectorized-kernel TD error: `δ = ρ·(r + γ·V'·nnt − V)` (ρ on the whole residual). -/
def deltaVec (γ ρ r V Vnext nnt : ℝ) : ℝ := ρ * (r + γ * Vnext * nnt - V)

/-- The exact discrepancy between the two kernels: `(ρ − 1)·(γ·V'·nnt − V)`. -/
theorem deltaVec_sub_deltaScalar (γ ρ r V Vnext nnt : ℝ) :
    deltaVec γ ρ r V Vnext nnt - deltaScalar γ ρ r V Vnext nnt
      = (ρ - 1) * (γ * Vnext * nnt - V) := by
  unfold deltaVec deltaScalar; ring

/-- **When the kernels agree.** The scalar and vectorized TD errors coincide iff the
    importance ratio is 1, or the (discounted-next-value − value) term vanishes. -/
theorem deltaVec_eq_deltaScalar_iff (γ ρ r V Vnext nnt : ℝ) :
    deltaVec γ ρ r V Vnext nnt = deltaScalar γ ρ r V Vnext nnt
      ↔ ρ = 1 ∨ γ * Vnext * nnt - V = 0 := by
  rw [← sub_eq_zero, deltaVec_sub_deltaScalar, mul_eq_zero, sub_eq_zero]

/-- Consequence: on the first minibatch the importance ratios are initialized to 1
    (`rollouts.ratio ← 1`), so both kernels compute the *same* advantages there. -/
theorem deltaVec_eq_deltaScalar_of_rho_one (γ r V Vnext nnt : ℝ) :
    deltaVec γ 1 r V Vnext nnt = deltaScalar γ 1 r V Vnext nnt := by
  rw [deltaVec_eq_deltaScalar_iff]; left; rfl

/-- **Terminal TD error has no bootstrap** (scalar kernel). At episode end the `nnt = 1−done` mask is `0`, so
    the discounted-next-value term vanishes: `deltaScalar γ ρ r V Vnext 0 = ρ·r − V`. -/
theorem deltaScalar_terminal (γ ρ r V Vnext : ℝ) :
    deltaScalar γ ρ r V Vnext 0 = ρ * r - V := by unfold deltaScalar; ring

/-- **Terminal TD error has no bootstrap** (vectorized kernel). `deltaVec γ ρ r V Vnext 0 = ρ·(r − V)`. -/
theorem deltaVec_terminal (γ ρ r V Vnext : ℝ) :
    deltaVec γ ρ r V Vnext 0 = ρ * (r - V) := by unfold deltaVec; ring

/-- **A zero importance ratio zeroes the vectorized TD error.** `deltaVec γ 0 r V Vnext nnt = 0` — with `ρ = 0`
    the off-policy correction discards the whole residual (V-trace's ratio scaling on the full TD). -/
theorem deltaVec_rho_zero (γ r V Vnext nnt : ℝ) :
    deltaVec γ 0 r V Vnext nnt = 0 := by unfold deltaVec; ring

/-- The discounted return `Σ_k w^k r_k` is the same closed form; the value target
    `returns = values + advantages` reuses `gaeHead` on the TD errors. -/
def discountedReturn (w : ℝ) (rs : List ℝ) : ℝ := gaeHead w rs

theorem discountedReturn_eq_geoSum (w : ℝ) (rs : List ℝ) :
    discountedReturn w rs = ∑ i ∈ range rs.length, w ^ i * rs.getD i 0 :=
  gaeHead_eq_geoSum w rs

/-! ### Constant reward/TD stream: the geometric series -/

/-- **Constant-stream discounted sum = `δ · Σwⁱ`.** For a length-`n` constant stream (every term `δ`), the
    recurrence sums to `δ` times the geometric series `∑_{i<n} wⁱ` — the analytic return of a constant per-step
    signal. -/
theorem gaeHead_replicate (w δ : ℝ) (n : Nat) :
    gaeHead w (List.replicate n δ) = δ * ∑ i ∈ Finset.range n, w ^ i := by
  rw [gaeHead_eq_geoSum, List.length_replicate, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i hi
  have hd : (List.replicate n δ).getD i 0 = δ := by
    have : i < (List.replicate n δ).length := by
      rw [List.length_replicate]; exact Finset.mem_range.mp hi
    rw [List.getD_eq_getElem _ _ this, List.getElem_replicate]
  rw [hd]; ring

/-- **Constant-stream closed form** (`w ≠ 1`): `gaeHead w (replicate n δ) = δ·(wⁿ − 1)/(w − 1)` — the finite
    geometric sum in closed form (`geom_sum_eq`). -/
theorem gaeHead_replicate_closed (w δ : ℝ) (n : Nat) (hw : w ≠ 1) :
    gaeHead w (List.replicate n δ) = δ * ((w ^ n - 1) / (w - 1)) := by
  rw [gaeHead_replicate, geom_sum_eq hw]

/-! ### Linearity and sign structure of the GAE recurrence -/

/-- **GAE is homogeneous in the TD errors.** Scaling every TD error by `c` scales the advantage by `c`:
    `gaeHead w (c·δ) = c · gaeHead w δ`. Since `δ` is linear in the rewards, this is the reward-scale covariance
    of the advantage — rescaling the reward units rescales advantages (and returns) by the same factor. -/
theorem gaeHead_smul (w c : ℝ) (δs : List ℝ) :
    gaeHead w (δs.map (fun δ => c * δ)) = c * gaeHead w δs := by
  induction δs with
  | nil => simp
  | cons δ rest ih => simp only [List.map_cons, gaeHead_cons, ih]; ring

/-- **GAE negates with the TD errors** (the `c = −1` case of `gaeHead_smul`): `gaeHead w (−δ) = − gaeHead w δ`. -/
theorem gaeHead_neg (w : ℝ) (δs : List ℝ) :
    gaeHead w (δs.map (fun δ => -δ)) = - gaeHead w δs := by
  induction δs with
  | nil => simp
  | cons δ rest ih => simp only [List.map_cons, gaeHead_cons, ih]; ring

/-- **GAE temporal composition law.** The advantage over a concatenated TD-error stream splits into the first
    segment plus the discounted second: `gaeHead w (xs ++ ys) = gaeHead w xs + w^|xs| · gaeHead w ys`. The weight
    `w^|xs| = (γλ)^|xs|` is exactly the discount accumulated over the first segment, so appending future TD errors
    contributes only through that geometric factor — the semigroup/temporal-composition structure of the backward
    GAE recurrence, generalizing `gaeHead_replicate`. Proved by induction with `pow_succ` on the segment length. -/
theorem gaeHead_append (w : ℝ) (xs ys : List ℝ) :
    gaeHead w (xs ++ ys) = gaeHead w xs + w ^ xs.length * gaeHead w ys := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
      simp only [List.cons_append, gaeHead_cons, ih, List.length_cons, pow_succ]
      ring

/-- **A nonnegative TD stream gives a nonnegative advantage.** If every TD error is `≥ 0` and `w ≥ 0`, then
    `gaeHead w δ ≥ 0` — the geometric accumulation of nonnegative signals stays nonnegative (a value estimate
    consistently below its bootstrap yields a nonnegative advantage). -/
theorem gaeHead_nonneg (w : ℝ) (hw : 0 ≤ w) (δs : List ℝ) (h : ∀ x ∈ δs, 0 ≤ x) :
    0 ≤ gaeHead w δs := by
  induction δs with
  | nil => simp
  | cons δ rest ih =>
      rw [gaeHead_cons]
      have hδ : 0 ≤ δ := h δ (List.mem_cons_self ..)
      have hrest : 0 ≤ gaeHead w rest := ih (fun x hx => h x (List.mem_cons_of_mem _ hx))
      have : 0 ≤ w * gaeHead w rest := mul_nonneg hw hrest
      linarith

/-- **A nonpositive TD stream gives a nonpositive advantage** (the sign-mirror of `gaeHead_nonneg`). -/
theorem gaeHead_nonpos (w : ℝ) (hw : 0 ≤ w) (δs : List ℝ) (h : ∀ x ∈ δs, x ≤ 0) :
    gaeHead w δs ≤ 0 := by
  induction δs with
  | nil => simp
  | cons δ rest ih =>
      rw [gaeHead_cons]
      have hδ : δ ≤ 0 := h δ (List.mem_cons_self ..)
      have hrest : gaeHead w rest ≤ 0 := ih (fun x hx => h x (List.mem_cons_of_mem _ hx))
      have : w * gaeHead w rest ≤ 0 := mul_nonpos_of_nonneg_of_nonpos hw hrest
      linarith

/-- **GAE advantage is monotone in the discount `w = γλ` for a nonnegative TD stream.** If every TD error is
    `≥ 0` and `0 ≤ w₁ ≤ w₂`, then `gaeHead w₁ δs ≤ gaeHead w₂ δs`. Raising the discount puts more weight on the
    (nonnegative) future TD errors, so the accumulated advantage can only grow — the comparative statics of the
    GAE recurrence in its bootstrapping factor. Proved by induction, using `gaeHead_nonneg` to keep the discounted
    tails nonnegative. -/
theorem gaeHead_mono_discount (w₁ w₂ : ℝ) (hw₁ : 0 ≤ w₁) (h₁₂ : w₁ ≤ w₂)
    (δs : List ℝ) (h : ∀ x ∈ δs, 0 ≤ x) :
    gaeHead w₁ δs ≤ gaeHead w₂ δs := by
  have hw₂ : 0 ≤ w₂ := le_trans hw₁ h₁₂
  induction δs with
  | nil => simp
  | cons δ rest ih =>
      have htail : ∀ x ∈ rest, 0 ≤ x := fun x hx => h x (List.mem_cons_of_mem _ hx)
      have hle : gaeHead w₁ rest ≤ gaeHead w₂ rest := ih htail
      have hb0 : 0 ≤ gaeHead w₂ rest := gaeHead_nonneg w₂ hw₂ rest htail
      rw [gaeHead_cons, gaeHead_cons]
      have hstep : w₁ * gaeHead w₁ rest ≤ w₂ * gaeHead w₂ rest :=
        calc w₁ * gaeHead w₁ rest
            ≤ w₁ * gaeHead w₂ rest := mul_le_mul_of_nonneg_left hle hw₁
          _ ≤ w₂ * gaeHead w₂ rest := mul_le_mul_of_nonneg_right h₁₂ hb0
      linarith

/-! ### Geometric boundedness: bounded rewards give a bounded discounted return -/

/-- **Discounted-return geometric bound.** If every reward/TD-error in `rs` has magnitude `≤ R` and the
    discount `w ∈ [0,1)`, then the GAE/return recurrence is bounded by the geometric sum `R/(1−w)` — regardless
    of horizon length. The fundamental RL stability estimate: a uniformly-bounded per-step signal yields a
    bounded return. Proved by induction on the recurrence `gaeHead w (δ::rest) = δ + w·gaeHead w rest`. -/
theorem gaeHead_bounded (w R : ℝ) (hw0 : 0 ≤ w) (hw1 : w < 1) (hR : 0 ≤ R) :
    ∀ (rs : List ℝ), (∀ x ∈ rs, |x| ≤ R) → |gaeHead w rs| ≤ R / (1 - w) := by
  have hden : 0 < 1 - w := by linarith
  have hne : (1 - w) ≠ 0 := ne_of_gt hden
  intro rs
  induction rs with
  | nil => intro _; rw [gaeHead_nil, abs_zero]; exact div_nonneg hR hden.le
  | cons δ rest ih =>
    intro h
    have hδ : |δ| ≤ R := h δ (List.mem_cons_self ..)
    have hrest : |gaeHead w rest| ≤ R / (1 - w) := ih (fun x hx => h x (List.mem_cons_of_mem _ hx))
    rw [gaeHead_cons]
    calc |δ + w * gaeHead w rest| ≤ |δ| + |w * gaeHead w rest| := abs_add_le _ _
      _ = |δ| + w * |gaeHead w rest| := by rw [abs_mul, abs_of_nonneg hw0]
      _ ≤ R + w * (R / (1 - w)) := add_le_add hδ (mul_le_mul_of_nonneg_left hrest hw0)
      _ = R / (1 - w) := by field_simp; ring

/-- **Difference of GAE recurrences = GAE of the pointwise differences.** For two equal-length TD-error streams,
    the advantage of the pointwise difference equals the difference of the advantages:
    `gaeHead w xs − gaeHead w ys = gaeHead w (zipWith (·−·) xs ys)`. This is the exact algebraic bridge that turns
    a perturbation of the value function into a single GAE recurrence. Proved by simultaneous induction on the two
    lists (the equal-length hypothesis rules out the length-mismatch branches via `omega`, and the cons/cons step
    reduces to the IH after `gaeHead_cons` + `ring`). The equal-length hypothesis is essential — with `xs = [1]`,
    `ys = [1,1]` the LHS is `1 − (1 + w) = −w` while the RHS (over the truncated `zipWith`) is `gaeHead w [0] = 0`. -/
theorem gaeHead_sub_zipWith (w : ℝ) :
    ∀ (xs ys : List ℝ), xs.length = ys.length →
      gaeHead w xs - gaeHead w ys = gaeHead w (List.zipWith (· - ·) xs ys) := by
  intro xs
  induction xs with
  | nil =>
    intro ys h
    cases ys with
    | nil => simp
    | cons y ys => simp only [List.length_nil, List.length_cons] at h; omega
  | cons x xs ih =>
    intro ys h
    cases ys with
    | nil => simp only [List.length_cons, List.length_nil] at h; omega
    | cons y ys =>
      have h' : xs.length = ys.length := by simp only [List.length_cons] at h; omega
      simp only [List.zipWith_cons_cons, gaeHead_cons]
      rw [← ih ys h']
      ring

/-- **GAE advantage is Lipschitz in the TD-error stream (horizon-free stability).** If two TD-error streams
    `xs, ys` have the same length and differ pointwise by at most `ε` (every entry of `xs − ys` has magnitude
    `≤ ε`), and the discount `w = γλ ∈ [0,1)`, then the resulting GAE advantages differ by at most `ε/(1−w)` —
    INDEPENDENT of the horizon. Concretely: perturbing the value function so each TD error moves by `≤ ε` moves the
    advantage by `≤ ε/(1−w)`, so the advantage map is `1/(1−w)`-Lipschitz in the sup metric on TD errors. This is
    the stability estimate that makes GAE well-behaved under value-function updates, and it strictly GENERALIZES
    `gaeHead_bounded` (take `ys` all-zero of the same length to recover `|A| ≤ R/(1−w)`). Proof: collapse the
    difference into one recurrence via `gaeHead_sub_zipWith`, then apply the geometric bound `gaeHead_bounded`. All
    hypotheses load-bearing: `hlen` is essential (`gaeHead_sub_zipWith` is false when lengths differ, and zipWith
    truncation would silently drop terms), `hw1 : w < 1` for the geometric-sum convergence (division by `1−w`), and
    `hw0`/`hε` are inherited from `gaeHead_bounded` and bite on the empty-list/degenerate cases. -/
theorem gaeHead_lipschitz (w ε : ℝ) (hw0 : 0 ≤ w) (hw1 : w < 1) (hε : 0 ≤ ε)
    (xs ys : List ℝ) (hlen : xs.length = ys.length)
    (h : ∀ z ∈ List.zipWith (· - ·) xs ys, |z| ≤ ε) :
    |gaeHead w xs - gaeHead w ys| ≤ ε / (1 - w) := by
  rw [gaeHead_sub_zipWith w xs ys hlen]
  exact gaeHead_bounded w ε hw0 hw1 hε _ h

/-- **Discounted return is geometrically bounded** (corollary): `|Σₖ wᵏ rₖ| ≤ R/(1−w)` for `|rₖ| ≤ R`,
    `w ∈ [0,1)`. -/
theorem discountedReturn_bounded (w R : ℝ) (rs : List ℝ) (hw0 : 0 ≤ w) (hw1 : w < 1)
    (hR : 0 ≤ R) (h : ∀ x ∈ rs, |x| ≤ R) :
    |discountedReturn w rs| ≤ R / (1 - w) := by
  rw [discountedReturn]
  exact gaeHead_bounded w R hw0 hw1 hR rs h

end Puffer.RL.GAE
