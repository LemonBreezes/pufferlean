/-
The scalar bound feeding the SVD spectral bridge (`Puffer/RL/SpectralBridge.lean`).

For the Muon Newton–Schulz map `p(X) = a·X + b·X³ + c·X⁵` (odd polynomial), factoring `X` out gives
`p(X) = X·q(XᵀX)` with `q(u) = a + b·u + c·u²`, and the induced scalar map on a singular value `σ`
is `p_scalar(σ) = a·σ + b·σ³ + c·σ⁵ = σ·q(σ²)`. Writing `t = σ²` (an eigenvalue of the Gram matrix
`XᵀX`), the SQUARED scalar map is

    p_scalar(σ)² = σ²·q(σ²)² = t · q(t)² =: h(t).

The spectral bridge `opNorm_le_of_gram_eigenvalue_bound` reduces `‖p(X)‖ ≤ C` to `∀i, λᵢ(p(X)ᴴp(X)) ≤ C²`,
and (via the still-to-be-formalized eigenvalue-mapping step) those Gram eigenvalues are exactly the
`h(μᵢ)` for `μᵢ = λᵢ(XᵀX) ∈ [0, ‖X‖²]`. So the analytic crux is the SCALAR inequality `h(t) ≤ C²`.

This file discharges that scalar bound on `t ∈ [0,1]` for the ACTUAL tuned coefficient schedule
`Puffer.Optim.Muon.muonCoeffs` (PufferLib `muon.cu:78–84`). The numeric maximum of `h` over `[0,1]`
is `≈ 1.62089` (attained by step 1 at `t ≈ 0.2373`, i.e. `|p_scalar| ≈ 1.2731`); we prove the
near-tight uniform bound

    ∀ (a,b,c) ∈ muonCoeffs, ∀ t ∈ [0,1],  t · (a + b·t + c·t²)²  ≤  1.63          (0.5 % margin)

so `C = √1.63 < 1.2767` bounds `|p_scalar(σ)|` (hence, modulo the eigenvalue-mapping step, the
per-step operator norm) uniformly across all five iterations. Each case is a degree-5 polynomial
inequality on `[0,1]` closed by `nlinarith` with a Positivstellensatz hint anchored at the interior
maximizer; axiom-clean (`propext`/`Classical.choice`/`Quot.sound`, no `sorry`).

SCOPE CAVEAT. This is the per-step bound on the UNIT interval `[0,1]`. The tuned schedule is designed
for the COMPOSITION: step 1 can map a singular value up to `≈1.273 > 1`, so a fully chained invariant
would need `h(t) ≤ C²` on a slightly larger interval `[0, C²]` (or a more careful composition
argument tracking the shrinking range). The `[0,1]` bound proved here is the clean, self-contained
scalar fact the user requested and the one that pins the per-step overshoot.
-/
import Mathlib
import Puffer.Optim.Muon

namespace Puffer.RL.MuonScalarBound

open Puffer.Optim.Muon (muonCoeffs)

/-- **Uniform Muon scalar bound.** For every coefficient triple `(a,b,c)` in the actual tuned
    schedule `muonCoeffs` and every `t ∈ [0,1]`, the squared induced scalar map satisfies
    `t · (a + b·t + c·t²)² ≤ 1.63`. Hence `C = √1.63 < 1.2767` bounds `|p_scalar(σ)|` for `σ² = t`
    uniformly across all five Newton–Schulz iterations. This is the analytic scalar input the SVD
    spectral bridge consumes (with `1.63 = C²`) once the Gram-eigenvalue mapping is in place. -/
theorem muon_scalar_bound {a b c : ℝ} (hmem : (a, b, c) ∈ muonCoeffs)
    (t : ℝ) (h0 : 0 ≤ t) (h1 : t ≤ 1) : t * (a + b * t + c * t ^ 2) ^ 2 ≤ 1.63 := by
  fin_cases hmem
  · nlinarith [mul_nonneg h0 (sq_nonneg (t - 0.2373)),
      mul_nonneg (sub_nonneg.2 h1) (sq_nonneg (t - 0.2373)), sq_nonneg (t - 0.2373),
      mul_nonneg h0 (sub_nonneg.2 h1), h0, sub_nonneg.2 h1,
      mul_nonneg (mul_nonneg h0 h0) (sub_nonneg.2 h1)]
  · nlinarith [mul_nonneg h0 (sq_nonneg (t - 0.2539)),
      mul_nonneg (sub_nonneg.2 h1) (sq_nonneg (t - 0.2539)), sq_nonneg (t - 0.2539),
      mul_nonneg h0 (sub_nonneg.2 h1), h0, sub_nonneg.2 h1,
      mul_nonneg (mul_nonneg h0 h0) (sub_nonneg.2 h1)]
  · nlinarith [mul_nonneg h0 (sq_nonneg (t - 0.2750)),
      mul_nonneg (sub_nonneg.2 h1) (sq_nonneg (t - 0.2750)), sq_nonneg (t - 0.2750),
      mul_nonneg h0 (sub_nonneg.2 h1), h0, sub_nonneg.2 h1,
      mul_nonneg (mul_nonneg h0 h0) (sub_nonneg.2 h1)]
  · nlinarith [mul_nonneg h0 (sq_nonneg (t - 0.4153)),
      mul_nonneg (sub_nonneg.2 h1) (sq_nonneg (t - 0.4153)), sq_nonneg (t - 0.4153),
      mul_nonneg h0 (sub_nonneg.2 h1), h0, sub_nonneg.2 h1,
      mul_nonneg (mul_nonneg h0 h0) (sub_nonneg.2 h1)]
  · nlinarith [mul_nonneg h0 (sq_nonneg (t - 0.4324)),
      mul_nonneg (sub_nonneg.2 h1) (sq_nonneg (t - 0.4324)), sq_nonneg (t - 0.4324),
      mul_nonneg h0 (sub_nonneg.2 h1), h0, sub_nonneg.2 h1,
      mul_nonneg (mul_nonneg h0 h0) (sub_nonneg.2 h1)]

end Puffer.RL.MuonScalarBound
