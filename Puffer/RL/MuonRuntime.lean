/-
Closing the trifecta on Muon's Newton–Schulz map.

`Puffer.Optim.Muon.nsScalar` (ℝ) is `φ(σ) = aσ+bσ³+cσ⁵`, whose classical special
case has the proved quadratic convergence (`nsClassical_quadratic`).
`Puffer.FloatR.nsScalarF` (Float) is the runnable version. Here we bound their gap
by a computable `nsScalarErrBnd`, obtained by pushing the `mulApprox_error` /
`addApprox_error` propagation helpers through the fixed 7-op circuit
`σ·(a + b·σ² + c·σ⁴)`. Same object, three layers: ℝ spec + convergence (proved) ·
running Float · proven error bound.

The two SCALAR Muon update steps get the same treatment (`nesterovMomentumF_error`,
`weightUpdateF_error`): the Nesterov momentum accumulator `m ← μ·m + g` (muon.cu:52)
and the fused weight update `w ← w·(1 − lr·wd) − lr·scale·update` (muon.cu:65) —
runnable `Float` kernels within computable bounds of their ℝ specs.
-/
import Mathlib
import Puffer.Float.Basic
import Puffer.Float.Exec
import Puffer.Optim.Muon

namespace Puffer.RL.MuonRuntime

open Puffer.FloatR
open Puffer.Optim.Muon (nsScalar nsScalar_factor nesterovMomentum weightUpdate weightUpdate_bound
  nesterovMomentum_bound nsScalar_bound)

/-! ### Per-intermediate certified error bounds -/

noncomputable def e2 (σ : Float) : ℝ := u64 * |toReal σ * toReal σ|
noncomputable def e4 (σ : Float) : ℝ :=
  u64 * |toReal (σ * σ) * toReal (σ * σ)| + |toReal (σ * σ)| * e2 σ + |toReal σ * toReal σ| * e2 σ
noncomputable def eBS (b σ : Float) : ℝ :=
  u64 * |toReal b * toReal (σ * σ)| + |toReal b| * e2 σ
noncomputable def eCS (c σ : Float) : ℝ :=
  u64 * |toReal c * toReal (σ * σ * (σ * σ))| + |toReal c| * e4 σ
noncomputable def eT1 (a b σ : Float) : ℝ :=
  u64 * |toReal a + toReal (b * (σ * σ))| + eBS b σ
noncomputable def eT2 (a b c σ : Float) : ℝ :=
  u64 * |toReal (a + b * (σ * σ)) + toReal (c * (σ * σ * (σ * σ)))| + eT1 a b σ + eCS c σ
noncomputable def nsScalarErrBnd (a b c σ : Float) : ℝ :=
  u64 * |toReal σ * toReal (a + b * (σ * σ) + c * (σ * σ * (σ * σ)))| + |toReal σ| * eT2 a b c σ

/-- **Muon Newton–Schulz runtime error.** The running `nsScalarF` deviates from the
    exact real `nsScalar` by at most the certified `nsScalarErrBnd`. -/
theorem nsScalarF_error (a b c σ : Float) :
    |toReal (nsScalarF a b c σ) - nsScalar (toReal a) (toReal b) (toReal c) (toReal σ)|
      ≤ nsScalarErrBnd a b c σ := by
  have h0 : ∀ z : Float, |toReal z - toReal z| ≤ 0 := fun z => by simp
  -- σ² and σ⁴
  have hs2 : |toReal (σ * σ) - toReal σ * toReal σ| ≤ e2 σ := by
    simpa [e2] using mulApprox_error σ σ (toReal σ) (toReal σ) 0 0 (h0 σ) (h0 σ)
  have hs4 : |toReal (σ * σ * (σ * σ)) - toReal σ * toReal σ * (toReal σ * toReal σ)| ≤ e4 σ := by
    simpa [e4] using mulApprox_error (σ * σ) (σ * σ) (toReal σ * toReal σ) (toReal σ * toReal σ)
      (e2 σ) (e2 σ) hs2 hs2
  -- b·σ² and c·σ⁴
  have hbs : |toReal (b * (σ * σ)) - toReal b * (toReal σ * toReal σ)| ≤ eBS b σ := by
    simpa [eBS] using mulApprox_error b (σ * σ) (toReal b) (toReal σ * toReal σ) 0 (e2 σ) (h0 b) hs2
  have hcs : |toReal (c * (σ * σ * (σ * σ))) - toReal c * (toReal σ * toReal σ * (toReal σ * toReal σ))|
      ≤ eCS c σ := by
    simpa [eCS] using mulApprox_error c (σ * σ * (σ * σ)) (toReal c)
      (toReal σ * toReal σ * (toReal σ * toReal σ)) 0 (e4 σ) (h0 c) hs4
  -- a + b·σ²  and  (a + b·σ²) + c·σ⁴
  have ht1 : |toReal (a + b * (σ * σ)) - (toReal a + toReal b * (toReal σ * toReal σ))| ≤ eT1 a b σ := by
    simpa [eT1] using addApprox_error a (b * (σ * σ)) (toReal a) (toReal b * (toReal σ * toReal σ))
      0 (eBS b σ) (h0 a) hbs
  have ht2 : |toReal (a + b * (σ * σ) + c * (σ * σ * (σ * σ)))
        - ((toReal a + toReal b * (toReal σ * toReal σ))
            + toReal c * (toReal σ * toReal σ * (toReal σ * toReal σ)))| ≤ eT2 a b c σ := by
    simpa [eT2] using addApprox_error (a + b * (σ * σ)) (c * (σ * σ * (σ * σ)))
      (toReal a + toReal b * (toReal σ * toReal σ))
      (toReal c * (toReal σ * toReal σ * (toReal σ * toReal σ))) (eT1 a b σ) (eCS c σ) ht1 hcs
  -- outer multiply by σ, then rewrite the ℝ target via the factorization
  have hres : |toReal (σ * (a + b * (σ * σ) + c * (σ * σ * (σ * σ))))
        - toReal σ * ((toReal a + toReal b * (toReal σ * toReal σ))
            + toReal c * (toReal σ * toReal σ * (toReal σ * toReal σ)))| ≤ nsScalarErrBnd a b c σ := by
    simpa [nsScalarErrBnd] using mulApprox_error σ (a + b * (σ * σ) + c * (σ * σ * (σ * σ)))
      (toReal σ)
      ((toReal a + toReal b * (toReal σ * toReal σ))
        + toReal c * (toReal σ * toReal σ * (toReal σ * toReal σ))) 0 (eT2 a b c σ) (h0 σ) ht2
  have hfactor : nsScalar (toReal a) (toReal b) (toReal c) (toReal σ)
      = toReal σ * ((toReal a + toReal b * (toReal σ * toReal σ))
          + toReal c * (toReal σ * toReal σ * (toReal σ * toReal σ))) := by
    rw [nsScalar_factor]; ring
  rw [nsScalarF, hfactor]
  exact hres

/-- **Runtime Newton–Schulz scalar magnitude bound.** The running `nsScalarF a b c σ` is bounded by the ℝ
    per-term-magnitude sum `|a|·|σ| + |b|·|σ|³ + |c|·|σ|⁵` plus the arithmetic-rounding budget `nsScalarErrBnd`
    — composes the ℝ `nsScalar_bound` with the runtime error `nsScalarF_error`. -/
theorem nsScalarF_bound (a b c σ : Float) :
    |toReal (nsScalarF a b c σ)|
      ≤ (|toReal a| * |toReal σ| + |toReal b| * |toReal σ|^3 + |toReal c| * |toReal σ|^5)
        + nsScalarErrBnd a b c σ := by
  have herr := nsScalarF_error a b c σ
  have hbnd := nsScalar_bound (toReal a) (toReal b) (toReal c) (toReal σ)
  have htri := abs_sub_abs_le_abs_sub (toReal (nsScalarF a b c σ))
    (nsScalar (toReal a) (toReal b) (toReal c) (toReal σ))
  linarith

/-! ### The two scalar Muon update steps -/

/-- Certified bound for the Nesterov momentum step (one multiply + one add). -/
noncomputable def nesterovErrBnd (m g μ : Float) : ℝ :=
  u64 * |toReal (μ * m) + toReal g| + u64 * |toReal μ * toReal m|

/-- **Muon Nesterov-momentum runtime error.** The running `nesterovMomentumF m g μ` deviates from the exact
    real `nesterovMomentum (toReal m) (toReal g) (toReal μ)` by at most `nesterovErrBnd`. -/
theorem nesterovMomentumF_error (m g μ : Float) :
    |toReal (nesterovMomentumF m g μ) - nesterovMomentum (toReal m) (toReal g) (toReal μ)|
      ≤ nesterovErrBnd m g μ := by
  have h0 : ∀ z : Float, |toReal z - toReal z| ≤ 0 := fun z => by simp
  have hp : |toReal (μ * m) - toReal μ * toReal m| ≤ u64 * |toReal μ * toReal m| := by
    simpa using mulApprox_error μ m (toReal μ) (toReal m) 0 0 (h0 μ) (h0 m)
  have hres := addApprox_error (μ * m) g (toReal μ * toReal m) (toReal g)
    (u64 * |toReal μ * toReal m|) 0 hp (h0 g)
  rw [nesterovMomentumF, nesterovMomentum]
  simpa [nesterovErrBnd] using hres

/-- **Runtime soundness: the executed Nesterov momentum is magnitude-bounded.** The running
    `nesterovMomentumF m g μ` satisfies `|toReal (nesterovMomentumF …)| ≤ |μ|·|m| + |g| + nesterovErrBnd` —
    composes the ℝ single-step bound `nesterovMomentum_bound` with the runtime error `nesterovMomentumF_error`.
    Together with the ℝ invariant `nesterovMomentum_bound_step`, the executed momentum accumulator stays bounded
    under a bounded gradient (up to arithmetic rounding). -/
theorem nesterovMomentumF_bound (m g μ : Float) :
    |toReal (nesterovMomentumF m g μ)|
      ≤ |toReal μ| * |toReal m| + |toReal g| + nesterovErrBnd m g μ := by
  have herr := nesterovMomentumF_error m g μ
  have hbnd := nesterovMomentum_bound (toReal m) (toReal g) (toReal μ)
  have htri := abs_sub_abs_le_abs_sub (toReal (nesterovMomentumF m g μ))
    (nesterovMomentum (toReal m) (toReal g) (toReal μ))
  linarith

/-- **Runtime Nesterov-momentum stability invariant (fully-real budget).** If the incoming momentum is
    bounded `|m| ≤ B`, the gradient is bounded `|g| ≤ G`, the momentum coefficient is nonnegative `0 ≤ μ`,
    and `B` absorbs the gradient (`G ≤ (1−μ)·B`), then the *executed* `nesterovMomentumF m g μ` stays inside
    the ball of radius `B` up to a first-order rounding budget expressed entirely in the real inputs:
    `|toReal (nesterovMomentumF …)| ≤ B + u·(2+u)·|μ·m| + u·|g|`. This is the runtime stability counterpart of
    `nesterovMomentumF_bound` (which pays the abstract `nesterovErrBnd`): with a uniformly-bounded gradient the
    executed accumulator cannot escape `B` — the extra term is `O(u)` rounding, and the certified
    `nesterovErrBnd` (which references the Float intermediate `toReal (μ·m)`) is dominated by the pure-ℝ budget
    `u·(2+u)·|μ·m| + u·|g|`. The `0 ≤ μ` and `G ≤ (1−μ)·B` hypotheses are load-bearing: they are exactly the
    contraction condition making `[−B, B]` a forward-invariant set of the real momentum map `μ·m + g`. -/
theorem nesterovMomentumF_bound_step (m g μ : Float) (B G : ℝ)
    (hm : |toReal m| ≤ B) (hg : |toReal g| ≤ G)
    (hμ0 : 0 ≤ toReal μ) (hinv : G ≤ (1 - toReal μ) * B) :
    |toReal (nesterovMomentumF m g μ)|
      ≤ B + u64 * (2 + u64) * |toReal μ * toReal m| + u64 * |toReal g| := by
  -- (1) the certified runtime error is dominated by a first-order, fully-real rounding budget
  have herr := nesterovMomentumF_error m g μ
  have hbudget : nesterovErrBnd m g μ
      ≤ u64 * (2 + u64) * |toReal μ * toReal m| + u64 * |toReal g| := by
    unfold nesterovErrBnd
    have hmul := mul_abs_le μ m
    have hu : (0 : ℝ) ≤ u64 := u64_pos.le
    have hP : |toReal (μ * m) + toReal g| ≤ (1 + u64) * |toReal μ * toReal m| + |toReal g| := by
      calc |toReal (μ * m) + toReal g| ≤ |toReal (μ * m)| + |toReal g| := abs_add_le _ _
        _ ≤ (1 + u64) * |toReal μ * toReal m| + |toReal g| := by linarith [hmul]
    nlinarith [mul_le_mul_of_nonneg_left hP hu,
      mul_nonneg hu (abs_nonneg (toReal μ * toReal m))]
  -- (2) the exact-ℝ momentum accumulator stays within B (the classic stability invariant)
  have hreal : |nesterovMomentum (toReal m) (toReal g) (toReal μ)| ≤ B := by
    rw [nesterovMomentum]
    have h1 : |toReal μ * toReal m + toReal g| ≤ |toReal μ * toReal m| + |toReal g| := abs_add_le _ _
    have h2 : |toReal μ * toReal m| = toReal μ * |toReal m| := by
      rw [abs_mul, abs_of_nonneg hμ0]
    have h3 : toReal μ * |toReal m| ≤ toReal μ * B := mul_le_mul_of_nonneg_left hm hμ0
    nlinarith [h1, h2, h3, hg, hinv]
  -- (3) combine via the reverse triangle inequality
  have htri2 := abs_sub_abs_le_abs_sub (toReal (nesterovMomentumF m g μ))
    (nesterovMomentum (toReal m) (toReal g) (toReal μ))
  linarith [herr, hbudget, hreal, htri2]

/-- Certified bound for the fused weight update (the `w·(1−lr·wd) − lr·scale·update` circuit). -/
noncomputable def wuErrBnd (w update lr wd scale : Float) : ℝ :=
  let ep1 := u64 * |toReal lr * toReal wd|
  let ef := u64 * |toReal (1 : Float) - toReal (lr * wd)| + ep1
  let et1 := u64 * |toReal w * toReal (1 - lr * wd)| + |toReal w| * ef
  let ep2 := u64 * |toReal lr * toReal scale|
  let ep3 := u64 * |toReal (lr * scale) * toReal update| + |toReal update| * ep2
  u64 * |toReal (w * (1 - lr * wd)) - toReal (lr * scale * update)| + et1 + ep3

/-- **Muon fused-weight-update runtime error.** The running `weightUpdateF w update lr wd scale` deviates
    from the exact real `weightUpdate (toReal w) …` by at most `wuErrBnd` — the `mul`/`sub`/`mul`/`mul`/`sub`
    circuit propagated (the `1 − lr·wd` factor via `toReal_one`). -/
theorem weightUpdateF_error (w update lr wd scale : Float) :
    |toReal (weightUpdateF w update lr wd scale)
        - weightUpdate (toReal w) (toReal update) (toReal lr) (toReal wd) (toReal scale)|
      ≤ wuErrBnd w update lr wd scale := by
  have h0 : ∀ z : Float, |toReal z - toReal z| ≤ 0 := fun z => by simp
  have hp1 : |toReal (lr * wd) - toReal lr * toReal wd| ≤ u64 * |toReal lr * toReal wd| := by
    simpa using mulApprox_error lr wd (toReal lr) (toReal wd) 0 0 (h0 lr) (h0 wd)
  have hf : |toReal (1 - lr * wd) - (1 - toReal lr * toReal wd)|
      ≤ u64 * |toReal (1 : Float) - toReal (lr * wd)| + u64 * |toReal lr * toReal wd| := by
    have := subApprox_error (1 : Float) (lr * wd) 1 (toReal lr * toReal wd)
      0 (u64 * |toReal lr * toReal wd|) (by simp) hp1
    simpa using this
  have ht1 : |toReal (w * (1 - lr * wd)) - toReal w * (1 - toReal lr * toReal wd)|
      ≤ u64 * |toReal w * toReal (1 - lr * wd)|
        + |toReal w| * (u64 * |toReal (1 : Float) - toReal (lr * wd)| + u64 * |toReal lr * toReal wd|) := by
    have := mulApprox_error w (1 - lr * wd) (toReal w) (1 - toReal lr * toReal wd)
      0 (u64 * |toReal (1 : Float) - toReal (lr * wd)| + u64 * |toReal lr * toReal wd|) (h0 w) hf
    simpa using this
  have hp2 : |toReal (lr * scale) - toReal lr * toReal scale| ≤ u64 * |toReal lr * toReal scale| := by
    simpa using mulApprox_error lr scale (toReal lr) (toReal scale) 0 0 (h0 lr) (h0 scale)
  have hp3 : |toReal (lr * scale * update) - toReal lr * toReal scale * toReal update|
      ≤ u64 * |toReal (lr * scale) * toReal update| + |toReal update| * (u64 * |toReal lr * toReal scale|) := by
    have := mulApprox_error (lr * scale) update (toReal lr * toReal scale) (toReal update)
      (u64 * |toReal lr * toReal scale|) 0 hp2 (h0 update)
    simpa using this
  have hres := subApprox_error (w * (1 - lr * wd)) (lr * scale * update)
    (toReal w * (1 - toReal lr * toReal wd)) (toReal lr * toReal scale * toReal update)
    _ _ ht1 hp3
  rw [weightUpdateF, weightUpdate]
  simpa [wuErrBnd] using hres

/-- **Runtime soundness: the executed Muon weight step is magnitude-bounded.** With the (real) weight-decay
    factor `1 − lr·wd ∈ [0,1]`, the running `weightUpdateF` moves `w` by at most the gradient term plus the
    arithmetic rounding: `|toReal (weightUpdateF …)| ≤ |w| + |lr·scale·update| + wuErrBnd`. Composes the ℝ
    stability bound `weightUpdate_bound` with the runtime error `weightUpdateF_error` — the optimizer step
    can't blow up a weight (the decayed weight is non-expansive), even accounting for Float rounding. -/
theorem weightUpdateF_bound (w update lr wd scale : Float)
    (h0 : 0 ≤ 1 - toReal lr * toReal wd) (h1 : 1 - toReal lr * toReal wd ≤ 1) :
    |toReal (weightUpdateF w update lr wd scale)|
      ≤ |toReal w| + |toReal lr * toReal scale * toReal update|
        + wuErrBnd w update lr wd scale := by
  have herr := weightUpdateF_error w update lr wd scale
  have hbnd := weightUpdate_bound (toReal w) (toReal update) (toReal lr) (toReal wd) (toReal scale) h0 h1
  have htri := abs_sub_abs_le_abs_sub (toReal (weightUpdateF w update lr wd scale))
    (weightUpdate (toReal w) (toReal update) (toReal lr) (toReal wd) (toReal scale))
  linarith

end Puffer.RL.MuonRuntime
