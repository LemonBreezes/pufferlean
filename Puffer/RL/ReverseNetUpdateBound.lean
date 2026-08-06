/-
The WHOLE-NET update bound fed by the fully-quantified reverse-mode gradient — the capstone that ties the
reverse-mode gradient thread to the trainer's actual parameter update.

`NetUpdateBound.vecAxpy_entrywise_error` bounds every component of the trainer's `vecAxpy lr p g` update
against the ideal real ascent, PARAMETERIZED by a per-component gradient error `εg`. `ReverseUpdateBound.
reverseGrad_errorF_bounded` supplies that error for the reverse-mode Float gradient with NO opaque `hbridge`:
`|revGrad σ e i − derivR e (envR σ) i| ≤ revGradBndF σ e i W + bridgeBnd e σ i` under `WD` (tight sweep
rounding + bounded primal rounding). Composing them:

  `vecAxpy_reverse_bounded_error` — every weight `i` of `vecAxpy lr p g` (the trainer's SGD update, with `g`
  holding the reverse gradients) is within `u64·|…| + u64·|lr·gᵢ| + |lr|·(revGradBndF + bridgeBnd)` of the
  ideal real gradient-ascent step against the symbolic derivative `derivR e (envR σ) i`.
  `vecAxpy_reverse_bounded_deriv_error` states it against the TRUE Mathlib derivative (`derivR_eq_deriv`,
  needs `PosR`), and `vecAxpy_reverse_bounded_sup_error` collapses it to one ℓ∞ scalar via uniform handles.

Every rounding layer is now quantified end-to-end — the forward value, the reverse-sweep gradient, the primal
bridge, and the SGD axpy step — with the only remaining conditions the honest, disclosed ones (`WD`
away-from-kink clearance, the edge-weight bound `hedges`, and `g` realizing the reverse gradient `hgeq`).
Axiom-clean beyond the trusted Float base.
-/
import Puffer.RL.NetUpdateBound
import Puffer.RL.ReverseUpdateBound

namespace Puffer.RL.ReverseNetUpdateBound

open Puffer.FloatR
open Puffer.FloatR.ADR (Expr evalR envR derivR derivR_eq_deriv WD PosR)
open Puffer.FloatR.ADReverse (revGrad revGradBndF bridgeBnd reverseGrad_errorF_bounded comp)
open Puffer.FloatR.AD (Tape)
open Puffer.RL.NNTrain (vecAxpy matAxpy)
open Puffer.RL.NetUpdateBound (vecAxpy_getElem frob_sum_le vecAxpy_sup_error matAxpy_sup_error)

/-- **Whole-net reverse-mode update bound (entrywise, vs `derivR`).** With the gradient array `g` holding the
    reverse gradients (`hgeq`), every weight's `vecAxpy` step is within the SGD-step bound of the ideal real
    ascent against `derivR e (envR σ) i`, the gradient error being the fully-quantified
    `revGradBndF + bridgeBnd` (no `hbridge`). -/
theorem vecAxpy_reverse_bounded_error (e : Expr) (σ : Nat → Float) (lr : Float) (p g : Array Float) (W : ℝ)
    (hW : 0 ≤ W) (hwd : WD e σ)
    (hedges : ∀ idx, idx < (comp σ e Tape.empty).2.val.size →
      ∀ ed ∈ (comp σ e Tape.empty).2.deps[idx]!,
        ed.1 < (comp σ e Tape.empty).2.val.size ∧ |toReal ed.2| ≤ W)
    (hgeq : ∀ i, i < p.size → toReal (g[i]!) = revGrad σ e i)
    (i : Nat) (hi : i < p.size) :
    |toReal ((vecAxpy lr p g)[i]!) - (toReal (p[i]!) + toReal lr * derivR e (envR σ) i)|
      ≤ u64 * |toReal (p[i]!) + toReal (lr * g[i]!)|
          + (u64 * |toReal lr * toReal (g[i]!)| + |toReal lr| * (revGradBndF σ e i W + bridgeBnd e σ i)) := by
  rw [vecAxpy_getElem lr p g i hi]
  have hg : |toReal (g[i]!) - derivR e (envR σ) i| ≤ revGradBndF σ e i W + bridgeBnd e σ i := by
    rw [hgeq i hi]; exact reverseGrad_errorF_bounded σ e i W hW hedges hwd
  exact axpyStep_error (p[i]!) lr (g[i]!) (derivR e (envR σ) i)
    (revGradBndF σ e i W + bridgeBnd e σ i) hg

/-- **Whole-net reverse-mode update bound vs the TRUE derivative.** As above, against
    `∂(evalR e)/∂(var i)` (`derivR_eq_deriv`, needs `PosR`). -/
theorem vecAxpy_reverse_bounded_deriv_error (e : Expr) (σ : Nat → Float) (lr : Float) (p g : Array Float)
    (W : ℝ) (hW : 0 ≤ W) (hwd : WD e σ) (hp : PosR e (envR σ))
    (hedges : ∀ idx, idx < (comp σ e Tape.empty).2.val.size →
      ∀ ed ∈ (comp σ e Tape.empty).2.deps[idx]!,
        ed.1 < (comp σ e Tape.empty).2.val.size ∧ |toReal ed.2| ≤ W)
    (hgeq : ∀ i, i < p.size → toReal (g[i]!) = revGrad σ e i)
    (i : Nat) (hi : i < p.size) :
    |toReal ((vecAxpy lr p g)[i]!)
        - (toReal (p[i]!) + toReal lr
            * deriv (fun t => evalR e (Function.update (envR σ) i t)) (envR σ i))|
      ≤ u64 * |toReal (p[i]!) + toReal (lr * g[i]!)|
          + (u64 * |toReal lr * toReal (g[i]!)| + |toReal lr| * (revGradBndF σ e i W + bridgeBnd e σ i)) := by
  rw [← derivR_eq_deriv e (envR σ) i hp]
  exact vecAxpy_reverse_bounded_error e σ lr p g W hW hwd hedges hgeq i hi

/-- **Whole-net reverse-mode update bound, ℓ∞ aggregate.** Uniform handles `Bmag`/`Bprod`/`Bε` collapse it to
    a single scalar bounding EVERY weight's update error. -/
theorem vecAxpy_reverse_bounded_sup_error (e : Expr) (σ : Nat → Float) (lr : Float) (p g : Array Float)
    (W : ℝ) (hW : 0 ≤ W) (hwd : WD e σ)
    (hedges : ∀ idx, idx < (comp σ e Tape.empty).2.val.size →
      ∀ ed ∈ (comp σ e Tape.empty).2.deps[idx]!,
        ed.1 < (comp σ e Tape.empty).2.val.size ∧ |toReal ed.2| ≤ W)
    (hgeq : ∀ i, i < p.size → toReal (g[i]!) = revGrad σ e i)
    (Bmag Bprod Bε : ℝ)
    (hmag : ∀ i, i < p.size → |toReal (p[i]!) + toReal (lr * g[i]!)| ≤ Bmag)
    (hprod : ∀ i, i < p.size → |toReal lr * toReal (g[i]!)| ≤ Bprod)
    (hεb : ∀ i, i < p.size → revGradBndF σ e i W + bridgeBnd e σ i ≤ Bε)
    (i : Nat) (hi : i < p.size) :
    |toReal ((vecAxpy lr p g)[i]!) - (toReal (p[i]!) + toReal lr * derivR e (envR σ) i)|
      ≤ u64 * Bmag + (u64 * Bprod + |toReal lr| * Bε) := by
  refine (vecAxpy_reverse_bounded_error e σ lr p g W hW hwd hedges hgeq i hi).trans ?_
  have hu : (0:ℝ) ≤ u64 := le_of_lt u64_pos
  exact add_le_add (mul_le_mul_of_nonneg_left (hmag i hi) hu)
    (add_le_add (mul_le_mul_of_nonneg_left (hprod i hi) hu)
      (mul_le_mul_of_nonneg_left (hεb i hi) (abs_nonneg _)))

/-- **Whole-net reverse-mode update bound, Frobenius (ℓ₂) aggregate.** The sum of squared per-weight update
    errors of the whole `vecAxpy` step is within `p.size · B²` for the uniform ℓ∞ bound
    `B = u64·Bmag + (u64·Bprod + |lr|·Bε)` — i.e. `‖update − ideal ascent‖₂ ≤ √(p.size)·B`, with the gradient
    error the fully-quantified `revGradBndF + bridgeBnd` (no `hbridge`). -/
theorem vecAxpy_reverse_bounded_frob_error (e : Expr) (σ : Nat → Float) (lr : Float) (p g : Array Float)
    (W : ℝ) (hW : 0 ≤ W) (hwd : WD e σ)
    (hedges : ∀ idx, idx < (comp σ e Tape.empty).2.val.size →
      ∀ ed ∈ (comp σ e Tape.empty).2.deps[idx]!,
        ed.1 < (comp σ e Tape.empty).2.val.size ∧ |toReal ed.2| ≤ W)
    (hgeq : ∀ i, i < p.size → toReal (g[i]!) = revGrad σ e i)
    (Bmag Bprod Bε : ℝ)
    (hmag : ∀ i, i < p.size → |toReal (p[i]!) + toReal (lr * g[i]!)| ≤ Bmag)
    (hprod : ∀ i, i < p.size → |toReal lr * toReal (g[i]!)| ≤ Bprod)
    (hεb : ∀ i, i < p.size → revGradBndF σ e i W + bridgeBnd e σ i ≤ Bε) :
    ∑ i ∈ Finset.range p.size,
        (toReal ((vecAxpy lr p g)[i]!) - (toReal (p[i]!) + toReal lr * derivR e (envR σ) i)) ^ 2
      ≤ (p.size : ℝ) * (u64 * Bmag + (u64 * Bprod + |toReal lr| * Bε)) ^ 2 :=
  frob_sum_le _ p.size _ (fun i hi =>
    vecAxpy_reverse_bounded_sup_error e σ lr p g W hW hwd hedges hgeq Bmag Bprod Bε hmag hprod hεb i hi)

/-! ### Whole-MLP single training-step composition (ℓ∞)

The bounds above are stated per parameter *array* (`vecAxpy`) with the flat variable index taken to be the
array index. The trainer's real update touches FOUR tensors — `W1`/`W2` (matrices, via `matAxpy`) and
`b1`/`b2` (vectors, via `vecAxpy`) — all differentiated from ONE loss expression `e` in a shared flat variable
space. These three declarations lift the per-array reverse bound to that whole-MLP step:
`RevGradData` bundles the one-loss/one-sweep hypotheses; the two `_idx_` lemmas route each tensor through its
own slice `vidx` of the flat parameter space; and `mlpStep_reverse_bounded_sup_error` composes all four into a
single computable ℓ∞ interval on the whole parameter→parameter map of one SGD step. -/

/-- The reverse-mode gradient data shared by every tensor of one loss expression `e` at env `σ`, per-edge
    weight bound `W`: the away-from-kink clearance (`WD e σ`) and the edge-weight bound (`hedges`). Bundled so
    the whole-MLP capstone states them ONCE (one loss ⇒ one reverse sweep ⇒ one set of edge data). -/
structure RevGradData (e : Expr) (σ : Nat → Float) (W : ℝ) : Prop where
  hW : 0 ≤ W
  hwd : WD e σ
  hedges : ∀ idx, idx < (comp σ e Tape.empty).2.val.size →
    ∀ ed ∈ (comp σ e Tape.empty).2.deps[idx]!,
      ed.1 < (comp σ e Tape.empty).2.val.size ∧ |toReal ed.2| ≤ W

/-- **Generalized-index vector reverse update bound (ℓ∞).** As `vecAxpy_reverse_bounded_sup_error`, but the
    flat variable index is routed by an arbitrary map `vidx : Nat → Nat`, so a bias vector can occupy its own
    slice of the shared flat parameter space. Instantiates `vecAxpy_sup_error` with the reverse-AD ideal
    `derivR e (envR σ) ∘ vidx` and error `(revGradBndF + bridgeBnd) ∘ vidx`, discharged per entry by
    `reverseGrad_errorF_bounded`. -/
theorem vecAxpy_reverse_idx_sup_error (e : Expr) (σ : Nat → Float) (lr : Float)
    (p g : Array Float) (vidx : Nat → Nat) (W : ℝ) (hd : RevGradData e σ W)
    (hgeq : ∀ i, i < p.size → toReal (g[i]!) = revGrad σ e (vidx i))
    (Bmag Bprod Bε : ℝ)
    (hmag : ∀ i, i < p.size → |toReal (p[i]!) + toReal (lr * g[i]!)| ≤ Bmag)
    (hprod : ∀ i, i < p.size → |toReal lr * toReal (g[i]!)| ≤ Bprod)
    (hεb : ∀ i, i < p.size → revGradBndF σ e (vidx i) W + bridgeBnd e σ (vidx i) ≤ Bε)
    (i : Nat) (hi : i < p.size) :
    |toReal ((vecAxpy lr p g)[i]!) - (toReal (p[i]!) + toReal lr * derivR e (envR σ) (vidx i))|
      ≤ u64 * Bmag + (u64 * Bprod + |toReal lr| * Bε) := by
  refine vecAxpy_sup_error lr p g (fun i => derivR e (envR σ) (vidx i))
    (fun i => revGradBndF σ e (vidx i) W + bridgeBnd e σ (vidx i))
    ?_ Bmag Bprod Bε hmag hprod hεb i hi
  intro i' hi'
  rw [hgeq i' hi']
  exact reverseGrad_errorF_bounded σ e (vidx i') W hd.hW hd.hedges hd.hwd

/-- **Generalized-index weight-MATRIX reverse update bound (ℓ∞), fully quantified.** For the trainer's actual
    weight update `matAxpy lr p g` (used for `W1`/`W2`), with each gradient entry `(g[i]!)[j]!` realizing the
    reverse gradient of the loss `e` at flat variable `vidx i j` (`hgeq`), every weight `(i,j)` is within
    `u64·Bmag + (u64·Bprod + |lr|·Bε)` of the ideal real ascent `p[i][j] + lr·derivR e (envR σ) (vidx i j)`,
    the gradient error the FULLY-QUANTIFIED `revGradBndF + bridgeBnd` (no opaque bridge). Matrix-level
    counterpart of `vecAxpy_reverse_idx_sup_error`; instantiates `matAxpy_sup_error` with the reverse-AD
    ideal/error and discharges each entry by `reverseGrad_errorF_bounded`. This is the piece the vector-only
    `vecAxpy_reverse_bounded_*` did not cover. -/
theorem matAxpy_reverse_idx_sup_error (e : Expr) (σ : Nat → Float) (lr : Float)
    (p g : Array (Array Float)) (vidx : Nat → Nat → Nat) (W : ℝ) (hd : RevGradData e σ W)
    (hgeq : ∀ i j, i < p.size → j < (p[i]!).size → toReal ((g[i]!)[j]!) = revGrad σ e (vidx i j))
    (Bmag Bprod Bε : ℝ)
    (hmag : ∀ i j, i < p.size → j < (p[i]!).size →
      |toReal ((p[i]!)[j]!) + toReal (lr * (g[i]!)[j]!)| ≤ Bmag)
    (hprod : ∀ i j, i < p.size → j < (p[i]!).size → |toReal lr * toReal ((g[i]!)[j]!)| ≤ Bprod)
    (hεb : ∀ i j, i < p.size → j < (p[i]!).size →
      revGradBndF σ e (vidx i j) W + bridgeBnd e σ (vidx i j) ≤ Bε)
    (i j : Nat) (hi : i < p.size) (hj : j < (p[i]!).size) :
    |toReal (((matAxpy lr p g)[i]!)[j]!)
        - (toReal ((p[i]!)[j]!) + toReal lr * derivR e (envR σ) (vidx i j))|
      ≤ u64 * Bmag + (u64 * Bprod + |toReal lr| * Bε) := by
  refine matAxpy_sup_error lr p g (fun i j => derivR e (envR σ) (vidx i j))
    (fun i j => revGradBndF σ e (vidx i j) W + bridgeBnd e σ (vidx i j))
    ?_ Bmag Bprod Bε hmag hprod hεb i j hi hj
  intro i' j' hi' hj'
  rw [hgeq i' j' hi' hj']
  exact reverseGrad_errorF_bounded σ e (vidx i' j') W hd.hW hd.hedges hd.hwd

/-- **Whole-MLP reverse-mode SGD single-step composition capstone (ℓ∞).** One runnable SGD training step
    updates the four MLP tensors as `W1 := matAxpy lr W1 gW1`, `b1 := vecAxpy lr b1 gb1`,
    `W2 := matAxpy lr W2 gW2`, `b2 := vecAxpy lr b2 gb2`, where the four gradient arrays hold the reverse-mode
    gradients of a SINGLE loss expression `e` (routed to the shared flat parameter space by the index maps
    `iW1`/`ib1`/`iW2`/`ib2`). Given ONE bundle of reverse-gradient data (`RevGradData` — one loss, one reverse
    sweep) and uniform ℓ∞ handles `Bmag`/`Bprod`/`Bε`, EVERY updated parameter across all four tensors lies
    within the SAME computable interval `B = u64·Bmag + (u64·Bprod + |lr|·Bε)` of the ideal exact-ℝ gradient
    ascent `θ + lr·derivR e (envR σ) (idx …)`, with the gradient error the FULLY-QUANTIFIED
    `revGradBndF + bridgeBnd` (no opaque bridge). This certifies the whole parameter→parameter map of one
    training step within a single error interval of the exact-ℝ ascent — composing the matrix bound
    `matAxpy_reverse_idx_sup_error` (`W1`, `W2`) with the vector bound `vecAxpy_reverse_idx_sup_error`
    (`b1`, `b2`). SCOPE (the layers above this one, deliberately out of scope here): the ideal target is ascent
    on the SYMBOLIC derivative `derivR e` of the loss the trainer differentiates — connecting `e` to the
    concrete forward-pass/PPO objective (so `derivR e` is *literally* the policy gradient) and iterating the
    local bound across N steps (accumulating how the θ-error perturbs the next step's gradient) remain to be
    composed on top. -/
theorem mlpStep_reverse_bounded_sup_error (e : Expr) (σ : Nat → Float) (lr : Float)
    (W1 gW1 : Array (Array Float)) (b1 gb1 : Array Float)
    (W2 gW2 : Array (Array Float)) (b2 gb2 : Array Float)
    (iW1 iW2 : Nat → Nat → Nat) (ib1 ib2 : Nat → Nat)
    (W : ℝ) (hd : RevGradData e σ W)
    (hgW1 : ∀ i j, i < W1.size → j < (W1[i]!).size → toReal ((gW1[i]!)[j]!) = revGrad σ e (iW1 i j))
    (hgb1 : ∀ i, i < b1.size → toReal (gb1[i]!) = revGrad σ e (ib1 i))
    (hgW2 : ∀ i j, i < W2.size → j < (W2[i]!).size → toReal ((gW2[i]!)[j]!) = revGrad σ e (iW2 i j))
    (hgb2 : ∀ i, i < b2.size → toReal (gb2[i]!) = revGrad σ e (ib2 i))
    (Bmag Bprod Bε : ℝ)
    (hmW1 : ∀ i j, i < W1.size → j < (W1[i]!).size → |toReal ((W1[i]!)[j]!) + toReal (lr * (gW1[i]!)[j]!)| ≤ Bmag)
    (hpW1 : ∀ i j, i < W1.size → j < (W1[i]!).size → |toReal lr * toReal ((gW1[i]!)[j]!)| ≤ Bprod)
    (heW1 : ∀ i j, i < W1.size → j < (W1[i]!).size → revGradBndF σ e (iW1 i j) W + bridgeBnd e σ (iW1 i j) ≤ Bε)
    (hmb1 : ∀ i, i < b1.size → |toReal (b1[i]!) + toReal (lr * gb1[i]!)| ≤ Bmag)
    (hpb1 : ∀ i, i < b1.size → |toReal lr * toReal (gb1[i]!)| ≤ Bprod)
    (heb1 : ∀ i, i < b1.size → revGradBndF σ e (ib1 i) W + bridgeBnd e σ (ib1 i) ≤ Bε)
    (hmW2 : ∀ i j, i < W2.size → j < (W2[i]!).size → |toReal ((W2[i]!)[j]!) + toReal (lr * (gW2[i]!)[j]!)| ≤ Bmag)
    (hpW2 : ∀ i j, i < W2.size → j < (W2[i]!).size → |toReal lr * toReal ((gW2[i]!)[j]!)| ≤ Bprod)
    (heW2 : ∀ i j, i < W2.size → j < (W2[i]!).size → revGradBndF σ e (iW2 i j) W + bridgeBnd e σ (iW2 i j) ≤ Bε)
    (hmb2 : ∀ i, i < b2.size → |toReal (b2[i]!) + toReal (lr * gb2[i]!)| ≤ Bmag)
    (hpb2 : ∀ i, i < b2.size → |toReal lr * toReal (gb2[i]!)| ≤ Bprod)
    (heb2 : ∀ i, i < b2.size → revGradBndF σ e (ib2 i) W + bridgeBnd e σ (ib2 i) ≤ Bε) :
    (∀ i j, i < W1.size → j < (W1[i]!).size →
        |toReal (((matAxpy lr W1 gW1)[i]!)[j]!) - (toReal ((W1[i]!)[j]!) + toReal lr * derivR e (envR σ) (iW1 i j))|
          ≤ u64 * Bmag + (u64 * Bprod + |toReal lr| * Bε))
    ∧ (∀ i, i < b1.size →
        |toReal ((vecAxpy lr b1 gb1)[i]!) - (toReal (b1[i]!) + toReal lr * derivR e (envR σ) (ib1 i))|
          ≤ u64 * Bmag + (u64 * Bprod + |toReal lr| * Bε))
    ∧ (∀ i j, i < W2.size → j < (W2[i]!).size →
        |toReal (((matAxpy lr W2 gW2)[i]!)[j]!) - (toReal ((W2[i]!)[j]!) + toReal lr * derivR e (envR σ) (iW2 i j))|
          ≤ u64 * Bmag + (u64 * Bprod + |toReal lr| * Bε))
    ∧ (∀ i, i < b2.size →
        |toReal ((vecAxpy lr b2 gb2)[i]!) - (toReal (b2[i]!) + toReal lr * derivR e (envR σ) (ib2 i))|
          ≤ u64 * Bmag + (u64 * Bprod + |toReal lr| * Bε)) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro i j hi hj
    exact matAxpy_reverse_idx_sup_error e σ lr W1 gW1 iW1 W hd hgW1 Bmag Bprod Bε hmW1 hpW1 heW1 i j hi hj
  · intro i hi
    exact vecAxpy_reverse_idx_sup_error e σ lr b1 gb1 ib1 W hd hgb1 Bmag Bprod Bε hmb1 hpb1 heb1 i hi
  · intro i j hi hj
    exact matAxpy_reverse_idx_sup_error e σ lr W2 gW2 iW2 W hd hgW2 Bmag Bprod Bε hmW2 hpW2 heW2 i j hi hj
  · intro i hi
    exact vecAxpy_reverse_idx_sup_error e σ lr b2 gb2 ib2 W hd hgb2 Bmag Bprod Bε hmb2 hpb2 heb2 i hi

end Puffer.RL.ReverseNetUpdateBound
