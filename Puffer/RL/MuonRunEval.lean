/-
# C88: the runnable Muon forward-pass certificate — closing the Muon-side Bool slate

C81 (`RunConstEval`) made the Muon WEIGHT run all-Bool (`muonTrainRun_all_finite_runnable`:
region/`checkAbsLe` data checks + ONE budget check on `runBoundF (muonStepCF …)`), but C64's
forward theorem `muonTrainRun_forward_all_finite` still consumes the ℝ-side budget

    `dotBound (min |x| |w|) (max Bx (runBound (muonStepC …) n B₀)) ≤ overflowBound`

— the ONE remaining numeric hypothesis on the Muon path with no runnable evaluator. This module
supplies it, closing the Muon-side slate: EVERY numeric hypothesis of both C64 Muon theorems
(weights AND forward pass) is now a runtime `Bool`.

* `muonFwdBoundF` — the forward-budget evaluator: C78's `dotBoundF` at the Float `max` of the
  input bound and the evaluated run budget `runBoundF (muonStepCF …) n B0F`. The `max` needs NO
  slack: `toReal_max` is an EXACT trusted-base axiom (the order-theoretic `maxOfLe` instance
  computes a bit-exact result), so the componentwise dominations pass straight through
  `max_le_max`.
* `muonFwdBound_le_overflow_of_check` — one `checkLe` on the evaluated forward budget certifies
  the exact ℝ-side hypothesis `muonTrainRun_forward_all_finite` consumes (domination → C70
  comparison soundness → `capF_le`).
* **`muonTrainRun_forward_all_finite_runnable`** — the forward pass `FiniteBound.dotF x (weights at step m)`
  is overflow-free at every step `m ≤ n`, from Bools alone.
* **`muonRun_fully_finite_runnable`** (capstone) — the COMBINED certificate: ELEVEN Bools (seven
  data checks, two threshold-nonnegativity checks, the weight-budget check, the forward-budget
  check) ⟹ every weight at every step `m ≤ n` is finite AND every step's forward pass is finite.
  The weight half is exactly C81's theorem; the forward half is this module's.
* `sgdFwdBoundF` / `trainRun_forward_all_finite_runnable` — the SGD twin (identical pattern;
  included so the forward-pass gap closes for BOTH C64 runs, disclosed as beyond the strict
  Muon directive).

**Scope (honestly disclosed).** (i) FINITE horizon `n`: `runBoundF (muonStepCF …) n B0F` grows
with `n` (each step inflates by `slackF` and adds the gradient-through-NS constant) — the Muon
path has NO contraction to exploit: the NS-circuit update ADDS `lr·ns(grad)` with no damping of
the weight term, so C67's horizon-free trick (the weight-decay factor `|1−wd| < 1` making the
budget map a contraction) has no analogue here; a horizon-free Muon budget would need the
C59/C62 BALL machinery (op-norm invariance of the NS iterate), which lives at the matrix level,
not this scalar per-coordinate circuit. (ii) The evaluators inherit C78's upward slack
(sound-not-complete, ≈0.1% per `slackF` factor — `muonStepCF` carries 11 raw `slackF`
occurrences per step (counted as 9 budget units in C81's convention, which counts the shared
σ² sub-budget once); these compound through
the run; negligible against `capF = 1e300`). (iii) The bounds fed to the ℝ theorems are `toReal`
of the CHECKED Float thresholds — the harness picks thresholds, the Bools bind the data to them;
nothing numeric is caller-trusted. What remains caller-supplied is exactly C64's MODEL scope
(not numbers): the fixed-batch repeated step, the self-contained linear-layer `gradW` backward,
and the scalar per-coordinate NS circuit. NO new axiom: everything routes through C78's
`slackF_key`/`capF_le`, C81's `muonStepCF_dominates`, C70's checkers, and the exact `toReal_max`.
-/
import Puffer.RL.RunConstEval
open Puffer.FloatR
open Puffer.RL.FiniteBound (overflowBound dotBound)
open Puffer.RL.WholeRunFinite (runBound sgdStepC sgdStepC_nonneg muonStepC muonStepC_nonneg
  trainRun muonTrainRun trainRun_forward_all_finite muonTrainRun_forward_all_finite)
open Puffer.RL.MarginCheck (checkLe checkLe_sound checkAbsLe checkAbsLe_sound checkRegion
  checkRegion_sound)
open Puffer.RL.BudgetEval (capF capF_le dotBoundF runBoundF runBoundF_dominates
  dotBound_le_overflow_of_check)
open Puffer.RL.RunConstEval (sgdStepCF sgdStepCF_dominates muonStepCF muonStepCF_dominates
  trainRun_all_finite_runnable muonTrainRun_all_finite_runnable)

namespace Puffer.RL.MuonRunEval

/-! ### The forward-budget evaluators -/

/-- **The Muon forward-budget evaluator**: C78's `dotBoundF` at the Float `max` of the input
    bound and the evaluated `n`-step Muon run budget. `k` is the dot length
    (`min x.length w.length` at the use site). The `max` is exact (`toReal_max`), so it spends
    no slack. -/
def muonFwdBoundF (BxF B0F BgsF BlrF BaF BbF BcF : Float) (n k : Nat) : Float :=
  dotBoundF (max BxF (runBoundF (muonStepCF BxF BgsF BlrF BaF BbF BcF) n B0F)) k

/-- The SGD twin: `dotBoundF` at the `max` of the input bound and the evaluated SGD run budget. -/
def sgdFwdBoundF (BxF B0F BgsF BlrF : Float) (n k : Nat) : Float :=
  dotBoundF (max BxF (runBoundF (sgdStepCF BxF BgsF BlrF) n B0F)) k

/-! ### The runnable discharges of C64's forward-budget hypotheses -/

/-- **The Muon forward discharge**: one passing `checkLe` on `muonFwdBoundF` certifies the exact
    ℝ-side hypothesis of C64's `muonTrainRun_forward_all_finite` — `dotBound k (max (toReal BxF)
    (runBound (muonStepC …) n (toReal B0F))) ≤ overflowBound`. The `max`'s domination is
    componentwise via the EXACT `toReal_max` (no slack); the run-budget component rides C81's
    `muonStepCF_dominates` through C78's `runBoundF_dominates`. -/
theorem muonFwdBound_le_overflow_of_check (BxF B0F BgsF BlrF BaF BbF BcF : Float) (n k : Nat)
    (hBx0 : (0 : ℝ) ≤ toReal BxF) (hB00 : (0 : ℝ) ≤ toReal B0F)
    (hgs0 : (0 : ℝ) ≤ toReal BgsF) (hlr0 : (0 : ℝ) ≤ toReal BlrF)
    (ha0 : (0 : ℝ) ≤ toReal BaF) (hb0 : (0 : ℝ) ≤ toReal BbF) (hc0 : (0 : ℝ) ≤ toReal BcF)
    (hchk : checkLe (muonFwdBoundF BxF B0F BgsF BlrF BaF BbF BcF n k) capF = true) :
    dotBound k (max (toReal BxF)
      (runBound (muonStepC (toReal BxF) (toReal BgsF) (toReal BlrF) (toReal BaF) (toReal BbF)
        (toReal BcF)) n (toReal B0F))) ≤ overflowBound := by
  have hC0 : 0 ≤ muonStepC (toReal BxF) (toReal BgsF) (toReal BlrF) (toReal BaF) (toReal BbF)
      (toReal BcF) := muonStepC_nonneg _ _ _ _ _ _ hBx0 hgs0 hlr0 ha0 hb0 hc0
  have hCc : muonStepC (toReal BxF) (toReal BgsF) (toReal BlrF) (toReal BaF) (toReal BbF)
      (toReal BcF) ≤ toReal (muonStepCF BxF BgsF BlrF BaF BbF BcF) :=
    muonStepCF_dominates BxF BgsF BlrF BaF BbF BcF _ _ _ _ _ _
      hBx0 le_rfl hgs0 le_rfl hlr0 le_rfl ha0 le_rfl hb0 le_rfl hc0 le_rfl
  have hrun : runBound (muonStepC (toReal BxF) (toReal BgsF) (toReal BlrF) (toReal BaF)
        (toReal BbF) (toReal BcF)) n (toReal B0F)
      ≤ toReal (runBoundF (muonStepCF BxF BgsF BlrF BaF BbF BcF) n B0F) :=
    runBoundF_dominates _ _ hC0 hCc n B0F _ hB00 le_rfl
  have hmax0 : 0 ≤ max (toReal BxF)
      (runBound (muonStepC (toReal BxF) (toReal BgsF) (toReal BlrF) (toReal BaF) (toReal BbF)
        (toReal BcF)) n (toReal B0F)) := hBx0.trans (le_max_left _ _)
  have hmaxdom : max (toReal BxF)
      (runBound (muonStepC (toReal BxF) (toReal BgsF) (toReal BlrF) (toReal BaF) (toReal BbF)
        (toReal BcF)) n (toReal B0F))
      ≤ toReal (max BxF (runBoundF (muonStepCF BxF BgsF BlrF BaF BbF BcF) n B0F)) := by
    rw [toReal_max]
    exact max_le_max le_rfl hrun
  have hchk' : checkLe (dotBoundF
      (max BxF (runBoundF (muonStepCF BxF BgsF BlrF BaF BbF BcF) n B0F)) k) capF = true := hchk
  exact dotBound_le_overflow_of_check _ _ k hmax0 hmaxdom hchk'

/-- The SGD twin of the forward discharge. -/
theorem sgdFwdBound_le_overflow_of_check (BxF B0F BgsF BlrF : Float) (n k : Nat)
    (hBx0 : (0 : ℝ) ≤ toReal BxF) (hB00 : (0 : ℝ) ≤ toReal B0F)
    (hgs0 : (0 : ℝ) ≤ toReal BgsF) (hlr0 : (0 : ℝ) ≤ toReal BlrF)
    (hchk : checkLe (sgdFwdBoundF BxF B0F BgsF BlrF n k) capF = true) :
    dotBound k (max (toReal BxF)
      (runBound (sgdStepC (toReal BxF) (toReal BgsF) (toReal BlrF)) n (toReal B0F)))
      ≤ overflowBound := by
  have hC0 : 0 ≤ sgdStepC (toReal BxF) (toReal BgsF) (toReal BlrF) :=
    sgdStepC_nonneg _ _ _ hBx0 hgs0 hlr0
  have hCc : sgdStepC (toReal BxF) (toReal BgsF) (toReal BlrF)
      ≤ toReal (sgdStepCF BxF BgsF BlrF) :=
    sgdStepCF_dominates BxF BgsF BlrF _ _ _ hBx0 le_rfl hgs0 le_rfl hlr0 le_rfl
  have hrun : runBound (sgdStepC (toReal BxF) (toReal BgsF) (toReal BlrF)) n (toReal B0F)
      ≤ toReal (runBoundF (sgdStepCF BxF BgsF BlrF) n B0F) :=
    runBoundF_dominates _ _ hC0 hCc n B0F _ hB00 le_rfl
  have hmax0 : 0 ≤ max (toReal BxF)
      (runBound (sgdStepC (toReal BxF) (toReal BgsF) (toReal BlrF)) n (toReal B0F)) :=
    hBx0.trans (le_max_left _ _)
  have hmaxdom : max (toReal BxF)
      (runBound (sgdStepC (toReal BxF) (toReal BgsF) (toReal BlrF)) n (toReal B0F))
      ≤ toReal (max BxF (runBoundF (sgdStepCF BxF BgsF BlrF) n B0F)) := by
    rw [toReal_max]
    exact max_le_max le_rfl hrun
  have hchk' : checkLe (dotBoundF
      (max BxF (runBoundF (sgdStepCF BxF BgsF BlrF) n B0F)) k) capF = true := hchk
  exact dotBound_le_overflow_of_check _ _ k hmax0 hmaxdom hchk'

/-! ### The runnable forward capstones -/

/-- **THE RUNNABLE MUON FORWARD CERTIFICATE**: the forward pass `FiniteBound.dotF x (weights at step m)` is
    overflow-free at every step `m ≤ n` of the Muon run, from Bools alone — the data checks bind
    the run to the thresholds, and ONE forward-budget check discharges C64's `hfwd`. -/
theorem muonTrainRun_forward_all_finite_runnable (x w : List Float) (gseed lr a b c : Float)
    (n : Nat) (BxF B0F BgsF BlrF BaF BbF BcF : Float)
    (hx : checkRegion x BxF = true) (hw : checkRegion w B0F = true)
    (hgs : checkAbsLe gseed BgsF = true) (hlr : checkAbsLe lr BlrF = true)
    (ha : checkAbsLe a BaF = true) (hb : checkAbsLe b BbF = true)
    (hc : checkAbsLe c BcF = true)
    (hBx0 : checkLe 0.0 BxF = true) (hB00 : checkLe 0.0 B0F = true)
    (hfwd : checkLe (muonFwdBoundF BxF B0F BgsF BlrF BaF BbF BcF n (min x.length w.length))
      capF = true) :
    ∀ m, m ≤ n → (FiniteBound.dotF x (muonTrainRun x gseed lr a b c m w)).isFinite = true := by
  have hBx0' : (0 : ℝ) ≤ toReal BxF := by
    have := checkLe_sound hBx0; rwa [toReal_zeroLit] at this
  have hB00' : (0 : ℝ) ≤ toReal B0F := by
    have := checkLe_sound hB00; rwa [toReal_zeroLit] at this
  have hgs' := checkAbsLe_sound hgs
  have hlr' := checkAbsLe_sound hlr
  have ha' := checkAbsLe_sound ha
  have hb' := checkAbsLe_sound hb
  have hc' := checkAbsLe_sound hc
  have hgs0 : (0 : ℝ) ≤ toReal BgsF := (abs_nonneg _).trans hgs'
  have hlr0 : (0 : ℝ) ≤ toReal BlrF := (abs_nonneg _).trans hlr'
  have ha0 : (0 : ℝ) ≤ toReal BaF := (abs_nonneg _).trans ha'
  have hb0 : (0 : ℝ) ≤ toReal BbF := (abs_nonneg _).trans hb'
  have hc0 : (0 : ℝ) ≤ toReal BcF := (abs_nonneg _).trans hc'
  exact muonTrainRun_forward_all_finite x w gseed lr a b c n (toReal BxF) (toReal B0F)
    (toReal BgsF) (toReal BlrF) (toReal BaF) (toReal BbF) (toReal BcF) hBx0' hB00'
    (checkRegion_sound hx) (checkRegion_sound hw) hgs' hlr' ha' hb' hc'
    (muonFwdBound_le_overflow_of_check BxF B0F BgsF BlrF BaF BbF BcF n _
      hBx0' hB00' hgs0 hlr0 ha0 hb0 hc0 hfwd)

/-- The SGD twin of the runnable forward certificate. -/
theorem trainRun_forward_all_finite_runnable (x w : List Float) (gseed lr : Float) (n : Nat)
    (BxF B0F BgsF BlrF : Float)
    (hx : checkRegion x BxF = true) (hw : checkRegion w B0F = true)
    (hgs : checkAbsLe gseed BgsF = true) (hlr : checkAbsLe lr BlrF = true)
    (hBx0 : checkLe 0.0 BxF = true) (hB00 : checkLe 0.0 B0F = true)
    (hfwd : checkLe (sgdFwdBoundF BxF B0F BgsF BlrF n (min x.length w.length)) capF = true) :
    ∀ m, m ≤ n → (FiniteBound.dotF x (trainRun x gseed lr m w)).isFinite = true := by
  have hBx0' : (0 : ℝ) ≤ toReal BxF := by
    have := checkLe_sound hBx0; rwa [toReal_zeroLit] at this
  have hB00' : (0 : ℝ) ≤ toReal B0F := by
    have := checkLe_sound hB00; rwa [toReal_zeroLit] at this
  have hgs' := checkAbsLe_sound hgs
  have hlr' := checkAbsLe_sound hlr
  have hgs0 : (0 : ℝ) ≤ toReal BgsF := (abs_nonneg _).trans hgs'
  have hlr0 : (0 : ℝ) ≤ toReal BlrF := (abs_nonneg _).trans hlr'
  exact trainRun_forward_all_finite x w gseed lr n (toReal BxF) (toReal B0F) (toReal BgsF)
    (toReal BlrF) hBx0' hB00' (checkRegion_sound hx) (checkRegion_sound hw) hgs' hlr'
    (sgdFwdBound_le_overflow_of_check BxF B0F BgsF BlrF n _ hBx0' hB00' hgs0 hlr0 hfwd)

/-! ### The combined Muon capstone -/

/-- **THE FULLY-RUNNABLE MUON RUN CERTIFICATE** (the Muon-side slate, closed): ELEVEN runtime
    Bools — seven data checks (`x`/`w` regions, seed, step-size, three NS coefficients), two
    threshold-nonnegativity checks, the weight-budget check (C81's), and the forward-budget
    check (this module's) — certify the COMPLETE C64 Muon conclusion: every weight at every
    step `m ≤ n` is finite AND every step's forward pass `FiniteBound.dotF x (weights at step m)` is
    finite. Nothing numeric is caller-trusted. -/
theorem muonRun_fully_finite_runnable (x w : List Float) (gseed lr a b c : Float) (n : Nat)
    (BxF B0F BgsF BlrF BaF BbF BcF : Float)
    (hx : checkRegion x BxF = true) (hw : checkRegion w B0F = true)
    (hgs : checkAbsLe gseed BgsF = true) (hlr : checkAbsLe lr BlrF = true)
    (ha : checkAbsLe a BaF = true) (hb : checkAbsLe b BbF = true)
    (hc : checkAbsLe c BcF = true)
    (hBx0 : checkLe 0.0 BxF = true) (hB00 : checkLe 0.0 B0F = true)
    (hbudget : checkLe (runBoundF (muonStepCF BxF BgsF BlrF BaF BbF BcF) n B0F) capF = true)
    (hfwd : checkLe (muonFwdBoundF BxF B0F BgsF BlrF BaF BbF BcF n (min x.length w.length))
      capF = true) :
    (∀ m, m ≤ n → ∀ u ∈ muonTrainRun x gseed lr a b c m w, u.isFinite = true) ∧
    (∀ m, m ≤ n → (FiniteBound.dotF x (muonTrainRun x gseed lr a b c m w)).isFinite = true) :=
  ⟨muonTrainRun_all_finite_runnable x w gseed lr a b c n BxF B0F BgsF BlrF BaF BbF BcF
      hx hw hgs hlr ha hb hc hBx0 hB00 hbudget,
    muonTrainRun_forward_all_finite_runnable x w gseed lr a b c n BxF B0F BgsF BlrF BaF BbF BcF
      hx hw hgs hlr ha hb hc hBx0 hB00 hfwd⟩

/-! ### Measured demos (the repo's actual first-row NS schedule `(4.0848, −6.8946, 2.9270)`
    from `Puffer/Float/Muon.lean`, dominated by thresholds `4.1`/`6.9`/`2.93`; `n = 3` steps,
    dot length `k = 2`). -/

/-- info: true -/
#guard_msgs in #eval checkLe (muonFwdBoundF 1.0 1.0 1.0 0.1 4.1 6.9 2.93 3 2) capF

-- The measured budget values themselves, pinned (RunnableDemo practice).
/-- info: 5.225911 -/
#guard_msgs in #eval runBoundF (muonStepCF 1.0 1.0 0.1 4.1 6.9 2.93) 3 1.0

/-- info: 54.756947 -/
#guard_msgs in #eval muonFwdBoundF 1.0 1.0 1.0 0.1 4.1 6.9 2.93 3 2

-- The repo's actual first NS row `(4.0848, −6.8946, 2.9270)` is inside the demo thresholds.
/-- info: true -/
#guard_msgs in #eval checkAbsLe 4.0848 4.1 && checkAbsLe (-6.8946) 6.9 && checkAbsLe 2.9270 2.93

-- The gate genuinely rejects: an input bound past the cap square-overflows the dot budget.
/-- info: false -/
#guard_msgs in #eval checkLe (muonFwdBoundF 1e200 1.0 1.0 0.1 4.1 6.9 2.93 3 2) capF

/-! ### Non-vacuity: the combined capstone composes (Bool hypotheses in hypothesis form —
    Float comparison is kernel-opaque, the C70/C79 split; the harness's native evaluation
    supplies the witnesses). -/

example (x w : List Float) (gseed lr a b c : Float) (BxF B0F BgsF BlrF BaF BbF BcF : Float)
    (hx : checkRegion x BxF = true) (hw : checkRegion w B0F = true)
    (hgs : checkAbsLe gseed BgsF = true) (hlr : checkAbsLe lr BlrF = true)
    (ha : checkAbsLe a BaF = true) (hb : checkAbsLe b BbF = true)
    (hc : checkAbsLe c BcF = true)
    (hBx0 : checkLe 0.0 BxF = true) (hB00 : checkLe 0.0 B0F = true)
    (hbudget : checkLe (runBoundF (muonStepCF BxF BgsF BlrF BaF BbF BcF) 100 B0F) capF = true)
    (hfwd : checkLe (muonFwdBoundF BxF B0F BgsF BlrF BaF BbF BcF 100 (min x.length w.length))
      capF = true) :
    (FiniteBound.dotF x (muonTrainRun x gseed lr a b c 100 w)).isFinite = true :=
  (muonRun_fully_finite_runnable x w gseed lr a b c 100 BxF B0F BgsF BlrF BaF BbF BcF
    hx hw hgs hlr ha hb hc hBx0 hB00 hbudget hfwd).2 100 le_rfl

end Puffer.RL.MuonRunEval
