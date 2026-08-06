/-
Discharging the reverse-mode primal bridge `hbridge : revE_RF e σ 1 k = revE_R e (envR σ) 1 k`.

`bsweepR_reads_derivR` (Stage C) — the reverse ℝ sweep reads off `derivR` — was CONDITIONAL on `hbridge`,
the equality between the reverse pass using the tape's ROUNDED Float edge weights (`revE_RF`: `toReal (evalF
b σ)`, `toReal (Float.exp …)`, Float kink signs) and the one using the IDEAL real weights (`revE_R`:
`evalR b σ`, `Real.exp`, real kink signs). `revE_RF` and `revE_R` are structurally identical folds; they
differ ONLY on the continuous edges (`mul`/`exp`/`log`) and the kink signs (`relu`/`max`/`min`). `add`/`sub`
(weights `toReal (±1.0) = ±1`), `scale` (both `toReal c`), `var`/`const` all match EXACTLY.

This file discharges `hbridge` into concrete, local, checkable conditions:

  • `IsLinear` + `revE_RF_eq_revE_R_of_linear` — for the LINEAR fragment (`var`/`const`/`add`/`sub`/`scale`,
    the linear layers/readouts) `hbridge` holds UNCONDITIONALLY: no `mul`/`exp`/`log`/kink, so every edge
    weight matches. `hbridge` is a THEOREM, not an assumption, for linear objectives.
  • `BridgeExact` + `revE_RF_eq_revE_R_of_bridgeExact` — for ALL ops, `hbridge` holds given per-node
    conditions: primal exactness `toReal (evalF · σ) = evalR · (envR σ)` on each `mul`/`exp`/`log` edge, and
    kink-branch agreement on each `relu`/`max`/`min`. `bridge_of_bridgeExact` is the `ā = 1` corollary that
    directly supplies `hbridge`.

The `mul`/`exp`/`log` conditions demand the primal round exactly — true absent rounding, false in general;
the under-rounding regime is the BOUNDED bridge below (`revE_RF_sub_revE_R_le`): `|revE_RF − revE_R| ≤
|ā|·bridgeBnd` for EVERY op under `WD` alone (no `hbridge`), covering the full PPO objective.

Axiom scope: the EXACT-discharge theorems (`revE_RF_eq_revE_R_of_linear`/`_of_bridgeExact` and their `bridge_
of_*` corollaries) are pure/structural — only the sanctioned `toReal` literal axioms, NO float-model axioms.
The BOUNDED-bridge theorems (`revE_RF_sub_revE_R_le`/`bridge_gap_le`) additionally use the Float (1+δ) model
axioms (`add`/`sub`/`mul`/`div`/`exp`/`log_model` via `evalF_error`/`divApprox_error`) — the same trusted base
as the rest of the error tower; still no `sorry`.
-/
import Puffer.Float.ADReverse

namespace Puffer.FloatR.ADReverse

open Puffer.FloatR (toReal)
open Puffer.FloatR.ADR (Expr evalF evalR envR derivR evalErrBnd evalErrBnd_nonneg WD evalF_error occurs)

/-! ### The linear fragment: `hbridge` unconditionally -/

/-- The linear fragment: `var`/`const`/`add`/`sub`/`scale` only — no `mul`/`exp`/`log`/kink. On it every
    reverse edge weight matches (`±1`, `toReal c`), so `revE_RF = revE_R` with NO hypothesis. -/
def IsLinear : Expr → Prop
  | .var _ => True
  | .const _ => True
  | .add a b => IsLinear a ∧ IsLinear b
  | .sub a b => IsLinear a ∧ IsLinear b
  | .scale _ a => IsLinear a
  | _ => False

/-- **`hbridge` holds unconditionally on the linear fragment.** -/
theorem revE_RF_eq_revE_R_of_linear (e : Expr) (σ : Nat → Float) :
    IsLinear e → ∀ (ā : ℝ) (k : Nat), revE_RF e σ ā k = revE_R e (envR σ) ā k := by
  induction e with
  | var i => intro _ ā k; rfl
  | const c => intro _ ā k; rfl
  | add a b iha ihb => intro h ā k
                       obtain ⟨ha, hb⟩ := h
                       simp only [revE_RF, revE_R, toReal_oneLit, mul_one]
                       rw [iha ha, ihb hb]
  | sub a b iha ihb => intro h ā k
                       obtain ⟨ha, hb⟩ := h
                       simp only [revE_RF, revE_R, toReal_oneLit, mul_one, toReal_neg, mul_neg_one]
                       rw [iha ha, ihb hb]
  | scale c a iha => intro h ā k
                     simp only [revE_RF, revE_R]
                     rw [iha h]
  | exp a iha => intro h; exact absurd h (by simp [IsLinear])
  | log a iha => intro h; exact absurd h (by simp [IsLinear])
  | mul a b iha ihb => intro h; exact absurd h (by simp [IsLinear])
  | relu a iha => intro h; exact absurd h (by simp [IsLinear])
  | max a b iha ihb => intro h; exact absurd h (by simp [IsLinear])
  | min a b iha ihb => intro h; exact absurd h (by simp [IsLinear])

/-- **Reverse-pass locality (Float-faithful sweep).** The reverse-mode pass over the tape's ACTUAL
    rounded Float edge weights, `revE_RF`, routes ZERO gradient to any variable `k` that does not
    syntactically occur in `e` — for every seed adjoint `ā`, and *unconditionally* (no `WD`, no
    primal-exactness): the Float edge weights are arbitrary reals, yet a leaf never present in the
    circuit receives no contribution. This is the reverse-mode analogue of
    `derivR_eq_zero_of_not_occurs` (the same locality for the ideal ℝ derivative), but for the
    rounded reverse SWEEP rather than the ideal derivative — so it certifies that primal rounding
    cannot manufacture spurious gradient flow to absent inputs (the trainer's gradient sparsity
    pattern is preserved bit-for-bit under rounding). Proved by one structural induction over all
    `Expr` nodes; at each interior node the child IH is instantiated at the node's own
    (rounded, adjoint-scaled) Float edge weight, so the recursion collapses to `0`. The `¬ occurs`
    hypothesis is load-bearing — an occurring variable receives nonzero flow (`revE_RF (var 0) σ 1 0 = 1`). -/
theorem revE_RF_eq_zero_of_not_occurs (e : Expr) (σ : Nat → Float) (k : Nat) :
    ¬ occurs k e → ∀ (ā : ℝ), revE_RF e σ ā k = 0 := by
  induction e with
  | var i => intro h ā
             simp only [occurs] at h
             simp only [revE_RF]
             rw [if_neg (fun hk : k = i => h hk.symm)]
  | const c => intro _ ā; simp only [revE_RF]
  | add a b iha ihb => intro h ā
                       simp only [occurs, not_or] at h
                       simp only [revE_RF, iha h.1, ihb h.2, add_zero]
  | sub a b iha ihb => intro h ā
                       simp only [occurs, not_or] at h
                       simp only [revE_RF, iha h.1, ihb h.2, add_zero]
  | mul a b iha ihb => intro h ā
                       simp only [occurs, not_or] at h
                       simp only [revE_RF, iha h.1, ihb h.2, add_zero]
  | scale c a iha => intro h ā
                     simp only [occurs] at h
                     simp only [revE_RF, iha h]
  | exp a iha => intro h ā
                 simp only [occurs] at h
                 simp only [revE_RF, iha h]
  | log a iha => intro h ā
                 simp only [occurs] at h
                 simp only [revE_RF, iha h]
  | relu a iha => intro h ā
                  simp only [occurs] at h
                  simp only [revE_RF, iha h]
  | max a b iha ihb => intro h ā
                       simp only [occurs, not_or] at h
                       simp only [revE_RF, iha h.1, ihb h.2, add_zero]
  | min a b iha ihb => intro h ā
                       simp only [occurs, not_or] at h
                       simp only [revE_RF, iha h.1, ihb h.2, add_zero]

/-! ### The general per-node conditions: `BridgeExact` -/

/-- Per-node conditions under which `hbridge` holds exactly: on each `mul`/`exp`/`log` edge the primal is
    computed with NO rounding (`toReal (evalF · σ) = evalR · (envR σ)`, `Float.exp = Real.exp`, `1.0/· =
    1/·`), and on each `relu`/`max`/`min` the Float kink branch agrees with the real one. `add`/`sub`/`scale`
    /`var`/`const` need nothing (their edge weights always match). -/
def BridgeExact : Expr → (Nat → Float) → Prop
  | .var _, _ => True
  | .const _, _ => True
  | .add a b, σ => BridgeExact a σ ∧ BridgeExact b σ
  | .sub a b, σ => BridgeExact a σ ∧ BridgeExact b σ
  | .mul a b, σ => BridgeExact a σ ∧ BridgeExact b σ
      ∧ toReal (evalF a σ) = evalR a (envR σ) ∧ toReal (evalF b σ) = evalR b (envR σ)
  | .scale _ a, σ => BridgeExact a σ
  | .exp a, σ => BridgeExact a σ ∧ toReal (Float.exp (evalF a σ)) = Real.exp (evalR a (envR σ))
  | .log a, σ => BridgeExact a σ ∧ toReal (1.0 / evalF a σ) = 1 / evalR a (envR σ)
  | .relu a, σ => BridgeExact a σ
      ∧ toReal (if evalF a σ < 0.0 then (0.0 : Float) else 1.0)
          = (if 0 < evalR a (envR σ) then (1 : ℝ) else 0)
  | .max a b, σ => BridgeExact a σ ∧ BridgeExact b σ
      ∧ toReal (if evalF a σ ≤ evalF b σ then (0.0 : Float) else 1.0)
          = (if evalR a (envR σ) ≤ evalR b (envR σ) then (0 : ℝ) else 1)
      ∧ toReal (if evalF a σ ≤ evalF b σ then (1.0 : Float) else 0.0)
          = (if evalR a (envR σ) ≤ evalR b (envR σ) then (1 : ℝ) else 0)
  | .min a b, σ => BridgeExact a σ ∧ BridgeExact b σ
      ∧ toReal (if evalF a σ ≤ evalF b σ then (1.0 : Float) else 0.0)
          = (if evalR a (envR σ) ≤ evalR b (envR σ) then (1 : ℝ) else 0)
      ∧ toReal (if evalF a σ ≤ evalF b σ then (0.0 : Float) else 1.0)
          = (if evalR a (envR σ) ≤ evalR b (envR σ) then (0 : ℝ) else 1)

/-- **`hbridge` holds given the per-node `BridgeExact` conditions** (for all ops). Structural induction:
    `revE_RF` and `revE_R` are the same fold, and each `BridgeExact` conjunct rewrites one side's edge
    weight to the other's. -/
theorem revE_RF_eq_revE_R_of_bridgeExact (e : Expr) (σ : Nat → Float) :
    BridgeExact e σ → ∀ (ā : ℝ) (k : Nat), revE_RF e σ ā k = revE_R e (envR σ) ā k := by
  induction e with
  | var i => intro _ ā k; rfl
  | const c => intro _ ā k; rfl
  | add a b iha ihb => intro h ā k
                       obtain ⟨ha, hb⟩ := h
                       simp only [revE_RF, revE_R, toReal_oneLit, mul_one]
                       rw [iha ha, ihb hb]
  | sub a b iha ihb => intro h ā k
                       obtain ⟨ha, hb⟩ := h
                       simp only [revE_RF, revE_R, toReal_oneLit, mul_one, toReal_neg, mul_neg_one]
                       rw [iha ha, ihb hb]
  | mul a b iha ihb => intro h ā k
                       obtain ⟨ha, hb, hea, heb⟩ := h
                       simp only [revE_RF, revE_R]
                       rw [hea, heb, iha ha, ihb hb]
  | scale c a iha => intro h ā k
                     simp only [revE_RF, revE_R]
                     rw [iha h]
  | exp a iha => intro h ā k
                 obtain ⟨ha, hexp⟩ := h
                 simp only [revE_RF, revE_R]
                 rw [hexp, iha ha]
  | log a iha => intro h ā k
                 obtain ⟨ha, hlog⟩ := h
                 simp only [revE_RF, revE_R]
                 rw [hlog, iha ha]
  | relu a iha => intro h ā k
                  obtain ⟨ha, hk⟩ := h
                  simp only [revE_RF, revE_R]
                  rw [hk, iha ha]
  | max a b iha ihb => intro h ā k
                       obtain ⟨ha, hb, hka, hkb⟩ := h
                       simp only [revE_RF, revE_R]
                       rw [hka, hkb, iha ha, ihb hb]
  | min a b iha ihb => intro h ā k
                       obtain ⟨ha, hb, hka, hkb⟩ := h
                       simp only [revE_RF, revE_R]
                       rw [hka, hkb, iha ha, ihb hb]

/-- **`hbridge` from `BridgeExact`** (the `ā = 1` corollary that directly discharges the hypothesis of
    `bsweepR_reads_derivR`). -/
theorem bridge_of_bridgeExact (e : Expr) (σ : Nat → Float) (k : Nat) (h : BridgeExact e σ) :
    revE_RF e σ 1 k = revE_R e (envR σ) 1 k :=
  revE_RF_eq_revE_R_of_bridgeExact e σ h 1 k

/-- **`hbridge` from `IsLinear`** (the `ā = 1` corollary; no primal-exactness needed). -/
theorem bridge_of_linear (e : Expr) (σ : Nat → Float) (k : Nat) (h : IsLinear e) :
    revE_RF e σ 1 k = revE_R e (envR σ) 1 k :=
  revE_RF_eq_revE_R_of_linear e σ h 1 k

/-! ### The BOUNDED bridge (all ops): `hbridge` under rounding, quantified

The exact equality `revE_RF = revE_R` fails under primal rounding, but the DIFFERENCE is bounded. Each op
recurses on its children with a scaled adjoint whose edge weight is rounded (`revE_RF`) vs ideal (`revE_R`);
the difference telescopes into (a) the child's own bridge error and (b) a cross term = (edge-weight error)·
(ideal child derivative), via `revE_R`'s linearity in the adjoint (`revE_R_eq`). `unary_step` packages that
telescoping. The per-op edge-weight errors: `mul`/`exp` reduce to `evalErrBnd` (`evalF_error`), `log` to a
division bound (`divApprox_error`), `add`/`sub`/`scale` are exact (0), and — under `WD`'s away-from-kink
clearance — `relu`/`max`/`min` kink signs AGREE (also 0). So `revE_RF_sub_revE_R_le` proves
`|revE_RF − revE_R| ≤ |ā|·bridgeBnd` for EVERY op under `WD` — no `hbridge`, covering the full RL objective
(softmax `exp`/ratio `log`/`relu`). `bridge_gap_le` is the `ā = 1` corollary against `derivR`. -/

/-- **The telescoping step for one child.** Given the child's bridge IH and an edge-weight error bound
    `|wF − wR| ≤ ewErr`, the scaled-adjoint difference splits into the child error (`|ā·wF|·Ba`) plus the
    cross term (`|ā|·ewErr·|derivR child|`), the latter exact by `revE_R`'s adjoint-linearity. -/
theorem unary_step (a : Expr) (σ : Nat → Float) (k : Nat) (ā wF wR Ba ewErr : ℝ)
    (ih : ∀ ā', |revE_RF a σ ā' k - revE_R a (envR σ) ā' k| ≤ |ā'| * Ba)
    (hew : |wF - wR| ≤ ewErr) :
    |revE_RF a σ (ā * wF) k - revE_R a (envR σ) (ā * wR) k|
      ≤ |ā| * (|wF| * Ba + ewErr * |derivR a (envR σ) k|) := by
  have hcross : revE_R a (envR σ) (ā * wF) k - revE_R a (envR σ) (ā * wR) k
      = ā * (wF - wR) * derivR a (envR σ) k := by rw [revE_R_eq, revE_R_eq]; ring
  have hsplit : |revE_RF a σ (ā * wF) k - revE_R a (envR σ) (ā * wR) k|
      ≤ |revE_RF a σ (ā * wF) k - revE_R a (envR σ) (ā * wF) k|
        + |ā * (wF - wR) * derivR a (envR σ) k| := by
    rw [← hcross, show revE_RF a σ (ā * wF) k - revE_R a (envR σ) (ā * wR) k
        = (revE_RF a σ (ā * wF) k - revE_R a (envR σ) (ā * wF) k)
          + (revE_R a (envR σ) (ā * wF) k - revE_R a (envR σ) (ā * wR) k) from by ring]
    exact abs_add_le _ _
  refine hsplit.trans ?_
  have h1 := ih (ā * wF); rw [abs_mul] at h1
  have h2 : |ā * (wF - wR) * derivR a (envR σ) k| ≤ |ā| * (ewErr * |derivR a (envR σ) k|) := by
    rw [abs_mul, abs_mul]
    have hstep := mul_le_mul_of_nonneg_right hew (abs_nonneg (derivR a (envR σ) k))
    calc |ā| * |wF - wR| * |derivR a (envR σ) k|
        = |ā| * (|wF - wR| * |derivR a (envR σ) k|) := by ring
      _ ≤ |ā| * (ewErr * |derivR a (envR σ) k|) := mul_le_mul_of_nonneg_left hstep (abs_nonneg ā)
  calc |revE_RF a σ (ā * wF) k - revE_R a (envR σ) (ā * wF) k|
        + |ā * (wF - wR) * derivR a (envR σ) k|
      ≤ |ā| * |wF| * Ba + |ā| * (ewErr * |derivR a (envR σ) k|) := add_le_add h1 h2
    _ = |ā| * (|wF| * Ba + ewErr * |derivR a (envR σ) k|) := by ring

/-- **Kink order-agreement under `WD`.** The `WD` gap (`εa + εb < |da − db|`) forces the Float order of the
    arguments to match their real order — so `max`/`min`'s discrete edge choices agree (edge error 0). -/
theorem order_agree (a b : Expr) (σ : Nat → Float) (ha : WD a σ) (hb : WD b σ)
    (hgap : evalErrBnd a σ + evalErrBnd b σ < |toReal (evalF a σ) - toReal (evalF b σ)|) :
    (evalF a σ ≤ evalF b σ) ↔ (evalR a (envR σ) ≤ evalR b (envR σ)) := by
  obtain ⟨hva1, hva2⟩ := abs_le.mp (evalF_error a σ ha)
  obtain ⟨hvb1, hvb2⟩ := abs_le.mp (evalF_error b σ hb)
  constructor
  · intro h
    have hdab := le_of_float_le h
    have habs : |toReal (evalF a σ) - toReal (evalF b σ)| = toReal (evalF b σ) - toReal (evalF a σ) := by
      rw [abs_of_nonpos (by linarith only [hdab])]; ring
    rw [habs] at hgap; linarith only [hva1, hva2, hvb1, hvb2, hgap]
  · intro hR
    by_contra h
    have hdba := le_of_not_float_le h
    have habs : |toReal (evalF a σ) - toReal (evalF b σ)| = toReal (evalF a σ) - toReal (evalF b σ) := by
      rw [abs_of_nonneg (by linarith only [hdba])]
    rw [habs] at hgap
    exact absurd hR (by rw [not_le]; linarith only [hva1, hva2, hvb1, hvb2, hgap])

/-- **Kink edge-agreement for `relu` under `WD`.** The Float subgradient branch matches the real one. -/
theorem relu_edge_eq (a : Expr) (σ : Nat → Float) (hwd : WD (Expr.relu a) σ) :
    toReal (if evalF a σ < 0.0 then (0.0 : Float) else 1.0) = (if 0 < evalR a (envR σ) then (1 : ℝ) else 0) := by
  obtain ⟨ha, hlo⟩ := hwd
  have hea : (0 : ℝ) ≤ evalErrBnd a σ := evalErrBnd_nonneg a σ ha
  have hval := evalF_error a σ ha
  by_cases h : evalF a σ < 0.0
  · rw [if_pos h]
    have hxle : toReal (evalF a σ) ≤ 0 := by
      have hr := toReal_reluF (evalF a σ); unfold reluF at hr
      rw [if_pos h, toReal_zeroLit] at hr; exact le_of_max_le_left hr.symm.le
    have hxne : toReal (evalF a σ) ≠ 0 := by
      intro hz; rw [hz, abs_zero] at hlo; exact absurd hlo (not_lt.mpr hea)
    have hxlt : toReal (evalF a σ) < 0 := lt_of_le_of_ne hxle hxne
    have hRlt : evalR a (envR σ) < 0 := by
      have h1 := (abs_le.mp hval).1; rw [abs_of_neg hxlt] at hlo; linarith only [h1, hlo]
    rw [if_neg (not_lt.mpr hRlt.le)]; exact toReal_zeroLit
  · rw [if_neg h]
    have hxge : 0 ≤ toReal (evalF a σ) := by
      have hr := toReal_reluF (evalF a σ); unfold reluF at hr
      rw [if_neg h] at hr
      calc (0 : ℝ) ≤ max (toReal (evalF a σ)) 0 := le_max_right _ _
        _ = toReal (evalF a σ) := hr.symm
    have hxne : toReal (evalF a σ) ≠ 0 := by
      intro hz; rw [hz, abs_zero] at hlo; exact absurd hlo (not_lt.mpr hea)
    have hxgt : 0 < toReal (evalF a σ) := lt_of_le_of_ne hxge (Ne.symm hxne)
    have hRgt : 0 < evalR a (envR σ) := by
      have h2 := (abs_le.mp hval).2; rw [abs_of_pos hxgt] at hlo; linarith only [h2, hlo]
    rw [if_pos hRgt]; exact toReal_oneLit

/-- Structural bound on `|revE_RF e σ ā k − revE_R e (envR σ) ā k| / |ā|`, all ops: each `mul`/`exp`/`log`
    edge's primal-rounding error (`evalErrBnd`/division) weighted by `|derivR|`; kink/linear edges exact. -/
noncomputable def bridgeBnd : Expr → (Nat → Float) → Nat → ℝ
  | .var _, _, _ => 0
  | .const _, _, _ => 0
  | .add a b, σ, k => bridgeBnd a σ k + bridgeBnd b σ k
  | .sub a b, σ, k => bridgeBnd a σ k + bridgeBnd b σ k
  | .mul a b, σ, k => |toReal (evalF b σ)| * bridgeBnd a σ k + evalErrBnd b σ * |derivR a (envR σ) k|
      + (|toReal (evalF a σ)| * bridgeBnd b σ k + evalErrBnd a σ * |derivR b (envR σ) k|)
  | .scale c a, σ, k => |toReal c| * bridgeBnd a σ k
  | .exp a, σ, k => |toReal (Float.exp (evalF a σ))| * bridgeBnd a σ k
      + evalErrBnd (Expr.exp a) σ * |derivR a (envR σ) k|
  | .log a, σ, k => |toReal (1.0 / evalF a σ)| * bridgeBnd a σ k
      + (u64 * |toReal (1.0 : Float) / toReal (evalF a σ)|
          + (0 + |(1 : ℝ) / evalR a (envR σ)| * evalErrBnd a σ) / (toReal (evalF a σ) - evalErrBnd a σ))
        * |derivR a (envR σ) k|
  | .relu a, σ, k => |toReal (if evalF a σ < 0.0 then (0.0 : Float) else 1.0)| * bridgeBnd a σ k
  | .max a b, σ, k => |toReal (if evalF a σ ≤ evalF b σ then (0.0 : Float) else 1.0)| * bridgeBnd a σ k
      + |toReal (if evalF a σ ≤ evalF b σ then (1.0 : Float) else 0.0)| * bridgeBnd b σ k
  | .min a b, σ, k => |toReal (if evalF a σ ≤ evalF b σ then (1.0 : Float) else 0.0)| * bridgeBnd a σ k
      + |toReal (if evalF a σ ≤ evalF b σ then (0.0 : Float) else 1.0)| * bridgeBnd b σ k

/-- **The bounded bridge (all ops).** For any `e` under `WD`, the rounded-primal reverse pass differs from
    the ideal by at most `|ā|·bridgeBnd` — unconditional (no `hbridge`, no `BridgeExact`), covering
    `exp`/`log`/kink and hence the full PPO objective. -/
theorem revE_RF_sub_revE_R_le (e : Expr) (σ : Nat → Float) :
    WD e σ → ∀ (ā : ℝ) (k : Nat),
      |revE_RF e σ ā k - revE_R e (envR σ) ā k| ≤ |ā| * bridgeBnd e σ k := by
  induction e with
  | var i => intro _ ā k; simp only [revE_RF, revE_R, bridgeBnd, mul_zero, sub_self, abs_zero, le_refl]
  | const c => intro _ ā k; simp only [revE_RF, revE_R, bridgeBnd, mul_zero, sub_self, abs_zero, le_refl]
  | add a b iha ihb => intro hwd ā k
                       obtain ⟨hwa, hwb⟩ := hwd
                       simp only [revE_RF, revE_R, toReal_oneLit, mul_one, bridgeBnd]
                       calc |revE_RF a σ ā k + revE_RF b σ ā k
                              - (revE_R a (envR σ) ā k + revE_R b (envR σ) ā k)|
                            ≤ |revE_RF a σ ā k - revE_R a (envR σ) ā k|
                              + |revE_RF b σ ā k - revE_R b (envR σ) ā k| := by
                              rw [show revE_RF a σ ā k + revE_RF b σ ā k
                                    - (revE_R a (envR σ) ā k + revE_R b (envR σ) ā k)
                                  = (revE_RF a σ ā k - revE_R a (envR σ) ā k)
                                    + (revE_RF b σ ā k - revE_R b (envR σ) ā k) from by ring]
                              exact abs_add_le _ _
                          _ ≤ |ā| * bridgeBnd a σ k + |ā| * bridgeBnd b σ k :=
                              add_le_add (iha hwa ā k) (ihb hwb ā k)
                          _ = |ā| * (bridgeBnd a σ k + bridgeBnd b σ k) := by ring
  | sub a b iha ihb => intro hwd ā k
                       obtain ⟨hwa, hwb⟩ := hwd
                       simp only [revE_RF, revE_R, toReal_oneLit, mul_one, toReal_neg, mul_neg_one, bridgeBnd]
                       calc |revE_RF a σ ā k + revE_RF b σ (-ā) k
                              - (revE_R a (envR σ) ā k + revE_R b (envR σ) (-ā) k)|
                            ≤ |revE_RF a σ ā k - revE_R a (envR σ) ā k|
                              + |revE_RF b σ (-ā) k - revE_R b (envR σ) (-ā) k| := by
                              rw [show revE_RF a σ ā k + revE_RF b σ (-ā) k
                                    - (revE_R a (envR σ) ā k + revE_R b (envR σ) (-ā) k)
                                  = (revE_RF a σ ā k - revE_R a (envR σ) ā k)
                                    + (revE_RF b σ (-ā) k - revE_R b (envR σ) (-ā) k) from by ring]
                              exact abs_add_le _ _
                          _ ≤ |ā| * bridgeBnd a σ k + |(-ā)| * bridgeBnd b σ k :=
                              add_le_add (iha hwa ā k) (ihb hwb (-ā) k)
                          _ = |ā| * (bridgeBnd a σ k + bridgeBnd b σ k) := by rw [abs_neg]; ring
  | mul a b iha ihb => intro hwd ā k
                       obtain ⟨hwa, hwb⟩ := hwd
                       have heb := evalF_error b σ hwb
                       have hea := evalF_error a σ hwa
                       simp only [revE_RF, revE_R, bridgeBnd]
                       calc |revE_RF a σ (ā * toReal (evalF b σ)) k + revE_RF b σ (ā * toReal (evalF a σ)) k
                              - (revE_R a (envR σ) (ā * evalR b (envR σ)) k
                                + revE_R b (envR σ) (ā * evalR a (envR σ)) k)|
                            ≤ |revE_RF a σ (ā * toReal (evalF b σ)) k
                                - revE_R a (envR σ) (ā * evalR b (envR σ)) k|
                              + |revE_RF b σ (ā * toReal (evalF a σ)) k
                                - revE_R b (envR σ) (ā * evalR a (envR σ)) k| := by
                              rw [show revE_RF a σ (ā * toReal (evalF b σ)) k
                                    + revE_RF b σ (ā * toReal (evalF a σ)) k
                                    - (revE_R a (envR σ) (ā * evalR b (envR σ)) k
                                      + revE_R b (envR σ) (ā * evalR a (envR σ)) k)
                                  = (revE_RF a σ (ā * toReal (evalF b σ)) k
                                      - revE_R a (envR σ) (ā * evalR b (envR σ)) k)
                                    + (revE_RF b σ (ā * toReal (evalF a σ)) k
                                      - revE_R b (envR σ) (ā * evalR a (envR σ)) k) from by ring]
                              exact abs_add_le _ _
                          _ ≤ |ā| * (|toReal (evalF b σ)| * bridgeBnd a σ k
                                + evalErrBnd b σ * |derivR a (envR σ) k|)
                              + |ā| * (|toReal (evalF a σ)| * bridgeBnd b σ k
                                + evalErrBnd a σ * |derivR b (envR σ) k|) :=
                              add_le_add
                                (unary_step a σ k ā (toReal (evalF b σ)) (evalR b (envR σ))
                                  (bridgeBnd a σ k) (evalErrBnd b σ) (fun ā' => iha hwa ā' k) heb)
                                (unary_step b σ k ā (toReal (evalF a σ)) (evalR a (envR σ))
                                  (bridgeBnd b σ k) (evalErrBnd a σ) (fun ā' => ihb hwb ā' k) hea)
                          _ = |ā| * (|toReal (evalF b σ)| * bridgeBnd a σ k
                                + evalErrBnd b σ * |derivR a (envR σ) k|
                              + (|toReal (evalF a σ)| * bridgeBnd b σ k
                                + evalErrBnd a σ * |derivR b (envR σ) k|)) := by ring
  | scale c a iha => intro hwd ā k
                     simp only [revE_RF, revE_R, bridgeBnd]
                     calc |revE_RF a σ (ā * toReal c) k - revE_R a (envR σ) (ā * toReal c) k|
                          ≤ |ā * toReal c| * bridgeBnd a σ k := iha hwd (ā * toReal c) k
                        _ = |ā| * (|toReal c| * bridgeBnd a σ k) := by rw [abs_mul]; ring
  | exp a iha => intro hwd ā k
                 have hedge := evalF_error (Expr.exp a) σ hwd
                 simp only [evalF, evalR] at hedge
                 simp only [revE_RF, revE_R, bridgeBnd]
                 exact unary_step a σ k ā (toReal (Float.exp (evalF a σ))) (Real.exp (evalR a (envR σ)))
                   (bridgeBnd a σ k) (evalErrBnd (Expr.exp a) σ) (fun ā' => iha hwd ā' k) hedge
  | log a iha => intro hwd ā k
                 obtain ⟨hwa, hlo⟩ := hwd
                 have hea : (0 : ℝ) ≤ evalErrBnd a σ := evalErrBnd_nonneg a σ hwa
                 have hpos : 0 < toReal (evalF a σ) := lt_of_le_of_lt hea hlo
                 have hval := evalF_error a σ hwa
                 have hdmin : 0 < toReal (evalF a σ) - evalErrBnd a σ := by linarith
                 have hdy : toReal (evalF a σ) - evalErrBnd a σ ≤ |toReal (evalF a σ)| := by
                   rw [abs_of_pos hpos]; linarith
                 have hyR : evalR a (envR σ) ≠ 0 := by
                   have hle : toReal (evalF a σ) - evalErrBnd a σ ≤ evalR a (envR σ) := by
                     have := (abs_le.mp hval).2; linarith
                   exact ne_of_gt (lt_of_lt_of_le hdmin hle)
                 have hx : |toReal (1.0 : Float) - (1 : ℝ)| ≤ 0 := by rw [toReal_oneLit]; simp
                 have hew := divApprox_error (1.0 : Float) (evalF a σ) (1 : ℝ) (evalR a (envR σ)) 0
                   (evalErrBnd a σ) (toReal (evalF a σ) - evalErrBnd a σ) hx hval hdmin hdy hyR
                 simp only [revE_RF, revE_R, bridgeBnd]
                 exact unary_step a σ k ā (toReal (1.0 / evalF a σ)) (1 / evalR a (envR σ))
                   (bridgeBnd a σ k)
                   (u64 * |toReal (1.0 : Float) / toReal (evalF a σ)|
                     + (0 + |(1 : ℝ) / evalR a (envR σ)| * evalErrBnd a σ)
                       / (toReal (evalF a σ) - evalErrBnd a σ))
                   (fun ā' => iha hwa ā' k) hew
  | relu a iha => intro hwd ā k
                  have hwa : WD a σ := hwd.1
                  have hedge := relu_edge_eq a σ hwd
                  simp only [revE_RF, revE_R, bridgeBnd]
                  rw [← hedge]
                  calc |revE_RF a σ (ā * toReal (if evalF a σ < 0.0 then (0.0 : Float) else 1.0)) k
                         - revE_R a (envR σ) (ā * toReal (if evalF a σ < 0.0 then (0.0 : Float) else 1.0)) k|
                        ≤ |ā * toReal (if evalF a σ < 0.0 then (0.0 : Float) else 1.0)| * bridgeBnd a σ k :=
                        iha hwa _ k
                      _ = |ā| * (|toReal (if evalF a σ < 0.0 then (0.0 : Float) else 1.0)| * bridgeBnd a σ k) := by
                        rw [abs_mul]; ring
  | max a b iha ihb => intro hwd ā k
                       obtain ⟨hwa, hwb, hgap⟩ := hwd
                       have hord := order_agree a b σ hwa hwb hgap
                       have hea : toReal (if evalF a σ ≤ evalF b σ then (0.0 : Float) else 1.0)
                           = (if evalR a (envR σ) ≤ evalR b (envR σ) then (0 : ℝ) else 1) := by
                         by_cases h : evalF a σ ≤ evalF b σ
                         · rw [if_pos h, if_pos (hord.mp h)]; exact toReal_zeroLit
                         · rw [if_neg h, if_neg (fun hR => h (hord.mpr hR))]; exact toReal_oneLit
                       have heb : toReal (if evalF a σ ≤ evalF b σ then (1.0 : Float) else 0.0)
                           = (if evalR a (envR σ) ≤ evalR b (envR σ) then (1 : ℝ) else 0) := by
                         by_cases h : evalF a σ ≤ evalF b σ
                         · rw [if_pos h, if_pos (hord.mp h)]; exact toReal_oneLit
                         · rw [if_neg h, if_neg (fun hR => h (hord.mpr hR))]; exact toReal_zeroLit
                       simp only [revE_RF, revE_R, bridgeBnd]
                       rw [← hea, ← heb]
                       calc |revE_RF a σ (ā * toReal (if evalF a σ ≤ evalF b σ then (0.0 : Float) else 1.0)) k
                              + revE_RF b σ (ā * toReal (if evalF a σ ≤ evalF b σ then (1.0 : Float) else 0.0)) k
                              - (revE_R a (envR σ) (ā * toReal (if evalF a σ ≤ evalF b σ then (0.0 : Float) else 1.0)) k
                                + revE_R b (envR σ) (ā * toReal (if evalF a σ ≤ evalF b σ then (1.0 : Float) else 0.0)) k)|
                            ≤ |revE_RF a σ (ā * toReal (if evalF a σ ≤ evalF b σ then (0.0 : Float) else 1.0)) k
                                - revE_R a (envR σ) (ā * toReal (if evalF a σ ≤ evalF b σ then (0.0 : Float) else 1.0)) k|
                              + |revE_RF b σ (ā * toReal (if evalF a σ ≤ evalF b σ then (1.0 : Float) else 0.0)) k
                                - revE_R b (envR σ) (ā * toReal (if evalF a σ ≤ evalF b σ then (1.0 : Float) else 0.0)) k| := by
                              rw [show revE_RF a σ (ā * toReal (if evalF a σ ≤ evalF b σ then (0.0 : Float) else 1.0)) k
                                    + revE_RF b σ (ā * toReal (if evalF a σ ≤ evalF b σ then (1.0 : Float) else 0.0)) k
                                    - (revE_R a (envR σ) (ā * toReal (if evalF a σ ≤ evalF b σ then (0.0 : Float) else 1.0)) k
                                      + revE_R b (envR σ) (ā * toReal (if evalF a σ ≤ evalF b σ then (1.0 : Float) else 0.0)) k)
                                  = (revE_RF a σ (ā * toReal (if evalF a σ ≤ evalF b σ then (0.0 : Float) else 1.0)) k
                                      - revE_R a (envR σ) (ā * toReal (if evalF a σ ≤ evalF b σ then (0.0 : Float) else 1.0)) k)
                                    + (revE_RF b σ (ā * toReal (if evalF a σ ≤ evalF b σ then (1.0 : Float) else 0.0)) k
                                      - revE_R b (envR σ) (ā * toReal (if evalF a σ ≤ evalF b σ then (1.0 : Float) else 0.0)) k) from by ring]
                              exact abs_add_le _ _
                          _ ≤ |ā * toReal (if evalF a σ ≤ evalF b σ then (0.0 : Float) else 1.0)| * bridgeBnd a σ k
                              + |ā * toReal (if evalF a σ ≤ evalF b σ then (1.0 : Float) else 0.0)| * bridgeBnd b σ k :=
                              add_le_add (iha hwa _ k) (ihb hwb _ k)
                          _ = |ā| * (|toReal (if evalF a σ ≤ evalF b σ then (0.0 : Float) else 1.0)| * bridgeBnd a σ k
                              + |toReal (if evalF a σ ≤ evalF b σ then (1.0 : Float) else 0.0)| * bridgeBnd b σ k) := by
                              rw [abs_mul, abs_mul]; ring
  | min a b iha ihb => intro hwd ā k
                       obtain ⟨hwa, hwb, hgap⟩ := hwd
                       have hord := order_agree a b σ hwa hwb hgap
                       have hea : toReal (if evalF a σ ≤ evalF b σ then (1.0 : Float) else 0.0)
                           = (if evalR a (envR σ) ≤ evalR b (envR σ) then (1 : ℝ) else 0) := by
                         by_cases h : evalF a σ ≤ evalF b σ
                         · rw [if_pos h, if_pos (hord.mp h)]; exact toReal_oneLit
                         · rw [if_neg h, if_neg (fun hR => h (hord.mpr hR))]; exact toReal_zeroLit
                       have heb : toReal (if evalF a σ ≤ evalF b σ then (0.0 : Float) else 1.0)
                           = (if evalR a (envR σ) ≤ evalR b (envR σ) then (0 : ℝ) else 1) := by
                         by_cases h : evalF a σ ≤ evalF b σ
                         · rw [if_pos h, if_pos (hord.mp h)]; exact toReal_zeroLit
                         · rw [if_neg h, if_neg (fun hR => h (hord.mpr hR))]; exact toReal_oneLit
                       simp only [revE_RF, revE_R, bridgeBnd]
                       rw [← hea, ← heb]
                       calc |revE_RF a σ (ā * toReal (if evalF a σ ≤ evalF b σ then (1.0 : Float) else 0.0)) k
                              + revE_RF b σ (ā * toReal (if evalF a σ ≤ evalF b σ then (0.0 : Float) else 1.0)) k
                              - (revE_R a (envR σ) (ā * toReal (if evalF a σ ≤ evalF b σ then (1.0 : Float) else 0.0)) k
                                + revE_R b (envR σ) (ā * toReal (if evalF a σ ≤ evalF b σ then (0.0 : Float) else 1.0)) k)|
                            ≤ |revE_RF a σ (ā * toReal (if evalF a σ ≤ evalF b σ then (1.0 : Float) else 0.0)) k
                                - revE_R a (envR σ) (ā * toReal (if evalF a σ ≤ evalF b σ then (1.0 : Float) else 0.0)) k|
                              + |revE_RF b σ (ā * toReal (if evalF a σ ≤ evalF b σ then (0.0 : Float) else 1.0)) k
                                - revE_R b (envR σ) (ā * toReal (if evalF a σ ≤ evalF b σ then (0.0 : Float) else 1.0)) k| := by
                              rw [show revE_RF a σ (ā * toReal (if evalF a σ ≤ evalF b σ then (1.0 : Float) else 0.0)) k
                                    + revE_RF b σ (ā * toReal (if evalF a σ ≤ evalF b σ then (0.0 : Float) else 1.0)) k
                                    - (revE_R a (envR σ) (ā * toReal (if evalF a σ ≤ evalF b σ then (1.0 : Float) else 0.0)) k
                                      + revE_R b (envR σ) (ā * toReal (if evalF a σ ≤ evalF b σ then (0.0 : Float) else 1.0)) k)
                                  = (revE_RF a σ (ā * toReal (if evalF a σ ≤ evalF b σ then (1.0 : Float) else 0.0)) k
                                      - revE_R a (envR σ) (ā * toReal (if evalF a σ ≤ evalF b σ then (1.0 : Float) else 0.0)) k)
                                    + (revE_RF b σ (ā * toReal (if evalF a σ ≤ evalF b σ then (0.0 : Float) else 1.0)) k
                                      - revE_R b (envR σ) (ā * toReal (if evalF a σ ≤ evalF b σ then (0.0 : Float) else 1.0)) k) from by ring]
                              exact abs_add_le _ _
                          _ ≤ |ā * toReal (if evalF a σ ≤ evalF b σ then (1.0 : Float) else 0.0)| * bridgeBnd a σ k
                              + |ā * toReal (if evalF a σ ≤ evalF b σ then (0.0 : Float) else 1.0)| * bridgeBnd b σ k :=
                              add_le_add (iha hwa _ k) (ihb hwb _ k)
                          _ = |ā| * (|toReal (if evalF a σ ≤ evalF b σ then (1.0 : Float) else 0.0)| * bridgeBnd a σ k
                              + |toReal (if evalF a σ ≤ evalF b σ then (0.0 : Float) else 1.0)| * bridgeBnd b σ k) := by
                              rw [abs_mul, abs_mul]; ring

/-- **The bounded `hbridge` gap at the seed adjoint `ā = 1`** (all ops): the rounded-primal reverse
    derivative is within `bridgeBnd` of `derivR`, unconditionally under `WD` — the quantitative form of the
    reverse-mode gradient's primal-rounding error for the full objective. -/
theorem bridge_gap_le (e : Expr) (σ : Nat → Float) (k : Nat) (hwd : WD e σ) :
    |revE_RF e σ 1 k - derivR e (envR σ) k| ≤ bridgeBnd e σ k := by
  have h := revE_RF_sub_revE_R_le e σ hwd 1 k
  rwa [revE_R_eq, one_mul, abs_one, one_mul] at h

end Puffer.FloatR.ADReverse
