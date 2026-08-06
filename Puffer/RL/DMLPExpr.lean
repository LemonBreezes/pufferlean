/-
# Dependent-typed varying-width MLP: active-region gradient-Lipschitz for ANY depth AND ANY widths

C20 (`MLPDepthExpr`) proved the active-region gradient-Lipschitz for a fixed 3-layer MLP, and C22
(`MLPDepthNExpr`) generalized to arbitrary depth — but only for a UNIFORM width `d` (every layer `Fin d → Fin d`),
because a plain `List` of layers forces one type. This module removes the uniform-width restriction with a
DEPENDENT-TYPED stack: each layer changes the `Fin` dimension, chained through a dimension-indexed inductive.

`DMLP m n` is a length-indexed, dimension-threaded stack of layers from input width `m` to output width `n`
(`nil : DMLP d d`; `cons (L : DLayer m h) (rest : DMLP h n) : DMLP m n` — a layer `Fin m → Fin h` then the rest
`Fin h → Fin n`, widths VARYING). `dmlpE stack x` folds `relu ∘ linLayerE` per layer; `dmlpLin` is the
relu-stripped linearization (a composition of linear layers, hence `Smooth`).

The proof reuses C20's `relu_linLayer_match` — which is already DIMENSION-POLYMORPHIC (`{n m}`), taking `Fin m`
activations to `Fin n` — so it composes across the changing widths without any new machinery. The dependent
induction `dmlp_active_eq_aux` iterates it once per layer (the `LayerMatch` invariant's dimension changing each
step), giving `dmlp_active_gradient_lipschitz`:

    |derivR (dmlpE stack x i) σ k − derivR (dmlpE stack x i) σ' k|  ≤  dLip R (dmlpLin stack x i)·δ

for a stack of ANY depth and ANY per-layer widths, on the all-hidden-active region (both points).

**Scope (honestly disclosed):** this is the full generalization of C20/C22 — arbitrary depth (`DMLP` of any
length) AND arbitrary per-layer widths (each `DLayer m h` with independent `m`, `h`). The `AllActiveD` hypothesis
(every intermediate pre-activation positive at both points) is REAL and load-bearing (off it a ReLU crosses its
kink — C7's intrinsic non-Lipschitzness), exactly as in C20/C22. A ReLU is applied after every layer; a final
linear readout composes as one more `linLayerE` (as in C22). The `DMLP` is the honest dependent-typed structure
the C22 docstring said "a symbolic-`n` capstone for varying widths would need."
-/
import Puffer.RL.MLPDepthExpr

open Puffer.FloatR.ADR
open Puffer.RL.LinearLayerExpr
open Puffer.RL.MLPDepthExpr

namespace Puffer.RL.DMLPExpr

/-- A single layer from width `m` to width `n`: a weight-index map `Fin n → Fin m → Nat` and a bias-index map
    `Fin n → Nat`. Distinct `m`, `n` — the widths vary per layer. -/
abbrev DLayer (m n : Nat) : Type := (Fin n → Fin m → Nat) × (Fin n → Nat)

/-- **A dimension-chained stack of layers** from input width `m` to output width `n`. `nil` is the empty stack
    (`m = n`); `cons (L : DLayer m h) rest` prepends a layer `Fin m → Fin h` to a stack `rest : DMLP h n`. The
    intermediate widths `h` are existentially threaded, so widths VARY freely per layer (unlike C22's uniform `d`). -/
inductive DMLP : Nat → Nat → Type where
  | nil {d : Nat} : DMLP d d
  | cons {m h n : Nat} (L : DLayer m h) (rest : DMLP h n) : DMLP m n

/-- The varying-width ReLU MLP: fold `relu ∘ linLayerE` over the dependent stack (a ReLU after every layer). -/
def dmlpE : {m n : Nat} → DMLP m n → (Fin m → Expr) → (Fin n → Expr)
  | _, _, .nil, x => x
  | _, _, .cons L rest, x => dmlpE rest (fun i => Expr.relu (linLayerE L.1 x L.2 i))

/-- The relu-stripped LINEARIZATION of `dmlpE` — a composition of linear layers, hence `Smooth`. -/
def dmlpLin : {m n : Nat} → DMLP m n → (Fin m → Expr) → (Fin n → Expr)
  | _, _, .nil, x => x
  | _, _, .cons L rest, x => dmlpLin rest (fun i => linLayerE L.1 x L.2 i)

/-- The linearization of a varying-width stack over `Smooth` inputs is `Smooth` (dependent induction, each layer
    `linLayerE_smooth`). So C4's `derivR_lip` applies to it at any depth and any widths. -/
theorem dmlpLin_smooth : {m n : Nat} → (stack : DMLP m n) → ∀ (x : Fin m → Expr),
    (∀ l, Smooth (x l)) → ∀ i, Smooth (dmlpLin stack x i)
  | _, _, .nil, _, hx, i => hx i
  | _, _, .cons L rest, x, hx, i =>
      dmlpLin_smooth rest (fun j => linLayerE L.1 x L.2 j) (fun j => linLayerE_smooth L.1 x L.2 j hx) i

/-- **All-intermediate-active predicate** over the dependent stack: at each layer the pre-activation over the
    running (relu) activations is strictly positive. Threaded through the fold exactly as the forward pass runs. -/
def AllActiveD (σ : Nat → ℝ) : {m n : Nat} → DMLP m n → (Fin m → Expr) → Prop
  | _, _, .nil, _ => True
  | _, _, .cons L rest, a =>
      (∀ i, 0 < evalR (linLayerE L.1 a L.2 i) σ)
      ∧ AllActiveD σ rest (fun i => Expr.relu (linLayerE L.1 a L.2 i))

/-- **The varying-width reduction (dependent induction).** Given a `LayerMatch` between the running relu
    activations `a` and their linearization `a'` (value + gradient), and `AllActiveD` for the stack, the whole
    stack's relu activations match the linearization's. Iterates C20's DIMENSION-POLYMORPHIC `relu_linLayer_match`
    once per layer — the `LayerMatch` invariant's `Fin`-dimension changing each step, which the polymorphic step
    handles with no new machinery. This is what makes varying widths work where C22's uniform `List` could not. -/
theorem dmlp_active_eq_aux (σ : Nat → ℝ) (k : Nat) :
    {m n : Nat} → (stack : DMLP m n) → ∀ (a a' : Fin m → Expr),
      LayerMatch σ k a a' → AllActiveD σ stack a →
        LayerMatch σ k (dmlpE stack a) (dmlpLin stack a')
  | _, _, .nil, _, _, hm, _ => hm
  | _, _, .cons L rest, a, a', hm, hact =>
      dmlp_active_eq_aux σ k rest _ _ (relu_linLayer_match L.1 a a' L.2 σ k hm hact.1) hact.2

/-- **CAPSTONE: varying-width, arbitrary-depth ReLU MLP is gradient-Lipschitz on the all-active region.** The full
    generalization of C20 (fixed 3-layer) and C22 (uniform width): for a `DMLP` stack of ANY depth and ANY
    per-layer widths, with `Smooth` input and every intermediate pre-activation positive at BOTH `σ` and `σ'`,
    `|derivR (dmlpE stack x i) σ k − derivR (dmlpE stack x i) σ' k| ≤ dLip R (dmlpLin stack x i)·δ`. On the
    all-active region the net collapses to its `Smooth` linearization (`dmlp_active_eq_aux`), which inherits C4's
    `derivR_lip`. The `AllActiveD` hypothesis is load-bearing (off it a ReLU crosses its kink). -/
theorem dmlp_active_gradient_lipschitz {m n : Nat} (stack : DMLP m n) (x : Fin m → Expr) (i : Fin n)
    (σ σ' : Nat → ℝ) (R δ : ℝ) (k : Nat) (hx : ∀ l, Smooth (x l))
    (hσ : ∀ j, |σ j| ≤ R) (hσ' : ∀ j, |σ' j| ≤ R) (hδ : ∀ j, |σ j - σ' j| ≤ δ) (hR : 0 ≤ R)
    (hact : AllActiveD σ stack x) (hact' : AllActiveD σ' stack x) :
    |derivR (dmlpE stack x i) σ k - derivR (dmlpE stack x i) σ' k|
      ≤ dLip R (dmlpLin stack x i) * δ := by
  rw [(dmlp_active_eq_aux σ k stack x x (LayerMatch.rfl' σ k x) hact).2 i,
      (dmlp_active_eq_aux σ' k stack x x (LayerMatch.rfl' σ' k x) hact').2 i]
  exact derivR_lip (dmlpLin_smooth stack x hx i) σ σ' R δ k hσ hσ' hδ hR

end Puffer.RL.DMLPExpr
