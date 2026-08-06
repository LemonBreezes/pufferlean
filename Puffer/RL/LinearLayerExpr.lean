/-
# Linear layer → `Expr` builder

The last "compile the forward pass" piece: turns a neural-network linear layer `out_i = Σⱼ W[i][j]·aⱼ + b_i`
into the AD grammar `Expr`, so the abstract logit expressions consumed by `SoftmaxExpr` (C9) become concrete
`W·x + b` combinations of the network parameters.

The var-indexing convention: `σ : Nat → ℝ` is the flat parameter vector and `derivR e σ k = ∂/∂(param k)`.
Since the gradient we care about is the policy gradient w.r.t. the network WEIGHTS, the weights/biases are the
`var`s (`var (w i j)`, `var (b i)`, at caller-supplied flat slots), and the input activations `aⱼ` are `Expr`s
— `const`s for the first layer (fixed observations), previous-layer output `Expr`s for hidden layers. So a
neuron is `Σⱼ (var wⱼ)·aⱼ + var b`, a fold of `mul`/`add` (the grammar has no n-ary sum).

Results:
* `evalR_dotBiasE` / `evalR_linLayerE` — the compiled layer evaluates to `Σⱼ σ(wⱼ)·evalR(aⱼ)σ + σ(b)`, the
  real linear-layer computation (weight value × activation value, plus bias value).
* `dotBiasE_smooth` / `linLayerE_smooth` — a layer of `Smooth` activations is itself `Smooth` (`var`, `mul`,
  `add` are all `Smooth` constructors), so C4's `derivR_lip` gives it a concrete gradient-Lipschitz constant
  with NO free hypotheses. `linLayerE_const_smooth` specializes to the first layer (constant inputs), which is
  `Smooth` unconditionally.

The gradient rule falls out of the grammar automatically: `derivR (dotBiasE …) σ (wⱼ) = evalR aⱼ σ` — the
gradient w.r.t. a weight slot is exactly its input activation (`∂L/∂Wᵢⱼ = aⱼ`).

**Scope (honestly disclosed):** this compiles one linear layer's VALUE + smoothness; stacking layers with
`relu` between them (a multi-layer MLP) composes `linLayerE` with `relu` `Expr`s — the ReLU nodes leave the
`Smooth` fragment (their gradient-Lipschitz is C7's away-from-kink, per-hidden-unit active region), so a deep
net's gradient-Lipschitz is assembled from C4 (linear, here) + C7 (relu) downstream. With `SoftmaxExpr` (C9),
`linLayerE w a b` supplies the logit vector: `logSoftmaxE (linLayerE w a b i) (List.ofFn (linLayerE w a b))`.
-/
import Puffer.Float.AutoDiffR

open Puffer.FloatR.ADR
open Puffer.FloatR (toReal)

namespace Puffer.RL.LinearLayerExpr

/-- **A single neuron `Σⱼ Wⱼ·aⱼ + b` as an `Expr`.** Weights are `var`-indices (the `.1` of each pair), input
    activations are `Expr`s (the `.2`), bias is a `var`-index. A fold of `add` over `mul (var wⱼ) aⱼ`, based at
    the bias `var b` (the grammar has no n-ary sum). -/
def dotBiasE : List (Nat × Expr) → Nat → Expr
  | [], b => .var b
  | (wj, aj) :: rest, b => .add (.mul (.var wj) aj) (dotBiasE rest b)

/-- `evalR (dotBiasE wa b) σ = Σ σ(wⱼ)·evalR(aⱼ)σ + σ(b)` — the real dot-product-plus-bias (weight value ×
    activation value, summed, plus bias value). -/
theorem evalR_dotBiasE (wa : List (Nat × Expr)) (b : Nat) (σ : Nat → ℝ) :
    evalR (dotBiasE wa b) σ = (wa.map (fun p => σ p.1 * evalR p.2 σ)).sum + σ b := by
  induction wa with
  | nil => simp [dotBiasE, evalR]
  | cons p rest ih =>
      obtain ⟨wj, aj⟩ := p
      simp only [dotBiasE, evalR, List.map_cons, List.sum_cons, ih]
      ring

/-- **A neuron of `Smooth` activations is `Smooth`.** Since `var`, `mul`, `add` are all `Smooth` constructors,
    `dotBiasE wa b` lands in the `Smooth` fragment whenever every activation `Expr` in `wa` is `Smooth` — so
    C4's `derivR_lip` gives it a concrete gradient-Lipschitz constant with no free hypotheses. -/
theorem dotBiasE_smooth (wa : List (Nat × Expr)) (b : Nat)
    (ha : ∀ p ∈ wa, Smooth p.2) : Smooth (dotBiasE wa b) := by
  induction wa with
  | nil => exact Smooth.var b
  | cons p rest ih =>
      obtain ⟨wj, aj⟩ := p
      refine Smooth.add (Smooth.mul (Smooth.var wj) ?_) (ih ?_)
      · exact ha (wj, aj) (List.mem_cons.mpr (Or.inl rfl))
      · exact fun q hq => ha q (List.mem_cons.mpr (Or.inr hq))

/-- **A full linear layer `out_i = Σⱼ W[i][j]·aⱼ + b_i`.** Produces one `Expr` per output neuron `i`, over the
    weight-index map `w`, input activations `a`, and bias-index map `b`. This is the logit vector `Fin n → Expr`
    that `SoftmaxExpr.logSoftmaxE`/`logSoftmaxE_ofFn` consume. -/
def linLayerE {n m : Nat} (w : Fin n → Fin m → Nat) (a : Fin m → Expr) (b : Fin n → Nat) :
    Fin n → Expr :=
  fun i => dotBiasE (List.ofFn (fun j => (w i j, a j))) (b i)

/-- `evalR (linLayerE w a b i) σ = Σⱼ σ(w i j)·evalR(aⱼ)σ + σ(b i)` — output neuron `i` computes the real
    linear combination of weight values and activation values plus its bias. -/
theorem evalR_linLayerE {n m : Nat} (w : Fin n → Fin m → Nat) (a : Fin m → Expr) (b : Fin n → Nat)
    (i : Fin n) (σ : Nat → ℝ) :
    evalR (linLayerE w a b i) σ = (∑ j : Fin m, σ (w i j) * evalR (a j) σ) + σ (b i) := by
  simp only [linLayerE, evalR_dotBiasE, List.map_ofFn, List.sum_ofFn, Function.comp]

/-- Each output neuron of a layer of `Smooth` activations is `Smooth` (so C4's `derivR_lip` applies to every
    logit). -/
theorem linLayerE_smooth {n m : Nat} (w : Fin n → Fin m → Nat) (a : Fin m → Expr) (b : Fin n → Nat)
    (i : Fin n) (ha : ∀ j, Smooth (a j)) : Smooth (linLayerE w a b i) := by
  apply dotBiasE_smooth
  rw [List.forall_mem_ofFn_iff]
  exact fun j => ha j

/-- **The first layer over raw (constant) inputs is `Smooth` unconditionally.** Its activations are `const`s
    (fixed observations), which are `Smooth`, so every first-layer neuron lands in the `Smooth` fragment. -/
theorem linLayerE_const_smooth {n m : Nat} (w : Fin n → Fin m → Nat) (x : Fin m → Float)
    (b : Fin n → Nat) (i : Fin n) : Smooth (linLayerE w (fun j => .const (x j)) b i) :=
  linLayerE_smooth w _ b i (fun j => Smooth.const (x j))

end Puffer.RL.LinearLayerExpr
