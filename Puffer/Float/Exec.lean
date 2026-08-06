/-
Executable Float kernels — Mathlib-FREE so the `puffer` binary stays lean (no
Mathlib linked). The proofs about these (error bounds vs ℝ) live in
`Puffer/Float/Net.lean`, which imports this module and Mathlib.
-/
namespace Puffer.FloatR

/-- Executable dot product `Σ xᵢ·wᵢ`, evaluated left-to-right in `Float`. -/
def dotF : List Float → List Float → Float
  | [], _ => 0
  | _, [] => 0
  | x :: xs, w :: ws => x * w + dotF xs ws

/-- Executable ReLU. -/
def reluF (x : Float) : Float := if x < 0.0 then 0.0 else x

/-- Executable linear unit `bias + Σ xᵢ·wᵢ`. -/
def linearF (x w : List Float) (bias : Float) : Float := dotF x w + bias

/-- One dense layer with ReLU: `reluF (linearF x wᵢ bᵢ)` for each neuron `(wᵢ, bᵢ)`. -/
def denseRelu (x : List Float) (w : List (List Float)) (b : List Float) : List Float :=
  (w.zip b).map (fun wb => reluF (linearF x wb.1 wb.2))

/-- GAE/discounted backward recurrence in `Float`: `A₀ = δ₀ + w·A₁` unrolled over
    the TD errors (`w = γλ`). The executable counterpart of `Puffer.RL.GAE.gaeHead`
    (its error vs ℝ is bounded in `Puffer/RL/GAERuntime.lean`). -/
def gaeHeadF (w : Float) : List Float → Float
  | [] => 0
  | δ :: rest => δ + w * gaeHeadF w rest

/-- Executable Newton–Schulz singular-value map `φ(σ) = a·σ + b·σ³ + c·σ⁵`, factored
    as `σ·(a + b·σ² + c·σ⁴)` so `σ²` is shared. Runnable counterpart of
    `Puffer.Optim.Muon.nsScalar` (its error vs ℝ is bounded in `Puffer/RL/MuonRuntime.lean`). -/
def nsScalarF (a b c σ : Float) : Float :=
  σ * (a + b * (σ * σ) + c * ((σ * σ) * (σ * σ)))

/-- Executable Nesterov momentum accumulator `m ← μ·m + g` (muon.cu:52). Runnable counterpart of
    `Puffer.Optim.Muon.nesterovMomentum` (its error vs ℝ is bounded in `Puffer/RL/MuonRuntime.lean`). -/
def nesterovMomentumF (m g μ : Float) : Float := μ * m + g

/-- Executable fused Muon weight update `w ← w·(1 − lr·wd) − lr·scale·update` (muon.cu:65). Runnable
    counterpart of `Puffer.Optim.Muon.weightUpdate` (its error vs ℝ is bounded in `Puffer/RL/MuonRuntime.lean`). -/
def weightUpdateF (w update lr wd scale : Float) : Float :=
  w * (1 - lr * wd) - lr * scale * update

/-! ### Adam optimizer step (byte-identical to `Puffer.RL.AdamStep`'s defs; the bound
    theorems there certify these, the rfl bridge `Puffer.RL.AdamExecBridge` proves the
    equality, and `puffer verify-adam` runs them). Per weight, `c₁ = 1−β₁`, `c₂ = 1−β₂`. -/

/-- Adam 1st-moment update `β₁·m + c₁·g`. -/
def adamM1F (m g b1 c1 : Float) : Float := b1 * m + c1 * g
/-- Adam 2nd-moment update `β₂·v + c₂·g²`. -/
def adamM2F (v g b2 c2 : Float) : Float := b2 * v + c2 * (g * g)
/-- Adam direction `m / (√v + ε)`. -/
def adamDirF (m v eps : Float) : Float := m / (Float.sqrt v + eps)
/-- One Adam parameter update `p − lr·(m'/(√v'+ε))`. -/
def adamStepF (p m v g lr b1 c1 b2 c2 eps : Float) : Float :=
  p - lr * adamDirF (adamM1F m g b1 c1) (adamM2F v g b2 c2) eps

/-- Bias-corrected 1st moment `m̂ = m'/c₁ₜ` (`c₁ₜ = 1−β₁ᵗ`). -/
def adamM1HatF (m c1t : Float) : Float := m / c1t
/-- Bias-corrected 2nd moment `v̂ = v'/c₂ₜ` (`c₂ₜ = 1−β₂ᵗ`). -/
def adamM2HatF (v c2t : Float) : Float := v / c2t
/-- Bias-corrected Adam step `p − lr·(m̂/(√v̂+ε))`. -/
def adamStepBcF (p m v g lr b1 c1 b2 c2 eps c1t c2t : Float) : Float :=
  p - lr * adamDirF (adamM1HatF (adamM1F m g b1 c1) c1t) (adamM2HatF (adamM2F v g b2 c2) c2t) eps

end Puffer.FloatR
