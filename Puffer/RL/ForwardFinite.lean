/-
# Forward-pass finiteness: extending C43's kernel certificate to a linear layer and an MLP forward pass

C43 (`Puffer/RL/FiniteBound.lean`) proved a single **bounded linear kernel** (`dotF`) overflow-free, from the one
trusted no-overflow axiom `isFinite_of_bounded` plus the `(1+δ)` base. This module lifts that certificate up the
network: a **linear layer** (`layerF`), a **ReLU layer** (`reluLayerF`), and a full **multi-layer MLP forward pass**
(`mlpF`) — showing that with magnitude-bounded weights and inputs, every layer's activations stay magnitude-bounded
and finite (overflow-free), through to the output.

* `reluF_mag_le` / `reluF_isFinite` — the executable ReLU (`Puffer.FloatR.reluF`, via the existing trusted axiom
  `toReal_reluF : toReal (reluF x) = max (toReal x) 0`) does NOT increase magnitude (`|toReal (reluF x)| ≤ |toReal
  x|`), so it preserves both the magnitude bound and finiteness.
* `dotBound_nonneg` / `dotBound_le_succ` / `dotBound_mono` — C43's `dotBound` is nonnegative and monotone in the term
  count, so a per-neuron `dotBound (min …) B` is bounded by the layer-uniform `dotBound d B` (`d` a width bound).
* `layerF` / `layerF_mag_le` / `layerF_isFinite` — a linear layer `W.map (dotF x ·)`: every neuron is a `dotF`, so
  each output magnitude is `≤ dotBound d B` (via C43's `dotF_mag_le`) and overflow-free when `dotBound d B ≤
  overflowBound`.
* `reluLayerF` / `reluLayerF_mag_le` / `reluLayerF_isFinite` — linear layer followed by ReLU (relu preserves the
  bound).
* `mlpF` / `mlpMagBound` / `mlpF_mag_le` / `mlpF_isFinite` — the forward pass as a left fold of `reluLayerF`. By
  induction over the layers, given per-layer weight bound `Bw`, input bound `B0`, and a uniform width bound `d`, every
  output activation's magnitude is `≤ mlpMagBound d Bw B0 Ws` (an explicit per-layer-growing bound), and the output is
  overflow-free when that bound is `≤ overflowBound`.

**Scope (honestly disclosed).** This covers the FORWARD PASS only — linear layers + ReLU — and rests on C43's
`isFinite_of_bounded` (the single trusted no-overflow fact) plus the `(1+δ)` base; NO new axiom is added. It does NOT
cover the loss, the backward/AD pass, or the optimizer — a full-trainer finiteness would compose those next. The
magnitude bound `mlpMagBound d Bw B0 Ws` GROWS with depth and width (each layer applies `B ↦ dotBound d (max B Bw)`,
roughly squaring), which is realistic — deep/wide nets with large weights genuinely can overflow, and the hypothesis
`mlpMagBound … ≤ overflowBound` is exactly the checkable condition under which the forward pass provably does not.
`mlpF_isFinite` certifies the OUTPUT activations; each individual layer is certified by `layerF_isFinite` /
`reluLayerF_isFinite` when its own bound is within threshold. A uniform width bound `d` (input length and every layer's
neuron count `≤ d`) is assumed; the weight bound `Bw` and input bound `B0` are the network's magnitude budgets.
-/
import Puffer.RL.FiniteBound
open Puffer.FloatR
open Puffer.RL.FiniteBound

namespace Puffer.RL.ForwardFinite

/-- The executable ReLU does not increase magnitude: `|toReal (reluF x)| ≤ |toReal x|` (via the existing trusted
    `toReal_reluF : toReal (reluF x) = max (toReal x) 0` — `|max r 0| ≤ |r|`). -/
theorem reluF_mag_le (x : Float) : |toReal (reluF x)| ≤ |toReal x| := by
  rw [toReal_reluF]
  rcases le_total (0 : ℝ) (toReal x) with h | h
  · rw [max_eq_left h]
  · rw [max_eq_right h, abs_zero]; exact abs_nonneg _

/-- A ReLU output is overflow-free when its input magnitude is within the overflow threshold (relu only shrinks
    magnitude, so `isFinite_of_bounded` applies through it). -/
theorem reluF_isFinite (x : Float) (h : |toReal x| ≤ overflowBound) : (reluF x).isFinite = true :=
  isFinite_of_bounded _ ((reluF_mag_le x).trans h)

/-- C43's `dotBound` is nonnegative (each accumulation step is a product/sum of nonnegatives). -/
theorem dotBound_nonneg (B : ℝ) (n : ℕ) : 0 ≤ dotBound n B := by
  have h1u : (0 : ℝ) ≤ 1 + u64 := by have := u64_pos; linarith
  induction n with
  | zero => simp [dotBound]
  | succ m ih =>
      simp only [dotBound]
      exact mul_nonneg h1u (add_nonneg (mul_nonneg h1u (mul_self_nonneg B)) ih)

/-- `dotBound` grows by one accumulation step: `dotBound n B ≤ dotBound (n+1) B` (the extra `(1+u)·B²` and the `(1+u)`
    inflation only increase the bound, since everything is nonnegative). -/
theorem dotBound_le_succ (B : ℝ) (n : ℕ) : dotBound n B ≤ dotBound (n + 1) B := by
  have hu : 0 < u64 := u64_pos
  have h1u : (0 : ℝ) ≤ 1 + u64 := by linarith
  have hbb : (0 : ℝ) ≤ (1 + u64) * (B * B) := mul_nonneg h1u (mul_self_nonneg B)
  have hdn : 0 ≤ dotBound n B := dotBound_nonneg B n
  simp only [dotBound]
  nlinarith [hbb, hdn, mul_nonneg (le_of_lt hu) hdn, mul_nonneg h1u hbb]

/-- `dotBound` is monotone in the term count: `m ≤ n → dotBound m B ≤ dotBound n B`. So a per-neuron
    `dotBound (min …) B` is bounded by the layer-uniform `dotBound d B` for any width bound `d`. -/
theorem dotBound_mono (B : ℝ) {m n : ℕ} (h : m ≤ n) : dotBound m B ≤ dotBound n B := by
  induction h with
  | refl => exact le_refl _
  | step _ ih => exact ih.trans (dotBound_le_succ B _)

/-- **A linear layer**: one `dotF` of the shared input `x` with each neuron's weight row. -/
def layerF (x : List Float) (W : List (List Float)) : List Float := W.map (fun row => FiniteBound.dotF x row)

/-- A linear layer has one output per neuron (weight row). -/
theorem layerF_length (x : List Float) (W : List (List Float)) : (layerF x W).length = W.length := by
  simp [layerF]

/-- **Linear-layer magnitude propagation.** With input and weight magnitudes `≤ B` and every row's length `≤ d` (via
    `x.length ≤ d`), every output neuron has `|toReal y| ≤ dotBound d B` — each is a `dotF` bounded by C43's
    `dotF_mag_le`, then lifted to the layer-uniform width `d` by `dotBound_mono`. -/
theorem layerF_mag_le (B : ℝ) (hB : 0 ≤ B) (d : ℕ) (x : List Float) (W : List (List Float))
    (hxd : x.length ≤ d) (hx : ∀ v ∈ x, |toReal v| ≤ B)
    (hW : ∀ row ∈ W, ∀ w ∈ row, |toReal w| ≤ B) :
    ∀ y ∈ layerF x W, |toReal y| ≤ dotBound d B := by
  intro y hy
  simp only [layerF, List.mem_map] at hy
  obtain ⟨row, hrow, rfl⟩ := hy
  calc |toReal (FiniteBound.dotF x row)|
      ≤ dotBound (min x.length row.length) B := dotF_mag_le B hB x row hx (hW row hrow)
    _ ≤ dotBound d B := dotBound_mono B ((min_le_left _ _).trans hxd)

/-- **A linear layer is overflow-free** when its uniform bound `dotBound d B ≤ overflowBound` (chaining
    `layerF_mag_le` + `isFinite_of_bounded`, exactly as C43's `dotF_isFinite`). -/
theorem layerF_isFinite (B : ℝ) (hB : 0 ≤ B) (d : ℕ) (x : List Float) (W : List (List Float))
    (hxd : x.length ≤ d) (hx : ∀ v ∈ x, |toReal v| ≤ B) (hW : ∀ row ∈ W, ∀ w ∈ row, |toReal w| ≤ B)
    (hbound : dotBound d B ≤ overflowBound) :
    ∀ y ∈ layerF x W, y.isFinite = true := by
  intro y hy
  exact isFinite_of_bounded _ ((layerF_mag_le B hB d x W hxd hx hW y hy).trans hbound)

/-- **A ReLU layer**: a linear layer followed by a per-neuron ReLU. -/
def reluLayerF (x : List Float) (W : List (List Float)) : List Float := (layerF x W).map reluF

/-- A ReLU layer has one output per neuron. -/
theorem reluLayerF_length (x : List Float) (W : List (List Float)) : (reluLayerF x W).length = W.length := by
  simp [reluLayerF, layerF]

/-- **ReLU-layer magnitude propagation** — same bound as the linear layer, since ReLU does not increase magnitude
    (`reluF_mag_le`). -/
theorem reluLayerF_mag_le (B : ℝ) (hB : 0 ≤ B) (d : ℕ) (x : List Float) (W : List (List Float))
    (hxd : x.length ≤ d) (hx : ∀ v ∈ x, |toReal v| ≤ B)
    (hW : ∀ row ∈ W, ∀ w ∈ row, |toReal w| ≤ B) :
    ∀ y ∈ reluLayerF x W, |toReal y| ≤ dotBound d B := by
  intro y hy
  simp only [reluLayerF, List.mem_map] at hy
  obtain ⟨z, hz, rfl⟩ := hy
  exact (reluF_mag_le z).trans (layerF_mag_le B hB d x W hxd hx hW z hz)

/-- **A ReLU layer is overflow-free** when its uniform bound is within threshold. -/
theorem reluLayerF_isFinite (B : ℝ) (hB : 0 ≤ B) (d : ℕ) (x : List Float) (W : List (List Float))
    (hxd : x.length ≤ d) (hx : ∀ v ∈ x, |toReal v| ≤ B) (hW : ∀ row ∈ W, ∀ w ∈ row, |toReal w| ≤ B)
    (hbound : dotBound d B ≤ overflowBound) :
    ∀ y ∈ reluLayerF x W, y.isFinite = true := by
  intro y hy
  exact isFinite_of_bounded _ ((reluLayerF_mag_le B hB d x W hxd hx hW y hy).trans hbound)

/-- **The MLP forward pass**: a left fold of `reluLayerF` over the weight matrices (a ReLU after every layer). -/
def mlpF (x : List Float) (Ws : List (List (List Float))) : List Float := Ws.foldl reluLayerF x

/-- The propagated activation-magnitude bound through the forward pass: each layer maps the running bound `B` to
    `dotBound d (max B Bw)` (input bound `B`, weight bound `Bw`, width bound `d`). Grows with depth. -/
noncomputable def mlpMagBound (d : ℕ) (Bw : ℝ) : ℝ → List (List (List Float)) → ℝ
  | B, [] => B
  | B, _ :: Ws => mlpMagBound d Bw (dotBound d (max B Bw)) Ws

/-- **MLP forward-pass magnitude propagation.** With every weight magnitude `≤ Bw`, every layer's neuron count `≤ d`,
    an input of length `≤ d` with magnitudes `≤ B0`, every output activation has `|toReal y| ≤ mlpMagBound d Bw B0 Ws`.
    By induction over the layers: each `reluLayerF` step bounds its outputs by `dotBound d (max B Bw)` (via
    `reluLayerF_mag_le` with the common bound `max B Bw`) and preserves the width invariant (`reluLayerF_length` +
    the neuron-count bound), feeding the next layer. -/
theorem mlpF_mag_le (d : ℕ) (Bw : ℝ) (hBw : 0 ≤ Bw) :
    ∀ (Ws : List (List (List Float))),
      (∀ W ∈ Ws, W.length ≤ d) → (∀ W ∈ Ws, ∀ row ∈ W, ∀ w ∈ row, |toReal w| ≤ Bw) →
      ∀ (x : List Float) (B0 : ℝ), 0 ≤ B0 → x.length ≤ d → (∀ v ∈ x, |toReal v| ≤ B0) →
        ∀ y ∈ mlpF x Ws, |toReal y| ≤ mlpMagBound d Bw B0 Ws := by
  intro Ws
  induction Ws with
  | nil =>
      intro _ _ x B0 _ _ hx y hy
      simp only [mlpF, List.foldl_nil] at hy
      simpa [mlpMagBound] using hx y hy
  | cons W rest ih =>
      intro hwidth hWs x B0 _ hxd hx y hy
      have hWmem : W ∈ W :: rest := List.mem_cons.mpr (Or.inl rfl)
      have hBmax : (0 : ℝ) ≤ max B0 Bw := le_max_of_le_right hBw
      have hax : ∀ v ∈ x, |toReal v| ≤ max B0 Bw := fun v hv => (hx v hv).trans (le_max_left _ _)
      have haW : ∀ row ∈ W, ∀ w ∈ row, |toReal w| ≤ max B0 Bw :=
        fun row hrow w hw => (hWs W hWmem row hrow w hw).trans (le_max_right _ _)
      have hamag : ∀ z ∈ reluLayerF x W, |toReal z| ≤ dotBound d (max B0 Bw) :=
        reluLayerF_mag_le (max B0 Bw) hBmax d x W hxd hax haW
      have halen : (reluLayerF x W).length ≤ d := by rw [reluLayerF_length]; exact hwidth W hWmem
      have hB0' : 0 ≤ dotBound d (max B0 Bw) := dotBound_nonneg _ _
      have hy' : y ∈ mlpF (reluLayerF x W) rest := hy
      have key := ih (fun W' hW' => hwidth W' (List.mem_cons.mpr (Or.inr hW')))
        (fun W' hW' => hWs W' (List.mem_cons.mpr (Or.inr hW')))
        (reluLayerF x W) (dotBound d (max B0 Bw)) hB0' halen hamag y hy'
      simpa [mlpMagBound] using key

/-- **THE FORWARD-PASS NO-OVERFLOW CERTIFICATE.** With magnitude-bounded weights (`≤ Bw`) and input (`≤ B0`), a
    uniform width bound `d`, and the propagated bound within threshold (`mlpMagBound d Bw B0 Ws ≤ overflowBound`), the
    MLP forward-pass output activations are overflow-free. The payoff: a runnable overflow-freedom guarantee for a
    bounded MLP forward pass (`mlpF_mag_le` for the magnitude, `isFinite_of_bounded` for the finiteness step). -/
theorem mlpF_isFinite (d : ℕ) (Bw : ℝ) (hBw : 0 ≤ Bw) (Ws : List (List (List Float)))
    (hwidth : ∀ W ∈ Ws, W.length ≤ d) (hWs : ∀ W ∈ Ws, ∀ row ∈ W, ∀ w ∈ row, |toReal w| ≤ Bw)
    (x : List Float) (B0 : ℝ) (hB0 : 0 ≤ B0) (hxd : x.length ≤ d) (hx : ∀ v ∈ x, |toReal v| ≤ B0)
    (hbound : mlpMagBound d Bw B0 Ws ≤ overflowBound) :
    ∀ y ∈ mlpF x Ws, y.isFinite = true := by
  intro y hy
  exact isFinite_of_bounded _
    ((mlpF_mag_le d Bw hBw Ws hwidth hWs x B0 hB0 hxd hx y hy).trans hbound)

end Puffer.RL.ForwardFinite
