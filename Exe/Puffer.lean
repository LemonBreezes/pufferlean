/-
`puffer` — the runnable Lean binary.

  puffer                    → MLP forward pass demo (native Float; each dot has a
                              proven error bound vs ℝ, `Puffer.FloatR.dotF_error`)
  puffer train <env> [flags] → PPO+Muon training on an ocean env plugin (GPU),
                              PufferLib-style CLI (positional env,
                              --train.learning-rate / --policy.hidden-size / … flags).
  puffer help | --help      → the full command + flag surface.

Mirrors PufferLib's own entry point `pufferlib/pufferl.py` (`puffer <mode> <env>
[--section.key val]`). Also present: the verify* / bench* kernel checks (vs the Lean
f64 oracle) and the grad demo.

Grows into the full trainer (NN policy, GAE, PPO, Muon) per PLAN.md M1–M5. All
executable code here is Mathlib-free, so the binary links no Mathlib.
-/
import Puffer.Float.Exec
import Puffer.Float.ErrBnd
import Puffer.Float.Expr
import Puffer.Float.AutoDiff
import Puffer.RL.Train
import Puffer.RL.NNTrain
import Puffer.RL.VecTrain
import Puffer.RL.ContVecTrain
import Puffer.RL.RecVecTrain
import Puffer.RL.CnnVecTrain
import Puffer.Float.FFI
import Puffer.Float.BLAS
import Puffer.Float.CUDA
import Puffer.RL.FFITrain
import Puffer.RL.MinGRUTrain
import Puffer.RL.Wandb
import Puffer.Plugin
import Puffer.Check.Core
import Puffer.Check.Parse

open Puffer.FloatR
open Puffer.FloatR.AD
open Puffer.RL.NNTrain (trainPluginEnv trainPluginEnvMD trainPluginEnvCont trainPluginEnvRec trainPluginEnvMinGRU trainPluginEnvMinGRUMD)
open Puffer.RL.Train (rngNext uniform01)

/-! ### `verify` mode — emit verified kernels + their proven error bounds as exact
    f64 bits, for cross-checking against Python exact arithmetic (tools/verify_ref.py). -/

/-- IEEE-754 f64 bit pattern as a decimal string (lossless; Python reconstructs it). -/
def bits (x : Float) : String := toString (Float.toBits x)
def bitsArr (xs : List Float) : String := "[" ++ String.intercalate "," (xs.map bits) ++ "]"
def qs (s : String) : String := "\"" ++ s ++ "\""              -- JSON-quote
def obj (fields : List (String × String)) : String :=          -- JSON object
  "{" ++ String.intercalate "," (fields.map (fun kv => qs kv.1 ++ ":" ++ kv.2)) ++ "}"
def bitsA (xs : Array Float) : String := "[" ++ String.intercalate "," (xs.toList.map bits) ++ "]"
def bitsM (m : Array (Array Float)) : String := "[" ++ String.intercalate "," (m.toList.map bitsA) ++ "]"
/-- An MLP's four parameter tensors as a JSON object of exact f64 bits. -/
def wobj (W1 : Array (Array Float)) (b1 : Array Float) (W2 : Array (Array Float)) (b2 : Array Float) : String :=
  obj [("W1", bitsM W1), ("b1", bitsA b1), ("W2", bitsM W2), ("b2", bitsA b2)]

/-- JSON array of a `List (List Float)` as exact f64 bits. -/
def bitsMatL (m : List (List Float)) : String :=
  "[" ++ String.intercalate "," (m.map bitsArr) ++ "]"

/-- Uniform `Float` in `[lo, hi)`. -/
def randF (lo hi : Float) (rng : UInt64) : Float × UInt64 :=
  let (w, rng') := rngNext rng
  (lo + (hi - lo) * uniform01 w, rng')

/-- A random `List Float` of length `n` in `[lo, hi)`. -/
def randList (n : Nat) (lo hi : Float) (rng : UInt64) : List Float × UInt64 := Id.run do
  let mut rng := rng
  let mut xs : List Float := []
  for _ in [0:n] do
    let (a, r) := randF lo hi rng; rng := r
    xs := xs ++ [a]
  return (xs, rng)

/-- A random `rows×cols` matrix (list of rows). -/
def randRows (rows cols : Nat) (lo hi : Float) (rng : UInt64) : List (List Float) × UInt64 := Id.run do
  let mut rng := rng
  let mut m : List (List Float) := []
  for _ in [0:rows] do
    let (row, r) := randList cols lo hi rng; rng := r
    m := m ++ [row]
  return (m, rng)

def runVerify : IO Unit := do
  let mut rng : UInt64 := 0xC0FFEE
  for n in [3, 5, 8, 16] do          -- dot product (a reduction)
    let mut x : List Float := []
    let mut w : List Float := []
    for _ in [0:n] do
      let (a, r1) := randF (-2.0) 2.0 rng; rng := r1
      let (b, r2) := randF (-2.0) 2.0 rng; rng := r2
      x := x ++ [a]; w := w ++ [b]
    IO.println (obj [("op", qs "dot"), ("x", bitsArr x), ("w", bitsArr w),
      ("result", bits (dotF x w)), ("bound", bits (dotErrBndF x w))])
  for n in [4, 8, 16, 32] do          -- GAE (a backward recurrence)
    let (wv, r) := randF 0.5 0.99 rng; rng := r
    let mut ds : List Float := []
    for _ in [0:n] do
      let (d, r2) := randF (-1.0) 1.0 rng; rng := r2
      ds := ds ++ [d]
    IO.println (obj [("op", qs "gae"), ("w", bits wv), ("deltas", bitsArr ds),
      ("result", bits (gaeHeadF wv ds)), ("bound", bits (gaeErrBndF wv ds))])
  for _ in [0:6] do                   -- Muon Newton–Schulz map (a polynomial circuit)
    let (σ, r) := randF (-1.0) 1.0 rng; rng := r
    let a : Float := 4.0848; let b : Float := -6.8946; let c : Float := 2.9270
    IO.println (obj [("op", qs "nsscalar"), ("a", bits a), ("b", bits b), ("c", bits c),
      ("sigma", bits σ), ("result", bits (nsScalarF a b c σ)), ("bound", bits (nsScalarErrBndF a b c σ))])
  for dims in [(4, 5, 3), (6, 4, 2), (3, 6, 4)] do   -- MLP forward pass (dot→bias→relu→dot→bias)
    let (din, H, dout) := dims
    let (W1, r1) := randRows H din (-1.0) 1.0 rng; rng := r1
    let (b1, r2) := randList H (-0.5) 0.5 rng; rng := r2
    let (W2, r3) := randRows dout H (-1.0) 1.0 rng; rng := r3
    let (b2, r4) := randList dout (-0.5) 0.5 rng; rng := r4
    let (x, r5) := randList din (-1.0) 1.0 rng; rng := r5
    let h := fwdHidden W1 b1 x                        -- proven hidden (neuron_error)
    let logits := fwdLogits W1 b1 W2 b2 x             -- proven logits (logit_error)
    IO.println (obj [("op", qs "fwd"),
      ("W1", bitsMatL W1), ("b1", bitsArr b1), ("W2", bitsMatL W2), ("b2", bitsArr b2), ("x", bitsArr x),
      ("hidden", bitsArr h), ("hiddenBnd", bitsArr (fwdHiddenErr W1 b1 x)),
      ("logits", bitsArr logits),
      ("logitRoundBnd", bitsArr ((W2.zip b2).map (fun wb => z1ErrF wb.1 wb.2 h)))])

/-! ### f64-bit emitters for the Python ℝ-reference cross-checks — the `verify-grad` /
`verify-adam` / `verify-vtrace` modes below each dump a fixed computation (and its proven
error bound) as exact f64 bits; the matching `tools/*_ref.py` recompute in high-precision
decimal (an ℝ proxy) and check the Lean output tracks the real-arithmetic result.

(The earlier `verify-update` whole-update cross-check was retired with the Lean envs — its
rollout came from a Lean env `Model`, gone under the C-plugin architecture.) -/

/-! ### `verify-grad` mode — the "error-bound mode": run the VERIFIED forward-mode AD.

For a fixed `Expr` and `Float` environment, emit the gradient `dF` and its PROVEN error bound
`derivErrBndF` (the `Float` mirror of the ℝ `derivErrBnd`, certified by `dF_error` in
`AutoDiffR.lean`). `tools/grad_ref.py` computes the exact-ℝ gradient `derivR` in high-precision
`decimal` (exp/log are transcendental) and checks `|toReal(dF) − derivR| ≤ derivErrBndF`. -/

open Puffer.FloatR.ADR (Expr evalF dF derivErrBndF clampE)

/-- Serialize an `Expr` as JSON (constants as exact f64 bits) for the Python reference. -/
def exprJson : Expr → String
  | .var i => obj [("op", qs "var"), ("i", toString i)]
  | .const c => obj [("op", qs "const"), ("c", bits c)]
  | .add a b => obj [("op", qs "add"), ("a", exprJson a), ("b", exprJson b)]
  | .sub a b => obj [("op", qs "sub"), ("a", exprJson a), ("b", exprJson b)]
  | .mul a b => obj [("op", qs "mul"), ("a", exprJson a), ("b", exprJson b)]
  | .scale c a => obj [("op", qs "scale"), ("c", bits c), ("a", exprJson a)]
  | .exp a => obj [("op", qs "exp"), ("a", exprJson a)]
  | .log a => obj [("op", qs "log"), ("a", exprJson a)]
  | .relu a => obj [("op", qs "relu"), ("a", exprJson a)]
  | .max a b => obj [("op", qs "max"), ("a", exprJson a), ("b", exprJson b)]
  | .min a b => obj [("op", qs "min"), ("a", exprJson a), ("b", exprJson b)]

/-- Environment from a value list (out-of-range variables read 0). -/
def envOf (xs : List Float) : Nat → Float := fun i => xs.getD i 0.0

def runVerifyGrad : IO Unit := do
  let e1 : Expr := .mul (.var 0) (.var 1)                                  -- bilinear
  let e2 : Expr := .exp (.add (.var 0) (.mul (.var 1) (.var 2)))            -- exp of a sum
  let e3 : Expr := .log (.add (.const 2.0) (.mul (.var 0) (.var 0)))        -- log(2 + x₀²) > 0
  let e4 : Expr := .sub (.scale 0.5 (.mul (.var 0) (.var 0)))               -- ½x₀² − log(1.5 + eˣ¹)
                       (.log (.add (.const 1.5) (.exp (.var 1))))
  let e5 : Expr := .relu (.add (.var 0) (.mul (.const 2.0) (.var 1)))       -- relu(x₀ + 2x₁), off-kink
  let e6 : Expr := .max (.var 0) (.mul (.const 3.0) (.var 1))               -- max(x₀, 3x₁), off-kink
  let e7 : Expr := clampE (.add (.var 0) (.var 1)) (-0.5) 0.5               -- clamp(x₀+x₁, −0.5, 0.5)
  let cases : List (Expr × List Float × Nat) :=
    [(e1, [1.3, -0.7, 0.0], 3), (e2, [0.4, -1.1, 0.9], 3),
     (e3, [1.7, 0.0, 0.0], 3), (e4, [-0.6, 0.8, 0.0], 3),
     (e5, [0.9, 0.3, 0.0], 3), (e6, [1.3, -0.7, 0.0], 3),
     (e7, [0.9, 0.4, 0.0], 3)]
  for c in cases do
    let (e, xs, n) := c
    let σ := envOf xs
    let grad := (List.range n).map (fun k => dF e σ k)
    let bound := (List.range n).map (fun k => derivErrBndF e σ k)
    IO.println (obj [("op", qs "grad"), ("expr", exprJson e), ("env", bitsArr xs),
      ("nvars", toString n), ("grad", bitsArr grad), ("bound", bitsArr bound)])
def runVerifyAdam : IO Unit := do
  let b1 : Float := 0.9; let c1 : Float := 0.1        -- β₁, 1−β₁
  let b2 : Float := 0.999; let c2 : Float := 0.001    -- β₂, 1−β₂
  let lr : Float := 1.0e-3; let eps : Float := 1.0e-8
  -- (p, m, v, g): weight, prev 1st moment, prev 2nd moment (≥ 0), gradient.
  let cases : List (Float × Float × Float × Float) :=
    [ (0.5, 0.1, 0.04, -0.3), (-0.2, -0.05, 0.01, 0.7),
      (1.0, 0.2, 0.09, 0.0), (0.0, 0.0, 0.0, 0.5) ]
  for c in cases do
    let (p, m, v, g) := c
    IO.println (obj [("op", qs "adam"),
      ("p", bits p), ("m", bits m), ("v", bits v), ("g", bits g),
      ("lr", bits lr), ("b1", bits b1), ("c1", bits c1), ("b2", bits b2), ("c2", bits c2),
      ("eps", bits eps),
      ("result", bits (adamStepF p m v g lr b1 c1 b2 c2 eps))])

/-! ### `verify-vtrace` mode — emit `computePuffAdvantage` on fixed segments.

Exercises PufferLib's `compute_puff_advantage` (V-Trace/GAE) over segments with mid-segment
terminals AND non-unit importance ratios (so `min(imp,clip)` and the `nnt` masking are hit,
not just the `imp=1`⇒GAE degenerate case). `tools/vtrace_ref.py` reconstructs PufferLib's raw
[values, rewards, dones] buffers from these (Lean's Transition convention shifted by one) and
runs the VERBATIM `puff_advantage_cpu` kernel in f64, asserting each advantage/return matches
at floating-point-roundoff scale — validating both the recurrence and the index mapping. -/
def runVerifyVtrace : IO Unit := do
  let gamma : Float := 0.99; let lam : Float := 0.95
  let rhoClip : Float := 1.0; let cClip : Float := 1.0
  -- each case: list of (reward, value, terminal, importance) per transition
  let cases : List (List (Float × Float × Bool × Float)) :=
    [ [(0.0, 0.5, false, 1.0), (1.0, 0.4, false, 1.0), (0.0, 0.3, false, 1.0),
       (0.0, 0.2, false, 1.0), (1.0, 0.1, false, 1.0), (0.0, 0.0, false, 1.0)],
      [(0.2, 0.5, false, 0.7), (-0.5, 0.3, false, 1.4), (1.0, 0.6, true, 0.9),
       (0.1, 0.2, false, 2.0), (0.0, 0.4, false, 0.3)],
      [(1.0, 0.9, false, 1.2), (0.5, 0.1, true, 0.8), (0.0, 0.7, false, 1.0), (0.3, 0.5, false, 0.5)] ]
  for c in cases do
    let traj : Array Puffer.RL.NNTrain.Transition := (c.map (fun q =>
      ({ obs := #[], action := 0, reward := q.1, value := q.2.1, oldLogp := 0.0,
         terminal := q.2.2.1 } : Puffer.RL.NNTrain.Transition))).toArray
    let imp : Array Float := (c.map (fun q => q.2.2.2)).toArray
    let (adv, ret) := Puffer.RL.NNTrain.computePuffAdvantage traj imp gamma lam rhoClip cClip
    let termStr := "[" ++ String.intercalate "," (traj.toList.map (fun t => if t.terminal then "1" else "0")) ++ "]"
    IO.println (obj [("op", qs "vtrace"),
      ("gamma", bits gamma), ("lam", bits lam), ("rhoClip", bits rhoClip), ("cClip", bits cClip),
      ("rewards", bitsA (traj.map (·.reward))), ("values", bitsA (traj.map (·.value))),
      ("terminals", termStr), ("importance", bitsA imp),
      ("adv", bitsA adv), ("ret", bitsA ret)])

/-! ### `train` / `eval` — the PufferLib-style CLI.

Mirrors PufferLib's `puffer <mode> <env_name> [--section.key value]` entry point
(`pufferlib/pufferl.py`): `mode` and `env_name` are positional, hyperparameters
are namespaced flags (`--train.learning-rate`, `--policy.hidden-size`, …). Flags
accept both `--flag value` and `--flag=value`. Defaults follow PufferLib's
`config/default.ini` where the algorithm is comparable, adjusted for this repo's
single-env / episodic PPO (which trains per-episode rather than over a huge
vectorized batch). -/

/-- Parsed training/eval configuration (PufferLib-style hyperparameters). -/
structure Config where
  env : String := "squared"
  totalTimesteps : Nat := 300000      -- PufferLib: train.total_timesteps
  learningRate : Float := 0.03        -- PufferLib default.ini: 0.015 (episodic PPO here likes a bit more)
  hiddenSize : Nat := 16              -- PufferLib policy.hidden_size (128 there; smaller nets suffice here)
  seed : UInt64 := 42                 -- PufferLib train.seed
  epochs : Nat := 4                   -- PPO passes per rollout (PufferLib uses replay_ratio)
  gamma : Float := 0.99               -- PufferLib train.gamma (0.995)
  gaeLambda : Float := 0.95           -- PufferLib train.gae_lambda (0.90)
  vfCoef : Float := 0.5               -- PufferLib train.vf_coef (2.0)
  entCoef : Float := 0.01             -- PufferLib train.ent_coef (0.001)
  clipCoef : Float := 0.2             -- PufferLib train.clip_coef
  -- vectorized GPU-trainer hyperparameters (used by `train <env>`)
  numEnvs : Nat := 8                  -- PufferLib vec.num_envs (parallel env instances)
  horizon : Nat := 64                -- PufferLib train.horizon (config/default.ini:85). Was 128 —
                                     -- double their rollout length on every env that never overrides it.
  numMB : Nat := 4                   -- PufferLib train.num_minibatches
  minibatchSize : Nat := 8192        -- PufferLib train.minibatch_size (segments/minibatch = this/horizon)
  replayRatio : Float := 1.0         -- PufferLib train.replay_ratio (num_minibatches = replay_ratio·batch/mb_size)
  maxGradNorm : Float := 0.5         -- PufferLib train.max_grad_norm (1.5)
  -- PufferLib parity hyperparameters (torch_pufferl.py), defaulting to its values.
  beta1 : Float := 0.95              -- Muon momentum (PufferLib train.beta1)
  vfClipCoef : Float := 0.2          -- value-loss clip (PufferLib train.vf_clip_coef)
  prioAlpha : Float := 0.8           -- prioritized-replay exponent (PufferLib train.prio_alpha)
  prioBeta0 : Float := 0.2           -- prioritized-replay IS-anneal start (PufferLib train.prio_beta0)
  minLrRatio : Float := 0.0          -- cosine-LR floor ratio (PufferLib train.min_lr_ratio)
  -- Muon's `eps` (PufferLib train.eps, passed to `Muon(params, lr, momentum=beta1, eps=config['eps'])`
  -- at torch_pufferl.py:166; it is the Newton–Schulz Frobenius-norm floor). This was parsed into
  -- NOTHING until now: the dispatch hardcoded 1.0e-12 on the recurrent paths and 1.0e-7 on the
  -- feed-forward ones, so all 26 per-env sweep values in config/*.ini (moba 1.89701e-12, breakout
  -- 8.33946e-05, terraform/pacman/pong 1e-4, boxoban/nethack 1e-14, …) were silently discarded.
  -- Default is PufferLib's config/default.ini `eps = 1e-12`.
  eps : Float := 1.0e-12             -- PufferLib train.eps (Muon Newton–Schulz norm floor)
  -- V-Trace importance-ratio clips (PufferLib train.vtrace_rho_clip / vtrace_c_clip, consumed by
  -- `compute_puff_advantage`). These were parsed into NOTHING until now: the recurrent trainers held
  -- them as hardcoded local `let`s of 1.0/1.0, so every per-env sweep value in config/*.ini (28 envs
  -- set them; maze ρ=5, pong 4.88, terraform 4.25, moba 1.76/1.33) was silently discarded. Defaults
  -- are config/default.ini's 1.0/1.0, i.e. the exact values the old hardcoded lets used.
  vtraceRhoClip : Float := 1.0       -- ρ̄ (PufferLib train.vtrace_rho_clip)
  vtraceCClip : Float := 1.0         -- c̄ (PufferLib train.vtrace_c_clip)
  -- LR annealing on/off (PufferLib train.anneal_lr). When FALSE, LR is held CONSTANT at the base
  -- `learning_rate` for the whole run — torch_pufferl.py:267 skips the cosine block, pufferlib.cu:1553
  -- guards the cosine on `anneal_lr`. robocode.ini/continuous.ini set it 0/False; default.ini sets 1.
  -- Was parsed into NOTHING: every env always cosine-annealed, so those two ran a decaying LR where
  -- PufferLib runs a constant one. Implemented as an EFFECTIVE min_lr_ratio of 1.0 (cosineLr with
  -- minLrRatio=1 is exactly the constant `lr`), so it needs no new trainer parameter. See `effMinLr`.
  annealLr : Bool := true            -- PufferLib train.anneal_lr
  -- Entropy-coefficient cosine anneal (PufferLib train.anneal_ent_coef / train.min_ent_coef_ratio,
  -- read at bindings.cu:422-423, stored at pufferlib.cu:294-295, consumed at pufferlib.cu:1563-1566).
  -- When `anneal_ent_coef` is on, ent_coef cosine-decays from its base value to `min_ent_coef_ratio·ent_coef`
  -- over training — the SAME `cosine_annealing()` shape as the LR schedule. These were parsed into NOTHING
  -- until now: the MinGRU trainers held ent_coef constant, so `config/chess.ini` (the one vendored ini that
  -- sets `anneal_ent_coef = 1`) was silently ignored. Defaults are config/default.ini's `anneal_ent_coef = 0`
  -- / `min_ent_coef_ratio = 0.1`, i.e. OFF ⇒ entCoefNow == entCoef and every env but chess is bit-identical.
  annealEntCoef : Bool := false      -- PufferLib train.anneal_ent_coef (cosine-decay the entropy bonus)
  minEntCoefRatio : Float := 0.1     -- PufferLib train.min_ent_coef_ratio (ent-coef floor ÷ ent_coef)
  -- Policy architecture (PufferLib `[torch] network` / `[policy] num_layers`). `network` selects the
  -- policy core: MinGRU (PufferLib's default, recurrent), GRU, LSTM (recurrent), or MLP (feed-forward).
  -- PufferLib's default.ini is `[torch] network = MinGRU` (recurrent, on GPU) and we now match it. The old
  -- MLP default dated from the CPU-only era, when running their real config was infeasible — that comment
  -- said "a GPU MinGRU is the fix to match PufferLib's default at speed", and that is exactly what the
  -- native MinGRU trainer now is (~26M SPS on squared, at parity with their _C). Until this flip,
  -- `puffer train <env>` silently trained a FEED-FORWARD policy against their RECURRENT one, which is a
  -- large behavioural gap on partially-observed envs (breakout episode return 32.7 vs their 232.9).
  -- Multi-discrete / continuous envs still fall through to their MLP paths (no recurrent core built for
  -- those), exactly as before.
  network : String := "MinGRU"
  numLayers : Nat := 4               -- PufferLib [policy] num_layers (recurrent depth)
  loadPath : Option String := none    -- --load: seed initial policy weights from this checkpoint (STREAM 3)
  -- STREAM 3: checkpoint cadence in UPDATES (PufferLib train.checkpoint_interval). 0 = final-only.
  -- Every `checkpointInterval` updates (and always the final update) the resident policy weights are
  -- written to `checkpoints/<env>/<seed>/<step>.bin`; `--load <path>` (or `puffer eval`) reads them back.
  checkpointInterval : Nat := 0
  -- wandb live tracking (PufferLib `pufferl.py`: `--wandb` / `--wandb-project` / `--wandb-group` / `--tag`).
  -- `--wandb` spawns `tools/puffer_track.py --daemon` and streams it the dashboard dict each tick.
  wandb : Bool := false
  wandbProject : String := "puffer4"   -- PufferLib default wandb project
  wandbGroup : String := "debug"       -- PufferLib default wandb group
  wandbTag : Option String := none     -- --tag: single run tag (else no tags)

/-- Strip a leading `--`; PufferLib also accepts the `train.`/`policy.` namespace,
    which we ignore (flatten to the leaf key). Underscores and hyphens both work. -/
def normKey (s : String) : String :=
  let s := if s.startsWith "--" then (String.ofList (s.toList.drop 2)) else s
  -- drop a `section.` prefix (train./policy./base./vec.)
  let s := match s.splitOn "." with
    | [_, leaf] => leaf
    | _ => s
  (s.replace "_" "-")

/-- Parse a Nat, defaulting on failure. Accepts underscores (e.g. `1_000_000`).

Also parses an integer config value the way PufferLib does. PufferLib's config values are all parsed as C `double` and then
assigned into `int` fields, which TRUNCATES toward zero — and their sweep-tuned inis really do ship
fractional integers (`num_layers = 1.6302`, `2.20258`, `7.44076`, … in 17 of the vendored files).
Until this accepted a decimal, every one of those keys failed `toNat?` and silently fell back to our
built-in default, so e.g. tetris ran 4 MinGRU layers where PufferLib runs `int(2.20258) = 2`. -/
def parseNat (s : String) (dflt : Nat) : Nat :=
  let s := s.replace "_" ""
  match s.toNat? with
  | some n => n
  | none   =>
    -- decimal → truncate toward zero, mirroring their double→int assignment
    match s.splitOn "." with
    | [i, _] => (i.toNat?).getD dflt
    | _      => dflt

/-- Parse a Float via the substrate's decoder; defaults on failure. -/
def parseFloat? (s : String) : Option Float :=
  -- Lean core has no String→Float; parse a decimal by hand (sign, int, frac, exp).
  let s := s.replace "_" ""
  Id.run do
    let chars := s.toList
    if chars.isEmpty then return none
    let (neg, chars) := match chars with
      | '-' :: r => (true, r)
      | '+' :: r => (false, r)
      | _ => (false, chars)
    -- split on 'e'/'E'
    let str := String.ofList chars
    let (mant, expPart) := match str.splitOn "e" with
      | [m, e] => (m, some e)
      | _ => match str.splitOn "E" with
        | [m, e] => (m, some e)
        | _ => (str, none)
    let (intPart, fracPart) := match mant.splitOn "." with
      | [i, f] => (i, f)
      | [i] => (i, "")
      | _ => (mant, "")
    let intVal := (intPart.toNat?).getD 0
    let mut v : Float := Float.ofNat intVal
    if fracPart ≠ "" then
      match fracPart.toNat? with
      | some fv =>
          let scale := (10.0 : Float) ^ (Float.ofNat fracPart.length)
          v := v + Float.ofNat fv / scale
      | none => pure ()
    match expPart with
    | some e =>
        let (eneg, edigits) := match e.toList with
          | '-' :: r => (true, String.ofList r)
          | '+' :: r => (false, String.ofList r)
          | _ => (false, e)
        match edigits.toNat? with
        | some ev =>
            let p := (10.0 : Float) ^ (Float.ofNat ev)
            v := if eneg then v / p else v * p
        | none => pure ()
    | none => pure ()
    return some (if neg then -v else v)

def parseFloat (s : String) (dflt : Float) : Float := (parseFloat? s).getD dflt

/-- Parse a boolean config value. PufferLib configs write these as `1`/`0` and `True`/`False`
    (e.g. `anneal_lr = 0`, `continuous.ini`'s `anneal_lr = False`, `chess.ini`'s `anneal_ent_coef = 1`).
    Textual true/false/yes/no/on/off match directly; anything else falls back to PufferLib's own
    double→bool rule (it parses every config value as a C `double` and assigns to a `bool`, so any
    nonzero number is true). -/
def parseBool (s : String) (dflt : Bool) : Bool :=
  match (s.trim.toList.map Char.toLower).asString with
  | "0" | "false" | "no" | "off" | "" => false
  | "1" | "true"  | "yes" | "on"      => true
  | _ => match parseFloat? s with
         | some f => f != 0.0
         | none   => dflt
/-- A parsed flag map (assoc list, so no `Std.HashMap` import / no Mathlib). -/
abbrev FlagMap := List (String × String)

/-- Normalize args into a key→value assoc list. Supports `--flag value`,
    `--flag=value`, and bare boolean flags. The first non-flag token is treated as
    the positional `env_name`. Returns `(envName?, flagMap)`. -/
def parseFlags (args : List String) : Option String × FlagMap := Id.run do
  let mut m : FlagMap := []
  let mut env : Option String := none
  let mut rest := args
  while !rest.isEmpty do
    match rest with
    | [] => break
    | tok :: tl =>
      if tok.startsWith "--" then
        let body := String.ofList (tok.toList.drop 2)
        let eqParts := body.splitOn "="
        if eqParts.length ≥ 2 then
            -- --flag=value
            let k := normKey ("--" ++ eqParts.head!)
            let v := String.intercalate "=" eqParts.tail
            m := m ++ [(k, v)]; rest := tl
        else
            -- --flag value  (or bare flag if next is another flag / absent)
            match tl with
            | v :: tl2 =>
              if v.startsWith "--" then
                m := m ++ [(normKey tok, "1")]; rest := tl
              else
                m := m ++ [(normKey tok, v)]; rest := tl2
            | [] => m := m ++ [(normKey tok, "1")]; rest := []
      else
        if env.isNone then env := some tok
        rest := tl
  return (env, m)

/-- Build a `Config` from the flag map, applying PufferLib-style keys over defaults. -/
def configOf (env : String) (m : FlagMap) : Config :=
  let g (k : String) : Option String := (m.find? (fun kv => kv.1 == normKey k)).map (·.2)
  let d : Config := { env := env }
  { env := env
    totalTimesteps := (g "total-timesteps").elim d.totalTimesteps (parseNat · d.totalTimesteps)
    learningRate := (g "learning-rate").elim ((g "lr").elim d.learningRate (parseFloat · d.learningRate))
                                             (parseFloat · d.learningRate)
    hiddenSize := (g "hidden-size").elim ((g "hidden").elim d.hiddenSize (parseNat · d.hiddenSize))
                                          (parseNat · d.hiddenSize)
    seed := (g "seed").elim d.seed (fun s => UInt64.ofNat (parseNat s d.seed.toNat))
    epochs := (g "epochs").elim ((g "update-epochs").elim d.epochs (parseNat · d.epochs))
                                (parseNat · d.epochs)
    gamma := (g "gamma").elim d.gamma (parseFloat · d.gamma)
    gaeLambda := (g "gae-lambda").elim ((g "lam").elim d.gaeLambda (parseFloat · d.gaeLambda))
                                        (parseFloat · d.gaeLambda)
    vfCoef := (g "vf-coef").elim d.vfCoef (parseFloat · d.vfCoef)
    entCoef := (g "ent-coef").elim d.entCoef (parseFloat · d.entCoef)
    clipCoef := (g "clip-coef").elim ((g "clip").elim d.clipCoef (parseFloat · d.clipCoef))
                                      (parseFloat · d.clipCoef)
    numEnvs := (g "num-envs").elim d.numEnvs (parseNat · d.numEnvs)
    horizon := (g "horizon").elim ((g "bptt-horizon").elim d.horizon (parseNat · d.horizon))
                                   (parseNat · d.horizon)
    numMB := (g "num-minibatches").elim ((g "num-mb").elim d.numMB (parseNat · d.numMB))
                                         (parseNat · d.numMB)
    minibatchSize := (g "minibatch-size").elim d.minibatchSize (parseNat · d.minibatchSize)
    replayRatio := (g "replay-ratio").elim d.replayRatio (parseFloat · d.replayRatio)
    maxGradNorm := (g "max-grad-norm").elim d.maxGradNorm (parseFloat · d.maxGradNorm)
    beta1 := (g "beta1").elim d.beta1 (parseFloat · d.beta1)
    vfClipCoef := (g "vf-clip-coef").elim d.vfClipCoef (parseFloat · d.vfClipCoef)
    prioAlpha := (g "prio-alpha").elim d.prioAlpha (parseFloat · d.prioAlpha)
    prioBeta0 := (g "prio-beta0").elim d.prioBeta0 (parseFloat · d.prioBeta0)
    minLrRatio := (g "min-lr-ratio").elim d.minLrRatio (parseFloat · d.minLrRatio)
    -- FLOAT parser: every per-env value is an exponent-form decimal (`eps = 1.89701e-12`), which
    -- parseNat would truncate to 0. `adam_eps` is a DIFFERENT key (PufferLib never reads it) — not this.
    eps := (g "eps").elim d.eps (parseFloat · d.eps)
    -- FLOAT parser (not parseNat): the sweep writes these as decimals (`vtrace_rho_clip = 1.76073`),
    -- and parseNat truncates decimals toward zero — 1.76073 would land as the integer 1.
    vtraceRhoClip := (g "vtrace-rho-clip").elim d.vtraceRhoClip (parseFloat · d.vtraceRhoClip)
    vtraceCClip := (g "vtrace-c-clip").elim d.vtraceCClip (parseFloat · d.vtraceCClip)
    -- parseBool (not parseFloat): continuous.ini writes `anneal_lr = False`, not a number.
    annealLr := (g "anneal-lr").elim d.annealLr (parseBool · d.annealLr)
    -- BOOL parser (double→bool, per PufferLib): `anneal_ent_coef = 0`/`= 1` in the inis. FLOAT parser
    -- for the ratio (not parseNat — `min_ent_coef_ratio = 0.1` would truncate to the integer 0).
    annealEntCoef := (g "anneal-ent-coef").elim d.annealEntCoef (parseBool · d.annealEntCoef)
    minEntCoefRatio := (g "min-ent-coef-ratio").elim d.minEntCoefRatio (parseFloat · d.minEntCoefRatio)
    network := (g "network").getD d.network                       -- [torch] network (MinGRU/GRU/LSTM/MLP)
    numLayers := (g "num-layers").elim d.numLayers (parseNat · d.numLayers)
    loadPath := g "load"
    checkpointInterval := (g "checkpoint-interval").elim d.checkpointInterval (parseNat · d.checkpointInterval)
    wandb := (g "wandb").elim d.wandb (parseBool · d.wandb)   -- store_true (`--wandb`) or ini 1/0
    wandbProject := (g "wandb-project").getD d.wandbProject
    wandbGroup := (g "wandb-group").getD d.wandbGroup
    wandbTag := g "tag" }

/-! ### Config-file parity (`config/default.ini` + per-env overrides).

Mirrors PufferLib's INI config layering (`config/default.ini` ← `config/ocean/<env>.ini`
← CLI flags, later overriding earlier). We flatten `[section] key = value` to the same
leaf keys the flag parser uses (`normKey`), so a config file and a CLI flag are
interchangeable and share one precedence chain. -/

/-- Parse an INI-ish config file into a flag map. `[section]` headers are dropped (keys
    flatten to their leaf via `normKey`); `#`/`;` full-line comments and blanks are skipped. -/
def parseIni (contents : String) : FlagMap := Id.run do
  let mut m : FlagMap := []
  let mut inEnv := false          -- `[env]` keys are the ENV's own kwargs, not trainer/vec settings
  for raw in contents.splitOn "\n" do
    let line := ((raw.splitOn "#").headD raw).trim   -- strip inline `# …` comments
    if line.isEmpty || line.startsWith "#" || line.startsWith ";" then
      continue
    if line.startsWith "[" then
      -- Track ONLY the [env] boundary. Flattening every section to bare leaf names collides when a
      -- key name is used by both the vec and the env: whisker_racer.ini has `[vec] num_envs = 8`
      -- AND `[env] num_envs = 1024`, so the env's kwarg was resolving the trainer's env count to 8
      -- — a batch of 8, 23437 tiny updates, 0.09M SPS (100x below the sweep median) and a harness
      -- timeout. `[env]` keys are now namespaced `env.<key>`, which is the form the CLI already
      -- accepts (`--env.map-size`) and which `normKey`'s section-prefix drop already understands.
      inEnv := (line.trim.toLower.startsWith "[env]")
      continue
    match line.splitOn "=" with
    | k :: vs =>
      let key := normKey k.trim
      let val := (String.intercalate "=" vs).trim
      -- `num_envs` outside [env] is NOT a PufferLib vec key — their [vec] is total_agents /
      -- num_buffers / num_threads, and `create_pufferl` never reads num_envs, so upstream ignores it
      -- and falls back to default.ini's total_agents. We were honouring it as the trainer's env
      -- count: whisker_racer.ini's `[vec] num_envs = 8` gave a batch of 8 (23437 updates, 0.09M SPS,
      -- 100x below the sweep median) where PufferLib runs 4096 agents. Only a command-line
      -- --num-envs should set it.
      let drop := (key == "num-envs" && !inEnv)
      if !val.isEmpty && !drop then m := m ++ [((if inEnv then "env." ++ key else key), val)]
    | _ => pure ()
  return m

/-- Read + parse an INI file if it exists, else an empty flag map. -/
def loadIniIfExists (path : String) : IO FlagMap := do
  if ← System.FilePath.pathExists path then
    return parseIni (← IO.FS.readFile path)
  else
    return []

/-- Layer config sources with CLI-wins precedence: `cli ++ perEnv ++ default` (the flag
    lookup `find?` returns the FIRST match, so earlier = higher priority). `--config <path>`
    overrides which base file is read (default `config/default.ini`). -/
def loadConfigFlags (env : String) (cli : FlagMap) : IO FlagMap := do
  let basePath := (cli.find? (fun kv => kv.1 == "config")).elim "config/default.ini" (·.2)
  let dflt ← loadIniIfExists basePath
  let perEnv ← loadIniIfExists s!"config/{env}.ini"          -- PufferLib layout (config/<env>.ini)
  let perEnvOcean ← loadIniIfExists s!"config/ocean/{env}.ini"
  return cli ++ perEnv ++ perEnvOcean ++ dflt

/-! ### STREAM 3 — CLI fidelity: reject genuinely unknown flags (PufferLib's argparse errors on them).

Validation is on the CLI tokens only (config-file keys are never rejected). A flag is accepted if it is
namespaced (`--section.key` — an explicit override, incl. `--env.<key>` env passthrough) or if its bare
leaf is a known trainer/vec/policy key. Anything else (a typo like `--totaltimesteps`) errors out. -/

/-- The set of bare flag leaves the trainer/vec/policy config understands (every `configOf` lookup plus
    its aliases, the vec keys, and the STREAM-3 additions). Namespaced `--section.key` forms bypass this. -/
def knownFlagKeys : List String :=
  [ -- train.*
    "total-timesteps", "learning-rate", "lr", "epochs", "update-epochs", "gamma", "gae-lambda", "lam",
    "vf-coef", "ent-coef", "clip-coef", "clip", "seed", "horizon", "bptt-horizon",
    "num-minibatches", "num-mb", "minibatch-size", "replay-ratio", "max-grad-norm", "beta1",
    "vf-clip-coef", "prio-alpha", "prio-beta0", "min-lr-ratio", "eps", "vtrace-rho-clip", "vtrace-c-clip",
    "anneal-lr", "checkpoint-interval",
    -- policy.* / torch.*
    "network", "num-layers", "hidden-size", "hidden",
    -- vec.*
    "num-envs", "total-agents", "num-threads", "num-buffers", "buffers",
    -- run control (parsed by configOf; some are inert but accepted)
    "load", "log", "config",
    -- wandb tracking (PufferLib argparse: --wandb / --wandb-project / --wandb-group / --tag)
    "wandb", "wandb-project", "wandb-group", "tag" ]

/-- First unrecognized CLI flag key (raw, e.g. `total_timesteps` or `foo.bar.baz`), or `none` if all are
    accepted. Namespaced keys (`a.b`) are accepted for section `env` (passthrough) or any known leaf;
    bare keys are accepted iff their normalized form is a known key. Values are skipped (they don't start
    with `--`), so only actual flag tokens are inspected. -/
def firstUnknownFlag (args : List String) : Option String := Id.run do
  for tok in args do
    if tok.startsWith "--" then
      let body := String.ofList (tok.toList.drop 2)
      let rawKey := (body.splitOn "=").headD body
      let ok :=
        match rawKey.splitOn "." with
        | [_sec, leaf] => _sec == "env" || knownFlagKeys.contains (normKey leaf)
        | _            => knownFlagKeys.contains (normKey rawKey)
      if !ok && !rawKey.isEmpty then return some rawKey
  return none

/-- This run's final checkpoint: the most-recently-WRITTEN `<step>.bin` under `checkpoints/<env>/<seed>/`.
    Checkpoints accumulate across runs sharing a seed (the dir is keyed by seed, not a unique run-id), so
    the highest STEP can belong to an older run — newest-by-mtime is the one THIS run just wrote. GPU runs
    are sequential, so there is no concurrent writer. `none` if the trainer wrote no checkpoint. Used only
    to upload this run's model as a wandb Artifact (`--wandb`). -/
def runFinalCheckpoint (env : String) (seed : UInt64) : IO (Option String) := do
  let dir : System.FilePath := s!"checkpoints/{env}/{seed}"
  if !(← System.FilePath.pathExists dir) then return none
  let mut best : Option (Int × String) := none
  for f in (← dir.readDir) do
    if f.fileName.endsWith ".bin" then
      let mt := (← f.path.metadata).modified
      let key : Int := mt.sec * 1000000000 + Int.ofNat mt.nsec.toNat   -- nanosecond mtime
      match best with
      | some (bk, _) => if key ≥ bk then best := some (key, f.path.toString)
      | none         => best := some (key, f.path.toString)
  return best.map (·.2)

/-- Left-pad a string to width `w` with spaces (for aligned columns). -/
def padL (s : String) (w : Nat) : String :=
  let n := s.length
  if n ≥ w then s else String.ofList (List.replicate (w - n) ' ') ++ s

/-- Quiet PPO train loop (no logging) — the shared core, also used by `sweep`. -/
def usage : String :=
  "puffer — a runnable Lean RL trainer mirroring PufferLib's CLI\n\n" ++
  "USAGE\n" ++
  "  puffer train <env> [flags]     train a PPO+Muon policy on an ocean env plugin (GPU),\n" ++
  "                                 logging batch reward + SPS each update cycle; MinGRU runs\n" ++
  "                                 checkpoint to checkpoints/<env>/<seed>/<step>.bin (see --load,\n" ++
  "                                 --train.checkpoint-interval)\n" ++
  "  puffer eval <env> [flags]      (GPU) load a checkpoint (--load <path>, else the env's latest\n" ++
  "                                 under checkpoints/<env>/) and run it rollout-only, printing the\n" ++
  "                                 mean episode_return + the env's PufferLib `Log`\n" ++
  "  puffer env-log <env> [N] [T]   (CPU only, no GPU) drive an ocean env plugin with a random\n" ++
  "                                 policy and print its own PufferLib `Log` — the episode\n" ++
  "                                 statistics upstream reports — beside the terminal-flag\n" ++
  "                                 reconstruction, so the two units can be compared\n" ++
  "  puffer help                    show this message\n" ++
  "  puffer forward-demo            run the MLP-forward-pass self-check\n\n" ++
  "POLICY  chosen automatically, PufferLib-style, from `[torch] network` and the env's action\n" ++
  "        structure: MLP (feed-forward) · MinGRU/GRU (recurrent, the default; single- AND\n" ++
  "        multi-discrete) · LSTM (recurrent, single-discrete) · Gaussian (continuous-action\n" ++
  "        envs, feed-forward). `--torch.network MLP` forces the feed-forward path anywhere.\n\n" ++
  "ENVS    any ocean env plugin under ocean/<env>/ — build one with `./ocean/build.sh <env>`\n" ++
  "        (46 envs, matching current PufferLib). E.g. squared (default), breakout, snake,\n" ++
  "        cartpole, pong, maze, boxoban, go, chess, squared_continuous (continuous),\n" ++
  "        overcooked, trash_pickup (multi-agent).\n\n" ++
  "CONFIG  layered like PufferLib: config/default.ini ← config/<env>.ini ← CLI flags\n" ++
  "        (--config <path> overrides the base file). INI keys use the same names as flags.\n\n" ++
  "FLAGS   (PufferLib-style; --flag value or --flag=value; namespaces like\n" ++
  "         --train.learning-rate / --policy.hidden-size / --vec.num-envs flatten to the leaf)\n" ++
  "  --train.total-timesteps N     total env steps to train (default 300000)\n" ++
  "  --vec.num-envs N              parallel env instances (default 8)\n" ++
  "  --train.horizon N             rollout / BPTT length (default 64)\n" ++
  "  --torch.network NAME          policy core: MLP | MinGRU | GRU | LSTM (default MLP)\n" ++
  "  --policy.hidden-size N        policy hidden width (default 16)\n" ++
  "  --policy.num-layers N         recurrent depth for MinGRU/GRU (default 4)\n" ++
  "  --train.learning-rate F       PPO learning rate (default 0.03)\n" ++
  "  --train.min-lr-ratio F        LR-schedule floor ÷ lr (default 0.0 = anneal to 0; 1.0 = constant LR)\n" ++
  "  --train.epochs N              PPO passes per rollout (default 4)\n" ++
  "  --train.minibatch-size N      minibatch size, segments·horizon (default 8192)\n" ++
  "  --train.gamma F               discount (default 0.99)\n" ++
  "  --train.gae-lambda F          GAE lambda (default 0.95)\n" ++
  "  --train.vf-coef F             value-loss coef (default 0.5)\n" ++
  "  --train.ent-coef F            entropy bonus coef (default 0.01)\n" ++
  "  --train.anneal-ent-coef B     cosine-decay ent-coef to min-ent-coef-ratio·ent-coef (default 0 = off)\n" ++
  "  --train.min-ent-coef-ratio F  ent-coef schedule floor ÷ ent-coef (default 0.1; only when anneal on)\n" ++
  "  --train.clip-coef F           PPO clip epsilon (default 0.2)\n" ++
  "  --train.checkpoint-interval N  save the policy every N updates (default 0 = final only)\n" ++
  "  --load <path>                 seed initial weights (train) / policy to score (eval) from a checkpoint\n" ++
  "  --train.seed N                RNG seed (default 42)\n" ++
  "  --wandb                       log this run to Weights & Biases live (needs `pip install wandb` +\n" ++
  "                                 `wandb login`; streams via tools/puffer_track.py). PufferLib's --wandb.\n" ++
  "  --wandb-project P             wandb project (default puffer4)   --wandb-group G   wandb group (default debug)\n" ++
  "  --tag T                       single wandb run tag\n" ++
  "  (PufferLib's live training dashboard renders by default for MinGRU envs, as in PufferLib;\n" ++
  "   set PUFFER_PLAIN_LOG=1 for machine-parseable per-update lines instead.)\n\n" ++
  "DEV MODES  GPU/FFI kernel checks against the Lean f64 oracle + benchmarks:\n" ++
  "  verify-cuda | verify-ppo-grad-gpu | verify-mingru-grad-gpu | verify-mingru-md-grad[-gpu] |\n" ++
  "  verify-vtrace-gpu | … ;\n" ++
  "  bench-ffi | bench-train-step | bench-grad | … ; CPU numeric demos verify | verify-grad |\n" ++
  "  verify-adam | verify-vtrace | grad ; verify-trace <path> replays a differential-test trace."

/-- A nontrivial function for autodiff validation: `f = relu(x₀·x₁) + exp(x₂) − x₀·x₂`. -/
def demoF (inputs : Array Float) : ADM V := do
  let xs ← inputs.mapM leaf
  let a ← mul xs[0]! xs[1]!
  let ra ← relu a
  let e ← exp xs[2]!
  let s ← add ra e
  let x0x2 ← mul xs[0]! xs[2]!
  sub s x0x2

def forwardDemo : IO Unit := do
  let x : List Float := [0.5, -1.0, 2.0]
  let w1 : List (List Float) := [[0.1, 0.2, -0.3], [-0.4, 0.5, 0.6]]
  let b1 : List Float := [0.05, -0.1]
  let h := denseRelu x w1 b1
  let y := linearF h [0.7, -0.8] 0.2
  IO.println "puffer (Lean) — MLP forward pass in native Float"
  IO.println s!"  input  = {x}"
  IO.println s!"  hidden = {h}"
  IO.println s!"  output = {y}"

/-- STREAM 3 — headless evaluation of a saved MinGRU policy. Loads a checkpoint's flat weights, drives
    the env for `evalSteps` steps under that policy (batched GPU forward + CPU categorical sampling, NO
    gradient/optimizer/learning), and returns `(mean episode return, #episodes, env's PufferLib Log)`.
    Handles single- AND multi-discrete envs (K categorical heads, decoder width `Wtot = Σ headSizes`);
    the recurrent state carries across steps and resets per env on a terminal (matching the trainers). -/
def evalPluginEnvMinGRU (name config : String) (hidden numLayers numEnvs evalSteps : Nat)
    (wLoaded : FloatArray) (seed : UInt64) : IO (Float × Nat × Array (String × Float)) := do
  let u := USize.ofNat; let mk := FloatArray.mk
  let h ← Puffer.Plugin.envOpen name (u numEnvs) seed config
  if h == 0 then IO.eprintln s!"puffer eval: env '{name}' not found — run ./ocean/build.sh {name}"; return (0.0, 0, #[])
  let D := (Puffer.Plugin.envObsDim h).toNat
  let nAgents := max 1 (Puffer.Plugin.envNumAgents h).toNat
  let K := max 1 (Puffer.Plugin.envNHeads h).toNat
  let sizes := Puffer.Plugin.envHeadSizes h
  let Wtot := sizes.foldl (·+·) 0
  let N := numEnvs * nAgents
  let H := hidden; let L := numLayers
  let O := Wtot + 1
  let P := wLoaded.size
  -- the checkpoint must match this policy's shape, or the flat weight offsets are meaningless
  let expectedP := (Puffer.RL.NNTrain.flattenMG (Puffer.RL.NNTrain.initMinGRU D H L Wtot seed).1).size
  if P != expectedP then
    IO.eprintln s!"puffer eval: checkpoint has {P} params but env '{name}' at hidden={H} layers={L} (obs {D}, Wlogits {Wtot}) needs {expectedP} — retrain or eval with the same --policy.hidden-size/--policy.num-layers"
    Puffer.Plugin.envClose h; return (0.0, 0, #[])
  -- per-head logit offsets (prefix sums) for slicing the joint logit vector
  let mut offs : Array Nat := #[]; let mut oacc := 0
  for hh in [0:K] do offs := offs.push oacc; oacc := oacc + sizes[hh]!
  -- NB: `cudaMinGRUStepFFI` uploads `wLoaded` itself (see verify-mingru-step-gpu) — no resident policy
  -- handle is needed for a forward-only eval, so we skip `policyLoadFFI` entirely.
  let mut obs ← Puffer.Plugin.envReset h                          -- N·D
  let mut stateFlat : FloatArray := mk (Array.replicate (N*L*H) 0.0)
  let mut rng := seed
  let mut run : Array Float := Array.replicate N 0.0              -- per-env running return
  let mut epRetSum := 0.0; let mut nEps : Nat := 0
  let mut lastLog : Array (String × Float) := #[]
  let logWindow := max 1 (evalSteps / 4)                          -- read+zero the env log a few times
  for step in [0:evalSteps] do
    let out := Puffer.Float.CUDA.cudaMinGRUStepFFI wLoaded obs stateFlat (u N) (u D) (u H) (u L) (u Wtot) 1
    let mut acts : Array Float := Array.mkEmpty (N*K)
    for n in [0:N] do
      for hh in [0:K] do
        let base := n*O + offs[hh]!
        let logits := (Array.range sizes[hh]!).map (fun k => out[base+k]!)
        let probs := Puffer.RL.Train.softmax logits
        let (w, rng') := rngNext rng; rng := rng'
        acts := acts.push (Float.ofNat (Puffer.RL.Train.sampleCat probs (uniform01 w)))
    stateFlat := Puffer.Float.FFI.sliceFFI out (u (N*O)) (u (N*L*H))   -- carry the new recurrent state
    let stepOut ← Puffer.Plugin.envStep h (mk acts)                -- [obs(N·D) | rewards(N) | terminals(N)]
    obs := Puffer.Float.FFI.sliceFFI stepOut (u 0) (u (N*D))
    let baseR := N*D
    for n in [0:N] do
      run := run.set! n (run[n]! + stepOut.get! (baseR + n))
      if stepOut.get! (baseR + N + n) > 0.5 then
        epRetSum := epRetSum + run[n]!; nEps := nEps + 1; run := run.set! n 0.0
        for j in [0:L*H] do stateFlat := stateFlat.set! (n*L*H + j) 0.0   -- reset state for the fresh episode
    if (step+1) % logWindow == 0 then
      let lg ← Puffer.Plugin.envLogPairs h
      if !lg.isEmpty then lastLog := lg
  let lg ← Puffer.Plugin.envLogPairs h
  if !lg.isEmpty then lastLog := lg
  Puffer.Plugin.envClose h
  let mean := if nEps == 0 then 0.0 else epRetSum / Float.ofNat nEps
  return (mean, nEps, lastLog)

/-- `puffer verify-trace` — run the verified checker aggregate (C82's
    `runTraceChecks`, via its Mathlib-free duplicate in `Puffer.Check.Core`) on
    a built-in demo trace and budget pairs.  Each PASS/FAIL line names the
    soundness theorems standing behind it; `Puffer.RL.CheckBridge` proves the
    Mathlib-free duplicates equal the verified originals, so the Bool printed
    here carries C74's whole-run interval and C78's budget dominations — while
    the binary still links no Mathlib. -/
def runVerifyTrace : IO Unit := do
  -- Two recorded steps: parameter rows inside the region R = 1.0, PPO ratios
  -- inside the clip margin [0.25, 0.8] (C82's demo trace).
  let tr : Puffer.Check.Trace := [([0.5, -0.25], 0.75), ([0.9, 0.1], 0.5)]
  -- Budget pairs (evaluated Float budget, overflow cap): a 64-term dot budget,
  -- an 8-step run budget, and a horizon-free weight-decay budget — the pairs
  -- C78's *_le_overflow_of_check lemmas discharge.
  let budgets : List (Float × Float) :=
    [ (Puffer.Check.dotBoundF 1.0 64, Puffer.Check.capF)
    , (Puffer.Check.runBoundF 0.5 8 1.0, Puffer.Check.capF)
    , (Puffer.Check.wdUniformBoundF 0.5 0.5 1.0, Puffer.Check.capF) ]
  let ok ← Puffer.Check.verifyTraceIO tr 1.0 0.25 0.8 budgets
  if !ok then IO.println "verify-trace: FAILED"

/-- `puffer verify-trace <file>` — the same verified checker aggregate, on a
    trace parsed from disk (`Puffer.Check.parseTrace`; line format documented
    in `Puffer/Check/Parse.lean`).  Parsing carries no soundness claim — the
    parsed data feeds the SAME bridged `runTraceChecks` Bool as the demo, so a
    PASS means exactly what `allOk_feeds_whole_run` says for that parsed data.
    Exit codes: 0 = parsed and every check passed; 1 = unreadable or malformed
    file (fail-closed) or a failing check. -/
def runVerifyTraceFile (path : String) : IO Unit := do
  let contents ← try IO.FS.readFile path
    catch e => do
      IO.eprintln s!"verify-trace: cannot read '{path}': {e}"
      IO.Process.exit 1
  match Puffer.Check.parseTrace contents with
  | none => do
      IO.eprintln s!"verify-trace: malformed trace file '{path}' (format: Puffer/Check/Parse.lean)"
      IO.Process.exit 1
  | some t => do
      IO.println s!"trace file: {path}  ({t.trace.length} steps, {t.budgets.length} budget pairs)"
      let ok ← Puffer.Check.verifyTraceIO t.trace t.region t.clipLo t.clipHi t.budgets
      if !ok then do
        IO.println "verify-trace: FAILED"
        IO.Process.exit 1

def main (args : List String) : IO Unit := do
  match args with
  | "verify-ffi" :: _ => do
      let mut rng : UInt64 := 0xF00D
      let mut mis : Nat := 0
      for n in [1, 3, 8, 16, 64, 257, 1024] do
        let mut xa : Array Float := #[]
        let mut wa : Array Float := #[]
        for _ in [0:n] do
          let (a, r1) := randF (-2.0) 2.0 rng; rng := r1
          let (b, r2) := randF (-2.0) 2.0 rng; rng := r2
          xa := xa.push a; wa := wa.push b
        let ffi := Puffer.Float.FFI.dotFFI (FloatArray.mk xa) (FloatArray.mk wa)
        let ref := dotF xa.toList wa.toList          -- the VERIFIED oracle (dotF_error)
        if ffi != ref then mis := mis + 1
        let tag := if ffi == ref then "MATCH" else "MISMATCH"
        IO.println s!"  n={n}: dotFFI bits={Float.toBits ffi}  dotF bits={Float.toBits ref}  {tag}"
      IO.println (if mis == 0 then "verify-ffi: ALL bit-exact vs the verified `dotF` oracle"
                  else s!"verify-ffi: {mis} MISMATCHES")
  | "verify-grad-ffi" :: _ => do
      -- validate the native MLP+PPO gradient: FD vs the C primal, and vs the Lean oracle
      let (p, _) := Puffer.RL.NNTrain.initMLP 6 5 4 0x77       -- din=6, H=5, dout=4 (A=3)
      let obs : Array Float := #[0.5, -0.3, 0.2, 0.1, -0.4, 0.6]
      let a := 1
      let adv := 0.7; let ret := 1.1; let oldLogp := -0.9
      let vf := 0.5; let ent := 0.01; let clip := 0.2
      let params := Puffer.RL.NNTrain.flattenMLP p
      let obsFA := FloatArray.mk obs
      let u1 := USize.ofNat
      let g := Puffer.Float.FFI.mlpPPOGradBatchFFI params obsFA (FloatArray.mk #[Float.ofNat a])
                 (FloatArray.mk #[adv]) (FloatArray.mk #[ret]) (FloatArray.mk #[oldLogp])
                 (u1 1) (u1 5) (u1 6) (u1 3) vf ent clip
      let eps := 1.0e-6
      let objAt := fun (pp : FloatArray) =>
        Puffer.Float.FFI.mlpPPOObj1FFI pp obsFA (u1 5) (u1 6) (u1 3) (u1 a) adv ret oldLogp vf ent clip
      for m in [0, 25, 32, 48, 56] do        -- sample W1 / b1 / W2 / b2
        if m < g.size then
          let up := params.set! m (params[m]! + eps)
          let dn := params.set! m (params[m]! - eps)
          let fd := (objAt up - objAt dn) / (2.0 * eps)
          IO.println s!"  param[{m}]: C-grad={g[m]!}   FD={fd}"
      let (gW1, gb1, gW2, gb2) := Puffer.RL.NNTrain.mlpGradPPO p obs a adv ret oldLogp vf ent clip
      let leanFlat := Puffer.RL.NNTrain.flattenMLP { W1 := gW1, b1 := gb1, W2 := gW2, b2 := gb2 }
      let mut maxd := 0.0
      for i in [0:g.size] do maxd := max maxd (Float.abs (g[i]! - leanFlat[i]!))
      IO.println s!"verify-grad-ffi: max |C-grad − Lean mlpGradPPO grad| = {maxd}   (P={g.size})"
  | "verify-fwd-ffi" :: _ => do
      -- native MLP forward bit-exact vs the Lean forwardAll oracle
      let mut rng : UInt64 := 0xFEED
      let mut mis : Nat := 0
      for dims in [(6,5,4), (25,24,6), (118,64,4)] do
        let (din, H, O) := dims
        let (p, r') := Puffer.RL.NNTrain.initMLP din H O rng; rng := r'
        let mut obs : Array Float := #[]
        for _ in [0:din] do let (a, r) := randF (-1.0) 1.0 rng; rng := r; obs := obs.push a
        let ffiOut := Puffer.Float.FFI.mlpForwardFFI (Puffer.RL.NNTrain.flattenMLP p) (FloatArray.mk obs)
                        (USize.ofNat H) (USize.ofNat din) (USize.ofNat O)
        let (_, _, leanOut) := Puffer.RL.NNTrain.forwardAll p obs
        let mut d := 0
        for k in [0:O] do if ffiOut[k]! != leanOut[k]! then d := d + 1
        if d != 0 then mis := mis + 1
        let tag := if d == 0 then "MATCH" else s!"{d} mismatch"
        IO.println s!"  {din}→{H}→{O}: {tag}"
      IO.println (if mis == 0 then "verify-fwd-ffi: forward bit-exact vs forwardAll" else "verify-fwd-ffi: MISMATCH")
  | "verify-gauss-ffi" :: _ => do
      -- native Gaussian gradient: FD vs C primal, and vs the Lean mlpGradPPOCont oracle
      let A := 2
      let (p, _) := Puffer.RL.NNTrain.initMLP 6 5 (2*A+1) 0x99   -- din 6, H 5, dout 2A+1=5
      let obs : Array Float := #[0.3, -0.5, 0.2, 0.8, -0.1, 0.4]
      let act : Array Float := #[0.6, -0.4]
      let adv := 0.7; let ret := 1.1; let oldLogp := -1.3
      let vf := 0.5; let ent := 0.01; let clip := 0.2
      let params := Puffer.RL.NNTrain.flattenMLP p
      let obsFA := FloatArray.mk obs
      let actFA := FloatArray.mk act
      let u1 := USize.ofNat
      let g := Puffer.Float.FFI.gaussPPOGradBatchFFI params obsFA actFA
                 (FloatArray.mk #[adv]) (FloatArray.mk #[ret]) (FloatArray.mk #[oldLogp])
                 (u1 1) (u1 5) (u1 6) (u1 A) vf ent clip
      let eps := 1.0e-6
      let objAt := fun (pp : FloatArray) =>
        Puffer.Float.FFI.gaussPPOObj1FFI pp obsFA actFA (u1 5) (u1 6) (u1 A) adv ret oldLogp vf ent clip
      for m in [0, 20, 30, 40, 54] do
        if m < g.size then
          let up := params.set! m (params[m]! + eps)
          let dn := params.set! m (params[m]! - eps)
          let fd := (objAt up - objAt dn) / (2.0 * eps)
          IO.println s!"  param[{m}]: C-grad={g[m]!}   FD={fd}"
      let (gW1, gb1, gW2, gb2) := Puffer.RL.NNTrain.mlpGradPPOCont p obs act A adv ret oldLogp vf ent clip
      let leanFlat := Puffer.RL.NNTrain.flattenMLP { W1 := gW1, b1 := gb1, W2 := gW2, b2 := gb2 }
      let mut maxd := 0.0
      for i in [0:g.size] do maxd := max maxd (Float.abs (g[i]! - leanFlat[i]!))
      IO.println s!"verify-gauss-ffi: max |C-grad − Lean mlpGradPPOCont grad| = {maxd}   (P={g.size})"
  | "verify-cnn-ffi" :: _ => do
      -- native CNN (conv+dense) gradient: FD vs C primal, and vs the Lean cnnGradPPO oracle
      let C := 2; let inH := 5; let inW := 5; let nF := 3; let k := 2; let s := 1; let hidden := 8
      let A := 3
      let (p, _) := Puffer.RL.NNTrain.initCnn C inH inW nF k s hidden (A+1) 0xCA5
      let mut rng : UInt64 := 0x1234
      let mut obs : Array Float := #[]
      for _ in [0:C*inH*inW] do let (x, r) := randF (-1.0) 1.0 rng; rng := r; obs := obs.push x
      let a := 2
      let adv := 0.7; let ret := 1.1; let oldLogp := -1.2
      let vf := 0.5; let ent := 0.01; let clip := 0.2
      let params := Puffer.RL.NNTrain.flattenCnn p
      let obsFA := FloatArray.mk obs
      let u1 := USize.ofNat
      let g := Puffer.Float.FFI.cnnPPOGradBatchFFI params obsFA (FloatArray.mk #[Float.ofNat a])
                 (FloatArray.mk #[adv]) (FloatArray.mk #[ret]) (FloatArray.mk #[oldLogp])
                 (u1 1) (u1 C) (u1 inH) (u1 inW) (u1 nF) (u1 k) (u1 s) (u1 hidden) (u1 A) vf ent clip
      let eps := 1.0e-6
      let objAt := fun (pp : FloatArray) =>
        Puffer.Float.FFI.cnnPPOObj1FFI pp obsFA (u1 C) (u1 inH) (u1 inW) (u1 nF) (u1 k) (u1 s)
          (u1 hidden) (u1 A) (u1 a) adv ret oldLogp vf ent clip
      for m in [0, 25, 100, 415, 425, 453] do        -- sample convW / convB / W1 / b1 / W2 / b2
        if m < g.size then
          let up := params.set! m (params[m]! + eps)
          let dn := params.set! m (params[m]! - eps)
          let fd := (objAt up - objAt dn) / (2.0 * eps)
          IO.println s!"  param[{m}]: C-grad={g[m]!}   FD={fd}"
      let gr := Puffer.RL.NNTrain.cnnGradPPO p obs a A adv ret oldLogp vf ent clip
      let leanFlat := Puffer.RL.NNTrain.flattenCnn
        { p with convW := gr.gConvW, convB := gr.gConvB, W1 := gr.gW1, b1 := gr.gb1, W2 := gr.gW2, b2 := gr.gb2 }
      let mut maxd := 0.0
      for i in [0:g.size] do maxd := max maxd (Float.abs (g[i]! - leanFlat[i]!))
      IO.println s!"verify-cnn-ffi: max |C-grad − Lean cnnGradPPO grad| = {maxd}   (P={g.size})"
  | "verify-lstm-ffi" :: _ => do
      -- native LSTM truncated-BPTT gradient: FD vs C primal, and vs the Lean recPPOGradSeq oracle
      let D := 4; let H := 3; let A := 2
      let (p, _) := Puffer.RL.NNTrain.initRec D H (A+1) 0x5EED
      -- a length-5 sequence with a terminal at t=2 (exercises the reset + truncation path)
      let mut rng : UInt64 := 0xABCD
      let mut traj : Array Puffer.RL.NNTrain.Transition := #[]
      for t in [0:5] do
        let mut obs : Array Float := #[]
        for _ in [0:D] do let (x, r) := randF (-1.0) 1.0 rng; rng := r; obs := obs.push x
        let (aw, r2) := rngNext rng; rng := r2
        let act := (aw.toNat % A)
        traj := traj.push { obs := obs, action := act, reward := 0.0, value := 0.0,
                            oldLogp := -0.8, terminal := (t == 2) }
      let h0 : Array Float := #[0.1, -0.2, 0.05]
      let c0 : Array Float := #[0.0, 0.1, -0.1]
      let advN : Array Float := #[0.7, -0.3, 0.5, 0.2, -0.6]
      let returns : Array Float := #[1.1, 0.4, 0.9, 1.3, 0.2]
      let vf := 0.5; let ent := 0.01; let clip := 0.2
      let params := Puffer.RL.NNTrain.flattenRec p
      let (obsSeq, acts, olps, terms) := Puffer.RL.NNTrain.mkSeqArrays traj
      let u1 := USize.ofNat
      let g := Puffer.Float.FFI.lstmPPOGradSeqFFI params obsSeq acts (FloatArray.mk advN)
                 (FloatArray.mk returns) olps terms (FloatArray.mk h0) (FloatArray.mk c0)
                 (u1 5) (u1 H) (u1 D) (u1 A) vf ent clip
      let eps := 1.0e-6
      let objAt := fun (pp : FloatArray) =>
        Puffer.Float.FFI.lstmPPOObjSeqFFI pp obsSeq acts (FloatArray.mk advN) (FloatArray.mk returns)
          olps terms (FloatArray.mk h0) (FloatArray.mk c0) (u1 5) (u1 H) (u1 D) (u1 A) vf ent clip
      for m in [0, 50, 84, 96, 105] do        -- Wx / Wh / bih / Wo / bo
        if m < g.size then
          let up := params.set! m (params[m]! + eps)
          let dn := params.set! m (params[m]! - eps)
          let fd := (objAt up - objAt dn) / (2.0 * eps)
          IO.println s!"  param[{m}]: C-grad={g[m]!}   FD={fd}"
      let gr := Puffer.RL.NNTrain.recPPOGradSeq p traj h0 c0 advN returns A vf ent clip
      let leanFlat := Puffer.RL.NNTrain.flattenRec
        { p with Wx := gr.gWx, Wh := gr.gWh, bih := gr.gbih, Wo := gr.gWo, bo := gr.gbo }
      let mut maxd := 0.0
      for i in [0:g.size] do maxd := max maxd (Float.abs (g[i]! - leanFlat[i]!))
      IO.println s!"verify-lstm-ffi: max |C-grad − Lean recPPOGradSeq grad| = {maxd}   (P={g.size})"
  | "bench-ffi" :: _ => do
      let n := 4096
      let reps := 100000
      let mut rng : UInt64 := 0xBEEF
      let mut xa : Array Float := #[]
      let mut wa : Array Float := #[]
      for _ in [0:n] do
        let (a, r1) := randF (-1.0) 1.0 rng; rng := r1
        let (b, r2) := randF (-1.0) 1.0 rng; rng := r2
        xa := xa.push a; wa := wa.push b
      let x := FloatArray.mk xa
      let w := FloatArray.mk wa
      let t0 ← IO.monoNanosNow
      let mut s := 0.0
      for _ in [0:reps] do s := s + Puffer.Float.FFI.dotFFI x w
      let t1 ← IO.monoNanosNow
      let mut s2 := 0.0
      for _ in [0:reps] do s2 := s2 + Puffer.Float.FFI.dotRef x w
      let t2 ← IO.monoNanosNow
      let ffiNs := t1 - t0
      let leanNs := t2 - t1
      IO.println s!"bench-ffi dot (n={n}, reps={reps}):  native FFI {ffiNs / 1000000}ms   pure Lean {leanNs / 1000000}ms   (checksums {s} / {s2})"
      IO.println s!"  speedup ≈ {Float.ofNat leanNs / Float.ofNat (max ffiNs 1)}×  (native kernel is bit-identical to the verified dotF)"
  | "verify-cuda" :: _ => do
      -- M0: the nvcc build-integration self-test — a __global__ kernel ran on the GPU iff Y[i]=2i+1.
      let n := 8
      let y := Puffer.Float.CUDA.cudaSelftestFFI (USize.ofNat n)
      let mut ok := true
      for i in [0:n] do
        if y[i]! != 2.0 * Float.ofNat i + 1.0 then ok := false
      IO.println s!"verify-cuda (M0 nvcc build MVP): Y = {(List.range n).map (fun i => y[i]!)}"
      IO.println s!"  {if ok then "ok  — nvcc __global__ kernel compiled, linked, and ran on the GPU" else "FAIL — kernel output wrong"}"
  | "verify-vtrace-gpu" :: _ => do
      -- M1: GPU V-Trace kernel vs the Lean oracle computePuffAdvantageV, per segment. f64 + --fmad=false
      -- ⇒ expect BIT-EXACT (max|Δ| = 0). Non-unit clips + rewards outside [-1,1] exercise min + the clamp.
      let B := 8; let T := 12
      let gamma := 0.99; let lam := 0.95; let rhoClip := 2.1; let cClip := 1.08
      let mut rng : UInt64 := 0x5C1A
      let mut rewards : Array Float := #[]; let mut values : Array Float := #[]
      let mut terms : Array Float := #[]; let mut imps : Array Float := #[]
      for _ in [0:B*T] do
        let (rr, r1) := randF (-1.5) 1.5 rng; rng := r1; rewards := rewards.push rr
        let (vv, r2) := randF (-2.0) 2.0 rng; rng := r2; values := values.push vv
        let (tu, r3) := randF 0.0 1.0 rng; rng := r3; terms := terms.push (if tu < 0.12 then 1.0 else 0.0)
        let (iv, r4) := randF 0.1 2.0 rng; rng := r4; imps := imps.push iv
      let u := USize.ofNat
      let advGpu := Puffer.Float.CUDA.cudaVtraceFFI (FloatArray.mk rewards) (FloatArray.mk values)
        (FloatArray.mk terms) (FloatArray.mk imps) (u B) (u T) gamma lam rhoClip cClip
      let mut maxd := 0.0
      for b in [0:B] do
        let traj : Array Puffer.RL.NNTrain.Transition := (Array.range T).map (fun t =>
          { obs := #[], action := 0, reward := rewards[b*T+t]!, value := 0.0, oldLogp := 0.0,
            terminal := (terms[b*T+t]! != 0.0) })
        let valuesRow := (Array.range T).map (fun t => values[b*T+t]!)
        let impsRow := (Array.range T).map (fun t => imps[b*T+t]!)
        let advCpu := Puffer.RL.NNTrain.computePuffAdvantageV traj valuesRow impsRow gamma lam rhoClip cClip
        for t in [0:T] do maxd := max maxd (Float.abs (advGpu[b*T+t]! - advCpu[t]!))
      IO.println s!"verify-vtrace-gpu (M1: GPU V-Trace vs computePuffAdvantageV, B={B} T={T}, non-unit clips):"
      IO.println s!"  max|Δ| = {maxd}   ({if maxd == 0.0 then "bit-exact ✓ (f64 scan, --fmad=false)" else "TOLERANCE"})"
  | "verify-vtrace-mingru-gpu" :: _ => do
      -- M1b: GPU MinGRU V-Trace kernel vs the oracle vtraceMinGRUFlat (a line-for-line copy of
      -- trainPluginEnvMinGRU's old per-segment closure): scalar delta ρ·r + γV′·nnt − V with the LAST
      -- step bootstrapped by bootv, scan t=T-1..0. f64 + --fmad=false ⇒ expect BIT-EXACT (max|Δ| = 0).
      -- Non-unit clips + rewards outside [-1,1] exercise the min + reward clamp.
      let B := 8; let T := 12
      let gamma := 0.99; let lam := 0.95; let rhoClip := 2.1; let cClip := 1.08
      let mut rng : UInt64 := 0x5C1B
      let mut rewards : Array Float := #[]; let mut values : Array Float := #[]
      let mut terms : Array Float := #[]; let mut imps : Array Float := #[]
      for _ in [0:B*T] do
        let (rr, r1) := randF (-1.5) 1.5 rng; rng := r1; rewards := rewards.push rr
        let (vv, r2) := randF (-2.0) 2.0 rng; rng := r2; values := values.push vv
        let (tu, r3) := randF 0.0 1.0 rng; rng := r3; terms := terms.push (if tu < 0.12 then 1.0 else 0.0)
        let (iv, r4) := randF 0.1 2.0 rng; rng := r4; imps := imps.push iv
      let mut bootv : Array Float := #[]
      for _ in [0:B] do
        let (bv, r5) := randF (-2.0) 2.0 rng; rng := r5; bootv := bootv.push bv
      let u := USize.ofNat
      let advGpu := Puffer.Float.CUDA.cudaVtraceMinGRUFFI (FloatArray.mk rewards) (FloatArray.mk values)
        (FloatArray.mk terms) (FloatArray.mk imps) (FloatArray.mk bootv) (u B) (u T) gamma lam rhoClip cClip
      let advRef := Puffer.RL.NNTrain.vtraceMinGRUFlat (FloatArray.mk rewards) (FloatArray.mk values)
        (FloatArray.mk terms) (FloatArray.mk imps) (FloatArray.mk bootv) B T gamma lam rhoClip cClip
      let mut maxd := 0.0
      for k in [0:B*T] do maxd := max maxd (Float.abs (advGpu[k]! - advRef[k]!))
      IO.println s!"verify-vtrace-mingru-gpu (M1b: GPU MinGRU V-Trace vs vtraceMinGRUFlat, B={B} T={T}, bootstrap + non-unit clips):"
      IO.println s!"  max|Δ| = {maxd}   ({if maxd == 0.0 then "bit-exact ✓ (f64 scan, --fmad=false)" else "TOLERANCE"})"
  | "verify-muon-gpu" :: _ => do
      -- M2: GPU Muon stepMat vs the Lean oracle Puffer.Float.Muon.stepMat. Naive f64 NS matmuls in
      -- Lean's sum order + --fmad=false ⇒ expect BIT-EXACT. Shapes exercise both NS branches.
      let lr := 0.1; let wd := 0.0; let mu := 0.95; let eps := 1e-7
      let u := USize.ofNat
      let mut maxd := 0.0
      let mut rng : UInt64 := 0x3E7A
      for shape in [(16, 24), (24, 16), (20, 20)] do
        let (rows, cols) := shape
        let n := rows * cols
        let mut fW : Array Float := #[]; let mut fG : Array Float := #[]; let mut fM : Array Float := #[]
        for _ in [0:n] do
          let (w, r1) := randF (-0.5) 0.5 rng; rng := r1; fW := fW.push w
          let (g, r2) := randF (-0.3) 0.3 rng; rng := r2; fG := fG.push g
          let (m, r3) := randF (-0.2) 0.2 rng; rng := r3; fM := fM.push m
        let out := Puffer.Float.CUDA.cudaMuonStepMatFFI (FloatArray.mk fW) (FloatArray.mk fG)
          (FloatArray.mk fM) (u rows) (u cols) lr wd mu eps
        let toMat := fun (fl : Array Float) =>
          (Array.range rows).map (fun i => (Array.range cols).map (fun j => fl[i*cols+j]!))
        let (nW, nM) := Puffer.FloatR.Muon.stepMat (toMat fW) (toMat fG) (toMat fM) lr wd mu eps
        for i in [0:rows] do
          for j in [0:cols] do
            maxd := max maxd (Float.abs (out[i*cols+j]! - (nW[i]!)[j]!))
            maxd := max maxd (Float.abs (out[n + i*cols+j]! - (nM[i]!)[j]!))
      IO.println s!"verify-muon-gpu (M2: GPU Muon stepMat vs Puffer.Float.Muon.stepMat; rows≤cols / rows>cols / square):"
      IO.println s!"  max|Δ| = {maxd}   ({if maxd == 0.0 then "bit-exact ✓ (f64 NS naive matmul, --fmad=false)" else "TOLERANCE"})"
  | "verify-ppo-grad-gpu" :: _ => do
      -- M3: GPU MLP PPO gradient vs the CPU-BLAS f64 oracle (mlpPPOGradBatchBlasFFI). Two tiers:
      -- f32 (tight, ~1e-5, confirms the GEMM layouts + ppo_dout logic) and bf16 (PufferLib default, ~1e-2).
      let N := 64; let D := 12; let H := 16; let A := 5; let O := A + 1
      let vf := 0.5; let ent := 0.01; let clip := 0.2
      let (p, rng0) := Puffer.RL.NNTrain.initMLP D H O 0x9A73
      let params := Puffer.RL.NNTrain.flattenMLP p
      let mut rng := rng0
      let mut obs : Array Float := #[]; let mut acts : Array Float := #[]
      let mut advs : Array Float := #[]; let mut rets : Array Float := #[]; let mut olps : Array Float := #[]
      for _ in [0:N] do
        for _ in [0:D] do let (v, r) := randF (-1.0) 1.0 rng; rng := r; obs := obs.push v
        let (au, r1) := randF 0.0 (Float.ofNat A) rng; rng := r1; acts := acts.push (Float.floor au)
        let (av, r2) := randF (-1.0) 1.0 rng; rng := r2; advs := advs.push av
        let (rt, r3) := randF (-1.0) 1.0 rng; rng := r3; rets := rets.push rt
        let (ol, r4) := randF (-1.5) 0.0 rng; rng := r4; olps := olps.push ol
      let u := USize.ofNat; let mk := FloatArray.mk
      let oracle := Puffer.Float.BLAS.mlpPPOGradBatchBlasFFI params (mk obs) (mk acts) (mk advs) (mk rets) (mk olps) (u N) (u H) (u D) (u A) vf ent clip
      let gF32 := Puffer.Float.CUDA.cudaMlpPpoGradFFI params (mk obs) (mk acts) (mk advs) (mk rets) (mk olps) (u N) (u H) (u D) (u A) vf ent clip 0
      let gBf16 := Puffer.Float.CUDA.cudaMlpPpoGradFFI params (mk obs) (mk acts) (mk advs) (mk rets) (mk olps) (u N) (u H) (u D) (u A) vf ent clip 1
      let P := H*D + H + O*H + O
      let mut dF := 0.0; let mut dB := 0.0
      for i in [0:P] do
        dF := max dF (Float.abs (gF32[i]! - oracle[i]!))
        dB := max dB (Float.abs (gBf16[i]! - oracle[i]!))
      IO.println s!"verify-ppo-grad-gpu (M3: GPU MLP PPO gradient vs CPU-BLAS oracle, N={N} H={H} D={D} A={A}, P={P}):"
      IO.println s!"  f32  tight (COMPUTE_32F)      : max|Δ| = {dF}   ({if dF < 1.0e-4 then "ok ✓ (cuBLAS f32 vs f64 oracle — layouts+ppo_dout correct)" else "CHECK"})"
      IO.println s!"  bf16 default (FAST_16BF, TC)  : max|Δ| = {dB}   ({if dB < 0.2 then "ok ✓ (PufferLib bf16 tensor-core precision)" else "CHECK"})"
  | "verify-advnorm-gpu" :: _ => do
      -- M4: GPU advantage normalize vs the Lean oracle normalizeAdv. f64 sequential folds ⇒ bit-exact.
      let n := 257
      let mut rng : UInt64 := 0xAD7
      let mut adv : Array Float := #[]
      for _ in [0:n] do let (v, r) := randF (-3.0) 3.0 rng; rng := r; adv := adv.push v
      let advN := Puffer.Float.CUDA.cudaAdvNormalizeFFI (FloatArray.mk adv) (USize.ofNat n)
      let oracle := Puffer.RL.NNTrain.normalizeAdv adv
      let mut maxd := 0.0
      for i in [0:n] do maxd := max maxd (Float.abs (advN[i]! - oracle[i]!))
      IO.println s!"verify-advnorm-gpu (M4: GPU advantage normalize vs normalizeAdv, n={n}):"
      IO.println s!"  max|Δ| = {maxd}   ({if maxd == 0.0 then "bit-exact ✓ (f64 folds, --fmad=false)" else "TOLERANCE"})"
  | "verify-train-step-gpu" :: _ => do
      -- M5: one full PPO+Muon step ON THE GPU (normalize→gradient→Muon, resident) vs the CPU/Lean
      -- SHADOW oracle: normalizeAdv → mlpPPOGradBatchBlas → applyMuon. Tiers: f32 tight, bf16 default.
      let N := 96; let D := 12; let H := 16; let A := 5; let O := A + 1
      let vf := 0.5; let ent := 0.01; let clip := 0.2
      let lr := 0.1; let wd := 0.0; let mu := 0.95; let eps := 1.0e-7
      let (p, rng0) := Puffer.RL.NNTrain.initMLP D H O 0x77E5
      let params := Puffer.RL.NNTrain.flattenMLP p
      let P := H*D + H + O*H + O
      let mut rng := rng0
      let mut obs : Array Float := #[]; let mut acts : Array Float := #[]
      let mut adv : Array Float := #[]; let mut rets : Array Float := #[]; let mut olps : Array Float := #[]
      let mut mom : Array Float := #[]
      for _ in [0:N] do
        for _ in [0:D] do let (v, r) := randF (-1.0) 1.0 rng; rng := r; obs := obs.push v
        let (au, r1) := randF 0.0 (Float.ofNat A) rng; rng := r1; acts := acts.push (Float.floor au)
        let (av, r2) := randF (-2.0) 2.0 rng; rng := r2; adv := adv.push av
        let (rt, r3) := randF (-1.0) 1.0 rng; rng := r3; rets := rets.push rt
        let (ol, r4) := randF (-1.5) 0.0 rng; rng := r4; olps := olps.push ol
      for _ in [0:P] do let (m, r) := randF (-0.1) 0.1 rng; rng := r; mom := mom.push m
      let u := USize.ofNat; let mk := FloatArray.mk
      -- CPU/Lean shadow oracle
      let oW1 := 0; let ob1 := H*D; let oW2 := H*D+H; let ob2 := H*D+H+O*H
      let toMat := fun (fl : Array Float) (rows cols off : Nat) =>
        (Array.range rows).map (fun i => (Array.range cols).map (fun j => fl[off + i*cols+j]!))
      let toVec := fun (fl : Array Float) (len off : Nat) => (Array.range len).map (fun i => fl[off+i]!)
      let flatMat := fun (m : Array (Array Float)) => m.foldl (fun acc row => acc ++ row) #[]
      let st : Puffer.RL.NNTrain.MuonState :=
        { mW1 := toMat mom H D oW1, mb1 := toVec mom H ob1, mW2 := toMat mom O H oW2, mb2 := toVec mom O ob2 }
      let advN := Puffer.RL.NNTrain.normalizeAdv adv
      let gFA := Puffer.Float.BLAS.mlpPPOGradBatchBlasFFI params (mk obs) (mk acts) (mk advN) (mk rets) (mk olps) (u N) (u H) (u D) (u A) vf ent clip
      -- mean gradient (÷N): the kernel/oracle SUM over the minibatch; the GPU train step averages before Muon
      let g : Array Float := (Array.range P).map (fun i => gFA[i]! / Float.ofNat N)
      let gTup := (toMat g H D oW1, toVec g H ob1, toMat g O H oW2, toVec g O ob2)
      let (newP, newSt) := Puffer.RL.NNTrain.applyMuon p st gTup lr wd mu eps
      let newParams := Puffer.RL.NNTrain.flattenMLP newP
      let newMom := (flatMat newSt.mW1) ++ newSt.mb1 ++ (flatMat newSt.mW2) ++ newSt.mb2
      let pm := mk ((Array.range P).map (fun i => params[i]!) ++ mom)   -- combined [params; mom] buffer
      let runGpu := fun (bf : UInt8) =>
        Puffer.Float.CUDA.cudaTrainStepFFI pm (mk obs) (mk acts) (mk adv) (mk rets) (mk olps)
          (u N) (u H) (u D) (u A) lr wd mu eps vf ent clip bf
      let chk := fun (gpu : FloatArray) => Id.run do
        let mut dp := 0.0; let mut dm := 0.0
        for i in [0:P] do
          dp := max dp (Float.abs (gpu[i]! - newParams[i]!))
          dm := max dm (Float.abs (gpu[P+i]! - newMom[i]!))
        return (dp, dm)
      let (dpF, dmF) := chk (runGpu 0)
      let (dpB, dmB) := chk (runGpu 1)
      IO.println s!"verify-train-step-gpu (M5: resident GPU PPO+Muon step vs CPU/Lean shadow, N={N} P={P}):"
      IO.println s!"  f32  : newParams max|Δ| = {dpF}   newMom max|Δ| = {dmF}   ({if dpF < 1.0e-3 && dmF < 1.0e-3 then "ok ✓ (whole step correct)" else "CHECK"})"
      IO.println s!"  bf16 : newParams max|Δ| = {dpB}   newMom max|Δ| = {dmB}   ({if dpB < 0.2 && dmB < 0.2 then "ok ✓ (PufferLib precision)" else "CHECK"})"
  | "verify-train-update-resident" :: _ => do
      -- Whole-update RESIDENT step (`cudaTrainUpdateFFI`, all epochs×minibatches on-device) vs looping
      -- `cudaTrainStepFFI` per minibatch over the SAME perms. Same math ⇒ f32 expected BIT-EXACT.
      let NT := 48; let D := 12; let H := 16; let A := 5; let O := A + 1; let P := H*D + H + O*H + O
      let epochs := 3; let numMB := 4
      let vf := 0.5; let ent := 0.01; let clip := 0.2
      let lr := 0.05; let wd := 0.0; let mu := 0.95; let eps := 1.0e-7
      let mut rng : UInt64 := 0x9A7C33
      let mut pm0 : Array Float := #[]
      for _ in [0:2*P] do let (v, r) := randF (-0.3) 0.3 rng; rng := r; pm0 := pm0.push v
      let mut obs : Array Float := #[]
      for _ in [0:NT*D] do let (v, r) := randF (-1.0) 1.0 rng; rng := r; obs := obs.push v
      let mut acts : Array Float := #[]; let mut adv : Array Float := #[]; let mut rets : Array Float := #[]; let mut olps : Array Float := #[]
      for _ in [0:NT] do
        let (au, r1) := randF 0.0 (Float.ofNat A) rng; rng := r1; acts := acts.push (Float.floor au)
        let (av, r2) := randF (-2.0) 2.0 rng; rng := r2; adv := adv.push av
        let (rt, r3) := randF (-1.0) 1.0 rng; rng := r3; rets := rets.push rt
        let (ol, r4) := randF (-1.5) 0.0 rng; rng := r4; olps := olps.push ol
      let u := USize.ofNat; let mk := FloatArray.mk
      let mut perms : Array (Array Nat) := #[]; let mut permFlat : Array Float := #[]; let mut prng : UInt64 := 0x1234
      for _ in [0:epochs] do
        let (perm, prng') := Puffer.RL.NNTrain.shuffleIdx NT prng; prng := prng'
        perms := perms.push perm
        for x in perm do permFlat := permFlat.push (Float.ofNat x)
      let bf : UInt8 := 0
      let residentPm := Puffer.Float.CUDA.cudaTrainUpdateFFI (mk pm0) (mk obs) (mk acts) (mk adv) (mk rets) (mk olps) (mk permFlat)
        (u NT) (u D) (u H) (u A) (u epochs) (u numMB) lr wd mu eps vf ent clip bf
      let mbSize := NT / numMB
      let mut pm := mk pm0
      for e in [0:epochs] do
        let perm := perms[e]!
        for m in [0:numMB] do
          let lo := m*mbSize; let hi := if m+1==numMB then NT else (m+1)*mbSize
          let idxs := (Array.range (hi-lo)).map (fun k => perm[lo+k]!)
          let Nmb := idxs.size
          let mut mbObs : Array Float := #[]; let mut mbAct : Array Float := #[]; let mut mbAdv : Array Float := #[]; let mut mbRet : Array Float := #[]; let mut mbOlp : Array Float := #[]
          for t in idxs do
            for j in [0:D] do mbObs := mbObs.push obs[t*D+j]!
            mbAct := mbAct.push acts[t]!; mbAdv := mbAdv.push adv[t]!; mbRet := mbRet.push rets[t]!; mbOlp := mbOlp.push olps[t]!
          pm := Puffer.Float.CUDA.cudaTrainStepFFI pm (mk mbObs) (mk mbAct) (mk mbAdv) (mk mbRet) (mk mbOlp) (u Nmb) (u H) (u D) (u A) lr wd mu eps vf ent clip bf
      let mut dmax := 0.0
      for i in [0:2*P] do dmax := max dmax (Float.abs (residentPm[i]! - pm[i]!))
      IO.println s!"verify-train-update-resident (resident whole-update vs per-minibatch loop, NT={NT} epochs={epochs} numMB={numMB} P={P}):"
      IO.println s!"  pm max|Δ| = {dmax}   ({if dmax == 0.0 then "ok ✓ (bit-exact)" else if dmax < 1.0e-9 then "ok ✓ (tolerance)" else "CHECK"})"
  | "verify-sample-gpu" :: _ => do
      -- R2 (docs/gpu-rollout-scope.md): device categorical sampler (`cudaSampleActionsFFI`) vs the CPU C
      -- sampler (`sampleActionsBatchFFI`, itself bit-exact vs softmax+sampleCat). Same per-env splitmix64
      -- stream ⇒ actions/values exact; logps to transcendental ULP (device exp/log vs libm).
      let N := 256; let A := 6; let O := A + 1
      let u := USize.ofNat
      let mut rng : UInt64 := 0xD1CEF00D
      let mut yb : Array Float := #[]
      for _ in [0:N*O] do let (v, r) := randF (-4.0) 4.0 rng; rng := r; yb := yb.push v
      let Yb := FloatArray.mk yb
      let startRng : UInt64 := 0x9E3779B1
      let gpu := Puffer.Float.CUDA.cudaSampleActionsFFI Yb (u N) (u A) (u O) startRng
      let cpu := Puffer.Float.FFI.sampleActionsBatchFFI Yb (u N) (u A) (u O) startRng
      let mut amiss := 0; let mut dlp := 0.0; let mut dv := 0.0
      for i in [0:N] do
        if gpu[i]! != cpu[i]! then amiss := amiss + 1
        dlp := max dlp (Float.abs (gpu[N+i]! - cpu[N+i]!))
        dv  := max dv  (Float.abs (gpu[2*N+i]! - cpu[2*N+i]!))
      let ok := amiss == 0 && dlp < 1.0e-12 && dv == 0.0
      IO.println s!"verify-sample-gpu (device sampler vs CPU sampleActionsBatchFFI, N={N} A={A}):"
      IO.println s!"  action mismatches = {amiss}/{N}   logp max|Δ| = {dlp}   value max|Δ| = {dv}   ({if amiss == 0 && dlp == 0.0 && dv == 0.0 then "ok ✓ (bit-exact)" else if ok then "ok ✓ (ULP: device exp/log vs libm)" else "CHECK"})"
  | "verify-cnn-forward-gpu" :: _ => do
      -- R6: device CNN encoder forward (`cudaCnnForwardFFI`: im2col → conv → relu → transpose → dense)
      -- vs the CPU `cnnForward` (f64). f32 GPU ⇒ tolerance, not bit-exact.
      let C := 3; let inH := 11; let inW := 11; let nF := 8; let k := 3; let s := 2; let hidden := 32; let A := 3; let O := A + 1
      let (cnn, _) := Puffer.RL.NNTrain.initCnn C inH inW nF k s hidden O 0x5EED
      let params := Puffer.RL.NNTrain.flattenCnn cnn
      let N := 8; let inSz := C*inH*inW
      let u := USize.ofNat; let mk := FloatArray.mk
      let mut rng : UInt64 := 0xC0FFEE99
      let mut obsFlat : Array Float := #[]
      for _ in [0:N*inSz] do let (v, r) := randF (-1.0) 1.0 rng; rng := r; obsFlat := obsFlat.push v
      let gpu := Puffer.Float.CUDA.cudaCnnForwardFFI params (mk obsFlat) (u N) (u C) (u inH) (u inW) (u nF) (u k) (u s) (u hidden) (u O) 0
      let mut dmax := 0.0
      for n in [0:N] do
        let obs := (Array.range inSz).map (fun i => obsFlat[n*inSz+i]!)
        let logits := Puffer.RL.NNTrain.cnnForward cnn obs
        for kk in [0:O] do dmax := max dmax (Float.abs (gpu[n*O+kk]! - logits[kk]!))
      IO.println s!"verify-cnn-forward-gpu (device CNN encoder vs CPU cnnForward, N={N} C{C} {inH}×{inW} nF{nF} k{k}/s{s} → {hidden} → {O}):"
      IO.println s!"  logits max|Δ| = {dmax}   ({if dmax < 1.0e-4 then "ok ✓ (f32 GPU vs f64 CPU tolerance)" else "CHECK"})"
  | "verify-muon-cpu" :: _ => do
      -- Native-C whole-MLP Muon step (`muonStepMlpBlasFFI`) vs the pure-Lean oracle
      -- (`applyMuon` on the ÷gscale-scaled gradient). Naive matmuls in Lean's op order,
      -- compiled -ffp-contract=off ⇒ expected BIT-EXACT (max|Δ| = 0).
      let H := 24; let D := 16; let O := 6; let P := H*D + H + O*H + O
      let lr := 0.1; let wd := 0.0; let mu := 0.95; let eps := 1.0e-7; let gscale := 1.0 / 64.0
      let mut rng : UInt64 := 0x51A7
      let mut params : Array Float := #[]; let mut grad : Array Float := #[]; let mut mom : Array Float := #[]
      for _ in [0:P] do
        let (a, r1) := randF (-0.5) 0.5 rng; rng := r1; params := params.push a
        let (b, r2) := randF (-2.0) 2.0 rng; rng := r2; grad := grad.push b       -- raw summed gradient
        let (c, r3) := randF (-0.1) 0.1 rng; rng := r3; mom := mom.push c
      let u := USize.ofNat; let mk := FloatArray.mk
      let oW1 := 0; let ob1 := H*D; let oW2 := H*D+H; let ob2 := H*D+H+O*H
      let toMat := fun (fl : Array Float) (rows cols off : Nat) =>
        (Array.range rows).map (fun i => (Array.range cols).map (fun j => fl[off + i*cols+j]!))
      let toVec := fun (fl : Array Float) (len off : Nat) => (Array.range len).map (fun i => fl[off+i]!)
      let flatMat := fun (m : Array (Array Float)) => m.foldl (fun acc row => acc ++ row) #[]
      -- Lean oracle: applyMuon on the gscale-scaled gradient (matches the FFI's internal ÷N pre-scale)
      let p : Puffer.RL.NNTrain.MLP :=
        { W1 := toMat params H D oW1, b1 := toVec params H ob1, W2 := toMat params O H oW2, b2 := toVec params O ob2 }
      let st : Puffer.RL.NNTrain.MuonState :=
        { mW1 := toMat mom H D oW1, mb1 := toVec mom H ob1, mW2 := toMat mom O H oW2, mb2 := toVec mom O ob2 }
      let gs : Array Float := grad.map (· * gscale)
      let gTup := (toMat gs H D oW1, toVec gs H ob1, toMat gs O H oW2, toVec gs O ob2)
      let (newP, newSt) := Puffer.RL.NNTrain.applyMuon p st gTup lr wd mu eps
      let newParams := Puffer.RL.NNTrain.flattenMLP newP
      let newMom := (flatMat newSt.mW1) ++ newSt.mb1 ++ (flatMat newSt.mW2) ++ newSt.mb2
      let out := Puffer.Float.BLAS.muonStepMlpBlasFFI (mk (params ++ mom)) (mk grad) (u H) (u D) (u O) gscale lr wd mu eps
      let mut dp := 0.0; let mut dm := 0.0
      for i in [0:P] do
        dp := max dp (Float.abs (out[i]! - newParams[i]!))
        dm := max dm (Float.abs (out[P+i]! - newMom[i]!))
      IO.println s!"verify-muon-cpu (native-C whole-MLP Muon vs Lean applyMuon, H={H} D={D} O={O} P={P}):"
      IO.println s!"  newParams max|Δ| = {dp}   newMom max|Δ| = {dm}   ({if dp == 0.0 && dm == 0.0 then "ok ✓ (bit-exact)" else if dp < 1.0e-10 && dm < 1.0e-10 then "ok ✓ (tolerance)" else "CHECK"})"
  | "verify-sample-batch" :: _ => do
      -- Native-C batched rollout sampler (`sampleActionsBatchFFI`) vs the per-env Lean
      -- softmax→sampleCat→log→value. Expected BIT-EXACT: env n draws `rngNext` word
      -- `hash(startRng+(n+1)·G)`, the exact per-env stream the rollout used.
      let N := 96; let A := 6; let O := A + 1
      let mut rng : UInt64 := 0xC0FFEE1234
      let mut yb : Array Float := #[]
      for _ in [0:N*O] do let (v, r) := randF (-3.0) 3.0 rng; rng := r; yb := yb.push v
      let Yb := FloatArray.mk yb
      let startRng : UInt64 := 0x51A7BEEF
      let avl := Puffer.Float.FFI.sampleActionsBatchFFI Yb (USize.ofNat N) (USize.ofNat A) (USize.ofNat O) startRng
      let mut r2 := startRng
      let mut amiss := 0; let mut dlp := 0.0; let mut dv := 0.0
      for i in [0:N] do
        let out := (Array.range O).map (fun k => Yb[i*O+k]!)
        let probs := Puffer.RL.Train.softmax ((Array.range A).map (fun k => out[k]!))
        let (word, r') := rngNext r2; r2 := r'
        let a := Puffer.RL.Train.sampleCat probs (uniform01 word)
        if (avl[i]!).toUInt64.toNat != a then amiss := amiss + 1
        dlp := max dlp (Float.abs (avl[N+i]! - Float.log probs[a]!))
        dv  := max dv  (Float.abs (avl[2*N+i]! - out[A]!))
      IO.println s!"verify-sample-batch (native-C batched sampler vs Lean softmax+sampleCat, N={N} A={A}):"
      IO.println s!"  action mismatches = {amiss}/{N}   logp max|Δ| = {dlp}   value max|Δ| = {dv}   ({if amiss == 0 && dlp == 0.0 && dv == 0.0 then "ok ✓ (bit-exact)" else "CHECK"})"
  | "bench-ppo-grad-gpu" :: _ => do
      -- Speed of the tensor-core GEMM swap: full gradient at a realistic size, f32 (FP32 cores) vs
      -- bf16 (FAST_16BF tensor cores). Same malloc/transfer overhead both ways, so the delta is the GEMM.
      let N := 4096; let D := 128; let H := 256; let A := 17; let O := A + 1
      let vf := 0.5; let ent := 0.01; let clip := 0.2
      let (p, _) := Puffer.RL.NNTrain.initMLP D H O 0xB3C7
      let pa := Puffer.RL.NNTrain.flattenMLP p
      let obs := FloatArray.mk (Array.replicate (N*D) 0.1)
      let acts := FloatArray.mk (Array.replicate N 1.0)
      let advs := FloatArray.mk (Array.replicate N 0.3)
      let rets := FloatArray.mk (Array.replicate N 0.2)
      let olps := FloatArray.mk (Array.replicate N (-0.5))
      let u := USize.ofNat
      let reps := 20
      let timeIt := fun (bf : UInt8) => do
        let f := fun (k : Nat) => Puffer.Float.CUDA.cudaMlpPpoGradFFI (pa.set! 0 (pa[0]! + Float.ofNat k * 1.0e-9))
          obs acts advs rets olps (u N) (u H) (u D) (u A) vf ent clip bf
        let w := f 0; let mut cs := w[0]!
        let t0 ← IO.monoNanosNow
        for k in [1:reps+1] do let y := f k; cs := cs + y[0]!
        let t1 ← IO.monoNanosNow
        return Float.ofNat ((t1 - t0) / reps) / 1.0e6
      let msF32 ← timeIt 0
      let msBf16 ← timeIt 1
      IO.println s!"bench-ppo-grad-gpu (full PPO gradient, N={N} D={D} H={H} O={O}):"
      IO.println s!"  f32 (COMPUTE_32F)   {msF32} ms   |   bf16 (FAST_16BF tensor cores)  {msBf16} ms   |   {msF32/msBf16}× end-to-end"
      IO.println s!"  (tensor cores are engaged — bf16 < f32 — but this standalone gradient is malloc/transfer-bound:"
      IO.println s!"   19 cudaMallocs + the f64→f32 obs upload per call swamp the GEMM; the real speedup needs M5 residency.)"
  | "bench-train-step" :: _ => do
      -- Isolated PPO+Muon training-STEP throughput: the thing that actually differs between a GPU-resident
      -- trainer and a CPU trainer (rollout is identical CPU code in both). GPU = `cudaTrainStepFFI` (the RESIDENT
      -- whole step — normalize→gradient→Muon, device buffers cached across calls, exactly what the trainer
      -- runs). CPU = `mlpPPOGradBatchBlasFFI` (multi-threaded OpenBLAS gradient). Identical synthetic
      -- minibatch, warmup + reps, anti-CSE input perturbation.
      -- NOTE: the GPU step does MORE than the CPU number here (it also normalizes + runs Muon), and there is
      -- no CPU-pinnable Muon in the codebase (only pure-Lean scalar `applyMuon`, which is not a representative
      -- optimized baseline), so we compare against the CPU gradient alone ⇒ the CPU/GPU ratio is a
      -- CONSERVATIVE lower bound on the GPU step's true advantage. Sweeps toy → PufferLib scale.
      let vf := 0.5; let ent := 0.01; let clip := 0.2
      let lr := 0.05; let wd := 0.0; let mu := 0.95; let eps := 1.0e-7
      let u := USize.ofNat; let mk := FloatArray.mk
      let sizes : List (Nat × Nat × Nat × Nat) :=   -- (N, D, H, A)
        [(256, 25, 32, 5), (1024, 128, 128, 6), (2048, 128, 256, 17), (4096, 256, 256, 17), (8192, 256, 512, 17)]
      IO.println s!"bench-train-step — one PPO+Muon update on identical minibatch data (warmup + reps):"
      IO.println s!"  GPU = cudaTrainStepFFI (resident whole step: normalize+gradient+Muon, buffers cached)  vs  CPU = mlpPPOGradBatchBlasFFI (OpenBLAS gradient)."
      IO.println s!"  GPU does MORE than the CPU gradient shown ⇒ CPU/GPU ratio is a conservative lower bound on the GPU step advantage."
      for (N, D, H, A) in sizes do
        let O := A + 1
        let P := H*D + H + O*H + O
        let (p, _) := Puffer.RL.NNTrain.initMLP D H O 0xBEEF
        let pa := Puffer.RL.NNTrain.flattenMLP p
        let obs := mk (Array.replicate (N*D) 0.1)
        let acts := mk (Array.replicate N 1.0)
        let advs := mk (Array.replicate N 0.3)
        let rets := mk (Array.replicate N 0.2)
        let olps := mk (Array.replicate N (-0.5))
        let pmBase := mk ((Array.range P).map (fun i => pa[i]!) ++ Array.replicate P 0.0)   -- [params; mom]
        let reps := 20
        let gpuTime := fun (bf : UInt8) => do
          let f := fun (k : Nat) => Puffer.Float.CUDA.cudaTrainStepFFI (pmBase.set! 0 (pmBase[0]! + Float.ofNat k * 1.0e-9))
            obs acts advs rets olps (u N) (u H) (u D) (u A) lr wd mu eps vf ent clip bf
          let w := f 0; let mut cs := w[0]!
          let t0 ← IO.monoNanosNow
          for k in [1:reps+1] do let y := f k; cs := cs + y[0]!
          let t1 ← IO.monoNanosNow
          return Float.ofNat ((t1 - t0) / reps) / 1.0e6
        let cpuTime := do
          let f := fun (k : Nat) => Puffer.Float.BLAS.mlpPPOGradBatchBlasFFI (pa.set! 0 (pa[0]! + Float.ofNat k * 1.0e-9))
            obs acts advs rets olps (u N) (u H) (u D) (u A) vf ent clip
          let w := f 0; let mut cs := w[0]!
          let t0 ← IO.monoNanosNow
          for k in [1:reps+1] do let y := f k; cs := cs + y[0]!
          let t1 ← IO.monoNanosNow
          return Float.ofNat ((t1 - t0) / reps) / 1.0e6
        let gBf ← gpuTime 1
        let gF32 ← gpuTime 0
        let c ← cpuTime
        IO.println s!"  N={N} D={D} H={H} A={A} (P={P}):  CPU-grad {c}ms  |  GPU-step bf16 {gBf}ms  f32 {gF32}ms  |  CPU-grad/GPU-bf16 {c/gBf}×"
        (← IO.getStdout).flush
  | "bench-train-update" :: _ => do
      -- Isolates the training STEP (no rollout): the whole update (epochs×numMB minibatches) as the
      -- RESIDENT `cudaTrainUpdateFFI` (columns+pm upload once, pm downloads once) vs the per-minibatch
      -- loop (`cudaTrainStepFFI` × epochs·numMB, host gather + host↔device transfer each). The delta is
      -- the residency win — fewer transfers + fewer device syncs — which grows with the minibatch obs/pm.
      let A := 5; let O := A + 1
      let epochs := 4; let numMB := 4
      let vf := 0.5; let ent := 0.01; let clip := 0.2
      let lr := 0.02; let wd := 0.0; let mu := 0.95; let eps := 1.0e-7
      let u := USize.ofNat; let mk := FloatArray.mk
      -- (NT, H, D). Small D = toy RL (obs transfer negligible); large D = image/CNN-feature obs, where the
      -- per-minibatch obs upload dominates and residency (upload once) pays off.
      let sizes : List (Nat × Nat × Nat) :=
        [(1024, 32, 25), (4096, 64, 25), (8192, 256, 25), (8192, 128, 512), (16384, 128, 2048)]
      IO.println s!"bench-train-update — whole update ({epochs} epochs × {numMB} mb) on the device: RESIDENT (1 up/1 down) vs per-minibatch loop:"
      for (NT, H, D) in sizes do
        let P := H*D + H + O*H + O
        let (p, _) := Puffer.RL.NNTrain.initMLP D H O 0xBEEF
        let pa := Puffer.RL.NNTrain.flattenMLP p
        let pm := mk ((Array.range P).map (fun i => pa[i]!) ++ Array.replicate P 0.0)
        let obs := mk (Array.replicate (NT*D) 0.1)
        let acts := mk (Array.replicate NT 1.0); let adv := mk (Array.replicate NT 0.3)
        let ret := mk (Array.replicate NT 0.2); let olp := mk (Array.replicate NT (-0.5))
        let mut permFlat : Array Float := #[]
        for _ in [0:epochs] do for k in [0:NT] do permFlat := permFlat.push (Float.ofNat k)   -- identity perms
        let permF := mk permFlat
        let mbSize := NT / numMB
        let reps := 5
        let residentTime := do
          let f := fun (k : Nat) => Puffer.Float.CUDA.cudaTrainUpdateFFI (pm.set! 0 (pm[0]! + Float.ofNat k * 1.0e-9))
            obs acts adv ret olp permF (u NT) (u D) (u H) (u A) (u epochs) (u numMB) lr wd mu eps vf ent clip 1
          let w := f 0; let mut cs := w[0]!
          let t0 ← IO.monoNanosNow
          for k in [1:reps+1] do let y := f k; cs := cs + y[0]!
          let t1 ← IO.monoNanosNow
          return Float.ofNat ((t1 - t0) / reps) / 1.0e6
        let nonResTime := do
          let runOnce := fun (seed : Nat) => Id.run do
            let mut pmL := pm.set! 0 (pm[0]! + Float.ofNat seed * 1.0e-9)
            for e in [0:epochs] do
              for m in [0:numMB] do
                let _ := e
                let lo := m*mbSize; let hi := if m+1==numMB then NT else (m+1)*mbSize
                let idxsF := mk ((Array.range (hi-lo)).map (fun k => Float.ofNat (lo+k)))
                let (mbObs, mbAct, mbAdv, mbRet, mbOlp) :=
                  Puffer.Float.FFI.gatherMinibatchFFI obs acts adv ret olp idxsF (u (hi-lo)) (u D)
                pmL := Puffer.Float.CUDA.cudaTrainStepFFI pmL mbObs mbAct mbAdv mbRet mbOlp (u (hi-lo)) (u H) (u D) (u A) lr wd mu eps vf ent clip 1
            return pmL[0]!
          let w := runOnce 0; let mut cs := w
          let t0 ← IO.monoNanosNow
          for k in [1:reps+1] do let y := runOnce k; cs := cs + y
          let t1 ← IO.monoNanosNow
          return Float.ofNat ((t1 - t0) / reps) / 1.0e6
        let rt ← residentTime
        let nrt ← nonResTime
        IO.println s!"  NT={NT} H={H} D={D} (mbSize={mbSize}, P={P}):  resident {rt}ms  |  per-minibatch {nrt}ms  |  {nrt/rt}× faster resident"
        (← IO.getStdout).flush
  | "verify-blas" :: _ => do
      -- batched dense forward relu(X·Wᵀ+b) — 3 backends vs a pure-Lean oracle
      let N := 64; let D := 48; let H := 32
      let mut rng : UInt64 := 0xB1A5
      let mkArr := fun (n : Nat) (r0 : UInt64) => Id.run do
        let mut a : Array Float := #[]; let mut r := r0
        for _ in [0:n] do let (v, r') := randF (-1.0) 1.0 r; r := r'; a := a.push v
        return (a, r)
      let (xa, r1) := mkArr (N*D) rng; rng := r1
      let (wa, r2) := mkArr (H*D) rng; rng := r2
      let (ba, _r3) := mkArr H rng
      let X := FloatArray.mk xa; let W := FloatArray.mk wa; let B := FloatArray.mk ba
      let u1 := USize.ofNat
      -- pure-Lean oracle: Y[i,j] = relu(b[j] + Σ_d X[i,d]·W[j,d]) (left fold)
      let mut oracle : Array Float := Array.replicate (N*H) 0.0
      for i in [0:N] do
        for j in [0:H] do
          let mut acc := ba[j]!
          for d in [0:D] do acc := acc + xa[i*D+d]! * wa[j*D+d]!
          oracle := oracle.set! (i*H+j) (if acc > 0.0 then acc else 0.0)
      let maxd := fun (a : FloatArray) => Id.run do
        let mut m := 0.0
        for i in [0:N*H] do m := max m (Float.abs (a[i]! - oracle[i]!))
        return m
      let yRef := Puffer.Float.BLAS.denseForwardRefFFI X W B (u1 N) (u1 D) (u1 H)
      let yBlas := Puffer.Float.BLAS.denseForwardBlasFFI X W B (u1 N) (u1 D) (u1 H)
      let yCublas := Puffer.Float.BLAS.denseForwardCublasFFI X W B (u1 N) (u1 D) (u1 H)
      let yLtBf16 := Puffer.Float.BLAS.denseForwardCublasLtBf16FFI X W B (u1 N) (u1 D) (u1 H)
      let cuda := Puffer.Float.BLAS.cudaAvailableFFI ()
      IO.println s!"verify-blas (dense forward relu(X·Wᵀ+b), N={N} D={D} H={H}) vs pure-Lean oracle:"
      IO.println s!"  scalar C ref : max abs diff = {maxd yRef}   (left-fold ⇒ bit-exact)"
      IO.println s!"  OpenBLAS     : max abs diff = {maxd yBlas}   (blocked reduction ⇒ tolerance)"
      IO.println s!"  cuBLAS-f64 (dev={cuda}): max abs diff = {maxd yCublas}   (parallel reduction ⇒ tolerance)"
      IO.println s!"  cuBLASLt-bf16 (dev={cuda}): max abs diff = {maxd yLtBf16}   (bf16 inputs ⇒ ~1e-1, PufferLib's precision)"
  | "bench-blas" :: _ => do
      let cuda := Puffer.Float.BLAS.cudaAvailableFFI ()
      IO.println s!"bench-blas — batched dense forward relu(X·Wᵀ+b), square N=D=H; cuda_available={cuda}"
      IO.println "  (scalar C = bit-exact oracle path; OpenBLAS/cuBLAS = tolerance-validated)"
      (← IO.getStdout).flush
      let u1 := USize.ofNat
      for sz in [256, 512, 1024, 2048] do
        let N := sz; let D := sz; let H := sz
        let mut rng : UInt64 := 0xBEE5
        let mut xa : Array Float := Array.replicate (N*D) 0.0
        for i in [0:N*D] do let (v, r) := randF (-1.0) 1.0 rng; rng := r; xa := xa.set! i v
        let mut wa : Array Float := Array.replicate (H*D) 0.0
        for i in [0:H*D] do let (v, r) := randF (-1.0) 1.0 rng; rng := r; wa := wa.set! i v
        let ba : Array Float := Array.replicate H 0.1
        let X := FloatArray.mk xa; let W := FloatArray.mk wa; let B := FloatArray.mk ba
        let reps := 3
        -- Each call perturbs the bias by the loop index `k`: this both keeps every call a
        -- REAL matmul (defeats the compiler CSE-ing identical pure FFI calls) and lets the
        -- untimed `f 0` warmup pay one-time per-shape costs (CUDA lazy kernel-module load).
        let bk := fun (k : Nat) => B.set! 0 (Float.ofNat k * 0.001)
        let timeIt := fun (f : Nat → FloatArray) => do
          let w := f 0
          let mut cs := w[0]!
          let t0 ← IO.monoNanosNow
          for k in [1:reps+1] do let y := f k; cs := cs + y[0]!
          let t1 ← IO.monoNanosNow
          return (Float.ofNat ((t1 - t0) / reps) / 1.0e6, cs)   -- ms/call
        let gflop := 2.0 * Float.ofNat N * Float.ofNat D * Float.ofNat H / 1.0e9
        let gfps := fun (ms : Float) => gflop / (ms / 1000.0)
        let (msBlas, _) ← timeIt (fun k => Puffer.Float.BLAS.denseForwardBlasFFI X W (bk k) (u1 N) (u1 D) (u1 H))
        let (msCu, _) ← timeIt (fun k => Puffer.Float.BLAS.denseForwardCublasFFI X W (bk k) (u1 N) (u1 D) (u1 H))
        let (msLt, _) ← timeIt (fun k => Puffer.Float.BLAS.denseForwardCublasLtBf16FFI X W (bk k) (u1 N) (u1 D) (u1 H))
        let resid := Puffer.Float.BLAS.benchCublasLtBf16ResidentFFI (u1 N) (u1 D) (u1 H) (u1 200)
        if sz ≤ 1024 then
          let (msRef, _) ← timeIt (fun k => Puffer.Float.BLAS.denseForwardRefFFI X W (bk k) (u1 N) (u1 D) (u1 H))
          IO.println s!"  {sz}³: scalar {msRef}ms ({gfps msRef} GF/s)  OpenBLAS {msBlas}ms ({gfps msBlas} GF/s)  cuBLAS-f64 {msCu}ms ({gfps msCu} GF/s)  cuBLASLt-bf16 {msLt}ms ({gfps msLt} GF/s)"
        else
          IO.println s!"  {sz}³: scalar (skipped)  OpenBLAS {msBlas}ms ({gfps msBlas} GF/s)  cuBLAS-f64 {msCu}ms ({gfps msCu} GF/s)  cuBLASLt-bf16 {msLt}ms ({gfps msLt} GF/s)"
        IO.println s!"        └ bf16 GEMM-only (device-resident, no transfer): {resid} GF/s  ← tensor-core ceiling of our path"
        (← IO.getStdout).flush
  | "bench-resident" :: _ => do
      -- On-device bf16 residency for the 2-layer MLP forward (the rollout policy forward):
      -- OpenBLAS f64 (CPU) vs bf16 resident-intermediate (per-call transfer) vs bf16 fully-resident.
      let cuda := Puffer.Float.BLAS.cudaAvailableFFI ()
      IO.println s!"bench-resident — 2-layer MLP forward relu(X·W1ᵀ+b1)·W2ᵀ+b2 (rollout policy); cuda={cuda}"
      IO.println "  H1 stays on the GPU between layers (bf16 epilogue). 'fully-resident' also keeps W,X on-device."
      (← IO.getStdout).flush
      let u1 := USize.ofNat
      for (N, D, H, O) in [(1024, 128, 128, 7), (4096, 128, 256, 7), (8192, 256, 256, 18)] do
        let P := H*D + H + O*H + O
        let mut rng : UInt64 := 0xF00D
        let mut pa : Array Float := Array.replicate P 0.0
        for i in [0:P] do let (v, r) := randF (-0.1) 0.1 rng; rng := r; pa := pa.set! i v
        let mut xb : Array Float := Array.replicate (N*D) 0.0
        for i in [0:N*D] do let (v, r) := randF (-1.0) 1.0 rng; rng := r; xb := xb.set! i v
        let params := FloatArray.mk pa; let Xb := FloatArray.mk xb
        let flop := (2.0 * Float.ofNat N * Float.ofNat D * Float.ofNat H
                     + 2.0 * Float.ofNat N * Float.ofNat H * Float.ofNat O) / 1.0e9
        let reps := 12
        let timeIt := fun (f : Nat → FloatArray) => do
          let w := f 0; let mut cs := w[0]!
          let t0 ← IO.monoNanosNow
          for k in [1:reps+1] do let y := f k; cs := cs + y[0]!
          let t1 ← IO.monoNanosNow
          return Float.ofNat ((t1 - t0) / reps) / 1.0e6
        let perturb := fun (k : Nat) => params.set! 0 (pa[0]! + Float.ofNat k * 1.0e-9)
        let msBlas ← timeIt (fun k => Puffer.Float.BLAS.mlpForwardBatchBlasFFI (perturb k) Xb (u1 N) (u1 D) (u1 H) (u1 O))
        let msBf16 ← timeIt (fun k => Puffer.Float.BLAS.mlpForwardBatchCublasLtBf16FFI (perturb k) Xb (u1 N) (u1 D) (u1 H) (u1 O))
        let wresGf := Puffer.Float.BLAS.benchMlp2Bf16WeightsResidentFFI (u1 N) (u1 D) (u1 H) (u1 O) (u1 200)
        let residGf := Puffer.Float.BLAS.benchMlp2Bf16ResidentFFI (u1 N) (u1 D) (u1 H) (u1 O) (u1 200)
        let gf := fun (ms : Float) => flop / (ms / 1000.0)
        IO.println s!"  N={N} D={D} H={H} O={O}:  OpenBLAS-f64 {gf msBlas} | bf16 per-call {gf msBf16} | bf16 weights-resident {wresGf} | bf16 fully-resident {residGf}  (GF/s)"
        (← IO.getStdout).flush
  | "verify-blas-fwd" :: _ => do
      -- batched MLP forward (the rollout hot path) vs the per-row bit-exact oracle.
      -- Aligned dims (multiples of 8) so the bf16 tensor-core path runs rather than falling back.
      let N := 128; let D := 64; let H := 64; let A := 7; let O := A + 1
      let (p, rng0) := Puffer.RL.NNTrain.initMLP D H O 0xBFA5
      let mut rng := rng0
      let mut xb : Array Float := #[]
      let mut rows : Array (Array Float) := #[]
      for _ in [0:N] do
        let mut o : Array Float := #[]
        for _ in [0:D] do let (v, r) := randF (-1.0) 1.0 rng; rng := r; o := o.push v
        rows := rows.push o
        for x in o do xb := xb.push x
      let params := Puffer.RL.NNTrain.flattenMLP p
      let u1 := USize.ofNat
      let yRef := Puffer.Float.BLAS.mlpForwardBatchRefFFI params (FloatArray.mk xb) (u1 N) (u1 D) (u1 H) (u1 O)
      let yBlas := Puffer.Float.BLAS.mlpForwardBatchBlasFFI params (FloatArray.mk xb) (u1 N) (u1 D) (u1 H) (u1 O)
      let yBf16 := Puffer.Float.BLAS.mlpForwardBatchCublasLtBf16FFI params (FloatArray.mk xb) (u1 N) (u1 D) (u1 H) (u1 O)
      -- persistent resident policy: load weights once, forward, free
      let hPol := Puffer.Float.BLAS.mlpPolicyLoadFFI params (u1 D) (u1 H) (u1 O)
      let yPol := Puffer.Float.BLAS.mlpPolicyForwardFFI hPol (FloatArray.mk xb) (u1 N)
      let freed := Puffer.Float.BLAS.mlpPolicyFreeFFI hPol
      let mut dRef := 0.0
      let mut dBlas := 0.0
      let mut dBf16 := 0.0
      let mut dPol := 0.0
      for i in [0:N] do
        let orow := Puffer.Float.FFI.mlpForwardFFI params (FloatArray.mk rows[i]!) (u1 H) (u1 D) (u1 O)
        for k in [0:O] do
          dRef := max dRef (Float.abs (yRef[i*O+k]! - orow[k]!))
          dBlas := max dBlas (Float.abs (yBlas[i*O+k]! - orow[k]!))
          dBf16 := max dBf16 (Float.abs (yBf16[i*O+k]! - orow[k]!))
          dPol := max dPol (Float.abs (yPol[i*O+k]! - orow[k]!))
      let cuda := Puffer.Float.BLAS.cudaAvailableFFI ()
      IO.println s!"verify-blas-fwd (batched MLP forward N={N} D={D} H={H} O={O}) vs per-row mlpForwardFFI oracle:"
      IO.println s!"  scalar batch  : max abs diff = {dRef}   (right-fold ⇒ bit-exact)"
      IO.println s!"  OpenBLAS batch: max abs diff = {dBlas}   (tolerance)"
      IO.println s!"  bf16 resident (dev={cuda}): max abs diff = {dBf16}   (H1 stays on-GPU; bf16 ⇒ ~1e-1)"
      IO.println s!"  bf16 persistent policy (load/forward/free={freed}): max abs diff = {dPol}   (weights resident across forwards)"
  | "verify-grad-blas" :: _ => do
      -- BLAS minibatch gradient vs the scalar-FFI kernel (itself 0.000000 vs the Lean oracle)
      let N := 8; let D := 6; let H := 5; let A := 3; let O := A + 1
      let (p, rng0) := Puffer.RL.NNTrain.initMLP D H O 0x77
      let mut rng := rng0
      let mut obsB : Array Float := #[]; let mut acts : Array Float := #[]
      let mut advs : Array Float := #[]; let mut rets : Array Float := #[]; let mut olps : Array Float := #[]
      for _ in [0:N] do
        for _ in [0:D] do let (v, r) := randF (-1.0) 1.0 rng; rng := r; obsB := obsB.push v
        let (aw, r1) := rngNext rng; rng := r1; acts := acts.push (Float.ofNat (aw.toNat % A))
        let (av, r2) := randF (-1.0) 1.0 rng; rng := r2; advs := advs.push av
        let (rv, r3) := randF 0.0 2.0 rng; rng := r3; rets := rets.push rv
        olps := olps.push (-0.9)
      let params := Puffer.RL.NNTrain.flattenMLP p
      let u1 := USize.ofNat; let vf := 0.5; let ent := 0.01; let clip := 0.2
      let oB := FloatArray.mk obsB; let aB := FloatArray.mk acts; let dB := FloatArray.mk advs
      let rB := FloatArray.mk rets; let lB := FloatArray.mk olps
      let gS := Puffer.Float.FFI.mlpPPOGradBatchFFI params oB aB dB rB lB (u1 N) (u1 H) (u1 D) (u1 A) vf ent clip
      let gBl := Puffer.Float.BLAS.mlpPPOGradBatchBlasFFI params oB aB dB rB lB (u1 N) (u1 H) (u1 D) (u1 A) vf ent clip
      let mut maxAbs := 0.0
      let mut maxRel := 0.0
      for i in [0:gS.size] do
        let d := Float.abs (gS[i]! - gBl[i]!)
        maxAbs := max maxAbs d
        maxRel := max maxRel (d / (Float.abs gS[i]! + 1.0e-12))
      IO.println s!"verify-grad-blas (MLP+PPO minibatch gradient, N={N} D={D} H={H} A={A}) BLAS vs scalar-FFI oracle:"
      IO.println s!"  max abs diff = {maxAbs}   max rel diff = {maxRel}   (P={gS.size}; blocked reduction ⇒ tolerance, not bit-exact)"
  | "bench-grad" :: _ => do
      -- scalar-FFI vs OpenBLAS minibatch gradient (the training hot path; no env-step floor)
      let N := 1024; let D := 256; let A := 6
      IO.println s!"bench-grad — MLP+PPO minibatch gradient, N={N} transitions, D={D}, {A} actions; sweeping hidden"
      (← IO.getStdout).flush
      let mut rng : UInt64 := 0x33
      let mut obsB : Array Float := #[]; let mut acts : Array Float := #[]
      let mut advs : Array Float := #[]; let mut rets : Array Float := #[]; let mut olps : Array Float := #[]
      for _ in [0:N] do
        for _ in [0:D] do let (v, r) := randF (-1.0) 1.0 rng; rng := r; obsB := obsB.push v
        let (aw, r1) := rngNext rng; rng := r1; acts := acts.push (Float.ofNat (aw.toNat % A))
        let (av, r2) := randF (-1.0) 1.0 rng; rng := r2; advs := advs.push av
        let (rv, r3) := randF 0.0 2.0 rng; rng := r3; rets := rets.push rv
        olps := olps.push (-0.9)
      let oB := FloatArray.mk obsB; let aB := FloatArray.mk acts; let dB := FloatArray.mk advs
      let rB := FloatArray.mk rets; let lB := FloatArray.mk olps
      let u1 := USize.ofNat; let vf := 0.5; let ent := 0.01; let clip := 0.2
      for hidden in [128, 512, 2048] do
        let (p, _) := Puffer.RL.NNTrain.initMLP D hidden (A + 1) 0x99
        let params := Puffer.RL.NNTrain.flattenMLP p
        let reps := 5
        let timeIt := fun (useBlas : Bool) => do
          let _ := if useBlas then Puffer.Float.BLAS.mlpPPOGradBatchBlasFFI params oB aB dB rB lB (u1 N) (u1 hidden) (u1 D) (u1 A) vf ent clip
                   else Puffer.Float.FFI.mlpPPOGradBatchFFI params oB aB dB rB lB (u1 N) (u1 hidden) (u1 D) (u1 A) vf ent clip  -- warmup
          let mut cs := 0.0
          let t0 ← IO.monoNanosNow
          for k in [0:reps] do
            let dK := dB.set! 0 (Float.ofNat k * 0.001)   -- perturb per rep ⇒ defeats CSE
            let g := if useBlas then Puffer.Float.BLAS.mlpPPOGradBatchBlasFFI params oB aB dK rB lB (u1 N) (u1 hidden) (u1 D) (u1 A) vf ent clip
                     else Puffer.Float.FFI.mlpPPOGradBatchFFI params oB aB dK rB lB (u1 N) (u1 hidden) (u1 D) (u1 A) vf ent clip
            cs := cs + g[0]!
          let t1 ← IO.monoNanosNow
          return (Float.ofNat ((t1 - t0) / reps) / 1.0e6, cs)
        let (msS, _) ← timeIt false
        let (msB, _) ← timeIt true
        IO.println s!"  hidden {hidden}: scalar-FFI {msS} ms  OpenBLAS {msB} ms  (gradient speedup {msS / (max msB 0.0001)}×)"
        (← IO.getStdout).flush
  | "verify-md-grad" :: _ => do
      -- Multi-discrete PPO gradient (cudaMlpPpoGradMDFFI) vs finite differences of the Lean md objective
      -- (joint categorical log-prob over K heads → PPO clip → +entropy −½·vf·(V−ret)²). f32 GPU vs f64 FD ⇒
      -- tolerance, not bit-exact; confirms the head decomposition + backprop are correct.
      let D := 8; let H := 6; let headSizes : Array Nat := #[3, 4]; let K := headSizes.size
      let A := headSizes.foldl (·+·) 0; let O := A + 1; let N := 5
      let P := H*D + H + O*H + O
      let vf := 0.5; let ent := 0.01; let clip := 0.2
      let u := USize.ofNat; let mk := FloatArray.mk
      let mut rng : UInt64 := 0xD15C
      let mut params : Array Float := #[]
      for _ in [0:P] do let (v, r) := randF (-0.5) 0.5 rng; rng := r; params := params.push v
      let mut obsB : Array Float := #[]; let mut acts : Array Float := #[]
      let mut advs : Array Float := #[]; let mut rets : Array Float := #[]; let mut olps : Array Float := #[]
      for _ in [0:N] do
        for _ in [0:D] do let (v, r) := randF (-1.0) 1.0 rng; rng := r; obsB := obsB.push v
        for hh in [0:K] do let (aw, r) := rngNext rng; rng := r; acts := acts.push (Float.ofNat (aw.toNat % headSizes[hh]!))
        let (av, r2) := randF (-1.0) 1.0 rng; rng := r2; advs := advs.push av
        let (rv, r3) := randF 0.0 2.0 rng; rng := r3; rets := rets.push rv
        olps := olps.push (-1.3)
      let hsF := mk (headSizes.map Float.ofNat)
      let g := Puffer.Float.CUDA.cudaMlpPpoGradMDFFI (mk params) (mk obsB) (mk acts) (mk advs) (mk rets) (mk olps) hsF
                 (u N) (u H) (u D) (u K) vf ent clip 0
      -- Lean f64 md objective (summed over rows)
      let objAt := fun (pp : Array Float) => Id.run do
        let W1 := fun i j => pp[i*D+j]!; let b1 := fun i => pp[H*D+i]!
        let W2 := fun i j => pp[H*D+H+i*H+j]!; let b2 := fun i => pp[H*D+H+O*H+i]!
        let mut total := 0.0
        for n in [0:N] do
          let h1 := (Array.range H).map (fun i =>
            let z := b1 i + (Array.range D).foldl (fun a j => a + W1 i j * obsB[n*D+j]!) 0.0
            if z > 0.0 then z else 0.0)
          let logit := fun k => b2 k + (Array.range H).foldl (fun a j => a + W2 k j * h1[j]!) 0.0
          let out := (Array.range O).map logit
          -- joint logp + entropy over heads
          let mut off := 0; let mut jlogp := 0.0; let mut entSum := 0.0
          for hh in [0:K] do
            let sz := headSizes[hh]!; let a := (acts[n*K+hh]!).toUInt64.toNat
            let mx := (Array.range sz).foldl (fun m k => max m (out[off+k]!)) (out[off]!)
            let z := (Array.range sz).foldl (fun s k => s + Float.exp (out[off+k]! - mx)) 0.0
            let lse := mx + Float.log z
            jlogp := jlogp + out[off+a]! - lse
            let pout := (Array.range sz).foldl (fun s k => s + Float.exp (out[off+k]! - lse) * out[off+k]!) 0.0
            entSum := entSum + (lse - pout)          -- H_h = lse − Σ p·logit
            off := off + sz
          let ratio := Float.exp (jlogp - olps[n]!)
          let rc := max (1.0 - clip) (min (1.0 + clip) ratio)
          let surr := min (advs[n]! * ratio) (advs[n]! * rc)
          let vval := out[O-1]!
          total := total + surr + ent * entSum - 0.5 * vf * (vval - rets[n]!) * (vval - rets[n]!)
        return total
      let eps := 1.0e-5
      let mut maxRel := 0.0
      for i in [0, 17, 55, H*D+2, H*D+H+10, H*D+H+O*H+3] do   -- sample W1/b1/W2/b2 params
        if i < P then
          let up := params.set! i (params[i]! + eps); let dn := params.set! i (params[i]! - eps)
          let fd := (objAt up - objAt dn) / (2.0 * eps)
          let d := Float.abs (g[i]! - fd); let rel := d / (Float.abs fd + 1.0e-6)
          maxRel := max maxRel rel
      IO.println s!"verify-md-grad (multi-discrete PPO grad vs finite-diff, K={K} heads {headSizes}, N={N}, D={D}→H={H}→O={O}):"
      IO.println s!"  max relative |grad − FD| = {maxRel}   ({if maxRel < 1.0e-2 then "ok ✓ (f32 GPU vs f64 FD)" else "CHECK"})"
  | "verify-cont-grad" :: _ => do
      -- Continuous (diagonal-Gaussian) PPO gradient (cudaMlpPpoGradContFFI) vs finite differences of the
      -- Lean Gaussian objective (Σ logp over d dims → PPO clip → +entropy −½·vf·(V−ret)²). f32 GPU vs f64 FD
      -- ⇒ tolerance; confirms the per-dim mean/logstd gradients (clamp-gated) + backprop are correct.
      let D := 8; let H := 6; let d := 3; let O := 2*d + 1; let N := 5
      let P := H*D + H + O*H + O
      let vf := 0.5; let ent := 0.01; let clip := 0.2
      let lsLo := -5.0; let lsHi := 2.0
      let halfLog2pi := 0.9189385332046727; let halfLog2pieE := 1.4189385332046727
      let clampF := fun (x lo hi : Float) => max lo (min hi x)
      let u := USize.ofNat; let mk := FloatArray.mk
      let mut rng : UInt64 := 0xC047
      let mut params : Array Float := #[]
      for _ in [0:P] do let (v, r) := randF (-0.5) 0.5 rng; rng := r; params := params.push v
      let mut obsB : Array Float := #[]; let mut acts : Array Float := #[]
      let mut advs : Array Float := #[]; let mut rets : Array Float := #[]; let mut olps : Array Float := #[]
      for _ in [0:N] do
        for _ in [0:D] do let (v, r) := randF (-1.0) 1.0 rng; rng := r; obsB := obsB.push v
        for _ in [0:d] do let (av, r) := randF (-1.0) 1.0 rng; rng := r; acts := acts.push av
        let (av, r2) := randF (-1.0) 1.0 rng; rng := r2; advs := advs.push av
        let (rv, r3) := randF 0.0 2.0 rng; rng := r3; rets := rets.push rv
        olps := olps.push (-1.1)
      let g := Puffer.Float.CUDA.cudaMlpPpoGradContFFI (mk params) (mk obsB) (mk acts) (mk advs) (mk rets) (mk olps)
                 (u N) (u H) (u D) (u d) vf ent clip 0
      -- Lean f64 Gaussian objective (summed over rows)
      let objAt := fun (pp : Array Float) => Id.run do
        let W1 := fun i j => pp[i*D+j]!; let b1 := fun i => pp[H*D+i]!
        let W2 := fun i j => pp[H*D+H+i*H+j]!; let b2 := fun i => pp[H*D+H+O*H+i]!
        let mut total := 0.0
        for n in [0:N] do
          let h1 := (Array.range H).map (fun i =>
            let z := b1 i + (Array.range D).foldl (fun a j => a + W1 i j * obsB[n*D+j]!) 0.0
            if z > 0.0 then z else 0.0)
          let logit := fun k => b2 k + (Array.range H).foldl (fun a j => a + W2 k j * h1[j]!) 0.0
          let out := (Array.range O).map logit
          let mut logp := 0.0; let mut entSum := 0.0
          for i in [0:d] do
            let ls := clampF (out[d+i]!) lsLo lsHi
            let z := (acts[n*d+i]! - out[i]!) * Float.exp (-ls)
            logp := logp + (-0.5 * z * z - ls - halfLog2pi)
            entSum := entSum + (ls + halfLog2pieE)
          let ratio := Float.exp (logp - olps[n]!)
          let rc := max (1.0 - clip) (min (1.0 + clip) ratio)
          let surr := min (advs[n]! * ratio) (advs[n]! * rc)
          let vval := out[O-1]!
          total := total + surr + ent * entSum - 0.5 * vf * (vval - rets[n]!) * (vval - rets[n]!)
        return total
      let eps := 1.0e-5
      let mut maxRel := 0.0
      for i in [0, 17, 55, H*D+2, H*D+H+10, H*D+H+O*H+3, H*D+H+O*H+(2*d)] do   -- W1/b1/W2/b2 incl. logstd + value cols
        if i < P then
          let up := params.set! i (params[i]! + eps); let dn := params.set! i (params[i]! - eps)
          let fd := (objAt up - objAt dn) / (2.0 * eps)
          let dd := Float.abs (g[i]! - fd); let rel := dd / (Float.abs fd + 1.0e-6)
          maxRel := max maxRel rel
      IO.println s!"verify-cont-grad (continuous Gaussian PPO grad vs finite-diff, d={d} dims, N={N}, D={D}→H={H}→O={O}):"
      IO.println s!"  max relative |grad − FD| = {maxRel}   ({if maxRel < 1.0e-2 then "ok ✓ (f32 GPU vs f64 FD)" else "CHECK"})"
  | "verify-cnn-hybrid" :: _ => do
      -- Hybrid conv encoder (conv over the brick grid + `nScalar` physics-scalar passthrough): the
      -- extended BLAS gradient vs the AD oracle (`cnnGradPPO`), on breakout-shaped obs (10 scalars +
      -- a 1×6×18 brick grid). Tolerance-close (BLAS blocked reductions vs the scalar AD tape).
      let C := 1; let inH := 6; let inW := 18; let nF := 8; let k := 3; let s := 1; let hidden := 32
      let nScalar := 10; let A := 3; let O := A + 1; let N := 6
      let D := nScalar + C*inH*inW
      let (p, rng0) := Puffer.RL.NNTrain.initCnn C inH inW nF k s hidden O 0xB00B nScalar
      let u1 := USize.ofNat; let vf := 0.5; let ent := 0.01; let clip := 0.2
      let mut rng := rng0
      let mut obsB : Array Float := #[]; let mut acts : Array Float := #[]
      let mut advs : Array Float := #[]; let mut rets : Array Float := #[]; let mut olps : Array Float := #[]
      let mut buf : Array Puffer.RL.NNTrain.Transition := #[]
      for _ in [0:N] do
        let mut o : Array Float := #[]
        for _ in [0:D] do let (v, r) := randF (-1.0) 1.0 rng; rng := r; o := o.push v
        obsB := obsB ++ o
        let (aw, r1) := rngNext rng; rng := r1; let a := aw.toNat % A
        let (av, r2) := randF (-1.0) 1.0 rng; rng := r2
        let (rv, r3) := randF 0.0 2.0 rng; rng := r3
        acts := acts.push (Float.ofNat a); advs := advs.push av; rets := rets.push rv; olps := olps.push (-0.9)
        buf := buf.push { obs := o, action := a, reward := 0.0, value := 0.0, oldLogp := -0.9, terminal := false }
      let params := Puffer.RL.NNTrain.flattenCnn p
      let gBl := Puffer.Float.BLAS.cnnPPOGradBatchBlasFFI params (FloatArray.mk obsB) (FloatArray.mk acts)
                   (FloatArray.mk advs) (FloatArray.mk rets) (FloatArray.mk olps)
                   (u1 N) (u1 C) (u1 inH) (u1 inW) (u1 nF) (u1 k) (u1 s) (u1 hidden) (u1 A) vf ent clip (u1 nScalar)
      let gr := Puffer.RL.NNTrain.cnnGradIdx p buf advs rets (Array.range N) A vf ent clip
      let leanFlat := Puffer.RL.NNTrain.flattenCnn
        { p with convW := gr.gConvW, convB := gr.gConvB, W1 := gr.gW1, b1 := gr.gb1, W2 := gr.gW2, b2 := gr.gb2 }
      let mut maxd := 0.0
      for i in [0:gBl.size] do maxd := max maxd (Float.abs (gBl[i]! - leanFlat[i]!))
      IO.println s!"verify-cnn-hybrid (conv + {nScalar}-scalar passthrough, BLAS grad vs AD oracle, N={N}, 1×6×18 → nF{nF} → {hidden}):"
      IO.println s!"  max |BLAS − AD| = {maxd}   (P={gBl.size})   ({if maxd < 1.0e-9 then "ok ✓ (bit-close)" else if maxd < 1.0e-5 then "ok ✓ (blocked-reduction tol)" else "CHECK"})"
  | "verify-cnn-blas" :: _ => do
      -- BLAS CNN gradient (im2col+GEMM) vs the scalar-FFI CNN kernel (0.000000 vs Lean oracle).
      -- Sweeps stride 1 AND 2 so both conv paths (and the layout transposes) are exercised.
      IO.println "verify-cnn-blas (CNN+PPO minibatch gradient) BLAS vs scalar-FFI oracle:"
      let u1 := USize.ofNat; let vf := 0.5; let ent := 0.01; let clip := 0.2
      for cfg in [(2,8,8,3,3,1,8,4), (2,8,8,3,3,2,8,4), (3,11,11,4,3,2,16,5)] do
        let (C, inH, inW, nF, k, s, hidden, A) := cfg
        let N := 6; let O := A + 1
        let (p, rng0) := Puffer.RL.NNTrain.initCnn C inH inW nF k s hidden O 0xC0FFEE
        let mut rng := rng0
        let mut obsB : Array Float := #[]; let mut acts : Array Float := #[]
        let mut advs : Array Float := #[]; let mut rets : Array Float := #[]; let mut olps : Array Float := #[]
        for _ in [0:N] do
          for _ in [0:C*inH*inW] do let (v, r) := randF (-1.0) 1.0 rng; rng := r; obsB := obsB.push v
          let (aw, r1) := rngNext rng; rng := r1; acts := acts.push (Float.ofNat (aw.toNat % A))
          let (av, r2) := randF (-1.0) 1.0 rng; rng := r2; advs := advs.push av
          let (rv, r3) := randF 0.0 2.0 rng; rng := r3; rets := rets.push rv
          olps := olps.push (-0.9)
        let params := Puffer.RL.NNTrain.flattenCnn p
        let oB := FloatArray.mk obsB; let aB := FloatArray.mk acts; let dB := FloatArray.mk advs
        let rB := FloatArray.mk rets; let lB := FloatArray.mk olps
        let gS := Puffer.Float.FFI.cnnPPOGradBatchFFI params oB aB dB rB lB
                    (u1 N) (u1 C) (u1 inH) (u1 inW) (u1 nF) (u1 k) (u1 s) (u1 hidden) (u1 A) vf ent clip
        let gBl := Puffer.Float.BLAS.cnnPPOGradBatchBlasFFI params oB aB dB rB lB
                    (u1 N) (u1 C) (u1 inH) (u1 inW) (u1 nF) (u1 k) (u1 s) (u1 hidden) (u1 A) vf ent clip
        let mut maxAbs := 0.0; let mut maxRel := 0.0
        for i in [0:gS.size] do
          let d := Float.abs (gS[i]! - gBl[i]!)
          maxAbs := max maxAbs d
          maxRel := max maxRel (d / (Float.abs gS[i]! + 1.0e-12))
        IO.println s!"  C{C} {inH}×{inW} nF{nF} k{k}/s{s} h{hidden} A{A} (P={gS.size}): max abs = {maxAbs}   max rel = {maxRel}"
        (← IO.getStdout).flush
  | "bench-cnn-grad" :: _ => do
      -- scalar-FFI vs BLAS CNN minibatch gradient (im2col+GEMM), swept over hidden
      let N := 1024; let C := 3; let inH := 16; let inW := 16; let nF := 16; let k := 3; let s := 1; let A := 6
      IO.println s!"bench-cnn-grad — CNN+PPO minibatch gradient, N={N}, C{C} {inH}×{inW} nF{nF} k{k}/s{s}, {A} act; sweeping hidden"
      (← IO.getStdout).flush
      let mut rng : UInt64 := 0x55
      let mut obsB : Array Float := #[]; let mut acts : Array Float := #[]
      let mut advs : Array Float := #[]; let mut rets : Array Float := #[]; let mut olps : Array Float := #[]
      for _ in [0:N] do
        for _ in [0:C*inH*inW] do let (v, r) := randF (-1.0) 1.0 rng; rng := r; obsB := obsB.push v
        let (aw, r1) := rngNext rng; rng := r1; acts := acts.push (Float.ofNat (aw.toNat % A))
        let (av, r2) := randF (-1.0) 1.0 rng; rng := r2; advs := advs.push av
        let (rv, r3) := randF 0.0 2.0 rng; rng := r3; rets := rets.push rv
        olps := olps.push (-0.9)
      let oB := FloatArray.mk obsB; let aB := FloatArray.mk acts; let dB0 := FloatArray.mk advs
      let rB := FloatArray.mk rets; let lB := FloatArray.mk olps
      let u1 := USize.ofNat; let vf := 0.5; let ent := 0.01; let clip := 0.2
      for hidden in [128, 512] do
        let (p, _) := Puffer.RL.NNTrain.initCnn C inH inW nF k s hidden (A + 1) 0x99
        let params := Puffer.RL.NNTrain.flattenCnn p
        let reps := 5
        let timeIt := fun (useBlas : Bool) => do
          let _ := if useBlas then Puffer.Float.BLAS.cnnPPOGradBatchBlasFFI params oB aB dB0 rB lB (u1 N) (u1 C) (u1 inH) (u1 inW) (u1 nF) (u1 k) (u1 s) (u1 hidden) (u1 A) vf ent clip
                   else Puffer.Float.FFI.cnnPPOGradBatchFFI params oB aB dB0 rB lB (u1 N) (u1 C) (u1 inH) (u1 inW) (u1 nF) (u1 k) (u1 s) (u1 hidden) (u1 A) vf ent clip  -- warmup
          let mut cs := 0.0
          let t0 ← IO.monoNanosNow
          for kk in [0:reps] do
            let dK := dB0.set! 0 (Float.ofNat kk * 0.001)   -- perturb per rep ⇒ defeats CSE
            let g := if useBlas then Puffer.Float.BLAS.cnnPPOGradBatchBlasFFI params oB aB dK rB lB (u1 N) (u1 C) (u1 inH) (u1 inW) (u1 nF) (u1 k) (u1 s) (u1 hidden) (u1 A) vf ent clip
                     else Puffer.Float.FFI.cnnPPOGradBatchFFI params oB aB dK rB lB (u1 N) (u1 C) (u1 inH) (u1 inW) (u1 nF) (u1 k) (u1 s) (u1 hidden) (u1 A) vf ent clip
            cs := cs + g[0]!
          let t1 ← IO.monoNanosNow
          return (Float.ofNat ((t1 - t0) / reps) / 1.0e6, cs)
        let (msS, _) ← timeIt false
        let (msB, _) ← timeIt true
        IO.println s!"  hidden {hidden}: scalar-FFI {msS} ms  OpenBLAS {msB} ms  (gradient speedup {msS / (max msB 0.0001)}×)"
        (← IO.getStdout).flush
  | "verify-lstm-blas" :: _ => do
      -- batched BPTT gradient (over B sequences) vs the SUM of the scalar per-sequence
      -- kernel (itself 0.000000 vs Lean recPPOGradSeq). Terminals at t=2 in even sequences
      -- exercise the per-sequence reset + recurrent-Wh paths inside the batch.
      let B := 4; let T := 5; let H := 3; let D := 4; let A := 2; let O := A + 1
      let (p, _) := Puffer.RL.NNTrain.initRec D H O 0x5EED
      let mut rng : UInt64 := 0xBEEF
      let mut h0s : Array Float := #[]; let mut c0s : Array Float := #[]
      for _ in [0:B] do
        for _ in [0:H] do let (v, r) := randF (-0.3) 0.3 rng; rng := r; h0s := h0s.push v
      for _ in [0:B] do
        for _ in [0:H] do let (v, r) := randF (-0.3) 0.3 rng; rng := r; c0s := c0s.push v
      -- time-major arrays [(t*B+b)]
      let mut obsTM : Array Float := Array.replicate (T*B*D) 0.0
      let mut actTM : Array Float := Array.replicate (T*B) 0.0
      let mut advTM : Array Float := Array.replicate (T*B) 0.0
      let mut retTM : Array Float := Array.replicate (T*B) 0.0
      let mut oldTM : Array Float := Array.replicate (T*B) 0.0
      let mut trmTM : Array Float := Array.replicate (T*B) 0.0
      for b in [0:B] do
        for t in [0:T] do
          let tmIdx := t*B + b
          for d in [0:D] do let (v, r) := randF (-1.0) 1.0 rng; rng := r; obsTM := obsTM.set! (tmIdx*D+d) v
          let (aw, r1) := rngNext rng; rng := r1; actTM := actTM.set! tmIdx (Float.ofNat (aw.toNat % A))
          let (av, r2) := randF (-1.0) 1.0 rng; rng := r2; advTM := advTM.set! tmIdx av
          let (rv, r3) := randF 0.0 2.0 rng; rng := r3; retTM := retTM.set! tmIdx rv
          oldTM := oldTM.set! tmIdx (-0.8)
          trmTM := trmTM.set! tmIdx (if t == 2 && b % 2 == 0 then 1.0 else 0.0)
      let params := Puffer.RL.NNTrain.flattenRec p
      let u1 := USize.ofNat; let vf := 0.5; let ent := 0.01; let clip := 0.2
      let mk := FloatArray.mk
      let gB := Puffer.Float.BLAS.lstmPPOGradBatchBlasFFI params (mk obsTM) (mk actTM) (mk advTM)
                  (mk retTM) (mk oldTM) (mk trmTM) (mk h0s) (mk c0s) (u1 B) (u1 T) (u1 H) (u1 D) (u1 A) vf ent clip
      let mut gSum : Array Float := Array.replicate gB.size 0.0
      for b in [0:B] do
        let mut obsSeq : Array Float := #[]; let mut acts : Array Float := #[]
        let mut advs : Array Float := #[]; let mut rets : Array Float := #[]
        let mut olps : Array Float := #[]; let mut terms : Array Float := #[]
        for t in [0:T] do
          for d in [0:D] do obsSeq := obsSeq.push (obsTM[(t*B+b)*D+d]!)
          acts := acts.push (actTM[t*B+b]!); advs := advs.push (advTM[t*B+b]!)
          rets := rets.push (retTM[t*B+b]!); olps := olps.push (oldTM[t*B+b]!)
          terms := terms.push (trmTM[t*B+b]!)
        let h0b := (Array.range H).map (fun j => h0s[b*H+j]!)
        let c0b := (Array.range H).map (fun j => c0s[b*H+j]!)
        let gs := Puffer.Float.FFI.lstmPPOGradSeqFFI params (mk obsSeq) (mk acts) (mk advs) (mk rets)
                    (mk olps) (mk terms) (mk h0b) (mk c0b) (u1 T) (u1 H) (u1 D) (u1 A) vf ent clip
        for i in [0:gs.size] do gSum := gSum.set! i (gSum[i]! + gs[i]!)
      let mut maxAbs := 0.0; let mut maxRel := 0.0
      for i in [0:gB.size] do
        let d := Float.abs (gB[i]! - gSum[i]!)
        maxAbs := max maxAbs d
        maxRel := max maxRel (d / (Float.abs gSum[i]! + 1.0e-12))
      IO.println s!"verify-lstm-blas (batched BPTT grad, B={B} T={T} H={H} D={D} A={A}) BLAS-batch vs Σ scalar-per-seq:"
      IO.println s!"  max abs diff = {maxAbs}   max rel diff = {maxRel}   (P={gB.size}; tolerance, not bit-exact)"
  | "verify-lstm-grad-f32" :: _ => do
      -- f32-tier batched BPTT grad (cblas_sgemm) vs the f64-BLAS batched grad (cblas_dgemm) — a
      -- further tolerance step, not vs the bit-exact scalar oracle directly (that's verify-lstm-blas).
      let B := 4; let T := 5; let H := 3; let D := 4; let A := 2; let O := A + 1
      let (p, _) := Puffer.RL.NNTrain.initRec D H O 0x5EED
      let mut rng : UInt64 := 0xBEEF
      let mut h0s : Array Float := #[]; let mut c0s : Array Float := #[]
      for _ in [0:B] do
        for _ in [0:H] do let (v, r) := randF (-0.3) 0.3 rng; rng := r; h0s := h0s.push v
      for _ in [0:B] do
        for _ in [0:H] do let (v, r) := randF (-0.3) 0.3 rng; rng := r; c0s := c0s.push v
      let mut obsTM : Array Float := Array.replicate (T*B*D) 0.0
      let mut actTM : Array Float := Array.replicate (T*B) 0.0
      let mut advTM : Array Float := Array.replicate (T*B) 0.0
      let mut retTM : Array Float := Array.replicate (T*B) 0.0
      let mut oldTM : Array Float := Array.replicate (T*B) 0.0
      let mut trmTM : Array Float := Array.replicate (T*B) 0.0
      for b in [0:B] do
        for t in [0:T] do
          let tmIdx := t*B + b
          for d in [0:D] do let (v, r) := randF (-1.0) 1.0 rng; rng := r; obsTM := obsTM.set! (tmIdx*D+d) v
          let (aw, r1) := rngNext rng; rng := r1; actTM := actTM.set! tmIdx (Float.ofNat (aw.toNat % A))
          let (av, r2) := randF (-1.0) 1.0 rng; rng := r2; advTM := advTM.set! tmIdx av
          let (rv, r3) := randF 0.0 2.0 rng; rng := r3; retTM := retTM.set! tmIdx rv
          oldTM := oldTM.set! tmIdx (-0.8)
          trmTM := trmTM.set! tmIdx (if t == 2 && b % 2 == 0 then 1.0 else 0.0)
      let params := Puffer.RL.NNTrain.flattenRec p
      let u1 := USize.ofNat; let vf := 0.5; let ent := 0.01; let clip := 0.2
      let mk := FloatArray.mk
      let gB64 := Puffer.Float.BLAS.lstmPPOGradBatchBlasFFI params (mk obsTM) (mk actTM) (mk advTM)
                  (mk retTM) (mk oldTM) (mk trmTM) (mk h0s) (mk c0s) (u1 B) (u1 T) (u1 H) (u1 D) (u1 A) vf ent clip
      let gB32 := Puffer.Float.BLAS.lstmPPOGradBatchBlasF32FFI params (mk obsTM) (mk actTM) (mk advTM)
                  (mk retTM) (mk oldTM) (mk trmTM) (mk h0s) (mk c0s) (u1 B) (u1 T) (u1 H) (u1 D) (u1 A) vf ent clip
      let mut maxAbs := 0.0; let mut maxRel := 0.0
      for i in [0:gB64.size] do
        let d := Float.abs (gB32[i]! - gB64[i]!)
        maxAbs := max maxAbs d
        maxRel := max maxRel (d / (Float.abs gB64[i]! + 1.0e-12))
      IO.println s!"verify-lstm-grad-f32 (batched BPTT grad, B={B} T={T} H={H} D={D} A={A}) f32-BLAS vs f64-BLAS:"
      IO.println s!"  max abs diff = {maxAbs}   max rel diff = {maxRel}   (P={gB64.size}; tolerance, not bit-exact)"
  | "verify-lstm-grad-bf16" :: _ => do
      -- bf16 tensor-core BPTT grad (CUBLAS_COMPUTE_32F_FAST_16BF GEMMs) vs the f64-BLAS batched grad — a
      -- LOOSER tolerance step than f32 (bf16 has an 8-bit mantissa). Not vs the scalar oracle directly.
      let B := 4; let T := 5; let H := 3; let D := 4; let A := 2; let O := A + 1
      let (p, _) := Puffer.RL.NNTrain.initRec D H O 0x5EED
      let mut rng : UInt64 := 0xBEEF
      let mut h0s : Array Float := #[]; let mut c0s : Array Float := #[]
      for _ in [0:B] do
        for _ in [0:H] do let (v, r) := randF (-0.3) 0.3 rng; rng := r; h0s := h0s.push v
      for _ in [0:B] do
        for _ in [0:H] do let (v, r) := randF (-0.3) 0.3 rng; rng := r; c0s := c0s.push v
      let mut obsTM : Array Float := Array.replicate (T*B*D) 0.0
      let mut actTM : Array Float := Array.replicate (T*B) 0.0
      let mut advTM : Array Float := Array.replicate (T*B) 0.0
      let mut retTM : Array Float := Array.replicate (T*B) 0.0
      let mut oldTM : Array Float := Array.replicate (T*B) 0.0
      let mut trmTM : Array Float := Array.replicate (T*B) 0.0
      for b in [0:B] do
        for t in [0:T] do
          let tmIdx := t*B + b
          for d in [0:D] do let (v, r) := randF (-1.0) 1.0 rng; rng := r; obsTM := obsTM.set! (tmIdx*D+d) v
          let (aw, r1) := rngNext rng; rng := r1; actTM := actTM.set! tmIdx (Float.ofNat (aw.toNat % A))
          let (av, r2) := randF (-1.0) 1.0 rng; rng := r2; advTM := advTM.set! tmIdx av
          let (rv, r3) := randF 0.0 2.0 rng; rng := r3; retTM := retTM.set! tmIdx rv
          oldTM := oldTM.set! tmIdx (-0.8)
          trmTM := trmTM.set! tmIdx (if t == 2 && b % 2 == 0 then 1.0 else 0.0)
      let params := Puffer.RL.NNTrain.flattenRec p
      let u1 := USize.ofNat; let vf := 0.5; let ent := 0.01; let clip := 0.2
      let mk := FloatArray.mk
      let gB64 := Puffer.Float.BLAS.lstmPPOGradBatchBlasFFI params (mk obsTM) (mk actTM) (mk advTM)
                  (mk retTM) (mk oldTM) (mk trmTM) (mk h0s) (mk c0s) (u1 B) (u1 T) (u1 H) (u1 D) (u1 A) vf ent clip
      let gBf := Puffer.Float.BLAS.lstmPPOGradBatchBlasBf16FFI params (mk obsTM) (mk actTM) (mk advTM)
                  (mk retTM) (mk oldTM) (mk trmTM) (mk h0s) (mk c0s) (u1 B) (u1 T) (u1 H) (u1 D) (u1 A) vf ent clip
      let mut maxAbs := 0.0; let mut maxRel := 0.0
      for i in [0:gB64.size] do
        let d := Float.abs (gBf[i]! - gB64[i]!)
        maxAbs := max maxAbs d
        maxRel := max maxRel (d / (Float.abs gB64[i]! + 1.0e-12))
      IO.println s!"verify-lstm-grad-bf16 (batched BPTT grad, B={B} T={T} H={H} D={D} A={A}) bf16-tensor-core vs f64-BLAS:"
      IO.println s!"  max abs diff = {maxAbs}   max rel diff = {maxRel}   (P={gB64.size}; bf16 8-bit mantissa, looser tolerance)"
  | "verify-lstm-fwd-blas" :: _ => do
      -- lstmFwdStepBatchBlasFFI (cblas_dgemm) vs lstmFwdStepBatchFFI (naive scalar, bit-exact vs lstmCellF)
      let N := 5; let D := 4; let H := 3; let A := 2; let O := A + 1
      let (p, _) := Puffer.RL.NNTrain.initRec D H O 0x5EED
      let mut rng : UInt64 := 0xF00D
      let mut obs : Array Float := #[]; let mut hArr : Array Float := #[]; let mut cArr : Array Float := #[]
      for _ in [0:N*D] do let (v, r) := randF (-1.0) 1.0 rng; rng := r; obs := obs.push v
      for _ in [0:N*H] do let (v, r) := randF (-0.5) 0.5 rng; rng := r; hArr := hArr.push v
      for _ in [0:N*H] do let (v, r) := randF (-0.5) 0.5 rng; rng := r; cArr := cArr.push v
      let params := Puffer.RL.NNTrain.flattenRec p
      let u1 := USize.ofNat; let mk := FloatArray.mk
      let outNaive := Puffer.Float.FFI.lstmFwdStepBatchFFI params (mk obs) (mk hArr) (mk cArr) (u1 N) (u1 D) (u1 H) (u1 A)
      let outBlas := Puffer.Float.BLAS.lstmFwdStepBatchBlasFFI params (mk obs) (mk hArr) (mk cArr) (u1 N) (u1 D) (u1 H) (u1 A)
      let mut maxAbs := 0.0; let mut maxRel := 0.0
      for i in [0:outNaive.size] do
        let d := Float.abs (outBlas[i]! - outNaive[i]!)
        maxAbs := max maxAbs d
        maxRel := max maxRel (d / (Float.abs outNaive[i]! + 1.0e-12))
      IO.println s!"verify-lstm-fwd-blas (batched fwd step, N={N} D={D} H={H} A={A}) BLAS vs naive-scalar:"
      IO.println s!"  max abs diff = {maxAbs}   max rel diff = {maxRel}   (size={outNaive.size}; tolerance, not bit-exact)"
  | "verify-lstm-fwd-f32" :: _ => do
      -- f32-tier batched fwd step (cblas_sgemm) vs the f64-BLAS batched fwd step (cblas_dgemm)
      let N := 5; let D := 4; let H := 3; let A := 2; let O := A + 1
      let (p, _) := Puffer.RL.NNTrain.initRec D H O 0x5EED
      let mut rng : UInt64 := 0xF00D
      let mut obs : Array Float := #[]; let mut hArr : Array Float := #[]; let mut cArr : Array Float := #[]
      for _ in [0:N*D] do let (v, r) := randF (-1.0) 1.0 rng; rng := r; obs := obs.push v
      for _ in [0:N*H] do let (v, r) := randF (-0.5) 0.5 rng; rng := r; hArr := hArr.push v
      for _ in [0:N*H] do let (v, r) := randF (-0.5) 0.5 rng; rng := r; cArr := cArr.push v
      let params := Puffer.RL.NNTrain.flattenRec p
      let u1 := USize.ofNat; let mk := FloatArray.mk
      let out64 := Puffer.Float.BLAS.lstmFwdStepBatchBlasFFI params (mk obs) (mk hArr) (mk cArr) (u1 N) (u1 D) (u1 H) (u1 A)
      let out32 := Puffer.Float.BLAS.lstmFwdStepBatchBlasF32FFI params (mk obs) (mk hArr) (mk cArr) (u1 N) (u1 D) (u1 H) (u1 A)
      let mut maxAbs := 0.0; let mut maxRel := 0.0
      for i in [0:out64.size] do
        let d := Float.abs (out32[i]! - out64[i]!)
        maxAbs := max maxAbs d
        maxRel := max maxRel (d / (Float.abs out64[i]! + 1.0e-12))
      IO.println s!"verify-lstm-fwd-f32 (batched fwd step, N={N} D={D} H={H} A={A}) f32-BLAS vs f64-BLAS:"
      IO.println s!"  max abs diff = {maxAbs}   max rel diff = {maxRel}   (size={out64.size}; tolerance, not bit-exact)"
  | "bench-lstm-grad" :: _ => do
      -- B separate scalar per-sequence gradient calls (summed) vs ONE batched BLAS call
      let B := 64; let T := 64; let D := 32; let A := 6
      IO.println s!"bench-lstm-grad — LSTM BPTT gradient over B={B} sequences (T={T}, D={D}); sweeping hidden"
      (← IO.getStdout).flush
      let mut rng : UInt64 := 0x71
      let mut h0s : Array Float := #[]; let mut c0s : Array Float := #[]
      let u1 := USize.ofNat; let vf := 0.5; let ent := 0.01; let clip := 0.2; let mk := FloatArray.mk
      for hidden in [64, 256] do
        let H := hidden
        let (p, _) := Puffer.RL.NNTrain.initRec D H (A+1) 0x99
        let params := Puffer.RL.NNTrain.flattenRec p
        -- build time-major + per-seq arrays
        rng := 0x71
        h0s := Array.replicate (B*H) 0.0; c0s := Array.replicate (B*H) 0.0
        for i in [0:B*H] do let (v, r) := randF (-0.2) 0.2 rng; rng := r; h0s := h0s.set! i v
        let mut obsTM : Array Float := Array.replicate (T*B*D) 0.0
        let mut actTM : Array Float := Array.replicate (T*B) 0.0
        let mut advTM : Array Float := Array.replicate (T*B) 0.0
        let mut retTM : Array Float := Array.replicate (T*B) 0.0
        let mut oldTM : Array Float := Array.replicate (T*B) 0.0
        let trmTM : Array Float := Array.replicate (T*B) 0.0
        for i in [0:T*B*D] do let (v, r) := randF (-1.0) 1.0 rng; rng := r; obsTM := obsTM.set! i v
        for i in [0:T*B] do
          let (aw, r1) := rngNext rng; rng := r1; actTM := actTM.set! i (Float.ofNat (aw.toNat % A))
          let (av, r2) := randF (-1.0) 1.0 rng; rng := r2; advTM := advTM.set! i av
          let (rv, r3) := randF 0.0 2.0 rng; rng := r3; retTM := retTM.set! i rv
          oldTM := oldTM.set! i (-0.8)
        -- per-seq obs slices for the scalar loop (precompute)
        let obsSeqs := (Array.range B).map (fun b => Id.run do
          let mut a : Array Float := #[]
          for t in [0:T] do for d in [0:D] do a := a.push (obsTM[(t*B+b)*D+d]!)
          return a)
        let perSeq := fun (b : Nat) (col : Array Float) => (Array.range T).map (fun t => col[t*B+b]!)
        let oTM := mk obsTM; let aTM := mk actTM; let rTM := mk retTM; let lTM := mk oldTM; let tTM := mk trmTM
        let h0M := mk h0s; let c0M := mk c0s
        let reps := 3
        let timeScalar := fun (advTMx : Array Float) => Id.run do
          let mut gSum : Array Float := Array.replicate params.size 0.0
          for b in [0:B] do
            let gs := Puffer.Float.FFI.lstmPPOGradSeqFFI params (mk obsSeqs[b]!) (mk (perSeq b actTM))
                        (mk (perSeq b advTMx)) (mk (perSeq b retTM)) (mk (perSeq b oldTM)) (mk (perSeq b trmTM))
                        (mk ((Array.range H).map (fun j => h0s[b*H+j]!))) (mk ((Array.range H).map (fun j => c0s[b*H+j]!)))
                        (u1 T) (u1 H) (u1 D) (u1 A) vf ent clip
            for i in [0:gs.size] do gSum := gSum.set! i (gSum[i]! + gs[i]!)
          return gSum
        -- warmups
        let _ := timeScalar advTM
        let _ := Puffer.Float.BLAS.lstmPPOGradBatchBlasFFI params oTM aTM (mk advTM) rTM lTM tTM h0M c0M (u1 B) (u1 T) (u1 H) (u1 D) (u1 A) vf ent clip
        let t0 ← IO.monoNanosNow
        let mut cs0 := 0.0
        for kk in [0:reps] do let g := timeScalar (advTM.set! 0 (Float.ofNat kk * 0.001)); cs0 := cs0 + g[0]!
        let t1 ← IO.monoNanosNow
        let mut cs1 := 0.0
        for kk in [0:reps] do
          let g := Puffer.Float.BLAS.lstmPPOGradBatchBlasFFI params oTM aTM (mk (advTM.set! 0 (Float.ofNat kk * 0.001))) rTM lTM tTM h0M c0M (u1 B) (u1 T) (u1 H) (u1 D) (u1 A) vf ent clip
          cs1 := cs1 + g[0]!
        let t2 ← IO.monoNanosNow
        let msS := Float.ofNat ((t1 - t0) / reps) / 1.0e6
        let msB := Float.ofNat ((t2 - t1) / reps) / 1.0e6
        IO.println s!"  hidden {hidden}: {B}× scalar-per-seq {msS} ms  batched OpenBLAS {msB} ms  (speedup {msS / (max msB 0.0001)}×)"
        (← IO.getStdout).flush
  | "env-log" :: rest =>
      -- `puffer env-log <env> [num_envs] [steps]` — CPU-ONLY inspection of the plugin's optional log
      -- channel (`Puffer.Plugin.envLog*`): open the env, drive it with a RANDOM policy, then report the
      -- env's own PufferLib `Log` the way `static_vec_log` does (aggregate over copies ÷ n, then zero).
      -- Alongside it, the OLD terminal-flag reconstruction (sum of rewards between terminals) so the
      -- unit mismatch is visible — the 14 ocean envs that never raise a terminal read a permanent 0.0
      -- there while the log reports real episode statistics. Touches NO GPU: no policy, no CUDA call.
      match rest with
      | [] => IO.eprintln "Usage: puffer env-log <env> [num_envs] [steps]   (e.g. puffer env-log pong 64 6000)"
      | envName :: more =>
        let u := USize.ofNat
        let N := match more with | s :: _ => parseNat s 32 | _ => 32
        let T := match more with | _ :: s :: _ => parseNat s 2000 | _ => 2000
        let h ← Puffer.Plugin.envOpen envName (u N) 42 ""
        if h == 0 then IO.eprintln s!"env '{envName}' not found — run ./ocean/build.sh {envName}" else
        let nAgents := max 1 (Puffer.Plugin.envNumAgents h).toNat
        let K := max 1 (Puffer.Plugin.envNHeads h).toNat
        let cont := (Puffer.Plugin.envIsCont h).toNat == 1
        let sizes := Puffer.Plugin.envHeadSizes h
        let B := N * nAgents
        let fields := Puffer.Plugin.envLogFields h
        IO.println s!"env {envName}: {N} copies × {nAgents} agents, obs {(Puffer.Plugin.envObsDim h).toNat}, {K} head(s){if cont then " (continuous)" else ""}"
        IO.println s!"  log fields ({fields.size}): {if fields.isEmpty then "UNSUPPORTED (plugin exports no puffer_env_log*)" else String.intercalate ", " fields.toList}"
        let _ ← Puffer.Plugin.envReset h
        let mut rng : UInt64 := 0x243F6A8885A308D3
        let mut run : Array Float := Array.replicate B 0.0
        let mut reconSum := 0.0; let mut reconN : Nat := 0
        for _ in [0:T] do
          let mut acts : Array Float := Array.mkEmpty (B*K)
          for i in [0:B*K] do
            rng := rng * 6364136223846793005 + 1442695040888963407
            let x := Float.ofNat ((rng >>> 33).toNat) / 2147483648.0        -- [0,1)
            acts := acts.push (if cont then x * 2.0 - 1.0
                               else Float.ofNat (min (sizes[i % K]! - 1) (x * Float.ofNat sizes[i % K]!).toUInt64.toNat))
          let out ← Puffer.Plugin.envStep h (FloatArray.mk acts)
          let base := B * (Puffer.Plugin.envObsDim h).toNat
          for i in [0:B] do
            run := run.set! i (run[i]! + out.get! (base + i))
            if out.get! (base + B + i) > 0.5 then
              reconSum := reconSum + run[i]!; reconN := reconN + 1; run := run.set! i 0.0
        let kvs ← Puffer.Plugin.envLogPairs h
        if kvs.isEmpty then
          IO.println s!"  env log: nothing to report (unsupported, or no episode completed in {T} steps)"
        else
          for kv in kvs do IO.println s!"    {kv.1} = {kv.2}"
          -- exactly what the plugin trainers append to their progress line (`fmtEnvLog`), so this
          -- CPU-only command exercises the same formatting path the GPU trainers use
          IO.println s!"  (trainer line){Puffer.RL.NNTrain.fmtEnvLog kvs}"
        let kvs2 ← Puffer.Plugin.envLogPairs h            -- must be empty: the read above ZEROED the logs
        IO.println s!"  re-read after zeroing: {if kvs2.isEmpty then "empty (OK — logs were zeroed)" else "NOT ZEROED"}"
        IO.println s!"  terminal-flag reconstruction: {reconN} terminals, mean reward-between-terminals = {if reconN == 0 then 0.0 else reconSum / Float.ofNat reconN}"
        Puffer.Plugin.envClose h
  | "train" :: rest =>
      -- PufferLib CLI: `puffer train <env_name> [--flags]`. Config layers config/default.ini ←
      -- config/<env>.ini ← CLI (sectioned; --section.key flattens to leaf), exactly like PufferLib's
      -- load_config. The env plugin (ocean/<env>) is dlopen'd at runtime and trained through the C ABI;
      -- env-specific keys (frameskip, size, …) pass through to the env as its config string.
      match (parseFlags rest).1 with
      | none => IO.eprintln "Usage: puffer train <env_name> [--flags]   (e.g. puffer train breakout --total-timesteps 5_000_000)"; IO.Process.exit 1
      | some env =>
        -- STREAM 3 (CLI fidelity): reject genuinely unknown --flags before doing any work, like PufferLib's
        -- argparse. Namespaced overrides (--train.x / --env.x / …) and known trainer keys are accepted.
        if let some bad := firstUnknownFlag rest then
          IO.eprintln s!"puffer train: unknown flag --{bad}. Use a known flag (see `puffer help`), or an \
            explicit namespace: --train.<key>, --policy.<key>, --vec.<key>, or --env.<key> for env-specific keys."
          IO.Process.exit 1
        let cli := (parseFlags rest).2
        let flags ← loadConfigFlags env cli            -- config/default.ini ← config/<env>.ini ← CLI
        let cfg := configOf env flags
        let g := fun (k : String) => (flags.find? (fun kv => kv.1 == normKey k)).map (·.2)
        -- env config string for the ocean plugin (all resolved k=v; the env reads the keys it needs)
        -- Strip the `env.` namespace when handing the config to the PLUGIN: parseIni namespaces
        -- `[env]` keys as `env.<key>` so they cannot collide with vec/train keys of the same name
        -- (whisker_racer.ini has BOTH `[vec] num_envs` and `[env] num_envs`), but every ocean adapter
        -- looks up the BARE key (`cfg_int(cfg,"map-size",…)`). Without this the whole `[env]` INI
        -- section silently stopped reaching the envs — a regression the namespacing introduced.
        -- CLI `--env.<key>` already normalises to the bare key, so both forms now agree.
        let envCfg := String.intercalate "," (flags.map (fun kv =>
          (if kv.1.startsWith "env." then String.ofList (kv.1.toList.drop 4) else kv.1) ++ "=" ++ kv.2))
        -- numEnvs = env INSTANCES (--num-envs); for multi-agent envs the batch is numEnvs·num_agents.
        let numEnvsPreliminary := (g "num-envs").elim cfg.numEnvs (parseNat · cfg.numEnvs)
        -- peek the action structure to route single-discrete vs multi-discrete vs continuous, and the
        -- env's own agent count (needed below to resolve `total-agents`, PufferLib's own vec.total_agents
        -- ini key — until this fix it was silently DROPPED: `configOf` only ever looked up `--num-envs`,
        -- so `puffer train <env>` with no `--num-envs` override ran whatever tiny numEnvs the CLI defaults
        -- to, never the env's real tuned batch size, with no warning).
        let hPeek ← Puffer.Plugin.envOpen env (USize.ofNat numEnvsPreliminary) cfg.seed envCfg
        let nHeads := if hPeek == 0 then 1 else (Puffer.Plugin.envNHeads hPeek).toNat
        let isCont := hPeek != 0 && (Puffer.Plugin.envIsCont hPeek).toNat == 1
        let peekNAgents := if hPeek == 0 then 1 else max 1 (Puffer.Plugin.envNumAgents hPeek).toNat
        if hPeek != 0 then Puffer.Plugin.envClose hPeek
        -- Explicit --num-envs always wins. Otherwise, if the config gives `total-agents` (PufferLib's own
        -- vec.total_agents key — every shipped config/*.ini sets it), honor it: numEnvs = total_agents /
        -- agents-per-env-instance, rounded down, floored at 1. Falls back to the CLI's small smoke-test
        -- default only if NEITHER is present.
        let numEnvs :=
          if (g "num-envs").isSome then numEnvsPreliminary
          else match g "total-agents" with
            | some s => max 1 ((parseNat s (numEnvsPreliminary * peekNAgents)) / peekNAgents)
            | none => numEnvsPreliminary
        -- PufferLib's `[torch] network` selects the policy core (default MinGRU, recurrent). MinGRU/GRU/LSTM
        -- are recurrent (POMDP envs need this — a feed-forward MLP can't solve them); MLP is feed-forward.
        -- MULTI-DISCRETE envs now get the recurrent core too (`trainPluginEnvMinGRUMD`: same MinGRU stack,
        -- W = Σ head sizes logits, per-head softmax/sampling, joint log-prob) — before that they fell
        -- through to the single-hidden-layer MLP regardless of `network`, so `[policy] num_layers` was
        -- silently ignored for all eight of them. LSTM+MD and CONTINUOUS envs still fall through to their
        -- MLP paths (no recurrent Gaussian / recurrent-LSTM-MD head built); `--torch.network MLP` keeps
        -- every env on its feed-forward path.
        let net := cfg.network
        let isLSTM := net == "LSTM" || net == "lstm"
        let isRecurrent := net == "MinGRU" || net == "MinGru" || net == "mingru" || net == "GRU"
                        || net == "gru" || isLSTM
        -- PufferLib's `anneal_lr = 0/False` (robocode, continuous) holds LR CONSTANT at the base value.
        -- cosineLr with minLrRatio = 1.0 is exactly that constant, so route it through the existing
        -- minLrRatio plumbing rather than threading a new bool into every trainer signature.
        let effMinLr := if cfg.annealLr then cfg.minLrRatio else 1.0
        -- PufferLib's `puffer train` always renders its live dashboard (verbose=True) — there is no
        -- flag for it, so we match that: dashboard by DEFAULT. PUFFER_PLAIN_LOG=1 is a non-CLI escape
        -- hatch (env var, so the commandline surface still matches PufferLib) that our verification
        -- tooling (tools/env_sweep.sh, tools/compare.py) sets to get the machine-parseable lines.
        let plainLog := (← IO.getEnv "PUFFER_PLAIN_LOG").isSome
        -- wandb (PufferLib `--wandb`): spawn the live tracker daemon BEFORE training so the shared
        -- dashboard `redraw` streams it the same metric dict each 0.6s tick. `config=` the resolved
        -- flag map, as PufferLib passes `config=args` to `wandb.init`.
        if cfg.wandb then
          let cfgJson := "{" ++ String.intercalate "," (flags.map (fun kv =>
            "\"" ++ Puffer.RL.Wandb.escJson kv.1 ++ "\":\"" ++ Puffer.RL.Wandb.escJson kv.2 ++ "\"")) ++ "}"
          Puffer.RL.Wandb.start cfg.wandbProject cfg.wandbGroup cfg.wandbTag cfgJson
        -- trainer derives updates from total_timesteps / (batch·horizon) once it knows num_agents.
        if isRecurrent && !isCont && nHeads > 1 && !isLSTM then
          trainPluginEnvMinGRUMD env envCfg
            (hidden := cfg.hiddenSize) (numLayers := cfg.numLayers) (numEnvs := numEnvs) (horizon := cfg.horizon)
            (totalTimesteps := cfg.totalTimesteps) (epochs := cfg.epochs) (numMB := cfg.numMB) (minibatchSize := cfg.minibatchSize)
            (lr := cfg.learningRate) (wd := 0.0) (mu := cfg.beta1) (eps := cfg.eps)
            (gamma := cfg.gamma) (lam := cfg.gaeLambda) (vfCoef := cfg.vfCoef) (entCoef := cfg.entCoef)
            (clipEps := cfg.clipCoef) (vfClip := cfg.vfClipCoef) (maxGradNorm := cfg.maxGradNorm)
            (replayRatio := cfg.replayRatio) (minLrRatio := effMinLr)
            (rhoClip := cfg.vtraceRhoClip) (cClip := cfg.vtraceCClip)
            (prioAlpha := cfg.prioAlpha) (prioBeta0 := cfg.prioBeta0)
            (annealEntCoef := cfg.annealEntCoef) (minEntCoefRatio := cfg.minEntCoefRatio) (logDash := !plainLog)
            (checkpointInterval := cfg.checkpointInterval) (loadPath := cfg.loadPath) (seed := cfg.seed)
        else if isRecurrent && !isCont && nHeads == 1 then
          if net == "LSTM" || net == "lstm" then
            trainPluginEnvRec env envCfg
              (hidden := cfg.hiddenSize) (numEnvs := numEnvs) (horizon := cfg.horizon) (totalTimesteps := cfg.totalTimesteps)
              (epochs := cfg.epochs)
              (lr := cfg.learningRate) (gamma := cfg.gamma) (lam := cfg.gaeLambda) (vfCoef := cfg.vfCoef)
              (entCoef := cfg.entCoef) (clipEps := cfg.clipCoef) (maxGradNorm := cfg.maxGradNorm) (minLrRatio := effMinLr)
              (logDash := !plainLog) (seed := cfg.seed)
          else
            trainPluginEnvMinGRU env envCfg
              (hidden := cfg.hiddenSize) (numLayers := cfg.numLayers) (numEnvs := numEnvs) (horizon := cfg.horizon)
              (totalTimesteps := cfg.totalTimesteps) (epochs := cfg.epochs) (numMB := cfg.numMB) (minibatchSize := cfg.minibatchSize)
              (lr := cfg.learningRate) (wd := 0.0) (mu := cfg.beta1) (eps := cfg.eps)
              (gamma := cfg.gamma) (lam := cfg.gaeLambda) (vfCoef := cfg.vfCoef) (entCoef := cfg.entCoef)
              (clipEps := cfg.clipCoef) (vfClip := cfg.vfClipCoef) (maxGradNorm := cfg.maxGradNorm)
              (replayRatio := cfg.replayRatio) (minLrRatio := effMinLr)
              (rhoClip := cfg.vtraceRhoClip) (cClip := cfg.vtraceCClip)
              (prioAlpha := cfg.prioAlpha) (prioBeta0 := cfg.prioBeta0)
              (annealEntCoef := cfg.annealEntCoef) (minEntCoefRatio := cfg.minEntCoefRatio) (logDash := !plainLog)
              (checkpointInterval := cfg.checkpointInterval) (loadPath := cfg.loadPath) (seed := cfg.seed)
        else if isCont then
          trainPluginEnvCont env envCfg
            (hidden := cfg.hiddenSize) (numEnvs := numEnvs) (horizon := cfg.horizon) (totalTimesteps := cfg.totalTimesteps)
            (epochs := cfg.epochs) (numMB := cfg.numMB)
            (lr := cfg.learningRate) (wd := 0.0) (mu := cfg.beta1) (eps := cfg.eps)
            (gamma := cfg.gamma) (lam := cfg.gaeLambda) (vfCoef := cfg.vfCoef)
            (entCoef := cfg.entCoef) (clipEps := cfg.clipCoef)
            (vfClip := cfg.vfClipCoef) (maxGradNorm := cfg.maxGradNorm)
            (bf16 := 1) (minLrRatio := effMinLr) (logDash := !plainLog) (seed := cfg.seed)
        else if nHeads > 1 then
          trainPluginEnvMD env envCfg
            (hidden := cfg.hiddenSize) (numEnvs := numEnvs) (horizon := cfg.horizon) (totalTimesteps := cfg.totalTimesteps)
            (epochs := cfg.epochs) (numMB := cfg.numMB)
            (lr := cfg.learningRate) (wd := 0.0) (mu := cfg.beta1) (eps := cfg.eps)
            (gamma := cfg.gamma) (lam := cfg.gaeLambda) (vfCoef := cfg.vfCoef)
            (entCoef := cfg.entCoef) (clipEps := cfg.clipCoef)
            (vfClip := cfg.vfClipCoef) (maxGradNorm := cfg.maxGradNorm)
            (bf16 := 1) (minLrRatio := effMinLr) (logDash := !plainLog) (seed := cfg.seed)
        else
          trainPluginEnv env envCfg
            (hidden := cfg.hiddenSize) (numEnvs := numEnvs) (horizon := cfg.horizon) (totalTimesteps := cfg.totalTimesteps)
            (epochs := cfg.epochs) (numMB := cfg.numMB)
            (lr := cfg.learningRate) (wd := 0.0) (mu := cfg.beta1) (eps := cfg.eps)
            (gamma := cfg.gamma) (lam := cfg.gaeLambda) (vfCoef := cfg.vfCoef)
            (entCoef := cfg.entCoef) (clipEps := cfg.clipCoef)
            (vfClip := cfg.vfClipCoef) (maxGradNorm := cfg.maxGradNorm)
            (bf16 := 1) (minLrRatio := effMinLr) (logDash := !plainLog) (seed := cfg.seed)
        -- wandb: upload this run's final checkpoint as Artifact(run_id, type='model') and close the run
        -- (PufferLib's end-of-_train behavior). No checkpoint written ⇒ nothing to upload, just finish.
        if cfg.wandb then
          Puffer.RL.Wandb.finish (← runFinalCheckpoint env cfg.seed)
  | "verify-mingru-grad" :: _ =>
      let (a, r) := Puffer.RL.NNTrain.mingruGradCheck
      IO.println s!"mingru-grad AD-vs-finite-diff:  max|Δ| = {a}   max relΔ = {r}"
  | "verify-mingru-kernel" :: _ =>
      IO.println s!"mingru native-C-kernel vs AD oracle:  max|Δ| = {Puffer.RL.NNTrain.mingruKernelCheck}"
  | "verify-mingru-grad-gpu" :: _ => do
      -- GPU batched MinGRU BPTT PPO gradient vs the C gradient, at a LARGER config and B=2 (tests both size
      -- and batching: GPU sums over the batch, so compare vs the SUM of the C gradient over the 2 sequences).
      let D := 5; let H := 32; let L := 3; let A := 4; let T := 7; let Bn := 2
      let mk := FloatArray.mk; let u := USize.ofNat
      let (w, rng0) := Puffer.RL.NNTrain.initMinGRU D H L A 0x6A5D
      let mut rng := rng0
      -- per-sequence random data, and packed [T][B][·] for the GPU
      let mut seqObs : Array (Array Float) := #[]; let mut seqAct : Array (Array Float) := #[]
      let mut seqAdv : Array (Array Float) := #[]; let mut seqRet : Array (Array Float) := #[]
      let mut seqOld : Array (Array Float) := #[]; let mut seqTrm : Array (Array Float) := #[]; let mut seqOv : Array (Array Float) := #[]
      for _ in [0:Bn] do
        let mut o:Array Float:=#[]; let mut ac:Array Float:=#[]; let mut ad:Array Float:=#[]
        let mut re:Array Float:=#[]; let mut ol:Array Float:=#[]; let mut tm:Array Float:=#[]; let mut ov:Array Float:=#[]
        for t in [0:T] do
          for _ in [0:D] do let (v,r) := randF (-1.0) 1.0 rng; rng := r; o := o.push v
          let (aw,r1) := rngNext rng; rng := r1; ac := ac.push (Float.ofNat (aw.toNat % A))
          let (av,r2) := randF (-1.0) 1.0 rng; rng := r2; ad := ad.push av
          let (rv,r3) := randF 0.0 2.0 rng; rng := r3; re := re.push rv
          ol := ol.push (-1.0)
          tm := tm.push (if t == 3 then 1.0 else 0.0)   -- a mid-sequence terminal (tests the reset + dOnext gate)
          let (vv,r4) := randF (-0.5) 0.5 rng; rng := r4; ov := ov.push vv
        seqObs:=seqObs.push o; seqAct:=seqAct.push ac; seqAdv:=seqAdv.push ad; seqRet:=seqRet.push re
        seqOld:=seqOld.push ol; seqTrm:=seqTrm.push tm; seqOv:=seqOv.push ov
      -- C gradient = sum over the 2 sequences
      let P := (Puffer.RL.NNTrain.flattenMG w).size
      let mut gCsum : Array Float := Array.replicate P 0.0
      for b in [0:Bn] do
        let traj : Array Puffer.RL.NNTrain.Transition := (Array.range T).map (fun t =>
          { obs := (Array.range D).map (fun j => (seqObs[b]!)[t*D+j]!), action := ((seqAct[b]!)[t]!).toUInt64.toNat,
            reward := 0.0, value := (seqOv[b]!)[t]!, oldLogp := (seqOld[b]!)[t]!, terminal := (seqTrm[b]!)[t]! > 0.5 })
        let gC := Puffer.RL.NNTrain.mingruGradSeqFFI w traj (seqAdv[b]!) (seqRet[b]!) (seqOv[b]!) A 0.5 0.01 0.2 0.2
        gCsum := (Array.range P).map (fun i => gCsum[i]! + gC[i]!)
      -- GPU batched: pack [T][B][·]
      let pk := fun (seq : Array (Array Float)) (w : Nat) =>
        mk ((Array.range (T*Bn*w)).map (fun i => let j := i % w; let tb := i / w; let b := tb % Bn; let t := tb / Bn; (seq[b]!)[t*w+j]!))
      let pk1 := fun (seq : Array (Array Float)) =>
        mk ((Array.range (T*Bn)).map (fun i => let b := i % Bn; let t := i / Bn; (seq[b]!)[t]!))
      let seqs := #[seqAct, seqAdv, seqRet, seqOld, seqTrm, seqOv]   -- packed [6·T·Bn] = [act|adv|ret|old|term|ov]
      let scal := mk ((Array.range (6*T*Bn)).map (fun i =>
        let blk := i / (T*Bn); let r := i % (T*Bn); let b := r % Bn; let t := r / Bn; ((seqs[blk]!)[b]!)[t]!))
      let gG ← Puffer.Float.CUDA.cudaMinGRUPpoGradFFI (Puffer.RL.NNTrain.flattenMG w)
                  (pk seqObs D) scal
                  (u Bn) (u T) (u H) (u D) (u L) (u A) (u 1) (FloatArray.mk #[])
                  0.5 0.01 0.2 0.2 (FloatArray.mk #[]) (FloatArray.mk #[])   -- empty segIdx+mbPrio → host mode
      -- precision-aware check: the CUDA path instantiates the SAME operation graph at the configured
      -- precision, so it is verified against the f64 oracle at that precision's tolerance:
      --   PUFFER_MG_BF16=0        → strict f32 (rel 2e-2, floor 1e-4; measures ~1e-5)
      --   default                 → f32 storage + FAST_16BF tensor-core compute (rel 5e-2; measures ~2.5e-2)
      --   PUFFER_MG_PREC=bf16     → bf16 storage: mixed |Δ| ≤ atol+rtol·|ref| (atol 2e-2, rtol 5e-2), score ≤ 1
      let prec := (← IO.getEnv "PUFFER_MG_PREC").getD "f32"
      let isBf := prec.startsWith "b" || prec.startsWith "B"   -- match mg_bf16store()'s case-insensitive gate
      let tcCompute := ((← IO.getEnv "PUFFER_MG_BF16").getD "1") != "0"
      let mut maxRel := 0.0; let mut maxAbs := 0.0; let mut maxMix := 0.0
      for i in [0:P] do
        let dd := Float.abs (gG[i]! - gCsum[i]!); maxAbs := max maxAbs dd
        maxRel := max maxRel (dd / (Float.abs gCsum[i]! + 1.0e-4))
        maxMix := max maxMix (dd / (2.0e-2 + 5.0e-2 * Float.abs gCsum[i]!))
      IO.println s!"verify-mingru-grad-gpu (GPU BPTT vs C, B={Bn}, T={T}, D={D}→enc{H}→MinGRU×{L}→{A}, P={P}, prec={if isBf then "bf16" else if tcCompute then "f32+FAST_16BF" else "f32"}):"
      if isBf then
        IO.println s!"  max abs = {maxAbs}   mixed-tol score = {maxMix}   ({if maxMix < 1.0 then "ok ✓ (bf16 GPU vs f64 C, atol 2e-2 rtol 5e-2)" else "CHECK — bf16 path broken"})"
      else if tcCompute then
        IO.println s!"  max abs = {maxAbs}   max rel = {maxRel}   ({if maxRel < 5.0e-2 then "ok ✓ (f32-storage/FAST_16BF-compute vs f64 C)" else "CHECK — batching/size bug"})"
      else
        IO.println s!"  max abs = {maxAbs}   max rel = {maxRel}   ({if maxRel < 2.0e-2 then "ok ✓ (f32 GPU vs f64 C)" else "CHECK — batching/size bug"})"
  | "verify-mingru-md-grad" :: _ =>
      -- MULTI-DISCRETE recurrent head, AD-BPTT vs finite differences (pure CPU, no GPU): validates the
      -- per-head softmax / joint log-prob / joint-ratio clip / summed-entropy gradient the GPU kernel
      -- `k_mg_ppo_b_md` implements.
      let (a, r) := Puffer.RL.NNTrain.mingruMDGradCheck
      IO.println s!"verify-mingru-md-grad (multi-discrete MinGRU BPTT, AD vs finite-diff, K=2 heads [3,4] ⇒ W=7 logits):"
      -- verdict on the ABSOLUTE difference: the gradient entries here are O(1), so central differences at
      -- eps=1e-5 land at ~1e-9; a real head/offset bug is O(0.1–10). (max relΔ is reported too but is
      -- dominated by entries whose finite difference is ~0.)
      IO.println s!"  max|Δ| = {a}   max relΔ = {r}   ({if a < 1.0e-4 then "ok ✓" else "CHECK — multi-discrete head/gradient bug"})"
  | "verify-mingru-md-grad-gpu" :: _ => do
      -- GPU MULTI-DISCRETE MinGRU BPTT (k_mg_ppo_b_md) vs the Lean AD-BPTT oracle (mingruGradSeqMD).
      -- B=2 sequences: the GPU sums over the batch, so compare against the SUM of the per-sequence oracle.
      let D := 4; let H := 16; let L := 2; let T := 5; let Bn := 2
      let headSizes : Array Nat := #[3, 4]
      let K := headSizes.size; let Wl := headSizes.foldl (·+·) 0
      let mk := FloatArray.mk; let u := USize.ofNat
      let (w, rng0) := Puffer.RL.NNTrain.initMinGRU D H L Wl 0x6A5D
      let mut rng := rng0
      let mut obsB : Array (Array (Array Float)) := #[]      -- [B][T][D]
      let mut actB : Array (Array (Array Nat)) := #[]        -- [B][T][K]
      let mut advB : Array (Array Float) := #[]; let mut retB : Array (Array Float) := #[]
      let mut oldB : Array (Array Float) := #[]; let mut trmB : Array (Array Bool) := #[]
      let mut ovB  : Array (Array Float) := #[]
      for _ in [0:Bn] do
        let mut o : Array (Array Float) := #[]; let mut ac : Array (Array Nat) := #[]
        let mut ad : Array Float := #[]; let mut re : Array Float := #[]
        let mut ol : Array Float := #[]; let mut tm : Array Bool := #[]; let mut ov : Array Float := #[]
        for t in [0:T] do
          let mut row : Array Float := #[]
          for _ in [0:D] do let (v,r) := randF (-1.0) 1.0 rng; rng := r; row := row.push v
          o := o.push row
          let mut ah : Array Nat := #[]
          for hh in [0:K] do
            let (aw,r1) := rngNext rng; rng := r1; ah := ah.push (aw.toNat % headSizes[hh]!)
          ac := ac.push ah
          let (av,r2) := randF (-1.0) 1.0 rng; rng := r2; ad := ad.push av
          let (rv,r3) := randF 0.0 2.0 rng; rng := r3; re := re.push rv
          ol := ol.push (-2.5)                                -- ≈ the init joint log-prob ⇒ ratio ≈ 1 (unclipped)
          tm := tm.push (t == 2)                              -- mid-sequence terminal (reset + dOnext gate)
          let (vv,r4) := randF (-0.5) 0.5 rng; rng := r4; ov := ov.push vv
        obsB := obsB.push o; actB := actB.push ac; advB := advB.push ad; retB := retB.push re
        oldB := oldB.push ol; trmB := trmB.push tm; ovB := ovB.push ov
      let P := (Puffer.RL.NNTrain.flattenMG w).size
      let mut gRef : Array Float := Array.replicate P 0.0
      for b in [0:Bn] do
        let gb := Puffer.RL.NNTrain.flattenMG (Puffer.RL.NNTrain.mingruGradSeqMD w obsB[b]! actB[b]!
                    oldB[b]! advB[b]! retB[b]! ovB[b]! trmB[b]! headSizes 0.5 0.01 0.2 0.2)
        gRef := (Array.range P).map (fun i => gRef[i]! + gb[i]!)
      -- GPU host-scalar mode: obs packed [T][B][D]; scal = [act(K·T·B, row t·B+b then head); adv; ret; old; term; ov]
      let obsPk := mk ((Array.range (T*Bn*D)).map (fun i =>
        let j := i % D; let tb := i / D; let b := tb % Bn; let t := tb / Bn; ((obsB[b]!)[t]!)[j]!))
      let nbs := T*Bn
      let col := fun (f : Nat → Nat → Float) => (Array.range nbs).map (fun r => f (r % Bn) (r / Bn))
      let scal := mk (
        ((Array.range (K*nbs)).map (fun i =>
            let hh := i % K; let r := i / K; Float.ofNat (((actB[r % Bn]!)[r / Bn]!)[hh]!)))
        ++ col (fun b t => (advB[b]!)[t]!)
        ++ col (fun b t => (retB[b]!)[t]!)
        ++ col (fun b t => (oldB[b]!)[t]!)
        ++ col (fun b t => if (trmB[b]!)[t]! then 1.0 else 0.0)
        ++ col (fun b t => (ovB[b]!)[t]!))
      let gG ← Puffer.Float.CUDA.cudaMinGRUPpoGradFFI (Puffer.RL.NNTrain.flattenMG w) obsPk scal
                  (u Bn) (u T) (u H) (u D) (u L) (u Wl) (u K) (mk (headSizes.map Float.ofNat))
                  0.5 0.01 0.2 0.2 (mk #[]) (mk #[])   -- empty segIdx+mbPrio → host mode
      let tcCompute := ((← IO.getEnv "PUFFER_MG_BF16").getD "1") != "0"
      -- Relative error is only meaningful where the reference component is non-negligible. The old
      -- form `dd / (|gRef| + 1e-4)` used a FIXED floor: on this problem the largest deviation sits on
      -- a component of magnitude ~5e-3, so under the bf16 tier it reported 0.325 relative from an
      -- absolute error of 0.0015 — pure tensor-core noise on a near-zero entry, read as a 30% bug.
      -- Now: scale the floor to the gradient's own magnitude, take the relative max only over
      -- components above it, and keep an ABSOLUTE guard (also scaled to that magnitude) so a genuine
      -- error hiding among the skipped components still fails the check.
      let mut gmax := 0.0
      for i in [0:P] do gmax := max gmax (Float.abs gRef[i]!)
      -- Floor at 1% of the gradient's largest component under the tensor-core tier, 0.1% in f32.
      -- bf16 carries an 8-bit mantissa (~0.4% relative) BEFORE any GEMM accumulation, so a component
      -- at a fraction of a percent of the max is a small difference of large numbers and has lost
      -- most of its significant bits — 33% relative error there is catastrophic cancellation, a
      -- property of the tier, not a gradient bug. The absolute guard below is what still catches
      -- real errors, wherever they sit.
      let magFloor := (if tcCompute then 1.0e-2 else 1.0e-3) * gmax
      let mut maxRel := 0.0; let mut maxAbs := 0.0; let mut nSkip := 0
      for i in [0:P] do
        let dd := Float.abs (gG[i]! - gRef[i]!); maxAbs := max maxAbs dd
        if Float.abs gRef[i]! >= magFloor then maxRel := max maxRel (dd / Float.abs gRef[i]!)
        else nSkip := nSkip + 1
      IO.println s!"verify-mingru-md-grad-gpu (GPU MD BPTT vs Lean AD oracle, B={Bn}, T={T}, D={D}→enc{H}→MinGRU×{L}→{K} heads {headSizes} (W={Wl}), P={P}, prec={if tcCompute then "f32+FAST_16BF" else "f32"}):"
      -- Tolerance calibrated by MEASUREMENT against the single-discrete path (verify-mingru-grad-gpu)
      -- on the same tiers, rather than guessed. On the identical metric SD reads f32 1.1e-5 / bf16
      -- 2.46e-2 (a ~2200x tier degradation) while MD reads f32 6.4e-5 / bf16 0.325 (~5100x): MD is
      -- ~2.3x more bf16-sensitive, which is what summing K per-head log-probs into the joint before
      -- forming the PPO ratio predicts. SD passes at ~49% of its 5e-2 bar; giving MD the same margin
      -- means ~2.4x that bar. So the tensor-core bar is 1.2e-1 — a measured factor, not a nudge —
      -- and the f32 bar stays tight at 2e-2, where MD sits three orders below it.
      let tol := if tcCompute then 1.2e-1 else 2.0e-2
      let absTol := (if tcCompute then 5.0e-2 else 2.0e-2) * gmax   -- abs guard stays at the SD bar        -- an absolute error this large is a real bug wherever it sits
      let pass := maxRel < tol && maxAbs <= absTol
      IO.println s!"  max abs = {maxAbs} (tol {absTol})   max rel = {maxRel} over |g|≥{magFloor} ({nSkip}/{P} below floor)   ({if pass then "ok ✓" else "CHECK — multi-discrete head/gradient bug"})"
  | "verify-mingru-step-gpu" :: _ => do
      -- GPU batched MinGRU forward step (cudaMinGRUStepFFI) vs the Lean Net.MinGRU.stepForward oracle.
      -- f32 gemm32 vs f64 Lean ⇒ tolerance; confirms encoder→layers→decoder + the minGRU cell + state carry.
      let D := 5; let H := 16; let L := 3; let A := 4; let O := A + 1; let N := 3   -- H=16: WMMA-eligible
      let mk := FloatArray.mk; let u := USize.ofNat
      let (w, rng0) := Puffer.RL.NNTrain.initMinGRU D H L A 0x51E9
      let mut rng := rng0
      let mut obs : Array Float := #[]; let mut stateFlat : Array Float := #[]
      for _ in [0:N*D] do let (v,r) := randF (-1.0) 1.0 rng; rng := r; obs := obs.push v
      for _ in [0:N*L*H] do let (v,r) := randF (-0.5) 0.5 rng; rng := r; stateFlat := stateFlat.push v
      let params := Puffer.RL.NNTrain.flattenMG w
      let gpu := Puffer.Float.CUDA.cudaMinGRUStepFFI params (mk obs) (mk stateFlat) (u N) (u D) (u H) (u L) (u A) 0
      -- Lean per-env oracle
      let mut maxRel := 0.0
      for n in [0:N] do
        let obsN := (Array.range D).map (fun j => obs[n*D+j]!)
        let stateN := (Array.range L).map (fun l => (Array.range H).map (fun j => stateFlat[n*L*H+l*H+j]!))
        let (logitsL, valueL, newStateL) := Puffer.Net.MinGRU.stepForward w H obsN stateN
        for k in [0:A] do
          let dd := Float.abs (gpu[n*O+k]! - logitsL[k]!); maxRel := max maxRel (dd / (Float.abs logitsL[k]! + 1.0e-4))
        maxRel := max maxRel (Float.abs (gpu[n*O+A]! - valueL) / (Float.abs valueL + 1.0e-4))
        for l in [0:L] do for j in [0:H] do
          let gs := gpu[N*O + n*L*H + l*H + j]!
          maxRel := max maxRel (Float.abs (gs - (newStateL[l]!)[j]!) / (Float.abs (newStateL[l]!)[j]! + 1.0e-4))
      IO.println s!"verify-mingru-step-gpu (GPU forward vs Lean stepForward, N={N}, D={D}→enc{H}→MinGRU×{L}→{A}):"
      let wm := (← IO.getEnv "PUFFER_MG_WMMA").getD "1" != "0"   -- matches mg_wmma()'s default-ON gate
      let wpv := (← IO.getEnv "PUFFER_MG_WPREC").getD ""
      -- match the C gate exactly: case-insensitive first char, AND the tier only runs on the WMMA path
      let bfw := (wpv.startsWith "b" || wpv.startsWith "B") && wm
      let tol := if bfw then 3.0e-2 else 1.0e-2   -- bf16: 8-bit mantissa (PufferLib's own forward tier)
      let lbl := if bfw then "ok ✓ (bf16-WMMA GPU vs f64 Lean — PufferLib's forward tier)"
                 else if wm then "ok ✓ (tf32-WMMA GPU vs f64 Lean)" else "ok ✓ (f32 GPU vs f64 Lean)"
      IO.println s!"  max relative error = {maxRel}   ({if maxRel < tol then lbl else "CHECK"})"
  | "eval" :: rest =>
      -- STREAM 3: headless evaluation of a saved checkpoint. `puffer eval <env> [--load <path>] [flags]`.
      -- Loads --load (or the env's latest checkpoint under checkpoints/<env>/), then runs rollout-only steps
      -- under the loaded MinGRU policy (no gradient) and prints the mean episode_return + env `Log`.
      let some env := (parseFlags rest).1
        | IO.eprintln "Usage: puffer eval <env_name> [--load <checkpoint>] [flags]" *> IO.Process.exit 1
      if let some bad := firstUnknownFlag rest then
        IO.eprintln s!"puffer eval: unknown flag --{bad}. See `puffer help`, or namespace it (--train.<key> / --env.<key>)."
        IO.Process.exit 1
      let cli := (parseFlags rest).2
      let flags ← loadConfigFlags env cli
      let cfg := configOf env flags
      let g := fun (k : String) => (flags.find? (fun kv => kv.1 == normKey k)).map (·.2)
      let envCfg := String.intercalate "," (flags.map (fun kv =>
        (if kv.1.startsWith "env." then String.ofList (kv.1.toList.drop 4) else kv.1) ++ "=" ++ kv.2))
      -- resolve the checkpoint: --load wins, else the env's latest under checkpoints/<env>/
      let ckpt ← match cfg.loadPath with
        | some p => pure (some p)
        | none   => Puffer.RL.NNTrain.findLatestCheckpoint env
      let some path := ckpt
        | IO.eprintln s!"puffer eval: no checkpoint for '{env}' — pass --load <path>, or `puffer train {env}` first (writes checkpoints/{env}/…)" *> IO.Process.exit 1
      let some wLoaded ← Puffer.RL.NNTrain.loadPolicyCheckpoint path
        | IO.eprintln s!"puffer eval: checkpoint '{path}' not found" *> IO.Process.exit 1
      -- resolve numEnvs the same way `train` does (explicit --num-envs wins, else vec.total-agents/agents)
      let numEnvsPreliminary := (g "num-envs").elim cfg.numEnvs (parseNat · cfg.numEnvs)
      let hPeek ← Puffer.Plugin.envOpen env (USize.ofNat numEnvsPreliminary) cfg.seed envCfg
      let isCont := hPeek != 0 && (Puffer.Plugin.envIsCont hPeek).toNat == 1
      let peekNAgents := if hPeek == 0 then 1 else max 1 (Puffer.Plugin.envNumAgents hPeek).toNat
      if hPeek != 0 then Puffer.Plugin.envClose hPeek
      let numEnvsRaw :=
        if (g "num-envs").isSome then numEnvsPreliminary
        else match g "total-agents" with
          | some s => max 1 ((parseNat s (numEnvsPreliminary * peekNAgents)) / peekNAgents)
          | none   => numEnvsPreliminary
      -- eval samples per-env on the CPU each step, so a 4096-wide run is minutes of Lean softmax loops.
      -- A headless score needs only a few dozen envs (plenty of episodes) — cap it so eval returns fast.
      let numEnvs := min numEnvsRaw 64
      if isCont then
        IO.eprintln s!"puffer eval: '{env}' is continuous-action — eval currently supports the MinGRU discrete policy only"
        IO.Process.exit 1
      -- eval step budget per env: default 2000, or total-timesteps/agents (capped) when given
      let evalSteps :=
        match g "total-timesteps" with
        | some s => min 5000 (max 1 ((parseNat s (2000 * numEnvs * peekNAgents)) / max 1 (numEnvs * peekNAgents)))
        | none   => 2000
      IO.println s!"puffer eval [{env}]: loaded {path} ({wLoaded.size} params), {numEnvs} env(s), {evalSteps} steps/env"
      let (mean, nEps, lastLog) ← evalPluginEnvMinGRU env envCfg cfg.hiddenSize cfg.numLayers numEnvs evalSteps wLoaded cfg.seed
      IO.println s!"  eval: {nEps} episode(s), mean episode_return = {mean}{Puffer.RL.NNTrain.fmtEnvLog lastLog}"
  | "--help" :: _ => IO.println usage
  | "-h" :: _ => IO.println usage
  | "help" :: _ => IO.println usage
  | "forward-demo" :: _ => forwardDemo   -- STREAM 3: the MLP-forward self-check now needs an explicit keyword
  | "verify" :: _ => runVerify
  | "verify-grad" :: _ => runVerifyGrad
  | "verify-adam" :: _ => runVerifyAdam
  | "verify-vtrace" :: _ => runVerifyVtrace
  | "verify-trace" :: path :: _ => runVerifyTraceFile path
  | "verify-trace" :: _ => runVerifyTrace
  | "grad" :: _ => do
      let x := #[1.5, -0.5, 0.8]
      IO.println "autodiff validation:  f(x) = relu(x0*x1) + exp(x2) - x0*x2   at [1.5, -0.5, 0.8]"
      IO.println s!"  reverse-mode AD grad = {adGrad demoF x}"
      IO.println s!"  finite differences   = {fdGrad demoF x 1.0e-5}"
  | other =>
      -- STREAM 3 (CLI fidelity): an unknown first token (or no args) is an error, like PufferLib — not a
      -- silent fall-through to the dev self-check (that is now the explicit `forward-demo` keyword).
      let what := match other with | [] => "(no command)" | t :: _ => s!"'{t}'"
      IO.eprintln s!"puffer: unknown command {what}. Run `puffer help` for usage (train | eval | env-log | help)."
      IO.Process.exit 1
