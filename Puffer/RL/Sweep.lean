/-
Hyperparameter SWEEP subsystem — the Lean port of PufferLib's `pufferlib/sweep.py` +
`pufferl.py::sweep()`.  Drives `puffer sweep <env>` / `puffer paretosweep <env>`.

WHAT THIS MIRRORS (get the formulas right — faithfulness is the point):
  * `sweep.py` `Space`/`Linear`/`Pow2`/`Log`/`Logit` (L46-140): the normalize/unnormalize transforms and
    the `auto`/`time` scale resolution.  `_params_from_puffer_sweep` (L141-183): the `[sweep.<g>.<p>]`
    param-spec → `Space` mapping.  `Hyperparameters` (L184-255): `sample`/`to_dict`/`from_dict`, the
    `search_centers`/`min_bounds`/`max_bounds`/`search_scales` arrays, `num`, `get_flat_idx`.
    `pareto_points` (L256-277).  `Random` (L309-340) and `ParetoGenetic` (L342-392) — reimplemented
    natively (NO Python).  `Protein` (L522+, the gpytorch optimizer) is BRIDGED to the real
    `pufferlib.sweep.Protein` via a Python co-process (below), falling back to native `Random` on
    `ImportError`.
  * `pufferl.py::sweep()` (L406-466) + `_train`'s result extraction (L326-377): the trial loop
    (suggest → train → observe each downsampled (score, cost, timestep) point), where
    scores = `metrics['env/<metric>']`, costs = `metrics['uptime']`, timesteps = `metrics['agent_steps']`.

HOW A TRIAL RUNS (reuses the existing self-log — see `Puffer.RL.SelfLog`): a trial spawns
  `puffer train <env> <base-flags> <suggested-hyper-flags> --train.total-timesteps <T>` as a subprocess
  (stdout → /dev/null so its dashboard doesn't clutter; the dashboard redraw still records the self-log),
  waits, then reads the NEW `logs/<env>/<id>.json` and pulls the three downsampled metric arrays.

DELIBERATE DIVERGENCES from PufferLib (all for the sequential single-GPU + subprocess-trial design):
  * `train/total_timesteps` is a swept `Space` (so `num`/bounds/`to_dict` match PufferLib bit-for-bit and
    Protein still sees it as its `cost_param`), but it is NOT emitted as a training flag — each trial's
    budget is the explicit `--train.total-timesteps <T>` (PufferLib's `sweep()` likewise fixes it, via
    `fixed_total_timesteps`, in pareto mode).  Its OBSERVED value per downsampled point is still the real
    `agent_steps`, exactly like PufferLib's `done_args['train']['total_timesteps'] = t`.
  * Only trial 0 uses the base config; trials ≥1 call `suggest()` (PufferLib's `idx > 1` guard runs 1-2
    default trials because of its multi-GPU in-flight pipeline; sequentially there is never an in-flight
    trial, so we start suggesting at trial 1 — this is what makes `paretosweep`/`ParetoGenetic` use the
    pareto front "after the first").
  * `paretosweep` geomspaces the per-trial budget over `[T/8, T]` (`T` = `--train.total-timesteps`)
    rather than PufferLib's fixed `[3e7, 1e11]`, for practicality; the pareto cost-pinning mechanism
    (fixing `fixed_total_timesteps` so the cost axis is held constant per trial) is identical.
  * The Protein bridge forces `use_gpu=False` (the GP is tiny; CPU avoids any GPU contention with the
    training subprocess).  The GP math is float64 either way.
-/
import Puffer.Check.Parse

namespace Puffer.RL.Sweep

open Puffer.Check (parseFloat?)

/-! ### Small self-contained helpers (this is a library module — no Exe deps). -/

@[inline] def negInf : Float := (-1.0) / 0.0
@[inline] def posInf : Float := 1.0 / 0.0
@[inline] def ln2 : Float := Float.log 2.0
@[inline] def ln10 : Float := Float.log 10.0
/-- `math.log(x, 2)` = `log(x)/log(2)` (CPython does NOT special-case base 2/10 in `math.log(x, base)`). -/
@[inline] def log2 (x : Float) : Float := Float.log x / ln2
/-- `math.log(x, 10)` = `log(x)/log(10)`. -/
@[inline] def log10 (x : Float) : Float := Float.log x / ln10

/-- Python's builtin `round()` / `np.round`: round half to EVEN.  (Verified: 2.5→2, 3.5→4, −2.5→−2.) -/
def pyRound (x : Float) : Float :=
  let f := Float.floor x
  let d := x - f
  if d < 0.5 then f
  else if d > 0.5 then f + 1.0
  else if Float.floor (f / 2.0) == f / 2.0 then f else f + 1.0  -- tie → nearest even

/-- Strip a leading `--`; drop a single `section.` prefix; underscores → hyphens (matches Exe's `normKey`). -/
def normKey (s : String) : String :=
  let s := if s.startsWith "--" then String.ofList (s.toList.drop 2) else s
  let s := match s.splitOn "." with | [_, leaf] => leaf | _ => s
  s.replace "_" "-"

def parseNat? (s : String) : Option Nat := (s.replace "_" "").toNat?
def parseNat (s : String) (d : Nat) : Nat :=
  let s := s.replace "_" ""
  match s.toNat? with
  | some n => n
  | none => match s.splitOn "." with | i :: _ => (i.toNat?).getD d | _ => d
def parseBool (s : String) (d : Bool) : Bool :=
  match s.trim.toLower with
  | "1" | "true" | "yes" | "on" => true
  | "0" | "false" | "no" | "off" | "" => false
  | _ => (parseFloat? s |>.map (· != 0.0)).getD d
/-- parseFloat that also tolerates PufferLib's `1_000_000` underscores. -/
@[inline] def pf? (s : String) : Option Float := parseFloat? (s.replace "_" "")

/-- Splitmix-ish uniform in `[0,1)` (24-bit).  The native samplers are stochastic, so the exact RNG is
    irrelevant to faithfulness (that is checked on the DETERMINISTIC transforms, `verifySweepSpaces`). -/
def randUnit (rng : UInt64) : Float × UInt64 :=
  let rng := rng * 6364136223846793005 + 1442695040888963407
  (Float.ofNat ((rng >>> 40).toNat) / 16777216.0, rng)

/-- Full-precision-ish float→string that survives our own `parseFloat?` at any magnitude (Lean's
    `toString` truncates to 6 decimals and collapses `1e-12`→`"0.000000"`).  Uses `mantissa e exp` so
    tiny/huge values round-trip; ~7 significant figures, which is ample for (stochastic) hyperparameters
    and JSON numbers.  Integer-valued params are formatted separately (as `Nat`). -/
def fmtFloat (x : Float) : String :=
  if x != x then "0"
  else if x == 0.0 then "0"
  else
    let neg := x < 0.0
    let a := Float.abs x
    let e := Float.floor (log10 a)
    let mant := a / ((10.0 : Float) ^ e)
    let estr := if e < 0.0 then "-" ++ toString ((Float.abs e).toUInt64.toNat)
                else toString (e.toUInt64.toNat)
    (if neg then "-" else "") ++ toString mant ++ "e" ++ estr

/-! ### Spaces — the normalize/unnormalize transforms (`sweep.py` L46-140). -/

inductive Kind | linear | pow2 | log | logit
  deriving BEq, Inhabited, Repr

structure Space where
  kind : Kind
  lo : Float          -- min
  hi : Float          -- max
  scale : Float       -- RESOLVED search scale (auto/time already turned into a number)
  isInt : Bool
  deriving Inhabited

/-- Resolve `scale` exactly as PufferLib: `time` → `1/(log2 max − log2 min)` (Log only), `auto` → 0.5,
    else the numeric value. -/
def resolveScale (kind : Kind) (lo hi : Float) (raw : Option Float) (rawStr : String) : Float :=
  match raw with
  | some x => x
  | none =>
    match kind with
    | .log => if rawStr.trim == "time" then 1.0 / (log2 hi - log2 lo) else 0.5
    | _ => 0.5

/-- Build a `Space` from a `[sweep.*]` distribution spec (`_params_from_puffer_sweep`, L167-178). -/
def mkSpace (dist : String) (lo hi : Float) (scaleRaw : Option Float) (scaleStr : String) : Option Space :=
  let build := fun k i => some { kind := k, lo, hi, scale := resolveScale k lo hi scaleRaw scaleStr, isInt := i }
  match dist.trim with
  | "uniform"      => build .linear false
  | "int_uniform"  => build .linear true
  | "uniform_pow2" => build .pow2 true
  | "log_normal"   => build .log false
  | "logit_normal" => build .logit false
  | _ => none

/-- `distribution` string for a `Space` (inverse of `mkSpace`; used to rebuild the spec for the bridge). -/
def distString (s : Space) : String :=
  match s.kind, s.isInt with
  | .linear, false => "uniform"
  | .linear, true  => "int_uniform"
  | .pow2, _       => "uniform_pow2"
  | .log, _        => "log_normal"
  | .logit, _      => "logit_normal"

/-- `Space.normalize` (L64-134): value → [−1, 1]. -/
def normalize (s : Space) (v : Float) : Float :=
  match s.kind with
  | .linear => 2.0 * ((v - s.lo) / (s.hi - s.lo)) - 1.0
  | .pow2   => 2.0 * ((log2 v - log2 s.lo) / (log2 s.hi - log2 s.lo)) - 1.0
  | .log    => 2.0 * ((log10 v - log10 s.lo) / (log10 s.hi - log10 s.lo)) - 1.0
  | .logit  =>
      let v := max s.lo (min v s.hi)
      2.0 * ((log10 (1.0 - v) - log10 (1.0 - s.lo)) / (log10 (1.0 - s.hi) - log10 (1.0 - s.lo))) - 1.0

/-- `Space.unnormalize` (L69-139): [−1, 1] → value (with the integer rounding PufferLib applies). -/
def unnormalize (s : Space) (v : Float) : Float :=
  match s.kind with
  | .linear =>
      let x := ((v + 1.0) / 2.0) * (s.hi - s.lo) + s.lo
      if s.isInt then pyRound x else x
  | .pow2   =>
      let ls := ((v + 1.0) / 2.0) * (log2 s.hi - log2 s.lo) + log2 s.lo
      (2.0 : Float) ^ (pyRound ls)
  | .log    =>
      let ls := ((v + 1.0) / 2.0) * (log10 s.hi - log10 s.lo) + log10 s.lo
      let x := (10.0 : Float) ^ ls
      if s.isInt then pyRound x else x
  | .logit  =>
      let ls := ((v + 1.0) / 2.0) * (log10 (1.0 - s.hi) - log10 (1.0 - s.lo)) + log10 (1.0 - s.lo)
      1.0 - (10.0 : Float) ^ ls

/-! ### The parsed sweep config (PufferLib's `Hyperparameters`, but ordered explicitly). -/

structure Param where
  group : String        -- e.g. "train"
  name  : String        -- e.g. "learning_rate"
  space : Space
  deriving Inhabited

/-- `unroll_nested_dict` flat key: `train/learning_rate`. -/
def Param.flatKey (p : Param) : String := p.group ++ "/" ++ p.name
/-- The CLI flag form; the subprocess's `normKey` turns `train.learning_rate` → the leaf, `_`→`-`. -/
def Param.flag (p : Param) : String := "--" ++ p.group ++ "." ++ p.name

structure SweepConfig where
  method : String
  metric : String
  goal   : String
  metricDistribution : String
  downsample : Nat
  maxRuns : Nat
  earlyStopQuantile : Float
  maxSuggestionCost : Float
  useGpu : Bool
  prunePareto : Bool
  params : Array Param
  deriving Inhabited

def SweepConfig.num (c : SweepConfig) : Nat := c.params.size
def SweepConfig.normMins (c : SweepConfig) : Array Float := c.params.map (fun p => normalize p.space p.space.lo)
def SweepConfig.normMaxs (c : SweepConfig) : Array Float := c.params.map (fun p => normalize p.space p.space.hi)
def SweepConfig.scales (c : SweepConfig) : Array Float := c.params.map (fun p => p.space.scale)
def SweepConfig.centers (c : SweepConfig) : Array Float := c.params.map (fun _ => 0.0)  -- norm_mean = 0
/-- Index of `train/total_timesteps` in the flat param vector (Protein's `cost_param_idx`). -/
def SweepConfig.costIdx (c : SweepConfig) : Option Nat := c.params.findIdx? (fun p => p.flatKey == "train/total_timesteps")

/-- `Hyperparameters.sample` for n=1 (L213-224): `clip(scale·(2·rand−1) + mu, min, max)` per dim.
    Drawing one point is statistically identical to `random.choice` over `random_suggestions` i.i.d.
    draws, which is what `Random`/`ParetoGenetic` do. -/
def sampleOne (c : SweepConfig) (mu : Array Float) (globalScale : Float) (rng : UInt64) : Array Float × UInt64 := Id.run do
  let mins := c.normMins; let maxs := c.normMaxs; let scales := c.scales
  let mut r := rng
  let mut out : Array Float := Array.mkEmpty c.num
  for i in [0:c.num] do
    let (u, r') := randUnit r; r := r'
    let v := (scales[i]! * globalScale) * (2.0 * u - 1.0) + (mu[i]?.getD 0.0)
    out := out.push (max mins[i]! (min v maxs[i]!))
  return (out, r)

/-- `Hyperparameters.to_dict` → the training flags for one normalized sample (`_fill` + `unnormalize`,
    L237-250).  `train/total_timesteps` is dropped (budget-controlled); integer spaces format as `Nat`. -/
def valuesToFlags (c : SweepConfig) (sample : Array Float) : Array String := Id.run do
  let mut fs : Array String := #[]
  for i in [0:c.num] do
    let p := c.params[i]!
    if p.flatKey == "train/total_timesteps" then continue
    let v := unnormalize p.space (sample[i]?.getD 0.0)
    let vs := if p.space.isInt then toString v.toUInt64.toNat else fmtFloat v
    fs := (fs.push p.flag).push vs
  return fs

/-! ### INI parsing for `[sweep]` scalars (from the layered flag map) + `[sweep.<g>.<p>]` param specs
    (re-read from the ini files, because Exe's `parseIni` flattens every `[sweep.x.y]` section to bare
    `distribution`/`min`/`max`/`scale` keys, which all collide). -/

/-- Ordered `(sectionName, [(key,val)])` for every `[...]` section of one ini's text (file order). -/
def iniSections (contents : String) : Array (String × Array (String × String)) := Id.run do
  let mut out : Array (String × Array (String × String)) := #[]
  let mut cur : Option String := none
  let mut kvs : Array (String × String) := #[]
  for raw in contents.splitOn "\n" do
    let line := ((raw.splitOn "#").headD raw).trim   -- strip inline `# …` comments
    if line.isEmpty || line.startsWith "#" || line.startsWith ";" then continue
    if line.startsWith "[" then
      match cur with | some s => out := out.push (s, kvs) | none => pure ()
      cur := some (String.ofList (line.toList.drop 1 |>.takeWhile (· != ']')) |>.trim)
      kvs := #[]
    else
      match line.splitOn "=" with
      | k :: vs => kvs := kvs.push (k.trim, (String.intercalate "=" vs).trim)
      | _ => pure ()
  match cur with | some s => out := out.push (s, kvs) | none => pure ()
  return out

/-- Collect the `[sweep]` scalars AND `[sweep.<group>.<param>]` param specs across the layered ini files
    (default ← per-env), preserving first-appearance order and merging a later file's keys over an
    earlier one's for the same section — exactly `configparser.read([default, env])` deep-merge semantics.
    (Read here, NOT via Exe's `parseIni`, because that flattens every `[sweep.x.y]` to bare colliding
    `distribution`/`min`/`max`/`scale` keys — and Exe's parseIni now skips `[sweep*]` so those keys don't
    pollute the train/env flag map, e.g. `[sweep] method` collided with whisker_racer's env `method`.) -/
def collectSweep (files : List String) :
    IO (Array (String × String) × Array (String × String × Array (String × String))) := do
  let mut scalars : Array (String × String) := #[]   -- normKey'd, later files override
  let mut acc : Array (String × String × Array (String × String)) := #[]
  for f in files do
    if ← System.FilePath.pathExists f then
      for (sec, kv) in iniSections (← IO.FS.readFile f) do
        if sec == "sweep" then
          for (k, v) in kv do
            let nk := normKey k
            scalars := (scalars.filter (·.1 != nk)).push (nk, v)
        else if sec.startsWith "sweep." then
          match (String.ofList (sec.toList.drop 6)).splitOn "." with
          | [grp, nm] =>
            match acc.findIdx? (fun (gr, pn, _) => gr == grp && pn == nm) with
            | some i =>
                let (_, _, old) := acc[i]!
                let merged := kv.foldl (fun (o : Array (String × String)) p => (o.filter (·.1 != p.1)).push p) old
                acc := acc.set! i (grp, nm, merged)
            | none => acc := acc.push (grp, nm, kv)
          | _ => pure ()   -- non 2-level sweep sections don't occur in the shipped configs
  return (scalars, acc)

/-- Read `method`/`metric`/`goal`/`max_runs`/`downsample`/… (CLI `--sweep.*` overriding the ini `[sweep]`
    defaults) and the param `Space`s from the ini files, into a `SweepConfig`. `flags` is Exe's
    `loadConfigFlags` result: it carries CLI `--sweep.method` (→ `method`) but NOT the ini `[sweep]`
    scalars (Exe's parseIni skips them), so the ini defaults come from `collectSweep`'s own parse. -/
def parseSweep (env : String) (basePath : String) (flags : List (String × String)) : IO SweepConfig := do
  let (iniScalars, secs) ← collectSweep [basePath, s!"config/{env}.ini", s!"config/ocean/{env}.ini"]
  let mut params : Array Param := #[]
  for (grp, nm, kv) in secs do
    let get := fun k => (kv.find? (·.1 == k)).map (·.2)
    let dist := (get "distribution").getD ""
    let lo := ((get "min").bind pf?).getD 0.0
    let hi := ((get "max").bind pf?).getD 1.0
    let scStr := (get "scale").getD "auto"
    match mkSpace dist lo hi (pf? scStr) scStr with
    | some sp => params := params.push { group := grp, name := nm, space := sp }
    | none => pure ()
  -- scalar lookup: CLI --sweep.* (in `flags`) first, else the ini [sweep] default (iniScalars)
  let look := fun (m : List (String × String)) (k : String) =>
    (m.find? (fun p => p.1 == k)).map (fun p => p.2)
  let g := fun (k : String) => (look flags (normKey k)).orElse (fun _ => look iniScalars.toList (normKey k))
  -- Reorder to PufferLib's `unroll_nested_dict` order: params grouped by first-appearance GROUP
  -- (all train/*, then policy/*, then vec/*, …), preserving within-group section order. The ini gives
  -- section order (train, policy, policy, vec, train, …); the nested dict groups them.
  let mut groups : Array String := #[]
  for p in params do if !groups.contains p.group then groups := groups.push p.group
  params := groups.foldl (fun acc grp => acc ++ params.filter (·.group == grp)) #[]
  return {
    method := (g "method").getD "Protein" |>.trim
    metric := (g "metric").getD "score" |>.trim
    goal := (g "goal").getD "maximize" |>.trim
    metricDistribution := (g "metric-distribution").getD "linear" |>.trim
    downsample := (g "downsample").elim 5 (parseNat · 5)
    maxRuns := (g "max-runs").elim 1200 (parseNat · 1200)
    earlyStopQuantile := (g "early-stop-quantile").elim 0.3 (fun s => (pf? s).getD 0.3)
    maxSuggestionCost := (g "max-suggestion-cost").elim 3600.0 (fun s => (pf? s).getD 3600.0)
    useGpu := (g "use-gpu").elim true (parseBool · true)
    prunePareto := (g "prune-pareto").elim true (parseBool · true)
    params }

/-- Per-param DEFAULT values (from the layered config, the value each swept param has when unswept),
    clamped into `[lo,hi]`.  Feeds trial 0's observed input and the Protein bridge's `fill`. -/
def defaultValues (c : SweepConfig) (flags : List (String × String)) : Array Float :=
  c.params.map (fun p =>
    let raw := (flags.find? (·.1 == normKey (p.group ++ "." ++ p.name))).map (·.2)
    let v := (raw.bind pf?).getD (unnormalize p.space 0.0)
    max p.space.lo (min v p.space.hi))

/-- Trial 0's normalized input vector = `from_dict(defaults)`. -/
def defaultsNorm (c : SweepConfig) (defaults : Array Float) : Array Float :=
  (Array.range c.num).map (fun i => normalize c.params[i]!.space (defaults[i]?.getD 0.0))

/-! ### Native optimizers: `Random` and `ParetoGenetic` (`sweep.py` L309-392). -/

structure Obs where
  input : Array Float
  output : Float
  cost : Float
  deriving Inhabited

/-- `pareto_points` (L256-277): non-dominated points, ascending cost, score strictly increasing (+ε). -/
def paretoPoints (obs : Array Obs) : Array Obs := Id.run do
  if obs.isEmpty then return #[]
  let idxs := (Array.range obs.size).qsort (fun a b => obs[a]!.cost < obs[b]!.cost)
  let mut out : Array Obs := #[]
  let mut mx := negInf
  for i in idxs do
    if obs[i]!.output > mx + 1.0e-6 then out := out.push obs[i]!; mx := obs[i]!.output
  return out

/-- `ParetoGenetic.suggest`'s search-center pick (L363-374, `bias_cost=True`, `log_bias=False`):
    the pareto point whose nearest-in-cost neighbour is farthest (`argmax_i min_j |c_i−c_j|`). -/
def paretoCenter (cands : Array Obs) : Array Float := Id.run do
  if cands.size ≤ 1 then return (cands[0]?.map (·.input)).getD #[]
  let costs := cands.map (·.cost)
  let mx := costs.foldl max costs[0]!
  let mut bestIdx := 0; let mut bestVal := negInf
  for i in [0:cands.size] do
    let mut mn := posInf
    for j in [0:cands.size] do
      let d := if i == j then mx + 1.0 else Float.abs (costs[i]! - costs[j]!)
      mn := min mn d
    if mn > bestVal then bestVal := mn; bestIdx := i
  return cands[bestIdx]!.input

/-! ### The Protein bridge — an inline `python3 -c` co-process driving the REAL `pufferlib.sweep.Protein`.
    Mirrors `Puffer.RL.Wandb`'s inline-python pattern: single-quoted Python so the Lean string needs no
    escaping; `PUFFER_PYTHON` selects the interpreter (default `python3`). -/

/-- Line protocol on the co-process's stdin/stdout (its own library chatter is redirected to devnull so it
    can't corrupt the channel; notes/errors go to stderr which we inherit):
      line 1 (stdin): the sweep config JSON `{__method__, __fill__, metric, goal, <group>:{<param>:spec}}`
      → `ready` (stdout) once the optimizer is built, else the process exits (→ Lean falls back to Random).
      `{"cmd":"suggest","fixed_total_timesteps":T|null}` → the flattened suggested hypers `{"g/p":v,…}`.
      `{"cmd":"observe"|"observe_default","score":s,"cost":c,"timestep":t}` → `ok`  (observe uses the last
      suggestion's hypers, or the defaults for `observe_default`; `train/total_timesteps` ← t, as pufferl).
      `{"cmd":"quit"}` → exit. -/
def bridgePy : String :=
"import sys, os, json
_out = sys.stdout
sys.stdout = open(os.devnull, 'w')
def send(o):
    _out.write(json.dumps(o) + '\\n'); _out.flush()
def sendraw(s):
    _out.write(s + '\\n'); _out.flush()
def flat(d, pre=''):
    o = {}
    for k, v in d.items():
        if isinstance(v, dict): o.update(flat(v, pre + k + '/'))
        else: o[pre + k] = v
    return o
try:
    import pufferlib.sweep as S
except Exception as e:
    sys.stderr.write('puffer sweep: import pufferlib.sweep failed (' + str(e) + '); falling back to native Random\\n')
    sys.exit(3)
try:
    cfg = json.loads(sys.stdin.readline())
    method = cfg.pop('__method__')
    fill = cfg.pop('__fill__')
    obj = getattr(S, method)(cfg)
except Exception as e:
    sys.stderr.write('puffer sweep: could not build pufferlib.sweep.' + str(locals().get('method','?')) + ' (' + str(e) + '); falling back to native Random\\n')
    sys.exit(3)
from copy import deepcopy
last = deepcopy(fill)
sendraw('ready')
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try:
        req = json.loads(line)
    except Exception:
        continue
    cmd = req.get('cmd')
    if cmd == 'suggest':
        fixed = req.get('fixed_total_timesteps')
        try:
            hypers, _info = obj.suggest(deepcopy(fill), fixed_total_timesteps=fixed)
        except Exception as e:
            sys.stderr.write('puffer sweep: Protein.suggest failed (' + str(e) + ')\\n')
            hypers = deepcopy(fill)
        last = hypers
        send(flat(hypers))
    elif cmd == 'observe' or cmd == 'observe_default':
        h = deepcopy(fill) if cmd == 'observe_default' else last
        t = req.get('timestep', None)
        if t is not None and isinstance(h.get('train'), dict) and 'total_timesteps' in h['train']:
            h['train']['total_timesteps'] = t
        try:
            obj.observe(h, req['score'], req['cost'], is_failure=req.get('is_failure', False))
        except Exception as e:
            sys.stderr.write('puffer sweep: Protein.observe failed (' + str(e) + ')\\n')
        sendraw('ok')
    elif cmd == 'quit':
        break
"

structure Bridge where
  stdin : IO.FS.Handle
  stdout : IO.FS.Handle
  wait : IO Unit

/-- `{"group":{"param":<valOf i>,…},…}` inner (no outer braces), groups in first-appearance order. -/
def groupedInner (c : SweepConfig) (valOf : Nat → String) : String := Id.run do
  let mut groups : Array String := #[]
  for p in c.params do if !groups.contains p.group then groups := groups.push p.group
  let mut parts : Array String := #[]
  for grp in groups do
    let mut inner : Array String := #[]
    for i in [0:c.num] do
      let p := c.params[i]!
      if p.group == grp then inner := inner.push ("\"" ++ p.name ++ "\":" ++ valOf i)
    parts := parts.push ("\"" ++ grp ++ "\":{" ++ String.intercalate "," inner.toList ++ "}")
  return String.intercalate "," parts.toList

/-- The sweep config JSON for the bridge.  Param specs carry the RESOLVED numeric scale (pufferlib's Space
    uses a numeric scale verbatim, so `auto`→0.5 / `time`→number reconstruct the identical Space).
    `use_gpu` is forced false. -/
def bridgeConfigJson (c : SweepConfig) (defaults : Array Float) : String :=
  let scalars := String.intercalate "," [
    "\"__method__\":\"" ++ c.method ++ "\"",
    "\"metric\":\"" ++ c.metric ++ "\"",
    "\"metric_distribution\":\"" ++ c.metricDistribution ++ "\"",
    "\"goal\":\"" ++ c.goal ++ "\"",
    "\"downsample\":" ++ toString c.downsample,
    "\"max_runs\":" ++ toString c.maxRuns,
    "\"use_gpu\":false",
    "\"prune_pareto\":" ++ (if c.prunePareto then "true" else "false"),
    "\"early_stop_quantile\":" ++ fmtFloat c.earlyStopQuantile,
    "\"max_suggestion_cost\":" ++ fmtFloat c.maxSuggestionCost ]
  let specVal := fun i =>
    let p := c.params[i]!
    "{\"distribution\":\"" ++ distString p.space ++ "\",\"min\":" ++ fmtFloat p.space.lo
      ++ ",\"max\":" ++ fmtFloat p.space.hi ++ ",\"scale\":" ++ fmtFloat p.space.scale ++ "}"
  let fillVal := fun i =>
    let p := c.params[i]!
    if p.space.isInt then toString (defaults[i]!.toUInt64.toNat) else fmtFloat (defaults[i]!)
  "{" ++ scalars ++ "," ++ groupedInner c specVal ++ ",\"__fill__\":{" ++ groupedInner c fillVal ++ "}}"

/-- Spawn the bridge, hand it the config, and wait for `ready`.  `none` ⇒ import/build failed (the note is
    already on stderr) ⇒ the caller falls back to native Random. -/
def startBridge (cfgJson : String) : IO (Option Bridge) := do
  try
    let py := (← IO.getEnv "PUFFER_PYTHON").getD "python3"
    let child ← IO.Process.spawn {
      cmd := py, args := #["-c", bridgePy], stdin := .piped, stdout := .piped, stderr := .inherit }
    let hIn := child.stdin
    let hOut := child.stdout
    hIn.putStr (cfgJson ++ "\n"); hIn.flush
    let ready ← hOut.getLine
    if ready.trim == "ready" then
      return some { stdin := hIn, stdout := hOut, wait := do let _ ← child.wait; pure () }
    else
      return none   -- the co-process exited (import/build failed); note is already on stderr
  catch _ => return none

/-- The text between the first two double-quotes of a JSON string token (the key). -/
def stripQuotes (s : String) : String := ((s.splitOn "\"")[1]?).getD s.trim

/-- Parse the bridge's flat suggestion `{"g/p":v,…}` (one JSON object on a line) into `(flatKey, value)`. -/
def parseFlatJson (line : String) : Array (String × Float) := Id.run do
  let inner := match line.splitOn "{" with | _ :: t :: _ => t | _ => line   -- after first '{'
  let inner := (inner.splitOn "}").headD inner                              -- before first '}'
  let mut out : Array (String × Float) := #[]
  for piece in inner.splitOn "," do
    match piece.splitOn ":" with
    | k :: v :: _ =>
        match pf? v.trim with
        | some f => out := out.push (stripQuotes k, f)
        | none => pure ()
    | _ => pure ()
  return out

/-- Suggested flags from the bridge (skip `train/total_timesteps`; integer spaces → `Nat`). -/
def flagsFromBridge (c : SweepConfig) (vals : Array (String × Float)) : Array String := Id.run do
  let mut fs : Array String := #[]
  for p in c.params do
    if p.flatKey == "train/total_timesteps" then continue
    match vals.find? (·.1 == p.flatKey) with
    | some (_, v) => fs := (fs.push p.flag).push (if p.space.isInt then toString v.toUInt64.toNat else fmtFloat v)
    | none => pure ()
  return fs

def bridgeSuggest (b : Bridge) (fixedTs : Option Float) : IO (Array (String × Float)) := do
  let fx := match fixedTs with | some t => fmtFloat t | none => "null"
  b.stdin.putStr ("{\"cmd\":\"suggest\",\"fixed_total_timesteps\":" ++ fx ++ "}\n"); b.stdin.flush
  return parseFlatJson (← b.stdout.getLine)

def bridgeObserve (b : Bridge) (defaultTrial : Bool) (score cost : Float)
    (timestep : Option Float) (isFailure : Bool) : IO Unit := do
  let cmd := if defaultTrial then "observe_default" else "observe"
  let ts := match timestep with | some t => ",\"timestep\":" ++ fmtFloat t | none => ""
  let fl := if isFailure then ",\"is_failure\":true" else ""
  b.stdin.putStr ("{\"cmd\":\"" ++ cmd ++ "\",\"score\":" ++ fmtFloat score
    ++ ",\"cost\":" ++ fmtFloat cost ++ ts ++ fl ++ "}\n")
  b.stdin.flush
  let _ ← b.stdout.getLine   -- ack

/-! ### Self-log reading (the metric arrays a training subprocess wrote — `Puffer.RL.SelfLog`). -/

def logFilenames (env : String) : IO (Array String) := do
  let dir : System.FilePath := s!"logs/{env}"
  if !(← dir.pathExists) then return #[]
  let mut out : Array String := #[]
  for f in ← dir.readDir do
    if f.fileName.endsWith ".json" then out := out.push f.fileName
  return out

/-- The NEWEST-by-mtime `logs/<env>/*.json` whose filename is not in `before` (i.e. the one this trial
    just wrote — each run's id is a fresh monotonic-nanos stamp).  `none` ⇒ the trial produced no log. -/
def readNewLog (env : String) (before : Array String) : IO (Option String) := do
  let dir : System.FilePath := s!"logs/{env}"
  if !(← dir.pathExists) then return none
  let mut best : Option (Int × System.FilePath) := none
  for f in ← dir.readDir do
    if f.fileName.endsWith ".json" && !before.contains f.fileName then
      let mt := (← f.path.metadata).modified
      let key : Int := mt.sec * 1000000000 + Int.ofNat mt.nsec.toNat
      match best with
      | some (bk, _) => if key ≥ bk then best := some (key, f.path)
      | none => best := some (key, f.path)
  match best with
  | some (_, p) => return some (← IO.FS.readFile p)
  | none => return none

/-- Pull one downsampled metric array (`metrics[key]`) from a self-log JSON.  Searches only after the
    `"metrics":{` marker and reads the bracketed comma list after `"key":[`. -/
def extractMetric (json key : String) : Array Float := Id.run do
  let afterM := match (json.splitOn "\"metrics\":{") with | _ :: t :: _ => t | _ => json
  match afterM.splitOn ("\"" ++ key ++ "\":[") with
  | _ :: rest :: _ =>
      let inner := (rest.splitOn "]").headD ""
      return (inner.splitOn ",").toArray.filterMap (fun t => pf? t.trim)
  | _ => return #[]

/-! ### The sweep loop. -/

def geomspace (lo hi : Float) (n : Nat) : Array Float :=
  if n == 0 then #[]
  else if n == 1 then #[hi]
  else (Array.range n).map (fun i => lo * ((hi / lo) ^ (Float.ofNat i / Float.ofNat (n - 1))))

/-- Pretty per-trial hyper summary: `learning-rate=2.34e-2 hidden-size=256 …`. -/
def prettyFlags (fs : Array String) : String := Id.run do
  let mut parts : Array String := #[]
  let mut i := 0
  while i + 1 < fs.size do
    parts := parts.push (((fs[i]!.replace "--" "").replace "." "/") ++ "=" ++ fs[i+1]!)
    i := i + 2
  return String.intercalate " " parts.toList

/-- The trial loop.  `budget` = resolved `--train.total-timesteps`; `baseFlags` = the user's forwarded
    flags (no `--sweep.*`, no `--train.total-timesteps`, no env positional); `flags` = the layered map. -/
def runSweep (env : String) (pareto : Bool) (pufferBin : String) (baseFlags : List String)
    (budget : Nat) (seed : UInt64) (sc : SweepConfig) (flags : List (String × String)) : IO Unit := do
  let maximize := sc.goal == "maximize"
  IO.println s!"puffer {if pareto then "paretosweep" else "sweep"} [{env}]: method={sc.method} \
    metric=env/{sc.metric} goal={sc.goal} max_runs={sc.maxRuns} downsample={sc.downsample} params={sc.num} \
    budget={budget}"
  let defaults := defaultValues sc flags
  let x0 := defaultsNorm sc defaults
  -- Optimizer: Protein → bridge (fall back to native Random on failure); Random/ParetoGenetic → native.
  let mut bridge? : Option Bridge := none
  let mut effMethod := sc.method
  if sc.method == "Protein" then
    bridge? ← startBridge (bridgeConfigJson sc defaults)
    if bridge?.isNone then
      IO.eprintln "puffer sweep: Protein bridge unavailable; falling back to native Random \
        (set PUFFER_PYTHON to an interpreter with pufferlib installed to bridge the real Protein)"
      effMethod := "Random"
    else
      IO.println "puffer sweep: bridged to pufferlib.sweep.Protein (Python co-process)"
  let tsList := if pareto then geomspace (Float.ofNat budget / 8.0) (Float.ofNat budget) sc.maxRuns else #[]
  let mut rng : UInt64 := seed * 2862933555777941757 + 3037000493
  let mut obs : Array Obs := #[]
  let mut bestScore := if maximize then negInf else posInf
  let mut bestFlags : Array String := #[]
  let mut bestTrial : Nat := 0
  let mut completed : Nat := 0
  for trial in [0:sc.maxRuns] do
    let T := if pareto then max 1 (tsList[trial]!).toUInt64.toNat else budget
    -- suggest (trial 0 uses the base config)
    let mut appliedFlags : Array String := #[]
    let mut suggestionVec : Array Float := x0
    let defaultTrial := trial == 0
    if !defaultTrial then
      match effMethod, bridge? with
      | "Protein", some b =>
          let fixedTs := if pareto then some (Float.ofNat T) else none
          appliedFlags := flagsFromBridge sc (← bridgeSuggest b fixedTs)
      | "ParetoGenetic", _ =>
          let cands := paretoPoints obs
          if cands.isEmpty then
            suggestionVec := sc.centers
          else
            let (s, r) := sampleOne sc (paretoCenter cands) 1.0 rng
            rng := r; suggestionVec := s
          appliedFlags := valuesToFlags sc suggestionVec
      | _, _ =>   -- native Random
          let (s, r) := sampleOne sc sc.centers 1.0 rng
          rng := r; suggestionVec := s
          appliedFlags := valuesToFlags sc suggestionVec
    -- run the trial as a subprocess; its self-log is the score/cost source
    let before ← logFilenames env
    let args := (["train", env] ++ baseFlags ++ appliedFlags.toList
                  ++ ["--train.total-timesteps", toString T]).toArray
    if (← IO.getEnv "PUFFER_SWEEP_VERBOSE").isSome then
      IO.eprintln s!"[sweep] trial {trial+1} cmd: {pufferBin} {String.intercalate " " args.toList}"
    let child ← IO.Process.spawn { cmd := pufferBin, args := args, stdout := .null, stderr := .inherit }
    let _ ← child.wait
    match ← readNewLog env before with
    | none =>
        -- No completed-episode self-log (no env/score) — e.g. a hyper combo whose episodes never finish
        -- inside the budget. PufferLib treats this as a FAILED trial (`observe(args, 0, 0, is_failure)`).
        match effMethod, bridge? with
        | "Protein", some b => bridgeObserve b defaultTrial 0.0 0.0 none true
        | _, _ => obs := obs.push { input := suggestionVec, output := 0.0, cost := 0.0 }
        IO.println s!"trial {trial+1}/{sc.maxRuns}: FAILED (no env/score in budget) — observed as failure (0,0)"
    | some json =>
        let scores := extractMetric json ("env/" ++ sc.metric)
        let costs := extractMetric json "uptime"
        let steps := extractMetric json "agent_steps"
        let np := min scores.size (min costs.size steps.size)
        -- observe each downsampled point (score, cost=uptime, timestep=agent_steps)
        for k in [0:np] do
          let s := scores[k]!; let c := costs[k]!; let t := steps[k]!
          match effMethod, bridge? with
          | "Protein", some b => bridgeObserve b defaultTrial s c (some t) false
          | _, _ =>
              let inp := match sc.costIdx with
                | some ci => if ci < suggestionVec.size then suggestionVec.set! ci (normalize sc.params[ci]!.space (max sc.params[ci]!.space.lo (min t sc.params[ci]!.space.hi))) else suggestionVec
                | none => suggestionVec
              obs := obs.push { input := inp, output := s, cost := c }
        let finalScore := if scores.isEmpty then (if maximize then negInf else posInf) else scores.back!
        let finalCost := if costs.isEmpty then 0.0 else costs.back!
        if !scores.isEmpty then completed := completed + 1
        if (if maximize then finalScore > bestScore else finalScore < bestScore) then
          bestScore := finalScore; bestFlags := appliedFlags; bestTrial := trial
        let hy := if appliedFlags.isEmpty then "(base config)" else prettyFlags appliedFlags
        IO.println s!"trial {trial+1}/{sc.maxRuns}: env/{sc.metric}={finalScore} cost={finalCost}s \
          T={T} pts={np}{if defaultTrial then " (defaults)" else ""}  hypers=[{hy}]"
  -- teardown + summary
  match bridge? with
  | some b => (try b.stdin.putStr "{\"cmd\":\"quit\"}\n"; b.stdin.flush; b.wait catch _ => pure ())
  | none => pure ()
  IO.println "─────────────────────────────────────────────"
  if bestScore == negInf || bestScore == posInf then
    IO.println s!"sweep done: {completed} successful trial(s); no scored trial to report as best"
  else
    IO.println s!"sweep done: {completed}/{sc.maxRuns} scored.  BEST = trial {bestTrial+1}  env/{sc.metric}={bestScore}"
    IO.println s!"  best hypers: {if bestFlags.isEmpty then "(base config)" else prettyFlags bestFlags}"

/-! ### Faithfulness self-check — emit the DETERMINISTIC transforms as exact float bits for
    `tools/sweep_parity.py` to compare against `pufferlib.sweep`.  See the task's FAITHFULNESS CHECK. -/

@[inline] def bits (x : Float) : String := toString (Float.toBits x)

def verifySweepSpaces : IO Unit := do
  -- (1) Per-Space battery (scale resolution + normalize + unnormalize) at fixed points.
  let mk := fun d lo hi sc => (mkSpace d lo hi (pf? sc) sc).getD default
  let battery : Array (String × Space) := #[
    ("uniform",      mk "uniform" 0.1 5.0 "0.5"),
    ("int_uniform",  mk "int_uniform" 1.0 8.0 "auto"),
    ("uniform_pow2", mk "uniform_pow2" 32.0 1024.0 "auto"),
    ("log_normal",   mk "log_normal" 0.00001 0.1 "0.5"),
    ("log_time",     mk "log_normal" 3.0e7 1.0e11 "time"),
    ("logit_normal", mk "logit_normal" 0.8 0.9999 "auto")]
  let normPts := #[-1.0, -0.3, 0.4, 1.0]
  for (nm, sp) in battery do
    IO.println s!"SCALE {nm} {bits sp.scale}"
    let mids := #[sp.lo, sp.hi, (sp.lo + sp.hi) / 2.0]
    for k in [0:3] do IO.println s!"NORM {nm} {k} {bits (normalize sp mids[k]!)}"
    for k in [0:4] do IO.println s!"UNNORM {nm} {k} {bits (unnormalize sp normPts[k]!)}"
  -- (2) The full default.ini sweep config (flat keys/order, bounds, scales, to_dict, from_dict).
  let sc ← parseSweep "default" "config/default.ini" []
  IO.println s!"NUM {sc.num}"
  let sample := (Array.range sc.num).map (fun i => Float.sin (Float.ofNat i))
  for i in [0:sc.num] do
    IO.println s!"KEY {i} {sc.params[i]!.flatKey}"
    IO.println s!"NMIN {i} {bits sc.normMins[i]!}"
    IO.println s!"NMAX {i} {bits sc.normMaxs[i]!}"
    IO.println s!"SCL {i} {bits sc.scales[i]!}"
    let v := unnormalize sc.params[i]!.space sample[i]!
    IO.println s!"TODICT {i} {bits v}"
    IO.println s!"FROMDICT {i} {bits (normalize sc.params[i]!.space v)}"

end Puffer.RL.Sweep
