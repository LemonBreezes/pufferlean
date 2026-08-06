/-
Live training dashboard — a faithful reproduction of PufferLib's `print_dashboard`
(`pufferlib/pufferl.py`), rendered from the Lean trainers when `--log` is passed.

PufferLib draws its monitor with `rich` (a ROUNDED box Table nesting three sub-tables:
Summary | Performance | Losses, then a two-column User Stats grid). rich renders at 80
columns when piped to a non-TTY, and PufferLib redraws in place each update with a
cursor-home escape (`\033[0;0H`), rate-limited to every 0.6s.

Rather than re-implement rich's generic table layout, this module reproduces the EXACT
fixed 80-column layout of a captured `puffer train moba` frame (see the column anchors
below, measured cell-for-cell from that frame). The number formatters (`abbreviate`,
`duration`, `fmtPerf`) mirror PufferLib's helpers op-for-op.

This renderer is OPT-IN: without `--log` the trainers keep their ad-hoc per-update lines
byte-for-byte (tooling parses them). See `Puffer/RL/MinGRUTrain.lean` for the wiring.
-/

namespace Puffer.RL.Dashboard

/-- GPU util % + VRAM (used/total GB) via NVML (`dlopen`'d, microsecond library call — the header's
    live stats, replacing a ~25ms `nvidia-smi` subprocess). Returns `#[gpu%, usedGB, totalGB]`, or
    zeros if NVML is unavailable. Defined in `ffi/puffercuda.cu` (`lean_nvml_stats`). -/
@[extern "lean_nvml_stats"]
opaque nvmlStatsFFI : IO FloatArray

/-! ### Number formatting (mirrors PufferLib's `pufferl.py` helpers). -/

/-- Fixed-decimal formatter (`f'{x:.{p}f}'`). Rounds half-up (Python uses half-to-even; the
    two differ only on exact ties, which float data almost never hits). Values fed here are
    bounded (`abbreviate` divides to <1000 first; durations are <1000ms; losses/stats are
    small), so `scaled` stays well within `UInt64`. `-0.000` is normalized to `0.000`. -/
def fmtFixed (x : Float) (p : Nat) : String :=
  if x.isNaN then "nan"
  else if x == (1.0/0.0) then "inf"
  else if x == (-1.0/0.0) then "-inf"
  else
    let neg := x < 0.0
    let a := x.abs
    let pw := 10 ^ p
    let scaled := a * Float.ofNat pw
    let nTot := (scaled + 0.5).toUInt64.toNat
    let ip := nTot / pw
    let fp := nTot % pw
    let frac :=
      if p == 0 then ""
      else
        let fs := toString fp
        "." ++ String.ofList (List.replicate (p - fs.length) '0') ++ fs
    (if neg && nTot != 0 then "-" else "") ++ toString ip ++ frac

/-- PufferLib `abbreviate`: divide by 1000 until <1000, suffix from ['','K','M','B','T'],
    format `{num:.1f}{suffix}` (SPS 3.4M, Steps 38.9M, Params 95.6K). -/
def abbreviate (x : Float) : String := Id.run do
  let prefixes := #["", "K", "M", "B", "T"]
  let mut num := x
  let mut pref := ""
  for i in [0:5] do
    pref := prefixes[i]!
    if num < 1000.0 then break
    num := num / 1000.0
  return fmtFixed num 1 ++ pref

/-- PufferLib `duration`: <0 → `0s`; <1s → `{s*1000:.0f}ms`; else `{d}d {h}h {m}m {s}s`. -/
def duration (seconds : Float) : String :=
  if seconds < 0.0 then "0s"
  else if seconds < 1.0 then fmtFixed (seconds * 1000.0) 0 ++ "ms"
  else
    let s := (Float.floor seconds).toUInt64.toNat
    s!"{s / 86400}d {(s / 3600) % 24}h {(s / 60) % 60}m {s % 60}s"

/-- PufferLib `fmt_perf`'s percent: `int(100*elapsed/delta - 1e-5)`, formatted `{:2d}%`
    (space-padded to width 2). `delta = rollout + train`. -/
def fmtPct (elapsed delta : Float) : String :=
  let pc := if delta == 0.0 then 0
            else (Float.floor (100.0 * elapsed / delta - 1.0e-5)).toUInt64.toNat
  let ps := toString pc
  (if ps.length ≥ 2 then ps else String.ofList (List.replicate (2 - ps.length) ' ') ++ ps) ++ "%"

/-! ### 80-column row layout (cell-exact anchors measured from a captured frame). -/

/-- Display width of one char. The blowfish emoji (U+1F421) is double-width; everything
    else we render (ASCII + `…`) is single-width. -/
@[inline] def charW (c : Char) : Nat := if c.toNat ≥ 0x1F000 then 2 else 1
/-- Display width of a string (sum of `charW`). -/
def dispWidth (s : String) : Nat := s.foldl (fun acc c => acc + charW c) 0

/-- Truncate `s` to display width `w` with a trailing `…` (rich's ellipsis truncation). -/
def truncW (s : String) (w : Nat) : String :=
  if s.length ≤ w then s
  else String.ofList (s.toList.take (w - 1)) ++ "…"

/-- A placed field: absolute start display column (col 0 = left border) and its string. -/
abbrev Field := Nat × String

/-- Left-justified field starting at display column `l`, truncated to width `w`. -/
@[inline] def fL (l w : Nat) (s : String) : Field := (l, truncW s w)
/-- Right-justified field ending (exclusive) at display column `r`. -/
@[inline] def fR (r : Nat) (s : String) : Field :=
  let dw := dispWidth s
  (if dw ≥ r then 1 else r - dw, s)

/-- Assemble one 80-column content row (`│` + 78 cells + `│`) from placed fields. Fields are
    sorted by start column and laid down left-to-right, padding with spaces between them. -/
def emitRow (fields : Array Field) : String := Id.run do
  let ps := fields.qsort (fun a b => a.1 < b.1)
  let mut out := "│"
  let mut col := 1
  for (start, s) in ps do
    while col < start do out := out.push ' '; col := col + 1
    out := out ++ s
    col := col + dispWidth s
  while col < 79 do out := out.push ' '; col := col + 1
  return out.push '│'

-- Column anchors (measured from `/tmp/pl_dash_frame.txt`, cell-for-cell).
-- Monitor: Summary | Performance | Losses
def sumLblL : Nat := 3
def sumLblW : Nat := 8
def sumValR : Nat := 25
def perfNmL : Nat := 29
def perfNmW : Nat := 8
def perfTmR : Nat := 45
def perfPcR : Nat := 50
def lossLblL : Nat := 54
def lossLblW : Nat := 16
def lossValR : Nat := 77
-- User Stats grid (two columns)
def usLLblL : Nat := 3
def usLblW : Nat := 22
def usLValR : Nat := 38
def usRLblL : Nat := 42
def usRValR : Nat := 77

def topBorder : String := "╭" ++ String.ofList (List.replicate 78 '─') ++ "╮"
def botBorder : String := "╰" ++ String.ofList (List.replicate 78 '─') ++ "╯"
def blankRow  : String := "│" ++ String.ofList (List.replicate 78 ' ') ++ "│"

/-! ### The dashboard payload + renderer. -/

/-- Everything the dashboard displays. Timings are in SECONDS; `remaining < 0` ⇒ unknown. -/
structure DashInput where
  envName      : String
  params       : Nat
  steps        : Nat
  sps          : Float
  epoch        : Nat
  uptime       : Float
  remaining    : Float
  rollout      : Float              -- Evaluate (total)
  train        : Float              -- Train (total)
  evalGpu      : Float
  evalEnv      : Float
  trainMisc    : Float
  trainForward : Float
  losses       : Array (String × Float)      -- ordered: policy,value,entropy,total,old_kl,kl,clipfrac
  userStats    : Array (String × Float)      -- ordered; `n` already excluded
  gpuPct       : Float
  vramUsed     : Float             -- GB
  vramTotal    : Float             -- GB
  ramGb        : Float             -- GB
  idx          : Nat               -- blowfish spinner offset

/-- Render the full multi-line frame (no trailing newline). -/
def render (d : DashInput) : String := Id.run do
  let mut rows : Array String := #[topBorder]
  -- Header row: `PufferLib 4.0 <idx spaces>🐡`  +  GPU / VRAM / RAM.
  let hdrLeft := "PufferLib 4.0 " ++ String.ofList (List.replicate d.idx ' ') ++ "🐡"
  rows := rows.push (emitRow #[
    (sumLblL, hdrLeft),
    (35, s!"GPU: {fmtFixed d.gpuPct 0}%"),
    (49, s!"VRAM: {fmtFixed d.vramUsed 1}/{fmtFixed d.vramTotal 0}G"),
    fR usRValR s!"RAM: {fmtFixed d.ramGb 1}G"])
  rows := rows.push blankRow
  -- Monitor: 8 rows. Summary (header+7), Performance (6 data, no header), Losses (header+7).
  let delta := d.rollout + d.train
  let sumRows : Array (String × String) := #[
    ("Summary", "Value"),
    ("Env", d.envName),
    ("Params", abbreviate (Float.ofNat d.params)),
    ("Steps", abbreviate (Float.ofNat d.steps)),
    ("SPS", abbreviate d.sps),
    ("Epoch", toString d.epoch),
    ("Uptime", duration d.uptime),
    ("Remaining", if d.remaining < 0.0 then "-" else duration d.remaining)]
  -- (name, elapsed) — name carries its own 2-space indent for sub-rows.
  let perfRows : Array (String × Float) := #[
    ("Evaluate", d.rollout), ("  GPU", d.evalGpu), ("  Env", d.evalEnv),
    ("Train", d.train), ("  Misc", d.trainMisc), ("  Forward", d.trainForward)]
  let lossRows : Array (String × String) := Id.run do
    let mut a : Array (String × String) := #[("Losses", "Value")]
    for (k, v) in d.losses do a := a.push (k, fmtFixed v 3)
    return a
  for i in [0:8] do
    let mut f : Array Field := #[]
    if i < sumRows.size then
      let (l, v) := sumRows[i]!
      f := f ++ #[fL sumLblL sumLblW l, fR sumValR v]
    if i < perfRows.size then
      let (nm, el) := perfRows[i]!
      f := f ++ #[fL perfNmL perfNmW nm, fR perfTmR (duration el), fR perfPcR (fmtPct el delta)]
    if i < lossRows.size then
      let (l, v) := lossRows[i]!
      f := f ++ #[fL lossLblL lossLblW l, fR lossValR v]
    rows := rows.push (emitRow f)
  rows := rows.push blankRow
  -- User Stats: header + up to 30 stats in two alternating columns (even→left, odd→right).
  let stats := if d.userStats.size > 30 then d.userStats.extract 0 30 else d.userStats
  rows := rows.push (emitRow #[
    fL usLLblL usLblW "User Stats", fR usLValR "Value",
    fL usRLblL usLblW "User Stats", fR usRValR "Value"])
  let nRows := (stats.size + 1) / 2
  for r in [0:nRows] do
    let mut f : Array Field := #[]
    let li := 2 * r
    if li < stats.size then
      let (k, v) := stats[li]!
      f := f ++ #[fL usLLblL usLblW k, fR usLValR (fmtFixed v 3)]
    let ri := 2 * r + 1
    if ri < stats.size then
      let (k, v) := stats[ri]!
      f := f ++ #[fL usRLblL usLblW k, fR usRValR (fmtFixed v 3)]
    rows := rows.push (emitRow f)
  rows := rows.push botBorder
  return String.intercalate "\n" rows.toList

/-! ### System stats (GPU / VRAM / RAM) — queried by the trainer at render time. -/

/-- GPU utilization %, VRAM used (GB), VRAM total (GB) via NVML. Microsecond library call (a
    `dlopen`'d driver query), so it is queried every render like PufferLib does — no subprocess, no
    cache needed. Zeros if NVML is unavailable. -/
def gpuStats : IO (Float × Float × Float) := do
  try
    let a ← nvmlStatsFFI
    if a.size ≥ 3 then return (a[0]!, a[1]!, a[2]!) else return (0.0, 0.0, 0.0)
  catch _ => return (0.0, 0.0, 0.0)

/-- Process resident memory (GB) from `/proc/self/status` (`VmRSS`, in kB). 0 on failure. -/
def ramGb : IO Float := do
  try
    let txt ← IO.FS.readFile "/proc/self/status"
    for line in txt.splitOn "\n" do
      if line.startsWith "VmRSS:" then
        let toks := (line.splitOn "\t").flatMap (·.splitOn " ") |>.filter (· != "")
        -- toks = ["VmRSS:", "5528", "kB"]
        match toks[1]? with
        | some kb => return (Float.ofNat (kb.toNat?.getD 0)) / 1048576.0
        | none => pure ()
    return 0.0
  catch _ => return 0.0

/-- Query GPU (NVML, microseconds) + RAM, render, and redraw in place (cursor-home, matching PufferLib). -/
def redraw (d : DashInput) : IO Unit := do
  let (gpu, vu, vt) ← gpuStats
  let ram ← ramGb
  let frame := render { d with gpuPct := gpu, vramUsed := vu, vramTotal := vt, ramGb := ram }
  IO.print ("\x1b[0;0H" ++ frame ++ "\n")
  (← IO.getStdout).flush

/-- Assemble a `DashInput` from the trainers' raw scalars and redraw. Shared by both MinGRU
    trainers. `lossArr` is `cudaMgReadLossesFFI`'s 7-element return; the per-phase seconds are
    `roll`/`vt`/`grad`/`muon` (Evaluate = roll; Train = vt+grad+muon; the GPU/Env and Misc/Forward
    sub-splits are approximated — see `MinGRUTrain.lean`). `now`/`t0` are monotonic nanos. -/
def redrawFrom (name : String) (params steps epoch t0 now totalTimesteps : Nat)
    (roll vt grad muon : Float) (lossArr : FloatArray)
    (userStats : Array (String × Float)) (idx : Nat) : IO Unit := do
  let upS := Float.ofNat (now - t0) / 1.0e9
  let sps := if now == t0 then 0.0 else Float.ofNat steps * 1.0e9 / Float.ofNat (now - t0)
  let remaining := if steps ≥ totalTimesteps || sps ≤ 0.0 then 0.0
                   else Float.ofNat (totalTimesteps - steps) / sps
  let gl := fun (i : Nat) => if i < lossArr.size then lossArr[i]! else 0.0
  redraw {
    envName := name, params := params, steps := steps, sps := sps, epoch := epoch,
    uptime := upS, remaining := remaining,
    rollout := roll, train := vt + grad + muon,
    evalGpu := 0.0, evalEnv := roll, trainMisc := vt + muon, trainForward := grad,
    losses := #[("policy", gl 0), ("value", gl 1), ("entropy", gl 2), ("total", gl 3),
                ("old_kl", gl 4), ("kl", gl 5), ("clipfrac", gl 6)],
    userStats := userStats,
    gpuPct := 0.0, vramUsed := 0.0, vramTotal := 0.0, ramGb := 0.0, idx := idx }

end Puffer.RL.Dashboard
