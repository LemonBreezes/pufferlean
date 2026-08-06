import Puffer.Check.Core

/-!
# C87: Mathlib-free trace-file parser for `puffer verify-trace <file>`

C84 wired the Mathlib-free checker core (`Puffer.Check.Core`) into the `puffer`
binary, but only over a built-in demo trace.  This module lets the binary check
a REAL recorded trace from disk: a line-oriented text format plus a fail-closed
parser producing exactly what `runTraceChecks` consumes.

Import discipline: this file imports **only** `Puffer.Check.Core` (transitive
closure `Core → Float.Expr → ErrBnd → Exec → ∅`), so it stays inside the exe's
documented Mathlib-free closure.  No `axiom`, no `sorry`, no `native_decide`.

## Trace file format

Line-oriented UTF-8 text.  Each line is trimmed; blank lines and lines starting
with `#` are skipped.  Tokens are separated by spaces/tabs.  The remaining
lines must be, in any order:

* `region <R>` — exactly once: the parameter-region radius (C73 `checkRegion`).
* `clip <tLo> <tHi>` — exactly once: the strict clip margin interval
  (C73 `checkClipMargin`).
* `budget <bound> <cap>` — zero or more: one `(bound, cap)` pair each, order
  preserved, fed to the `checkLe` half of `runTraceChecks` (C78's
  `*_le_overflow_of_check` discharge shape).
* `step <ratio> <p1> … <pk>` — zero or more, one per recorded training step,
  order preserved: the recorded PPO ratio FIRST, then the parameter row
  (`k ≥ 0`; the row is variable-length, which is why the ratio leads).
  Produces the `StepRec` `([p1, …, pk], ratio)`.

Any other line, any malformed Float token, a duplicated `region`/`clip` line,
or a missing `region`/`clip` line makes the whole parse return `none`
(fail-closed — a malformed file can never produce a PASS).

Float literals: optional sign, decimal digits, optional `.` fraction, optional
`e`/`E` exponent with optional sign (exponent limited to ≤ 4 digits — a
superset of f64's ±308 range; longer exponent TOKENS are rejected fail-closed).
Note that a ≤4-digit exponent whose VALUE overflows f64 (e.g. `1e9999`) is
accepted as its correctly-rounded value `±inf`, not rejected: an `inf` budget
bound then correctly FAILS its cap check, while an `inf` region radius makes
the region check vacuously permissive — harmless to soundness, since the
transferred theorems' ℝ-side hypotheses (`toReal Rf ≤ R` etc.) still gate any
real conclusion, but callers should not rely on the parser to reject
overflowing literals.  The
value is built by `Float.ofScientific` on the full decimal mantissa — the same
correctly-rounded decimal→f64 path Lean uses for source literals, and the one
the trusted base's `toReal_ofScientific_close` speaks about.

## Honest scope

Parsing carries NO soundness claim: nothing here checks that the file honestly
records a real training run (garbage in, garbage out).  The verified surface is
unchanged — whatever parses feeds the SAME bridged `runTraceChecks` Bool
(C83 core / C84 bridge), so a PASS on a parsed trace means exactly what
`VerifyTrace.allOk_feeds_whole_run` says FOR THAT PARSED DATA: if the recorded
rows/ratios/budget pairs are what the run actually produced, the whole-run
error interval follows (with C74's disclosed plumbing hypotheses).
-/

namespace Puffer.Check

/-! ### Decimal Float literals -/

/-- Consume leading decimal digits from a char list, accumulating their value:
    returns `(value, digitCount, rest)`.  The digit COUNT (not the value's
    magnitude) is what positions the decimal point, so leading zeros in a
    fraction (`1.05`) are handled correctly. -/
def takeDigits : List Char → Nat → Nat → Nat × Nat × List Char
  | [], acc, n => (acc, n, [])
  | c :: cs, acc, n =>
    if c.isDigit then takeDigits cs (acc * 10 + (c.toNat - '0'.toNat)) (n + 1)
    else (acc, n, c :: cs)

/-- Parse a decimal Float literal: `[+-]? digits* (. digits*)? ([eE][+-]?digits{1,4})?`
    with at least one mantissa digit required.  Built on `Float.ofScientific`
    over the FULL decimal mantissa (integer and fraction digits concatenated),
    so the result is the correctly-rounded f64 of the written decimal — e.g.
    `parseFloat? "1e300"` is bit-identical to the source literal `1e300`.
    Returns `none` on any malformed input (empty, stray characters, missing
    digits, exponent longer than 4 digits). -/
def parseFloat? (s : String) : Option Float :=
  let cs := s.toList
  let (neg, cs) := match cs with
    | '-' :: r => (true, r)
    | '+' :: r => (false, r)
    | _ => (false, cs)
  let (ip, ipLen, cs) := takeDigits cs 0 0
  let (fp, fpLen, cs) := match cs with
    | '.' :: r => takeDigits r 0 0
    | _ => (0, 0, cs)
  if ipLen == 0 && fpLen == 0 then none
  else
    let m := ip * 10 ^ fpLen + fp
    let mk (decExp : Int) : Float :=
      let f := if decExp < 0 then Float.ofScientific m true decExp.natAbs
               else Float.ofScientific m false decExp.toNat
      if neg then -f else f
    match cs with
    | [] => some (mk (-(fpLen : Int)))
    | c :: r =>
      if c == 'e' || c == 'E' then
        let (eneg, r) := match r with
          | '-' :: r2 => (true, r2)
          | '+' :: r2 => (false, r2)
          | _ => (false, r)
        let (ev, evLen, r) := takeDigits r 0 0
        if evLen == 0 || evLen > 4 || !r.isEmpty then none
        else some (mk ((if eneg then -(ev : Int) else (ev : Int)) - (fpLen : Int)))
      else none

/-! ### The trace file layer -/

/-- Everything `runTraceChecks` consumes, parsed from one trace file. -/
structure TraceInput where
  /-- Parameter-region radius `R` (the `region` line). -/
  region  : Float
  /-- Clip-margin lower bound `tLo` (the `clip` line). -/
  clipLo  : Float
  /-- Clip-margin upper bound `tHi` (the `clip` line). -/
  clipHi  : Float
  /-- The recorded steps, in file order (the `step` lines). -/
  trace   : Trace
  /-- The `(bound, cap)` budget pairs, in file order (the `budget` lines). -/
  budgets : List (Float × Float)
  deriving Repr

/-- Split a line into whitespace-separated tokens (spaces, tabs, and a
    trailing CR all separate; empty tokens dropped). -/
def tokenize (line : String) : List String :=
  ((((line.replace "\t" " ").replace "\r" " ").splitOn " ").filter (· != ""))

/-- Fold the tokenized significant lines into a `TraceInput`, fail-closed:
    any unrecognized line shape, malformed Float, duplicate `region`/`clip`,
    or (at the end) missing `region`/`clip` yields `none`. -/
def parseLines : List (List String) → Option Float → Option (Float × Float) →
    List StepRec → List (Float × Float) → Option TraceInput
  | [], some R, some lohi, steps, budgets =>
      some { region := R, clipLo := lohi.1, clipHi := lohi.2,
             trace := steps.reverse, budgets := budgets.reverse }
  | [], _, _, _, _ => none
  | toks :: rest, reg?, clip?, steps, budgets =>
    match toks with
    | ["region", r] =>
        if reg?.isSome then none else
        (parseFloat? r).bind fun R => parseLines rest (some R) clip? steps budgets
    | ["clip", lo, hi] =>
        if clip?.isSome then none else
        (parseFloat? lo).bind fun l => (parseFloat? hi).bind fun h =>
          parseLines rest reg? (some (l, h)) steps budgets
    | ["budget", b, c] =>
        (parseFloat? b).bind fun bv => (parseFloat? c).bind fun cv =>
          parseLines rest reg? clip? steps ((bv, cv) :: budgets)
    | "step" :: ratio :: row =>
        (parseFloat? ratio).bind fun rv =>
          (row.mapM parseFloat?).bind fun rowv =>
            parseLines rest reg? clip? ((rowv, rv) :: steps) budgets
    | _ => none

/-- Parse a whole trace file (format in the module docstring).  Fail-closed:
    `none` on ANY malformed content.  Blank lines and lines whose first token
    starts with `#` are skipped. -/
def parseTrace (contents : String) : Option TraceInput :=
  let toks := (contents.splitOn "\n").map tokenize
  let sig := toks.filter fun t => match t with
    | [] => false
    | w :: _ => !w.startsWith "#"
  parseLines sig none none [] []

/-! ### Build-asserted demos

The round-trip tests use Bool `==` (bit-level via `Float.beq`), so `true` below
certifies EXACT agreement with the corresponding source literal — not just a
matching printout. -/

-- Exact round-trips through `Float.ofScientific`.
/-- info: true -/
#guard_msgs in
#eval parseFloat? "1.5" == some 1.5

/-- info: true -/
#guard_msgs in
#eval parseFloat? "-0.25" == some (-0.25)

/-- info: true -/
#guard_msgs in
#eval parseFloat? "1e-3" == some 0.001

/-- info: true -/
#guard_msgs in
#eval parseFloat? "6.25E2" == some 625.0

-- Malformed literals are rejected.
/-- info: [none, none, none, none, none] -/
#guard_msgs in
#eval [parseFloat? "abc", parseFloat? "1.2.3", parseFloat? "1e",
       parseFloat? "", parseFloat? "--5"]

/-- The C82/C84 demo data as a trace file: same trace, region, clip margins,
    and (first two) budget pairs as `Core.lean`'s first build-asserted demo. -/
def demoTraceText : String :=
  "# puffer verify-trace demo (same data as the built-in demo)\n" ++
  "region 1.0\n" ++
  "clip 0.25 0.8\n" ++
  "step 0.75 0.5 -0.25\n" ++
  "step 0.5 0.9 0.1\n" ++
  "budget 1.0 1e300\n" ++
  "budget 2048.0 1e300\n"

-- The parsed file reproduces the demo aggregate Bool…
/-- info: some true -/
#guard_msgs in
#eval (parseTrace demoTraceText).map fun t =>
  (runTraceChecks t.trace t.region t.clipLo t.clipHi t.budgets).allOk

-- …and the parsed pieces are BIT-IDENTICAL to the demo's source literals
-- (trace rows and budget caps included: `parseFloat? "1e300" == capF`).
/-- info: true -/
#guard_msgs in
#eval (parseTrace demoTraceText).map (fun t =>
    (t.trace, t.region, t.clipLo, t.clipHi, t.budgets)) ==
  some ([([0.5, -0.25], 0.75), ([0.9, 0.1], 0.5)],
        1.0, 0.25, 0.8, [(1.0, capF), (2048.0, capF)])

-- Fail-closed: missing `clip` line, malformed Float, duplicate `region`.
/-- info: [false, false, false] -/
#guard_msgs in
#eval [ (parseTrace "region 1.0\nstep 0.5 0.1\n").isSome
      , (parseTrace "region 1.0\nclip 0.25 0.8\nstep zork 0.1\n").isSome
      , (parseTrace "region 1.0\nregion 1.0\nclip 0.25 0.8\n").isSome ]

end Puffer.Check
