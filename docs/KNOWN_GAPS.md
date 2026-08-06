# Known gaps vs PufferLib (measured 2026-08-04, same box, their `_C` rebuilt per env)

## laser_puzzle — FIXED (was a real gap; root cause was our action space)
RESOLVED 2026-08-04. Our adapter hardcoded `ocean_numactions() = 5`, but upstream declares
`#define ACT_SIZES {NUM_ACTIONS}` with `NUM_ACTIONS = ACTIONS_PER_CELL * INNER_ROWS * INNER_COLS
= 3*4*4 = 48` (6x6 board, 4x4 inner). Since `c_step` decodes `cell_idx = action / ACTIONS_PER_CELL`
(laser_puzzle.h:216), a policy limited to actions 0..4 could reach only 2 of the 16 inner cells —
so it plateaued and then decayed. After setting the action space to NUM_ACTIONS:
    ours 0.16 -> 0.76 @12M, and 1.43-1.49 @40M  vs  PufferLib +1.454 @12M
and after ALSO fixing the per-copy RNG (below): **1.52-1.62 @12M across seeds, EXCEEDING theirs.**

Second defect, same env: `allocate()` sets `env->rng = 0` (laser_puzzle.h:124), and our adapter
seeded `rng` BEFORE calling it — so every copy shared rng=0. `c_reset` picks its level with
`rand_r(&env->rng) % num_levels` (:260), so all 1024 copies drew the SAME level and, with every
episode truncating at exactly max_steps, stayed in lockstep: 1 distinct board across 256 copies for
the first ~480 steps. The reported episode_return was one level's value replicated across the batch,
and `--train.seed` did nothing whatsoever (which is why the original "consistent across 3 seeds"
finding was vacuous). Fixed by seeding after allocate(); seeds now genuinely diverge.
CAUTION: do not hoist or remove that `allocate()` call — it is also what makes max_steps correct
(NUM_ACTIONS=48); the adapter's own `max-steps` cfg line is dead code.

Verified the same class across every env whose ACT_SIZES is macro-defined
(double_pendulum 3, hex 121, lightsout 25, whackamole 25) — laser_puzzle was the only wrong one.

METHOD NOTE: an earlier audit compared ACT_SIZES as literal TEXT and passed this env, because
`{NUM_ACTIONS}` vs our `5` was not expanded. Compare RESOLVED VALUES, not macro spellings.

## Declines that are NOT defects — all three now checked against the reference
Measured at 12M steps, both sides on their own (byte-identical) configs, plus the random-policy
floor from `puffer env-log <env> 256 4000`:

| env | random floor | ours | PufferLib | verdict |
|---|---|---|---|---|
| lightsout | — | −0.65 | **−1.752** | we LEAD |
| maze | 0.058 | 0.100–0.124 | 0.126 | parity (both ~2x random) |
| drmario | −2.382 | −2.57 | **−2.531** | parity — *both below random* |

- **maze**: config verified identical (MinGRU 512x5L, 512 envs, horizon 256 — note `num_layers =
  5.88042` truncates to 5 on both sides). We sit at 0.10-0.12 against their 0.126, and both are
  ~2x the 0.058 random floor. Not a defect; the env is simply hard at this budget (upstream trains
  it for 3.4e8 steps).
- **drmario**: config identical (128x1L, 2048 envs, horizon 128). Ours −2.57, theirs −2.531 — the
  same number within noise. Both finish BELOW the −2.382 random floor, i.e. PufferLib's own trainer
  also ends worse than random on this env at this budget. That is an env/reward property, not an
  implementation gap on either side.

No outstanding learning-gap leads remain. Every env that looked like a defect has now been either
fixed (laser_puzzle) or explained by measurement.

## Zero-reward envs — all three diagnosed, none is a defect (source-verified)
- **convert**: `log.episode_return` is declared (`convert.h:15`) and **never assigned anywhere** —
  upstream's `my_log` exports it regardless, so PufferLib prints 0.0 for this env too. Its `score/n`
  and `perf/n` are pinned to exactly 1.0 by construction (incremented in the same block as `n`), so
  only the batch reward carries a monotone signal. Sweep now uses it: reads 50 → 54 over 183 updates.
- **slimevolley**: the field IS written (`slimevolley.h:573`, `episode_return = 5 - right->lives` =
  points scored). With `num_agents=1` the opponent is the hand-tuned built-in bot, so a weak policy
  genuinely scores ~0. Widening the sample makes it appear (0.0 at 32 copies, 0.0063 at 256). A zero
  here is a true statement about the policy, not a bug. Caveat: these fields are ASSIGNED not
  accumulated while `n++`, so the aggregate divides a last-episode value by total episodes —
  magnitudes are diluted, though a zero stays a zero.
- **robocode**: zero-sum confirmed exactly — both reward sources are antisymmetric between the two
  slots (`robocode.h:380-386`, `:584-591`) and `add_log` sums both (`:179`), so `episode_return ≡ 0`
  in exact arithmetic. The adapter's declared log fields are already complete. The right metric is
  `score` (mean damage dealt, rises with skill), NOT `slot_0_score` — that is a win rate that
  converges to 0.5 by symmetry under selfplay.

## Metric artifacts, not learning problems
- **minimal** reports 0.0 because its `Log` has no `episode_return` field at all (perf/score/n only).
- **target** reports exactly 1.0 and **rware** exactly 2.0 every update: their logs record a per-goal
  constant rather than an accumulating return. Report `score`/`perf` for these instead.

## Correct-by-design zeros
- **robocode**: 1v1 selfplay with exactly zero-sum rewards, so any summed reward is identically 0.
  Needs a win-rate / per-slot-score metric to say anything useful.
- **chess**: no action-mask channel in our plugin ABI — see the banner in `ocean/chess/adapter.c`
  for the cost/benefit (only 2 upstream envs use masks, and the other one is unported).


## double_pendulum — NOT a defect (I called this a bug; the reference refutes it)
Measured 2026-08-04, 12M steps, matched configs:
    random-policy floor 11.7-12.5   |   OURS ~6.3   |   PufferLib 4.328   |   ceiling 900
Both implementations finish BELOW the random baseline, and ours is ~1.5x PufferLib's. So the flat
~6.3 is a property of this env under its shipped PPO config, not a gap in our trainer. It is a
swing-up task (the pendulum starts hanging down, theta = pi +/- 0.08, double_pendulum.h:107-109) and
neither implementation escapes the local optimum of a constant push within 12M steps.

I recorded this as a "flat NON-LEARNER / real bug" on the strength of the random-floor comparison
ALONE, without first measuring the reference — the exact discipline I had just written down after
lightsout. Below-random is a red flag worth investigating; it is not by itself evidence of a defect
on our side. Both checks are needed: the random floor tells you whether learning happened, the
reference tells you whose problem it is.

## CLOSED (2026-08-05): minimal — the recurrent MD core reached parity
Re-measured after the recurrent multi-discrete core (`trainPluginEnvMinGRUMD`) shipped and
`Exe/Puffer.lean:2028` began routing MD envs through it. minimal now trains as a `128x4L MinGRU`
MD policy (header: `MULTI-DISCRETE (2 heads #[9,5]) MinGRU (recurrent 128x4L)`), not the old
`66->128->15 MLP`. At the identical config (8192 agents, horizon 64, 12M steps), 3 seeds:
    ours = -207 / -204 / -230   vs   PufferLib reference -213.4
i.e. at parity, up from -468.6. The gap was exactly the diagnosed cause below (1-layer MLP vs
4-layer recurrent), and building the recurrent MD core removed it. Original finding retained below
for the record.

### (historical) OPEN (CONFIRMED, real): minimal — we are ~2.2x worse than the reference
Measured 2026-08-04 at MATCHED config (both 8192 agents, horizon 64, 12M steps):
    PufferLib score = -213.4   |   ours = -468.6
`score` is -(mean steps between catches), so less negative is better: they catch about twice as
often. Unlike double_pendulum/maze/drmario/lightsout, the reference here is clearly ahead, so this
is OUR gap. (Note `score` drifts with elapsed time, so only compare runs at the SAME agent count and
step budget — a 1024-agent run is not comparable to an 8192-agent one.)

LIKELY CAUSE — architecture, not numerics. Our trainer header for this run reads
`66->128->15 MLP`: a single-hidden-layer feed-forward net. PufferLib runs minimal with
`hidden_size 128, num_layers 4` through its MinGRU stack. minimal is MULTI-DISCRETE (2 heads {9,5}),
and our multi-discrete path has no recurrent option — `Exe/Puffer.lean` routes MD/continuous envs to
their MLP paths "regardless of `network`", so `num_layers = 4` is silently ignored for every MD env.
So we are training a 1-layer MLP against their 4-layer recurrent policy.

That would affect EVERY multi-discrete env, not just minimal (convert, drive, target, terraform,
moba, robocode, slimevolley, minimal). Worth checking those against the reference before assuming
the size of the problem. The fix is a recurrent multi-discrete core, which is a real feature, not a
bug fix — scope it deliberately.

## robocode: CLOSED — its win-rate metric is structurally unreachable here
`score` bounces because `n` is tiny (an episode closes only every ~30-45 updates at max_ticks 3000 vs
horizon 64) — real small-sample noise, not an aggregation bug. `perf`, `episode_return` and the slot
scores are all structurally dead in symmetric selfplay (episode_return is exactly zero-sum).

The recommended fix was `hist_score / hist_n` — win rate against a frozen historical opponent. That
is UNREACHABLE: those fields only accumulate when `env->tag > 0` (robocode.h:465), and NOTHING sets
`tag` — not our adapter, not upstream's `ocean/robocode/binding.c`, not `src/vecenv.h`. It is set by
PufferLib's PYTHON-side selfplay pool, which this port does not have. So hist_score/hist_n are
permanently 0 on both sides.

Conclusion: robocode has no absolute progress metric available to either implementation's C path.
`draw_rate` and `episode_length` are the only stable signals, and their direction is genuinely
ambiguous under symmetric selfplay (better aiming pushes them down, better evasion pushes them up) —
diagnostics, not a pass/fail. Its permanent DECLINES/noise verdict in the sweep is expected and
should not be chased further without porting the selfplay pool.


## RESOLVED: the MD BPTT gradient under bf16 — measured against single-discrete, tolerance calibrated
The MD GPU gradient check used to report max rel 0.325 on the bf16 tier and print CHECK. Settled by
running the SAME metric against the single-discrete path (`verify-mingru-grad-gpu`) on both tiers:

| path | f32 | bf16 | tier degradation |
|---|---|---|---|
| single-discrete | 1.1e-5 | 2.46e-2 | ~2200x |
| multi-discrete  | 6.4e-5 | 3.25e-1 | ~5100x |

So MD is ~2.3x more bf16-sensitive than SD — real, modest, and exactly what summing K per-head
log-probs into the joint before forming the PPO ratio predicts. Not a bug.

Two fixes came out of it, both evidence-led:
1. The check's relative error used a FIXED floor (`dd/(|g|+1e-4)`), so tensor-core noise on a
   component ~0.3% of the gradient max read as 32% error. It now floors at a fraction of the
   gradient's own magnitude (1% under the tensor-core tier, 0.1% in f32 — bf16's 8-bit mantissa is
   ~0.4% relative before GEMM accumulation), keeps an ABSOLUTE guard at the SD bar so a genuine
   error among skipped components still fails, and reports the floor and skip count.
2. The tensor-core tolerance is 1.2e-1, which is the 2.4x SD needs to give MD the same margin SD
   enjoys (SD passes at ~49% of its 5e-2 bar). The f32 bar stays tight at 2e-2, where MD sits three
   orders below it.
Also fixed along the way: `k_mg_ppo_b_md` accumulated the joint log-prob in float as `x-(m+log z)`
while `k_sample_md` (the source of `oldLogp`) accumulates in double as `log(exp(x-m)/z)`. Now matched
— it did NOT move the bf16 number, but it removes a real precision asymmetry and improved f32
6.4e-5 -> 2.6e-5.

## robocode: CLOSED — its win-rate metric is structurally unreachable here
`score` bounces because `n` is tiny (an episode closes only every ~30-45 updates at max_ticks 3000 vs
horizon 64) — real small-sample noise, not an aggregation bug. `perf`, `episode_return` and the slot
scores are all structurally dead in symmetric selfplay (episode_return is exactly zero-sum).

The recommended fix was `hist_score / hist_n` — win rate against a frozen historical opponent. That
is UNREACHABLE: those fields only accumulate when `env->tag > 0` (robocode.h:465), and NOTHING sets
`tag` — not our adapter, not upstream's `ocean/robocode/binding.c`, not `src/vecenv.h`. It is set by
PufferLib's PYTHON-side selfplay pool, which this port does not have. So hist_score/hist_n are
permanently 0 on both sides.

Conclusion: robocode has no absolute progress metric available to either implementation's C path.
`draw_rate` and `episode_length` are the only stable signals, and their direction is genuinely
ambiguous under symmetric selfplay (better aiming pushes them down, better evasion pushes them up) —
diagnostics, not a pass/fail. Its permanent DECLINES/noise verdict in the sweep is expected and
should not be chased further without porting the selfplay pool.


## OPEN: the MD BPTT gradient is much noisier under the bf16 tier than single-discrete is
`verify-mingru-md-grad-gpu` reports max rel **0.325** on the default (bf16) tier, against its own
5e-2 tensor-core tolerance — so it prints CHECK. With `PUFFER_MG_BF16=0` it is 6.4e-5, and the CPU
AD-vs-finite-difference check (`verify-mingru-md-grad`) is exact.

This is PRE-EXISTING and NOT caused by the fused-rollout migration: main and the migrated branch
report the identical 0.324689 unpinned and the identical 0.000064 pinned. It arrived with the
recurrent MD core itself.

The project's existing stance for the single-discrete BPTT is to verify the LOGIC in f32 and treat
bf16 as "that logic plus tensor-core GEMMs" — `tools/compare.py` pins `PUFFER_MG_BF16=0` for exactly
that check. The same stance arguably applies here, but 0.325 is far worse than the single-discrete
path's bf16 behaviour, and MD has a plausible amplifier: the joint log-prob sums K per-head
log-probs, so per-head rounding accumulates before the ratio is formed. Worth deciding whether that
is acceptable numerically or whether the MD head should accumulate its joint log-prob in f32.
Until then, run that check with `PUFFER_MG_BF16=0`.

UPDATE 2026-08-04 — the joint-log-prob hypothesis is REFUTED. There WAS a real asymmetry (the
gradient accumulated the joint log-prob in float as `x-(m+log z)` while `k_sample_md`, which produced
the `oldLogp` being differentiated, accumulates in double as `log(exp(x-m)/z)`), and it is now fixed
in `k_mg_ppo_b_md`. That improved the f32-pinned check 6.4e-5 -> 4.7e-5 but left the bf16 number
**bit-identical at 0.324689**. So accumulation precision is not the cause.

What the numbers actually say: max ABS error is only 0.001541 while max REL is 0.325, which implies
the offending component has magnitude ~0.005 — i.e. this is bf16 GEMM noise on a near-zero gradient
entry, where relative error is meaningless. That matches how the single-discrete BPTT check is
treated (`tools/compare.py` pins `PUFFER_MG_BF16=0` and calls bf16 "this logic + tensor-core GEMMs").
The likely correct fix is the CHECK, not the kernel: gate on absolute error, or on relative error
only for components above a magnitude floor. I have NOT made that change — changing a tolerance to
make a check pass needs more evidence than one session's inference, and the current CHECK output is
at least honest about the discrepancy.

## OPEN (REAL, found 2026-08-04): target and overcooked COLLAPSE late in training
Not classifier noise — I mis-called it twice by reading the sweep's 46-char truncated trend, which
cut off the ending. Full series at 12M steps, 1024 agents:

    target     (n)         404 1833 1783 2348 1011 2520 2204 2331 1955 1391 **212**
    overcooked (mean/step) 0.00108 0.00278 0.00494 0.0108 0.0124 0.0123 0.0125 0.00275 **0 0**

Both climb strongly and then fall off a cliff at the very end — overcooked to EXACTLY 0. A clean
rise-then-zero is not a plateau or seed variance; something is destroying the policy late in the run.

Both are MULTI-DISCRETE envs on the new recurrent/fused MD path, which makes a shared cause likely.
Candidates, cheapest first:
  1. LR annealing: `anneal_lr = 1` with cosine to `min_lr_ratio = 0.0` — the tail of training runs at
     ~0 LR, which should FREEZE a policy, not destroy one. If a collapse coincides with the last
     updates, suspect something that scales with 1/lr or divides by it.
  2. Entropy collapse in the joint multi-head distribution (K heads multiply, so the joint can sharpen
     far faster than any single head).
  3. Something in the MD fused rollout that degrades once the policy becomes near-deterministic
     (e.g. a sampler path exercised only when one head's probability mass saturates).
DISCRIMINATORS: my first run was INVALID — retracted. It reported four configs as bit-identical and
I concluded the flags were being ignored. Tested in ISOLATION (one flag per invocation, no shell
wrapper) every flag demonstrably works on this trainer at 4M steps:
    --train.learning-rate 0.0001 -> 0.000906   |   0.01 -> 0.00439
    --train.ent-coef      0.0    -> 0.00436    |   0.5  -> 0.00151
    --train.min-lr-ratio  0.0    -> 0.00439    |   1.0  -> 0.0108
So flag plumbing is FINE; the bit-identical result came from a bug in my own test harness (a shell
function wrapping `env`), not from the trainer. Both earlier conclusions drawn from it — "all three
hypotheses refuted" and "trainPluginEnvMinGRUMD ignores its settings" — are withdrawn.

LEADING HYPOTHESIS (back in play, now with evidence FOR it): the LR anneal. `--train.min-lr-ratio 1.0`
(constant LR) more than DOUBLES performance at 4M steps, 0.00439 -> 0.0108. The default anneals
cosine to `min_lr_ratio = 0.0`, so the tail of a run trains at ~0 LR — consistent with a curve that
climbs and then dies at the end. NEXT: run the full 12M with `--train.min-lr-ratio 1.0` and check
whether the terminal 0 0 disappears. That is a single run and it either confirms or kills it.

(superseded) Original discriminators: re-run with `--train.min-lr-ratio 1.0` (constant LR) to test (1); raise
`--train.ent-coef` to test (2); `PUFFER_MG_MD_FUSED=0` to test (3) — that last one isolates the fused
arm from the recurrent core, since the non-fused MD path predates it.
robocode's DECLINES verdict is separate and remains the known small-`n` noise on `score`
(14.8 14.9 9.55 13.6 10.5 11.3 9.4 9.6 10.9 11.8 — flat within its sampling error); its only real
metric is `hist_score/hist_n`, still unread by the sweep.

METHOD NOTE: the sweep truncates its trend to 46 characters, which HID a terminal collapse behind a
healthy-looking opening. Always inspect the full series before judging a verdict.


## RESOLVED (not a defect): target/overcooked tail collapse is the LR anneal freezing a transient dip
Chain of evidence, 12M steps, overcooked:
  default (cosine -> min_lr_ratio 0.0):  0.0011 ... 0.0125 0.0028 0 0     (dies)
  --train.min-lr-ratio 1.0 (constant):   0.0011 ... 0.0124 1.2e-5 4.5e-5 0.0079 0.0124  (recovers)
  lr floored at 1e-8*lr so no step is literally zero:  IDENTICAL to default (0 0)

The floor test is the decisive one: a literal lr==0 step is NOT the cause. Note the constant-LR run
also dips hard (1.2e-5, 4.5e-5) and then climbs out. So the policy passes through a transient bad
state in both cases; with a constant LR it escapes, and with the cosine anneal it is frozen there
because the tail of the schedule has almost no learning rate left. That is ordinary RL dynamics
meeting an aggressive schedule — upstream's own schedule, which we inherit deliberately
(anneal_lr = 1, min_lr_ratio = 0.0, and overcooked.ini sets anneal_lr explicitly).

So there is nothing to fix in the trainer. What IS worth knowing:
  - Reading the FINAL update as "the result" is wrong under a cosine-to-zero schedule; the peak or a
    late-band median is the honest summary. The sweep already classifies on band medians.
  - PufferLib's own overcooked SEGFAULTS on this box (rc=139 building/running their _C at 12M), so no
    cross-implementation reference exists for this env. Worth revisiting if their build is fixed.
  - If a user wants this env to converge rather than freeze, --train.min-lr-ratio 1.0 more than
    doubles its final result.


## OPEN (upstream-shared, gated OFF): moba does not reproduce at a fixed seed
Five reruns of ONE binary, seed 1003, 12M steps: 19.4 / 101.3 / 68.5 / 105.2 / 84.0. Not the
trainer, not multi-discrete/MinGRU/the fused arm (breakout, target and drive — the other MD MinGRU
envs — are bit-identical across reruns; `PUFFER_MG_MD_FUSED=0` does not remove it), and not the GPU
(compute-sanitizer memcheck/initcheck/racecheck are all clean on a moba run — correctly so, the bug
is host C). It is moba's own env code reaching PROCESS-GLOBAL mutable state from inside `c_step`,
which only misbehaves because the rollout driver steps disjoint env ranges from `nbuf` concurrent
pthreads (ffi/puffercuda.cu, `mg_buf_worker` / `mgsub` pool). moba is the only env of ours whose
STEP path touches process-global state.

Two independent causes. Measured separately, EACH ALONE still leaves moba nondeterministic:
1. **libc `rand()`** — ocean/moba/moba.h:669 (`spawn_player`, respawn), :862 (`spawn_creep`),
   :892 (`spawn_neutral`), plus `init_moba`'s CachedRNG fill. One global LCG, many consumer
   threads => which env receives which draw is scheduler-dependent.
2. **The shared 256 MB `ai_paths` BFS cache** — ocean/moba/moba.h:608 in `move_towards`; allocated
   once and shared in ocean/moba/adapter.c, exactly as upstream binding.c does. Sharing it is
   sound: `bfs` treats only WALL cells as obstacles and the wall layout is identical in every env
   copy and constant in time, so a *filled* block is a pure function of its destination (verified:
   filling the same 400 blocks from four differently-advanced envs, forward and reverse order, gives
   one identical hash). The race is in the FILL. Two threads bfs the same block and bfs's
   `paths[r][c] != 255` prune makes the stored direction interleaving-dependent; and — the dominant
   term — a thread READS a block another is still filling. `bfs` writes the destination cell TWICE,
   `atn` (0) when it pops it and 8 after the loop, so a reader catching the transient gets "step
   north" where the finished block says "you are already there".

**The default is upstream-compatible and therefore still racy, by deliberate choice.** The ocean
envs are fixed by PufferLib and are our comparison baseline; a "fixed" moba is a different
environment from theirs, and swapping `rand()` for a per-env generator changes the actual draws, not
just the synchronisation. Note also that **PufferLib's own config/moba.ini ships `num_threads = 16`,
so upstream almost certainly hits both races too** — the racy behaviour IS the baseline. Worth
reporting upstream: both are ~10-line fixes there, and cause 2 in particular (the torn read of the
twice-written destination cell) is a genuine bug rather than a reproducibility nicety.

`PUFFER_MOBA_DETERMINISTIC=1` turns both fixes on: per-env xorshift64* for `rand()`, and a publish
protocol for the cache (fill each block once under a striped mutex, publish with a release store,
never read a block before its acquire load says published). Read once at env construction, never on
the step path.

    default (OFF): byte-identical to main — same `env-log moba` hash, and same trainer output on the
                   serial path (PUFFER_MG_ROLL_BUFFERS=1) at seeds 1003/1007/1011/1042. Still
                   nondeterministic under threads, exactly like main. SPS 1.0025x main (free).
    ON:            10 reruns at seed 1003, 12M steps -> ONE distinct output. A GPU-free 7-thread env
                   driver reproduces the 1-thread trajectory exactly at all 300 steps, cache content
                   and filled-block set included. SPS 1.0179x main.
    Score is unaffected either way: 16 seeds each, before median 148.8 / mean 176.0, after median
    100.2 / mean 140.1, Mann-Whitney p=0.29 — the seed-to-seed spread (9 to 439) dwarfs it. This
    buys reproducibility, not performance.

NEXT: moba's real open problem is a peak-then-collapse — episode_return climbs to ~390 by update 100
and falls to ~24 by update 183. With the flag on this is now deterministic and therefore bisectable
for the first time; that is the tractable next target.

## 2026-08-05 — PufferLib behavior-parity sweep (audit + fixes)

A three-dimension audit (config keys / CLI+capability / algorithm) then a parallel fix pass. CLOSED and verified (parity 22/22 throughout):
- **anneal_ent_coef / min_ent_coef_ratio** — were dropped; the only config key that changed a default-path result (chess). Now parsed + cosine-annealed in both MinGRU trainers (mirrors anneal_lr). Default off ⇒ all non-chess envs bit-identical.
- **Dashboard for ALL trainers** — was MinGRU-only; now the 4 feed-forward trainers (MLP/MD/Cont/LSTM) render the live dashboard too, SPS-neutral (render-cadence loss readback) and determinism-preserving. Default-on, PUFFER_PLAIN_LOG=1 for the parseable lines.
- **Checkpointing** — `puffer train` now writes `checkpoints/<env>/<seed>/<step>.bin` (our format), `--load` resumes, `--train.checkpoint-interval N`. Fixed two bugs found in verification: the save must DOWNLOAD device-resident weights (host wFlat is empty on the resident path) and slice the first P (not 2P incl. momentum).
- **eval mode** — `puffer eval <env> [--load]` loads a checkpoint and prints a headless score (MinGRU discrete). Capped to 64 envs / ≤5000 steps so the CPU-side per-step sampling returns in seconds.
- **CLI fidelity** — unknown flag/mode now error + exit 1 (was a silent forwardDemo); startup probe lines "Detected discrete action space with N heads" / "Num workers: N" on the MinGRU path.

DEFERRED (deliberate, with reasons):
- **Newton–Schulz form** `1/(‖x‖+eps)` vs PufferLib's `1/max(‖x‖,eps)`: numerically immaterial (differs only when ‖update‖≲eps, never) AND matching it breaks the formal-proof tower in Puffer/RL/ (theorems restate the old seed, close by rfl). Not worth trading the verification layer for zero measurable change. Branch preserved.
- **Feed-forward config keys** (replay_ratio / prio_alpha / prio_beta0 / num_layers on continuous/MLP/LSTM): those trainers push the PPO loop into a C kernel with no clean seam, have no prioritized-replay sampler, and are single-hidden-layer by construction. Honoring these = building prioritized-replay + a recurrent-Gaussian/deep core (a feature) for a path that already BEATS PufferLib's numbers. Deferred.
- **eval** for continuous/LSTM; PufferLib `.bin` interop; the interactive render/gif loop; sweep/match/wandb/multi-GPU — out of scope (workflows we don't ship).
- **Startup probe lines / checkpoint save** on the feed-forward trainers (MinGRU pair only) — to bound merge scope; easy follow-up.
