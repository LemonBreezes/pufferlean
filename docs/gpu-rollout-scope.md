> **SUPERSEDED (2026-08-02): the device-env capability this document scopes was built, benchmarked
> (worth ~1.7× on breakout), and then REMOVED by product decision — the library steps envs on CPU
> worker threads, placement-matched with PufferLib. Kept as history; the commands/FFIs below no longer exist.**

# Scope: a GPU-resident rollout

The training step is now the whole update resident on-device (`cudaTrainUpdateFFI`): one upload, one
download per update. The **rollout is the last CPU boundary.** At large `numEnvs` it is ~80% of the update
(`squared_big`: hidden 256, 512 envs × 32 → the GPU step runs in ~1.4 s but the CPU rollout dominates), so
it caps the large-batch regime the resident step unlocks. PufferLib's throughput
comes from running the rollout on the GPU too — resident policy + device envs + device trajectory, no
host↔device transfer in the timestep loop. This scopes that.

> **Status — R0/R1/R2 done.** `squared`'s dynamics run on the device: `k_sq_reset` / `k_sq_observe` /
> `k_sq_step` in `ffi/puffercuda.cu` (state `N × [r,c,tick]`, one thread per env), bound as
> `cudaEnvSq{Reset,Observe,Step}FFI`. `verify-env-gpu` diffs them against the Lean `Puffer.Env.Squared`
> model over a `(r,c,tick)×action` grid (move / target / off-grid / timeout) — **bit-exact**: 750 step
> cases + 150 observe states + reset all Δ = 0 (integer arithmetic + one-hot are exact). **R2:** the
> categorical sampler `k_sample` (splitmix64 `hash(rng+(n+1)·G)` + max/sumexp/cumulative/logp), bound as
> `cudaSampleActionsFFI`; `verify-sample-gpu` vs the CPU `sampleActionsBatchFFI` — **actions + values
> bit-exact**, logp to transcendental ULP (~1e-16, device `exp/log` vs libm). The env-ABI generalization
> (envId dispatch, R5) and residency (`ts_buf`, R3) are deferred; these test FFIs round-trip host↔device.
>
> **R3 done.** `cudaRolloutSquaredFFI` (`ffi/puffercuda.cu`) runs the whole `squared` rollout on the
> device: per timestep `k_sq_observe → mlp_forward_dev (f32/bf16 gemm32) → k_sample → k_sq_step`
> (+ auto-reset), state resident, filling the env-major SoA columns `[obs; act; val; logp; rew; term]`
> (the layout `cudaTrainUpdateFFI` reads). `verify-rollout-gpu` diffs it against a Lean orchestration of
> the SAME kernels (round-tripped per timestep via `cudaMlpForwardFFI`) — **bit-exact** (obs + all scalar
> columns Δ = 0), isolating the driver's chaining/RNG (`rng+s·N·G`)/indexing/auto-reset from the (expected)
> f32-forward drift vs the CPU f64 rollout. `bench-rollout-gpu`: the device rollout is **3–10× the CPU
> `vecRolloutBatchedSoA`** (N=1024 → 10.2×), the gap widening with N. Buffers are fresh `cudaMalloc` per call.
>
> **R4 done — the squared PoC is complete: the whole PPO loop runs on the GPU.** `train-ppo-gpu-resident`
> (`trainPPOGpuResident`): per update `cudaRolloutSquaredFFI` (GPU rollout → SoA columns) → `buildBatchSoA`
> (GAE) → `cudaTrainUpdateFFI` (whole update resident on-device), `pm=[params;mom]` threaded across BOTH
> (the rollout forward reads `pm[0:P]`), host only does the GAE + column slicing + periodic eval.
> **Converges 200/200** (the CPU trainer's optimum; the f32/bf16 device rollout diverges from the CPU f64
> path at tolerance but trains the same PPO+Muon). **174k SPS vs `trainPPOGpu`'s 152k** (CPU rollout) at
> toy scale — the rollout forward/sampler/env are off the CPU; the gap widens with N (`bench-rollout-gpu`
> 3–10×). The column round-trip (rollout downloads / train uploads) + Lean GAE stay on the host — a
> fully-fused rollout→GAE→train FFI (columns never leaving the device) is a further optimization.
>
> **R5 done — the ABI generalizes.** A generic per-env dispatch `env_{reset,observe,step}_dev(envId, …)`
> in `ffi/puffercuda.cu` (0=squared, 1=chain_mdp) is now the single source of truth; the squared wrappers
> (`k_sq_*`) delegate to it, and a generic driver `cudaRolloutFFI(envId, D, sw, size, ic0, ic1, …)` runs any
> registered env. `verify-env-gpu` diffs BOTH envs vs their Lean models — **bit-exact** (squared 750
> cases; chain_mdp 120 cases + observe + reset, all Δ = 0). `trainPPOGpuResident` is generic
> (`train-ppo-gpu-resident {squared|chain_mdp}`): the whole loop on the GPU converges for **both** —
> `squared` 200/200 @ 173k SPS, `chain_mdp` 12.0 200/200 @ 169k SPS. **Device RNG for rng-reset envs is
> NOT built:** both toy envs reset deterministically (no RNG), so nothing exercises it; the ABI leaves room
> for a per-env RNG word in the state, but `memory`/`tmaze` (the reset-consuming envs) aren't ported — that
> lands with R6.
>
> **R6 (partial) — the CNN path is built.** `cudaCnnForwardFFI` (`ffi/puffercuda.cu`) is the resident CNN
> encoder forward: im2col → conv `gemm32` → relu+convB → pixel→filter transpose → dense → logits, the GPU
> twin of the CPU `cnnForward`. `verify-cnn-forward-gpu` diffs it vs the f64 CPU forward — **Δ < 1e-6**
> (f32 GPU vs f64 CPU), for multi-channel + strided (`C=3, 11×11, k3/s2, nF8`). This is the reusable
> conv-encoder any CNN env's rollout would slot in place of `mlp_forward_dev`.
>
> **Correction — breakout is an MLP env, not CNN.** `breakoutEnv.observe` is a flat `10 + numBricks`
> feature vector (physics scalars ++ brick-flags), so breakout uses the MLP forward (`mlp_forward_dev`),
> not the conv encoder. The R6 CNN encoder above serves the actual *image* envs (`maze`/`snake`);
> breakout's port is **physics-only**.
>
> **Breakout physics — DONE, bit-exact (`verify-breakout-gpu`).** The full swept-collision step is ported
> to the device (`ffi/puffercuda.cu`, the `bk_*` block): `bk_calcVline`/`bk_calcHline` swept segments →
> `bk_calcBrickCollision` (4-edge) → `bk_calcAllBrick` (swept-AABB rectangle sweep) → `bk_calcAllWall` →
> `bk_calcPaddle` (sin/cos bounce via the verified `d_sinf`/`d_cosf`) → `bk_destroyBrick`/scoring →
> `bk_checkWallBounds` → `bk_handleCollisions` → `bk_stepFrame` (fire+`d_randR` / paddle / Euler / ball-loss
> / terminal) → `bk_resetRound`/`bk_cReset`/`bk_init` → `bk_observe`, plus `d_f2i` (trunc) and `d_min`/`d_max`.
> Every op uses `double` arithmetic with `d_r32` (=f32) at exactly the Lean model's `f32(...)` sites; the
> pure-CI **discard** semantics (a non-winning paddle `calcHline` must not mutate the shared collision) are
> preserved under pointer mutation via a `tmp` copy. State layout `N × [16 scalars | numBricks brick-flags]`,
> config `double[17]`; FFI `lean_cuda_breakout_{reset,step,observe}` + Lean `cudaBreakout*FFI`. **The precision
> crux (`d_sincos_core`, a bit-exact port of `Puffer.Numeric.SinCosF`; `verify-sincosf-gpu` 0/4201) is the
> foundation.** Verification: `verify-breakout-gpu` steps N=16 envs (varied seeds; even envs track the ball,
> odd flee it) in lockstep from `init` for T=12000 frameskip-1 frames — **0 mismatches** across all 124 state
> fields + reward + terminal (192k steps) + `observe`, plus a crafted-state battery that deterministically
> fires the `score==maxScore` terminal and all three `checkWallBounds` escape branches. An adversarial
> 5-agent faithfulness audit (device↔Lean, one per cluster) found **zero divergences**; the coverage-gaps it
> flagged (observe, terminal/cReset, half-max reset, BACKWALL, speed cap) are now all **proven exercised**
> (`bounces=1492 terminals=200 halfMaxResets=4 backwalls=8 maxSpeed=448 maxScoreTerm=true`).
>
> **Breakout resident rollout — WIRED (`verify-rollout-breakout-gpu`, `train-ppo-gpu-resident breakout`).**
> `lean_cuda_rollout_breakout` runs the whole breakout rollout on-device (bk_observe → f32 → mlp_forward_dev
> → k_sample → bk_step, env-major SoA columns), and `trainPPOGpuResidentBreakout` drives the whole PPO loop
> on the GPU (resident rollout → GAE → `cudaTrainUpdateFFI`). Two breakout-specific pieces resolved: (1) the
> **17-field config** rides a `double[17]` array (`breakoutCfgArr`), the ABI's config-array extension — breakout
> doesn't fit the `(size,ic0,ic1)` slot; (2) **state residency** — breakout episodes span the horizon, so the
> env state is threaded IN/OUT across updates (the toy envs re-center each update). On the rng-reset question:
> the device uses PufferLib-faithful **internal auto-reset** (bk_step's cReset keeps the env's `rand_r` stream
> — C `c_step` self-resets keeping `env->rng`), so terminals consume NO trainer rng; only the N per-env seeds
> are drawn at init. (The CPU Lean `vecRolloutBatchedSoA` instead calls `env.reset` on terminal, re-seeding —
> a harness artifact that diverges from PufferLib; the resident device path is the more faithful one.)
> `verify-rollout-breakout-gpu`: the fused driver is **bit-exact** vs a Lean orchestration of the same kernels
> (obs/scalars/final-state max|Δ|=0). Smoke run (64 envs × 128 × 150 updates): 1.23M env-steps in 4.8s =
> **254k SPS** for the whole GPU loop, batch reward climbs 2→45, eval 19/20 episodes survive — trains
> end-to-end on the device.
>
> **Convergence-parity study (GPU-resident vs CPU trainer, identical config/seed, 200 updates).** Both
> `train-ppo-gpu-resident breakout` and `train-ppo-cpu breakout` (hidden 128, 64 envs × 128, epochs 2,
> numMB 4, lr 0.01, γ0.99 λ0.95, seed 0x1234) track the SAME greedy-return band (5.8–8.4) throughout and
> finish within noise: **GPU final eval 6.93** (len 1440, 29/30 survive) vs **CPU 5.70** (len 1290, 30/30)
> — a 30-episode mean on a high-variance return. Same policy quality, same ~1300–1450 episode lengths. The
> GPU-resident loop reaches it **9.4× faster**: 1.64M env-steps in **6.6 s (248k SPS)** vs **61.9 s (26k
> SPS)** on CPU. (Neither return climbs much in 200 updates — expected for hard sparse-reward breakout with
> a flat-feature MLP and horizon T=128 ≪ episode length ~1400; the point here is that the two backends
> converge IDENTICALLY, which they do. Absolute performance is a curriculum/capacity/horizon question.)
>
> **Breakout ACTUALLY LEARNS (tuned run).** The plateau above was a horizon/credit-assignment problem, not
> a wiring one. With **frameskip=4** (episode ~350 agent-steps instead of ~1400, so γ0.99 no longer discounts
> episode-end rewards to ≈0), **hidden 256**, **lr 0.008 / epochs 3**, 128 envs × 1500 updates (~25M
> agent-steps), the GPU-resident trainer takes greedy return from **7 (random init) → 320** (final eval, 50
> greedy eps, 49/50 survive, mean episode length 1697) — a **~45× improvement**, reproducible (313→321
> across runs), of a max 864. Batch reward Σ climbs 207→3098. Run via `tune-breakout-gpu 128 128 1500 256 8
> 100 3 4 1 150 6 50 990 4` (~130 s at 188k SPS). Key finding: the cosine-LR schedule length = `updates`
> matters — a SHORT/fast-anneal schedule (1500) consolidates into a good policy, while a LONG one (3000, all
> else equal) holds lr high, thrashes, and ends far worse (greedy 13.8). Tuning infra: `mkBreakoutEnv cfg`
> (varies frameskip without touching the verified physics — bk_step already loops `cfg.frameskip`) and the
> parametrized `tune-breakout-gpu` CLI.
>
> **Host-glue C-slice (large-env regime unblocked).** The resident trainers extracted the SoA columns from
> the rollout result with an interpreted `mk ((Array.range (NT·D)).map …)` walk that dominated host time and
> capped batch size. `Puffer.Float.FFI.sliceFFI` (`lean_ffi_slice`, a memcpy) replaces it: glue/update drops
> 44→19 ms at 128 envs (2.3×), 74→31 ms at 256 envs (2.4×), and 512 envs (NT=65536) now runs at **507k SPS**
> with glue a minority (GPU-train dominant). Bit-identical training data (L1 update-1 batch reward = 207;
> `verify-rollout-breakout-gpu` still max|Δ|=0). Residual glue is now GAE + perm-build. This makes the
> large-env runs (the stable path to chasing the 864 ceiling) practical.
>
> **Large-env run — 320 → 460 (53% of the 864 ceiling).** With big batches now cheap, the key was scaling
> the learning rate WITH the batch: at 512 envs (4× L1's 128), L1's lr 0.008 *missed* the takeoff (eval 27),
> but **sqrt-scaling to lr 0.016** restored it → eval 345. Adding capacity (**hidden 512**) → 396, and the
> longer **1500-update** fast-anneal → **final eval 460.2 / 864** (50/50 greedy episodes survive, mean
> episode length 2114; greedy curve 4 → 102 → 271 → 411). Progression 128→512 envs: 320 → 345 → 396 → 460 —
> a 66× lift over the random-init baseline (~7). 98M agent-steps in **232 s at 424k SPS** (GPU-train now the
> dominant cost, 54%; glue 33%). Winning config: `tune-breakout-gpu 512 128 1500 512 16 100 3 4 1 150 6 50
> 990 4`. Levers, in order of impact: frameskip 4 (credit assignment) → batch+lr scaling → capacity → schedule
> length. Further headroom toward 864 likely needs a conv/recurrent encoder and/or curriculum — an ML
> question now, not an infra one.

## Where the rollout stands today (`vecRolloutBatchedSoA`)

Per timestep, over the `N` env instances:

| Step | Impl today | Resident-ready? |
|---|---|---|
| `env.observe s → obs` | **Lean, env-specific** | no — must port |
| batched policy forward `obs → logits` | OpenBLAS (`mlpForwardBatchBlasFFI`); a resident-weights cuBLASLt bf16 forward already exists (`lean_ffi_mlp_policy_load/forward`) | weights yes; obs/out stream today |
| categorical sample → `action, logp, value` | **C** (`sampleActionsBatchFFI`, host) | logic portable to a device kernel |
| `env.step s a → s', reward, terminal` (+ auto-reset) | **Lean, env-specific** | no — must port |
| append `obs/act/val/logp/rew/term` → SoA columns | Lean + `scatterObsFFI` (C) | columns already flat/resident-shaped |

Then `buildBatchSoA` (GAE, `ρ=c=1`) → advantages; `cudaTrainUpdateFFI` consumes the SoA columns. **The
training step is already resident. The two things that are genuinely CPU-only are `env.observe` and
`env.step` (Lean, env-specific) — plus the env `State` and the reset RNG.**

## The architecture — nothing crosses the bus in the timestep loop

Device buffers, allocated once (via the `ts_buf` cache) and living across the whole rollout, then handed to
the training step **in place**:

- `state` — `N × stateWords` (int32/float32; env-specific compact layout)
- `obs` — `N·D` f32 (one timestep's batch; or write straight into the traj column)
- trajectory SoA columns — `obs (NT·D)`, `act/val/logp/rew/term (NT)` — the **same layout
  `cudaTrainUpdateFFI` already reads**

Per timestep (host launches kernels + the one GEMM; no `cudaMemcpy`):

1. `k_observe<env>` : `state → obs` block for this timestep
2. **forward** : `obs → logits` — resident weights, **device-in/device-out** (extend the resident-policy
   forward so it reads/writes device buffers instead of streaming)
3. `k_sample` : `logits + device-RNG → action, logp, value`
4. `k_step<env>` : `state + action → state', reward, terminal`; **auto-reset on terminal**; write
   `reward/terminal` into the columns
5. append `obs/action/value/logp` into the resident columns at the timestep block

After `horizon`: `k_vtrace` (already built as M1 / `cudaVtraceFFI`) over the resident `rew/val/term`
columns → advantages, resident; then `cudaTrainUpdateFFI` runs on the columns with **no upload**. Params
stay resident across rollout **and** train — the host only orchestrates and reads `[params;mom]` out once
per update for logging/eval.

## The hard part: env dynamics on the GPU

`env.observe/step/reset` are Lean closures over an env-specific `State`. Each must become a `__device__`
function over a flat device state. This is the main cost and it is **O(#envs)**.

- **Toy envs are trivial integer dynamics.** `squared`: `State = {r,c,tick}` (3 int32/env); `step` = move +
  out-of-bounds/timeout check (−1) + target check (+1); `observe` = one-hot of the flattened cell; `reset` =
  center (deterministic, no RNG). `chain_mdp`: `{pos,tick}` (2 int32), a 1-D chain, one-hot obs, deterministic
  reset. Both port to ~30-line device kernels.
- **A device env ABI.** A `stateWords`-wide flat state + `__device__` `observe/step/reset`, driven by a
  generic per-timestep loop. Dispatch by an `envId` switch (or a per-env kernel). Start concrete (squared),
  factor the ABI once a second env (chain_mdp) lands.
- **Complex grids defer.** `overcooked / rware / breakout(CNN) / snake` have large/structured state and (for
  breakout) a conv encoder; port after the toy loop proves out. Breakout also needs the CNN forward resident,
  which is a separate encoder-residency task.

**Bit-exactness:** each device env kernel is bit-exact vs its Lean model on fixed `(state, action)` — that is
the unit test. The **full** device rollout is *not* bit-exact vs the CPU one (bf16/f32 device forward differs
from OpenBLAS at tolerance, and the float op order differs), but it is a valid, same-distribution rollout;
correctness rests on per-kernel bit-exactness + end-to-end convergence + a tolerance shadow diff.

## Milestones

| # | Deliverable | Effort | Verification | Dep |
|---|---|---|---|---|
| **R0** | Device `state` buffer + env-ABI skeleton (`envId` dispatch); a no-op `k_step`/`k_observe` callable via `@[extern]`, `ts_buf`-resident | **S** | `lake build puffer` green; a trivial device kernel round-trips a state buffer | — |
| **R1** | Device `observe/step/reset` for **squared** (integer dynamics, deterministic reset) | **M** | `verify-env-gpu squared`: fixed `(state, action)` grid → device `step/observe/reset` **bit-exact** vs `Puffer.Env.Squared.step/observe/init` (Nat/Int + one-hot are exact) | R0 |
| **R2** | Device **categorical sampler** kernel (splitmix64 `hash(s+(n+1)·G)` + max/sumexp/cumulative/logp), reading resident logits | **M** | `verify-sample-gpu`: fixed logits+`rng` → **bit-exact** vs `sampleActionsBatchFFI` / `softmax`+`sampleCat` (already the CPU reference) | R0 |
| **R3** | **Resident rollout driver**: per-timestep loop (`k_observe` → resident forward → `k_sample` → `k_step` + auto-reset), filling the resident SoA columns; device-in/device-out resident forward | **L** | `verify-rollout-gpu` (shadow): device columns vs the CPU `vecRolloutBatchedSoA` at **tolerance** (bf16 forward ⇒ ~1e-2 trajectory divergence, so compare distributions + a fixed-`u` deterministic-forward path bit-exact); env-steps/s vs CPU | R1, R2 |
| **R4** | **Fully-on-GPU train loop** (`train-ppo-gpu-resident`): device GAE (`k_vtrace` over resident cols) → `cudaTrainUpdateFFI` with params resident across rollout **and** train; host only reads `pm` out for eval | **M** | converges to the optimum on `squared` (win 200/200, matching the CPU trainer); SPS decomposition shows rollout no longer CPU-bound | R3 + `cudaTrainUpdateFFI` (done) |
| **R5** | Generalize the ABI to **chain_mdp** (2nd env) + a per-env **device RNG** stream for rng-reset envs (`memory/tmaze` sample a cue at reset) | **M** | `verify-env-gpu chain_mdp` bit-exact; device-RNG reset stream matches the Lean `rngNext` order for a reset-consuming env | R1, R4 |
| **R6** | Complex envs — `snake`, then the **CNN** path (`breakout`: resident conv encoder + device env) *(deferred)* | **L+** | per-env bit-exact + convergence; CNN encoder residency is its own sub-task | R5 |

**R0–R4 (squared, the fully-resident loop, proof of concept): ~2–3 focused weeks.** R5 (generalize + 2nd
env + device RNG): ~1 week. R6: open-ended, per-env.

## Precision / RNG / verification notes

- **RNG.** The sampler already uses device-portable splitmix64 with the O(1) state advance
  (`state += steps·G`), so the sample stream is reproducible on-device (R2 is a straight port). Envs whose
  `reset` consumes RNG (`memory/tmaze`) need a **per-env** device stream (R5); the deterministic-reset toy
  envs (squared/chain_mdp) need none — do those first.
- **Precision.** The device forward is bf16 (PufferLib default) or f32. Neither is bit-exact vs OpenBLAS, so
  the rollout trajectory diverges from the CPU one at tolerance — expected and fine (`train-ppo-vec-blas`
  already trains on a tolerance-close rollout). Keep an f32 **deterministic-forward** path so R3's shadow can
  isolate env-kernel bugs (bit-exact `step/observe/sample`) from forward-precision drift.
- **Verification ladder** (mirrors the training scope's, which held): unit — each device env/sampler kernel
  bit-exact vs its Lean/C oracle on fixed inputs; integration — R3 shadow diffs device vs CPU columns at
  tolerance; end-to-end — R4 converges 200/200. No new oracles needed: `Squared.step/observe`,
  `softmax`/`sampleCat`, `computePuffAdvantage`, and the existing `verify-*` harness are the references.

## Top risks

1. **Env porting is O(#envs) and env-specific** — the real cost, not the driver kernel. Mitigate: land the
   two trivial envs + a clean ABI first; only then decide whether the complex grids are worth it (they may
   stay CPU — a hybrid: GPU rollout for simple envs, CPU for the rest).
2. **State-layout generality.** `squared` is 3 ints; `overcooked/rware` are structured grids. A single flat
   `stateWords` ABI may not fit the complex envs → per-env kernels there. Don't over-abstract early.
3. **Warp divergence** in `env.step` branches (bounds/target/reset). N envs are independent (one thread each),
   so it's just branch divergence within a warp — a minor throughput hit, not a correctness issue.
4. **Losing bit-exactness vs the CPU rollout.** Accept it (bf16 forward), and lean on per-kernel bit-exactness
   + convergence. This is the same trade the BLAS rollout already made.
5. **Auto-reset + variable episode length** → a per-env state machine in `k_step`. Manageable (the CPU rollout
   already does exactly this per-env), but the terminal/reset RNG interleaving must match the intended
   semantics (deterministic-reset envs sidestep it).

## Alternatives (cheaper, partial)

- **Async CPU-rollout / GPU-train overlap.** Pipeline: while the GPU trains on update *k*'s buffer, the CPU
  rolls out update *k+1*. Hides the rollout behind the GPU step without porting any env. Caps at
  `max(rollout, step)` instead of `rollout + step` — a real win when they're comparable, but the rollout rate
  still bounds throughput. **Cheapest; no CUDA env work.**
- **Multi-thread the CPU env stepping.** The rollout's `env.step`/`env.observe` loop is single-threaded Lean;
  a task-parallel sweep over the N envs would use the 16 cores. Partial (the forward is already BLAS), no GPU.
- **Forward-only residency.** Use the existing resident-policy FFI (`mlp_policy_forward`) so the rollout
  forward doesn't re-upload weights — but `env.step` stays on the CPU, so obs/actions still cross the bus each
  timestep. Small win; already most of the way there via BLAS.

## Recommendation

Do **R0–R4 for `squared`** — the fully-resident rollout+train loop as a proof of concept — verifying
per-kernel bit-exactness and end-to-end convergence, and measure the large-batch rollout speedup (the
`squared_big` config is the target: rollout should fall from ~80% of the update toward the step's own time).
Then **R5** to prove the ABI generalizes (chain_mdp + device RNG). **Defer R6** (complex/CNN envs) until the
toy loop demonstrates the win — and seriously consider a **hybrid** (GPU rollout for the simple envs that
matter for throughput benchmarking, CPU rollout for the long tail of complex envs) rather than porting
everything. If the appetite for CUDA env-porting is low, the **async overlap** alternative captures much of
the end-to-end win for a fraction of the effort and is the pragmatic first move.
