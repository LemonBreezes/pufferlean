# Native rollout loop — design document

*2026-08-02 · status: LADDER CLIMBED; batched-across-buffers SCOPED — no build recommended, see §13-14.*

## 1. Why

The placement-matched benchmark (docs/benchmark.html, commit `3570286`) leaves us at
**0.76× / 0.74× / 0.57×** of PufferLib's native trainer on breakout / pong / squared at identical
hyperparameters, CPU envs both sides. The update phase (GPU-resident V-Trace/prio/BPTT/Muon,
de-synced, side-streamed) is at or near parity; **the gap is the rollout loop**. Incremental
attacks on the current loop are exhausted — the span-dilation finding (project log, 2026-08-02)
showed our per-launch profiles were co-residency mirages and only aggregate work/structure changes
move the wall.

## 2. What their loop actually is (dissected, with line refs in the workflow corpus)

PufferLib's native rollout, per buffer per step, in steady state:

1. **One `cudaGraphLaunch`** — the whole forward (obs cast → encoder GEMM → MinGRU layers →
   fused logits+value GEMM → multinomial sampler → action cast) is captured as **one graph per
   (t, buffer) pair** at init, via a fake warmup rollout with no env stepping. `horizon ×
   num_buffers` graph execs, replayed forever. Per-step pointers are *baked* (that is why
   per-(t,buf) graphs); `lr`/`ent_coef` are read through **device pointers** so annealing never
   invalidates a graph; sampling uses **persistent curand Philox states in device memory** so no
   per-launch seed argument exists.
2. One `cudaMemcpyAsync` D2H of f32 actions into pinned host memory.
3. **One `cudaStreamSynchronize`** — the only sync per step per buffer.
4. OpenMP env stepping (8 threads per buffer) writing obs/rew/term **directly into pinned slices**.
5. Three un-synchronized `cudaMemcpyAsync` H2D returns — stream ordering fences them before the
   next step's graph launch; the host immediately proceeds.

Structure: **2 persistent buffer threads** (atomic spin handshake with the main thread), each
owning 2048 agents → **one 2048-row forward per step per buffer**. All rollout/train storage is
**bf16** (f32 master weights, f32 accumulate). Host API calls per step per buffer: 1 graph launch
+ 4 memcpyAsync + 1 sync. Ours today: ~16 launches + 16 syncs + worker barriers across 16 threads.

Two of our past refutations are *explained, not contradicted*, by their design: our rollout-graph
attempt failed because we passed per-step rng seeds as kernel arguments (they keep rng state
device-resident); our wide-forward refutation predates the WMMA kernel and never tested width
*decoupled* from env fan-out (their env parallelism is nested OMP inside a buffer, not more
buffers).

## 3. Hard constraints (any design must honor or explicitly renegotiate)

1. **Lean is the captain.** Per-update orchestration (lr anneal, rng threading, cadence, config)
   stays in Lean; no training formula leaves Lean.
2. **Env plugin ABI unchanged.** 46 dlopen'd envs; `step_range_{f64,f32,u8,bf16}`; obs kinds;
   reset semantics.
3. **Verification discipline.** f64 Lean/AD oracle → C twin → GPU tier chain; the byte-identity
   harness (fixed-seed multi-update runs, `max|Δ| = 0` for bit-identical claims); determinism;
   sampler rng keyed by global row (buffer count is provably bit-neutral).
4. **Rollout contracts:** returned `roll` layout (logging cadence rows), `g_dc*` device-direct
   columns + `colsReady` stamp, `g_dMGObsTraj`, bootstrap values, state threading, `finalObs`
   exactness. A replacement honors each or renegotiates it *with the Lean side in the same commit*.

## 4. Considered designs

Three designs were produced independently and judged (workflow `wf_d85d0bbd-573`, 8 agents;
full corpus in the session transcript):

| design | essence | judged |
|---|---|---|
| **Wide-Lane** | Evolve `mg_buf_worker`: decouple forward width from env fan-out (2–4 wide buffers, nested per-buffer env thread pools). ~200–300 LoC, one file, bit-identical stages. | **1st** — best SPS-per-risk; the flat nbuf sweep never tested the decoupled cell |
| **Mirrorstep** | Full mirror of their loop: persistent buffer threads, device-staged obs H2D, wide GEMM forward, per-(t,buf) graphs. ~600–900 LoC. | 2nd — right end-state pieces, wrong first step; width alone does not cut aggregate fused-forward work |
| **Conn** | Maximal: one native C driver runs 20-update chunks; Lean specifies rather than executes. ~1200–1800 LoC. | 3rd — its transport wins are capturable per-update without the identity cost; **rejected under standing rule D4** |

## 5. Recommended: the escalation ladder

Wide-Lane as the foundation; Mirrorstep's staging/width/graphs as measurement-gated rungs;
Conn's orchestration wins re-scoped to per-update FFI changes. Every rung is env-gated,
independently byte-verified, with the previous path as fallback.

**End-state per-step sequence (per buffer b):** one forward launch on `stream_b` (fused WMMA
default; wide-GEMM arm post-M6) reading device-staged obs → `cudaMemcpyAsync` D2H of actions
only → **one** `cudaStreamSynchronize` → barrier-release E env sub-workers
(`step_range_*` over disjoint sub-ranges into pinned ping-pong, host-tail shares included) →
un-synchronized H2D of obs/terms staging → ping-pong flip. System-wide: 2 syncs/step.

| rung | change | verification | gate |
|---|---|---|---|
| **M1** | Per-buffer env sub-pools (E threads inside a buffer; E=1 ≡ today) | bit-identical; byte-harness across E×{FUSED}×{dcOK} | — |
| **M2** | Device-staged obs+terms H2D replacing zero-copy PCIe reads in the kernel | bit-identical (pointer swap); nsys span A/B | **D0**: default-on iff span drops, SPS ≥ 0 |
| **M3** | nbuf default 2; sweep {2,4,8,16} with total env threads pinned | bit-identical (documented bit-neutrality) | **D1**: ≥8–10% → continue ladder; flat → M6 is the priority (wall = aggregate forward work) |
| **M4** | Conn-lite per-update wins: persistent buffer threads; wFlat D2H last-minibatch-only; compact `roll` transport (obsCol elided when resident); resident-state reuse flag; muon writes f32 `dP` in place | bit-identical incl. multi-update weight bytes; small Lean deltas, contracts amended explicitly | — |
| **M5** | Zero code: re-bench `PUFFER_MG_WPREC=bf16` at the new width | existing tier harness | **D2**: forward-tier default chosen on SPS at matched 3-seed learning curves; any flip = declared re-baseline |
| **M6** | Wide-GEMM forward arm (existing verified non-fused lineage at width, FAST_16BF) targeting their ~9.5µs/256-row class | tolerance-class at the existing tier; wide-shape verify extension; 3-seed curves | gated on D1 |
| **M7** | Per-(t,buf,parity) graph capture, warmup-rollout pattern; rng base + lr via device pointers | **must byte-equal M6 eager** (new verify mode: eager vs replay compare) | **D3**: only if the GEMM arm shows ≥10% launch-gap overhead; the fused arm never needs it |
| **M8** | Default flips, docs, memory, benchmark-artifact re-baseline | placement-matched, 3 envs, per-tier, sweep table incl. null results | — |

**Standing rule D4 (the Conn rule):** the chunked C driver is rejected unless, after M4,
profiling still attributes ≥10% of update wall to FFI/transport unreachable per-update. The
identity cost (Lean demoted from executing to specifying; permanent dual-implementation drift
tax) is payable only for measured, otherwise-unreachable wall time. Facts today say the bar is
not met.

## 6. Expected outcome and cost

- Bit-identical rungs M1–M4: **+10–20% SPS** (~400–600 LoC, ~1 week), zero verification debt.
- Tolerance rungs M5–M7: further **+10–20%** if the wide forward lands in their per-row class →
  placement gap 0.76× → **0.85–1.0×** (~1 more week, gated on D1/D2/D3).
- Worst case: a published null at M3 with M1/M2/M4 retained as permanent risk-free wins — which
  is itself this project's honest-benchmarking identity.

## 7. Invariants carried throughout

No new kernel math anywhere (M6 reuses verified lineage at a new shape, checked at its existing
tier). Every new env var is either provably bit-neutral (byte-harness documents it empirically)
or a declared, deterministic, off-gated tolerance-class change mirrored in the verify harness.
V-Trace, Muon, u8/f32 transport, device-direct columns remain `max|Δ| = 0` throughout.

## 8. Work breakdown (implementation scope)

All C work in `ffi/puffercuda.cu` unless noted. "Verify" = the byte-harness protocol from §5
(fixed-seed multi-update, `max|Δ|=0` for bit-identical claims, across breakout/pong/squared and
the precision gates) plus the standard adversarial review per rung.

| rung | work items | LoC | effort | risk |
|---|---|---|---|---|
| **M1** env sub-pools | Clone the `g_rp` pool pattern into per-buffer sub-pools (struct, barriers, spawn/teardown); `mg_buf_worker`'s env-step section becomes barrier-release/join of E sub-workers, each calling `step_range_*` on its disjoint sub-range plus its share of the host tail (rewCol/termCol scatter, plane memcpys). `PUFFER_MG_ENVTHREADS`. E=1 short-circuits to today's inline path. | 100–150 | ~1 day | low — disjoint ranges, unchanged ABI |
| **M2** device-staged obs | `dObsStage_b`/`dTermStage_b` (rb2 slots); post-env-step async H2D from the pinned ping-pong; the fused kernel's obs source flips from zero-copy pinned to device staging under `PUFFER_MG_H2DOBS`. WAR safety rests on the documented stream-ordering argument (step s+1's sync precedes the twin rewrite at s+2) — review focus. | 80–120 | ~1 day | low-med — ping-pong reuse hazard |
| **M3** width sweep | Default nbuf→2; sweep {2,4,8,16} × E pinned at 16 total × M2 on/off × 3 envs × 2–3 reps ≈ 40–70 sequential runs. Pulls **D1**. | ~10 | ~half day (measurement) | none |
| **M4a** persistent buffer threads | RUN/WAIT atomic handshake replacing per-update spawn/join (PufferLib `vecenv.h` precedent; bounded spin + usleep backoff). | 80–120 | ~half day | low |
| **M4b** wFlat last-mb-only | `returnWeights` flag on the muon FFI; Lean passes it on the final minibatch only. | ~30 C + ~10 Lean | ~1 h | low |
| **M4c** compact roll | Rollout returns the compact layout when traj+dcOK (obsCol elided — kills the full-size sarray alloc/zero per update); full-fat layout on logging updates and as fallback. Contract 1 amended in the same commit (Lean slicing gated). | 60–100 C + ~20 Lean | ~half day | med — layout contract; the device-env-removal reviews showed where these bite |
| **M4d** resident-state flag | Per-update skip of the finalState f64 round-trip. NOTE: the CPU path ping-pongs `dSa`/`dSb` — the flag must track which buffer holds the final state (the removed genv handoff never had this hazard; its reviews are the checklist). | 40–60 | ~half day | med |
| **M4e** muon writes `dP` | Muon epilogue writes the f32 cast in place, retiring the per-rollout `k_d2f`. Order-independent cast ⇒ byte-equal. | ~20 | ~1 h | low |
| **M5** bf16 at width | Zero code. Re-bench `PUFFER_MG_WPREC=bf16` across the M3 sweep grid. Pulls **D2**. | 0 | ~2 h | none |
| **M6** wide-GEMM arm | New branch in the buffer worker assembling existing verified pieces at nb-width: `mingru_fwd_dev` + `k_mingru_asm` + width-parallel sampler + `k_mg_cols_alv` + `k_scatter_mg_obs` + state-reset fold, per-buffer cuBLAS handles (exist). Wide-shape extension of `verify-mingru-step-gpu`. 3-seed learning-curve acceptance. | 300–400 (+~50 verify) | 2–3 days | med — new composition of verified parts, tolerance-class |
| **M7** graph capture | Gated on **D3**. Warmup-capture pass at first rollout (fake pass, no env stepping); exec array `[T×nbuf×parity]`; rng base via device pointer (our global-row splitmix64 needs only a device base slot — no Philox migration); capture-key over baked pointers/shapes with recapture; eager-vs-replay byte-compare verify mode. | 200–300 | 2–3 days | high — only rung with real machinery risk |
| **M8** re-baseline | Default flips; docs; memory; placement-matched benchmark + artifact re-render incl. the sweep table and nulls. | ~50 + docs | ~1 day | none |

**Totals.** Bit-identical span M1–M4: ~450–600 LoC, **4–5 working days** including verification
runs — expected +10–20% at zero verification debt. Tolerance span M5–M7 (gated on D1/D2/D3):
~500–750 LoC, **4–6 days** — further +10–20% if the wide forward lands in their per-row class.
Endpoint projection 0.85–1.0× placement-matched. Sequencing is strict M1→M2→M3 (D1 decides
whether M6 jumps the queue); M4a–e are independent and can interleave anywhere after M1.

## 9. Progress log

**M1 landed (commit `f35a484`).** Env sub-pools built exactly as scoped, bit-identical (gate off
reproduces every baseline; gate on at ENVTHREADS 4/8/16 is *also* byte-identical on all three
envs — the scatter is disjoint-row regardless of partition depth).

**Decision D1: PULLED, and it says skip ahead.** The nbuf×ENVTHREADS sweep (total env threads
pinned near 16) is flat-to-slightly-negative on all three envs:

| env | nbuf=16,E=1 (today) | nbuf=2,E=8 (decoupled) |
|---|---|---|
| breakout | 16.88M | 16.78M |
| squared | 16.69M | 16.47M |
| pong | 11.97M | 11.93M |

This is exactly what §5's M3 row predicted for the flat case: *"the wall is aggregate forward
work"* — the fused kernel's 16-row/96KB-smem blocks do constant per-row work regardless of
launch width, so decoupling fan-out from width had nothing to win. Not a bug; a closed question.

**Re-sequencing.** M2 and M4 remain worth landing on their own merits (they target different
costs — PCIe reads inside the kernel, per-update transport). **M6 (the wide-GEMM forward arm)
is promoted ahead of M3-gated sequencing** — its own gate (`gated on D1`) is now satisfied, and
it is the only rung that targets the actual bottleneck D1 identified: PufferLib's per-row-cheaper
forward organization (GEMM + elementwise, not a fused mega-kernel), not our launch/sync
structure. M7 (graphs) stays behind M6's own follow-up gate D3.

Next: M6, or M2 first if the maintainer wants the cheap PCIe win banked before the bigger
composition. Recommendation: **M6** — it is what D1 actually points at.

## 10. M6 refuted — the wide-GEMM arm is a dead end, no new code needed to know it

M6 was scoped as "assemble existing verified pieces... at width" — and the existing pieces
(`mingru_fwd_dev` + `k_mingru_asm` + `k_sample_seg`) are *already* the live fallback arm behind
`PUFFER_MG_FUSED=0`, already running inside `mg_buf_worker` at whatever width `nbuf` gives it.
M1's decoupling makes "at width" directly measurable with zero new code — so before writing the
~300-400 LoC M6 was budgeted at, the premise was tested directly:

| env (H) | fused (default) | unfused, nbuf=16 (today's width) | unfused, nbuf=2 (wide) | unfused, nbuf=1 (single whole-batch GEMM, no buffer overlap) |
|---|---|---|---|---|
| pong (32) | 11.97M | 6.59M | — | — |
| breakout (64) | 16.9M | 9.5M | 9.5M | 9.5M |
| squared (128) | 16.7M | 16.7M | 16.8M | 11.1M |

Three findings, all decisive:

1. **The fused kernel already beats the GEMM arm by ~1.8× at H≤64** (pong, breakout) — consistent
   with the code's own standing comment on `mingru_fwd_dev`'s call site: 5 GEMMs + 8 kernels per
   (step, buffer) pay *kernel floors*, not compute, and the fused kernel collapses all of that
   into one launch.
2. **Widening the GEMM arm (nbuf 16→2) does not help** — flat on breakout, flat-to-slightly-up
   on squared. Confirms D1's mechanism generalizes: more rows per GEMM call doesn't fix a
   launch-floor-bound design when the floor count per step doesn't drop enough to matter, or when
   the GEMMs are already past their efficient-width knee.
3. **Going all the way to nbuf=1 (one true whole-batch GEMM per stage, matching PufferLib's
   per-stage width most closely) makes it *worse*, not better** — squared drops 16.7M→11.1M. The
   buffer-level env/GPU overlap `nbuf` also controls is worth more than any wide-GEMM efficiency
   gain, so the naive trade is net negative.

**Conclusion: M6 as scoped — reusing the existing per-buffer GEMM lineage at width — is refuted.**
It cannot reach PufferLib's per-row efficiency; at H≤64 it doesn't even reach our own fused
kernel's. The only way to approach PufferLib's organization (one big GEMM per stage, batched
across ALL agents, still overlapped with env-stepping across 2 persistent buffers, launch-count
collapsed via captured graphs) is a **new architecture**, not an assembly of verified parts —
batch the GEMM stage across buffers while preserving overlap, which the current per-buffer-owns-
its-whole-forward structure does not support without a genuine rewrite of the buffer/stream
relationship. That is a scope and risk class equivalent to the rejected Conn design, and by the
same standing logic (§5, rule D4) it should be a maintainer decision, not something to build
under a "climb the ladder" mandate.

**Ladder status:** M1 landed (real, zero-risk, reusable). D1 and M6 both answered by measurement.
M2 (device-staged obs — a different cost, PCIe reads inside the fused kernel) and M4 (per-update
transport wins) remain live, small, and worth landing on their own merits. M7 (graph capture) is
now doubly closed — both the fused-kernel graph attempt (main session log) and this GEMM-arm
line refute the premise that launch count is the fused kernel's bottleneck. **Recommendation:**
land M2 + M4 as the remaining cheap, safe wins (~150-250 LoC, ~1-2 days, incremental but real);
treat "design a new batched-across-buffers architecture" as a separate, explicitly-authorized
decision — this document's honest conclusion, not a deferred TODO.

## 11. M2 — already settled by the existing zero-copy gate

M2 proposed device-staged obs+terms H2D to replace zero-copy PCIe reads inside the fused kernel.
The obs half already exists (`PUFFER_MG_ZCOBS`, default ON = zero-copy) — re-measured fresh
under M1 to check for interaction:

| env | zero-copy (default) | device-staged H2D (`ZCOBS=0`) |
|---|---|---|
| breakout | 16.93M | 16.75M |
| squared | 16.61M | 16.70M |
| pong | 11.98M | 11.70M |

The standing default wins or ties everywhere (±1%, within run-to-run noise) — the same
conclusion the original campaign reached when this gate was built. Nothing new to land.

The one untested piece is the terminal gate in device-column mode, which always reads zero-copy
from the pinned `termPlane` with no alternative — but its payload (8B/row) is ~1/118th of
breakout's obs traffic (472B/row), so any win there is bounded well under 1%. Not worth building.

**M2 closed: already answered, no code to write.**

## 12. M4 — surveyed item by item, one landed

Before building, each M4 sub-item was bounded against measurement rather than assumed worth it:

| item | finding | action |
|---|---|---|
| M4e (muon writes `dP` in place) | `k_d2f`'s own kernel is **0.02% of GPU time** (150µs total across 180 launches, nsys) | not worth building — negligible even before counting the D2H-sync risk of a freshness check |
| M4b (wFlat D2H last-minibatch-only) | already delivered by the minibatch de-sync attack earlier this session (commit `eb9c873`) — the resident-grad path already returns an empty wFlat with no D2H | already done, nothing to land |
| M4c/M4d (compact roll, resident-state skip) | the per-step sample D2H — necessary for CPU env-stepping, not waste — is **88.8% of all memcpy time** (nsys); eliminating any of it means reopening the device-env decision this session already made deliberately | blocked by a standing decision, not a gap in the ladder |
| **M4a (persistent buffer threads)** | pure fixed overhead (pthread_create+join every update), zero data-flow risk, bounded positive | **built** — commit `0152628`, +1.4–4.8% across envs |

**Ladder status: fully climbed.** M1 landed (real, currently a measured no-op pending a future
width-sensitive kernel). D1 pulled cleanly. M2 settled by the existing gate. M6 refuted by direct
measurement before writing its budgeted LoC. M4 surveyed exhaustively; its one live item landed.
M7 was gated on M6 and inherits its refutation.

**What's left is not on this ladder.** The remaining gap to PufferLib requires either (a) a new
batched-across-buffers forward architecture (§10 — flagged, not started, same risk class as the
rejected Conn design) or (b) accepting the current per-row kernel organization as the ceiling of
this design. Both are calls for the maintainer, not the next rung to climb.

## 13. Scoping "batched-across-buffers" — two more levers tested, both refuted

§10 flagged a new batched-across-buffers architecture (mirroring PufferLib's wide-GEMM,
graph-captured, bf16-everywhere organization) as the remaining path to close the h128/h64 gap
further, explicitly requiring a separate maintainer decision before starting. This section is
that scoping — and the honest conclusion is that the premise doesn't survive measurement.

### 13.1 Root-causing the per-row cost: occupancy, not launch count

`ptxas -v` on the default rollout kernel (`k_mg_fused_step_w`): **80 registers/thread, 0 spills**,
at `blockDim=512`. This GPU (RTX 5090, sm_120): 65,536 registers/SM, 100KB shared/SM, 1536 max
threads/SM. 80×512 = 40,960 registers/block — only **1 block/SM** fits (2× needs 81,920 >
65,536). Shared memory independently confirms the same ceiling for breakout (full-layout
footprint 53.5KB; 2×53.5KB=107KB > 100KB), while squared's big-H reduced layout (43.5KB) would
allow 2 blocks by shared memory alone — registers still cap it at 1 there too.

**The fused kernel runs at 1 block/SM = 33% of max thread occupancy on both flagship shapes.**

### 13.2 The occupancy experiment (built, measured, reverted — no commit)

Tested `__launch_bounds__(512,2)` on the kernel — a one-line compiler hint forcing ≤64
registers/thread, the exact budget for 2 blocks/SM by registers. `ptxas` confirmed 64 registers
with small spills (40B store / 44B load per thread). Bit-identical by construction (spilling
relocates values, doesn't change them — confirmed via the byte-harness on both envs).

**Measured (interleaved A/B, 3 pairs each): breakout −1 to −2%, squared −2 to −3%.** A net loss
on both — not the free latency-hiding win occupancy theory predicts.

**Why:** two things are true at once. For breakout, shared memory *also* caps at 1 block/SM
regardless of the register change, so its regression is pure spill overhead with no occupancy
gain to offset it. For squared, where shared memory *would* allow 2 blocks, the loss persists
anyway — meaning the kernel isn't actually latency-hiding-limited. This matches and sharpens an
earlier finding from this campaign (the FR=32 tiling refutation, pre-dating tonight): *"the
fused kernel is NOT weight-load-bound — it's SCALAR-STAGE-bound (encoder's serial FMA dot, the
gate cell's sigmoid/tanh, the f64 categorical sampler's exp/log)."* Scalar ALU/SFU throughput,
not memory latency, is the binding resource — more concurrent blocks per SM doesn't help because
there's no idle memory-latency window to fill; it just adds spill traffic contending for the
same load/store units.

### 13.3 The batched-across-buffers premise doesn't survive

The premise was: PufferLib's organization (wide per-buffer GEMMs, one graph-captured launch per
step per buffer, bf16 everywhere) must be inherently more efficient, so adopting its shape
should close the gap. Two independent, decisive tests now contradict that for this kernel:

- **M6** (§10): the existing GEMM-based lineage, run wide, loses to our own fused kernel by
  ~1.8× at h≤64, and going wider makes it *worse* by sacrificing buffer-level overlap.
- **Occupancy tuning** (this section): our fused kernel is *already* at PufferLib's launch-count
  parity (one launch per step per buffer) and is scalar-throughput-bound, not memory- or
  occupancy-bound — the lever that helps a memory-bound kernel does nothing here.

**There is no scoped, promising batched-across-buffers architecture to build.** What's actually
binding is scalar compute throughput on stages that cannot be GEMM-ized: the gate cell's
elementwise nonlinearities (every row, every layer) and the categorical sampler's f64
transcendentals (every row). Reorganizing buffers, launches, or occupancy cannot touch either.

### 13.4 What would actually move it — and why it's a different kind of decision

The only levers that touch the real bottleneck:

1. **Reduce scalar FLOPs/row** — e.g. a cheaper sampler. The f64 categorical draw is
   exact-vs-oracle by design; a faster approximation is a verification-tier decision, not a perf
   patch, and a more sensitive one than the bf16 forward tier (which touches the *gradient* —
   this would touch the *sampled action itself*: a wrong action changes the trajectory, not just
   its gradient).
2. **SFU throughput is a hardware ceiling** — sigmoid/tanh/exp/log run at a fixed per-SM rate
   independent of occupancy or memory-system tuning. No design lever here beyond "do less of it."
3. **Amortize the sampler across less-frequent draws** — architecturally invasive (changes the
   rollout's per-step semantics), out of scope for a kernel-level attack.

None of these are incremental, low-risk rungs like the ones already climbed (§5–§12) — each is a
precision/semantics decision with verification-tier consequences, not a structural rewrite.
**Recommendation: do not build. The gap's remaining shape is now understood precisely** (scalar
SFU throughput on the sampler and gate cell, not launch count, not occupancy, not GEMM
organization) **even though no further attack is recommended** — the ladder's conclusion (§12)
stands, and this section closes the one item it left explicitly open.

## 14. A third refutation — fast-transcendental gate intrinsics, and what it actually pins the wall

§13's occupancy experiment established the fused kernel is scalar-throughput-bound, not memory- or
occupancy-bound, and named the gate cell's sigmoid/tanh and the sampler's f64 exp/log as the binding
cost. The natural next test: does the gate cell's `expf()` — CUDA's accurate multi-instruction SFU
path, since this build does not pass `--use-fast-math` — cost meaningfully more than the hardware
`__expf()` intrinsic (a single SFU instruction, ~2 ulp)?

**Built, verified, measured, reverted — zero committed code.** A `d_sigf_fast`/`d_gactf_fast` pair
using `__expf` was wired into the default rollout kernel's gate stage only (BPTT forward/backward/
gradient untouched — a pure rollout-forward precision decision, gated `PUFFER_MG_FASTEXP`, default
off, independent of and no riskier than the tf32-WMMA tier it composes with). Gate off: byte-identical
to every baseline. Gate on: confirmed genuinely active (trajectory diverges from the very first step —
the verify harness's *displayed* max-relative-error happened to round identically at H=16 since tf32's
own ~4.5e-3 error already dominates and swamps `__expf`'s ~1e-6-to-1e-4-scale contribution; not a bug),
deterministic, learning curve healthy and tracking the baseline.

**Measured (interleaved A/B, 3 pairs each): breakout ≈ −1%, squared ≈ −0.2%.** Flat to slightly
negative — not the win a naive instruction-count argument predicts.

**This sharpens, rather than just repeats, §13's conclusion.** It isn't that the gate's transcendental
evaluation is cheap and irrelevant — occupancy analysis shows it's the dominant scalar cost. It's that
*reducing that cost doesn't reach the wall*, for the same reason occupancy tuning didn't: under this
architecture's heavy multi-stream co-residency (up to 16 concurrent buffer streams sharing the GPU),
the SM's execution resources are saturated by the *aggregate* kernel mix across all buffers, not by any
one kernel's own instruction count. Freeing SFU cycles in one buffer's kernel doesn't translate to
wall-clock gains because the other ~15 concurrent streams' kernels are already using whatever capacity
would be freed — consistent with the earlier span-dilation finding (nsys per-launch spans under this
regime are ~5× co-residency artifacts, not exclusive cost). **The wall is not any single kernel's
resource footprint; it is aggregate cross-stream GPU saturation at the current total kernel-work volume.**

**Standing recommendation unchanged, now on firmer ground.** Three independent, kernel-level micro-
attacks on this specific bottleneck — wide-GEMM reorganization (M6), occupancy/register tuning (§13),
and scalar-instruction reduction (this section) — all refuted. The remaining gap is not reachable by
further attacks *at the single-kernel level* under the current multi-buffer architecture; the only
levers left are the ones §13.4 already named (reduce total scalar work — e.g. amortizing the sampler,
a semantics-level change) or reducing the aggregate kernel-work volume itself (fewer, larger launches
across buffers — which is what M6 already tried and lost, since it sacrifices the overlap that makes
16-way co-residency worthwhile in the first place). **Do not attack this kernel again without a
structurally different idea; the class of "make this one kernel cheaper" is closed.**
