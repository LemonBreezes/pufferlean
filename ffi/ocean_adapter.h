/* ocean_adapter.h — bridge a real PufferLib Ocean C env to puffer-lean's plugin ABI (puffer_env.h).
 *
 * Each env's adapter.c does (mirroring PufferLib's binding.c):
 *     #include "<env>.h"                     // the real ocean env (c_reset/c_step + the struct)
 *     #define ENV_T <StructName>
 *     #define OBS_T <obs element type>       // unsigned char (ByteTensor) or float
 *     static void ocean_setup(ENV_T*, uint64_t seed, int idx, const char* cfg);  // config + rng + buffers
 *     static int  ocean_obsdim(ENV_T*);       // per-agent obs length
 *     static int  ocean_numactions(ENV_T*);
 *     static int  ocean_maxsteps(ENV_T*);
 *     static void ocean_teardown(ENV_T*);
 *     #include "../../ffi/ocean_adapter.h"
 *
 * We hold N single-agent env copies and drive them with the ocean env's OWN c_reset/c_step (which
 * auto-reset on terminal), converting the env's obs type → double and our double actions → float. So the
 * trainer runs PufferLib's actual env code, unchanged.
 */
#include "puffer_env.h"
#include <stdlib.h>
#include <string.h>
#include <stddef.h>   /* offsetof — how each adapter locates its own Log fields generically */

/* obs/action/reward/terminal buffer element types vary per ocean env (e.g. some envs have int actions +
   uint8 terminals; squared/breakout are float). Default to float; each adapter overrides as needed. */
#ifndef ACT_T
#define ACT_T float
#endif
#ifndef REW_T
#define REW_T float
#endif
#ifndef TERM_T
#define TERM_T float
#endif
/* Action components per agent (NUM_ATNS). 1 = single discrete action (works with the single-action
   policy). >1 = multi-discrete/continuous — the env's actions buffer is num_agents·ACT_N. */
#ifndef ACT_N
#define ACT_N 1
#endif
/* Agents per env instance. Single-agent envs (whose struct may lack a num_agents field) leave this at 1;
   a multi-agent adapter #defines OCEAN_NAGENTS(e) ((e)->num_agents) after setting it in ocean_setup. */
#ifndef OCEAN_NAGENTS
#define OCEAN_NAGENTS(e) 1
#endif
/* Action heads: single-discrete = 1 head (headsize = num_actions). A multi-discrete adapter #defines
   OCEAN_NHEADS <k> and OCEAN_HEADSIZES {s0,…,s_{k-1}} (PufferLib ACT_SIZES); ACT_N follows OCEAN_NHEADS. */
#ifndef OCEAN_NHEADS
#define OCEAN_NHEADS 1
#endif
#undef ACT_N
#define ACT_N OCEAN_NHEADS

/* nAgents = agents PER env instance (env->num_agents, set in ocean_setup). Each agent is a training row,
   so the batch is N·nAgents rows (instance i, agent a -> row i·nAgents+a). Single-agent envs keep nAgents=1. */
typedef struct { int N, nAgents, obsDim, numActions, maxSteps; ENV_T* envs; } OceanWrap;

void* puffer_env_make(int num_envs, uint64_t seed, const char* config) {
  OceanWrap* w = (OceanWrap*)calloc(1, sizeof(OceanWrap));
  w->N = num_envs;
  w->envs = (ENV_T*)calloc(num_envs, sizeof(ENV_T));
  for (int n = 0; n < num_envs; n++) ocean_setup(&w->envs[n], seed, n, config);
  w->nAgents    = OCEAN_NAGENTS(&w->envs[0]) > 0 ? OCEAN_NAGENTS(&w->envs[0]) : 1;
  w->obsDim     = ocean_obsdim(&w->envs[0]);      /* per agent */
  w->numActions = ocean_numactions(&w->envs[0]);
  w->maxSteps   = ocean_maxsteps(&w->envs[0]);
  return w;
}

void puffer_env_spec(void* env, int* obs_dim, int* num_actions, int* max_steps, int* num_agents) {
  OceanWrap* w = (OceanWrap*)env;
  *obs_dim = w->obsDim; *num_actions = w->numActions; *max_steps = w->maxSteps; *num_agents = w->nAgents;
}

void puffer_env_actinfo(void* env, int* n_heads, int* head_sizes) {
  OceanWrap* w = (OceanWrap*)env; (void)w; *n_heads = OCEAN_NHEADS;
#ifdef OCEAN_HEADSIZES
  static const int hs[] = OCEAN_HEADSIZES;
  for (int i = 0; i < OCEAN_NHEADS; i++) head_sizes[i] = hs[i];
#else
  head_sizes[0] = w->numActions;   /* single discrete head */
#endif
}

/* Continuous (diagonal-Gaussian) action space? A continuous adapter #defines OCEAN_CONTINUOUS 1 with
   OCEAN_NHEADS = action dim d and OCEAN_HEADSIZES {1,…,1}; the trainer then runs the Gaussian PPO path. */
int puffer_env_iscont(void* env) { (void)env;
#ifdef OCEAN_CONTINUOUS
  return 1;
#else
  return 0;
#endif
}

/* Copy every agent's obs into row (i·nAgents + a) of the flattened batch. */
static void ocean_copy_obs(OceanWrap* w, double* obs) {
  int A = w->nAgents, D = w->obsDim;
  for (int i = 0; i < w->N; i++) {
    OBS_T* o = (OBS_T*)w->envs[i].observations;   /* laid out nAgents·D contiguous */
    for (int a = 0; a < A; a++) {
      double* d = obs + (long)(i * A + a) * D;
      const OBS_T* oa = o + (long)a * D;
      for (int j = 0; j < D; j++) d[j] = (double)oa[j];
    }
  }
}

void puffer_env_reset(void* env, double* obs) {
  OceanWrap* w = (OceanWrap*)env;
  for (int n = 0; n < w->N; n++) c_reset(&w->envs[n]);
  ocean_copy_obs(w, obs);
}

/* Step only envs [start, start+count) and write their obs/rewards/terminals into the full arrays at the
   matching offsets — the range primitive the native rollout driver's persistent-thread env-step calls
   (each worker owns a disjoint env range, so no locking). Env-agnostic; names no specific env. */
void puffer_env_step_range(void* env, const double* actions, double* obs, double* rewards, double* terminals,
                           int start, int count) {
  OceanWrap* w = (OceanWrap*)env;
  int A = w->nAgents, D = w->obsDim;
  for (int i = start; i < start + count; i++) {
    ENV_T* e = &w->envs[i];
    ACT_T* ea = (ACT_T*)e->actions;
    for (int a = 0; a < A; a++)
      for (int c = 0; c < ACT_N; c++)
        ea[a * ACT_N + c] = (ACT_T)actions[(long)(i * A + a) * ACT_N + c];
    /* PufferLib's vecenv.h zeros rewards/terminals before each c_step (src/vecenv.h:288-289): envs that
       write a reward only on an event (and never clear it) rely on this. Replicate it for faithfulness. */
    memset(e->rewards,   0, (size_t)A * sizeof(REW_T));
    memset(e->terminals, 0, (size_t)A * sizeof(TERM_T));
    c_step(e);                                     /* real ocean step (auto-resets terminated agents) */
    for (int a = 0; a < A; a++) {
      rewards[i * A + a]   = (double)((REW_T*)e->rewards)[a];
      terminals[i * A + a] = (double)((TERM_T*)e->terminals)[a];
    }
    OBS_T* o = (OBS_T*)e->observations;            /* obs → double for this env's agents (ocean_copy_obs, ranged) */
    for (int a = 0; a < A; a++) {
      double* d = obs + (long)(i * A + a) * D;
      const OBS_T* oa = o + (long)a * D;
      for (int j = 0; j < D; j++) d[j] = (double)oa[j];
    }
  }
}

void puffer_env_step(void* env, const double* actions, double* obs, double* rewards, double* terminals) {
  puffer_env_step_range(env, actions, obs, rewards, terminals, 0, ((OceanWrap*)env)->N);
}

/* f32-obs twin of step_range: identical stepping, obs written as FLOAT. The env's native obs are float or
   uint8 (OBS_T), so (float)oa[j] == (float)(double)oa[j] — BIT-IDENTICAL to the f64 path followed by the
   trainer's downstream f32 cast, with the double intermediate + cast pass removed from the rollout's
   critical path (the GPU forward consumes f32). Env-agnostic. */
void puffer_env_step_range_f32(void* env, const double* actions, float* obs, double* rewards, double* terminals,
                               int start, int count) {
  OceanWrap* w = (OceanWrap*)env;
  int A = w->nAgents, D = w->obsDim;
  for (int i = start; i < start + count; i++) {
    ENV_T* e = &w->envs[i];
    ACT_T* ea = (ACT_T*)e->actions;
    for (int a = 0; a < A; a++)
      for (int c = 0; c < ACT_N; c++)
        ea[a * ACT_N + c] = (ACT_T)actions[(long)(i * A + a) * ACT_N + c];
    memset(e->rewards,   0, (size_t)A * sizeof(REW_T));     /* zero-before-step, as in step_range */
    memset(e->terminals, 0, (size_t)A * sizeof(TERM_T));
    c_step(e);
    for (int a = 0; a < A; a++) {
      rewards[i * A + a]   = (double)((REW_T*)e->rewards)[a];
      terminals[i * A + a] = (double)((TERM_T*)e->terminals)[a];
    }
    OBS_T* o = (OBS_T*)e->observations;
    for (int a = 0; a < A; a++) {
      float* d = obs + (long)(i * A + a) * D;
      const OBS_T* oa = o + (long)a * D;
      for (int j = 0; j < D; j++) d[j] = (float)oa[j];
    }
  }
}

/* ---- reduced-precision obs TRANSPORT (optional; the trainer picks by puffer_env_obs_kind) ----------
   kind 1 (u8): OBS_T is a byte type — raw byte transport, and the trainer's u8→f32 widen is EXACT, so
   this is BIT-IDENTICAL to the f32 path at 4x fewer bytes. kind 0: float-native — u8 would truncate;
   the trainer must not use step_range_u8 (it exists but is only valid for kind 1).
   bf16: valid for ANY env — obs rounded to bf16 (round-to-nearest-even; bit-equal to CUDA's
   __float2bfloat16 for all FINITE values — NaN canonicalization differs, but envs never emit NaN obs),
   a TOLERANCE-class transport matching PufferLib's own bf16 obs storage. */
int puffer_env_obs_kind(void* env) { (void)env;
  OBS_T m1 = (OBS_T)-1;                          /* unsignedness test: u8 transport would WRAP a signed byte */
  return (sizeof(OBS_T)==1 && m1 > 0) ? 1 : 0; }

static inline unsigned short ocean_f32_to_bf16(float f) {
  union { float f; unsigned int u; } v; v.f = f;
  unsigned int u = v.u;
  return (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);   /* RNE, == __float2bfloat16 */
}

void puffer_env_step_range_u8(void* env, const double* actions, unsigned char* obs, double* rewards,
                              double* terminals, int start, int count) {
  OceanWrap* w = (OceanWrap*)env;
  int A = w->nAgents, D = w->obsDim;
  for (int i = start; i < start + count; i++) {
    ENV_T* e = &w->envs[i];
    ACT_T* ea = (ACT_T*)e->actions;
    for (int a = 0; a < A; a++)
      for (int c = 0; c < ACT_N; c++)
        ea[a * ACT_N + c] = (ACT_T)actions[(long)(i * A + a) * ACT_N + c];
    memset(e->rewards,   0, (size_t)A * sizeof(REW_T));
    memset(e->terminals, 0, (size_t)A * sizeof(TERM_T));
    c_step(e);
    for (int a = 0; a < A; a++) {
      rewards[i * A + a]   = (double)((REW_T*)e->rewards)[a];
      terminals[i * A + a] = (double)((TERM_T*)e->terminals)[a];
    }
    OBS_T* o = (OBS_T*)e->observations;
    for (int a = 0; a < A; a++) {
      unsigned char* d = obs + (long)(i * A + a) * D;
      const OBS_T* oa = o + (long)a * D;
      for (int j = 0; j < D; j++) d[j] = (unsigned char)oa[j];
    }
  }
}

void puffer_env_step_range_bf16(void* env, const double* actions, unsigned short* obs, double* rewards,
                                double* terminals, int start, int count) {
  OceanWrap* w = (OceanWrap*)env;
  int A = w->nAgents, D = w->obsDim;
  for (int i = start; i < start + count; i++) {
    ENV_T* e = &w->envs[i];
    ACT_T* ea = (ACT_T*)e->actions;
    for (int a = 0; a < A; a++)
      for (int c = 0; c < ACT_N; c++)
        ea[a * ACT_N + c] = (ACT_T)actions[(long)(i * A + a) * ACT_N + c];
    memset(e->rewards,   0, (size_t)A * sizeof(REW_T));
    memset(e->terminals, 0, (size_t)A * sizeof(TERM_T));
    c_step(e);
    for (int a = 0; a < A; a++) {
      rewards[i * A + a]   = (double)((REW_T*)e->rewards)[a];
      terminals[i * A + a] = (double)((TERM_T*)e->terminals)[a];
    }
    OBS_T* o = (OBS_T*)e->observations;
    for (int a = 0; a < A; a++) {
      unsigned short* d = obs + (long)(i * A + a) * D;
      const OBS_T* oa = o + (long)a * D;
      for (int j = 0; j < D; j++) d[j] = ocean_f32_to_bf16((float)oa[j]);
    }
  }
}


void puffer_env_free(void* env) {
  OceanWrap* w = (OceanWrap*)env;
  for (int n = 0; n < w->N; n++) ocean_teardown(&w->envs[n]);
  free(w->envs); free(w);
}

/* ---- OPTIONAL log channel (puffer_env.h) — PufferLib's per-env `Log` reported verbatim -------------
 *
 * Every ocean env carries its own `Log log;` on the env struct, but the fields differ per env in NAME,
 * COUNT and ORDER, so nothing generic can name them. An adapter therefore lists ITS OWN scalar float
 * fields once, as an X-macro:
 *
 *     #define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(n)
 *
 * and this header turns each name into a (name, slot) pair with `offsetof(Log, f)/sizeof(float)` — the
 * adapter's own `#include "<env>.h"` makes `offsetof` resolve against the RIGHT struct, so no layout is
 * ever assumed here. An env that lacks a field just does not list it: the field is then absent from the
 * exported names and the caller sees "unsupported" for it, with no compile break for any other env.
 * An adapter that defines no OCEAN_LOG_FIELDS at all exports none of the three symbols, and the loader
 * sees the env as having no log channel (exactly like a plugin without puffer_env_iscont).
 *
 * `n` is REQUIRED when the channel is enabled: PufferLib's aggregation is defined in terms of it.
 * Envs whose `Log` has non-scalar members (craftax_classic's `achievements[]`, drone's `TaskLog
 * task[]`, robocode's `hist_*_bank[]`) simply omit those from the list — they are still summed as part
 * of the whole-struct fold below (which is byte-for-byte PufferLib's), just not individually named. */
#ifdef OCEAN_LOG_FIELDS

#ifndef OCEAN_LOG_T
#define OCEAN_LOG_T Log            /* every ocean env names it `Log`; overridable per adapter */
#endif
#ifndef OCEAN_LOG_MEMBER
#define OCEAN_LOG_MEMBER log       /* ...and stores it as `Log log;` on the env struct */
#endif

#define OCEAN_LOG_NAME_(f) #f,
#define OCEAN_LOG_SLOT_(f) (int)(offsetof(OCEAN_LOG_T, f) / sizeof(float)),
static const char* const ocean_log_names[] = { OCEAN_LOG_FIELDS(OCEAN_LOG_NAME_) };
static const int         ocean_log_slots[] = { OCEAN_LOG_FIELDS(OCEAN_LOG_SLOT_) };
#define OCEAN_LOG_NF     ((int)(sizeof(ocean_log_slots) / sizeof(ocean_log_slots[0])))
/* PufferLib reads the Log as a flat float array: `num_keys = sizeof(Log)/sizeof(float)`. Same here, so
   the fold below sums the SAME slots in the SAME order as static_vec_aggregate_logs — bit-identical. */
#define OCEAN_LOG_NKEYS  ((int)(sizeof(OCEAN_LOG_T) / sizeof(float)))
#define OCEAN_LOG_NSLOT  ((int)(offsetof(OCEAN_LOG_T, n) / sizeof(float)))

int puffer_env_log_nfields(void* env) { (void)env; return OCEAN_LOG_NF; }

const char* puffer_env_log_name(void* env, int i) { (void)env;
  return (i >= 0 && i < OCEAN_LOG_NF) ? ocean_log_names[i] : ""; }

/* static_vec_aggregate_logs + static_vec_log (PufferLib src/vecenv.h), transcribed:
     sum every float slot over the copies with log.n > 0 → divide ALL slots by the summed n → zero every
     copy's log → report. "n" itself is reported UNDIVIDED (vecenv's `dict_set(out,"n",n)` overwrites
     my_log's divided one). n == 0 ⇒ early return, out untouched, logs left alone. */
double puffer_env_log(void* env, double* out, int max) {
  OceanWrap* w = (OceanWrap*)env;
  float agg[OCEAN_LOG_NKEYS];
  memset(agg, 0, sizeof(agg));
  for (int i = 0; i < w->N; i++) {
    const float* lg = (const float*)(const void*)&w->envs[i].OCEAN_LOG_MEMBER;
    if (lg[OCEAN_LOG_NSLOT] == 0.0f) continue;            /* PufferLib skips never-logged copies */
    for (int j = 0; j < OCEAN_LOG_NKEYS; j++) agg[j] += lg[j];
  }
  float n = agg[OCEAN_LOG_NSLOT];
  if (n == 0.0f) return 0.0;                              /* no episode finished — nothing to report */
  for (int j = 0; j < OCEAN_LOG_NKEYS; j++) agg[j] /= n;
  for (int i = 0; i < w->N; i++) memset(&w->envs[i].OCEAN_LOG_MEMBER, 0, sizeof(OCEAN_LOG_T));
  int k = (OCEAN_LOG_NF < max) ? OCEAN_LOG_NF : max;
  for (int i = 0; i < k; i++) {
    int s = ocean_log_slots[i];
    out[i] = (s == OCEAN_LOG_NSLOT) ? (double)n : (double)agg[s];
  }
  return (double)n;
}
#endif  /* OCEAN_LOG_FIELDS */
