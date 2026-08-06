/* Adapter: PufferLib Ocean `moba` -> pufferlean plugin ABI.
 * MOBA (Dota-like) env. MULTI-DISCRETE: 6 heads {7,7,3,2,2,2}
 *   head0/1: move vel_y/vel_x (0..6), head2: attack target mode (0..2),
 *   heads 3/4/5: use Q/W/E (0..1).
 * ByteTensor obs (unsigned char, 510 = 11*11*4 + 26 per agent), float actions/rewards/terminals.
 *
 * script_opponents (default 1, from config/moba.ini): when 1, dire team (pids 5-9) is
 * driven by an internal scripted AI (creep_ai) and only the 5 radiant heroes are learning
 * agents -> num_agents = 5, a FLAT batch. When 0, all 10 heroes learn (num_agents = 10).
 * Either way the batch is flat; we keep the .ini default of 1 (5 agents, cheaper).
 *
 * init_moba() mallocs the internal game state (entities/map/rng); the 5 caller buffers
 * (obs/actions/rewards/terminals/truncations) plus the per-env BFS scratch (ai_path_buffer)
 * are malloc'd here. ai_paths (256 MB BFS cache) is identical for every env (same procedural
 * map) so it is allocated once and SHARED across all env copies, exactly as binding.c does.
 *
 * DETERMINISM — PUFFER_MOBA_DETERMINISTIC, DEFAULT OFF (2026-08-05).
 * moba does not reproduce at a fixed seed. The MinGRU rollout driver steps disjoint env ranges from
 * `nbuf` concurrent pthreads (ffi/puffercuda.cu, mg_buf_worker / mgsub pool), and moba — alone among
 * our envs — reaches PROCESS-GLOBAL mutable state from inside c_step, in two places:
 *   (1) libc rand(). moba.h draws from it in spawn_player (moba.h:669), spawn_creep (:862),
 *       spawn_neutral (:892) and init_moba's CachedRNG fill. One global LCG, many consumer threads
 *       => which env gets which draw is scheduler-dependent.
 *   (2) the shared 256 MB ai_paths BFS cache, filled lazily during stepping (moba.h move_towards).
 *       Threads race to fill the same destination block, AND read blocks still being filled — bfs
 *       writes the destination cell TWICE (atn, then 8), so a reader catching the transient gets a
 *       bogus direction. This one is the dominant term.
 * Each alone still leaves moba nondeterministic (measured); both are needed. Neither is a bug in a
 * serial driver — `env-log moba` is byte-identical either way.
 *
 * The ocean envs are FIXED BY PUFFERLIB and are our comparison baseline, so fixing this by default
 * would make our moba a different environment from theirs. Worse, PufferLib's own config/moba.ini
 * ships num_threads = 16, so upstream very likely races the same way — the racy behaviour IS the
 * baseline. So both fixes are gated OFF by default and the default path is byte-identical to
 * upstream. Set PUFFER_MOBA_DETERMINISTIC=1 to get a moba that reproduces at a fixed seed (useful
 * for bisecting the return collapse); accept that its trajectories then differ from PufferLib's.
 * The flag is read ONCE at env construction — never on the step path. */
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>

/* Read once from ocean_setup (env construction is serial and strictly precedes any stepping). Both
   hot-path hooks below branch on it; the branch is perfectly predicted and measures free. */
static int g_moba_det = 0;
static void moba_det_init(void) {
  static int done = 0;
  if (done) return;
  done = 1;
  const char* e = getenv("PUFFER_MOBA_DETERMINISTIC");
  g_moba_det = (e && e[0] && e[0] != '0') ? 1 : 0;
}

/* --- (2) publish protocol for the shared ai_paths cache ---------------------------------------
   ai_paths is 128*128 destination blocks of 128*128 bytes. Under the flag, each block is filled by
   exactly one thread under the block's mutex and published with a release store, and NOTHING reads
   a block before the matching acquire load says published — that second half is what kills the torn
   read of the twice-written destination cell. The mutex is STRIPED so fills of different
   destinations still run in parallel (a single global lock serialized every BFS sweep and cost ~10%
   SPS); same block => same stripe => still exactly one filler. Once published a block is immutable,
   so the steady-state hot path is one byte load and no lock.
   Flag off: upstream's exact code — read the cell, bfs on a 255 miss, no synchronisation. */
#define MOBA_NBLK    (128*128)
#define MOBA_NSTRIPE 64
static _Atomic unsigned char g_moba_bfs_done[MOBA_NBLK];
static pthread_mutex_t g_moba_bfs_mu[MOBA_NSTRIPE];
#define MOBA_BFS_ENSURE(env, blk, cell, ydst, xdst) do {                                  \
    if (!g_moba_det) {                       /* upstream (default) */                     \
      if ((env)->ai_paths[cell] == 255)                                                   \
        bfs((env)->map, &(env)->ai_paths[blk], (env)->ai_path_buffer, (ydst), (xdst));    \
    } else {                                                                              \
      int _b = (blk) / MOBA_NBLK;                                                         \
      if (!atomic_load_explicit(&g_moba_bfs_done[_b], memory_order_acquire)) {            \
        pthread_mutex_lock(&g_moba_bfs_mu[_b % MOBA_NSTRIPE]);                            \
        if (!atomic_load_explicit(&g_moba_bfs_done[_b], memory_order_relaxed)) {          \
          bfs((env)->map, &(env)->ai_paths[blk], (env)->ai_path_buffer, (ydst), (xdst));  \
          atomic_store_explicit(&g_moba_bfs_done[_b], 1, memory_order_release);           \
        }                                                                                 \
        pthread_mutex_unlock(&g_moba_bfs_mu[_b % MOBA_NSTRIPE]);                          \
      }                                                                                   \
    }                                                                                     \
  } while (0)

/* --- (1) per-env rand(). Range [0, 2^31-1] == [0, RAND_MAX], matching what moba.h's `rand()%k` and
   `(float)rand()/RAND_MAX` uses expect. g_moba_rs points at the CURRENT env's state and is set by
   the c_step/c_reset wrappers below (and by ocean_setup around init_moba); the fallback only ever
   serves a call made outside those, which no moba code path does.
   Flag off: the real libc rand(), captured here BEFORE the macro shadows the name. ------------- */
static int moba_libc_rand(void) { return rand(); }
static _Thread_local unsigned long long* g_moba_rs = NULL;
static unsigned long long g_moba_rs_fallback = 0x9E3779B97F4A7C15ULL;
static int moba_rand(void) {
  if (!g_moba_det) return moba_libc_rand();
  unsigned long long* s = g_moba_rs ? g_moba_rs : &g_moba_rs_fallback;
  unsigned long long x = *s;
  x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
  *s = x;
  return (int)((x * 0x2545F4914F6CDD1DULL) >> 33);
}
#define rand() moba_rand()

#include "moba.h"

/* Point the per-env generator at `e` before any moba code that can draw from rand() runs. Both
   macros expand their argument twice, so callers must pass a side-effect-free expression (they do:
   `e` / `&w->envs[n]` in ffi/ocean_adapter.h). The inner c_step/c_reset are not re-expanded (C
   macros do not recurse), so they stay the real functions. Harmless when the flag is off — one
   dead store per env-step, and moba_rand ignores g_moba_rs in that case. */
static inline void moba_enter(MOBA* e) { g_moba_rs = &e->rand_state; }
#define c_step(e)  (moba_enter(e), c_step(e))
#define c_reset(e) (moba_enter(e), c_reset(e))

#define ENV_T MOBA
#define OBS_T unsigned char
#define ACT_T float
#define OCEAN_NHEADS 6
#define OCEAN_HEADSIZES {7, 7, 3, 2, 2, 2}
#define OCEAN_NAGENTS(e) ((e)->num_agents)

#define MOBA_OBS_PER_AGENT (11*11*4 + 26)   /* 510 */

static int cfg_int(const char* cfg, const char* key, int def) {
  if (!cfg) return def;
  size_t klen = strlen(key);
  const char* p = cfg;
  while (*p) {
    const char* q = p; while (*q && *q != ',') q++;
    if ((size_t)(q - p) > klen && strncmp(p, key, klen) == 0 && p[klen] == '=') return atoi(p + klen + 1);
    p = (*q == ',') ? q + 1 : q;
  }
  return def;
}
static float cfg_float(const char* cfg, const char* key, float def) {
  if (!cfg) return def;
  size_t klen = strlen(key);
  const char* p = cfg;
  while (*p) {
    const char* q = p; while (*q && *q != ',') q++;
    if ((size_t)(q - p) > klen && strncmp(p, key, klen) == 0 && p[klen] == '=') return (float)atof(p + klen + 1);
    p = (*q == ',') ? q + 1 : q;
  }
  return def;
}

/* 256 MB BFS path cache — same procedural map for every env copy, so allocate once and share. */
static unsigned char* g_ai_paths = NULL;

static void ocean_setup(MOBA* e, uint64_t seed, int idx, const char* cfg) {
  moba_det_init();                /* the ONE read of PUFFER_MOBA_DETERMINISTIC, before any stepping */
  e->vision_range     = cfg_int(cfg,   "vision-range",     5);
  e->agent_speed      = cfg_float(cfg, "agent-speed",      1.0f);
  e->reward_death     = cfg_float(cfg, "reward-death",     -0.163764f);
  e->reward_xp        = cfg_float(cfg, "reward-xp",        0.00665677f);
  e->reward_distance  = cfg_float(cfg, "reward-distance",  0.0f);
  e->reward_tower     = cfg_float(cfg, "reward-tower",     0.642119f);
  e->script_opponents = cfg_int(cfg,   "script-opponents", 1);
  /* num_agents is derived from script_opponents exactly like binding.c my_init. */
  e->num_agents = e->script_opponents ? (NUM_PLAYERS / 2) : NUM_PLAYERS;

  int agents = e->num_agents;
  e->observations = (unsigned char*)calloc((size_t)agents * MOBA_OBS_PER_AGENT, sizeof(unsigned char));
  e->actions      = (float*)calloc((size_t)agents * OCEAN_NHEADS, sizeof(float));
  e->rewards      = (float*)calloc((size_t)agents, sizeof(float));
  e->terminals    = (float*)calloc((size_t)agents, sizeof(float));
  e->truncations  = (unsigned char*)calloc((size_t)agents, sizeof(unsigned char));

  e->ai_path_buffer = (int*)calloc(3 * 8 * 128 * 128, sizeof(int));
  if (!g_ai_paths) {
    g_ai_paths = (unsigned char*)malloc((size_t)128 * 128 * 128 * 128);
    memset(g_ai_paths, 255, (size_t)128 * 128 * 128 * 128);
    /* env construction is serial and strictly precedes any stepping, so this is the safe place */
    for (int i = 0; i < MOBA_NSTRIPE; i++) pthread_mutex_init(&g_moba_bfs_mu[i], NULL);
  }
  e->ai_paths = g_ai_paths;

  /* Vary the per-env cached RNG (init_moba fills env->rng from rand()). Default: srand() on the
     process-global generator, exactly as upstream. Under the flag rand() is per-env, so seed THIS
     env's state instead — same intent, no global state, and the draws no longer depend on which
     thread later steps the env. splitmix64 finalizer, forced nonzero (xorshift64* fixes 0). */
  if (!g_moba_det) {
    srand((unsigned int)(seed + (uint64_t)idx * 0x9E3779B9u));
  } else {
    uint64_t z = (seed + (uint64_t)idx * 0x9E3779B97F4A7C15ULL) + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    e->rand_state = (z ^ (z >> 31)) | 1ULL;
    g_moba_rs = &e->rand_state;  /* init_moba draws from rand() (CachedRNG fill + entity spawns) */
  }
  init_moba(e, game_map_npy);   /* mallocs entities/map/rng — NOT the caller buffers above */
}

static int  ocean_obsdim(MOBA* e)     { (void)e; return MOBA_OBS_PER_AGENT; }        /* 510 */
static int  ocean_numactions(MOBA* e) { (void)e; return 7 + 7 + 3 + 2 + 2 + 2; }     /* 23 total logits */
static int  ocean_maxsteps(MOBA* e)   { (void)e; return 1000; }

static void ocean_teardown(MOBA* e) {
  c_close(e);   /* frees entities, reward_components, map(+grid/pids), orig_grid, rng, ai_path_buffer */
  free(e->observations);
  free(e->actions);
  free(e->rewards);
  free(e->terminals);
  free(e->truncations);
  /* e->ai_paths is the shared g_ai_paths — leaked once at process exit, never per-env freed. */
}

/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(reward_death) X(reward_xp) X(reward_distance) \
  X(reward_tower) X(radiant_victory) X(radiant_level) X(radiant_towers_alive) X(dire_victory) \
  X(dire_level) X(dire_towers_alive) X(radiant_support_episode_return) X(radiant_support_reward_death) \
  X(radiant_support_reward_xp) X(radiant_support_reward_distance) X(radiant_support_reward_tower) \
  X(radiant_support_level) X(radiant_support_kills) X(radiant_support_deaths) \
  X(radiant_support_damage_dealt) X(radiant_support_damage_received) X(radiant_support_healing_dealt) \
  X(radiant_support_healing_received) X(radiant_support_creeps_killed) X(radiant_support_neutrals_killed) \
  X(radiant_support_towers_killed) X(radiant_support_usage_auto) X(radiant_support_usage_q) \
  X(radiant_support_usage_w) X(radiant_support_usage_e) X(n)
#include "../../ffi/ocean_adapter.h"
