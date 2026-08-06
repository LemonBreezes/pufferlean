/* Adapter: PufferLib Ocean `robocode` → pufferlean plugin ABI. MULTI-AGENT + MULTI-DISCRETE:
 * num_agents robots per battle, 5 action heads (ACT_SIZES {4,9,11,11,6}), FloatTensor 16-dim obs.
 * Replicates binding.c my_init (config → init) then drives the REAL robocode c_reset/c_step
 * (all agents share one episode; end_episode resets the whole battle + sets every terminal). */
#include "robocode.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#define ENV_T Robocode
#define OBS_T float
#define ACT_T float                        /* env->actions is float* */
#define OCEAN_NHEADS 5
#define OCEAN_HEADSIZES {4, 9, 11, 11, 6}
#define OCEAN_NAGENTS(e) ((e)->num_agents)

static int cfg_int(const char* cfg, const char* key, int def) {
  if (!cfg) return def; size_t k = strlen(key); const char* p = cfg;
  while (*p) { const char* q = p; while (*q && *q != ',') q++;
    if ((size_t)(q - p) > k && strncmp(p, key, k) == 0 && p[k] == '=') return atoi(p + k + 1);
    p = (*q == ',') ? q + 1 : q; } return def;
}
static float cfg_float(const char* cfg, const char* key, float def) {
  if (!cfg) return def; size_t k = strlen(key); const char* p = cfg;
  while (*p) { const char* q = p; while (*q && *q != ',') q++;
    if ((size_t)(q - p) > k && strncmp(p, key, k) == 0 && p[k] == '=') return (float)atof(p + k + 1);
    p = (*q == ',') ? q + 1 : q; } return def;
}

static void ocean_setup(Robocode* e, uint64_t seed, int idx, const char* cfg) {
  /* config (dash-normalized keys), defaults from config/robocode.ini */
  e->width         = cfg_int(cfg, "width", 800);
  e->height        = cfg_int(cfg, "height", 600);
  /* HARD CAP 2: robocode's per-slot pointer arrays (obs_ptr/action_ptr/reward_ptr/terminal_ptr)
   * are declared size [2] in the struct — it is a 1v1 selfplay env. allocate_env loops s<num_agents
   * writing those arrays, so num_agents>2 smashes the struct (verified: crash w/ action_ptr[6]=garbage).
   * Clamp to [1,2]; the smoke test's --num-agents 8 therefore trains at 2. */
  e->num_agents    = cfg_int(cfg, "num-agents", 2);
  if (e->num_agents < 1) e->num_agents = 1;
  if (e->num_agents > 2) e->num_agents = 2;
  e->num_bots      = cfg_int(cfg, "num-bots", 0);
  e->max_ticks     = cfg_int(cfg, "max-ticks", 3000);
  e->reward_damage = cfg_float(cfg, "reward-damage", 0.0938468f);
  e->reward_spot   = cfg_float(cfg, "reward-spot", 0.00526184f);
  e->bot_policy    = cfg_int(cfg, "bot-policy", 3);
  e->client        = NULL;
  e->rng = (unsigned int)(seed + (uint64_t)idx * 0x9E3779B9u);   /* distinct per copy */

  /* allocate_env = init(env) (robots/bullets/logs/bot_mems) + mallocs the 4 caller-owned buffers
   * (observations 16·N, actions 5·N, rewards N, terminals N) AND wires the per-slot obs_ptr/
   * action_ptr/reward_ptr/terminal_ptr that compute_observations & c_step deref — layout matches
   * exactly what the wrapper reads/writes. */
  allocate_env(e);
}
static int  ocean_obsdim(Robocode* e)     { (void)e; return EGO_FEATURES + OTHER_FEATURES; }  /* 16 */
static int  ocean_numactions(Robocode* e) { (void)e; return 4 + 9 + 11 + 11 + 6; }            /* 41 = Σ heads */
static int  ocean_maxsteps(Robocode* e)   { return e->max_ticks; }
static void ocean_teardown(Robocode* e) {                          /* = c_close + free the 4 buffers */
  c_close(e);                                                      /* frees robots/bullets/logs/bot_mems */
  free(e->observations); free(e->actions); free(e->rewards); free(e->terminals);
}

/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
/* non-scalar Log members not individually named (still folded into the aggregate): hist_score_bank[ROBOCODE_MAX_BANKS], hist_n_bank[ROBOCODE_MAX_BANKS] */
#define OCEAN_LOG_FIELDS(X) X(perf) X(episode_return) X(episode_length) X(score) X(damage_received) X(hist_score) X(hist_n) \
  X(slot_0_score) X(slot_1_score) X(draw_rate) X(n)
#include "../../ffi/ocean_adapter.h"
