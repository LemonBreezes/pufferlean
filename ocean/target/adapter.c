/* Adapter: PufferLib Ocean `target` -> puffer-lean plugin ABI.
 * Multi-agent env (puffers chasing stars). MULTI-DISCRETE: 2 heads {9,5}
 * (heading adjust 0-8, speed adjust 0-4). float obs/actions/rewards/terminals.
 * init() mallocs internal agents[num_agents] + goals[num_goals]; the 4 caller
 * buffers are malloc'd here, each sized num_agents. */
#include "target.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#define ENV_T Target
#define OBS_T float
#define ACT_T float
#define OCEAN_NHEADS 2
#define OCEAN_HEADSIZES {9, 5}
#define OCEAN_NAGENTS(e) ((e)->num_agents)

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

/* per-agent obs length: num_goals*2 + num_agents*2 + 4  (see compute_observations) */
static int obs_per_agent(Target* e) { return e->num_goals * 2 + e->num_agents * 2 + 4; }

static void ocean_setup(Target* e, uint64_t seed, int idx, const char* cfg) {
  e->width      = cfg_int(cfg, "width", 952);
  e->height     = cfg_int(cfg, "height", 592);
  e->num_agents = cfg_int(cfg, "num-agents", 8);
  e->num_goals  = cfg_int(cfg, "num-goals", 4);
  e->rng = (unsigned int)(seed + (uint64_t)idx * 0x9E3779B9u);   /* distinct per copy */

  int nA = e->num_agents;
  e->observations = (float*)calloc((size_t)nA * obs_per_agent(e), sizeof(float));
  e->actions      = (float*)calloc((size_t)nA * OCEAN_NHEADS, sizeof(float));
  e->rewards      = (float*)calloc((size_t)nA, sizeof(float));
  e->terminals    = (float*)calloc((size_t)nA, sizeof(float));

  init(e);   /* mallocs env->agents[num_agents] and env->goals[num_goals] */
}
static int  ocean_obsdim(Target* e)     { return obs_per_agent(e); }
static int  ocean_numactions(Target* e) { (void)e; return 9 + 5; }   /* sum of head sizes (total logits) */
static int  ocean_maxsteps(Target* e)   { (void)e; return 1000; }
static void ocean_teardown(Target* e) {
  c_close(e);   /* frees agents + goals (client is NULL, never rendered) */
  free(e->observations); free(e->actions); free(e->rewards); free(e->terminals);
}

/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(n)
#include "../../ffi/ocean_adapter.h"
