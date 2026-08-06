/* Adapter: PufferLib Ocean `convert` -> pufferlean plugin ABI.
 * MULTI-DISCRETE: 2 heads {9,5}. float obs (28 per agent when num_resources=8),
 * DOUBLE actions (Convert.actions is double*), float rewards/terminals.
 * Multi-agent: struct has int num_agents. init() mallocs agents/factories. */
#include "convert.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#define ENV_T Convert
#define OBS_T float
#define ACT_T double
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

static int obs_per_agent(Convert* e) { return 3 * e->num_resources + 4; }

static void ocean_setup(Convert* e, uint64_t seed, int idx, const char* cfg) {
  (void)seed; (void)idx;
  e->num_agents    = cfg_int(cfg, "num-agents", 8);
  e->width         = cfg_int(cfg, "width", 1920);
  e->height        = cfg_int(cfg, "height", 1080);
  e->num_factories = cfg_int(cfg, "num-factories", 32);
  e->num_resources = cfg_int(cfg, "num-resources", 8);

  int nA = e->num_agents;
  e->observations = (float*)calloc((size_t)nA * obs_per_agent(e), sizeof(float));
  e->actions      = (double*)calloc((size_t)nA * OCEAN_NHEADS, sizeof(double));
  e->rewards      = (float*)calloc(nA, sizeof(float));
  e->terminals    = (float*)calloc(nA, sizeof(float));
  init(e);   /* mallocs internal agents[num_agents] + factories[num_factories] */
}
static int  ocean_obsdim(Convert* e)     { return obs_per_agent(e); }
static int  ocean_numactions(Convert* e) { (void)e; return 9 + 5; }   /* Sum of head sizes */
static int  ocean_maxsteps(Convert* e)   { (void)e; return 1000; }
static void ocean_teardown(Convert* e) {
  c_close(e);   /* frees agents + factories (client is NULL, so untouched) */
  free(e->observations); free(e->actions); free(e->rewards); free(e->terminals);
}

/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(n)
#include "../../ffi/ocean_adapter.h"
