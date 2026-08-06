/* Adapter: PufferLib Ocean `terraform` -> puffer-lean plugin ABI. MULTI-AGENT MULTI-DISCRETE:
 * num_agents dozers per instance, 3 discrete heads (ACT_SIZES {5,5,3}), FloatTensor 319-dim obs.
 * Mirrors binding.c my_init (config -> rng -> init) then drives the REAL terraform c_reset/c_step
 * (which auto-resets internally on reset_frequency / when the map is solved). */
#include "terraform.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#define ENV_T Terraform
#define OBS_T float
#define ACT_T float
#define REW_T float
#define TERM_T float
#define OCEAN_NHEADS 3
#define OCEAN_HEADSIZES {5, 5, 3}
#define OCEAN_NAGENTS(e) ((e)->num_agents)

#define TF_OBS 319

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

static void ocean_setup(Terraform* e, uint64_t seed, int idx, const char* cfg) {
  /* config (mirror my_init). NOTE: size must stay 64 -> num_quadrants=36 -> obs exactly 319. */
  e->num_agents      = cfg_int(cfg, "num-agents", 8);
  e->size            = cfg_int(cfg, "size", 64);
  e->reset_frequency = cfg_int(cfg, "reset-frequency", 1024);
  e->reward_scale    = cfg_float(cfg, "reward-scale", 0.0907157f);
  e->client          = NULL;
  e->rng             = (unsigned int)(seed + (uint64_t)idx * 0x9E3779B9u);   /* distinct per copy */

  /* caller-owned flat buffers (nAgents laid out contiguous) */
  e->observations = (float*)calloc((size_t)e->num_agents * TF_OBS, sizeof(float));
  e->actions      = (float*)calloc((size_t)e->num_agents * OCEAN_NHEADS, sizeof(float));
  e->rewards      = (float*)calloc((size_t)e->num_agents, sizeof(float));
  e->terminals    = (float*)calloc((size_t)e->num_agents, sizeof(float));

  init(e);   /* allocates internal maps/dozers/returns; uses rng + config set above */
}
static int  ocean_obsdim(Terraform* e)     { (void)e; return TF_OBS; }
static int  ocean_numactions(Terraform* e) { (void)e; return 13; }   /* Sum of head sizes 5+5+3 */
static int  ocean_maxsteps(Terraform* e)   { return e->reset_frequency > 0 ? e->reset_frequency : 1024; }
static void ocean_teardown(Terraform* e) {
  free_initialized(e);   /* frees the internal arrays init() allocated (incl. returns) */
  free(e->observations); free(e->actions); free(e->rewards); free(e->terminals);
}

/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(n) X(quadrant_progress)
#include "../../ffi/ocean_adapter.h"
