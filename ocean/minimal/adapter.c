/* Adapter: PufferLib Ocean `minimal` -> puffer-lean plugin ABI.
 * Multi-agent coordination env (AGENTS puffers catching TARGETS of matching type).
 * MULTI-DISCRETE: 2 heads {9,5} (actions[2*i] heading adjust 0-8, actions[2*i+1]
 * speed adjust 0-4). float obs/actions/rewards/terminals.
 * No init()/allocate(): the Entity array lives inline in the struct, so we only
 * malloc the 4 caller-owned buffers, each sized num_agents.
 * num_agents is FIXED at AGENTS (compute_observations/c_step loop over the AGENTS
 * macro, not env->num_agents), so config can't change it. */
#include "minimal.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#define ENV_T Env
#define OBS_T float
#define ACT_T float
#define OCEAN_NHEADS 2
#define OCEAN_HEADSIZES {9, 5}
#define OCEAN_NAGENTS(e) ((e)->num_agents)

/* per-agent obs length (== binding.c OBS_SIZE) = 2 + 4*(AGENTS+TARGETS) = 66 */
#define OBS_PER_AGENT (2 + 4*(AGENTS + TARGETS))

static void ocean_setup(Env* e, uint64_t seed, int idx, const char* cfg) { (void)cfg;
  e->num_agents = AGENTS;   /* env loops are hardcoded to the AGENTS macro; must stay 8 */
  e->rng = (unsigned int)(seed + (uint64_t)idx * 0x9E3779B9u);   /* distinct per copy */

  int nA = e->num_agents;
  e->observations = (float*)calloc((size_t)nA * OBS_PER_AGENT, sizeof(float));
  e->actions      = (float*)calloc((size_t)nA * OCEAN_NHEADS, sizeof(float));
  e->rewards      = (float*)calloc((size_t)nA, sizeof(float));
  e->terminals    = (float*)calloc((size_t)nA, sizeof(float));
  /* no init()/allocate(): Entity entities[AGENTS+TARGETS] is inline in the struct */
}
static int  ocean_obsdim(Env* e)     { (void)e; return OBS_PER_AGENT; }
static int  ocean_numactions(Env* e) { (void)e; return 9 + 5; }   /* sum of head sizes (total logits) */
static int  ocean_maxsteps(Env* e)   { (void)e; return 1000; }
static void ocean_teardown(Env* e) {
  free(e->observations); free(e->actions); free(e->rewards); free(e->terminals);
}

/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(n)
#include "../../ffi/ocean_adapter.h"
