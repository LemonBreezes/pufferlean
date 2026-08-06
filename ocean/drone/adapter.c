/* Adapter: PufferLib Ocean `drone` → puffer-lean plugin ABI. MULTI-AGENT + CONTINUOUS: num_drones agents
 * per instance, 4 diagonal-Gaussian action dims (ACT_SIZES {1,1,1,1}), FloatTensor obs (DRONE_OBS_SIZE).
 * Replicates binding.c my_init (config → rng task pick → hover/race config → task_init → init) without the
 * Python Dict layer, then drives the REAL drone c_reset/c_step (physics + task reward, auto-reset on done). */
#include "drone.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#define ENV_T DroneEnv
#define OBS_T float
#define OCEAN_NHEADS 4                 /* action dim d */
#define OCEAN_HEADSIZES {1, 1, 1, 1}
#define OCEAN_CONTINUOUS 1             /* → Gaussian PPO path */
#define OCEAN_NAGENTS(e) ((e)->num_agents)

/* render.h (which defines c_close_client) is only pulled in by binding.c; stub it since we never render. */
void c_close_client(Client* client) { (void)client; }

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

static void ocean_setup(DroneEnv* e, uint64_t seed, int idx, const char* cfg) {
  e->num_agents  = cfg_int(cfg, "num-drones", 64);
  e->rng         = (unsigned int)(seed + (uint64_t)idx * 0x9E3779B9u);   /* distinct per copy */
  e->alpha_vel   = cfg_float(cfg, "alpha-vel", 0.0f);
  e->alpha_omega = cfg_float(cfg, "alpha-omega", 0.0f);
  e->alpha_action= cfg_float(cfg, "alpha-action", 0.0f);
  e->dr          = cfg_float(cfg, "dr", 0.05f);
  e->integrator  = cfg_int(cfg, "use-rk2", 0);
  e->client      = NULL;

  /* task fractions → pick this copy's task from its rng (mirrors my_init) */
  float fr[NUM_TASKS];
  fr[TASK_HOVER]  = cfg_float(cfg, "hover-frac", 0.8332634358752755f);
  fr[TASK_RACE]   = cfg_float(cfg, "race-frac",  0.6878962707768624f);
  fr[TASK_SPHERE] = cfg_float(cfg, "sphere-frac", 0.0f);
  fr[TASK_CUBE]   = cfg_float(cfg, "cube-frac", 0.0f);
  fr[TASK_FLAG]   = cfg_float(cfg, "flag-frac", 0.0f);
  float total = 0.0f; for (int t = 0; t < NUM_TASKS; t++) total += fr[t];
  int ridx = (int)e->rng; float cum = 0.0f; e->task = TASK_HOVER;
  for (int t = 0; t < NUM_TASKS; t++) {
    cum += fr[t] / total;
    if ((int)floorf((ridx + 1) * cum) > (int)floorf(ridx * cum)) { e->task = (TaskType)t; break; }
  }

  /* caller-owned flat buffers (nAgents laid out contiguous per the template) */
  e->observations = (float*)calloc((size_t)e->num_agents * DRONE_OBS_SIZE, sizeof(float));
  e->actions      = (float*)calloc((size_t)e->num_agents * 4, sizeof(float));
  e->rewards      = (float*)calloc((size_t)e->num_agents, sizeof(float));
  e->terminals    = (float*)calloc((size_t)e->num_agents, sizeof(float));

  /* task_config (mirror hover_config/race_config), then task_init + init */
  if (e->task == TASK_RACE) {
    RaceConfig* c = (RaceConfig*)calloc(1, sizeof(RaceConfig));
    c->max_rings  = cfg_int(cfg, "max-rings", 10);
    c->ring_reward= cfg_float(cfg, "ring-reward", 2.4450236350884f);
    c->alpha_dist = cfg_float(cfg, "race-alpha-dist", 2.8630645575928786f);
    c->horizon    = cfg_int(cfg, "race-horizon", 2048);
    e->task_config = c;
  } else {
    HoverConfig* c = (HoverConfig*)calloc(1, sizeof(HoverConfig));
    c->target_dist  = cfg_float(cfg, "hover-target-dist", 5.0f);
    c->alpha_hover  = cfg_float(cfg, "alpha-hover", 1.0f);
    c->alpha_dist   = cfg_float(cfg, "hover-alpha-dist", 0.8120191629018807f);
    c->sphere_radius= cfg_float(cfg, "sphere-radius", 4.0f);
    c->horizon      = cfg_int(cfg, "hover-horizon", 1024);
    e->task_config = c;
  }
  task_init(e);
  init(e);
}
static int  ocean_obsdim(DroneEnv* e)     { (void)e; return DRONE_OBS_SIZE; }
static int  ocean_numactions(DroneEnv* e) { (void)e; return 4; }        /* action dim (continuous) */
static int  ocean_maxsteps(DroneEnv* e)   { return task_horizon(e); }
static void ocean_teardown(DroneEnv* e) {                               /* = c_close minus client */
  task_close(e);                                                        /* frees task_state + task_config */
  for (int i = 0; i < e->num_agents; i++) free(e->agents[i].target);
  free(e->agents); physics_close(&e->physics);
  free(e->observations); free(e->actions); free(e->rewards); free(e->terminals);
}

/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
/* non-scalar Log members not individually named (still folded into the aggregate): task[NUM_TASKS] */
#define OCEAN_LOG_FIELDS(X) X(episode_return) X(episode_length) X(n)
#include "../../ffi/ocean_adapter.h"
