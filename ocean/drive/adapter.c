/* Adapter: PufferLib Ocean `drive` -> puffer-lean plugin ABI. Runs the REAL ocean/drive/drive.h
 * c_reset/c_step (Waymo-derived driving, float obs, MULTI-DISCRETE {7,13} actions).
 *
 * drive is multi-agent: the number of controlled agents (active_agent_count) is determined by the
 * binary MAP FILE, not by config. allocate() calls init() (which reads the map + sizes the agent
 * set) and THEN sizes the 4 caller buffers by active_agent_count -- so we call allocate() rather
 * than mallocing buffers ourselves. Every env instance is pointed at the SAME map so they all share
 * one uniform per-env agent count (the flat batch is N * active_agent_count rows).
 *
 * The map binaries are not in this repo; they live in the sibling PufferLib checkout. */
#include "drive.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#define ENV_T Drive
#define OBS_T float
#define ACT_T float               /* env->actions is float* (accel index, steering index) */
#define REW_T float
#define TERM_T float
#define OCEAN_NHEADS 2
#define OCEAN_HEADSIZES {7, 13}
#define OCEAN_NAGENTS(e) ((e)->active_agent_count)

/* Binary maps (map_%03d.bin) are vendored next to this adapter (ocean/drive/); fall back to the sibling
   PufferLib checkout, or $DRIVE_MAP_DIR, if the vendored copy isn't found (e.g. run from another cwd). */
#ifndef DRIVE_MAP_DIR
#define DRIVE_MAP_DIR "ocean/drive"
#endif
#include <unistd.h>
/* Return the first "%s/map_%03d.bin" that exists, into out; tries $DRIVE_MAP_DIR then the vendored dir. */
static void drive_map_path(char* out, size_t n, int map_id) {
  const char* dirs[3]; int k = 0;
  const char* envd = getenv("DRIVE_MAP_DIR");
  if (envd) dirs[k++] = envd;
  dirs[k++] = DRIVE_MAP_DIR;
  for (int i = 0; i < k; i++) {
    snprintf(out, n, "%s/map_%03d.bin", dirs[i], map_id);
    if (access(out, R_OK) == 0) return;
  }
  snprintf(out, n, "%s/map_%03d.bin", DRIVE_MAP_DIR, map_id);   /* last resort: let fopen report */
}

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

static void ocean_setup(Drive* e, uint64_t seed, int idx, const char* cfg) {
  /* config fields mirror ocean/drive/binding.c my_init (dash-normalized keys) */
  e->human_agent_idx = cfg_int(cfg, "human-agent-idx", 0);
  e->reward_vehicle_collision = cfg_float(cfg, "reward-vehicle-collision", -0.2f);
  e->reward_offroad_collision = cfg_float(cfg, "reward-offroad-collision", -0.2f);
  e->reward_goal_post_respawn = cfg_float(cfg, "reward-goal-post-respawn", 0.0f);
  e->reward_vehicle_collision_post_respawn =
      cfg_float(cfg, "reward-vehicle-collision-post-respawn", 0.0f);
  /* Cap the controlled-agent set so smoke tests stay cheap; active_agent_count = min(valid, cap). */
  e->max_agents = cfg_int(cfg, "max-agents", 8);

  int map_id = cfg_int(cfg, "map-id", 10);   /* map_010.bin (also 942 available) */
  char path[512];
  drive_map_path(path, sizeof(path), map_id);
  e->map_name = strdup(path);

  e->rng = (unsigned int)(seed + (uint64_t)idx * 0x9E3779B9u);

  allocate(e);   /* init(): reads map, builds agent set; then sizes the 4 buffers by active_agent_count */
  e->num_agents = e->active_agent_count;
}

static int  ocean_obsdim(Drive* e)     { (void)e; return OBS_SIZE; }        /* 973 */
static int  ocean_numactions(Drive* e) { (void)e; return 7 + 13; }          /* sum of head sizes */
static int  ocean_maxsteps(Drive* e)   { (void)e; return 1000; }            /* env self-resets at 91 */
static void ocean_teardown(Drive* e) {
  free_allocated(e);   /* frees observations/actions/rewards/terminals + c_close (internal arrays) */
  free(e->map_name);
}

/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(episode_return) X(episode_length) X(perf) X(score) X(offroad_rate) X(collision_rate) \
  X(clean_collision_rate) X(completion_rate) X(dnf_rate) X(n)
#include "../../ffi/ocean_adapter.h"
