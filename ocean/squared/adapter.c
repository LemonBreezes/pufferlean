/* Adapter: PufferLib Ocean `squared` → pufferlean plugin ABI. Runs the REAL ocean/squared/squared.h
 * c_reset/c_step (random target via rand_r, uint8 grid obs). Build via ocean/build.sh. */
#include "squared.h"
#include <string.h>
#include <stdint.h>

#define ENV_T Squared
#define OBS_T unsigned char

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

static void ocean_setup(Squared* e, uint64_t seed, int idx, const char* cfg) {
  e->num_agents = 1;
  e->size = cfg_int(cfg, "size", 11);
  e->rng = (unsigned int)(seed + (uint64_t)idx * 0x9E3779B9u);   /* distinct per copy */
  int tiles = e->size * e->size;
  e->observations = (unsigned char*)calloc(tiles, sizeof(unsigned char));
  e->actions   = (float*)calloc(1, sizeof(float));
  e->rewards   = (float*)calloc(1, sizeof(float));
  e->terminals = (float*)calloc(1, sizeof(float));
}
static int  ocean_obsdim(Squared* e)     { return e->size * e->size; }
static int  ocean_numactions(Squared* e) { (void)e; return 5; }
static int  ocean_maxsteps(Squared* e)   { return 3 * e->size + 2; }
static void ocean_teardown(Squared* e)   { free(e->observations); free(e->actions); free(e->rewards); free(e->terminals); }

/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(n)
#include "../../ffi/ocean_adapter.h"
