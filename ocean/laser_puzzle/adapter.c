#include "laser_puzzle.h"
#include <string.h>
#include <stdint.h>
#define ENV_T LaserPuzzle
#define OBS_T unsigned char
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
static float cfg_flt(const char* cfg,const char* key,float def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return (float)atof(p+k+1); p=(*q==',')?q+1:q; } return def; }
static void ocean_setup(LaserPuzzle* e,uint64_t seed,int idx,const char* cfg){
  e->num_agents=1;
  /* allocate() sets ROWS/COLS, max_steps=NUM_ACTIONS, num_agents AND `rng = 0`
     (laser_puzzle.h:120-124), so it MUST run before the per-copy seed is written — otherwise every
     copy shares rng=0. That is what used to happen: `c_reset` picks its level with
     rand_r(&env->rng) % num_levels (laser_puzzle.h:260), so all copies drew the SAME level and, with
     every episode truncating at exactly max_steps, stayed in lockstep — 1 distinct board across 256
     copies for the first ~480 steps. The reported episode_return was then one level's value
     replicated across the batch, and --train.seed did nothing at all.
     Do NOT hoist allocate() or drop it: it is also what makes max_steps correct (NUM_ACTIONS=48).
     Upstream seeds per copy the same way (src/vecenv.h:364, `envs[i].rng = i`). */
  allocate(e);
  e->rng=(unsigned int)(seed+(uint64_t)idx*0x9E3779B9u);
}
static int ocean_obsdim(LaserPuzzle* e){ (void)e; return INIT_ROWS*INIT_COLS; }
/* Action space is NUM_ACTIONS = ACTIONS_PER_CELL * INNER_ROWS * INNER_COLS = 3*4*4 = 48, matching
   upstream's `#define ACT_SIZES {NUM_ACTIONS}` (binding.c:5). This said 5, so the policy could only
   ever emit actions 0..4 — and c_step decodes `cell_idx = action / ACTIONS_PER_CELL` (laser_puzzle.h:216),
   so it could reach just 2 of the 16 inner cells. That is why training plateaued at 0.30 and decayed
   while PufferLib reached +1.454 on an identical config. */
static int ocean_numactions(LaserPuzzle* e){ (void)e; return NUM_ACTIONS; }
static int ocean_maxsteps(LaserPuzzle* e){ return e->max_steps; }
static void ocean_teardown(LaserPuzzle* e){ deallocate(e); }
/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(n)
#include "../../ffi/ocean_adapter.h"
