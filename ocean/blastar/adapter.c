/* Adapter: PufferLib Ocean `blastar` → pufferlean plugin ABI. Runs the REAL ocean/blastar/blastar.h
 * c_reset/c_step (float obs, double actions, float terminals). Single-agent, one discrete action of 6. */
#include "blastar.h"
#include <string.h>
#include <stdint.h>
#define ENV_T Blastar
#define OBS_T float
#define ACT_T double
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
static void ocean_setup(Blastar* e,uint64_t seed,int idx,const char* cfg){ (void)seed;(void)idx;
  e->num_agents=1;
  int num_obs=cfg_int(cfg,"num-obs",10);
  allocate(e,num_obs); }
static int  ocean_obsdim(Blastar* e){ return e->num_obs; }
static int  ocean_numactions(Blastar* e){ (void)e; return 6; }
static int  ocean_maxsteps(Blastar* e){ (void)e; return 100000; }
static void ocean_teardown(Blastar* e){ free_allocated(e); }
/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(lives) X(vertical_closeness_rew) \
  X(fired_bullet_rew) X(kill_streak) X(hit_enemy_with_bullet_rew) X(avg_score_difference) X(n)
#include "../../ffi/ocean_adapter.h"
