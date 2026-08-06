#include "g2048.h"
#include <string.h>
#include <stdint.h>
#define ENV_T Game
#define OBS_T unsigned char
/* g2048 is SINGLE-AGENT: binding.c my_init hardcodes num_agents=1 and c_step only touches [0]. PufferLib's
   num_agents=1024 is its VECTORIZATION count -> maps to --num-envs here. No OCEAN_NAGENTS override (default 1). */
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
static float cfg_flt(const char* cfg,const char* key,float def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return (float)atof(p+k+1); p=(*q==',')?q+1:q; } return def; }
static void ocean_setup(Game* e,uint64_t seed,int idx,const char* cfg){
  e->num_agents=1;                                   /* single-agent (matches binding.c my_init) */
  e->scaffolding_ratio=cfg_flt(cfg,"scaffolding-ratio",0.0f);
  e->rng=(unsigned int)(seed+(uint64_t)idx*0x9E3779B9u);
  e->observations=(unsigned char*)calloc(16,sizeof(unsigned char));   /* one 4x4 board */
  e->actions=(float*)calloc(1,sizeof(float));
  e->rewards=(float*)calloc(1,sizeof(float));
  e->terminals=(float*)calloc(1,sizeof(float));
  init(e);
}
static int ocean_obsdim(Game* e){ (void)e; return 16; }
static int ocean_numactions(Game* e){ (void)e; return 4; }
static int ocean_maxsteps(Game* e){ (void)e; return 100000; }
static void ocean_teardown(Game* e){ free(e->observations);free(e->actions);free(e->rewards);free(e->terminals); }
/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(merge_score) X(episode_return) X(episode_length) X(lifetime_max_tile) \
  X(reached_16384) X(reached_32768) X(reached_65536) X(reached_131072) X(n)
#include "../../ffi/ocean_adapter.h"
