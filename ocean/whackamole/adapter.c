/* Adapter: PufferLib Ocean whackamole -> pufferlean plugin ABI. Runs the REAL ocean/whackamole/whackamole.h c_reset/c_step. */
#include "whackamole.h"
#include <string.h>
#include <stdint.h>
#define ENV_T Whackamole
#define OBS_T float
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
static void ocean_setup(Whackamole* e,uint64_t seed,int idx,const char* cfg){ (void)cfg;
  e->num_agents=1;
  e->hits=0;
  e->tick=0;
  e->client=NULL;
  e->rng=(unsigned int)(seed+(uint64_t)idx*0x9E3779B9u);
  e->observations=(float*)calloc(TOTAL_CELLS,sizeof(float));
  e->actions=(float*)calloc(1,sizeof(float));
  e->rewards=(float*)calloc(1,sizeof(float));
  e->terminals=(float*)calloc(1,sizeof(float));
}
static int  ocean_obsdim(Whackamole* e){ (void)e; return TOTAL_CELLS; }
static int  ocean_numactions(Whackamole* e){ (void)e; return TOTAL_CELLS; }
static int  ocean_maxsteps(Whackamole* e){ (void)e; return ATTEMPTS_PER_EPISODE; }
static void ocean_teardown(Whackamole* e){ free(e->observations);free(e->actions);free(e->rewards);free(e->terminals); }
/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(n)
#include "../../ffi/ocean_adapter.h"
