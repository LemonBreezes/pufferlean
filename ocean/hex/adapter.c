/* Adapter: PufferLib Ocean hex -> pufferlean plugin ABI. Runs the REAL ocean/hex/hex.h c_reset/c_step. */
#include "hex.h"
#include <string.h>
#include <stdint.h>
#define ENV_T Hex
#define OBS_T float
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
static void ocean_setup(Hex* e,uint64_t seed,int idx,const char* cfg){
  e->num_agents=1;
  e->random_opponent=cfg_int(cfg,"random-opponent",0);
  e->rng=(unsigned int)(seed+(uint64_t)idx*0x9E3779B9u);
  e->observations=(float*)calloc(2*TOTAL_CELLS,sizeof(float));
  e->actions=(float*)calloc(1,sizeof(float));
  e->rewards=(float*)calloc(1,sizeof(float));
  e->terminals=(float*)calloc(1,sizeof(float));
}
static int  ocean_obsdim(Hex* e){ (void)e; return 2*TOTAL_CELLS; }
static int  ocean_numactions(Hex* e){ (void)e; return TOTAL_CELLS; }
static int  ocean_maxsteps(Hex* e){ (void)e; return TOTAL_CELLS+1; }
static void ocean_teardown(Hex* e){ free(e->observations);free(e->actions);free(e->rewards);free(e->terminals); }
/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(n)
#include "../../ffi/ocean_adapter.h"
