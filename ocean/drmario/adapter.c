/* Adapter: PufferLib Ocean drmario -> pufferlean plugin ABI. Runs the REAL ocean/drmario/drmario.h c_reset/c_step. */
#include "drmario.h"
#include <string.h>
#include <stdint.h>
#define ENV_T DrMario
#define OBS_T float
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
static void ocean_setup(DrMario* e,uint64_t seed,int idx,const char* cfg){
  e->num_agents=1;
  e->n_rows=cfg_int(cfg,"n-rows",16);
  e->n_cols=cfg_int(cfg,"n-cols",8);
  e->n_init_viruses=cfg_int(cfg,"n-init-viruses",14);
  e->rng=(unsigned int)(seed+(uint64_t)idx*0x9E3779B9u);
  allocate(e);
}
static int  ocean_obsdim(DrMario* e){ return e->n_rows*e->n_cols*N_OBS_PLANES + N_SCALAR_OBS; }
static int  ocean_numactions(DrMario* e){ (void)e; return 7; }
static int  ocean_maxsteps(DrMario* e){ (void)e; return 100000; }
static void ocean_teardown(DrMario* e){ free_allocated(e); }
/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(viruses_cleared) X(n)
#include "../../ffi/ocean_adapter.h"
