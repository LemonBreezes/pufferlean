/* Adapter: PufferLib Ocean craftax_classic -> puffer-lean plugin ABI. Runs the REAL ocean/craftax_classic/craftax_classic.h c_reset/c_step. */
#include "craftax_classic.h"
#include <string.h>
#include <stdint.h>
#define ENV_T CraftaxClassic
#define OBS_T float
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
static void ocean_setup(CraftaxClassic* e,uint64_t seed,int idx,const char* cfg){
  (void)cfg;
  e->num_agents=1;
  e->rng=(unsigned int)(seed+(uint64_t)idx*0x9E3779B9u);
  e->observations=(float*)calloc(OBS_DIM,sizeof(float));
  e->actions=(float*)calloc(1,sizeof(float));
  e->rewards=(float*)calloc(1,sizeof(float));
  e->terminals=(float*)calloc(1,sizeof(float));
  c_init(e);
}
static int  ocean_obsdim(CraftaxClassic* e){ (void)e; return OBS_DIM; }
static int  ocean_numactions(CraftaxClassic* e){ (void)e; return 17; }
static int  ocean_maxsteps(CraftaxClassic* e){ (void)e; return MAX_TIMESTEPS; }
static void ocean_teardown(CraftaxClassic* e){ free(e->observations);free(e->actions);free(e->rewards);free(e->terminals); }
/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
/* non-scalar Log members not individually named (still folded into the aggregate): achievements[NUM_ACHIEVEMENTS] */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(n)
#include "../../ffi/ocean_adapter.h"
