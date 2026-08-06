/* Adapter: PufferLib Ocean tripletriad -> puffer-lean plugin ABI. Runs the REAL ocean/tripletriad/tripletriad.h c_reset/c_step. */
#include "tripletriad.h"
#include <string.h>
#include <stdint.h>
#define ENV_T CTripleTriad
#define OBS_T float
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
static void ocean_setup(CTripleTriad* e,uint64_t seed,int idx,const char* cfg){
  e->num_agents=1;
  e->width=cfg_int(cfg,"width",990);
  e->height=cfg_int(cfg,"height",690);
  e->card_width=cfg_int(cfg,"card-width",192);
  e->card_height=cfg_int(cfg,"card-height",224);
  e->rng=(unsigned int)(seed+(uint64_t)idx*0x9E3779B9u);
  allocate_ctripletriad(e);
}
static int  ocean_obsdim(CTripleTriad* e){ (void)e; return 114; }
static int  ocean_numactions(CTripleTriad* e){ (void)e; return 14; }
static int  ocean_maxsteps(CTripleTriad* e){ (void)e; return MAX_EPISODE_LENGTH; }
static void ocean_teardown(CTripleTriad* e){ free_allocated_ctripletriad(e); }
/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(n)
#include "../../ffi/ocean_adapter.h"
