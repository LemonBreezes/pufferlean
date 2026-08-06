#include "rware.h"
#include <string.h>
#include <stdint.h>
#define ENV_T CRware
#define OBS_T float
#define OCEAN_NAGENTS(e) ((e)->num_agents)
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
static float cfg_flt(const char* cfg,const char* key,float def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return (float)atof(p+k+1); p=(*q==',')?q+1:q; } return def; }
static void ocean_setup(CRware* e,uint64_t seed,int idx,const char* cfg){
  e->num_agents=cfg_int(cfg,"num-agents",8);
  e->width=cfg_int(cfg,"width",640); e->height=cfg_int(cfg,"height",360);
  e->grid_square_size=cfg_int(cfg,"grid-square-size",64); e->map_choice=cfg_int(cfg,"map-choice",1);
  e->num_requested_shelves=cfg_int(cfg,"num-requested-shelves",4); e->human_agent_idx=cfg_int(cfg,"human-agent-idx",0);
  e->rng=(unsigned int)(seed+(uint64_t)idx*0x9E3779B9u);
  allocate(e);
}
static int ocean_obsdim(CRware* e){ (void)e; return 27; }
static int ocean_numactions(CRware* e){ (void)e; return 5; }
static int ocean_maxsteps(CRware* e){ (void)e; return 100000; }
static void ocean_teardown(CRware* e){ free_allocated(e); }
/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(n)
#include "../../ffi/ocean_adapter.h"
