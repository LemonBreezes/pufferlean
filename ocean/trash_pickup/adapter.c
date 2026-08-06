/* Adapter: PufferLib Ocean trash_pickup -> puffer-lean plugin ABI. Runs the REAL ocean/trash_pickup/trash_pickup.h c_reset/c_step. */
#include "trash_pickup.h"
#include <string.h>
#include <stdint.h>
#define ENV_T CTrashPickupEnv
#define OBS_T unsigned char
/* ACT_T: actions field is float* -> float (no #define) */
/* TERM_T: terminals field is float* -> float (no #define) */
#define OCEAN_NAGENTS(e) ((e)->num_agents)   /* multi-agent: num_agents>1 */
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
static float cfg_flt(const char* cfg,const char* key,float def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return (float)atof(p+k+1); p=(*q==',')?q+1:q; } return def; }
static void ocean_setup(ENV_T* e,uint64_t seed,int idx,const char* cfg){
  e->num_agents        = cfg_int(cfg,"num-agents",8);
  e->grid_size         = cfg_int(cfg,"grid-size",20);
  e->num_trash         = cfg_int(cfg,"num-trash",40);
  e->num_bins          = cfg_int(cfg,"num-bins",2);
  e->max_steps         = cfg_int(cfg,"max-steps",500);
  e->agent_sight_range = cfg_int(cfg,"agent-sight-range",5);
  e->rng = (unsigned int)(seed + (uint64_t)idx*0x9E3779B9u);
  allocate(e);   /* sizes the four buffers by num_agents */
}
static int  ocean_obsdim(ENV_T* e){ int d=2*e->agent_sight_range+1; return d*d*5; }
static int  ocean_numactions(ENV_T* e){ (void)e; return 4; }
static int  ocean_maxsteps(ENV_T* e){ return e->max_steps; }
static void ocean_teardown(ENV_T* e){ free_allocated(e); }
/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(trash_collected) X(n)
#include "../../ffi/ocean_adapter.h"
