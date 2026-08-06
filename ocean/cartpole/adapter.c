/* Adapter: PufferLib Ocean cartpole -> pufferlean plugin ABI. Runs the REAL ocean/cartpole/cartpole.h c_reset/c_step. */
#include "cartpole.h"
#include <string.h>
#include <stdint.h>
#define ENV_T Cartpole
#define OBS_T float
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
static float cfg_flt(const char* cfg,const char* key,float def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return (float)atof(p+k+1); p=(*q==',')?q+1:q; } return def; }
static void ocean_setup(Cartpole* e,uint64_t seed,int idx,const char* cfg){
  e->num_agents=1;
  e->cart_mass=cfg_flt(cfg,"cart-mass",1.0f);
  e->pole_mass=cfg_flt(cfg,"pole-mass",0.1f);
  e->pole_length=cfg_flt(cfg,"pole-length",0.5f);
  e->gravity=cfg_flt(cfg,"gravity",9.8f);
  e->force_mag=cfg_flt(cfg,"force-mag",10.0f);
  e->tau=cfg_flt(cfg,"dt",0.02f);
  e->continuous=cfg_int(cfg,"continuous",0);
  e->rng=(unsigned int)(seed+(uint64_t)idx*0x9E3779B9u);
  allocate(e); }
static int  ocean_obsdim(Cartpole* e){ (void)e; return 4; }
static int  ocean_numactions(Cartpole* e){ (void)e; return 2; }
static int  ocean_maxsteps(Cartpole* e){ (void)e; return MAX_STEPS; }
static void ocean_teardown(Cartpole* e){ free_allocated(e); }
/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(episode_length) X(x_threshold_termination) X(pole_angle_termination) X(max_steps_termination) \
  X(n) X(score)
#include "../../ffi/ocean_adapter.h"
