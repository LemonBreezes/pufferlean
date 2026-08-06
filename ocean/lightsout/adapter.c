/* Adapter: PufferLib Ocean lightsout -> pufferlean plugin ABI. Runs the REAL ocean/lightsout/lightsout.h c_reset/c_step. */
#include "lightsout.h"
#include <string.h>
#include <stdint.h>
#define ENV_T LightsOut
#define OBS_T unsigned char
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
static void ocean_setup(LightsOut* e,uint64_t seed,int idx,const char* cfg){
  e->num_agents=1;
  e->grid_size=5;
  e->cell_size=1280/e->grid_size; if(1280%e->grid_size!=0) e->cell_size++;
  e->max_steps=cfg_int(cfg,"max-steps",100);
  e->observation_size=e->grid_size*e->grid_size;
  e->ema=0.5f;
  e->score_ema=0.0f;
  e->scramble_prob=0.15f;
  e->grid=NULL;
  e->client=NULL;
  e->rng=(unsigned int)(seed+(uint64_t)idx*0x9E3779B9u);
  int t=e->grid_size*e->grid_size;
  e->observations=(unsigned char*)calloc(t,sizeof(unsigned char));
  e->actions=(float*)calloc(1,sizeof(float));
  e->rewards=(float*)calloc(1,sizeof(float));
  e->terminals=(float*)calloc(1,sizeof(float));
}
static int  ocean_obsdim(LightsOut* e){ return e->grid_size*e->grid_size; }
static int  ocean_numactions(LightsOut* e){ return e->grid_size*e->grid_size; }
static int  ocean_maxsteps(LightsOut* e){ return e->max_steps; }
static void ocean_teardown(LightsOut* e){ free(e->grid); e->grid=NULL; free(e->observations); free(e->actions); free(e->rewards); free(e->terminals); }
/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(scramble_p) X(n)
#include "../../ffi/ocean_adapter.h"
