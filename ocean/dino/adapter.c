/* Adapter: PufferLib Ocean dino -> pufferlean plugin ABI. Runs the REAL ocean/dino/dino.h c_reset/c_step. */
#include "dino.h"
#include <string.h>
#include <stdint.h>
#define ENV_T Dinosaur
#define OBS_T float
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
static void ocean_setup(Dinosaur* e,uint64_t seed,int idx,const char* cfg){
  e->num_agents=1;
  e->width=cfg_int(cfg,"width",800);
  e->height=cfg_int(cfg,"height",400);
  e->speed_init=cfg_int(cfg,"speed-init",6);
  e->speed_max=cfg_int(cfg,"speed-max",15);
  e->spawn_rate_min=cfg_int(cfg,"spawn-rate-min",45);
  e->spawn_rate_max=cfg_int(cfg,"spawn-rate-max",65);
  e->rate_increment_rate=cfg_int(cfg,"rate-increment-rate",600);
  e->rng=(unsigned int)(seed+(uint64_t)idx*0x9E3779B9u);
  c_init(e); /* allocates the four buffers + agent, sets gravity/spawn_rate/max_obstacles */
}
static int  ocean_obsdim(Dinosaur* e){ (void)e; return OBS_SIZE; }
static int  ocean_numactions(Dinosaur* e){ (void)e; return 3; }
static int  ocean_maxsteps(Dinosaur* e){ (void)e; return 100000; }
static void ocean_teardown(Dinosaur* e){ c_close(e); free(e->observations);free(e->actions);free(e->rewards);free(e->terminals); }
/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_length) X(episode_return) X(n)
#include "../../ffi/ocean_adapter.h"
