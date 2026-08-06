/* Adapter: PufferLib Ocean pong -> pufferlean plugin ABI. Runs the REAL ocean/pong/pong.h c_reset/c_step. */
#include "pong.h"
#include <string.h>
#include <stdint.h>
#define ENV_T Pong
#define OBS_T float
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
static void ocean_setup(Pong* e,uint64_t seed,int idx,const char* cfg){
  e->num_agents=1;
  e->width=cfg_int(cfg,"width",500);
  e->height=cfg_int(cfg,"height",640);
  e->paddle_width=cfg_int(cfg,"paddle-width",20);
  e->paddle_height=cfg_int(cfg,"paddle-height",70);
  e->ball_width=cfg_int(cfg,"ball-width",32);
  e->ball_height=cfg_int(cfg,"ball-height",32);
  e->paddle_speed=cfg_int(cfg,"paddle-speed",8);
  e->ball_initial_speed_x=cfg_int(cfg,"ball-initial-speed-x",10);
  e->ball_initial_speed_y=cfg_int(cfg,"ball-initial-speed-y",1);
  e->ball_max_speed_y=cfg_int(cfg,"ball-max-speed-y",13);
  e->ball_speed_y_increment=cfg_int(cfg,"ball-speed-y-increment",3);
  e->max_score=cfg_int(cfg,"max-score",21);
  e->frameskip=cfg_int(cfg,"frameskip",8);
  e->continuous=cfg_int(cfg,"continuous",0);
  e->rng=(unsigned int)(seed+(uint64_t)idx*0x9E3779B9u);
  allocate(e); }
static int  ocean_obsdim(Pong* e){ (void)e; return 8; }
static int  ocean_numactions(Pong* e){ (void)e; return 3; }
static int  ocean_maxsteps(Pong* e){ (void)e; return 100000; }
static void ocean_teardown(Pong* e){ free_allocated(e); }
/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(n)
#include "../../ffi/ocean_adapter.h"
