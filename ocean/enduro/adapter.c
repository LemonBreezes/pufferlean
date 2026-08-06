/* Adapter: PufferLib Ocean enduro -> pufferlean plugin ABI. Runs the REAL ocean/enduro/enduro.h c_reset/c_step. */
#include "enduro.h"
#include <string.h>
#include <stdint.h>
#define ENV_T Enduro
#define OBS_T float
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
static void ocean_setup(Enduro* e,uint64_t seed,int idx,const char* cfg){
  e->num_agents=1;
  e->width=cfg_int(cfg,"width",152);
  e->height=cfg_int(cfg,"height",210);
  e->car_width=cfg_int(cfg,"car-width",16);
  e->car_height=cfg_int(cfg,"car-height",11);
  e->max_enemies=cfg_int(cfg,"max-enemies",10);
  e->continuous=cfg_int(cfg,"continuous",0);
  e->seed=(int)(seed+(uint64_t)idx*0x9E3779B9u);
  e->rng=(unsigned int)(seed+(uint64_t)idx*0x9E3779B9u);
  e->rng_state=(unsigned int)(seed+(uint64_t)idx*0x9E3779B9u);
  e->obs_size=OBSERVATIONS_MAX_SIZE;
  allocate(e);
}
static int  ocean_obsdim(Enduro* e){ (void)e; return OBSERVATIONS_MAX_SIZE; }
static int  ocean_numactions(Enduro* e){ (void)e; return 9; }
static int  ocean_maxsteps(Enduro* e){ (void)e; return 100000; }
static void ocean_teardown(Enduro* e){ free_allocated(e); }
/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(reward) X(step_rew_car_passed_no_crash) \
  X(crashed_penalty) X(passed_cars) X(passed_by_enemy) X(cars_to_pass) X(days_completed) X(days_failed) \
  X(collisions_player_vs_car) X(collisions_player_vs_road) X(n)
#include "../../ffi/ocean_adapter.h"
