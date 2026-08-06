/* Adapter: PufferLib Ocean snake -> pufferlean plugin ABI. Runs the REAL ocean/snake/snake.h c_reset/c_step. */
#include "snake.h"
#include <string.h>
#include <stdint.h>
#define ENV_T CSnake
#define OBS_T unsigned char       /* observations is char* (ByteTensor / OBS_TYPE CHAR) */
#define ACT_T double              /* actions field is double* (ACT_TYPE DOUBLE) */
/* terminals is float* -> TERM_T defaults to float */
#define OCEAN_NAGENTS(e) ((e)->num_agents)   /* multi-agent: num_agents snakes per instance */
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
static float cfg_flt(const char* cfg,const char* key,float def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return (float)atof(p+k+1); p=(*q==',')?q+1:q; } return def; }
static void ocean_setup(CSnake* e,uint64_t seed,int idx,const char* cfg){
  (void)seed; (void)idx;
  /* config fields my_init sets, hyphenated keys, defaults from config/snake.ini [env] */
  e->width=cfg_int(cfg,"width",640);
  e->height=cfg_int(cfg,"height",360);
  e->num_agents=cfg_int(cfg,"num-agents",256);
  e->vision=cfg_int(cfg,"vision",5);
  e->leave_corpse_on_death=(unsigned char)cfg_int(cfg,"leave-corpse-on-death",1);
  e->food=cfg_int(cfg,"num-food",4096);
  e->reward_food=cfg_flt(cfg,"reward-food",0.1f);
  e->reward_corpse=cfg_flt(cfg,"reward-corpse",0.1f);
  e->reward_death=cfg_flt(cfg,"reward-death",-1.0f);
  e->max_snake_length=cfg_int(cfg,"max-snake-length",1024);
  e->cell_size=cfg_int(cfg,"cell-size",2);
  allocate_csnake(e);   /* mallocs obs/actions/rewards/terminals (sized by num_agents) + init_csnake */
}
static int  ocean_obsdim(CSnake* e){ return (2*e->vision+1)*(2*e->vision+1); }  /* per agent */
static int  ocean_numactions(CSnake* e){ (void)e; return 4; }
static int  ocean_maxsteps(CSnake* e){ (void)e; return 1000; }
static void ocean_teardown(CSnake* e){ free_csnake(e); }
/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(n)
#include "../../ffi/ocean_adapter.h"
