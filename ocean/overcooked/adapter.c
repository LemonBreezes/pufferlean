/* Adapter: PufferLib Ocean overcooked -> pufferlean plugin ABI. Runs the REAL ocean/overcooked/overcooked.h c_reset/c_step. */
#include "overcooked.h"
#include <string.h>
#include <stdint.h>
#define ENV_T Overcooked
#define OBS_T float
/* ACT_T: actions field is float* -> float (no #define) */
/* TERM_T: terminals field is float* -> float (no #define) */
#define OCEAN_NAGENTS(e) ((e)->num_agents)   /* multi-agent: num_agents>1 */
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
static float cfg_flt(const char* cfg,const char* key,float def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return (float)atof(p+k+1); p=(*q==',')?q+1:q; } return def; }
static void ocean_setup(ENV_T* e,uint64_t seed,int idx,const char* cfg){
  e->num_agents       = cfg_int(cfg,"num-agents",2);
  e->layout_id        = (LayoutType)cfg_int(cfg,"layout",0);
  e->grid_size        = cfg_int(cfg,"grid-size",100);
  e->observation_size = 43;
  e->rewards_config.dish_served_whole_team = cfg_flt(cfg,"reward-dish-served-whole-team",1.0f);
  e->rewards_config.dish_served_agent      = cfg_flt(cfg,"reward-dish-served-agent",0.0f);
  e->rewards_config.pot_started            = cfg_flt(cfg,"reward-pot-started",0.15f);
  e->rewards_config.ingredient_added       = cfg_flt(cfg,"reward-ingredient-added",0.15f);
  e->rewards_config.ingredient_picked      = cfg_flt(cfg,"reward-ingredient-picked",0.05f);
  e->rewards_config.plate_picked           = cfg_flt(cfg,"reward-plate-picked",0.05f);
  e->rewards_config.soup_plated            = cfg_flt(cfg,"reward-soup-plated",0.20f);
  e->rewards_config.wrong_dish_served      = cfg_flt(cfg,"reward-wrong-dish-served",0.0f);
  e->rewards_config.step_penalty           = cfg_flt(cfg,"reward-step-penalty",0.0f);
  e->rng = (unsigned int)(seed + (uint64_t)idx*0x9E3779B9u);
  int A = e->num_agents;
  e->observations = (float*)calloc((size_t)A*43, sizeof(float));
  e->actions      = (float*)calloc((size_t)A, sizeof(float));
  e->rewards      = (float*)calloc((size_t)A, sizeof(float));
  e->terminals    = (float*)calloc((size_t)A, sizeof(float));
  init(e);   /* sizes grid/agents/pots by num_agents+layout and sets derived state */
}
static int  ocean_obsdim(ENV_T* e){ (void)e; return 43; }
static int  ocean_numactions(ENV_T* e){ (void)e; return 6; }
static int  ocean_maxsteps(ENV_T* e){ (void)e; return 512; }
static void ocean_teardown(ENV_T* e){ c_close(e); free(e->observations); free(e->actions); free(e->rewards); free(e->terminals); }
/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(dishes_served) X(correct_dishes) X(wrong_dishes) \
  X(ingredients_picked) X(pots_started) X(items_dropped) X(agent_collisions) X(n)
#include "../../ffi/ocean_adapter.h"
