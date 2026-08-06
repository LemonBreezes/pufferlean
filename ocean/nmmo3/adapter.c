/* Adapter: PufferLib Ocean nmmo3 -> pufferlean plugin ABI. Runs the REAL ocean/nmmo3/nmmo3.h c_reset/c_step. */
#include "nmmo3.h"
#include <string.h>
#include <stdint.h>
#define ENV_T MMO
#define OBS_T unsigned char
/* ACT_T: actions field is float* -> float (no #define) */
/* TERM_T: terminals field is float* -> float (no #define) */
#define OCEAN_NAGENTS(e) ((e)->num_agents)   /* multi-agent: num_agents>1 */
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
static float cfg_flt(const char* cfg,const char* key,float def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return (float)atof(p+k+1); p=(*q==',')?q+1:q; } return def; }
static void ocean_setup(ENV_T* e,uint64_t seed,int idx,const char* cfg){
  e->width               = cfg_int(cfg,"width",512);
  e->height              = cfg_int(cfg,"height",512);
  e->num_agents          = cfg_int(cfg,"num-agents",1024);
  e->num_enemies         = cfg_int(cfg,"num-enemies",2048);
  e->num_resources       = cfg_int(cfg,"num-resources",2048);
  e->num_weapons         = cfg_int(cfg,"num-weapons",1024);
  e->num_gems            = cfg_int(cfg,"num-gems",512);
  e->tiers               = cfg_int(cfg,"tiers",5);
  e->levels              = cfg_int(cfg,"levels",40);
  e->teleportitis_prob   = cfg_flt(cfg,"teleportitis-prob",0.001f);
  e->enemy_respawn_ticks = cfg_int(cfg,"enemy-respawn-ticks",2);
  e->item_respawn_ticks  = cfg_int(cfg,"item-respawn-ticks",100);
  e->x_window            = cfg_int(cfg,"x-window",7);
  e->y_window            = cfg_int(cfg,"y-window",5);
  e->reward_combat_level = cfg_flt(cfg,"reward-combat-level",1.0f);
  e->reward_prof_level   = cfg_flt(cfg,"reward-prof-level",1.0f);
  e->reward_item_level   = cfg_flt(cfg,"reward-item-level",1.0f);
  e->reward_market       = cfg_flt(cfg,"reward-market",0.0f);
  e->reward_death        = cfg_flt(cfg,"reward-death",-1.0f);
  e->rng = (unsigned int)(seed + (uint64_t)idx*0x9E3779B9u);
  allocate_mmo(e);   /* sizes the four buffers by num_agents; calls init(e) */
}
static int  ocean_obsdim(ENV_T* e){ (void)e; return 11*15*10+47+10; }   /* 1707 per agent */
static int  ocean_numactions(ENV_T* e){ (void)e; return 26; }
static int  ocean_maxsteps(ENV_T* e){ (void)e; return 100000; }
static void ocean_teardown(ENV_T* e){ free_allocated_mmo(e); }
/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(n) X(return_comb_lvl) X(return_prof_lvl) \
  X(return_item_atk_lvl) X(return_item_def_lvl) X(return_market_buy) X(return_market_sell) X(return_death) \
  X(min_comb_prof) X(purchases) X(sales) X(equip_attack) X(equip_defense) X(r) X(c)
#include "../../ffi/ocean_adapter.h"
