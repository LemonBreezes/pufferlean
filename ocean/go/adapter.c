/* Adapter: PufferLib Ocean go -> puffer-lean plugin ABI. Runs the REAL ocean/go/go.h c_reset/c_step. */
#include "go.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#define ENV_T CGo
#define OBS_T float
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
static float cfg_flt(const char* cfg,const char* key,float def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return (float)atof(p+k+1); p=(*q==',')?q+1:q; } return def; }
static void ocean_setup(CGo* e,uint64_t seed,int idx,const char* cfg){
  e->num_agents=1;
  e->rng=(unsigned int)(seed+(uint64_t)idx*0x9E3779B9u);
  e->selfplay=cfg_int(cfg,"selfplay",0);
  e->width=cfg_int(cfg,"width",950);
  e->height=cfg_int(cfg,"height",750);
  e->grid_size=cfg_int(cfg,"grid-size",9);
  e->board_width=cfg_int(cfg,"board-width",600);
  e->board_height=cfg_int(cfg,"board-height",600);
  e->grid_square_size=cfg_int(cfg,"grid-square-size",64);
  e->komi=cfg_flt(cfg,"komi",7.5f);
  e->reward_move_pass=cfg_flt(cfg,"reward-move-pass",-0.518441f);
  e->reward_move_invalid=cfg_flt(cfg,"reward-move-invalid",-0.0864746f);
  e->reward_move_valid=cfg_flt(cfg,"reward-move-valid",0.0f);
  e->reward_player_capture=cfg_flt(cfg,"reward-player-capture",0.553628f);
  e->reward_opponent_capture=cfg_flt(cfg,"reward-opponent-capture",-0.102283f);
  e->side=(rand_r(&e->rng)%2)+1;
  allocate(e);
}
static int  ocean_obsdim(CGo* e){ return e->grid_size*e->grid_size*4+2; }
static int  ocean_numactions(CGo* e){ return e->grid_size*e->grid_size+1; }
static int  ocean_maxsteps(CGo* e){ return 3*e->grid_size*e->grid_size+2; }
static void ocean_teardown(CGo* e){ free_allocated(e); }
/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(n) X(illegal_move_count) X(legal_move_count) \
  X(pass_move_count) X(white_wins) X(black_wins)
#include "../../ffi/ocean_adapter.h"
