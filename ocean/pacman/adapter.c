/* Adapter: PufferLib Ocean `pacman` -> puffer-lean plugin ABI. Runs the REAL ocean/pacman/pacman.h
 * c_reset/c_step (28x31 maze, 4 ghosts with the original scatter/chase/frightened AI).
 * float obs (291 = 11 player + 9*4 ghost + 240 dots + 4 powerups), 4 discrete actions
 * (0=DOWN 1=UP 2=RIGHT 3=LEFT), float actions/rewards/terminals.
 *
 * Field-for-field vs upstream binding.c my_init (PufferLib/ocean/pacman/binding.c):
 *   num_agents, randomize_starting_position, min_start_timeout, max_start_timeout,
 *   frightened_time, max_mode_changes, scatter_mode_length, chase_mode_length, init(env)
 * are ALL set below (defaults from config/pacman.ini [env]). We call allocate() instead of
 * init(): allocate() = init() + the four caller buffers PufferLib's vecenv would have supplied.
 * NB frightened_time is a DIVISOR in compute_observations (obs[10] = frightened_time_left /
 * (float)frightened_time) — leaving it 0 would make every obs NaN, so it must never default to 0.
 * env->rng is the one field upstream leaves at calloc-0 (so all its copies draw the same player
 * spawn / frightened-ghost turns); we seed it per copy like every other adapter here. */
#include "pacman.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#define ENV_T PacmanEnv
#define OBS_T float
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
static void ocean_setup(PacmanEnv* e,uint64_t seed,int idx,const char* cfg){
  e->num_agents=1;
  e->randomize_starting_position=cfg_int(cfg,"randomize-starting-position",1)?true:false;
  e->min_start_timeout=cfg_int(cfg,"min-start-timeout",0);
  e->max_start_timeout=cfg_int(cfg,"max-start-timeout",49);
  e->frightened_time=cfg_int(cfg,"frightened-time",35);
  e->max_mode_changes=cfg_int(cfg,"max-mode-changes",6);
  e->scatter_mode_length=cfg_int(cfg,"scatter-mode-length",70);
  e->chase_mode_length=cfg_int(cfg,"chase-mode-length",140);
  if(e->frightened_time<1) e->frightened_time=1;       /* never 0: it divides obs[10] */
  if(e->max_start_timeout<e->min_start_timeout) e->max_start_timeout=e->min_start_timeout;
  e->rng=(unsigned int)(seed+(uint64_t)idx*0x9E3779B9u);
  allocate(e);   /* init() (map + spawns + pickup_obs) + observations/actions/rewards/terminals */
}
static int  ocean_obsdim(PacmanEnv* e){ (void)e; return OBSERVATIONS_COUNT; }
static int  ocean_numactions(PacmanEnv* e){ (void)e; return 4; }
static int  ocean_maxsteps(PacmanEnv* e){ (void)e; return MAX_STEPS; }
static void ocean_teardown(PacmanEnv* e){ free_allocated(e); }   /* frees the 4 buffers + c_close */
/* env's own PufferLib `Log` fields, in declaration order — expanded to offsetof by the shared
   adapter so the log channel reports the same statistics upstream's my_log does. */
#define OCEAN_LOG_FIELDS(X) X(episode_return) X(episode_length) X(score) X(perf) X(n)
#include "../../ffi/ocean_adapter.h"
