/* Adapter: PufferLib Ocean whisker_racer -> puffer-lean plugin ABI. Runs the REAL ocean/whisker_racer/whisker_racer.h c_reset/c_step. */
#include "whisker_racer.h"
#include <string.h>
#include <stdint.h>
#define ENV_T WhiskerRacer
#define OBS_T float
#define ACT_T double   /* actions field is double* (ACT_TYPE DOUBLE) */
static int cfg_int(const char* cfg,const char* key,int def){ if(!cfg)return def; size_t k=strlen(key); const char* p=cfg;
  while(*p){ const char* q=p; while(*q&&*q!=',')q++; if((size_t)(q-p)>k&&strncmp(p,key,k)==0&&p[k]=='=')return atoi(p+k+1); p=(*q==',')?q+1:q; } return def; }
static void ocean_setup(WhiskerRacer* e,uint64_t seed,int idx,const char* cfg){
  e->num_agents=1;
  /* integer config fields (hyphenated keys), defaults from config/whisker_racer.ini [env] */
  e->frameskip=cfg_int(cfg,"frameskip",4);
  e->width=cfg_int(cfg,"width",1080);
  e->height=cfg_int(cfg,"height",720);
  e->track_width=cfg_int(cfg,"track-width",75);
  e->num_radial_sectors=cfg_int(cfg,"num-radial-sectors",180);
  e->num_points=cfg_int(cfg,"num-points",16);
  e->bezier_resolution=cfg_int(cfg,"bezier-resolution",4);
  e->mode7=cfg_int(cfg,"mode7",0);
  e->render_many=cfg_int(cfg,"render-many",0);
  e->method=cfg_int(cfg,"method",2);
  e->continuous=cfg_int(cfg,"continuous",0);
  e->render=0;
  /* float config fields: cfg_int can't parse floats, set from .ini defaults */
  e->turn_pi_frac=40.0f;
  e->maxv=5.0f;
  e->w_ang=0.777f;
  e->max_whisker_length=100.0f;
  e->reward_yellow=0.2f;
  e->reward_green=-0.001f;
  e->gamma=0.9f;
  e->corner_thresh=0.5f;
  e->ftmp1=0.5f; e->ftmp2=3.0f; e->ftmp3=0.3f; e->ftmp4=0.0f;
  e->i=idx;
  e->rng=(unsigned int)(seed+(uint64_t)idx*0x9E3779B9u);
  allocate(e);  /* calls init(e) (builds track using the fields above) + mallocs obs/actions/rewards/terminals */
}
static int  ocean_obsdim(WhiskerRacer* e){ (void)e; return 3; }
static int  ocean_numactions(WhiskerRacer* e){ (void)e; return 3; }
static int  ocean_maxsteps(WhiskerRacer* e){ (void)e; return 100000; }
static void ocean_teardown(WhiskerRacer* e){ free_allocated(e); }
/* Log channel (ffi/puffer_env.h): this env's OWN Log fields, located via offsetof in <env>.h. */
#define OCEAN_LOG_FIELDS(X) X(perf) X(score) X(episode_return) X(episode_length) X(n)
#include "../../ffi/ocean_adapter.h"
