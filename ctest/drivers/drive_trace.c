// Headless trace driver for ocean/drive (Waymo-style kinematic-bicycle driving).
//
// The full env loads a binary Waymo map, mean-centers it, builds a spatial grid,
// selects active/static agents, and runs collision detection against road edges
// and other vehicles. NONE of that involves RNG in c_step/c_reset/init — the only
// rand_r calls are render-only (car colors / FPV camera), so the *simulation* is
// fully deterministic. The portable, map-independent core is `move_dynamics` (the
// per-agent kinematic-bicycle Euler step) plus the goal-distance reward, respawn,
// and the 91-step episode reset.
//
// We isolate that core the `squared` way: build a Drive by hand with a SINGLE
// active-agent vehicle, no other actors, and no roads. `map_corners` is left all
// zero so getGridIndex() returns -1 for every query — this makes checkNeighbors()
// find no road edges and the vehicle-vehicle loop find no partners, so
// collision_check() always reports NO_COLLISION without ever touching the (unbuilt)
// grid. The trajectory is then determined entirely by the initial pose, the vehicle
// length, the goal position, and the (accel_idx, steer_idx) action stream.
//
// Usage:
//   drive_trace X Y HEADING VX VY LENGTH WIDTH GOAL_X GOAL_Y \
//       REW_VEH REW_OFF REW_GOAL_POST REW_VEH_POST  A0 S0 A1 S1 A2 S2 ...
//   each c_step consumes one (accel_idx in [0,6], steer_idx in [0,12]) pair.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "drive.h"

int main(int argc, char** argv) {
    if (argc < 14 || ((argc - 14) % 2) != 0) {
        fprintf(stderr,
            "usage: %s X Y HEADING VX VY LENGTH WIDTH GOAL_X GOAL_Y "
            "REW_VEH REW_OFF REW_GOAL_POST REW_VEH_POST  ACCEL STEER [ACCEL STEER ...]\n",
            argv[0]);
        return 2;
    }

    float x0       = (float)atof(argv[1]);
    float y0       = (float)atof(argv[2]);
    float heading0 = (float)atof(argv[3]);
    float vx0      = (float)atof(argv[4]);
    float vy0      = (float)atof(argv[5]);
    float length0  = (float)atof(argv[6]);
    float width0   = (float)atof(argv[7]);
    float goal_x   = (float)atof(argv[8]);
    float goal_y   = (float)atof(argv[9]);

    Drive env;
    memset(&env, 0, sizeof(env));
    env.dynamics_model = CLASSIC;
    env.reward_vehicle_collision              = (float)atof(argv[10]);
    env.reward_offroad_collision              = (float)atof(argv[11]);
    env.reward_goal_post_respawn              = (float)atof(argv[12]);
    env.reward_vehicle_collision_post_respawn = (float)atof(argv[13]);
    env.timestep = 0;

    // Single active-agent vehicle.
    int n = TRAJECTORY_LENGTH;
    env.num_entities = 1;
    env.num_objects  = 1;
    env.num_roads    = 0;
    Entity* entities = (Entity*)calloc(1, sizeof(Entity));
    env.entities = entities;
    Entity* e = &entities[0];
    e->type = VEHICLE;
    e->array_size = n;
    e->traj_x = (float*)calloc(n, sizeof(float));
    e->traj_y = (float*)calloc(n, sizeof(float));
    e->traj_z = (float*)calloc(n, sizeof(float));
    e->traj_vx = (float*)calloc(n, sizeof(float));
    e->traj_vy = (float*)calloc(n, sizeof(float));
    e->traj_vz = (float*)calloc(n, sizeof(float));
    e->traj_heading = (float*)calloc(n, sizeof(float));
    e->traj_valid = (int*)calloc(n, sizeof(int));
    // traj[0] is the reset/respawn target (set_start_position / respawn_agent).
    e->traj_x[0] = x0; e->traj_y[0] = y0; e->traj_z[0] = 0.0f;
    e->traj_vx[0] = vx0; e->traj_vy[0] = vy0; e->traj_vz[0] = 0.0f;
    e->traj_heading[0] = heading0;
    e->traj_valid[0] = 1;
    e->width  = width0;
    e->length = length0;
    e->height = 0.0f;
    e->goal_position_x = goal_x;
    e->goal_position_y = goal_y;
    e->goal_position_z = 0.0f;
    e->mark_as_expert = 0;
    e->x = x0; e->y = y0; e->z = 0.0f;
    e->vx = vx0; e->vy = vy0; e->vz = 0.0f;
    e->heading = heading0;
    e->heading_x = cosf(heading0);
    e->heading_y = sinf(heading0);
    e->valid = 1;
    e->reached_goal = 0;
    e->respawn_timestep = -1;
    e->collided_before_goal = 0;
    e->reached_goal_this_episode = 0;
    e->active_agent = 1;
    e->collision_state = NO_COLLISION;

    env.active_agent_count = 1;
    env.active_agent_indices = (int*)calloc(1, sizeof(int));
    env.active_agent_indices[0] = 0;
    env.num_actors = 1;
    env.static_agent_count = 0;
    env.static_agent_indices = NULL;
    env.expert_static_agent_count = 0;
    env.expert_static_agent_indices = NULL;
    env.max_agents = MAX_AGENTS;

    // Grid disabled: all-zero corners => getGridIndex() returns -1 everywhere,
    // so collision_check() and compute_observations() never touch the grid.
    env.map_corners = (float*)calloc(4, sizeof(float));
    env.grid_cols = 0;
    env.grid_rows = 0;
    env.grid_cells = NULL;
    env.neighbor_cache_indices = NULL;
    env.neighbor_cache_entities = NULL;
    env.vision_range = VISION_RANGE;

    env.observations = (float*)calloc(OBS_SIZE, sizeof(float));
    env.actions      = (float*)calloc(2, sizeof(float));
    env.rewards      = (float*)calloc(1, sizeof(float));
    env.terminals    = (float*)calloc(1, sizeof(float));
    env.logs         = (Log*)calloc(1, sizeof(Log));

    printf("step\tx\ty\theading\tvx\tvy\treward\trespawn\treached\n");
    printf("0\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%d\t%d\n",
        e->x, e->y, e->heading, e->vx, e->vy, env.rewards[0],
        e->respawn_timestep, e->reached_goal_this_episode);

    int nsteps = (argc - 14) / 2;
    for (int i = 0; i < nsteps; i++) {
        int accel_idx = atoi(argv[14 + i * 2]);
        int steer_idx = atoi(argv[14 + i * 2 + 1]);
        env.actions[0] = (float)accel_idx;
        env.actions[1] = (float)steer_idx;
        c_step(&env);
        printf("%d\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%d\t%d\n",
            i + 1, e->x, e->y, e->heading, e->vx, e->vy, env.rewards[0],
            e->respawn_timestep, e->reached_goal_this_episode);
    }
    return 0;
}
