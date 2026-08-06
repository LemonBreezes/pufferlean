// Headless trace driver for ocean/robocode (multi-robot tank-combat physics).
//
// Configuration: num_agents = 2, num_bots = 0 (so bots.h logic is never
// exercised: bot_step runs only for indices [num_agents, num_agents+num_bots)).
// Two player-controlled robots on a width x height field: each step both robots
// move / turn body+gun+radar / fire, bullets fly and collide, robots ram, radar
// scans award reward_spot, bullet hits deal damage (+/- reward_damage).
//
// RNG (rand_r on env->rng) is used ONLY by c_reset for random placement; c_step
// contains no RNG. So (as in `squared`/`docking`) we bypass c_reset and set the
// robot state directly, then drive c_step from that fixed state. The episode-end
// path (end_episode -> c_reset) would re-randomise placement, so we STOP at the
// first terminal. Terminals here come only from the timeout branch
// (tick > max_ticks); we pre-empt it in the driver so c_reset never runs, and
// print the pre-timeout state with terminal=1 (the timeout branch does no robot
// physics before resetting). The test cases keep energy >= 0, so the death
// branch (which also resets) never fires.
//
// Usage:
//   robocode_trace WIDTH HEIGHT MAX_TICKS REWARD_DAMAGE REWARD_SPOT \
//       R0X R0Y R0H R0GH R0RH R0E R0GHEAT \
//       R1X R1Y R1H R1GH R1RH R1E R1GHEAT \
//       A0 A1 A2 A3 A4  B0 B1 B2 B3 B4  [ ... 10 action indices per step ]
//   Robot init per robot: x y heading gun_heading radar_heading energy gun_heat
//       (v=0, radar_heading_prev=radar_heading, bullet_idx=0 are fixed)
//   Actions per step: robot0 (move[0..3] turn[0..8] gun[0..10] radar[0..10]
//       fire[0..5]) then robot1 (same 5).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "robocode.h"

#define NFIX 19   // number of fixed (non-action) args after argv[0]

static void print_row(int step, Robocode* env, int term) {
    Robot* a = &env->robots[0];
    Robot* b = &env->robots[1];
    printf("%d\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%d\t%d"
           "\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%d\t%d"
           "\t%.9g\t%.9g\t%d\n",
        step,
        a->x, a->y, a->v, a->heading, a->gun_heading, a->radar_heading,
        a->radar_heading_prev, a->gun_heat, a->energy, a->bullet_idx,
        b->x, b->y, b->v, b->heading, b->gun_heading, b->radar_heading,
        b->radar_heading_prev, b->gun_heat, b->energy, b->bullet_idx,
        env->rewards[0], env->rewards[1], term);
}

int main(int argc, char** argv) {
    if (argc < 1 + NFIX) {
        fprintf(stderr, "usage: %s WIDTH HEIGHT MAX_TICKS REWARD_DAMAGE REWARD_SPOT "
            "R0X R0Y R0H R0GH R0RH R0E R0GHEAT R1X R1Y R1H R1GH R1RH R1E R1GHEAT "
            "A0..A4 B0..B4 [...]\n", argv[0]);
        return 2;
    }
    Robocode env;
    memset(&env, 0, sizeof(env));
    env.num_agents    = 2;
    env.num_bots      = 0;
    env.width         = atoi(argv[1]);
    env.height        = atoi(argv[2]);
    env.max_ticks     = atoi(argv[3]);
    env.reward_damage = (float)atof(argv[4]);
    env.reward_spot   = (float)atof(argv[5]);
    env.bot_policy    = 0;
    env.tag           = 0;
    env.rng           = 12345u;   // unused by c_step

    allocate_env(&env);   // allocates robots/bullets/logs + wires per-slot pointers

    // Set robot state directly (bypass c_reset's rand_r placement).
    for (int r = 0; r < 2; r++) {
        int o = 6 + r * 7;
        Robot* R = &env.robots[r];
        R->x                 = (float)atof(argv[o + 0]);
        R->y                 = (float)atof(argv[o + 1]);
        R->heading           = (float)atof(argv[o + 2]);
        R->gun_heading       = (float)atof(argv[o + 3]);
        R->radar_heading     = (float)atof(argv[o + 4]);
        R->radar_heading_prev = R->radar_heading;
        R->energy            = atoi(argv[o + 5]);
        R->gun_heat          = (float)atof(argv[o + 6]);
        R->v                 = 0.0f;
        R->bullet_idx        = 0;
    }
    env.tick = 0;
    env.rewards[0] = 0.0f; env.rewards[1] = 0.0f;
    env.terminals[0] = 0.0f; env.terminals[1] = 0.0f;

    printf("step\tr0x\tr0y\tr0v\tr0h\tr0gh\tr0rh\tr0rhp\tr0gheat\tr0e\tr0bi"
           "\tr1x\tr1y\tr1v\tr1h\tr1gh\tr1rh\tr1rhp\tr1gheat\tr1e\tr1bi"
           "\trew0\trew1\tterm\n");
    print_row(0, &env, 0);

    int step = 0;
    for (int ai = 1 + NFIX; ai + 9 < argc; ai += 10) {
        if (env.tick + 1 > env.max_ticks) {
            // timeout terminal: pre-empt c_step so c_reset never scrambles state.
            print_row(step + 1, &env, 1);
            return 0;
        }
        for (int k = 0; k < 5; k++) env.action_ptr[0][k] = (float)atoi(argv[ai + k]);
        for (int k = 0; k < 5; k++) env.action_ptr[1][k] = (float)atoi(argv[ai + 5 + k]);
        c_step(&env);
        step += 1;
        int term = (int)env.terminals[0];
        print_row(step, &env, term);
        if (term) return 0;   // (death terminal: state already scrambled) safety stop
    }
    return 0;
}
