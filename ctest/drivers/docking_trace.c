// Headless trace driver for ocean/docking (single-agent continuous ship-docking
// physics). Bypasses c_reset's rand_r placement by setting the ship/dock state
// directly (the `squared` pattern): the RNG is used ONLY in c_reset, and c_step
// contains no RNG, so a trace from a fixed initial state is fully deterministic.
// We stop at the first terminal (whose reset_pending would re-randomize).
//
// The config values passed here are all in c_init's "pass-through" regime (they
// are positive / non-zero and drag in [0,1]), so c_init leaves them unchanged and
// the Lean model can consume them directly.
//
// Usage:
//   docking_trace WIDTH HEIGHT MAX_TICKS MAX_SPEED TURN_RATE ACCEL DRAG \
//       DOCK_RADIUS DOCK_SPEED_THR DOCK_HEAD_THR STEP_PENALTY PROGRESS_SCALE \
//       SHIP_X SHIP_Y HEADING SPEED DOCK_X DOCK_Y  ACTION [ACTION ...]
//   actions: 0=NOOP 1=TURN_LEFT 2=TURN_RIGHT 3=THRUST 4=BRAKE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "docking.h"

int main(int argc, char** argv) {
    if (argc < 20) {
        fprintf(stderr,
            "usage: %s WIDTH HEIGHT MAX_TICKS MAX_SPEED TURN_RATE ACCEL DRAG "
            "DOCK_RADIUS DOCK_SPEED_THR DOCK_HEAD_THR STEP_PENALTY PROGRESS_SCALE "
            "SHIP_X SHIP_Y HEADING SPEED DOCK_X DOCK_Y ACTION...\n", argv[0]);
        return 2;
    }
    Docking env;
    memset(&env, 0, sizeof(env));
    env.width                  = atoi(argv[1]);
    env.height                 = atoi(argv[2]);
    env.max_ticks              = atoi(argv[3]);
    env.max_speed              = (float)atof(argv[4]);
    env.turn_rate              = (float)atof(argv[5]);
    env.accel                  = (float)atof(argv[6]);
    env.drag                   = (float)atof(argv[7]);
    env.dock_radius            = (float)atof(argv[8]);
    env.dock_speed_threshold   = (float)atof(argv[9]);
    env.dock_heading_threshold = (float)atof(argv[10]);
    env.step_penalty           = (float)atof(argv[11]);
    env.progress_reward_scale  = (float)atof(argv[12]);

    float obs[DOCKING_OBS_SIZE];
    float act = 0.0f, rew = 0.0f, term = 0.0f;
    env.observations = obs;
    env.actions = &act;
    env.rewards = &rew;
    env.terminals = &term;

    c_init(&env);   // pass-through regime: leaves the tunables above unchanged

    // Set the initial state directly (bypass c_reset's rand_r placement).
    env.ship_x       = (float)atof(argv[13]);
    env.ship_y       = (float)atof(argv[14]);
    env.ship_heading = (float)atof(argv[15]);
    env.ship_speed   = (float)atof(argv[16]);
    env.dock_x       = (float)atof(argv[17]);
    env.dock_y       = (float)atof(argv[18]);
    env.dock_heading = 0.0f;               // c_reset always sets dock_heading = 0
    env.tick = 0;
    env.episode_return = 0.0f;
    env.reset_pending = 0;
    env.feedback_timer = 0;
    env.prev_distance = docking_distance(&env);

    printf("step\tship_x\tship_y\theading\tspeed\treward\tep_return\tterminal\tresult\n");
    printf("0\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%d\t%d\n",
        env.ship_x, env.ship_y, env.ship_heading, env.ship_speed,
        env.rewards[0], env.episode_return, (int)env.terminals[0], env.last_result);

    for (int i = 19; i < argc; i++) {
        env.actions[0] = (float)atoi(argv[i]);
        c_step(&env);
        printf("%d\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%d\t%d\n",
            i - 18, env.ship_x, env.ship_y, env.ship_heading, env.ship_speed,
            env.rewards[0], env.episode_return, (int)env.terminals[0], env.last_result);
        if (env.terminals[0]) break;
    }
    return 0;
}
