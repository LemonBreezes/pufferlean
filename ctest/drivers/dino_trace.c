// Headless trace driver for ocean/dino (the Chrome-dinosaur runner).
//
// Determinism: dino's spawn logic draws from TWO PRNGs. `rand_r(&env->rng)` (spawn
// count/type) is glibc's reentrant LCG, reproduced exactly on the Lean side and
// seeded here. stdlib `rand()` (the next spawn_rate) is NOT reproducible, so we
// NEUTRALISE it by setting spawn_rate_max = spawn_rate_min + 1, making
// `rand() % (max-min) == rand() % 1 == 0` for any rand() value. We also keep
// rate_increment_rate large so `speed` never increments (making the spawn_rate
// division exactly 1.0). Under those conventions the trajectory is a deterministic
// function of the rng seed, matching Puffer/Env/Dino/Model.lean.
//
// Rewards are zeroed before each c_step (mirroring vecenv.h's per-step memset), so
// the printed reward is the single-step value (+0.01 survive, +0.01-1.0 collide).
//
// Usage: dino_trace WIDTH SPEED_INIT SPAWN_RATE_MIN RATE_INC RNG_SEED ACTION [ACTION ...]
//        actions: 0=NOOP 1=JUMP 2=CROUCH
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dino.h"

static void print_row(Dinosaur* env, int step) {
    double sumx = 0.0;
    for (int o = 0; o < env->num_obstacles; o++) sumx += env->obstacles[o].x;
    double ox0 = env->num_obstacles > 0 ? env->obstacles[0].x : -999.0;
    printf("%d\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%d\t%d\t%.9g\t%.9g\t%.9g\t%d\n",
        step, env->agent->y, env->agent->y_velocity, env->agent->height,
        env->agent->width, env->agent->x_offset, env->speed, env->num_obstacles,
        ox0, sumx, env->rewards[0], (int)env->terminals[0]);
}

int main(int argc, char** argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s WIDTH SPEED_INIT SPAWN_RATE_MIN RATE_INC RNG_SEED ACTION...\n", argv[0]);
        return 2;
    }
    Dinosaur env;
    memset(&env, 0, sizeof(env));
    env.width               = atoi(argv[1]);
    env.height              = 400;
    env.speed_init          = atoi(argv[2]);
    env.speed_max           = 100;
    env.spawn_rate_min      = atoi(argv[3]);
    env.spawn_rate_max      = atoi(argv[3]) + 1;   // neutralise stdlib rand(): rand()%1==0
    env.rate_increment_rate = atoi(argv[4]);
    unsigned int seed       = (unsigned int)strtoul(argv[5], NULL, 10);

    c_init(&env);        // allocates obs/act/rew/term + agent, sets gravity/shape
    env.rng = seed;      // seed the reproducible rand_r stream
    c_reset(&env);       // resets speed/agent/obstacles (does not touch env.rng)
    srand(0);            // rand()'s value is neutralised, but keep C deterministic

    printf("step\ty\tvy\th\tw\txoff\tspeed\tnobs\tox0\tsumx\treward\tterminal\n");
    print_row(&env, 0);
    for (int i = 6; i < argc; i++) {
        env.rewards[0] = 0.0f;              // vecenv zeros rewards/terminals before each step
        env.terminals[0] = 0.0f;
        env.actions[0] = (float)atoi(argv[i]);
        c_step(&env);
        print_row(&env, i - 5);
        if (env.terminals[0]) break;
    }
    free(env.agent);
    free(env.obstacles);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    return 0;
}
