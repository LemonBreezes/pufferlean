// Headless trace driver for ocean/cartpole.
//
// Sets up the physics state directly (bypassing c_reset's rand_r initial state)
// and runs c_step with the demo() config (discrete mode). On an ending step
// (reward == 0, i.e. terminated or truncated) c_step internally calls c_reset,
// which re-randomises x/x_dot/theta/theta_dot via rand_r — so the live state read
// back on the ending row is stochastic. Exactly like the snake driver, we report
// the PRE-step physics on that row (the last deterministic observable) and stop.
// Every non-ending row reports the post-step physics read back from env.
//
// The four initial-state args are integers in 1024ths (n -> n/1024.0f, exact in
// f32, so no parse-time rounding). Actions: 0 = push left, 1 = push right.
//
// Usage: cartpole_trace X0 XD0 TH0 THD0 ACTION [ACTION ...]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cartpole.h"

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s X0 XD0 TH0 THD0 [ACTION...]\n", argv[0]);
        return 2;
    }
    Cartpole env;
    memset(&env, 0, sizeof(env));
    float obs[4] = {0}, act = 0.0f, rew = 0.0f, term = 0.0f;
    env.observations = obs; env.actions = &act; env.rewards = &rew; env.terminals = &term;
    env.num_agents = 1;
    // demo() config (discrete mode).
    env.cart_mass = 1.0f;
    env.pole_mass = 0.1f;
    env.pole_length = 0.5f;
    env.gravity = 9.8f;
    env.force_mag = 10.0f;
    env.tau = 0.02f;
    env.continuous = 0;
    env.episode_return = 0.0f;
    env.rng = 0;
    // Initial physics, bypassing c_reset's rand_r.
    env.x = (float)atoi(argv[1]) / 1024.0f;
    env.x_dot = (float)atoi(argv[2]) / 1024.0f;
    env.theta = (float)atoi(argv[3]) / 1024.0f;
    env.theta_dot = (float)atoi(argv[4]) / 1024.0f;
    env.tick = 0;

    printf("step\tx\tx_dot\ttheta\ttheta_dot\treward\tterminal\n");
    printf("0\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%d\n",
           env.x, env.x_dot, env.theta, env.theta_dot, env.rewards[0], (int)env.terminals[0]);

    int step = 1;
    for (int i = 5; i < argc; i++) {
        // Pre-step physics (used for the ending row, where c_step's c_reset has
        // already re-randomised the live state).
        float pre_x = env.x, pre_xd = env.x_dot, pre_th = env.theta, pre_thd = env.theta_dot;

        env.actions[0] = (float)atoi(argv[i]);
        c_step(&env);

        if (env.rewards[0] == 0.0f) {   // ending step (terminated or truncated)
            printf("%d\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%d\n",
                   step, pre_x, pre_xd, pre_th, pre_thd, env.rewards[0], (int)env.terminals[0]);
            break;
        }
        printf("%d\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%d\n",
               step, env.x, env.x_dot, env.theta, env.theta_dot, env.rewards[0], (int)env.terminals[0]);
        step++;
    }
    return 0;
}
