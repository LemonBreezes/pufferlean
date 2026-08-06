// Headless trace driver for ocean/double_pendulum. Sets the physics state
// directly to a fixed deterministic start (bypassing c_reset's rand_r noise) and
// applies the CLI action list, replicating c_step's body but STOPPING before the
// terminal c_reset (which would re-randomize the state and zero reward/terminal).
// Emits the deterministic physics-state / reward / terminal fields as TSV.
//
// Usage: double_pendulum_trace ACTION [ACTION ...]
//        actions: 0=push left (-force_mag), 1=noop (0), 2=push right (+force_mag)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "double_pendulum.h"

int main(int argc, char** argv) {
    float observations[DP_OBS_SIZE] = {0};
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};

    DoublePendulum env;
    memset(&env, 0, sizeof(env));
    env.observations = observations;
    env.actions = actions;
    env.rewards = rewards;
    env.terminals = terminals;
    env.num_agents = 1;
    env.rng = 1;
    env.cart_mass = 1.0f;
    env.link1_mass = 0.1f;
    env.link2_mass = 0.1f;
    env.link1_length = 0.5f;
    env.link2_length = 0.5f;
    env.gravity = 9.8f;
    env.force_mag = 10.0f;
    env.dt = 0.02f;

    // Deterministic initial state (bypass c_reset RNG): cart centered at rest,
    // both links hanging straight down (theta == pi).
    env.x = 0.0f;
    env.x_dot = 0.0f;
    env.theta1 = (float)M_PI;
    env.theta1_dot = 0.0f;
    env.theta2 = (float)M_PI;
    env.theta2_dot = 0.0f;
    env.tick = 0;
    env.episode_return = 0.0f;
    env.upright_steps = 0;
    env.max_upright_steps = 0;

    printf("step\tx\tx_dot\ttheta1\ttheta1_dot\ttheta2\ttheta2_dot\treward\tterminal\n");
    printf("0\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%d\n",
           env.x, env.x_dot, env.theta1, env.theta1_dot, env.theta2, env.theta2_dot,
           0.0, 0);

    for (int i = 1; i < argc; i++) {
        float a = (float)atoi(argv[i]);
        env.actions[0] = a;
        // --- replicate c_step body, minus the terminal c_reset ---
        if (!isfinite(a)) a = 1.0f;
        int action = (int)a;
        if ((unsigned)action >= DP_ACTIONS) action = 1;
        float force = 0.0f;
        if (action == 0) force = -env.force_mag;
        else if (action == 2) force = env.force_mag;

        integrate_physics(&env, force);
        env.tick += 1;

        bool invalid = !isfinite(env.x) || !isfinite(env.x_dot)
            || !isfinite(env.theta1) || !isfinite(env.theta1_dot)
            || !isfinite(env.theta2) || !isfinite(env.theta2_dot);
        bool x_done = env.x < -DP_X_THRESHOLD || env.x > DP_X_THRESHOLD;
        bool timeout = env.tick >= DP_MAX_STEPS;
        bool done = invalid || x_done || timeout;
        env.rewards[0] = upright_reward(&env, force);
        env.episode_return += env.rewards[0];
        env.terminals[0] = (invalid || x_done) ? 1.0f : 0.0f;

        printf("%d\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%d\n",
               i, env.x, env.x_dot, env.theta1, env.theta1_dot, env.theta2, env.theta2_dot,
               env.rewards[0], (int)env.terminals[0]);
        if (done) break;
    }
    return 0;
}
