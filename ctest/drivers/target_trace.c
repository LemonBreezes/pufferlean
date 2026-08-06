// Headless trace driver for ocean/target (puffers eating stars, multi-agent
// continuous physics). Bypasses c_reset's rand_r placement by setting the agent
// and goal state directly (the `squared` pattern), sets env->rng to a supplied
// seed so the goal-collection rand_r draws are reproducible, applies the action
// list one step (2*num_agents action floats) at a time, and prints the
// deterministic per-agent / per-goal state each step.
//
// Usage:
//   target_trace WIDTH HEIGHT NA NG RNG \
//       <NA * (X Y HEADING SPEED)>  <NG * (GX GY)>  <ACTIONS...>
// where ACTIONS is consumed 2*NA at a time (a[2*i]=heading adj 0-8 center 4,
// a[2*i+1]=speed adj 0-4 center 2); the number of steps is
// (remaining args)/(2*NA).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "target.h"

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s WIDTH HEIGHT NA NG RNG <agents...> <goals...> <actions...>\n", argv[0]);
        return 2;
    }
    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    int na = atoi(argv[3]);
    int ng = atoi(argv[4]);
    unsigned int rng = (unsigned int)strtoul(argv[5], NULL, 10);

    Target env;
    memset(&env, 0, sizeof(env));
    env.width = width;
    env.height = height;
    env.num_agents = na;
    env.num_goals = ng;
    env.rng = rng;
    init(&env);  // callocs env.agents (na) and env.goals (ng)

    float* obs = (float*)calloc((size_t)(na * (ng*2 + na*2 + 4)), sizeof(float));
    float* act = (float*)calloc((size_t)(2*na), sizeof(float));
    float* rew = (float*)calloc((size_t)na, sizeof(float));
    float* term = (float*)calloc((size_t)na, sizeof(float));
    env.observations = obs; env.actions = act; env.rewards = rew; env.terminals = term;

    int idx = 6;
    for (int i = 0; i < na; i++) {
        env.agents[i].x = (float)atof(argv[idx++]);
        env.agents[i].y = (float)atof(argv[idx++]);
        env.agents[i].heading = (float)atof(argv[idx++]);
        env.agents[i].speed = (float)atof(argv[idx++]);
        env.agents[i].ticks_since_reward = 0;
    }
    for (int i = 0; i < ng; i++) {
        env.goals[i].x = (float)atof(argv[idx++]);
        env.goals[i].y = (float)atof(argv[idx++]);
    }

    // Header
    printf("step");
    for (int i = 0; i < na; i++)
        printf("\ta%d_x\ta%d_y\ta%d_h\ta%d_spd\ta%d_tsr\ta%d_rew", i, i, i, i, i, i);
    for (int g = 0; g < ng; g++)
        printf("\tg%d_x\tg%d_y", g, g);
    printf("\n");

    // Emit one row: `s` is the step index.
    #define EMIT_ROW(s) do { \
        printf("%d", (s)); \
        for (int i = 0; i < na; i++) \
            printf("\t%.9g\t%.9g\t%.9g\t%.9g\t%d\t%.9g", env.agents[i].x, env.agents[i].y, \
                   env.agents[i].heading, env.agents[i].speed, env.agents[i].ticks_since_reward, \
                   env.rewards[i]); \
        for (int g = 0; g < ng; g++) \
            printf("\t%.9g\t%.9g", env.goals[g].x, env.goals[g].y); \
        printf("\n"); \
    } while (0)

    EMIT_ROW(0);

    int per_step = 2 * na;
    int step = 1;
    while (idx + per_step <= argc) {
        for (int k = 0; k < per_step; k++)
            env.actions[k] = (float)atof(argv[idx++]);
        c_step(&env);
        EMIT_ROW(step);
        step++;
    }

    free(obs); free(act); free(rew); free(term);
    free(env.agents); free(env.goals);
    return 0;
}
