// Headless trace driver for ocean/scape.
//
// Scape is deterministic: c_reset hardcodes all entity positions (NO RNG), and
// c_step never reads env->actions (the agent's destination is only ever changed
// by a mouse click inside c_render, which we never call). It also never writes
// env->rewards or env->terminals, so both stay 0 for the whole episode. Hence the
// trajectory is a pure function of the number of steps taken. We take one c_step
// per action arg (the action VALUE is ignored, exactly as c_step ignores it) and
// print all 7 entity positions plus reward/terminal each step.
//
// Usage: scape_trace ACTION [ACTION ...]   (action values are ignored by c_step)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "scape.h"

static void print_row(int step, Scape* env) {
    printf("%d", step);
    int n = env->num_agents + env->num_npcs;
    for (int i = 0; i < n; i++) {
        printf("\t%.9g\t%.9g", env->entities[i].x, env->entities[i].y);
    }
    printf("\t%.9g\t%d\n", env->rewards[0], (int)env->terminals[0]);
}

int main(int argc, char** argv) {
    Scape env;
    memset(&env, 0, sizeof(env));
    env.width = 1080;
    env.height = 720;
    env.num_agents = 1;
    init(&env);   // sets num_agents=1, num_npcs=6, allocates entities

    int num_obs = 1;
    env.observations = calloc((size_t)(env.num_agents * num_obs), sizeof(float));
    env.actions = calloc((size_t)env.num_agents, sizeof(double));
    env.rewards = calloc((size_t)env.num_agents, sizeof(float));
    env.terminals = calloc((size_t)env.num_agents, sizeof(float));

    c_reset(&env);

    printf("step\te0x\te0y\te1x\te1y\te2x\te2y\te3x\te3y"
           "\te4x\te4y\te5x\te5y\te6x\te6y\treward\tterminal\n");
    print_row(0, &env);

    for (int i = 1; i < argc; i++) {
        // action value is ignored by c_step; the arg count sets the step count.
        c_step(&env);
        print_row(i, &env);
        if (env.terminals[0]) break;   // never fires (scape never terminates)
    }

    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    return 0;
}
