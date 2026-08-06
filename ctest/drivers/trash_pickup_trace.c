// Headless trace driver for ocean/trash_pickup. Sets the RNG seed directly, then
// calls c_reset (which places agents/bins/trash with glibc rand_r(&env->rng) — a
// readable, fully-reproducible LCG), applies the given per-step actions (num_agents
// actions per step, in agent order), and prints the observable deterministic fields
// each step (agent x/y/carrying, bin x/y, trash presence, per-agent reward, terminal,
// total episode reward). Stops at the first terminal (whose c_step internally calls
// c_reset, re-randomizing positions with the CONTINUED rng stream — reproduced in the
// Lean model). agent_sight_range is fixed to 1 (it only affects the obs buffer size,
// never the printed trace).
//
// Usage: trash_pickup_trace GRID NA NB NT MAXSTEPS SEED [ACTION ...]
//        actions: 0=UP 1=DOWN 2=LEFT 3=RIGHT
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "trash_pickup.h"

static void print_header(CTrashPickupEnv* env) {
    int na = env->num_agents, nb = env->num_bins, nt = env->num_trash;
    printf("step\tcstep");
    for (int a = 0; a < na; a++) printf("\ta%dx\ta%dy\ta%dc", a, a, a);
    for (int b = 0; b < nb; b++) printf("\tb%dx\tb%dy", b, b);
    for (int t = 0; t < nt; t++) printf("\tt%dp", t);
    for (int a = 0; a < na; a++) printf("\tr%d", a);
    printf("\tterm\ttotrew\n");
}

static void print_row(CTrashPickupEnv* env, int row) {
    int na = env->num_agents, nb = env->num_bins, nt = env->num_trash;
    int bin_start = na, trash_start = na + nb;
    printf("%d\t%d", row, env->current_step);
    for (int a = 0; a < na; a++)
        printf("\t%d\t%d\t%d", env->entities[a].pos_x, env->entities[a].pos_y,
               env->entities[a].carrying ? 1 : 0);
    for (int b = 0; b < nb; b++)
        printf("\t%d\t%d", env->entities[bin_start + b].pos_x, env->entities[bin_start + b].pos_y);
    for (int t = 0; t < nt; t++)
        printf("\t%d", env->entities[trash_start + t].presence ? 1 : 0);
    for (int a = 0; a < na; a++)
        printf("\t%.9g", env->rewards[a]);
    printf("\t%d\t%.9g\n", (int)env->terminals[0], env->total_episode_reward);
}

int main(int argc, char** argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s GRID NA NB NT MAXSTEPS SEED [ACTION...]\n", argv[0]);
        return 2;
    }
    CTrashPickupEnv env;
    memset(&env, 0, sizeof(env));
    env.grid_size = atoi(argv[1]);
    env.num_agents = atoi(argv[2]);
    env.num_bins = atoi(argv[3]);
    env.num_trash = atoi(argv[4]);
    env.max_steps = atoi(argv[5]);
    env.rng = (unsigned int)strtoul(argv[6], NULL, 10);
    env.agent_sight_range = 1;

    allocate(&env);
    c_reset(&env);

    int na = env.num_agents;
    print_header(&env);
    print_row(&env, 0);

    int nact = argc - 7;
    int nsteps = nact / na;
    for (int step = 1; step <= nsteps; step++) {
        for (int a = 0; a < na; a++)
            env.actions[a] = (float)atoi(argv[7 + (step - 1) * na + a]);
        c_step(&env);
        print_row(&env, step);
        if (env.terminals[0]) break;
    }

    free_allocated(&env);
    return 0;
}
