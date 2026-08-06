// Headless trace driver for ocean/squared. Sets up the grid directly (AGENT at
// center, TARGET at TARGET_IDX), bypassing c_reset's rand() target placement, and
// stops at the first terminal (whose c_reset re-randomizes). Emits the
// deterministic r/c/reward/terminal fields.
//
// Usage: squared_trace SIZE TARGET_IDX ACTION [ACTION ...]
//        actions: 0=NOOP 1=DOWN 2=UP 3=LEFT 4=RIGHT
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "squared.h"

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s SIZE TARGET_IDX [ACTION...]\n", argv[0]); return 2; }
    int size = atoi(argv[1]);
    int target_idx = atoi(argv[2]);
    Squared env;
    memset(&env, 0, sizeof(env));
    unsigned char* obs = (unsigned char*)calloc((size_t)(size * size), 1);
    float act = 0.0f, rew = 0.0f, term = 0.0f;
    env.observations = obs; env.actions = &act; env.rewards = &rew; env.terminals = &term;
    env.size = size; env.tick = 0; env.r = size / 2; env.c = size / 2;
    obs[(size * size) / 2] = 1;   // AGENT
    obs[target_idx] = 2;          // TARGET

    printf("step\tr\tc\treward\tterminal\n");
    printf("0\t%d\t%d\t%.9g\t%d\n", env.r, env.c, env.rewards[0], (int)env.terminals[0]);
    for (int i = 3; i < argc; i++) {
        env.actions[0] = (float)atoi(argv[i]);
        c_step(&env);
        printf("%d\t%d\t%d\t%.9g\t%d\n", i - 2, env.r, env.c, env.rewards[0], (int)env.terminals[0]);
        if (env.terminals[0]) break;
    }
    free(obs);
    return 0;
}
