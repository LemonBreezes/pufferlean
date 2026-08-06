// Headless trace driver for ocean/tripletriad. Allocates the env, seeds the
// PRNG (env->rng) to SEED, and calls c_reset so the whole card deal comes from
// rand_r(&env->rng) starting at SEED — a readable glibc LCG, fully reproducible.
// The built-in opponent (a random legal card + random empty cell each placement)
// draws from the same rand_r stream, so the entire trace is deterministic given
// SEED and the action list.
//
// Emits the 3x3 board_states (b0..b8, row-major, 0/1/-1), both scores, the
// player's selected card (card_selected[0]), reward and terminal each step, and
// stops at the first episode end: a natural terminal (terminals[0]==1) or the
// MAX_EPISODE_LENGTH=30 timeout (which c_step handles by resetting; we stop then
// too, before the re-randomized board is observed as a fresh game).
//
// Usage: tripletriad_trace SEED ACTION [ACTION ...]
//   actions: 0-4 = SELECT card 1-5, 5-13 = PLACE at board cell 1-9
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tripletriad.h"

static void print_row(CTripleTriad* env, int step) {
    printf("%d", step);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            printf("\t%d", env->board_states[i][j]);
    printf("\t%d\t%d\t%d\t%.9g\t%d\n",
           env->score[0], env->score[1], env->card_selected[0],
           env->rewards[0], (int)env->terminals[0]);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s SEED [ACTION...]\n", argv[0]); return 2; }
    unsigned int seed = (unsigned int)strtoul(argv[1], NULL, 10);

    CTripleTriad env;
    memset(&env, 0, sizeof(env));
    env.width = 114;
    env.height = 1;
    env.num_cards = 10;
    allocate_ctripletriad(&env);   // allocates arrays + obs/act/rew/term

    env.rng = seed;
    c_reset(&env);                 // deal cards from rand_r starting at SEED

    printf("step\tb0\tb1\tb2\tb3\tb4\tb5\tb6\tb7\tb8\tscore0\tscore1\tsel0\treward\tterminal\n");
    print_row(&env, 0);

    for (int i = 2; i < argc; i++) {
        env.actions[0] = (float)atoi(argv[i]);
        c_step(&env);
        int step = i - 1;
        print_row(&env, step);
        if ((int)env.terminals[0] == 1 || step >= 30) break;
    }

    free_allocated_ctripletriad(&env);
    return 0;
}
