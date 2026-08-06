// Headless trace driver for ocean/connect4. Sets up an empty board directly
// (player_pieces=0, env_pieces=0, tick=0, rng=0), bypassing any reset RNG, and
// applies the given player column actions. The built-in negamax opponent is
// deterministic apart from its tie-break, which uses rand_r(&env->rng) — a
// readable glibc LCG seeded to 0 here, so the whole trace is reproducible.
// Emits tick / player_pieces / env_pieces / reward / terminal each step and
// stops at the first terminal (whose c_reset would re-randomize nothing but the
// board back to empty).
//
// Usage: connect4_trace ACTION [ACTION ...]   (each action is a column 0..6)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "connect4.h"

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s ACTION [ACTION...]\n", argv[0]); return 2; }
    Connect4 env;
    memset(&env, 0, sizeof(env));
    float obs[42];
    memset(obs, 0, sizeof(obs));
    float act = 0.0f, rew = 0.0f, term = 0.0f;
    env.observations = obs; env.actions = &act; env.rewards = &rew; env.terminals = &term;
    env.player_pieces = 0; env.env_pieces = 0;
    env.tick = 0; env.end_game = 0; env.rng = 0;

    printf("step\ttick\tplayer\tenv\treward\tterminal\n");
    printf("0\t%d\t%llu\t%llu\t%.9g\t%d\n", env.tick,
           (unsigned long long)env.player_pieces, (unsigned long long)env.env_pieces,
           env.rewards[0], (int)env.terminals[0]);
    for (int i = 1; i < argc; i++) {
        env.actions[0] = (float)atoi(argv[i]);
        c_step(&env);
        printf("%d\t%d\t%llu\t%llu\t%.9g\t%d\n", i, env.tick,
               (unsigned long long)env.player_pieces, (unsigned long long)env.env_pieces,
               env.rewards[0], (int)env.terminals[0]);
        if (env.terminals[0]) break;
    }
    return 0;
}
