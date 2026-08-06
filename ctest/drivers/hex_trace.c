// Headless trace driver for ocean/hex. Sets up a deterministic initial state
// directly (empty board, union-find reset, rng = 0), bypassing nothing that is
// random at reset (c_reset uses no RNG here — the opponent's rand_r() is the
// only randomness, and env->rng starts at 0), then applies the action list and
// prints the observable deterministic fields per step, stopping at the first
// terminal.
//
// hex's c_reset sets env->terminals[0] = 0, so the terminal flag does NOT survive
// a terminal step (it is cleared by the c_reset that c_step calls). The reward
// DOES survive (c_reset never touches rewards[0]) and is nonzero exactly on a
// terminal step, so we detect termination via rewards[0] != 0.
//
// The opponent (env->random_opponent == false) is compute_env_move: a Fisher-Yates
// shuffle of the 6 hex directions driven by rand_r(&env->rng), picking the first
// empty neighbour of the player's last move (falling back to compute_legal_move,
// a rand_r rejection sample over the whole board). With random_opponent == true it
// is compute_legal_move directly. Both consume the same reproducible rand_r stream.
//
// env_move column: the opponent's chosen cell on a NON-terminal step (read back by
// diffing the board). On any terminal row it is printed as -1 — c_reset has already
// wiped the board (env-win case) or no env move happened (invalid / player-win).
//
// Usage: hex_trace OPP ACTION [ACTION ...]
//        OPP: 0 = compute_env_move (neighbour heuristic), 1 = random_opponent
//        ACTION: a cell index 0..120 (out-of-range or occupied => invalid => lose)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "hex.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s OPP [ACTION...]\n", argv[0]);
        return 2;
    }
    int opp = atoi(argv[1]);

    Hex env;
    memset(&env, 0, sizeof(env));
    float* obs = (float*)calloc((size_t)(2 * TOTAL_CELLS), sizeof(float));
    float act = 0.0f, rew = 0.0f, term = 0.0f;
    env.observations = obs;
    env.actions = &act;
    env.rewards = &rew;
    env.terminals = &term;
    env.num_agents = 1;
    env.random_opponent = (opp != 0);
    env.rng = 0;                 // explicit deterministic seed

    c_reset(&env);               // empty board, uf_init, tick=0, terminals=0; rng untouched

    printf("step\taction\tenv_move\treward\tterminal\n");
    printf("0\t-1\t-1\t%.9g\t0\n", 0.0);

    int8_t prev[TOTAL_CELLS];
    for (int i = 2; i < argc; i++) {
        int action = atoi(argv[i]);
        memcpy(prev, env.board, sizeof(prev));
        env.actions[0] = (float)action;
        env.rewards[0] = 0.0f;   // c_step only writes rewards on a terminal branch
        c_step(&env);
        int terminal = (env.rewards[0] != 0.0f) ? 1 : 0;
        int env_move = -1;
        if (!terminal) {
            for (int k = 0; k < TOTAL_CELLS; k++) {
                if (prev[k] == 0 && env.board[k] == ENV_COLOR) { env_move = k; break; }
            }
        }
        printf("%d\t%d\t%d\t%.9g\t%d\n", i - 1, action, env_move, env.rewards[0], terminal);
        if (terminal) break;
    }

    free(obs);
    return 0;
}
