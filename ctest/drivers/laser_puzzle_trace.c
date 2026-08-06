// Headless trace driver for ocean/laser_puzzle. Sets up the 6x6 board directly
// from a BOARD string (bypassing c_reset's rand_r() level pick + its dependence
// on a levels .bin file), then runs scripted mirror-placement actions and prints
// the deterministic observable fields, stopping at the first terminal.
//
// We call the env's REAL transition `c_step`. To keep the trace deterministic we
// install a NON-NULL dummy client: on a terminal step c_step then only sets
// pending_reset=1 (deferred reset) instead of calling c_reset. That avoids
// c_reset's `rand_r % num_levels` (num_levels==0 here) AND avoids c_reset
// clobbering sinks_found/mirrors_placed, so the terminal row shows the true
// terminal-time state. Nothing in c_step dereferences client (only truthiness),
// and c_render/make_client are never called, so no window opens.
//
// BOARD is a ROWS*COLS (6*6=36) char string, row-major (r outer, c inner):
//   '.'      empty
//   '0'..'7' laser  with that id   (sources; always on the border)
//   'a'..'h' sensor with id 0..7   (sinks;   'a'=0 .. 'h'=7)
// Initial boards carry no mirrors (levels are mirror-stripped); the agent places
// mirrors via actions.
//
// Usage: laser_puzzle_trace TOTAL_SINKS OPTIMAL_MIRRORS MAX_STEPS BOARD ACTION...
//   ACTION in 0..3*INNER_ROWS*INNER_COLS-1 = 0..47: cell_idx=action/3 (0..15),
//   mirror=action%3 (0 none, 1 right, 2 left); cell = interior (cell_idx/4+1, cell_idx%4+1).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "laser_puzzle.h"

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr,
            "usage: %s TOTAL_SINKS OPTIMAL_MIRRORS MAX_STEPS BOARD [ACTION...]\n",
            argv[0]);
        return 2;
    }
    int total_sinks = atoi(argv[1]);
    int optimal_mirrors = atoi(argv[2]);
    int max_steps = atoi(argv[3]);
    const char* board_str = argv[4];
    int rows = INIT_ROWS, cols = INIT_COLS;   // 6 x 6, fixed by the env
    int ncells = rows * cols;
    if ((int)strlen(board_str) != ncells) {
        fprintf(stderr, "BOARD has %zu chars, expected %d (ROWS*COLS)\n",
                strlen(board_str), ncells);
        return 2;
    }

    LaserPuzzle env;
    memset(&env, 0, sizeof(env));

    unsigned char* obs = (unsigned char*)calloc((size_t)ncells, 1);
    Cell* board = (Cell*)calloc((size_t)ncells, sizeof(Cell));
    float act = 0.0f, rew = 0.0f, term = 0.0f;
    env.observations = obs;
    env.actions = &act;
    env.rewards = &rew;
    env.terminals = &term;

    env.ROWS = rows;
    env.COLS = cols;
    env.max_steps = max_steps;
    env.num_agents = 1;
    env.board = board;
    env.total_sinks = total_sinks;
    env.optimal_mirrors = optimal_mirrors;
    env.sinks_found = 0;
    env.mirrors_placed = 0;
    env.moves_made = 0;
    env.episode_length = 0;
    env.episode_return = 0.0f;
    env.pending_reset = 0;
    // memset already zeroed sink_hit_before[].

    // Non-NULL dummy client: makes the terminal branch defer reset (see header).
    Client dummy;
    memset(&dummy, 0, sizeof(dummy));
    env.client = &dummy;

    // Decode BOARD into env->board.
    for (int i = 0; i < ncells; i++) {
        char ch = board_str[i];
        Cell cell = { .type = EMPTY, .mirror = MIRROR_NONE, .id = 0 };
        if (ch >= '0' && ch <= '7') {
            cell.type = LASER;
            cell.id = ch - '0';
        } else if (ch >= 'a' && ch <= 'h') {
            cell.type = SENSOR;
            cell.id = ch - 'a';
        }
        board[i] = cell;
    }

    printf("step\tsinks_found\tmirrors_placed\treward\tterminal\n");
    printf("0\t%d\t%d\t%.9g\t%d\n",
           env.sinks_found, env.mirrors_placed, env.rewards[0], (int)env.terminals[0]);

    for (int i = 5; i < argc; i++) {
        env.actions[0] = (float)atoi(argv[i]);
        c_step(&env);
        printf("%d\t%d\t%d\t%.9g\t%d\n",
               i - 4, env.sinks_found, env.mirrors_placed, env.rewards[0], (int)env.terminals[0]);
        if (env.terminals[0]) break;   // stop at first terminal (reset deferred)
    }

    free(obs);
    free(board);
    return 0;
}
