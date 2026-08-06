// Headless trace driver for ocean/maze. Builds a FIXED maze level directly
// (bypassing my_vec_init's rand_r() maze generation and c_reset's rand_r()
// level pick), then drives scripted actions through c_step and emits only the
// deterministic observable fields (x, y, tick, reward, terminal). On the first
// terminal, c_step internally calls c_reset which re-randomizes the level, so
// we print that row and break before the RNG can matter.
//
// The maze grid lives on a MAX_SIZE(47)-stride board; only s->maze[y*47+x] is
// read by move_to (WALL=1 blocks, GOAL=4 rewards+ends). We pass the level as:
//   WIDTH HEIGHT SPAWN_X SPAWN_Y MAZE  ACTION [ACTION ...]
// where MAZE is HEIGHT rows of WIDTH chars concatenated (row-major), each char a
// tile digit 0=EMPTY 1=WALL 2=AGENT 4=GOAL. Actions: 0=PASS 1=EAST 2=NORTH
// 3=WEST 4=SOUTH.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "maze.h"

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s WIDTH HEIGHT SPAWN_X SPAWN_Y MAZE [ACTION...]\n", argv[0]);
        return 2;
    }
    int width  = atoi(argv[1]);
    int height = atoi(argv[2]);
    int spawn_x = atoi(argv[3]);
    int spawn_y = atoi(argv[4]);
    const char* maze_str = argv[5];

    Grid env;
    memset(&env, 0, sizeof(env));

    unsigned char* obs = (unsigned char*)calloc((size_t)(WINDOW * WINDOW), 1);
    float act = 0.0f, rew = 0.0f, term = 0.0f;
    env.observations = obs;
    env.actions = &act;
    env.rewards = &rew;
    env.terminals = &term;
    env.num_agents = 1;
    env.tick = 0;

    // Build the fixed initial State directly on the MAX_SIZE-stride board.
    State* s = &env.state;
    s->width = width;
    s->height = height;
    s->spawn_x = spawn_x;
    s->spawn_y = spawn_y;
    s->x = spawn_x;
    s->y = spawn_y;
    s->direction = 0;
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int tile = maze_str[r * width + c] - '0';
            s->maze[maze_offset(r, c)] = (unsigned char)tile;
        }
    }

    // A valid single-level pool so the terminal-triggered c_reset does not
    // dereference NULL / divide by zero. We break right after that row anyway.
    State* levels = (State*)calloc(1, sizeof(State));
    levels[0] = env.state;
    env.levels = levels;
    env.num_levels = 1;
    env.rng = 0;

    printf("step\tx\ty\ttick\treward\tterminal\n");
    printf("0\t%d\t%d\t%d\t%.9g\t%d\n",
           env.state.x, env.state.y, env.tick, env.rewards[0], (int)env.terminals[0]);
    for (int i = 6; i < argc; i++) {
        env.actions[0] = (float)atoi(argv[i]);
        c_step(&env);
        printf("%d\t%d\t%d\t%d\t%.9g\t%d\n",
               i - 5, env.state.x, env.state.y, env.tick, env.rewards[0], (int)env.terminals[0]);
        if (env.terminals[0]) break;   // c_reset re-randomized the level; stop here
    }
    free(obs);
    free(levels);
    return 0;
}
