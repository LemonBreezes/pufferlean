// Headless trace driver for ocean/snake (single-agent slice).
//
// The real env NEVER sets terminals[i] (the `terminals[i] = 1` line in
// step_snake is commented out): on death it silently respawns the snake with
// rand() (spawn_snake), and on eating FOOD it places new food with rand()
// (spawn_food). Both are the ONLY rand() sites reachable during a step. So we
// build a deterministic initial scene directly (bypassing c_reset's rand), run
// c_step, and STOP the trace at (and including) the first row whose reward is
// reward_death (a death → spawn_snake next) or reward_food (a food → spawn_food
// next). Eating a CORPSE (reward_corpse) uses no rand, so the trace continues
// through it. On a death step spawn_snake has already scrambled the head, so we
// report the PRE-step head/length (the last valid observable) for that row;
// otherwise we report the post-step head/length read back from env->snake.
//
// Observable deterministic fields per step: head row r, head col c, reward,
// snake length, and our synthetic terminal flag.
//
// CLI (all coords are (row,col), row in [0,height), col in [0,width)):
//   snake_trace W H V MAXLEN  NB r0 c0 r1 c1 ...   (NB body cells, head first)
//                             NF fr0 fc0 ...        (NF food cells)
//                             NC cr0 cc0 ...        (NC corpse cells)
//                             ACTION [ACTION ...]   (0=up 1=down 2=left 3=right)
// A wall border of thickness V lines all four grid edges (as in c_reset).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "snake.h"

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s W H V MAXLEN NB r c... NF r c... NC r c... ACTION...\n", argv[0]);
        return 2;
    }
    int W = atoi(argv[1]);
    int H = atoi(argv[2]);
    int V = atoi(argv[3]);
    int MAXLEN = atoi(argv[4]);

    CSnake env;
    memset(&env, 0, sizeof(env));
    env.num_agents = 1;
    env.width = W;
    env.height = H;
    env.vision = V;
    env.max_snake_length = MAXLEN;
    env.food = 0;
    env.leave_corpse_on_death = 1;
    env.reward_food = 1.0f;
    env.reward_corpse = 0.5f;
    env.reward_death = -1.0f;
    env.window = 2 * V + 1;
    env.obs_size = env.window * env.window;
    env.tick = 0;

    // Buffers (single agent).
    env.grid = (char*)calloc((size_t)(W * H), sizeof(char));
    env.snake = (int*)calloc((size_t)(2 * MAXLEN), sizeof(int));
    env.snake_lengths = (int*)calloc(1, sizeof(int));
    env.snake_ptr = (int*)calloc(1, sizeof(int));
    env.snake_lifetimes = (int*)calloc(1, sizeof(int));
    env.snake_colors = (int*)calloc(1, sizeof(int));
    env.snake_logs = (Log*)calloc(1, sizeof(Log));
    env.observations = (char*)calloc((size_t)env.obs_size, sizeof(char));
    double act = 0.0; float rew = 0.0f, term = 0.0f;
    env.actions = &act; env.rewards = &rew; env.terminals = &term;
    env.snake_colors[0] = 7;   // as init_csnake sets snake 0

    // Wall border of thickness V (matches c_reset's wall placement).
    for (int r = 0; r < V; r++)
        for (int c = 0; c < W; c++) env.grid[r * W + c] = WALL;
    for (int r = H - V; r < H; r++)
        for (int c = 0; c < W; c++) env.grid[r * W + c] = WALL;
    for (int r = 0; r < H; r++) {
        for (int c = 0; c < V; c++) env.grid[r * W + c] = WALL;
        for (int c = W - V; c < W; c++) env.grid[r * W + c] = WALL;
    }

    int k = 5;
    // Snake body, head first.
    int NB = atoi(argv[k++]);
    int* br = (int*)calloc((size_t)NB, sizeof(int));
    int* bc = (int*)calloc((size_t)NB, sizeof(int));
    for (int j = 0; j < NB; j++) { br[j] = atoi(argv[k++]); bc[j] = atoi(argv[k++]); }
    // Lay body into the circular buffer: buffer slot j = body[(NB-1)-j], so
    // slot 0 = tail ... slot NB-1 = head, with snake_ptr = NB-1.
    for (int s = 0; s < 2 * MAXLEN; s++) env.snake[s] = -1;   // sentinel for unused slots
    for (int j = 0; j < NB; j++) {
        int bidx = (NB - 1) - j;
        env.snake[2 * j]     = br[bidx];
        env.snake[2 * j + 1] = bc[bidx];
        env.grid[br[bidx] * W + bc[bidx]] = (char)env.snake_colors[0];
    }
    env.snake_ptr[0] = NB - 1;
    env.snake_lengths[0] = NB;
    env.snake_lifetimes[0] = 0;

    // Food cells.
    int NF = atoi(argv[k++]);
    for (int j = 0; j < NF; j++) {
        int fr = atoi(argv[k++]); int fc = atoi(argv[k++]);
        env.grid[fr * W + fc] = FOOD;
    }
    // Corpse cells.
    int NC = atoi(argv[k++]);
    for (int j = 0; j < NC; j++) {
        int cr = atoi(argv[k++]); int cc = atoi(argv[k++]);
        env.grid[cr * W + cc] = CORPSE;
    }

    printf("step\tr\tc\treward\tlength\tterminal\n");
    // Row 0: the supplied initial head/length, reward 0, terminal 0.
    {
        int hp = env.snake_ptr[0];
        printf("0\t%d\t%d\t%.9g\t%d\t%d\n",
               env.snake[2 * hp], env.snake[2 * hp + 1], 0.0, env.snake_lengths[0], 0);
    }

    int step = 1;
    for (int i = k; i < argc; i++) {
        // Pre-step head/length (used for the death row, where c_step's
        // spawn_snake has already scrambled the live head).
        int hp = env.snake_ptr[0];
        int pre_r = env.snake[2 * hp];
        int pre_c = env.snake[2 * hp + 1];
        int pre_len = env.snake_lengths[0];

        env.actions[0] = (double)atoi(argv[i]);
        c_step(&env);
        float r = env.rewards[0];

        if (r == env.reward_death) {
            printf("%d\t%d\t%d\t%.9g\t%d\t%d\n", step, pre_r, pre_c, r, pre_len, 1);
            step++;
            break;
        }
        int hp2 = env.snake_ptr[0];
        int post_r = env.snake[2 * hp2];
        int post_c = env.snake[2 * hp2 + 1];
        int post_len = env.snake_lengths[0];
        int t = (r == env.reward_food) ? 1 : 0;
        printf("%d\t%d\t%d\t%.9g\t%d\t%d\n", step, post_r, post_c, r, post_len, t);
        step++;
        if (t) break;
    }

    free(br); free(bc);
    free(env.grid); free(env.snake); free(env.snake_lengths); free(env.snake_ptr);
    free(env.snake_lifetimes); free(env.snake_colors); free(env.snake_logs);
    free(env.observations);
    return 0;
}
