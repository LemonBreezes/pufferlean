// Headless trace driver for ocean/slimevolley. Runs the real c_reset/c_step in
// two-agent mode (num_agents=2, so no scripted bot). The RIGHT agent is idle; the
// LEFT agent is driven by a per-step 3-bit action code (bit0=forward, bit1=backward,
// bit2=jump). All physics is float (f32); the only RNG is randf(env) =
// (float)rand_r(&env->rng)/(float)RAND_MAX, which we seed from argv[1] and let run
// exactly (glibc's reentrant LCG is reproducible). It fires on the initial serve
// (c_reset), on each re-serve after a point (new_match), and on the terminal
// c_reset. We print the raw physics/score state each step and stop at the first
// terminal (whose c_reset re-serves & re-seeds, printed on the terminal row).
//
// Usage: slimevolley_trace SEED ACTION [ACTION ...]
//        ACTION = 0..7  (bit0=forward bit1=backward bit2=jump for the LEFT agent)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "slimevolley.h"

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s SEED [ACTION...]\n", argv[0]); return 2; }
    SlimeVolley env;
    memset(&env, 0, sizeof(env));
    env.num_agents = 2;
    init(&env);   // allocates ground/fence/fence_stub/agents[2]/ball

    env.observations = (float*)calloc(24, sizeof(float));  // 12 per agent, unread here
    env.actions      = (float*)calloc(6, sizeof(float));
    env.rewards      = (float*)calloc(2, sizeof(float));
    env.terminals    = (float*)calloc(2, sizeof(float));

    env.rng = (unsigned int)strtoul(argv[1], NULL, 10);
    c_reset(&env);  // initial serve: draws ball vx/vy from rng, sets up agents

    printf("step\ttick\tdelay\tball_x\tball_y\tball_vx\tball_vy\tlx\tly\tlvx\tlvy\trx\try\trvx\trvy\tllives\trlives\treward\tterminal\n");
    printf("0\t%d\t%d\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%d\t%d\t%.9g\t%d\n",
        env.tick, env.delay_frames,
        env.ball->x, env.ball->y, env.ball->vx, env.ball->vy,
        env.agents[0].x, env.agents[0].y, env.agents[0].vx, env.agents[0].vy,
        env.agents[1].x, env.agents[1].y, env.agents[1].vx, env.agents[1].vy,
        env.agents[0].lives, env.agents[1].lives,
        env.rewards[0], (int)env.terminals[0]);

    for (int i = 2; i < argc; i++) {
        int code = atoi(argv[i]);
        env.actions[0] = (code & 1) ? 1.0f : 0.0f;   // LEFT forward
        env.actions[1] = (code & 2) ? 1.0f : 0.0f;   // LEFT backward
        env.actions[2] = (code & 4) ? 1.0f : 0.0f;   // LEFT jump
        env.actions[3] = 0.0f; env.actions[4] = 0.0f; env.actions[5] = 0.0f;  // RIGHT idle
        c_step(&env);
        printf("%d\t%d\t%d\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%d\t%d\t%.9g\t%d\n",
            i - 1, env.tick, env.delay_frames,
            env.ball->x, env.ball->y, env.ball->vx, env.ball->vy,
            env.agents[0].x, env.agents[0].y, env.agents[0].vx, env.agents[0].vy,
            env.agents[1].x, env.agents[1].y, env.agents[1].vx, env.agents[1].vy,
            env.agents[0].lives, env.agents[1].lives,
            env.rewards[0], (int)env.terminals[0]);
        if (env.terminals[0]) break;
    }

    free(env.observations); free(env.actions); free(env.rewards); free(env.terminals);
    c_close(&env);
    return 0;
}
