// Headless trace driver for ocean/chess (CHESS_MODE_RANDOM = single learner vs a
// built-in random opponent). Builds the Chess struct directly, bypassing binding.c:
//   - starting position is the FEN passed on the CLI (random_fen off, no curriculum),
//     so c_reset uses NO rng (fen_curriculum == NULL, random_fen == 0);
//   - the ONLY randomness in the episode is the opponent's move choice
//     `rand_r(&env->rng) % legal_count`, a readable glibc LCG seeded explicitly, so
//     the whole trace is deterministic and reproducible in Lean.
//
// The learner plays a two-phase action (pick a from-square, then a to-square/promo)
// exactly as the trained policy would; the opponent replies with one full random move
// per c_step. We feed one action per c_step (opponent steps ignore their action).
//
// c_reset is called by c_step at game end (non-human modes), which re-parses the
// starting FEN and flips learner_color — so on a terminal row the position fields
// reflect the RESET (fresh) state; the reward/terminal fields (which c_reset does not
// touch) carry the outcome. We stop at the first terminal.
//
// LEARNER_COLOR is the color the learner plays THIS episode (post-reset). Because
// c_reset flips learner_color, we seed env.learner_color = 1 - LEARNER_COLOR so the
// flip lands on the requested color.
//
// Hard-coded config (must match Puffer/Env/Chess/Model.lean): reward_draw = 0,
// reward_invalid_piece = -0.5, reward_invalid_move = -0.25, reward_repetition = 0,
// enable_50_move_rule = 1, enable_threefold_repetition = 0 (so the zobrist key /
// repetition machinery is never observable and need not be reproduced).
//
// Usage: chess_trace SEED LEARNER_COLOR MAX_MOVES FEN(6 fields) ACTION [ACTION ...]
//   SEED           : unsigned int, the opponent rng seed
//   LEARNER_COLOR  : 0 = learner plays White, 1 = learner plays Black (post-reset)
//   MAX_MOVES      : timeout in plies (chess_moves >= MAX_MOVES => draw/timeout)
//   FEN            : a full FEN as its SIX space-separated fields, i.e. SIX argv
//                    tokens (piece-placement, side, castling, ep, halfmove,
//                    fullmove). The differential-test harness word-splits each
//                    case into individual argv tokens, so a FEN's internal spaces
//                    make it span six tokens; we rejoin them here. (pos_set only
//                    reads the first four, but all six are accepted.)
//   ACTION         : integer per c_step. 0..63 square, 64..95 promotion cell,
//                    96 pass, <0 or >=96 invalid. Opponent steps ignore their action.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "chess.h"

static const char PCHARS[16] = ".PNBRQK\?\?pnbrqk\?";

// 64-char (+7 '/') board string, rank 7 (top) down to rank 0, files a..h.
static void board_str(const Position* pos, char* out) {
    char* p = out;
    for (int rank = 7; rank >= 0; rank--) {
        for (int file = 0; file < 8; file++) {
            Piece pc = pos->board[(rank << 3) + file];
            *p++ = PCHARS[pc & 15];
        }
        if (rank > 0) *p++ = '/';
    }
    *p = '\0';
}

int main(int argc, char** argv) {
    if (argc < 10) {
        fprintf(stderr, "usage: %s SEED LEARNER_COLOR MAX_MOVES FEN(6 fields) [ACTION...]\n", argv[0]);
        return 2;
    }
    unsigned int seed = (unsigned int)strtoul(argv[1], NULL, 10);
    int learner_color = atoi(argv[2]);
    int max_moves = atoi(argv[3]);
    // The FEN is six space-separated fields, delivered as argv[4..9] because the
    // harness word-splits the case; rejoin them into one FEN string.
    char fenbuf[128];
    snprintf(fenbuf, sizeof(fenbuf), "%s %s %s %s %s %s",
             argv[4], argv[5], argv[6], argv[7], argv[8], argv[9]);
    const char* fen = fenbuf;

    init_bitboards();

    Chess env;
    memset(&env, 0, sizeof(env));

    uint8_t obs[OBS_SIZE];
    float act = 0.0f, rew = 0.0f, term = 0.0f;
    env.obs_ptr[0] = obs;
    env.action_ptr[0] = &act;
    env.reward_ptr[0] = &rew;
    env.terminal_ptr[0] = &term;
    env.action_mask = NULL;

    env.mode = CHESS_MODE_RANDOM;
    env.num_agents = 1;
    env.rng = seed;
    env.learner_color = 1 - learner_color;   // c_reset flips it to learner_color
    env.slot_for_color[CHESS_WHITE] = 0;
    env.slot_for_color[CHESS_BLACK] = 1;

    env.max_moves = max_moves;
    env.reward_draw = 0.0f;
    env.reward_invalid_piece = -0.5f;
    env.reward_invalid_move = -0.25f;
    env.reward_repetition = 0.0f;
    env.enable_50_move_rule = 1;
    env.enable_threefold_repetition = 0;
    env.random_fen = 0;
    env.fen_curric_pct = 0.0f;
    env.fen_curriculum = NULL;
    env.num_fens = 0;
    env.legal_dirty = 1;
    env.client = NULL;
    env.human_color = -1;
    env.log_pgn = 0;
    env.log_pgn_choice_made = 1;
    strncpy(env.starting_fen, fen, sizeof(env.starting_fen) - 1);

    c_reset(&env);

    char bs[80];
    board_str(&env.pos, bs);
    printf("step\tside\tcastle\tep\trule50\tcmoves\tboard\treward\tterminal\n");
    printf("0\t%d\t%d\t%d\t%d\t%d\t%s\t%.9g\t%d\n",
           (int)env.pos.sideToMove, (int)env.pos.castlingRights, (int)env.pos.epSquare,
           (int)env.pos.rule50, env.chess_moves, bs, 0.0, 0);

    for (int i = 10; i < argc; i++) {
        act = (float)atoi(argv[i]);
        rew = 0.0f;
        term = 0.0f;
        c_step(&env);
        board_str(&env.pos, bs);
        int terminal = (term != 0.0f) ? 1 : 0;
        printf("%d\t%d\t%d\t%d\t%d\t%d\t%s\t%.9g\t%d\n",
               i - 9, (int)env.pos.sideToMove, (int)env.pos.castlingRights,
               (int)env.pos.epSquare, (int)env.pos.rule50, env.chess_moves, bs,
               rew, terminal);
        if (terminal) break;
    }
    return 0;
}
