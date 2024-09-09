#include <stdio.h>

// Assuming Move and MoveList are already defined as earlier
// Assuming bitboards, blockers, and other board state are already set up

// Function to apply a move (for now, just update the bitboards)
void make_move(Bitboard *bitboards, Move move, int piece_type) {
    clear_bit(&bitboards[piece_type], move.from_square);
    set_bit(&bitboards[piece_type], move.to_square);
}

// Function to undo a move (restore the bitboards)
void unmake_move(Bitboard *bitboards, Move move, int piece_type) {
    clear_bit(&bitboards[piece_type], move.to_square);
    set_bit(&bitboards[piece_type], move.from_square);
}

// Function to generate all legal moves for the current position
void generate_all_moves(MoveList *move_list) {
    move_list->count = 0; // Reset move list

    // Assuming blockers, rooks, queens, pawns, etc. are already defined
    Bitboard blockers = 0ULL; // Define this as needed

    // Example: Generate rook moves
    generate_rook_moves(bitboards[WHITE_ROOKS], blockers, move_list);
    // Add similar lines for generating other moves (queens, knights, bishops, etc.)
    generate_queen_moves(bitboards[WHITE_QUEENS], blockers, move_list);
    // Add knight, bishop, pawn moves similarly
}

// Perft test at a given depth
uint64_t perft(Bitboard *bitboards, int depth, int piece) {
    if (depth == 0) return 1; // If depth is 0, just return 1 (leaf node)

    MoveList move_list;
    generate_all_moves(&move_list);

    uint64_t nodes = 0;
    for (int i = 0; i < move_list.count; i++) {
        Move move = move_list.moves[i];

        // Make the move
        make_move(bitboards, move, piece);  // , use correct piece type

        // Recur to the next depth
        nodes += perft(bitboards, depth - 1);

        // Unmake the move
        unmake_move(bitboards, move, piece); // , use correct piece type
    }

    return nodes;
}

// Perft test driver function
void perft_test(int depth) {
    uint64_t nodes = perft(bitboards, depth);
    printf("Perft(%d) = %llu\n", depth, nodes);
}
