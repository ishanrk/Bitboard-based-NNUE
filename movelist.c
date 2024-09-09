#define MAX_MOVES 256

// Move structure
typedef struct {
    int from_square;
    int to_square;
} Move;

// Move list structure
typedef struct {
    Move moves[MAX_MOVES];
    int count;
} MoveList;

// Function to add a move to the move list
void add_move(MoveList *move_list, int from, int to) {
    move_list->moves[move_list->count].from_square = from;
    move_list->moves[move_list->count].to_square = to;
    move_list->count++;
}

// Function to generate rook moves and populate move list
void generate_rook_moves(Bitboard rooks, Bitboard blockers, MoveList *move_list) {
    Bitboard rook_moves = rook_attacks(rooks, blockers);
    while (rook_moves) {
        int to_square = __builtin_ffsll(rook_moves) - 1;
        clear_bit(&rook_moves, to_square);

        // Find the corresponding from square and add the move
        Bitboard current_rook = rooks;
        while (current_rook) {
            int from_square = __builtin_ffsll(current_rook) - 1;
            clear_bit(&current_rook, from_square);
            add_move(move_list, from_square, to_square);
        }
    }
}

// Function to generate queen moves and populate move list
void generate_queen_moves(Bitboard queens, Bitboard blockers, MoveList *move_list) {
    Bitboard queen_moves = queen_attacks(queens, blockers);
    while (queen_moves) {
        int to_square = __builtin_ffsll(queen_moves) - 1;
        clear_bit(&queen_moves, to_square);

        // Find the corresponding from square and add the move
        Bitboard current_queen = queens;
        while (current_queen) {
            int from_square = __builtin_ffsll(current_queen) - 1;
            clear_bit(&current_queen, from_square);
            add_move(move_list, from_square, to_square);
        }
    }
}

// Function to print moves in the move list (for debugging)
void print_moves(MoveList *move_list) {
    for (int i = 0; i < move_list->count; i++) {
        printf("Move from %d to %d\n", move_list->moves[i].from_square, move_list->moves[i].to_square);
    }
}
