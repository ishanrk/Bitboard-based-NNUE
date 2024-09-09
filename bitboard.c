#include <stdint.h>
#include <stdio.h>

// Define a bitboard type (64-bit unsigned integer)
typedef uint64_t Bitboard;

// Constants to represent the pieces on the board
// (assuming we represent each piece with a separate bitboard)
enum {
    WHITE_PAWNS, WHITE_KNIGHTS, WHITE_BISHOPS, WHITE_ROOKS, WHITE_QUEENS, WHITE_KINGS,
    BLACK_PAWNS, BLACK_KNIGHTS, BLACK_BISHOPS, BLACK_ROOKS, BLACK_QUEENS, BLACK_KINGS,
    ALL_PIECES
};

// Define bitboards for each piece type
Bitboard bitboards[12] = {0};

// Utility masks
Bitboard fileA = 0x0101010101010101ULL;
Bitboard fileH = 0x8080808080808080ULL;
Bitboard rank1 = 0x00000000000000FFULL;
Bitboard rank8 = 0xFF00000000000000ULL;

// Function to set a bit at a specific square
void set_bit(Bitboard *bb, int square) {
    *bb |= (1ULL << square);
}

// Function to clear a bit at a specific square
void clear_bit(Bitboard *bb, int square) {
    *bb &= ~(1ULL << square);
}

// Function to get a bit at a specific square (1 if occupied, 0 if not)
int get_bit(Bitboard bb, int square) {
    return (bb >> square) & 1;
}

// Function to print the bitboard (for debugging)
void print_bitboard(Bitboard bb) {
    for (int rank = 7; rank >= 0; rank--) {
        for (int file = 0; file < 8; file++) {
            int square = rank * 8 + file;
            printf("%d ", get_bit(bb, square));
        }
        printf("\n");
    }
    printf("\n");
}

// Move masks
Bitboard north(Bitboard bb) {
    return bb << 8;
}

Bitboard south(Bitboard bb) {
    return bb >> 8;
}

Bitboard east(Bitboard bb) {
    return (bb & ~fileH) << 1;
}

Bitboard west(Bitboard bb) {
    return (bb & ~fileA) >> 1;
}

Bitboard northeast(Bitboard bb) {
    return (bb & ~fileH) << 9;
}

Bitboard northwest(Bitboard bb) {
    return (bb & ~fileA) << 7;
}

Bitboard southeast(Bitboard bb) {
    return (bb & ~fileH) >> 7;
}

Bitboard southwest(Bitboard bb) {
    return (bb & ~fileA) >> 9;
}

// Knight move masks (pre-calculated)
Bitboard knight_moves[64];

void init_knight_moves() {
    for (int square = 0; square < 64; square++) {
        Bitboard bb = 1ULL << square;
        knight_moves[square] = (north(north(east(bb))) | north(north(west(bb))) |
                                south(south(east(bb))) | south(south(west(bb))) |
                                east(east(north(bb))) | east(east(south(bb))) |
                                west(west(north(bb))) | west(west(south(bb))));
    }
}

// Example: Pawn move masks (1 square forward for white)
Bitboard white_pawn_moves(Bitboard pawns) {
    return north(pawns);
}

Bitboard black_pawn_moves(Bitboard pawns) {
    return south(pawns);
}

// Example: Pawn attack masks (pre-calculated)
Bitboard white_pawn_attacks(Bitboard pawns) {
    return northeast(pawns) | northwest(pawns);
}

Bitboard black_pawn_attacks(Bitboard pawns) {
    return southeast(pawns) | southwest(pawns);
}

// Board initialization (setup chess board position)
void init_bitboards() {
    // Set white pawns on rank 2
    bitboards[WHITE_PAWNS] = rank1 << 8;
    // Set black pawns on rank 7
    bitboards[BLACK_PAWNS] = rank8 >> 8;
    // Add other piece initialization as needed
}

// Rook attacks (horizontal and vertical)
Bitboard rook_attacks(Bitboard rooks, Bitboard blockers) {
    Bitboard attacks = 0ULL;

    Bitboard current_rook = rooks;
    while (current_rook) {
        int square = __builtin_ffsll(current_rook) - 1; // find the first rook
        clear_bit(&current_rook, square);

        // Generate attacks in all 4 directions
        Bitboard attack_north = north(1ULL << square);
        while (attack_north && !(attack_north & blockers)) {
            attacks |= attack_north;
            attack_north = north(attack_north);
        }

        Bitboard attack_south = south(1ULL << square);
        while (attack_south && !(attack_south & blockers)) {
            attacks |= attack_south;
            attack_south = south(attack_south);
        }

        Bitboard attack_east = east(1ULL << square);
        while (attack_east && !(attack_east & blockers)) {
            attacks |= attack_east;
            attack_east = east(attack_east);
        }

        Bitboard attack_west = west(1ULL << square);
        while (attack_west && !(attack_west & blockers)) {
            attacks |= attack_west;
            attack_west = west(attack_west);
        }
    }
    
    return attacks;
}

// Bishop attacks (for diagonal sliding)
Bitboard bishop_attacks(Bitboard bishops, Bitboard blockers) {
    Bitboard attacks = 0ULL;

    Bitboard current_bishop = bishops;
    while (current_bishop) {
        int square = __builtin_ffsll(current_bishop) - 1; // find the first bishop
        clear_bit(&current_bishop, square);

        // Generate attacks in all 4 diagonal directions
        Bitboard attack_northeast = northeast(1ULL << square);
        while (attack_northeast && !(attack_northeast & blockers)) {
            attacks |= attack_northeast;
            attack_northeast = northeast(attack_northeast);
        }

        Bitboard attack_northwest = northwest(1ULL << square);
        while (attack_northwest && !(attack_northwest & blockers)) {
            attacks |= attack_northwest;
            attack_northwest = northwest(attack_northwest);
        }

        Bitboard attack_southeast = southeast(1ULL << square);
        while (attack_southeast && !(attack_southeast & blockers)) {
            attacks |= attack_southeast;
            attack_southeast = southeast(attack_southeast);
        }

        Bitboard attack_southwest = southwest(1ULL << square);
        while (attack_southwest && !(attack_southwest & blockers)) {
            attacks |= attack_southwest;
            attack_southwest = southwest(attack_southwest);
        }
    }

    return attacks;
}

// Queen attacks (combination of rook and bishop attacks)
Bitboard queen_attacks(Bitboard queens, Bitboard blockers) {
    return rook_attacks(queens, blockers) | bishop_attacks(queens, blockers);
}


