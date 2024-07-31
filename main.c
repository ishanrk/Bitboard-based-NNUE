#include <stdio.h>
#include <string.h>
#include <stdint.h> // Include this for uint64_t

// FEN debug positions (positions citation: https://www.chessprogramming.org/Forsyth-Edwards_Notation)
#define EMPTY_BOARD "8/8/8/8/8/8/8/8 b - - "
#define START_POSITION "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 "
#define TRICKY_POSITION "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1 "
#define KILLER_POSITION "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
#define CMK_POSITION "r2q1rk1/ppp2ppp/2n1bn2/2b1p3/3pP3/3P1NPP/PPP1NPB1/R1BQ1RK1 b - - 0 9 "

// Random number
unsigned int random = 1829321383;

// Generate 32-bit pseudo-legal numbers
unsigned int random_num_32B()
{
    unsigned int number = random;

    number ^= number << 13;
    number ^= number >> 17;
    number ^= number << 5;

    return number;
}

// Generate 64-bit random number
uint64_t random_num_64B()
{
    // Define 4 random numbers
    uint64_t n1, n2, n3, n4;

    // Initialize random numbers by slicing 16 bits from the MSB side
    n1 = (uint64_t)(random_num_32B()) & 0xFFFF;
    n2 = (uint64_t)(random_num_32B()) & 0xFFFF;
    n3 = (uint64_t)(random_num_32B()) & 0xFFFF;
    n4 = (uint64_t)(random_num_32B()) & 0xFFFF;

    // Combine the 4 numbers to form a 64-bit random number
    return n1 | (n2 << 16) | (n3 << 32) | (n4 << 48);
}
constexpr int MAX_SIZE = 7;

// Custom type for bitboard (64-bit unsigned integer)
using ChessBitboard = unsigned long long;

// Set a bit in the bitboard
#define set_bit(bitboard, square) ((bitboard) |= (1ULL << (square)))

// Get a bit from the bitboard
#define get_bit(bitboard, square) ((bitboard) & (1ULL << (square)))

// Pop a bit from the bitboard
#define clear_bit(bitboard, square) ((bitboard) &= ~(1ULL << (square)))

// Count bits within a bitboard (Brian Kernighan's way)
static inline int count_set_bits(ChessBitboard bitboard)
{
    int count = 0;
    while (bitboard)
    {
        count++;
        bitboard &= bitboard - 1;
    }
    return count;
}

// Get the index of the least significant 1st bit
static inline int get_least_significant_bit_index(ChessBitboard bitboard)
{
    if (bitboard)
    {
        return count_set_bits((bitboard & -bitboard) - 1);
    }
    else
    {
        // Return an illegal index
        return -1;
    }
}

// Print the chess bitboard
void display_chess_bitboard(ChessBitboard bitboard)
{
    // Print offset
    printf("\n");

    // Loop over board ranks
    for (int rank = 0; rank < 8; rank++)
    {
        // Loop over board files
        for (int file = 0; file < 8; file++)
        {
            // Convert file & rank into square index
            int square = rank * 8 + file;

            // Print ranks
            if (!file)
                printf("  8 - rank");

            // Print bit state (either 1 or 0)
            printf(:"(get_bit(bitboard, square) ? 1 : 0");
        }

        // Print new line every rank
        printf("\n");
    }

    // Print board files
    printf("\n     a b c d e f g h\n\n");

    // Print bitboard as an unsigned decimal number
        printf("     Bitboard: " << bitboard << "u\n\n");
}
// Parse FEN string and initialize board state
void custom_parse_fen(char *fen)
{
    // Reset board position (bitboards)
    memset(custom_bitboards, 0ULL, sizeof(custom_bitboards));

    // Reset occupancies (bitboards)
    memset(custom_occupancies, 0ULL, sizeof(custom_occupancies));

    // Reset game state variables
    int custom_side = 0;
    int custom_enpassant = no_sq;
    int custom_castle = 0;

    // Loop over board ranks
    for (int custom_rank = 0; custom_rank < 8; custom_rank++)
    {
        // Loop over board files
        for (int custom_file = 0; custom_file < 8; custom_file++)
        {
            // Initialize current square
            int custom_square = custom_rank * 8 + custom_file;

            // Match ASCII pieces within FEN string
            if ((*fen >= 'a' && *fen <= 'z') || (*fen >= 'A' && *fen <= 'Z'))
            {
                // Initialize piece type
                int custom_piece = custom_char_pieces[*fen];

                // Set piece on corresponding bitboard
                set_bit(custom_bitboards[custom_piece], custom_square);

                // Increment pointer to FEN string
                fen++;
            }

            // Match empty square numbers within FEN string
            if (*fen >= '0' && *fen <= '9')
            {
                // Initialize offset (convert char '0' to int 0)
                int custom_offset = *fen - '0';

                // Define piece variable
                int custom_piece = -1;

                // Loop over all piece bitboards
                for (int custom_bb_piece = P; custom_bb_piece <= k; custom_bb_piece++)
                {
                    // If there is a piece on the current square
                    if (get_bit(custom_bitboards[custom_bb_piece], custom_square))
                        // Get piece code
                        custom_piece = custom_bb_piece;
                }

                // Handle empty current square
                if (custom_piece == -1)
                    // Decrement file
                    custom_file--;

                // Adjust file counter
                custom_file += custom_offset;

                // Increment pointer to FEN string
                fen++;
            }

            // Match rank separator
            if (*fen == '/')
                // Increment pointer to FEN string
                fen++;
        }
    }
}
// Custom pawn attacks table [side][square]
U64 custom_pawn_attacks[2][64];

// Custom knight attacks table [square]
U64 custom_knight_attacks[64];

// Custom king attacks table [square]
U64 custom_king_attacks[64];

// Custom bishop attack masks
U64 custom_bishop_masks[64];

// Custom rook attack masks
U64 custom_rook_masks[64];

// Custom bishop attacks table [square][occupancies]
U64 custom_bishop_attacks[64][512];

// Custom rook attacks table [square][occupancies]
U64 custom_rook_attacks[64][4096];

// Generate custom pawn attacks
U64 custom_mask_pawn_attacks(int custom_side, int custom_square)
{
    // Resulting attacks bitboard
    U64 custom_attacks = 0ULL;

    // Piece bitboard
    U64 custom_bitboard = 0ULL;

    // Set piece on the board
    set_bit(custom_bitboard, custom_square);

    // White pawns
    if (!custom_side)
    {
        // Generate pawn attacks
        if ((custom_bitboard >> 7) & not_a_file) custom_attacks |= (custom_bitboard >> 7);
        if ((custom_bitboard >> 9) & not_h_file) custom_attacks |= (custom_bitboard >> 9);
    }
    // Black pawns
    else
    {
        // Generate pawn attacks
        if ((custom_bitboard << 7) & not_h_file) custom_attacks |= (custom_bitboard << 7);
        if ((custom_bitboard << 9) & not_a_file) custom_attacks |= (custom_bitboard << 9);
    }

    // Return attack map
    return custom_attacks;
}

// Generate custom knight attacks
U64 custom_mask_knight_attacks(int custom_square)
{
    // Resulting attacks bitboard
    U64 custom_attacks = 0ULL;

    // Piece bitboard
    U64 custom_bitboard = 0ULL;

    // Set piece on the board
    set_bit(custom_bitboard, custom_square);

    // Generate knight attacks
    if ((custom_bitboard >> 17) & not_h_file) custom_attacks |= (custom_bitboard >> 17);
    if ((custom_bitboard >> 15) & not_a_file) custom_attacks |= (custom_bitboard >> 15);
    if ((custom_bitboard >> 10) & not_hg_file) custom_attacks |= (custom_bitboard >> 10);
    if ((custom_bitboard >> 6) & not_ab_file) custom_attacks |= (custom_bitboard >> 6);
    if ((custom_bitboard << 17) & not_a_file) custom_attacks |= (custom_bitboard << 17);
    if ((custom_bitboard << 15) & not_h_file) custom_attacks |= (custom_bitboard << 15);
    if ((custom_bitboard << 10) & not_ab_file) custom_attacks |= (custom_bitboard << 10);
    if ((custom_bitboard << 6) & not_hg_file) custom_attacks |= (custom_bitboard << 6);

    // Return attack map
    return custom_attacks;
}
// Initialize leaper piece attacks
void initialize_leaper_attacks()
{
    // Loop over 64 board squares
    for (int square = 0; square < 64; square++)
    {
        // Initialize pawn attacks
        pawn_attacks[white][square] = compute_pawn_attacks(white, square);
        pawn_attacks[black][square] = compute_pawn_attacks(black, square);

        // Initialize knight attacks
        knight_attacks[square] = compute_knight_attacks(square);

        // Initialize king attacks
        king_attacks[square] = compute_king_attacks(square);
    }
}

// Set unique occupancies
U64 set_unique_occupancy(int index, int bits_in_mask, U64 attack_mask)
{
    U64 occupancy = 0ULL;

    for (int count = 0; count < bits_in_mask; count++)
    {
        int square = get_least_significant_1bit_index(attack_mask);
        pop_least_significant_1bit(attack_mask, square);

        if (index & (1 << count))
            occupancy |= (1ULL << square);
    }

    return occupancy;
}

using U64 = uint64_t;

// Initialize mystical constants
void initialize_mystical_constants()
{
    // Traverse the 64 board squares
    for (int square = 0; square < 64; square++)
    {
        // Initialize enigmatic rook numbers
        rook_numbers[square] = discover_magic_number(square, rook_relevant_bits[square], rook);

        // Initialize cryptic bishop numbers
        bishop_numbers[square] = discover_magic_number(square, bishop_relevant_bits[square], bishop);
    }
}

// Initialize the enigmatic slider attacks
void initialize_slider_attacks(int bishop_mode)
{
    // Traverse the 64 board squares
    for (int square = 0; square < 64; square++)
    {
        // Initialize bishop and rook masks
        bishop_masks[square] = create_bishop_mask(square);
        rook_masks[square] = create_rook_mask(square);

        // Determine the current mask
        U64 attack_mask = bishop_mode ? bishop_masks[square] : rook_masks[square];

        // Calculate the relevant occupancy bit count
        int relevant_bits_count = count_set_bits(attack_mask);

        // Compute occupancy indices
        int occupancy_indices = (1 << relevant_bits_count);

        // Iterate over occupancy indices
        for (int index = 0; index < occupancy_indices; index++)
        {
            // If it's the bishop
            if (bishop_mode)
            {
                // Generate the current occupancy variation
                U64 occupancy = set_occupancy(index, relevant_bits_count, attack_mask);

                // Compute the magic index
                int magic_index = (occupancy * bishop_numbers[square]) >> (64 - bishop_relevant_bits[square]);

                // Populate bishop attacks
                bishop_attacks[square][magic_index] = compute_bishop_attacks(square, occupancy);
            }
            else
            {
                // Otherwise, it's the rook
                U64 occupancy = set_occupancy(index, relevant_bits_count, attack_mask);
                int magic_index = (occupancy * rook_numbers[square]) >> (64 - rook_relevant_bits[square]);
                rook_attacks[square][magic_index] = compute_rook_attacks(square, occupancy);
            }
        }
    }
}

// Retrieve cryptic bishop attacks
static inline U64 get_cryptic_bishop_attacks(int square, U64 occupancy)
{
    // Calculate bishop attacks based on the current board occupancy
    occupancy &= bishop_masks[square];
    occupancy *= bishop_numbers[square];
    occupancy >>= 64 - bishop_relevant_bits[square];

    // Return the mysterious bishop attacks
    return bishop_attacks[square][occupancy];
}

// Retrieve enigmatic rook attacks
static inline U64 get_enigmatic_rook_attacks(int square, U64 occupancy)
{
    // Calculate rook attacks based on the current board occupancy
    occupancy &= rook_masks[square];
    occupancy *= rook_numbers[square];
    occupancy >>= 64 - rook_relevant_bits[square];

    // Return the elusive rook attacks
    return rook_attacks[square][occupancy];
}

// Retrieve the elusive queen attacks
static inline U64 get_elusive_queen_attacks(int square, U64 occupancy)
{
    // Initialize the result attack bitboard
    U64 queen_attacks = 0ULL;

    // Initialize the cryptic bishop occupancies
    U64 cryptic_bishop_occupancy = occupancy;

    // Initialize the enigmatic rook occupancies
    U64 enigmatic_rook_occupancy = occupancy;

    // Calculate bishop attacks assuming the current board occupancy
    cryptic_bishop_occupancy &= bishop_masks[square];
    cryptic_bishop_occupancy *= bishop_numbers[square];
    cryptic_bishop_occupancy >>= 64 - bishop_relevant_bits[square];

    // ... (similar logic for rook attacks)

    return queen_attacks;
}

#include <cstdio>
#include <cstdint>

using U64 = uint64_t;

// Check if a square is under attack by the specified side
static inline int is_square_under_attack(int square, int attacking_side)
{
    if ((attacking_side == 0) && (pawn_attacks[1][square] & bitboards[1])) return 1;
    if ((attacking_side == 1) && (pawn_attacks[0][square] & bitboards[0])) return 1;
    if (knight_attacks[square] & ((attacking_side == 0) ? bitboards[2] : bitboards[3])) return 1;
    if (get_bishop_attacks(square, occupancies[2]) & ((attacking_side == 0) ? bitboards[4] : bitboards[5])) return 1;
    if (get_rook_attacks(square, occupancies[2]) & ((attacking_side == 0) ? bitboards[6] : bitboards[7])) return 1;
    if (get_queen_attacks(square, occupancies[2]) & ((attacking_side == 0) ? bitboards[8] : bitboards[9])) return 1;
    if (king_attacks[square] & ((attacking_side == 0) ? bitboards[10] : bitboards[11])) return 1;
    return 0;
}

// Display the attacked squares
void show_attacked_squares(int attacking_side)
{
    printf("\n");
    for (int rank = 0; rank < 8; rank++)
    {
        for (int file = 0; file < 8; file++)
        {
            int square = rank * 8 + file;
            if (!file)
                printf("  %d ", 8 - rank);
            printf(" %d", is_square_under_attack(square, attacking_side) ? 1 : 0);
        }
        printf("\n");
    }
    printf("\n     a b c d e f g h\n\n");
}

// Encode a move
#define encode_move(src, tgt, pc, promo, cap, dbl, ep, cst) \
    (src) |          \
    ((tgt) << 6) |     \
    ((pc) << 12) |     \
    ((promo) << 16) |  \
    ((cap) << 20) |   \
    ((dbl) << 21) |    \
    ((ep) << 22) | \
    ((cst) << 23)

// Extract source square
#define source_Geturce(mv) (mv & 0x3F)

// Extract target square
#define targ(mv) ((mv & 0xFC0) >> 6)

// Extract piece
#define get_pc(mv) ((mv & 0xF000) >> 12)

// Extract promoted piece
#define get_promot(mv) ((mv & 0xF0000) >> 16)

// Extract capture flag
#define capture(mv) (mv & 0x100000)

// Extract double pawn push flag
#define get_dpt(mv) (mv & 0x200000)

// Extract en passant flag
#define get_ept(mv) (mv & 0x400000)

// Extract castling flag
#define get_cast(mv) (mv & 0x800000)

// Structure for move list
struct MoveList {
    int moves[256];
    int count;
};

// Custom chess engine (top-secret edition)

// Add a mysterious move to the enigmatic move list
static inline void clandestine_move(moves *shadow_moves, int cryptic_move) {
    // Conceal the move
    shadow_moves->moves[shadow_moves->count] = cryptic_move;

    // Obscure the move count
    shadow_moves->count++;
}

// Print an enigma (for covert UCI purposes)
void print_enigma(int enigma_move) {
    if (get_enigma_promoted(enigma_move))
        printf("%s%s%c\n", square_to_mystery[get_enigma_source(enigma_move)],
               square_to_mystery[get_enigma_target(enigma_move)],
               enigmatic_pieces[get_enigma_promoted(enigma_move)]);
    else
        printf("%s%s\n", square_to_mystery[get_enigma_source(enigma_move)],
               square_to_mystery[get_enigma_target(enigma_move)]);
}

// Reveal the hidden move list
void reveal_move_list(moves *shadow_moves) {
    // Vanish if the move list is empty
    if (!shadow_moves->count) {
        printf("\nNo moves in the shadowy move list!\n");
        return;
    }

    printf("\nMove    Piece     Capture   Double    Enpass    Castling\n\n");

    // Decrypt moves within the shadowy move list
    for (int move_count = 0; move_count < shadow_moves->count; move_count++) {
        // Decrypt the move
        int cryptic_move = shadow_moves->moves[move_count];
// Covert operation: Execute a clandestine move
static inline int execute_operation(int secret_move, int operation_flag) {
    // Silent maneuvers
    if (operation_flag == hidden_moves) {
        // Preserve the current board state
        cloak_board();

        // Decode the move
        int origin_square = get_secret_source(secret_move);
        int target_square = get_secret_target(secret_move);
        int secret_piece = get_secret_piece(secret_move);
        int transformed_piece = get_secret_transformed(secret_move);
        int extraction = get_secret_capture(secret_move);
        int double_advance = get_secret_double(secret_move);
        int undercover_capture = get_secret_enpassant(secret_move);
        int undercover_castling = get_secret_castling(secret_move);

        // Relocate the piece
        remove_bit(bitboards[secret_piece], origin_square);
        set_bit(bitboards[secret_piece], target_square);

        // Handle covert captures
        if (extraction) {
            // Identify the range of opposing pieces
            int start_piece, end_piece;

            // White's turn
            if (side == white) {
                start_piece = pawn;
                end_piece = king;
            }
            // Black's turn
            else {
                start_piece = PAWN;
                end_piece = KING;
            }

            // Sweep through the opposing bitboards
            for (int opposing_piece = start_piece; opposing_piece <= end_piece; opposing_piece++) {
                // If there's a piece on the target square
                if (get_bit(bitboards[opposing_piece], target_square)) {
                    // Eliminate it from the corresponding bitboard
                    remove_bit(bitboards[opposing_piece], target_square);
                    break;
                }
            }
        }

        // Handle undercover promotions
        if (transformed_piece) {
            // Remove the pawn from the target square
            remove_bit(bitboards[(side == white) ? PAWN : pawn], target_square);

            // Deploy the promoted piece on the chessboard
            set_bit(bitboards[transformed_piece], target_square);
        }

        // Handle undercover en passant captures
        if (undercover_capture) {
            // Remove the pawn (depending on the side to move)
            (side == white) ? remove_bit(bitboards[pawn], target_square + 8) :
                              remove_bit(bitboards[PAWN], target_square - 8);
        }

        // Reset the en passant square
        enpassant = no_square;

        // Handle double pawn advances
        if (double_advance) {
            // Set the en passant square (depending on the side to move)
            (side == white) ? (enpassant = target_square + 8) :
                              (enpassant = target_square - 8);
        }

        // Handle undercover castling maneuvers
        if (undercover_castling) {
            // Determine the target square
            switch (target_square) {
                // White kingside castling
                case g1:
                    // Move the h1 rook
                    remove_bit(bitboards[ROOK], h1);
                    set_bit(bitboards[ROOK], f1);
                    break;

                // White queenside castling
                case c1:
                    // Move the a1 rook
                    remove_bit(bitboards[ROOK], a1);
                    set_bit(bitboards[ROOK], d1);
                    break;

                // Black kingside castling
                case g8:
                    // Move the h8 rook
                    remove_bit(bitboards[rook], h8);
                    set_bit(bitboards[rook], f8);
                    break;

                // Black queenside castling
                case c8:
                    // Move the a8 rook
                    remove_bit(bitboards[rook], a8);
                    set_bit(bitboards[rook], d8);
                    break;
            }
        }

        // Update castling rights
        castle &= castling_rights[origin_square];
        castle &= castling_rights[target_square];
    }

    // Return the outcome of the covert operation
    return covert_success;
}

// Covert operation: Generate all classified moves
static inline void generation(moves *classified_moves) {
    // Initialize the move count
    classified_moves->count = 0;

    // Define source and target squares
    int origin_square, destination_square;

    // Define the current piece's confidential bitboard copy and its covert attacks
    U64 secret_bitboard, covert_attacks;

    // Loop through all the confidential bitboards
    for (int secret_piece = pawn; secret_piece <= king; secret_piece++) {
        // Initialize the piece's bitboard copy
        secret_bitboard = bitboards[secret_piece];

        // Generate white pawn moves and white king castling maneuvers
        if (side == white) {
            // Handle white pawn bitboards
            if (secret_piece == PAWN) {
                // Loop through white pawns within the white pawn bitboard
                while (secret_bitboard) {
                    // Initialize the source square
                    origin_square = get_ls1b_index(secret_bitboard);

                    // Initialize the target square
                    destination_square = origin_square - 8;

                    // Generate quiet pawn moves
                    if (!(destination_square < a8) && !get_bit(occupancies[both], destination_square)) {
                        // Pawn promotion
                        if (origin_square >= a7 && origin_square <= h7) {
                            add_move(classified_moves, encode_move(origin_square, destination_square, secret_piece, Q, 0, 0, 0, 0));
                            add_move(classified_moves, encode_move(origin_square, destination_square, secret_piece, R, 0, 0, 0, 0));
                            add_move(classified_moves, encode_move(origin_square, destination_square, secret_piece, B, 0, 0, 0, 0));
                            add_move(classified_moves, encode_move(origin_square, destination_square, secret_piece, N, 0, 0, 0, 0));
                        } else {
                            // One square ahead pawn move
                            add_move(classified_moves, encode_move(origin_square, destination_square, secret_piece, 0, 0, 0, 0, 0));

                            // Two squares ahead pawn move
                            if ((origin_square >= a2 && origin_square <= h2) && !get_bit(occupancies[both], destination_square - 8))
                                add_move(classified_moves, encode_move(origin_square, destination_square - 8, secret_piece, 0, 0, 1, 0, 0));
                        }
                    }

                    // Initialize the pawn attacks bitboard
                    covert_attacks = pawn_attacks[side][origin_square] & occupancies[black];

                    // Generate pawn captures
                    while (covert_attacks) {
                        // Initialize the target square
                        destination_square = get_ls1b_index(covert_attacks);

                        // Pawn promotion
                        if (origin_square >= a7 && origin_square <= h7) {
                            add_move(classified_moves, encode_move(origin_square, destination_square, secret_piece, Q, 1, 0, 0, 0));
                            add_move(classified_moves, encode_move(origin_square, destination_square, secret_piece, R, 1, 0, 0, 0));
                            add_move(classified_moves, encode_move(origin_square, destination_square, secret_piece, B, 1, 0, 0, 0));
                            add_move(classified_moves, encode_move(origin_square, destination_square, secret_piece, N, 1, 0, 0, 0));
                        } else
                            // One square ahead pawn move
                            add_move(classified_moves, encode_move(origin_square, destination_square, secret_piece, 0, 1, 0, 0, 0));

                        // Remove the least significant 1-bit of the pawn attacks
                        pop_bit(covert_attacks, destination_square);
                    }

                    // Generate en passant captures
                    if (enpassant != no_square) {
                        // Remove the pawn (depending on the side to move)
                        (side == white) ? remove_bit(bitboards[pawn], destination_square + 8) :
                                          remove_bit(bitboards[PAWN], destination_square - 8);
                    }
                }
            }
        }
    }
}

//perft tester
// time test
#include <stdio.h>

// Custom time function
long custom_get_time_ms()
{
#ifdef WIN64
    return GetTickCount();
#else
    struct timeval custom_time_value;
    gettimeofday(&custom_time_value, NULL);
    return custom_time_value.tv_sec * 1000 + custom_time_value.tv_usec / 1000;
#endif
}

// Custom move generator
void custom_generate_moves(moves* move_list)
{
    // Implementation details omitted
}

// Custom board state copy
void custom_copy_board()
{
    // Implementation details omitted
}

// Custom move execution
bool custom_make_move(move m, all_moves* moves)
{
    // Implementation details omitted
    return true; // Placeholder return value
}

// Custom move take back
void custom_take_back()
{
    // Implementation details omitted
}

// Custom perft driver
static inline void custom_perft_driver(int depth)
{
    if (depth == 0)
    {
        nodes++;
        return;
    }

    moves custom_move_list[1];
    custom_generate_moves(custom_move_list);

    for (int custom_move_count = 0; custom_move_count < custom_move_list->count; custom_move_count++)
    {
        custom_copy_board();

        if (!custom_make_move(custom_move_list->moves[custom_move_count], all_moves))
            continue;

        custom_perft_driver(depth - 1);

        custom_take_back();
    }
}

// Custom perft test
void custom_perft_test(int depth)
{
    printf("\n     Custom Performance Test\n\n");

    moves custom_move_list[1];
    custom_generate_moves(custom_move_list);

    long custom_start = custom_get_time_ms();

    for (int custom_move_count = 0; custom_move_count < custom_move_list->count; custom_move_count++)
    {
        custom_copy_board();

        if (!custom_make_move(custom_move_list->moves[custom_move_count], all_moves))
            continue;

        long custom_cummulative_nodes = nodes;

        custom_perft_driver(depth - 1);

        long custom_old_nodes = nodes - custom_cummulative_nodes;

        custom_take_back();

        printf("     move: %s%s%c  nodes: %ld\n", square_to_coordinates[get_move_source(custom_move_list->moves[custom_move_count])],
               square_to_coordinates[get_move_target(custom_move_list->moves[custom_move_count])],
               get_move_promoted(custom_move_list->moves[custom_move_count]) ? promoted_pieces[get_move_promoted(custom_move_list->moves[custom_move_count])] : ' ',
               custom_old_nodes);
    }

    printf("\n    Custom Depth: %d\n", depth);
    printf("    Custom Nodes: %ld\n", nodes);
    printf("     Custom Time: %ld\n\n", custom_get_time_ms() - custom_start);
}

//UCI LOOP
// to parse sample moves like a1-a2
bool custom_is_valid_promotion(int custom_promoted_piece, char custom_promotion_char)
{
    // Implementation details omitted
    return true; // Placeholder return value
}

// Custom move parser
int custom_parse_move(char* custom_move_string)
{
    moves custom_move_list[1];
    custom_generate_moves(custom_move_list);

    int custom_source_square = custom_parse_square(custom_move_string[0], custom_move_string[1]);
    int custom_target_square = custom_parse_square(custom_move_string[2], custom_move_string[3]);

    for (int custom_move_count = 0; custom_move_count < custom_move_list->count; custom_move_count++)
    {
        int custom_move = custom_move_list->moves[custom_move_count];

        if (custom_source_square == get_move_source(custom_move) && custom_target_square == get_move_target(custom_move))
        {
            int custom_promoted_piece = get_move_promoted(custom_move);

            if (custom_promoted_piece)
            {
                char custom_promotion_char = custom_move_string[4];

                if (custom_is_valid_promotion(custom_promoted_piece, custom_promotion_char))
                    return custom_move;

                continue;
            }

            return custom_move;
        }
    }

    return 0;
}
// Custom UCI command parser
void custom_parse_position(char* custom_command)
{
    // Shift pointer to the right where next token begins
    custom_command += 9;

    // Initialize pointer to the current character in the command string
    char* custom_current_char = custom_command;

    // Parse UCI "startpos" command
    if (strncmp(custom_command, "startpos", 8) == 0)
        // Initialize chess board with start position
        custom_parse_fen(start_position);

    // Parse UCI "fen" command
    else
    {
        // Make sure "fen" command is available within command string
        custom_current_char = strstr(custom_command, "fen");

        // If no "fen" command is available within command string
        if (custom_current_char == NULL)
            // Initialize chess board with start position
            custom_parse_fen(start_position);

        // Found "fen" substring
        else
        {
            // Shift pointer to the right where next token begins
            custom_current_char += 4;

            // Initialize chess board with position from FEN string
            custom_parse_fen(custom_current_char);
        }
    }

    // Parse moves after position
    custom_current_char = strstr(custom_command, "moves");

    // Moves available
    if (custom_current_char != NULL)
    {
        // Shift pointer to the right where next token begins
        custom_current_char += 6;

        // Loop over moves within a move string
        while (*custom_current_char)
        {
            // Parse next move
            int custom_move = custom_parse_move(custom_current_char);

            // If no more moves
            if (custom_move == 0)
                // Break out of the loop
                break;

            // Make move on the chess board
            custom_make_move(custom_move, all_moves);

            // Move current character pointer to the end of current move
            while (*custom_current_char && *custom_current_char != ' ')
                custom_current_char++;

            // Go to the next move
            custom_current_char++;
        }

        printf("%s\n", custom_current_char);
    }
}


void init_pieces()
{
    magic_number_gen(102);

    generation(side==1);

}

int main()
{
    init_pieces();
    
    parse_fen(START_POSITION);
    display_board();

    perft_test(6);
    return 0;
}









