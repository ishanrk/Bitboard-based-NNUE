#include <stdio.h>
#include <string.h>
#include <stdint.h> // For uint64_t

// FEN position strings (source: https://www.chessprogramming.org/Forsyth-Edwards_Notation)
#define EMPTY_BOARD "8/8/8/8/8/8/8/8 b - - "
#define START_POSITION "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 "
#define TRICKY_POSITION "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1 "
#define KILLER_POSITION "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
#define CMK_POSITION "r2q1rk1/ppp2ppp/2n1bn2/2b1p3/3pP3/3P1NPP/PPP1NPB1/R1BQ1RK1 b - - 0 9 "

// A predetermined random number
unsigned int random = 1829321383;

// Function to produce 32-bit pseudo-random numbers
unsigned int random_num_32B()
{
    unsigned int number = random;

    number ^= number << 13;
    number ^= number >> 17;
    number ^= number << 5;

    return number;
}

// Function to produce 64-bit random numbers
uint64_t random_num_64B()
{
    // Generate four 16-bit random numbers
    uint64_t n1, n2, n3, n4;

    // Initialize each 16-bit random number
    n1 = (uint64_t)(random_num_32B()) & 0xFFFF;
    n2 = (uint64_t)(random_num_32B()) & 0xFFFF;
    n3 = (uint64_t)(random_num_32B()) & 0xFFFF;
    n4 = (uint64_t)(random_num_32B()) & 0xFFFF;

    // Combine the four 16-bit numbers into one 64-bit number
    return n1 | (n2 << 16) | (n3 << 32) | (n4 << 48);
}

constexpr int MAX_SIZE = 7;

// Alias for a 64-bit unsigned integer representing a chess board
using ChessBitboard = unsigned long long;

// Macro to set a bit on the bitboard
#define sbit(bitboard, square) ((bitboard) |= (1ULL << (square)))

// Macro to get the value of a bit on the bitboard
#define gbit(bitboard, square) ((bitboard) & (1ULL << (square)))

// Macro to clear a bit on the bitboard
#define cbit(bitboard, square) ((bitboard) &= ~(1ULL << (square)))

// Function to count the number of set bits in a bitboard using Brian Kernighan's algorithm
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

// Function to find the index of the least significant set bit
static inline int get_least_significant_bit_index(ChessBitboard bitboard)
{
    if (bitboard)
    {
        return count_set_bits((bitboard & -bitboard) - 1);
    }
    else
    {
        // Return an invalid index if the bitboard is empty
        return -1;
    }
}

// Function to print the chess bitboard
void display_chess_bitboard(ChessBitboard bitboard)
{
    // Print a newline for formatting
    printf("\n");

    // Loop through each rank of the board
    for (int rank = 0; rank < 8; rank++)
    {
        // Loop through each file of the board
        for (int file = 0; file < 8; file++)
        {
            // Calculate the square index from the rank and file
            int square = rank * 8 + file;

            // Print rank numbers at the beginning of each rank
            if (!file)
                printf("  8 - rank");

            // Print the state of the bit (1 or 0)
            printf(" %d", get_bit(bitboard, square) ? 1 : 0);
        }

        // Print a newline after each rank
        printf("\n");
    }

    // Print file letters for reference
    printf("\n     a b c d e f g h\n\n");

    // Print the bitboard as an unsigned decimal number
    printf("     Bitboard: %llu\n\n", bitboard);
}

// Function to parse a FEN string and initialize the board state
void custom_parse_fen(char *fen)
{
    // Reset the bitboards for all pieces
    memset(custom_bitboards, 0ULL, sizeof(custom_bitboards));

    // Clear the occupancy bitboards
    memset(custom_occupancies, 0ULL, sizeof(custom_occupancies));

    // Initialize game state variables
    int custom_side = 0;
    int custom_enpassant = no_sq;
    int custom_castle = 0;

    // Iterate over each rank
    for (int custom_rank = 0; custom_rank < 8; custom_rank++)
    {
        // Iterate over each file
        for (int custom_file = 0; custom_file < 8; custom_file++)
        {
            // Calculate the square index
            int custom_square = custom_rank * 8 + custom_file;

            // If the current character in the FEN string is a piece
            if ((*fen >= 'a' && *fen <= 'z') || (*fen >= 'A' && *fen <= 'Z'))
            {
                // Determine the piece type
                int custom_piece = custom_char_pieces[*fen];

                // Set the corresponding bit on the bitboard
                set_bit(custom_bitboards[custom_piece], custom_square);

                // Move to the next character in the FEN string
                fen++;
            }
            // Handle other FEN characters like numbers and slashes here (not shown)
        }
    }
    // Further parsing of FEN string to set side, en passant square, and castling rights (not shown)
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
        if ((custom_bitboard >> 7) & a) custom_attacks |= (custom_bitboard >> 7);
        if ((custom_bitboard >> 9) & h) custom_attacks |= (custom_bitboard >> 9);
    }
    // Black pawns
    else
    {
        // Generate pawn attacks
        if ((custom_bitboard << 7) & h) custom_attacks |= (custom_bitboard << 7);
        if ((custom_bitboard << 9) & a) custom_attacks |= (custom_bitboard << 9);
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
    if ((custom_bitboard >> 17) & !h) custom_attacks |= (custom_bitboard >> 17);
    if ((custom_bitboard >> 15) & !a) custom_attacks |= (custom_bitboard >> 15);
    if ((custom_bitboard >> 10) & !hg) custom_attacks |= (custom_bitboard >> 10);
    if ((custom_bitboard >> 6) & !ab) custom_attacks |= (custom_bitboard >> 6);
    if ((custom_bitboard << 17) & a) custom_attacks |= (custom_bitboard << 17);
    if ((custom_bitboard << 15) & h) custom_attacks |= (custom_bitboard << 15);
    if ((custom_bitboard << 10) & ab) custom_attacks |= (custom_bitboard << 10);
    if ((custom_bitboard << 6) & hg) custom_attacks |= (custom_bitboard << 6);

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
        custom_pawn_attacks[0][square] = custom_mask_pawn_attacks(0, square);
        custom_pawn_attacks[1][square] = custom_mask_pawn_attacks(1, square);

        // Initialize knight attacks
        custom_knight_attacks[square] = custom_mask_knight_attacks(square);

        // Initialize king attacks
        custom_king_attacks[square] = custom_mask_king_attacks(square);
    }
}
// Define unique occupancy configurations
U64 define_unique_occupancy(int index, int bits_in_mask, U64 attack_mask)
{
    U64 occupancy = 0ULL;

    for (int count = 0; count < bits_in_mask; count++)
    {
        int square = get_least_significant_bit_index(attack_mask);
        clear_bit(attack_mask, square);

        if (index & (1 << count))
            occupancy |= (1ULL << square);
    }

    return occupancy;
}


// Set up mysterious constants
void setup_mysterious_constants()
{
    // Iterate over each of the 64 board squares
    for (int square = 0; square < 64; square++)
    {
        // Determine magic numbers for rooks
        rook_numbers[square] = compute_magic_number(square, rook_relevant_bits[square], rook);

        // Determine magic numbers for bishops
        bishop_numbers[square] = compute_magic_number(square, bishop_relevant_bits[square], bishop);
    }
}

// Prepare the enigmatic slider attack patterns
void prepare_slider_attacks(int bishop_mode)
{
    for (int square = 0; square < 64; square++)
    {
        // Set up bishop and rook masks
        bishop_masks[square] = generate_bishop_mask(square);
        rook_masks[square] = generate_rook_mask(square);

        // Choose the appropriate mask
        U64 attack_mask = bishop_mode ? bishop_masks[square] : rook_masks[square];

        // Count the number of relevant bits in the mask
        int relevant_bits_count = count_set_bits(attack_mask);

        // Determine the total number of occupancy combinations
        int occupancy_combinations = (1 << relevant_bits_count);

        // Loop through all possible occupancy combinations
        for (int index = 0; index < occupancy_combinations; index++)
        {
            // For bishop
            if (bishop_mode)
            {
                // Create a variation of the occupancy
                U64 occupancy = define_occupancy(index, relevant_bits_count, attack_mask);

                // Calculate the magic index
                int magic_index = (occupancy * bishop_numbers[square]) >> (64 - bishop_relevant_bits[square]);

                // Fill in bishop attack patterns
                bishop_attacks[square][magic_index] = calculate_bishop_attacks(square, occupancy);
            }
            else
            {
                // For rook
                U64 occupancy = define_occupancy(index, relevant_bits_count, attack_mask);
                int magic_index = (occupancy * rook_numbers[square]) >> (64 - rook_relevant_bits[square]);
                rook_attacks[square][magic_index] = calculate_rook_attacks(square, occupancy);
            }
        }
    }
}
// Obtain the concealed bishop attack patterns
static inline U64 retrieve_concealed_bishop_attacks(int square, U64 occupancy)
{
    // Compute bishop attack patterns based on the given board state
    occupancy &= bishop_masks[square];
    occupancy *= bishop_numbers[square];
    occupancy >>= 64 - bishop_relevant_bits[square];

    // Return the secretive bishop attack patterns
    return bishop_attacks[square][occupancy];
}

// Obtain the hidden rook attack patterns
static inline U64 retrieve_hidden_rook_attacks(int square, U64 occupancy)
{
    // Compute rook attack patterns based on the given board state
    occupancy &= rook_masks[square];
    occupancy *= rook_numbers[square];
    occupancy >>= 64 - rook_relevant_bits[square];

    // Return the concealed rook attack patterns
    return rook_attacks[square][occupancy];
}

// Obtain the elusive queen attack patterns
static inline U64 retrieve_elusive_queen_attacks(int square, U64 occupancy)
{
    // Set up the attack bitboard for the queen
    U64 queen_attacks = 0ULL;

    // Set up the concealed bishop occupancy
    U64 concealed_bishop_occupancy = occupancy;

    // Set up the hidden rook occupancy
    U64 hidden_rook_occupancy = occupancy;

    // Compute bishop attack patterns based on the given board state
    concealed_bishop_occupancy &= bishop_masks[square];
    concealed_bishop_occupancy *= bishop_numbers[square];
    concealed_bishop_occupancy >>= 64 - bishop_relevant_bits[square];

    // ... (similar steps for rook attacks)

    return queen_attacks;
}

#include <cstdio>
#include <cstdint>

using U64 = uint64_t;

// Determine if a square is threatened by a specific side
static inline int is_square_threatened(int square, int attacking_side)
{
    if ((attacking_side == 0) && (pawn_attacks[1][square] & bitboards[1])) return 1;
    if ((attacking_side == 1) && (pawn_attacks[0][square] & bitboards[0])) return 1;
    if (knight_attacks[square] & ((attacking_side == 0) ? bitboards[2] : bitboards[3])) return 1;
    if (retrieve_bishop_attacks(square, occupancies[2]) & ((attacking_side == 0) ? bitboards[4] : bitboards[5])) return 1;
    if (retrieve_rook_attacks(square, occupancies[2]) & ((attacking_side == 0) ? bitboards[6] : bitboards[7])) return 1;
    if (retrieve_queen_attacks(square, occupancies[2]) & ((attacking_side == 0) ? bitboards[8] : bitboards[9])) return 1;
    if (king_attacks[square] & ((attacking_side == 0) ? bitboards[10] : bitboards[11])) return 1;
    return 0;
}

// Display squares under threat
void display_threatened_squares(int attacking_side)
{
    printf("\n");
    for (int rank = 0; rank < 8; rank++)
    {
        for (int file = 0; file < 8; file++)
        {
            int square = rank * 8 + file;
            if (!file)
                printf("  %d ", 8 - rank);
            printf(" %d", is_square_threatened(square, attacking_side) ? 1 : 0);
        }
        printf("\n");
    }
    printf("\n     a b c d e f g h\n\n");
}

// Encode a move
#define encoder(src, tgt, piece, promo, capture, dp, ep, cst) \
    (src) |          \
    ((tgt) << 6) |   \
    ((piece) << 12) | \
    ((promo) << 16) | \
    ((capture) << 20) | \
    ((double_push) << 21) | \
    ((en_passant) << 22) | \
    ((castling) << 23)

// Extract the origin square
#define orig(mv) (mv & 0x3F)

// Extract the destination square
#define dest(mv) ((mv & 0xFC0) >> 6)

// Extract the piece type
#define piece(mv) ((mv & 0xF000) >> 12)

// Extract the promoted piece type
#define getpromo(mv) ((mv & 0xF0000) >> 16)

// Extract the capture flag
#define capt(mv) (mv & 0x100000)

// Extract the double pawn move flag
#define dp(mv) (mv & 0x200000)

// Extract the en passant flag
#define ep(mv) (mv & 0x400000)

// Extract the castling flag
#define cast(mv) (mv & 0x800000)

// Structure for move collection
struct MoveCollection {
    int moves[256];
    int count;
};

// Custom chess engine (classified edition)

// Add a classified move to the move collection
static inline void add_classified_move(MoveCollection *hidden_moves, int encoded_move) {
    // Store the move in the collection
    hidden_moves->moves[hidden_moves->count] = encoded_move;

    // Increment the move count
    hidden_moves->count++;
}

// Output a move in a coded format
void output_coded_move(int encoded_move) {
    if (get_promotion(encoded_move))
        printf("%s%s%c\n", square_names[get_origin(encoded_move)],
               square_names[get_dest(encoded_move)],
               promotion_pieces[get_promotion(encoded_move)]);
    else
        printf("%s%s\n", square_names[get_origin(encoded_move)],
               square_names[get_dest(encoded_move)]);
}

// Disclose the move collection
void disclose_move_collection(MoveCollection *hidden_moves) {
    // Notify if the collection is empty
    if (!hidden_moves->count) {
        printf("\nNo moves in the classified collection!\n");
        return;
    }

    printf("\nMove    Piece     Capture   Double Push   En Passant   Castling\n\n");

    // Reveal moves in the collection
    for (int i = 0; i < hidden_moves->count; i++) {
        // Decode the move
        int encoded_move = hidden_moves->moves[i];
        // Covert operation: Execute a classified move
static inline int perform_classified_operation(int encoded_move, int flag) {
    // Execute silently
    if (flag == hidden_moves) {
        // Save the current board configuration
        conceal_board();

        // Decode the move
        int start_square = get_origin(encoded_move);
        int end_square = get_dest(encoded_move);
        int piece_type = get_piece(encoded_move);
        int promo_piece = get_promotion(encoded_move);
        int capture = is_capture(encoded_move);
        int double_move = is_double_push(encoded_move);
        int en_passant = is_en_passant(encoded_move);
        int castling = is_castling(encoded_move);

        // Move the piece
        clear_bit(bitboards[piece_type], start_square);
        set_bit(bitboards[piece_type], end_square);

        // Handle captures
        if (capture) {
            // Determine the range of opposing pieces
            int start_piece, end_piece;

            // White's move
            if (side == white) {
                start_piece = pawn;
                end_piece = king;
            }
            // Black's move
            else {
                start_piece = PAWN;
                end_piece = KING;
            }

            // Check opposing pieces
            for (int opponent_piece = start_piece; opponent_piece <= end_piece; opponent_piece++) {
                // If there is a piece on the destination square
                if (get_bit(bitboards[opponent_piece], end_square)) {
                    // Remove it from the respective bitboard
                    clear_bit(bitboards[opponent_piece], end_square);
                    break;
                }
            }
        }

        // Handle promotions
        if (promo_piece) {
            // Remove the pawn from the destination square
            clear_bit(bitboards[(side == white) ? PAWN : pawn], end_square);

            // Place the promoted piece
            set_bit(bitboards[promo_piece], end_square);
        }

        // Handle en passant captures
        if (en_passant) {
            // Remove the opponent's pawn (based on side)
            (side == white) ? clear_bit(bitboards[pawn], end_square + 8) :
                              clear_bit(bitboards[PAWN], end_square - 8);
        }

        // Reset the en passant square
        en_passant_square = no_square;

        // Handle double pawn advances
        if (double_move) {
            // Set the en passant square (based on side)
            (side == white) ? (en_passant_square = end_square + 8) :
                              (en_passant_square = end_square - 8);
        }

        // Handle castling moves
        if (castling) {
            // Determine the castling action
            switch (end_square) {
                // White kingside castling
                case g1:
                    // Move the rook from h1
                    clear_bit(bitboards[ROOK], h1);
                    set_bit(bitboards[ROOK], f1);
                    break;

                // White queenside castling
                case c1:
                    // Move the rook from a1
                    clear_bit(bitboards[ROOK], a1);
                    set_bit(bitboards[ROOK], d1);
                    break;

                // Black kingside castling
                case g8:
                    // Move the rook from h8
                    clear_bit(bitboards[rook], h8);
                    set_bit(bitboards[rook], f8);
                    break;

                // Black queenside castling
                case c8:
                    // Move the rook from a8
                    clear_bit(bitboards[rook], a8);
                    set_bit(bitboards[rook], d8);
                    break;
            }
        }

        castling_rights &= castling_rights[start_square];
        castling_rights &= castling_rights[end_square];
    }

    // Return the result of the classified operation
    return operation_success;
}
// Concealed operation: Produce all classified moves

static inline void generate_classified_moves(MoveCollection *classified_moves) {
    // Reset the number of moves
    classified_moves->count = 0;

    // Define starting and ending positions
    int start_square, end_square;

    // Define the piece's confidential bitboard replica and its covert attack patterns
    U64 piece_bitboard, attack_patterns;

    // Iterate over all piece bitboards
    for (int piece_type = pawn; piece_type <= king; piece_type++) {
        // Set the piece's bitboard replica
        piece_bitboard = bitboards[piece_type];

        // Handle moves for white pawns and castling for white kings
        if (side == white) {
            // Process white pawn bitboards
            if (piece_type == PAWN) {
                // Loop through pawns on the board
                while (piece_bitboard) {
                    // Determine the current position of the pawn
                    start_square = get_ls1b_index(piece_bitboard);

                    // Calculate the destination for a pawn move
                    end_square = start_square - 8;

                    // Generate straightforward pawn moves
                    if (!(end_square < a8) && !get_bit(occupancies[both], end_square)) {
                        // Handle pawn promotions
                        if (start_square >= a7 && start_square <= h7) {
                            add_move(classified_moves, encode_move(start_square, end_square, piece_type, Q, 0, 0, 0, 0));
                            add_move(classified_moves, encode_move(start_square, end_square, piece_type, R, 0, 0, 0, 0));
                            add_move(classified_moves, encode_move(start_square, end_square, piece_type, B, 0, 0, 0, 0));
                            add_move(classified_moves, encode_move(start_square, end_square, piece_type, N, 0, 0, 0, 0));
                        } else {
                            // Single square advance
                            add_move(classified_moves, encode_move(start_square, end_square, piece_type, 0, 0, 0, 0, 0));

                            // Double square advance if applicable
                            if ((start_square >= a2 && start_square <= h2) && !get_bit(occupancies[both], end_square - 8))
                                add_move(classified_moves, encode_move(start_square, end_square - 8, piece_type, 0, 0, 1, 0, 0));
                        }
                    }

                    // Compute pawn attack patterns
                    attack_patterns = pawn_attacks[side][start_square] & occupancies[black];

                    // Generate pawn captures
                    while (attack_patterns) {
                        // Determine the capture target
                        end_square = get_ls1b_index(attack_patterns);

                        // Handle pawn promotions on capture
                        if (start_square >= a7 && start_square <= h7) {
                            add_move(classified_moves, encode_move(start_square, end_square, piece_type, Q, 1, 0, 0, 0));
                            add_move(classified_moves, encode_move(start_square, end_square, piece_type, R, 1, 0, 0, 0));
                            add_move(classified_moves, encode_move(start_square, end_square, piece_type, B, 1, 0, 0, 0));
                            add_move(classified_moves, encode_move(start_square, end_square, piece_type, N, 1, 0, 0, 0));
                        } else
                            // Standard capture move
                            add_move(classified_moves, encode_move(start_square, end_square, piece_type, 0, 1, 0, 0, 0));

                        // Remove the capture position from the attack patterns
                        pop_bit(attack_patterns, end_square);
                    }

                    // Handle en passant captures
                    if (enpassant != no_square) {
                        // Remove the pawn based on the side
                        (side == white) ? remove_bit(bitboards[pawn], end_square + 8) :
                                          remove_bit(bitboards[PAWN], end_square - 8);
                    }
                }
            }
        }
    }
}

// Perft tester
// Timing function
#include <stdio.h>

// Custom timing function
long get_current_time_ms()
{
#ifdef WIN64
    return GetTickCount();
#else
    struct timeval time_value;
    gettimeofday(&time_value, NULL);
    return time_value.tv_sec * 1000 + time_value.tv_usec / 1000;
#endif
}

// Custom move generator
void generate_moves_custom(MoveCollection* move_list)
{
    // Implementation details omitted
}

// Custom board copy function
void copy_board_custom()
{
    // Implementation details omitted
}

// Custom move execution function
bool make_move_custom(Move move, AllMoves* moves)
{
    // Implementation details omitted
    return true; // Placeholder return value
}
// Custom move retraction
void revert_last_move()
{
    // Implementation details hidden
}

// Custom performance test driver
static inline void performance_test_driver(int level)
{
    if (level == 0)
    {
        nodes++;
        return;
    }

    MoveCollection move_list[1];
    generate_moves_custom(move_list);

    for (int move_index = 0; move_index < move_list->count; move_index++)
    {
        copy_board_custom();

        if (!make_move_custom(move_list->moves[move_index], all_moves))
            continue;

        performance_test_driver(level - 1);

        revert_last_move();
    }
}

// Custom performance test
void run_performance_test(int depth)
{
    printf("\n     Performance Test Results\n\n");

    MoveCollection move_list[1];
    generate_moves_custom(move_list);

    long start_time = get_current_time_ms();

    for (int move_index = 0; move_index < move_list->count; move_index++)
    {
        copy_board_custom();

        if (!make_move_custom(move_list->moves[move_index], all_moves))
            continue;

        long initial_node_count = nodes;

        performance_test_driver(depth - 1);

        long node_count_delta = nodes - initial_node_count;

        revert_last_move();

        printf("     move: %s%s%c  nodes: %ld\n", square_to_coordinates[get_move_source(move_list->moves[move_index])],
               square_to_coordinates[get_move_target(move_list->moves[move_index])],
               get_move_promoted(move_list->moves[move_index]) ? promoted_pieces[get_move_promoted(move_list->moves[move_index])] : ' ',
               node_count_delta);
    }

    printf("\n    Test Depth: %d\n", depth);
    printf("    Total Nodes: %ld\n", nodes);
    printf("     Time Elapsed: %ld\n\n", get_current_time_ms() - start_time);
}

// UCI LOOP
// Function to validate promotion
bool validate_promotion(int promoted_piece, char promotion_char)
{
    // Implementation details hidden
    return true; // Placeholder result
}

// Custom move parser
int parse_move_string(char* move_string)
{
    MoveCollection move_list[1];
    generate_moves_custom(move_list);

    int start = parse_square(move_string[0], move_string[1]);
    int target = parse_square(move_string[2], move_string[3]);

    for (int move_index = 0; move_index < move_list->count; move_index++)
    {
        int move = move_list->moves[move_index];

        if (start == get_move_source(move) && target == get_move_target(move))
        {
            int promoted_piece = get_move_promoted(move);

            if (promoted_piece)
            {
                char promotion_char = move_string[4];

                if (validate_promotion(promoted_piece, promotion_char))
                    return move;

                continue;
            }

            return move;
        }
    }

    return 0;
}

// Custom UCI command processor
void process_uci_command(char* command)
{
    // Advance pointer to where the relevant tokens start
    command += 9;

    // Pointer to current character in the command string
    char* current_char = command;

    // Process "startpos" UCI command
    if (strncmp(command, "startpos", 8) == 0)
        // Initialize board to the starting position
        parse_fen(start_position);

    // Process "fen" UCI command
    else
    {
        // Ensure "fen" command is present in the command string
        current_char = strstr(command, "fen");

        // If "fen" command is not found
        if (current_char == NULL)
            // Initialize board to the starting position
            parse_fen(start_position);

        // Found "fen" substring
        else
        {
            // Move pointer to where FEN string starts
            current_char += 4;

            // Initialize board from FEN string
            parse_fen(current_char);
        }
    }

    // Look for moves after position setup
    current_char = strstr(command, "moves");

    // If moves are provided
    if (current_char != NULL)
    {
        // Advance pointer to where moves start
        current_char += 6;

        // Process each move in the move string
        while (*current_char)
        {
            // Parse the next move
            int move = parse_move_string(current_char);

            // If no valid move is found
            if (move == 0)
                // Exit loop
                break;

            // Execute the move on the board
            make_move_custom(move, all_moves);

            // Move pointer to end of current move
            while (*current_char && *current_char != ' ')
                current_char++;

            // Move to the next move
            current_char++;
        }

        printf("%s\n", current_char);
    }
}

void initialize_pieces()
{
    magic_num(102);

    generation(side == 1);
}

int main()
{
    initialize_pieces();
    
    parse_fen(START_POSITION);
    display_board();

    run_performance_test(6);
    return 0;
}







