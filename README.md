# Bitboard based NNUE
## Introduction
### Bitboards (64 int)
Bitboards are a specific representation of chess boards using 64 bit integers. Bit 1 at the position i out of 64 indicates that there is a piece at row i%8 and column (i//8)+1. The 64 bit integer is a compact representation of a board. AND, OR, XOR and other bit operations can be used to quickly perform bit operations between bitboards. 
For eg. Bitboard representation for possible rook attacks for white = Bitboard for all black pieces AND Bitboard for all white rook moves. This will simply be the AND of two 64 bit integers.
![bitboard-chess](https://github.com/user-attachments/assets/07b66da7-5f2b-4b8f-b98f-e7032f6ed6cf)
### NNUE
The NNUE architecture is a shallow 4 layered fully connected neural network. Chess positions for input are provided in the form of a vector representing that position. Index i of the vector is represented by the tuple (King position, piece position). Total king positions = 64 x Total piece positions  = 64 * 64 (11) = 41024 x 1 vector. This input vector is sparse and barely changes from one position to the next. This allows for matrix multiplications to not happen at each iteration unless necessary and is computationally efficient. The training is done through non-NNUE Stockfish engine evaluations.
![1024px-StockfishNNUELayers](https://github.com/user-attachments/assets/88494c27-851d-4ab0-8a1f-a6b420dd1aa7)
### Libraries
Pytorch is used for creating the neural networks and evaluations.
### Actual Engine
An alpha-beta minmax tree is used for evaluating positions 6 to 7 moves forwards to make an optimal move given that your opponent plays the optimal move for the prior  6-7 moves. A move shortlister is also used to determine which moves at each level of the tree are considered for evaluation.
### Resources
The chessprogramming.wiki is a good resources to learn about most chess programming concepts. https://www.youtube.com/@chessprogramming591 is also a good resource for creating bitbaord based chess engines and I will loosely follow him for my based engine architecture used for my nnue evaluation.
