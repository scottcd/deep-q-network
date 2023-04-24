#include "Environment.h"
#include "TicTacToeEnvironment.h"

TicTacToeEnvironment::TicTacToeEnvironment() : Environment(9, 9),  board(9, 0)
{
}

TicTacToeEnvironment::~TicTacToeEnvironment()
{}

std::vector<double> TicTacToeEnvironment::reset()
{
    return board;
}

std::tuple<std::vector<double>, float, bool> TicTacToeEnvironment::step(int action)
{
        std::vector<double> board_vec(board.begin(), board.end());
        return std::make_tuple(board_vec, 0.0f, false);
}

void TicTacToeEnvironment::render()
{
    std::cout << " " << board[0] << " | " << board[1] << " | " << board[2] << " \n";
    std::cout << "---+---+---\n";
    std::cout << " " << board[3] << " | " << board[4] << " | " << board[5] << " \n";
    std::cout << "---+---+---\n";
    std::cout << " " << board[6] << " | " << board[7] << " | " << board[8] << " \n";
    std::cout << "\n";
}

void TicTacToeEnvironment::close()
{
}