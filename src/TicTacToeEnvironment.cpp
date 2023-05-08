#include "TicTacToeEnvironment.h"

TicTacToeEnvironment::TicTacToeEnvironment(float illegalMoveReward, float legalNonEndingMoveReward,
                                           float winReward, float drawReward,
                                           float lossReward)
    : Environment(9, 9),
      wins(0),
      losses(0),
      draws(0),
      outcome(0),
      legalMoves(0),
      illegalMoves(0),
      totalInvalidMoves(0),
      totalMoves(0),
      illegalMoveReward(illegalMoveReward),
      legalNonEndingMoveReward(legalNonEndingMoveReward),
      winReward(winReward),
      drawReward(drawReward),
      lossReward(lossReward)
{
}

TicTacToeEnvironment::~TicTacToeEnvironment()
{
}

torch::Tensor TicTacToeEnvironment::reset()
{
    outcome = 0;
    illegalMoves = 0;
    legalMoves = 0;

    std::cout << "Resetting board.\n";
    std::iota(actionSpace.begin(), actionSpace.end(), 0);
    fill(observationSpace.begin(), observationSpace.end(), 0);

    return torch::from_blob(observationSpace.data(), {1, static_cast<long int>(observationSpace.size())}, torch::kFloat32);
}

std::tuple<torch::Tensor, torch::Tensor, bool> TicTacToeEnvironment::step(int action)
{
    // play move if space is empty.
    if (observationSpace[action] == 0)
    {
        // env-specific statistics
        legalMoves++;
        totalMoves++;

        observationSpace[action] = 1;
    }
    else
    {
        // env-specific statistics
        illegalMoves++;
        totalInvalidMoves++;

        return std::make_tuple(torch::from_blob(observationSpace.data(), {static_cast<long int>(observationSpace.size())}, torch::kFloat32),
                               torch::from_blob(&illegalMoveReward, {1}, torch::kFloat32).clone(), false);
    }
    // check if we won
    bool won = checkWin(1);
    bool draw = checkDraw();
    if (won == 1)
    {
        wins++;
        outcome = 1;
        return std::make_tuple(torch::from_blob(observationSpace.data(), {static_cast<long int>(observationSpace.size())}, torch::kFloat32),
                               torch::from_blob(&winReward, {1}, torch::kFloat32).clone(), true);
    }
    if (draw == 1)
    {
        draws++;
        outcome = 0;
        return std::make_tuple(torch::from_blob(observationSpace.data(), {static_cast<long int>(observationSpace.size())}, torch::kFloat32),
                               torch::from_blob(&drawReward, {1}, torch::kFloat32).clone(), true);
    }
    // opponent play move
    int i = opponentSelectAction();
    observationSpace[i] = -1;

    // check if they won
    bool lost = checkWin(-1);
    if (lost == 1)
    {
        losses++;
        outcome = -1;
        return std::make_tuple(torch::from_blob(observationSpace.data(), {static_cast<long int>(observationSpace.size())}, torch::kFloat32),
                               torch::from_blob(&lossReward, {1}, torch::kFloat32).clone(), true);
    }
    if (draw == 1)
    {
        draws++;
        outcome = 0;
        return std::make_tuple(torch::from_blob(observationSpace.data(), {static_cast<long int>(observationSpace.size())}, torch::kFloat32),
                               torch::from_blob(&drawReward, {1}, torch::kFloat32).clone(), true);
    }

    // could optimize here by adding the appropriate reward for blocking a move, getting two in a row, etc.

    return std::make_tuple(torch::from_blob(observationSpace.data(), {static_cast<long int>(observationSpace.size())}, torch::kFloat32),
                           torch::from_blob(&legalNonEndingMoveReward, {1}, torch::kFloat32).clone(), false);
}

int TicTacToeEnvironment::opponentSelectAction()
{
    if (observationSpace[4] == 0) {
        return 4;
    }
    // Play a corner if available
    if (observationSpace[0] == 0) {
        return 0;
    }
    if (observationSpace[2] == 0) {
        return 2;
    }
    if (observationSpace[6] == 0) {
        return 6;
    }
    if (observationSpace[8] == 0) {
        return 8;
    }
    // Play an edge if available
    if (observationSpace[1] == 0) {
        return 1;
    }
    if (observationSpace[3] == 0) {
        return 3;
    }
    if (observationSpace[5] == 0) {
        return 5;
    }
    if (observationSpace[7] == 0) {
        return 7;
    }
    return -1;
}

bool TicTacToeEnvironment::checkDraw()
{
    for (int i = 0; i < observationSpace.size(); i++)
    {
        if (observationSpace[i] == 0)
            return false;
    }
    return true;
}

bool TicTacToeEnvironment::checkWin(double value)
{
    // Check horizontal lines
    for (int i = 0; i < 3; i++)
    {
        if (observationSpace[i * 3] == observationSpace[i * 3 + 1] && observationSpace[i * 3 + 1] == observationSpace[i * 3 + 2] && observationSpace[i * 3] == value)
        {
            return true;
        }
    }

    // Check vertical lines
    for (int i = 0; i < 3; i++)
    {
        if (observationSpace[i] == observationSpace[i + 3] && observationSpace[i + 3] == observationSpace[i + 6] && observationSpace[i] == value)
        {
            return true;
        }
    }

    // Check diagonal lines
    if (observationSpace[0] == observationSpace[4] && observationSpace[4] == observationSpace[8] && observationSpace[0] == value)
    {
        return true;
    }
    if (observationSpace[2] == observationSpace[4] && observationSpace[4] == observationSpace[6] && observationSpace[2] == value)
    {
        return true;
    }

    // No winner
    return false;
}

void TicTacToeEnvironment::render()
{
    system("clear");

    std::cout << "Playing as O\n"
              << "\n";

    // convert int values to X's and O's
    auto to_char = [](double val)
    {
        switch ((int)val)
        {
        case 1:
            return 'O';
        case -1:
            return 'X';
        default:
            return ' ';
        }
    };

    std::cout << "\n";

    std::cout << " " << to_char(observationSpace[0]) << " | " << to_char(observationSpace[1]) << " | " << to_char(observationSpace[2]) << " \n";
    std::cout << "---+---+---\n";
    std::cout << " " << to_char(observationSpace[3]) << " | " << to_char(observationSpace[4]) << " | " << to_char(observationSpace[5]) << " \n";
    std::cout << "---+---+---\n";
    std::cout << " " << to_char(observationSpace[6]) << " | " << to_char(observationSpace[7]) << " | " << to_char(observationSpace[8]) << " \n";
    std::cout << "\n";

    float averageInvalidMoves = totalInvalidMoves / (wins + losses + draws + 1);
    float averageMovesPerGame = totalMoves / (wins + losses + draws + 1);

    std::cout << "Legal this Game:\t\t" << legalMoves << "\n";
    std::cout << "Illegal Moves this Game:\t" << illegalMoves << "\n\n";

    std::cout << "Total Wins:\t\t\t" << wins << "\n";
    std::cout << "Total Losses:\t\t\t" << losses << "\n";
    std::cout << "Total Draws:\t\t\t" << draws << "\n";
    std::cout << "Average Invalid Moves:\t\t" << averageInvalidMoves << "\n";
    std::cout << "Average Moves per Game:\t\t" << averageMovesPerGame << "\n";

    std::cout << "\n";

    // std::this_thread::sleep_for(std::chrono::milliseconds(300));
}

void TicTacToeEnvironment::close()
{
}