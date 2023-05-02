#include "TicTacToeEnvironment.h"

TicTacToeEnvironment::TicTacToeEnvironment() : Environment(9, 9)
{
}

TicTacToeEnvironment::~TicTacToeEnvironment()
{
}

torch::Tensor TicTacToeEnvironment::reset()
{
    std::cout << "Resetting board.\n";
    std::iota(actionSpace.begin(), actionSpace.end(), 0);
    fill(observationSpace.begin(), observationSpace.end(), 0);

    return torch::from_blob(observationSpace.data(), {1, static_cast<long int>(observationSpace.size())}, torch::kFloat32);
}

std::tuple<torch::Tensor, torch::Tensor, bool> TicTacToeEnvironment::step(int action)
{
    // play move if space is empty.
    if (observationSpace[action] == 0)
        observationSpace[action] = 1;
    // else, move is invalid. Reward: -1
    else
    {
        float reward = -1.0f;
        return std::make_tuple(torch::from_blob(observationSpace.data(), {static_cast<long int>(observationSpace.size())}, torch::kFloat32),
                               torch::from_blob(&reward, {1}, torch::kFloat32).clone(), false);
    }
    // check if we won
    bool won = checkWin(1);
    bool draw = checkDraw();
    if (won == 1)
    {
        float reward = 100.0f;
        return std::make_tuple(torch::from_blob(observationSpace.data(), {static_cast<long int>(observationSpace.size())}, torch::kFloat32),
                               torch::from_blob(&reward, {1}, torch::kFloat32).clone(), true);
    }
    if (draw == 1)
    {
        float reward = 0.0f;
        return std::make_tuple(torch::from_blob(observationSpace.data(), {static_cast<long int>(observationSpace.size())}, torch::kFloat32),
                               torch::from_blob(&reward, {1}, torch::kFloat32).clone(), true);
    }
    // opponent play move
    int i = opponentSelectAction();
    observationSpace[i] = -1;

    // check if they won
    bool lost = checkWin(-1);
    if (lost == 1)
    {
        float reward = -100.0f;
        return std::make_tuple(torch::from_blob(observationSpace.data(), {static_cast<long int>(observationSpace.size())}, torch::kFloat32),
                               torch::from_blob(&reward, {1}, torch::kFloat32).clone(), true);
    }
    if (draw == 1)
    {
        float reward = 0.0f;
        return std::make_tuple(torch::from_blob(observationSpace.data(), {static_cast<long int>(observationSpace.size())}, torch::kFloat32),
                               torch::from_blob(&reward, {1}, torch::kFloat32).clone(), true);
    }

    // could optimize here by adding the appropriate reward for blocking a move, getting two in a row, etc.

    float reward = 1.0f;
    return std::make_tuple(torch::from_blob(observationSpace.data(), {static_cast<long int>(observationSpace.size())}, torch::kFloat32),
                           torch::from_blob(&reward, {1}, torch::kFloat32).clone(), false);
}

int TicTacToeEnvironment::opponentSelectAction()
{
    // Generate a random index in the vector that has the value 0.0
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, observationSpace.size() - 1);
    int index = dis(gen);
    while (observationSpace[index] != 0.0)
    {
        index = dis(gen);
    }
    return index;
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
    // std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // wins, losses, draws, avg repeat moves, avg moves to end
}

void TicTacToeEnvironment::close()
{
}