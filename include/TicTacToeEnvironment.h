#ifndef TICTACTOE_ENVIRONMENT_H
#define TICTACTOE_ENVIRONMENT_H

#include "Environment.h"
#include <array>
#include <iostream>
#include <numeric>
#include <random>

class TicTacToeEnvironment : public Environment
{
public:
    TicTacToeEnvironment();
    virtual ~TicTacToeEnvironment();
    virtual std::tuple<torch::Tensor, torch::Tensor, bool> step(int action) override;
    virtual torch::Tensor reset() override;
    virtual void render() override;
    virtual void close() override;
    bool checkWin(double value);
    bool checkDraw();
    int opponentSelectAction();
};

#endif // TICTACTOE_ENVIRONMENT_H
