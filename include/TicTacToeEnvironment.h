#ifndef TICTACTOE_ENVIRONMENT_H
#define TICTACTOE_ENVIRONMENT_H

#include "Environment.h"
#include <array>
#include <iostream>


class TicTacToeEnvironment : public Environment {
public:
    TicTacToeEnvironment();
    virtual ~TicTacToeEnvironment();
    virtual std::tuple<std::vector<double>, float, bool> step(int action) override;
    virtual std::vector<double> reset() override;
    virtual void render() override;
    virtual void close() override;

private:
    std::vector<double> board; // TicTacToe board
};

#endif // TICTACTOE_ENVIRONMENT_H
