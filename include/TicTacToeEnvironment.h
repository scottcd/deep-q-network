#ifndef TICTACTOE_ENVIRONMENT_H
#define TICTACTOE_ENVIRONMENT_H

#include "Environment.h"
#include <array>
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <random>
#include <thread> 
#include <chrono> 

/**
 * Tic-Tac-Toe Environment
 * 
 * A 9x9 board where players can place
 * an X or O until a player gets 3
 * in-a-row or the board fills up.
*/
class TicTacToeEnvironment : public Environment
{
public:
    /**
     * Tic-Tac-Toe Environment Constructor
    */
    TicTacToeEnvironment();

    /**
     * Tic-Tac-Toe Environment Destructor
    */
    virtual ~TicTacToeEnvironment();
    
    /**
     * Step in the environment
     * 
    */
    virtual std::tuple<torch::Tensor, torch::Tensor, bool> step(int action) override;
    
    /**
     * Reset the environment to its initial state
     * 
     * @return A tensor representing the environment 
    */
    virtual torch::Tensor reset() override;
    
    /**
     * Render the Tic-Tac-Toe board
    */
    virtual void render() override;

    /**
     * Close the Tic-Tac-Toe board
    */
    virtual void close() override;

    /**
     * Check if the game has ended in a win for the agent
     * 
     * @return bool if win occured
    */
    bool checkWin(double value);
    
    /**
     * Check if the game has ended in a draw
     * 
     * @return bool if draw occured
    */
    bool checkDraw();

    /**
     * Have the opponent select a position to move to
     * 
     * @return position for the opponent to move
    */
    int opponentSelectAction();
};

#endif // TICTACTOE_ENVIRONMENT_H
