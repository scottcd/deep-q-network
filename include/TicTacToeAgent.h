#ifndef TICTACTOE_AGENT_H
#define TICTACTOE_AGENT_H

#include <string>
#include <ostream>
#include <vector>
#include <tuple>
#include <torch/torch.h>
#include "DQN.h"
#include "ReplayMemory.h"
#include "Environment.h"
#include "TicTacToeEnvironment.h"
#include "Agent.h"
#include "Transition.h"

/*
* Deep Q Learning Agent for Tic-Tac-Toe
*/
class TicTacToeAgent : public Agent
{
public:
    /**
     * Tic-Tac-Toe Agent Constructor
     *
     * Agent that learns to play Tic-Tac-Toe 
     * over time. The agent selects an action, 
     * acts in the environment, and learns the
     * expected cumulative reward of each action
     * for a given state.
     */
    TicTacToeAgent();

    /**
     * Tic-Tac-Toe Agent Deconstructor
     *
     * Deconstructor for the Tic-Tac-Toe
     * Agent. Nothing really fancy here.
     */
    virtual ~TicTacToeAgent();

    /**
     * Act in the Tic-Tac-Toe Environment
     *
     * The agent selects an action and steps in the environment.
     *
     * @return a boolearn whether someone won the game
     */
    virtual bool act() override;

    /**
     * Learn from the updated environment.
     *
     * The agent learns how their action
     * updates state.
     */
    virtual void learn() override;

    /**
     * Train the agent.
     *
     * The agent trains on the environment 
     * to move optimally in any given state.
     */
    virtual void train() override;
};

#endif // TICTACTOE_AGENT_H