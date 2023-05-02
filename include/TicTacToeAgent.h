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
     * 
     * note: make sure to sent environment and filepaths!
     */
    TicTacToeAgent() : Agent(9, 9)
    {
        env = new TicTacToeEnvironment();
        policyFilePath="results/tic-tac-toe-policy.pt";
        targetFilePath="results/tic-tac-toe-target.pt";
    }
    /**
     * Tic-Tac-Toe Agent Deconstructor
     *
     * Deconstructor for the Tic-Tac-Toe
     * Agent. Nothing really fancy here.
     */
    ~TicTacToeAgent()
    {
        delete env;
    }
};

#endif // TICTACTOE_AGENT_H