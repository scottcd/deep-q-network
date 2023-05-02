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
     * @param numEpisodes Number of episodes to run
     * @param cleanStart Whether to load model from file or start anew
     * @param policyFilePath File path for policy neural network
     * @param targetFilePath File path for target neural network
     */
    TicTacToeAgent(int numEpisodes = -1, bool cleanStart = false,
                   std::string policyFilePath = "out/tic-tac-toe-policy.pt",
                   std::string targetFilePath = "out/tic-tac-toe-target.pt")
        : Agent(9, 9)
    {
        env = new TicTacToeEnvironment();
        this->policyFilePath = policyFilePath;
        this->targetFilePath = targetFilePath;
        this->cleanStart = cleanStart;
        if (numEpisodes > 0)
        {
            this->numEpisodes = numEpisodes;
        }
        statsFilePath = "out/tic-tac-toe-results.csv";
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


    void updateStatsParameters() override 
    {
        statsParameters = {{to_string(dynamic_cast<TicTacToeEnvironment *>(env)->outcome),
                            to_string(dynamic_cast<TicTacToeEnvironment *>(env)->legalMoves),
                            to_string(dynamic_cast<TicTacToeEnvironment *>(env)->illegalMoves),
                            to_string(dynamic_cast<TicTacToeEnvironment *>(env)->illegalMoveReward),
                            to_string(dynamic_cast<TicTacToeEnvironment *>(env)->legalNonEndingMoveReward),
                            to_string(dynamic_cast<TicTacToeEnvironment *>(env)->winReward),
                            to_string(dynamic_cast<TicTacToeEnvironment *>(env)->lossReward),
                            to_string(dynamic_cast<TicTacToeEnvironment *>(env)->drawReward),
                            to_string(epsilonStart), to_string(epsilonEnd), to_string(epsilonDecay),
                            to_string(batchSize), to_string(gamma), to_string(tau),
                            to_string(learningRate), to_string(cleanStart), to_string(numEpisodes)}};
    }


    /**
     * Setters for command line arguments
     */
    void setNumberEpisodes(int value)
    {
        numEpisodes = value;
    }
    /**
     * Setters for command line arguments
     */
    void setCleanStart(bool value)
    {
        cleanStart = value;
    }
    void setIllegalMoveReward(float value)
    {
        dynamic_cast<TicTacToeEnvironment *>(env)->illegalMoveReward = value;
    }
    void setLegalMoveReward(float value)
    {
        dynamic_cast<TicTacToeEnvironment *>(env)->legalNonEndingMoveReward = value;
    }
    void setWinReward(float value)
    {
        dynamic_cast<TicTacToeEnvironment *>(env)->winReward = value;
    }
    void setLossReward(float value)
    {
        dynamic_cast<TicTacToeEnvironment *>(env)->lossReward = value;
    }
    void setDrawReward(float value)
    {
        dynamic_cast<TicTacToeEnvironment *>(env)->drawReward = value;
    }
    void setEpsilonStart(float value)
    {
        epsilonStart = value;
    }
    void setEpsilonEnd(float value)
    {
        epsilonEnd = value;
    }
    void setEpsilonDecay(float value)
    {
        epsilonDecay = value;
    }
    void setBatchSize(int value)
    {
        batchSize = value;
    }
    void setGamma(float value)
    {
        gamma = value;
    }
    void setTau(float value)
    {
        tau = value;
    }
    void setLearningRate(double value)
    {
        learningRate = value;
    }
};

#endif // TICTACTOE_AGENT_H