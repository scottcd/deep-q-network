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
        updateStatsParameters();
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

    /**
     * Set stats parameters to parameters
     * we want to record (output to csv).
     */
    void updateStatsParameters() override
    {
        statsParameters["Outcome"] = to_string(dynamic_cast<TicTacToeEnvironment *>(env)->outcome);
        statsParameters["LegalMoves"] = to_string(dynamic_cast<TicTacToeEnvironment *>(env)->legalMoves);
        statsParameters["IllegalMoves"] = to_string(dynamic_cast<TicTacToeEnvironment *>(env)->illegalMoves);
        statsParameters["IllegalMoveReward"] = to_string(dynamic_cast<TicTacToeEnvironment *>(env)->illegalMoveReward);
        statsParameters["LegalNonEndingMoveReward"] = to_string(dynamic_cast<TicTacToeEnvironment *>(env)->legalNonEndingMoveReward);
        statsParameters["WinReward"] = to_string(dynamic_cast<TicTacToeEnvironment *>(env)->winReward);
        statsParameters["LossReward"] = to_string(dynamic_cast<TicTacToeEnvironment *>(env)->lossReward);
        statsParameters["DrawReward"] = to_string(dynamic_cast<TicTacToeEnvironment *>(env)->drawReward);
        statsParameters["EpsilonStart"] = to_string(epsilonStart);
        statsParameters["EpsilonEnd"] = to_string(epsilonEnd);
        statsParameters["EpsilonDecay"] = to_string(epsilonDecay);
        statsParameters["BatchSize"] = to_string(batchSize);
        statsParameters["Gamma"] = to_string(gamma);
        statsParameters["Tau"] = to_string(tau);
        statsParameters["LearningRate"] = to_string(learningRate);
        statsParameters["CleanStart"] = to_string(cleanStart);
        statsParameters["NumEpisodes"] = to_string(numEpisodes);
    }

    /**
     * Setters for command line arguments
     */
    void setNumberEpisodes(int value)
    {
        numEpisodes = value;
    }
    void setStatsFilePath(std::string value)
    {
        statsFilePath = value;
    }
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