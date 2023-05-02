#ifndef AGENT_H
#define AGENT_H

#include <string>
#include <ostream>
#include <vector>
#include <cmath>
#include <filesystem>
#include <tuple>
#include <torch/torch.h>
#include "DQN.h"
#include "ReplayMemory.h"
#include "Environment.h"
#include "Transition.h"
#include "TransposedTransition.h"

/**
 * Generic Deep Q Learning Agent
 *
 * Override Agent with a scenario-specific Agent. 
 * Derived classes must set an environment and 
 * filepaths for policy and target
 */
class Agent
{
public:
    /**
     * Generic Deep Q Agent Constructor
     *
     * Detailed description of the method.
     * This can span multiple lines.
     *
     * @param observationSpace Number of input observations
     * @param actionSpace Number of possible actions
     * @param memorySize Number of transitions to hold in Replay Memory
     * @param epsilonStart Value to start epsilon threshold
     * @param epsilonEnd Value to end epsilon threshold
     * @param epsilonDecay Rate at which we lower epsilon threshold
     * @param gamma Weight for future states in Q value calculation
     * @param tau Weight for target network's soft update; that is, how much we update target from policy
     * @param batchSize Number of transitions required before learning
     * @param learningRate Optimizer's learning rate
     */
    Agent(int observationSpace, int actionSpace, int memorySize = 10000, float epsilonStart = 0.9,
          float epsilonEnd = 0.05, float epsilonDecay = 1000, float gamma = 0.99, float tau = 0.005,
          int batchSize = 128, double learningRate = 1e-4);
    /**
     * Select an action
     *
     * Agent selects an action using an Annealing Epsilon Greedy
     * Function. Using a threshold, we either act randomly or optimally.
     * At the beginning, the agent is more likely to randomly
     * choose an action from the action space. As time progresses, we lower our
     * threshold, and the agent is more likely to choose the action
     * with the maximum expected return.
     *
     * @param state Description of the parameter.
     * @return return
     */
    torch::Tensor selectAction(torch::Tensor state);

    /**
     * Given some state, act on the environment.
     *
     * @param state current state
     */
    bool act(torch::Tensor &state);

    /**
     * Learn from the environment.
     *
     * If our memory is large enough, optimize our policy
     * using Adam's optimizer.
     */
    void learn(torch::optim::AdamW &optimizer);

    /**
     * Train on the environment.
     *
     * Run through the environment until termination
     * for the specified number of episodes.
     *
     *
     * @param numEpisodes number of episodes to train
     */
    void train(int numEpisodes);

protected:
    //  Deep Q Policy Network
    DQN policyNetwork;
    // Deep Q Target Network
    DQN targetNetwork;
    // Memory to keep track of state change and action/reward
    ReplayMemory memory;
    // Environment the agent observes and acts on
    Environment *env;
    // filepath for policy network
    std::string policyFilePath;
    // filepath for target network
    std::string targetFilePath;
    // Number of actions taken
    int actionsTaken;
    // Number of transitions required before learning
    int batchSize;
    // Value epsilon threshold starts on
    float epsilonStart;
    // Value epsilon threshold ends on
    float epsilonEnd;
    // Rate at which we lower epsilon threshold
    float epsilonDecay;
    // Weight for future states in Q value calculation
    float gamma;
    // Weight for target network's soft update
    float tau;
    // Optimizer's learning rate
    double learningRate;
};

#endif // AGENT_H