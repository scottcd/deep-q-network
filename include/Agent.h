#ifndef AGENT_H
#define AGENT_H

#include <string>
#include <ostream>
#include <vector>
#include <cmath>
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
 */
class Agent
{
    // TODO: add argument to save/load in constructor/deconstructor

public:
    /**
     * Generic Deep Q Agent Constructor
     *
     * Detailed description of the method.
     * This can span multiple lines.
     *
     * @param observationSpace Description of the parameter.
     * @param actionSpace Description of the parameter.
     * @param memorySize Description of the parameter.
     * @param epsilonStart Description of the parameter.
     * @param epsilonEnd Description of the parameter.
     * @param epsilonDecay Description of the parameter.
     * @param batchSize Description of the parameter.
     */
    Agent(int observationSpace, int actionSpace, int memorySize = 10000, float epsilonStart = 0.9,
          float epsilonEnd = 0.05, float epsilonDecay = 1000, int batchSize = 128);
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
    void learn();

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
    DQN policyNetwork;
    DQN targetNetwork;
    ReplayMemory memory;
    Environment *env;
    int actionsTaken;
    int batchSize;
    float epsilonStart;
    float epsilonEnd;
    float epsilonDecay;
};

#endif // AGENT_H