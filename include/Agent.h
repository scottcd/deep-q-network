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

class Agent
{
public:
    /**
     * Brief description of the method.
     *
     * Detailed description of the method.
     * This can span multiple lines.
     *
     * @param observationSpace Description of the parameter.
     * @return Description of the return value.
     */
    Agent(int observationSpace, int actionSpace, int memorySize=10000, float epsilonStart = 0.9,
          float epsilonEnd = 0.05, float epsilonDecay = 1000, int batchSize = 128);
    /**
     * Select a space to play
     *
     * The agent selects a space to place their
     * O.
     *
     * @return an integer (0-8) corresponding to the
     * space to play.
     */
    torch::Tensor selectAction(torch::Tensor state);
    bool act(torch::Tensor &state);
    void learn();
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