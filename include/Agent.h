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
    Agent(int observationSpace, int actionSpace, int memorySize, float epsilonStart = 0.9, float epsilonEnd = 0.05, float epsilonDecay = 1000)
        : policyNetwork(observationSpace, actionSpace),
          targetNetwork(observationSpace, actionSpace),
          memory(memorySize),
          epsilonStart(epsilonStart),
          epsilonEnd(epsilonEnd),
          epsilonDecay(epsilonDecay)
    {
        actionsTaken = 0;
    }
    /**
     * Select a space to play
     *
     * The agent selects a space to place their
     * O.
     *
     * @return an integer (0-8) corresponding to the
     * space to play.
     */
    torch::Tensor selectAction(torch::Tensor state)
    {
        // get random double
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        double sample = dis(gen);

        // calculate epsilon threshold
        double eps_threshold = epsilonEnd + (epsilonStart - epsilonEnd) * std::exp(-1.0 * actionsTaken / epsilonDecay);

        actionsTaken++;

        if (sample > eps_threshold)
        {
            torch::NoGradGuard no_grad;

            // get the output tensor from the policy net
            torch::Tensor output = policyNetwork.forward(state);

            // find the index of the element with the largest value in each row
            torch::Tensor max_indices = std::get<1>(output.max(1, true));

            // reshape the tensor to have shape (1, 1)
            torch::Tensor result = max_indices.view({1, 1});

            return result;
        }
        else
            return torch::randint(env->getNumActions(), {1, 1});
    }
    virtual bool act() = 0;
    virtual void learn() = 0;
    virtual void train() = 0;

protected:
    DQN policyNetwork;
    DQN targetNetwork;
    ReplayMemory memory;
    Environment *env;
    int actionsTaken;
    float epsilonStart;
    float epsilonEnd;
    float epsilonDecay;
};

#endif // AGENT_H