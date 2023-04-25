#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <string>
#include <ostream>
#include <vector>
#include <tuple>
#include <torch/torch.h>

class Environment
{
public:
    Environment(int numObservations, int numActions) : observationSpace(numObservations), actionSpace(numActions)
    {
    }
    virtual std::tuple<torch::Tensor, torch::Tensor, bool> step(int action) = 0;
    virtual torch::Tensor reset() = 0;
    virtual void render() = 0;
    virtual void close() = 0;
    int getNumActions() { return actionSpace.size(); }

protected:
    // structure for each possible action
    std::vector<int> actionSpace;
    // structure for current state of the environment
    std::vector<double> observationSpace;
};

#endif // ENVIRONMENT_H