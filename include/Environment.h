#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <string>
#include <ostream>
#include <vector>
#include <tuple>
#include <torch/torch.h>

/*
* Generic Environment to represent the 
* environment that the Agent perceives.
*/
class Environment
{
public:
    /**
     * Environment Constructor
     *
     * Generic Environment to be
     * overriden by a scenario-specific
     * environment.
     */
    Environment(int numObservations, int numActions) : observationSpace(numObservations), actionSpace(numActions)
    {
    }
    /**
     * Step through the environment with some action
     * 
     * @param action action to take
     * @return tuple for state, reward, and terminated
    */
    virtual std::tuple<torch::Tensor, torch::Tensor, bool> step(int action) = 0;
    
    /**
     * Reset the environment to its initial state
    */
    virtual torch::Tensor reset() = 0;
    
    /**
     * Display the environment
    */
    virtual void render() = 0;
    
    /**
     * Do any cleanup to close the environment
    */
    virtual void close() = 0;

    /**
     * Get the number of possible actions
    */
    int getNumActions() { return actionSpace.size(); }

    /**
     * Get the number of observations
    */
    int getNumObservations() { return observationSpace.size(); }

protected:
    // structure for each possible action
    std::vector<int> actionSpace;
    // structure for current state of the environment
    std::vector<double> observationSpace;
};

#endif // ENVIRONMENT_H