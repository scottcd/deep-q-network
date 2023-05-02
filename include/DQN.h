#ifndef DQN_H
#define DQN_H

#include <string>
#include <ostream>
#include <deque>
#include <random>
#include <algorithm>
#include <torch/torch.h>

using namespace std;
using namespace torch;

/**
 * Deep Q Neural Network
 *
 * Derived from PyTorch's neural
 * network module.
 */
class DQNImpl : public torch::nn::Module
{
public:
    /**
     * Deep Q Neural Network Constructor
     *
     * Constructor to initialize a Deep Q
     * Neural Network
     *
     * @param numObservations Number of observations (NN input)
     * @param numActions Number of possible actions (NN output)
     */
    DQNImpl(int numObservations, int numActions);

    /**
     * Forward propagate through the neural network
     *
     * @param x
     * @return x
     */
    torch::Tensor &forward(torch::Tensor &x);

    /**
     * get named parameters
     * 
     * @return a dictionary of names, values for each named parameter
    */
    std::unordered_map<std::string, torch::Tensor> stateDictionary() const;

    /**
     * update named parameters with a state dictionary
     * 
     * @param stateDictionary state dictionary to update our named_parameters with
    */
    void loadStateDictionary(const std::unordered_map<std::string, torch::Tensor> &stateDictionary);

private:
    // Neural Network input layer
    torch::nn::Linear layer1 = nullptr;
    // Neural Network hidden layer
    torch::nn::Linear layer2 = nullptr;
    // Neural Network output layer
    torch::nn::Linear layer3 = nullptr;
    // number of observations
    int numObservations;
    // number of possible actions
    int numActions;
};
TORCH_MODULE(DQN); // creates module holder for NetImpl

#endif // DQN_H