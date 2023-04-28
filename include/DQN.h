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
class DQN  : public torch::nn::Module{
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
    DQN(int numObservations, int numActions);

    /**
     * Forward propagate through the neural network
     */
    torch::Tensor& forward(torch::Tensor& x);

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

#endif // DQN_H