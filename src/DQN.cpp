#include "DQN.h"

DQN::DQN(int numObservations, int numActions) : 
numObservations(numObservations), numActions(numActions) 
{
    torch::nn::Module();
    layer1 = torch::nn::Linear(numObservations, 128);
    layer2 = torch::nn::Linear(128, 128);
    layer3 = torch::nn::Linear(128, numActions);
}

torch::Tensor& DQN::forward(torch::Tensor& x) {
    x = torch::relu(layer1(x));
    x = torch::relu(layer2(x));
    x = layer3(x);
    return x;
}