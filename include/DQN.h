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

class DQN  : public torch::nn::Module{
public:
    DQN(int numObservations, int numActions);
    torch::Tensor& forward(torch::Tensor& x);

    friend std::ostream& operator<<(std::ostream& os, const DQN& dqn) {
        os << "DQN(numObservations=" << dqn.numObservations << ", numActions=" << dqn.numActions <<  ")";
        return os;
    }

private:
    torch::nn::Linear layer1 = nullptr;
    torch::nn::Linear layer2 = nullptr;
    torch::nn::Linear layer3 = nullptr;
    int numObservations;
    int numActions;
};

#endif // DQN_H