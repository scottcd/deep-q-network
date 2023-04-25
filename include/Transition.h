#ifndef TRANSITION_H
#define TRANSITION_H

#include <string>
#include <ostream>
#include <deque>
#include <random>
#include <algorithm>
#include <torch/torch.h>


class Transition {
public:
    Transition() {}
    Transition(torch::Tensor state, torch::Tensor nextState, torch::Tensor action, torch::Tensor reward) 
    : state(state), nextState(nextState), action(action), reward(reward)
    {}
    ~Transition() {}

private:
    torch::Tensor state;
    torch::Tensor nextState;
    torch::Tensor action;
    torch::Tensor reward;
};

#endif // TRANSITION_H