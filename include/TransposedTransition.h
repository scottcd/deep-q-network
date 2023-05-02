#ifndef TRANSPOSEDTRANSITION_H
#define TRANSPOSEDTRANSITION_H

#include <string>
#include <ostream>
#include <deque>
#include <random>
#include <algorithm>
#include <torch/torch.h>
#include "Transition.h"
/**
 * Transposed Transition
 *
 * Transpose Transition's cols and rows to
 * rows and cols respectively
 */
class TransposedTransition
{
public:
    /**
     * Transposed Transition Default Constructor
     */
    TransposedTransition() {}

    /**
     * Transposed Transition Constructor
     *
     * Transposes a given vector of Transitions
     * to vectors of states, rewards, next states,
     * and actions
     *
     * e.g.,
     * ([a, 1], [b, 2], [c, 3]) ->
     * ([a, b, c], [1, 2, 3])
     */
    TransposedTransition(std::vector<Transition> transitions)
        : states(), nextStates(), actions(), rewards()

    {
        // Transpose the batch of Transitions
        for (const auto &transition : transitions)
        {
            states.push_back(transition.state);
            nextStates.push_back(transition.nextState);
            actions.push_back(transition.action);
            rewards.push_back(transition.reward);
        }
    }

    // vector of states
    std::vector<torch::Tensor> states;
    // vector of next states
    std::vector<torch::Tensor> nextStates;
    // vector of actions
    std::vector<torch::Tensor> actions;
    // vector of rewards
    std::vector<torch::Tensor> rewards;
};

#endif // TRANSPOSEDTRANSITION_H