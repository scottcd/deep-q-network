#ifndef TRANSPOSEDTRANSITION_H
#define TRANSPOSEDTRANSITION_H

#include <string>
#include <ostream>
#include <deque>
#include <random>
#include <algorithm>
#include <torch/torch.h>
#include "Transition.h"

class TransposedTransition
{
public:
    TransposedTransition() {}
    TransposedTransition(std::vector<Transition> transitions)
        : states(), nextStates(), actions(), rewards()

    {
        // Transpose the batch of Transitions
        std::vector<std::vector<torch::Tensor>> transposedTensors;
        for (int i = 0; i < 4; i++)
        {
            std::vector<torch::Tensor> tensorBatch;
            for (const auto &transition : transitions)
            {
                tensorBatch.push_back(torch::transpose(transition.state, 0, 1));
                tensorBatch.push_back(torch::transpose(transition.nextState, 0, 1));
                tensorBatch.push_back(torch::transpose(transition.action, 0, 1));
                tensorBatch.push_back(torch::transpose(transition.reward, 0, 1));
            }
            transposedTensors.push_back(tensorBatch);
        }

        // Stack the transposed tensors and store them in the corresponding member variables
        for (const auto &tensorBatch : transposedTensors)
        {
            states.push_back(torch::stack(tensorBatch[0]));
            nextStates.push_back(torch::stack(tensorBatch[1]));
            actions.push_back(torch::stack(tensorBatch[2]));
            rewards.push_back(torch::stack(tensorBatch[3]));
        }
    }
    ~TransposedTransition() {}

    std::vector<torch::Tensor> states;
    std::vector<torch::Tensor> nextStates;
    std::vector<torch::Tensor> actions;
    std::vector<torch::Tensor> rewards;
};

#endif // TRANSPOSEDTRANSITION_H