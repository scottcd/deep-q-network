#include "DQN.h"

DQNImpl::DQNImpl(int numObservations, int numActions) : torch::nn::Module(),
                                                numObservations(numObservations), numActions(numActions)
{
    layer1 = register_module("layer1", torch::nn::Linear(numObservations, 128));
    layer2 = register_module("layer2", torch::nn::Linear(128, 128));
    layer3 = register_module("layer3", torch::nn::Linear(128, numActions));
}

torch::Tensor &DQNImpl::forward(torch::Tensor &x)
{
    x = torch::relu(layer1(x));
    x = torch::relu(layer2(x));
    x = layer3(x);
    return x;
}

std::unordered_map<std::string, torch::Tensor> DQNImpl::stateDictionary() const
{
    std::unordered_map<std::string, torch::Tensor> dict;

    for (const auto &pair : named_parameters())
    {
        const std::string &name = pair.key();
        const torch::Tensor &tensor = pair.value();

        dict[name] = tensor.clone();
    }

    return dict;
}


void DQNImpl::loadStateDictionary(const std::unordered_map<std::string, torch::Tensor> &stateDictionary)
{
    auto params = named_parameters();
    for (auto &pair : stateDictionary)
    {
        const std::string &key = pair.first;
        const torch::Tensor &value = pair.second;

        auto *param_it = params.find(key);
        if (param_it != nullptr)
        {
            continue;
        }
        torch::NoGradGuard noGrad;
        param_it->copy_(value);
    }
}