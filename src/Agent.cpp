#include "Agent.h"

Agent::Agent(int observationSpace, int actionSpace, int memorySize, float epsilonStart,
             float epsilonEnd, float epsilonDecay, float gamma, float tau, int batchSize,
             double learningRate)
    : policyNetwork(observationSpace, actionSpace),
      targetNetwork(observationSpace, actionSpace),
      memory(memorySize),
      epsilonStart(epsilonStart),
      epsilonEnd(epsilonEnd),
      epsilonDecay(epsilonDecay),
      gamma(gamma),
      tau(tau),
      batchSize(batchSize),
      learningRate(learningRate),
      actionsTaken(0),
      cleanStart(false),
      numEpisodes(10)

{
}

void Agent::saveStatistics(const std::map<std::string, std::string> &data)
{
    if (statsFilePath.empty())
    {
        std::cout << "No write file specified. Statistics will not be written to a file." << std::endl;
        return;
    }
    std::ifstream file(statsFilePath);
    if (!file)
    {
        // create file
        std::ofstream newFile(statsFilePath);
        if (!newFile)
        {
            std::cerr << "Error creating file: " << statsFilePath << std::endl;
            return;
        }
        bool isFirstKey = true;
        for (const auto &entry : data)
        {
            if (!isFirstKey)
                newFile << ",";
            isFirstKey = false;
            newFile << entry.first;
        }
        newFile << std::endl;
    }

    // append file
    std::ofstream appendFile(statsFilePath, std::ios::app);
    if (!appendFile)
    {
        std::cerr << "Error opening file for appending: " << statsFilePath << std::endl;
        return;
    }
    bool isFirstValue = true;
    for (const auto &entry : data)
    {
        if (!isFirstValue)
            appendFile << ",";
        isFirstValue = false;
        appendFile << entry.second; 
    }
    appendFile << std::endl;
}

void Agent::loadModel()
{
    if (policyFilePath.empty() or targetFilePath.empty())
    {
        std::cout << "No model file specified. Will not load model." << std::endl;
        return;
    }
    if (std::filesystem::exists(policyFilePath))
    {
        torch::load(policyNetwork, policyFilePath);
    }
    if (std::filesystem::exists(targetFilePath))
    {
        torch::load(targetNetwork, targetFilePath);
    }
}

void Agent::saveModel()
{
    if (policyFilePath.empty() or targetFilePath.empty())
    {
        std::cout << "No model file specified. Will not save model." << std::endl;
        return;
    }
    torch::save(policyNetwork, policyFilePath);
    torch::save(targetNetwork, targetFilePath);
}

torch::Tensor Agent::selectAction(torch::Tensor state)
{
    // get random double
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    double sample = dis(gen);

    // calculate epsilon threshold
    double eps_threshold = epsilonEnd + (epsilonStart - epsilonEnd) * std::exp(-1.0 * actionsTaken / epsilonDecay);

    actionsTaken++;

    if (sample > eps_threshold)
    {
        torch::NoGradGuard no_grad;

        // get the output tensor from the policy net
        torch::Tensor output = policyNetwork(state);

        // find the index of the element with the largest value in each row
        torch::Tensor max_indices = std::get<1>(output.flatten().max(0));

        // reshape the tensor to have shape (1, 1)
        torch::Tensor result = max_indices.view({1, 1});
        return result;
    }
    else
        return torch::randint(env->getNumActions(), {1, 1});
}

bool Agent::act(torch::Tensor &state)
{
    torch::Tensor action = selectAction(state);
    auto [observation, reward, terminated] = env->step(action.item<int>());

    // if terminated, set tensor to zeros, else set to observation
    torch::Tensor nextState;
    if (terminated)
    {
        nextState = torch::empty({env->getNumObservations()}).unsqueeze(0);
    }
    else
    {
        nextState = observation.to(torch::kFloat32).unsqueeze(0);
    }

    // Create transition and push to memory
    Transition transition = Transition(state, nextState, action, reward);

    memory.push(transition);

    state = nextState;

    return terminated;
}

void Agent::learn(torch::optim::AdamW &optimizer)
{
    // if our memory is not large enough, don't optimize
    if (memory.size() < batchSize)
        return;
    // get a random sample of memory
    std::vector<Transition> transitions = memory.sample(batchSize);
    TransposedTransition transposedTransitions = TransposedTransition(transitions);

    // stuff
    std::vector<torch::Tensor> resultVector;
    std::vector<torch::Tensor> maskVector;

    for (const auto &tensor : transposedTransitions.nextStates)
    {
        // if tensor is empty (final state)
        if (torch::equal(tensor, torch::empty({9}).unsqueeze(0)))
        {
            resultVector.push_back(tensor);
            maskVector.push_back(torch::zeros({1}, torch::kBool));
        }
        else
        {
            resultVector.push_back(tensor);
            maskVector.push_back(torch::ones({1}, torch::kBool));
        }
    }

    torch::Tensor nonFinalMask = torch::cat(maskVector);
    torch::Tensor nonFinalNextStates = torch::cat(resultVector);

    torch::Tensor states = torch::cat(transposedTransitions.states);
    torch::Tensor actions = torch::cat(transposedTransitions.actions);
    torch::Tensor rewards = torch::cat(transposedTransitions.rewards);

    // compute Q(s_t, a)
    torch::Tensor stateActionValues = policyNetwork(states).gather(1, actions);

    // Compute V(s_{t+1}) for all next states
    torch::Tensor nextStateValues = torch::zeros(batchSize);
    {
        // Forward non-final states and get the max value for each
        torch::NoGradGuard no_grad;
        torch::Tensor nonZeroIndices = nonFinalMask.nonzero().squeeze();
        torch::Tensor targetNetworkOutputNonFinal = targetNetwork(nonFinalNextStates);
        torch::Tensor maxValuesNonFinal = std::get<0>(targetNetworkOutputNonFinal.index({nonZeroIndices}).max(1));

        nextStateValues.index_put_({nonZeroIndices},
                                   maxValuesNonFinal);
    }
    // compute the expected Q values
    torch::Tensor expectedStateActionValues = (nextStateValues * gamma) + rewards;

    // compute Huber loss
    torch::nn::SmoothL1Loss criterion;
    torch::Tensor loss = criterion(stateActionValues, expectedStateActionValues.unsqueeze(1));

    // optimize the model
    optimizer.zero_grad();

    loss.backward();

    // in-place gradient clipping
    torch::nn::utils::clip_grad_value_(policyNetwork->parameters(), 100);
    optimizer.step();
}

void Agent::train()
{
    // create optimizer for learn()
    torch::optim::AdamW optimizer = torch::optim::AdamW(policyNetwork->parameters(),
                                                        torch::optim::AdamWOptions(learningRate).amsgrad(true));
    if (cleanStart == false)
    {
        loadModel();
    }

    // train for n episodes
    for (int i = 0; i < numEpisodes; i++)
    {
        torch::Tensor state = env->reset();
        while (1)
        {
            // act on the current state
            bool terminated = act(state);

            // learn how action impacted state and optimize
            learn(optimizer);

            // Soft update the target network's weights
            auto targetNetworkStateDictionary = targetNetwork->stateDictionary();
            auto policyNetworkStateDictionary = policyNetwork->stateDictionary();

            for (auto &pair : policyNetworkStateDictionary)
            {
                const std::string &key = pair.first;
                const torch::Tensor &policyTensor = pair.second;
                torch::Tensor &targetTensor = targetNetworkStateDictionary[key];

                targetTensor = (policyTensor * tau) + (targetTensor * (1 - tau));
            }

            targetNetwork->loadStateDictionary(targetNetworkStateDictionary);

            env->render();

            if (terminated)
            {
                // episode finished
                // convert outcomes value to win/loss/draw
                auto to_string = [](int val) -> std::string
                {
                    switch (static_cast<int>(val))
                    {
                    case 1:
                        return "Win";
                    case -1:
                        return "Loss";
                    default:
                        return "Draw";
                    }
                };
                // outcome, # legal moves, # illegal moves,
                if (statsParameters.size() != 0)
                {
                    updateStatsParameters();
                    saveStatistics(statsParameters);
                }

                break;
            }
        }
    }
    saveModel();
}