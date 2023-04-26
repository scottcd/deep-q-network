#include "Agent.h"
#include <torch/torch.h> // include the PyTorch header file

Agent::Agent(int observationSpace, int actionSpace, int memorySize, float epsilonStart,
             float epsilonEnd, float epsilonDecay, int batchSize)
    : policyNetwork(observationSpace, actionSpace),
      targetNetwork(observationSpace, actionSpace),
      memory(memorySize),
      epsilonStart(epsilonStart),
      epsilonEnd(epsilonEnd),
      epsilonDecay(epsilonDecay),
      batchSize(batchSize),
      actionsTaken(0) {}

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
        torch::Tensor output = policyNetwork.forward(state);

        // find the index of the element with the largest value in each row
        torch::Tensor max_indices = std::get<1>(output.max(1, true));

        // reshape the tensor to have shape (1, 1)
        torch::Tensor result = max_indices.view({1, 1});

        return result;
    }
    else
        return torch::randint(env->getNumActions(), {1, 1});
}

bool Agent::act(torch::Tensor &state)
{
    // select an action and step in the environment
    torch::Tensor action = selectAction(state);
    auto [observation, reward, terminated] = env->step(action.item<int>());

    // if terminated, set tensor to zeros, else set to observation
    torch::Tensor nextState;
    if (terminated)
    {
        nextState = torch::zeros({1, env->getNumActions()});
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

void Agent::learn()
{
    // if our memory is not large enough, don't optimize
    if (memory.size() < batchSize)
        return;

    // get a random sample of memory
    std::vector<Transition> transitions = memory.sample(batchSize);
    TransposedTransition transposedTransitions = TransposedTransition(transitions);

    // do some fancy work for nextStates

    torch::Tensor states = torch::cat(transposedTransitions.states);
    torch::Tensor actions = torch::cat(transposedTransitions.actions);
    torch::Tensor rewards = torch::cat(transposedTransitions.rewards);

}

void Agent::train(int numEpisodes)
{
    for (int i = 0; i < numEpisodes; i++)
    {
        torch::Tensor state = env->reset();

        while (1)
        {
            // act on the current state
            bool terminated = act(state);
            
            // learn how action impacted state and optimize
            learn();

            // Soft update the target network's weights
            // end soft update

            env->render();

            if (terminated)
                // episode finished
                break;
        }
    }
}