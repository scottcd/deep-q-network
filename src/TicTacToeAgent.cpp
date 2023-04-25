#include "TicTacToeAgent.h"
#include <torch/torch.h> // include the PyTorch header file
#include <chrono> // for std::chrono::seconds
#include <thread> // for std::this_thread::sleep_for

TicTacToeAgent::TicTacToeAgent() : Agent(9, 9, 10000)
{
    // TODO: add argument to save/load in constructor/deconstructor
    env = new TicTacToeEnvironment();
}

TicTacToeAgent::~TicTacToeAgent()
{
    delete env;
}

bool TicTacToeAgent::act()
{
    return 1;
}

void TicTacToeAgent::learn()
{
}

void TicTacToeAgent::train()
{
    torch::Tensor state = env->reset();

    while (1)
    {
        torch::Tensor action = selectAction(state);
        std::tuple<torch::Tensor, torch::Tensor, bool> data = env->step(action.item<int>());
        auto [observation, reward, terminated] = data;

        torch::Tensor nextState;
        if (terminated)
        {
            nextState = torch::zeros({1, 9});
        }
        else
        {
            nextState = observation.to(torch::kFloat32).unsqueeze(0);
        }

        Transition transition = Transition(state, nextState, action, reward);
        memory.push(transition);
        state = nextState;

        learn(); // or, optimize

        // soft update of the target network's weights
        
        env->render();
        std::this_thread::sleep_for(std::chrono::seconds(1));

        if (terminated)
            break;
    }

}