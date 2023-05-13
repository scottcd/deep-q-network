from environment import Environment

class TicTacToeEnvironment(Environment):
    def __init__(self, action_space, observation_space):
        super().__init__(action_space, observation_space)
    
    # implment these
    def reset(self):
        pass

    def step(self):
        pass

    def close(self):
        pass

    def render(self):
        pass
