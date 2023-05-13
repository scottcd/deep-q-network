class Environment:
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space

    def reset(self):
        pass

    def step(self):
        pass

    def close(self):
        pass

    def render(self):
        pass