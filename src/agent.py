import math
import random

import torch
from dqn import DQN
from memory import ReplayMemory


class Agent():
    def __init__(self, number_observations, number_actions, memory_size=10000, batch_size=128,
                 epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=1000,
                 tau=0.005, gamma=0.99, learning_rate=0.1,
                 policy_output=None, target_output=None, statistics_output=None):
        self.policy_network = DQN(number_observations, number_actions)
        self.target_network = DQN(number_observations, number_actions)
        self.memory = ReplayMemory(memory_size)
        self.batch_size = batch_size
        self.policy_output = policy_output
        self.target_output = target_output
        self.statistics_output = statistics_output
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.steps_taken = 0

    def load_model(self):
        return

    def save_model(self):
        return

    def save_statistics(self):
        return

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.steps_taken / self.epsilon_decay)
        self.steps_taken += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_network(state).max(1)[1].view(1, 1)
        else:
            # return torch.tensor([[env.action_space.sample()]], dtype=torch.long)
            return

    def act(self):
        return

    def learn(self):
        return

    def train(self):
        return
