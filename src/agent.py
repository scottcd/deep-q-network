import math
import random
import io
import os
import re

import torch
import torch.nn as nn
import torch.optim as optim

from dqn import DQN
from memory import ReplayMemory
from ttt_environment import TicTacToeEnvironment
from transition import Transition
from device_manager import DeviceManager


class Agent():
    def __init__(self, n_observations, n_actions, number_episodes=1,
                 memory_size=10000, batch_size=128,
                 epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=1000,
                 tau=0.005, gamma=0.99, learning_rate=0.1,
                 policy_output=None, target_output=None, statistics_output=None,
                 policy_input=None, target_input=None, n_hidden_layers=1, n_neurons=128):
        self.env = TicTacToeEnvironment(
            n_observations, n_actions)
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.number_episodes = number_episodes
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.policy_output = policy_output
        self.target_output = target_output
        self.statistics_output = statistics_output
        self.policy_input = policy_input
        self.target_input = target_input
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.steps_taken = 0
        self.explores = 0
        self.exploits = 0
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.start = 0

        device_manager = DeviceManager.get_instance()
        self.device = device_manager.get_device()

        match = None

        if target_input is not None:
            match = re.search(r"\b\d+\b", target_input)
        
        if match:
            self.start = int(match.group())




    def configure(self):
        self.memory = ReplayMemory(self.memory_size)
        self.policy_network = DQN(
            self.n_observations, self.n_actions, self.n_hidden_layers, self.n_neurons).to(self.device)
        self.target_network = DQN(
            self.n_observations, self.n_actions, self.n_hidden_layers, self.n_neurons).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.optimizer = optim.AdamW(
            self.policy_network.parameters(), lr=self.learning_rate, amsgrad=True)

    def load_model(self):
        if self.policy_input is not None:
            with open(self.policy_input, 'rb') as f:
                buffer = io.BytesIO(f.read())
            state_dict = torch.load(buffer)
            self.policy_network.load_state_dict(state_dict)
        if self.target_input is not None:
            with open(self.target_input, 'rb') as f:
                buffer = io.BytesIO(f.read())
            state_dict = torch.load(buffer)
            self.target_network.load_state_dict(state_dict)


    def save_model(self):
        if self.policy_output is not None:
            torch.save(self.policy_network.state_dict(), self.policy_output)
        if self.target_output is not None:
            torch.save(self.target_network.state_dict(), self.target_output)

    def save_statistics(self, i):
        # excluding draw!
        headers = (
            'EpisodeNumber,NumberEpisodes,MemorySize,BatchSize,'
            'EpsilonStart,EpsilonEnd,EpsilonDecay,Tau,Gamma,LearningRate,'
            'WinReward,DrawReward,LossReward,legalMoveReward,IllegalMoveReward,'
            'NumberNeurons', 'NumberHiddenLayers'
            'Explore/Exploit,Outcome,NumberIllegalMoves'
        )

        run_number = i + self.start

        out = (
            f'{run_number},{self.number_episodes},{self.memory.capacity},{self.batch_size},'
            f'{self.epsilon_start},{self.epsilon_end},{self.epsilon_decay},{self.tau},{self.gamma},{self.learning_rate},'
            f'{self.env.win_reward},{self.env.draw_reward},{self.env.loss_reward},'
            f'{self.env.legal_move_reward},{self.env.illegal_move_reward},'
            f'{self.n_neurons},{self.n_hidden_layers},'
            f'{self.explores/(self.exploits + self.explores)},{self.env.outcome},{self.env.illegal_moves}\n'
        )

        if os.path.isfile(self.statistics_output):
            # file exists, append
            with open(self.statistics_output, 'a') as f:
                f.write(out)
        else:
            # file does not exist, create and append
            with open(self.statistics_output, 'w') as f:
                f.write(f'{headers}\n{out}')

    def select_action(self):
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.steps_taken / self.epsilon_decay)
        self.steps_taken += 1
        if sample > eps_threshold:
            with torch.no_grad():
                self.exploits += 1
                return self.policy_network(self.env.state).max(1)[1].view(1, 1)
        else:
            self.explores += 1
            return torch.tensor([[random.randint(0, self.env.action_space-1)]], device=self.device, dtype=torch.long)

    def act(self):
        action = self.select_action()

        reward = self.env.step(
            action.item())

        return (action, reward)

    def observe_state_change(self, action, reward):
        self.memory.push(self.env.state, action,
                         self.env.next_state, reward)
        
        self.env.state = self.env.next_state

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        #
        state_action_values = self.policy_network(
            state_batch).gather(1, action_batch)
        #
        next_state_values = torch.zeros(self.batch_size, device=self.device, )

        #
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(
                non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values,
                         expected_state_action_values.unsqueeze(1))


        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
        self.optimizer.step()


    def update_target_network(self):
        # soft update target network
        target_network_state_dict = self.target_network.state_dict()
        policy_network_state_dict = self.policy_network.state_dict()
        for key in policy_network_state_dict:
            target_network_state_dict[key] = policy_network_state_dict[key] * \
                self.tau + target_network_state_dict[key]*(1-self.tau)
        self.target_network.load_state_dict(target_network_state_dict)

    def train(self):
        if self.policy_input is not None:
            self.load_model()

        for i in range(self.number_episodes):
            self.env.reset()
            self.exploits = 0
            self.explores = 0

            while True:
                # select an action and step in the environment
                action, reward = self.act()
                reward = torch.tensor([reward], device=self.device)

                # self.env.render()

                # observe and remember state change
                self.observe_state_change(action, reward)

                # optimize model from what we learn
                self.optimize_model()

                # soft update target network
                self.update_target_network()

                # if episode has ended
                if self.env.terminated:
                    if self.statistics_output is not None:
                        self.save_statistics(i)
                    break

        if self.policy_output is not None:
            self.save_model()
