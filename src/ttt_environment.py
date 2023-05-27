import torch
from device_manager import DeviceManager
from environment import Environment


class TicTacToeEnvironment(Environment):
    def __init__(self, action_space, observation_space, legal_move_reward=1,
                 illegal_move_reward=-1, win_reward=10, loss_reward=-10, draw_reward=0):
        super().__init__(action_space, observation_space)
        self.state = None
        self.next_state = None
        self.terminated = False
        self.legal_move_reward = legal_move_reward
        self.illegal_move_reward = illegal_move_reward
        self.win_reward = win_reward
        self.loss_reward = loss_reward
        self.draw_reward = draw_reward
        self.illegal_moves = 0
        self.legal_moves = 0
        self.outcome = 0

        device_manager = DeviceManager.get_instance()
        self.device = device_manager.get_device()

    def reset(self):
        self.terminated = False
        self.illegal_moves = 0
        self.legal_moves = 0
        self.state = torch.zeros((1, self.observation_space))

    def update_next_state(self, observation):
        if self.terminated:
            self.next_state = None
        else:
            self.next_state = observation.clone().detach()


    def opponent_move(self):
        if (self.state[0, 4] == 0):
            self.state[0, 4] = -1
            return

        # Play a corner if available
        if (self.state[0, 0] == 0):
            self.state[0, 0] = -1
            return
        if (self.state[0, 2] == 0):
            self.state[0, 2] = -1
            return
        if (self.state[0, 6] == 0):
            self.state[0, 6] = -1
            return
        if (self.state[0, 8] == 0):
            self.state[0, 8] = -1
            return
        # Play an edge if available
        if (self.state[0, 1] == 0):
            self.state[0, 1] = -1
            return
        if (self.state[0, 3] == 0):
            self.state[0, 3] = -1
            return
        if (self.state[0, 5] == 0):
            self.state[0, 5] = -1
            return
        if (self.state[0, 7] == 0):
            self.state[0, 7] = -1
            return

    def check_end(self, value):
        state_flat = self.state.flatten()  # flatten the 2D tensor into a 1D tensor
        for i in range(3):
            if torch.all(state_flat[i*3:i*3+3] == value):
                return True  # horizontal check
            if torch.all(state_flat[i:9:3] == value):
                return True  # vertical check
        if torch.all(state_flat[0:9:4] == value):
            return True  # diagonal check from top-left to bottom-right
        if torch.all(state_flat[2:7:2] == value):
            return True  # diagonal check from top-right to bottom-left
        return False
            
    def check_draw(self):
        return False if torch.any(self.state == 0) else True


    # implement this
    def step(self, action):
        # invalid move
        if self.state[0, action] != 0:
            self.illegal_moves += 1
            self.update_next_state(self.state)
            return torch.Tensor([self.illegal_move_reward], device=self.device)

        # play move
        self.state[0, action] = 1
        self.legal_moves += 1

        # check finish
        if self.check_end(1):
            self.outcome = 1
            self.terminated = True
            self.update_next_state(self.state)
            return torch.Tensor([self.win_reward], device=self.device)
        if self.check_draw():
            self.terminated = True
            self.update_next_state(self.state)
            return torch.Tensor([self.draw_reward], device=self.device)

        # opponent play
        self.opponent_move()

        # check finish
        if self.check_draw():
            self.terminated = True
            self.update_next_state(self.state)
            return torch.Tensor([self.draw_reward], device=self.device)
        if self.check_end(-1):
            self.outcome = -1
            self.terminated = True
            self.update_next_state(self.state)
            return torch.Tensor([self.loss_reward], device=self.device)

        self.update_next_state(self.state)
        return torch.Tensor([self.legal_move_reward], device=self.device)

    def render(self):
        # Create a dictionary to map the values to symbols
        symbols = {
            1: 'X',
            -1: 'O',
            0: ' '
        }
        # Print the board
        print(
            f'\n {symbols[self.state[0, 0].item()]} | {symbols[self.state[0, 1].item()]} | {symbols[self.state[0, 2].item()]} ')
        print('---+---+---')
        print(
            f' {symbols[self.state[0, 3].item()]} | {symbols[self.state[0, 4].item()]} | {symbols[self.state[0, 5].item()]} ')
        print('---+---+---')
        print(
            f' {symbols[self.state[0, 6].item()]} | {symbols[self.state[0, 7].item()]} | {symbols[self.state[0, 8].item()]} ')
