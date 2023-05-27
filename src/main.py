import argparse

from agent import Agent

def add_args():
    parser.add_argument('-e', '--number_episodes',  type=int, dest='number_episodes',
                        help='Number of episodes to train the agent.')
    parser.add_argument('-m', '--memory_size',  type=int, dest='memory_size', 
                        help='Size of replay memory to store transition.')
    parser.add_argument('-b', '--batch_size',  type=int, dest='batch_size',
                        help='Batch size of memory to use for optimization.')
    parser.add_argument('-es', '--epsilon_start',  type=float, dest='epsilon_start',
                        help='Epsilon start value for select action')
    parser.add_argument('-ee', '--epsilon_end',  type=float, dest='epsilon_end',
                        help='Epsilon end value for select action')
    parser.add_argument('-ed', '--epsilon_decay',  type=float, dest='epsilon_decay',
                        help='Epsilon decay value for select action')
    parser.add_argument('-t', '--tau',  type=float, dest='tau',
                        help='Tau value for soft update')
    parser.add_argument('-g', '--gamma',  type=float, dest='gamma',
                        help='Gamma value for Q value calculation')
    parser.add_argument('-l', '--learning_rate',  type=float, dest='learning_rate',
                        help='Learning rate for agent to learn the environment.')
    parser.add_argument('-po', '--policy_output',  type=str, dest='policy_output',
                        help='Filepath for policy network output')
    parser.add_argument('-pi', '--policy_input',  type=str, dest='policy_input',
                        help='Filepath for policy network input')
    parser.add_argument('-to', '--target_output',  type=str, dest='target_output',
                        help='Filepath for target network output')
    parser.add_argument('-ti', '--target_input',  type=str, dest='target_input',
                        help='Filepath for target network input')
    parser.add_argument('-so', '--statistics_output',  type=str, dest='statistics_output',
                        help='Filepath for statistics output')
    parser.add_argument('-lmr', '--legal_move_reward',  type=float, dest='legal_move_reward',
                        help='Reward for legal move.')
    parser.add_argument('-ilr', '--illegal_move_reward',  type=float, dest='illegal_move_reward',
                        help='Reward for illegal move.')
    parser.add_argument('-wr', '--win_reward',  type=float, dest='win_reward',
                        help='Reward for winning the game.')
    parser.add_argument('-lr', '--loss_reward',  type=float, dest='loss_reward',
                        help='Reward for losing the game.')
    parser.add_argument('-dr', '--draw_reward',  type=float, dest='draw_reward',
                        help='Reward for a draw.')
    parser.add_argument('-nl', '--number_h_layers',  type=int, dest='number_hidden_layers',
                        help='Number of hidden layers in the neural network.')
    parser.add_argument('-nn', '--number_neurons',  type=int, dest='number_neurons',
                    help='Number of neurons in the hidden layers of the neural network.')


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    add_args()

    agent = Agent(9, 9)

    parsed_args = parser.parse_args()

    if parsed_args.number_episodes is not None:
        agent.number_episodes = parsed_args.number_episodes
    if parsed_args.memory_size is not None:
        agent.memory_size = parsed_args.memory_size
    if parsed_args.batch_size is not None:
        agent.batch_size = parsed_args.batch_size
    if parsed_args.epsilon_start is not None:
        agent.epsilon_start = parsed_args.epsilon_start
    if parsed_args.epsilon_end is not None:
        agent.epsilon_end = parsed_args.epsilon_end
    if parsed_args.epsilon_decay is not None:
        agent.epsilon_decay = parsed_args.epsilon_decay
    if parsed_args.tau is not None:
        agent.tau = parsed_args.tau
    if parsed_args.gamma is not None:
        agent.gamma = parsed_args.gamma
    if parsed_args.learning_rate is not None:
        agent.learning_rate = parsed_args.learning_rate
    if parsed_args.policy_output is not None:
        agent.policy_output = parsed_args.policy_output
    if parsed_args.target_output is not None:
        agent.target_output = parsed_args.target_output
    if parsed_args.statistics_output is not None:
        agent.statistics_output = parsed_args.statistics_output
    if parsed_args.policy_input is not None:
        agent.policy_input = parsed_args.policy_input
    if parsed_args.target_input is not None:
        agent.target_input = parsed_args.target_input
    if parsed_args.legal_move_reward is not None:
        agent.env.legal_move_reward = parsed_args.legal_move_reward
    if parsed_args.illegal_move_reward is not None:
        agent.env.illegal_move_reward = parsed_args.illegal_move_reward
    if parsed_args.win_reward is not None:
        agent.env.win_reward = parsed_args.win_reward
    if parsed_args.loss_reward is not None:
        agent.env.loss_reward = parsed_args.loss_reward
    if parsed_args.draw_reward is not None:
        agent.env.draw_reward = parsed_args.draw_reward
    if parsed_args.number_hidden_layers is not None:
        agent.n_hidden_layers = parsed_args.number_hidden_layers
    if parsed_args.number_neurons is not None:
        agent.n_neurons = parsed_args.number_neurons

    agent.configure()

    agent.train()
