#!/bin/bash

num_episodes=500
loss_reward=0
win_reward=0
draw_reward=0
legal_move_reward=1000
illegal_move_reward=-1000
learning_rate=0.01
batch_size=128
memory_size=10000


./build/app --batchSize "$batch_size" --memorySize "$memory_size" --numEpisodes "$num_episodes" \
--winReward "$win_reward" --lossReward "$loss_reward" --drawReward "$draw_reward" \
--legalMoveReward "$legal_move_reward" --illegalMoveReward "$illegal_move_reward" \
--learningRate "$learning_rate"

# set parameters
legal_move_reward=-1000
illegal_move_reward=1000

./build/app --batchSize "$batch_size" --memorySize "$memory_size" --numEpisodes "$num_episodes" \
--winReward "$win_reward" --lossReward "$loss_reward" --drawReward "$draw_reward" \
--legalMoveReward "$legal_move_reward" --illegalMoveReward "$illegal_move_reward" \
--learningRate "$learning_rate"