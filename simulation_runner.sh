cd src;

# TODO: code start
start=0
num=0
increment=0

default_episodes=100000
default_training_sessions=20
default_neurons=128
default_hidden_layers=1
n_episodes=$default_episodes    
n_training_sessions=$default_training_sessions
n_neurons=$default_neurons
n_hidden_layers=$default_hidden_layers


## parse args
args=("$@")
i=0

while [ $i -lt $# ]; do
    if [[ "${args[$i]}" == "-h" || "${args[$i]}" == "--help" ]]; then
        echo "Help:"
        echo "  -h, --help             Display this help message"
        echo "  --n_neurons <value>    Set the number of neurons (default=$default_neurons)"
        echo "  --n_hidden_layers <value>  Set the number of hidden layers (default=$default_hidden_layers)"
        echo "  --n_episodes <value>  Set the number of episodes per session (default=$default_episodes)"
        echo "  --n_training_sessions <value>  Set the number of training sessions (default=$default_training_sessions)"
        exit 0  # Optional: Exit the script after displaying the help message
    fi
    if [[ "${args[$i]}" == "--n_neurons" && "$(($i+1))" -lt $# ]]; then
        n_neurons="${args[$i+1]}"
        ((i+=2))
        continue
    fi
    if [[ "${args[$i]}" == "--n_hidden_layers" && "$(($i+1))" -lt $# ]]; then
        n_hidden_layers="${args[$i+1]}"
        ((i+=2))
        continue
    fi
    if [[ "${args[$i]}" == "--n_episodes" && "$(($i+1))" -lt $# ]]; then
        n_episodes="${args[$i+1]}"
        ((i+=2))
        continue
    fi
    if [[ "${args[$i]}" == "--n_training_sessions" && "$(($i+1))" -lt $# ]]; then
        n_training_sessions="${args[$i+1]}"
        ((i+=2))
        continue
    fi
    ((i++))
done

tag="${n_hidden_layers}layer${n_neurons}neuron"
stats_out="../out/$tag-stats.csv"
policy_out="../out/$tag-policy"
target_out="../out/$tag-target"

end=$(($start + ($n_episodes * $n_training_sessions)))
## parse args end



echo "--------------------------------------------------------------------"
echo "                     Training Deep Q Network"
echo "--------------------------------------------------------------------"
echo -e "Episode Range:\t\t\tEpisodes $(($start+1))-$end"
echo -e "Training Schedule:\t\t$n_training_sessions sessions @ $n_episodes episode(s)"
echo -e "Number of Hidden Layers:\t$n_hidden_layers"
echo -e "Number of Neurons per Layer:\t$n_neurons"
echo "--------------------------------------------------------------------"


echo "$(date '+%m/%d %H:%M:%S') | 0/$n_training_sessions training sessions complete"

if [ $start -eq 0 ]; then
    # # try to make legal moves
    python3 main.py --number_episodes $n_episodes --tau 0.005 --memory_size 10000 --batch_size 128 \
    --win_reward 10 --loss_reward 10 --draw_reward 10 --legal_move_reward 10 --illegal_move_reward -10 \
    --number_neurons $n_neurons  --number_h_layers $n_hidden_layers \
    --statistics_output $stats_out  --policy_output "$policy_out-$n_episodes.pth" \
    --target_output "$target_out-$n_episodes.pth" 
    
    num=$(($num + $n_episodes))
    increment=1
    
    wait
    echo "$(date '+%m/%d %H:%M:%S') | $increment/$n_training_sessions training sessions complete"
    increment=$((increment+1))
fi

# ## use this block to use saved network
for (( i=$(($start+$num)); i<$end; i+=$n_episodes)); do
    next=$(($i+$n_episodes))
    
    # # try to make legal moves
    python3 main.py --number_episodes $n_episodes --tau 0.005 --memory_size 10000 --batch_size 128 \
    --win_reward 10 --loss_reward 10 --draw_reward 10 --legal_move_reward 10 --illegal_move_reward -10 \
    --number_neurons $n_neurons  --number_h_layers $n_hidden_layers \
    --statistics_output $stats_out  --policy_output "$policy_out-$next.pth" \
    --policy_input "$policy_out-$i.pth" --target_input "$target_out-$i.pth" \
    --target_output "$target_out-$next.pth" 
    
    
    echo "$(date '+%m/%d %H:%M:%S') | $increment/$n_training_sessions training sessions complete"
    increment=$((increment+1))
done