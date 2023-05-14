cd src;

for (( i=100000; i<=2000000; i+=100000)); do
    for tau in 0.005 0.01 0.05 0.1; do
        for memory in 10000 20000 50000; do
            for batch in 128 256 512; do
                # try to make legal moves
                python3 main.py --number_episodes $i --tau $tau --memory_size $memory --batch_size $batch \
                --win_reward 10 --loss_reward 10 --draw_reward 10 --legal_move_reward 10 --illegal_move_reward -10 &
                # try to make illegal moves
                python3 main.py --number_episodes $i --tau $tau --memory_size $memory --batch_size $batch \
                --win_reward -10 --loss_reward -10 --draw_reward -10 --legal_move_reward -10 --illegal_move_reward 10 &
            done
        done
    done
done