

Ts, rewards, Qs, best_avg_reward = [], [], [], -1e10

def test(args, T, dqn, val_mem,
         skip_frames=1, evaluate=False, realtime=False, env=None):

    global Ts, rewards, Qs, best_avg_reward

    own_env = False

    Ts.append(T)
    T_rewards = []
    T_Qs = []

    # Test performance over several episodes
    done = True
    for _ in range(args.evaluation_episodes):
        while True:
            if done:
                state = env.reset()
                reward_sum = 0
                done = False

            action = dqn.act_e_greedy(state)  # Choose an action ε-greedily
            state, reward, done, _ = env.step(action)  # Step
            reward_sum += reward

            if done:
                T_rewards.append(reward_sum)
                break


    # Test Q-values over validation memory
    for state in val_mem:  # Iterate over valid states
        T_Qs.append(dqn.evaluate_q(state))

    avg_reward = sum(T_rewards) / len(T_rewards)
    avg_Q = sum(T_Qs) / len(T_Qs)
    if not evaluate:
        # Append to results
        rewards.append(T_rewards)
        Qs.append(T_Qs)

        # Save model parameters if improved
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            dqn.save('results'+str(T))

    # Return average reward and Q-value
    return avg_reward, avg_Q