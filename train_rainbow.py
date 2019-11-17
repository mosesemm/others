from environments.obstacle_tower.obstacle_tower_env import ObstacleTowerEnv
from matplotlib import pyplot as plt
import torch
from train.rainbow_agent import Agent
from datetime import datetime
from train.util import test
from train.memory import ReplayMemory
from train.wrappers import ToTorchTensors

class Args():
    def __init__(self):
        self.seed = int(1)
        self.T_max = int(50e6)
        self.max_episode_length = int(108e3)
        self.history_length= int(1)
        self.hidden_size = int(512)
        self.noisy_std = 0.1
        self.atoms = int(51)
        self.V_min = -10
        self.V_max = int(10)
        self.memory_capacity = int(1e6)
        self.replay_frequency = int(4)
        self.priority_exponent = 0.5
        self.priority_weight = 0.4
        self.multi_step = int(3)
        self.discount= 0.99
        self.target_update = int(32e3)
        self.reward_clip = 1
        self.lr=0.0000625
        self.adam_eps=1.5e-4
        self.batch_size=int(32)
        self.learn_start=int(80e3)
        self.evaluation_size = int(500)
        self.evaluation_episodes = int(10)
        self.evaluation_interval = int(100000)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def log(s):
    print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)

def main():

    args = Args()
    args.device = device

    args.large = False
    args.skip_frames = 0
    args.random_aug = 0.

    config = {'starting-floor': 0, 'total-floors': 9, 'dense-reward': 1,
              'lighting-type': 0, 'visual-theme': 0, 'default-theme': 0, 'agent-perspective': 1, 'allowed-rooms': 0,
              'allowed-modules': 0,
              'allowed-floors': 0,
              }
    env = ObstacleTowerEnv('./ObstacleTower/obstacletower',
                           worker_id=1, retro=True, realtime_mode=False, config=config)

    env = ToTorchTensors(env, device=device)

    test_env = ObstacleTowerEnv('./ObstacleTower/obstacletower',
                           worker_id=2, retro=True, realtime_mode=False, config=config)

    test_env = ToTorchTensors(test_env, device=device)

    action_space = env.action_space
    env.seed(1)
    print(env.observation_space)
    print(env.action_space)

    mem = ReplayMemory(args, args.memory_capacity, obs_space=env.observation_space)
    val_mem = ReplayMemory(args, args.evaluation_size, obs_space=test_env.observation_space)

    dqn = Agent(args, env)

    priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)
    time_step = 0
    done = True
    state = None

    while time_step < args.evaluation_size:
        if done:
            state = env.reset()
            done = False

        next_state, _, done, _ = env.step(action_space.sample())
        val_mem.append(state, None, None, done)
        state = next_state
        time_step += 1


    dqn.train()
    done = True

    for time_step in range(args.T_max):
        if done:
            state = env.reset()
            done = False

        if time_step % args.replay_frequency == 0:
            dqn.reset_noise()  # Draw a new set of noisy weights

        action = dqn.act(state)  # Choose an action greedily (with noisy weights)
        next_state, reward, done, info = env.step(action)  # Step
        if args.reward_clip > 0:
            reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
        mem.append(state, action, reward, done)  # Append transition to memory

        # Train and test
        if time_step >= args.learn_start:
            # Anneal importance sampling weight β to 1
            mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)

            if time_step % args.replay_frequency == 0:
                dqn.learn(mem)  # Train with n-step distributional double-Q learning

            if time_step % args.evaluation_interval == 0:
                dqn.eval()  # Set DQN (online network) to evaluation mode
                avg_reward, avg_Q = test(args, time_step, dqn, val_mem, env=test_env)  # Test
                log('T = ' + str(time_step) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(
                    avg_reward) + ' | Avg. Q: ' + str(avg_Q))
                dqn.train()  # Set DQN (online network) back to training mode

            # Update target network
            if time_step % args.target_update == 0:
                dqn.update_target_net()

        state = next_state


    env.close()


if __name__ == '__main__':
    main()
