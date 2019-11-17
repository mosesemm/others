from environments.obstacle_tower.obstacle_tower_env import ObstacleTowerEnv
from matplotlib import pyplot as plt
import time
import os
from train.model import Policy, CNNBase
from train.ppo_agent import PPO
from train.memory import RolloutStorage
from train.wrappers import *

import torch
from collections import deque

import numpy as np
import gym



# args
args = {
    'num_updates': 10,
    'num_steps' : 10000,  # 10e6
    'clip_param': 0.2,
    'ppo_epoch': 4,
    'num_mini_batch':32 ,
    'value_loss_coef': 0.5,
    'entropy_coef': 0.01,
    'lr': 7e-4,
    'eps': 1e-5,
    'max_grad_norm': 0.5,
    'num_processes': 1,
    'save_interval': 100,
    'use_gae': False,
    'use_proper_time_limits': False,
    'gae_lambda': 0.95,
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_otc_env(args,
                 device,
                 start_index=0,
                 allow_early_resets=True,
                 start_method=None):

    def make_env(rank, env_count, total_env):
        def _thunk():

            config = {'starting-floor': 0, 'total-floors': 5, 'dense-reward': 1,
                      'lighting-type': 0, 'visual-theme': 0, 'default-theme': 0, 'agent-perspective': 1,
                      'allowed-rooms': 0,
                      'allowed-modules': 0,
                      'allowed-floors': 0,
                      }
            env = ObstacleTowerEnv('./ObstacleTower/obstacletower',
                                   worker_id=1, retro=True, realtime_mode=False, config=config)
            # env.seed(seed + rank)
            # env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
            #               allow_early_resets=allow_early_resets)
            # return wrap_deepmind(env, **wrapper_kwargs)
            return env

        return _thunk

    envs = [
        make_env(i + start_index + 1, i, args['num_processes'])
        for i in range(args['num_processes'])
    ]

    envs = DummyVecEnv(envs)
    # if args.normalize_visual_obs:
    #     envs = VecNormalize(envs, True, ret=False,clipob=1.)
    return envs



def main():
    env = make_otc_env(args, device)

    env = VecPyTorch(env, device)
    #env = gym.wrappers.Monitor(env, "recording", video_callable=lambda episode_id: episode_id % 60 == 0, force=True)
    #env = PyTorchFrame(env)
    #env = FrameStack(env, 4)
    #env.seed(1)
    print(env.observation_space)
    print(env.action_space)

    obs_shape = env.observation_space.spaces['visual'].shape
    vector_obs_len = env.observation_space.spaces['vector'].shape[0]
    actor_critic = Policy(
        obs_shape,
        env.action_space,
        vector_obs_len=vector_obs_len)

    agent = PPO(
        actor_critic,
        args['clip_param'],
        args['ppo_epoch'],
        args['num_mini_batch'],
        args['value_loss_coef'],
        args['entropy_coef'],
        lr=args['lr'],
        eps=args['eps'],
        max_grad_norm=args['max_grad_norm'])

    rollouts = RolloutStorage(args['num_steps'], args['num_processes'],
                              env.observation_space.shape, [vector_obs_len], env.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs  = env.reset()
    obs_tensor = torch.from_numpy(obs).float().to(device)
    rollouts.obs[0].copy_(obs_tensor)
    rollouts.vector_obs[0].copy_(obs_tensor)
    rollouts.to(device)

    start = time.time()

    episode_rewards = [0.0]
    episode_floors = deque(maxlen=100)
    episode_times = deque(maxlen=100)


    for j in range(args['num_updates']):

        for step in range(args['num_steps']):

            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.vector_obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            obs, vector_obs, reward, done, infos = env.step(action)

            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])

            rollouts.insert(obs, vector_obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1],
                rollouts.vector_obs[-1],
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args['use_gae'], args.gamma,
                                 args.gae_lambda, args['use_proper_time_limits'])

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        episode_rewards[-1] += reward

        total_num_steps = (j + 1) * args['num_processes'] * args['num_steps']

        if (j % args['save_interval'] == 0
                or j == args['num_updates'] - 1):

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(total_num_steps))
            print("episodes: {}".format(j))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("********************************************************")
            torch.save({"model_state_dict": actor_critic.state_dict()}, 'checkpoint_pl.pth')
            np.savetxt("rewards.csv", episode_rewards, delimiter=",", fmt='%1.3f')

    '''

    plt.imshow(obs)
    plt.show()

    obs, reward, done, info = env.step(env.action_space.sample())
    print('obs', obs)
    print('reward', reward)
    print('done', done)
    print('info', info)

    plt.imshow(obs)
    plt.show()
    env.close()
    
    '''


if __name__ == '__main__':
    main()
