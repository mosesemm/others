import random
import numpy as np
import gym

from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':

    hyper_params = {
        "seed": 42,  # which seed to use
        "env": "PongNoFrameskip-v4",  # name of the game
        "replay-buffer-size": int(5e3),  # replay buffer size
        "learning-rate": 1e-4,  # learning rate for Adam optimizer
        "discount-factor": 0.99,  # discount factor
        "num-steps": int(1e6),  # total number of steps to run the environment for
        "batch-size": 32,  # number of transitions to optimize at the same time
        "learning-starts": 10000,  # number of steps before learning starts
        "learning-freq": 1,  # number of iterations between every optimization step
        "use-double-dqn": True,  # use double deep Q-learning
        "target-update-freq": 1000,  # number of iterations between every target network update
        "eps-start": 1.0,  # e-greedy start threshold
        "eps-end": 0.01,  # e-greedy end threshold
        "eps-fraction": 0.1,  # fraction of num-steps
        "print-freq": 10
    }

    np.random.seed(hyper_params["seed"])
    random.seed(hyper_params["seed"])

    assert "NoFrameskip" in hyper_params["env"], "Require environment with no frameskip"
    env = gym.make(hyper_params["env"])
    env.seed(hyper_params["seed"])

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    # TODO Pick Gym wrappers to use
    #
    #
    #

    env = WarpFrame(env)
    env = PyTorchFrame(env)
    #env = FrameStack(env)

    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])

    # TODO Create dqn agent

    agent = DQNAgent(env.observation_space, env.action_space, replay_buffer, hyper_params["use-double-dqn"],
                        hyper_params["learning-rate"], hyper_params["batch-size"], hyper_params["discount-factor"])

    eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"])
    episode_rewards = [0.0]

    state = env.reset()
    for t in range(hyper_params["num-steps"]):
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fraction * (hyper_params["eps-end"] - hyper_params["eps-start"])
        sample = random.random()
        # TODO 
        #  select random action if sample is less equal than eps_threshold
        action = random.choice(np.arange(env.action_space.n)) if sample <= eps_threshold else  agent.act(state)
        # take step in env
        next_state, reward, done, _ = env.step(action)
        # add state, action, reward, next_state, float(done) to reply memory - cast done to float
        replay_buffer.add(state, action, reward, next_state, float(done))

        # add reward to episode_reward
        state = next_state
        episode_rewards[-1] += reward
        if done:
            state = env.reset()
            episode_rewards.append(0.0)

        if t > hyper_params["learning-starts"] and t % hyper_params["learning-freq"] == 0:
            agent.optimise_td_loss()

        if t > hyper_params["learning-starts"] and t % hyper_params["target-update-freq"] == 0:
            agent.update_target_network()

        num_episodes = len(episode_rewards)

        if done and hyper_params["print-freq"] is not None and len(episode_rewards) % hyper_params[
            "print-freq"] == 0:
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(t))
            print("episodes: {}".format(num_episodes))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("% time spent exploring: {}".format(int(100 * eps_threshold)))
            print("********************************************************")
            torch.save(agent.policy_network.state_dict(), 'checkpoint_pl.pth')
            torch.save(agent.target_network.state_dict(), 'checkpoint_tgt.pth')

    #plot reward per episode
    fig = plt.figure()
    #ax = fig.add_subplot(111)
    plt.plot(np.arange(len(episode_rewards)), episode_rewards)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.save("espisode_reward.png")