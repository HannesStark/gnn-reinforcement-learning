from datetime import datetime
import numpy as np
import gym
from gym import wrappers
from stable_baselines3 import A2C
import pybullet_envs  # register pybullet envs from bullet3


def evaluate(model, num_steps=1000):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """
    episode_rewards = [0.0]
    obs = env.reset()
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs)

        obs, reward, done, info = env.step(action)

        # Stats
        episode_rewards[-1] += reward
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)
    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))

    return mean_100ep_reward


env_name = 'AntBulletEnv-v0'

env = gym.make(env_name)

model = A2C('MlpPolicy', env, verbose=1, tensorboard_log="runs")
mean_reward_before_train = evaluate(model, num_steps=1000)
model.learn(total_timesteps=10000, tb_log_name='{}_{}'.format(env_name, datetime.now().strftime('%d-%m_%H-%M-%S')))
model.save("a2c_ant")
mean_reward = evaluate(model, num_steps=1000)
print(mean_reward_before_train)
print(mean_reward)
