from datetime import datetime
import numpy as np
import gym
from gym import wrappers
from stable_baselines3 import A2C, PPO
import pybullet_envs  # register pybullet envs from bullet3


def evaluate(model, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward


env_name = 'AntBulletEnv-v0'

env = gym.make(env_name)

model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log = "gs://adlr-train-logs/runs", batch_size=1792)
mean_reward_before_train = evaluate(model, num_episodes=4)
model.learn(total_timesteps=300000, tb_log_name='{}_{}'.format(env_name, datetime.now().strftime('%d-%m_%H-%M-%S')))
model.save("a2c_ant")
mean_reward = evaluate(model, num_episodes=4)
print(mean_reward_before_train)
print(mean_reward)
