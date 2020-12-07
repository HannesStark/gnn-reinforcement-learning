from typing import Type, Tuple

import gym
import torch as th
from gym import wrappers
from stable_baselines3 import PPO
import pybullet_envs  # register pybullet envs from bullet3
from stable_baselines3.common.policies import ActorCriticPolicy


class ActorCriticGNNPolicy(ActorCriticPolicy):
    def _get_latent(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.
        :param obs: Observation
        :return: Latent codes
            for the actor, the value function and for gSDE function
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        print(features)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Features for sde
        latent_sde = latent_pi
        if self.sde_features_extractor is not None:
            latent_sde = self.sde_features_extractor(features)
        return latent_pi, latent_vf, latent_sde

env = gym.make('AntBulletEnv-v0')

model = PPO(ActorCriticGNNPolicy, env, verbose=1)
model.learn(total_timesteps=10000)
model.save("a2c_ant")
