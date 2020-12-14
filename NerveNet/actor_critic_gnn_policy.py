from typing import Any, Dict, List, Optional, Tuple, Type, Union

from pathlib import Path
import gym
import numpy as np
import torch as th
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, MlpExtractor, NatureCNN, create_mlp
from graph_util.mujoco_parser_nervenet import parse_mujoco_graph


class NerveNetGNN(BaseFeaturesExtractor):
    """
    GNN from NerveNet paper:
        Wang, et al.
        "Nervenet: Learning structured policy with graph neural networks"
        6th International Conference on Learning Representations, ICLR 2018

    :param observation_space:
    :param env: 
    """

    def __init__(self,
                 observation_space: gym.Space,
                 env=None,
                 task_name: str = None,
                 xml_name: str = None,
                 xml_assets_path: Path = None):
        super(NerveNetGNN, self).__init__(observation_space,
                                          get_flattened_obs_dim(observation_space))
        # TODO: either require number of features to be given as argument or extract them from env
        self.task_name = task_name
        self.xml_name = xml_name
        self.xml_assets_path = xml_assets_path

        if self.xml_name is None:
            if isinstance(env, gym.Wrapper):
                env = env.env
            self.xml_name = env.robot.model_xml

        # graph = parse_mujoco_graph(task_name=self.task_name,
        #                            xml_name=self.xml_name,
        #                            xml_assets_path=self.xml_assets_path)
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.flatten(observations)


# @Hannes: I think we should be able to use ActorCriticPolicy class without any (major) changes
# we might just get away with just initialising the class with our own features_extractor_class
# as defined above.
# The only issue I currently see with that approach is, that I'm not a 100% sure about the
# calculation of the controler outputs. The NerveNet paper proposed to use a different MLPs for
# each type of controler node (hips, feet, knees, etc.) to calculate the mean of the policy
# distribution. However, they to say that in practice they found that one unified controller
# doesn't hurt the performance, so we might be able to get away with that.
#

# TODO: Next step: decide which data structure we should use to pass the robot structure to NerveNetGNN

class ActorCriticGNNPolicy(ActorCriticPolicy):
    """
    GNN policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = NerveNetGNN,
        # TODO: use this to pass the robot structure to the NerveNetGNN
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(ActorCriticGNNPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            sde_net_arch=sde_net_arch,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

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
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Features for sde
        latent_sde = latent_pi
        if self.sde_features_extractor is not None:
            latent_sde = self.sde_features_extractor(features)
        return latent_pi, latent_vf, latent_sde
