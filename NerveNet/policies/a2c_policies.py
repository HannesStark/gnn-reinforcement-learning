from typing import Any, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
import gym
import torch
from stable_baselines3.common.preprocessing import get_action_dim
from torch import nn
from torch_geometric.nn import GCNConv

from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.utils import get_device
from NerveNet.models.nerve_net_gnn import NerveNetGNN, NerveNetGNN_V0
from NerveNet.models.nerve_net_conv import NerveNetConv
from NerveNet.graph_util.mujoco_parser import parse_mujoco_graph


class ActorCriticGnnPolicy(ActorCriticPolicy):
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
        ``torch.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            net_arch: Dict[str, List[Tuple[nn.Module, int]]] = {
                "input": [
                    (nn.Linear, 12)
                ],
                "propagate": [
                    (NerveNetConv, 12),
                    (NerveNetConv, 12),
                    (NerveNetConv, 12)
                ],
                "policy": [
                    (nn.Linear, 64)
                ],
                "value": [
                    (nn.Linear, 64)
                ]
            },
            activation_fn: Type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            sde_net_arch: Optional[List[int]] = None,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            # use these to pass arguments to the NerveNetGNN
            mlp_extractor_class: Type[nn.Module] = NerveNetGNN,
            mlp_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):

        self.mlp_extractor_class = mlp_extractor_class
        self.mlp_extractor_kwargs = mlp_extractor_kwargs
        super(ActorCriticGnnPolicy, self).__init__(
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
        self.action_net = torch.nn.Identity()

    @property
    def device(self) -> torch.device:
        """Infer which device this policy lives on by inspecting its parameters.
        If it has no parameters, the 'auto' device is used as a fallback.

        Note: The BasePolicy class for some reason returns cpu and not auto as fallback.
        However, when we use the FlattenExtractor as FeatureExtractor Network there won't have
        been any parameters defined from which we could infere the correct device, always leading to the fallback.
        Which means, we wouldn't be able to use the GPU if we also use FlattenExtractor
        :return:"""
        for param in self.parameters():
            return param.device
        return get_device("auto")

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        """
        self.mlp_extractor = self.mlp_extractor_class(self.features_dim, net_arch=self.net_arch, activation_fn=self.activation_fn,
                                                      **self.mlp_extractor_kwargs
                                                      )

    def _get_latent(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.
        :param obs: Observation
        :return: Latent codes
            for the actor, the value function and for gSDE function
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, log_std_action, latent_vf = self.mlp_extractor(features)

        # Features for sde
        latent_sde = latent_pi
        if self.sde_features_extractor is not None:
            latent_sde = self.sde_features_extractor(features)
        return latent_pi, log_std_action, latent_vf, latent_sde

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        latent_pi, log_std_action, _, latent_sde = self._get_latent(
            observation)
        distribution = self._get_action_dist_from_latent(
            latent_pi, log_std_action, latent_sde)
        return distribution.get_actions(deterministic=deterministic)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        latent_pi, log_std_action, latent_vf, latent_sde = self._get_latent(
            obs)
        mean_actions = latent_pi
        values = latent_vf  # nervenet GNN already returns the values
        # Evaluate the values for the given observations
        distribution = self._get_action_dist_from_latent(
            mean_actions, log_std_action, latent_sde=latent_sde)
        actions = distribution.get_actions(deterministic=deterministic)

        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        latent_pi, log_std_action, latent_vf, latent_sde = self._get_latent(
            obs)
        distribution = self._get_action_dist_from_latent(
            latent_pi, log_std_action, latent_sde)
        log_prob = distribution.log_prob(actions)
        values = latent_vf  # nervenet GNN already returns the values
        return values, log_prob, distribution.entropy()

    def _get_action_dist_from_latent(self, mean_actions: torch.Tensor, log_std_action: torch.Tensor, latent_sde: Optional[torch.Tensor] = None) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :param latent_sde: Latent code for the gSDE exploration function
        :return: Action distribution
        """

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, log_std_action)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, log_std_action, latent_sde)
        else:
            raise ValueError("Invalid action distribution")


class ActorCriticGnnPolicy_V0(ActorCriticGnnPolicy):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        if "mlp_extractor_class" in kwargs.keys():
            kwargs.pop("mlp_extractor_class")
        super(ActorCriticGnnPolicy_V0, self).__init__(
            *args,
            mlp_extractor_class=NerveNetGNN_V0,
            **kwargs
        )


class ActorCriticMLPPolicyTransfer(ActorCriticPolicy):
    """
    Wrapper of policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.
    This policy allows to use another MLP Policy for a different environment.
    It is suited both for disability and size transfer tasks

    Takes the same parameters as ActorCriticPolicy, but only the following to are not overwritten when 
    loading the trained policy that should be transfered to a new environment.
    :param observation_space: Observation space
    :param action_space: Action space

    :param base_policy: The ActorCriticPolicy that should be transfered to the new environment

    There are also a few arguments which are not persisted for ActorCriticPolicy, so they need to be provided again:

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
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            base_policy: Type[ActorCriticPolicy] = None,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            use_expln: bool = False,
            base_env_task_name=None,
            base_env_xml_assets_path=None,
            transfer_env_task_name=None,
            transfer_env_xml_assets_path=None,
            ** kwargs,
    ):

        super(ActorCriticMLPPolicyTransfer, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=base_policy.net_arch,
            activation_fn=base_policy.activation_fn,
            ortho_init=base_policy.ortho_init,
            use_sde=base_policy.use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            sde_net_arch=base_policy.sde_net_arch,
            use_expln=use_expln,
            squash_output=base_policy.squash_output,
            features_extractor_class=base_policy.features_extractor_class,
            features_extractor_kwargs=base_policy.features_extractor_kwargs,
            normalize_images=base_policy.normalize_images,
            optimizer_class=base_policy.optimizer_class,
            optimizer_kwargs=base_policy.optimizer_kwargs,
        )
        self.base_policy = base_policy
        self.base_env_task_name = base_env_task_name
        self.base_env_xml_assets_path = base_env_xml_assets_path
        self.transfer_env_task_name = transfer_env_task_name
        self.transfer_env_xml_assets_path = transfer_env_xml_assets_path

        self.base_env_info = parse_mujoco_graph(task_name=self.base_env_task_name,
                                                xml_assets_path=self.base_env_xml_assets_path)

        self.transfer_env_info = parse_mujoco_graph(task_name=self.transfer_env_task_name,
                                                    xml_assets_path=self.transfer_env_xml_assets_path)

        base_output_mapping = dict()
        transfer_output_mapping = dict()

        for out_id, node_id in enumerate(self.base_env_info["output_list"]):
            node_key = self.base_env_info["tree"][node_id]["name"]
            base_output_mapping[node_key] = out_id

        for out_id, node_id in enumerate(self.transfer_env_info["output_list"]):
            node_key = self.transfer_env_info["tree"][node_id]["name"]
            transfer_output_mapping[node_key] = out_id

        self.out_base_mask = []
        self.out_transfer_mask = []
        for node_key in transfer_output_mapping.keys():
            if node_key in base_output_mapping.keys():
                self.out_base_mask += [transfer_output_mapping[node_key]]
                self.out_transfer_mask += [base_output_mapping[node_key]]

        # current action dist is 12
        # transfered action dist is 8.
        # we need padding here
        #self.action_dist = base_policy.action_dist
        # self.log_std = torch.zeros( self.base_policy.action_space.shape[0]))
        with torch.no_grad():
            self.log_std[self.out_base_mask] = base_policy.log_std.detach()[
                self.out_transfer_mask]

        # old: feature_dim: 38
        # new: feature_dim: 28
        self.features_extractor = base_policy.features_extractor

        # old and new have latent_dim for pi and vf: 64
        self.mlp_extractor = base_policy.mlp_extractor

        # in_features for both = 64
        # out is 12 and 8
        self.base_action_net = base_policy.action_net
        self.action_net = ActionNetWrapper(self.base_action_net,
                                           self.action_space.shape[0],
                                           self.out_base_mask,
                                           self.out_transfer_mask)

        # this fully matches: in 64 and out 1
        self.value_net = base_policy.value_net

    def _get_latent(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def _preprocess_observations(self, transfer_env_obs: torch.Tensor) -> torch.Tensor:
        batch_size = transfer_env_obs.shape[0]
        base_input_mapping = dict()
        transfer_input_mapping = dict()

        for node_id, obs_idx in self.base_env_info["obs_input_mapping"].items():
            node_key = self.base_env_info["tree"][node_id]["name"]
            if not "ignore" in node_key:
                base_input_mapping[node_key] = obs_idx

        for node_id, obs_idx in self.transfer_env_info["obs_input_mapping"].items():
            node_key = self.transfer_env_info["tree"][node_id]["name"]
            if not "ignore" in node_key:
                transfer_input_mapping[node_key] = obs_idx

        # the transfer env mask allows us to filter only for those observations (of joints/body parts)
        # that are in the transfer env as well as in the base env. E.g. if the transfer env has more legs,
        # their input would be discarded using this mask
        mask_transfer = []

        # the base_mask specifies to which joint/body part the given observations from the transfer env should be mapped to
        # this is required in case the order of joints/body parts is switched up
        # But it can also be used to filter out joints/body parts of the base env, that the transfer env doesn't have
        # e.g. for disabilities the base env has elements, the transfer env doesn't have. In those cases the observation for these elements will be zero
        mask_base = []

        for node_key in base_input_mapping.keys():
            if node_key in transfer_input_mapping.keys():
                mask_transfer += transfer_input_mapping[node_key]
                mask_base += base_input_mapping[node_key]

        base_env_obs = torch.zeros(
            (batch_size, self.base_policy.observation_space.shape[0]))
        base_env_obs[:, mask_base] = transfer_env_obs[:, mask_transfer]

        return base_env_obs

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        latent_pi, latent_vf, latent_sde = self._get_latent(
            self._preprocess_observations(obs))
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(
            latent_pi, latent_sde=latent_sde)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        # todo: add 0 padding for actions
        return actions, values, log_prob

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        latent_pi, _, latent_sde = self._get_latent(
            self._preprocess_observations(observation))
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        return distribution.get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # TODO: might need to throw away additional actions
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()


class ActionNetWrapper(torch.nn.Module):
    def __init__(self,
                 base_action_net: torch.nn.Module,
                 output_size: int,
                 out_base_mask,
                 out_transfer_mask):
        super(ActionNetWrapper, self).__init__()
        self.base_action_net = base_action_net
        self.output_size = output_size
        self.out_base_mask = out_base_mask
        self.out_transfer_mask = out_transfer_mask

    def forward(self, latent_pi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = latent_pi.shape[0]
        base_mean_actions = self.base_action_net(latent_pi)
        mean_actions = torch.zeros(batch_size, self.output_size)
        mean_actions[:, self.out_base_mask] = base_mean_actions[:,
                                                                self.out_transfer_mask]

        return mean_actions
