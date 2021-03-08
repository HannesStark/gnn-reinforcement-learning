from stable_baselines3.common.policies import register_policy
from NerveNet.policies.a2c_policies import ActorCriticGnnPolicy, ActorCriticGnnPolicy_V0,  ActorCriticMLPPolicyTransfer

register_policy("GnnPolicy", ActorCriticGnnPolicy)
register_policy("GnnPolicy_V0", ActorCriticGnnPolicy_V0)
register_policy("MLPTransferPolicy", ActorCriticMLPPolicyTransfer)
