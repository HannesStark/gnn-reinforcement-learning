from stable_baselines3.common.policies import register_policy
from NerveNet.policies.a2c_policies import ActorCriticGnnPolicy, ActorCriticMLPPolicyTransfer

GnnPolicy = ActorCriticGnnPolicy

register_policy("GnnPolicy", GnnPolicy)
register_policy("MLPTransferPolicy", ActorCriticMLPPolicyTransfer)
