import os

from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnRewardThreshold
import numpy as np


class LoggingCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, verbose=0, logpath=None):
        super(LoggingCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.saved_reward = []
        self.rolling_reward = []
        self.every_step_outfile = open(os.path.join(logpath, 'rolling_mean_every_step.txt'), 'w+')
        self.rolling_outfile = open(os.path.join(logpath, 'rolling_mean.txt'), 'w+')
        self.outfile = open(os.path.join(logpath, 'rollout_mean.txt'), 'w+')

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        self.saved_reward.append(self.locals['rewards'])
        self.rolling_reward.append(self.locals['rewards'])
        self.every_step_outfile.write(str(np.array(self.rolling_reward).mean() * 1000) + '\n')
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self.rolling_outfile.write(str(np.array(self.rolling_reward).mean() * 1000) + '\n')
        self.rolling_outfile.flush()

        mean_reward = np.array(self.saved_reward).mean() * 1000
        self.outfile.write(str(mean_reward) + '\n')
        self.outfile.flush()
        self.saved_reward = []
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
