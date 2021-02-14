from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv
from NerveNet.gym_envs.pybullet import locomotion_robots


class AntCpBackBulletEnv(WalkerBaseBulletEnv):

    def __init__(self, render=False):
        self.robot = locomotion_robots.Ant_Crippled_BackLegs()
        WalkerBaseBulletEnv.__init__(self, self.robot, render)


class AntCpLeftBackBulletEnv(WalkerBaseBulletEnv):

    def __init__(self, render=False):
        self.robot = locomotion_robots.Ant_Crippled_LeftBackLeg()
        WalkerBaseBulletEnv.__init__(self, self.robot, render)


class AntCpRightBackBulletEnv(WalkerBaseBulletEnv):

    def __init__(self, render=False):
        self.robot = locomotion_robots.Ant_Crippled_RightBackLeg()
        WalkerBaseBulletEnv.__init__(self, self.robot, render)
