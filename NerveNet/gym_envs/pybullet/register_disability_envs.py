from pybullet_envs import register

register(id='AntCpBackBulletEnv-v0',
         entry_point='NerveNet.gym_envs.pybullet.locomotion_gym:AntCpBackBulletEnv',
         max_episode_steps=1000,
         reward_threshold=2500.0)

register(id='AntCpLeftBackBulletEnv-v0',
         entry_point='NerveNet.gym_envs.pybullet.locomotion_gym:AntCpLeftBackBulletEnv',
         max_episode_steps=1000,
         reward_threshold=2500.0)

register(id='AntCpRightBackBulletEnv-v0',
         entry_point='NerveNet.gym_envs.pybullet.locomotion_gym:AntCpRightBackBulletEnv',
         max_episode_steps=1000,
         reward_threshold=2500.0)
