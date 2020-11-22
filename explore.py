import gym
import pybullet as p
import time
import pybullet_data
import pybullet_envs

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setGravity(0, 0, -10)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
planeId = p.loadURDF('plane.urdf')
r2d2_start_pos = [0, 0, 1]
cheetah_start_pos = [1, 0, 1]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
r2d2_id = p.loadURDF('r2d2.urdf', r2d2_start_pos, cubeStartOrientation)
cheetah_id = p.loadURDF("mini_cheetah/mini_cheetah.urdf", cheetah_start_pos, cubeStartOrientation)
numJoints = p.getNumJoints(cheetah_id)
print(cheetah_id)
print(numJoints)
maxForce = 500
p.setRealTimeSimulation(1)

while (True):
    p.stepSimulation()
    time.sleep(1. / 240.)
    p.setJointMotorControl2(
        r2d2_id,
        0,
        p.VELOCITY_CONTROL,
        force=1.5,
        # maxVelocity=5
    )
