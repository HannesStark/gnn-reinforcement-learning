import pybullet as p
import time
import pybullet_data
import math
import pprint
pp = pprint.PrettyPrinter(indent=4)

# setup physics simulation and environment
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -10)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
planeId = p.loadURDF("plane.urdf")


# setup robot
r2d2StartOrientation = p.getQuaternionFromEuler([0, 0, 0])

r2d2StartPos = [0.5, -0.5, 2]
r2d2_id = p.loadURDF("r2d2.urdf", r2d2StartPos, r2d2StartOrientation)


# lets get some information about the robot
num_joints = p.getNumJoints(r2d2_id)
# getJointInfo only returns a list... not very convenient :(
# so let's manually name the information that is returned...
joint_infos_names = ["jointIndex", "jointName", "jointType", "qIndex", "uIndex", "flags", "jointDamping", "jointFriction ", "jointLowerLimit ",
                     "jointUpperLimit ", "jointMaxForce ", "jointMaxVelocity ", "linkName ", "jointAxis", "parentFramePos", "parentFrameOrn", "parentIndex"]
joint_infos = [dict(zip(joint_infos_names, p.getJointInfo(r2d2_id, j)))
               for j in range(num_joints)]
joint_infos = {joint_info["jointName"]               : joint_info for joint_info in joint_infos}

print("Number of joints:", num_joints)
print("Joint information:")
pp.pprint(joint_infos)


### Controll the robot! ###

# First of all: Robots are controlled by controlling their joint motors
# Note: 1) Every joint as between 0 and 6 degrees of fredom
#       2) Not all joints have a motor. However by default
#          each revolute joint and prismatic joint is motorized
# Secondly, to control a given motor one must specify the controlMode:
# POSITION_CONTROL, VELOCITY_CONTROL, TORQUE_CONTROL and some others... 


for i in range(1000):
    # stepSimulation perform all actions (collision detection,
    # constraing solving, integration, etc.) in a single step.
    p.stepSimulation()
    # per default stepSimulation has a timestep of 1/240 seconds
    time.sleep(1./240.)


# finally we need to disconnect from
p.disconnect()
