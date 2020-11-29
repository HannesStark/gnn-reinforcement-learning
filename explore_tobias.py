import pybullet as p
import time
import pybullet_data
import math

# PyBullet works with a client-server driven API
# so first we need to specify which physics server we're interested in
# Both GUI and DIRECT run the simulation/rendering in the same process as pybullet
# However DIRECT won't show a visual window / graphical user interface
# and it also doesn't allow access to OpenGL and VR hardware functions
# Additionally there is also SHARED_Memory as well as UDP, TCP which
# allows to connect to either an physics server instance of a different process
# on the same machine, or to a physics server over TCP or UDP networking.
physicsClient = p.connect(p.GUI)

# by default there is no graviational forace enable
p.setGravity(0, 0, -10)

# provide data files to use, like urdfs, picutes, etc.
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally

# The loadURDF will send a command to the physics server to load a physics model
# let's load a simple plane for stuff to stand/move on
planeId = p.loadURDF("plane.urdf")


# For representing positions pybullet uses cartesian coordinates [x, y, z]
# For representing orientation/rotation it uses quaterions [x,y,z,w]
# However, because these are not intuitive using getQuaternionFromEuler
# and getEulerFromQuaternion euler angles [yaw, pitch, roll]
# or a 3x3 matric can also be used
# Note: euler angles are provided in radients not degree
cubeStartOrientation = p.getQuaternionFromEuler([0, 1/6 * math.pi, 0])

cubeStartPos = [0, 0, 1]
boxId = p.loadURDF("r2d2.urdf", cubeStartPos, cubeStartOrientation)

# By default the pyhsics server doesnt do a step unless explicitly requestes
# with stepSimulation. But you can also run a simulation in real time by letting
# the server automatically step the simulation according to its real-time-clock:
# p.setRealTimeSimulation(1)

for i in range(1000):
    # stepSimulation perform all actions (collision detection,
    # constraing solving, integration, etc.) in a single step.
    p.stepSimulation()
    # per default stepSimulation has a timestep of 1/240 seconds
    time.sleep(1./240.)

    cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)


# finally we need to disconnect from
p.disconnect()
