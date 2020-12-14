"""
    
    @author:
        Tobias Schmidt, Hannes Stark
"""

from enum import IntEnum, Enum


class NodeType:
    ROOT = 'root'
    JOINT = 'joint'
    BODY = 'body'


class EdgeType(IntEnum):
    JOINT_JOINT = 1
    JOINT_BODY = 6
    BODY_JOINT = -6
    BODY_BODY = 3
    TENDON = 4
    ROOT_JOINT = 9
    JOINT_ROOT = -9
    ROOT_BODY = 10
    BODY_ROOT = -10


XML_DICT = {"Walker2DBulletEnv-v0": "walker2d.xml",
            "HalfCheetahBulletEnv-v0": "half_cheetah.xml",
            "AntBulletEnv-v0": "ant.xml",
            "HopperBulletEnv-v0": "hopper.xml",
            "HumanoidBulletEnv-v0": "humanoid_symmetric.xml",
            "HumanoidFlagrunBulletEnv-v0": "humanoid_symmetric.xml",
            "HumanoidFlagrunHarderBulletEnv-v0": "humanoid_symmetric.xml"}

ALLOWED_NODE_TYPES = [NodeType.ROOT, NodeType.JOINT, NodeType.BODY]
