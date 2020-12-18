"""
    
    @author:
        Tobias Schmidt, Hannes Stark
"""

from enum import IntEnum, Enum


ALLOWED_NODE_TYPES = ["root", "joint", "body"]
EDGE_TYPES = {
    ("root", "root"): 0,
    ("joint", "joint"): 1,
    ("joint", "body"): 6,
    ("body", "joint"): -6,
    ("root", "joint"): 9,
    ("joint", "root"): -9,
    ("root", "body"): 10,
    ("body", "root"): -10,
    ("body", "body"): 3
}


XML_DICT = {"Walker2DBulletEnv-v0": "walker2d.xml",
            "HalfCheetahBulletEnv-v0": "half_cheetah.xml",
            "AntBulletEnv-v0": "ant.xml",
            "HopperBulletEnv-v0": "hopper.xml",
            "HumanoidBulletEnv-v0": "humanoid_symmetric.xml",
            "HumanoidFlagrunBulletEnv-v0": "humanoid_symmetric.xml",
            "HumanoidFlagrunHarderBulletEnv-v0": "humanoid_symmetric.xml"}


class ControllerType(IntEnum):
    SHARED = 0,
    SEPERATE = 1,
    UNIFIED = 2


class RootRelationOption(IntEnum):
    NONE = 0,
    BODY = 1,
    ALL = 2,
