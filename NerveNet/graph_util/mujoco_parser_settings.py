"""
    
    @author:
        Tobias Schmidt, Hannes Stark
"""

from enum import IntEnum, Enum


ALLOWED_NODE_TYPES = ["root", "joint", "body"]

# ["ball", "slide", "free"] are not supported
# to be able to support them, we'd need to adjust the input_mapping
# of the parser to take into account that these different types have
# different degrees of freedom
# TODO: We need to add support for slide in order to use the halfCheetah env
SUPPORTED_JOINT_TYPES = ["hinge"]

# fixed joint attributes that we support
SUPPORTED_JOINT_ATTRIBUTES = ["armature",
                              "damping",
                              "limited",
                              "axis",
                              "pos",
                              "range",
                              "stiffness"]


# fixed body attributes that we support
SUPPORTED_BODY_ATTRIBUTES = ["pos"]

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


SHARED_EMBEDDING_GROUPS = ["root",
                           "aux",
                           "cart",
                           "pole",
                           "torso",
                           "pelvis",
                           "updown",
                           "leg",
                           "foot",
                           "hip",
                           "ankle",
                           "knee",
                           "thigh",
                           "arm",
                           "elbow",
                           "waist",
                           "shoulder",
                           "hand",
                           "abdomen",
                           "tail",
                           "shin",
                           "pod",
                           "mid",
                           "back",
                           "f_"
                           ]


class ControllerOption(IntEnum):
    SHARED = 0,
    SEPERATE = 1,
    UNIFIED = 2


class EmbeddingOption(IntEnum):
    SHARED = 0,
    SEPERATE = 1,
    UNIFIED = 2


class RootRelationOption(IntEnum):
    NONE = 0,
    BODY = 1,
    ALL = 2,
