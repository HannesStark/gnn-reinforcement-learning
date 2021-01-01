"""
    @author:
        Tobias Schmidt, Hannes Stark
"""

from enum import IntEnum


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
    ("root", "root"): 1,  # and for self loops
    ("joint", "joint"): 2,
    ("joint", "body"): 3,
    ("body", "joint"): -3,
    ("root", "joint"): 4,
    ("joint", "root"): -4,
    ("root", "body"): 5,
    ("body", "root"): -5,
    ("body", "body"): 6
}


XML_DICT = {"Walker2DBulletEnv-v0": "walker2d.xml",
            "HalfCheetahBulletEnv-v0": "half_cheetah.xml",
            "AntBulletEnv-v0": "ant.xml",
            "HopperBulletEnv-v0": "hopper.xml",
            "HumanoidBulletEnv-v0": "humanoid_symmetric.xml",
            "HumanoidFlagrunBulletEnv-v0": "humanoid_symmetric.xml",
            "HumanoidFlagrunHarderBulletEnv-v0": "humanoid_symmetric.xml"}


SHARED_EMBEDDING_GROUPS = ["root_mujocoroot",
                           "aux",
                           "cart",
                           "pole",
                           "torso",
                           "pelvis",
                           "updown",
                           "leg",
                           "foot",
                           "f_",
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
                           "neck",
                           "head",
                           "tail",
                           "shin",
                           "pod",
                           "slider",
                           "rot",
                           "mid",
                           "back",
                           "body",
                           "joint",
                           "root",  # for joints named rootx, rooty and so on
                           "ignore"
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
