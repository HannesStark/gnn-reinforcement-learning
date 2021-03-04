
"""
    Some helper functions to parse the mujoco xml template files

    @author:
        Tobias Schmidt, Hannes Stark, modified from the code of Tingwu Wang.
"""

import numpy as np
import pybullet_data
import pybullet_envs  # register pybullet envs from bullet3

from pathlib import Path
from enum import IntEnum, Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from bs4 import BeautifulSoup


from NerveNet.graph_util.mujoco_parser_nervenet import XML_DICT as NERVENET_XML_DICT
from NerveNet.graph_util.mujoco_parser_settings import XML_DICT as PYBULLET_XML_DICT
from NerveNet.graph_util.mujoco_parser_settings import ALLOWED_NODE_TYPES, SUPPORTED_JOINT_TYPES,\
    SUPPORTED_JOINT_ATTRIBUTES, SUPPORTED_BODY_ATTRIBUTES, EDGE_TYPES, SHARED_EMBEDDING_GROUPS, \
    CUSTOM_SHARED_EMBEDDING_GROUPS, ControllerOption, RootRelationOption, EmbeddingOption


__all__ = ["parse_mujoco_graph"]

XML_ASSETS_DIR = Path(pybullet_data.getDataPath()) / "mjcf"


def parse_mujoco_graph(task_name: str = None,
                       xml_name: str = None,
                       xml_assets_path: Path = None,
                       use_sibling_relations: bool = True,
                       root_relation_option: RootRelationOption = RootRelationOption.BODY,
                       controller_option: ControllerOption = ControllerOption.SHARED,
                       embedding_option: EmbeddingOption = EmbeddingOption.SHARED,
                       foot_list: List[str] = [],
                       absorb_root_joints: bool = True):
    '''
    TODO: add documentation

    Parameters:
        "task_name":
            The name of the task to parse the graph structure from.
            Takes priority over xml_name.
        "xml_name":
            The name of the xml file to parse the graph structure from.
            Either xml_name or task_name must be specified.
        "xml_assets_path":
            Specifies in which directory to look for the mujoco (mjcf) xml file.
            If none, default will be set to the pybullet data path.
        "use_sibling_relations":
            Whether to use sibling relations between nodes to build the relation (adjacency) matrix
            or to only use parent-child relationships
        "root_relation_option":
                Specifies which other nodes the root node should additionally be connected to:
                    - RootRelationOption.NONE:
                        No other nodes
                    - RootRelationOption.BODY:
                        All nodes of type "body" are connected to root
                    - RootRelationOption.ALL:
                        All nodes are connected to root

    Returns:
        A dictionary containing details about the parsed graph structure:
            "tree": dict
                A dictionary representation of the parsed XML
            "relation_matrix": ndarray of shape (num_nodes, num_nodes)
                A representation of the adjacency matrix for the parsed graph.
                Non-Zero entries are edge conections of different types as defined by EDGE_TYPES
            "node_type_dict": dict
                Keys: The names of the node types
                Values: The list of the node ids that are of this node type
            "output_type_dict": dict
                Specifies which outputs should use the same controller network
                Keys: The name of the control group (e.g. "hip", "ankle")
                Values: The list of the node ids that are of this motor type
            "output_list": list
                The list of node ids that correspond to each of the motors.
                The order exactly matches the motor order specified in the xml file.
            "input_dict": dict
                Keys: node ids
                Value: list of ids in observation vector that belong to the node
            "node_parameters"
    '''

    if task_name is not None:  # task_name takes priority
        if task_name in NERVENET_XML_DICT:
            xml_name = NERVENET_XML_DICT[task_name]
        elif task_name in PYBULLET_XML_DICT:
            xml_name = PYBULLET_XML_DICT[task_name]
        else:
            raise NotImplementedError(f"No task named {task_name} defined.")

        if len(foot_list) == 0 and task_name in pybullet_envs.registry.env_specs:
            # if we have the task_name we can try to get the foot_list from the gym environment
            task_env = pybullet_envs.gym.make(task_name)
            if isinstance(task_env, pybullet_envs.gym.Wrapper):
                task_env = task_env.env
            robot = task_env.robot
            if hasattr(robot, "foot_list"):
                foot_list = robot.foot_list

    assert xml_name is not None, "Either task_name or xml_name must be given."

    if xml_assets_path is None:
        xml_assets_path = XML_ASSETS_DIR

    xml_path = xml_assets_path / xml_name

    with open(str(xml_path), "r") as xml_file:
        xml_soup = BeautifulSoup(xml_file.read(), "xml")

    tree = __extract_tree(xml_soup, foot_list, absorb_root_joints)

    relation_matrix = __build_relation_matrix(
        tree,
        use_sibling_relations=use_sibling_relations,
        root_relation_option=root_relation_option)

    # group nodes by node type
    node_type_dict = {node_type: [node["id"]
                                  for node in tree if node["type"] == node_type]
                      for node_type in ALLOWED_NODE_TYPES}
    print(node_type_dict)

    output_type_dict, output_list = __get_output_mapping(
        tree,
        controller_option=controller_option)

    obs_input_mapping, static_input_mapping, input_type_dict = __get_input_mapping(
        task_name,
        tree,
        embedding_option=embedding_option)

    num_nodes = len(obs_input_mapping)

    return dict(tree=tree,
                relation_matrix=relation_matrix,
                node_type_dict=node_type_dict,
                output_type_dict=output_type_dict,
                output_list=output_list,
                obs_input_mapping=obs_input_mapping,
                static_input_mapping=static_input_mapping,
                input_type_dict=input_type_dict,
                num_nodes=num_nodes)


def __extract_tree(xml_soup: BeautifulSoup,
                   foot_list: List[str],
                   absorb_root_joints: bool):
    '''
    TODO: Add docstring
    '''

    motor_names = __get_motor_names(xml_soup)
    default_body_soup = xml_soup.find("default").find("body")
    default_joint_soup = xml_soup.find("default").find("joint")
    robot_body_soup = xml_soup.find("worldbody").find("body")

    root_joints = robot_body_soup.find_all('joint', recursive=False)

    root_node = {"type": "root",
                 "is_output_node": False,
                 "name": "root_mujocoroot",
                 "raw_name": "root_mujocoroot",
                 "neighbour": [],
                 "id": 0,
                 "parent": 0,
                 "info": robot_body_soup.attrs,
                 "attached_joint_name": ["joint_" + j["name"] for j in root_joints if j["name"] not in motor_names],
                 "attached_joint_info": [],
                 "motor_names": motor_names,
                 "foot_list": foot_list,
                 "default": {
                     "body": {} if default_body_soup is None else default_body_soup.attrs,
                     "joint": {} if default_joint_soup is None else default_joint_soup.attrs,
                 }
                 }

    tree = list(__unpack_node(robot_body_soup,
                              current_tree={0: root_node},
                              parent_id=0,
                              motor_names=motor_names,
                              foot_list=foot_list).values())

    if absorb_root_joints:
        # absorb joints that are directly attached to the root node into the root node
        # we need to absorb the root joints because they are usually "dead" joints
        # meaning, they do not produce any observations. If we keep them in anyways,
        # we'd expect to get a bigger observation vector than we actually do.

        tree[0]["attached_joint_info"] = [node
                                          for node in tree
                                          if node["name"] in tree[0]["attached_joint_name"]]
        tree = [node
                for node in tree
                if node["name"] not in tree[0]["attached_joint_name"]]

        # after absorbing the root joints we need to adjust the indicies accordingly
        index_offset = len(tree[0]["attached_joint_name"])
        for i in range(len(tree)):
            # apply only for indicies that came after the ones that were removed
            if tree[i]["id"] - index_offset > 0:
                tree[i]["id"] = tree[i]["id"] - index_offset
            if tree[i]["parent"] - index_offset > 0:
                tree[i]["parent"] = tree[i]["parent"] - index_offset

    return tree


def __unpack_node(node: BeautifulSoup,
                  current_tree: dict,
                  parent_id: int,
                  motor_names: List[str],
                  foot_list: List[str]) -> dict:
    '''
    This function is used to recursively unpack the xml graph structure of a given node.

    Parameters:
        node:
            The root node from which to unpack the underlying xml tree into a dictionary.
        current_tree:
            The dictionary represenation of the tree that has already been unpacked.
            - Required to determin the new id to use for the current node.
            - Will be updated with the root node and its decendent nodes.
        parent_id:
            The id of the parent node for the current node.
        motor_names:
            The list of joint names that are supposed be output_nodes.

    Returns:
        A dictionary representation of the xml tree rooted at the given node
        with "id" and "parent_id" references to encode the relationship structure.
    '''
    id = max(current_tree.keys()) + 1
    node_type = node.name

    current_tree[id] = {
        "type": node_type,
        "is_output_node": node["name"] in motor_names,
        "is_foot": node["name"] in foot_list,
        "raw_name": node["name"],
        "name": node_type + "_" + node["name"],
        "id": id,
        "parent": parent_id,  # to be set later
        "info": node.attrs
    }

    if current_tree[id]["is_foot"]:
        current_tree[id]["foot_id"] = foot_list.index(node["name"])

    if current_tree[id]["type"] == "body":
        geoms = node.find_all('geom', recursive=False)
        current_tree[id].update({"geoms": [geom.attrs for geom in geoms]})

    child_soups = [child
                   for allowed_type in ALLOWED_NODE_TYPES
                   for child in node.find_all(allowed_type, recursive=False)
                   ]

    for child in child_soups:
        current_tree.update(__unpack_node(child,
                                          current_tree=current_tree,
                                          parent_id=id,
                                          motor_names=motor_names,
                                          foot_list=foot_list)
                            )

    return current_tree


def __get_motor_names(xml_soup: BeautifulSoup) -> List[str]:
    motors = xml_soup.find('actuator').find_all('motor')
    name_list = [motor['joint'] for motor in motors]
    return name_list


def __build_relation_matrix(tree: List[dict],
                            use_sibling_relations: bool,
                            root_relation_option: RootRelationOption) -> np.ndarray:
    '''
    TODO better docstring

    Parameters:
        "tree":
            The dictionary representation of the parsed XML to build the relation matrix from.
        "use_sibling_relations":
            Whether to use sibling relations between nodes to build the relation (adjacency) matrix
            or to only use parent-child relationships
        "root_relation_option":
            The root node is an abstract node and not part of the kinematic tree described by the XML structure.
            Therefore we need to define which nodes it is in relation/connected to.
            At a minimum it should have the same relations as the main body (e.g. "torso") node does.
            With this option we can additionally specify which other nodes it should also be connected to:
                - RootRelationOption.NONE:
                    No other nodes
                - RootRelationOption.BODY:
                    All nodes of type "body" are connected to root
                - RootRelationOption.ALL:
                    All nodes are connected to root
    Returns:
        "relation_matrix": ndarray of shape (num_nodes, num_nodes)
                A representation of the adjacency matrix for the parsed graph.
                Non-Zero entries are edge conections of different types as defined by EDGE_TYPES

    '''
    num_node = len(tree)
    relation_matrix = np.zeros([num_node, num_node], dtype=np.int)

    # nodes with outgoing edge
    for node_out in tree:
        # nodes with incoming edge
        for node_in in tree:
            if node_in["parent"] == node_out["id"]:
                # direction out -> in
                relation_matrix[node_out["id"]][node_in["id"]] = EDGE_TYPES[(
                    node_out["type"], node_in["type"])]
                # direction in -> out
                relation_matrix[node_in["id"]][node_out["id"]] = EDGE_TYPES[(
                    node_in["type"], node_out["type"])]

    # always connect the root to its grand-children, but not to its children
    # for whatever reasons... again that's just how the reference implementation works
    root_children_ids = [node["id"]
                         for node in tree
                         if node["parent"] == 0
                         and node["id"] != 0]
    # we only care about the children of type body
    root_grandchildren = [node
                          for node in tree
                          if node["parent"] in root_children_ids
                          and node["type"] == "body"]

    # disconnect root with its children
    for child_node_id in root_children_ids:
        relation_matrix[child_node_id][0] = 0
        relation_matrix[0][child_node_id] = 0

    # connect root with its grand-children
    for grandchild_node in root_grandchildren:
        relation_matrix[grandchild_node["id"]][0] = EDGE_TYPES[(
            grandchild_node["type"], "root")]
        relation_matrix[0][grandchild_node["id"]] = EDGE_TYPES[(
            "root", grandchild_node["type"])]

    if use_sibling_relations:
        sibling_pairs = [(node_1, node_2)
                         for node_1 in tree
                         for node_2 in tree
                         if node_1["parent"] == node_2["parent"]
                         and node_1["id"] != node_2["id"]
                         and node_1["type"] != "root"
                         and node_2["type"] != "root"]
        for node_1, node_2 in sibling_pairs:
            relation_matrix[node_1["id"]][node_2["id"]] = EDGE_TYPES[(
                node_1["type"], node_2["type"])]
            relation_matrix[node_2["id"]][node_1["id"]] = EDGE_TYPES[(
                node_2["type"], node_1["type"])]

    if root_relation_option != RootRelationOption.NONE:
        if root_relation_option == RootRelationOption.ALL:
            root_relation_nodes = tree
        elif root_relation_option == RootRelationOption.BODY:
            root_relation_nodes = [node
                                   for node in tree
                                   if node["type"] == "body"]
        else:
            raise NotImplementedError("Unknown root relation option")

        for node in root_relation_nodes:
            if node["type"] != "root":
                relation_matrix[node["id"]][0] = EDGE_TYPES[(
                    node["type"], "root")]
                relation_matrix[0][node["id"]] = EDGE_TYPES[(
                    "root", node["type"])]

    return relation_matrix


def __get_output_mapping(tree: List[dict], controller_option: ControllerOption) -> Tuple[dict, List[int]]:
    '''
    Parameters:
        "tree":
            The dictionary representation of the parsed XML to extract the output mapping from.
        "controller_option":
            Wang et al. propose different sharing options for the controller networks that generate
            the action from the final hidden presentation of actuator/motor/output nodes.
                - shared
                    The same controller network for the same/similar type of motors TODO: specify what exactly is "similiar"
                - seperate
                    Different controller networks for every output
                - unified:
                    The same controller network for all outputs
    Returns:
        "output_type_dict": dict
            A mapping of output nodes into controll groups.
            Nodes of the same controll group are supposed to share the same controller network.
            Keys: The identifier of the group
            Values: The list of nodes of this group
        "output_list": list[str]
            The list of all output nodes in the order
    '''

    # we need to use the exact same order as motors are defined in the xml
    # --> names in the correct order are in root node
    # --> indexing first improves sorting afterwards
    motor_index_map = {v: i for i, v in enumerate(tree[0]["motor_names"])}

    output_nodes = {node["raw_name"]: node
                    for node in tree if node["is_output_node"]}
    output_nodes = dict(sorted(output_nodes.items(),
                               key=lambda pair: motor_index_map[pair[0]]))

    output_list = [node["id"] for node in output_nodes.values()]

    # TODO currently we just type/group together based on the name prefix....
    # either we need to document this as "feature" or find a better way to do this
    joint_types = set([node_name.split("_")[0]
                       for node_name in output_nodes.keys()])

    if controller_option == ControllerOption.SHARED:
        output_type_dict = {
            joint_type: [node["id"]
                         for node_name, node in output_nodes.items() if joint_type in node_name]
            for joint_type in joint_types
        }
    elif controller_option == ControllerOption.SEPERATE:
        output_type_dict = {node["raw_name"]: node["id"]
                            for node in output_nodes}
    elif controller_option == ControllerOption.UNIFIED:
        output_type_dict['unified'] = output_list
    else:
        raise NotImplementedError(
            "Unknown controller option: %s" % controller_option)

    return output_type_dict, output_list


def __get_input_mapping(task_name: str, tree: List[dict], embedding_option: EmbeddingOption) -> Tuple[dict, dict, dict]:
    '''
    Parameters:
        "tree":
            The dictionary representation of the parsed XML to extract the output mapping from.
        "embedding_option":
                - shared
                    The same embedding function for the same/similar type of nodes TODO: specify what exactly is "similiar"
                - seperate
                    Different embedding functions for every output
                - unified:
                    The same embedding function for all outputs
    '''
    # mapping from node ids to parts of the observations vector
    obs_input_mapping = {}
    # mapping from node ids to static structural information (as defined by
    # the kinematic tree) / fixed node features
    static_input_mapping = {}
    # grouping of nodes that use the same embedding function
    input_type_dict = {}

    # the joint observations we get from the environment are in the same order as the joints are defined in the kinematic tree
    # to be able to more easily map them we enumerate them again, because except for the order we don't know anything else about
    # the ids of the node joints
    joints = dict(enumerate([node
                             for node in tree
                             if node["type"] == "joint"]))
    # the same goes for body nodes
    bodies = dict(enumerate([node
                             for node in tree
                             if node["type"] == "body"]))

    # this is how many observations we have for the root node
    # the observations for the joints follow after these
    # TODO: varify that this can indeed be a constant
    _root_obs_size = 8
    _num_obs_per_joint = 2
    # we can immediately set the first observations to be used by the root node
    obs_input_mapping[0] = list(range(0, _root_obs_size))

    for joint_key, node in joints.items():
        # the id of the observations vector where the observations for this joint start from
        obs_idx_offset = _root_obs_size + joint_key * _num_obs_per_joint
        # in case we'll add more complex joints later that don't have scalar observations use a list here
        # joint position
        position_obs = [obs_idx_offset]
        # angular velocity
        velocity_obs = [obs_idx_offset + 1]
        obs_input_mapping[node["id"]] = position_obs + velocity_obs

        # get default values for attributes and update them with node attributes
        # overwriting the default attributes if neccessary
        attrs = tree[0]["default"]["joint"].copy()
        attrs.update(node["info"])
        static_input_mapping[node["id"]] = {attr_name: __format_attr(attrs[attr_name])
                                            for attr_name in SUPPORTED_JOINT_ATTRIBUTES
                                            if attr_name in attrs.keys()}

    # this is how many observations we have for all joint nodes combined
    joints_obs_size = sum([len(obs_input_mapping[node["id"]])
                           for _, node in joints.items()])

    for _, node in bodies.items():
        # for pybullet envs there are no observations generated for body nodes
        # except the root observations
        obs_input_mapping[node["id"]] = []

        # and except for body nodes which are feets
        if node["is_foot"]:
            # every foot has one boolean observation to tell whether the foot is on the ground
            obs_input_mapping[node["id"]] += [_root_obs_size +
                                              joints_obs_size + node["foot_id"]]

        # get default values for attributes and update them with node attributes
        # overwriting the default attributes if neccessary
        attrs = tree[0]["default"]["body"].copy()
        attrs.update(node["info"])
        static_input_mapping[node["id"]] = {attr_name: __format_attr(attrs[attr_name])
                                            for attr_name in SUPPORTED_JOINT_ATTRIBUTES
                                            if attr_name in attrs.keys()}

    if embedding_option == EmbeddingOption.SHARED:
        if task_name in CUSTOM_SHARED_EMBEDDING_GROUPS.keys():
            for node_type, node_names in CUSTOM_SHARED_EMBEDDING_GROUPS[task_name].items():
                input_type_dict[node_type] = [node["id"]
                                              for node in tree
                                              if node["name"] in node_names
                                              # make sure we don't already have this id in another group
                                              and node["id"] not in [i for l in input_type_dict.values() for i in l]]
        else:
            for node_type in SHARED_EMBEDDING_GROUPS:
                input_type_dict[node_type] = [node["id"]
                                              for node in tree
                                              if node_type in node["raw_name"].lower()
                                              # make sure we don't already have this id in another group
                                              and node["id"] not in [i for l in input_type_dict.values() for i in l]]

        # drop empty groups / only keep non-empty groups
        input_type_dict = {group_name: group_nodes
                           for group_name, group_nodes in input_type_dict.items()
                           if len(group_nodes) > 0}

    elif embedding_option == EmbeddingOption.SEPERATE:
        input_type_dict = {node["raw_name"]: node["id"]
                           for node in tree}
    elif embedding_option == EmbeddingOption.UNIFIED:
        input_type_dict['unified'] = [node["id"] for node in tree]
    else:
        raise NotImplementedError(
            "Unknown embedding option: %s" % embedding_option)

    # verify that we actually assign every node here into some group
    assert sum([len(node_list) for node_list in input_type_dict.values()]) == len(
        tree), "Every node must be assigned to exactly one group!"
    # also verify that we have mappings for every noed
    assert len(obs_input_mapping) == len(
        tree), "Every node must have an observation input mapping!"
    assert len(static_input_mapping) == len(
        tree) - 1, "Every node must have a static input mapping, except the root node!"

    for nodes in input_type_dict.values():
        assert all([len(obs_input_mapping[nodes[0]]) == len(obs_input_mapping[node])
                    for node in nodes]), "Every node in a group must have the same number of observations"

    return obs_input_mapping, static_input_mapping, input_type_dict


def __format_attr(slist: str) -> List[float]:
    '''
        Attributes in xml mujoco files may contain integer, floats, booleans and arithmetic expressions.
        This function converts these attribute strings into lists of floats
    '''
    # kinda hacky way to convert string to float
    # by evaluating the sub-strings as python expressions

    # Adding true and false as variables so the strings may contain lowercase true/false
    # which will be interpreted as calls to these variables
    true = True
    false = False
    return list(map(float, map(eval, slist.split(" "))))


if __name__ == "__main__":
    assert False
