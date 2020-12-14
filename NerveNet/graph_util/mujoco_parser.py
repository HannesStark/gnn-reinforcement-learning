
"""
    Some helper functions to parse the mujoco xml template files

    @author:
        Tobias Schmidt, Hannes Stark, modified from the code of Tingwu Wang.
"""


import os
import logging
import numpy as np
import pybullet_data

from pathlib import Path
from enum import IntEnum, Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from bs4 import BeautifulSoup

from graph_util.mujoco_parser_settings import XML_DICT, ALLOWED_NODE_TYPES, NodeType, EdgeType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = ["parse_mujoco_graph"]

XML_ASSETS_DIR = Path(pybullet_data.getDataPath()) / "mjcf"


def parse_mujoco_graph(task_name: str = None,
                       xml_name: str = None,
                       xml_assets_path: Path = None):
    '''
    TODO: add documentation

    Parameters:
        task_name:
            The name of the task to parse the graph structure from.
            Takes priority over xml_name.
        xml_name:
            The name of the xml file to parse the graph structure from.
            Either xml_name or task_name must be specified.
        xml_assets_path:
            Specifies in which directory to look for the mujoco (mjcf) xml file.
            If none, default will be set to the pybullet data path.

    Returns:
        A dictionary containing details about the parsed graph structure:
            "tree": dict
                A dictionary representation of the parsed XML
            "relation_matrix": ndarray of shape (num_nodes, num_nodes)
                A representation of the adjacency matrix for the parsed graph.
                Non-Zero entries are edge conections of different types as defined by EdgeType
            "node_type_dict": dict
                Keys: The names of the node types
                Values: The list of the node ids that are of this node type
            "output_type_dict": dict
                Keys: The prefix of the motor names (e.g. "hip", "ankle")
                Values: The list of the node ids that are of this motor type
            "output_list": list
                The list of node ids that correspond to each of the motors.
                The order exactly matches the motor order  specified in the xml file.
            "input_dict": dict
                Keys: node ids
                Value: list of ids in observation vector that belong to the node
            "node_parameters"


    '''

    if task_name is not None:  # task_name takes priority
        xml_name = XML_DICT[task_name]

    assert xml_name is not None, "Either task_name or xml_name must be given."

    if xml_assets_path is None:
        xml_assets_path = XML_ASSETS_DIR

    xml_path = xml_assets_path / xml_name

    with open(str(xml_path), "r") as xml_file:
        xml_soup = BeautifulSoup(xml_file.read(), "xml")

    _extract_tree(xml_soup)

    return dict(tree=tree,
                relation_matrix=relation_matrix,
                node_type_dict=node_type_dict,
                output_type_dict=output_type_dict,
                input_dict=input_dict,
                output_list=output_list,
                debug_info=debug_info,
                node_parameters=node_parameters,
                para_size_dict=para_size_dict,
                num_nodes=len(tree))


def _extract_tree(xml_soup: BeautifulSoup,
                  allowed_node_types: List[str] = ALLOWED_NODE_TYPES):
    '''
    TODO: Add docstring
    '''
    tree = []
    tree_id = 0

    motor_names = _get_motor_names(xml_soup)
    robot_body_soup = xml_soup.find("worldbody").find("body")

    root_joints = robot_body_soup.find_all('joint', recursive=False)

    tree.append({"type": "root",
                 "is_output_node": False,
                 "name": "root_mujocoroot",
                 "neighbour": [],
                 "id": tree_id,
                 "info": robot_body_soup.attrs,
                 "attached_joint_name": [j["name"] for j in root_joints if j["name"] not in motor_names],
                 "attached_joint_info": []
                 })
    # TODO: Add observation size information

    sub_tree = _unpack_node(robot_body_soup, parent_id=0)


def _unpack_node(node: BeautifulSoup,
                 id: int,
                 parent_id: int,
                 motor_names: List[str],
                 allowed_node_types: List[str] = ALLOWED_NODE_TYPES) -> list:
    '''
    TODO: Add docstring
    '''
    sub_tree = [{
        "type": node.name,
        "is_output_node": node["name"] in motor_names,
        "raw_name": node["name"],
        "name": node.name + node["name"],
        "id": id,
        "parent": parent_id,  # to be set later
        "info": node.attrs
    }]

    child_soups = [child
                   for child in node.find_all(node_type, recursive=False)
                   for node_type in allowed_node_types]
    next_id = id + 1
    for child in child_soups:
        sub_tree = _unpack_node(child,
                                id=next_id,
                                parent_id=id,
                                motor_names=motor_names,
                                allowed_node_types=allowed_node_types)
        next_id = max([node["id"] for node in sub_tree]) + 1

    return []


def _get_motor_names(xml_soup):
    motors = xml_soup.find('actuator').find_all('motor')
    name_list = [motor['joint'] for motor in motors]
    return name_list


if __name__ == "__main__":
    assert False
