import os
import json
from pathlib import Path

from NerveNet.graph_util.mujoco_parser_settings import XML_DICT as PYBULLET_XML_DICT
from NerveNet.graph_util.mujoco_parser_nervenet import XML_DICT as NERVENET_XML_DICT


def generate_graph_log(task_name: str,
                       xml_assets_path: Path = None,
                       dump_dir: Path = Path("logs_parser"),
                       use_nervenet_parser: bool = False):

    dump_dir.mkdir(parents=True, exist_ok=True)

    if use_nervenet_parser:
        from graph_util.mujoco_parser_nervenet import parse_mujoco_graph
    else:
        from graph_util.mujoco_parser import parse_mujoco_graph

    graph = parse_mujoco_graph(
        task_name=task_name, xml_assets_path=xml_assets_path)

    # make bs4.element.Tag type of xml document serializable
    for i in range(len(graph["tree"])):
        if "raw" in graph["tree"][i]:
            graph["tree"][i]["raw"] = str(graph["tree"][i]["raw"])
            if "attached_joint_info" in graph["tree"][i]:
                for j in range(len(graph["tree"][i]["attached_joint_info"])):
                    graph["tree"][i]["attached_joint_info"][j]["raw"] = str(
                        graph["tree"][i]["attached_joint_info"][j]["raw"])

    # make numpy array serializable
    graph['relation_matrix'] = graph['relation_matrix'].tolist()
    with open(str(dump_dir / f"{task_name}.json"), 'w') as f:
        json.dump(graph, f, indent=4)
