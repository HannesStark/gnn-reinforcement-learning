from graph_util.mujoco_parser_nervenet import parse_mujoco_graph, OB_MAP, XML_DICT
import json
from pathlib import Path


def generate_graph_log(task_name: str):
    assert task_name in OB_MAP.keys(
    ), f"{task_name} is not yet supported, try one of the following: " + str(OB_MAP.keys())

    dump_dir = Path("graph_logs")
    dump_dir.mkdir(parents=True, exist_ok=True)

    # to be able to load the MUJOXO xml definitions you need to point to your NerveNet repository
    nervenet_dir = Path("C:\\Users\\tsbau\\git\\NerveNet")

    graph = parse_mujoco_graph(xml_name=XML_DICT[task_name], xml_assets_path=(
        nervenet_dir / "environments" / "assets"))

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
    for key in graph['node_parameters'].keys():
        graph['node_parameters'][key] = graph['node_parameters'][key].tolist()

    with open(str(dump_dir / f"{task_name}.json"), 'w') as f:
        json.dump(graph, f, indent=4)


for task_name in XML_DICT.keys():
    try:
        generate_graph_log(task_name)
    except:
        print(f"Could not generate graph log for {task_name}.")
