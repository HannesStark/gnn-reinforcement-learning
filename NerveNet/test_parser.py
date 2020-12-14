from graph_util.mujoco_parser import parse_mujoco_graph
from graph_util.mujoco_parser_settings import XML_DICT
import json
from pathlib import Path


def generate_graph_log(task_name: str):
    dump_dir = Path("graph_logs_new")
    dump_dir.mkdir(parents=True, exist_ok=True)
    graph = parse_mujoco_graph(task_name=task_name)

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

parse_mujoco_graph(task_name="AntBulletEnv-v0")
