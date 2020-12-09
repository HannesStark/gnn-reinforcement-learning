from graph_util.mujoco_parser import parse_mujoco_graph
import json
from pathlib import Path

task_name = "AntS-v1"
dump_dir = Path("graph_logs")
dump_dir.mkdir(parents=True, exist_ok=True)
graph = parse_mujoco_graph(task_name=task_name)

# make bs4.element.Tag type of xml document serializable
for i in range(len(graph["tree"])):
    graph["tree"][i]["raw"] = str(graph["tree"][i]["raw"])


# make numpy array serializable
graph['relation_matrix'] = graph['relation_matrix'].tolist()
for key in graph['node_parameters'].keys():
    graph['node_parameters'][key] = graph['node_parameters'][key].tolist()

with open(str(dump_dir / f"{task_name}.json"), 'w') as f:
    json.dump(graph, f, indent=4)
