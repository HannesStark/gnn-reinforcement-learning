from NerveNet.graph_util.mujoco_parser import parse_mujoco_graph
from NerveNet.graph_util.mujoco_parser_settings import RootRelationOption, EmbeddingOption
import pybullet_data
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt


def visualize_task_graph(task_name: str,
                         figsize=(10, 10),
                         use_sibling_relations: bool = True,
                         drop_body_nodes=True,
                         root_relation_option: RootRelationOption = RootRelationOption.BODY):
    fig = plt.figure(figsize=figsize)

    task_log = parse_mujoco_graph(task_name=task_name,
                                  use_sibling_relations=use_sibling_relations,
                                  root_relation_option=root_relation_option,
                                  drop_body_nodes=drop_body_nodes)

    node_colors = {
        "red": task_log["node_type_dict"]["root"],
        "blue": task_log["node_type_dict"]["joint"],
        "black": task_log["node_type_dict"]["body"],
    }

    node_names = {node["id"]: node["raw_name"] for node in task_log["tree"] if "raw_name" in node.keys()}
    node_names[0] = "root"

    # Generate graph structure
    G = nx.Graph()
    for i in range(task_log["num_nodes"]):
        G.add_node(i)
        for j in range(i, task_log["num_nodes"]):
            if task_log["relation_matrix"][i][j] != 0:
                G.add_edge(i, j)

    pos = nx.spring_layout(G)  # , pos={0: np.array([ 0, 0])})

    options = {"node_size": 600, "alpha": 0.8}
    for color, nodes in node_colors.items():
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color, **options)
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color, **options)

    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    for p in pos:  # raise text positions
        pos[p][1] += 0.055
    nx.draw_networkx_labels(G, pos, node_names, font_size=20)

    plt.savefig('robot.png')
    plt.show()
    return task_log


task_name = "AntBulletEnv-v0"

xml_assets_path = Path(pybullet_data.getDataPath()) / "mjcf"

# info = parse_mujoco_graph(task_name=task_name,
#                                       xml_name=task_name,
#                                       root_relation_option=RootRelationOption.ALL,
#                                       xml_assets_path=xml_assets_path,
#                                       embedding_option=EmbeddingOption.SHARED)

visualize_task_graph("AntBulletEnv-v0",
                     figsize=(20, 20),
                     use_sibling_relations=True,
                     drop_body_nodes=True,
                     root_relation_option=RootRelationOption.NONE
                     )
