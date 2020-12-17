import os
from pathlib import Path
from parser_logs_util import generate_graph_log
from graph_util.mujoco_parser_settings import XML_DICT as PYBULLET_XML_DICT
from graph_util.mujoco_parser_nervenet import XML_DICT as NERVENET_XML_DICT


nervenet_assets_dir = Path(os.getcwd()).parent / \
    "NerveNet" / "environments" / "assets"

generate_graph_log("AntS-v1",
                   xml_assets_path=nervenet_assets_dir)

for task_name in NERVENET_XML_DICT.keys():
    try:
        generate_graph_log(
            task_name,
            xml_assets_path=nervenet_assets_dir)
    except:
        print(f"Could not generate graph log for {task_name}.")

for task_name in PYBULLET_XML_DICT.keys():
    generate_graph_log(task_name)
    # try:
    #     generate_graph_log(task_name)
    # except:
    #     print(f"Could not generate graph log for {task_name}.")
