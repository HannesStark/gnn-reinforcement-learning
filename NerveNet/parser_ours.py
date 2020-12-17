import os
from pathlib import Path
from parser_logs_util import generate_graph_log
from graph_util.mujoco_parser_settings import XML_DICT as PYBULLET_XML_DICT
from graph_util.mujoco_parser_nervenet import XML_DICT as NERVENET_XML_DICT


nervenet_assets_dir = Path(os.getcwd()).parent / \
    "NerveNet" / "environments" / "assets"

for task_name in NERVENET_XML_DICT.keys():
    try:
        generate_graph_log(
            task_name,
            xml_assets_path=nervenet_assets_dir,
            use_nervenet_parser=True)
    except:
        print(f"Could not generate graph log for {task_name}.")

for task_name in PYBULLET_XML_DICT.keys():
    try:
        generate_graph_log(task_name,
                           dump_dir=Path("graph_logs_new"))
    except:
        print(f"Could not generate graph log for {task_name}.")
