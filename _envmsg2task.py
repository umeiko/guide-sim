"""
把旧任务转换为新版本task的脚本
"""
import os
import json
import time
from  env.metadata import GuideSimMetadata, HyperParams

DATASET_PATH = "./datas/vivo_single_branch"

BACKGROUND_PATH = os.path.join(DATASET_PATH, "images")
MASKS_PATH = os.path.join(DATASET_PATH, "label")
TASK_PATH = os.path.join(DATASET_PATH, "task")
OLD_TASKS_PATH = os.path.join(DATASET_PATH, "envmsgs")

MASKS = os.listdir(MASKS_PATH)
TASKS = os.listdir(TASK_PATH)
OLD_TASKS = os.listdir(OLD_TASKS_PATH)

def _sortfunc(name:str):
    return int(name.split(".")[0])

MASKS.sort(key=_sortfunc)
TASKS.sort(key=_sortfunc)

def open_old_msg(path:str):
    with open(path, "r") as f:
        return json.load(f)

def main():
    for i in MASKS:
        old_msg_f_name = i.split(".")[0] + ".json"
        old_msg = open_old_msg(os.path.join(OLD_TASKS_PATH, old_msg_f_name))
        matadata = GuideSimMetadata()
        matadata.mask_path = os.path.join("label", i)
        matadata.background_path = os.path.join("images", i)
        matadata.insert_pos = [old_msg[0][0] //2 , old_msg[0][1] // 2]
        matadata.direct_pos = [old_msg[1][0] //2 , old_msg[1][1] // 2]
        matadata.target_pos = [old_msg[2][0] //2 , old_msg[2][1] // 2]
        matadata.save_to_json(os.path.join(TASK_PATH, i.split(".")[0] + ".json"))

if __name__ == '__main__':
    main()
