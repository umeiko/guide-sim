import os
import json
import cv2
from  tqdm import tqdm


os.chdir("/home/umeko/ws/guide-sim/datas")
PATH = "./train"
TASK_PATH = os.path.join(PATH, "task")
TASKS = [os.path.join(TASK_PATH, name) for name in os.listdir(TASK_PATH)]
NOANG_IMAGES = [os.path.join(PATH, "no_angiography", name) for name in os.listdir(TASK_PATH)]

for i in tqdm(TASKS):
    with open(i, "r") as f:
        js = json.load(f)
    mask_path = js["mask_path"].replace("\\", "/")
    bg_path = js["background_path"].replace("\\", "/")
    # 额外的检查：查看 os.path.join(PATH, "no_angiography", xxx.png) 是否是存在的
    # 如果存在的话, 在配置文件中添加 "no_angiography" 属性
    if os.path.exists(os.path.join(PATH, "no_angiography", os.path.basename(mask_path))):
        js["no_angiography"] = os.path.join("no_angiography", os.path.basename(mask_path))
        jsname = os.path.basename(i)
        print(f"给{i}写入no_angiography属性")
        with open(i, "w") as f:
            json.dump(js, f, ensure_ascii=False, indent=4)