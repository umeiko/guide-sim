import os
import json
from  tqdm import tqdm


os.chdir("/home/umeko/maohaolin/guide-sim/datas")
PATH = "./plastic_tube"
task_dir = os.path.join(PATH, "task")
orig_tasks = os.listdir(task_dir)

# 第一步：重命名为临时名称
temp_files = []
for k, i in enumerate(tqdm(orig_tasks)):
    src = os.path.join(task_dir, i)
    temp_name = f"temp_{k}.json"
    dst = os.path.join(task_dir, temp_name)
    print(src, "->", dst)
    os.rename(src, dst)
    temp_files.append(temp_name)

# 第二步：重命名为最终名称
for k, temp_name in enumerate(tqdm(temp_files)):
    src = os.path.join(task_dir, temp_name)
    dst = os.path.join(task_dir, f"{k}.json")
    print(src, "->", dst)
    os.rename(src, dst)