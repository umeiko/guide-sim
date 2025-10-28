import os
import json

PATH_A = "./merged_tasks"
PATH_B = "./merged_tasks3"
PATH_MERGED = "./merged_tasks2"

tsk_a_mask_lst = [os.path.join(PATH_A, "label", i) \
    for i in os.listdir(
    os.path.join(PATH_A, "label"))]

tsk_a_img_lst = [os.path.join(PATH_A, "images", i) \
    for i in os.listdir(
    os.path.join(PATH_A, "images"))]

tsk_b_mask_lst = [os.path.join(PATH_B, "label", i) \
    for i in os.listdir(
    os.path.join(PATH_B, "label"))]

tsk_b_img_lst = [os.path.join(PATH_B, "images", i) \
    for i in os.listdir(
    os.path.join(PATH_B, "images"))]

mapping = {}
cnt = 0
for i in tsk_a_mask_lst:
    fname = i.split("/")[-1]
    image_path = os.path.join(PATH_A, "images", fname)
    mapping[i] = os.path.join("label", f"{cnt}.png")
    mapping[image_path] = os.path.join("images", f"{cnt}.png")
    cnt += 1

for i in tsk_b_mask_lst:
    fname = i.split("/")[-1]
    image_path = os.path.join(PATH_B, "images", fname)
    mapping[i] = os.path.join("label", f"{cnt}.png")
    mapping[image_path] = os.path.join("images", f"{cnt}.png")
    cnt += 1

os.makedirs(os.path.join(PATH_MERGED, "task"), exist_ok=True)
os.makedirs(os.path.join(PATH_MERGED, "images"), exist_ok=True)
os.makedirs(os.path.join(PATH_MERGED, "label"), exist_ok=True)

print("="*20, "copy images", "="*20)
for i in mapping:
    print(i, "->", mapping[i])
    os.system(f"cp {i} {os.path.join(PATH_MERGED, mapping[i])}")

tsk_a_task_lst = os.listdir(
    os.path.join(PATH_A, "task"))
tsk_b_task_lst = os.listdir(
    os.path.join(PATH_B, "task"))

print("="*20, "mapping tasks", "="*20)

cnt_tsk = 0

for tsk in tsk_a_task_lst:
    _p = os.path.join(PATH_A, "task", tsk)
    js = json.load(open(_p))
    # 有可能是 windows 类型的路径分隔符 "background_path": "images\\4.png"
    # 也有可能是 unix 类型的路径分隔符 "background_path": "images/4.png"
    # 都需要处理
    print(_p)
    js["mask_path"] = js["mask_path"].replace("\\", "/")
    js["background_path"] = js["background_path"].replace("\\", "/")
    old_mask_name = js["mask_path"].split("/")[-1]
    old_image_name = js["background_path"].split("/")[-1]
    
    old_mask_path = os.path.join(PATH_A, "label", old_mask_name)
    old_image_path = os.path.join(PATH_A, "images", old_image_name)
    
    print(old_mask_path, "->", mapping[old_mask_path])
    print(old_image_path, "->", mapping[old_image_path])
    js["mask_path"] = mapping[old_mask_path]
    js["background_path"] = mapping[old_image_path]

    with open(os.path.join(PATH_MERGED, "task", f"{cnt_tsk}.json"), "w") as f:
        json.dump(js, f, ensure_ascii=False, indent=4)
        print(f"write task to {os.path.join(PATH_MERGED, 'task', f'{cnt_tsk}.json')}")
    cnt_tsk += 1

for tsk in tsk_b_task_lst:
    _p = os.path.join(PATH_B, "task", tsk)
    js = json.load(open(_p))
    # 有可能是 windows 类型的路径分隔符 "background_path": "images\\4.png"
    # 也有可能是 unix 类型的路径分隔符 "background_path": "images/4.png"
    # 都需要处理
    print(_p)
    js["mask_path"] = js["mask_path"].replace("\\", "/")
    js["background_path"] = js["background_path"].replace("\\", "/")
    old_mask_name = js["mask_path"].split("/")[-1]
    old_image_name = js["background_path"].split("/")[-1]
    
    old_mask_path = os.path.join(PATH_B, "label", old_mask_name)
    old_image_path = os.path.join(PATH_B, "images", old_image_name)
    
    print(old_mask_path, "->", mapping[old_mask_path])
    print(old_image_path, "->", mapping[old_image_path])
    js["mask_path"] = mapping[old_mask_path]
    js["background_path"] = mapping[old_image_path]

    with open(os.path.join(PATH_MERGED, "task", f"{cnt_tsk}.json"), "w") as f:
        json.dump(js, f, ensure_ascii=False, indent=4)
        print(f"write task to {os.path.join(PATH_MERGED, 'task', f'{cnt_tsk}.json')}")
    cnt_tsk += 1