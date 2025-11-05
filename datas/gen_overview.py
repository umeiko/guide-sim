import os
import json
import cv2
from  tqdm import tqdm

os.chdir("/home/umeko/ws/guide-sim/datas")
PATH = "./train"
TASK_PATH = os.path.join(PATH, "task")
TASKS = [os.path.join(TASK_PATH, name) for name in os.listdir(TASK_PATH)]
NOVAS_TASKS = [os.path.join(PATH, "novas", name) for name in os.listdir(TASK_PATH)]

os.makedirs(os.path.join(PATH, "overview"), exist_ok=True)
os.makedirs(os.path.join(PATH, "novas"), exist_ok=True)

for i in tqdm(TASKS):
    with open(i, "r") as f:
        js = json.load(f)
    mask_path = js["mask_path"].replace("\\", "/")
    bg_path = js["background_path"].replace("\\", "/")
    # 额外的检查：查看 os.path.join(PATH, "novas", xxx.png) 是否是存在的
    # 如果存在的话, 在配置文件中添加 "novas" 属性
    if os.path.exists(os.path.join(PATH, "novas", os.path.basename(mask_path))):
        js["novas"] = os.path.join("novas", os.path.basename(mask_path))
        jsname = os.path.basename(i)
        with open(os.path.join("novas", jsname), "w") as f:
            js = json.dump(f)
    
    # 读取图像和掩码, 并将掩码以0.25的透明度叠加到图像上
    img = cv2.imread(os.path.join(PATH, bg_path))
    mask = cv2.imread(os.path.join(PATH, mask_path))
    # 图像尺寸重设到和mask一致
    img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
    overlay = img.copy()
    alpha = 0.25
    cv2.addWeighted(mask, alpha, overlay, 1 - alpha, 0, overlay)
    # 读取导丝起点坐标，在图上标注"start"和绿点
    start_x, start_y = js["insert_pos"]
    # 如果 guide_pos_lst 属性存在且不为空，则使用最后一点作为起点
    if "guide_pos_lst" in js and js["guide_pos_lst"]:
        # 读取的值是浮点，需要转换为整数
        start_x, start_y = js["guide_pos_lst"][-1]
        start_x, start_y = int(start_x), int(start_y)
    cv2.circle(overlay, (start_x, start_y), 5, (0, 255, 0), -1)
    cv2.putText(overlay, "start", (start_x + 10, start_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # 读取导丝终点坐标，在图上标注"goal"和红点
    goal_x, goal_y = js["target_pos"]
    cv2.circle(overlay, (goal_x, goal_y), 5, (0, 0, 255), -1)
    cv2.putText(overlay, "goal", (goal_x + 10, goal_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # 保存结果图像
    save_path = os.path.join(PATH, "overview", os.path.basename(i).replace(".json", ".png"))
    cv2.imwrite(save_path, overlay)
    # print(f"Saved overview image to {save_path}")
