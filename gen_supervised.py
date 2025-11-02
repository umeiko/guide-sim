#!/bin/python3

# 此代码用于生成监督学习数据集
# 用法：wsad 控制导丝
#      按住空格重设导丝的初始位置等参数
#      鼠标左键重设目标位置
#      , 和 . 键用于切换任务
from env.simulation import GuideWireEngine
from env.guide_sim import GuidewireEnv, ndarray_gray_to_surf, surf_to_ndarray
from env.metadata import GuideSimMetadata, HyperParams
import pygame
import os
import math
import json
from pygame.key import *
from pygame.locals import *
from pygame.color import *
import cv2

import numpy as np
import io

import pymunk
import pymunk.constraints
import pymunk.pygame_util

from env.metadata import GuideSimMetadata, HyperParams
import logging
import sys

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG) # 设置日志级别为DEBUG，以便输出所有级别的日志
# 创建一个控制台处理器，并设置级别为DEBUG
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setLevel(logging.DEBUG)
# 设置日志格式
formatter = logging.Formatter('[%(filename)s:%(lineno)d][%(levelname)s] %(message)s')
console_handler.setFormatter(formatter)

def load_json(path:str):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    else:
        return {"num_datas" : 0,
                "datas" : {}
                }
def write_json(path:str, data:dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

LOGGER.addHandler(console_handler)

DATASET_PATH = "./datas/exvivo"
OUTPUT_PATH = "./supervised_datas/train"
os.makedirs(OUTPUT_PATH, exist_ok=True)
TASK_PATH = os.path.join(DATASET_PATH, "task")
TASK_PATH = os.path.join(DATASET_PATH, "task")
TASKS = os.listdir(TASK_PATH)
OUTPUTS = os.listdir(OUTPUT_PATH)


def _sortfunc(name:str):
    return int(name.split(".")[0])
TASKS.sort(key=_sortfunc)
TASKS = [os.path.join(DATASET_PATH, "task", t) for t in TASKS]
OUTPUTS = load_json(os.path.join(OUTPUT_PATH, "datas.json"))

HYPER = HyperParams()
HYPER.load_from_json("./hyper.json")

image_path = os.path.join(OUTPUT_PATH, "image")
affordance_path = os.path.join(OUTPUT_PATH, "affordance")

os.makedirs(image_path, exist_ok=True)
os.makedirs(affordance_path, exist_ok=True)

class PygameWindow():
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(HYPER.img_size)
        self.clock = pygame.time.Clock()
        self.task_id = 0
        self.env = GuidewireEnv(TASKS[self.task_id])
        self.bg_buffer = pygame.Surface(self.screen.get_size())
        state = self.env.reset()
        self.bg_buffer.blit(ndarray_gray_to_surf(state), (0, 0))
        self._bgs    = [state]
        self.affordance = []
        self.done = False
        self.actions = []
        self.metadata = {
            "task": TASKS[self.task_id],
            "num_actions": 0
        }

    def act(self, key):
        """ 控制导丝，同时记录动作和数据 """
        flag_save = True
        if self.done:
            if key in [K_w, K_s, K_a, K_d]:
                return
        d = None
        s = None
        r = 0
        
        if key == K_w:
            s, r, d, _ = self.env.step(0)
            self.actions.append(0)
        elif key == K_s:
            s, r, d, _ = self.env.step(1)
            self.actions.append(1)
        elif key == K_a:
            s, r, d, _ = self.env.step(2)
            self.actions.append(2)
        elif key == K_d:
            s, r, d, _ = self.env.step(3)
            self.actions.append(3)
        elif key == K_SPACE:
            flag_save = False
            state = self.env.reset()
            self.bg_buffer.blit(ndarray_gray_to_surf(state), (0, 0))
            self.metadata["num_actions"] = 0
            self.actions = []
            self._bgs    = [state]
            self.done = False
        if key in [K_w, K_s, K_a, K_d]:
            print(r, _)
        if flag_save:
            if not self.done:
                self.bg_buffer.blit(ndarray_gray_to_surf(s), (0, 0))
                self._bgs.append(s)
                self.affordance.append(self.env.get_a_star_path())
                self.metadata["num_actions"] += 1
                self.done = d
            if r > 3.8:
                self.save_data()

    def switch_task(self, key):
        """切换任务"""
        if key == K_COMMA:
            if self.task_id > 0:
                self.task_id -= 1
                self.load_task(self.task_id)
        elif key == K_PERIOD:
            if self.task_id < len(TASKS) - 1:
                self.task_id += 1
                self.load_task(self.task_id)
    
    def load_task(self, task_id):
        print("加载任务：")
        print(TASKS[task_id])
        self.env = GuidewireEnv(TASKS[task_id])
        self.metadata["task"] = TASKS[task_id]
        self.metadata["num_actions"] = 0
        self.actions = []
        state = self.env.reset()
        self.bg_buffer.blit(ndarray_gray_to_surf(state), (0, 0))
        self._bgs    = [state]
        self.done = False

    def run(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key in [K_w, K_s, K_a, K_d, K_SPACE]:
                        self.act(event.key)
                    if event.key in [K_COMMA, K_PERIOD]:
                        self.switch_task(event.key)
            
            self.screen.blit(self.bg_buffer, (0, 0))
            self.clock.tick(60)
            pygame.display.flip()
    
    def save_data(self):
        """保存数据"""
        print("数据保存了！")
        for bg, aff, act in zip(self._bgs, self.affordance, self.actions):
            bg = np.array(bg, dtype=np.uint8).swapaxes(0, 1).swapaxes(1, 2)
            aff = np.array(aff, dtype=np.uint8)
            OUTPUTS["num_datas"] += 1
            cv2.imwrite(os.path.join(image_path,f"{OUTPUTS['num_datas']}.png"),
                        bg,
                        )
            cv2.imwrite(os.path.join(affordance_path, f"{OUTPUTS['num_datas']}.png"),
                        aff,
                        )
            OUTPUTS["datas"][OUTPUTS["num_datas"]] = {
                "task": self.metadata["task"],
                "action": act,
            }
        write_json(os.path.join(OUTPUT_PATH, "datas.json"), OUTPUTS)

if __name__ == '__main__':
    app = PygameWindow()
    app.run()