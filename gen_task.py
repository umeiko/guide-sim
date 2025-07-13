#!/bin/python3

# 此代码用于手动建图
# 用法：wsad 控制导丝
#      按住空格重设导丝的初始位置等参数
#      鼠标左键重设目标位置
#      t保存一个不带形态的任务
#      r键保存一个完整的导丝任务
#      , 和 . 键用于切换已保存的任务
#      o 和 p 键用于切换任务的背景图
from env.simulation import GuideWireEngine
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
LOGGER.addHandler(console_handler)

DATASET_PATH = "./datas/exvivo"

BACKGROUND_PATH = os.path.join(DATASET_PATH, "images")
MASKS_PATH = os.path.join(DATASET_PATH, "label")
TASK_PATH = os.path.join(DATASET_PATH, "task")
MASKS = os.listdir(MASKS_PATH)
TASKS = os.listdir(TASK_PATH)
SIZE = (512, 512)
INSERT_LIMIT = 100


def _sortfunc(name:str):
    return int(name.split(".")[0])
def _angle_between_points(point1, point2):
    """计算两个点之间的角度"""
    x_diff = point2[0] - point1[0]
    y_diff = point2[1] - point1[1]
    return math.degrees(math.atan2(y_diff, x_diff))
MASKS.sort(key=_sortfunc)
TASKS.sort(key=_sortfunc)

class PygameWindow():
    """一个简单的Pygame窗口, 用于手动建图"""
    def __init__(self):
        if len(MASKS) == 0:
            raise FileNotFoundError("No masks found in " + MASKS_PATH)
        pygame.init()
        self.engine = GuideWireEngine()
        self.screen = pygame.display.set_mode(SIZE)
        self.clock = pygame.time.Clock()
        self.sim_metadata = GuideSimMetadata()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        pymunk.pygame_util.positive_y_is_up = False
        task_loaded = False
        self.task_id = 0
        if len(TASKS) == 0:
            LOGGER.info("No tasks found, creating a new one.")
            self.bg_surf = pygame.image.load(os.path.join(BACKGROUND_PATH, MASKS[0]))
            self.mask_surf = pygame.image.load(os.path.join(MASKS_PATH, MASKS[0]))
            self.sim_metadata.background_path = os.path.join("images", MASKS[0])
            self.sim_metadata.mask_path = os.path.join("label", MASKS[0])
            LOGGER.info("Loaded %s" , os.path.join(BACKGROUND_PATH, MASKS[0]))
            LOGGER.info("Loaded %s" , os.path.join(MASKS_PATH, MASKS[0]))
        else:
            self.sim_metadata.load_from_json(os.path.join(TASK_PATH, TASKS[-1]))
            LOGGER.info("Loaded %s" , os.path.join(TASK_PATH, TASKS[-1]))
            self.bg_surf = pygame.image.load(
                os.path.join(DATASET_PATH, self.sim_metadata.background_path))
            self.mask_surf = pygame.image.load(
                os.path.join(DATASET_PATH, self.sim_metadata.mask_path))
            task_loaded = True
            self.task_id = len(TASKS) -1
        self.bg_surf   = pygame.transform.scale(self.bg_surf, SIZE)
        self.mask_surf = pygame.transform.scale(self.mask_surf, SIZE)
        self.engine.draw_walls_by_mask(self.mask_surf, friction=0.1)
        self._is_setting_guide = False
        self._guide_setting_stage = 0
        self._angle = None
        self._guide_init_done = False
        if not task_loaded:
            ...
        else:
            if self.sim_metadata.guide_pos_lst is None:
                self.create_base_guide()
            else:
                self.load_task(self.task_id)
            self._guide_init_done = True


    def load_task(self, task_id):
        """加载任务"""
        self.engine.clear_all()
        LOGGER.info("Loading task %s", os.path.join(TASK_PATH, TASKS[task_id]))
        self.sim_metadata.load_from_json(os.path.join(TASK_PATH, TASKS[task_id]))
        self.bg_surf = pygame.image.load(
            os.path.join(DATASET_PATH, self.sim_metadata.background_path))
        self.mask_surf = pygame.image.load(
            os.path.join(DATASET_PATH, self.sim_metadata.mask_path))
        self.bg_surf   = pygame.transform.scale(self.bg_surf, SIZE)
        self.mask_surf = pygame.transform.scale(self.mask_surf, SIZE)
        self.engine.draw_walls_by_mask(self.mask_surf, friction=0.1)
        if self.sim_metadata.guide_pos_lst is None:
            self.create_base_guide()
        else:
            self._angle = _angle_between_points(self.sim_metadata.insert_pos, 
                                          self.sim_metadata.direct_pos)
            self.engine.set_guide_by_list(self.sim_metadata.guide_pos_lst,
                                          self.sim_metadata.radius,
                                          self._angle)
        self._guide_init_done = True


    def render(self):
        """渲染画面"""
        self.screen.blit(self.bg_surf, (0, 0))
        self.engine.space.debug_draw(self.draw_options)
        self.render_guide_set()
        pygame.display.flip()

    def set_guide(self, pos):
        """设置导丝"""
        if self._guide_setting_stage == 0:  # 设置起点
            self.sim_metadata.insert_pos = pos
            self._guide_setting_stage += 1
        elif self._guide_setting_stage == 1:  # 设置朝向
            self.sim_metadata.direct_pos = pos
            self._guide_setting_stage += 1
        elif self._guide_setting_stage == 2:  # 设置目标点
            self.sim_metadata.target_pos = pos
            self._guide_setting_stage += 1
    
    def clear_guide_set(self):
        """清除导丝设置"""
        self.sim_metadata.insert_pos = None
        self.sim_metadata.direct_pos = None
        self.sim_metadata.target_pos = None
    
    def render_guide_set(self):
        """渲染导丝设置"""
        if self.sim_metadata.insert_pos is not None:
            pygame.draw.circle(self.screen, (255, 0, 0), self.sim_metadata.insert_pos, 5)
        if self.sim_metadata.direct_pos is not None:
            pygame.draw.circle(self.screen, (0, 255, 0), self.sim_metadata.direct_pos, 5)
        if self.sim_metadata.target_pos is not None:
            pygame.draw.circle(self.screen, (0, 0, 255), self.sim_metadata.target_pos, 5)

    def create_base_guide(self):
        """创建基础导丝"""
        if self.sim_metadata.insert_pos is not None and \
              self.sim_metadata.direct_pos is not None and \
              self.sim_metadata.target_pos is not None:
            self._angle = _angle_between_points(self.sim_metadata.insert_pos, 
                                          self.sim_metadata.direct_pos)
            self.engine.create_ini_guidewire(
                self.sim_metadata.radius,
                self.sim_metadata.insert_pos,
                ini_len= 3,
                direct_angle=self._angle
            )
            self._guide_init_done = True
            LOGGER.info("Guide created")
    
    def act_guide(self, key):
        """导丝运动"""
        if self._guide_init_done:
            if key == K_w:
                if len(self.engine.balls) <= INSERT_LIMIT:
                    self.engine.insert_a_ball(radius=self.sim_metadata.radius,
                                               direct_angle=self._angle)
                    LOGGER.info("Insert a ball %.2f ", self.sim_metadata.radius)
            elif key == K_s:
                self.engine.pull_back_a_ball(direct_angle=self._angle)
                LOGGER.info("Pullback a ball")
            elif key == K_a:
                self.engine.bend_guidewire(d_angle=10)
                LOGGER.info("rotate guide")
            elif key == K_d:
                self.engine.bend_guidewire(d_angle=-10)
                LOGGER.info("rotate guide")

    def set_target(self, pos):
        """设置目标点的位置"""
        self.sim_metadata.target_pos = pos
        

    
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

    def switch_mask(self, key):
        """切换背景"""
        now_mask = os.path.basename(self.sim_metadata.mask_path)
        idx = MASKS.index(now_mask)
        if key == K_o:
            if idx > 0:
                idx -= 1
            else:
                return
        elif key == K_p:
            if idx < len(MASKS) - 1:
                idx += 1
            else:
                return
        
        self.sim_metadata = GuideSimMetadata()
        LOGGER.info("No tasks, creating a new one.")
        self.bg_surf = pygame.image.load(os.path.join(BACKGROUND_PATH, MASKS[idx]))
        self.mask_surf = pygame.image.load(os.path.join(MASKS_PATH, MASKS[idx]))
        self.sim_metadata.background_path = os.path.join("images", MASKS[idx])
        self.sim_metadata.mask_path = os.path.join("label", MASKS[idx])
        LOGGER.info("Loaded %s" , os.path.join(BACKGROUND_PATH, MASKS[idx]))
        LOGGER.info("Loaded %s" , os.path.join(MASKS_PATH, MASKS[idx]))
        self.engine.clear_all()
        self.bg_surf   = pygame.transform.scale(self.bg_surf, SIZE)
        self.mask_surf = pygame.transform.scale(self.mask_surf, SIZE)
        self.engine.draw_walls_by_mask(self.mask_surf, friction=0.1)
        
        

    def save_task(self, key):
        """保存任务"""
        if key == K_r:
            self.save_full_task()
        elif key == K_t:
            self.save_base_task()

    def save_full_task(self):
        """保存任务"""
        self.sim_metadata.guide_pos_lst = self.engine.get_guide_pos_list()
        self.sim_metadata.guide_angle = self.engine.angle
        self.sim_metadata.save_to_json(os.path.join(TASK_PATH, f"{len(TASKS)+1}.json"))
        TASKS.append(f"{len(TASKS)+1}.json")
        LOGGER.info(f"Full task saved, tasks: {TASKS}")
    def save_base_task(self):
        """保存任务"""
        base = GuideSimMetadata()
        base.background_path = self.sim_metadata.background_path
        base.mask_path = self.sim_metadata.mask_path
        base.insert_pos = self.sim_metadata.insert_pos
        base.direct_pos = self.sim_metadata.direct_pos
        base.target_pos = self.sim_metadata.target_pos
        base.radius = self.sim_metadata.radius
        base.save_to_json(os.path.join(TASK_PATH, f"{len(TASKS)+1}.json"))
        TASKS.append(f"{len(TASKS)+1}.json")
        LOGGER.info(f"Base task saved, tasks: {TASKS}")

    def run(self):
        fps = 60
        exact = 1000 / fps
        while True:
            for _ in range(int(exact)):
                self.engine.space.step(1 / fps / exact)
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                if event.type == KEYDOWN:
                    if event.key == K_SPACE:
                        LOGGER.info("Start setting guide")
                        self._is_setting_guide = True
                        self.clear_guide_set()
                        self.engine.clear_all()
                        self._guide_init_done = False
                    elif event.key in [K_COMMA, K_PERIOD]:
                        self.switch_task(event.key)
                    elif event.key in [K_r, K_t]:
                        self.save_task(event.key)
                    elif event.key in [K_o, K_p]:
                        self.switch_mask(event.key)
                    else:
                        self.act_guide(event.key)
                elif event.type == KEYUP:
                    if event.key == K_SPACE:
                        LOGGER.info("stop setting guide")
                        self._is_setting_guide = False
                        if self._guide_setting_stage != 3:
                            self.clear_guide_set()
                            self.engine.clear_all()
                            self._guide_init_done = False
                            LOGGER.info("Nothing changed")
                        else:
                            self.create_base_guide()
                        self._guide_setting_stage = 0
                elif event.type == MOUSEBUTTONDOWN:
                    if self._is_setting_guide:
                        self.set_guide(pygame.mouse.get_pos())
                    else:
                        self.set_target(pygame.mouse.get_pos())
                
            self.render()
            self.clock.tick(fps)




if __name__ == '__main__':
    app = PygameWindow()
    app.run()