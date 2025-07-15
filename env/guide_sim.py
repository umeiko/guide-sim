#!/bin/python3

import numpy as np

import sys
sys.path.append("..") 
import json
import os
import math
import random
import pygame
import logging
import cv2
import io
import time
import pymunk
import heapq
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from env.metadata import GuideSimMetadata, HyperParams

logger = logging.getLogger(__name__)
current_time = time.strftime("%m-%d_%H-%M", time.localtime())

try:
    import simulation
    logger.info("Importing simulation from current directory")
except ImportError:
    try:
        from env import simulation
        logger.info("Importing simulation from env directory")
    except ImportError as e:
        logger.error(
            "Failed to import simulation module.")
        raise e

SIZE = (512, 512)
def _angle_between_points(point1, point2):
    """计算两个点之间的角度"""
    x_diff = point2[0] - point1[0]
    y_diff = point2[1] - point1[1]
    return math.degrees(math.atan2(y_diff, x_diff))

G_RENDER_WIDTH  = 4
G_RENDER_RGB  = (43, 43, 43)
TARGET_RGB    = (255, 255, 255)
DEBUG_DRAW = False
SIMULATION_WIDTH = 512
SIMULATION_HIGHT = 512
MAX_NODES = 100

def rgb_to_gray(rgb:np.ndarray)->np.ndarray:
    return np.array(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))

def gray_to_rgb(gray:np.ndarray)->np.ndarray:
    if gray.shape[0] <= 3:
        gray = gray.swapaxes(0, 2)
        gray = gray.swapaxes(0, 1)
    return np.array(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))

def l2_distance_square(point1, point2):
    """
    没有开根的欧氏距离 (为了优化计算速度)
    """
    return (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2

def l2_is_done(point1, point2, threshold):
    """
    判断两个点的欧氏距离是否在阈值范围内
    """
    return l2_distance_square(point1, point2) <= threshold**2


def find_nearest_255(img_msk:np.ndarray, point):
    """
    找到离point最近的有效点
    img_msk: 二值图, 0为障碍, 255为通路
    """
    if img_msk[point[0], point[1]] == 255:
        return tuple(point)
    mask = (img_msk == 255)
    _, inds = distance_transform_edt(~mask, return_indices=True)
    return tuple(inds[:, point[0], point[1]])

def a_star(img_msk:np.ndarray, start, end):
    '''
    A*算法
    img_msk: 二值图, 0为障碍, 255为通路
    start: 起始点
    end: 终点
    return: 路径 (1为通路, 0为无效)
    '''
    start = find_nearest_255(img_msk, start)
    end = find_nearest_255(img_msk, end)
    h, w = img_msk.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: abs(start[0]-end[0]) + abs(start[1]-end[1])}
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == end:
            # 回溯路径
            path = np.zeros_like(img_msk, dtype=np.uint8)
            while current in came_from:
                path[current] = 1
                current = came_from[current]
            path[start] = 1
            path[end] = 1
            return path
        for d in directions:
            neighbor = (current[0]+d[0], current[1]+d[1])
            if 0<=neighbor[0]<h and 0<=neighbor[1]<w and img_msk[neighbor]==255:
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + abs(neighbor[0]-end[0]) + abs(neighbor[1]-end[1])
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    # 无路径
    return np.zeros_like(img_msk, dtype=np.uint8)

def a_star_distance(a_star_path:np.ndarray):
    """
    得到两个点之间的A*距离
    img_msk: 二值图, 0为障碍, 255为通路
    """
    if len(a_star_path.shape) > 2:
        raise ValueError(f"a_star_path should be a 2D array, got {a_star_path.shape}")
    return np.sum(a_star_path)

def surf_to_ndarray(surf:pygame.Surface, alpha=False):
    """pygame image to cv2 form image"""
    frame_np = np.array(pygame.surfarray.pixels3d(surf))
    if alpha:
        alpha_np = np.array(pygame.surfarray.pixels_alpha(surf))[:,:,np.newaxis]
        # alpha_np = np.array(pygame.surfarray.pixels_alpha(surf))
        frame_np = np.append(frame_np, alpha_np, axis=2)
        # frame_np = np.stack((frame_np, alpha_np), axis=2)
    frame_np = np.transpose(frame_np, (1, 0, 2))
    return frame_np


def ndarray_to_surf(ndarray:np.ndarray):
    conv_img = pygame.surfarray.make_surface(ndarray[:,:,:3])
    # conv_img.set_alpha(pygame.surfarray.)
    conv_img = pygame.transform.rotate(conv_img, 90)
    return pygame.transform.flip(conv_img,False,True)

def ndarray_gray_to_surf(ndarray:np.ndarray):
    ndarray = gray_to_rgb(ndarray)
    if ndarray.shape[0] <= 3:
        ndarray = ndarray.swapaxes(0, 2)
    conv_img = pygame.surfarray.make_surface(ndarray[:,:,:3])
    # conv_img.set_alpha(pygame.surfarray.)
    conv_img = pygame.transform.rotate(conv_img, 90)
    return pygame.transform.flip(conv_img,False,True)

class GuidewireEnv():
    def __init__(self, task_path:str=None, dataset_path: str = None):
        logging.info("Initializing GuidewireEnv ...")
        if dataset_path is None:
            self.dataset_path = os.path.dirname(os.path.dirname(task_path))
        else:
            self.dataset_path = dataset_path
        self.metadata = GuideSimMetadata()
        self.hyper_params = HyperParams()
        self.engine = simulation.GuideWireEngine()
        self.bg_surf = None
        self.mask_surf = None
        self.task_path = task_path
        self._angle = None
        self._mask_surf_np = None
        self.a_star_path_np = None
        self.load_task(task_path)
        logging.info("Load metadata from %s", task_path)
        
        self.step_punishment = 0.00
        self.image_size:tuple[int, int] = self.hyper_params.img_size

        self.now_step = 0
        self.g_surface:pygame.Surface = pygame.Surface(SIZE, pygame.SRCALPHA)
        self.display:pygame.Surface = pygame.Surface(SIZE)

        self._goal_rect_color = (255, 255, 255)
        self._goal_rect_width = 40
        self._goal_line_width = 4

        self._now_json = None
        self.draw_options = None
        self.gray = True

    def load_task(self, task_path:str, file_reload=True):
        """加载任务"""
        self.engine.clear_all()
        if file_reload:
            self.metadata.load_from_json(task_path)
            self.bg_surf = pygame.image.load(
                        os.path.join(self.dataset_path, self.metadata.background_path))
            self.mask_surf = pygame.image.load(
                        os.path.join(self.dataset_path, self.metadata.mask_path))
        self._mask_surf_np = rgb_to_gray(surf_to_ndarray(self.mask_surf))
        self.bg_surf   = pygame.transform.scale(self.bg_surf, SIZE)
        self.mask_surf = pygame.transform.scale(self.mask_surf, SIZE)
        
        self.engine.draw_walls_by_mask(self.mask_surf, friction=0.1)
        if self.metadata.guide_pos_lst is None:
            self.create_base_guide()
        else:
            self._angle = _angle_between_points(self.metadata.insert_pos, 
                                          self.metadata.direct_pos)
            self.engine.set_guide_by_list(self.metadata.guide_pos_lst,
                                          self.metadata.radius,
                                          self._angle)
    def create_base_guide(self):
        """创建基础导丝"""
        if self.metadata.insert_pos is not None and \
              self.metadata.direct_pos is not None and \
              self.metadata.target_pos is not None:
            self._angle = _angle_between_points(self.metadata.insert_pos, 
                                          self.metadata.direct_pos)
            self.engine.create_ini_guidewire(
                self.metadata.radius,
                self.metadata.insert_pos,
                ini_len= 3,
                direct_angle=self._angle
            )

    def step(self, action:int)->tuple[np.ndarray, float, bool, dict]:
        self.now_step += 1
        exact = 1000
        action = int(action)
        # 输入动作
        if action == 0:
            if len(self.engine.balls) <= MAX_NODES:
                self.engine.insert_a_ball(radius=self.metadata.radius,
                                           direct_angle=self._angle)
        elif action == 1:
            self.engine.pull_back_a_ball(direct_angle=self._angle)
        elif action == 2:
            self.engine.bend_guidewire(d_angle=10)
        elif action == 3:
            self.engine.bend_guidewire(d_angle=-10)
        elif action == 4:
            pass
        else:
            raise Exception("action error")

        # 物理引擎进行计算
        for _ in range(exact):
            self.engine.space.step(1 / exact)

        # 导丝以及终点的位置坐标
        reward = self.step_punishment
        pos_g_tip = self.get_now_tip_pos()
        pos_target = self.get_now_target_pos()
        done = False
        if l2_is_done(pos_g_tip, pos_target, 4 * self.metadata.radius):
            done = True
            reward = -math.log( l2_distance_square(pos_g_tip, pos_target) ** 0.5 * 0.001)
        else:
            if self.now_step >= self.hyper_params.max_steps:
                done = True
                self.a_star_path_np = a_star(self._mask_surf_np,
                                            pos_g_tip, 
                                            pos_target)
                reward = -math.log( a_star_distance(
                                    self.a_star_path_np
                                    ) * 0.001 )
    
        s = self.render()
        # s = np.array(s, dtype=np.float32) / 255.

        return s, reward, done, (pos_g_tip, pos_target, self.now_step)

    def get_a_star_path(self)->np.ndarray:
        a_star_path_np = a_star(self._mask_surf_np,
                                self.engine.get_guide_tip_pos(),
                                self.metadata.target_pos)
        return a_star_path_np

    def render(self, _return_array=True)->np.ndarray:
        # 绘制导丝
        self.g_surface.fill((255,255,255, 0))
        self._draw_guidewire(self.g_surface)
        # 对导丝进行高斯模糊
        blured_arr = cv2.GaussianBlur(surf_to_ndarray(self.g_surface, True), (5, 5), 0)
        img_encode_bytes = cv2.imencode(".png", blured_arr)[1].tobytes()
        picture_stream  = io.BytesIO(img_encode_bytes)
        surf_blured = pygame.image.load(picture_stream, ".png")

        # 输出背景
        self.display.blit(self.bg_surf, (0,0))
        if DEBUG_DRAW:        
            self.engine.space.debug_draw(self.draw_options)  # 绘制新图案
        self.display.blit(surf_blured, (0,0))

        target_pos2D = self.metadata.target_pos
        w, h = self._goal_rect_width, self._goal_rect_width
        rect = (target_pos2D[0] - w/2, target_pos2D[1] - h/2, w, h)
        pygame.draw.rect(self.display, TARGET_RGB, rect, self._goal_line_width)

        if _return_array:
            if self.display.get_size() != self.hyper_params.img_size\
                and self.hyper_params.img_size is not None:

                out = pygame.transform.scale(self.display, self.hyper_params.img_size)
                return surf_to_ndarray(out) if not self.gray \
                    else np.expand_dims(rgb_to_gray(surf_to_ndarray(out)), axis=0)
            else:
                return surf_to_ndarray(self.display) if not self.gray \
                    else np.expand_dims(
                        rgb_to_gray(surf_to_ndarray(self.display)),
                        axis=0)

    def get_now_info(self) -> str:
        return f"Metadata:\n{self.metadata}\nHyperParams:\n{self.hyper_params}"
    
    def reset(self)->np.ndarray:
        self.engine.clear_all()
        self.load_task(self.task_path, False)
        self.now_step = 0
        return self.render()

    def close(self)->None:
        ...

    def _draw_guidewire(self, surface:pygame.Surface, color=None, width=None,):
        """ 绘制导丝 """
        g_color = G_RENDER_RGB if color is None else color
        g_width = G_RENDER_WIDTH if width is None else width

        points = []
        # 获取导丝的位姿信息并绘制导丝
        for i in self.engine.balls:
            points.append(i.body.position.int_tuple)
        if len(points) > 1:
            pygame.draw.lines(surface, g_color, False, points, g_width)
            pygame.draw.circle(surface, g_color, points[-1], g_width//2)

    def get_now_tip_pos(self) -> list:
        """获取导丝尖部坐标"""
        # 防止出界
        pos = self.engine.get_guide_tip_pos()
        pos[0] = SIMULATION_HIGHT-1 if pos[0] > SIMULATION_HIGHT else pos[0]
        pos[0] = 0 if pos[0] < 0 else pos[0]
        pos[1] = SIMULATION_WIDTH-1 if pos[1] > SIMULATION_WIDTH else pos[1]
        pos[1] = 0 if pos[1] < 0 else pos[1]
        return pos
    
    def get_now_target_pos(self) -> list:
        """获取目标点坐标"""
        return self.metadata.target_pos[::-1]


if __name__ == "__main__":
    eval_env = GuidewireEnv()
