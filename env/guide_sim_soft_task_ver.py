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

def detect_self_collision(points, min_distance, ignore_adjacent=5):
    """
    检查给定点集是否发生自碰撞

    `points`: `np.array` of shape (N, 3) or (N, 2)
    
    `min_distance`: 或安全距离阈值
    
    `ignore_adjacent`: 忽略相邻的几个节点（如 ignore_adjacent=2 表示忽略 i±2 以内）
    """

    N = len(points)
    points = np.array(points)
    for i in range(N):
        for j in range(i + ignore_adjacent + 1, N):
            dist = np.linalg.norm(points[i] - points[j])
            if dist < min_distance:
                return True
    return False
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

class PointNormalLine:
    def __init__(self, p0, n1, n2):
        """
        p0: (x0, y0)  直线上一点
        n1, n2: (x1, y1), (x2, y2) 用两点定义法线方向
        """
        self.p0 = np.array(p0, dtype=np.float32)
        n1 = np.array(n1, dtype=np.float32)
        n2 = np.array(n2, dtype=np.float32)

        self.n = n2 - n1
        self.n = self.perp(self.n)
        norm = np.linalg.norm(self.n)
        if norm < 1e-6:
            raise ValueError("法线方向两点不能重合")

        self.n /= norm  # 归一化，方便数值稳定
    
    def point_side(self, p, eps=2):
        """
        判断点 p 在直线哪一侧
        返回：
          +1  -> 法线指向的一侧（约定为“右侧”）
          -1  -> 法线反方向的一侧（约定为“左侧”）
           0  -> 在直线上
        """
        p = np.array(p, dtype=np.float32)
        s = np.dot(self.n, p - self.p0)

        if s > eps:
            return True
        elif s < -eps:
            return False
        else:
            return False
    def perp(self, v):
        # 逆时针旋转90°（如果你觉得左右反了，就换成 [v[1], -v[0]]）
        return np.array([-v[1], v[0]], dtype=np.float32)
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
    
    return int(np.sum(a_star_path))

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
        # logging.info("Initializing GuidewireEnv ...")
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
        self._mask_surf_np = None  # [512 * 512]
        self.a_star_path_np = None
        self.load_task(task_path)
        # logging.info("Load metadata from %s", task_path)
        
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

        self.last_a_star = None
        self.inial_a_star = None
        self.soft_task_dist = 60
        self.iniallized = False
        self.line = None

    def load_task(self, task_path:str, file_reload=True):
        """加载任务"""
        self.engine.clear_all()
        
        if file_reload:
            self.metadata.load_from_json(task_path)
            # Win和Linux的兼容性考虑
            self.metadata.background_path = \
                self.metadata.background_path.replace("\\", "/")
            self.metadata.mask_path = \
                self.metadata.mask_path.replace("\\", "/")
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
        self.line = PointNormalLine(self.metadata.insert_pos, self.metadata.insert_pos, self.metadata.target_pos)
        self.iniallized = True

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
        pos_g_tip = self.get_now_tip_pos()
        pos_target = self.get_now_target_pos()
        done = False
        now_dis = self.get_a_star_distance()  # self.a_star_path_np也会被更新
        is_collision = detect_self_collision(self.engine.get_guide_pos_list(),
                                              2.5*self.metadata.radius,
                                              )
        if l2_is_done(pos_g_tip, pos_target, 4 * self.metadata.radius):
            done = True
            now_dis = 4 * self.metadata.radius
            # reward = -math.log( l2_distance_square(pos_g_tip, pos_target) ** 0.5 * 0.001)
            # 成功到达终点之后的稀疏奖励
            # reward = -math.log( ( 4 * self.metadata.radius / self.inial_a_star)  * 0.05 + 0.0001)
            reward = math.log(self.inial_a_star / ( now_dis**2 / self.inial_a_star + 1e-5) )
        else:
            d_penalty = 0
            if is_collision and action!=1: # 惩罚缠绕但是不执行导丝撤回的情况
                d_penalty = 4.0 * self.metadata.radius  # 相当于因缠绕损失了 5节点 的有效推进
            reward_dis = now_dis + d_penalty
            # 没有到达终点，但是已经超出了最大步数限制
            if self.now_step >= self.hyper_params.max_steps:
                done = True
                # 没有到达终点的最终稀疏奖励，使用最终距离
                reward = math.log(self.inial_a_star / ( (reward_dis)**2 / self.inial_a_star + 1e-5) )
            else:
                # 使用A*距离的差值作为密集奖励
                reward = (self.last_a_star - reward_dis) * 10. / self.inial_a_star
                reward = reward * 1.25 if reward < 0. else reward   # 远离终点时，会有额外的惩罚

        self.last_a_star = now_dis
        
        s = self.render()
        # s = np.array(s, dtype=np.float32) / 255.

        return s, reward, done, (pos_g_tip, pos_target, self.now_step, is_collision)

    def get_a_star_path(self)->np.ndarray:
        return self.a_star_path_np

    def get_a_star_distance(self):
        pos_g_tip = self.get_now_tip_pos()
        pos_target = self.get_now_target_pos()
        self.a_star_path_np = a_star(self._mask_surf_np,
                            pos_g_tip,
                            pos_target)
        return a_star_distance(self.a_star_path_np)

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
    

    def sample_goal(self, dist):
        """
        以当前导丝的尖端坐标为圆心，dist为半径，随机生成一个点，作为新的目标点.
        这个点必须位于MASK中的合法位置，否则重新采样，直到采样到合法的点。
        """
        # 最大采样次数，都失败的话就放弃采样
        max_sample_times = 200
        # 小于这个距离的点会被拒绝
        rejection_min_dist = max(4*self.metadata.radius, dist*0.5)
        entry = np.array(self.metadata.insert_pos, dtype=np.float32)
        # print(f"rejection_min_dist: {rejection_min_dist}")
        theta = self._angle * np.pi / 180.0
        forward = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        h, w = self._mask_surf_np.shape

        for i in range(max_sample_times):
            pos = self.get_now_tip_pos()
            # 逆高斯采样，即边缘的概率更高
            pos = (pos[0] + (1-np.random.normal(0, dist)), pos[1] + (1-np.random.normal(0, dist)))
            sampled_dist = l2_distance_square(pos, self.get_now_tip_pos())
            # 如果采样到的坐标离圆心太近，则拒绝采样
            if sampled_dist < rejection_min_dist**2:
                continue
            # 如果采样到的坐标在圆外，则依旧拒绝采样
            if sampled_dist > dist**2:
                continue
            # 越界直接拒绝
            x, y = int(pos[0]), int(pos[1])
            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            # 拒绝插入点背面区域
            # if self.line.point_side(pos):
            #     continue
            if len(self.engine.balls) < 6:
                if not self.line.point_side(pos[::-1]):
                    # print(f"拒绝采样到背后的点 {pos}")
                    continue
            if self._mask_surf_np[int(pos[0]), int(pos[1])] == 255:
                # print(f"Sample goal: {l2_distance_square(pos, self.get_now_tip_pos()) ** 0.5}")
                return pos[::-1]
        else:
            return None

    def reset(self, debug=True)->np.ndarray:
        # self.engine.clear_all()
        if self.iniallized:
            is_collision = detect_self_collision(self.engine.get_guide_pos_list(),
                                                2.5*self.metadata.radius,
                                                )
            resampled_target = self.sample_goal(self.soft_task_dist)
        else:
            if debug:
                print(f"not iniallized")
            is_collision = False
            resampled_target = None
        if not is_collision and resampled_target is not None:
            if debug:
                print(f"is_collision {is_collision} resampled_target {resampled_target}")
            self.set_now_target_pos(resampled_target)
        else:
            del self.engine
            self.engine = simulation.GuideWireEngine()
            self.load_task(self.task_path, False)
            resampled_target = self.sample_goal(self.soft_task_dist)
            if resampled_target is not None:
                self.set_now_target_pos(resampled_target)
            if debug:
                print(f"reset all")
        self.now_step = 0
        self.last_a_star = self.get_a_star_distance()
        self.inial_a_star = self.last_a_star
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

    def set_now_target_pos(self, pos_like) -> list:
        """获取目标点坐标"""
        pos_like = list(map(int, pos_like))
        self.metadata.target_pos = pos_like
    
    def set_soft_task_threashold(self, dist:float) -> None:
        """设置软任务生成距离"""
        self.soft_task_dist = dist


if __name__ == "__main__":
    eval_env = GuidewireEnv()
