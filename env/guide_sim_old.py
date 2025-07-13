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
log_format = "%(asctime)s - %(levelname)s - %(message)s"

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



class BaseEnv():
    done:bool     = True
    max_steps:int = None
    def __init__(self) -> None:
        ...
    def step(self, action:int)->tuple[np.ndarray, float, bool, dict]:
        ...
    def reset(self)->np.ndarray:
        ...
    def render(self)->np.ndarray:
        ...
    def close(self)->None:
        ...
    def set_params(self, param):
        """将param中的参数赋予本环境"""
        for key in dir(param):
            if not callable(getattr(param, key)) and not key.startswith('__'):
                setattr(self, key, getattr(param, key))


INSERT_LIMIT = 100
RADIUS = 5
DATASET_DIR = "./datas/"
GROUP = "real_x_ray"
G_RENDER_WIDTH  = 4 
G_RENDER_RGB  = (43, 43, 43) 
DEBUG_DRAW = False
SIMULATION_WIDTH = 512
SIMULATION_HIGHT = 512
CLIP_X = None
CLIP_Y = None

def rgb_to_gray(rgb:np.ndarray):
    return np.array(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))

def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

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

def angle_between_points(point1:tuple, point2:tuple):
    x_diff = point2[0] - point1[0]
    y_diff = point2[1] - point1[1]
    return math.degrees(math.atan2(y_diff, x_diff))

def num_compare(name:str):
    return int(name.split(".")[0])


class GuidewireEnv(BaseEnv):
    def __init__(self, group=None):
        logging.info("Initializing GuidewireEnv ...")
        # num_envirment = len(os.listdir())
        self.engine :simulation.GuideWireEngine  = None
        self.original_dir  :str = None
        self.mask_dir      :str = None 
        self.map_json_dir  :str = None 
        self.group = group
        self.original_images:list[str] = None
        self.mask_images:list[str] = None
        self.map_jsons:list[str] = None
        self.step_punishment = 0.05
        if group is not None:
            self._get_path_init(self.group)
            self.num_datas = len(self.map_jsons)
        else:
            self.num_datas = 0
        self.image_size:tuple[int, int] = None
        self.g_angle = None
        self.now_step = 0
        self.g_surface:pygame.Surface = None
        self.display:pygame.Surface = None
        self.goal_rect_color = (255, 255, 255)
        self.max_steps = 300
        self._now_json = None
        self.draw_options = None
        self.gray = True
        self.clip_x = CLIP_X
        self.clip_y = CLIP_Y
        self.point_scale = 1.0  # 导丝标记的缩放比例
        if group is not None:
            self.reset()


    def step(self, action:int, debug=False)->tuple[np.ndarray, float, bool, dict]:
        self.now_step += 1
        exact = 1000
        action = int(action)
        # 输入动作
        if action == 0:
            if len(self.engine.balls) <= INSERT_LIMIT:
                self.engine.insert_a_ball(radius=RADIUS, direct_angle=self.g_angle)
        elif action == 1:
            self.engine.pull_back_a_ball(direct_angle=self.g_angle)
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
        debug = self.engine.balls[-1].body.position.int_tuple
        debug2 = self.engine.goals[-1].body.position.int_tuple
        # if self.goal_rect_color is not None:
        #     w, h = 40, 40
        #     rect = (debug2[0] - w/2, debug2[1] - h/2, w, h)
        #     pygame.draw.rect(self.display, self.goal_rect_color, rect, 5)
        #     # rect=(debug[0] - w/2, debug[1] - h/2, w, h)
        #     # pygame.draw.rect(self.screen, (255, 0, 0), rect, 5)
        reward = (self.step_punishment) / self.max_steps
        done = False
        if self.engine.done:
            done = True	
    
        s = self.render()
        s = np.array(s, dtype=np.float32) / 255.
        # s = surf_to_ndarray(self.display)
        if self.now_step >= self.max_steps:
            done = True
        
        if done:
            distance = pow((pow(debug[0] - debug2[0], 2) + pow(debug[1] - debug2[1], 2)), 0.5) * 0.001		
            reward = - math.log(distance)
        # print("s",s.shape)
        return s, reward, done, (debug, debug2, self.now_step)
    
    def set_params(self, param):
        """将param中的参数赋予本环境"""
        super().set_params(param)
        self._get_path_init(self.group)
        self.num_datas = len(self.map_jsons)
        self.reset()

    def reset(self, index=None)->np.ndarray:
        if self.group is not None:
            self.now_step = 0
            del self.engine
            self.engine = simulation.GuideWireEngine()
            try:
                self._load_env(index)
            except Exception as e:
                logging.info(f"{e}")
                raise e
            return self.render()
        else:
            raise Exception("Dataset group is not set, please set it first.")
    
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
        self.display.blit(self.bg, (0,0))
        if DEBUG_DRAW:        
            self.engine.space.debug_draw(self.draw_options)  # 绘制新图案
        self.display.blit(surf_blured, (0,0))

        # debug = self.engine.balls[-1].body.position.int_tuple
        debug2 = self.engine.goals[-1].body.position.int_tuple
        if self.goal_rect_color is not None:
            w, h = 40, 40
            rect = (debug2[0] - w/2, debug2[1] - h/2, w, h)
            pygame.draw.rect(self.display, self.goal_rect_color, rect, 5)
        
        if _return_array:
            if self.display.get_size() != self.image_size and self.image_size is not None:
                out = pygame.transform.scale(self.display, self.image_size)
                return surf_to_ndarray(out) if not self.gray \
                    else np.expand_dims(rgb_to_gray(surf_to_ndarray(out)), axis=0)
                    # else rgb_to_gray(surf_to_ndarray(out))
            else:
                return surf_to_ndarray(self.display) if not self.gray \
                    else rgb_to_gray(surf_to_ndarray(self.display))
        
    def get_now_info(self) -> str:
        return self._now_json
    
    def close(self)->None:
        self.reset()

    def _load_env(self, index=None):
        if index is None:
            if self.num_datas==1:
                index = 0
            else:
                index = random.randint(0, self.num_datas-1)
        # f_name = self.map_jsons[index].split("\\")[-1].split(".")[0]
        f_name = os.path.splitext(os.path.basename(self.map_jsons[index]))[0]
        
        self._now_json = self.map_jsons[index]
        original_image = os.path.join(self.original_dir, f"{f_name}.png")
        mask_image = os.path.join(self.mask_dir, f"{f_name}.png")

        # logging.info(f"Loading {self.map_jsons[index]}, {original_image}, {mask_image}...")
        points = load_data(self.map_jsons[index])
        self._points = points
        self.engine.clear_all()
        self.engine.create_goal((0,0))
        # bg   = pygame.image.load(self.original_images[index])
        # mask = pygame.image.load(self.mask_images[index])
        bg   = pygame.image.load(original_image)
        mask = pygame.image.load(mask_image)
        if bg.get_size() != mask.get_size():
            raise Exception("The size of the background and mask is not the same.")
        bg, mask = self.load_image_mask(bg, mask)
        self.bg   = pygame.transform.scale(bg, (SIMULATION_WIDTH, SIMULATION_HIGHT))
        self.mask = pygame.transform.scale(mask, (SIMULATION_WIDTH, SIMULATION_HIGHT))
        self.engine.draw_walls_by_mask(self.mask, friction=0.1)
        self.engine.clear_goal()
        self.display = pygame.Surface(self.mask.get_size())
        self.g_surface = pygame.Surface(self.mask.get_size(), pygame.SRCALPHA)
        self.g_angle = angle_between_points(points[0], points[1])
        # logging.info(f"Map loaded , angle:{self.g_angle}, goal:{points[2]}")
        self.engine.create_ini_guidewire(RADIUS, (int(points[0][0] * self.point_scale),
                                                    int(points[0][1] * self.point_scale)),
                                        2, self.g_angle)
        self.engine.create_goal((int(points[2][0] * self.point_scale) ,
                                int(points[2][1] * self.point_scale)))
        if DEBUG_DRAW:
            self.draw_options = pymunk.pygame_util.DrawOptions(self.display)
    
    def load_image_mask(self, bg, mask):
        # self.screen.blit(self.mask, (0,0))
        # clip_rect = pygame.Rect(0, 0, 500, 500)
        # 裁剪表面
        if self.clip_x is not None or self.clip_y is not None:
            _bg = pygame.Surface((self.clip_x, self.clip_y if self.clip_y \
                                is not None else bg.get_height()))
            _msk = pygame.Surface((self.clip_x, self.clip_y if self.clip_y \
                                is not None else bg.get_height()))
            _bg.blit(bg, (0, 0))
            _msk.blit(mask, (0, 0))
            # self.bg.blit(_bg, (0, 0))
            # self.mask.blit(_msk, (0, 0))
            bg = _bg
            mask = _msk
        bg   = pygame.transform.scale(bg, (SIMULATION_WIDTH, SIMULATION_HIGHT))
        mask = pygame.transform.scale(mask, (SIMULATION_WIDTH, SIMULATION_HIGHT))
        return bg, mask
    
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


    def __len__(self):
        return len(self.map_jsons) if self.map_jsons is not None else 0

    
    def _get_path_init(self, group=None):
        """初始化路径存储器"""
        group_dir = os.path.join(DATASET_DIR, group)
        self.original_dir  = os.path.join(group_dir, "images")
        self.mask_dir      = os.path.join(group_dir, "label")
        self.map_json_dir = os.path.join(group_dir, "envmsgs")
        map_jsons     = os.listdir(self.map_json_dir)
        self.map_jsons = [os.path.join(self.map_json_dir, f_name) for f_name in map_jsons]
    
    def get_now_tip_pos(self):
        return self.engine.balls[-1].body.position.int_tuple


if __name__ == "__main__":
    eval_env = GuidewireEnv()

    import params
    ENV_NAME = "resnet50_vivo"
    env_param   = params.EnvParams()
    train_param = params.TrainParams()
    env_param   = params.EnvParams()
    run_param    = params.RuntimeParams()


    train_param.load_from_json(f'./env/{ENV_NAME}/params')
    env_param.load_from_json(f'./env/{ENV_NAME}/params')
    run_param.load_from_json(f'./env/{ENV_NAME}/params')
    eval_env.set_params(env_param)