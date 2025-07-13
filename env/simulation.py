

# 此代码提供二维环境下的介入导丝仿真
import pygame
import pymunk
import pymunk.constraints
import pymunk.pygame_util
import numpy as np
import matplotlib.pyplot as plt
import pymunk.autogeometry
from pymunk import BB
import math


def generate_geometry(surface:pygame.Surface, space, 
                      sample_rate = 8,
                      threshold = 50,
                      friction  = 0.5
                      ):
    '''利用图形掩膜生成几何学边框'''
    size_background = surface.get_size()
    for s in space.shapes:
        if hasattr(s, "generated") and s.generated:
            space.remove(s)

    def sample_func(point):
        try:
            p = int(point[0]), int(point[1])
            color = surface.get_at(p)
            # if color.hsla[2] < 100:
            #     print(point, color.hsla[2])
            return color.hsla[2]  # use lightness
        except Exception as e:
            print(e)
            return 0

    line_set = pymunk.autogeometry.march_soft(
        BB(0, 0, size_background[0]-1, size_background[1]-1), 
        size_background[0]//sample_rate, size_background[1]//sample_rate, threshold, sample_func
    )

    for polyline in line_set:
        line = pymunk.autogeometry.simplify_curves(polyline, 1.0)

        for i in range(len(line) - 1):
            p1 = line[i]
            p2 = line[i + 1]
            shape = pymunk.Segment(space.static_body, p1, p2, 1)
            shape.friction = friction
            shape.color = pygame.Color("red")
            shape.generated = True
            space.add(shape)


class GuideWireEngine:
    def __init__(self):    
        self.space = pymunk.Space()
        # self.handler = self.space.add_default_collision_handler()
        # self.handler.begin = self.coll_begin
        self.done = False  # 是否到达肿瘤
        self.guidwire_fric = 0.5
        self.wall_fric     = 0.3

        self.balls:list[pymunk.Shape] = []     # 节点装在这里
        self.positions = [] # 每个节点的位置装在这里
        self.connects = []  # 连接装在这里
        self.angle = 0.0    # 前端的弯曲角度
        self.len_of_bend = 5   # 弯曲的长度
        self.walls = []  # 血管壁装在这里
        self.goals = []  # 肿瘤装在这里

    def draw_walls_by_mask(self, Surf:pygame.Surface, 
                      sample_rate = 5,
                      threshold = 50,
                      friction  = 0.5 ):
        generate_geometry(Surf, self.space, sample_rate, threshold, friction)


    # def coll_begin(self, arbiter, space, data):
    #     if (arbiter.shapes[0] in self.goals) or (arbiter.shapes[1] in self.goals):
    #         self.done = True
    #     return True

    def create_a_ball(self, ball_type="dynamic", mass=1, radius=7, position=(300, 300),
                        elasticity=0.1, friction=0.3, insert=False):        
        inertia = pymunk.moment_for_circle(mass, 0, radius, position)  # 转动惯量
        # 创建对象
        if ball_type == 'dynamic':
            body = pymunk.Body(mass, inertia)
        elif ball_type == 'kinematic': 
            body = pymunk.Body(mass, inertia, body_type=pymunk.Body.KINEMATIC)
        else:
            raise Exception("ball_type must be dynamic or kinematic")
        body.position = position[0], position[1]
        # 载入形状
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = elasticity
        shape.friction = friction
        # 加入引擎空间中
        self.space.add(body, shape)
        if insert:
            self.balls.insert(1, shape)
        else:
            self.balls.append(shape)


    def create_connect(self, prev_shape, next_shape,
                        angle=0, limit=30, ela=1e8, dump=1e7, gen_angle=0):  # 创造两个球之间的连接
        prev_body = prev_shape.body
        next_body = next_shape.body

        dx = math.cos(gen_angle * math.pi / 180) * (prev_shape.radius+1)
        dy = math.sin(gen_angle * math.pi / 180) * (prev_shape.radius+1)
        string = pymunk.constraints.DampedRotarySpring(prev_body,
                                                       next_body,
                                                       angle * math.pi / 180, ela, dump)  # 弹性副
        pivot = pymunk.constraints.PivotJoint(prev_body, next_body, 
                                                (dx, dy), (-dx, -dy), )  # 旋转副
        pivot.collide_bodies = True
        joint = pymunk.constraints.RotaryLimitJoint(prev_body,
                                                    next_body,
                                                    -limit * math.pi / 180, limit * math.pi / 180)  # 限制副
        self.space.add(string)
        self.space.add(pivot)
        self.space.add(joint)
        return [string, pivot, joint]


    def insert_a_ball(self, radius=7, direct_angle=0):  # 插入一个球
        ini_pos = self.balls[0].body.position

        # 删除第一个连接
        for i in self.connects[0]:
            self.space.remove(i)
        self.connects.remove(self.connects[0])

        d = self.balls[0].radius+radius+1
        dx = math.cos(direct_angle* math.pi / 180) * d
        dy = math.sin(direct_angle* math.pi / 180) * d
        # 创建活动球
        self.create_a_ball(radius=radius,friction=self.guidwire_fric,position=(ini_pos[0]+dx, ini_pos[1]+dy), insert=True)
        # 创建左边的连接
        c1 = self.create_connect(self.balls[0], self.balls[1], angle=0, limit=90, gen_angle=direct_angle)
        # 创建右边的连接
        if len(self.balls) <= self.len_of_bend:
            c2 = self.create_connect(self.balls[1], self.balls[2], angle=self.angle, limit=90, gen_angle=direct_angle)
        else:
            c2 = self.create_connect(self.balls[1], self.balls[2], angle=0, limit=90, gen_angle=direct_angle)

        self.connects.insert(0, c2)
        self.connects.insert(0, c1) 

        
    def pull_back_a_ball(self, direct_angle=0):
        if len(self.balls) > 2:
            self.space.remove(self.balls[1])
            self.balls.pop(1)
            for i in self.connects[0]:
                self.space.remove(i)
            for i in self.connects[1]:
                self.space.remove(i)
            self.connects.pop(0)
            self.connects.pop(0)
            c = self.create_connect(self.balls[0], self.balls[1], 0, 90, gen_angle=direct_angle)
            self.connects.insert(0, c) 


    def create_ini_guidewire(self, radius=7, start_position=(0, 345), ini_len=20, direct_angle=0):  # 创建一个初始导丝
        """建立初始导丝"""
        self.create_a_ball(ball_type='kinematic', friction=self.guidwire_fric, radius=radius, position=start_position)
        now_x = start_position[0]
        now_y = start_position[1]
        for i in range(ini_len):
            d = 2 * radius + 1
            dx = math.cos(direct_angle* math.pi / 180) * d
            dy = math.sin(direct_angle* math.pi / 180) * d
            now_x += dx
            now_y += dy
            self.create_a_ball(radius=radius,position=(now_x, now_y), friction=self.guidwire_fric,)
            c = self.create_connect(self.balls[i], self.balls[i+1], 0, 90, gen_angle=direct_angle)
            self.connects.append(c)

    def set_guide_by_list(self, old_balls, radius, direct_angle):
        """通过元素列表绘制导丝"""
        self.clear_guide_wire()
        self.create_a_ball(ball_type='kinematic', friction=self.guidwire_fric, radius=radius, position=old_balls[0])
        for i in range(len(old_balls)-1):
            self.create_a_ball(radius=radius, friction=self.guidwire_fric, position=old_balls[i+1])
            c = self.create_connect(self.balls[i], self.balls[i+1], 0, 90, gen_angle=direct_angle)
            self.connects.append(c)
        for _ in range(3000):
            self.space.step(1 / 1000)

    def bend_guidewire(self, d_angle=10):
        self.angle += d_angle
        if self.angle >= 30:
            self.angle = 30
        elif self.angle <= -30:
            self.angle = -30
        for k, i in enumerate(self.connects[::-1]):
            if k <= self.len_of_bend:
                i[0].rest_angle = self.angle * 3.14 / 180

    def draw_a_line(self, p1, p2):
        static_body = self.space.static_body
        
        line = pymunk.Segment(static_body, p1, p2, 5.0)
        line.elasticity = 0.95  # 弹性系数 0-1
        line.friction = 0.4  # 摩擦系数 0-1        
        self.walls.append(line)
        self.space.add(line)    

    def draw_walls(self, wall_list):
        for i in wall_list:
            self.draw_a_line(i[0], i[1])

    def create_goal(self, position=(100, 100)):
        body = self.space.static_body
        body.position = position[0], position[1]
        shape = pymunk.Circle(body, 10, (0, 0))
        shape.elasticity = 0.95
        shape.friction = 0.4
        self.space.add(shape)
        self.goals.append(shape)


    def clear_all(self):  # 清空环境
        for i in self.balls:
            self.space.remove(i)
        for i in self.connects:
            for j in i:
                self.space.remove(j)
        for i in self.walls:
            self.space.remove(i)
        # for i in self.goals:
        #     self.space.remove(i)
        self.done = False  # 是否到达肿瘤
        self.balls.clear()     # 节点装在这里
        self.connects.clear()  # 连接装在这里
        self.angle = 0.0    # 前端的弯曲角度
        self.len_of_bend = 5   # 弯曲的长度
        self.walls.clear()  # 血管壁装在这里
        self.clear_goal()
        # self.goals.clear()
    
    def clear_goal(self):
        for i in self.goals:
            self.space.remove(i)
        self.goals.clear()

    def get_guide_pos_list(self) -> list:
        return [list(i.body.position) for i in self.balls]
    
    def get_guide_tip_pos(self) -> list:
        return list(self.balls[-1].body.position.int_tuple)[::-1]
    
    def clear_guide_wire(self):
        """删除导丝"""
        for i in self.balls:
            self.space.remove(i)
        for i in self.connects:
            for j in i:
                self.space.remove(j)
        self.balls.clear()
        self.connects.clear()
        self.done = False  # 是否到达肿瘤
        self.angle = 0.0    # 前端的弯曲角度


        

def main():
    env = GuideWireEngine()
    
    pygame.init()    
    # 创建一个绘图界面
    screen = pygame.Surface((1200, 600))  
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    exact = 100  # 每step计算100次
    
    env.create_ini_guidewire()
    env.bend_guidewire(10)

    while True:  # 主循环    
        # 计算下一秒的画面
        for x in range(exact):
            # step()输入为秒，即物理引擎中时间过去多少秒。过大的step会导致计算结果及其不稳定。
            env.space.step(1 / exact)
        
        screen.fill(pygame.Color("white"))  # 清空屏幕
        env.space.debug_draw(draw_options)  # 绘制新屏幕

        x3 = np.array(pygame.surfarray.pixels3d(screen))
        print(x3.shape)
        print(x3)
        plt.figure()
        plt.imshow(x3)
        plt.show()


if __name__ == "__main__":
    main()