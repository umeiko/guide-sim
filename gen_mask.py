# 按S保存掩码mask, 仅在windows下测试过
# 按R重置
# 按G生成掩码
# 按住鼠标左键画线
# 按住鼠标左键+shift放出物理小球
import pygame
import sys
import pymunk
import pymunk.autogeometry
import pymunk.pygame_util
from pymunk import BB
from tqdm import tqdm
import os


MOUSE_WIDTH = 12
FILE_NAME = "2.png"
IN_PATH  = "datas/exvivo4/images/"  # 输入的图片路径
OUT_PATH = "datas/exvivo4/label/"

# 图像尺寸重构
SIZE        = [512, 512]
def draw_helptext(screen):
    font = pygame.font.Font(None, 16)
    text = [
        "LMB(hold): Draw pink color",
        "LMB(hold) + Shift: Create balls",
        "g: Generate segments from pink color drawing",
        "r: Reset",
    ]
    y = 5
    for line in text:
        text = font.render(line, 1, pygame.Color("black"))
        screen.blit(text, (5, y))
        y += 10


def generate_geometry(surface, space):
    for s in space.shapes:
        if hasattr(s, "generated") and s.generated:
            space.remove(s)

    def sample_func(point):
        try:
            p = int(point[0]), int(point[1])
            color = surface.get_at(p)
            return color.hsla[2]  # use lightness
        except Exception as e:
            print(e)
            return 0

    line_set = pymunk.autogeometry.march_soft(
        BB(0, 0, SIZE[0]-1, SIZE[1]-1), SIZE[0]//10, SIZE[1]//10, 90, sample_func
    )

    for polyline in line_set:
        line = pymunk.autogeometry.simplify_curves(polyline, 1.0)

        for i in range(len(line) - 1):
            p1 = line[i]
            p2 = line[i + 1]
            shape = pymunk.Segment(space.static_body, p1, p2, 1)
            shape.friction = 0.5
            shape.color = pygame.Color("red")
            shape.generated = True
            space.add(shape)


def copy_and_modify_surface(terrain_surface):
    # 创建一个新的表面，大小与 terrain_surface 相同
    save_surf = pygame.Surface(terrain_surface.get_size())
    # 锁定表面以提高像素操作的效率
    terrain_surface.lock()
    save_surf.lock()
    # 遍历每个像素
    for x in tqdm(range(terrain_surface.get_width()), desc="Processing rows"):
        for y in range(terrain_surface.get_height()):
            color = terrain_surface.get_at((x, y))
            if color == pygame.Color("pink"):
                save_surf.set_at((x, y), pygame.Color("white"))
            else:
                save_surf.set_at((x, y), pygame.Color("black"))
    # 解锁表面
    terrain_surface.unlock()
    save_surf.unlock()
    return save_surf

def main():
    clock = pygame.time.Clock()
    pygame.init()
    bg_surface = pygame.image.load(os.path.join(IN_PATH, FILE_NAME))
    # bg_surface = pygame.transform.scale(bg_surface, SIZE)
    display_surface = pygame.display.set_mode(SIZE)
    bg_surface = pygame.transform.scale(bg_surface, SIZE)
    space = pymunk.Space()
    space.gravity = 0, 980
    def pre_solve(arb, space, data):
        s = arb.shapes[0]
        space.remove(s.body, s)
        return False

    space.add_collision_handler(0, 1).pre_solve = pre_solve
    terrain_surface = pygame.Surface(bg_surface.get_size())
    terrain_surface.fill(pygame.Color("white"))
    terrain_surface.set_alpha(50)
    color = pygame.color.THECOLORS["pink"]
    pygame.draw.circle(terrain_surface, color, (450, 120), 100)
    generate_geometry(terrain_surface, space)
    for _ in range(25):
        mass = 1
        moment = pymunk.moment_for_circle(mass, 0, 10)
        body = pymunk.Body(mass, moment)
        body.position = 450, 120
        shape = pymunk.Circle(body, 10)
        shape.friction = 0.5
        space.add(body, shape)
    
    draw_options = pymunk.pygame_util.DrawOptions(display_surface)
    pymunk.pygame_util.positive_y_is_up = False
    fps = 60
    pygame.display.set_caption(f"pymunk_{bg_surface.get_size()}")
    while True:
        for event in pygame.event.get():
            if (
                event.type == pygame.QUIT
                or event.type == pygame.KEYDOWN
                and (event.key in [pygame.K_ESCAPE, pygame.K_q])
            ):
                sys.exit(0)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                pass
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                terrain_surface.fill(pygame.Color("white"))
                for s in space.shapes:
                    if hasattr(s, "generated") and s.generated:
                        space.remove(s)

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_g:
                generate_geometry(terrain_surface, space)

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                pygame.image.save(display_surface, "deformable.png")
            
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                msk = copy_and_modify_surface(terrain_surface)
                pygame.image.save(msk, os.path.join(OUT_PATH, FILE_NAME))
            
            elif event.type == pygame.MOUSEMOTION:
                pygame.display.set_caption(f"pymunk_{bg_surface.get_size()} {event.pos}")
        
        if pygame.mouse.get_pressed()[0]:
            if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                mass = 1
                moment = pymunk.moment_for_circle(mass, 0, 10)
                body = pymunk.Body(mass, moment)
                body.position = pygame.mouse.get_pos()
                shape = pymunk.Circle(body, 10)
                shape.friction = 0.5
                space.add(body, shape)
            else:
                color = pygame.Color("pink")
                pos = pygame.mouse.get_pos()
                pygame.draw.circle(terrain_surface, color, pos, MOUSE_WIDTH)
        

        space.step(1.0 / fps)
        
        display_surface.blit(bg_surface, (0, 0))
        display_surface.blit(terrain_surface, (0, 0), )
        space.debug_draw(draw_options)
        draw_helptext(display_surface)
        pygame.display.flip()
        clock.tick(fps)


if __name__ == "__main__":
    sys.exit(main())