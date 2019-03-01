from typing import List, Deque, Tuple

import pygame as pg
import random as rnd
import os

from functools import reduce
from collections import defaultdict, deque


def dummy(*args, **kwargs):
    pass


info  = print
log   = print
debug = dummy
spam  = dummy


class Settings:
    hover = True


class Viewport:
    def __init__(self, w, hor_sz, ver_sz):
        self.x, self.y = 0, 0
        self.w, self.h = w, w
        self.hor_sz, self.ver_sz = hor_sz, ver_sz

    def get_coord(self, v, h):
        """Left-top corner coords"""
        return self.x + v * self.w, self.y + h * self.w

    def get_cell_coords(self, x, y) -> tuple:
        """Cell rect coords -- a pair of pairs"""
        return self.get_coord(x, y), self.get_coord(x+1, y+1)

    def get_cell(self, xm, ym):
        w = self.w
        return (xm - self.x) // w, (ym - self.y) // w


class Collision:
    NONE = 0
    GROUND = 1
    UNIT = 2
    FLIGHT = 4

class Entity:
    collision = Collision.NONE

    def __init__(self, x, y):
        self.x, self.y = x, y
        self.passed = 0
        self.start = x, y
        self.action = None

    def get_draw_point(self):
        sx, sy = self.start
        tx, ty = self.x, self.y
        p = self.passed
        return sx + (tx - sx) * p, sy + (ty - sy) * p


class Tile(Entity):
    color_name = None

    def __init__(self, x, y):
        super().__init__(x, y)
        self.color = pg.Color(self.color_name)
        self.hover_color = self.color.correct_gamma(0.7)
        self.hover = False

    def draw(self, surf: pg.Surface, viewport: Viewport):
        if self.hover:
            clr = self.hover_color
        else:
            clr = self.color
        x, y = viewport.get_coord(self.x, self.y)
        w, h = viewport.w, viewport.h
        pg.draw.rect(surf, clr, (x, y, w, h))


class Grassland(Tile):
    color_name = 'MEDIUMSEAGREEN'
    collision = Collision.NONE


class Tree(Tile):
    color_name = 'FORESTGREEN'
    collision = Collision.GROUND


class Rock(Tile):
    color_name = 'GRAY'

    collision = Collision.GROUND


class AssetManager:
    asset_requests = defaultdict(lambda: [])
    
    @classmethod
    def require_asset(cls, name, storage):
        cls.asset_requests[name].append(storage)
        log('require asset', name)

    @classmethod
    def load(cls, hor, ver):
        img = pg.image.load(os.path.join('assets', 'worker_01.png'))
        asset = pg.transform.scale(img, (hor, ver))
        cls.asset_requests['worker'][0][''] = asset


class UnitState:
    IDLE = 'IDLE'
    TRANSITION = 'TRANS'
    MOVE = 'MOVE'


class Unit(Entity):
    model = 'error'
    collision = Collision.NONE

    def __init__(self, x, y):
        super().__init__(x, y)
        self.img_state = ''
        self.state_queue = [(UnitState.IDLE, [])]
        self.target = None

    def draw(self, surf: pg.Surface, viewport: Viewport):
        draw_coords = self.get_draw_point()
        p = viewport.get_coord(*draw_coords)
        surf.blit(self.assets[self.img_state], p)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.assets = {}
        AssetManager.require_asset(cls.model, cls.assets)
        return True


class Worker(Unit):
    model = 'worker'
    collision = Collision.GROUND | Collision.UNIT



def generate_land(hor_sz, ver_sz, land: list, entities: list):
    for _ in range(int((hor_sz * ver_sz)**0.7)):
        x = rnd.randint(0, hor_sz - 1)
        y = rnd.randint(0, ver_sz - 1)
        tile = rnd.choice((Rock, Tree))(x, y)
        land[x][y] = tile
        spam('add tile', x, y, 'color', tile.color)

    for x in range(hor_sz):
        for y in range(ver_sz):
            if land[x][y] == 0:
                land[x][y] = Grassland(x, y)
                spam('add tile', x, y)

    worker = None
    while not worker:
        x = rnd.randint(0, hor_sz - 1)
        y = rnd.randint(0, ver_sz - 1)
        if land[x][y].collision & Worker.collision == 0:
            worker = Worker(x, y)
            entities[1].append(worker)


class Game:
    def __init__(self, hor_sz, ver_sz):
        self.hor_sz, self.ver_sz = hor_sz, ver_sz

        self.entities: List[List[Entity]] = [[], [], []]
        self.land = [[0] * ver_sz for _ in range(hor_sz)]

        generate_land(hor_sz, ver_sz, self.land, self.entities)

        FieldT = List[List[List[Entity]]]
        self.field: FieldT = [[[t] for t in tiles] for tiles in self.land]
        for l in self.entities:
            for e in l:
                self.field[e.x][e.y].append(e)

        for y in range(ver_sz):
            for x in range(hor_sz):
                debug(self.land[x][y].color_name[0], end='')
            debug()

        for y in range(ver_sz):
            for x in range(hor_sz):
                t = self.land[x][y]
                spam(f'({t.x},{t.y}) ', end='')
            spam()

    def __getitem__(self, key):
        x, y = key
        return self.field[x][y][-1]

    def get_path(self, ax, ay, bx, by, affinity) -> Deque[Tuple[int, int]]:
        import numpy as np
        from operator import or_

        h, v = self.hor_sz, self.ver_sz
        m = np.empty((h, v), np.uint32)
        for x in range(h):
            for y in range(v):
                es = self.field[x][y]
                m[x][y] = reduce(or_, (e.collision for e in es)) & affinity

        dst = np.zeros_like(m)
        d = deque()
        d.append((ax, ay))


        while d:
            x, y = d.popleft()
            for sx, sy in (x-1, y), (x, y-1), (x+1, y), (x, y+1):
                if 0 <= sx < h and 0 <= sy < v and not m[sx][sy]:
                    m[sx][sy] = 1
                    dst[sx][sy] = dst[x][y] + 1
                    if (sx, sy) == (bx, by):
                        d.clear()
                        break
                    d.append((sx, sy))

        path = deque()
        x, y = bx, by
        while dst[x][y]:
            path.append((x, y))
            for sx, sy in (x-1, y), (x, y-1), (x+1, y), (x, y+1):
                if dst[sx][sy] + 1 == dst[x][y]:
                    x, y = sx, sy
                    break
        return path


class Action:
    _ACTIONS = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Action._ACTIONS[cls.Model] = cls

    @staticmethod
    def get(cls):
        return Action._ACTIONS.get(cls)

    def __init__(self, game: Game, e: Entity, x, y):
        self.target = (x, y)
        self.game: Game = game
        self.entity: Entity = e

    def tick(self, elapsed_ms):
        """
        The only thing Action does is make a thing on every
        GameController's tick
        """
        pass


class UnitAction(Action):
    Model = Unit

    class State:
        IDLE = 0
        MOVING = 1
        TRANSITION = 2

    v = 1  # tile per sec

    def __init__(self, game: Game, e: Entity, x, y):
        super().__init__(game, e, x, y)
        self.path = game.get_path(e.x, e.y, x, y, e.collision)
        self.entity.start = self.entity.x, self.entity.y
        self.entity.passed = 1


    def tick(self, elapsed_ms):
        e = self.entity
        if e.passed >= 1:
            if self.path:
                sx, sy = e.x, e.y
                tx, ty = self.path.pop()
                e.passed -= 1
                e.x, e.y = tx, ty
                e.start = sx, sy
            else:
                e.start = e.x, e.y
        e.passed += self.v * elapsed_ms / 1000


class WorkerAction(UnitAction):
    Model = Worker


class GameController(Viewport):
    def __init__(self, w, game: Game):
        super().__init__(w, game.hor_sz, game.ver_sz)
        self.game = game
        self.prev_hover = game[0, 0]
        self.selected = None
        AssetManager.load(w, w)

    def draw(self, surf: pg.Surface):

        # Tiles
        for line in self.game.land:
            for e in line:
                e.draw(surf, self)

        # Other entities
        for l in self.game.entities:
            for e in l:
                e.draw(surf, self)

    def check_borders(self, cx, cy):
        for x, w in (cx, self.hor_sz), (cy, self.ver_sz):
            if x < 0 or x >= w:
                return False
        return True

    def hover(self, mx, my):
        self.prev_hover.hover = False

        cx, cy = self.get_cell(mx, my)
        if not self.check_borders(cx, cy):
            spam(f'hover {cx}, {cy} OOR')
            return

        spam(f'hover {cx}, {cy}')
        self.prev_hover = self.game[cx, cy]
        self.prev_hover.hover = True

    def select(self, mx, my):
        cx, cy = self.get_cell(mx, my)

        if not self.check_borders(cx, cy):
            debug(f'select {cx}, {cy} OOR')
            return

        obj = self.game[cx, cy]

        if isinstance(obj, Unit):
            debug(f'select {cx}, {cy} unit', obj.__class__.__name__)
            self.selected = obj

    def command(self, mx, my):
        if not self.selected:
            return
        cx, cy = self.get_cell(mx, my)

        if not self.check_borders(cx, cy):
            debug(f'command {cx}, {cy} OOR')
            return
        debug(f'command {cx}, {cy}')

        obj = self.game[cx, cy]
        A = Action.get(self.selected.__class__)
        self.selected.action = A(self.game, self.selected, cx, cy)

    def tick(self, elapsed_ms):
        for layer in self.game.entities:
            for e in layer:
                if e.action:
                    e.action.tick(elapsed_ms)



os.environ['SDL_VIDEO_CENTERED'] = "1"
pg.init()
running = True

game = Game(30, 20)
controller = GameController(35, game)
screen = pg.display.set_mode((1000, 600))
button_down = 0
while running:
    screen.fill((255,255,255))
    for ev in pg.event.get():
        if ev.type == pg.QUIT:
            running = False
        elif ev.type == pg.MOUSEBUTTONDOWN:
            if ev.button == 1:
                controller.hover(*ev.pos)
            button_down = 1
        elif ev.type == pg.MOUSEMOTION:
            if button_down == 1:
                controller.hover(*ev.pos)
        elif ev.type == pg.MOUSEBUTTONUP:
            if ev.button == 1:
                controller.select(*ev.pos)
            elif ev.button == 3:
                controller.command(*ev.pos)
        elif ev.type == pg.KEYDOWN:
            if ev.key == pg.K_ESCAPE:
                running = False

    controller.tick(7)

    controller.draw(screen)
    pg.display.flip()

pg.quit()
