import random

import numpy as np
import pygame

BLOCK = 20
COLS, ROWS = 32, 24
WIDTH, HEIGHT = COLS * BLOCK, ROWS * BLOCK  # 640 x 480

# Clockwise on screen (y grows downward): 0=RIGHT, 1=DOWN, 2=LEFT, 3=UP
DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]

BOARD_BG = (11, 15, 21)
GRID = (20, 27, 36)
BORDER = (30, 38, 50)
TEAL = (45, 212, 191)
TEAL_TAIL = (19, 110, 104)
HEAD = (214, 255, 249)
FOOD = (239, 68, 68)
FOOD_GLOW = (120, 38, 48)


def _lerp(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


class SnakeGame:
    """
    Snake with an RL-style interface (no window of its own, the Dashboard draws it).

    Actions (int): 0 = straight, 1 = turn left, 2 = turn right
    State (11 values):
      [danger_straight, danger_right, danger_left,
       dir_up, dir_down, dir_left, dir_right,
       food_up, food_down, food_left, food_right]
    Rewards: +10 food, -10 crash, 0 otherwise.
    """

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.dir_idx = 0  # start moving RIGHT
        head = (COLS // 2, ROWS // 2)
        self.snake = [head, (head[0] - 1, head[1]), (head[0] - 2, head[1])]
        self.score = 0
        self.frames = 0
        self.food = None
        self._place_food()

    def _place_food(self):
        occupied = set(self.snake)
        free = [(x, y) for x in range(COLS) for y in range(ROWS) if (x, y) not in occupied]
        if free:
            self.food = self.rng.choice(free)
        else:
            self.food = self.snake[0]  # board completely full

    def _is_danger(self, pt):
        x, y = pt
        if x < 0 or x >= COLS or y < 0 or y >= ROWS:
            return True
        return pt in self.snake

    def get_state(self):
        head = self.snake[0]
        d = self.dir_idx

        straight = DIRECTIONS[d]
        right = DIRECTIONS[(d + 1) % 4]
        left = DIRECTIONS[(d - 1) % 4]

        def ahead(v):
            return (head[0] + v[0], head[1] + v[1])

        state = [
            self._is_danger(ahead(straight)),
            self._is_danger(ahead(right)),
            self._is_danger(ahead(left)),
            d == 3,  # up
            d == 1,  # down
            d == 2,  # left
            d == 0,  # right
            self.food[1] < head[1],  # food up
            self.food[1] > head[1],  # food down
            self.food[0] < head[0],  # food left
            self.food[0] > head[0],  # food right
        ]
        return np.array(state, dtype=np.float32)

    def step(self, action):
        """Returns: reward, done, score"""
        self.frames += 1

        if action == 1:      # turn left
            self.dir_idx = (self.dir_idx - 1) % 4
        elif action == 2:    # turn right
            self.dir_idx = (self.dir_idx + 1) % 4

        dx, dy = DIRECTIONS[self.dir_idx]
        new_head = (self.snake[0][0] + dx, self.snake[0][1] + dy)
        self.snake.insert(0, new_head)

        out_of_bounds = (
            new_head[0] < 0 or new_head[0] >= COLS or
            new_head[1] < 0 or new_head[1] >= ROWS
        )
        hit_self = new_head in self.snake[1:]
        timeout = self.frames > 100 * len(self.snake)  # stops endless looping

        if out_of_bounds or hit_self or timeout:
            self.snake.pop(0)  # keep the last valid snake for drawing
            return -10, True, self.score

        if new_head == self.food:
            self.score += 1
            self.frames = 0
            if len(self.snake) >= COLS * ROWS:
                self.food = self.snake[0]
                return 10, True, self.score
            self._place_food()
            return 10, False, self.score

        self.snake.pop()
        return 0, False, self.score

    def draw(self, surface, ox, oy):
        pygame.draw.rect(surface, BOARD_BG, (ox, oy, WIDTH, HEIGHT))
        for x in range(0, WIDTH + 1, BLOCK):
            pygame.draw.line(surface, GRID, (ox + x, oy), (ox + x, oy + HEIGHT))
        for y in range(0, HEIGHT + 1, BLOCK):
            pygame.draw.line(surface, GRID, (ox, oy + y), (ox + WIDTH, oy + y))

        n = len(self.snake)
        for i, (x, y) in enumerate(self.snake):
            rect = pygame.Rect(ox + x * BLOCK + 1, oy + y * BLOCK + 1, BLOCK - 2, BLOCK - 2)
            if i == 0:
                color = HEAD
            else:
                color = _lerp(TEAL, TEAL_TAIL, (i / max(1, n - 1)) * 0.7)
            pygame.draw.rect(surface, color, rect, border_radius=3)

        fx, fy = self.food
        center = (ox + fx * BLOCK + BLOCK // 2, oy + fy * BLOCK + BLOCK // 2)
        pygame.draw.circle(surface, FOOD_GLOW, center, BLOCK // 2 + 1)
        pygame.draw.circle(surface, FOOD, center, BLOCK // 2 - 3)

        pygame.draw.rect(surface, BORDER, (ox, oy, WIDTH, HEIGHT), width=1)