from collections import namedtuple
import numpy as np


class Block:
    def __init__(self, x, y, w, color):
        rect = namedtuple('rect', ('x', 'y', 'w'))
        self.bounds = rect(x=x, y=y, w=w)
        self.color = color


class Snake:
    DIRECTIONS = {
        'UP': ((0, -1), 'LEFT', 'RIGHT'),
        'DOWN': ((0, 1), 'RIGHT', 'LEFT'),
        'LEFT': ((-1, 0), 'DOWN', 'UP'),
        'RIGHT': ((1, 0), 'UP', 'DOWN'),
    }

    def __init__(self, x, y, blocks, block_len, random):
        self.blockw = block_len
        self.blocks = blocks
        self.direction = self.DIRECTIONS['UP']
        self.game_over = False
        self.random = random

        self.body = []
        for i in range(4):
            self.body.append(
                Block(x * block_len,
                      (y + i + 1) * block_len,
                      block_len,
                      (0, 255, 0)))
        self.head = Block(x * block_len,
                          y * block_len,
                          block_len,
                          (0, 255, 255),
                          )
        self.apple = None
        self.generate_apple()
        self.apple_ate = False
        self.cnt_apples = 0
        self.cnt_steps = 0

    def generate_apple(self):
        while True:
            x, y = self.random.randint(0, self.blocks - 1), self.random.randint(0, self.blocks - 1)
            if self.head.bounds.x == x * self.blockw and self.head.bounds.y == y * self.blockw:
                continue

            flag = False
            for e in self.body:
                if e.bounds.x == x * self.blockw and e.bounds.y == y * self.blockw:
                    flag = True
                    continue
            if flag:
                continue

            self.apple = Block(
                x * self.blockw,
                y * self.blockw,
                self.blockw,
                (255, 0, 0)
            )
            break

    def update(self):
        for e in self.body:
            if e.bounds.x == self.head.bounds.x and e.bounds.y == self.head.bounds.y:
                self.game_over = True
                return

        if self.head.bounds.x < 0 or \
                self.head.bounds.x > (self.blocks - 1) * self.blockw or \
                self.head.bounds.y < 0 or \
                self.head.bounds.y > (self.blocks - 1) * self.blockw:
            self.game_over = True
            return

        self.head.color = (0, 255, 0)
        self.body = [self.head] + self.body[:]
        self.head = Block(self.head.bounds.x + self.direction[0][0] * self.blockw,
                          self.head.bounds.y + self.direction[0][1] * self.blockw,
                          self.blockw,
                          (0, 255, 255))

        if self.apple is None:
            self.generate_apple()

        if self.apple.bounds.x == self.head.bounds.x and self.apple.bounds.y == self.head.bounds.y:
            self.generate_apple()
            self.apple_ate = True
        else:
            self.body = self.body[:-1]

    def get_raw_state(self):
        reward = -0.25
        self.cnt_steps += 1
        if self.apple_ate:
            self.cnt_apples += 1
            self.apple_ate = False
            self.cnt_steps = 0
            reward = 3 * self.cnt_apples
        elif self.game_over:
            if self.cnt_steps < 15:
                reward = -100
            else:
                reward = -10

        state = [
            self.head.bounds.x // self.blockw,  # from head to left side
            self.blocks - (self.head.bounds.x // self.blockw) - 1,  # from head to right side
            self.head.bounds.y // self.blockw,  # from head to up side
            self.blocks - (self.head.bounds.y // self.blockw) - 1,  # from head to down side
        ]

        for e in self.body:
            if e.bounds.x == self.head.bounds.x:
                if e.bounds.y < self.head.bounds.y:
                    state[2] = min(state[2], (self.head.bounds.y - e.bounds.y) // self.blockw - 1)
                else:
                    state[3] = min(state[3], (- self.head.bounds.y + e.bounds.y) // self.blockw - 1)
            elif e.bounds.y == self.head.bounds.y:
                if e.bounds.x < self.head.bounds.x:
                    state[0] = min(state[0], (self.head.bounds.x - e.bounds.x) // self.blockw - 1)
                else:
                    state[1] = min(state[1], (- self.head.bounds.x + e.bounds.x) // self.blockw - 1)

        apple_crd = [
            (-self.head.bounds.x + self.apple.bounds.x) // self.blockw,
            (self.head.bounds.y - self.apple.bounds.y) // self.blockw,
        ]

        if self.direction == self.DIRECTIONS['UP']:
            state = [state[2], state[0], state[1]]
        if self.direction == self.DIRECTIONS['LEFT']:
            state = [state[0], state[3], state[2]]
            if apple_crd[0] * apple_crd[1] > 0:
                apple_crd[1] *= -1
            else:
                apple_crd[0] *= -1
        if self.direction == self.DIRECTIONS['DOWN']:
            state = [state[3], state[1], state[0]]
            apple_crd[0] *= -1
            apple_crd[1] *= -1
        if self.direction == self.DIRECTIONS['RIGHT']:
            state = [state[1], state[2], state[3]]
            if apple_crd[0] * apple_crd[1] > 0:
                apple_crd[0] *= -1
            else:
                apple_crd[1] *= -1

        state.extend(apple_crd)

        return state, reward
