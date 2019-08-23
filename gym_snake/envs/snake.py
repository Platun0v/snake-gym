from math import sqrt


class Block:
    def __init__(self, x, y, w, color):
        self.x = x
        self.y = y
        self.w = w
        self.color = color


class Snake:
    DIRECTIONS = {
        'UP': ((0, 1), 'LEFT', 'RIGHT'),
        'DOWN': ((0, -1), 'RIGHT', 'LEFT'),
        'LEFT': ((-1, 0), 'DOWN', 'UP'),
        'RIGHT': ((1, 0), 'UP', 'DOWN'),
    }

    def __init__(self, blocks, block_len, random,
                 rew_step=-0.25, rew_apple=3.5, rew_death=-10.0, rew_death2=-100.0,
                 rew_apple_func=lambda cnt, rew: sqrt(cnt) * rew):
        self.blockw = block_len
        self.blocks = blocks
        self.random = random

        x, y = self.blocks // 2, self.blocks // 2
        self.direction = self.DIRECTIONS['UP']
        self.game_over = False

        self.rew_step = rew_step
        self.rew_apple = rew_apple
        self.rew_death = rew_death
        self.rew_death2 = rew_death2
        self.rew_apple_func = rew_apple_func

        self.body = []
        for i in range(4):
            self.body.append(
                Block(x * block_len,
                      (y - i - 1) * block_len,
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
            if self.head.x == x * self.blockw and self.head.y == y * self.blockw:
                continue

            flag = False
            for e in self.body:
                if e.x == x * self.blockw and e.y == y * self.blockw:
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
        if self.head.x < 0 or \
                self.head.x > (self.blocks - 1) * self.blockw or \
                self.head.y < 0 or \
                self.head.y > (self.blocks - 1) * self.blockw:
            self.game_over = True
            return

        self.head.color = (0, 255, 0)
        self.body = [self.head] + self.body[:]
        self.head = Block(self.head.x + self.direction[0][0] * self.blockw,
                          self.head.y + self.direction[0][1] * self.blockw,
                          self.blockw,
                          (0, 255, 255))

        if self.apple is None:
            self.generate_apple()

        if self.apple.x == self.head.x and self.apple.y == self.head.y:
            self.generate_apple()
            self.apple_ate = True
        else:
            self.body = self.body[:-1]

        for e in self.body:
            if e.x == self.head.x and e.y == self.head.y:
                self.game_over = True

    def get_raw_state(self):
        reward = self.rew_step
        self.cnt_steps += 1
        if self.apple_ate:
            self.cnt_apples += 1
            self.apple_ate = False
            reward = self.rew_apple_func(self.cnt_apples, self.rew_apple)
        elif self.game_over:
            if self.cnt_steps < 15:
                reward = self.rew_death2
            else:
                reward = self.rew_death

        state = [
            self.head.x // self.blockw,  # from head to left side
            self.blocks - (self.head.x // self.blockw) - 1,  # from head to right side
            self.head.y // self.blockw,  # from head to down side
            self.blocks - (self.head.y // self.blockw) - 1,  # from head to up side
        ]

        for e in self.body:
            if e.x == self.head.x:
                if e.y < self.head.y:
                    state[2] = min(state[2], (self.head.y - e.y) // self.blockw - 1)
                else:
                    state[3] = min(state[3], (- self.head.y + e.y) // self.blockw - 1)
            elif e.y == self.head.y:
                if e.x < self.head.x:
                    state[0] = min(state[0], (self.head.x - e.x) // self.blockw - 1)
                else:
                    state[1] = min(state[1], (- self.head.x + e.x) // self.blockw - 1)

        apple_crd = [
            (-self.head.x + self.apple.x) // self.blockw,
            (-self.head.y + self.apple.y) // self.blockw,
        ]

        if self.direction == self.DIRECTIONS['UP']:
            state = [state[3], state[0], state[1]]
        if self.direction == self.DIRECTIONS['LEFT']:
            state = [state[0], state[2], state[3]]
            if apple_crd[0] * apple_crd[1] > 0:
                apple_crd[1] *= -1
            else:
                apple_crd[0] *= -1
            apple_crd[0], apple_crd[1] = apple_crd[1], apple_crd[0]
        if self.direction == self.DIRECTIONS['DOWN']:
            state = [state[2], state[1], state[0]]
            apple_crd[0] *= -1
            apple_crd[1] *= -1
        if self.direction == self.DIRECTIONS['RIGHT']:
            state = [state[1], state[3], state[2]]
            if apple_crd[0] * apple_crd[1] > 0:
                apple_crd[0] *= -1
            else:
                apple_crd[1] *= -1
            apple_crd[0], apple_crd[1] = apple_crd[1], apple_crd[0]

        state.extend(apple_crd)

        return state, reward, self.game_over
