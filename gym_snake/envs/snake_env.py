import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

from gym_snake.envs.snake import Snake

import logging

logger = logging.getLogger(__name__)


class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, blocks=10, block_size=50):
        self.blocks = blocks
        self.width = block_size * blocks
        self.snake = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            dtype=np.float32,
            low=np.array([0, 0, 0, -1, -1]),
            high=np.array([1, 1, 1, 1, 1]),
        )

        self.seed()
        self.viewer = None
        self.rewards = None

    def set_rewards(self, rew_step, rew_apple, rew_death, rew_death2, rew_apple_func):
        self.rewards = [rew_step, rew_apple, rew_death, rew_death2, rew_apple_func]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if action != 0:
            self.snake.direction = self.snake.DIRECTIONS[self.snake.direction[action]]

        info = {}

        self.snake.update()
        info['apple_ate'] = self.snake.apple_ate

        raw_state, reward, done = self.snake.get_raw_state()
        info['apples'] = self.snake.cnt_apples

        state = np.array(raw_state, dtype=np.float32)
        state /= self.blocks

        return state, reward, done, info

    def reset(self):
        if self.rewards:
            self.snake = Snake(self.blocks, self.width // self.blocks, self.np_random,
                               rew_step=self.rewards[0], rew_apple=self.rewards[1],
                               rew_death=self.rewards[2], rew_death2=self.rewards[3],
                               rew_apple_func=self.rewards[4],)
        else:
            self.snake = Snake(self.blocks, self.width // self.blocks, self.np_random)
        raw_state = self.snake.get_raw_state()

        state = np.array(raw_state[0], dtype=np.float32)
        state /= self.blocks

        return state

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        w = self.snake.blockw

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.width, self.width)
            apple = self._create_block(w)
            self.apple_trans = rendering.Transform()
            apple.add_attr(self.apple_trans)
            apple.set_color(*self.snake.apple.color)
            self.viewer.add_geom(apple)

            head = self._create_block(w)
            self.head_trans = rendering.Transform()
            head.add_attr(self.head_trans)
            head.set_color(*self.snake.head.color)
            self.viewer.add_geom(head)

            self.body = []
            for i in range(len(self.snake.body)):
                body = self._create_block(w)
                body_trans = rendering.Transform()
                body.add_attr(body_trans)
                body.set_color(*self.snake.body[0].color)

                self.body.append(body_trans)
                self.viewer.add_geom(body)

        self.apple_trans.set_translation(self.snake.apple.x, self.snake.apple.y)
        self.head_trans.set_translation(self.snake.head.x, self.snake.head.y)

        if len(self.snake.body) > len(self.body):
            body = self._create_block(w)
            body_trans = rendering.Transform()
            body.add_attr(body_trans)
            body.set_color(*self.snake.body[0].color)

            self.body.append(body_trans)
            self.viewer.add_geom(body)
        elif len(self.snake.body) < len(self.body):
            self.body, trash = self.body[len(self.body) - len(self.snake.body):], \
                               self.body[:len(self.body) - len(self.snake.body)]
            for i in range(len(trash)):
                trash[i].set_translation(-w, -w)

        for i in range(len(self.body)):
            self.body[i].set_translation(self.snake.body[i].x, self.snake.body[i].y)

        self.viewer.render()

    def _create_block(self, w):
        from gym.envs.classic_control import rendering
        return rendering.FilledPolygon([(0, 0), (0, w), (w, w), (w, 0)])

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
