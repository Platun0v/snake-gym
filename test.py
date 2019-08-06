import argparse
from time import sleep
import gym
import gym_snake
from rl.model import DQN
import torch
import numpy as np


class Agent:
    def __init__(self, pth_path, seed):
        self.model = DQN(seed)
        self.model.load_state_dict(torch.load(pth_path))

    def act(self, state):
        state = torch.from_numpy(state).float()
        self.model.eval()
        with torch.no_grad():
            action_values = self.model(state)

        return np.argmax(action_values.data.numpy())


def main(pth_path, render, times, seed):
    env = get_env()
    agent = Agent(pth_path, seed)
    watch_agent(agent, env, times, render)


def get_env():
    return gym.make('Snake-v0')


def watch_agent(agent, env, times, render):
    for i in range(times):
        state = env.reset()
        while True:
            if render:
                env.render()
                sleep(0.05)
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            if done:
                break
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test trained agent')
    parser.add_argument('--pth_path', default='rl/default_model.pth', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--times', default=3, type=int)
    parser.add_argument('--seed', default=0, type=int)

    args = parser.parse_args()
    main(
        pth_path=args.pth_path,
        render=args.render,
        times=args.times,
        seed=args.seed,
    )
