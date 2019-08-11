import argparse
from time import sleep
import gym
import gym_snake
import torch
import numpy as np
from rl.agent import Agent
from collections import deque


def main(save_path, render, seed, block_size, blocks, episodes, max_t, eps_start, eps_end, eps_decay):
    env = get_env(seed, block_size, blocks)
    agent = Agent(env.observation_space.shape[0], env.action_space.n, seed)
    agent = train_dqn(agent, env, episodes, max_t, eps_start, eps_end, eps_decay, render)
    torch.save(agent.qnetwork_local.state_dict(), save_path)


def get_env(seed, block_size, blocks):
    env = gym.make('Snake-v0', block_size=block_size, blocks=blocks)
    env.seed(seed)
    return env


def train_dqn(agent, env, episodes, max_t, eps_start, eps_end, eps_decay, render):
    scores = []
    apples = []
    scores_window = deque(maxlen=100)
    apples_window = deque(maxlen=100)
    eps = eps_start
    for i in range(1, episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, info = env.step(action)
            if render:
                env.render()
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)        # save most recent score
        apples_window.append(info['apples'])
        scores.append(score)               # save most recent score
        apples.append(info['apples'])
        eps = max(eps_end, eps_decay*eps)  # decrease epsilon
        print(f'\rEpisode {i}\t'
              f'Average apples: {np.mean(apples):.2f}\t'
              f'Average score: {np.mean(scores):.2f}', end='')
        if i % 100 == 0:
            print(f'\rEpisode {i}\t'
                  f'Average apples: {np.mean(apples):.2f}\t'
                  f'Average score: {np.mean(scores):.2f}')
    return agent


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train agent')
    parser.add_argument('--save_path', default='checkpoint.pth', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--blocks', default=10, type=int)
    parser.add_argument('--block_size', default=50, type=int)
    parser.add_argument('--episodes', default=2000, type=int)
    parser.add_argument('--max_t', default=1500, type=int)
    parser.add_argument('--eps_start', default=1.0, type=float)
    parser.add_argument('--eps_end', default=0.01, type=float)
    parser.add_argument('--eps_decay', default=0.995, type=float)

    args = parser.parse_args()
    main(
        save_path=args.save_path,
        render=args.render,
        seed=args.seed,
        block_size=args.block_size,
        blocks=args.blocks,
        episodes=args.episodes,
        max_t=args.max_t,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
    )
