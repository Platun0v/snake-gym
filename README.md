# gym-snake

## Description
gym-snake is implementation of the classic game snake that is made as an OpenAI gym environment

## Dependencies
+ gym
+ numpy
+ torch

## Installation
1. Clone repository: `$ git clone https://github.com/Platun0v/snake-gym.git`
2. `cd` into snake-gym and run: `pip install -r requirements.txt`

## Using ready-made programs

### Test network
To see pretrained snake run: `python test.py`

#### Parameters
+ _load_path_ - path to neural network
+ _render_ - render process
+ _times_ - how many times to run
+ _seed_ - seed
+ _blocks_ - number of blocks on square grid
+ _block_size_ - the size of block in pixels
 
### Train network
To train snake run: `python train.py`

#### Parameters
+ _save_path_ - path to save neural network
+ _render_ - render process
+ _episodes_ - how many times to run
+ _seed_ - seed
+ _blocks_ - number of blocks on square grid
+ _block_size_ - the size of block in pixels

## Using enviroment
```python
import gym
import gym_snake

env = gym.make('Snake-v0')

for i in range(100):
    env.reset()
    for t in range(1000):
        env.render()
        state, reward, done, info = env.step(env.action_space.sample())
        if done:
            print('episode {} finished after {} timesteps'.format(i, t))
            break
```