import gym
import numpy as np
import sys

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.insert(0, './packages/MO-highway-env/')
import highway_env

if __name__ == "__main__":
    env = gym.make("mo-highway-v0")
    env.seed(0)
    env.configure({
        "screen_width": 1400,
        "screen_height": 200,
        "centering_position": [0.5, 0.5],
        "scaling": 5.5,
        "duration": np.Infinity,
        "action": {
            "type": "ContinuousAction"
        }
    })
    NUM_TRAJECTORIES = 10
    for tau in range(NUM_TRAJECTORIES):
        obs, done = env.reset(), False
        total_reward = 0
        while not done:
            env.render()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward
        env.render()
        print('Return from trajectory', tau+1, ':', total_reward)