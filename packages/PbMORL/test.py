import gym
import numpy as np
from pprint import pprint
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
            "type": "DiscreteMetaAction"
        },
        "cur_reward": "speed"
    })
    NUM_TRAJECTORIES = 10
    for tau in range(NUM_TRAJECTORIES):
        obs, done = env.reset(), False
        while not done:
            env.render()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            for r in info:
                info[r] = round(info[r],3)
            print("Selected reward:", reward)
            pprint(info)
        env.render()