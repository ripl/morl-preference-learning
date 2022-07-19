from os import environ
import gym
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import DQN

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
sys.path.insert(0, './packages/MO-highway-env/')
import highway_env
# sys.path.insert(0, './packages/MO-highway-env/scripts/')
# from utils import record_videos

from gym import wrappers

NUM_TRAJECTORIES = 10

if __name__ == "__main__":
    model = DQN.load("./packages/PbMORL/models/speed_model")

    env = gym.make("mo-highway-v0")
    env.seed(0)
    env.configure({
        "duration": np.Infinity,
        "cur_reward": 0
    })
    # env = record_videos(env, "./packages/PbMORL/videos")
    env = wrappers.Monitor(env, "./packages/PbMORL/videos", force=True)

    for tau in range(NUM_TRAJECTORIES):
        obs, done = env.reset(), False
        retrn = 0
        while not done:
            env.render()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            retrn += reward
        env.render()
        print(retrn)

    env.close()