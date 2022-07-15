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

TRAJ_TO_COMPARE = 10

if __name__ == "__main__":
    model = DQN.load("./packages/PbMORL/models/speed_model")

    env = gym.make("highway-v0")
    env.configure({
        "duration": np.Infinity
    })
    # env = record_videos(env)

    for tau in range(TRAJ_TO_COMPARE):
        obs, done = env.reset(), False
        retrn = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            retrn += reward
        print(retrn)

    env.close()