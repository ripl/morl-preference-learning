import gym
import numpy as np
from stable_baselines3 import DQN

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
sys.path.insert(0, './packages/MO-highway-env/')
import highway_env


def train_policy(env, filename):
    model = DQN('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=32,
                gamma=0.9,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                exploration_fraction=0.7,
                verbose=1,
                tensorboard_log="./packages/PbMORL/highway_dqn/")
    model.learn(2e4)
    model.save("./packages/PbMORL/models/" + filename)
    return model

if __name__ == "__main__":
    env = gym.make("highway-v0")
    env.configure({
        "duration": 40,
        "reward_num": 2, # 0: speed, 1: right, 2: crash
        "reward_speed_range": [20, 30]
    })
    train_policy(env, "safe_model")