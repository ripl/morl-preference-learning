import gym
from gym.wrappers import RecordVideo
import numpy as np
from stable_baselines3 import DQN

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
sys.path.insert(0, './packages/MO-highway-env/')
import highway_env

NUM_TRAJECTORIES = 10

if __name__ == "__main__":
    model = DQN.load("./packages/PbMORL/models/speed_model")

    env = gym.make("mo-highway-v0")
    env.seed(0)
    env.configure({
        "screen_width": 1400,
        "screen_height": 200,
        "centering_position": [0.5, 0.5],
        "scaling": 5.5,
        "duration": np.Infinity,
        "cur_reward": 0,
    })
    
    # Video recording
    env = RecordVideo(env, video_folder="./packages/PbMORL/videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)

    # Run trajectories
    for tau in range(NUM_TRAJECTORIES):
        obs, done = env.reset(), False
        total_reward = 0
        while not done:
            env.render()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        env.render()
        print(total_reward)
    env.close()