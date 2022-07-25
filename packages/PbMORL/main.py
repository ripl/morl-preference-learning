import getopt
import gym
from gym.wrappers import RecordVideo
import numpy as np
import os
from stable_baselines3 import DQN
import sys

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.insert(0, './packages/MO-highway-env/')
import highway_env

sys.path.insert(0, './packages/')
from PbMORL.utils import clear_videos
from PbMORL.training import train_policy

VIDEO_PATH = './packages/PbMORL/videos'
MODEL_PATH = './packages/PbMORL/models'
LOG_PATH = './packages/PbMORL/highway_dqn/'

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"tr:p:v",["train", "reward=", "policy=", "video"])
    except:
        print('Invalid usage!')
        print('Use -t or --train to train policy on selected reward (0 by default)')
        print('Use -r <reward num> or --reward <reward num> to select a reward')
        print('Use -p <policy num> or --policy <policy num> to select a policy')
        print('Use -v or --video to enable video recording')
        sys.exit(2)

    env = gym.make("mo-highway-v0")
    env.seed(0)
    env.configure({
        "screen_width": 1400,
        "screen_height": 200,
        "centering_position": [0.5, 0.5],
        "scaling": 5.5,
        "duration": np.Infinity,
    })
    policy_num = 0

    for opt, arg in opts:
        if opt in ('-r', '--reward'):
            env.configure({"cur_reward": int(arg)})
            env.reset()
        if opt in ('-p', '--policy'):
            policy_num = arg
        if opt in ('-t', '--train'):
            print('Running in training mode...')
            print('Training policy ' + str(env.config["cur_reward"]) + '...')
            env.configure({"duration": 40})
            env.reset()
            train_policy(env, MODEL_PATH + "/policy_" + str(env.config["cur_reward"]), LOG_PATH)
        if opt in ('-v', '--video'):
            clear_videos(VIDEO_PATH)
            env = RecordVideo(env, video_folder=VIDEO_PATH, episode_trigger=lambda e: True)
            env.unwrapped.set_record_video_wrapper(env)
            
            model = DQN.load(MODEL_PATH + "/policy_" + str(policy_num))

            # Sample trajectories
            NUM_TRAJECTORIES = 10
            print('Sampling', NUM_TRAJECTORIES, 'trajectories from policy', policy_num)
            for tau in range(NUM_TRAJECTORIES):
                obs, done = env.reset(), False
                total_reward = 0
                while not done:
                    env.render()
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)
                    total_reward += reward
                env.render()
                print('Return from trajectory', tau+1, ':', total_reward)
            env.close()

            # Print Number of Videos Saved
            # _, _, files = next(os.walk("./packages/PbMORL/videos"))
            # print(len(files)/2)

if __name__ == "__main__":
    main(sys.argv[1:])