from collections import Counter
import getopt
import gym
from gym.wrappers import RecordVideo
import numpy as np
from stable_baselines3 import DQN
import sys

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.insert(0, './packages/MO-highway-env/')
import highway_env

sys.path.insert(0, './packages/')
from PbMORL.utils import clear_videos
from PbMORL.training import train_policy

VIDEO_PATH = './packages/PbMORL/videos/'
MODEL_PATH = './packages/PbMORL/models/'
LOG_PATH = './packages/PbMORL/highway_dqn/'
POLICY = 'speed'

def main(argv):
    try:
        opts, _ = getopt.getopt(argv,"tv",["train=", "video="])
    except:
        print('Invalid usage!')
        print('Use -t or --train to train policy')
        print('Use -v or --video to view and record sampled trajectories')
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

    for opt, _ in opts:
        if opt in ('-t', '--train'):
            print('Running in training mode...')
            print('Training policy:', POLICY)
            env.configure({
                "render_agent": False,
                "duration": 40
            })
            env.reset()
            train_policy(env, MODEL_PATH + POLICY, LOG_PATH)
        if opt in ('-v', '--video'):
            clear_videos(VIDEO_PATH)
            env = RecordVideo(env, video_folder=VIDEO_PATH, episode_trigger=lambda e: True)
            env.unwrapped.set_record_video_wrapper(env)
            model = DQN.load(MODEL_PATH + POLICY)
            # Sample trajectories
            NUM_TRAJECTORIES = 10
            print('Sampling', NUM_TRAJECTORIES, 'trajectories from policy:', POLICY)
            for tau in range(NUM_TRAJECTORIES):
                obs, done = env.reset(), False
                total_reward = Counter()
                while not done:
                    env.render()
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)
                    total_reward += Counter(info)
                env.render()
                print('Return from trajectory', tau+1, ':', total_reward)
            env.close()

if __name__ == "__main__":
    main(sys.argv[1:])