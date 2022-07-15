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

if __name__ == "__main__":
    print("Hello from Python")