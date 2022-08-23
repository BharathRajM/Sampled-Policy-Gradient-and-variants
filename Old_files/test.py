import numpy as np
import torch
import gym
import argparse
import os
import utils

import OurDDPG
import SPG

print("all imports are perfect")

env = gym.make("HalfCheetah-v4")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device")

action = env.action_space.sample()
next_state,reward,done,_ = env.step(action)

print("done")