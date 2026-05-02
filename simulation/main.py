import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

from agent.rl_agent import
from utils.logger import


def main():
    print("Starting...")

    env = UnityEnvironment(file_name=None)
    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]
    #bridge - name of the "brain" to send and receive data

    num_beams = 7    #may be changed
    agent = the_agent(input_size=num_beams)
    logger = TensorBoardLogger()



    #THE loop
    episodes = 10000
    for episode in range(episodes):
        env.reset()
        episode_reward=0
        done=False






