from __future__ import print_function

import os
import gymnasium as gym
import json
from datetime import datetime
from agent.dqn_agent import DQNAgent
from train_carracing import run_episode
from agent.networks import *
import numpy as np

np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("CarRacing-v3", render_mode="human")

    history_length = 0

    # TODO: Define networks and load agent
    # ....

    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(
            env, agent, deterministic=True, do_training=False, rendering=True
        )
        episode_rewards.append(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    if not os.path.exists("./results"):
        os.mkdir("./results")

    fname = f"./results/carracing_results_dqn-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(results, f)

    env.close()
    print("... finished")
