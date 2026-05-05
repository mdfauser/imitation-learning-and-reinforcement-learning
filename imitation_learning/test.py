import sys

sys.path.append(".")
from datetime import datetime
import numpy as np
import gymnasium as gym
import os
import json
import torch
from collections import deque

from agent.bc_agent import BCAgent
import utils

def run_episode(env, agent, rendering=True, max_timesteps=1000, history_length=4):

    episode_reward = 0
    step = 0

    state, _ = env.reset()
    stack = deque(maxlen=history_length)

    def preprocess(state):
        cropped_state = state[0:84, 6:90]
        return utils.rgb2gray(cropped_state)


    if rendering:
        # Gymnasium CarRacing uses pygame backend; render once after reset
        # to initialize/display a valid first frame.
        env.render()

    state = preprocess(state)
    for _ in range(history_length):
        stack.append(state)
    
    while True:
        # TODO: preprocess the state in the same way than in your preprocessing in train_agent.py
        state = np.stack(stack, axis=-1)
        state_tensor = torch.from_numpy(state).unsqueeze(0).float()

        # TODO: get the action from your agent! You need to transform the discretized actions to continuous
        # actions.
        # hints:
        #       - the action array fed into env.step() needs to have a shape like np.array([0.0, 0.0, 0.0])
        #       - just in case your agent misses the first turn because it is too fast: you are allowed to clip the acceleration in test_agent.py
        #       - you can use the softmax output to calculate the amount of lateral acceleration
        a = agent.predict(state_tensor)
        a = a.squeeze(0)
        a = a.detach().cpu().numpy()
        
        next_state, r, terminated, truncated, _ = env.step(a)
        episode_reward += r
        state = next_state
        step += 1

        state = preprocess(state)
        stack.append(state)

        if rendering:
            env.render()

        if terminated or truncated or step > max_timesteps:
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True

    n_test_episodes = 15  # number of episodes to test
    history_length=6
    # TODO: load agent
    agent = BCAgent(history_length=history_length, n_classes=3, batch_size=64)
    agent.load("models/bc_agent_hw6.pt") 

    env = gym.make("CarRacing-v3", render_mode="human")

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=rendering, history_length=history_length)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    fname = f"results/6hw_10k_results_bc_agent-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(results, f)

    env.close()
    print("... finished")
