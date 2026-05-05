import argparse
import gymnasium as gym
import numpy as np
import pickle
import os
from datetime import datetime
import gzip
import json
import pygame


def get_action_from_keyboard():
    """
    This method gets the action from the keyboard.
    It returns the action as a numpy array.
    The action is a tuple of (steering, gas, brake).
    The steering is a value between -1.0 and 1.0.
    The gas is a value between 0.0 and 1.0.
    The brake is a value between 0.0 and 1.0.
    """
    pygame.event.pump()
    keys = pygame.key.get_pressed()
    steering = -1.0 if keys[pygame.K_LEFT] else (1.0 if keys[pygame.K_RIGHT] else 0.0)
    gas = 1.0 if keys[pygame.K_UP] else 0.0
    brake = 0.2 if keys[pygame.K_DOWN] else 0.0
    return np.array([steering, gas, brake], dtype=np.float32)


def store_data(data, datasets_dir="./data"):
    # save data
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, "data.pkl.gzip")
    f = gzip.open(data_file, "wb")
    pickle.dump(data, f)


def save_results(episode_rewards, results_dir="./results"):
    # save results
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # save statistics in a dictionary and write them into a .json file
    results = dict()
    results["number_episodes"] = len(episode_rewards)
    results["episode_rewards"] = episode_rewards

    results["mean_all_episodes"] = np.array(episode_rewards).mean()
    results["std_all_episodes"] = np.array(episode_rewards).std()

    fname = os.path.join(
        results_dir,
        f"results_manually-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json",
    )
    fh = open(fname, "w")
    json.dump(results, fh)
    print("... finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--collect_data",
        action="store_true",
        default=False,
        help="Collect the data in a pickle file.",
    )

    args = parser.parse_args()

    samples = {
        "state": [],
        "next_state": [],
        "reward": [],
        "action": [],
        "terminal": [],
    }

    env = gym.make("CarRacing-v3", render_mode="human")
    env.reset()
    env.render()

    a = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    episode_rewards = []
    steps = 0
    while True:
        episode_reward = 0
        state, _ = env.reset()
        while True:
            a = get_action_from_keyboard()
            env.render()

            next_state, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            episode_reward += r

            samples["state"].append(state)  # state has shape (96, 96, 3)
            samples["action"].append(np.array(a))  # action has shape (1, 3)
            samples["next_state"].append(next_state)
            samples["reward"].append(r)
            samples["terminal"].append(done)

            state = next_state
            steps += 1

            if steps % 1000 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("\nstep {}".format(steps))

            if args.collect_data and steps % 5000 == 0:
                print("... saving data")
                store_data(samples, "./data")
                save_results(episode_rewards, "./results")

            if done:
                break

        episode_rewards.append(episode_reward)

    env.close()
