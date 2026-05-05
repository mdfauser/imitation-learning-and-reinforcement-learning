# export DISPLAY=:0

# import pyglet
# pyglet.options["headless"] = True

import sys
import os

sys.path.append("./")

import torch
import numpy as np
import gymnasium as gym
from tensorboard_evaluation import Evaluation
from utils import EpisodeStats, rgb2gray
from utils import action_to_id, id_to_action, LEFT, RIGHT, BRAKE, ACCELERATE, STRAIGHT
from agent.dqn_agent import DQNAgent
from agent.networks import CNN
from collections import deque
import time


def run_episode(
    env,
    agent,
    deterministic,
    skip_frames=1,
    do_training=True,
    rendering=False,
    max_timesteps=1000,
    history_length=1,
    fill_buffer=False,
    train_freq=1
):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    # image_hist = deque(maxlen=history_length)
    state_buffer = np.zeros((history_length, 84, 84), dtype=np.uint8)

    step = 0
    state, _ = env.reset()

    if rendering:
        env.render()

    # append image history to first state
    sg_state = state_preprocessing(state)
    # image_hist.extend([sg_state] * history_length)
    # state = np.array(image_hist)
    acc_reward = 0
    start_ep_time = time.time()
    train_time = 0
    for i in range(history_length):
        state_buffer[i] = sg_state

    while True:

        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly.

        # TODO before calling we need the history window
        # float_state = state / 255.0
        action_id = agent.act(state=state_buffer, deterministic=deterministic)

        action = id_to_action(action_id)

        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            sg_next_state, r, terminated, truncated, info = env.step(action)
            reward += r

            # if rendering:
            #     env.render()

            if terminated or truncated:
                break
        
        acc_reward += reward
        sg_next_state = state_preprocessing(sg_next_state)
        # image_hist.append(sg_next_state)
        # # image_hist.pop(0)
        # next_state = np.array(image_hist)
        state_buffer = np.roll(state_buffer, -1, axis=0)
        state_buffer[-1] = sg_next_state

        if fill_buffer or do_training:
            obs_uint8 = (torch.from_numpy(sg_state)).to(torch.uint8)
            # next_obs_uint8 = (torch.from_numpy(sg_next_state)).to(torch.uint8)
            data = {
                "observation": obs_uint8,
                "action": torch.tensor(action_id),
                # "next_observation": next_obs_uint8, # delete next_obs
                "reward": torch.tensor(reward),
                "terminal": torch.tensor(terminated, dtype=torch.bool),
                }
            agent.replay_buffer.append(data)

        if do_training:
            start_train = time.time()
            agent.train()
            train_time += time.time() - start_train

        stats.step(reward, action_id)

        # state = next_state
        sg_state = sg_next_state

        if terminated or truncated or (step * (skip_frames + 1)) > max_timesteps:
            break
        
        if acc_reward < -50:
            break

        step += 1
    run_ep_time = time.time() - start_ep_time

    print(f"Run Episode: {run_ep_time:.4f}s | Train: {train_time}")

    return stats


def train_online(
    env,
    agent,
    num_episodes,
    history_length,
    max_timesteps=1000,
    model_dir="./models",
    tensorboard_dir="./tensorboard",
):

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train agent")
    tensorboard = Evaluation(
        tensorboard_dir,
        "CarRacing",
        stats=[
            "train/episode_reward",
            "train/straight",
            "train/left",
            "train/right",
            "train/accel",
            "train/brake",
            "eval/mean_reward"
        ],
    )
    for i in range(10):
        print("fill buffer: ", i)
        stats = run_episode(env, agent, deterministic=False, do_training=False, max_timesteps=600, history_length=history_length, fill_buffer=True)

    for i in range(1, num_episodes + 1):
        print("epsiode %d" % i)

        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)

        stats = run_episode(
            env,
            agent,
            max_timesteps=max_timesteps,
            deterministic=False,
            do_training=True,
            history_length=history_length,
            train_freq = 1
        )

        tensorboard.write_episode_data(
            i,
            eval_dict={
                "train/episode_reward": stats.episode_reward,
                "train/straight": stats.get_action_usage(STRAIGHT),
                "train/left": stats.get_action_usage(LEFT),
                "train/right": stats.get_action_usage(RIGHT),
                "train/accel": stats.get_action_usage(ACCELERATE),
                "train/brake": stats.get_action_usage(BRAKE),
            },
        )

        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        if i % eval_cycle == 0:
            eval_rewards = [] 
            for j in range(num_eval_episodes):
                eval_stat = run_episode(env, agent, deterministic=True, do_training=False, max_timesteps=1000)
                eval_rewards.append(eval_stat.episode_reward)
            
            mean_reward = sum(eval_rewards) / len(eval_rewards)
            tensorboard.write_episode_data(i, eval_dict={"eval/mean_reward": mean_reward})
            
            print(f"[{i}] Evaluation Mean Reward: {mean_reward}")


        # store model.
        if i % eval_cycle == 0 or (i >= num_episodes - 1):
            agent.save(os.path.join(model_dir, "dqn_agent_carracing.pt"))

    tensorboard.close_session()


def state_preprocessing(state):
    cut_state = state[0:84, 6:90, :] 
    return rgb2gray(cut_state).astype(np.uint8) #rgb2gray(cut_state).reshape(84, 84) #/ 255.0


if __name__ == "__main__":

    num_eval_episodes = 5
    eval_cycle = 200
    history_length = 4
    batch_size = 128
    num_actions = 5

    env = gym.make("CarRacing-v3", render_mode="rgb_array")

    state_dim = (84,84)

    buffer_shapes = {
        "observation": state_dim,
        "action": [1], # for continous num_actions
        # "next_observation": state_dim,
        "reward": [1],
        "terminal":[1],
    }

    Q = CNN(history_length=history_length, 
            n_classes=num_actions, 
            batch_size=batch_size, 
            pool_stride=4 , 
            cnn_kernels =[(8,8), (4,4), (3,3)], 
            cnn_strides=[4,2,1]
            )

    agent = DQNAgent(Q=Q, Q_target=Q, num_actions=num_actions, gamma=0.95, batch_size=batch_size, epsilon=0.1, tau=0.01, 
                    lr=1e-4, history_length=history_length,  buffer_shapes=buffer_shapes, device="cuda:0")

    train_online(env, agent, num_episodes=600, history_length=history_length)
