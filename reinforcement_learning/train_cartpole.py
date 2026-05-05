import sys

sys.path.append("./")
import time
import os
import numpy as np
import torch
import gymnasium as gym
import itertools as it
from agent.dqn_agent import DQNAgent
from tensorboard_evaluation import Evaluation
from agent.networks import MLP
from utils import EpisodeStats


def run_episode(
    env, agent, deterministic, do_training=False, rendering=False, max_timesteps=1000, fill_buffer=False ,train_freq=1
):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """
    stats = EpisodeStats()  # save statistics like episode reward or action usage
    state, _ = env.reset()
    step = 0
    while True:
        action_id = agent.act(state=state, deterministic=deterministic)
        next_state, reward, terminated, truncated, info = env.step(action_id)

        if fill_buffer or do_training:
            data = {
            "observation": torch.from_numpy(state),
            "action": torch.tensor(action_id),
            "next_observation": torch.from_numpy(next_state),
            "reward": torch.tensor(reward),
            "terminal": torch.tensor(terminated),
            }
            agent.replay_buffer.append(data)

        if do_training and step % train_freq == 0:
            agent.train()

        stats.step(reward, action_id)
        state = next_state

        if rendering:
            env.render()

        if terminated or truncated or step > max_timesteps:
            break

        step += 1

    return stats


def train_online(
    env,
    agent,
    num_episodes,
    device_storage="cpu",
    device_calc="cuda",
    model_dir="./models",
    tensorboard_dir="./tensorboard",
):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train agent")

    tensorboard = Evaluation(
        tensorboard_dir,
        "CartPole",
        stats=["train/episode_reward", "train/a_0", "train/a_1", "eval/mean_reward"],
    )

    # training
    for i in range(100):
        print("fill buffer: ", i)
        stats = run_episode(env, agent, deterministic=False, do_training=False, fill_buffer=True)

    time_eval = 0
    for i in range(num_episodes):
        print("episode: ", i)
        start_run = time.time()
        stats = run_episode(env, agent, deterministic=False, do_training=True, train_freq=2)
        time_run = time.time() - start_run
        start_store = time.time()
        tensorboard.write_episode_data(
            i,
            eval_dict={
                "train/episode_reward": stats.episode_reward,
                "train/a_0": stats.get_action_usage(0),
                "train/a_1": stats.get_action_usage(1),
            },
        )
        time_store = time.time() - start_store
        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        # ...
        if i % eval_cycle == 0:
            start_eval = time.time()
            eval_rewards = [] 
            for j in range(num_eval_episodes):
                eval_stat = run_episode(env, agent, deterministic=True, do_training=False, max_timesteps=500)
                eval_rewards.append(eval_stat.episode_reward)
            
            mean_reward = sum(eval_rewards) / len(eval_rewards)
            tensorboard.write_episode_data(i, eval_dict={"eval/mean_reward": mean_reward})
            
            print(f"[{i}] Evaluation Mean Reward: {mean_reward}")
            time_eval = time.time() - start_eval
            print(f"Env: {time_run:.4f}s | Store: {time_store:.4f}s | Eval: {time_eval:.4f}s")

        # store model.
        if i % eval_cycle == 0 or i >= (num_episodes - 1):
            agent.save(os.path.join(model_dir, "dqn_agent_cartpole.pt"))

        

    tensorboard.close_session()


if __name__ == "__main__":

    num_eval_episodes = 5  # evaluate on 5 episodes
    eval_cycle = 200  # evaluate every 10 episodes

    # You find information about cartpole in
    # https://gymnasium.farama.org/environments/classic_control/cart_pole/
    # Hint: CartPole is considered solved when the average reward is greater than or equal to 490.0 over 100 consecutive trials.

    env = gym.make("CartPole-v1", render_mode="rgb_array")

    state_dim = 4
    num_actions = 2
    device_storage = "cpu"
    device_calc = "cuda"

    # TODO:
    # 1. init Q network and target network (see dqn/networks.py)
    # 2. init DQNAgent (see dqn/dqn_agent.py)
    # 3. train DQN agent with train_online(...)

    buffer_shapes = {
        "observation": [state_dim],
        "action": [1], # for continous num_actions
        "next_observation": [state_dim],
        "reward": [1],
        "terminal":[1],
    }
    Q = MLP(state_dim=state_dim, action_dim=num_actions, hidden_dim=400)

    agent = DQNAgent(Q=Q, Q_target=Q, num_actions=num_actions, 
                    gamma=0.95, batch_size=128, epsilon=0.1, 
                    tau=0.01, lr=1e-4, history_length=1, 
                    buffer_shapes=buffer_shapes
            )
    train_online(env, agent, num_episodes=2000, device_storage=device_storage, device_calc=device_calc)
