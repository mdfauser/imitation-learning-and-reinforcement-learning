import os
from datetime import datetime
import gymnasium as gym
import json
from agent.dqn_agent import DQNAgent
from train_cartpole import run_episode
from agent.networks import MLP
import numpy as np

np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("CartPole-v1", render_mode="human")

    # TODO: load DQN agent
    # ...
    observation_shape = env.observation_space
    action_shape = env.action_space

    buffer_shapes = {
        "observation": [observation_shape],
        "action": [action_shape],
        "next_observation": [observation_shape],
        "reward": [1],
        "terminal":[1],
    }
    Q = MLP(state_dim=observation_shape, action_dim=action_shape, hidden_dim=400)

    agent = DQNAgent(Q=Q, Q_target=Q, num_actions=action_shape, 
                    gamma=0.95, batch_size=64, epsilon=0.1, 
                    tau=0.01, lr=1e-4, history_length=0, 
                    buffer_shapes=buffer_shapes
                    )

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

    fname = f"./results/cartpole_results_dqn-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(results, f)

    env.close()
    print("... finished")
