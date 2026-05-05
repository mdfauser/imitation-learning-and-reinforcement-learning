import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from agent.replay_buffer import EpisodicBuffer, FunctionBuffer


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DQNAgent:

    def __init__(
        self,
        Q,
        Q_target,
        num_actions,
        gamma=0.95,
        batch_size=64,
        epsilon=0.1,
        tau=0.01,
        lr=1e-4,
        history_length=0,
        buffer_shapes={},
        device="cuda:0"
    ):
        """
        Q-Learning agent for off-policy TD control using Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.

        Args:
           Q: Action-Value function estimator (Neural Network)
           Q_target: Slowly updated target network to calculate the targets.
           num_actions: Number of actions of the environment.
           gamma: discount factor of future rewards.
           batch_size: Number of samples per batch.
           tau: indicates the speed of adjustment of the slowly updated target network.
           epsilon: Chance to sample a random action. Float betwen 0 and 1.
           lr: learning rate of the optimizer
        """
        # setup networks
        self.Q = Q.cuda(device="cuda:0")
        self.Q_target = Q_target.cuda(device="cuda:0")
        self.Q_target.load_state_dict(self.Q.state_dict())

        # define replay buffer
        self.replay_buffer = EpisodicBuffer(num_episodes=1000, max_episode_length=500, shapes=buffer_shapes) 

        # parameters
        self.batch_size = batch_size
        self.history_length = history_length
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.num_actions = num_actions
        self.device = device
        self.traj_len = history_length + 1

    def train(self):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """

        # 2. sample next batch and perform batch update:
         
        batch  = self.replay_buffer.sample(batch_size=self.batch_size, trajectory_len=self.traj_len) 
        #       2.1 compute td targets and loss
        #              td_target =  reward + discount * max_a Q_target(next_state_batch, a)
        all_states = batch["observation"].to(self.device).float() / 255.0
        actions = batch["action"].to(self.device).long()
        rewards = batch["reward"].to(self.device)
        dones = batch["terminal"].to(self.device)
        states = all_states[:, 0:self.history_length]
        next_states = all_states[:, 1:]
        last_actions = actions[:, -1] # only using the last action of each trajectory
        last_rewards = rewards[:, -1]
        last_dones = dones[:, -1]

        with torch.no_grad():
            max_target_q = self.Q_target(next_states).max(dim=-1, keepdim=True).values
            td_target = last_rewards + self.gamma * max_target_q * (1 - last_dones)
        
        cur_q = self.Q(states).gather(1, last_actions)
        loss = self.criterion(cur_q, td_target)
        
        #        2.2 update the Q network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #       2.3 call soft update for target network
        soft_update(self.Q_target, self.Q, self.tau)


    def act(self, state, deterministic):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0
        r = np.random.uniform()
        if deterministic or r > self.epsilon:
            # TODO: take greedy action (argmax)
            with torch.no_grad():
                action_id = torch.argmax(self.Q(state_t)).item()
        else:
            # TODO: sample random action
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work.
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
            # action_id = np.random.randint(self.num_actions) # cartpole
            action_id = np.random.choice([0, 1, 2, 3, 4], p=[0.1, 0.6, 0.1, 0.1, 0.1])

        return action_id

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name))
        self.Q_target.load_state_dict(torch.load(file_name))
