import torch.nn as nn
import torch
import torch.nn.functional as F

"""
CartPole network
"""


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=400):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CNN(nn.Module):
    def __init__(self, history_length, n_classes=3, batch_size=64, pool_stride=4 , cnn_kernels =[(8,8), (4,4), (3,3)], cnn_strides=[4,2,1]):
        super(CNN, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=history_length, out_channels=32, kernel_size=cnn_kernels[0], stride=cnn_strides[0]), 
            nn.ReLU(), 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=cnn_kernels[1], stride=cnn_strides[1]), 
            nn.ReLU(), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=cnn_kernels[2], stride=cnn_strides[2]), 
            nn.ReLU(), 
            nn.Flatten(), 
        )

        with torch.no_grad():
                size_input = torch.zeros(1, history_length, 84, 84)
                size_output = self.sequential(size_input)
                self.flattened_size = size_output.shape[1]

        self.linear = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )


    def forward(self, x):
        x = self.sequential(x.float())
        x = self.linear(x)
        return x

