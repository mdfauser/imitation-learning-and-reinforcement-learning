import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Imitation learning network
"""


class CNN(nn.Module):

    def __init__(self, history_length, n_classes=3, batch_size=64, pool_stride=4 , cnn_kernels =[(8,8), (4,4), (3,3)], cnn_strides=[4,2,1]):
        super(CNN, self).__init__()
        # TODO : define layers of a convolutional neural network
        self.sequential = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=history_length, out_channels=32, kernel_size=cnn_kernels[0], stride=cnn_strides[0]), 
            nn.ReLU(), 
            # Layer 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=cnn_kernels[1], stride=cnn_strides[1]), 
            nn.ReLU(), 
            # Layer 3
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
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.sequential(x)
        x = self.linear(x)
        return x
