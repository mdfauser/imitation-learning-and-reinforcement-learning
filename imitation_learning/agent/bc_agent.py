import torch
from agent.networks import CNN
import numpy as np


class BCAgent:

    def __init__(self, history_length, n_classes, batch_size):
        # TODO: Define network, loss function, optimizer
        self.net = CNN(history_length=history_length, n_classes=n_classes, batch_size=batch_size)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=3e-4)
        self.loss_function = torch.nn.MSELoss()

    def update(self, x, y):
        # TODO: forward + backward + optimize
        pred = self.predict(x)
        y = y.squeeze(1)
        loss = self.loss_function(pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
                
        return loss
    
    def to_tensor(self, x):
        x = torch.tensor(x)
        return x

    def predict(self, X):
        outputs = self.net(X)
        return outputs
    
    def calc_disc_accuracy(self, pred, y):
        # for discrete actions
        correct_predictions = torch.all(pred == y, axis=1)
        return len(correct_predictions) / len(pred)

    def calc_cont_accuracy(self, pred, y, tolerance=0.1):
        # for discrete actions
        y = y.squeeze(1)
        diff = torch.abs(y - pred)
        # feature wise
        correct_features = torch.where(diff < tolerance, True, False)
        f_accuracy = torch.sum(correct_features) / (len(correct_features) * correct_features.size(-1))
        # action wise
        correct_predictions = torch.all(diff < tolerance, axis=1)
        a_accuracy = torch.sum(correct_predictions) / len(correct_predictions)

        return a_accuracy, f_accuracy

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)
