import pickle
import torch
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt

import sys

sys.path.append(".")

import utils
from agent.bc_agent import BCAgent
from tensorboard_evaluation import Evaluation
from numpy.lib.stride_tricks import sliding_window_view


def read_data(datasets_dir="./data", frac=0.1):
    """
    This method reads the states and actions recorded in drive_manually.py
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, "data.pkl.gzip")

    f = gzip.open(data_file, "rb")
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype("float32")
    y = np.array(data["action"]).astype("float32")

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = (
        X[: int((1 - frac) * n_samples)],
        y[: int((1 - frac) * n_samples)],
    )
    X_valid, y_valid = (
        X[int((1 - frac) * n_samples) :],
        y[int((1 - frac) * n_samples) :],
    )
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can train your model with discrete actions (as you get them from read_data) by discretizing the action space
    #    using action_to_id() from utils.py.
    crop_X_train = X_train[:, 0:84, 6:90]
    crop_X_valid = X_valid[:, 0:84, 6:90]
    gs_X_train, gs_X_valid = utils.rgb2gray(crop_X_train), utils.rgb2gray(crop_X_valid)

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96, 1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).
    def stack_history(X, history_length):

        padding = ((history_length - 1, 0), (0, 0), (0, 0))
        X_padded = np.pad(X, padding, mode='edge')
        X_stacked = sliding_window_view(X_padded, window_shape=history_length, axis=0)

        return X_stacked

    stacked_X_train = stack_history(gs_X_train, history_length)
    stacked_X_valid = stack_history(gs_X_valid, history_length)

    return stacked_X_train, y_train, stacked_X_valid, y_valid


def train_model(
    X_train,
    y_train,
    X_valid,
    y_valid,
    n_minibatches,
    batch_size,
    lr,
    model_dir="./models",
    tensorboard_dir="./tensorboard",
    history_length=4,
):

    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train model")

    # TODO: specify your agent with the neural network in agents/bc_agent.py
    agent = BCAgent(history_length=history_length, n_classes=3, batch_size=64)

    tensorboard_eval = Evaluation(tensorboard_dir, "Imitation Learning", stats=["train_loss", "valid_loss", "train_accuracy", "validation_accuracy", "train_accuracy_f", "validation_accuracy_f"])

    # TODO: implement the training
    #
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training *during* the training in your web browser

    def sample_minibatch(size_batch, X_train, y_train):
        idx = torch.randint(high=len(X_train), size=(size_batch,))
        X_minibatch = X_train[idx]
        y_minibatch = y_train[idx]
        assert len(X_minibatch) == size_batch, "the length of minibatches is not what it should be"
        return X_minibatch, y_minibatch

    
    X_valid = agent.to_tensor(X_valid)
    y_valid = agent.to_tensor(y_valid)
    X_train = agent.to_tensor(X_train) 
    y_train = agent.to_tensor(y_train)
    # training loop
    for i in range(n_minibatches):
        eval_dict = {}
        X_batch, y_batch = sample_minibatch(64, X_train, y_train)
        train_loss = agent.update(X_batch, y_batch)

        if i % 10 == 0:
            # compute training/ validation accuracy and write it to tensorboard
            pred_train = agent.predict(X_batch)
            pred_valid = agent.predict(X_valid)
            valid_loss = agent.loss_function(pred_valid, y_valid)
            eval_dict["train_loss"] = train_loss
            eval_dict["valid_loss"] = valid_loss
            eval_dict["train_accuracy"], eval_dict["train_accuracy_f"] = agent.calc_cont_accuracy(pred_train, y_batch)
            eval_dict["validation_accuracy"], eval_dict["validation_accuracy_f"] = agent.calc_cont_accuracy(pred_valid, y_valid)

            tensorboard_eval.write_episode_data(i, eval_dict)
            print(f"Epoch: [{i}] - Train_loss: {train_loss:.4f}, Valid_loss: {valid_loss:.4f}, Train_acc: {eval_dict['train_accuracy_f']:.2f}, Valid_acc: {eval_dict['validation_accuracy_f']:.2f} ")

    # TODO: save your agent
    model_dir = agent.save(os.path.join(model_dir, "bc_agent_hw6.pt"))
    print("Model saved in file: %s" % model_dir)


if __name__ == "__main__":
    history_length=6
    # read data
    X_train, y_train, X_valid, y_valid = read_data("./data")
    
    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(
        X_train, y_train, X_valid, y_valid, history_length=history_length, 
    )

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid, n_minibatches=1000, batch_size=64, lr=1e-4, history_length=history_length)
