"""
models/dnn_baseline/model.py
==============================
The DNN model definition.
"""

import torch
import torch.nn as nn
import config

class DNN(nn.Module):
    """A simple Deep Neural Network (feed-forward) for regression.

    Architecture (Reuben, Table D.3, adapted for a standard DNN):
        input(n_features) -> 128 (LeakyReLU)
                          -> 128 (LeakyReLU)
                          -> 128 (LeakyReLU)
                          -> 1 (output)

    Parameters
    ----------
    n_features : int
        The number of input features the model should expect.
    hidden_size : int, optional
        The number of neurons in each hidden layer. If None, uses the
        value from this model's config.py.
    """
    def __init__(self, n_features, hidden_size=None):
        super().__init__()
        hidden_size = hidden_size or config.HIDDEN_SIZE
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size), nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        # The .squeeze() removes the trailing dimension of size 1 from the output,
        # changing the shape from (batch_size, 1) to (batch_size,).
        return self.net(x).squeeze()
