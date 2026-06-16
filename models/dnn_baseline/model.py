"""
models/dnn_baseline/model.py
==============================
The DNN model definition.
"""

import torch
import torch.nn as nn
import config

class DNN(nn.Module):
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
        return self.net(x).squeeze()
