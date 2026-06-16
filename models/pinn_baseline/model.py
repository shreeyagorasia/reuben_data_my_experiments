"""
models/pinn_baseline/model.py
===============================
The PINN (Physics-Informed Neural Network) model definition and its
physics-based loss term.
"""

import torch
import torch.nn as nn

import config


class PINN(nn.Module):
    """Feed-forward network that predicts tree height.

    Architecture (Reuben, Table D.3):

        input(n_other + 1) -> 128 (LeakyReLU)
                            -> 128 (LeakyReLU)
                            -> 128 (LeakyReLU)
                            -> 1

    The "+1" is the AGE feature. AGE is kept as a separate input
    (rather than just being one column among many) so that, during
    training, we can compute d(prediction)/d(AGE) with autograd and
    compare it to the Chapman-Richards growth rate -- this is the
    "physics" part of the physics-informed loss.
    """

    def __init__(self, n_other, hidden_size=None):
        super().__init__()
        hidden_size = hidden_size or config.HIDDEN_SIZE

        # n_other is the count of all non-AGE features (X, Y, land use, etc.)
        # We add + 1 to account for the separated AGE tensor being concatenated.
        self.net = nn.Sequential(
            nn.Linear(n_other + 1, hidden_size), nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x_other, t_age):
        # Concatenate the "other" features with AGE before feeding
        # them into the network.
        x = torch.cat([x_other, t_age], dim=1)
        return self.net(x).squeeze()


def cr_derivative(t_unscaled, cr_params):
    """Analytical derivative dH/dt of the Chapman-Richards curve.

    H(t)  = y_max * (1 - exp(-k*t)) ** p
    dH/dt = y_max * p * (1 - exp(-k*t)) ** (p-1) * k * exp(-k*t)

    `t_unscaled` is AGE in real years (a torch tensor).
    `cr_params` is a dict with keys "y_max", "k", "p".
    """
    y_max, k, p = cr_params["y_max"], cr_params["k"], cr_params["p"]
    e = torch.exp(-k * t_unscaled)
    return y_max * p * (1 - e) ** (p - 1) * k * e


def pinn_loss(pred, t_scaled, y_true, mse_fn, cr_params, sigma_y, sigma_age, age_mean):
    """Compute the two pieces of the PINN loss (Eq. 3.9 / 3.10):

        data_loss    = MSE(prediction, true height)            [usual ML loss]
        physics_loss = MSE(dPrediction/dAGE, dChapmanRichards/dAGE)

    The physics term encourages the network's predicted GROWTH RATE
    (with respect to AGE) to follow the Chapman-Richards curve, even
    though the network is not told that curve's formula directly.

    Parameters
    ----------
    pred       : model predictions (scaled units), shape (batch,)
    t_scaled   : scaled AGE tensor, requires_grad=True, shape (batch, 1)
    y_true     : true heights (scaled units), shape (batch,)
    mse_fn     : nn.MSELoss()
    cr_params  : dict with the fitted Chapman-Richards parameters
    sigma_y    : standard deviation used to scale the target (height)
    sigma_age  : standard deviation used to scale AGE
    age_mean   : mean used to scale AGE

    Returns
    -------
    data_loss, physics_loss  (both are scalar tensors)
    """
    data_loss = mse_fn(pred, y_true)

    # d(prediction)/d(AGE) in SCALED units, computed via autograd.
    dy_dt_scaled = torch.autograd.grad(
        outputs=pred,
        inputs=t_scaled,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True,
    )[0].squeeze()

    # Chain rule: convert the scaled derivative back to real units
    # (metres of height per year of age).
    dy_dt_unscaled = dy_dt_scaled * (sigma_y / sigma_age)

    # Convert the scaled AGE values back to real years, so we can
    # plug them into the Chapman-Richards derivative formula.
    t_years = t_scaled.squeeze() * sigma_age + age_mean

    physics_loss = mse_fn(dy_dt_unscaled, cr_derivative(t_years, cr_params))
    return data_loss, physics_loss
