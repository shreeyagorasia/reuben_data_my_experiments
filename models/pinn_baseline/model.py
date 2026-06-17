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
        # them into the network. `dim=1` concatenates along the feature axis.
        # Input shape becomes (batch_size, n_other + 1).
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
    # --------------------------------------------------------------------------
    # 1. DATA LOSS (L_data): Standard Mean Squared Error
    # This is the "usual" machine learning loss term that measures how far the
    # model's height predictions (pred) are from the true heights (y_true).
    # --------------------------------------------------------------------------
    data_loss = mse_fn(pred, y_true)

    # --------------------------------------------------------------------------
    # 2. PHYSICS LOSS (L_phys): Growth Rate Consistency
    # This term encourages the model's predicted growth rate to match the
    # analytical growth rate from the Chapman-Richards (CR) curve.
    # --------------------------------------------------------------------------

    # Step 2a: Compute the model's predicted growth rate using autograd.
    # `torch.autograd.grad` calculates the derivative of `pred` with respect to `t_scaled`.
    # This gives us d(Height_scaled) / d(Age_scaled).
    dy_dt_scaled = torch.autograd.grad(
        outputs=pred,
        inputs=t_scaled,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True,
    )[0].squeeze()

    # Step 2b: Convert the scaled derivative back to real-world units (m/year) using the chain rule.
    # dH/dt = (dH/dH_scaled) * (dH_scaled/dt_scaled) * (dt_scaled/dt)
    #       = sigma_y * dy_dt_scaled * (1/sigma_age)
    dy_dt_unscaled = dy_dt_scaled * (sigma_y / sigma_age)

    # Step 2c: Un-scale the age values from the batch back to real years.
    # This is needed to calculate the target CR growth rate at the correct age.
    t_years = t_scaled.squeeze() * sigma_age + age_mean

    # Step 2d: Calculate the final physics loss.
    # This is the MSE between the model's predicted growth rate (from autograd)
    # and the analytical CR growth rate at each plot's age.
    physics_loss = mse_fn(dy_dt_unscaled, cr_derivative(t_years, cr_params))
    return data_loss, physics_loss
