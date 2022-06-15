import os
from functools import partial
import numpy as np
import torch
from torch import nn


def extract(a, t, x_shape):
    """

    :param a:
    :param t:
    :param x_shape:
    :return:
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DiffusionModel(nn.Module):
    def __init__(
            self,
            image_size=(224, 224),
            channel_x=3,
            channel_y=1,
            timesteps=2000,
            seed=19520208
    ):
        super().__init__()

        self.seed = seed
        self.image_size = image_size
        self.channel_x = channel_x
        self.channel_y = channel_y
        self.timesteps = timesteps

        # The timesteps is controlled by a variance schedule \beta
        # alpha = 1 - beta
        # gamma = \hat{alpha} = \prod^T_{i=1}  alpha

        alphas = np.linspace(1e-6, 0.01, timesteps)  # a numberical seq from 1e-6 to 0.01
        gammas = np.cumprod(alphas, axis=0)  # get cummulative prod of alphas on aixs=0

        torch.manual_seed(self.seed)
        to_torch = partial(torch.sensor, dtype=torch.float32)

        # Calculation for q(y_t | y_{t-1})  -> Forward diffusion process
        self.register_buffer("gammas", to_torch(gammas))
        self.register_buffer("sqrt_one_minus_gammas", to_torch(np.sqrt(1 - gammas)))
        self.register_buffer('sqrt_gammas', to_torch(np.sqrt(gammas)))

        def noisy_image(self, t, y):
            """
            Create a noisy image from a given timestep and image
            :param self:
            :param t:
            :param y:
            :return:
            """
            # tensor the same size as input fill with random number for a normal distribution
            noise = torch.randn_like(y)
            y_noisy = extract(self.gammas, t, y.shape) * y + extract(self.sqrt_one_minus_gammas, t, noise.shape) * noise
            return y_noisy, noise

        def noise_prediction(self, denoise_fn, y_noisy, x, t):
            """

            :param self:
            :param denoise_fn:
            :param y_noisy:
            :param x:
            :param t:
            :return:
            """
            noise_pred = denoise_fn(y_noisy, x, t)
            return (noise_pred)


