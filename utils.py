import numpy as np


def gaussian(x, mu, sig):
    return np.exp(
        -np.power(x - mu, 2.) / (2 * np.power(sig, 2.))
    ) / (sig * np.sqrt(2. * np.pi))
