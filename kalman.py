import numpy as np

from utils import bayesian_update, gaussian


class Gaussian(object):
    def __init__(self, mu, sigma):
        self.mu = float(mu)
        self.sigma = float(sigma)

    def vectorize(self, xlims, step=0.1):
        x_array = np.arange(*xlims, step)
        y_array = [100 * gaussian(x, self.mu, self.sigma) for x in x_array]
        return x_array, y_array

    def bayesian_update(self, other):
        sigma_sum = self.sigma**2 + other.sigma**2
        self.mu = ((self.sigma**2) * other.mu + (other.sigma**2) * self.mu) / sigma_sum
        self.sigma = np.sqrt(((self.sigma * other.sigma)**2) / sigma_sum)


class Reading(Gaussian):

    def __init__(self, sensor, truth, timestamp=None):
        self.sensor = sensor
        self.timestamp = timestamp
        self.truth = truth
        self.value = sensor.read(truth)
        self.color = sensor.color

        self.mu = sensor.predictor(self.value)
        self.sigma = sensor.predictor_sigma

    # def plot(self, ax):
    #     x_range = np.arange(*ax.get_xlim(), 0.1)
    #     y_vector = [100 * gaussian(x, self.mu, self.sigma) for x in x_range]
    #     # print(self.mu)
    #     ax.plot(x_range, y_vector, color=self.color)
    #     ax.vlines(self.mu, 0, max(y_vector), linestyles="dotted")


class Estimate(Gaussian):

    def __init__(self, color='purple'):
        self.reading_vector = []
        self.mu = None
        self.sigma = None
        self.color = color

    def add_reading(self, reading):
        self.reading_vector.append(reading)
        self.update(reading)

    def reorder(self):
        self.reading_vector.sort(key=lambda x: x.timestamp)

    def update(self, reading):
        if self.mu is None or self.sigma is None:
            self.mu = reading.mu
            self.sigma = reading.sigma
        else:
            self.bayesian_update(reading)