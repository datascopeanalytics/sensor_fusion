import math
import random

import numpy as np
from matplotlib import pyplot as plt

import seaborn as sns


def generate(intersect, slope, sensor_sigma, occupant_range, occupant_sigma, datapoints=1000):
    """Generates fake sensor data"""

    fake_data = {
        'occupants': [],
        'reading': []}

    for _ in range(datapoints):
        real_occupants = random.randrange(occupant_range + 1)
        noisy_occupants = max(0, random.gauss(real_occupants, occupant_sigma))
        reading = intersect + noisy_occupants * slope
        reading = random.gauss(reading, sensor_sigma)
        fake_data['occupants'].append(real_occupants)
        fake_data['reading'].append(reading)

    return fake_data


def plot(data, filename):
    plt.cla()
    plt.scatter(x, y)
    slope, intercept = np.polyfit(data["occupants"], data["reading"], 1)
    occupant_range = np.array(
        range(min(data["occupants"]), max(data["occupants"]))
    )
    fit = occupant_range * slope + intercept
    plt.plot(occupant_range, fit)


def fit(data):

    slope, intercept = np.polyfit(data["occupants"], data["reading"], 1)
    error = 0.0
    n_samples = len(data["occupants"])
    for occupants, reading in zip(data["occupants"], data["reading"]):
        error += (occupants * slope + intercept - reading)**2

    sigma = np.sqrt(error / (n_samples - 1))

    def sensor_model(occupants):
        return occupants * slope + intercept, sigma

    return sensor_model


def round_up(number, scale):
    return int(math.ceil(number / float(scale)) * scale)


def round_down(number, scale):
    return int(math.floor(number / float(scale)) * scale)


def plot(data, sensor_model, filename):
    occupant_vector = np.array(
        range(min(data["occupants"]) - 1, max(data["occupants"]) + 2)
    )
    fit_vector = np.array([sensor_model(o)[0] for o in occupant_vector])
    _, sigma = sensor_model(occupant_vector[0])
    error_vector = [sigma] * len(occupant_vector)

    plt.cla()
    ax = plt.gca()
    ax.set_xlim([occupant_vector[0], occupant_vector[-1]])
    ax.set_ylim([round_down(min(data["reading"]), 200),
                 round_up(max(data["reading"]), 200)])

    ax = sns.regplot(
        x=np.array(data["occupants"]),
        y=np.array(data["reading"]),
        marker='x',
        fit_reg=False
    )
    ax.plot(occupant_vector, fit_vector)
    # ax.fill_between(
    #     occupant_vector,
    #     fit_vector - sigma,
    #     fit_vector + sigma,
    #     alpha=0.3,
    #     lw=0,
    # )

    x_range = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 50)
    y_range = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 50)

    xx, yy = np.meshgrid(x_range, y_range)
    zz = xx + yy

    for i in range(len(x_range)):
        for j in range(len(y_range)):
            zz[j, i] = gaussian(yy[j, i], *sensor_model(xx[j, i]))

    pal = sns.light_palette("green", as_cmap=True)
    ax.contourf(
        xx, yy, zz,
        alpha=0.5,
        cmap=pal,
        antialiased=True,
        levels=[
            gaussian(3 * sigma, 0, sigma),
            gaussian(2 * sigma, 0, sigma),
            gaussian(sigma, 0, sigma),
            gaussian(0, 0, sigma),
        ]
    )

    plt.savefig(filename)


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) / (sig * np.sqrt(2. * np.pi))

if __name__ == "__main__":

    data = generate(350, 60, 10, 15, 5, 250)
    sensor_model = fit(data)
    plot(data, sensor_model, 'test.pdf')
