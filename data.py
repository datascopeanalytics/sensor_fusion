import random

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def generate(intersect, slope, sensor_sigma, occupant_range, occupant_sigma, datapoints=1000):
    """Generates fake sensor data"""

    fake_data = {
        'occupants': [],
        'reading': []}

    for _ in xrange(datapoints):
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

if __name__ == "__main__":

    data = generate(350, 60, 10, 15, 5, 250)
    occupant_vector = np.array(
        range(min(data["occupants"])-1, max(data["occupants"]) + 2)
    )
    sensor_model = fit(data)
    fit_vector = np.array([sensor_model(o)[0] for o in occupant_vector])
    _, sigma = sensor_model(occupant_vector[0])
    error_vector = [sigma] * len(occupant_vector)

    plt.cla()
    ax = plt.gca()
    ax.set_xlim([occupant_vector[0], occupant_vector[-1]])
    ax.set_ylim([0,  2000])

    ax = sns.regplot(
        x=np.array(data["occupants"]),
        y=np.array(data["reading"]),
        marker='x',
        fit_reg=False
    )
    ax.plot(occupant_vector, fit_vector)
    ax.fill_between(
        occupant_vector,
        fit_vector - sigma,
        fit_vector + sigma,
        alpha=0.3,
        lw=0,
    )

    plt.savefig('test.pdf')
