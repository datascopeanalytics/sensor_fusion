import math
import random

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

import seaborn as sns
import traces
import scipy


def plot_linear_fit(ax, x_array, y_array, fit_function, fit_sigma, color, cmap):
    xlim = (min(x_array), max(x_array))
    ylim = (min(y_array), max(y_array))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    x_range = np.linspace(*xlim)
    y_range = np.linspace(*ylim)

    ax.scatter(x_array, y_array, lw=0, alpha=0.5, color=color)
    fit_line = [fit_function(x) for x in x_range]
    ax.plot(x_range, fit_line, color=color)

    xx, yy = np.meshgrid(x_range, y_range)
    zz = xx + yy

    for i in range(len(x_range)):
        for j in range(len(y_range)):
            zz[j, i] = gaussian(yy[j, i], fit_function(xx[j, i]), fit_sigma)

    im = ax.imshow(
        zz, origin='lower', interpolation='bilinear',
        cmap=cmap, alpha=0.5, aspect='auto',
        extent=(xlim[0], xlim[-1], ylim[0], ylim[-1]),
        vmin=0.0, vmax=gaussian(0, 0, fit_sigma)
    )

    return ax, im


class Sensor(object):

    def __init__(self, name, **kwargs):
        self.name = name
        for key, value in kwargs.items():
            setattr(self, key, value)

    def read(self, variable):
        variable = max(0, random.gauss(variable, self.proc_sigma))
        reading = self.intersect + variable * self.slope
        return random.gauss(reading, self.sigma)

    def fit(self, data):
        self.experiment_data = data
        n_samples = len(data)

        model_slope, model_intercept = np.polyfit(
            [o for o, r in data], [r for o, r in data], 1)

        def model(occupants):
            return occupants * model_slope + model_intercept
        self.model = model

        error = 0.0
        for occupants, reading in data:
            error += (model(occupants) - reading)**2
        sigma = np.sqrt(error / (n_samples - 1))
        self.model_sigma = sigma

        predictor_slope, predictor_intercept = np.polyfit(
            [r for o, r in data], [o for o, r in data], 1)

        def predictor(sensor_reading):
            return sensor_reading * predictor_slope + predictor_intercept
        self.predictor = predictor

        error = 0.0
        for occupants, reading in data:
            error += (predictor(reading) - occupants)**2
        sigma = np.sqrt(error / (n_samples - 1))
        self.predictor_sigma = sigma

    def _round_up(self, reading):
        scale = float(self.round_level)
        return int(math.ceil(reading / float(scale)) * scale)

    def _round_down(self, reading):
        scale = float(self.round_level)
        return int(math.floor(reading / float(scale)) * scale)

    def plot_experiment(self):
        color = self.color
        data = self.experiment_data
        cmap = sns.light_palette(color, as_cmap=True)

        fig, ax = plt.subplots()
        occupants, readings = (np.array(array) for array in zip(*data))

        # ax_left, im_left = plot_linear_fit(
        # ax_left, occupants, readings, self.model, self.model_sigma, color,
        # cmap)

        ax, im = plot_linear_fit(
            ax, readings, occupants, self.predictor, self.predictor_sigma, color, cmap)

        # cax, kw = mpl.colorbar.make_axes([ax_left, ax_right], location="bottom")

        # norm = mpl.colors.Normalize(vmin=0, vmax=1)
        # cbar = mpl.colorbar.ColorbarBase(
        #     ax, cmap=cmap, norm=norm, alpha=0.5)

        cbar = plt.colorbar(im, alpha=0.5, extend='neither', ticks=[
            gaussian(3 * self.predictor_sigma, 0, self.predictor_sigma),
            gaussian(2 * self.predictor_sigma, 0, self.predictor_sigma),
            gaussian(self.predictor_sigma, 0, self.predictor_sigma),
            gaussian(0, 0, self.predictor_sigma),
        ])
        # cbar.solids.set_edgecolor("face")

        cbar.set_ticklabels(
            ['$3 \sigma$', '$2 \sigma$', '$\sigma$', '{:.2%}'.format(
                gaussian(0, 0, self.predictor_sigma))],
            update_ticks=True
        )

        fig.savefig("experiment_plots/" + self.name + ".png")


class TrainCar(object):

    max_occupants = 120
    occupant_range = range(0, max_occupants + 1)

    def __init__(self, occupants=0):
        self.sigma = 0
        self.occupants = occupants
        self.sensor_array = [
            Sensor("co2", intersect=350, slope=15, sigma=10,
                   round_level=500, proc_sigma=30),
            Sensor("temp", intersect=0, slope=0.25,
                   sigma=5, round_level=10, proc_sigma=5)
        ]

    def generate_occupancy(self, start=0, end=30, stations=5):
        self.occupants_trace = traces.TimeSeries()
        self.occupants_trace[start] = random.randint(
            1, self.max_occupants/2)
        self.occupants_trace[end] = 0

        # at each station a certain number of people get on or off
        for _ in range(stations):
            minute = random.randint(start + 1, end - 1)
            current_val = self.occupants_trace[minute]
            new_val = max(0, int(random.gauss(current_val, 20)))
            self.occupants_trace[minute] = new_val

        return self.occupants_trace

    def read_sensors(self, experiment=True, timestamp=None):
        reading_dict = {}
        for sensor in self.sensor_array:
            if experiment:
                occupants = max(0, random.gauss(self.occupants, self.sigma))
            elif timestamp:
                occupants = self.occupants_trace[timestamp]
            else:
                raise AttributeError("gimme some moneeey!")
            reading_dict[sensor.name] = sensor.read(occupants)
        return reading_dict

    def plot_experiment(self, **kwargs):
        for i, sensor in enumerate(self.sensor_array):
            color = sns.color_palette()[i]
            sensor.color = color
            sensor.plot_experiment()

    def run_experiment(self, datapoints=1000):
        """Generates fake sensor data"""

        data = []

        for _ in range(datapoints):
            self.occupants = random.randrange(self.max_occupants + 1)
            data.append((self.occupants, self.read_sensors()))

        for i, sensor in enumerate(self.sensor_array):
            sensor_data = [(o, r[sensor.name]) for o, r in data]
            sensor.fit(sensor_data)

        self.experiment_data = data


def plot_readings(reading_vector, predictor, filename, x_range=[0, 15], fuse=False):
    plt.clf()
    ax = plt.gca()
    x_vector = np.linspace(*x_range)
    palette = sns.color_palette()
    fusion = None
    for reading in reading_vector:
        occupants, sigma = predictor(reading)
        if fuse:
            if not fusion:
                fusion = occupants, sigma
            else:
                fusion = bayesian_update(fusion, (occupants, sigma))
        y_vector = 100 * gaussian(x_vector, occupants, sigma)
        ax.plot(x_vector, y_vector, color=palette[1])
        ax.vlines(occupants, 0, max(y_vector), linestyles='dotted',
                  label='{}ppm'.format(reading))

    if fuse:
        y_vector = 100 * gaussian(x_vector, *fusion)
        ax.plot(x_vector, y_vector, color=palette[3])

    ax.set_ylim(0, 10)
    ax.set_xlim(*x_range)
    plt.savefig(filename)


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) / (sig * np.sqrt(2. * np.pi))


def bayesian_update(gaussian_a, gaussian_b):
    mu_a, sigma_a = gaussian_a
    mu_b, sigma_b = gaussian_b
    mu = ((sigma_a**2) * mu_b + (sigma_b**2)
          * mu_a) / (sigma_a**2 + sigma_b**2)
    sigma = np.sqrt(((sigma_a * sigma_b)**2) / (sigma_a**2 + sigma_b**2))

    return mu, sigma


class Reading(object):

    def __init__(self, sensor, truth, timestamp=None):
        self.sensor = sensor
        self.timestamp = timestamp
        self.truth = truth
        self.value = sensor.read(truth)
        self.mu = sensor.predictor(self.value)
        self.sigma = sensor.predictor_sigma
        self.color = sensor.color

    def plot(self, ax):
        x_range = np.arange(*ax.get_xlim(), 0.1)
        y_vector = [100 * gaussian(x, self.mu, self.sigma) for x in x_range]
        print(self.mu)
        ax.plot(x_range, y_vector, color=self.color)
        ax.vlines(self.mu, 0, max(y_vector), linestyles="dotted")


class Estimate(object):

    def __init__(self):
        self.reading_vector = []
        self.estimate = None

    def add_reading(self, reading_obj):
        self.reading_vector.append(reading_obj)

    def reorder(self):
        self.reading_vector.sort(key=lambda x: x.timestamp)

    def plot(self):
        pass

if __name__ == "__main__":

    train_car = TrainCar()
    co2_sensor, temp_sensor = train_car.sensor_array
    train_car.run_experiment(datapoints=250)
    train_car.plot_experiment()
    train_car.generate_occupancy()  # defaults to 5 stations and 30 minutes

    time_array = np.arange(-1, 31, 1.0/60)
    co2_array = []
    temp_array = []
    truth = []
    for t in time_array:
        reading = train_car.read_sensors(timestamp=t)
        co2_array.append(reading["co2"])
        temp_array.append(reading["temp"])
        truth.append(train_car.occupants_trace[t])

    plt.clf()
    plt.plot(time_array, truth)
    plt.savefig("truth.png")

    plt.clf()
    plt.plot(time_array, co2_array)
    plt.savefig("co2.png")

    plt.clf()
    ax = plt.gca()
    ax.set_xlim(0, train_car.max_occupants)
    reading = Reading(co2_sensor, 55)
    reading.plot(ax)
    plt.savefig("test.png")
