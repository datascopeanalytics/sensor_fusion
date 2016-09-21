import math
import random

import numpy as np
from matplotlib import pyplot as plt

import seaborn as sns


class Sensor(object):

    def __init__(self, name, **kwargs):
        self.name = name
        for key, value in kwargs.items():
            setattr(self, key, value)

    def read(self, variable):
        reading = self.intersect + variable * self.slope
        return random.gauss(reading, self.sigma)

    def fit(self, data):
        slope, intercept = np.polyfit(
            [o for o, r in data], [r for o, r in data], 1)
        error = 0.0
        n_samples = len(data)
        for occupants, reading in data:
            error += (occupants * slope + intercept - reading)**2

        sigma = np.sqrt(error / (n_samples - 1))

        def model(occupants):
            return occupants * slope + intercept

        def predictor(sensor_reading):
            return (sensor_reading - intercept) / slope

        self.model = model
        self.predictor = predictor
        self.model_sigma = sigma
        self.predictor_sigma = sigma / slope

    def round_up(self, reading):
        scale = float(self.round_level)
        return int(math.ceil(reading / float(scale)) * scale)

    def round_down(self, reading):
        scale = float(self.round_level)
        return int(math.floor(reading / float(scale)) * scale)


class TrainCar(object):

    max_occupants = 120
    occupant_range = range(0, max_occupants+1)

    def __init__(self, occupants=0):
        self.sigma = self.max_occupants / 5
        self.occupants = occupants
        self.sensor_array = [
            Sensor("co2", intersect=350, slope=15, sigma=10, round_level=500),
            Sensor("temp", intersect=19, slope=0.6, sigma=0.5, round_level=10)
        ]

    def read_sensors(self):
        reading_dict = {}
        for sensor in self.sensor_array:
            occupants = max(0, random.gauss(self.occupants, self.sigma))
            reading_dict[sensor.name] = sensor.read(occupants)
        return reading_dict

    def run_experiment(self, datapoints=1000):
        """Generates fake sensor data"""

        data = []

        for _ in range(datapoints):
            self.occupants = random.randrange(self.max_occupants + 1)
            data.append((self.occupants, self.read_sensors()))

        for sensor in self.sensor_array:
            sensor_data = [(o, r[sensor.name]) for o, r in data]
            sensor.fit(sensor_data)
            f, (ax_left, ax_right) = plt.subplots(1, 2)
            occupants, readings = (np.array(array) for array in zip(*sensor_data))
            ax_left.set_xlim(min(occupants), max(occupants))
            ax_left.set_ylim(
                sensor.round_down(min(readings)),
                sensor.round_up(max(readings))
            )
            ax_left.scatter(occupants, readings, marker='x')
            fit_line = [sensor.model(x) for x in self.occupant_range]
            ax_left.plot(self.occupant_range, fit_line)

            x_range = np.linspace(ax_left.get_xlim()[0], ax_left.get_xlim()[1], 100)
            y_range = np.linspace(ax_left.get_ylim()[0], ax_left.get_ylim()[1], 100)

            xx, yy = np.meshgrid(x_range, y_range)
            zz = xx + yy

            for i in range(len(x_range)):
                for j in range(len(y_range)):
                    zz[j, i] = gaussian(yy[j, i], sensor.model(xx[j, i]), sensor.model_sigma)

            pal = sns.light_palette("green", as_cmap=True)

            im = ax_left.imshow(zz,  interpolation='bilinear', origin='lower',
                            cmap=pal, alpha=0.5, aspect='auto',
                            extent=(x_range[0], x_range[-1], y_range[0], y_range[-1]))
            cb = f.colorbar(im, orientation='vertical', ticks=[
                gaussian(3 * sensor.model_sigma, 0, sensor.model_sigma),
                gaussian(2 * sensor.model_sigma, 0, sensor.model_sigma),
                gaussian(sensor.model_sigma, 0, sensor.model_sigma),
                gaussian(0, 0, sensor.model_sigma),
            ],
                drawedges=True
            )
            cb.set_ticklabels(['$3 \sigma$', '$2 \sigma$',
                               '$\sigma$', 'max'], update_ticks=True)

            f.savefig("experiment_plots/"+sensor.name+".png")

        self.experiment_data = data




def plot_sensor_model(data, sensor_model, filename, round_level=10):

    # ax = sns.regplot(x=x_array, y=y_array, marker='x', fit_reg=False )
    ax.plot(occupant_vector, fit_vector)

    x_range = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
    y_range = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 100)

    xx, yy = np.meshgrid(x_range, y_range)
    zz = xx + yy

    for i in range(len(x_range)):
        for j in range(len(y_range)):
            zz[j, i] = gaussian(yy[j, i], *sensor_model(xx[j, i]))

    pal = sns.light_palette("green", as_cmap=True)

    im = plt.imshow(zz,  interpolation='bilinear', origin='lower',
                    cmap=pal, alpha=0.5, aspect='auto',
                    extent=(x_range[0], x_range[-1], y_range[0], y_range[-1]))
    cb = plt.colorbar(im, orientation='vertical', ticks=[
        gaussian(3 * sigma, 0, sigma),
        gaussian(2 * sigma, 0, sigma),
        gaussian(sigma, 0, sigma),
        gaussian(0, 0, sigma),
    ],
        drawedges=True
    )
    cb.set_ticklabels(['$3 \sigma$', '$2 \sigma$',
                       '$\sigma$', 'max'], update_ticks=True)
    plt.savefig(filename)


def plot_predictor(reading_range, predictor, filename, round_level=1, readings=[]):

    palette = sns.color_palette()

    reading_vector = np.linspace(*reading_range)
    fit_vector = np.array([predictor(r)[0] for r in reading_vector])
    _, sigma = predictor(reading_vector[0])
    # error_vector = [sigma] * len(reading_vector)

    plt.clf()
    ax = plt.gca()
    ax.set_xlim([reading_vector[0], reading_vector[-1]])
    ax.set_ylim([round_down(fit_vector[0], round_level),
                 round_up(fit_vector[-1], round_level)])

    ax.plot(reading_vector, fit_vector, color=palette[1])

    x_range = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
    y_range = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 100)

    xx, yy = np.meshgrid(x_range, y_range)
    zz = xx + yy

    for i in range(len(x_range)):
        for j in range(len(y_range)):
            zz[j, i] = gaussian(yy[j, i], *predictor(xx[j, i]))

    if readings:
        ax.vlines(readings, ax.get_ylim()[
                  0], ax.get_ylim()[-1], linestyles='dotted')

    pal = sns.light_palette("green", as_cmap=True)

    im = plt.imshow(zz,  interpolation='bilinear', origin='lower',
                    cmap=pal, alpha=0.5, aspect='auto',
                    extent=(x_range[0], x_range[-1], y_range[0], y_range[-1]))
    cb = plt.colorbar(im, orientation='vertical', ticks=[
        gaussian(3 * sigma, 0, sigma),
        gaussian(2 * sigma, 0, sigma),
        gaussian(sigma, 0, sigma),
        gaussian(0, 0, sigma),
    ],
        drawedges=True
    )
    cb.set_ticklabels(['$3 \sigma$', '$2 \sigma$',
                       '$\sigma$', 'max'], update_ticks=True)
    plt.savefig(filename)


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

if __name__ == "__main__":

    train_car = TrainCar()
    co2_sensor = train_car.sensor_array[0]
    # experiment_data = 
    train_car.run_experiment(datapoints=250)
    # co2_data = [(o, r["co2"]) for o, r in experiment_data]

    # co2_data = generate(350, 60, 10, 15, 5, 250)
    # co2_sensor_model, co2_predictor = fit(co2_data)
    # plot_sensor_model(co2_data, co2_sensor.model,
    #                   'co2_experiment.png', round_level=500)
    # plot_predictor([0, 2500], co2_sensor.predictor,
    #                'co2_predictor.png', round_level=10)
    # plot_predictor([0, 2500], co2_sensor.predictor,
    #                'co2_predictor1.png', round_level=10, readings=[733])
    # plot_predictor([0, 2500], co2_sensor.predictor, 'co2_predictor2.png',
    #                round_level=10, readings=[733, 1037])
    # plot_predictor([0, 2500], co2_sensor.predictor, 'co2_predictor3.png',
    #                round_level=10, readings=[733, 790, 1037, 500, 699])

    # plot_readings([733], co2_sensor.predictor, 'co2_readings1.png', x_range=[0, train_car.max_occupants])
    # plot_readings([733, 1037], co2_sensor.predictor,
    #               'co2_readings2a.png', x_range=[0, train_car.max_occupants], fuse=False)
    # plot_readings([733, 1037], co2_sensor.predictor,
    #               'co2_readings2b.png', x_range=[0, train_car.max_occupants], fuse=True)
    # plot_readings([733, 790, 1037, 500, 699], co2_sensor.predictor,
    #               'co2_readings3.png', x_range=[0, train_car.max_occupants], fuse=True)

    # temp_data = generate(19, 0.6, 0.5, 15, 5, 250)
    # temp_sensor_model, temp_predictor = fit(temp_data)
    # plot_sensor_model(temp_data, temp_sensor_model,
    #                   'temp_experiment.png', round_level=5)
    # plot_predictor([10, 40], temp_predictor,
    #                'temp_predictor.png', round_level=10)
