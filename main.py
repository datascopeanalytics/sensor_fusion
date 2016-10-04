import copy
import datetime
import math
import random
from collections import defaultdict

import matplotlib as mpl
import matplotlib.animation
from matplotlib.ticker import FormatStrFormatter

import numpy as np
import scipy
import seaborn as sns
import traces

# local imports
from kalman import Estimate, Reading
from matplotlib import pyplot as plt
from sensor import Sensor
from traincar import TrainCar


class SensorAnimation(matplotlib.animation.FuncAnimation):

    def __init__(self, time_array, truth, reading_array, estimate_array):

        self.fig, (self.ax2, self.ax1) = plt.subplots(
            1, 2, sharey=True,
            gridspec_kw={"width_ratios":[3, 1]},
            figsize=(8, 4)
        )
        plt.tight_layout(pad=2.0)

        self.time_array = time_array
        self.estimate_array = estimate_array

        self.ax1.set_ylim(0, 120)
        self.ax1.set_xlim(0, 20)
        self.ax1.set_xlabel("Probability")
        self.ax1.xaxis.set_major_formatter(FormatStrFormatter('%d%%'))

        self.estimate_line = self.ax1.plot(
            [], [], color='purple', label='estimate')
        self.lines = []
        for sensor in reading_array:
            self.lines += self.ax1.plot(
                [], [], color=sensor.color, label=sensor.name)

        self.truth_line = self.ax1.hlines(truth[0], 0, 20, color='red', label='Occupancy')
        self.ax1.legend()

        self.ax2.plot(time_array, truth, color='red', label='Occupancy')
        # self.ax2.set_ylim(0, 150)
        self.ax2.set_title("Train car occupancy over time")
        self.ax2.set_xlabel("Time (minutes)")
        self.ax2.set_ylabel("Occupants")
        self.estimate_ts = self.ax2.plot(
            [], [], color='purple', label='estimate')
        self.fill_lines = self.ax2.fill_between(
            [], [], color='purple', alpha=0.5)

        self.truth = truth
        self.reading_array = reading_array

        super().__init__(
            self.fig, self.update,
            frames=len(time_array),
            blit=True
        )

    def update(self, i):
        """updates frame i of the animation"""
        self.ax1.set_title("{}".format(
            datetime.timedelta(minutes=self.time_array[i]))
        )
        for sensor, line in zip(self.reading_array.keys(), self.lines):
            reading = self.reading_array.get(sensor)[i]
            x, y = reading.vectorize(self.ax1.get_ylim())
            line.set_data(y, x)
        estimate = self.estimate_array[i]
        self.estimate_line[0].set_data(
            estimate.vectorize(self.ax1.get_ylim())[1],
            estimate.vectorize(self.ax1.get_ylim())[0],
            )

        self.truth_line.remove()
        self.truth_line = self.ax1.hlines(truth[i], 0, 20, color='red', label='Occupancy')

        self.estimate_ts[0].set_data(
            self.time_array[:i], self.estimate_array[:i])
        self.fill_lines.remove()
        self.fill_lines = self.ax2.fill_between(
            self.time_array[:i],
            [e.mu - 2 * e.sigma for e in self.estimate_array[:i]],
            [e.mu + 2 * e.sigma for e in self.estimate_array[:i]],
            color='purple',
            alpha=0.5
        )

        return tuple(self.lines + self.estimate_line + self.estimate_ts + [self.fill_lines] + [self.truth_line])


if __name__ == "__main__":

    # create some crappy sensors
    co2_sensor = Sensor("CO$_2$", intersect=350, slope=15,
                        sigma=10, round_level=500, proc_sigma=30, units="ppm")
    # sigma=500, round_level=500, proc_sigma=0)
    temp_sensor = Sensor("Temperature", intersect=0, slope=0.25,
                         sigma=5, round_level=10, proc_sigma=5, units="$^{\circ}$C")

    # put the sensors on a train car
    train_car = TrainCar(sensor_array=[co2_sensor, temp_sensor])

    # run some experiments to model/calibrate the sensors
    train_car.run_experiment(datapoints=250)
    train_car.plot_experiment(path="experiment_plots")

    # generate some "real" occupancy data
    train_car.generate_occupancy()  # defaults to 5 stations and 30 minutes

    time_array = np.arange(0, 30, 1.0 / 10)
    reading_array = defaultdict(list)
    truth = []
    estimate_array = []
    estimate = Estimate()
    for t in time_array:
        for reading in train_car.read_sensors(t):
            reading_array[reading.sensor].append(reading)
            estimate.add_reading(reading)
        estimate_array.append(copy.deepcopy(estimate))
        # if the last point was in a station
        if truth and train_car.occupants_trace[t] != truth[-1]:
            estimate = Estimate()
        truth.append(train_car.occupants_trace[t])

    # plt.clf()
    # plt.plot(time_array, reading_array[co2_sensor])
    # plt.savefig("co2.png")

    plt.clf()

    animation = SensorAnimation(
        time_array, truth, reading_array, estimate_array
    )
    animation.save("30minutes.mp4", fps=10, bitrate=1024)

    plt.clf()
    plt.xlabel("Number of people in the train car")
    plt.ylabel("Probability")
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f%%'))

    reading_1 = Reading(co2_sensor, 60)
    print("reading_1 = ", (reading_1.value, reading_1.mu))
    plt.plot(*reading_1.vectorize((0,120)), color=co2_sensor.color, label="CO$_2$ sensor")
    plt.vlines(reading_1, 0, max(reading_1.vectorize((0,120))[1]), linestyles='dashed')
    plt.legend()
    plt.savefig("reading_plots/1_co2.svg")

    reading_2 = Reading(co2_sensor, 60)
    print("reading_2 = ", (reading_2.value, reading_2.mu))
    plt.plot(*reading_2.vectorize((0,120)), color=co2_sensor.color)
    plt.vlines(reading_2, 0, max(reading_2.vectorize((0,120))[1]), linestyles='dashed')
    plt.savefig("reading_plots/2_co2.svg")

    estimate = Estimate()
    estimate.add_reading(reading_1)
    estimate.add_reading(reading_2)
    estimate_line = plt.plot(*estimate.vectorize((0,120)), color='purple', label="Estimate")
    plt.legend()
    plt.savefig("reading_plots/3_co2.svg")

    reading_3 = Reading(temp_sensor, 60)
    print("reading_3 = ", (reading_3.value, reading_3.mu))
    plt.plot(*reading_3.vectorize((0,120)), color=temp_sensor.color, label="Temperature sensor")
    plt.vlines(reading_3, 0, max(reading_3.vectorize((0,120))[1]), linestyles='dashed')
    estimate.add_reading(reading_3)
    estimate_line[0].remove()
    estimate_line = plt.plot(*estimate.vectorize((0,120)), color='purple', label="Estimate")
    plt.legend()
    plt.savefig("reading_plots/4_co2.svg")
