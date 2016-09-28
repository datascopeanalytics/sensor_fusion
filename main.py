import math
import random
from collections import defaultdict

import matplotlib as mpl
import matplotlib.animation as animation
import numpy as np
import scipy
import seaborn as sns
import traces
# local imports
from kalman import Estimate, Reading
from matplotlib import pyplot as plt
from sensor import Sensor
from traincar import TrainCar


class SensorAnimation(animation.FuncAnimation):
    def __init__(self, truth, reading_array):
        self.fig, self.ax = plt.subplots()

        self.truth = truth
        self.reading_array = reading_array

        for sensor in reading_array:
            self.line = self.ax.plot([], [], color=sensor.color, label=sensor.name)

        super().__init__(
            self.fig, self.update,
            frames=200, init_function=self.init, blit=True
        )

    def init():
        """Initializes the animation"""
        self.lines = []
        for sensor_name in self.reading_array:
            self.lines.append()
            self.line.set_data([], [])
        return line,

    def update(i, truth, reading_array):
        """updates frame i of the animation"""
        x = np.linspace(0, 2, 1000)
        y = np.sin(2 * np.pi * (x - 0.01 * i))
        line.set_data(x, y)
        return line,


if __name__ == "__main__":

    # create some crappy sensors
    co2_sensor = Sensor("co2", intersect=350, slope=15,
                        sigma=10, round_level=500, proc_sigma=30)
    temp_sensor = Sensor("temp", intersect=0, slope=0.25,
                         sigma=5, round_level=10, proc_sigma=5)

    # put the sensors on a train car
    train_car = TrainCar(sensor_array=[co2_sensor, temp_sensor])

    # run some experiments to model/calibrate the sensors
    train_car.run_experiment(datapoints=250)
    train_car.plot_experiment(path="experiment_plots")

    # generate some "real" occupancy data
    train_car.generate_occupancy()  # defaults to 5 stations and 30 minutes

    time_array = np.arange(-1, 31, 1.0 / 60)
    reading_array = defaultdict(list)
    truth = []
    for t in time_array:
        for reading in train_car.read_sensors(timestamp=t):
            reading_array[reading.sensor].append(reading)
        truth.append(train_car.occupants_trace[t])

    plt.clf()
    plt.plot(time_array, truth)
    plt.savefig("truth.png")

    plt.clf()
    plt.plot(time_array, reading_array[co2_sensor])
    plt.savefig("co2.png")

    plt.clf()



#     plt.clf()
#     ax = plt.gca()
#     ax.set_xlim(0, train_car.max_occupants)
#     line, = ax.plot([], [], lw=2)
#     estimate = Estimate(ax)
#     reading = Reading(co2_sensor, 55)
#     estimate.add_reading(reading)
#     # estimate.plot(ax)
#     # reading.plot(ax)
#
#     # plt.savefig("test.png")
#

#
#
