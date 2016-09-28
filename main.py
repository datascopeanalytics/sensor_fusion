import math
import random
from collections import defaultdict
import copy

import matplotlib as mpl
import matplotlib.animation
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
        self.fig, self.ax = plt.subplots()
        self.time_array = time_array
        self.estimate_array = estimate_array

        self.ax.set_xlim(0, 150)
        self.ax.set_ylim(0, 50)

        self.truth = truth
        self.reading_array = reading_array

        self.estimate_line = self.ax.plot([], [], color='purple', label='estimate')
        self.lines = []
        for sensor in reading_array:
            self.lines += self.ax.plot([], [], color=sensor.color, label=sensor.name)

        super().__init__(
            self.fig, self.update,
            frames=len(time_array), init_func=self.init, blit=True
        )

    def init(self):
        """Initializes the animation"""
        self.ax.set_title("t=0s")
        self.estimate_line[0].set_data([], [])

        for line in self.lines:
            line.set_data([], [])

        return tuple(self.lines + self.estimate_line)

    def update(self, i):
        """updates frame i of the animation"""
        self.ax.set_title("t={}s".format(i))

        estimate = self.estimate_array[i]
        self.estimate_line[0].set_data(*estimate.vectorize(self.ax.get_xlim()))

        for sensor, line in zip(self.reading_array.keys(), self.lines):
            reading = self.reading_array.get(sensor)[i]
            x, y = reading.vectorize(self.ax.get_xlim())
            line.set_data(x, y)
        return tuple(self.lines + self.estimate_line)


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

    time_array = np.arange(-1, 30, 1.0 / 30)
    reading_array = defaultdict(list)
    truth = []
    estimate_array = []
    estimate = Estimate()
    for t in time_array:
        for reading in train_car.read_sensors(experiment=False, timestamp=t):
            reading_array[reading.sensor].append(reading)
            estimate.add_reading(reading)
        estimate_array.append(copy.deepcopy(estimate))
        # if the last point was in a station
        if truth and train_car.occupants_trace[t] != truth[-1]:
            estimate = Estimate()
        truth.append(train_car.occupants_trace[t])

    plt.clf()
    plt.plot(time_array, truth)
    plt.savefig("truth.png")

    plt.clf()
    plt.plot(time_array, reading_array[co2_sensor])
    plt.savefig("co2.png")

    plt.clf()

    animation = SensorAnimation(
        time_array, truth, reading_array, estimate_array
    )
    animation.save("test.mp4", fps=30)
