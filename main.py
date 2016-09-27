import math
import random

import matplotlib as mpl
import numpy as np
import scipy
import seaborn as sns
import traces
from matplotlib import pyplot as plt
import matplotlib.animation as animation

# local imports
from kalman import Reading, Estimate
from traincar import TrainCar
from sensor import Sensor


if __name__ == "__main__":

    # create some crappy sensors
    co2_sensor = Sensor("co2", intersect=350, slope=15, sigma=10, round_level=500, proc_sigma=30)
    temp_sensor = Sensor("temp", intersect=0, slope=0.25, sigma=5, round_level=10, proc_sigma=5)

    # put the sensors on a train car
    train_car = TrainCar(sensor_array=[co2_sensor, temp_sensor])

    # run some experiments to model/calibrate the sensors
    train_car.run_experiment(datapoints=250)
    train_car.plot_experiment("experiment_plots")

    # generate some "real" occupancy data
    train_car.generate_occupancy()  # defaults to 5 stations and 30 minutes

    time_array = np.arange(-1, 31, 1.0 / 60)
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
    line, = ax.plot([], [], lw=2)
    estimate = Estimate(ax)
    reading = Reading(co2_sensor, 55)
    estimate.add_reading(reading)
    # estimate.plot(ax)
    # reading.plot(ax)

    # plt.savefig("test.png")

    animation.FuncAnimation(
        plt.gcf(), estimte.update_plot,
        frames=200, init_func=estimate.init_plot,
        blit=True, fargs=[train_car])

def init():
    line.set_data([], [])
    return line,

def animate(i):
    x = np.linspace(0, 2, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return line,
