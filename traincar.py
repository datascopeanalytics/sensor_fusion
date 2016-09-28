import random

import traces
import seaborn as sns

from sensor import Sensor
from kalman import Reading

class TrainCar(object):
    max_occupants = 120
    occupant_range = range(0, max_occupants + 1)

    def __init__(self, occupants=0, sensor_array=[]):
        self.sigma = 0
        self.occupants = occupants
        self.sensor_array = sensor_array

    def generate_occupancy(self, start=0, end=30, stations=5):
        self.occupants_trace = traces.TimeSeries()
        self.occupants_trace[start] = self.max_occupants / 2
        self.occupants_trace[end] = 0

        # at each station a certain number of people get on or off
        minute = start
        while minute < end:
            minute += random.gauss(float(end-start)/stations, 5)
            current_val = self.occupants_trace[minute]
            new_val = max(0, int(random.gauss(current_val, 20)))
            self.occupants_trace[minute] = new_val

        return self.occupants_trace

    def read_sensors(self, experiment=True, timestamp=None):
        for sensor in self.sensor_array:
            if experiment:
                occupants = max(0, random.gauss(self.occupants, self.sigma))
            elif timestamp:
                occupants = self.occupants_trace[timestamp]
            else:
                raise AttributeError("gimme some moneeey!")
            yield Reading(sensor, occupants, timestamp)


    def plot_experiment(self, **kwargs):
        for i, sensor in enumerate(self.sensor_array):
            color = sns.color_palette()[i]
            sensor.color = color
            sensor.plot_experiment(**kwargs)

    def run_experiment(self, datapoints=1000):
        """Generates fake sensor data and trains the sensor"""

        for sensor in self.sensor_array:
            sensor_data = []
            for _ in range(datapoints):
                self.occupants = random.randrange(self.max_occupants + 1)
                sensor_data.append((self.occupants, sensor.read(self.occupants)))
            sensor.fit(sensor_data)
