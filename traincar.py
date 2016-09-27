import random

import traces

from sensor import Sensor


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
            1, self.max_occupants / 2)
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
