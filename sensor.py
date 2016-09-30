import random
import os
import copy

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import linregress


from utils import gaussian


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
        reading = variable * self.slope + self.intersect
        return random.gauss(reading, self.sigma)

    def fit(self, data):
        self.experiment_data = copy.deepcopy(data)
        n_samples = len(data)

        model_slope, model_intercept = np.polyfit(
            [o for o, r in data], [r for o, r in data], 1)

        def model(occupants):
            return occupants * model_slope + model_intercept
        self.model = model

        def predictor(sensor_reading):
            return (sensor_reading-model_intercept)/model_slope
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

    def plot_experiment(self, path=""):
        color = self.color
        data = self.experiment_data
        cmap = sns.light_palette(color, as_cmap=True)

        fig, ax = plt.subplots()
        occupants, readings = (np.array(array) for array in zip(*data))

        # ax_left, im_left = plot_linear_fit(
        # ax_left, occupants, readings, self.model, self.model_sigma, color,
        # cmap)

        ax, im = plot_linear_fit(
            ax, readings, occupants,
            self.predictor, self.predictor_sigma,
            color, cmap
        )

        ax.set_xlabel("{} sensor readout ({})".format(self.name, self.units))
        ax.set_ylabel("Number of train car occupants")
        
        # cax, kw = mpl.colorbar.make_axes(
        # [ax_left, ax_right], location="bottom"
        # )

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

        fig.savefig(os.path.join(path, self.name+".png"))
