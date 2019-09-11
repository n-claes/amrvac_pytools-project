import numpy as np
import matplotlib.pyplot as plt

class plot():
    def __init__(self, dataset, var, **kwargs):
        self.dataset = dataset
        self.var = var

        self.cmap = kwargs.get("cmap", "jet")

        fig = kwargs.get("fig", None)
        # initialise figure and axis
        if fig is None:
            self.fig = plt.figure()
        else:
            self.fig = fig
        ax = kwargs.get("ax", None)
        if ax is None:
            self.ax = self.fig.gca()
        else:
            self.ax = ax

        self.title = kwargs.get("title", "plot: {}".format(var))

        # one dimensional dataset
        if self.dataset.header["ndim"] == 1:
            self.plot_1d()
        elif self.dataset.header["ndim"] == 2:
            self.plot_2d()
        else:
            raise NotImplementedError("3D plotting is not supported.")

    def plot_1d(self):
        pass

    def plot_2d(self):
        pass






