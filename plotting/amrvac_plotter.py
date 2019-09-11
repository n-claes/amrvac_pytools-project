import numpy as np
import matplotlib.pyplot as plt
import collections

class _plotsetup():
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset

        self.cmap = kwargs.get("cmap", "jet")

        # initialise figure and axis
        fig = kwargs.get("fig", None)
        if fig is None:
            self.fig = plt.figure()
        else:
            self.fig = fig
        ax = kwargs.get("ax", None)
        if ax is None:
            self.ax = self.fig.gca()
        else:
            self.ax = ax

        # initialise coordinate axis
        self.x_axis = kwargs.get("x_axis", None)
        self.y_axis = kwargs.get("y_axis", None)
        self.z_axis = kwargs.get("z_axis", None)

class amrplot(_plotsetup):
    def __init__(self, dataset, var, **kwargs):
        super().__init__(dataset, **kwargs)
        self.var = var

    def plot_1d(self):
        pass

    def plot_2d(self):
        pass


class rgplot(_plotsetup):
    def __init__(self, dataset, data, **kwargs):
        if dataset.data_dict is None:
            raise AttributeError("Make sure the regridded data is loaded when calling this class (ds.load_all_data)")
        if not isinstance(data, np.ndarray):
            raise Exception("Attribute 'data' passed should be a numpy array containing the data")
        super().__init__(dataset, **kwargs)

        self.data = data
        self.synthetic_view = kwargs.get("synthetic_view", False)

        if len(self.data.shape) == 1:
            self.plot_1d()
        elif len(self.data.shape) == 2:
            self.plot_2d()
        else:
            raise NotImplementedError("Plotting in 3 dimensions is not supported")


    def plot_1d(self):
        x = self.dataset.get_coordinate_arrays()[0]
        self.ax.plot(x, self.data)
        plt.show()

    def plot_2d(self):
        if not self.synthetic_view:
            bounds_x, bounds_y = self.dataset.get_bounds()
            im = self.ax.imshow(np.rot90(self.data), extent=[*bounds_x, *bounds_y])
            self.fig.colorbar(im)
            plt.show()
        else:
            pass


