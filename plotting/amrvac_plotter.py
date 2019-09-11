import numpy as np
import matplotlib.pyplot as plt
import collections

from reading import dat_reader

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

        if self.dataset.header["ndim"] == 1:
            self.plot_1d()
        elif self.dataset.header["ndim"] == 2:
            self.plot_2d()
        else:
            raise NotImplementedError("Plotting in 3D is not supported")

    def plot_1d(self):
        domain_width = self.dataset.header["xmax"] - self.dataset.header["xmin"]
        block_nx = self.dataset.header["block_nx"]
        # dx at coarsest grid level
        dx0 = domain_width / self.dataset.header["domain_nx"]

        try:
            var_idx = self.dataset.header['w_names'].index(self.var)
        except ValueError:
            raise NotImplementedError("Implement plotting of other variables than the ones in the dat file!")


        for ileaf, (lvl, morton_idx) in enumerate(zip(self.dataset.block_lvls, self.dataset.block_ixs)):
            # dx at certain lvl
            dx = dx0 * 0.5**(lvl-1)
            l_edge = self.dataset.header["xmin"] + (morton_idx - 1) * block_nx * dx
            r_edge = l_edge + block_nx * dx

            offset = self.dataset.block_offsets[ileaf]
            block = dat_reader.get_single_block_data(self.dataset.file, offset, self.dataset.block_shape)
            block_data = block[:, var_idx]
            x = np.linspace(l_edge, r_edge, self.dataset.header["block_nx"])
            self.ax.plot(x, block_data, '-k')

    def plot_2d(self):
        raise NotImplementedError


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
            raise NotImplementedError("Plotting in 3D is not supported")


    def plot_1d(self):
        x = self.dataset.get_coordinate_arrays()[0]
        self.ax.plot(x, self.data, '-k')

    def plot_2d(self):
        if not self.synthetic_view:
            bounds_x, bounds_y = self.dataset.get_bounds()
            im = self.ax.imshow(np.rot90(self.data), extent=[*bounds_x, *bounds_y], cmap=self.cmap)
            self.fig.colorbar(im)
        else:
            raise NotImplementedError


