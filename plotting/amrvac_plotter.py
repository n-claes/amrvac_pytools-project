import numpy as np
import matplotlib.pyplot as plt
import collections

from reading import datfile_utilities

class _plotsetup():
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset

        self.cmap = kwargs.get("cmap", "jet")

        # initialise figure and axis
        fig = kwargs.get("fig", None)
        ax = kwargs.get("ax", None)
        if fig is None or ax is None:
            self.fig, self.ax = plt.subplots(1, sharex="all", sharey="all")
        else:
            self.fig = fig
            self.ax = ax

class amrplot(_plotsetup):
    def __init__(self, dataset, var, **kwargs):
        super().__init__(dataset, **kwargs)
        self.var = var

        self.domain_width = self.dataset.header["xmax"] - self.dataset.header["xmin"]
        self.block_nx = self.dataset.header["block_nx"]
        # dx at coarsest grid level
        self.dx0 = self.domain_width / self.dataset.header["domain_nx"]

        # mesh-related parameters
        self.draw_mesh = kwargs.get("draw_mesh", False)
        if not isinstance(self.draw_mesh, bool):
            raise ValueError("'draw_mesh' argument should be True or False")
        self.mesh_color = kwargs.get("mesh_color", "black")
        self.mesh_linestyle = kwargs.get("mesh_linestyle", "solid")
        self.mesh_linewidth = kwargs.get("mesh_linewidth", 2)
        self.mesh_opacity = kwargs.get("mesh_opacity", 1)

        if self.dataset.header["ndim"] == 1:
            self.plot_1d()
        elif self.dataset.header["ndim"] == 2:
            self.plot_2d()
        else:
            raise NotImplementedError("Plotting in 3D is not supported")

    def plot_1d(self):
        try:
            var_idx = self.dataset.header['w_names'].index(self.var)
        except ValueError:
            raise NotImplementedError("Implement plotting of other variables than the ones in the dat file!")

        for ileaf, (lvl, morton_idx) in enumerate(zip(self.dataset.block_lvls, self.dataset.block_ixs)):
            # dx at current block level
            dx = self.dx0 * 0.5**(lvl-1)
            # calculate actual edges of the block
            l_edge = self.dataset.header["xmin"] + (morton_idx - 1) * self.block_nx * dx
            r_edge = l_edge + self.block_nx * dx

            # retrieve block offset in dat file
            offset = self.dataset.block_offsets[ileaf]
            # read in block data (contains all variables)
            block = datfile_utilities.get_single_block_data(self.dataset.file, offset, self.dataset.block_shape)
            # cut block data to contain only the desired variable
            block_data = block[:, var_idx]
            x = np.linspace(l_edge, r_edge, self.block_nx)
            self.ax.plot(x, block_data, '-k')

    def plot_2d(self):
        varmin, varmax = self.dataset.get_extrema(self.var)
        for ileaf, (lvl, morton_idx) in enumerate(zip(self.dataset.block_lvls, self.dataset.block_ixs)):
            dx = self.dx0 * 0.5**(lvl-1)
            l_edge = self.dataset.header["xmin"] + (morton_idx - 1) * self.block_nx * dx
            r_edge = l_edge + self.block_nx * dx
            offset = self.dataset.block_offsets[ileaf]
            block = datfile_utilities.get_single_block_data(self.dataset.file, offset, self.dataset.block_shape)
            block_data = block[:, :, self.dataset.header['w_names'].index(self.var)]
            x = np.linspace(l_edge[0], r_edge[0], self.block_nx[0])
            y = np.linspace(l_edge[1], r_edge[1], self.block_nx[1])
            im = self.ax.pcolormesh(y, x, block_data, cmap=self.cmap, vmin=varmin, vmax=varmax, snap=True)

            if self.draw_mesh:
                if not r_edge[1] == self.dataset.header["xmax"][1]:
                    self.ax.vlines(x=r_edge[1], ymin=l_edge[0], ymax=r_edge[0], color=self.mesh_color,
                                   lw=self.mesh_linewidth, linestyle=self.mesh_linestyle, alpha=self.mesh_opacity)
                if not r_edge[0] == self.dataset.header["xmax"][0]:
                    self.ax.hlines(y=r_edge[0], xmin=l_edge[1], xmax=r_edge[1], color=self.mesh_color,
                                   lw=self.mesh_linewidth, linestyle=self.mesh_linestyle, alpha=self.mesh_opacity)
        self.fig.colorbar(im)


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


