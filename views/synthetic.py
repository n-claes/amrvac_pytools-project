import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp

from reading import datfile_utilities
from processing import process_data
from physics import ionisation

class _syntheticmain():
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        if self.dataset.header['ndim'] == 1:
            print("synthetic views can only be created for 2D/3D data")
            sys.exit(1)

        # initialise figure and axis
        fig = kwargs.get("fig", None)
        ax = kwargs.get("ax", None)
        if fig is None or ax is None:
            self.fig, self.ax = plt.subplots(1)
        else:
            self.fig = fig
            self.ax = ax

        # initialise variables
        self.line_of_sight = kwargs.get("line_of_sight", "x")
        self.altitude = kwargs.get("altitude", 20000)
        self.simulation_type = kwargs.get("simulation_type", "prominence")
        self.f23 = 0.6407  # oscillator strength of H-alpha line
        ionisation.init_splines(self.altitude)
        self.integrated_block_list = []


    def _get_ne(self, block, block_fields, block_ion):
        block_p = block[..., block_fields.index("p")] * self.dataset.units.unit_pressure
        block_T = block[..., block_fields.index("T")] * self.dataset.units.unit_temperature
        block_ne = block_p / ((1 + 1.1 / block_ion) * self.dataset.units.k_B * block_T)
        return block_ne

    def _integrate_block(self, block, l_edge, r_edge):
        """
        Integrates a given block along the given line of sight.
        :param block: the block to integrate
        :param l_edge: contains the left edge of the block, as a ndim length list [x0(, y0, z0)]
        :param r_edge: contains the right edge of the block, as an ndim length list [x1(, y1, z1)]
        :return: 2D numpy array, containing the integrated block. If the block is originally 2D the block itself
                 is returned.
        """
        if self.dataset.header["ndim"] == 2:
            return block

        block_nx = self.dataset.header['block_nx']
        if self.line_of_sight == 'x':
            x = np.linspace(l_edge[0], r_edge[0], block_nx[0])
            result = np.zeros_like(block[0, :, :])
            for i, j in np.ndindex(result.shape):
                col = block[:, i, j]
                integrated_col = np.trapz(col, x)
                result[i, j] = integrated_col
            return result
        elif self.line_of_sight == 'y':
            y = np.linspace(l_edge[1], r_edge[1], block_nx[1])
            result = np.zeros_like(block[:, 0, :])
            for i, j in np.ndindex(result.shape):
                col = block[i, :, j]
                integrated_col = np.trapz(col, y)
                result[i, j] = integrated_col
            return result
        else:
            z = np.linspace(l_edge[2], r_edge[2], block_nx[2])
            result = np.zeros_like(block[:, :, 0])
            for i, j in np.ndindex(result.shape):
                col = block[i, j, :]
                integrated_col = np.trapz(col, z)
                result[i, j] = integrated_col
            return result

    def _regrid_2dblock(self, ileaf, block, max_lvl):
        block_lvl = self.dataset.block_lvls[ileaf]
        if block_lvl == max_lvl:
            return block
        block_nx = self.dataset.header['block_nx']
        if self.dataset.header['ndim'] == 3:
            # remove corresponding index from block_nx if dataset is 3D
            if self.line_of_sight == 'x':
                block_nx = np.asarray([block_nx[1], block_nx[2]])
            elif self.line_of_sight == 'y':
                block_nx = np.asarray([block_nx[0], block_nx[2]])
            else:
                block_nx = np.asarray(block_nx[:-1])

        regrid_width = block_nx * 2**(max_lvl - block_lvl)
        nb_elements = np.prod(block_nx)

        vals = np.reshape(block, nb_elements)
        pts = np.array([[i, j] for i in np.linspace(0, 1, block_nx[0])
                               for j in np.linspace(0, 1, block_nx[1])])
        grid_x, grid_y = np.mgrid[0:1:regrid_width[0]*1j,
                                  0:1:regrid_width[1]*1j]
        block_regridded = interp.griddata(pts, vals, (grid_x, grid_y), method='linear')
        return block_regridded

    def _merge_integrated_blocks(self):
        self.integrated_block_list = np.asarray(self.integrated_block_list)
        if self.dataset.header['ndim'] == 2:
            return

        print(self.dataset.block_ixs)

        # for b in self.integrated_block_list:
        #     print(b.shape)

        max_lvl = np.max(self.dataset.block_lvls)
        # dx0 = self.dataset.domain_width / self.dataset.header["domain_nx"]
        # min_dx = dx0 * 0.5 ** (max_lvl - 1)






class h_alpha(_syntheticmain):
    def __init__(self, dataset, **kwargs):
        print(">> Creating H-alpha view...")
        super().__init__(dataset, **kwargs)

        max_blocklvl = np.max(self.dataset.block_lvls)
        for ileaf, offset in enumerate(self.dataset.block_offsets):
            block = datfile_utilities.get_single_block_data(self.dataset.file, offset, self.dataset.block_shape)
            # this adds the temperature and pressure to the block
            block, block_fields = process_data.add_primitives_to_single_block(block, self.dataset)
            # interpolate ionisation and f parameter for each block
            block_ion, block_fpar = ionisation.block_interpolate_ionisation_f(block, block_fields, self.dataset)
            block_ne = super()._get_ne(block, block_fields, block_ion)
            n2 = block_ne**2 / (block_fpar * 1e16)              # parameter f is interpolated in units of 1e16 cm-3
            # calculate block opacity
            block_kappa = (np.pi * self.dataset.units.ec**2 / (self.dataset.units.m_e * self.dataset.units.c)) * \
                            self.f23 * n2 * self._gaussian(block, block_fields)
            # integrate block along line of sight to get opacity
            l_edge, r_edge = process_data.get_block_edges(ileaf, self.dataset)
            opacity = super()._integrate_block(block_kappa, l_edge, r_edge)

            S = self._source_function()
            intensity = S * (1 - np.exp(-opacity))
            if self.simulation_type == 'filament':
                Ibgr = 2.2 * S
                intensity += Ibgr * np.exp(-opacity)
            # if integrated 2D matrix is not at max_blocklvl, regrid it
            intensity = super()._regrid_2dblock(ileaf, intensity, max_blocklvl)
            self.integrated_block_list.append(intensity)

        # merge all integrated blocks into one single 2D array
        super()._merge_integrated_blocks()



    def _gaussian(self, block, block_fields):
        block_T = block[..., block_fields.index("T")] * self.dataset.units.unit_temperature
        ksi = 5 * 1e5  # microturbulence in cm/s
        nu_0 = self.dataset.units.c / (6562.8 * 1e-8)       # H-alpha wavelength is 6562.8 Angstrom
        delta_nu = 0
        delta_nuD = (nu_0 / self.dataset.units.c) * \
                    np.sqrt(2 * self.dataset.units.k_B * block_T / self.dataset.units.m_p + ksi ** 2)
        phi_nu = 1.0 / (np.sqrt(np.pi) * delta_nuD) * np.exp(-delta_nu / delta_nuD) ** 2
        return phi_nu

    def _source_function(self):
        H = self.altitude * 1e5     # altitude in cm
        # dilution factor W
        W = 0.5 * ( 1 - np.sqrt(1 - (self.dataset.units.Rsun**2 / (self.dataset.units.Rsun + H)**2)) )
        return W * 0.17 * 4.077 * 1e-5




class faraday(_syntheticmain):
    def __init__(self, dataset, **kwargs):
        print(">> Creating Faraday view...")
        super().__init__(dataset, **kwargs)