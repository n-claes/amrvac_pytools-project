"""
Class to process the MPI-AMRVAC raw data into a useful Python dictionary.

@author: Niels Claes
"""

import sys
import numpy as np
import copy
from processing import convert
from collections import OrderedDict


def add_primitives_to_single_block(block, dataset):
    """
    This function is used by the synthetic views to calculate pressure and temperature on-the-go.
    For now, only add pressure (last-to-one index) and temperature (last index)
    :param block: data of a single block, numpy array of shape dataset.block_shape
    :param dataset: amrvac_reader instance, containing the dataset info
    :return: block with pressure and temperature added
    """
    # calculate pressure
    ndim = dataset.header['ndim']
    phys_type = dataset.header['physics_type']
    block_fields = dataset.block_fields
    gamma = dataset.header['gamma']
    rho_idx = dataset.header['w_names'].index('rho')
    if ndim == 2:
        if phys_type == 'hd':
            v1, v2, p = convert.hd_to_primitive_2d(*(block[:, :, idx] for idx in range(0, len(block_fields))), gamma)
        else:
            v1, v2, p = convert.mhd_to_conserved_2d(*(block[:, :, idx] for idx in range(0, len(block_fields))), gamma)
        temp = p / block[:, :, rho_idx]
    else:
        if phys_type == 'hd':
            v1, v2, v3, p = convert.hd_to_primitive_3d(*(block[:, :, :, idx] for idx in range(0, len(block_fields))), gamma)
        else:
            v1, v2, v3, p = convert.mhd_to_primitive_3d(*(block[:, :, :, idx] for idx in range(0, len(block_fields))), gamma)
        temp = p / block[:, :, :, rho_idx]
    # add pressure and temperature to block fields, for consistent retrieval of index later on
    dataset.block_fields += ['p', 'T']
    block = np.concatenate((block, p[..., np.newaxis]), axis=ndim)      # add pressure
    block = np.concatenate((block, temp[..., np.newaxis]), axis=ndim)   # add temperature
    return block




class create_amrvac_dict():
    def __init__(self, raw_data, header):
        self._header  = header
        self._standard_fields = copy.deepcopy(header["w_names"])
        self.primitive_vars = None
        self._ndim    = header["ndim"]

        self.data = OrderedDict()

        for var in self._standard_fields:
            idx = self._standard_fields.index(var)
            if self._ndim == 1:
                self.data[var] = raw_data[:, idx]
            elif self._ndim == 2:
                self.data[var] = raw_data[:, :, idx]
            elif self._ndim == 3:
                self.data[var] = raw_data[:, :, :, idx]
            else:
                print("Something wrong with the ndim parameter in the header?")
                print("Current value ndim = {}".format(self._ndim))
                sys.exit(1)


    def add_primitives(self):
        phys_type = self._header["physics_type"]
        gamma = self._header["gamma"]

        if self._ndim == 1:
            if phys_type == "hd":
                v1, p = convert.hd_to_primitive_1d(*(self.data[var] for var in self._standard_fields), gamma=gamma)
            else:
                v1, p = convert.mhd_to_primitive_1d(*(self.data[var] for var in self._standard_fields), gamma=gamma)
            self.data.update({'v1': v1, 'p': p})
            self.primitive_vars = ['v1', 'p']
        elif self._ndim == 2:
            if phys_type == "hd":
                v1, v2, p = convert.hd_to_primitive_2d(*(self.data[var] for var in self._standard_fields), gamma=gamma)
            else:
                v1, v2, p = convert.mhd_to_primitive_2d(*(self.data[var] for var in self._standard_fields), gamma=gamma)
            self.data.update({'v1': v1, 'v2': v2, 'p': p})
            self.primitive_vars = ['v1', 'v2', 'p']
        else:
            if phys_type == "hd":
                v1, v2, v3, p = convert.hd_to_primitive_3d(*(self.data[var] for var in self._standard_fields), gamma=gamma)
            else:
                v1, v2, v3, p = convert.mhd_to_primitive_3d(*(self.data[var] for var in self._standard_fields), gamma=gamma)
            self.data.update({'v1': v1, 'v2': v2, 'v3': v3, 'p': p})
            self.primitive_vars = ['v1', 'v2', 'v3', 'p']