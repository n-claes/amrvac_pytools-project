"""
Class to process the MPI-AMRVAC raw data into a useful Python dictionary.

@author: Niels Claes
"""

import sys
from processing import convert

from collections import OrderedDict
import copy

class create_amrvac_dict():
    def __init__(self, raw_data, header):
        self._header  = header
        self._w_names = header["w_names"]
        self._ndim    = header["ndim"]

        self.data = OrderedDict()

        for var in self._w_names:
            idx = self._w_names.index(var)
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


    def convert_to_primitive(self):
        w_names_cons = copy.deepcopy(self._w_names)
        phys_type    = self._header["physics_type"]
        gamma        = self._header["gamma"]

        if self._ndim == 1:
            if phys_type == "hd":
                v1, p = convert.hd_to_primitive_1d(*(self.data[var] for var in self._w_names), gamma=gamma)
            else:
                v1, p = convert.mhd_to_primitive_1d(*(self.data[var] for var in self._w_names), gamma=gamma)
            prim_data = OrderedDict({"v1": v1, "p": p})
            self._replace_wname(["m1", "e"], ["v1", "p"])
        elif self._ndim == 2:
            if phys_type == "hd":
                v1, v2, p = convert.hd_to_primitive_2d(*(self.data[var] for var in self._w_names), gamma=gamma)
            else:
                v1, v2, p = convert.mhd_to_primitive_2d(*(self.data[var] for var in self._w_names), gamma=gamma)
            prim_data = OrderedDict({"v1": v1, "v2": v2, "p": p})
            self._replace_wname(["m1", "m2", "e"], ["v1", "v2", "p"])
        else:
            if phys_type == "hd":
                v1, v2, v3, p = convert.hd_to_primitive_3d(*(self.data[var] for var in self._w_names), gamma=gamma)
            else:
                v1, v2, v3, p = convert.mhd_to_primitive_3d(*(self.data[var] for var in self._w_names), gamma=gamma)
            prim_data = OrderedDict({"v1": v1, "v2": v2, "v3": v3, "p": p})
            self._replace_wname(["m1", "m2", "m3", "e"], ["v1", "v2", "v3", "p"])

        self._update_dict(w_names_cons, prim_data)


    def convert_to_conservative(self):
        w_names_prim = copy.deepcopy(self._w_names)
        phys_type    = self._header["physics_type"]
        gamma        = self._header["gamma"]

        if self._ndim == 1:
            if phys_type == "hd":
                m1, e = convert.hd_to_conserved_1d(*(self.data[var] for var in self._w_names), gamma=gamma)
            else:
                m1, e = convert.mhd_to_conserved_1d(*(self.data[var] for var in self._w_names), gamma=gamma)
            cons_data = OrderedDict({"m1": m1, "e": e})
            self._replace_wname(["v1", "p"], ["m1", "e"])
        elif self._ndim == 2:
            if phys_type == "hd":
                m1, m2, e = convert.hd_to_conserved_2d(*(self.data[var] for var in self._w_names), gamma=gamma)
            else:
                m1, m2, e = convert.mhd_to_conserved_2d(*(self.data[var] for var in self._w_names), gamma=gamma)
            cons_data = OrderedDict({"m1": m1, "m2": m2, "e": e})
            self._replace_wname(["v1", "v2", "p"], ["m1", "m2", "e"])
        else:
            if phys_type == "hd":
                m1, m2, m3, e = convert.hd_to_conserved_3d(*(self.data[var] for var in self._w_names), gamma=gamma)
            else:
                m1, m2, m3, e = convert.mhd_to_conserved_3d(*(self.data[var] for var in self._w_names), gamma=gamma)
            cons_data = OrderedDict({"m1": m1, "m2": m2, "m3": m3, "e": e})
            self._replace_wname(["v1", "v2", "v3", "p"], ["m1", "m2", "m3", "p"])

        self._update_dict(w_names_prim, cons_data)


    def _replace_wname(self, original, replacement):
        for i in range(len(original)):
            idx = self._w_names.index(original[i])
            self._w_names[idx] = replacement[i]
        return


    def _update_dict(self, w_names_old, new_data):
        updated_data = OrderedDict()

        for key, value in self.data.items():
            key_idx = tuple(self.data).index(key)
            # do not update unchanged keys
            if w_names_old[key_idx] == self._w_names[key_idx]:
                updated_data[key] = self.data[key]
                continue

            # w_names has already been changed to new values
            new_key = self._w_names[key_idx]
            updated_data[new_key] = new_data[new_key]

        self.data = copy.deepcopy(updated_data)
        del updated_data
        return
