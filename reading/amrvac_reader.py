"""
Class to load in MPI-AMRVAC .dat files.

@author: Niels Claes
         Last edit: 14 August 2019
"""
import sys, os
import numpy as np
from reading import process_data, dat_reader
import matplotlib.pyplot as plt

class load_file:
    def __init__(self, filename):
        try:
            file = open(filename, "rb")
        except IOError:
            print("Unable to open {}".format(filename))
            print("Is this an MPI-AMRVAC .dat file?")
            sys.exit(1)

        self._filename = filename
        # Trim filename, assumes format .../.../filenameXXXX.dat
        for c in range(len(filename) - 1, -1, -1):
            if filename[c] == '/' or filename[c] == '\\':
                self._filename = filename[c+1:]
                break
        print(">> Reading {}".format(filename))

        self._file = file
        self.header = dat_reader.get_header(file)
        self._uniform = self._data_is_uniform()
        self._conservative = self._data_is_conservative()
        self.data_dict = None
        self._regriddir = "regridded_files"

        # load blocktree information
        self.block_lvls, self.block_ixs, self.block_offsets = dat_reader.get_tree_info(file)


    def _data_is_uniform(self):
        """
        Checks if the data is uniformely refined.
        :return: True if grid is uniform, False otherwise.
        """
        refined_nx = 2 ** (self.header['levmax'] - 1) * self.header['domain_nx']
        nleafs_uniform = np.prod(refined_nx / self.header['block_nx'])
        if not self.header["nleafs"] == nleafs_uniform:
            return False
        return True

    def _data_is_conservative(self):
        """
        Checks if the current data is in the form of conservative variables.
        :return: True if 'm1' is in the list of variables, False otherwise
        """
        if "m1" in self.header["w_names"]:
            return True
        else:
            return False

    def get_info(self):
        """
        Prints the current snapshot info (obtained from the header) to the console.
        """
        print("")
        print("[INFO] Current file      : {}".format(self._filename))
        print("[INFO] Datfile version   : {}".format(self.header["datfile_version"]))
        print("[INFO] Current time      : {}".format(self.header["time"]))
        print("[INFO] Physics type      : {}".format(self.header["physics_type"]))
        print("[INFO] Conservative vars : {}".format(self._conservative))
        print("[INFO] Boundaries        : {} -> {}".format(self.header["xmin"],
                                                           self.header["xmax"]))
        print("[INFO] Max AMR level     : {}".format(self.header["levmax"]))
        print("[INFO] Block size        : {}".format(self.header["block_nx"]))
        print("-" * 40)
        print("Currently known variables:")
        print(self.header["w_names"])
        print("\n")

    def get_bounds(self):
        """
        Get the bounds of the calculation domain.
        :return: A list containing the bounds in the form [[x1, x2], [y1, y2], [z1, z2]].
                 Depending on the dimension the list size is either 1, 2 or 3.
        """
        bounds = []
        for i in range(len(self.header["xmin"])):
            bounds.append([self.header["xmin"][i], self.header["xmax"][i]])
        return bounds

    def get_coordinate_arrays(self):
        """
        Calculates the domain discretisation.
        :return: A list containing the coordinate axes in the form [x, y, z] where
                 eg. 'x' is a NumPy array representing the domain discretization along the x-axis.
        """
        self._check_datadict_exists()

        w_names = self.header["w_names"]
        xmax = self.header["xmax"]
        xmin = self.header["xmin"]
        nx = np.asarray(self.data_dict.data[w_names[0]].shape)

        coordinate_arrays = []
        for i in range(len(nx)):
            coordinate_arrays.append(np.linspace(xmin[i], xmax[i], nx[i]))

        return np.asarray(coordinate_arrays)

    def load_all_data(self, nbprocs=None, regriddir=None):
        """
        Loads in all the data to RAM, this can take up quite some space for huge datasets.
        Data will be regridded if this is not already done.
        """
        if self._uniform:
            data = dat_reader.get_uniform_data(self._file, self.header)
        else:
            data = self._regrid_data(nbprocs, regriddir)
        self.data_dict = process_data.create_amrvac_dict(data, self.header)
        return self.data_dict.data

    def _regrid_data(self, nbprocs, regriddir):
        """
        Regrids non-uniform data, which is saved as a Numpy file in the folder 'regridded_files'
        (created if not present). The regrid directory is searched first if the data is already present.
        If so it is loaded, if not regridding is started.
        :param nbprocs: Number of processors to do the regridding. Defaults to max number - 2
        :param regriddir: The directory to which the regridded file is saved. This directory is searched in order to
                          try and load the file if already present.
        """
        self._check_regrid_directory(regriddir)
        try:
            data = self._load_regridded_data()
        except FileNotFoundError:
            data = dat_reader.get_amr_data(self._file, self.header, nbprocs)
            self._save_regridded_data(data)
        return data

    def switch_variable_type(self):
        """
        Switches between conservative and primitive variables. Also changes the corresponding names in the
        header info, the order of the variables is unchanged.
        """
        self._check_datadict_exists()

        if not (self.header["physics_type"] == "hd" or
                self.header["physics_type"] == "mhd"):
            print("Switching variable types only possible in hd or mhd")
            return
        if self._conservative:
            self.data_dict.convert_to_primitive()
            self._conservative = False
        else:
            self.data_dict.convert_to_conservative()
            self._conservative = True
        return

    def get_time(self):
        """
        Gets the current snapshot time.
        :return: Current time in the simulation (in AMRVAC units).
        """
        return self.header["time"]

    def _check_regrid_directory(self, regriddir):
        """
        Checks if the specified save directory is present. Defaults to 'regridded_files', if this folder is not
        present it is created. If 'regriddir' is explicitly specified and the folder is not present, throws an error
        and terminates the program (to prevent creation of unwanted folders).
        :param regriddir: The directory to which to save the regridded files (and load them back in).
        """
        if regriddir is None:
            regriddir = self._regriddir
            if not os.path.isdir(regriddir):
                os.mkdir(regriddir)
                print("[INFO] Created directory: {}".format(regriddir))
        else:
            if not os.path.isdir(regriddir):
                sys.exit("Specified directory does not exist: {}".format(regriddir))
            self._regriddir = regriddir

    def _save_regridded_data(self, data):
        """
        Saves the regridded data as a NumPy file.
        :param data: The raw, regridded data.
        """
        filename_regridded = self._regriddir + "/" + self._filename[:-4] + "_regridded.npy"
        np.save(filename_regridded, data)
        print("[INFO] Regridded data saved to {}".format(filename_regridded))

    def _load_regridded_data(self):
        """
        Loads the regridded data file back in, if present.
        :return: The data file if it is present.
        :exception: FileNotFoundError if the file is not present. The dataset will be regridded in this case.
        """
        filename_regridded = self._regriddir + "/" + self._filename[:-4] + "_regridded.npy"

        if os.path.isfile(filename_regridded):
            try:
                data = np.load(filename_regridded)
                print("[INFO] Regridded file found and loaded ({}).".format(filename_regridded))
                print("-"*40)
                return data
            except:
                print("[INFO] File found but failed to load: {}".format(filename_regridded))
        else:
            raise FileNotFoundError

    def _check_datadict_exists(self):
        if self.data_dict is None:
            print("[INFO] Dataset must be loaded to do this, call load_all_data() first.")
            raise AttributeError