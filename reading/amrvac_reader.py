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
        self._header = dat_reader.get_header(file)
        self._uniform = self._data_is_uniform()
        self._conservative = self._data_is_conservative()
        self._data_dict = None
        self._savedir = "regridded_files"


    def _data_is_uniform(self):
        """
        Checks if the data is uniformely refined.
        :return: True if grid is uniform, False otherwise.
        """
        refined_nx = 2 ** (self._header['levmax'] - 1) * self._header['domain_nx']
        nleafs_uniform = np.prod(refined_nx / self._header['block_nx'])
        if not self._header["nleafs"] == nleafs_uniform:
            return False
        return True


    def _data_is_conservative(self):
        """
        Checks if the current data is in the form of conservative variables.
        :return: True if 'm1' is in the list of variables, False otherwise
        """
        if "m1" in self._header["w_names"]:
            return True
        else:
            return False


    def get_info(self):
        """
        Prints the current snapshot info (obtained from the header) to the console.
        """
        print("")
        print("[INFO] Current file      : {}".format(self._filename))
        print("[INFO] MPI-AMRVAC version: {}".format(self._header["version"]))
        print("[INFO] Current time      : {}".format(self._header["time"]))
        print("[INFO] Physics type      : {}".format(self._header["physics_type"]))
        print("[INFO] Conservative vars : {}".format(self._conservative))
        print("[INFO] Boundaries        : {} -> {}".format(self._header["xmin"],
                                                           self._header["xmax"]))
        print("[INFO] Max AMR level     : {}".format(self._header["levmax"]))
        print("[INFO] Block size        : {}".format(self._header["block_nx"]))
        print("-" * 40)
        print("Currently known variables:")
        print(self._header["w_names"])
        print("\n")


    def get_bounds(self):
        """
        Get the bounds of the calculation domain.
        :return: A list containing the bounds in the form [[x1, x2], [y1, y2], [z1, z2]].
                 Depending on the dimension the list size is either 1, 2 or 3.
        """
        bounds = []
        for i in range(len(self._header["xmin"])):
            bounds.append([self._header["xmin"][i], self._header["xmax"][i]])
        return bounds


    def get_coordinate_arrays(self):
        """
        Calculates the domain discretisation.
        :return: A list containing the coordinate axes in the form [x, y, z] where
                 eg. 'x' is a NumPy array representing the domain discretization along the x-axis.
        """
        if self._data_dict is None:
            print("{:-^40}".format("[INFO]"))
            print("The dataset must first be reduced to Python-readable format. \n"
                  "Call ds.reduce_data() first, then call ds.get_coordinate_arrays().")
            print("-"*40)
            sys.exit(1)

        w_names = self._header["w_names"]

        xmax = self._header["xmax"]
        xmin = self._header["xmin"]
        nx = np.asarray(self._data_dict.data[w_names[0]].shape)

        coordinate_arrays = []
        for i in range(len(nx)):
            coordinate_arrays.append(np.linspace(xmin[i], xmax[i], nx[i]))

        return np.asarray(coordinate_arrays)


    def reduce_data(self, nbprocs=None, savedir=None):
        """
        Starts the regridding process for non-uniformely refined data. This data is then saved as a
        Numpy file in the folder 'regridded_files', which is created if not present. This only works in Python 3,
        due to dependence on certain methods in the 'multiprocessing' module. The program will first search in the
        save directory (default 'regridded_files' unless specified) if the regridded data is already present. If so,
        it is loaded, if not, regridding is done.
        :param nbprocs: Number of processors to do the regridding. Defaults to max number - 2
        :param savedir: The directory to which the regridded file is saved. This directory is searched in order to
                        try and load the file if already present.
        """
        if self._uniform:
            data = dat_reader.get_uniform_data(self._file, self._header)
        else:
            self._check_save_directory(savedir)
            try:
                data = self._load_regridded_data()
            except FileNotFoundError:
                data = dat_reader.get_amr_data(self._file, self._header, nbprocs)
                self._save_regridded_data(data)
        self._data_dict = process_data.create_amrvac_dict(data, self._header)
        return


    def get_data(self):
        """
        Transforms the raw data into a Python dictionary. The data can be accessed using the known variables
        (see get_info).
        :return: Dictionary containing the raw data, with each variable as a NumPy array (with dimension depending on
                 the dimension of the problem).
        """
        if self._data_dict is None:
            print("{:-^40}".format("[INFO]"))
            print("The dataset must first be reduced to Python-readable format. \n"
                  "Call ds.reduce_data() first, then call ds.get_data().")
            print("-"*40)
            sys.exit(1)
        return self._data_dict.data


    def switch_variable_type(self):
        """
        Switches between conservative and primitive variables. Also changes the corresponding names in the
        header info, the order of the variables is unchanged.
        """
        if not (self._header["physics_type"] == "hd" or
                self._header["physics_type"] == "mhd"):
            print("Switching variable types only possible in hd or mhd")
            return

        if self._data_dict is None:
            print("{:-^40}".format("[INFO]"))
            print("The dataset must first be reduced to Python-readable format. \n"
                  "Call ds.reduce_data() first, then call ds.switch_variable_type().")
            print("-"*40)
            sys.exit(1)

        if self._conservative:
            self._data_dict.convert_to_primitive()
            self._conservative = False
        else:
            self._data_dict.convert_to_conservative()
            self._conservative = True
        return


    def get_time(self):
        """
        Gets the current snapshot time.
        :return: Current time in the simulation (in AMRVAC units).
        """
        return self._header["time"]


    def _check_save_directory(self, savedir):
        """
        Checks if the specified save directory is present. Defaults to 'regridded_files', if this folder is not
        present it is created. If 'savedir' is explicitly specified and the folder is not present, throws an error
        and terminates the program (to prevent creation of unwanted folders).
        :param savedir: The directory to which to save the regridded files (and load them back in).
        """
        print("-"*40)
        print("[INFO] Data is not uniformely refined and will be regridded.")
        if savedir is None:
            savedir = self._savedir

            if not os.path.isdir(savedir):
                os.mkdir(savedir)
                print("[INFO] Created directory: {}".format(savedir))
        else:
            if not os.path.isdir(savedir):
                sys.exit("Directory does not exist: {}".format(savedir))

            self._savedir = savedir

        print("-"*40)


    def _save_regridded_data(self, data):
        """
        Saves the regridded data as a NumPy file.
        :param data: The raw, regridded data.
        """
        filename_regridded = self._savedir + "/" + self._filename[:-4] + "_regridded.npy"
        np.save(filename_regridded, data)
        print("[INFO] Regridded data saved to {}".format(filename_regridded))


    def _load_regridded_data(self):
        """
        Loads the regridded data file back in, if present.
        :return: The data file if it is present.
        :exception: FileNotFoundError if the file is not present. The dataset will be regridded in this case.
        """
        filename_regridded = self._savedir + "/" + self._filename[:-4] + "_regridded.npy"

        if os.path.isfile(filename_regridded):
            try:
                data = np.load(filename_regridded)
                print("[INFO] Regridded file found and loaded ({}).".format(filename_regridded))
                print("-"*40)
                return data
            except:
                print("[INFO] File found but failed to load: {}".format(filename_regridded))
        else:
            print("[INFO] File not found: {}".format(filename_regridded))
        raise FileNotFoundError


    def plot(self, var, varname=None):
        """
        Makes a simple 1 or 2 dimensional plot of the given data. 3D plotting is not supported at the moment.
        :param var: Data of the variable to plot (eg. data_dict['rho']) if data is contained in 'data_dict'.
        :param varname: Names the variable on the plot.
        :exception: Terminates program if data is three-dimensional.
        """
        if self._header["ndim"] == 1:
            fig, ax = plt.subplots(1)
            x = self.get_coordinate_arrays()[0]
            ax.plot(x, var)
        elif self._header["ndim"] == 2:
            fig, ax = plt.subplots(1)
            bounds_x, bounds_y = self.get_bounds()
            im = ax.imshow(np.rot90(var), extent=[*bounds_x, *bounds_y])
            fig.colorbar(im)
            if varname is not None:
                ax.set_title("{} : {}".format(self._filename, varname))
            else:
                ax.set_title("{}".format(self._filename))
        else:
            print("3D plotting not implemented.")
            sys.exit(1)
        return
