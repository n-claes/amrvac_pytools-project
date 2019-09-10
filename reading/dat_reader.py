"""
Module containing reading and processing methods for an MPI-AMRVAC .dat file.

@author: Jannis Theunissen (original)
         Niels Claes (extensions)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import struct
import numpy as np
import sys
import scipy.interpolate as interp
import multiprocessing

# Size of basic types (in bytes)
size_logical = 4
size_int = 4
size_double = 8
name_len = 16

# For un-aligned data, use '=' (for aligned data set to '')
align = '='


def get_header(dat):
    """
    Reads header from MPI-AMRVAC 2.0 snapshot.
    :param dat: .dat file opened in binary mode.
    :return: Dictionary containing header data from snapshot.
    """

    dat.seek(0)
    h = {}


    fmt = align + 'i'
    [h['version']] = struct.unpack(fmt, dat.read(struct.calcsize(fmt)))

    if h['version'] < 3:
        print("Unsupported .dat file version: ", h['version'])
        sys.exit()

    # Read scalar data at beginning of file
    fmt = align + 9 * 'i' + 'd'
    hdr = struct.unpack(fmt, dat.read(struct.calcsize(fmt)))
    [h['offset_tree'], h['offset_blocks'], h['nw'],
     h['ndir'], h['ndim'], h['levmax'], h['nleafs'], h['nparents'],
     h['it'], h['time']] = hdr

    # Read min/max coordinates
    fmt = align + h['ndim'] * 'd'
    h['xmin'] = np.array(struct.unpack(fmt, dat.read(struct.calcsize(fmt))))
    h['xmax'] = np.array(struct.unpack(fmt, dat.read(struct.calcsize(fmt))))

    # Read domain and block size (in number of cells)
    fmt = align + h['ndim'] * 'i'
    h['domain_nx'] = np.array(
        struct.unpack(fmt, dat.read(struct.calcsize(fmt))))
    h['block_nx'] = np.array(
        struct.unpack(fmt, dat.read(struct.calcsize(fmt))))

    # Read w_names
    w_names = []
    for i in range(h['nw']):
        fmt = align + name_len * 'c'
        hdr = struct.unpack(fmt, dat.read(struct.calcsize(fmt)))
        w_names.append(b''.join(hdr).strip().decode())
    h['w_names'] = w_names

    # Read physics type
    fmt = align + name_len * 'c'
    hdr = struct.unpack(fmt, dat.read(struct.calcsize(fmt)))
    h['physics_type'] = b''.join(hdr).strip().decode()

    # Read number of physics-defined parameters
    fmt = align + 'i'
    [n_pars] = struct.unpack(fmt, dat.read(struct.calcsize(fmt)))

    # First physics-parameter values are given, then their names
    fmt = align + n_pars * 'd'
    vals = struct.unpack(fmt, dat.read(struct.calcsize(fmt)))

    fmt = align + n_pars * name_len * 'c'
    names = struct.unpack(fmt, dat.read(struct.calcsize(fmt)))
    # Split and join the name strings (from one character array)
    names = [b''.join(names[i:i+name_len]).strip().decode()
             for i in range(0, len(names), name_len)]
    # names = [str(b''.join(names[i:i + name_len]).strip())
    #          for i in range(0, len(names), name_len)]

    # Store the values corresponding to the names
    for val, name in zip(vals, names):
        h[name] = val
    return h


def get_block_data(dat):
    """
    Reads block data from an MPI-AMRVAC 2.0 snapshot.
    :param dat: .dat file opened in binary mode.
    :return: Dictionary containing block data.
    """

    dat.seek(0)
    h = get_header(dat)
    nw = h['nw']
    block_nx = np.array(h['block_nx'])
    nleafs = h['nleafs']
    nparents = h['nparents']

    # Read tree info. Skip 'leaf' array
    dat.seek(h['offset_tree'] + (nleafs + nparents) * size_logical)

    # Read block levels
    fmt = align + nleafs * 'i'
    block_lvls = np.array(struct.unpack(fmt, dat.read(struct.calcsize(fmt))))

    # Read block indices
    fmt = align + nleafs * h['ndim'] * 'i'
    block_ixs = np.reshape(
        struct.unpack(fmt, dat.read(struct.calcsize(fmt))),
        [nleafs, h['ndim']])

    # Start reading data blocks
    dat.seek(h['offset_blocks'])

    blocks = []

    for i in range(nleafs):
        lvl = block_lvls[i]
        ix = block_ixs[i]

        # Read number of ghost cells
        fmt = align + h['ndim'] * 'i'
        gc_lo = np.array(struct.unpack(fmt, dat.read(struct.calcsize(fmt))))
        gc_hi = np.array(struct.unpack(fmt, dat.read(struct.calcsize(fmt))))

        # Read actual data
        block_shape = np.append(gc_lo + block_nx + gc_hi, nw)
        fmt = align + np.prod(block_shape) * 'd'
        d = struct.unpack(fmt, dat.read(struct.calcsize(fmt)))
        w = np.reshape(d, block_shape, order='F')  # Fortran ordering

        b = {"lvl": lvl, "ix": ix, "w": w}
        blocks.append(b)

    return blocks


def get_uniform_data(dat, hdr):
    """
    Retrieves the data for a uniform data set.
    :param dat: .dat file, opened in binary mode.
    :param hdr: The .dat file header.
    :return: The raw data as a NumPy array.
    """
    blocks = get_block_data(dat)

    refined_nx = 2 ** (hdr['levmax'] - 1) * hdr['domain_nx']
    domain_shape = np.append(refined_nx, hdr['nw'])
    d = np.zeros(domain_shape, order='F')

    for b in blocks:
        i0 = (b['ix'] - 1) * hdr['block_nx']
        i1 = i0 + hdr['block_nx']
        if hdr['ndim'] == 1:
            d[i0[0]:i1[0], :] = b['w']
        elif hdr['ndim'] == 2:
            d[i0[0]:i1[0], i0[1]:i1[1], :] = b['w']
        elif hdr['ndim'] == 3:
            d[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2], :] = b['w']
    return d

def get_amr_data(dat, hdr, nbprocs):
    """
    Retrieves the data for a non-uniform data set by performing regridding.
    :param dat: .dat file, opened in binary mode.
    :param hdr: The .dat file header.
    :param nbprocs: The number of processors to use when regridding.
    :return: The raw data as a NumPy array.
    """
    # version check
    PY2 = sys.version_info[0] == 2
    if PY2:
        print("Only Python 3 is supported due to methods from the multiprocessing module.")
        sys.exit(1)

    if nbprocs is None:
        nbprocs = multiprocessing.cpu_count() - 2
    print("[INFO] Regridding will be done using {} processors.".format(nbprocs))
    print("-"*40)

    blocks = get_block_data(dat)
    refined_nx = 2 ** (hdr['levmax'] - 1) * hdr['domain_nx']
    domain_shape = np.append(refined_nx, hdr['nw'])
    d = np.zeros(domain_shape, order='F')

    max_lvl = hdr['levmax']
    block_iterable = [(b, hdr) for b in blocks]

    # Progress tracking
    init_progress = multiprocessing.Value("i", 0)
    nb_blocks  = multiprocessing.Value("i", len(blocks))

    print_progress(0, 100)

    # Initialize pool
    pool = multiprocessing.Pool(initializer=mp_init,
                                initargs=[init_progress, nb_blocks],
                                processes=nbprocs)

    # Execute multiprocessing pool
    blocks_regridded = np.array(pool.starmap(interpolate_block, block_iterable))
    pool.close()
    pool.join()
    print_progress(100, 100)
    print("")

    # fill arrays with regridded data
    for i in range(len(blocks)):
        b = blocks[i]
        block_lvl = b['lvl']
        block_idx = b['ix']

        grid_diff = 2 ** (max_lvl - block_lvl)

        max_idx = block_idx * grid_diff
        min_idx = max_idx - grid_diff

        idx0 = min_idx * hdr['block_nx']

        if hdr['ndim'] == 1:
            if block_lvl == max_lvl:
                idx1 = idx0 + hdr['block_nx']
                d[idx0[0]:idx1[0], :] = b['w']
            else:
                idx1 = idx0 + (hdr['block_nx'] * grid_diff)
                d[idx0[0]:idx1[0], :] = blocks_regridded[i]

        elif hdr['ndim'] == 2:
            if block_lvl == max_lvl:
                idx1 = idx0 + hdr['block_nx']
                d[idx0[0]:idx1[0], idx0[1]:idx1[1], :] = b['w']
            else:
                idx1 = idx0 + (hdr['block_nx'] * grid_diff)
                d[idx0[0]:idx1[0], idx0[1]:idx1[1], :] = blocks_regridded[i]
        else:
            if block_lvl == max_lvl:
                idx1 = idx0 + hdr['block_nx']
                d[idx0[0]:idx1[0], idx0[1]:idx1[1], idx0[2]:idx1[2], :] = b['w']
            else:
                idx1 = idx0 + (hdr['block_nx'] * grid_diff)
                d[idx0[0]:idx1[0], idx0[1]:idx1[1], idx0[2]:idx1[2], :] = blocks_regridded[i]

    return d


def print_progress(count, total):
    """
    Small method to print the current progress.
    :param count: Number of blocks done.
    :param total: Number of blocks in total.
    """
    percentage = round(100.0 * count / float(total), 1)
    print("Regridding...    {}%".format(percentage), end="\r")

def add_progress():
    """
    Adds progress to the multiprocessing variables
    """
    progress.value += 1
    if progress.value % 10 == 0:
        print_progress(progress.value, total_blocks.value)
    return

def mp_init(t, nb_blocks):
    """
    Initialiser method passed to the multiprocessing pool.
    :param t: progress
    :param nb_blocks: number of blocks
    """
    global progress, total_blocks
    progress = t
    total_blocks = nb_blocks

def interpolate_block(b, hdr):
    """
    Interpolates a given block to the maximum refinement level using flat interpolation.
    :param b: The block to refine.
    :param hdr: The .dat file header.
    :return: NumPy array containing the refined block data.
    """
    block_lvl = b['lvl']
    max_lvl   = hdr['levmax']
    if block_lvl == max_lvl:
        add_progress()
        return b
    ndim = hdr['ndim']
    curr_width = hdr['block_nx']

    grid_diff = 2**(max_lvl - block_lvl)
    regrid_width = curr_width * grid_diff
    nb_elements = np.prod(hdr['block_nx'])

    b_interpolated = np.zeros([*regrid_width, hdr['nw']])

    for var in range(0, hdr['nw']):
        if ndim == 1:
            block_spline = interp.interp1d(np.arange(curr_width), b['w'][:, var])
            block_result = block_spline(np.linspace(0, b['w'][:, var].size-1, regrid_width[0]))
            b_interpolated[:, var] = block_result
        elif ndim == 2:
            vals = np.reshape(b['w'][:, :, var], nb_elements)
            pts  = np.array(  [[i, j] for i in np.linspace(0, 1, curr_width[0])
                                      for j in np.linspace(0, 1, curr_width[1])]  )
            grid_x, grid_y = np.mgrid[0:1:regrid_width[0]*1j,
                                      0:1:regrid_width[1]*1j]
            grid_interpolated = interp.griddata(pts, vals, (grid_x, grid_y), method="linear")
            b_interpolated[:, :, var] = grid_interpolated
        else:
            vals = np.reshape(b['w'][:, :, :, var], nb_elements)
            pts  = np.array(  [[i, j, k] for i in np.linspace(0, 1, curr_width[0])
                                         for j in np.linspace(0, 1, curr_width[1])
                                         for k in np.linspace(0, 1, curr_width[2])]  )
            grid_x, grid_y, grid_z = np.mgrid[0:1:regrid_width[0]*1j,
                                              0:1:regrid_width[1]*1j,
                                              0:1:regrid_width[2]*1j]
            grid_interpolated = interp.griddata(pts, vals, (grid_x, grid_y, grid_z), method="linear")
            b_interpolated[:, :, :, var] = grid_interpolated

    add_progress()
    return b_interpolated


