import sys

from reading import datfile_utilities
from processing import process_data

class _syntheticsetup():
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        if self.dataset.header['ndim'] == 1:
            print("synthetic views can only be created for 2D/3D data")
            sys.exit(1)

        self.line_of_sight = kwargs.get("line_of_sight", "x")




class h_alpha(_syntheticsetup):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)

        for offset in self.dataset.block_offsets:
            block = datfile_utilities.get_single_block_data(self.dataset.file, offset, self.dataset.block_shape)
            process_data.add_primitives_to_single_block(block, self.dataset)
            break


class faraday(_syntheticsetup):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)