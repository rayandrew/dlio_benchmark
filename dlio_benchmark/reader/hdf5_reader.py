"""
   Copyright (c) 2024, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import logging

import h5py
import numpy as np

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.utils.utility import Profile
from dlio_benchmark.reader.reader_handler import FormatReader

dlp = Profile(MODULE_DATA_READER)

class HDF5Reader(FormatReader):
    """
    Reader for HDF5 files.
    """
    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)
        self.choices = list(range(self._args.num_dataset_per_record))
        

    @dlp.log
    def open(self, filename):
        filenames = super().open(filename)
        fds = []
        for filename in filenames:
            # self.logger.output(f"Opening {filename}")
            fds.append(h5py.File(filename, 'r'))
        return fds

    @dlp.log
    def close(self, filename):
        for fd in self.open_file_map[filename]:
            fd.close()

    @dlp.log
    def get_sample(self, filename, sample_index):
        super().get_sample(filename, sample_index)
        bytes = 0
        for fd in self.open_file_map[filename]:
            for i in self.choices:
                image = fd[f'records_{i}'][sample_index]
                bytes += image.nbytes
                del image
        dlp.update(image_size=int(bytes))

    def next(self):
        for batch in super().next():
            yield batch

    @dlp.log
    def read_index(self, image_idx, step):
        return super().read_index(image_idx, step)

    @dlp.log
    def finalize(self):
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True
