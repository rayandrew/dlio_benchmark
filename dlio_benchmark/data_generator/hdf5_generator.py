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

import h5py
import numpy as np
import math # is needed for chunking case.

from dlio_benchmark.common.enumerations import Compression
from dlio_benchmark.data_generator.data_generator import DataGenerator
from dlio_benchmark.utils.utility import progress
from dlio_benchmark.utils.utility import Profile
from shutil import copyfile

from dlio_benchmark.common.constants import MODULE_DATA_GENERATOR

dlp = Profile(MODULE_DATA_GENERATOR)

"""
Generator for creating data in HDF5 format.
"""
class HDF5Generator(DataGenerator):
    def __init__(self):
        super().__init__()

        self.record_element_dtype = self._args.bytes_to_np_dtype(self._args.record_element_bytes) if self._args.record_element_type == "" else np.dtype(self._args.record_element_type)

        self.record_labels = [0] * self.num_samples
        self.chunks = None
        if len(self._args.chunk_dims) > 0:
            self.chunks = self._args.chunk_dims

        self.hdf5_compression = None
        self.hdf5_compression_level = None
        if self.compression != Compression.NONE:
            self.hdf5_compression = str(self.compression)
            if self.compression == Compression.GZIP:
                self.hdf5_compression_level = self.compression_level

    def create_file(self, name, shape, records):
        hf = h5py.File(name, 'w', libver='latest')
        for dataset_id in range(self._args.num_dataset_per_record):
            hf.create_dataset(f'records_{dataset_id}', shape, chunks=self.chunks, compression=self.hdf5_compression,
                              compression_opts=self.hdf5_compression_level, dtype=self.record_element_dtype, data=records)
        hf.create_dataset('labels', data=self.record_labels)
        hf.close()

    @dlp.log    
    def generate(self):
        """
        Generate hdf5 data for training. It generates a 3d dataset and writes it to file.
        """
        super().generate()

        np.random.seed(10)

        rng = np.random.default_rng()

        dim = self.get_dimension(self.total_files_to_generate)
        if self._args.num_dataset_per_record > 1:
            dim = [[int(d[0] / self._args.num_dataset_per_record), *d[1:]] for d in dim]

        for i in dlp.iter(range(self.my_rank, int(self.total_files_to_generate), self.comm_size)):
            dim1 = dim[2*i]
            if isinstance(dim1, list):
                if dim1[0] == 1:
                    dim1 = dim1[1:]
                if self.num_samples > 1:
                    shape = (self.num_samples, *dim1)
                else:
                    shape = (1, *dim1)
                # if shape[0] == 1:
                #     shape = shape[1:]
                # records = np.random.randint(255, size=shape, dtype=self.record_element_dtype)
                records = rng.random(size=shape, dtype=self.record_element_dtype)
            else:
                dim2 = dim[2*i+1]
                if self.num_samples > 1:
                    shape = (self.num_samples, dim1, dim2)
                else:
                    shape = (1, dim1, dim2)
                records = rng.random(size=shape, dtype=self.record_element_dtype)
                # records = np.random.randint(255, size=shape, dtype=self.record_element_dtype)

            progress(i+1, self.total_files_to_generate, "Generating HDF5 Data")

            out_path_spec = self.storage.get_uri(self._file_list[i])
            if self._args.files_per_record is not None and self._args.files_per_record > 0:
                self.storage.create_node(out_path_spec, exist_ok=True)
                for j in range(self._args.files_per_record):
                    name = f"{self._file_list[i]}/{j}.part"
                    out_path_spec = self.storage.get_uri(name)
                    self.create_file(name=out_path_spec, shape=shape, records=records)
            else:
                self.create_file(name=out_path_spec, shape=shape, records=records)

        np.random.seed()
