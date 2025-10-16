"""
   Copyright (c) 2025, UChicago Argonne, LLC
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

from abc import ABC, abstractmethod
import numpy as np

from dlio_benchmark.common.enumerations import DatasetType
from dlio_benchmark.data_loader.data_loader_factory import DataLoaderFactory
from dlio_benchmark.storage.storage_factory import StorageFactory
from dlio_benchmark.utils.utility import DLIOMPI

from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.utils.utility import sleep, PerfTrace

class DummyTraceObject(object):
    def __init__(self, string, step, r):
        pass

    def __enter__(self):
        return 1

    def __exit__(self, string, step, r):
        pass

class Framework(ABC):
    def __init__(self):
        self.args = ConfigArguments.get_instance()
        self.output_folder = self.args.output_folder
        self.comm = DLIOMPI.get_instance().comm()
        self.rng = np.random.default_rng(self.args.seed + self.args.my_rank)


    @abstractmethod
    def init_loader(self, format_type, epoch, data_loader=None):
        self.reader_train = DataLoaderFactory.get_loader(data_loader, format_type,
                                                         dataset_type=DatasetType.TRAIN, epoch=epoch)
        self.reader_valid = DataLoaderFactory.get_loader(data_loader, format_type,
                                                         dataset_type=DatasetType.VALID, epoch=epoch)
        self.storage = StorageFactory().get_storage(self.args.storage_type, self.args.storage_root, self.args.framework)

    @abstractmethod 
    def get_type(self):
        pass
    
    @abstractmethod
    def start_framework_profiler(self):
        pass

    @abstractmethod
    def stop_framework_profiler(self):
        pass

    @abstractmethod
    def trace_object(self, string, step, r):
        pass

    def model(self, epoch, batch, computation_time, req=None):
        x = 1.0
        dur = sleep(computation_time, dry_run=True, rng=self.rng)
        trace = PerfTrace.get_instance()
        t_start = trace.get_time()
        t_end = t_start + (dur * 1e6)  # convert to microseconds
        while trace.get_time() < t_end:
            x = x * 0.9999999 + 0.0000001

    @abstractmethod
    def compute(self, batch, epoch_number, step, computation_time, backward_computation_time=None, backward_sync=False):
        pass

    @abstractmethod
    def get_loader(self, dataset_type):
        pass

    @abstractmethod
    def is_nativeio_available(self):
        pass
    # Metadata APIs
    def create_node(self, id, exist_ok=False):
        return False

    def get_node(self, id):
        return None

    def walk_node(self, id, use_pattern=False):
        return None

    def delete_node(self, id):
        return False

    # Data APIs
    def put_data(self, id, data, offset=None, length=None):
        return False

    def get_data(self, id, data, offset=None, length=None):
        return None

    def isfile(self, id):
        return False

