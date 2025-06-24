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

from dlio_benchmark.common.error_code import ErrorCodes
from dlio_benchmark.common.enumerations import FormatType, FrameworkType, DatasetType, DataLoaderType
from dlio_benchmark.data_loader.data_loader_factory import DataLoaderFactory
from dlio_benchmark.framework.framework import Framework, DummyTraceObject
from dlio_benchmark.common.constants import MODULE_AI_FRAMEWORK
import os
import torch
import numpy as np
import functools
import logging
from dlio_benchmark.utils.utility import utcnow, DLIOMPI
from dlio_benchmark.utils.utility import Profile, PerfTrace

from time import time

from dlio_benchmark.reader.reader_factory import ReaderFactory
from dlio_benchmark.storage.storage_factory import StorageFactory
from dlio_benchmark.utils.utility import sleep

HANDLED_FUNCTIONS = {}
dlp = Profile(MODULE_AI_FRAMEWORK)


def implements(torch_function):
    """Register a torch function override for ScalarTensor"""

    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


# Does this annotation mean that torch.mean will be replaced by torch_sleep?
@implements(torch.mean)
def torch_sleep(sleep_time):
    return sleep(sleep_time)

# def simulate_gpu_on_cpu(duration_sec, base_size=512):
#     size = int(base_size * (duration_sec / 0.1))  # scale size if longer duration
#     size = max(size, base_size)  # avoid too small
#     a = np.random.rand(size, size)
#     b = np.random.rand(size, size)

#     end_time = PerfTrace.get_instance().get_time() + duration_sec * 1e6 # convert to microseconds
#     while PerfTrace.get_instance().get_time() < end_time:
#         _ = np.dot(a, b)

def busy_compute_for(duration_sec):
    x = np.random.rand(1000, 1000)
    duration = duration_sec * 1e6  # convert to microseconds
    start = PerfTrace.get_instance().get_time()
    # # do one operation
    # y = np.dot(x, x)
    # z = np.sum(y)

    while (PerfTrace.get_instance().get_time() - start) < duration:
        y = np.dot(x, x)
        z = np.sum(y)

class TorchFramework(Framework):
    __instance = None

    @dlp.log_init
    def __init__(self, profiling):
        super().__init__()
        self.profiling = profiling
        self.reader_handler = None
        self.comm = DLIOMPI.get_instance().comm()

    @dlp.log
    def init_loader(self, format_type, epoch=0, data_loader=None):
        if data_loader is None:
            data_loader = DataLoaderType.PYTORCH
        super().init_loader(format_type, epoch, data_loader)

    @dlp.log
    def get_type(self):
        return FrameworkType.PYTORCH

    @staticmethod
    def get_instance(profiling):
        """ Static access method. """
        if TorchFramework.__instance is None:
            TorchFramework.__instance = TorchFramework(profiling)
        return TorchFramework.__instance

    @dlp.log
    def start_framework_profiler(self):
        pass

    @dlp.log
    def stop_framework_profiler(self):
        pass

    @dlp.log
    def trace_object(self, string, step, r):
        return DummyTraceObject(string, step, r)

    @dlp.log
    def compute(self, batch, epoch_number, step, computation_time, backward_computation_time=None, accumulate_grad_batches=1):
        self.forward(batch, epoch_number, step, computation_time)
        self.backward(batch, epoch_number, step, computation_time=backward_computation_time, accumulate_grad_batches=accumulate_grad_batches)

    @dlp.log
    def forward(self, batch, epoch_number, step, computation_time):
        self.model(batch, computation_time)

    @dlp.log
    def backward(self, batch, epoch_number, step, computation_time, accumulate_grad_batches=1):
        if computation_time:
            self.model(batch, computation_time)
        if step % accumulate_grad_batches == 0:
            self.comm.barrier()

    @dlp.log
    def transfer(self, batch, epoch_number, step, transfer_time=None):
        if transfer_time:
            duration = sleep(transfer_time, exec=False) * 1e6  # convert to microseconds
            start = PerfTrace.get_instance().get_time()
            while (PerfTrace.get_instance().get_time() - start) < duration:
                _ = np.random.rand(512, 512).mean()
            # sleep(transfer_time)

    @dlp.log
    def get_loader(self, dataset_type=DatasetType.TRAIN):
        if dataset_type == DatasetType.TRAIN:
            return self.reader_train
        else:
            return self.reader_valid

    @dlp.log
    def is_nativeio_available(self):
        return False
