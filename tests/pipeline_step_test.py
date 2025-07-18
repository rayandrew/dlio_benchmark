import torch
import shutil
import os
import time
import logging
import glob
import uuid

import unittest
import pytest

import numpy as np
from hydra import initialize_config_dir, compose

from mpi4py import MPI

import dlio_benchmark
from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.utils.utility import DLIOMPI, PerfTrace, Profile
from dlio_benchmark.main import DLIOBenchmark

config_dir=os.path.dirname(dlio_benchmark.__file__)+"/configs/"
OUTPUT_DIR="./outputs/"

comm = MPI.COMM_WORLD

NUM_TRAINING_STEPS = 5
NUM_EPOCH = 1
NUM_TRAINING_DATA = 64

class Dataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=10):
        self.data = np.zeros((num_samples, 3, 32, 32), dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
@pytest.fixture
def setup_data():
    DLIOMPI.get_instance().initialize()
    dlio_output = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}")
    os.makedirs(dlio_output, exist_ok=True)

    real_output = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}")
    os.makedirs(real_output, exist_ok=True)

    dataset = Dataset(NUM_TRAINING_DATA)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False)
    yield dataloader, dlio_output, real_output
    DLIOMPI.get_instance().finalize()


def run_benchmark(cfg):
    comm.Barrier()
    ConfigArguments.reset()
    benchmark = DLIOBenchmark(cfg['workload'])
    benchmark.initialize()
    benchmark.run()
    benchmark.finalize()
    return benchmark

def num_compute(output, keyword):
    num_compute = 0
    for p in glob.glob(os.path.join(output, "*.pfw")):
        with open(p, "r") as f:
            for line in f:
                if keyword in line:
                    num_compute += 1
    return num_compute

@pytest.mark.timeout(60, method="thread")
def test_training_step(setup_data):
    dataloader, dlio_output, real_output = setup_data
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='config', overrides=['++workload.workflow.train=True', \
                                                       '++workload.workflow.generate_data=True', \
                                                       '++workload.framework=pytorch', \
                                                       '++workload.reader.data_loader=pytorch', \
                                                       '++workload.dataset.format=png', \
                                                       f'++workload.dataset.num_files_train={NUM_TRAINING_DATA}', \
                                                       f'++workload.train.epochs={NUM_EPOCH}', \
                                                       f'++workload.train.total_training_steps={NUM_TRAINING_STEPS}', \
                                                       '++workload.train.computation_time=0.001', \
                                                       f'++workload.output.folder={dlio_output}'])
        run_benchmark(cfg)

    dlio_num_compute = num_compute(dlio_output, "TorchFramework.compute")
    print("DLIO num_compute:", dlio_num_compute)

    dataset = Dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)


    logfile = f"{real_output}/trace-{DLIOMPI.get_instance().rank()}-of-{DLIOMPI.get_instance().size()}.pfw"
    pfw = PerfTrace.initialize_log(logfile=logfile, data_dir="./data", process_id=-1)

    p = Profile(name="compute", cat="compute")

    @p.log
    def compute(batch):
        print('here')
        return batch.sum()

    step = 0
    for batch in dataloader:
        step += 1
        compute(batch)
        if step > NUM_TRAINING_STEPS:
            break
        
    real_num_compute = num_compute(real_output, "compute")
    print("REAL num_compute:", real_num_compute)
    pfw.finalize()

      
if __name__ == '__main__':
    unittest.main()
