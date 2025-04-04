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
import importlib
import inspect
import hydra

import logging
from time import time


from typing import Any, Dict, List, ClassVar

from dlio_benchmark.common.constants import MODULE_CONFIG
from dlio_benchmark.common.enumerations import StorageType, FormatType, Shuffle, ReadType, FileAccess, Compression, \
    FrameworkType, \
    DataLoaderType, Profiler, DatasetType, DataLoaderSampler, CheckpointLocationType, CheckpointMechanismType
from dlio_benchmark.utils.utility import DLIOMPI, get_trace_name, utcnow
from dlio_benchmark.utils.utility import Profile, PerfTrace, DFTRACER_ENABLE, DLIOLogger, OUTPUT_LEVEL
from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig
import math
import os
import numpy as np

dlp = Profile(MODULE_CONFIG)
@dataclass
class ConfigArguments:
    __instance = None

    # command line argument
    # Framework to use
    model: str = "default"
    framework: FrameworkType = FrameworkType.TENSORFLOW
    # Dataset format, such as PNG, JPEG
    format: FormatType = FormatType.TFRECORD
    # Shuffle type
    file_shuffle: Shuffle = Shuffle.OFF
    shuffle_size: int = 1024
    sample_shuffle: Shuffle = Shuffle.OFF
    read_type: ReadType = ReadType.ON_DEMAND
    file_access: FileAccess = FileAccess.MULTI
    # Set root as the current directory by default
    storage_root: str = "./"
    storage_type: StorageType = StorageType.LOCAL_FS
    record_length: int = 64 * 1024
    record_length_stdev: int = 0
    record_length_resize: int = 0
    num_files_train: int = 8
    num_samples_per_file: int = 1
    batch_size: int = 1
    epochs: int = 1
    seed_change_epoch: bool = True
    generate_data: bool = False
    generate_only: bool = False
    log_level: int = OUTPUT_LEVEL
    data_folder: str = "./data/"
    output_folder: str = None
    metric_exclude_start_steps: int = 1
    metric_exclude_end_steps: int = 0
    checkpoint_folder: str = "./checkpoints/"
    log_file: str = "dlio.log"
    file_prefix: str = "img"
    keep_files: bool = True
    do_profiling: bool = False
    profiler: Profiler = Profiler.IOSTAT
    seed: int = 123
    do_checkpoint: bool = False
    do_train: bool = True
    checkpoint_after_epoch: int = 1
    epochs_between_checkpoints: int = 1
    steps_between_checkpoints: int = -1
    transfer_size: int = None
    read_threads: int = 1
    dont_use_mmap: bool = False
    computation_threads: int = 1
    computation_time: ClassVar[Dict[str, Any]] = {}
    preprocess_time: ClassVar[Dict[str, Any]] = {}
    prefetch_size: int = 2
    enable_chunking: bool = False
    chunk_size: int = 0
    compression: Compression = Compression.NONE
    compression_level: int = 4
    total_training_steps: int = -1
    do_eval: bool = False
    batch_size_eval: int = 1
    num_files_eval: int = 0
    generation_buffer_size: int = 2 * 1073741824  # 2 GB
    eval_time: ClassVar[Dict[str, Any]] = {}
    eval_after_epoch: int = 1
    epochs_between_evals: int = 1
    checkpoint_type: CheckpointLocationType = CheckpointLocationType.RANK_ZERO
    checkpoint_mechanism: CheckpointMechanismType = CheckpointMechanismType.NONE
    model_datatype: str = "fp16"
    optimizer_datatype: str = "fp32"
    checkpoint_fsync: bool = False
    checkpoint_only: bool = False
    checkpoint_load_rank_shift: int = 0
    checkpoint_recovery_after_steps: int = -1
    time_between_checkpoints: float = -1
    num_checkpoints: int = -1
    model_size: int = 10240
    model_type: str = None
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_attention_heads: int = 32
    num_kv_heads: int = 8
    ffn_hidden_size: int = 8192
    zero_stage: int = 0
    optimization_groups: ClassVar[List[int]] = []
    num_layers: int = -1
    layer_parameters: ClassVar[List[int]] = []
    tensor_parallelism: int = 1
    pipeline_parallelism: int = 1
    data_loader: DataLoaderType = DataLoaderType.TENSORFLOW.value
    num_subfolders_train: int = 0
    num_subfolders_eval: int = 0
    iostat_devices: ClassVar[List[str]] = []
    data_loader_classname = None
    checkpoint_mechanism_classname = None
    data_loader_sampler: DataLoaderSampler = None
    reader_classname: str = None
    multiprocessing_context: str = "fork"
    pin_memory: bool = True

    # derived fields
    required_samples: int = 1
    total_samples_eval: int = 1
    total_samples_train: int = 1
    file_list_eval: ClassVar[List[str]] = []
    file_list_train: ClassVar[List[str]] = []
    max_dimension: int = 1
    storage = None
    dimension_stdev: float = 0.0
    dimension: int = 1
    training_steps: int = 0
    eval_steps: int = 0
    samples_per_thread: int = 1
    au: float = 0.90
    file_map = None
    global_index_map = None
    data_loader_class = None
    reader_class = None
    checkpoint_mechanism_class = None
    native_data_loader = False
    train_sample_index_sum = 1
    eval_sample_index_sum = 1

    def __init__(self):
        """ Virtually private constructor. """
        if ConfigArguments.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            self.comm_size = DLIOMPI.get_instance().size()
            self.my_rank = DLIOMPI.get_instance().rank()
            self.logger = DLIOLogger.get_instance()
            ConfigArguments.__instance = self

    def __setstate__(self, state):
        self.__dict__.update(state)
        DLIOLogger.reset()
        DLIOMPI.reset()  # in 'fork' case, clear parent's DLIOMPI
        DLIOMPI.get_instance().set_parent_values(self.my_rank, self.comm_size)
        ConfigArguments.__instance = self

    @staticmethod
    def get_instance():
        """ Static access method. """
        if ConfigArguments.__instance is None:
            ConfigArguments()
        return ConfigArguments.__instance

    def configure_dlio_logging(self, is_child=False):
        global DLIOLogger
        # with "multiprocessing_context=fork" the log file remains open in the child process
        if is_child and self.multiprocessing_context == "fork":
            return
        # Configure the logging library
        log_format_verbose = '[%(levelname)s] %(message)s [%(pathname)s:%(lineno)d]'
        log_format_simple = '[%(levelname)s] %(message)s'
        # Set logging format to be simple only when debug_level <= INFO
        log_format = log_format_simple
        if 'DLIO_LOG_LEVEL' in os.environ:
            log_level_str = os.environ["DLIO_LOG_LEVEL"]
        else:
            log_level_str = "warning"
        if log_level_str in ["info", "INFO"]:
            log_level = logging.INFO
        elif log_level_str in ["warning", "warn", "WARNING", "WARN"]:
            log_level = logging.WARNING
        elif log_level_str in ["error", "ERROR"]:
            log_level = logging.ERROR
        elif log_level_str in ["critical", "CRITICAL"]:
            log_level = logging.CRITICAL
        elif log_level_str in ["DEBUG", "debug"]:
            log_format = log_format_verbose
            log_level = logging.DEBUG
        logging.basicConfig(
            force = True,
            level=log_level,
            handlers=[
                logging.FileHandler(self.logfile_path, mode="a", encoding='utf-8'),
                logging.StreamHandler()
            ],
            format = log_format
            # logging's max timestamp resolution is msecs, we will pass in usecs in the message
        )

    def configure_dftracer(self, is_child=False, use_pid=False):
        # with "multiprocessing_context=fork" the profiler file remains open in the child process
        if is_child and self.multiprocessing_context == "fork":
            return
        # Configure the profiler
        if DFTRACER_ENABLE:
            dlp_trace = get_trace_name(self.output_folder, use_pid)
            if DLIOMPI.get_instance().rank() == 0:
                self.logger.info(f"{utcnow()} Profiling DLIO {dlp_trace}")
            return PerfTrace.initialize_log(logfile=dlp_trace,
                                                   data_dir=f"{os.path.abspath(self.data_folder)}:"
                                                            f"{self.data_folder}:./{self.data_folder}:"
                                                            f"{self.checkpoint_folder}:./{self.checkpoint_folder}:"
                                                            f"{os.path.abspath(self.checkpoint_folder)}",
                                                   process_id=self.my_rank)
        return None

    def finalize_dftracer(self, dlp_logger):
        if DFTRACER_ENABLE and dlp_logger:
            dlp_logger.finalize()

    @dlp.log
    def validate(self):
        """ validate whether the parameters are set correctly"""
        if (self.do_profiling == True) and (self.profiler == Profiler('darshan')):
            if ('LD_PRELOAD' not in os.environ or os.environ["LD_PRELOAD"].find("libdarshan") == -1):
                raise Exception("Please set darshan runtime library in LD_PRELOAD")
        if self.format is FormatType.TFRECORD and (self.data_loader is DataLoaderType.PYTORCH):
            raise Exception(f"{self.framework} support for tfrecord is not implemented for {self.data_loader}.")
        if (self.framework == FrameworkType.TENSORFLOW and self.data_loader == DataLoaderType.PYTORCH) or (
                self.framework == FrameworkType.PYTORCH and self.data_loader == DataLoaderType.TENSORFLOW):
            raise Exception("Imcompatible between framework and data_loader setup.")
        if len(self.file_list_train) != self.num_files_train:
            raise Exception(
                f"Expected {self.num_files_train} training files but {len(self.file_list_train)} found. Ensure data was generated correctly.")
        if len(self.file_list_eval) != self.num_files_eval:
            raise Exception(
                f"Expected {self.num_files_eval} evaluation files but {len(self.file_list_eval)} found. Ensure data was generated correctly.")
        if self.data_loader_classname is not None and self.data_loader_sampler is None:
            raise Exception(
                f"For custom data loaders workload.reader.data_loader_sampler needs to be defined as iter or index.")
        if self.read_threads > 1:
            import platform
            if platform.system() in ["Linux", "Windows"]:
                import psutil
                p = psutil.Process()
                cores_available = len(p.cpu_affinity())
                if cores_available < self.read_threads:
                    self.logger.warning(
                        f"Running DLIO with {self.read_threads} threads for I/O but core available {cores_available} "
                        f"are insufficient and can lead to lower performance.")
        if self.num_layers > 0 and self.num_layers < self.pipeline_parallelism:
            raise Exception(
                f"Expected model.num_layers {self.num_layers} should be larger than "
                f"model.parallelism.pipeline {self.pipeline_parallelism}.")
        if self.pipeline_parallelism > 1 and self.zero_stage == 3:
            raise Exception(f"ZeRO stage {self.zero_stage} is not compatible with pipeline parallelism.")
        if self.comm_size % (self.pipeline_parallelism * self.tensor_parallelism) != 0:
            raise Exception(f"Number of processes {self.comm_size} is not a multiple of model parallelism size: {self.pipeline_parallelism * self.tensor_parallelism}")

    @staticmethod
    def reset():
        ConfigArguments.__instance = None

    @dlp.log
    def derive_configurations(self, file_list_train=None, file_list_eval=None):
        self.dimension = int(math.sqrt(self.record_length))
        self.dimension_stdev = self.record_length_stdev / 2.0 / math.sqrt(self.record_length)
        self.max_dimension = self.dimension
        if self.checkpoint_mechanism == CheckpointMechanismType.NONE:
            if self.framework == FrameworkType.TENSORFLOW:
                self.checkpoint_mechanism = CheckpointMechanismType.TF_SAVE
            elif self.framework == FrameworkType.PYTORCH:
                self.checkpoint_mechanism = CheckpointMechanismType.PT_SAVE
        if (self.record_length_resize > 0):
            self.max_dimension = int(math.sqrt(self.record_length_resize))
        if (file_list_train != None and file_list_eval != None):
            self.resized_image = np.random.randint(255, size=(self.max_dimension, self.max_dimension), dtype=np.uint8)
            self.file_list_train = file_list_train
            self.file_list_eval = file_list_eval
            self.num_files_eval = len(file_list_eval)
            self.num_files_train = len(file_list_train)
            self.total_samples_train = self.num_samples_per_file * len(self.file_list_train)
            self.total_samples_eval = self.num_samples_per_file * len(self.file_list_eval)
            self.train_sample_index_sum = self.total_samples_train * (self.total_samples_train - 1) // 2
            self.eval_sample_index_sum = self.total_samples_eval * (self.total_samples_eval - 1) // 2
            self.required_samples = self.comm_size * self.batch_size
            if self.read_threads > 0:
                self.required_samples *= self.read_threads
            self.training_steps = int(math.ceil(self.total_samples_train / self.batch_size / self.comm_size))
            self.eval_steps = int(math.ceil(self.total_samples_eval / self.batch_size_eval / self.comm_size))
        if self.data_loader_sampler is None and self.data_loader_classname is None:
            if self.data_loader == DataLoaderType.TENSORFLOW:
                self.data_loader_sampler = DataLoaderSampler.ITERATIVE
            elif self.data_loader in [DataLoaderType.PYTORCH, DataLoaderType.DALI]:
                self.data_loader_sampler = DataLoaderSampler.INDEX
        if self.data_loader_classname is not None:
            from dlio_benchmark.data_loader.base_data_loader import BaseDataLoader
            classname = self.data_loader_classname.split(".")[-1]
            module = importlib.import_module(".".join(self.data_loader_classname.split(".")[:-1]))
            for class_name, obj in inspect.getmembers(module):
                if class_name == classname and issubclass(obj, BaseDataLoader):
                    if DLIOMPI.get_instance().rank() == 0:
                        self.logger.info(f"Discovered custom data loader {class_name}")
                    self.data_loader_class = obj
                    break
        if self.checkpoint_mechanism_classname is not None:
            from dlio_benchmark.checkpointing.base_checkpointing import BaseCheckpointing
            classname = self.checkpoint_mechanism_classname.split(".")[-1]
            module = importlib.import_module(".".join(self.checkpoint_mechanism_classname.split(".")[:-1]))
            for class_name, obj in inspect.getmembers(module):
                if class_name == classname and issubclass(obj, BaseCheckpointing):
                    if DLIOMPI.get_instance().rank() == 0:
                        self.logger.info(f"Discovered custom checkpointing mechanism {class_name}")
                    self.checkpoint_mechanism_class = obj
                    break
        if self.reader_classname is not None:
            from dlio_benchmark.reader.reader_handler import FormatReader
            classname = self.reader_classname.split(".")[-1]
            module = importlib.import_module(".".join(self.reader_classname.split(".")[:-1]))
            for class_name, obj in inspect.getmembers(module):
                if class_name == classname and issubclass(obj, FormatReader):
                    if DLIOMPI.get_instance().rank() == 0:
                        self.logger.info(f"Discovered custom data reader {class_name}")
                    self.reader_class = obj
                    break
        self.train_file_map = {self.my_rank : {}}
        self.val_file_map = {self.my_rank : {}}
        self.train_global_index_map = {}
        self.val_global_index_map = {}
        self.native_data_loader = False
        if self.data_loader == DataLoaderType.TENSORFLOW:
            if self.format == FormatType.TFRECORD:
                self.native_data_loader = True
        elif self.data_loader == DataLoaderType.NATIVE_DALI:
            if self.format in [FormatType.JPEG, FormatType.PNG, FormatType.NPY, FormatType.TFRECORD]:
                self.native_data_loader = True

    @dlp.log
    def build_sample_map_iter(self, file_list, total_samples, epoch_number):
        self.logger.debug(f"ranks {self.comm_size} threads {self.read_threads} tensors")
        
        num_files = len(file_list)
        samples_sum = 0
        process_thread_file_map = {}
        if num_files > 0:
            num_threads = 1
            if self.read_threads > 0 and self.data_loader is not DataLoaderType.DALI:
                num_threads = self.read_threads
            samples_per_proc = int(math.ceil(total_samples/self.comm_size)) 
            self.samples_per_thread = samples_per_proc // num_threads
            start_sample_index = samples_per_proc * self.my_rank
            end_sample_index = samples_per_proc * (self.my_rank + 1) - 1
            if end_sample_index > total_samples - 1:
                end_sample_index = total_samples - 1
            sample_list = np.arange(start_sample_index, end_sample_index + 1)
            self.logger.debug(f"{self.my_rank} {start_sample_index} {end_sample_index}")
            if self.sample_shuffle is not Shuffle.OFF:
                if self.seed_change_epoch:
                    np.random.seed(self.seed + epoch_number)
                else:
                    np.random.seed(self.seed)
                np.random.shuffle(sample_list)
            sample_index = 0
            if num_files > 0:
                files_per_rank = (num_files // self.comm_size) % num_files
                file_index = self.my_rank * files_per_rank
                for thread_index in range(num_threads):
                    process_thread_file_map[thread_index] = []
                for sample in sample_list:
                    samples_sum += sample
                    thread_index = (sample_index // self.samples_per_thread) % num_threads
                    abs_path = os.path.abspath(file_list[file_index])
                    process_thread_file_map[thread_index].append((sample,
                                                abs_path,
                                                sample_list[sample_index] % self.num_samples_per_file))
                    sample_index += 1
                    file_index = (sample_index // self.num_samples_per_file) % num_files
        return process_thread_file_map, samples_sum

    @dlp.log
    def get_global_map_index(self, file_list, total_samples, epoch_number):
        process_thread_file_map = {}
        num_files = len(file_list)
        start_sample = 0
        end_sample = 0
        samples_sum = 0
        if num_files > 0:
            end_sample = total_samples - 1
            samples_per_proc = int(math.ceil(total_samples/self.comm_size)) 
            start_sample = self.my_rank * samples_per_proc
            end_sample = (self.my_rank + 1) * samples_per_proc - 1
            if end_sample > total_samples - 1:
                end_sample = total_samples - 1
            self.logger.debug(f"my_rank: {self.my_rank}, start_sample: {start_sample}, end_sample: {end_sample}")
            sample_list = np.arange(start_sample, end_sample + 1)
            if self.sample_shuffle is not Shuffle.OFF:
                if self.seed_change_epoch:
                    np.random.seed(self.seed + epoch_number)
                else:
                    np.random.seed(self.seed)
                np.random.shuffle(sample_list)
            for sample_index in range(end_sample - start_sample + 1):
                global_sample_index = sample_list[sample_index]
                samples_sum += global_sample_index
                file_index = int(math.floor(global_sample_index/self.num_samples_per_file))
                abs_path = os.path.abspath(file_list[file_index])
                sample_index = global_sample_index % self.num_samples_per_file
                process_thread_file_map[global_sample_index] = (abs_path, sample_index)
        return process_thread_file_map, samples_sum

    @dlp.log
    def reconfigure(self, epoch_number):
        if self.data_loader_sampler == DataLoaderSampler.ITERATIVE:
            if self.file_shuffle is not Shuffle.OFF:
                if self.seed_change_epoch:
                    np.random.seed(self.seed + epoch_number)
                else:
                    np.random.seed(self.seed)
                np.random.shuffle(self.file_list_train) 
                np.random.shuffle(self.file_list_eval)
        if self.data_loader_sampler == DataLoaderSampler.ITERATIVE:
            self.train_file_map, local_train_sample_sum = self.build_sample_map_iter(self.file_list_train, self.total_samples_train,
                                                             epoch_number)
            self.val_file_map, local_eval_sample_sum = self.build_sample_map_iter(self.file_list_eval, self.total_samples_eval, epoch_number)
        elif self.data_loader_sampler == DataLoaderSampler.INDEX:
            self.train_global_index_map, local_train_sample_sum = self.get_global_map_index(self.file_list_train, self.total_samples_train,
                                                             epoch_number)
            self.val_global_index_map, local_eval_sample_sum = self.get_global_map_index(self.file_list_eval, self.total_samples_eval,
                                                             epoch_number)
        global_train_sample_sum = DLIOMPI.get_instance().reduce(local_train_sample_sum)
        global_eval_sample_sum = DLIOMPI.get_instance().reduce(local_eval_sample_sum)        
        if self.my_rank == 0:
            self.logger.info(f"{utcnow()} Total number of samples: train {global_train_sample_sum}, eval {global_eval_sample_sum}")
            if self.train_sample_index_sum != global_train_sample_sum:
                raise Exception(f"Sharding of train samples are missing samples got {global_train_sample_sum} but expected {self.train_sample_index_sum}")
            
            if self.eval_sample_index_sum != global_eval_sample_sum:
                raise Exception(f"Sharding of eval samples are missing samples got {global_eval_sample_sum} but expected {self.eval_sample_index_sum}")

def LoadConfig(args, config):
    '''
    Override the args by a system config (typically loaded from a YAML file)
    '''
    if 'framework' in config:
        args.framework = FrameworkType(config['framework'])

    if 'storage' in config:
        if 'storage_type' in config['storage']:
            args.storage_type = StorageType(config['storage']['storage_type'])
        if 'storage_root' in config['storage']:
            args.storage_root = config['storage']['storage_root']

    # dataset related settings
    if 'dataset' in config:
        if 'record_length_bytes' in config['dataset']:
            args.record_length = config['dataset']['record_length_bytes']
        if 'record_length_byte_stdev' in config['dataset']:
            args.record_length_stdev = config['dataset']['record_length_bytes_stdev']
        if 'record_length_bytes_resize' in config['dataset']:
            args.record_length_resize = config['dataset']['record_length_bytes_resize']
        if 'num_files_train' in config['dataset']:
            args.num_files_train = config['dataset']['num_files_train']
        if 'num_files_eval' in config['dataset']:
            args.num_files_eval = config['dataset']['num_files_eval']
        if 'generation_buffer_size' in config['dataset']:
            args.generation_buffer_size = config['dataset']['generation_buffer_size']
        if 'num_samples_per_file' in config['dataset']:
            args.num_samples_per_file = config['dataset']['num_samples_per_file']
        if 'data_folder' in config['dataset']:
            args.data_folder = config['dataset']['data_folder']
            args.data_folder = args.data_folder.rstrip('/')
        if 'num_subfolders_train' in config['dataset']:
            args.num_subfolders_train = config['dataset']['num_subfolders_train']
        if 'num_subfolders_eval' in config['dataset']:
            args.num_subfolders_eval = config['dataset']['num_subfolders_eval']
        if 'enable_chunking' in config['dataset']:
            args.enable_chunking = config['dataset']['enable_chunking']
        if 'chunk_size' in config['dataset']:
            args.chunk_size = config['dataset']['chunk_size']
        if 'compression' in config['dataset']:
            args.compression = config['dataset']['compression']
        if 'compression_level' in config['dataset']:
            args.compression_level = config['dataset']['compression_level']
        if 'file_prefix' in config['dataset']:
            args.file_prefix = config['dataset']['file_prefix']
        if 'format' in config['dataset']:
            args.format = FormatType(config['dataset']['format'])
        if 'keep_files' in config['dataset']:
            args.keep_files = config['dataset']['keep_files']

    # data reader
    reader = None
    if 'data_reader' in config:
        reader = config['data_reader']
    elif 'reader' in config:
        reader = config['reader']
    if reader is not None:
        if 'dont_use_mmap' in reader:
            args.dont_use_mmap = reader['dont_use_mmap']
        if 'reader_classname' in reader:
            args.reader_classname = reader['reader_classname']
        if 'multiprocessing_context' in reader:
            args.multiprocessing_context = reader['multiprocessing_context']
        if 'data_loader' in reader:
            args.data_loader = DataLoaderType(reader['data_loader'])
        if 'data_loader_classname' in reader:
            args.data_loader_classname = reader['data_loader_classname']
        if 'data_loader_sampler' in reader:
            args.data_loader_sampler = DataLoaderSampler(reader['data_loader_sampler'])
        if 'read_threads' in reader:
            args.read_threads = reader['read_threads']
        if 'computation_threads' in reader:
            args.computation_threads = reader['computation_threads']
        if 'batch_size' in reader:
            args.batch_size = reader['batch_size']
        if 'batch_size_eval' in reader:
            args.batch_size_eval = reader['batch_size_eval']
        if 'prefetch_size' in reader:
            args.prefetch_size = reader['prefetch_size']
        if 'file_shuffle' in reader:
            args.file_shuffle = reader['file_shuffle']
        if 'file_access' in reader:
            args.file_access = FileAccess(reader['file_access'])
        if 'shuffle_size' in reader:
            args.shuffle_size = reader['shuffle_size']
        if 'sample_shuffle' in reader:
            args.sample_shuffle = Shuffle(reader['sample_shuffle'])
        if 'read_type' in reader:
            args.read_type = reader['read_type']
        if 'transfer_size' in reader:
            args.transfer_size = reader['transfer_size']
        
        args.preprocess_time = {}
        if 'preprocess_time' in reader:
            preprocess_time = {}
            if isinstance(reader['preprocess_time'], dict):
                preprocess_time = reader['preprocess_time']
            elif isinstance(reader['preprocess_time'], (int, float)):
                preprocess_time["mean"] = reader['preprocess_time']
            elif isinstance(reader['preprocess_time'], DictConfig):
                preprocess_time = OmegaConf.to_container(reader['preprocess_time'])
            else:
                args.preprocess_time = reader['preprocess_time']
            args.preprocess_time = preprocess_time if preprocess_time is not None else {}
        if 'preprocess_time_stdev' in reader:
            args.preprocess_time["stdev"] = reader['preprocess_time_stdev']
        if 'pin_memory' in reader:
            args.pin_memory = reader['pin_memory']

    # training relevant setting
    if 'train' in config:
        if 'epochs' in config['train']:
            args.epochs = config['train']['epochs']
        if 'total_training_steps' in config['train']:
            args.total_training_steps = config['train']['total_training_steps']
        if 'seed_change_epoch' in config['train']:
            args.seed_change_epoch = config['train']['seed_change_epoch']
        args.computation_time = {}
        if 'computation_time' in config['train']:
            computation_time = {}
            if isinstance(config['train']['computation_time'], dict):
                computation_time = config['train']['computation_time']
            elif isinstance(config['train']['computation_time'], (int, float)):
                computation_time["mean"] = config['train']['computation_time']
            elif isinstance(config['train']['computation_time'], DictConfig):
                computation_time = OmegaConf.to_container(config['train']['computation_time'])
            else:
                args.computation_time = config['train']['computation_time']
            args.computation_time = computation_time if computation_time is not None else {}
        if 'computation_time_stdev' in config['train']:
            args.computation_time["stdev"] = config['train']['computation_time_stdev']
        if 'seed' in config['train']:
            args.seed = config['train']['seed']

    if 'evaluation' in config:
        args.eval_time = {}
        if 'eval_time' in config['evaluation']:
            eval_time = {}
            if isinstance(config['evaluation']['eval_time'], dict):
                eval_time = config['evaluation']['eval_time']
            elif isinstance(config['evaluation']['eval_time'], (int, float)):
                eval_time["mean"] = config['evaluation']['eval_time']
            elif isinstance(config['evaluation']['eval_time'], DictConfig):
                eval_time = OmegaConf.to_container(config['evaluation']['eval_time'])
            else:
                args.eval_time = config['evaluation']['eval_time']
            args.eval_time = eval_time if eval_time is not None else {}
                
        if 'eval_time_stdev' in config['evaluation']:
            args.eval_time["stdev"] = config['evaluation']['eval_time_stdev']
        if 'eval_after_epoch' in config['evaluation']:
            args.eval_after_epoch = config['evaluation']['eval_after_epoch']
        if 'epochs_between_evals' in config['evaluation']:
            args.epochs_between_evals = config['evaluation']['epochs_between_evals']

    if 'checkpoint' in config:
        if 'checkpoint_folder' in config['checkpoint']:
            args.checkpoint_folder = config['checkpoint']['checkpoint_folder']
            args.checkpoint_folder = args.checkpoint_folder.rstrip('/')
        if 'checkpoint_after_epoch' in config['checkpoint']:
            args.checkpoint_after_epoch = config['checkpoint']['checkpoint_after_epoch']
        if 'epochs_between_checkpoints' in config['checkpoint']:
            args.epochs_between_checkpoints = config['checkpoint']['epochs_between_checkpoints']
        if 'steps_between_checkpoints' in config['checkpoint']:
            args.steps_between_checkpoints = config['checkpoint']['steps_between_checkpoints']
        if 'type' in config['checkpoint']:
            args.checkpoint_type = CheckpointLocationType(config['checkpoint']['type'])
        if 'checkpoint_mechanism_classname' in config['checkpoint']:
            args.checkpoint_mechanism_classname = config['checkpoint']['checkpoint_mechanism_classname']
        if 'fsync' in config['checkpoint']:
            args.checkpoint_sync = config['checkpoint']['fsync']
        if 'time_between_checkpoints' in config['checkpoint']:
            args.time_between_checkpoints = config['checkpoint']['time_between_checkpoints']
        if 'num_checkpoints' in config['checkpoint']:
            args.num_checkpoints = config['checkpoint']['num_checkpoints']
        if 'load_rank_shift' in config['checkpoint']:
            args.checkpoint_load_rank_shift = config['checkpoint']['load_rank_shift']
        if 'recovery_after_steps' in config['checkpoint']:
            args.checkpoint_recovery_after_steps = config['checkpoint']['recovery_after_steps']

    if 'model' in config:
        if 'name' in config['model']:
            args.model = config['model']['name']
        if 'type' in config['model']:
            args.model_type = config['model']['type']
        if 'model_size_bytes' in config['model']:
            args.model_size = config['model']['model_size_bytes']
        if 'optimization_groups' in config['model']:
            args.optimization_groups = config['model']['optimization_groups']
        if 'num_layers' in config['model']:
            args.num_layers = config['model']['num_layers']
        if 'layer_parameters' in config['model']:
            args.layer_parameters = config['model']['layer_parameters']
        if 'model_datatype' in config['model']:
            args.model_datatype = config['model']['model_datatype']
        if 'optimizer_datatype' in config['model']:
            args.optimizer_datatype = config['model']['optimizer_datatype']

        if 'parallelism' in config['model']:
            if 'tensor' in config['model']['parallelism']:
                args.tensor_parallelism = config['model']['parallelism']['tensor']
            if 'pipeline' in config['model']['parallelism']:
                args.pipeline_parallelism = config['model']['parallelism']['pipeline']
            if 'zero_stage' in config['model']['parallelism']:
                args.zero_stage = config['model']['parallelism']['zero_stage']

        if 'transformer' in config['model']:
            if 'vocab_size' in config['model']['transformer']:
                args.vocab_size = config['model']['transformer']['vocab_size']
            if 'hidden_size' in config['model']['transformer']:
                args.hidden_size = config['model']['transformer']['hidden_size']
            if 'ffn_hidden_size' in config['model']['transformer']:
                args.ffn_hidden_size = config['model']['transformer']['ffn_hidden_size']
            if 'num_attention_heads' in config['model']['transformer']:
                args.num_attention_heads = config['model']['transformer']['num_attention_heads']
            if 'num_kv_heads' in config['model']['transformer']:
                args.num_kv_heads = config['model']['transformer']['num_kv_heads']
            
    if 'output' in config:
        if 'folder' in config['output']:
            args.output_folder = config['output']['folder']
        if 'log_file' in config['output']:
            args.log_file = config['output']['log_file']
        if 'metric' in config['output']:
            if 'exclude_start_steps' in config['output']['metric']:
                args.metric_exclude_start_steps = int(config['output']['metric']['exclude_start_steps'])
            if 'exclude_end_steps' in config['output']['metric']:
                args.metric_exclude_end_steps = int(config['output']['metric']['exclude_end_steps'])

    if args.output_folder is None:
        try:
            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
            args.output_folder = hydra_cfg['runtime']['output_dir']
        except:
            args.output_folder = 'output/'
    args.logfile_path = os.path.join(args.output_folder, args.log_file)

    if 'workflow' in config:
        if 'train' in config['workflow']:
            args.do_train = config['workflow']['train']
        if 'generate_data' in config['workflow']:
            args.generate_data = config['workflow']['generate_data']
        if 'evaluation' in config['workflow']:
            args.do_eval = config['workflow']['evaluation']
        if 'checkpoint' in config['workflow']:
            args.do_checkpoint = config['workflow']['checkpoint']
        if 'profiling' in config['workflow']:
            args.do_profiling = config['workflow']['profiling']
    
    if not args.do_train:
        if args.generate_data and (not args.do_checkpoint):
            args.generate_only = True
        if args.do_checkpoint:
            args.checkpoint_only = True

    if 'profiling' in config:
        if 'profiler' in config['profiling']:
            args.profiler = Profiler(config['profiling']['profiler'])
        if 'iostat_devices' in config['profiling']:
            args.iostat_devices = config['profiling']['iostat_devices']
            if isinstance(args.iostat_devices, str):
                args.iostat_devices = [args.iostat_devices]

    if 'metric' in config:
        if 'au' in config['metric']:
            args.au = config['metric']['au']
