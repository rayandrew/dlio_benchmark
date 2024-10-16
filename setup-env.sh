#!/usr/bin/env bash

module use /soft/modulefiles
module unload darshan
module load gcc-native
# module load conda
# conda activate base
module unload cray-mpich
# module load craype-accel-nvidia80
module load cudatoolkit-standalone
module load cudnn
export CC=cc
export CXX=CC
source "${VIRTUAL_ENV}/bin/activate"

# AWS NCCL OFI Plugin settings below
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=/soft/libraries/nccl/nccl_2.21.5-1+cuda12.2_x86_64/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072
