#!/bin/bash --login
#PBS -l select=1:system=polaris
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -A radix-io
#PBS -l filesystems=home:eagle:grand
#PBS -m bae
#PBS -N StormerProfiling

###############################################################################
# Check if running in DEBUG=1 mode.
#   - If so, this will print each command before it is ran and exit if any of
#   them return a nonzero exit status.
###############################################################################
if [[ -n "${DEBUG-}" ]]; then  # to use: `DEBUG=1 bash train_llama_alcf.sh`
    printf "\e[1;31m%s\e[0m\n" "!! RUNNING IN DEBUG MODE !!"
    set -euxo pipefail
fi

###############################################################################
# Print (but DO NOT EXECUTE !!) each command that would be ran.
#
# Enable with: NOOP=1 PBS_O_WORKDIR=$(pwd) bash train_llama_alcf.sh
###############################################################################
if [[ -v NOOP ]]; then         # to use: `NOOP=1 bash train_llama_alcf.sh`
  echo "Run NOOP mode"
  set -o noexec                # same as set -n
fi

export VIRTUAL_ENV="/lus/eagle/projects/MDClimSim/rayandrew/venvs/dlio-benchmark"
export IO_PROFILING_ENABLE=1
export MPICH_GPU_SUPPORT_ENABLED=0

#####################
# MAIN PROGRAM LOGIC
#####################
main() {
    cd "${PBS_O_WORKDIR}" || exit
    source ./setup-env.sh
    if [ "${IO_PROFILING_ENABLE}" -eq 1 ]; then
	    echo "IO_PROFILING_ENABLE=${IO_PROFILING_ENABLE}"
            IO_PROFILING_LD_PRELOAD="${VIRTUAL_ENV}/lib/python3.11/site-packages/dftracer/lib64/libdftracer_preload.so"
	    if [ ! -f "${IO_PROFILING_LD_PRELOAD}" ]; then
		    echo "Error: IO_PROFILING_LD_PRELOAD=${IO_PROFILING_LD_PRELOAD} does not exist"
		    exit 1
	    fi
	    echo "IO_PROFILING_LD_PRELOAD=${IO_PROFILING_LD_PRELOAD}"
	    mpiexec --verbose --envall -n 4 -ppn 4 --hostfile $PBS_NODEFILE --cpu-bind depth -d 16 --env DFTRACER_ENABLE ${IO_PROFILING_ENABLE} --env DFTRACER_SET_CORE_AFFINITY 1 --env DFTRACER_INC_METADATA 1 --env LD_PRELOAD ${IO_PROFILING_LD_PRELOAD} python3 -m dlio_benchmark.main workload=stormer_a100
    else
	    mpiexec --verbose --envall -n 4 -ppn 4 --hostfile $PBS_NODEFILE --cpu-bind depth -d 16 --genvall python3 -m dlio_benchmark.main workload=stormer_a100
    fi

    set +x
}

main
