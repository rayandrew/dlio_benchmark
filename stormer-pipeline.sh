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

# export VIRTUAL_ENV="/lus/eagle/projects/MDClimSim/rayandrew/venvs/dlio-benchmark"
export VIRTUAL_ENV="/lus/eagle/projects/MDClimSim/rayandrew/venvs/stormer"
export IO_PROFILING_ENABLE=1
export MPICH_GPU_SUPPORT_ENABLED=0

ezpz_get_num_hosts() {
    hostfile="${HOSTFILE:-${PBS_NODEFILE:-${NODEFILE}}}"
    if [[ -n "${hostfile}" ]]; then
        nhosts=$(wc -l <"${hostfile}")
    elif [[ -n "${SLURM_NNODES:-}" ]]; then
        nhosts=${SLURM_NNODES:-1}
    else
        nhosts=1
    fi
    if [[ -n "${nhosts}" ]]; then
        export NHOSTS="${nhosts}"
    fi
    echo "${nhosts}"
}

ezpz_get_num_gpus_per_host() {
    ngpu_per_host=4
    export NGPU_PER_HOST="${ngpu_per_host}"
    echo "${ngpu_per_host}"
}

exe() { echo "\$ $@" ; "$@" ; }

#####################
# MAIN PROGRAM LOGIC
#####################
main() {
    if [[ "${PBS_O_WORKDIR}" != "/lus/eagle/projects/MDClimSim/rayandrew/dlio-benchmark" ]]; then
        PBS_O_WORKDIR="/lus/eagle/projects/MDClimSim/rayandrew/dlio-benchmark"
    fi
    cd "${PBS_O_WORKDIR}" || exit


    source ./setup-env.sh

    # extract the data

    hostfile="${HOSTFILE:-${PBS_NODEFILE:-${NODEFILE}}}"
    num_hosts=$(ezpz_get_num_hosts "${hf}")
    num_gpus_per_host=$(ezpz_get_num_gpus_per_host)
    num_gpus="$((num_hosts * num_gpus_per_host))"

    num_cores_per_host=$(getconf _NPROCESSORS_ONLN)
    num_cpus_per_host=$((num_cores_per_host / 2))
    depth=$((num_cpus_per_host / num_gpus_per_host))

    dist_launch_cmd="mpiexec --verbose --envall -n ${num_gpus} -ppn ${num_gpus_per_host} --hostfile ${hostfile} --cpu-bind depth -d ${depth}"

    if [ "${IO_PROFILING_ENABLE}" -eq 1 ]; then
	    echo "IO_PROFILING_ENABLE=${IO_PROFILING_ENABLE}"
            IO_PROFILING_LD_PRELOAD="${VIRTUAL_ENV}/lib/python3.11/site-packages/dftracer/lib64/libdftracer_preload.so"
	    if [ ! -f "${IO_PROFILING_LD_PRELOAD}" ]; then
		    echo "Error: IO_PROFILING_LD_PRELOAD=${IO_PROFILING_LD_PRELOAD} does not exist"
		    exit 1
	    fi
	    echo "IO_PROFILING_LD_PRELOAD=${IO_PROFILING_LD_PRELOAD}"
	    exe ${dist_launch_cmd} --env DFTRACER_ENABLE ${IO_PROFILING_ENABLE} --env DFTRACER_SET_CORE_AFFINITY 1 --env DFTRACER_INC_METADATA 1 --env LD_PRELOAD ${IO_PROFILING_LD_PRELOAD} --genvall python3 -m dlio_benchmark.main workload=stormer_a100 ++workload.workflow.generate_data=False ++workload.workflow.train=True # ++workload.output.folder=/local/scratch/output
    else
	    exe ${dist_launch_cmd} --genvall python3 -m dlio_benchmark.main workload=stormer_a100 ++workload.workflow.generate_data=False # ++workload.workflow.train=True
    fi

    #

    set +x
}

main
