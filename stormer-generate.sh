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

export VIRTUAL_ENV="/lus/eagle/projects/MDClimSim/rayandrew/venvs/stormer"
export DFTRACER_ENABLE=0

#####################
# MAIN PROGRAM LOGIC
#####################
main() {
    cd "/lus/eagle/projects/MDClimSim/rayandrew/dlio-benchmark" || exit
    source ./setup-env.sh
    num_cores=$(nproc --all)
    mpiexec --verbose --envall -n $num_cores -ppn $num_cores --hostfile $PBS_NODEFILE --genvall python3 -m dlio_benchmark.main workload=stormer_a100 # ++workload.workflow.generate_data=True ++workload.workflow.train=False
}

main
