#!/bin/bash

#SBATCH -p mit_normal
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 05:00:00

# Create logs directory
mkdir -p logs

echo "Starting MCMC analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Arguments passed: $@"
echo ""

# Separate PyTensor compile cache per job, so concurrent jobs don't contend for its lock
export PYTENSOR_FLAGS="base_compiledir=${TMPDIR:-/tmp}/pytensor_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"

uv run python src/run.py "$@"
status=$?

echo ""
echo "Completed at: $(date) with exit status $status"
# Fail the job if Python failed, so jobs waiting on afterok (PSIS-LOO) don't run
exit $status
