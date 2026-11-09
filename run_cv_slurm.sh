#!/bin/bash

#SBATCH -p mit_normal
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 12:00:00

# Runs all folds one after another in one job: fine for the numpyro models, but
# 20 BART fits will not finish within the time limit; use
# run_cross_validation_parallel.sh for those.

# Create logs directory
mkdir -p logs

echo "Starting CV analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Arguments passed: $@"
echo ""

# Separate PyTensor compile cache per job, so concurrent jobs don't contend for its lock
export PYTENSOR_FLAGS="base_compiledir=${TMPDIR:-/tmp}/pytensor_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"

uv run python src/cross_validate.py "$@"

echo ""
echo "Completed at: $(date)"