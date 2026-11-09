#!/bin/bash
#SBATCH --job-name=cv_fold
#SBATCH --output=logs/cv_fold_%A_%a.out
#SBATCH --error=logs/cv_fold_%A_%a.err
#SBATCH -p mit_normal
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH -t 06:00:00

# This script runs a single fold of cross-validation
# Use with SLURM job arrays to parallelize across folds; the array range is set by
# run_cross_validation_parallel.sh (--array=0-(N_SPLITS-1))
# 
# Arguments are passed directly from sbatch command:
# $1 = MODEL, $2 = N_SPLITS, $3 = NUM_SAMPLES, $4 = NUM_WARMUP, 
# $5 = NUM_TREES, $6 = OUTPUT_DIR, $7 = RANDOM_SEED

MODEL=$1
N_SPLITS=$2
NUM_SAMPLES=$3
NUM_WARMUP=$4
NUM_TREES=$5
OUTPUT_DIR=$6
RANDOM_SEED=$7
FOLD_INDEX=$SLURM_ARRAY_TASK_ID

echo "=================================="
echo "Running fold $FOLD_INDEX for model $MODEL"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "=================================="

# Separate PyTensor compile cache per job, so concurrent jobs don't contend for its lock
export PYTENSOR_FLAGS="base_compiledir=${TMPDIR:-/tmp}/pytensor_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"

# Run the cross-validation for this specific fold
uv run python src/cross_validate.py \
    --model "$MODEL" \
    --n-splits "$N_SPLITS" \
    --num-samples "$NUM_SAMPLES" \
    --num-warmup "$NUM_WARMUP" \
    --num-trees "$NUM_TREES" \
    --output-dir "$OUTPUT_DIR" \
    --fold-index "$FOLD_INDEX" \
    --random-seed "$RANDOM_SEED"
status=$?

echo "Fold $FOLD_INDEX finished with exit status $status"
# Fail the array task if Python failed, so aggregation (afterok) doesn't run
exit $status
