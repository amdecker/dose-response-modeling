#!/bin/bash

# run: ./run_cross_validation.sh

# Settings
MODELS="linear linear_quadratic quadratic linear_bart linear_quadratic_bart quadratic_bart varying_coefficient_bart"
N_SPLITS="20"
NUM_SAMPLES="2500"
NUM_WARMUP="1000"
NUM_TREES="200"
OUTPUT_DIR="results/cross_validation"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Running cross-validation..."
echo "Models: $MODELS"
echo ""

# Run each combination
for model in $MODELS; do
    echo "Running: $model"
    job_name="cv_${model}"
    # numpyro: 20 folds of ~1.5 min. BART cannot finish 20 folds in one job (see run_cv_slurm.sh)
    case "$model" in
        *bart*) time_limit="12:00:00" ;;
        *) time_limit="01:00:00" ;;
    esac
    cmd_args="--model $model --n-splits $N_SPLITS --num-samples $NUM_SAMPLES --num-warmup $NUM_WARMUP --num-trees $NUM_TREES --output-dir $OUTPUT_DIR"
    sbatch --job-name="$job_name" --time="$time_limit" ./run_cv_slurm.sh $cmd_args
done