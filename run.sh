#!/bin/bash

# run: ./run.sh

# Settings
MODELS="linear linear_quadratic quadratic linear_bart linear_quadratic_bart quadratic_bart varying_coefficient_bart"
NUM_SAMPLES="2500"
NUM_WARMUP="1000"
NUM_CHAINS="4"
NUM_TREES="200"
OUTPUT_DIR="results/idata"
FIGURES_DIR="results/figures"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Running MCMC models..."
echo "Models: $MODELS"
echo "Samples: $NUM_SAMPLES, Warmup: $NUM_WARMUP, Chains: $NUM_CHAINS"
echo "Trees (BART models): $NUM_TREES"
echo ""

# Run each model
for model in $MODELS; do
    echo "Submitting: $model"
    job_name="mcmc_${model}"
    # Measured on the cluster: numpyro fits ~1 min, BART fits 1.9-3.3 h
    case "$model" in
        *bart*) time_limit="05:00:00" ;;
        *) time_limit="00:15:00" ;;
    esac
    
    cmd_args="--model $model --num-samples $NUM_SAMPLES --num-warmup $NUM_WARMUP --num-chains $NUM_CHAINS --num-trees $NUM_TREES --output-dir $OUTPUT_DIR --figures-dir $FIGURES_DIR"
    
    sbatch --job-name="$job_name" --time="$time_limit" ./run_slurm.sh $cmd_args
done

echo ""
echo "Results will be saved in: $OUTPUT_DIR"
