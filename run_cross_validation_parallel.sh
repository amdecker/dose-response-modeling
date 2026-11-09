#!/bin/bash

# run: ./run_cross_validation_parallel.sh

# Settings
MODELS="linear linear_quadratic quadratic linear_bart linear_quadratic_bart quadratic_bart varying_coefficient_bart"
N_SPLITS="20"
NUM_SAMPLES="2500"
NUM_WARMUP="1000"
NUM_TREES="200"
OUTPUT_DIR="results/cross_validation"
RANDOM_SEED="42"

# Create output directory and logs directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

echo "Running parallel cross-validation with SLURM job arrays..."
echo "Models: $MODELS"
echo "Folds: $N_SPLITS (parallelized)"
echo ""

# Array to store job IDs for dependency management
declare -a job_ids

# Submit job array for each model
for model in $MODELS; do
    echo "Submitting job array for model: $model"
    # Not yet timed per fold: a BART fold is a ~3 h fit plus prediction; numpyro ~2 min
    case "$model" in
        *bart*) time_limit="06:00:00" ;;
        *) time_limit="00:20:00" ;;
    esac
    
    # Submit the job array (one task per fold)
    # Arguments: MODEL N_SPLITS NUM_SAMPLES NUM_WARMUP NUM_TREES OUTPUT_DIR RANDOM_SEED
    job_output=$(sbatch \
        --array=0-$((N_SPLITS-1)) \
        --job-name="cv_${model}" \
        --time="$time_limit" \
        ./run_cv_parallel_slurm.sh \
        "$model" "$N_SPLITS" "$NUM_SAMPLES" "$NUM_WARMUP" "$NUM_TREES" "$OUTPUT_DIR" "$RANDOM_SEED")
    
    # Extract job ID
    job_id=$(echo "$job_output" | grep -oP 'Submitted batch job \K\d+')
    echo "  Submitted job array: $job_id"
    job_ids+=("$job_id")
    
    # Submit aggregation job that depends on the fold jobs completing
    agg_output=$(sbatch \
        --dependency=afterok:$job_id \
        --job-name="agg_${model}" \
        --output="logs/agg_${model}_%j.out" \
        --error="logs/agg_${model}_%j.err" \
        --time=00:10:00 \
        --mem=2G \
        --cpus-per-task=1 \
        --wrap="uv run python src/cross_validate.py --model $model --n-splits $N_SPLITS --output-dir $OUTPUT_DIR --aggregate")
    
    agg_job_id=$(echo "$agg_output" | grep -oP 'Submitted batch job \K\d+')
    echo "  Submitted aggregation job: $agg_job_id (depends on $job_id)"
    echo ""
done

echo "All jobs submitted!"
echo "Job IDs: ${job_ids[@]}"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Check logs in: logs/"
echo "Results will be in: $OUTPUT_DIR"
