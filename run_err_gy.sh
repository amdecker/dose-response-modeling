#!/bin/bash

# ERR/Gy for dose ranges (Ozasa et al. 2012, Fig. 5). One job per model and cut
# point. Collect them afterwards with
#   uv run python src/err_gy.py --task plot

MODELS=(
    "linear"
    # "linear_bart"
)

# Cut points in Gy (same as err_gy.DOSE_LIMITS), plus the full dose range
DOSE_LIMITS=$(uv run python -c "import sys; sys.path.insert(0, 'src'); from err_gy import DOSE_LIMITS; print(' '.join(f'{x:g}' for x in DOSE_LIMITS))")
DOSE_LIMITS="$DOSE_LIMITS full"

SETTINGS="--num-samples 2500 --num-warmup 1000 --num-chains 4 --num-trees 200"

mkdir -p logs

for model in "${MODELS[@]}"; do
    # One CPU per chain. Measured: numpyro fits ~1 min, BART fits 1.9-3.3 h
    case "$model" in
        *bart*) time_limit="05:00:00" ;;
        *) time_limit="00:15:00" ;;
    esac
    for dose_limit in $DOSE_LIMITS; do
        job_name="err_gy_${model}_${dose_limit}"
        echo "Submitting $job_name"
        # Separate PyTensor compile cache per job, so concurrent jobs don't contend for its lock
        sbatch --job-name="$job_name" \
               -p mit_normal \
               -c 4 --mem=32G -t "$time_limit" \
               --output="logs/${job_name}_%j.out" \
               --wrap="export PYTENSOR_FLAGS=base_compiledir=\${TMPDIR:-/tmp}/pytensor_\${SLURM_JOB_ID}; uv run python src/err_gy.py --task fit --model $model --dose-limit $dose_limit $SETTINGS"
    done
done

echo "All jobs submitted!"
echo "Monitor job status with: squeue -u \$USER"
echo "Progress lines (BART) and results are in logs/err_gy_*.out"
