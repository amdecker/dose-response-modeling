#!/bin/bash

# SLURM script to run PSIS-LOO computation for all models
# This script loops through all models and submits a SLURM job for each.
#
# If a fit for a model is queued or running (job name mcmc_<model>, as
# submitted by ./run.sh), its PSIS-LOO job waits for the fit to finish
# successfully, so ./run.sh and ./run_psis_loo.sh can be queued back to back.
# If the fit fails, the PSIS-LOO job is cancelled rather than left pending.

# List of models to compute PSIS-LOO for
MODELS=(
    "linear"
    "linear_quadratic"
    "quadratic"
    "linear_bart"
    "linear_quadratic_bart"
    "quadratic_bart"
    "varying_coefficient_bart"
)

# SLURM configuration
# Only the per-stratum log likelihood is loaded, so little memory is needed
MEM="4G"
TIME="00:15:00"

# Job name prefix
JOB_PREFIX="psis_loo"

echo "Submitting PSIS-LOO jobs for ${#MODELS[@]} models..."

# Loop through models and submit jobs
for model in "${MODELS[@]}"; do
    # Queued or running fits of this model (from ./run.sh), joined with ":"
    fit_jobs=$(squeue -u "$USER" -h -n "mcmc_${model}" -o %i | paste -sd: -)
    if [ -n "$fit_jobs" ]; then
        echo "Submitting job for model: $model (after fit job(s) $fit_jobs)"
        dependency=(--dependency="afterok:${fit_jobs}" --kill-on-invalid-dep=yes)
    else
        echo "Submitting job for model: $model (no fit queued, runs now)"
        dependency=()
    fi
    sbatch --job-name="${JOB_PREFIX}_${model}" \
           -p mit_normal \
           --mem="$MEM" \
           -t "$TIME" \
           "${dependency[@]}" \
           --wrap="uv run python src/compute_psis_loo.py $model"
done

echo "All jobs submitted!"
echo "Monitor job status with: squeue -u \$USER"
echo "Logs are written to slurm-<job id>.out in this directory"
