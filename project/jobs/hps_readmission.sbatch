#!/bin/bash
#SBATCH --job-name=readmission_hps
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=power-general
#SBATCH --account=itaym-users
#SBATCH --output=/groups/itay_mayrose/blyachman1/tau_courses/mlhc/final_proj/jobs_out/readmission_hps.out
#SBATCH --error=/groups/itay_mayrose/blyachman1/tau_courses/mlhc/final_proj/jobs_out/readmission_hps.err


CONDA_PATH=/groups/itay_mayrose/blyachman1/miniconda3


source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate res
echo "Current conda environment: $CONDA_PREFIX"
start_time=$(date +%s.%N)
echo "Job started at $(date)" >> "$LOG_FILE"
echo "--------------------------------" >> "$LOG_FILE"
cd "/groups/itay_mayrose/blyachman1/tau_courses/mlhc/final_proj/mlhc"

export SCIPY_ARRAY_API=1

python models/hyper_params/readm_hps.py \
  --input_matrix data/feature_mats/readmission_final_feature_matrix.pkl \
  --out_folder data/hps_res/readmission


end_time=$(date +%s.%N)
echo "Job finished at $(date)" >> "$LOG_FILE"
elapsed=$(echo "$end_time - $start_time" | bc)
echo "time: ${elapsed} seconds"
