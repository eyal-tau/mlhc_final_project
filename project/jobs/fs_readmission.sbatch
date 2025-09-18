#!/bin/bash
#SBATCH --job-name=readmission_fs
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=power-general
#SBATCH --account=itaym-users
#SBATCH --output=/groups/itay_mayrose/blyachman1/tau_courses/mlhc/final_proj/jobs_out/readmission_fs.out
#SBATCH --error=/groups/itay_mayrose/blyachman1/tau_courses/mlhc/final_proj/jobs_out/readmission_fs.err


CONDA_PATH=/groups/itay_mayrose/blyachman1/miniconda3


source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate res
echo "Current conda environment: $CONDA_PREFIX"
start_time=$(date +%s.%N)
echo "Job started at $(date)" >> "$LOG_FILE"
echo "--------------------------------" >> "$LOG_FILE"
cd "/groups/itay_mayrose/blyachman1/tau_courses/mlhc/final_proj/mlhc"

python models/xgboost_iterative_feature.py \
  --splits_pkl data/splits/readmission_splits.pkl \
  --target readmission \
  --out_dir data/feature_selection_fin \
  --max_features 50 \
  --patience 8

end_time=$(date +%s.%N)
echo "Job finished at $(date)" >> "$LOG_FILE"
elapsed=$(echo "$end_time - $start_time" | bc)
echo "time: ${elapsed} seconds"
