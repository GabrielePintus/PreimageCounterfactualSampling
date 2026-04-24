#!/usr/bin/env bash
#SBATCH --job-name=pcf-face-all
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-py13}"

CONFIGS=(
  "configs/benchmarks/benchmark_face_wisconsin_breast_cancer.yaml"
  "configs/benchmarks/benchmark_face_adult.yaml"
  "configs/benchmarks/benchmark_face_compas.yaml"
  "configs/benchmarks/benchmark_face_german_credit.yaml"
  "configs/benchmarks/benchmark_face_heloc.yaml"
  "configs/benchmarks/benchmark_face_give_me_some_credit.yaml"
  "configs/benchmarks/benchmark_face_lending_club.yaml"
)

cd "${REPO_DIR}"

echo "==> Host: $(hostname)"
echo "==> Repo: ${REPO_DIR}"
echo "==> Conda env: ${CONDA_ENV_NAME}"
echo "==> Start: $(date --iso-8601=seconds)"
echo "==> FACE configs:"
printf '    - %s\n' "${CONFIGS[@]}"

# Avoid nested BLAS/OpenMP oversubscription on small shared nodes.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found in batch environment" >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

echo "==> Python: $(which python)"
python --version

nvidia-smi || true

for config_path in "${CONFIGS[@]}"; do
  echo ""
  echo "==> Running: ${config_path}"
  echo "==> Time: $(date --iso-8601=seconds)"
  srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" python scripts/benchmark.py --config "${config_path}"
done

echo "==> End: $(date --iso-8601=seconds)"
