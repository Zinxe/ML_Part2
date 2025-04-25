#!/bin/bash
#SBATCH --job-name=snakemake_pipeline
#SBATCH --partition=hmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm_logs/snakemake_%j.out
#SBATCH --error=slurm_logs/snakemake_%j.err

cd "${SLURM_SUBMIT_DIR}"

# 1. set up environment
echo "Activating Conda environment..."
source ~/installConda.sh
source ~/initConda.sh

mamba env create -f environment.yml
mamba activate mdhds2025

mkdir -p slurm_logs
mkdir -p result

# 5. run Snakemake
echo "Starting Snakemake at $(date)"
snakemake --use-conda --cores ${SLURM_CPUS_PER_TASK} --latency-wait 60
echo "Finished at $(date)"

