#!/bin/bash
#SBATCH --job-name=snakemake_pipeline
#SBATCH --partition=standard   
#SBATCH --nodes=1
#SBATCH --ntasks=1              
#SBATCH --cpus-per-task=28      
#SBATCH --mem=128G              
#SBATCH --time=7-00:00:00      
#SBATCH --output=slurm_logs/snakemake_%j.out
#SBATCH --error=slurm_logs/snakemake_%j.err

cd "${SLURM_SUBMIT_DIR}"

# install conda
if [ ! -d "${WORK}/miniforge3" ]; then
    echo "Installing Miniforge to ${WORK}/miniforge3..."
    bash installConda.sh
fi
echo "Sourcing Conda init..."
source ~/initConda.sh

# activate enviroment
echo "Creating Conda env 'multimodal' if needed..."
mamba env create -f mdhds2025_environment.yml -y 2>/dev/null || true
echo "Activating 'multimodal'..."
mamba activate multimodal

mkdir -p slurm_logs
mkdir -p result

# 5. run Snakemake
echo "Starting Snakemake at $(date)"
snakemake --use-conda --cores ${SLURM_CPUS_PER_TASK} --latency-wait 60
echo "Finished at $(date)"

