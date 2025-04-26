#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=test.out
#SBATCH --error=test.err

hostname
date
