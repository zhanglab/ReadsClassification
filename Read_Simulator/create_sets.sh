#!/bin/sh
#SBATCH --time 24:00:00
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=40
#SBATCH --output=create-sets-output.%J
#SBATCH --workdir=/gpfs/u/home/TPMS/TPMScccr/scratch/run

echo START
date

cd /gpfs/u/home/TPMS/TPMScccr/scratch

source miniconda3x86/bin/activate tf-env

python /gpfs/u/home/TPMS/TPMScccr/barn/ReadsClassification/Read_Simulator/create_sets.py --input_path /gpfs/u/home/TPMS/TPMScccr/scratch/read_sim_run_1

echo END
date
