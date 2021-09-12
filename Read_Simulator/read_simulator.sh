#!/bin/sh
#SBATCH --time 24:00:00
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=40
#SBATCH --output=read-sim-output.%J
#SBATCH --workdir=/gpfs/u/home/TPMS/TPMScccr/scratch/run

echo START
date

cd /gpfs/u/home/TPMS/TPMScccr/scratch

source miniconda3x86/bin/activate tf-env

python /gpfs/u/home/TPMS/TPMScccr/barn/ReadsClassification/Read_Simulator/main.py --input_path /gpfs/u/home/TPMS/TPMScccr/scratch/read_sim_run_1 --gtdb_path /gpfs/u/home/TPMS/TPMScccr/scratch/gtdb_genomes_reps_r95 --ncbi_path /gpfs/u/home/TPMS/TPMScccr/scratch/NCBI-RefSeq-2021-07-20 --gtdb_info /gpfs/u/home/TPMS/TPMScccr/scratch/bac120_metadata_r95.tsv --genetic_code /gpfs/u/home/TPMS/TPMScccr/barn/ReadsClassification/Read_Simulator/codon_list.csv --mutate

echo END
date
