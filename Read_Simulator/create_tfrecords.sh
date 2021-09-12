#!/bin/sh
#SBATCH --time 24:00:00
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=40
#SBATCH --output=create-tfrecords-output.%J
#SBATCH --workdir=/gpfs/u/home/TPMS/TPMScccr/scratch/run

echo START
date

cd /gpfs/u/home/TPMS/TPMScccr/scratch

source miniconda3x86/bin/activate tf-env

python /gpfs/u/home/TPMS/TPMScccr/barn/ReadsClassification/Read_Simulator/create_tfrecords.py --input_path /gpfs/u/home/TPMS/TPMScccr/scratch/read_sim_run_1 --voc /gpfs/u/home/TPMS/TPMScccr/scratch/kmer_voc.txt --dataset_type 'train'

python /gpfs/u/home/TPMS/TPMScccr/barn/ReadsClassification/Read_Simulator/create_tfrecords.py --input_path /gpfs/u/home/TPMS/TPMScccr/scratch/read_sim_run_1 --voc /gpfs/u/home/TPMS/TPMScccr/scratch/kmer_voc.txt --dataset_type 'val'

python /gpfs/u/home/TPMS/TPMScccr/barn/ReadsClassification/Read_Simulator/create_tfrecords.py --input_path /gpfs/u/home/TPMS/TPMScccr/scratch/read_sim_run_1 --voc /gpfs/u/home/TPMS/TPMScccr/scratch/kmer_voc.txt --dataset_type 'test'

echo END
date
