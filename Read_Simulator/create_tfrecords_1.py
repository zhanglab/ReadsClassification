import os
import numpy as np
import tensorflow as tf
from mpi4py import MPI
import argparse
import sys
import random
import glob
import math
from utils import *

# create a communicator consisting of all the processors
comm = MPI.COMM_WORLD
# get the number of processors
size = comm.Get_size()
# get the rank of each processor
rank = comm.Get_rank()
print(comm, size, rank)

def get_reads(args, fq_file, fq_type=None):
    """ Loads and shuffle reads """
    with open(os.path.join(args.input_path, fq_file), 'r') as infile:
        content = infile.readlines()
        reads = [''.join(content[i:i+4]) for i in range(0, len(content), 4)]
        if fq_type == 'train' or fq_type == 'val':
            random.shuffle(reads)
        return reads

def split_fq_files(args, fq_file):
    # get label
    label = fq_file.split('/')[-1].split('-')[0]
    # get reads
    reads = get_reads(args, fq_file, args.dataset)
    # compute size of chunks
    chunk_size = math.ceil(len(reads)/args.num_tfrec)
    # split reads into x lists with x = number of tfrecords
    reads_per_tfrec = [reads[i:i+chunk_size] for i in range(0, len(reads), chunk_size)]
    # create fastq files for each tfrecords
    for i in range(args.num_tfrec):
        with open(os.path.join(args.input_path, 'fq_files', f'{args.dataset}-tfrec-{i}', f'{label}-{args.dataset}-reads.fq'), 'w') as f:
            f.write(''.join(reads_per_tfrec[i]))

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='path to fastq files')
    parser.add_argument('--dataset', type=str, help='type of dataset', choices=['train', 'val', 'test'])
    parser.add_argument('--num_reads', type=int, help='total number of reads in dataset')
    args = parser.parse_args()
    # define number of tfrecords to generate
    args.num_tfrec = math.ceil(args.num_reads/25000000)
    if rank == 0:
        # create directory to store fastq files
        for i in range(args.num_tfrec):
            # define output directory
            output_path = os.path.join(args.input_path, 'fq_files', f'{args.dataset}-tfrec-{i}')
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
        # get the list of fastq files
        list_fastq_files = sorted(glob.glob(os.path.join(args.input_path, f'*-{args.dataset}*-reads.fq')))
        # divide fastq files into number of processes available
        group_size = len(list_fastq_files)//size
        print(f'group size: {group_size}')
        print(list_fastq_files)
        fq_files_per_processes = [[] for i in range(size)]
        num_process = 0
        for i in range(len(list_fastq_files)):
            fq_files_per_processes[num_process].append(list_fastq_files[i])
            num_process += 1
            if num_process == size:
                num_process = 0
    else:
        fq_files_per_processes = None
    # scatter list of fastq files to all processes
    fq_files_per_processes = comm.scatter(fq_files_per_processes, root=0)
    print(f'Rank: {rank}\n{fq_files_per_processes}\n')
    for fq_file in fq_files_per_processes:
        split_fq_files(args, fq_file)

if __name__ == '__main__':
    main()
