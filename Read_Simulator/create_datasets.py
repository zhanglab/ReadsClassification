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


# create a communicator consisting of all the processes
comm = MPI.COMM_WORLD
# get the number of processes
size = comm.Get_size()
# get the rank of each process
rank = comm.Get_rank()
print(comm, size, rank)

def get_reads(args, fq_file, dataset=None):
    """ Loads and shuffle reads """
    with open(os.path.join(args.input_path, fq_file), 'r') as infile:
        content = infile.readlines()
        reads = [''.join(content[i:i+4]) for i in range(0, len(content), 4)]
        if dataset == 'train' or dataset == 'val':
            random.shuffle(reads)
        return reads

def split_fq_files(args, fq_file, set_info):
    """ Loads and shuffle reads """
    # get label and set type
    label = fq_file.split('/')[-1].split('-')[0]
    dataset = fq_file.split('/')[-1].split('-')[1]
    # get reads
    reads = get_reads(args, fq_file, dataset)
    # compute size of chunks
    chunk_size = math.ceil(len(reads)/set_info[dataset][1])
    # split reads into x lists with x = number of tfrecords
    reads_per_tfrec = [reads[i:i+chunk_size] for i in range(0, len(reads), chunk_size)]
    # create fastq files for each tfrecords
    for i in range(set_info[dataset][1]):
        with open(os.path.join(args.input_path, 'fq_files', f'{dataset}-tfrec-{i}', f'{label}-{dataset}-reads.fq'), 'w') as f:
            f.write(''.join(reads_per_tfrec[i]))

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='path to fastq files')
    parser.add_argument('--train_reads', type=int, help='number of reads in training set')
    parser.add_argument('--test_reads', type=int, help='number of reads in testing set')
    parser.add_argument('--val_reads', type=int, help='number of reads in validation set')
    args = parser.parse_args()

    # set_info = {}
    # set_info['train'] = math.ceil(args.train_reads/1000000)
    # set_info['test'] = math.ceil(args.test_reads/1000000)
    # set_info['val'] = math.ceil(args.val_reads/1000000)
    #
    # # get list of fq files per process
    # list_fastq_files = sorted(glob.glob(os.path.join(args.input_path, f'*-reads.fq')))
    # num_files_per_process = math.ceil(len(list_fastq_files)/size)
    # fq_files_per_process = list_fq_files[rank*num_files_per_process:(rank+1)*num_files_per_process] if rank < size else
    # for fq_file in fq_files_per_process:
    #     split_fq_files(args, fq_file, set_info)

    if rank == 0:
        # get number of reads per dataset
        set_info = {'train': [], 'val': [], 'test': []}
        count_reads_train_val(args.input_path, set_info)
        count_reads_test(args.input_path, set_info)
        # define number of tfrecords to generate for each dataset
        for key, value in set_info.items():
            num_tfrec = math.ceil(value[0]/1000000)
            value.append(num_tfrec)
            # create directory to store fastq files
            for i in range(num_tfrec):
                # create output directory to store tfrecords
                os.makedirs(os.path.join(args.input_path, 'fq_files', f'{key}-tfrec-{i}'))
        # get the list of fastq files
        list_fastq_files = sorted(glob.glob(os.path.join(args.input_path, f'*-reads.fq')))
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
        set_info = None

    # broadcast information on datasets to all processes
    set_info = comm.bcast(set_info, root=0)
    # scatter list of fastq files to all processes
    fq_files_per_processes = comm.scatter(fq_files_per_processes, root=0)
    print(f'Rank: {rank}\n{fq_files_per_processes}\n')
    for fq_file in fq_files_per_processes:
        split_fq_files(args, fq_file, set_info)

if __name__ == '__main__':
    main()
