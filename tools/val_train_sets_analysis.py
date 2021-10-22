""" script to process linclust tsv output file and identify the number of identical reads between the training and validation sets """
import sys
import os
import math
from collections import defaultdict
#from mpi4py import MPI
import glob
import multiprocess as mp

def get_reads(dataset, fq_file, fq_files_loc):
    with open(os.path.join(fq_files_loc, fq_file), 'r') as handle:
        content = handle.readlines()
        list_reads = [''.join(content[i:i+4]) for i in range(0, len(content), 4)]
        for read in list_reads:
            rec = read.split('\n')
            dataset[rec[0]] = fq_file
    print(mp.current_process(), fq_file, len(dataset))
    return

def parse_linclust(linclust_subset, training_set, validation_set, outfilename):
    outfile = open(outfilename, 'w')
    with open(linclust_subset, 'r') as f:
        for line in f:
            ref = line.rstrip().split('\t')[0]
            read = line.rstrip().split('\t')[1]
            if ref in training_set and read in validation_set:
                outfile.write(f'{ref}\t{training_set[ref]}\t{read}\t{validation_set[read]}\n')
            elif ref in validation_set and read in training_set:
                outfile.write(f'{ref}\t{validation_set[ref]}\t{read}\t{training_set[read]}\n')
            else:
                continue

def parse_data(filename):
    with open(os.path.join(filename), 'r') as f:
        content = f.readlines()
        content = [i.rstrip() for i in content]
    print(len(content))
    #chunk_size = math.ceil(len(content) // mp.cpu_count())
    #chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    #print(filename, len(content), len(chunks), chunk_size, mp.cpu_count())
    return content

def main():
    input_dir = sys.argv[1]
    print(mp.cpu_count())
    # parse data
    train_fq_files_sets = parse_data(os.path.join(input_dir, 'training_data_cov_7x', 'list_fq_files'))
    val_fq_files_sets = parse_data(os.path.join(input_dir, 'validation_data_cov_7x', 'list_fq_files'))
    print(train_fq_files_sets)
    print(val_fq_files_sets)
    # get reads in training and validation sets
    with mp.Manager() as manager:
        training_set = manager.dict()
        validation_set = manager.dict()
        # fill out training dictionaries
        processes = [mp.Process(target=get_reads, args=(training_set, fq_file, os.path.join(input_dir, 'training_data_cov_7x'))) for fq_file in train_fq_files_sets]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        print(f'size of training set dictionary: {len(training_set)}')
        # fill out validation dictionaries
        processes = [mp.Process(target=get_reads, args=(validation_set, fq_file, os.path.join(input_dir, 'validation_data_cov_7x'))) for fq_file in val_fq_files_sets]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        print(f'size of validation set dictionary: {len(validation_set)}')
        # get number of necessary processes
        num_processes = glob.glob(os.path.join(input_dir, f'linclust-subset-*'))
        # parse linclust output data
        processes = [mp.Process(target=parse_linclust, args=(os.path.join(input_dir, f'linclust-subset-{i}'), training_set, validation_set, os.path.join(input_dir, f'analysis-linclust-subset-{i}'))) for i in range(num_processes)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()



    # # create a communicator consisting of all the processors
    # comm = MPI.COMM_WORLD
    # # get the number of processors
    # size = comm.Get_size()
    # # get the rank of each processor
    # rank = comm.Get_rank()
    # print(comm, size, rank)
    # if rank == 0:
    #     # create dictionary for storing reads in training and validaiton sets
    #     training_set = get_reads(os.path.join(input_dir, 'training_data_cov_7x'))
    #     validation_set = get_reads(os.path.join(input_dir, 'validation_data_cov_7x'))
    # else:
    #     training_set = None
    #     validation_set = None
    #
    # # broadcast dictionaries to other processes
    # training_set = comm.bcast(training_set, root=0)
    # validation_set = comm.bcast(validation_set, root=0)
    #
    # # load linclust output subset
    # linclust_subset = open(os.path.join(input_dir, f'linclust-subset-{rank}'), 'r')
    # print(f'{rank} - {linclust-subset}')
    #
    # # create output file
    # outfile = open(f'analysis-linclust-subset-{rank}', 'w')
    #
    # # parse linclust output
    # parse_linclust(linclust_subset, training_set, validation_set, outfile)


if __name__ == "__main__":
    main()
