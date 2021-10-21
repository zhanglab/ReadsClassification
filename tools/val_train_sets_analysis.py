""" script to process linclust tsv output file and identify the number of identical reads between the training and validation sets """
import sys
import os
from collections import defaultdict
from mpi4py import MPI

def get_reads(fq_files_loc):
    dataset = defaultdict(list) # key = read id, value = set
    with open(os.path.join(fq_files_loc, 'list_fq_files'), 'r') as f:
        for line in f:
            with open(os.path.join(fq_files_loc, line.rstrip()), "r") as handle:
                content = handle.readlines()
                list_reads = [''.join(content[i:i+4]) for i in range(0, len(content), 4)]
                for read in list_reads:
                    rec = read.split('\n')
                    dataset[rec[0]] = line.rstrip()
    print(fq_files_loc, len(dataset))
    return dataset

def parse_linclust(linclust_subset, training_set, validation_set, outfile):
    for line in linclust_subset:
        ref = line.rstrip().split('\t')[0]
        read = line.rstrip().split('\t')[1]
        if ref in training_set and read in validation_set:
            outfile.write(f'{ref}\t{training_set[ref]}\t{read}\t{validation_set[read]}\n')
        elif ref in validaiton_set and read in training_set:
            outfile.write(f'{ref}\t{validation_set[ref]}\t{read}\t{training_set[read]}\n')
        else:
            continue

def main():
    input_dir = sys.argv[1]
    # create a communicator consisting of all the processors
    comm = MPI.COMM_WORLD
    # get the number of processors
    size = comm.Get_size()
    # get the rank of each processor
    rank = comm.Get_rank()
    print(comm, size, rank)
    if rank == 0:
        # create dictionary for storing reads in training and validaiton sets
        training_set = get_reads(os.path.join(input_dir, 'training_data_cov_7x'))
        validation_set = get_reads(os.path.join(input_dir, 'validation_data_cov_7x'))
    else:
        training_set = None
        validation_set = None

    # broadcast dictionaries to other processes
    training_set = comm.bcast(training_set, root=0)
    validation_set = comm.bcast(validation_set, root=0)

    # load linclust output subset
    linclust_subset = open(os.path.join(input_dir, f'linclust-subset-{rank}'), 'r')
    print(f'{rank} - {linclust-subset}')

    # create output file
    outfile = open(f'analysis-linclust-subset-{rank}', 'w')

    # parse linclust output
    parse_linclust(linclust_subset, training_set, validation_set, outfile)


if __name__ == "__main__":
    main()
