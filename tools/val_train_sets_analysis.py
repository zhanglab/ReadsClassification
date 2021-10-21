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

def split_data(filename, size):
    print(filename)
    with open(filename, 'r') as f:
        content = f.readlines()
        chunk_size = len(content)//size + 1
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        print(f'number of chunks: {len(chunks)} - size of chunks: {chunk_size}')
    return chunks

def parse_linclust(clusters, training_set, validation_set, outfile):
    for i in clusters:
        ref = i.rstrip().split('\t')[0]
        read = i.rstrip().split('\t')[1]
        if ref in training_set and read in validation_set:
            outfile.write(f'{ref}\t{training_set[ref]}\t{read}\t{validation_set[read]}\n')
        elif ref in validaiton_set and read in training_set:
            outfile.write(f'{ref}\t{validation_set[ref]}\t{read}\t{training_set[read]}\n')
        else:
            continue

def main():
    input_dir = sys.argv[1]
    linclust_subset = sys.argv[2]
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
        # load and split subset file of linclust output into N lists (N = number of processes)
        linclust_chunks = split_data(os.path.join(input_dir, linclust_subset), size)
    else:
        training_set = None
        validation_set = None
        linclust_chunks = None

    # create output file
    outfilename = '-'.join(['linclust_out_subset',linclust_subset.rstrip().split('-')[-1], 'rank', rank])
    outfile = open(outfilename, 'w')

    # broadcast dictionaries to other processes
    training_set = comm.bcast(training_set, root=0)
    validation_set = comm.bcast(validation_set, root=0)

    # scatter chunks of linclust output to other processes
    linclust_chunks = comm.scatter(linclust_chunks, root=0)

    # parse linclust output
    parse_linclust(linclust_chunks, training_set, validation_set, outfile)



if __name__ == "__main__":
    main()
