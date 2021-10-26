import os
import sys
import glob
import datetime
import numpy as np
import math
import io
import pandas as pd
import multiprocess as mp
from collections import defaultdict

def parse_linclust(linclust_subset, training_set, reads_of_interest):
    curr_process = mp.current_process()
    curr_process_id = str(curr_process._identity)
    curr_process_name = str(curr_process.name)
    curr_process_num_1 = curr_process_id.split('(')[1].split(',')[0]
    curr_process_num_2 = curr_process_name.split('-')[1]
    print(f'process id: {curr_process_id} - {curr_process_num_1} - {curr_process_num_2}')
    local_dict = {}
    df = pd.read_csv(linclust_subset, delimiter='\t', header=None)
    print(f'process id: {curr_process_id} - # reads: {len(df)}')
    # add read that is not in the training set as a key to reads_of_interest dictionary
    for row in df.itertuples():
        if row[1] in training_set and row[2] not in training_set:
            local_dict[row[2]] = row[1]
        elif row[1] not in training_set and row[2] in training_set:
            local_dict[row[1]] = row[2]
        else:
            continue

    reads_of_interest[curr_process_num_1] = local_dict

def verify_reads(reads_of_interest, validation_set, output_file):
    f = open(output_file, 'w')
    f.write(f'{len(reads_of_interest)}\n')
    for read, ref_read in reads_of_interest.items():
        if read in validation_set:
            f.write(f'{read}\t{ref_read}\n')
    f.close()

def get_read_ids(list_fq_files):
    dataset = []
    for fq_file in list_fq_files:
        with open(fq_file, 'r') as infile:
            content = infile.readlines()
            reads = [''.join(content[i:i+4]) for i in range(0, len(content), 4)]
            dataset += [j[0] for j in reads]
    return set(dataset)

def get_time(start, end):
    total_time = end - start
    hours, seconds = divmod(total_time.seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    print("\n%02d:%02d:%02d.%d\n" % (hours, minutes, seconds, total_time.microseconds))
    return end

def main():
    input_dir = sys.argv[1]
    linclust_out = sys.argv[2]
    # load training and validation tfrecords
    train_files = sorted(glob.glob(os.path.join(input_dir, 'training_data_cov_7x', '*.fq')))
    val_files = sorted(glob.glob(os.path.join(input_dir, 'validation_data_cov_7x', '*.fq')))

    start = datetime.datetime.now()
    # get training dataset
    train_set = get_read_ids(train_files)
    print('get training dataset')
    start = get_time(start, datetime.datetime.now())
    # parse linclust output, compare reads to training set
    num_processes = mp.cpu_count()
    print(f'Number of processes: {num_processes}')
    with mp.Manager() as manager:
        # store reads that haven't been found in the training set (key = read, value = reference read)
        reads_of_interest = manager.dict()
        processes_compare_train = [mp.Process(target=parse_linclust, args=(os.path.join(input_dir, f'linclust-subset-{i}'), train_set, reads_of_interest)) for i in range(num_processes)]
        for p in processes_compare_train:
            p.start()
        for p in processes_compare_train:
            p.join()
        print('compare linclust output to training set')
        start = get_time(start, datetime.datetime.now())
        print(f'number of processes: {len(reads_of_interest)}')
        val_set = get_read_ids(train_files)
        print('get validation set')
        start = get_time(start, datetime.datetime.now())
        # verify that reads of interest are part of the validation sets
        processes_compare_val = [mp.Process(target=verify_reads, args=(reads_of_interest[str(i)], val_set, os.path.join(input_dir, f'linclust-subset-{i}'))) for i in range(num_processes)]
        for p in processes_compare_val:
            p.start()
        for p in processes_compare_val:
            p.join()
        print('check reads in validation set')
        end = get_time(start, datetime.datetime.now())
    #
    #
    #
    # total_time = end - start
    # hours, seconds = divmod(total_time.seconds, 3600)
    # minutes, seconds = divmod(seconds, 60)
    # # get true and predicted classes
    # f.write(f'Testing accuracy: {test_accuracy.result().numpy()*100}\n')
    # f.write(f'Testing loss: {test_loss.result().numpy()}\n')
    # f.write("Testing took %02d:%02d:%02d.%d\n" % (hours, minutes, seconds, total_time.microseconds))
    # print("\nTesting took %02d:%02d:%02d.%d\n" % (hours, minutes, seconds, total_time.microseconds))



if __name__ == "__main__":
    main()


# """ script to process linclust tsv output file and identify the number of identical reads between the training and validation sets """
# import sys
# import os
# import math
# from collections import defaultdict
# #from mpi4py import MPI
# import glob
# import multiprocess as mp
#
# def get_reads(dataset, fq_file, fq_files_loc):
#     with open(os.path.join(fq_files_loc, fq_file), 'r') as handle:
#         content = handle.readlines()
#         list_reads = [''.join(content[i:i+4]) for i in range(0, len(content), 4)]
#         for read in list_reads:
#             rec = read.split('\n')
#             dataset[rec[0]] = fq_file
#     print(mp.current_process(), fq_file, len(dataset))
#     return
#
# def parse_linclust(linclust_subset, training_set, validation_set, outfilename):
#     outfile = open(outfilename, 'w')
#     with open(linclust_subset, 'r') as f:
#         for line in f:
#             ref = line.rstrip().split('\t')[0]
#             read = line.rstrip().split('\t')[1]
#             if ref in training_set and read in validation_set:
#                 outfile.write(f'{ref}\t{training_set[ref]}\t{read}\t{validation_set[read]}\n')
#             elif ref in validation_set and read in training_set:
#                 outfile.write(f'{ref}\t{validation_set[ref]}\t{read}\t{training_set[read]}\n')
#             else:
#                 continue
#
# def parse_data(filename):
#     with open(os.path.join(filename), 'r') as f:
#         content = f.readlines()
#         content = [i.rstrip() for i in content]
#     print(len(content))
#     #chunk_size = math.ceil(len(content) // mp.cpu_count())
#     #chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
#     #print(filename, len(content), len(chunks), chunk_size, mp.cpu_count())
#     return content
#
# def main():
#     input_dir = sys.argv[1]
#     print(mp.cpu_count())
#     # parse data
#     train_fq_files_sets = parse_data(os.path.join(input_dir, 'training_data_cov_7x', 'list_fq_files'))
#     val_fq_files_sets = parse_data(os.path.join(input_dir, 'validation_data_cov_7x', 'list_fq_files'))
#     print(train_fq_files_sets)
#     print(val_fq_files_sets)
#     # get reads in training and validation sets
#     with mp.Manager() as manager:
#         training_set = manager.dict()
#         validation_set = manager.dict()
#         # fill out training dictionaries
#         processes = [mp.Process(target=get_reads, args=(training_set, fq_file, os.path.join(input_dir, 'training_data_cov_7x'))) for fq_file in train_fq_files_sets]
#         for p in processes:
#             p.start()
#         for p in processes:
#             p.join()
#         print(f'size of training set dictionary: {len(training_set)}')
#         # fill out validation dictionaries
#         processes = [mp.Process(target=get_reads, args=(validation_set, fq_file, os.path.join(input_dir, 'validation_data_cov_7x'))) for fq_file in val_fq_files_sets]
#         for p in processes:
#             p.start()
#         for p in processes:
#             p.join()
#         print(f'size of validation set dictionary: {len(validation_set)}')
#         # get number of necessary processes
#         num_processes = glob.glob(os.path.join(input_dir, f'linclust-subset-*'))
#         # parse linclust output data
#         processes = [mp.Process(target=parse_linclust, args=(os.path.join(input_dir, f'linclust-subset-{i}'), training_set, validation_set, os.path.join(input_dir, f'analysis-linclust-subset-{i}'))) for i in range(num_processes)]
#         for p in processes:
#             p.start()
#         for p in processes:
#             p.join()
#
#
#
#     # # create a communicator consisting of all the processors
#     # comm = MPI.COMM_WORLD
#     # # get the number of processors
#     # size = comm.Get_size()
#     # # get the rank of each processor
#     # rank = comm.Get_rank()
#     # print(comm, size, rank)
#     # if rank == 0:
#     #     # create dictionary for storing reads in training and validaiton sets
#     #     training_set = get_reads(os.path.join(input_dir, 'training_data_cov_7x'))
#     #     validation_set = get_reads(os.path.join(input_dir, 'validation_data_cov_7x'))
#     # else:
#     #     training_set = None
#     #     validation_set = None
#     #
#     # # broadcast dictionaries to other processes
#     # training_set = comm.bcast(training_set, root=0)
#     # validation_set = comm.bcast(validation_set, root=0)
#     #
#     # # load linclust output subset
#     # linclust_subset = open(os.path.join(input_dir, f'linclust-subset-{rank}'), 'r')
#     # print(f'{rank} - {linclust-subset}')
#     #
#     # # create output file
#     # outfile = open(f'analysis-linclust-subset-{rank}', 'w')
#     #
#     # # parse linclust output
#     # parse_linclust(linclust_subset, training_set, validation_set, outfile)
#
#
# if __name__ == "__main__":
#     main()
