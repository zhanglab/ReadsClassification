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

# def parse_linclust(linclust_subset, outfilename, set_1, set_2, reads_of_interest_set_1, reads_of_interest_set_2, reads_in_set_1, reads_in_set_2):
#     curr_process = mp.current_process()
#     curr_process_name = str(curr_process.name)
#     curr_process_num = curr_process_name.split('-')[1]
#     outfile = open(outfilename, 'w')
#     local_dict_roi_set_2 = {} # key = read in set 2
#     local_dict_roi_set_1 = {} # key = read in set 1
#     local_dict_ri_set_1 = {}
#     local_dict_ri_set_2 = {}
#     df = pd.read_csv(linclust_subset, delimiter='\t', header=None)
#     print(f'process id: {curr_process_num} - # reads: {len(df)}')
#     # add reads that is not in the training set as a key to reads_of_interest dictionary
#     for row in df.itertuples():
#         if row[1] in set_1 and row[2] in set_2:
#             local_dict_roi_set_2[row[2]] = row[1]
#             outfile.write(f'{row[1]}\t{set_1[row[1]]}\t{row[2]}\t{set_2[row[2]]}\n')
#         elif row[1] in set_2 and row[2] in set_1:
#             local_dict_roi_set_1[row[2]] = row[1]
#             outfile.write(f'{row[2]}\t{set_1[row[2]]}\t{row[1]}\t{set_2[row[1]]}\n')
#         elif row[1] in set_1 and row[2] in set_1:
#             local_dict_ri_set_1[row[2]] = row[1]
#         elif row[1] in set_2 and row[2] in set_2:
#             local_dict_ri_set_2[row[2]] = row[1]
#
#     reads_of_interest_set_2[str(curr_process_num)] = local_dict_roi_set_2
#     reads_of_interest_set_1[str(curr_process_num)] = local_dict_roi_set_1
#     reads_in_set_1[str(curr_process_num)] = local_dict_ri_set_1
#     reads_in_set_2[str(curr_process_num)] = local_dict_ri_set_2

# def verify_reads_v1(reads_of_interest, validation_set, training_set, output_file):
#     f = open(output_file, 'w')
#     f.write(f'{len(reads_of_interest)}\n')
#     for t_read, v_read in reads_of_interest.items():
#         if v_read in validation_set:
#             f.write(f'{v_read}\t{validation_set[v_read]}\t{t_read}\t{training_set[t_read]}\n')
#     f.close()
#
# def verify_reads_v2(reads_of_interest, validation_set, training_set, output_file):
#     f = open(output_file, 'w')
#     f.write(f'{len(reads_of_interest)}\n')
#     for v_read, t_read in reads_of_interest.items():
#         if v_read in validation_set:
#             f.write(f'{v_read}\t{validation_set[v_read]}\t{t_read}\t{training_set[t_read]}\n')
#     f.close()

def parse_linclust(linclust_subset, linclust_data_dict):
    outfile = open(outfilename, 'w')
    with open(linclust_subset, 'r') as infile:
        for line in infile:
            read_1 = line.rstrip().split('\t')[0]
            read_2 = line.rstrip().split('\t')[1]
            if read_1 != read_2:
                linclust_data_dict[read_2] = read_1
                # outfile.write(f'{read_1}\t{read_2}\n')
            else:
                continue

def get_read_ids(list_fq_files):
    dataset = {}
    for fq_file in list_fq_files:
        with open(fq_file, 'r') as infile:
            content = infile.readlines()
            reads = [''.join(content[i:i+4]) for i in range(0, len(content), 4)]
            for j in reads:
                dataset[j.split('\n')[0].rstrip()[1:]] = j.split('\n')[1].rstrip()
    return dataset

def get_time(start, end):
    total_time = end - start
    hours, seconds = divmod(total_time.seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    print("\n%02d:%02d:%02d.%d\n" % (hours, minutes, seconds, total_time.microseconds))
    return end

def get_num_reads(data, set_name):
    num_reads = 0
    print(f'Number of reads in {set_name} dataset:')
    for key, value in data.items():
        print(f'{key}\t{len(value)}')
        num_reads += len(value)
    print(num_reads)

def main():
    input_dir = sys.argv[1]
    set_1 = sys.argv[2]
    set_2 = sys.argv[3]
    set_1_name = sys.argv[4]
    set_2_name = sys.argv[5]

    # find the remaining reads (reads identical but with different read ids) in the training set
    # set_1_files = sorted(glob.glob(os.path.join(path_set_1, '*-reads.fq')))
    # print(f'Number of fastq files in set #1: {len(set_1_files)}')
    # set_1 = get_read_ids(set_1_files)
    # print(f'get set #1 - {len(set_1)}')
    # start = get_time(start, datetime.datetime.now())

    # filter reads in cluster results that are identical and with the same read ids
    with mp.Manager() as manager:
        linclust_subset_data_dict = manager.dict()
        num_processes = len(glob.glob(os.path.join(input_dir, f'linclust-subset-*')))
        processes_compare_train = [mp.Process(target=parse_linclust, args=(os.path.join(input_dir, f'linclust-subset-{i}'), linclust_data_dict)) for i in range(num_processes)]
        for p in processes_compare_train:
            p.start()
        for p in processes_compare_train:
            p.join()
    print(len(linclust_subset_data_dict))







    # # define number of processes requested
    # pool_size = int(os.environ['SLURM_CPUS_ON_NODE'])
    # # load training and validation tfrecords
    # set_1_files = sorted(glob.glob(os.path.join(path_set_1, '*-reads.fq')))
    # set_2_files = sorted(glob.glob(os.path.join(path_set_2, '*-reads.fq')))
    # # define the number of processes required
    # num_processes = len(glob.glob(os.path.join(input_dir, f'linclust-subset-*')))
    # print(f'Number of processes: {num_processes}')
    # print(f'Number of fastq files in set #1: {len(set_1_files)}')
    # print(f'Number of fastq files in set #2: {len(set_2_files)}')
    # start = datetime.datetime.now()
    # get reads from set #1
    # set_1 = get_read_ids(set_1_files)
    # print(f'get set #1 - {len(set_1)}')
    # start = get_time(start, datetime.datetime.now())
    # get reads from set #2
    # set_2 = get_read_ids(set_2_files)
    # print(f'get set #2 - {len(set_2)}')
    # start = get_time(start, datetime.datetime.now())
    # parse linclust output, compare reads to training set
    # with mp.Manager() as manager:
        # store reads that haven't been found in the training set (key = read, value = reference read)
        # reads_of_interest_set_1 = manager.dict()
        # reads_of_interest_set_2 = manager.dict()
        # reads_in_set_1 = manager.dict()
        # reads_in_set_2 = manager.dict()
        # processes_compare_train = [mp.Process(target=parse_linclust, args=(os.path.join(input_dir, f'linclust-subset-{i}'), os.path.join(input_dir, f'linclust-subset-{i}'), set_1, set_2, reads_of_interest_set_1, reads_of_interest_set_2, reads_in_set_1, reads_in_set_2)) for i in range(num_processes)]
        # processes_compare_train = [mp.Process(target=parse_linclust, args=(os.path.join(input_dir, f'linclust-subset-{i}'), os.path.join(input_dir, f'linclust-subset-{i}'))) for i in range(num_processes)]
        # for p in processes_compare_train:
        #     p.start()
        # for p in processes_compare_train:
        #     p.join()
        # print('compare linclust output to training set')
        # start = get_time(start, datetime.datetime.now())
        # print('number of reads in training set with reference read in validation set')
        # get_num_reads(reads_of_interest_set_1, set_1_name)
        # print('number of reads in validation set with reference read in training set')
        # get_num_reads(reads_of_interest_set_2, set_2_name)
        # print('number of reads in training set with reference read also in training set')
        # get_num_reads(reads_in_set_1, set_1_name)
        # print('number of reads in validation set with reference read also in validation set')
        # get_num_reads(reads_in_set_2, set_2_name)
        # # val_set = get_read_ids(val_files)
        # # print(f'get validation set - {len(val_set)}')
        # end = get_time(start, datetime.datetime.now())
        # verify that reads of interest are part of the validation sets
        # processes_compare_val_1 = [mp.Process(target=verify_reads_v1, args=(reads_of_interest_train[key], val_set, train_set, os.path.join(input_dir, f'linclust-results-subset-{key}-train'))) for key in reads_of_interest_train.keys()]
        # for p in processes_compare_val_1:
        #     p.start()
        # for p in processes_compare_val_1:
        #     p.join()
        # print('check reads in validation set - 1')
        # start = get_time(start, datetime.datetime.now())
        # # verify that reads of interest are part of the validation sets
        # processes_compare_val_2 = [mp.Process(target=verify_reads_v2, args=(value, val_set, train_set, os.path.join(input_dir, f'linclust-results-subset-{key}-val'))) for key, value in reads_of_interest_val.items()]
        # for p in processes_compare_val_2:
        #     p.start()
        # for p in processes_compare_val_2:
        #     p.join()
        # print('check reads in validation set - 2')
        # end = get_time(start, datetime.datetime.now())


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
