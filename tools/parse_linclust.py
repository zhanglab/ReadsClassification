import os
import sys
import glob
import datetime
import numpy as np
import argparse
import math
import io
import pandas as pd
from collections import defaultdict

def parse_linclust(args, set_1, set_2, outputfile, dict_seq_ids):
    # linclust_clusters = defaultdict(list)  # key = representative read of cluster, value = list of reads part of the cluster
    num_clusters_single_reads = 0 # number of clusters with a unique read
    num_clusters_multiple_reads = 0 # number of clusters with multiple reads
    num_reads_in_clusters_w_multiple_reads = 0
    num_clusters_w_reads_set_2_in_set_2 = 0 # number of clusters with only testing reads (single and multiple reads clusters)
    num_clusters_w_reads_set_1_in_set_1 = 0 # number of clusters with only training reads (single and multiple reads clusters)
    num_reads_set_2_w_reads_in_set_1 = 0 # number of testing reads identical to training reads
    num_reads_set_1_w_reads_in_set_2 = 0 # number of training reads identical to testing reads
    num_reads_set_2_in_clusters_w_only_set_2 = 0 # number of testing reads in clusters containing only testing reads (single and multiple reads clusters)
    num_reads_set_1_in_clusters_w_only_set_1 = 0 # number of training reads in clusters containing only training reads (single and multiple reads clusters)
    reads_id_for_new_testing_set = []
    with open(args.linclust_out, 'r') as infile:
        content = infile.readlines()
        current_cluster = content[0].rstrip().split('\t')[0]
        reads_in_cluster = [content[0].rstrip().split('\t')[1]]
        # print(f'inital cluster: {content[0].rstrip()}')
        for i in range(1, len(content)):
            next_cluster = content[i].rstrip().split('\t')[0]
            if next_cluster == current_cluster:
                # print(f'same cluster: {content[i].rstrip()}')
                reads_in_cluster.append(content[i].rstrip().split('\t')[1])
            else:
                # print(f'different cluster: {content[i].rstrip()}')
                # check how many reads are in the current cluster
                if len(reads_in_cluster) > 1:
                    num_clusters_multiple_reads += 1
                    num_reads_in_clusters_w_multiple_reads += len(reads_in_cluster)
                    reads_in_set_1 = 0
                    reads_in_set_2 = 0
                    # check if cluster contains only testing reads
                    for r in reads_in_cluster:
                        if r in set_1:
                            reads_in_set_1 += 1
                        elif r in set_2:
                            reads_in_set_2 += 1
                    if reads_in_set_2 > 0 and reads_in_set_1 > 0:
                        num_reads_set_2_w_reads_in_set_1 += reads_in_set_2
                        num_reads_set_1_w_reads_in_set_2 += reads_in_set_1
                    elif reads_in_set_2 > 0 and reads_in_set_1 == 0:
                        num_clusters_w_reads_set_2_in_set_2 += 1
                        num_reads_set_2_in_clusters_w_only_set_2 += reads_in_set_2
                        reads_id_for_new_testing_set += reads_in_cluster
                    elif reads_in_set_2 == 0 and reads_in_set_1 > 0:
                        num_clusters_w_reads_set_1_in_set_1 += 1
                        num_reads_set_1_in_clusters_w_only_set_1 += reads_in_set_1
                else:
                    num_clusters_single_reads += 1
                    # check if single read is in testing set
                    if reads_in_cluster[0] in set_2:
                        num_clusters_w_reads_set_2_in_set_2 += 1
                        num_reads_set_2_in_clusters_w_only_set_2 += 1
                        reads_id_for_new_testing_set.append(reads_in_cluster[0])
                    elif reads_in_cluster[0] in set_1:
                        num_clusters_w_reads_set_1_in_set_1 += 1
                        num_reads_set_1_in_clusters_w_only_set_1 += 1
                # update the current cluster
                current_cluster = next_cluster
                reads_in_cluster = [content[i].rstrip().split('\t')[1]]

    # get read sequences for new updated testing set
    # reads_seq_for_testing_set = [reads_for_new_testing_set[i:i+25000000] for i in range(0, len(reads_for_new_testing_set), 25000000)]
    # print(f'number of fastq files: {len(reads_seq_for_testing_set)} - {float(len(reads_for_new_testing_set))/25000000}')


    # save results of parsing
    with open(outputfile, 'a') as f:
        f.write(f'number of clusters with a single read:\t{num_clusters_single_reads}\nnumber of clusters with multiple reads:\t{num_clusters_multiple_reads}\nnumber of reads in clusters with multiple reads:\t{num_reads_in_clusters_w_multiple_reads}\nnumber of clusters with only testing reads:\t{num_clusters_w_reads_set_2_in_set_2}\nnumber of clusters with only training reads:\t{num_clusters_w_reads_set_1_in_set_1}\nnumber of testing reads identical to training reads:\t{num_reads_set_2_w_reads_in_set_1}\t{(float(num_reads_set_2_w_reads_in_set_1)/len(set_2))*100}\nnumber of training reads identical to testing reads:\t{num_reads_set_1_w_reads_in_set_2}\nnumber of testing reads in clusters containing only testing reads:\t{num_reads_set_2_in_clusters_w_only_set_2}\t{(float(num_reads_set_2_in_clusters_w_only_set_2)/len(set_2))*100}\nnumber of testing reads in new updated testing set:\t{len(reads_id_for_new_testing_set)}\nnumber of training reads in clusters containing only training reads:\t{num_reads_set_1_in_clusters_w_only_set_1}\n')

    # save new testing set to fastq files for testing (one fastq file per testing genome)
    for i in range(len(reads_id_for_new_testing_set)):
        # retrieve seq id and genome
        seq_id = reads_id_for_new_testing_set[i].split('-')[0]
        genome_output_fq_file = os.path.join(args.input_dir, 'fq_files_genomes', f'{dict_seq_ids[seq_id]}-reads.fq')
        with open(genome_output_fq_file, 'a') as new_fq:
            new_fq.write(set_2[reads_id_for_new_testing_set[i]])

    #     with open(os.path.join(args.input_dir, f'updated-testing-set-{i}.fq'), 'w') as f:
    #         reads = [set_2[j] for j in reads_seq_for_testing_set[i]]
    #         print(f'subset {i} - # reads: {len(reads)}')
    #         for k in range(len(reads)):
    #             f.write(f'{reads[k]}')


def get_read_ids(list_fq_files, set_type):
    dataset = {}
    for fq_file in list_fq_files:
        with open(fq_file, 'r') as infile:
            content = infile.readlines()
            reads = [''.join(content[i:i+4]) for i in range(0, len(content), 4)]
            for j in reads:
                if set_type == 'testing':
                    dataset[j.split('\n')[0].rstrip()[1:]] = j
                else:
                    dataset[j.split('\n')[0].rstrip()[1:]] = j.split('\n')[1].rstrip()
    return dataset

def get_seq_ids(args):
    dict_seq_ids = {}
    with open(args.seq_ids, 'r') as f:
        for line in f:
            dict_seq_ids[line.rstrip().split('\t')[1]] = line.rstrip().split('\t')[0]
    return dict_seq_ids

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='path to input files')
    parser.add_argument('--path_set_1', type=str, help='path to fastq files from set 1')
    parser.add_argument('--path_set_2', type=str, help='path to fastq files from set 2')
    parser.add_argument('--set_1_name', type=str, help='name of set 1')
    parser.add_argument('--set_2_name', type=str, help='name of set 2')
    parser.add_argument('--linclust_out', type=str, help='path to linclust output')
    parser.add_argument('--seq_ids', type=str, help='path to file containing list of sequence id')
    args = parser.parse_args()
    # create output file
    outputfile = os.path.join(args.input_dir, f'{args.set_1_name}-{args.set_2_name}-linclust-output-parsing')
    # get reads in set 1
    set_1_files = sorted(glob.glob(os.path.join(args.path_set_1, '*-reads.fq')))
    print(f'Number of fastq files in set #1: {len(set_1_files)}')
    set_1 = get_read_ids(set_1_files, args.set_1_name)
    print(f'get set #1 - {len(set_1)}')
    # get reads in set 2
    set_2_files = sorted(glob.glob(os.path.join(args.path_set_2, '*-reads.fq')))
    print(f'Number of fastq files in set #2: {len(set_2_files)}')
    set_2 = get_read_ids(set_2_files, args.set_2_name)
    print(f'get set #2 - {len(set_2)}')
    with open(outputfile, 'w') as f:
        f.write(f'number of fastq files in {args.set_1_name} set:\t{len(set_1_files)}\nnumber of fastq files in {args.set_2_name} set:\t{len(set_2_files)}\nnumber of reads in {args.set_1_name} set:\t{len(set_1)}\nnumber of reads in {args.set_2_name} set:\t{len(set_2)}\n')
    # get list of sequence ids
    dict_seq_ids = get_seq_ids(args)
    # filter reads in cluster results that are identical and with the same read ids
    parse_linclust(args, set_1, set_2, outputfile, dict_seq_ids)

if __name__ == "__main__":
    main()
