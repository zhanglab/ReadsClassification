import os
import sys
import glob
import datetime
import numpy as np
import math
import io
import pandas as pd
from collections import defaultdict

def parse_linclust(linclust_out, set_1, set_2, outputfile, input_dir):
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
    reads_for_new_testing_set = []
    with open(linclust_out, 'r') as infile:
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
                        reads_for_new_testing_set += reads_in_cluster
                    elif reads_in_set_2 == 0 and reads_in_set_1 > 0:
                        num_clusters_w_reads_set_1_in_set_1 += 1
                        num_reads_set_1_in_clusters_w_only_set_1 += reads_in_set_1
                else:
                    num_clusters_single_reads += 1
                    # check if single read is in testing set
                    if reads_in_cluster[0] in set_2:
                        num_clusters_w_reads_set_2_in_set_2 += 1
                        num_reads_set_2_in_clusters_w_only_set_2 += 1
                        reads_for_new_testing_set.append(reads_in_cluster[0])
                    elif reads_in_cluster[0] in set_1:
                        num_clusters_w_reads_set_1_in_set_1 += 1
                        num_reads_set_1_in_clusters_w_only_set_1 += 1
                # update the current cluster
                current_cluster = next_cluster
                reads_in_cluster = [content[i].rstrip().split('\t')[1]]

    # get read sequences for new updated testing set
    reads_seq_for_testing_set = [set_2[i] for i in reads_for_new_testing_set]

    # save results of parsing
    with open(outputfile, 'a') as f:
        f.write(f'number of clusters with a single read:\t{num_clusters_single_reads}\nnumber of clusters with multiple reads:\t{num_clusters_multiple_reads}\nnumber of reads in clusters with multiple reads:\t{num_reads_in_clusters_w_multiple_reads}\nnumber of clusters with only testing reads:\t{num_clusters_w_reads_set_2_in_set_2}\nnumber of clusters with only training reads:\t{num_clusters_w_reads_set_1_in_set_1}\nnumber of testing reads identical to training reads:\t{num_reads_set_2_w_reads_in_set_1}-{(float(num_reads_set_2_w_reads_in_set_1)/len(set_2))*100}\nnumber of training reads identical to testing reads:\t{num_reads_set_1_w_reads_in_set_2}\nnumber of testing reads in clusters containing only testing reads:\t{num_reads_set_2_in_clusters_w_only_set_2}-{(float(num_reads_set_2_in_clusters_w_only_set_2)/len(set_2))*100}\nnumber of testing reads in new updated testing set:\t{len(reads_for_new_testing_set)}\nnumber of training reads in clusters containing only training reads:\t{num_reads_set_1_in_clusters_w_only_set_1}\n')

    # save new testing set to fastq file
    with open(os.path.join(input_dir, 'updated-testing-set.fq'), 'w') as f:
        for i in range(len(reads_seq_for_testing_set)):
            f.write(f'{reads_seq_for_testing_set[i]}')


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

def main():
    input_dir = sys.argv[1]
    path_set_1 = sys.argv[2]
    set_1_name = sys.argv[3]
    path_set_2 = sys.argv[4]
    set_2_name = sys.argv[5]
    linclust_out = sys.argv[6]
    # create output file
    outputfile = os.path.join(input_dir, f'{set_1_name}-{set_2_name}-linclust-output-parsing')
    # get reads in set 1
    set_1_files = sorted(glob.glob(os.path.join(path_set_1, '*-reads.fq')))
    print(f'Number of fastq files in set #1: {len(set_1_files)}')
    set_1 = get_read_ids(set_1_files, set_1_name)
    print(f'get set #1 - {len(set_1)}')
    # get reads in set 2
    set_2_files = sorted(glob.glob(os.path.join(path_set_2, '*-reads.fq')))
    print(f'Number of fastq files in set #2: {len(set_2_files)}')
    set_2 = get_read_ids(set_2_files, set_2_name)
    print(f'get set #2 - {len(set_2)}')
    with open(outputfile, 'w') as f:
        f.write(f'number of fastq files in {set_1_name} set:\t{len(set_1_files)}\nnumber of fastq files in {set_2_name} set:\t{len(set_2_files)}\nnumber of reads in {set_1_name} set:\t{len(set_1)}\nnumber of reads in {set_2_name} set:\t{len(set_2)}\n')
    # filter reads in cluster results that are identical and with the same read ids
    parse_linclust(linclust_out, set_1, set_2, outputfile, input_dir)

if __name__ == "__main__":
    main()
