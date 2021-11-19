import os
import sys
import glob
import datetime
import numpy as np
import math
import io
import pandas as pd
from collections import defaultdict

def compare_to_sets(linclust_data, set_1, set_2, outputfile, set_1_name, set_2_name):
    num_reads_set_2 = 0 # get number of testing reads in a cluster with at least one training read
    for cluster_rep, cluster_list in linclust_data.items():
        # find clusters with reads in both sets
        reads_in_set_1 = 0
        reads_in_set_2 = 0
        for r in cluster_list:
            if r in set_1:
                reads_in_set_1 += 1
            elif r in set_2:
                reads_in_set_2 += 1
        # report the number of reads in test of interest
        if reads_in_set_2 > 0 and reads_in_set_1 > 0:
            num_reads_set_2 += len(reads_in_set_2)
    with open(outputfile, 'a') as f:
        f.write(f'number of reads in {set_2_name} identical to reads in {set_1_name}\t{num_reads_set_2}')

def parse_linclust(linclust_out):
    linclust_clusters = defaultdict(list)  # key = representative read of cluster, value = list of reads part of the cluster
    with open(linclust_out, 'r') as infile:
        for line in infile:
            read_1 = line.rstrip().split('\t')[0]
            read_2 = line.rstrip().split('\t')[1]
            linclust_data_dict[read_1].append(read_2)
    return linclust_clusters

def get_read_ids(list_fq_files):
    dataset = {}
    for fq_file in list_fq_files:
        with open(fq_file, 'r') as infile:
            content = infile.readlines()
            reads = [''.join(content[i:i+4]) for i in range(0, len(content), 4)]
            for j in reads:
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
    # filter reads in cluster results that are identical and with the same read ids
    linclust_clusters = parse_linclust(linclust_out)
    print(f'number of clusters: {len(linclust_clusters)}')
    # get reads in set 1
    set_1_files = sorted(glob.glob(os.path.join(path_set_1, '*-reads.fq')))
    print(f'Number of fastq files in set #1: {len(set_1_files)}')
    set_1 = get_read_ids(set_1_files)
    print(f'get set #1 - {len(set_1)}')
    # get reads in set 2
    set_2_files = sorted(glob.glob(os.path.join(path_set_2, '*-reads.fq')))
    print(f'Number of fastq files in set #2: {len(set_2_files)}')
    set_2 = get_read_ids(set_2_files)
    print(f'get set #2 - {len(set_2)}')
    with open(outputfile, 'w') as f:
        f.write(f'number of clusters\t{len(linclust_clusters)}\nnumber of fastq files in {set_1_name}\t{len(set_1_files)}\nnumber of fastq files in {set_2_name}\t{len(set_2_files)}\nnumber of reads in {set_1_name}\t{len(set_1)}\nnumber of reads in {set_2_name}\t{len(set_2)}\n')
    # compare reads with set # 1 and set # 2
    compare_to_sets(linclust_clusters, set_1, set_2, outputfile, set_1_name, set_2_name)

if __name__ == "__main__":
    main()
