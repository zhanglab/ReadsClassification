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

def compare_to_sets(linclust_data, set_1, set_2, outfilename_1, outfilename_2, outfilename_3, set_1_name, set_2_name):
    outfile_1 = open(outfilename_1, 'w') # store reads that are in training set but cluster is not
    outfile_2 = open(outfilename_2, 'w') # store reads that are not in training set but cluster is
    outfile_3 = open(outfilename_3, 'w')
    roi_1 = 0
    roi_2 = 0
    r_both_in_set_1 = 0
    r_both_in_set_2 = 0
    for read, cluster in linclust_data.items():
        if read in set_1 and cluster in set_2:
            outfile_1.write(f'{cluster}\t{set_2[cluster]}\t{read}\t{set_1[read]}\t')
            roi_1 += 1
        elif read in set_2 and cluster in set_1:
            outfile_2.write(f'{cluster}\t{set_1[cluster]}\t{read}\t{set_2[read]}\t')
            roi_2 += 1
        elif read in set_1 and cluster in set_1:
            r_both_in_set_1 += 1
        elif read in set_2 and cluster in set_2:
            r_both_in_set_2 += 1
    print(f'# pair of reads in roi_1: {roi_1}')
    print(f'# pair of reads in roi_2: {roi_2}')
    print(f'# pair of reads in r_both_in_set_1: {r_both_in_set_1}')
    print(f'# pair of reads in r_both_not_in_set_1: {r_both_in_set_2}')
    outfile_3.write(f'set 1: {set_1_name}\nset 2: {set_2_name}\nnumber of reads with identical sequences: {len(linclust_data)}\n# reads in set 1: {len(set_1)}\n# reads in set 2: {len(set_2)}\n# pair of reads in roi_1: {roi_1}\n# pair of reads in roi_2: {roi_2}\n# pair of reads in r_both_in_set_1: {r_both_in_set_1}\n# pair of reads in r_both_not_in_set_1: {r_both_in_set_2}\n')

def parse_linclust(linclust_out):
    # outfile = open(outfilename, 'w')
    linclust_data_dict = {}
    with open(linclust_out, 'r') as infile:
        for line in infile:
            read_1 = line.rstrip().split('\t')[0]
            read_2 = line.rstrip().split('\t')[1]
            if read_1 != read_2:
                linclust_data_dict[read_2] = read_1
                # outfile.write(f'{read_1}\t{read_2}\n')
            else:
                continue
    return linclust_data_dict

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

    # filter reads in cluster results that are identical and with the same read ids
    linclust_data_dict = parse_linclust(linclust_out)
    print(f'number of reads with identical sequences: {len(linclust_data_dict)}')
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
    # compare reads with set # 1 and set # 2
    compare_to_sets(linclust_data_dict, set_1, set_2, os.path.join(input_dir, f'linclust-reads-in-{set_1_name}'), os.path.join(input_dir, f'linclust-reads-in-{set_2_name}', os.path.join(input_dir, f'linclust-summary-{set_1_name}-{set_2_name}', set_1_name, set_2_name)))

if __name__ == "__main__":
    main()
