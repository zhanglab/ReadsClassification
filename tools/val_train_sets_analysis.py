""" script to process linclust tsv output file and identify the number of identical reads between the training and validation sets """
import sys
import os
from collections import defaultdict

def get_sets(fq_files_loc, cluster_reads):
    dataset = defaultdict(list) # key = subset, value = list of reads id and sequence in cluster
    with open(os.path.join(fq_files_loc, 'list_fq_files'), 'r') as f:
        for line in f:
            with open(os.path.join(fq_files_loc, line.rstrip()), "r") as handle:
                content = handle.readlines()
                list_reads = [''.join(content[i:i+4]) for i in range(0, len(content), 4)]
                for read in list_reads:
                    rec = read.split('\n')
                    if rec[0] in cluster_reads:
                        dataset[line.rstrip()].append([rec[0], rec[1]])
    return dataset

def main():
    linclust_out = sys.argv[1]
    train_fq = sys.argv[2]
    val_fq = sys.argv[3]

    # define output file
    outfile = os.path.join(os.getcwd(), '-'.join(['linclust-analysis-x', linclust_out.split('x')[1]]))
    print(outfile, linclust_out)
    # process linclust output file
    clusters = defaultdict(list)
    with open(os.path.join(os.getcwd(), linclust_out), 'r') as infile:
        for line in infile:
            clusters[line.rstrip().split('\t')[0]].append(line.rstrip().split('\t')[1])

    # check whether reads are part of the validation or the training set for clusters with more than 1 read
    for rep_cluster, reads_cluster in clusters.items():
        if len(reads_cluster) > 1:
            # get training sets and validations sets in which those reads can be found
            training_sets = get_sets(train_fq, reads_cluster)
            validation_sets = get_sets(val_fq, reads_cluster)
            # write output to file if there are reads in both the training and validation sets
            if len(training_sets) > 0 and len(validation_sets) > 0:
                with open(outfile, 'a') as handle:
                    handle.write(f'training sets:\t{len(training_sets)}\n')
                    for key, value in training_sets.items():
                        for i in range(len(value)):
                            handle.write(f'{key}\t{value[i][0]}\t{value[i][1]}\n')

                    handle.write(f'validation sets:\t{len(validation_sets)}\n')
                    for key, value in validation_sets.items():
                        for i in range(len(value)):
                            handle.write(f'{key}\t{value[i][0]}\t{value[i][1]}\n')

if __name__ == "__main__":
    main()
