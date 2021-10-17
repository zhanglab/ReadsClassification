""" script to process linclust tsv output file and identify the number of identical reads between the training and validation sets """
import sys
import os
from Bio import SeqIO
from collections import defaultdict

def get_reads(fq_files_loc, cluster_reads):
    # reads = {}
    with open(os.path.join(fq_files_loc, 'list_fq_files'), 'r') as f:
        for line in f:
            print(line)
            with open(os.path.join(fq_files_loc, line), "r") as handle:
            #     for rec in SeqIO.parse(handle, "fastq"):
            #         reads[rec.id] = rec.seq
                content = f.readlines()
                list_reads = [''.join(content[i:i+4]) for i in range(0, len(content), 4)]
                for read in list_reads:
                    rec = read.split('\n')
                    if rec[0] in cluster_reads:
                        print(line, rec[0], rec[1])
                    # reads[rec[0]] = str(rec[1])

    # return reads

def main():
    linclust_out = sys.argv[1]
    train_fq = sys.argv[2]
    val_fq = sys.argv[3]
    # get set of read ids in training and validation sets
    # train_reads = get_reads(list_train_fq, train_fq)
    # val_reads = get_reads(list_val_fq, val_fq)
    # print(f'Number of reads: {len(train_reads)+len(val_reads)}')
    # process linclust output file
    clusters = defaultdict(list)
    num_lines = 0
    with open(os.path.join(os.getcwd(), linclust_out), 'r') as infile:
        for line in infile:
            num_lines += 1
            # clusters[line.rstrip().split('\t')[0]].append(line.rstrip().split('\t')[1])
            # if len(clusters[line.rstrip().split("\t")[0]]) >= 3:
            #     print(f'rep: {line.rstrip().split('\t')[0]}\nreads: {clusters[line.rstrip().split("\t")[0]]}\n')
            #     get_reads(train_fq, clusters[line.rstrip().split('\t')[0]])
            #     get_reads(val_fq, clusters[line.rstrip().split('\t')[0]])
            #     # reads_in_train = set(clusters[line.rstrip().split('\t')[0]]).intersection(set(train_reads))
            #     # reads_in_val = set(clusters[line.rstrip().split('\t')[0]]).intersection(set(val_reads))
            #     # print(f'reads in training set:\n')
            #     # for r in reads_in_val:
            #     #     print(r, train_reads[r])
            #     # print(f'reads in validation set:\n')
            #     # for r in reads_in_val:
            #     #     print(r, val_reads[r])
            #
            #     break
    print(f'number of lines in tsv file: {num_lines}')

if __name__ == "__main__":
    main()
