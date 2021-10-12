""" script to process linclust tsv output file and identify the number of identical reads between the training and validation sets """
import sys
import os
from Bio import SeqIO
from collections import defaultdict

def get_reads(list_fq_files, fq_files_loc):
    reads = {}
    with open(list_fq_files, 'r') as f:
        for line in f:
            with open(os.path.join(fq_files_loc, line.rstrip()), "r") as handle:
                for rec in SeqIO.parse(handle, "fastq"):
                    reads[rec.id] = rec.seq
    return reads

def main():
    linclust_out = sys.argv[1]
    list_train_fq = sys.argv[2]
    list_val_fq = sys.argv[3]
    train_fq = sys.argv[4]
    val_fq = sys.argv[5]
    # get set of read ids in training and validation sets
    train_reads = get_reads(list_train_fq, train_fq)
    val_reads = get_reads(list_val_fq, val_fq)
    print(f'Number of reads: {len(train_reads)+len(val_reads)}')
    # process linclust output file
    clusters = defaultdict(list)
    with open(os.path.join(os.getcwd(), linclust_out), 'r') as infile:
        for line in infile:
            clusters[line.rstrip().split('\t')[0]].append(line.rstrip().split('\t')[1])
            if len(clusters[line.rstrip().split('\t')[0]]) > 1:
                rep = line.rstrip().split('\t')[0]
                reads = clusters[rep]
                print(f'rep: {rep}\nread: {reads}\n')
                reads_in_train = set(rep).intersection(set(train_reads))
                reads_in_val = set(clusters[rep]).intersection(set(val_reads))
                print(f'reads in training set:\n')
                for r in reads_in_val:
                    print(r, train_reads[r])
                print(f'reads in validation set:\n')
                for r in reads_in_val:
                    print(r, val_reads[r])
                break

if __name__ == "__main__":
    main()
