import sys
import os
import math
import gzip
import argparse

def split_fq_file(reads, out_filename, n_reads):
    # compute number of output fq files
    n_files = math.ceil(len(reads)/n_reads)
    for i in range(n_files):
        start = i*n_reads
        end = (i*n_reads) + n_reads if i < n_files - 1 else (i*n_reads) + len(reads)
        with gzip.open('-'.join([out_filename, f'{i}.fastq.gz']), 'wt') as outfile:
            outfile.write(''.join(reads[start:end]))


def get_reads(fq_file):
    with gzip.open(fq_file, "rt") as infile:
        content = infile.readlines()
        reads = [''.join(content[i:i+4]) for i in range(0, len(content), 4)]
    return reads

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fq_file', help="Path to the input fastq file")
    parser.add_argument('--output_dir', help="Path to the output directory")
    parser.add_argument('--n_reads', default=500000, type=int, help="number of reads per fastq file")
    args = parser.parse_args()
    # get list of reads
    reads = get_reads(args.fq_file)
    # split list of fq file if
    if len(reads) > args.n_reads:
        out_filename = os.path.join(args.output_dir, ''.join(fq_file.split('/')[-1].split('.')[0]))
        split_fq_file(reads, out_filename, args.n_reads)

if __name__ == "__main__":
    main()
