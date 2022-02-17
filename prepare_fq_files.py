import sys
import os
import math
import gzip


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
    fq_file = sys.argv[1]
    output_dir = sys.argv[2]
    n_reads = sys.argv[3]
    # get list of reads
    reads = get_reads(fq_file)
    # split list of fq file if
    if len(reads) > n_reads:
        out_filename = os.path.join(output_dir, ''.join(fq_file.split('/')[-1].split('.')[0]))
        split_fq_file(reads, out_filename, n_reads)

if __name__ == "__main__":
    main()
