import os
import argparse
import sys
import math

def split_fq_file(args):
    num_reads_per_fq = 500000
    with open(args.input_fastq, 'r') as infile:
        content = infile.readlines()
        reads = [''.join(content[i:i+4]) for i in range(0, len(content), 4)]
        reads_in_fq = [''.join(reads[i:i+num_reads_per_fq]) for i in range(0, len(reads), num_reads_per_fq)]
        num_fq_per_gpu = math.ceil(len(reads_in_fq)/args.num_gpus)
        print(num_fq_per_gpu, len(reads_in_fq), len(reads))
        for i in range(1, args.num_gpus+1, 1):
            if not os.path.isdir(os.path.join(args.output_dir, f'tfrecords-{i}')):
                os.makedirs(os.path.join(args.output_dir, f'tfrecords-{i}'))
            f = open(os.path.join(args.output_dir, f'tfrecords-{i}', 'count-reads'), 'w')
            for j in range(num_fq_per_gpu):
                with open(os.path.join(args.output_dir, f'tfrecords-{i}' , f'testing-set-{j}.fq'), 'w') as outfile:
                    outfile.write(reads_in_fq[i+j-1])
                    f.write(f'testing-set-{j}.fq\t{len(reads_in_fq[i+j-1])}\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fastq', help="Path to the input fastq file")
    parser.add_argument('--output_dir', help="Path to store fastq files")
    parser.add_argument('--num_gpus', type=int, help="number of gpus used for classification")

    args = parser.parse_args()
    split_fq_file(args)

if __name__ == "__main__":
    main()
