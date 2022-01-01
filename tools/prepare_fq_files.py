import os
import argparse
import sys
import math

def split_fq_file(args):
    with open(args.input_fastq, 'r') as infile:
        content = infile.readlines()
        reads = [''.join(content[i:i+4]) for i in range(0, len(content), 4)]
        reads_in_fq = []
        for i in range(0,len(reads),500000):
            reads_in_fq.append(reads[i:i+500000])
        num_fq_per_gpu = math.ceil(len(reads_in_fq)/args.num_gpus)
        print(num_fq_per_gpu, len(reads_in_fq), len(reads))
        f = open(os.path.join(args.output_dir, 'count-reads'), 'w')
        for count, i in enumerate(range(0, len(reads_in_fq), num_fq_per_gpu)):
            if not os.path.isdir(os.path.join(args.output_dir, f'tfrecords-{count}')):
                os.makedirs(os.path.join(args.output_dir, f'tfrecords-{count}'))
            num_reads = 0
            print(count, i)
            for j in range(i, i+num_fq_per_gpu, 1):
                num_reads += len(reads_in_fq[j])
                with open(os.path.join(args.output_dir, f'tfrecords-{count}' , f'testing-set-{j}.fq'), 'w') as outfile:
                    outfile.write(''.join(reads_in_fq[j]))
            print(count, num_reads)
            f.write(f'gpu {count}\t{num_reads}\n')
        f.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fastq', help="Path to the input fastq file")
    parser.add_argument('--output_dir', help="Path to store fastq files")
    parser.add_argument('--num_gpus', type=int, help="number of gpus used for classification")

    args = parser.parse_args()
    split_fq_file(args)

if __name__ == "__main__":
    main()
