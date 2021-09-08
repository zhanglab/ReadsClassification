import os
import sys
import argparse
import glob
import random
import math

def write_to_fq(args, reads, type_set):
    if len(reads.splitlines()) % 4 != 0:
        sys.exit(f'incorrect fastq file - odd # lines: {len(reads.splitlines())}')
    with open(os.path.join(args.output_dir, f'{args.species_label}-{type_set}.fq'), 'a') as outfile:
        outfile.write(reads)

def get_reads(args, fastq_file, type_set):
    with open(fastq_file, 'r') as infile:
        content = infile.readlines()
        reads = [''.join(content[i:i+4]) for i in range(0, len(content), 4)]
        if type_set == 'training':
            random.shuffle(reads)
            # divide sequences into training and validation sets
            num_train = math.ceil(len(reads)*0.7)
            train_data = ''.join(reads[:num_train])
            val_data = ''.join(reads[num_train:])
            write_to_fq(args, train_data, 'training')
            write_to_fq(args, val_data, 'validation')
        else:
            return reads

def create_train_val_sets(args, train_genomes):
    for genome in train_genomes:
        # split forward and reverse reads between training and validation sets
        get_reads(args, os.path.join(args.input_dir, f'{args.species_label}-{genome}-fw-reads.fq', 'training')
        get_reads(args, os.path.join(args.input_dir, f'{args.species_label}-{genome}-rv-reads.fq', 'training')

def create_test_set(args, test_genomes):
    for genome in test_genomes:
        fw_reads = get_reads(args, os.path.join(args.input_dir, f'{args.species_label}-{genome}-fw-reads.fq', 'testing')
        rv_reads = get_reads(args, os.path.join(args.input_dir, f'{args.species_label}-{genome}-rv-reads.fq', 'testing')
        write_to_fq(args, fw_reads, 'testing')
        write_to_fq(args, rv_reads, 'testing')

def create_sets(args, genomes):
    random.shuffle(genomes)
    # define number of genomes in the training (+ validation sets)
    num_train_genomes = math.ceil(len(genomes)*0.7)
    # define training and testing sets
    train_genomes = genomes[:num_train_genomes]
    test_genomes = genomes[num_train_genomes:]

    create_train_val_sets(args, train_genomes)
    create_test_set(args, test_genomes)

def get_genomes(args):
    # get all fastq files of forward reads in the input directory
    list_fw_fastq_files = glob.glob(os.path.join(args.input_dir, f'{args.species_label}-*fw-reads.fq'))
    # extract genome IDs from fastq filenames
    genomes = ['-'.join(i.split('-')[1:3]) for i in list_fw_fastq_files]
    return genomes

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_dir', type=str, help='path to fastq files')
    parser.add_argument('-species_label', type=str, gelp='label ID of species')
    args = parser.parser_args()
    # define and create output path
    args.output_dir = os.path.join(args.input_dir, f'{args.species_label}-sets')
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    # get genomes of species with label args.species_label
    genomes = get_genomes(args)
    # create training, validation and testing sets
    create_sets(args, genomes)

if __name__ == "__main__":
    main()
