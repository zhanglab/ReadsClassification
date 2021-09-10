import argparse
import multiprocess as mp
import glob
import math
import random
from utils import *

def create_test_set(args, list_genomes, label):
    with open(os.path.join(args.input_path, f'{label}-test-reads.fq') as outfile:
        for g in list_genomes:
            # write forward reads to fq file
            with open(os.path.join(args.input_path, f'{label}-{g}-fw-reads.fq')) as infile:
                outfile.write(infile.read())
            # write reverse reads to fq file
            with open(os.path.join(args.input_path, f'{label}-{g}-rv-reads.fq')) as infile:
                outfile.write(infile.read())
    return

def split_reads(in_fq_file, out_train_fq_file, out_val_fq_file):
    with open(fq_file, 'r') as infile:
        content = infile.readlines()
        reads = [''.join(content[i:i+4]) for i in range(0, len(content), 4)]
        random.shuffle(reads)
        num_train_reads = math.ceil(0.7*len(reads))
        train_reads = reads[:num_train_reads]
        val_reads = reads[num_train_reads:]
        out_train_fq_file.write(''.join(train_reads))
        out_val_fq_file.write(''.join(val_reads))
    return

def create_train_val_sets(args, list_genomes, label):
    t_outfile = open(os.path.join(args.input_path, f'{label}-train-reads.fq')
    v_outfile = open(os.path.join(args.input_path, f'{label}-val-reads.fq')
    for g in list_genomes:
        # split forward reads
        split_reads(os.path.join(args.input_path, f'{label}-{g}-fw-reads.fq'), t_outfile, v_outfile)
        # split reverse reads
        split_reads(os.path.join(args.input_path, f'{label}-{g}-rv-reads.fq'), t_outfile, v_outfile)
    return

def create_sets(args, label):
    # get genomes
    list_genomes = get_genomes(args, label)
    # divide list of genomes into training and testing
    num_train_genomes = math.ceil(0.7*len(list_genomes))
    random.shuffle(list_genomes)
    train_genomes = list_genomes[:num_train_genomes]
    test_genomes = list_genomes[num_train_genomes:]
    # create testing set
    create_test_set(args, test_genomes, label)
    # create training and validation sets
    create_train_val_sets(args, train_genomes, label)
    return

def get_genomes(args, label):
    """ returns list of genomes (fastq files) of a species """
    list_fq_files = glob.glob(os.path.join(args.input_path, f'{label}-*-fw-read.fq'))
    list_genome_id = [i.split('/')[-1].split('-')[1:3] for i in list_fq_files]
    return list_genome_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='path containing fastq files')
    args = parser.parse_args()
    # load class_mapping dictionary
    args.class_mapping = load_class_mapping(os.path.join(args.input_path, 'class_mapping.json'))
    # generate reads for each species in parallel
    with mp.Manager() as manager:
        # create new processes
        processes = [mp.Process(target=create_sets, args=(args, label)) for label in args.class_mapping.keys()]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

if __name__ == '__main__':
    main()
