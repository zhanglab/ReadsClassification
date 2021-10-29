import matplotlib as plt
import os
import sys
import argparse
import glob

def get_plot(args, m, dict_metrics):
    dict_labels = {'average mash distance': 'average mash distance', 'GC content': '%GC content', 'genome size': 'Genome size (bp)', 'training size': 'number of reads in training set'}
    index = 0 if m == 'precision' else 1
    genomes = list(dict_metrics.keys())
    list_metrics = [dict_metrics[i][index] for i in genomes]
    list_data = [args.dict_data[i] for i in genomes]
    plt.scatter(list_data, list_metrics)
    plt.xlabel(f'{dict_labels[args.parameter]}')
    plt.ylabel(f'{m}')
    plt.savefig(os.path.join(args.output_path, f'{args.dataset_type}-genomes-{m}.png'))

# def get_tsv():

def get_test_results(args):
    reports = glob.glob(os.path.join(args.input_test, 'classification-report-species-*'))
    dict_data = {}
    for r in reports:
        genome = r.split('-')[-1]
        with open(r, 'r') as infile:
            content = infile.readlines()
            for i in content[1:]:
                if float(i.rstrip().split('\t')[3]) != 0.0:
                    dict_data[genome] = [float(i.rstrip().split('\t')[1]), float(i.rstrip().split('\t')[2])]
                    break
    return dict_data


def get_mash_distances(args):
    list_mash_dist_files = glob.glob(os.path.join(args.path_mash_dist, '*-avg-mash-dist'))
    dict_data = {}
    for mash_dist_file in list_mash_dist_files:
        with open(mash_dist_file, 'r') as f:
            for line in f:
                dict_data[line.rstrip().split('\t')[0]] = float(line.rstrip().split('\t')[1])
    return dict_data

def count_GC(seq):
    num_GC = 0
    for nt in seq:
        if nt == 'G' or nt == 'C':
            num_GC += 1
    return num_GC

def get_genome_info(args):
    dict_data = {}
    with open(args.genomes, 'r') as f:
        for line in f:
            genome = line.rstrip().split('\t')[0]
            fasta_file = line.rstrip().split('\t')[1]
            data = 0
            with open(fasta_file, 'r') as infile:
                for seq in infile:
                    if seq[0] != '>':
                        if args.parameter == 'GC content':
                            data += count_GC(seq.rstrip())
                        elif args.parameter == 'genome size':
                            data += len(seq.rstrip())
            dict_data[genome] = data
    return dict_data

def get_read_num(args):
    dict_reads = {}
    with open(args.file_num_reads, 'r') as f:
        for line in f:
            line = line.rstrip()
            dict_reads[line.split('\t')[0][:-3]] = line.split('\t')[1:]
    return dict_reads

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_test', help="Path to testing results")
    parser.add_argument('--dataset_type', help="type of dataset", choices=['training', 'testing', 'validation'])
    parser.add_argument('--genomes', help="File with list of genomes ID and path to fasta files")
    parser.add_argument('--parameter', help="Parameter used for x-axis", choices=['average mash distance', 'GC content', 'genome size', 'training size', 'taxonomy'])
    parser.add_argument('--file_num_reads', help="Path to file with number of reads in datasets", required=('training size' in sys.argv))
    parser.add_argument('--path_mash_dist', help="Path to directory with files containing average mash distances", required=('mash distance' in sys.argv and 'testing' in sys.argv))
    parser.add_argument('--file_taxonomy', help="Path to file containing taxonomy information", required=('taxonomy' in sys.argv))
    args = parser.parse_args()

    if args.parameter == 'training size':
        # get number of reads
        args.dict_data = get_read_num(args)
    elif args.parameter == 'GC content' or args.parameter == 'genome size':
        # get GC content or genome size per genome
        args.dict_data = get_genome_info(args)
    elif args.parameter == 'average mash distance':
        # get average mash distance between each testing genome and all training genomes
        args.dict_data = get_mash_distances(args)

    # with open(args.genomes, 'r') as infile:
    #     for line in infile:
    #         genome = line.rstrip().split('\t')[0]
    #         GC_count, genome_size = get_GC_content(line.rstrip().split('\t')[1]))
    #         outfile.write(f'{line.rstrip()}\t{GC_count}\t{genome_size}\t{dict_num_reads[genome][0]}\t{dict_num_reads[genome][1]}\t{dict_num_reads[genome][2]}\n')

    #ranks = ['species', 'genus', 'family', 'order', 'class']
    # define output path
    args.output_path = os.path.join(args.input_test, 'png')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # get testing results
    dict_metrics = get_test_results(args)
    metrics = ['precision', 'recall']
    for m in metrics:
        get_plot(args, m, dict_metrics)



if __name__ == "__main__":
    main()
