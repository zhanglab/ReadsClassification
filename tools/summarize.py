import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import statistics
import os
import sys
import numpy as np
import random
import argparse
import glob
import pandas as pd
import json
import csv

def get_plot(args, m, r, dict_metrics):
    dict_labels = {'average mash distance': 'average mash distance to training genomes', 'GC content': '%GC content in testing genome', 'genome size': 'Genome size (bp) of testing genome', 'training size': 'number of reads in training set', 'training genomes': 'number of training genomes'}
    index = 0 if m == 'precision' else 1
    # define x and y data
    genomes = list(dict_metrics.keys())
    list_metrics = [dict_metrics[i][index] for i in genomes]
    list_data = [args.dict_data[i] for i in genomes]

    # assign colors
    if r is not None:
        # define colors
        taxa = set(args.taxa_rank.values())
        random_colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(len(taxa))]
        rank_colors = dict(zip(list(taxa), random_colors))
        colors = [rank_colors[args.taxa_rank[i]] for i in genomes]
        labels = [args.taxa_rank[i] for i in genomes]
        figname = os.path.join(args.output_path, f'{args.dataset_type}-genomes-{m}-{r}.png')
        legendname = os.path.join(args.output_path, f'{args.dataset_type}-genomes-{m}-{r}-legend.png')
        # create tsv file to store results for R plots
        tsv_filename = os.path.join(args.output_path, f'{args.dataset_type}-genomes-{m}-{r}.tsv')
        with open(tsv_filename, 'w') as f:
            wr = csv.writer(f, delimiter='\t')
            wr.writerow(['genomes', m, dict_labels[args.parameter], 'taxa'])
            for i in range(len(genomes)):
                wr.writerow([genomes[i], list_metrics[i], list_data[i], args.taxa_rank[genomes[i]]])
        # report testing genomes with unusual results (average mash distance to training genomes of 0 and low recall)
    else:
        colors = 'black'
        figname = os.path.join(args.output_path, f'{args.dataset_type}-genomes-{m}.png')
        legendname = os.path.join(args.output_path, f'{args.dataset_type}-genomes-{m}-legend.png')

    # generate plot
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(list_data, list_metrics, color=colors, label=labels)
    plt.xlabel(f'{dict_labels[args.parameter]}')
    plt.ylabel(f'{m}')
    fig.savefig(figname, bbox_inches='tight')
    fig_legend = plt.figure(10,13)
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.legend(*ax.get_legend_handles_labels(), loc='center')
    ax_legend.axis('off')
    fig_legend.savefig(legendname, bbox_inches='tight')

def get_stats(args):
    with open(os.path.join(args.output_path,f'{args.dataset_type}-stats'), 'a') as f:
        f.write(f'data:\t{args.parameter}\n')
        f.write(f'mean:\t{statistics.mean(list(args.dict_data.values()))}\nmedian\t{statistics.median(list(args.dict_data.values()))}\nmin\t{min(list(args.dict_data.values()))}\nmax\t{max(list(args.dict_data.values()))}\n')

def get_taxonomy(args):
    # load gtdb information
    gtdb_df = pd.read_csv(args.gtdb_info, delimiter='\t', low_memory=False)
    # get species in database
    taxonomy = [i.split(';') for i in list(gtdb_df['gtdb_taxonomy'])]
    # get genome accession ids
    accession_id_list = [i[3:] for i in list(gtdb_df.accession)]
    # get taxonomy
    genomes_taxonomy = {accession_id_list[i]: taxonomy[i][::-1] for i in range(len(accession_id_list)) if accession_id_list[i] in args.dict_data}
    return genomes_taxonomy

def get_test_results(args):
    reports = glob.glob(os.path.join(args.input_test, 'classification-report-species-*'))
    dict_data = {}
    for r in reports:
        genome = r.split('-')[-1]
        with open(r, 'r') as infile:
            content = infile.readlines()
            for line in content[1:-1]:
                if float(line.rstrip().split('\t')[3]) != 0.0:
                    dict_data[genome] = [float(line.rstrip().split('\t')[1]), float(line.rstrip().split('\t')[2])]
                    break
    return dict_data

def get_mash_distances(args):
    list_mash_dist_files = glob.glob(os.path.join(args.path_mash_dist, '*-avg-mash-dist'))
    dict_data = {}
    for mash_dist_file in list_mash_dist_files:
        with open(mash_dist_file, 'r') as f:
            content = f.readlines()
            msh_dist = float(content[0].split('\t')[1])
            genome = content[0].split('\t')[0]
            dict_data[genome] = msh_dist
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
    parser.add_argument('--parameter', help="Parameter used for x-axis", choices=['average mash distance', 'GC content', 'genome size', 'training size', 'training genomes', 'taxonomy'])
    parser.add_argument('--file_num_reads', help="Path to file with number of reads in datasets", required=('training size' in sys.argv))
    parser.add_argument('--path_mash_dist', help="Path to directory with files containing average mash distances", required=('mash distance' in sys.argv and 'testing' in sys.argv))
    parser.add_argument('--gtdb_info', help="File with information on GTDB database", required=('taxonomy' in sys.argv or 'training size' in sys.argv))
    # parser.add_argument('--class_mapping_file', help="File containing json dictionary with labels mapped to species", required=('taxonomy'))
    args = parser.parse_args()
    if args.parameter == 'training size':
        # load dictionary mapping species to labels
        # with open(args.class_mapping_file) as f:
        #     args.class_mapping = json.load(f)
        # get number of reads
        args.dict_data = get_read_num(args)
    elif args.parameter == 'GC content' or args.parameter == 'genome size':
        # get GC content or genome size per genome
        args.dict_data = get_genome_info(args)
    elif args.parameter == 'average mash distance':
        # get average mash distance between each testing genome and all training genomes
        args.dict_data = get_mash_distances(args)
    # get taxonomy
    args.taxonomy = get_taxonomy(args)
    # with open(args.genomes, 'r') as infile:
    #     for line in infile:
    #         genome = line.rstrip().split('\t')[0]
    #         GC_count, genome_size = get_GC_content(line.rstrip().split('\t')[1]))
    #         outfile.write(f'{line.rstrip()}\t{GC_count}\t{genome_size}\t{dict_num_reads[genome][0]}\t{dict_num_reads[genome][1]}\t{dict_num_reads[genome][2]}\n')

    ranks = {'0': 'species', '1': 'genus', '2': 'family', '3': 'order', '4': 'class'}
    # define output path
    args.output_path = os.path.join(args.input_test, 'results', '-'.join(args.parameter.split(' ')))
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    # get testing results
    dict_metrics = get_test_results(args)
    metrics = ['precision', 'recall']
    for m in metrics:
        for r_index, r_name in ranks.items():
            # get taxa
            args.taxa_rank = {genome: genome_taxonomy[int(r_index)].split('__')[1] for genome, genome_taxonomy in args.taxonomy.items()}
            get_plot(args, m, r_name, dict_metrics)
    # get statistics on data
    get_stats(args)




if __name__ == "__main__":
    main()
