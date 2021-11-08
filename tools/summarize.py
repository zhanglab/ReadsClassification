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
from collections import defaultdict

def get_tsv(args, m, dict_metrics, index):
    genomes = list(dict_metrics.keys())
    list_metrics = [dict_metrics[i][index] for i in genomes]
    list_data = [args.dict_data[i] for i in genomes]
    # create tsv file to store results for R plots
    tsv_filename = os.path.join(args.output_path, f'{args.dataset_type}-genomes-{m}.tsv')
    with open(tsv_filename, 'w') as f:
        wr = csv.writer(f, delimiter='\t')
        wr.writerow(['genomes', m, args.dict_labels[args.parameter], 'class', 'order', 'family', 'genus', 'species'])
        for i in range(len(genomes)):
            taxonomy = args.taxonomy[genomes[i]]
            wr.writerow([genomes[i], list_metrics[i], list_data[i], taxonomy[4].split('__')[1], taxonomy[3].split('__')[1],
            taxonomy[2].split('__')[1], taxonomy[1].split('__')[1], taxonomy[0].split('__')[1]])

def get_plot_taxon_level(args, taxon, genomes, taxon_metrics, taxon_data, color, m, r_name):
    figname = os.path.join(args.output_path, r_name, f'{args.dataset_type}-genomes-{m}-{r_name}-{"-".join(taxon.split(" "))}.png')
    plt.clf()
    fig, ax = plt.subplots()
    ax.scatter(taxon_data, taxon_metrics, color=color, label=taxon)
    plt.xlabel(f'{args.dict_labels[args.parameter]}')
    plt.ylabel(f'{m}')
    fig.savefig(figname, bbox_inches='tight')

def get_plot_rank_level(args, m, r_name, r_index, dict_metrics, index):
    # get taxa
    taxa_rank = {genome: genome_taxonomy[int(r_index)].split('__')[1] for genome, genome_taxonomy in args.taxonomy.items()}
    # get the taxa with 2 or more genomes
    dict_taxa = defaultdict(list)
    for g, t in taxa_rank.items():
        dict_taxa[t].append(g)
    taxa = [t for t, list_g in dict_taxa.items() if len(list_g) >= 2]
    # define colors
    random_colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
         for i in range(len(taxa))]
    rank_colors = dict(zip(list(taxa), random_colors))
    # define name of figures
    figname = os.path.join(args.output_path, r_name, f'{args.dataset_type}-genomes-{m}-{r_name}.png')
    legendname = os.path.join(args.output_path, r_name, f'{args.dataset_type}-genomes-{m}-{r_name}-legend.png')
    # generate a plot for each taxon
    for taxon, color in rank_colors.items():
        # get testing genomes with given taxon
        genomes = [genome_id for genome_id, genome_taxon in taxa_rank.items() if genome_taxon == taxon]
        taxon_metrics = [dict_metrics[g][index] for g in genomes]
        taxon_data = [args.dict_data[g] for g in genomes]
        get_plot_taxon_level(args, taxon, genomes, taxon_metrics, taxon_data, color, m, r_name)

    # generate plot
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for taxon, color in rank_colors.items():
        # get testing genomes with given taxon
        genomes = [genome_id for genome_id, genome_taxon in taxa_rank.items() if genome_taxon == taxon]
        taxon_metrics = [dict_metrics[g][index] for g in genomes]
        taxon_data = [args.dict_data[g] for g in genomes]
        ax.scatter(taxon_data, taxon_metrics, color=color, label=taxon)
    plt.xlabel(f'{args.dict_labels[args.parameter]}')
    plt.ylabel(f'{m}')
    fig.savefig(figname, bbox_inches='tight')
    fig_legend = plt.figure()
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
    # sort dictionary by value
    dict_data_sorted = {k: v for k, v in sorted(dict_data.items(), key=lambda item: item[1])}
    return dict_data_sorted

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
    args.dict_labels = {'average mash distance': 'average mash distance to training genomes', 'GC content': '%GC content in testing genome', 'genome size': 'Genome size (bp) of testing genome', 'training size': 'number of reads in training set', 'training genomes': 'number of training genomes'}
    ranks = {'0': 'species', '1': 'genus', '2': 'family', '3': 'order', '4': 'class'}
    # define output path
    args.output_path = os.path.join(args.input_test, 'results', '-'.join(args.parameter.split(' ')))
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    # get testing results
    dict_metrics = get_test_results(args)
    metrics = ['precision', 'recall']
    for m in metrics:
        index = 0 if m == 'precision' else 1
        # generate tsv file with information
        get_tsv(args, m, dict_metrics, index)
        # generate a plot for each taxonomic level
        for r_index, r_name in ranks.items():
            get_plot_rank_level(args, m, r_name, r_index, dict_metrics, index)
    # get statistics on data
    get_stats(args)




if __name__ == "__main__":
    main()
