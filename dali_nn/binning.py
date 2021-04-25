import json
import os
import sys
import argparse
import statistics
import itertools
import numpy as np
from random import randint
from Bio import SeqIO
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

def summarize_cs(args, class_conf_scores_R1, class_conf_scores_R2):
    for class_id, class_name in class_mapping.items():
        with open(os.path.join(args.output_path, f'{args.sample}_model{args.model_num}_epoch{args.epoch_num}_cs_stats_per_class_{args.threshold}.txt'), 'a') as f:
            f.write(f'{class_id}\t{class_name}\n'
                    f'max fw/rv: {max(class_conf_scores_R1[class_id])}\t{max(class_conf_scores_R2[class_id])}\t'
                    f'min fw/rv: {min(class_conf_scores_R1[class_id])}\t{min(class_conf_scores_R2[class_id])}\t'
                    f'mean fw/rv: {statistics.mean(class_conf_scores_R1[class_id])}\t{statistics.mean(class_conf_scores_R2[class_id])}\t'
                    f'median fw/rv: {statistics.median(class_conf_scores_R1[class_id])}\t{statistics.median(class_conf_scores_R2[class_id])}\t'
                    f'standard deviation fw/rv: {statistics.stdev(class_conf_scores_R1[class_id])}\t{statistics.stdev(class_conf_scores_R2[class_id])}\n'
                    )

    taxa = [args.class_mapping[str(i)] for i in range(len(args.class_mapping))]
    conf_scores_R1 = [statistics.mean(class_conf_scores_R1[str(i)]) for i in range(len(args.class_mapping))]
    conf_scores_R2 = [statistics.mean(class_conf_scores_R2[str(i)]) for i in range(len(args.class_mapping))]
    x_pos = np.arange(len(axa))
    width = 0.35
    print(x_pos)
    print(taxa)
    plt.clf()
    fig, ax = plt.subplots(figsize=(18, 15))
    ax.bar(x_pos - width/2, conf_scores_R1, width, color='lightcoral', label='forward')
    ax.bar(x_pos + width/2, conf_scores_R2, width, color='dodgerblue', label='reverse')
    ax.legend(loc='upper right', fontsize=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(taxa, rotation=90, fontsize=15)
    ax.set_ylabel('mean confidence scores', fontsize=15)
    plt.savefig(os.path.join(args.output_path, f'{args.sample}_model{args.model_num}_epoch{args.epoch_num}_cs_class.png'), bbox_inches='tight')

def histogram(args, conf_scores_R1, conf_scores_R2):
    plt.clf()
    fig, ax = plt.subplots(figsize=(7, 6))
    n, _, _ = ax.hist(x=[conf_scores_R1, conf_scores_R2], label=['forward reads', 'reverse reads'])
    print(f'max frequence: {n.max()}')

    with open(os.path.join(args.input_path, f'{args.sample}_cs_stats_{args.threshold}.txt'), 'w') as f:
        f.write(f'max fw/rv: {max(conf_scores_R1)}\t{max(conf_scores_R2)}\n'
                f'min fw/rv: {min(conf_scores_R1)}\t{min(conf_scores_R2)}\n'
                f'mean fw/rv: {statistics.mean(conf_scores_R1)}\t{statistics.mean(conf_scores_R2)}\n'
                f'median fw/rv: {statistics.median(conf_scores_R1)}\t{statistics.median(conf_scores_R2)}\n'
                f'standard deviation fw/rv: {statistics.stdev(conf_scores_R1)}\t{statistics.stdev(conf_scores_R2)}'
                )

    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax.grid(axis='y', alpha=0.75)
    ax.legend(loc='upper left')
    ax.set_xlabel('Confidence score')
    ax.set_ylabel('Frequency')
    plt.savefig(os.path.join(args.output_path, f'{args.sample}_model{args.model_num}_epoch{args.epoch_num}_cs_hist.png'))


def generate_fastq_files(args, class_id, list_records, type_reads):
    SeqIO.write(list_records, os.path.join(args.output_path, f'sample-{args.sample}-bins', f"{args.sample}-{class_id}-{type_reads}.fq"), "fastq")

def create_bins(args, fw_read_ids, rv_read_ids, fw_predicted_classes, rv_predicted_classes, fw_dict_conf_scores, rv_dict_conf_scores):
    print(len(fw_read_ids), len(fw_predicted_classes), len(fw_dict_conf_scores), len(rv_read_ids), len(rv_predicted_classes), len(rv_dict_conf_scores))
    assert len(fw_read_ids) == len(fw_predicted_classes) == len(fw_dict_conf_scores) == len(rv_read_ids) == len(rv_predicted_classes) == len(rv_dict_conf_scores)

    dict_reads_R1 = default_dict(list)   # {predicted class: list read ids forward reads}
    dict_reads_R2 = default_dict(list)   # {predicted class: list read ids reverse reads}

    # retrieve records
    records_R1 = {record.id.split(' ')[0]: record for record in
                  SeqIO.parse(os.path.join(args.input_path, 'fastq', f'{args.sample}_R1.fq'), 'fastq')}
    records_R2 = {record.id.split(' ')[0]: record for record in
                  SeqIO.parse(os.path.join(args.input_path, 'fastq', f'{args.sample}_R2.fq'), 'fastq')}

    # sort reads based on the class they were assigned to
    for i in range(len(fw_predicted_classes)):
        if fw_dict_conf_scores[i] >= args.threshold:
            read_id = fw_read_ids[i]
            dict_reads_R1[fw_predicted_classes[i]].append(read_id.split(' ')[0])
            

    for i in range(len(rv_predicted_classes)):
        if rv_dict_conf_scores[i] >= args.threshold:
            read_id = rv_read_ids[i]
            dict_reads_R2[rv_predicted_classes[i]].append(read_id.split(' ')[0])

    class_conf_scores_R1 = {}    # {class id: list of confidence scores of forward reads}
    class_conf_scores_R2 = {}  # {class id: list of confidence scores of reverse reads}
    conf_scores_R1 = []     # confidence scores of all forward reads regardless of class
    conf_scores_R2 = []     # confidence scores of all reverse reads regardless of class
    num_classified_paired_reads = 0
    num_classified_fw_unpaired_reads = 0
    num_classified_rv_unpaired_reads = 0
    for class_id, class_name in args.class_mapping.items():
        # filter reads in pairs that are not assigned to the same class
        set_reads_R1 = set(dict_reads_R1[class_id])
        set_reads_R2 = set(dict_reads_R2[class_id])
        # take the intersection of both sets (reads in pairs)
        reads_in_pairs = list(set_reads_R1.intersection(set_reads_R2))
        num_classified_paired_reads += len(reads_in_pairs)
        # get records for reads in pairs
        list_records_paired_reads = [records_R1[read_id] for read_id in reads_in_pairs] + [records_R2[read_id] for read_id in reads_in_pairs]
        # get list of confidence scores
        class_conf_scores_R1[class_id] = [fw_dict_conf_scores[read_id + '1:N:0:AGATCC'] for read_id in reads_in_pairs]
        class_conf_scores_R2[class_id] = [rv_dict_conf_scores[read_id + '2:N:0:AGATCC'] for read_id in reads_in_pairs]
        conf_scores_R1 += class_conf_scores_R1[class_id]
        conf_scores_R2 += class_conf_scores_R2[class_id]
        # create fastq files
        generate_fastq_files(args, class_id, list_records_paired_reads, 'paired')
        # get classified forward unpaired reads
        fw_reads = list(set_reads_R1.difference(set_reads_R2))
        num_classified_fw_unpaired_reads += len(fw_reads)
        list_records_fw_unpaired_reads = [records_R1[read_id] for read_id in fw_reads]
        conf_scores_R1 += [fw_dict_conf_scores[read_id + '1:N:0:AGATCC'] for read_id in fw_reads]
        # create fastq files
        generate_fastq_files(args, class_id, list_records_fw_unpaired_reads, 'fw_unpaired')
        # get classified forward unpaired reads
        rv_reads = list(set_reads_R2.difference(set_reads_R1))
        num_classified_rv_unpaired_reads += len(rv_reads)
        list_records_fw_unpaired_reads = [records_R2[read_id] for read_id in rv_reads] 
        conf_scores_R2 += [rv_dict_conf_scores[read_id + '2:N:0:AGATCC'] for read_id in rv_reads]
        # create fastq files
        generate_fastq_files(args, class_id, list_records_rv_unpaired_reads, 'rv_unpaired')
        # summarize results
        with open(os.path.join(args.output_path, f'sample_{args.sample}_model{args.model_num}_epoch{args.epoch_num}_classification_summary'), 'a') as f:
            f.write(f'{class_id}\t{class_name}\tpairs: {len(reads_in_pairs)}\tfw unpaired: {len(fw_reads)}\trv unpaired: {len(rv_reads)}\t{len(dict_reads_R1[class_id])+len(dict_reads_R2[class_id])}\n')

    # create histogram for confidence scores
    histogram(args, conf_scores_R1, conf_scores_R2)
    # create barplot with confidence scores stats for each class
    summarize_cs(args, class_conf_scores_R1, class_conf_scores_R2)
    print(f'unpaired fw: {num_classified_fw_unpaired_reads}')
    print(f'unpaired rv: {num_classified_rv_unpaired_reads}')
    print(f'paired: {num_classified_paired_reads}')
    print(f'total: {len(fw_read_ids) + len(rv_read_ids)}')
    print(f'total classified fw: {len(conf_scores_R1)}\ttotal classified rv:{len(conf_scores_R2)}')
    print(f'total unclassified: {len(fw_read_ids) + len(rv_read_ids) - len(conf_scores_R1) - len(conf_scores_R2)}')

    with open(os.path.join(args.output_path, f'sample_{args.sample}_model{args.model_num}_epoch{args.epoch_num}_classification_summary'), 'a') as f:
        f.write(f'total reads: {len(fw_read_ids) + len(rv_read_ids)}\tunclassified reads: {len(fw_read_ids) + len(rv_read_ids) - 2*num_classified_reads - num_classified_fw_unpaired_reads - num_classified_rv_unpaired_reads}\tclassified fw unpaired: {num_classified_fw_unpaired_reads}\tclassified rv unpaired: {num_classified_rv_unpaired_reads}\tclassified pairs of reads: {num_classified_reads}')
