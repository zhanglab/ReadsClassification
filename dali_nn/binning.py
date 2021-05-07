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
    for class_id, class_name in args.class_mapping.items():
        with open(os.path.join(args.output_path, f'{args.sample}_model{args.model_num}_epoch{args.epoch_num}_cs_stats_per_class_{args.threshold}.txt'), 'a') as f:
            f.write(f'{class_id}\t{class_name}\n')
            if len(class_conf_scores_R1[class_id]) != 0:
                f.write(f'max fw: {max(class_conf_scores_R1[class_id])}\t'
                f'min fw: {min(class_conf_scores_R1[class_id])}\t'
                f'mean fw: {statistics.mean(class_conf_scores_R1[class_id])}\t'
                f'median fw: {statistics.median(class_conf_scores_R1[class_id])}\t'
                f'standard deviation fw: {np.std(class_conf_scores_R1[class_id])}\n')
            if len(class_conf_scores_R2[class_id]) != 0:
                f.write(f'max rv: {max(class_conf_scores_R2[class_id])}\t'
                f'min rv: {min(class_conf_scores_R2[class_id])}\t'
                f'mean rv: {statistics.mean(class_conf_scores_R2[class_id])}\t'
                f'median rv: {statistics.median(class_conf_scores_R2[class_id])}\t'
                f'standard deviation rv: {np.std(class_conf_scores_R2[class_id])}\n')

    taxa = [args.class_mapping[str(i)] for i in range(len(args.class_mapping))]
    conf_scores_R1 = [np.mean(class_conf_scores_R1[str(i)]) if len(class_conf_scores_R1[str(i)]) != 0 else 0 for i in range(len(args.class_mapping))]
    conf_scores_R2 = [np.mean(class_conf_scores_R2[str(i)]) if len(class_conf_scores_R2[str(i)]) != 0 else 0 for i in range(len(args.class_mapping))]
    x_pos = np.arange(len(taxa))
    width = 0.35
    print(f'size x_pos: {len(x_pos)}')
    print(f'size taxa: {len(taxa)}')
    print(f'size r1: {len(conf_scores_R1)}')
    print(f'size r2: {len(conf_scores_R2)}')
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

    with open(os.path.join(args.output_path, f'{args.sample}_cs_stats_{args.threshold}.txt'), 'w') as f:
        f.write(f'max fw/rv: {max(conf_scores_R1)}\t{max(conf_scores_R2)}\n'
                f'min fw/rv: {min(conf_scores_R1)}\t{min(conf_scores_R2)}\n'
                f'mean fw/rv: {statistics.mean(conf_scores_R1)}\t{statistics.mean(conf_scores_R2)}\n'
                f'median fw/rv: {statistics.median(conf_scores_R1)}\t{statistics.median(conf_scores_R2)}\n'
                f'standard deviation fw/rv: {np.std(conf_scores_R1)}\t{np.std(conf_scores_R2)}'
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

    dict_reads_R1 = defaultdict(list)   # {predicted class: list read ids forward reads}
    dict_reads_R2 = defaultdict(list)   # {predicted class: list read ids reverse reads}
    
    # retrieve records
    records_R1 = {record.id.split(' ')[0]: record for record in
                  SeqIO.parse(os.path.join(args.path_to_sample, 'fastq', f'{args.sample}_R1.fq'), 'fastq')}
    records_R2 = {record.id.split(' ')[0]: record for record in
                  SeqIO.parse(os.path.join(args.path_to_sample, 'fastq', f'{args.sample}_R2.fq'), 'fastq')}
    
    # load decision thresholds
    decision_thresholds = {}
    with open(args.decision_thresholds, 'r') as f:
        for line in f:
            line = line.rstrip().split('\t')
            decision_thresholds[line[0]] = float(line[2])

    # sort reads based on their predicted class
    for i in range(len(fw_predicted_classes)):
        read_id = fw_read_ids[i]
        fw_pred_class = str(fw_predicted_classes[i])
        if fw_dict_conf_scores[read_id] >= decision_thresholds[fw_pred_class]:
            dict_reads_R1[fw_pred_class].append(read_id)
            
    for i in range(len(rv_predicted_classes)):
        read_id = rv_read_ids[i]
        rv_pred_class = str(rv_predicted_classes[i])
        if rv_dict_conf_scores[read_id] >= decision_thresholds[rv_pred_class]:
            dict_reads_R2[rv_pred_class].append(read_id)
    
    class_conf_scores_R1 = {}    # {class id: list of confidence scores of forward reads}
    class_conf_scores_R2 = {}  # {class id: list of confidence scores of reverse reads}
    conf_scores_R1 = []     # confidence scores of all forward reads regardless of class
    conf_scores_R2 = []     # confidence scores of all reverse reads regardless of class
    num_classified_paired_reads = 0
    num_classified_fw_unpaired_reads = 0
    num_classified_rv_unpaired_reads = 0
    for class_id, class_name in args.class_mapping.items():
        print(class_id, class_name, len(dict_reads_R1[class_id]), len(dict_reads_R2[class_id]))
        # filter reads in pairs that are not assigned to the same class
        set_reads_R1 = set(dict_reads_R1[class_id])
        set_reads_R2 = set(dict_reads_R2[class_id])
        # take the intersection of both sets (reads in pairs)
        reads_in_pairs = list(set_reads_R1.intersection(set_reads_R2))
        num_classified_paired_reads += len(reads_in_pairs)
        # get records for reads in pairs
        list_records_paired_reads_fw = [records_R1[read_id] for read_id in reads_in_pairs]
        list_records_paired_reads_rv = [records_R2[read_id] for read_id in reads_in_pairs]
        # get list of confidence scores
        class_conf_scores_R1[class_id] = [fw_dict_conf_scores[read_id] for read_id in reads_in_pairs]
        class_conf_scores_R2[class_id] = [rv_dict_conf_scores[read_id] for read_id in reads_in_pairs]
        conf_scores_R1 += class_conf_scores_R1[class_id]
        conf_scores_R2 += class_conf_scores_R2[class_id]
        # create fastq files
        #generate_fastq_files(args, class_id, list_records_paired_reads_fw, 'fw_paired')
        #generate_fastq_files(args, class_id, list_records_paired_reads_rv, 'rv_paired')
        # get classified forward unpaired reads
        fw_reads = list(set_reads_R1.difference(set_reads_R2))
        #num_classified_fw_unpaired_reads += len(fw_reads)
        #list_records_unpaired_reads = [records_R1[read_id] for read_id in fw_reads]
        #class_conf_scores_R1[class_id] += [fw_dict_conf_scores[read_id] for read_id in fw_reads]
        #conf_scores_R1 += [fw_dict_conf_scores[read_id] for read_id in fw_reads]
        # add reverse unclassified reads 
        num_classified_paired_reads += len(fw_reads)
        list_records_paired_reads_rv += [records_R2[read_id] for read_id in fw_reads]
        class_conf_scores_R2[class_id] += [rv_dict_conf_scores[read_id] for read_id in fw_reads]
        conf_scores_R2 += [rv_dict_conf_scores[read_id] for read_id in fw_reads]
        # get classified forward unpaired reads
        rv_reads = list(set_reads_R2.difference(set_reads_R1))
        #num_classified_rv_unpaired_reads += len(rv_reads)
        #list_records_unpaired_reads += [records_R2[read_id] for read_id in rv_reads]
        #class_conf_scores_R2[class_id] += [rv_dict_conf_scores[read_id] for read_id in rv_reads]
        #conf_scores_R2 += [rv_dict_conf_scores[read_id] for read_id in rv_reads]
        # add forward unclassified reads
        num_classified_paired_reads += len(rv_reads)
        list_records_paired_reads_fw += [records_R1[read_id] for read_id in rv_reads]
        class_conf_scores_R1[class_id] += [fw_dict_conf_scores[read_id] for read_id in rv_reads]
        conf_scores_R1 += [fw_dict_conf_scores[read_id] for read_id in rv_reads]
        # create fastq files
        generate_fastq_files(args, class_id, list_records_paired_reads_fw, 'fw_paired')
        generate_fastq_files(args, class_id, list_records_paired_reads_rv, 'rv_paired')
        #generate_fastq_files(args, class_id, list_records_unpaired_reads, 'unpaired')
        # compute relative abundance
        #ra = round((len(fw_reads)+len(rv_reads)+len(reads_in_pairs))/(len(fw_read_ids) + len(rv_read_ids)), 3)
        ra = round((2*len(fw_reads)+2*len(rv_reads)+len(reads_in_pairs))/(len(fw_read_ids) + len(rv_read_ids)), 3)
        # summarize results
        with open(os.path.join(args.output_path, f'sample_{args.sample}_model{args.model_num}_epoch{args.epoch_num}_classification_summary'), 'a') as f:
            #f.write(f'{class_id}\t{class_name}\tpairs: {len(reads_in_pairs)}\tfw unpaired: {len(fw_reads)}\trv unpaired: {len(rv_reads)}\treads classified: {len(dict_reads_R1[class_id])+len(dict_reads_R2[class_id])}\trelative abundance: {ra}\n')
            f.write(f'{class_id}\t{class_name}\treads classified: {len(reads_in_pairs) + 2*len(fw_reads) + 2*len(rv_reads)}\trelative abundance: {ra}\n')
        if len(dict_reads_R1[class_id]) + len(dict_reads_R2[class_id]) >= args.min_num_reads:
            with open(os.path.join(args.output_path, f'list_bins'), 'a') as f:
                f.write(f'{class_id}\n')

    # create histogram for confidence scores
    histogram(args, conf_scores_R1, conf_scores_R2)
    # create barplot with confidence scores stats for each class
    summarize_cs(args, class_conf_scores_R1, class_conf_scores_R2)
    #print(f'unpaired fw: {num_classified_fw_unpaired_reads}')
    #print(f'unpaired rv: {num_classified_rv_unpaired_reads}')
    print(f'paired: {num_classified_paired_reads}')
    print(f'total: {len(fw_read_ids) + len(rv_read_ids)}')
    print(f'total classified fw: {len(conf_scores_R1)}\ttotal classified rv:{len(conf_scores_R2)}')
    print(f'total unclassified: {len(fw_read_ids) + len(rv_read_ids) - len(conf_scores_R1) - len(conf_scores_R2)}')

    with open(os.path.join(args.output_path, f'sample_{args.sample}_model{args.model_num}_epoch{args.epoch_num}_classification_summary'), 'a') as f:
        #f.write(f'total reads: {len(fw_read_ids) + len(rv_read_ids)}\tunclassified reads: {len(fw_read_ids) + len(rv_read_ids) - 2*num_classified_paired_reads - num_classified_fw_unpaired_reads - num_classified_rv_unpaired_reads}\tclassified fw unpaired: {num_classified_fw_unpaired_reads}\tclassified rv unpaired: {num_classified_rv_unpaired_reads}\tclassified pairs of reads: {num_classified_paired_reads}')
        f.write(f'total reads: {len(fw_read_ids) + len(rv_read_ids)}\tunclassified reads: {len(fw_read_ids) + len(rv_read_ids) - 2*num_classified_paired_reads}\tclassified pairs of reads: {num_classified_paired_reads}')
