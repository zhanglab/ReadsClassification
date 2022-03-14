import glob
import pandas as pd
import sys
import os
import argparse
import numpy as np
import json
import time
import random
from summarize import *


def get_results(args, list_tsv_files):
    pred_species = []
    labels = []
    probs = []
    for in_file in list_tsv_files:
        df = pd.read_csv(in_file, header=None, sep='\t')
        labels += df.iloc[:,0].tolist()
        pred_species += df.iloc[:,1].tolist()
        probs += df.iloc[:,2].tolist()

    if args.data_type == 'test':
        if args.sample_size == None:
            args.sample_size = len(labels)
        # select sample of data
        sample = list(zip(labels, probs))
        random.shuffle(sample)
        s_labels, s_probs = zip(*sample)

        return pred_species, labels, list(s_probs)[:args.sample_size], list(s_labels)[:args.sample_size]

    elif args.data_type == 'meta':
        return pred_species, probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='path to temporary directory containing tsv files obtained from running read_classifier.py', required=True)
    parser.add_argument('--data_type', type=str, help='input data type', required=True, choices=['test', 'meta'])
    parser.add_argument('--sample_size', type=int, help='size of sample for ROC curve analysis')
    parser.add_argument('--rank_mapping_dir', type=str, help='path to json files containing dictionaries mapping taxa to labels', required=True)
    parser.add_argument('--decision_thresholds', type=str, help='path to file containing decision thresholds', required=('--meta' in sys.argv))
    args = parser.parse_args()

    if args.data_type == 'test':
        list_tsv_files = sorted(glob.glob(os.path.join(args.input_dir, 'tmp', '*.tsv')))
        # get predictions and ground truth at species level
        start_time = time.time()
        pred_species, true_species, s_probs, s_true_species = get_results(args, list_tsv_files)
        print(f'# unique predicted probabilities: {len(set(s_probs))}')
        end_time = time.time()
        print(end_time - start_time)
        # load species mapping labels dictionary
        with open(os.path.join(args.rank_mapping_dir, 'species_labels.json'), 'r') as f:
            species_mapping_dict = json.load(f)
        # get confusion matrix
        cm, accuracy = get_cm(true_species, pred_species, species_mapping_dict, 'species')
        # get decision thresholds
        ROCcurve(args, s_true_species, s_probs, species_mapping_dict, set(true_species), 'species')
        # get precision and recall
        get_metrics(args, cm, species_mapping_dict, set(true_species), 'species')
        # analyze results at higher taxonomic levels
        for r in ['genus', 'family', 'order', 'class']:
            # load dictionary mapping species labels to other ranks labels
            with open(os.path.join(args.rank_mapping_dir, f'{r}_species_labels.json')) as f_json:
                rank_species_mapping = json.load(f_json)
            # get vectors of predicted and true labels at given rank
            rank_pred_taxa = [rank_species_mapping[str(i)] for i in pred_species]
            rank_true_taxa = [rank_species_mapping[str(i)] for i in true_species]
            # load dictionary mapping labels to taxa at given rank
            with open(os.path.join(args.rank_mapping_dir, f'{r}_labels.json')) as f_json:
                rank_mapping_dict = json.load(f_json)
            # get confusion matrix
            cm, accuracy = get_cm(rank_true_taxa, rank_pred_taxa, rank_mapping_dict, r)
            # get decision thresholds
            ROCcurve(args, rank_true_taxa, s_probs, rank_mapping_dict, set(rank_true_taxa), r)
            # get precision and recall
            get_metrics(args, cm, rank_mapping_dict, set(rank_true_taxa), r)
    elif args.data_type == 'meta':
        list_tsv_files = sorted(glob.glob(os.path.join(args.input_dir, '*.tsv')))
        for tsv_f in list_tsv_files:
            print(tsv_f)
            pred_species, probs = get_results(args, [tsv_f])
            outfile = os.path.join(args.input_dir, tsv_f.split('.')[0])
            get_taxa_occurrences(args, 'species', pred_species, probs, outfile)
            # load species mapping labels dictionary
            # with open(os.path.join(args.rank_mapping_dir, 'species_labels.json'), 'r') as f:
            #     species_mapping_dict = json.load(f)
            # inv_species_mapping_dict = {v:k for k, v in species_mapping_dict.items()}
            # pred_labels_sp = [inv_species_mapping_dict[i] for i in pred_species]
            # for r in ['genus', 'family', 'order', 'class']:
            #     # load dictionary mapping species labels to other ranks labels
            #     with open(os.path.join(args.rank_mapping_dir, f'{r}_species_labels.json')) as f_json:
            #         rank_species_mapping = json.load(f_json)
            #     # load dictionary mapping labels to taxa at given rank
            #     with open(os.path.join(args.rank_mapping_dir, f'{r}_labels.json')) as f_json:
            #         rank_mapping_dict = json.load(f_json)
            #     # get predicted taxa labels at given rank
            #     rank_pred_taxa = [rank_species_mapping[str(i)] for i in pred_labels_sp]
            #     # get predicted taxa names at given rank
            #     rank_pred_taxa = [rank_mapping_dict[str(i)] for i in rank_pred_taxa]
            #     get_taxa_occurrences(args, r, rank_pred_taxa, probs, outfile)




if __name__ == "__main__":
    main()
