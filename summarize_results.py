import glob
import pandas as pd
import sys
import os
import argparse
import numpy as np
import json
import time
import random
from summarize import get_cm, ROCcurve, get_metrics


def get_results(args, list_tsv_files):
    pred_species = []
    true_species = []
    probs = []
    for in_file in list_tsv_files:
        print(in_file)
        df = pd.read_csv(in_file, header=None, sep='\t')
        true_species += df.iloc[:,0].tolist()
        pred_species += df.iloc[:,1].tolist()
        probs += df.iloc[:,2].tolist()

    # select sample of data
    sample = list(zip(true_species, probs))
    random.shuffle(sample)
    s_true_species, s_probs = zip(*sample)

    return pred_species, true_species, list(s_probs)[:args.sample_size], list(s_true_species)[:args.sample_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='path to temporary directory containing tsv files obtained from running read_classifier.py', required=True)
    parser.add_argument('--data_type', type=str, help='input data type', required=True, choices=['test', 'meta'])
    parser.add_argument('--sample_size', type=int, help='size of sample for ROC curve analysis', default=10000000)
    parser.add_argument('--rank_mapping_dir', type=str, help='path to json files containing dictionaries mapping taxa to labels', required=True)
    args = parser.parse_args()

    with open(os.path.join(args.rank_mapping_dir, 'species_labels.json'), 'r') as f:
        species_mapping_dict = json.load(f)

    list_tsv_files = sorted(glob.glob(os.path.join(args.input_dir, 'tmp', '*.tsv')))

    if args.data_type == 'test':
        # get predictions and ground truth at species level
        start_time = time.time()
        pred_species, true_species, s_probs, s_true_species = get_results(args, list_tsv_files)
        print(f'# unique predicted probabilities: {len(set(s_probs))}')
        end_time = time.time()
        print(end_time - start_time)
        # get confusion matrix
        cm, accuracy = get_cm(true_species, pred_species, species_mapping_dict, 'species')
        # get decision thresholds
        ROCcurve(args, s_true_species, s_probs, species_mapping_dict, set(true_species), 'species')
        # get precision and recall
        get_metrics(args, cm, species_mapping_dict, set(true_species), 'species')
        # analyze results at higher taxonomic levels
        # for r in ['genus', 'family', 'order', 'class']:
        #     # load dictionary mapping species labels to other ranks labels
        #     with open(os.path.join(args.rank_mapping_dir, f'{r}_species_labels.json')) as f_json:
        #         rank_species_mapping = json.load(f_json)
        #     # get vectors of predicted and true labels at given rank
        #     rank_pred_taxa = [rank_species_mapping[str(i)] for i in pred_species]
        #     rank_true_taxa = [rank_species_mapping[str(i)] for i in true_species]
        #     # load dictionary mapping labels to taxa at given rank
        #     with open(os.path.join(args.rank_mapping_dir, f'{r}_labels.json')) as f_json:
        #         rank_mapping_dict = json.load(f_json)
        #     # get confusion matrix
        #     cm, accuracy = get_cm(rank_true_taxa, rank_pred_taxa, rank_mapping_dict, r)
        #     # get decision thresholds
        #     ROCcurve(args, rank_true_taxa, probs, rank_mapping_dict, set(rank_true_taxa), r)
        #     # get precision and recall
        #     get_metrics(args, cm, rank_mapping_dict, set(rank_true_taxa), r)

if __name__ == "__main__":
    main()
