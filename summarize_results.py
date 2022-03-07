import glob
import pandas as pd
import sys
import os
import argparse
import json
from summarize import get_cm, ROCcurve, get_metrics


def get_results(list_tsv_files):
    pred_species = []
    true_species = []
    probs = []
    for file_ in list_tsv_files:
        df = pd.read_csv(file_, header=None)
        true_species += df.iloc[:,0].tolist()
        pred_species += df.iloc[:,1].tolist()
        probs += df.iloc[:,2].tolist()

    return pred_species, true_species, probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='path to temporary directory containing tsv files obtained from running read_classifier.py', required=True)
    parser.add_argument('--data_type', type=str, help='input data type', required=True, choices=['test', 'meta'])
    parser.add_argument('--rank_mapping_dir', type=str, help='path to json files containing dictionaries mapping taxa to labels', required=True)
    args = parser.parse_args()

    with open(os.path.join(args.rank_mapping_dir, 'species_labels.json'), 'r') as f:
        species_mapping_dict = json.load(f)

    list_tsv_files = sorted(glob.glob(os.path.join(args.input_dir, 'tmp', '*.tsv')))

    if args.data_type == 'test':
        # get predictions and ground truth at species level
        pred_species, true_species, probs = get_results(list_tsv_files)
        # get confusion matrix
        cm, accuracy = get_cm(true_species, pred_species, species_mapping_dict)
        # get decision thresholds
        ROCcurve(args, true_species, probs, species_mapping_dict, set(true_species), 'species')
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
            cm, accuracy = get_cm(rank_true_taxa, rank_pred_taxa, rank_mapping_dict)
            # get decision thresholds
            ROCcurve(args, rank_true_taxa, probs, rank_mapping_dict, set(rank_true_taxa), r)
            # get precision and recall
            get_metrics(args, cm, rank_mapping_dict, set(rank_true_taxa), r)







if __name__ == "__main__":
    main()
