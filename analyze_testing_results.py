import pandas as pd
import os
import sys
import glob
import numpy as np
import json
from summarize import get_metrics, ROCcurve
import argparse

def write_cm_to_file(cm, class_mapping_dict, results_dir, rank):
    with open(os.path.join(results_dir, f'{rank}-confusion-matrix.tsv'), 'w') as f:
            for i in range(len(class_mapping_dict)):
                f.write(f'\t{class_mapping_dict[str(i)]}')
            f.write('\n')
            for i in range(len(class_mapping_dict)):
                f.write(f'{class_mapping_dict[str(i)]}')
                for j in range(len(class_mapping_dict)):
                    f.write(f'\t{cm[i,j]}')
                f.write('\n')


def get_prob(results, results_dir, class_mapping_dict):
    pred_species = []
    true_species = []
    probs = []
    # with open(os.path.join(results_dir, 'all-probabilities-species.tsv'), 'w') as out_f:
    for r in results:
        df = pd.read_csv(r, delimiter='\t', header=None)
        pred_species += df.iloc[:,2].tolist()
        true_species += df.iloc[:,1].tolist()
        probs += df.iloc[:,3].tolist()
        # with open(r, 'r') as in_f:
        #     for line in in_f:
        #         true_taxon = line.rstrip().split('\t')[1]
        #         pred_taxon = line.rstrip().split('\t')[2]
                # out_f.write(f'{line.rstrip()}\t{class_mapping_dict[true_taxon]}\t{class_mapping_dict[pred_taxon]}\n')
    # print(f'size of pred and true species lists: {len(pred_species)}\t{len(true_species)}')
    return pred_species, true_species, probs

def create_prob_file(results_dir, pred_classes, true_classes, probs, class_mapping_dict, rank):
    with open(os.path.join(results_dir, f'all-probabilities-{rank}.tsv'), 'w') as f:
        for i in range(len(pred_classes)):
            if pred_classes[i] == true_classes[i]:
                f.write(f'correct\t')
            else:
                f.write(f'incorrect\t')
            f.write(f'{true_classes[i]}\t{pred_classes[i]}\t{probs[i]}\t{class_mapping_dict[str(true_classes[i])]}\t{class_mapping_dict[str(pred_classes[i])]}\n')

def get_cm(true_classes, predicted_classes, results_dir, class_mapping, rank):
    # create empty confusion matrix with rows = true classes and columns = predicted classes
    cm = np.zeros((len(class_mapping), len(class_mapping)))
    num_correct_pred = 0
    num_incorrect_pred = 0
    # fill out confusion matrix
    for i in range(len(true_classes)):
        cm[true_classes[i], predicted_classes[i]] += 1
        if true_classes[i] == predicted_classes[i]:
            num_correct_pred += 1
        else:
            num_incorrect_pred += 1

    accuracy = round(float(num_correct_pred)/len(true_classes), 5)
    print(f'{rank}\t{accuracy}')
    return cm


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--class_mapping', type=str, help='directory containing json dictionary files mapping taxa to labels', required=True)
    parser.add_argument('--results_dir', type=str, help='directory containing testing results', required=True)
    parser.add_argument('--output_dir', type=str, help='output directory', required=True)
    args = parser.parse_args()

    # load dictionary mapping labels to species
    with open(os.path.join(args.class_mapping, 'class_mapping.json'), 'r') as f:
        class_mapping_dict = json.load(f)

    # get softmax layer output from each GPU
    results_prob = sorted(glob.glob(os.path.join(args.results_dir, 'probabilities-*.tsv')))

    # analyze results at the species level
    # get confusion matrix (rows = true classes, columns = predicted classes)
    cm = np.zeros((len(class_mapping_dict), len(class_mapping_dict)))
    # get predictions and ground truth
    pred_species, true_species, probs = get_prob(results_prob, args.results_dir, class_mapping_dict)
    # fill out confusion matrix at the species level
    cm = get_cm(true_species, pred_species, args.results_dir, class_mapping_dict, 'species')
    write_cm_to_file(cm, class_mapping_dict, args.results_dir, 'species')
    # get precision and recall for each species
    sp_accuracy = get_metrics(cm, class_mapping_dict, args.results_dir, 'species')

    # analyze results at higher taxonomic levels
    for r in ['genus', 'family', 'order', 'class']:
        # load dictionary mapping species labels to other ranks labels
        with open(os.path.join(args.class_mapping, f'{r}_species_mapping_dict.json')) as f_json:
            rank_species_mapping = json.load(f_json)
        # get vectors of predicted and true labels at given rank
        rank_pred_classes = [rank_species_mapping[str(i)] for i in pred_species]
        rank_true_classes = [rank_species_mapping[str(i)] for i in true_species]
        # load dictionary mapping labels to taxa at given rank
        with open(os.path.join(args.class_mapping, f'{r}_mapping_dict.json')) as f_json:
            rank_mapping_dict = json.load(f_json)
        # get confusion matrix
        cm = get_cm(rank_true_classes, rank_pred_classes, args.results_dir, class_mapping_dict, r)
        write_cm_to_file(cm, rank_mapping_dict, args.results_dir, r)
        # get precision and recall
        get_metrics(cm, rank_mapping_dict, args.results_dir, r)
        # add taxonomy to file with probabilities
        create_prob_file(args.results_dir, rank_pred_classes, rank_true_classes, probs, rank_mapping_dict, r)

if __name__ == "__main__":
    main()
