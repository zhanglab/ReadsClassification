import pandas as pd
import os
import sys
import glob
import numpy as np
import json

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
    with open(os.path.join(results_dir, 'all-probabilities-species.tsv'), 'w') as out_f:
        for r in results:
            print(r)
            df = pd.read_csv(r, delimiter='\t', header=None)
            print(df.shape)
            print(df.iloc[:,2])
            pred_species += df.iloc[:,2].tolist()
            true_species += df.iloc[:,1].tolist()
            probs += df.iloc[:,3].tolist()
            with open(r, 'r') as in_f:
                for line in in_f:
                    true_taxon = line.rstrip().split('\t')[1]
                    pred_taxon = line.rstrip().split('\t')[2]
                    out_f.write(f'{line.rstrip()}\t{class_mapping_dict[true_taxon]}\t{class_mapping_dict[pred_taxon]}\n')
    print(f'size of pred and true species lists: {len(pred_species)}\t{len(true_species)}')
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


def get_metrics(cm, class_mapping_dict, results_dir, rank):
    """ precision = True Positives / (True Positives + False Positives) """
    """ recall = True Positives / (True Positives + False Negatives) """
    """ accuracy = number of correct predictions / (number of reads in testing set) """

    correct_predictions = 0

    f = open(os.path.join(results_dir, f'{rank}-metrics-classification-report.tsv'), 'w')
    f.write(f'{rank}\tprecision\trecall\tnumber\n')

    # get labels of species present only in testing set and total number of reads in testing set
    labels_in_test_set = []
    total_num_reads = 0
    for i in range(len(class_mapping_dict)):
        num_testing_reads = sum([cm[i, j] for j in range(len(class_mapping_dict))])
        total_num_reads += num_testing_reads
        if num_testing_reads != 0:
            labels_in_test_set.append(i)

    # get precision and recall for all species in testing set
    for i in labels_in_test_set:
        true_positives = cm[i, i]
        other_labels = [j for j in labels_in_test_set if j != i]
        false_positives = sum([cm[j, i] for j in other_labels])
        false_negatives = sum([cm[i, j] for j in other_labels])
        correct_predictions += true_positives
        num_testing_reads = sum([cm[i, j] for j in range(len(class_mapping_dict))])
        precision = float(true_positives)/(true_positives+false_positives)
        recall = float(true_positives)/(true_positives+false_negatives)
        f.write(f'{class_mapping_dict[str(i)]}\t{round(precision,5)}\t{round(recall,5)}\t{num_testing_reads}\n')
    
    accuracy =  float(correct_predictions)/total_num_reads
    f.write(f'Accuracy: {round(accuracy,5)}')
    f.close()

def main():
    input_dir = sys.argv[1]
    results_dir = sys.argv[2]

    # load dictionary mapping labels to species
    with open(os.path.join(input_dir, 'class_mapping.json'), 'r') as f:
        class_mapping_dict = json.load(f)

    # get softmax layer output from each GPU
    results_prob = sorted(glob.glob(os.path.join(results_dir, 'results-gpu-*', 'probabilities-gpu-*.tsv')))

    # analyze results at the species level
    # get confusion matrix (rows = true classes, columns = predicted classes)
    cm = np.zeros((len(class_mapping_dict), len(class_mapping_dict)))        
    # get predictions and ground truth
    pred_species, true_species, probs = get_prob(results_prob, results_dir, class_mapping_dict)
    # fill out confusion matrix at the species level
    cm = get_cm(true_species, pred_species, results_dir, class_mapping_dict, 'species')
    write_cm_to_file(cm, class_mapping_dict, results_dir, 'species')
    # get precision and recall for each species
    get_metrics(cm, class_mapping_dict, results_dir, 'species')

    # analyze results at higher taxonomic levels
    for r in ['genus', 'family', 'order', 'class']:    
        # load dictionary mapping species labels to other ranks labels
        with open(os.path.join(input_dir, f'{r}_species_mapping_dict.json')) as f_json:
            rank_species_mapping = json.load(f_json)
        # get vectors of predicted and true labels at given rank
        rank_pred_classes = [rank_species_mapping[str(i)] for i in pred_species]
        rank_true_classes = [rank_species_mapping[str(i)] for i in true_species]
        # load dictionary mapping labels to taxa at given rank
        with open(os.path.join(input_dir, f'{r}_mapping_dict.json')) as f_json:
            rank_mapping_dict = json.load(f_json)
        # get confusion matrix
        cm = get_cm(rank_true_classes, rank_pred_classes, results_dir, class_mapping_dict, r)
        write_cm_to_file(cm, rank_mapping_dict, results_dir, r)
        # get precision and recall
        get_metrics(cm, rank_mapping_dict, results_dir, r)
        # add taxonomy to file with probabilities
        create_prob_file(results_dir, rank_pred_classes, rank_true_classes, probs, rank_mapping_dict, r)

if __name__ == "__main__":
    main()
