import os
from random import randint
from collections import defaultdict
import itertools
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# plt.ioff()

def get_cm(true_taxa, predicted_taxa, rank_mapping_dict, rank):
    # create empty confusion matrix with rows = true classes and columns = predicted classes
    cm = np.zeros((len(rank_mapping_dict), len(rank_mapping_dict)))
    num_correct_pred = 0
    num_incorrect_pred = 0
    # fill out confusion matrix
    for i in range(len(true_taxa)):
        cm[true_taxa[i], predicted_taxa[i]] += 1
        if true_taxa[i] == predicted_taxa[i]:
            num_correct_pred += 1
        else:
            num_incorrect_pred += 1

    accuracy = round(float(num_correct_pred)/len(true_taxa), 5)
    print(f'{rank}\taccuracy 1: {accuracy}')
    return cm, accuracy

def get_metrics(args, cm, rank_mapping_dict, labels_in_test_set, rank):
    """ precision = True Positives / (True Positives + False Positives) """
    """ recall = True Positives / (True Positives + False Negatives) """
    """ accuracy = number of correct predictions / (number of reads in testing set) """

    f = open(os.path.join(args.input_dir, f'{rank}-metrics-classification-report.tsv'), 'w')
    f.write(f'{rank}\tprecision\trecall\tnumber\n')
    correct_predictions = 0
    total_num_reads = 0
    # get precision and recall for all species in testing set
    for i in labels_in_test_set:
        true_positives = cm[i, i]
        other_labels = [j for j in range(len(rank_mapping_dict)) if j != i]
        false_positives = sum([cm[j, i] for j in other_labels])
        false_negatives = sum([cm[i, j] for j in other_labels])
        correct_predictions += true_positives
        num_testing_reads = sum([cm[i, j] for j in range(len(rank_mapping_dict))])
        total_num_reads += num_testing_reads
        precision = float(true_positives)/(true_positives+false_positives)
        recall = float(true_positives)/(true_positives+false_negatives)
        f.write(f'{rank_mapping_dict[str(i)]}\t{round(precision,5)}\t{round(recall,5)}\t{num_testing_reads}\n')

    accuracy =  round(float(correct_predictions)/total_num_reads, 5)
    print(f'{rank}\taccuracy 2: {accuracy}')
    f.close()

def ROCcurve(args, true_taxa, probs, rank_mapping_dict, labels_in_test_set, rank):
    fpr = {}
    tpr = {}
    thresholds = {}
    roc_auc = {}
    J_stats = [None] * len(rank_mapping_dict)
    opt_thresholds = [None] * len(rank_mapping_dict)
    jstat_opt_thresholds = [None] * len(rank_mapping_dict)

    print(f'{rank}\t{len(true_taxa)}\t{len(probs)}')
    dict_probs = defaultdict(list)
    for i in range(len(true_taxa)):
        dict_probs[true_taxa[i]].append(probs[i])

    f = open(os.path.join(args.input_dir, f'decision_thresholds_{rank}.tsv'), 'w')
    for i in range(len(rank_mapping_dict)):
        if i in labels_in_test_set:
            in_arr = np.array([i]*len(dict_probs[i]))
            print(i, len(dict_probs[i]), len(in_arr))
            # fpr[i], tpr[i], thresholds[i] = roc_curve(true_arr[:, i], pred_arr[:, i])
            fpr[i], tpr[i], thresholds[i] = roc_curve(in_arr, np.array(dict_probs[i]), pos_label=i)
            roc_auc[i] = auc(fpr[i], tpr[i])
            # Compute Youden's J statistics for each species:
            # get optimal cut off corresponding to a high TPR and low FPR
            J_stats[i] = tpr[i] - fpr[i]
            jstat_optimal_index = np.argmax(J_stats[i])
            opt_thresholds[i] = thresholds[i][jstat_optimal_index]
            jstat_opt_thresholds[i] = round(J_stats[i][jstat_optimal_index], 2)
            f.write(f'{i}\t{rank_mapping_dict[str(i)]}\t{jstat_opt_thresholds[i]}\t{opt_thresholds[i]}\n')
        else:
            f.write(f'{i}\t{rank_mapping_dict[str(i)]}\t0.5\n')

        # Compute micro-average ROC curve and ROC area
        # fpr["micro"], tpr["micro"], _ = roc_curve(true_arr.ravel(), pred_arr.ravel())
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        #
        # # Aggregate all false positive rates
        # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(class_mapping))]))
        # # Interpolate all ROC curves at this points
        # mean_tpr = np.zeros_like(all_fpr)
        # for j in range(len(class_mapping)):
        #     mean_tpr += np.interp(all_fpr, fpr[j], tpr[j])
        #
        # # Finally average it and compute AUC
        # mean_tpr /= len(class_mapping)

    # fpr["macro"] = all_fpr
    # tpr["macro"] = mean_tpr
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    #
    # # Plot all ROC curves
    # plt.clf()
    # ax = plt.gca()
    # plt.figure()
    # lw = 2
    # plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)
    #
    # plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),color='navy', linestyle=':', linewidth=4)
