import os
import json
import glob
from random import randint
from collections import defaultdict
import itertools
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
# from sklearn.metrics import jaccard_score
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

# def get_colors(tax_rank, list_taxa, input_path):
#     colors = ['salmon', 'red', 'orange', 'limegreen', 'deepskyblue', 'dodgerblue',
#     'slategrey', 'royalblue', 'darkorchid', 'violet', 'magenta', 'navy', 'green', 'gold',
#     'chocolate', 'black', 'yellow', 'deeppink', 'blue', 'cyan']
#     # select a number of colors equal to the number of taxa
#     colors_s = random.choices(colors, k=(len(list_taxa)))
#     dict_colors = dict(zip(list_taxa, colors_s))
#     with open(os.path.join(input_path, f'{tax_rank}-colors.json'), "w") as f:
#         json.dump(dict_colors, f)
#     return dict_colors

# def create_barplot_training(train_data, val_data, filename, class_mapping):
#     list_taxa = [class_mapping[str(i)] for i in range(len(class_mapping))]
#     new_train_data = [train_data[i] for i in range(len(list_taxa))]
#     new_val_data = [val_data[i] for i in range(len(list_taxa))]
#     x_pos = np.arange(0, len(list_taxa)*2, 2)
#     width = 0.5
#
#     plt.clf()
#     fig, ax = plt.subplots(figsize=(18, 15))
#     ax.bar(list_taxa, new_train_data, width, align='center', color='lightcoral', label='training')
#     ax.bar(list_taxa, new_val_data, width, align='center', color='dodgerblue', label='validation')
#     ax.legend(loc='upper right', fontsize=15)
#     ax.set_xticklabels(list_taxa, rotation=90, fontsize=7)
#     ax.set_ylabel('# reads', fontsize=15)
#     plt.savefig(f'{filename}', bbox_inches='tight')
#
#
# def create_barplot_testing(test_data, filename, class_mapping):
#     list_taxa = [class_mapping[str(i)] for i in range(len(class_mapping))]
#     new_test_data = [test_data[i] if i in test_data else 0 for i in range(len(list_taxa))]
#     x_pos = np.arange(0, len(list_taxa) * 2, 2)
#     width = 0.5
#
#     plt.clf()
#     fig, ax = plt.subplots(figsize=(18, 15))
#     ax.bar(list_taxa, new_test_data, width, align='center', color='lightcoral', label='testing')
#     ax.legend(loc='upper right', fontsize=15)
#     ax.set_xticklabels(list_taxa, rotation=90, fontsize=7)
#     ax.set_ylabel('# reads', fontsize=15)
#     plt.savefig(f'{filename}', bbox_inches='tight')

# def learning_curves(history, filename, mode='std'):
#     # plot learning curves
#     plt.clf()
#     fig = plt.figure(dpi=500)
#     ax1 = fig.add_subplot(111)
#     ax2 = ax1.twinx()
#     # Add some extra space for the second axis at the bottom
#     fig.subplots_adjust(bottom=0.2)
#     fig.set_size_inches(7, 7)
#     # Set range of y axis to 0.0 - max of the four lists
#     if mode == 'std':
#         combined_lists = history.history['loss'] + history.history['val_loss'] + history.history['accuracy'] + \
#                      history.history['val_accuracy']
#     elif mode == 'hp':
#         combined_lists = history.history['loss'] + history.history['accuracy']
#
#     max_range = max(combined_lists)
#     ax2.set_ylim(0, 1)
#     ax1.set_ylim(0, max_range)
#     # Get x-axis values
#     list_num_epochs = list(range(1, len(history.history['loss']) + 1, 1))
#     x_coords = [i for i in range(1, len(history.history['loss']) + 1)]
#     if mode == 'std':
#         ax2.plot(list_num_epochs, history.history['val_accuracy'], color='salmon', linewidth=2.0,
#              label='Average Validation Accuracy')
#         ax1.plot(list_num_epochs, history.history['val_loss'], color='dodgerblue', linewidth=2.0, label='Validation Loss')
#     ax2.plot(list_num_epochs, history.history['accuracy'], color='lightsalmon', linewidth=2.0, label='Average Training Accuracy')
#     ax1.plot(list_num_epochs, history.history['loss'], color='skyblue', linewidth=2.0, label='Training Loss')
#     ax1.set_ylabel('Loss', fontsize=14, labelpad=12)
#     ax1.set_xlabel('Number of epochs', fontsize=14, labelpad=12)
#     ax2.set_ylabel('Average Accuracy', fontsize=14, labelpad=12)
#     ax2.ticklabel_format(useOffset=True, style='sci')
#     ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
#                fancybox=True, shadow=False, ncol=5)
#     ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
#                fancybox=True, shadow=False, ncol=5)
#     # Add vertical lines to represent each epoch
#     for xcoord in x_coords:
#         plt.axvline(x=xcoord, color='gray', linestyle='--')
#     plt.savefig(f'{filename}', bbox_inches='tight')


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
    f.close()


def ROCcurve(args, class_mapping, species_in_test_set):

    list_pred_files = sorted(glob.glob(os.path.join(args.results_dir, 'pred-probs-*.npy')))
    list_true_files = sorted(glob.glob(os.path.join(args.results_dir, 'true-probs-*.npy')))

    print(len(list_true_files), len(list_pred_files))
    print(len(species_in_test_set), species_in_test_set)

    fpr = {}
    tpr = {}
    thresholds = {}
    roc_auc = {}
    J_stats = [None] * len(species_in_test_set)
    opt_thresholds = [None] * len(species_in_test_set)
    jstat_opt_thresholds = [None] * len(species_in_test_set)
    # for i in range(len(list_pred_files)):
    pred_arr = np.load(list_pred_files[0])
    true_arr = np.load(list_true_files[0])
    f = open(os.path.join(args.output_dir, f'decision_thresholds.tsv'), 'w')
    for j in range(len(class_mapping)):
        if j in species_in_test_set:
            fpr[j], tpr[j], thresholds[j] = roc_curve(true_arr[:, j], pred_arr[:, j])
            roc_auc[j] = auc(fpr[j], tpr[j])
            # Compute Youden's J statistics for each species:
            # get optimal cut off corresponding to a high TPR and low FPR
            J_stats[j] = tpr[j] - fpr[j]
            jstat_optimal_index = np.argmax(J_stats[j])
            opt_thresholds[j] = thresholds[j][jstat_optimal_index]
            jstat_opt_thresholds[j] = round(J_stats[j][jstat_optimal_index], 2)
            if j == 0:
                print(j, opt_thresholds[j], jstat_opt_thresholds[j])
                print(fpr[j])
                print(tpr[j])
                print(J_stats[j])
                print(thresholds[j])
                print(jstat_optimal_index)
                print(opt_thresholds[j])
                print(jstat_opt_thresholds[j])
            f.write(f'{j}\t{class_mapping[str(j)]}\t{jstat_opt_thresholds[j]}\n')
        else:
            f.write(f'{j}\t{class_mapping[str(j)]}\t0.5\n')

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

    # only plot ROC curves for test sets with 20 species
    if len(species_in_test_set) <= 20:
        colors = np.random.rand(len(species_in_test_set), 3)
        plt.clf()
        ax = plt.gca()
        plt.figure()
        lw = 2
        for i, color in zip(species_in_test_set, colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,label='ROC curve of species {0} (area = {1:0.2f})'.format(class_mapping[str(i)], roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        #plt.legend(loc=(0, -.6), prop=dict(size=9))
        plt.savefig(os.path.join(args.output_dir, f'ROCcurves.png'),bbox_inches='tight')
        figlegend = plt.figure()
        plt.figlegend(*ax.get_legend_handles_labels())
        figlegend.savefig(os.path.join(args.output_dir, f'ROCcurves_Legend.png'), bbox_inches='tight')
