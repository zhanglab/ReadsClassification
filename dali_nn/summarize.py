import os
from random import randint
import itertools
import numpy as np
from Bio import SeqIO
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()


def create_barplot_training(train_data, val_data, filename, class_mapping):
    list_taxa = [class_mapping[str(i)] for i in range(len(class_mapping))]
    new_train_data = [train_data[i] for i in range(len(list_taxa))]
    new_val_data = [val_data[i] for i in range(len(list_taxa))]
    x_pos = np.arange(0, len(list_taxa)*2, 2)
    width = 0.5
 
    plt.clf()
    fig, ax = plt.subplots(figsize=(18, 15))
    ax.bar(list_taxa, new_train_data, width, align='center', color='lightcoral', label='training')
    ax.bar(list_taxa, new_val_data, width, align='center', color='dodgerblue', label='validation')
    ax.legend(loc='upper right', fontsize=15)
    ax.set_xticklabels(list_taxa, rotation=90, fontsize=7)
    ax.set_ylabel('# reads', fontsize=15)
    plt.savefig(f'{filename}', bbox_inches='tight')


def create_barplot_testing(test_data, filename, class_mapping):
    list_taxa = [class_mapping[str(i)] for i in range(len(class_mapping))]
    new_test_data = [test_data[i] for i in range(len(list_taxa))]
    x_pos = np.arange(0, len(list_taxa) * 2, 2)
    width = 0.5

    plt.clf()
    fig, ax = plt.subplots(figsize=(18, 15))
    ax.bar(list_taxa, new_test_data, width, align='center', color='lightcoral', label='testing')
    ax.legend(loc='upper right', fontsize=15)
    ax.set_xticklabels(list_taxa, rotation=90, fontsize=7)
    ax.set_ylabel('# reads', fontsize=15)
    plt.savefig(f'{filename}', bbox_inches='tight')

def learning_curves(history, filename):
    # plot learning curves
    plt.clf()
    fig = plt.figure(dpi=500)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    # Add some extra space for the second axis at the bottom
    fig.subplots_adjust(bottom=0.2)
    fig.set_size_inches(7, 7)
    # Set range of y axis to 0.0 - max of the four lists
    combined_lists = history.history['loss'] + history.history['val_loss'] + history.history['accuracy'] + \
                     history.history['val_accuracy']
    max_range = max(combined_lists)
    ax2.set_ylim(0, 1)
    ax1.set_ylim(0, max_range)
    # Get x-axis values
    list_num_epochs = list(range(1, len(history.history['loss']) + 1, 1))
    x_coords = [i for i in range(1, len(history.history['loss']) + 1)]

    ax2.plot(list_num_epochs, history.history['val_accuracy'], color='salmon', linewidth=2.0,
             label='Average Validation Accuracy')
    ax2.plot(list_num_epochs, history.history['accuracy'], color='lightsalmon', linewidth=2.0,
             label='Average Training Accuracy')
    ax1.plot(list_num_epochs, history.history['val_loss'], color='dodgerblue', linewidth=2.0, label='Validation Loss')
    ax1.plot(list_num_epochs, history.history['loss'], color='skyblue', linewidth=2.0, label='Training Loss')
    ax1.set_ylabel('Loss', fontsize=14, labelpad=12)
    ax1.set_xlabel('Number of epochs', fontsize=14, labelpad=12)
    ax2.set_ylabel('Average Accuracy', fontsize=14, labelpad=12)
    ax2.ticklabel_format(useOffset=True, style='sci')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
               fancybox=True, shadow=False, ncol=5)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
               fancybox=True, shadow=False, ncol=5)
    # Add vertical lines to represent each epoch
    for xcoord in x_coords:
        plt.axvline(x=xcoord, color='gray', linestyle='--')
    plt.savefig(f'{filename}', bbox_inches='tight')

def ROCcurve(test_dict_classes, trueclasses, predictions, class_mapping, output_path, epoch, colors):
    true_arr = np.asarray(trueclasses)
    pred_arr = np.asarray(predictions)
    fpr = {}
    tpr = {}
    roc_auc = {}
    thresholds = {}
    for i in range(len(class_mapping)):
        fpr[i], tpr[i], thresholds[i] = roc_curve(true_arr[:, i], pred_arr[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    J_stats = [None] * len(class_mapping)
    opt_thresholds = [None] * len(class_mapping)
    f = open(os.path.join(output_path, f'decision_thresholds_epoch{epoch}'), 'w')
    # Compute Youden's J statistics for each species
    for i in range(len(class_mapping)):
        if i in test_dict_classes:
            J_stats[i] = tpr[i] - fpr[i]
            jstat_max_index = np.argmax(J_stats[i])
            opt_thresholds[i] = thresholds[i][jstat_max_index]
            jstat_decision_threshold = round(J_stats[i][jstat_max_index], 2)
            f.write(f'{i}\t{class_mapping[str(i)]}\t{jstat_decision_threshold}\n')
        else:
            f.write(f'{i}\t{class_mapping[str(i)]}\t0.5\n')
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(true_arr.ravel(), pred_arr.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(class_mapping))]))
    # Interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(class_mapping)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(class_mapping)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.clf()
    ax = plt.gca()
    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),color='navy', linestyle=':', linewidth=4)


    for i, color in zip(range(len(class_mapping)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,label='ROC curve of class {0} (area = {1:0.2f})'.format(class_mapping[str(i)], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.legend(loc=(0, -.6), prop=dict(size=9))
    plt.savefig(os.path.join(output_path, f'ROCcurves_epoch_{epoch}.png'),bbox_inches='tight')
    figlegend = plt.figure()
    plt.figlegend(*ax.get_legend_handles_labels())
    figlegend.savefig(os.path.join(output_path, f'ROCcurves_Legend_epoch_{epoch}.png'), bbox_inches='tight')


def confusion_matrix(test_true_classes_int, predicted_classes, list_labels, output_path, class_mapping, epoch):
    num_classes = len(list_labels)
    # create empty confusion matrix with rows = true classes and columns = predicted classes
    cm = np.zeros((num_classes, num_classes))
    num_correct_pred = 0
    num_incorrect_pred = 0
    # fill out confusion matrix
    for i in range(len(test_true_classes_int)):
        cm[test_true_classes_int[i], predicted_classes[i]] += 1
        if test_true_classes_int[i] == predicted_classes[i]:
            num_correct_pred += 1
        else:
            num_incorrect_pred += 1

    with open(os.path.join(output_path, f'confusion_matrix_epoch{epoch}'), 'w') as f:
        for i in range(num_classes):
            f.write(f'\t{class_mapping[str(i)]}')
        f.write('\n')
        for i in range(num_classes):
            f.write(f'{class_mapping[str(i)]}')
            for j in range(num_classes):
                f.write(f'\t{cm[i,j]}')
            f.write('\n')
    accuracy = round(float(num_correct_pred)/len(test_true_classes_int), 5)
    print(accuracy, num_correct_pred, len(test_true_classes_int))
    return cm, accuracy


def create_summary_report(dict_results, class_mapping, output_path, epoch, metric, train_records, val_records):
    # create file to store the report
    f = open(os.path.join(output_path, f'summary_report_{metric}_epoch_{epoch}'), 'w')
    # sort dictionary based on values
    sorted_dict = {k: v for k, v in sorted(dict_results.items(), key=lambda item: item[1])}
    print(sorted_dict)
    for key, value in sorted_dict.items():
        # get number of reads in training fastq file
        train_num_reads = train_records.count(str(key))
        val_num_reads = val_records.count(str(key))
        f.write(f'{class_mapping[str(key)]}\t{value}\t{train_num_reads}\t{val_num_reads}\n')
    f.close()

def metrics_report(test_true_classes_int, predicted_classes, list_labels, output_path, class_mapping, epoch, path_to_data):
    """ precision = True Positives / (True Positives + False Positives) """
    """ recall = True Positives / (True Positives + False Negatives) """
    list_all_labels = list(range(len(list_labels)))
    # get confusion matrix
    cm, accuracy = confusion_matrix(test_true_classes_int, predicted_classes, list_labels, input_path, class_mapping, epoch)
    f = open(os.path.join(output_path, f'classification_report_epoch{epoch}'), 'w') 
    f.write('species\tprecision\trecall\tnumber\n')
    dict_precision = {}
    dict_recall = {}
    # get precision and recall for each class
    for i in range(len(list_labels)):
        true_positives = cm[i, i]
        other_labels = [j for j in range(len(list_labels)) if j != i]
        false_positives = sum([cm[j, i] for j in other_labels])
        false_negatives = sum([cm[i, j] for j in other_labels])
        num_testing_reads = sum([cm[i, j] for j in list_all_labels])
        precision = 0.0
        recall = 0.0
        if num_testing_reads != 0:
            precision = float(true_positives)/(true_positives+false_positives)
            recall = float(true_positives)/(true_positives+false_negatives)
        dict_precision[i] = precision
        dict_recall[i] = recall
        f.write(f'{class_mapping[str(i)]}\t{round(precision,5)}\t{round(recall,5)}\t{num_testing_reads}\n')
    f.write(f'Accuracy: {accuracy}')
    f.close()

    # get records in training and validation sets
    train_records = [record.id.split('|')[1] for record in SeqIO.parse(os.path.join(path_to_data, 'training_data_fold0.fq'), 'fastq')]
    val_records = [record.id.split('|')[1] for record in SeqIO.parse(os.path.join(path_to_data, 'validation_data_fold0.fq'), 'fastq')]
    # sort species according to precision and recall and report number of reads in training set
    create_summary_report(dict_recall, class_mapping, output_path, epoch, 'recall', train_records, val_records)
    create_summary_report(dict_precision, class_mapping, output_path, epoch, 'precision', train_records, val_records)
