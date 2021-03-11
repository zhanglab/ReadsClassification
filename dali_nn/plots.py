import os
import itertools
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()


def create_barplot(args, train_data, val_data):
    list_taxa = [args.class_mapping[str(i)] for i in range(len(args.class_mapping))]
    new_train_data = [train_data[i] for i in range(len(list_taxa))]
    new_val_data = [val_data[i] for i in range(len(list_taxa))]
    x_pos = np.arange(len(list_taxa))
    width = 0.35

    plt.clf()
    fig, ax = plt.subplots(figsize=(18, 15))
    ax.bar(x_pos - width / 2, new_train_data, width, color='lightcoral', label='training')
    ax.bar(x_pos + width / 2, new_val_data, width, color='dodgerblue', label='testing')
    ax.legend(loc='upper right', fontsize=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(list_taxa, rotation=90, fontsize=15)
    ax.set_ylabel('# reads', fontsize=15)
    plt.savefig(os.path.join(args.output, args.model_name, f'training-barplots'), bbox_inches='tight')

def learning_curves(args, history):
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
    plt.savefig(os.path.join(args.output, args.model_name, 'LearningCurves.png'), bbox_inches='tight')
