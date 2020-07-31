import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import pandas as pd

def LearningCurvesPlot(self, train_epoch_loss, train_epoch_accuracy, test_epoch_loss, test_epoch_accuracy, epochs):
    plt.clf()
    fig = plt.figure(dpi=500)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    # Add some extra space for the second axis at the bottom
    fig.subplots_adjust(bottom=0.2)
    fig.set_size_inches(7, 7)
    # Set range of y axis to 0.0 - max of the four lists
    combined_lists = train_epoch_loss + train_epoch_accuracy + test_epoch_loss + test_epoch_accuracy
    max_range = max(combined_lists)
    ax2.set_ylim(0, max_range)
    ax1.set_ylim(0, max_range)
    # Get x-axis values
    list_num_epochs = list(range(1, epochs + 1, 1))
    x_coords = [i for i in range(1, epochs + 1)]

    ax2.plot(list_num_epochs, test_epoch_accuracy, color='black', linewidth=2.0, label='Average Validation Accuracy')
    ax2.plot(list_num_epochs, train_epoch_accuracy, color='green', linewidth=2.0, label='Average Training Accuracy')
    ax1.plot(list_num_epochs, test_epoch_loss, color='red', linewidth=2.0, label='Validation Loss')
    ax1.plot(list_num_epochs, train_epoch_loss, color='blue', linewidth=2.0, label='Training Loss')
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
    plt.savefig(os.path.join(self.hparams.output, 'LearningCurves-{0}-model.png'.format(self.hparams.model)),
        bbox_inches='tight')