# DL-TODA: A Deep learning tool for omics data analysis

## About
DL-TODA is a program to perform taxonomic classification and binning of metagenomic data. 
DL-TODA can be used to train a deep learning model based on a list of taxa from the user's choice.
Alternatively, a model able to classify reads into 3313 different bacterial species is provided.

## Install DL-TODA
- Download DL-TODA using git
- Setup conda environment
- Installation of dependencies
- Hardware architecture

## Run DL-TODA
- Create tfrecords
- Training model
- Testing model
- Classification of metagenomic data

## 3313 bacterial species model

## Tutorial to build a toy model
DL-TODA installation can be verified by training and testing a small model of 6 bacterial species.

The data (fastq files and tfrecords) along with the trained model, testing results and accessory files can be found here: [toy model] (github link to directory)

Reads were simulated using ART read simulator (give parameters) and converted to tfrecords. Each training, validation and testing subsets contains an even number of reads per genome and species. 

The complete dataset information is given in the following table:

| Genome | Species | Training | Validation | Testing |


## Contact
