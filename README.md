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
|:------:|:-------:|:--------:|:----------:|:-------:|
|GCA_000495505.1| Campylobacter testudinum |   34796  |   14912    |         |
|GCA_000832905.1| Bacillus_E coagulans |   65989  |   28281    |         |




0       GCF_000814265.1 34816   14920
1       GCA_000832905.1 65989   28281
2       GCA_001697145.1 39226   16810
2       GCF_900636525.1 36393   15597
3       GCA_002243705.1 55116   23620
4       GCA_003325215.1 106463  45627
4       GCF_001629755.1 93674   40146
4       GCF_001629775.1 93674   40146
4       GCF_002211865.1 95237   40815
4       GCF_002211965.1 99505   42645
4       GCF_002794415.1 99711   42733
4       GCF_003994555.1 95193   40797
4       GCF_003994635.1 95266   40828
4       GCF_004126495.1 94717   40593
4       GCF_900475895.1 93835   40215
5       GCA_006974205.1 100264  42970
5       GCF_003182655.1 100793  43197
5       GCF_003355135.1 99456   42624



0 Campylobacter testudinum d__Bacteria;p__Campylobacterota;c__Campylobacteria;o__Campylobacterales;f__Campylobacteraceae;g__Campylobacter;s__Campylobacter testudinum
1 Bacillus_E coagulans d__Bacteria;p__Firmicutes;c__Bacilli;o__Bacillales;f__Bacillaceae_C;g__Bacillus_E;s__Bacillus_E coagulans
2 Streptococcus anginosus_C d__Bacteria;p__Firmicutes;c__Bacilli;o__Lactobacillales;f__Streptococcaceae;g__Streptococcus;s__Streptococcus anginosus_C
3 Anoxybacillus flavithermus d__Bacteria;p__Firmicutes;c__Bacilli;o__Bacillales;f__Bacillaceae;g__Anoxybacillus;s__Anoxybacillus flavithermus
4 Salmonella diarizonae d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Salmonella;s__Salmonella diarizonae
5 Serratia marcescens d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Serratia;s__Serratia marcescens

genus
0 Campylobacter
1 Bacillus_E
2 Streptococcus
3 Anoxybacillus
4 Salmonella
5 Serratia

Family
0 Campylobacteraceae
1 Bacillaceae_C
2 Streptococcaceae
3 Bacillaceae
4 Enterobacteriaceae

Order
0 Campylobacterales
1 Bacillales
2 Lactobacillales
3 Enterobacterales

Class
0 Campylobacteria
1 Bacilli
2 Gammaproteobacteria


## Contact
