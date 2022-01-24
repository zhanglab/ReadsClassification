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

The complete dataset information with the number of reads in the training, validation adntesting sets is given in the following table:

| GTDB Genome ID | Species | Training | Validation | Testing |
|:------:|:-------:|:--------:|:----------:|:-------:|
|GCA_000495505.1| Campylobacter testudinum |   34796  |   14912    |    -    |
|GCF_000814265.1| Bacillus_E coagulans |   34816  |   14920    |    -    |
|GCA_000832905.1| Bacillus_E coagulans |   65989  |   28281    |    -    |
|GCA_001697145.1|Streptococcus anginosus_C|39226|16810|-|
|GCF_900636525.1|Streptococcus anginosus_C| 36393|   15597|-|
|GCA_002243705.1|Anoxybacillus flavithermus| 55116 |  23620|-|
|GCA_003325215.1|Salmonella diarizonae| 106463 | 45627|-|
|GCF_001629755.1|Salmonella diarizonae| 93674  | 40146|-|
|GCF_001629775.1|Salmonella diarizonae| 93674  | 40146|-|
|GCF_002211865.1|Salmonella diarizonae| 95237  | 40815|-|
|GCF_002211965.1|Salmonella diarizonae| 99505  | 42645|-|
|GCF_002794415.1|Salmonella diarizonae| 99711  | 42733|-|
|GCF_003994555.1|Salmonella diarizonae| 95193  | 40797|-|
|GCF_003994635.1|Salmonella diarizonae| 95266  | 40828|-|
|GCF_004126495.1|Salmonella diarizonae| 94717  | 40593|-|
|GCF_900475895.1|Salmonella diarizonae| 93835  | 40215|-|
|GCA_006974205.1|Serratia marcescens| 100264 | 42970|-|
|GCF_003182655.1|Serratia marcescens| 100793 | 43197|-|
|GCF_003355135.1|Serratia marcescens| 99456  | 42624|-|
|GCF_001484645.1|Campylobacter testudinum|-|-| 21828|
|GCF_002973655.1|Campylobacter testudinum|-|-| 22362|
|GCF_000217835.1|Bacillus_E coagulans |-|-| 36862|
|GCF_000831165.1|Streptococcus anginosus_C|-|-| 24436|
|GCF_000019045.1|Anoxybacillus flavithermus|-|-| 34158|
|GCF_001628755.1|Salmonella diarizonae|-|-| 57352|
|GCF_002220535.1|Serratia marcescens|-|-| 61860|
|GCF_003967055.1|Serratia marcescens|-|-|60862|


## Contact
