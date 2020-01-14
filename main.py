from createKmers import *
from deeplearning import *
from datetime import datetime
# Libraries for creating kmers
from Bio import SeqIO
from os import path
# Libraries for creating model
import tensorflow as tf

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Incorrect number of arguments provided.\nCorrect format: [script] [read] [num of processes]\n")

    print(f'\nProcess reads: START -- {datetime.now()}')
    # Get fastq file name
    FastqFile = str(sys.argv[1])      
    # Get filename
    FileName = FastqFile.split(".gz")[0]
    # Get the number of threads
    numThreads = int(sys.argv[2])

    print(f'Fastq File: {FastqFile}')
    print(f'File Name: {FileName}')
    print(f'Number of processes/threads: {numThreads}')

    # Unzip the file if it does not already exist
    if not path.exists(FileName):
        print("\tUnzipped file does not already exist")
        unzipFile(FastqFile, FileName)

    # Parse Fastq file and create dictionary of 10-mer vectors
    Reads_dict = SeqIO.index(FileName, 'fastq')
    kmers_dict = GetKmersDictionary()
    DictVectors = ParseFastq(Reads_dict, kmers_dict)

    # Create a list with the reads id
    ListReadsID = list(DictVectors.keys())

    # Create matrix
    X_test = createMatrix(DictVectors)

    print(f'Process reads: END -- {datetime.now()}\n')

    # Get order names mapped to integers
    class_mapping = {0: 'Enterobacterales', 1: 'Mycoplasmatales', 2: 'Chlamydiales', 3: 'Vibrionales',
                     4: 'Fusobacteriales', 5: 'Spirochaetales', 6: 'Rhodobacterales', 7: 'Unclassified'}
    print(f'class_mapping dictionary: {class_mapping}\n')

    # reset graph
    tf.compat.v1.reset_default_graph()
    createGraph(numThreads)
