#!/bin/bash

conda activate tf-env

echo "START at $(date)"

DATA_SET_DIR=/home/uri/data/species-data/imbalanced/tfrecords

cd $DATA_SET_DIR

TFRECORDS=$(ls --ignore *\.txt)
echo $TFRECORDS

# create directory to store index files 
if [ ! -e idx_files ]; then 
	echo "create directory idx_files" 
	mkdir idx_files
else
	echo "idx_files already exists!"
fi

# run tfrecord2idx script for each tfrecord in DATA_SET_DIR
for file in $TFRECORDS;
do
	echo ${file}
	python /home/uri/scripts/tfrecord2idx.py $DATA_SET_DIR/${file} \
		$DATA_SET_DIR/idx_files/${file}.idx;
done

echo "END at $(date)"

