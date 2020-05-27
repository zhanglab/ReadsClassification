#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: ./createFiles.sh [Folder Name]"
    exit 2
fi

FOLDER=$1
COV=1
Path="/data/zhanglab/esameth/"
INI="default_config.ini"

function genomeSize {
    chars=$(grep -v ">" "$FILE" | wc -c)
    lines=$(grep -v ">" "$FILE" | wc -l)
    totalchars=$((chars-lines))
    size_gbp=$(bc <<< "scale=5 ; $COV * $totalchars / 1000000000")
    echo "Genome size: $size_gbp"
}

function taxID {
    ID=$(grep "$SPECIES" "/data/zhanglab/cecile_cres/RefSeq-03-07-2020/assembly_summary.txt" | awk -F'\t' '{print $6}')
    echo "taxID: $ID"
}

function CAMISIMfiles {
    echo "genome_ID OTU novelty_category    NCBI_ID source" > "$dir"/metadata.tsv
    echo "$SPECIES  0   known strain    $ID " >> "$dir"/metadata.tsv
    echo "$SPECIES  $FILE" > "$dir"/genome_to_id.tsv
    cp $Path$INI $Path$dir$INI
    sed -i "/^output_directory=/ c\output_directory="$Path$dir"output" $Path$dir$INI
    sed -i "/^metadata=/ c\metadata="$Path$dir"metadata.tsv" $Path$dir$INI
    sed -i "/^id_to_genome_file=/ c\id_to_genome_file="$Path$dir"genome_to_id.tsv" $Path$dir$INI
}

for dir in $FOLDER/*/ ; do
    SPECIES="$(basename $dir)"
    FILE=$Path$(ls "$dir"*.fna)
    echo "$SPECIES"
    echo "$FILE"
    genomeSize
    taxID
    CAMISIMfiles
done
