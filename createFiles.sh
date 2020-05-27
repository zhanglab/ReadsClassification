#!/bin/bash

#SBATCH --partition=general,zhanglab
#SBATCH --time=01:00:00 # walltime limit (HH:MM:SS)
#SBATCH --ntasks-per-node=2 # processor core(s) per node
#SBATCH --nodes=1 # number of nodes
module load CAMISIM/1.1.0-foss-2016b-Python-2.7.12

if [ "$#" -ne 1 ]; then
    echo "Usage: ./createFiles.sh [Folder Name]"
    exit 2
fi

FOLDER=$1
COV=1
Path="/data/zhanglab/esameth/"
INI="default_config.ini"
Genomes="/data/zhanglab/cecile_cres/RefSeq-03-07-2020/RefSeq-03-07-2020"

# Calculate genome size 
function genomeSize {
    chars=$(grep -v ">" "$FILE" | wc -c)
    lines=$(grep -v ">" "$FILE" | wc -l)
    totalchars=$((chars-lines))
    size_gbp=$(bc <<< "scale=5 ; $COV * $totalchars / 1000000000")
    echo "Genome size: $size_gbp"
}

# Get the species tax id for metadata.tsv
function taxID {
    ID=$(grep "$SPECIES" "/data/zhanglab/cecile_cres/RefSeq-03-07-2020/assembly_summary.txt" | awk -F'\t' '{print $6}')
    echo "taxID: $ID"
}

# Create the metadata.tsv, genome_to_id.tsv, and default_config.ini files
function CAMISIMfiles {
    echo "genome_ID OTU novelty_category    NCBI_ID source" > "$dir"/metadata.tsv
    echo "$SPECIES  0   known strain    $ID " >> "$dir"/metadata.tsv
    echo "$SPECIES  $FILE" > "$dir"/genome_to_id.tsv
    cp $Path$INI $Path$dir$INI
    # Replace the output_directory, metadata, and id_to_genome_file paths to the correct path
    sed -i "/^output_directory=/ c\output_directory="$Path$dir"output" $Path$dir$INI
    sed -i "/^metadata=/ c\metadata="$Path$dir"metadata.tsv" $Path$dir$INI
    sed -i "/^id_to_genome_file=/ c\id_to_genome_file="$Path$dir"genome_to_id.tsv" $Path$dir$INI
}

# Create the CAMISIM files for every species folder
for dir in $FOLDER/*/ ; do
    # Get the species name based on the folder name
    SPECIES="$(basename $dir)"
    echo "$SPECIES"
    # Get the name of the fna file for the species
    FILE=$(find $Genomes -name "$SPECIES"*.fna)
    echo "$FILE"
    genomeSize
    taxID
    CAMISIMfiles
    # Run CAMISIM
    /opt/software/CAMISIM/1.1.0-foss-2016b-Python-2.7.12/bin/metagenomesimulation.py $Path$dir$INI
    echo "CAMISIM files for $SPECIES complete\n"
done
