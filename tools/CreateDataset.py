import sys
import os
import re
import json
import pandas as pd
from collections import defaultdict
from datetime import datetime
from .TaxonomyGraph import *


class Dataset:
    def __init__(self):
        self.dict_genomes = dict()
        self.df = pd.read_csv(os.path.join(os.getcwd(), 'ReadsClassification/tools/bac120_metadata_r95.tsv'),
                              delimiter='\t', usecols=['accession', 'ncbi_assembly_name', 'ncbi_species_taxid',
                                                       'ncbi_taxid', 'gtdb_genome_representative', 'gtdb_taxonomy',
                                                       'gtdb_representative'])
        # Drop rows where there is not a representative genome
        self.df = self.df[self.df.gtdb_representative == 't']
        self.all_genomes = dict(zip(self.df.accession, self.df.ncbi_assembly_name))

        # Dataframe to hold rankedlineage.dmp data
        self.ranked_lineage = pd.read_csv(os.path.join(os.getcwd(), 'ReadsClassification/tools/rankedlineage.dmp'),
                                          delimiter='|',
                                          usecols=['1\t', '\troot\t'])
        self.ranked_lineage = self.ranked_lineage.replace(r'\t', '', regex=True)

    def get_ncbi_taxid(self, target_species):
        # Get the tax_id. Return None if the species is not in the file
        row = self.ranked_lineage.loc[self.ranked_lineage['\troot\t'] == target_species, '1\t']
        return None if row.empty else row.iloc[0]

    def get_rep_genome(self, target_taxid):
        representative_genome = str()
        taxonomy = str()
        # Get all rows that contain the taxid
        contain_taxid = self.df[(self.df['ncbi_taxid'] == target_taxid) |
                                (self.df['ncbi_species_taxid'] == target_taxid)]
        # Drop duplicates
        unique_genomes = contain_taxid.drop_duplicates(subset='gtdb_genome_representative')
        if unique_genomes.empty:
            return representative_genome, taxonomy
        # Choose a random genome if there are more than one representative
        taxid_row = unique_genomes.sample()
        representative_genome = taxid_row.gtdb_genome_representative.item()
        taxonomy = taxid_row.gtdb_taxonomy.item()
        ncbi_assembly_name = re.sub(r'\s+', '_', self.all_genomes[representative_genome])
        return '_'.join([representative_genome, ncbi_assembly_name]), taxonomy

    def clean_taxonomy(self, lineage):
        ranks = {0: 'species', 1: 'genus', 2: 'family', 3: 'order', 4: 'class'}
        new_lineage = []
        # Remove tabs and replace unknown taxa by a specific name
        lineage = lineage.split(';')
        lineage.reverse()
        for i in range(len(lineage) - 2):
            taxon = lineage[i].split('__')[1]
            if taxon == '':
                taxon = ' '.join([ranks[i], 'unknown'])
            new_lineage.append(taxon)
        return new_lineage

    def GetFastaFiles(self):
        set_fasta_avail = set()
        for root, dirs, files in os.walk('/project/projectdirs/m3513/srp/ncbi/genomes/all/'):
            for file in files:
                if 'genomic.fna.gz' in file:
                    set_fasta_avail.add(file)
        return set_fasta_avail

    def CreateDataset(self, args):
        # Get all genomes available in a dictionary
        set_fasta_avail = self.GetFastaFiles()
        print(len(set_fasta_avail))
        graph = Graph()

        dataset_absent_species = open(os.path.join(args.output, 'Dataset-absent'), 'w')
        dataset_present_species = open(os.path.join(args.output, 'Dataset-present'), 'w')
        dataset_duplicate_species = open(os.path.join(args.output, 'Dataset-duplicate'), 'w')
        with open(os.path.join(args.input, 'Species')) as f:
            for line in f:
                line = line.strip('\n')
                taxid = self.get_ncbi_taxid(line)
                if not taxid:
                    continue
                representative_genome, taxonomy = self.get_rep_genome(taxid)
                # Only consider species with genomes that have a representative genome
                if representative_genome and taxonomy:
                    # verify that fasta file exists
                    if '_'.join([representative_genome[3:],'genomic.fna.gz']) in set_fasta_avail:
                        lineage = self.clean_taxonomy(taxonomy)
                        if representative_genome not in self.dict_genomes:
                            # initialize graph with first lineage
                            dataset_present_species.write(
                                '{}\t{}\t{}\t{}\n'.format(line, taxid, lineage[0], representative_genome))
                            if graph.track == 0:
                                graph.graph = graph.AddSpecies(lineage, representative_genome)
                                lineage.reverse()
                                graph.CountGenomes(lineage, representative_genome)
                            else:
                                # build up graph with next species
                                lineage.reverse()
                                graph.BuildGraph(lineage, representative_genome)
                            self.dict_genomes[representative_genome] = line
                        else:
                            dataset_duplicate_species.write(
                                '{}\t{}\t{}\t{}\t{}\n'.format(line, taxid, lineage[0], representative_genome,
                                                                   self.dict_genomes[representative_genome]))
                    # consider as absent genomes without fasta file in records
                    else:
                        dataset_absent_species.write('{}\t{}\n'.format(line, taxid))
                # consider as absent genomes without any representative genomes
                else:
                    dataset_absent_species.write('{}\t{}\n'.format(line, taxid))

        with open(os.path.join(args.output, 'graph.json'), 'w', encoding='utf-8') as f:
            json.dump(graph.graph, f, ensure_ascii=False, indent=4)

        return graph
