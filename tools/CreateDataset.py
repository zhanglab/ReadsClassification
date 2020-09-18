import sys
import os
import json
from collections import defaultdict
from datetime import datetime
from .TaxonomyGraph import *


class Dataset:

    def __init__(self):
        self.dict_genomes = dict()

    def GetNCBITaxID(self, target_species):
        with open(os.path.join(os.getcwd(), 'ReadsClassification/tools/rankedlineage.dmp'), 'r') as f:
            for line in f:
                line = line.strip('\n')
                fields = line.split('|')
                taxid = fields[0].strip('\t')
                species = fields[1].strip('\t')
                if species == target_species:
                    return taxid

    def GetRepresentativeGenome(self, target_taxid):
        representative_genome = str()
        taxonomy = str()
        all_genomes = {}  # key = genome id in gtdb and value = ncbi_assembly_name
        with open(os.path.join(os.getcwd(), 'ReadsClassification/tools/bac120_metadata_r89.tsv'),
                  'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                line = line.strip('\n')
                fields = line.split('\t')
                all_genomes[fields[0]] = fields[46]
                if fields[73] == target_taxid or fields[72] == target_taxid:
                    representative_genome = fields[14]
                    taxonomy = fields[16]
        if representative_genome != '':
            ncbi_assembly_name = all_genomes[representative_genome]
            # verify that genome
            return '_'.join([representative_genome, ncbi_assembly_name]), taxonomy
        else:
            return representative_genome, taxonomy

    def CleanTaxonomy(self, lineage):
        ranks = {0: 'species', 1: 'genus', 2: 'family', 3: 'order', 4: 'class'}
        new_lineage = []
        # remove tabs and replace unknown taxa by a specific name
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
        # get all genomes available in a dictionary
        set_fasta_avail = self.GetFastaFiles()
        print(len(set_fasta_avail))
        graph = Graph()

        dataset_absent_species = open(os.path.join(args.output, 'Dataset-absent'), 'w')
        dataset_present_species = open(os.path.join(args.output, 'Dataset-present'), 'w')
        dataset_duplicate_species = open(os.path.join(args.output, 'Dataset-duplicate'), 'w')
        with open(os.path.join(args.input, 'Species')) as f:
            for line in f:
                line = line.strip('\n')
                taxid = self.GetNCBITaxID(line)
                representative_genome, taxonomy = self.GetRepresentativeGenome(taxid)
                # only consider species with genomes that have a representative genome
                if representative_genome != '' and taxonomy != '':
                    # verify that fasta file exists
                    if '_'.join([representative_genome[3:],'genomic.fna.gz'])  in set_fasta_avail:
                        # if target_file != None:
                        lineage = self.CleanTaxonomy(taxonomy)
                        if representative_genome not in self.dict_genomes:
                            # initialize graph with first lineage
                            dataset_present_species.write(
                                '{0}\t{1}\t{2}\t{3}\n'.format(line, taxid, lineage[0], representative_genome))
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
                                '{0}\t{1}\t{2}\t{3}\t{4}\n'.format(line, taxid, lineage[0], representative_genome,
                                                                   self.dict_genomes[representative_genome]))
                    # consider as absent genomes without fasta file in records
                    else:
                        dataset_absent_species.write('{0}\t{1}\n'.format(line, taxid))
                # consider as absent genomes without any representative genomes
                else:
                    dataset_absent_species.write('{0}\t{1}\n'.format(line, taxid))

        with open(os.path.join(args.output, 'graph.json'), 'w', encoding='utf-8') as f:
            json.dump(graph.graph, f, ensure_ascii=False, indent=4)

        return graph
