import os
import json
from collections import defaultdict

class Graph:

    def __init__(self):
        self.graph = {}
        self.track = 0
        self.ranks = {0: 'class', 1: 'order', 2: 'family', 3: 'genus', 4: 'species'}
        self.records = {0: defaultdict(list), 1: defaultdict(list), 2: defaultdict(list),
                        3: defaultdict(list), 4: defaultdict(list)}

    def AddSpecies(self, lineage, genome):
        branch = {}
        for i in range(len(lineage)):
            if i == 0:
                # initialize with a dictionary mapping the species to an empty list of genomes
                branch = {lineage[i]: genome}
            # progressively add the highest level taxonomic ranks
            elif i >= 1:  # if i == 2 to 5
                branch = {lineage[i]: branch}
        self.track += 1
        return branch

    def CountGenomes(self, lineage, genome):
        iteration = 0
        for i in range(len(lineage)):
            self.records[iteration][lineage[i]].append(genome)
            iteration += 1

    def BuildGraph(self, lineage, genome):
        dictionary = self.graph
        # add count of genomes to records
        self.CountGenomes(lineage, genome)
        for i in range(len(lineage)):
            if lineage[i] not in dictionary:
                # create a branch
                taxa_to_add = lineage[i:len(lineage)]
                taxa_to_add.reverse()
                branch = self.AddSpecies(taxa_to_add, genome)
                # add the branch to the rank
                dictionary[lineage[i]] = branch[lineage[i]]
                break
            else:
                # update dictionary variable and lookup for taxa in lower taxonomic levels
                dictionary = dictionary[lineage[i]]

    def GetLabelWeights(self, args):
        # function to retrieve the weights to apply to each label during training for the class
        # order, family and genus ranks/classifiers
        for key, value in self.ranks.items():
            class_weights = {}
            class_mapping = {}
            total_num_genomes = 0
            for list_genomes in self.records[key].values():
                total_num_genomes += len(list_genomes)
            index = 0
            for taxon, list_genomes in self.records[key].items():
                class_mapping[index] = taxon
                class_weights[index] = total_num_genomes / (len(self.records[key]) * len(list_genomes))
                index += 1

            # create a json file to store the weights and another one to the labels mapped to integers

            with open(os.path.join(args.output, '{}-weights.json'.format(value)), 'w', encoding='utf-8') as f:
                json.dump(class_weights, f, ensure_ascii=False, indent=4)

            with open(os.path.join(args.output, '{}-integers.json'.format(value)), 'w', encoding='utf-8') as f:
                json.dump(class_mapping, f, ensure_ascii=False, indent=4)

            with open(os.path.join(args.output, '{}-records.json'.format(value)), 'w', encoding='utf-8') as f:
                json.dump(self.records, f, ensure_ascii=False, indent=4)



