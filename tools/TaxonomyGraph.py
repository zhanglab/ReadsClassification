import os
import json
from collections import defaultdict

from .utils import create_json

class Graph:
    def __init__(self):
        self.graph = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
        self.ranks = {0: 'class', 1: 'order', 2: 'family', 3: 'genus', 4: 'species'}
        self.records = {0: defaultdict(list), 1: defaultdict(list), 2: defaultdict(list),
                        3: defaultdict(list), 4: defaultdict(list)}

    def add_species(self, lineage, genome):
        s_class, order, family, genus, species = lineage
        self.graph[s_class][order][family][genus][species] = genome
        for rank in range(len(lineage)):
            self.records[rank][lineage[rank]].append(genome)

    def get_label_weights(self, output):
        # function to retrieve the weights to apply to each label during training for the class
        # order, family and genus ranks/classifiers
        for rank, rank_values in self.records.items():
            class_weights = {}
            class_mapping = {}
            total_num_genomes = 0
            for index, (taxon, genomes) in enumerate(rank_values.items()):
                class_mapping[index] = taxon
                class_weights[index] = len(genomes)
                total_num_genomes += len(genomes)

            for index, weight in class_weights.items():
                class_weights[index] = total_num_genomes / (len(self.records[rank]) * weight)

            # create a json file to store the weights and another one to the labels mapped to integers
            create_json(output, '{}-weights.json'.format(value), class_weights)
            create_json(output, '{}-integers.json'.format(value), class_mapping)
            create_json(output, '{}-records.json'.format(value), self.records[rank])