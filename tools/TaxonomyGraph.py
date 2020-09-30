import os
import json
from collections import defaultdict

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