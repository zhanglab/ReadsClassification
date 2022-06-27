import json
import os
import csv


def load_json_dict(filename):
    with open(os.path.join(filename), 'r') as f:
        dict = json.load(f)
    return dict

def create_json_dict(dict, filename):
    f = open(os.path.join(filename), 'w')
    json.dump(dict, f)

def write_list_to_csv(list, filename):
    with open(filename,"w") as f:
        wr = csv.writer(f, delimiter="\n")
        wr.writerow(list)

def get_list_from_file(file):
    with open(file, 'r') as f:
        return [i.rstrip() for i in f.readlines()]
