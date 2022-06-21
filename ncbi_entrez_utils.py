from Bio import Entrez
from ncbi_tax_utils import get_ncbi_taxonomy, parse_nodes_file, parse_names_file
import argparse

def retrieve_protein_seq(uid):

def retrieve_genome_sp(uid):
    request = Entrez.esummary(db="assembly", id=uid, rettype="uilist", retmode="text")
    result = Entrez.read(request, validate=False)
    request.close()

    return result['DocumentSummarySet']['DocumentSummary'][0]['SpeciesName'], result['DocumentSummarySet']['DocumentSummary'][0]['SpeciesTaxid']

def retrieve_uid(genome_id):
    request = Entrez.esearch(db="assembly", retmax=10, term=genome_id, id_type="acc")
    result = Entrez.read(request)
    request.close()
    if int(result["Count"]) > 1:
        sys.exit(-1)
    else:
        uid = result["IdList"][0]

    return uid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--email', type=str, help='email required to access NCBI Entrez')
    parser.add_argument('--output_dir', type=str, help='path to output directory')
    parser.add_argument('--genomes', type=str, help='path to tab separated file containing list of genomes id')
    parser.add_argument('--ncbi_db', type=str, help='path to directory containing nodes.dmp and names.dmp NCBI taxonomy files')
    parser.add_argument('--taxonomy', help='taxonomy of genome', action='store_true', required=('--ncbi_db' in sys.argv))
    parser.add_argument('--protein', help='taxonomy of genome', action='store_true')
    args = parser.parse_args()

    Entrez.email = args.email

    with open(args.genomes, 'r') as f:
        content = f.readlines()
        genomes = [i.rstrip().split('\t')[0] for i in content]

    list_uids = []
    for genome_id in genomes:
        list_uids.append(retrieve_uid(genome_id))

    if args.taxonomy:
        d_nodes = parse_nodes_file(os.path.join(ncbi_db, 'nodes.dmp'))
        d_names = parse_names_file(os.path.join(ncbi_db, 'names.dmp'))

        with open(os.path.join(args.output_dir, 'genomes-taxonomy.tsv'), 'w') as out_f:
            for i in range(len(list_uids)):
                species_name, species_taxid = retrieve_genome_sp(uid)
                list_taxids, list_taxa, _ = get_ncbi_taxonomy(species_taxid, d_nodes, d_names)
                out_f.write(f'{genomes[i]}\t{list_uids[i]}\t{list_taxa}\t{list_taxids}\n')

if __name__ == "__main__":
    main()
