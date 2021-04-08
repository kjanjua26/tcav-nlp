"""
Aggregate the learned CAVs to run it with TCAV.
"""

import argparse, os
import glob, json, pickle
import numpy as np

def read_json(json_file):
    """
    Read the JSON file.
    """
    with open(json_file) as jp:
        data = json.load(jp)
        return data

def write_pickle_file(layer_wise_cavs, output_path):
    """
    Dump the data to a pickle file.
    """
    layer_wise_cavs_path = output_path + "/layer_wise_cavs.pickle"

    with open(layer_wise_cavs_path, "wb") as writer:
        pickle.dump(layer_wise_cavs, writer)

def combine(aggregation_folder, list_of_concepts, 
            num_of_layers, output_directory):

    cavs = {str(k): {} for k in range(1, num_of_layers+1)}

    # fill-up the CAVs.    
    for l, v in cavs.items():
        for concept in list_of_concepts:
            v[concept] = {}

    for fp in glob.glob(aggregation_folder + "*.json"):
        
        concept = fp.split('/')[-1].split('.')[0].split('_')[-3]
        try:
            layer = str(int(fp.split('/')[-1].split('.')[0].split('_')[-1])+1)
        except:
            pass
        if concept in list_of_concepts:
            json_cav = read_json(fp)
            json_cav = np.array(json_cav['weights'])
            cavs[layer][concept] = json_cav

    write_pickle_file(cavs, output_directory)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--aggregation_folder",
        help="The folder that contains the separate CAVs.")
    parser.add_argument("-lc", "--lst_concepts",
        help="The list of concepts.")
    parser.add_argument("-o", "--output_directory",
        help="The output directory to write the data to.")

    args = parser.parse_args()
    aggregation_folder = args.aggregation_folder
    list_of_concepts = args.lst_concepts.split(' ')
    output_directory = args.output_directory
    num_of_layers = 13

    combine(aggregation_folder, list_of_concepts, num_of_layers, output_directory)

if __name__ == '__main__':
    main()