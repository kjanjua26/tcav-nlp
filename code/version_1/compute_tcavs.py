"""
    Compute the TCAVs for the given CAVs against each concept for each sentence in the base_data.txt.
    Use base_data_labels.txt for word analysis.
    Computes both ways: sentence TCAV and word TCAV.
    
    python compute_tcavs.py -b ../../../data/extractions-professions-mask.json -c ../../../data/results/layer_wise_cavs.pickle -r ../../../data/results/layer_wise_random_cavs.pickle -o ../../../data/results/
"""

import argparse, sys, os
import matplotlib.pyplot as plt
from scipy import stats
import pickle
import numpy as np
from time import perf_counter
from prepare_concepts import segregate_acts_for_each_layer

sys.path.append("/Users/Janjua/Desktop/QCRI/Work/aux_classifier/")
import aux_classifier.data_loader as data_loader

def load_pickle_files(pickled_file):
    """
    Load the pickle files.
    
    Arguments
        pickled_file (pickle): the file to un-pickle.

    Returns
        unpicked_file: the unpickled file.
    """
    
    with open(pickled_file, "rb") as reader:
        unpicked_file = pickle.load(reader)
    
    return unpicked_file

def directional_derivative(ft, cav):
    """
    Implements the directional derivative (dot product) of two vectors.

    Arguments
        ft (np.array): the ft vector.
        cav (np.array): the computed CAV.
    
    Returns:
        bool value: if < 0.
    """
    dot = np.dot(ft, cav)
    return dot < 0

def compute_sentence_tcav(concept_cav_per_layer, random_cav_per_layer, bottleneck_base_per_layer):
    """
    Compute TCAV scores for a given CAV of a concept and each sentence in the base corpus.

    Arguments
        concept_cav_per_layer (dict): the CAV of a specific layer for every concept.
        random_cav_per_layer (dict): the random CAV of a specific layer for every concept.
        bottleneck_base_per_layer (dict): the bottleneck base (acts) of a specific layer.
    
    Returns
        tcavs_sentence_dict (dict): the TCAV score for a specific layer of every concept.
    """
    def compute_tcav(cav_per_layer):
        tcavs = {}
        for concept, cav in cav_per_layer.items():
            print(f"For Concept - {concept}")
            count = 0
            for ix, acts in enumerate(bottleneck_base_per_layer):
                for ft in acts:
                    dydx = directional_derivative(ft, cav)
                    if dydx: count += 1
            tcav = float(count)/float(len(bottleneck_base_per_layer)*len(acts))
            tcavs[concept] = tcav
        return tcavs
    
    tcavs_concept = compute_tcav(concept_cav_per_layer)
    tcavs_random = compute_tcav(random_cav_per_layer)

    return (tcavs_concept, tcavs_random)

def compute_word_tcav():
    """
    Compute TCAV scores for a given CAV of a concept and every word in each sentence in the base corpus.
    """
    pass

def run(concept_cavs, random_cavs, bottleneck_base):
    """
    Compute the TCAV of each concept for every layer.

    Arguments
        concept_cavs (dict): the concept CAVs of each layer.
        random_cavs (dict): the random concept CAVs of each layer.
        bottleneck_base (dict): the bottleneck base activations of each layer.
    """
    layer_wise_tcavs = {}
    layer_wise_random_tcavs = {}

    for layer, concept_cav_per_layer in concept_cavs.items():
        print(f"For Layer - {layer}")
        random_cav_per_layer = random_cavs[layer]
        base_act_per_layer = bottleneck_base[layer]

        tcavs_sentence_concept, tcavs_sentence_random = compute_sentence_tcav(concept_cav_per_layer, random_cav_per_layer, base_act_per_layer)
        layer_wise_tcavs[layer] = tcavs_sentence_concept
        layer_wise_random_tcavs[layer] = tcavs_sentence_random
    
    return (layer_wise_tcavs, layer_wise_random_tcavs)

def layer_wise_plots(layer_wise_tcavs, num_layers, output_folder):
    """
    Layer wise plots of all the concepts i.e. a plot for each layer.
    
    Arguments
        layer_wise_tcavs (dict): the layer wise tcavs.
        num_layers (int): the number of layers in the model.
        output_folder (str): the folder to store the images in.
    """
    for ix in range(num_layers):
        scores = layer_wise_tcavs[f"{ix+1}"]
        x_points = [x for x in range(len(scores.keys()))]
        keys = list(scores.keys())
        plt.rcParams["figure.figsize"] = (8,8)
        plt.bar(x_points, list(scores.values()), width=0.2, color='k')
        plt.xticks(x_points, keys, rotation=70)
        plt.title(f"TCAV Scores for Various Concepts at Layer {ix+1}")
        plt.ylabel('TCAV Score')
        plt.xlabel('Concepts')
        plt.savefig(output_folder + f"layer_wise/{ix+1}.png")
        plt.close()
    
def concept_wise_bar_plots(layer_wise_tcavs, num_layers, output_folder):
    """
    A better plot - a single concept at every layer (all layers in one plot.)

    Arguments
        layer_wise_tcavs (dict): the layer wise tcavs.
        num_layers (int): the number of layers in the model.
        output_folder (str): the folder to store the images in.
    """
    concepts_all_layers_plots = {}

    # rearrange the dicts.
    for ix in range(1, num_layers+1):
        scores = layer_wise_tcavs[f"{ix}"]
        for k, v in scores.items():
            if k not in list(concepts_all_layers_plots.keys()):
                concepts_all_layers_plots[k] = []
                concepts_all_layers_plots[k].append(v)
            else:
                concepts_all_layers_plots[k].append(v)

    # construct the plots
    for k, v in concepts_all_layers_plots.items():
        x_points = [x for x in range(len(v))]
        plt.rcParams["figure.figsize"] = (8,8)
        plt.bar(x_points, v, width=0.2, color='k')
        plt.xticks(x_points, [f"Layer-{ix+1}" for ix in range(len(v))], rotation=70)
        plt.title(f"TCAV Scores for Concept {k} at various layers")
        plt.ylabel('TCAV Score')
        plt.xlabel('Layers')
        plt.savefig(output_folder + f"concept_wise/{k}.png")
        plt.close()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--base_corpus", 
        help="Path to base corpus to compute the derivative and results against.")
    parser.add_argument("-c", "--concepts_cavs", 
        help="The concept CAV pickle file computed.")
    parser.add_argument("-r", "--randoms_cavs", 
        help="The random CAV pickle file computed for the t-test.")
    parser.add_argument("-o", "--output_directory",
        help="The output directory to store the results in.")

    args = parser.parse_args()
    num_neurons = 768

    base_corpus = args.base_corpus
    output_directory = args.output_directory

    start = perf_counter()
    concept_cavs = load_pickle_files(args.concepts_cavs)
    random_cavs = load_pickle_files(args.randoms_cavs)

    base_acts, base_num_layers = data_loader.load_activations(base_corpus, num_neurons)
    bottleneck_base = segregate_acts_for_each_layer(base_acts, base_num_layers)

    layer_wise_tcavs, layer_wise_random_tcavs = run(concept_cavs, random_cavs, bottleneck_base)
    end = perf_counter()
    print(f"Computed in {end-start}s")

    print("Plotting Now.")
    concept_wise_bar_plots(layer_wise_tcavs, base_num_layers, output_directory)

    layer_wise_plots(layer_wise_tcavs, base_num_layers, output_directory)

if __name__ == '__main__':
    main()