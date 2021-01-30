"""
    Compute the TCAVs for the given CAVs against each concept for each sentence in the base_data.txt.
    Use base_data_labels.txt for word analysis.
    Computes both ways: sentence TCAV and word TCAV.
    
    python compute_tcavs.py -b ../../../data/extractions-professions-mask.json -c ../../../data/results/layer_wise_cavs.pickle -r ../../../data/results/layer_wise_random_cavs.pickle -o ../../../data/results/

    # real
    python compute_tcavs.py -b ../../../data/extractions-professions-mask.json -c ../../../data/results/layer_wise_cavs.pickle -r ../../../data/results/layer_wise_random_cavs.pickle -o ../../../data/results/ -bs ../../../data/mask.txt -m "w"
"""

import argparse, sys, os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import stats
import pickle
import numpy as np
from time import perf_counter
from scipy.stats import shapiro
from prepare_concepts import segregate_acts_for_each_layer
from collections import defaultdict

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

def read_sentences(sents_path):
    """
    Reads all the sentences to a list.
    """
    sents = []
    with open(sents_path) as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.strip()
            sents.append(line)
    return sents

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
        tcavs_concept (dict): the TCAV score for a specific layer of every concept.
        tcavs_random (dict): the TCAV score for a specific layer of every concept (random CAV).
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
            is_normal = is_normality_shapiro(tcav)
            
            if is_normal:
                tcavs[concept] = tcav
            else:
                tcavs[concept] = list(np.zeros(np.array(tcav).shape)) # since the TCAV is not normal, make it 0.

        return tcavs
    
    tcavs_concept = compute_tcav(concept_cav_per_layer)
    tcavs_random = compute_tcav(random_cav_per_layer)

    return (tcavs_concept, tcavs_random)

def run_sent(concept_cavs, random_cavs, bottleneck_base):
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

def compute_word_tcav(concept_cavs, bottleneck_base, sentences, num_layers):
    """
    Compute TCAV scores for a given CAV of a concept and every word in each sentence in the base corpus.

    Arguments
        concept_cavs (dict): the dictionary of CAVs of each concept for every layer.
        bottleneck_base (dict): the dictionary of base activations.
        sentences (list): the list of sentences.
        num_layers (int): the number of layers.
    
    Returns
        word_tcav (dict): the dictionary of word weightage for each layer for each word for each concept.
    """

    word_tcav = defaultdict(dict)
    word_tcav = {str(k): {} for k in range(1, num_layers+1)}
    
    for jx, sent in enumerate(sentences):
        words = list(sent.split(' '))
        for ix in range(1, num_layers+1):
            act_per_layer_per_sent = bottleneck_base[str(ix)][jx]
            layer_cavs = concept_cavs[str(ix)]
            for concept, cav in layer_cavs.items():
                word_tcav[str(ix)][concept] = []
                count = 0
                for fx, act in enumerate(act_per_layer_per_sent):
                    dydx = directional_derivative(act, cav)
                    
                    if dydx: count += 1
                    tcav = float(count)/float(len(act_per_layer_per_sent))
                    word_tcav[str(ix)][concept].append((words[fx], tcav))

    return word_tcav

def get_specific_word_weightage(word_tcav, word_to_pick):
    """
    Returns the weightage (TCAV) of a specific word for all the concepts at all layers.

    Arguments
        word_tcav (dict): the dictionary of word weightage for each layer for each word for each concept.
        word_to_pick (str): the word to get the weightage of for each layer.
    
    Returns
        word_weightage (dict): the dictionary of the specific word with weightage for each layer.
    """
    word_weightage = defaultdict(dict)
    for layer, word_concept_dict in word_tcav.items():
        word_weightage[layer] = {}
        for concept, word_weight_list in word_concept_dict.items():
            
            for ix in word_weight_list:
                word, weight = ix
                if word == word:
                    word_weightage[layer][concept] = weight
    return word_weightage

def is_normality_shapiro(tcav_to_test):
    """
    Perform the test for normality (Shapiro-Wilk) for TCAVs.
    The assumption is: the correct CAV's corresponding TCAV should be normal.
    """
    alpha = 0.05
    stat, p = shapiro(tcav_to_test)
    if p > alpha: return False
    else: return True

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

def word_layer_wise_plots(word_weightage, output_folder, word):
    """
    Plot a specific word for each concept at each layer.

    Arguments
        word_weightage (dict): the dictionary of the specific word with weightage for each layer.
    """
    layers = list(word_weightage.keys())
    concepts = list(word_weightage["1"].keys())
    lst_of_colors = ["xkcd:black", "xkcd:red", "blue", "xkcd:green", "xkcd:eggplant purple", "xkcd:deep pink",
                    "xkcd:topaz", "xkcd:chocolate", "xkcd:ocean blue", "xkcd:dark maroon", 
                    "xkcd:violet", "xkcd:poo brown", "xkcd:gunmetal"]

    data = []
    for layer in layers:
        layer_concept_weights = []
        for concept in concepts:
            weight = word_weightage[layer][concept]
            layer_concept_weights.append(weight)

        data.append(layer_concept_weights)

    gap = .8 / len(data)

    plt.figure(figsize=(20,10))
    for i, row in enumerate(data):
        X = np.arange(len(row))
        rect = plt.bar(X + i * gap, row, width = gap, color = lst_of_colors[i])
        plt.xticks(X, [ix for ix in concepts], rotation=70)
        for j in range(len(row)):
            height = rect[j].get_height()
            plt.text(rect[j].get_x() + rect[j].get_width()/2.0, height, f"{i+1}", ha='center', va='bottom')
    
    plt.ylabel("TCAV Scores")
    plt.title(f"TCAV Scores for each layer for word {word}")
    plt.savefig(output_folder + f"{word}_all_layers.png")
    
    plt.show()
    
def write_the_tcavs(output_path, layer_wise_tcavs, layer_wise_random_tcavs):
    """
    Write the computed CAVs (for both random and concept) in pickle files for further use.

    Arguments
        output_path (str): the output folder name.
        layer_wise_tcavs (dict): the layerwise computed TCAVs.
        layer_wise_random_tcavs (dict): the layerwise computed random TCAVs.
    """
    layer_wise_tcavs_path = output_path + "/layer_wise_tcavs.pickle"
    layer_wise_random_tcavs_path = output_path + "/layer_wise_random_tcavs.pickle"

    with open(layer_wise_tcavs_path, "wb") as writer:
        pickle.dump(layer_wise_tcavs, writer)
    
    with open(layer_wise_random_tcavs_path, "wb") as writer:
        pickle.dump(layer_wise_random_tcavs, writer)

def write_word_tcavs(output_path, word_layer_wise_tcavs):
    """
    Write the word layer wise TCAV to pickle files.

    Arguments
        word_layer_wise_tcavs (dict): the dictionary of word weightage for each layer for each word for each concept.
    """
    word_layer_wise_path = output_path + "/word_layer_wise.pickle"

    with open(word_layer_wise_path, "wb") as writer:
        pickle.dump(word_layer_wise_tcavs, writer)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--base_acts_to_compute", 
        help="Path to base acts to compute the derivative and results against.")
    parser.add_argument("-bs", "--base_sentences", 
        help="Path to base sentences to compute the word TCAV.")
    parser.add_argument("-c", "--concepts_cavs", 
        help="The concept CAV pickle file computed.")
    parser.add_argument("-r", "--randoms_cavs", 
        help="The random CAV pickle file computed for the t-test.")
    parser.add_argument("-o", "--output_directory",
        help="The output directory to store the results in.")
    parser.add_argument("-m", "--compute_mode",
        help="The compute mode to compute the TCAV in (w for word, s for sentence).")

    args = parser.parse_args()
    num_neurons = 768
    word = "[MASK]"

    base_acts_to_compute = args.base_acts_to_compute
    base_sentences = args.base_sentences
    output_directory = args.output_directory
    compute_mode = args.compute_mode

    start = perf_counter()
    concept_cavs = load_pickle_files(args.concepts_cavs)
    random_cavs = load_pickle_files(args.randoms_cavs)
    sents = read_sentences(base_sentences)
    
    base_acts, base_num_layers = data_loader.load_activations(base_acts_to_compute, num_neurons)
    bottleneck_base = segregate_acts_for_each_layer(base_acts, base_num_layers)

    if compute_mode == "s":
        print("Computing Sentence Level Results.")
        layer_wise_tcavs, layer_wise_random_tcavs = run_sent(concept_cavs, random_cavs, bottleneck_base)
        write_the_tcavs(output_directory, layer_wise_tcavs, layer_wise_random_tcavs)
        end = perf_counter()
        print(f"Computed in {end-start}s")

        print("Plotting Now.")
        concept_wise_bar_plots(layer_wise_tcavs, base_num_layers, output_directory)
        layer_wise_plots(layer_wise_tcavs, base_num_layers, output_directory)

    elif compute_mode == "w":
        print("Computing Word Level Results.")
        word_layer_wise_tcavs = compute_word_tcav(concept_cavs, bottleneck_base, sents, base_num_layers)
        write_word_tcavs(output_directory, word_layer_wise_tcavs)
        weight_dict = get_specific_word_weightage(word_layer_wise_tcavs, word)
        word_layer_wise_plots(weight_dict, output_directory, word)


if __name__ == '__main__':
    main()