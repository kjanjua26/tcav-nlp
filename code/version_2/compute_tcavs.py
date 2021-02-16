"""
    Compute the TCAVs for the given CAVs against each concept for each sentence in the base_data.txt.
    Use base_data_labels.txt for word analysis.
    Computes both ways: sentence TCAV and word TCAV.
"""
import sys
sys.path.append("/Users/Janjua/Desktop/QCRI/Work/aux_classifier/")
import aux_classifier.data_loader as data_loader

import argparse, os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import stats
import pickle
import numpy as np
import pandas as pd
from time import perf_counter
from IPython.display import HTML
from scipy.stats import shapiro
from prepare_concepts import segregate_acts_for_each_layer, get_concepts_dict,  assign_labels_to_concepts
from collections import defaultdict

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

def mask_out_each_concept_in_base_sentences(sents, base_labels, concepts):
    """
    Masks out each concept in base sentences to compute the TCAV for each word for analysis.
    """
    sents_masked = {}

    for concept in concepts:
        sents_masked[concept] = []
        for ix, sent in enumerate(sents):
            label = base_labels[ix]
            try:
                index = label.split(' ').index(concept)
                word_in_sent = sent.split(' ')[index]
                sent = sent.replace(word_in_sent, '[MASK]', 1)
                sents_masked[concept].append(sent)
            except:
                sents_masked[concept].append(sent)
        
    return sents_masked

def compute_word_tcav(concept_cavs, bottleneck_base,
                      sentences, num_layers, word,
                      num_of_runs):
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
    

    for ix in range(1, num_layers+1):
        
        check_for_spurious_cavs = {}
        layer_cavs = concept_cavs[str(ix)]
        for concept, cavs_of_runs in layer_cavs.items():
            print(f"[INFO] Test Concept - {concept}")
            
            tcavs_per_run = []
            for run in range(num_of_runs):
                count = 0
                for cav_key, cav in cavs_of_runs[run].items():
                    #print(cav_key)
                    for jx, sent in enumerate(sentences):
                        act_per_layer_per_sent = bottleneck_base[str(ix)][jx]
                        words = list(sent.split(' '))
                        if word in words:
                            selected_word_index = words.index(word)
                            selected_word_acts = act_per_layer_per_sent[selected_word_index]
                            dydx = directional_derivative(selected_word_acts, cav)
                            if dydx:
                                count += 1

                    tcav = float(count)/float(len(sentences))
                    print(f"Layer - {ix} CAV - {cav_key} TCAV - {tcav}")
                    tcavs_per_run.append(tcav)
        
            max_tcav = max(tcavs_per_run)
            mean_tcav = np.mean(tcavs_per_run)

            print(f"[INFO] Concept - {concept} Mean - {mean_tcav} Max - {max_tcav}")
            if abs(max_tcav - mean_tcav) <= 0.05:
                check_for_spurious_cavs[concept] = max_tcav
            else:
                check_for_spurious_cavs[concept] = 0.0
        
        word_tcav[str(ix)] = check_for_spurious_cavs
    
    return word_tcav

def run_for_chosen_word_write_to_pickle(sentences, concept_cavs,
                                        bottleneck_base, num_layers,
                                        word, output_directory, num_of_runs):
    """
    Computes the TCAV for each word on [MASKED] sentences for each concept.

    Arguments
        sentences (dict): dict of [MASKED] sentences against each concept.
        concept_cavs (dict): the dictionary of CAVs of each concept for every layer.
        bottleneck_base (dict): the dictionary of base activations.
        num_layers (int): the number of layers.
        word (str): the word to get TCAVs for.
    """

    write_file_path = f"{output_directory}" + "/inference_word_mode.pickle"
    concept_masked_tcav_dict = defaultdict(dict)

    for concept_masked, sents in sentences.items(): # these are concept_wise masked sentences.
        if concept_masked in list(concept_cavs["1"].keys()):
            print(f"[INFO] Masked Concept - {concept_masked}")
            word_layer_wise_tcavs = compute_word_tcav(concept_cavs, bottleneck_base, sents, num_layers, word, num_of_runs)
            concept_masked_tcav_dict[concept_masked] = word_layer_wise_tcavs
        
    print(concept_masked_tcav_dict)

    # write to pickle file.
    with open(write_file_path, "wb") as writer:
        pickle.dump(concept_masked_tcav_dict, writer)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--base_acts_to_compute",
        help="Path to base acts to compute the derivative and results against.")
    parser.add_argument("-bs", "--base_sentences",
        help="Path to base sentences to compute the word TCAV.")
    parser.add_argument("-bl", "--base_labels",
        help="Path to base labels.")
    parser.add_argument("-c", "--concepts_cavs",
        help="The concept CAV pickle file computed.")
    parser.add_argument("-o", "--output_directory",
        help="The output directory to store the results in.")
    #parser.add_argument("-m", "--compute_mode", default="wm",
    #    help="The compute mode to compute the TCAV in (w for word, s for sentence, wm for testing for [MASK] only.).")
    parser.add_argument("-w", "--word", default="[MASK]",
        help="The word to compute the TCAVs for in w mode.")
    parser.add_argument("-rs", "--runs",
        help="The number of runs used to get CAVs.")

    args = parser.parse_args()
    num_neurons = 768

    base_acts_to_compute = args.base_acts_to_compute
    base_sentences = args.base_sentences
    base_labels = args.base_labels
    output_directory = args.output_directory
    #compute_mode = args.compute_mode
    num_of_runs = int(args.runs)
    word = args.word

    #start = perf_counter()
    
    concept_cavs = load_pickle_files(args.concepts_cavs)
    
    sents = read_sentences(base_sentences)
    base_labels = read_sentences(base_labels)

    concepts_dict = get_concepts_dict(base_labels)
    concepts2class = assign_labels_to_concepts(concepts_dict)

    base_acts, base_num_layers = data_loader.load_activations(base_acts_to_compute, num_neurons)
    bottleneck_base = segregate_acts_for_each_layer(base_acts, base_num_layers)

    print("[INFO] Computing TCAVs.")
    masked_sents = mask_out_each_concept_in_base_sentences(sents, base_labels, list(concepts2class.values()))
    run_for_chosen_word_write_to_pickle(masked_sents, concept_cavs, bottleneck_base, base_num_layers, word, output_directory, num_of_runs)

if __name__ == '__main__':
    main()