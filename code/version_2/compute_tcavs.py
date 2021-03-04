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
from prepare_concepts import segregate_acts_for_each_layer, get_concepts_dict, assign_labels_to_concepts
from collections import defaultdict
from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize
from scipy import stats

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

def get_model_unmasker(model_type):
    """
    Returns the model unmasker.
    """
    unmasker = pipeline('fill-mask', model=model_type)
    return unmasker

def get_top_prediction_from_unmasker(unmasker, sent):
    """
    Passes the sentence to the unmasker model and returns the prediction.

    Arguments
        sent (str): the input sentence.
        model_type (str): the type of the model to load unmasker for.

    Returns
        pred_sent (str): the predicted sentence string.
        pred_token (str): the predicted token against MASK.
        pred_score (float): the score of the prediction.
    """
    top_pred = unmasker(sent)[0]
   
    pred_sent = top_pred['sequence']
    pred_score = top_pred['score']
    pred_token = top_pred['token_str']

    pred_sent = pred_sent.replace("[CLS] ", '').replace(" [SEP]", '')
    return (pred_sent, pred_token, pred_score)

def get_pos_label_for_MASK(pred_sent, pred_token):
    """
    Tags the predicted sentence output from unmasker, and returns the tag against [MASK].

    Arguments
        pred_sent (str): the sentence output from unmasker.
        pred_token (str): the token which was predicted in place of MASK.

    Returns
        tag (str): the tag of the predicted MASK.
    """
    tokenized_pred_sent = word_tokenize(pred_sent)
    tagged_pred_sent = nltk.pos_tag(tokenized_pred_sent)
    for tok, tag in tagged_pred_sent:
        if tok == pred_token:
            return tag

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

    Arguments
        sents (str): the list of sentences to MASK concepts in.
        base_labels (list): the list of base labels.
        concepts (list): the list of concepts.
    
    Returns
        sents_masked (dict): the dictionary of masked sentences against concepts.
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

def t_test(tcavs, r_tcavs):
    """
    Computes the t-test between two distributions.
    """
    _, p = stats.ttest_ind(tcavs, r_tcavs)
    return p

def apply_bonferroni_correction(num_of_runs, p):
    """
    Apply the Bonferroni correction and check whether to accept or not.

    Arguments
        num_of_runs (int): the number of runs.
        p (float): the computed p-value using t-test.
    Returns
        to_accept (bool): whether to accept the result or not.
    """
    default = 0.05 # the base value to reject the null-hypothesis.
    alpha = float(default)/float(num_of_runs)
    return p <= alpha

def compute_word_tcav(concept_cavs, random_cavs,
                      bottleneck_base,
                      sentences, num_layers, word,
                      num_of_runs, model_type, 
                      gold_label, if_rand):
    """
    Compute TCAV scores for a given CAV of a concept and every word in each sentence in the base corpus.

    Arguments
        concept_cavs (dict): the dictionary of CAVs of each concept for every layer.
        bottleneck_base (dict): the dictionary of base activations.
        sentences (list): the list of sentences.
        num_layers (int): the number of layers.
        model_type (str): the type of the model to load the unmasker for.

    Returns
        word_tcav (dict): the dictionary of word weightage for each layer for each word for each concept.
    """

    word_tcav = defaultdict(dict)
    word_tcav = {str(k): {} for k in range(1, num_layers+1)}
    unmasker = get_model_unmasker(model_type)

    for ix in range(1, num_layers+1):
        check_for_spurious_cavs = {}
        layer_cavs = concept_cavs[str(ix)]
        if if_rand:
            random_layer_cavs = random_cavs[str(ix)]

        for concept, cavs_of_runs in layer_cavs.items():
            print(f"Running for Concept - {concept}")
            tcavs_per_run = []

            #if if_rand:
            #    random_tcavs_per_run = []
            #    random_layer_cavs_per_concept = random_layer_cavs[concept]

            for run in range(num_of_runs):
                count = 0
                random_count = 0
                score_of_tag_preds = 0

                for cav_key, cav in cavs_of_runs[run].items():
                    if if_rand:
                        random_cav = random_layer_cavs_per_concept[run]["r-"+cav_key]
                    for jx, sent in enumerate(sentences):
                        words = list(sent.split(' '))
                        if word in words:
                        
                            pred_sent, pred_token, pred_score = get_top_prediction_from_unmasker(unmasker, sent)
                            print(f"[INFO] Model {model_type} predicted {pred_token} in place of [MASK] with conf. score of {round(pred_score, 3)}.")
                            tag_of_prediction = get_pos_label_for_MASK(pred_sent, pred_token)
                            act_per_layer_per_sent = bottleneck_base[str(ix)][jx]
                            
                            print(f"[INFO] The actual label was {gold_label} and the tag of prediction is {tag_of_prediction}.")
                            print(f"[INFO] This was tested on concept {concept}.")

                            if tag_of_prediction == gold_label:
                                score_of_tag_preds += 1
                                
                                selected_word_index = words.index(word)
                                selected_word_acts = act_per_layer_per_sent[selected_word_index]
                                dydx = directional_derivative(selected_word_acts, cav)
                                
                                if dydx: 
                                    count += 1

                                #if if_rand:
                                #    r_dydx = directional_derivative(selected_word_acts, random_cav)
                                #    if r_dydx: 
                                #        random_count += 1

                    tcav = float(count)/float(score_of_tag_preds)
                    tcavs_per_run.append(tcav)

                    #if if_rand:
                    #    r_tcav = float(random_count)/float(score_of_tag_preds)
                    #    random_tcavs_per_run.append(r_tcav)

                    accuracy = score_of_tag_preds/len(sentences)

                    print(f"For run {run}, the concept {gold_label} achieved an accuracy (tag matching with gt) of {accuracy}.")
                    
            
            print(f"[INFO] Layer - {ix} TCAVs - {tcavs_per_run}, Masked Concept - {gold_label} Tested Concept - {concept}.")
            #print(f"[INFO] Random TCAVs - {random_tcavs_per_run}.")
            
            # perform the t-test here and then proceed.
            #p = t_test(tcavs_per_run, random_tcavs_per_run)
            #print(f"[INFO] For Concept {concept}, the p-value is {p} at layer {ix}.")

            # apply the bonferroni correction and check if the p_value is still less.
            #to_accept = apply_bonferroni_correction(num_of_runs, p)
            
            #if to_accept: # the test is passed.
            check_for_spurious_cavs[concept] = np.mean(tcavs_per_run)

        word_tcav[str(ix)] = check_for_spurious_cavs

    return word_tcav

def run_for_chosen_word_write_to_pickle(sentences, concept_cavs, random_cavs,
                                        bottleneck_base, num_layers,
                                        word, output_directory, 
                                        num_of_runs, model_type, if_rand):
    """
    Computes the TCAV for each word on [MASKED] sentences for each concept.

    Arguments
        sentences (dict): dict of [MASKED] sentences against each concept.
        concept_cavs (dict): the dictionary of CAVs of each concept for every layer.
        bottleneck_base (dict): the dictionary of base activations.
        num_layers (int): the number of layers.
        word (str): the word to get TCAVs for.
        output_directory (str): the directory to store the pickle file in.
        num_of_runs (int): the number of times to run the experiment for.
        model_type (str): the model type to load the unmasker.
    """

    write_file_path = f"{output_directory}" + "/inference_word_mode.pickle"
    concept_masked_tcav_dict = defaultdict(dict)

    for concept_masked, sents in sentences.items(): # these are concept_wise masked sentences.
        #if concept_masked in ["NN", "JJ", "NNS", "JJR", "JJS", "DT", "CC", "CD", "VB", "VBP"]:
        print(f"[INFO] Masked Concept - {concept_masked}")
        word_layer_wise_tcavs = compute_word_tcav(concept_cavs, random_cavs, bottleneck_base, sents, num_layers, word, num_of_runs, model_type, concept_masked, if_rand)
        concept_masked_tcav_dict[concept_masked] = word_layer_wise_tcavs

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
    parser.add_argument("-rc", "--random_concepts_cavs",
        help="The random CAV pickle file computed.")
    parser.add_argument("-o", "--output_directory",
        help="The output directory to store the results in.")
    parser.add_argument("-w", "--word", default="[MASK]",
        help="The word to compute the TCAVs for in w mode.")
    parser.add_argument("-rs", "--runs",
        help="The number of runs used to get CAVs.")
    parser.add_argument("-m", "--model",
        help="The model type to load the unmasker for.")
    parser.add_argument("-ir", "--if_random", default="0",
        help="Whether to compute the random vs random or not.")

    args = parser.parse_args()
    num_neurons = 768

    base_acts_to_compute = args.base_acts_to_compute
    base_sentences = args.base_sentences
    base_labels = args.base_labels
    output_directory = args.output_directory
    num_of_runs = int(args.runs)
    word = args.word
    model_type = args.model
    if_rand = int(args.if_random)

    concept_cavs = load_pickle_files(args.concepts_cavs)


    if if_rand:
        random_cavs = load_pickle_files(args.random_concepts_cavs)
    else:
        random_cavs = {}

    sents = read_sentences(base_sentences)
    base_labels = read_sentences(base_labels)

    concepts_dict = get_concepts_dict(base_labels)
    concepts2class = assign_labels_to_concepts(concepts_dict)

    base_acts, base_num_layers = data_loader.load_activations(base_acts_to_compute, num_neurons)
    bottleneck_base = segregate_acts_for_each_layer(base_acts, base_num_layers)

    print("[INFO] Computing TCAVs.")
    masked_sents = mask_out_each_concept_in_base_sentences(sents, base_labels, list(concepts2class.values()))
    run_for_chosen_word_write_to_pickle(masked_sents, concept_cavs, random_cavs, bottleneck_base, base_num_layers, word, output_directory, num_of_runs, model_type, if_rand)

if __name__ == '__main__':
    main()