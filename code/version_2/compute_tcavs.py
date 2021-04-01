"""
    Compute the TCAVs for the given CAVs against each concept for each sentence in the base_data.txt.
    Use base_data_labels.txt for word analysis.
    Computes both ways: sentence TCAV and word TCAV.
"""
import sys
sys.path.append("/alt/mt/tcav/NeuroX/")
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
import extraction
import json
import torch
import logging
logging.disable(logging.INFO)

def write_pickle_file(layer_tcav, output_path, name):
    """
    Write the pickle file.

    Arguments
        output_path (str): the path of output directory.
        name (str): the name of the file.
    """
    write_file_path = f"{output_path}/{name}"
    with open(write_file_path, "wb") as writer:
        pickle.dump(layer_tcav, writer)


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

def mask_out_each_concept_in_base_sentences(sents, base_labels, concepts, word):
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
                sent = sent.replace(word_in_sent, word, 1)
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

def get_gradients(model, tokenizer, original_sentence, 
                  masked_sentence, layer):
    """
    Get the gradient using the activations.

    Arguments
        model_type (str): the name of the model.
        input_sent (str): the input sentence.
        masked_sent (str): the masked sentence.
        layer (int): the layer to get the grad of.

    Returns
        grads (torch.tensor): the gradients of layer (at the masked token position) w.r.t to the current prediction
    """
    print("[INFO] Sentence: ", original_sentence)
    print("[INFO] Masked Sentence: ", masked_sentence)

    inputs = tokenizer(masked_sentence, return_tensors="pt")
    label = tokenizer(original_sentence, return_tensors="pt")["input_ids"]

    masked_idx = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1][0].item()

    model_output = model(**inputs, labels=label)

    gradients = torch.autograd.grad(model_output.loss, model_output.hidden_states[int(layer)-1])[0][0, masked_idx, :]
    return gradients

def get_token_loss_gradients(model, tokenizer, original_sentence, 
                            masked_sentence, layer):

    """
    Get the gradient using the activations.

    Arguments
        model_type (str): the name of the model.
        input_sent (str): the input sentence.
        masked_sent (str): the masked sentence.
        layer (int): the layer to get the grad of.

    Returns
        grads (torch.tensor): the gradients of layer (at the masked token position) w.r.t token (MASK) loss.
    """
    print("[INFO] Sentence: ", original_sentence)
    print("[INFO] Masked Sentence: ", masked_sentence)              

    inputs = tokenizer(masked_sentence, return_tensors="pt")
    masked_idx = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1][0].item()
    model_output = model(**inputs)
    predicted_labels = torch.argmax(model_output.logits, axis=-1)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    predicted_token_loss = loss_fct(model_output.logits.view(-1, tokenizer.vocab_size), predicted_labels.view(-1))[masked_idx]
    gradients = torch.autograd.grad(predicted_token_loss, model_output.hidden_states[int(layer)-1])[0][0, masked_idx, :]
    return gradients

def compute_word_tcav(concept_cavs, random_cavs,
                      bottleneck_base,
                      sentences, num_layers, 
                      word, num_of_runs, 
                      model_type, gold_label, 
                      if_rand, output_directory, use_grad):
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
    model, tokenizer, _ = extraction.get_model_and_tokenizer(model_type, mtype="grad")
    # normal => list(word_tcav.keys())

    # test for only last layer.
    for lx in list(["13"]):
        check_for_spurious_cavs = {}
        layer_cavs = concept_cavs[str(lx)]
        if if_rand:
            random_layer_cavs = random_cavs[str(lx)]

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
                            #print(f"[INFO] Model {model_type} predicted {pred_token} in place of [MASK] with conf. score of {round(pred_score, 3)}.")
                            
                            tag_of_prediction = get_pos_label_for_MASK(pred_sent, pred_token)
                            predicted_sentence = sent.replace(word, pred_token)

                            act_per_layer_per_sent = bottleneck_base[str(lx)][jx]
                            
                            print(f"[INFO] The actual label was {gold_label} and the tag of prediction is {tag_of_prediction}.")
                            print(f"[INFO] This was tested on concept {concept}.")

                            if tag_of_prediction == gold_label:
                                score_of_tag_preds += 1
                                
                                selected_word_index = words.index(word)
                                selected_word_acts = act_per_layer_per_sent[selected_word_index]

                                if use_grad:
                                    # use gradients computed w.r.t to the loss.
                                    grads = get_token_loss_gradients(model, tokenizer, predicted_sentence, sent, lx)
                                    dydx = directional_derivative(grads, cav)
                                else:
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

                    #accuracy = score_of_tag_preds/len(sentences)
                    print(f"[INFO] Concept {gold_label}\n[INFO] Total # of sentences {len(sentences)}\n[INFO] Matched Tag # of Sentences {score_of_tag_preds}")
                    print("="*100)
                    #print(f"For run {run}, the concept {gold_label} achieved an accuracy (tag matching with gt) of {accuracy}.")
                    
            print(f"[INFO] Layer - {lx} TCAVs - {tcavs_per_run}, Masked Concept - {gold_label} Tested Concept - {concept}.")
            #print(f"[INFO] Random TCAVs - {random_tcavs_per_run}.")
            
            # perform the t-test here and then proceed.
            #p = t_test(tcavs_per_run, random_tcavs_per_run)
            #print(f"[INFO] For Concept {concept}, the p-value is {p} at layer {lx}.")

            # apply the bonferroni correction and check if the p_value is still less.
            #to_accept = apply_bonferroni_correction(num_of_runs, p)
            
            #if to_accept: # the test is passed.
            check_for_spurious_cavs[concept] = np.mean(tcavs_per_run)

        word_tcav[str(lx)] = check_for_spurious_cavs
    
        # write the layer_wise results to pickle file.
        write_pickle_file(check_for_spurious_cavs, output_directory, f"layer-{str(lx)}-{gold_label}-results.pickle")

    return word_tcav

def load_the_acts(concept_masked_representations, num_neurons):
    activations = []

    out = concept_masked_representations.split('\n')
    for line in out:
        if line != "":
            token_acts = []
            sentence_activations = json.loads(line)['features']
            for act in sentence_activations:
                token_acts.append(np.concatenate([l['values'] for l in act['layers']]))
            activations.append(np.vstack(token_acts))

            num_layers = activations[0].shape[1] / num_neurons
        
    return activations, int(num_layers)

def modify_extractions_with_added_masked_word(model_name, masked_sents,
                                            num_neurons, concept_cavs,
                                            random_cavs, word,
                                            num_of_runs, if_rand,
                                            output_directory, use_grad):

    masked_reps_with_sents = {}
    concept_masked_tcav_dict = defaultdict(dict)
    write_file_path = f"{output_directory}" + "/inference_word_mode_masked_test.pickle"
    print("[INFO] Masked Activations Test.")

    for concept_masked, sents in masked_sents.items():
        if concept_masked in ["NN", "JJ", "NNS", "JJR", "JJS", "DT", "CC", "CD", "VB", "VBP"]:
            print("[INFO] Concept Masked - ", concept_masked)

            concept_masked_representations = extraction.extract_representations_from_sents(model_name, sents)
            base_acts, base_num_layers = load_the_acts(concept_masked_representations, num_neurons)
            bottleneck_base = segregate_acts_for_each_layer(base_acts, base_num_layers)

            word_layer_wise_tcavs = compute_word_tcav(concept_cavs, random_cavs, 
                                                    bottleneck_base, sents, 
                                                    base_num_layers, word, num_of_runs, 
                                                    model_name, concept_masked, 
                                                    if_rand, output_directory, use_grad)
        
            concept_masked_tcav_dict[concept_masked] = word_layer_wise_tcavs

    with open(write_file_path, "wb") as writer:
        pickle.dump(concept_masked_tcav_dict, writer)

def run_for_chosen_word_write_to_pickle(sentences, concept_cavs, random_cavs,
                                        bottleneck_base, num_layers,
                                        word, output_directory, 
                                        num_of_runs, model_type, 
                                        if_rand, use_grad):
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

    write_file_path = f"{output_directory}" + "/inference_word_mode_unmasked_test.pickle"
    concept_masked_tcav_dict = defaultdict(dict)

    for concept_masked, sents in sentences.items(): # these are concept_wise masked sentences.
        if concept_masked in ["NN", "JJ", "NNS", "JJR", "JJS", "DT", "CC", "CD", "VB", "VBP"]:
            print(f"[INFO] Masked Concept - {concept_masked}")

            word_layer_wise_tcavs = compute_word_tcav(concept_cavs, random_cavs, 
                                                    bottleneck_base, sents, 
                                                    num_layers, word, num_of_runs, 
                                                    model_type, concept_masked,
                                                    if_rand, output_directory, use_grad)

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
    parser.add_argument("-w", "--word",
        help="The word to compute the TCAVs for in w mode.")
    parser.add_argument("-rs", "--runs",
        help="The number of runs used to get CAVs.")
    parser.add_argument("-m", "--model",
        help="The model type to load the unmasker for.")
    parser.add_argument("-ir", "--if_random", default="0",
        help="Whether to compute the random vs random or not.")
    parser.add_argument("-pm", "--process_mode", default="1",
        help="The processing mode: MASK or no-MASK base sentences testing.")
    parser.add_argument("-g", "--use_grad", default="1",
        help="Use gradients to compute CAV.")

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
    process_mode = int(args.process_mode)
    use_grad = int(args.use_grad)
    
    concept_cavs = load_pickle_files(args.concepts_cavs)


    if if_rand:
        random_cavs = load_pickle_files(args.random_concepts_cavs)
    else:
        random_cavs = {}

    sents = read_sentences(base_sentences)
    base_labels = read_sentences(base_labels)

    concepts_dict = get_concepts_dict(base_labels)
    concepts2class = assign_labels_to_concepts(concepts_dict)

    print("[INFO] Computing TCAVs.")
    masked_sents = mask_out_each_concept_in_base_sentences(sents, base_labels, list(concepts2class.values()), word)

    if process_mode: # do the MASKED word acts testing.
        modify_extractions_with_added_masked_word(model_type, masked_sents, 
                                            num_neurons, concept_cavs, random_cavs,
                                            word, num_of_runs, 
                                            if_rand, output_directory, use_grad)

    else: # do the unmasked word testing.
        base_acts, base_num_layers = data_loader.load_activations(base_acts_to_compute, num_neurons)
        bottleneck_base = segregate_acts_for_each_layer(base_acts, base_num_layers)

        run_for_chosen_word_write_to_pickle(masked_sents, concept_cavs, 
                                            random_cavs, bottleneck_base, 
                                            base_num_layers, word, 
                                            output_directory, num_of_runs, 
                                            model_type, if_rand, use_grad)

if __name__ == '__main__':
    main()