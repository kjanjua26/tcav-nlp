"""
    Prepares the concepts layer wise to compute CAVs.

    python3 prepare_concepts.py -i ../../../data/subsample.in -l ../../../data/subsample_label.in -e ../../../data/subsample_concept_acts.json -c "1" -o .
    python3 prepare_concepts.py -i ../../../data/dconcept.in -l ../../../data/dconcept_labels.in -e ../../../data/dconcept_acts.json -c "NN" -o .
    python prepare_concepts.py -i ../../../data/subsample.in -l ../../../data/subsample_label.in -e ../../../data/subsample_concept_acts.json -c "gender" -o ../../../data/results

"""

import argparse, sys, os
import cavs
import numpy as np
from time import perf_counter 
import random
import pickle

sys.path.append("/Users/Janjua/Desktop/QCRI/Work/aux_classifier/")
import aux_classifier.extraction as extraction
import aux_classifier.data_loader as data_loader
import aux_classifier.utils as utils

def read_file(fp):
    """
    Reads the file and store it in a list.

    Arguments
        fp (the file path): the file path to read.
    Returns
        list_of_lines (list): the list containing the lines (read from the file).
    """

    list_of_lines = []
    
    data = open(fp, 'r')
    lines = data.readlines()
    for line in lines:
        line = line.strip()
        list_of_lines.append(line)
    
    return list_of_lines

def get_concepts_dict(lst_labels):
    """
    Get the concepts. If gender type labels are passed, two concepts: male, female similarly gets for all parts of speech, etc.
    
    Arguments
        labels (list): the list of labels containing tags.
    Returns
        concept_dict (dict): the concept dictionary with keys one for each unique label.
    """

    concepts = {}
    for labels in lst_labels:
        labels = labels.split(' ') # since each line is separated, corresponding to the text (sentence).
        for label in labels:
            if label not in concepts.keys():
                concepts[label] = {}
    
    return concepts

def load_activations(activations, num_neurons):
    """
    Given the .json file of activations, it loads them in memory for processing.

    Arguements
        activations (json): the activations json file.
        num_neurons (int): the number of neurons.
    
    Returns
        acts (list): the list of arrays containing the activations.
        num_layers (int): the number of layers in the network.
    """
    acts, num_layers = data_loader.load_activations(activations, num_neurons)
    return acts, num_layers

def segregate_acts_for_each_layer(acts, num_layers):
    """
    Separate acts for each layer - 1 to 13.

    Arguements
        acts (list): contains the activations.
        num_layers (int): the total number of layers.
    Returns
        bottlenecks (dict): contains layer-wise acts.
    """
    bottleneck = {str(k+1): [] for k in range(num_layers)}
    np_acts = np.array(acts)
    nrof_samples = np_acts.shape[0]

    for ix, sample in enumerate(acts):
        sample = np.reshape(sample, (-1, 13, 768)) # reshape to each layer.
        sample = sample.transpose(1, 0, 2) # it is (-1, 13, 768), we want it to be (13, -1, 768)
        for layer_no, act in enumerate(sample):
            bottleneck[str(layer_no+1)].append(act)
    return bottleneck

def segregate_tokens_for_each_layer(activations, concept_sents, concept_labels, max_sentence_l):
    """
    Segregates the tokens for each layer.

    Arguements
        activations (dict): contains the activations for each layer.
        concept_sents (list): path where the sentences are stored.
        concept_labels (list): the labels of the sentences.
        max_sentence_l (int): the maximum length of a sentence.

    Return
        bottleneck_tokens (dict): the dictionary of tokens, separated by layer.
    """
    
    bottleneck_tokens = {}
    for k, v in activations.items():
        tokens = data_loader.load_data(concept_sents,
                                       concept_labels,
                                       v, max_sentence_l)
        bottleneck_tokens[k] = tokens

    return bottleneck_tokens

def modify_labels_according_to_concept(tokens, activations, idx):
    """
    Modify the label according to the concept.
    For any concept, this modifies the entire label vector accordingly.
    
    Arguments
        idx (str): the label for the concept required.
        tokens (dict): source (words), targets (labels).
        activations (list): the activations extracted.
    
    Returns
        tokens_up (dict): source (words), targets (labels).
    """
    source = tokens["source"]
    targets = tokens["target"]
    
    tokens_up = {"source": [], "target": []}
    
    assert (len(source) == len(targets) == len(activations))
    
    # 1 is the label you want to be in the labels against the idx.
    # mask all others to '0'
    for ix, sentence in enumerate(source):
        labels = targets[ix]
        labels = ['1' if x == idx else '0' for x in labels]
        tokens_up["source"].append(sentence)
        tokens_up["target"].append(labels)
                    
    return tokens_up

def assign_labels_to_concepts(concepts):
    """
    Given a concept dict, it maps the keys in the dict to str(int) values.

    Arguments
        concepts (dict): the dictionary of concepts.
    
    Returns
        concepts2class (dict): the dictionary of labels assigned to each concept.
    """
    concepts2class = {}
    for ix, key in enumerate(concepts.keys()):
        concepts2class[str(ix)] = key

    return concepts2class

def choose_a_concept_other_than_current(current, concepts):
    """
    Choose a concept from the list other than the one CAV is computed for.

    Arguments
        current (the current chosen concept).
        concepts (list): the list of all the conecpts.
    
    Returns
        chosen (the chosen concept other than current).
    """
    while True:
        chosen = random.choice(concepts)
        if current != chosen:
            return chosen

def get_bottlenecks(concept_acts, concept_sents, concept_labels, num_layers, max_sentence_l):
    """
    Converts the extracted embeddings, tokens to layer wise for easy layer-by-layer analysis.

    Arguments
        concept_acts (list): the list of activations of the concepts.
        concept_sents (list): the list of sentences of the concepts.
        concept_labels (list): the list of labels of the concepts corresponding to concept_sents].
        num_layers (int): the number of layers.
        max_sentence_l (int): the maximum sentence length.

    Returns
        concept_bottleneck (dict): the dictionary of activations segregated layer wise.
        concept_tokens (dict): the dictionary of tokens segregated layer wise.
    """
    concept_bottleneck = segregate_acts_for_each_layer(concept_acts, num_layers) # layer wise concepts.
    tokens_bottleneck = segregate_tokens_for_each_layer(concept_bottleneck, concept_sents, concept_labels, max_sentence_l) # layer wise tokens

    return (concept_bottleneck, tokens_bottleneck)

def run_for_each_layer(bottleneck_concept_acts_per_layer, bottleneck_tokens, concepts2class):
    """
    Perform the test.

    Arguments
        bottleneck_concept_acts_per_layer (list): activations of a specific layer.
        bottleneck_tokens (dict): dictionary of source (words) and target (labels) of a layer.
        concepts2class (dict): the concepts.
    
    Returns
        cavs_for_layers (dict): the dictionary containing cav of each concept for a layer.
    """
    cavs_for_layers = {}
    random_cavs_for_layers = {}

    labels = list(concepts2class.values())
    
    for idx, concept in concepts2class.items():
        print(f"Concept - {concept}")
        random_concept = choose_a_concept_other_than_current(concept, labels)
        
        # train CAV for the actual data.
        toks = modify_labels_according_to_concept(bottleneck_tokens, bottleneck_concept_acts_per_layer, concept) # pass in the concept token vector
        X, y, _ = utils.create_tensors(toks, bottleneck_concept_acts_per_layer, concept)
        cav, accuracy = cavs.run(X, y, model_type="LR")
        print(f"The trained model achieved an accuracy of - {accuracy}")
        if accuracy > 0.8: # only pick the CAV with accuracy higher than 80%.
            cavs_for_layers[concept] = cav

        print(f"For Random Concept - {random_concept}")
        # train CAV for the random data.
        random_toks = modify_labels_according_to_concept(bottleneck_tokens, bottleneck_concept_acts_per_layer, random_concept) # pass in the random concept token vector
        random_X, random_y, _ = utils.create_tensors(random_toks, bottleneck_concept_acts_per_layer, random_concept)
        random_cav, random_accuracy = cavs.run(random_X, random_y, model_type="LR")
        random_cavs_for_layers[random_concept] = random_cav
        
    return cavs_for_layers, random_cavs_for_layers

def run(concept_bottleneck, tokens_bottleneck, concepts2class):
    """
    Performs the run_for_each_layer() function for each layer.

    Arguments
        bottleneck_concept_acts_per_layer (list): activations of a specific layer.
        tokens_bottleneck (dict): dictionary of source (words) and target (labels) of a layer.
        concepts2class (dict): the concepts.
    
    Returns
        layer_wise_cavs (dict of dict): cavs of each layer for each concept.
    """
    layer_wise_cavs = {}
    layer_wise_random_cavs = {}

    for layer, bottleneck_acts_per_layer in concept_bottleneck.items():
        print(f"For layer - {layer}")
        token_bottleneck = tokens_bottleneck[layer]
        cavs_for_layer, random_cavs_for_layer = run_for_each_layer(bottleneck_acts_per_layer, token_bottleneck, concepts2class)
        layer_wise_cavs[layer] = cavs_for_layer
        layer_wise_random_cavs[layer] = random_cavs_for_layer
        print("="*50)
        print()

    return layer_wise_cavs, layer_wise_random_cavs

def write_the_cavs(output_path, layer_wise_cavs, layer_wise_random_cavs):
    """
    Write the computed CAVs (for both random and concept) in pickle files for further use.

    Arguments
        output_path (str): the output folder name.
        layer_wise_cavs (dict): the layerwise computed CAVs.
        layer_wise_random_cavs (dict): the layerwise computed random CAVs.
    """
    layer_wise_cavs_path = output_path + "/layer_wise_cavs.pickle"
    layer_wise_random_cavs_path = output_path + "/layer_wise_random_cavs.pickle"

    with open(layer_wise_cavs_path, "wb") as writer:
        pickle.dump(layer_wise_cavs, writer)
    
    with open(layer_wise_random_cavs_path, "wb") as writer:
        pickle.dump(layer_wise_random_cavs, writer)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_corpus", 
        help="File path to prepare concepts from.")
    parser.add_argument("-l", "--label_corpus", 
        help="File path to labels for concepts.")
    parser.add_argument("-e", "--embeddings",
        help="The path to the embeddings required.")
    parser.add_argument("-c", "--concepts",
        help="Perform test on individual concept or all the concepts in the dataset.\nPass in the concept name (pos, gender) if you want to do it for all.")
    parser.add_argument("-o", "--output_folder",
        help="The output folder to store the concepts data in.")
    
    args = parser.parse_args()
    num_neurons = 768
    max_sentence_l = 512
    concept_type = ["pos", "gender"] # currently only dealing with two types of concepts entirely - pos, gender (m, f).

    concept_sents = args.input_corpus
    concept_labels = args.label_corpus
    concepts = args.concepts
    output_folder = args.output_folder

    start = perf_counter()
    concepts_labels_data = read_file(concept_labels)
    concept_acts, num_layers = load_activations(args.embeddings, num_neurons)
    
    concepts_dict = get_concepts_dict(concepts_labels_data)
    concepts2class = assign_labels_to_concepts(concepts_dict)

    if concepts not in concept_type:
        concepts2class = dict(filter(lambda i:i[0] in concepts.split(' '), concepts2class.items()))

    concept_bottleneck, tokens_bottleneck = get_bottlenecks(concept_acts, 
                                                            concept_sents, 
                                                            concept_labels, 
                                                            num_layers, max_sentence_l)
    
    layer_wise_cavs, layer_wise_random_cavs = run(concept_bottleneck, tokens_bottleneck, concepts2class)
    write_the_cavs(output_folder, layer_wise_cavs, layer_wise_random_cavs)
    end = perf_counter()

    print(f"Completed the process in {end-start}s")

if __name__ == '__main__':
    main()