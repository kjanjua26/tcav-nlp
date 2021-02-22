"""
    Prepares the concepts layer wise to compute CAVs.
"""

import sys
sys.path.append("/Users/Janjua/Desktop/QCRI/Work/aux_classifier/")
import aux_classifier.extraction as extraction
import aux_classifier.data_loader as data_loader
import aux_classifier.utils as utils

import argparse, os
import cavs
import numpy as np
from time import perf_counter
import random
import pickle
from pathos.multiprocessing import ProcessingPool
from sklearn.utils import shuffle

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

def get_x_y_of_concept(X, Y):
    """
    Get's the chunk of X and Y who belong to the concept.

    Arguments
        X (list): the features
        y (list): the labels
    """
    X_concept, y_concept = [], []
    X_other, y_other = [], []

    for ix, x in enumerate(X):
        y = Y[ix]
        if y == 1:
            X_concept.append(x)
            y_concept.append(y)
        else:
            X_other.append(x)
            y_other.append(y)
    
    return (np.array(X_concept), np.array(y_concept), np.array(X_other), np.array(y_other))

def get_random_index(total_len, n):
    """
    Get the random indices.

    Arguments
        total_len (int): the total length of the other array.
        n (int): the max to randomly sample.
    """
    index = np.random.choice(total_len, n, replace=False)
    return index

def get_cav_key(run, concept, layerno):
    """
    Get the key to store the CAV against.

    Arguments
        run (int): the # of run.
        concept (str): the concept running.
        layerno (str): the # of layer.
    """
    return '-'.join([f"layer[{layerno}]", str(run), concept])

def parallelized_nrof_runs(X, y, cav_key, model, process_type):
    """
    Parallelize the run to utilize the multiprocessing pool.
    
    Arguments
        X (lst): the list of nd.arrays of X.
        y (lst): the list of nd.arrays of y.
        cav_key (str): the list of keys.
        model (str): the model type.
    """
    cavs_for_layers = {}
    cav, accuracy = cavs.run(X, y, model_type=model)
    print(f"[INFO] Test Accuracy - {accuracy} for CAV - {cav_key}")
    
    if process_type == "main":
        if accuracy > 0.8:
            cavs_for_layers[cav_key] = cav
    else:
        cavs_for_layers[cav_key] = cav

    return cavs_for_layers

def run_for_each_layer(layerno, bottleneck_concept_acts_per_layer,
                    bottleneck_tokens, concepts2class, model_type, num_workers, no_of_runs):
    """
    Perform the test.

    Arguments
        layerno (str): the layer no.
        bottleneck_concept_acts_per_layer (list): activations of a specific layer.
        bottleneck_tokens (dict): dictionary of source (words) and target (labels) of a layer.
        concepts2class (dict): the concepts.
        model_type (str): the type of the model.
        num_workers (int): the number of workers for parallelizing.
        no_of_runs (int): the number of runs.
    Returns
        cavs_for_layers (dict): the dictionary containing cav of each concept for a layer.
    """
    concept_cavs = {}

    for _, concept in concepts2class.items():
        print(f"[INFO] Concept - {concept}")
        toks = modify_labels_according_to_concept(bottleneck_tokens, bottleneck_concept_acts_per_layer, concept) # pass in the concept token vector
        X, y, _ = utils.create_tensors(toks, bottleneck_concept_acts_per_layer, concept)
        X_concept, y_concept, X_other, y_other = get_x_y_of_concept(X, y)

        X, y, cav_keys, model_types, ops_type = [], [], [], [], []  # maintain lists to parallelize the run.

        for run in range(no_of_runs):
            if (X_other.shape[0] > len(X_concept)):
                index = get_random_index(X_other.shape[0], len(X_concept))
                X_other_ = X_other[index]
                y_other_ = y_other[index]
             
                X_ = np.concatenate((X_concept, X_other_))
                y_ = np.concatenate((y_concept, y_other_))
            else:
                X_ = np.concatenate((X_concept, X_other))
                y_ = np.concatenate((y_concept, y_other))
            
            # shuffle the two lists.
            X_, y_ = shuffle(X_, y_)

            cav_key = get_cav_key(run, concept, layerno)
            X.append(X_)
            y.append(y_)
            cav_keys.append(cav_key)
            model_types.append(model_type)
            ops_type.append("main")
        
        pool = ProcessingPool(num_workers)
        cavs_for_layers = pool.map(parallelized_nrof_runs, X, y, cav_keys, model_types, ops_type)

        concept_cavs[concept] = cavs_for_layers

    return concept_cavs

def run_random_for_each_layer(layerno, bottleneck_concept_acts_per_layer,
                    bottleneck_tokens, concepts2class, model_type, num_workers, no_of_runs):
    """
    Perform the test.

    Arguments
        layerno (str): the layer no.
        bottleneck_concept_acts_per_layer (list): activations of a specific layer.
        bottleneck_tokens (dict): dictionary of source (words) and target (labels) of a layer.
        concepts2class (dict): the concepts.
        model_type (str): the type of the model.
        num_workers (int): the number of workers for parallelizing.
        no_of_runs (int): the number of runs.
    Returns
        random_concept_cavs (dict): the dictionary containing random CAV for each concept for a layer.
    """
    random_concept_cavs = {}

    for _, concept in concepts2class.items():
        toks = modify_labels_according_to_concept(bottleneck_tokens, bottleneck_concept_acts_per_layer, concept) # pass in the concept token vector
        X, y, _ = utils.create_tensors(toks, bottleneck_concept_acts_per_layer, concept)
        X_concept, y_concept, X_other, y_other = get_x_y_of_concept(X, y)

        X, y, cav_keys, model_types, ops_type = [], [], [], [], []  # maintain lists to parallelize the run.

        for run in range(no_of_runs):
            index = get_random_index(X_other.shape[0], len(X_other))
            random_index = get_random_index(X_other.shape[0], len(X_other))
            
            X_random = X_other[random_index]
            y_random = y_other[random_index]

            X_other_ = X_other[index]
            y_other_ = np.ones_like(y_random)

            X_ = np.concatenate((X_random, X_other_))
            y_ = np.concatenate((y_random, y_other_))
            
            # shuffle the two lists.
            X_, y_ = shuffle(X_, y_)

            cav_key = get_cav_key(run, concept, layerno)
            random_cav_key = "r-" + cav_key
            X.append(X_)
            y.append(y_)
            cav_keys.append(random_cav_key)
            model_types.append(model_type)
            ops_type.append("random")

        pool_r = ProcessingPool(num_workers)
        random_cavs_for_layers = pool_r.map(parallelized_nrof_runs, X, y, cav_keys, model_types, ops_type)

        random_concept_cavs[concept] = random_cavs_for_layers

    return random_concept_cavs

def run(concept_bottleneck, tokens_bottleneck, concepts2class, model_type, num_workers, no_of_runs):
    """
    Performs the run_for_each_layer() function for each layer.

    Arguments
        bottleneck_concept_acts_per_layer (list): activations of a specific layer.
        tokens_bottleneck (dict): dictionary of source (words) and target (labels) of a layer.
        concepts2class (dict): the concepts.
        model_type (str): the type of linear model to train to get CAVs.
    Returns
        layer_wise_cavs (dict of dict): cavs of each layer for each concept.
    """
    layer_wise_cavs = {}
    random_layer_wise_cavs = {}

    for layer, bottleneck_acts_per_layer in concept_bottleneck.items():

        print(f"[INFO] For layer - {layer}")
        token_bottleneck = tokens_bottleneck[layer]

        cavs_for_layer = run_for_each_layer(layer, bottleneck_acts_per_layer,
                                            token_bottleneck, concepts2class, model_type,
                                            num_workers, no_of_runs)
        layer_wise_cavs[layer] = cavs_for_layer
        
        print("[INFO] Running for Random.")
        random_cavs_for_layer = run_random_for_each_layer(layer, bottleneck_acts_per_layer,
                                            token_bottleneck, concepts2class, model_type,
                                            num_workers, no_of_runs)
        random_layer_wise_cavs[layer] =  random_cavs_for_layer
        
        print("="*50)
        print()

    return layer_wise_cavs, random_layer_wise_cavs

def write_the_cavs(output_path, layer_wise_cavs, random_layer_wise_cavs):
    """
    Write the computed CAVs (for both random and concept) in pickle files for further use.

    Arguments
        output_path (str): the output folder name.
        layer_wise_cavs (dict): the layerwise computed CAVs.
    """
    layer_wise_cavs_path = output_path + "/layer_wise_cavs.pickle"
    random_layer_wise_cavs_path = output_path + "/random_layer_wise_cavs.pickle"

    with open(layer_wise_cavs_path, "wb") as writer:
        pickle.dump(layer_wise_cavs, writer)

    with open(random_layer_wise_cavs_path, "wb") as writer:
        pickle.dump(random_layer_wise_cavs, writer)


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
    parser.add_argument("-lm", "--linear_model_type", default="SGDC",
        help="The type of linear model to train.")
    parser.add_argument("-w", "--workers",
        help="The number of workers to parallelize.")
    parser.add_argument("-rs", "--runs",
        help="The number of runs.")
    
    args = parser.parse_args()
    num_neurons = 768
    max_sentence_l = 512
    concept_type = ["pos", "gender"] # currently only dealing with two types of concepts entirely - pos, gender (m, f).

    concept_sents = args.input_corpus
    concept_labels = args.label_corpus
    concepts = args.concepts
    output_folder = args.output_folder
    model_type = args.linear_model_type
    num_workers = int(args.workers)
    no_of_runs = int(args.runs)

    start = perf_counter()
    concepts_labels_data = read_file(concept_labels)
    concept_acts, num_layers = load_activations(args.embeddings, num_neurons)
    
    concepts_dict = get_concepts_dict(concepts_labels_data)
    concepts2class = assign_labels_to_concepts(concepts_dict)

    if concepts not in concept_type:
        concepts2class = {i[0]: i[1] for i in enumerate(concepts.split(' '))}

    print("[INFO] Concepts: ", list(concepts2class.values()))

    concept_bottleneck, tokens_bottleneck = get_bottlenecks(concept_acts,
                                                            concept_sents,
                                                            concept_labels,
                                                            num_layers, max_sentence_l)
    
    layer_wise_cavs, random_layer_wise_cavs = run(concept_bottleneck, tokens_bottleneck, concepts2class, model_type, num_workers, no_of_runs)
    write_the_cavs(output_folder, layer_wise_cavs, random_layer_wise_cavs)
    end = perf_counter()

    print(f"Completed the process in {end-start}s")

if __name__ == '__main__':
    main()