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
from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sns
import logging

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
    
    # '1' is the label you want to be in the labels against the idx.
    # mask all others to '0'
    for ix, word in enumerate(source):
        labels = targets[ix]
        labels = ["1" if x == idx else "0" for x in labels]
        tokens_up["source"].append(word)
        tokens_up["target"].append(labels)

    return tokens_up

def modify_labels_for_controlled_experiments(tokens, activations, idx):
    """
    Modify the labels to perform the controlled experiments.
    Reference paper: https://www.aclweb.org/anthology/D19-1275.pdf

    Arguments
        idx (str): the label for the concept required.
        tokens (dict): source (words), targets (labels).
        activations (list): the activations extracted.
    
    Returns
        tokens_up (dict): source (words), targets (labels) modified according to the controlled experiment.
    """
    source = tokens["source"]
    targets = tokens["target"]
    
    tokens_up = {"source": [], "target": []}
    
    assert (len(source) == len(targets) == len(activations))

    # maintain a map of the labels assigned to words
    word2lbl = {}

    for ix, tokenzied_sentence in enumerate(source):
        labels = targets[ix]

        modified_labels = []
        
        for word in tokenzied_sentence:
            if word in word2lbl.keys():
                modified_labels.append(word2lbl[word])
            else:
                lbl = random.choice(["0", "1"])
                word2lbl[word] = lbl
                modified_labels.append(lbl)
        
        tokens_up["source"].append(tokenzied_sentence)
        tokens_up["target"].append(modified_labels)

    # get the count of words assigned 0s and 1s
    logging.info(f"[INFO] 0 Labels are {len([x for x in tokens_up['target'] for y in x if y == '0'])}.")
    logging.info(f"[INFO] 1 Labels are {len([x for x in tokens_up['target'] for y in x if y == '1'])}.")
    print(f"[INFO] 0 Labels are {len([x for x in tokens_up['target'] for y in x if y == '0'])}.")
    print(f"[INFO] 1 Labels are {len([x for x in tokens_up['target'] for y in x if y == '1'])}.")
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

    c = 0
    for ix, x in enumerate(X):
        y = Y[ix]
        if y == 1:
            c += 1
            X_concept.append(x)
            y_concept.append(y)
        else:
            X_other.append(x)
            y_other.append(y)

    print("Len of Samples: ", c)
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
    accuracies = {}
    to_return = []

    cav, accuracy = cavs.run(X, y, model_type=model)
    logging.info(f"[INFO] Test Accuracy - {accuracy} for CAV - {cav_key}")
    print(f"[INFO] Test Accuracy - {accuracy} for CAV - {cav_key}")
    
    if process_type == "main":
        if accuracy > 0.8:
            cavs_for_layers[cav_key] = cav
            accuracies[cav_key] = accuracy
    else:
        cavs_for_layers[cav_key] = cav
        accuracies[cav_key] = accuracy

    to_return.append(cavs_for_layers)
    to_return.append(accuracies)

    return to_return

def run_for_each_layer(layerno, bottleneck_concept_acts_per_layer,
                        bottleneck_tokens, concepts2class, 
                        model_type, num_workers, 
                        no_of_runs):
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
    concept_accuracies = {}

    for _, concept in concepts2class.items():
        concept_cavs[concept] = []

        X, y, cav_keys, model_types, ops_type = [], [], [], [], []
        for run in range(no_of_runs):
            toks = modify_labels_according_to_concept(bottleneck_tokens, bottleneck_concept_acts_per_layer, concept) # pass in the concept token vector

            X_, y_, mappings = utils.create_tensors(toks, bottleneck_concept_acts_per_layer, concept)
            label2idx, idx2label, src2idx, idx2src = mappings

            if label2idx["1"] != 1:
                # the mapping was reverse, logical_not(y).
                y_ = np.logical_not(y_).astype(int)
                
            print(f"[INFO] Concept - {concept}, Len - {len([x for x in y_ if x == 1])}")
            print(f"[INFO] Other Len - {len([x for x in y_ if x == 0])}")
            
            X_, y_ = utils.balance_binary_class_data(X_, y_)
            
            logging.info(f"Concept - {concept}, Len - {len([x for x in y_ if x == 1])}")
            logging.info(f"Other Len - {len([x for x in y_ if x == 0])}")

            print(f"Concept - {concept}, Len - {len([x for x in y_ if x == 1])}")
            print(f"Other Len - {len([x for x in y_ if x == 0])}")

            cav_key = get_cav_key(run, concept, layerno)
            cav_keys.append(cav_key)
            model_types.append(model_type)
            ops_type.append("main")
            X.append(X_)
            y.append(y_)

        pool = ProcessingPool(num_workers)
        to_return = pool.map(parallelized_nrof_runs, X, y, cav_keys, model_types, ops_type)

        for run in range(no_of_runs):
            concept_cavs[concept].append(to_return[run][0])

        concept_accuracies[concept] = to_return[0][1]

    return concept_cavs, concept_accuracies

def run_controlled_experiment_for_each_layer(layerno, bottleneck_concept_acts_per_layer,
                                            bottleneck_tokens, concepts2class, 
                                            model_type, num_workers, 
                                            no_of_runs):

    concept_cavs = {}
    concept_accuracies = {}

    for _, concept in concepts2class.items():
        X, y, cav_keys, model_types, ops_type = [], [], [], [], []
        concept_cavs[concept] = []

        for run in range(no_of_runs):
            controlled_toks = modify_labels_for_controlled_experiments(bottleneck_tokens, bottleneck_concept_acts_per_layer, concept)
            X_controlled, y_controlled, controlled_mappings = utils.create_tensors(controlled_toks, bottleneck_concept_acts_per_layer, concept)
            c_label2idx, _, _, _ = controlled_mappings
            
            if c_label2idx["1"] != 1:
                # the mapping was reverse, logical_not(y).
                y_controlled = np.logical_not(y_controlled).astype(int)
            
            X_, y_ = utils.balance_binary_class_data(X_controlled, y_controlled)

            print(f"Concept - {concept}, Len - {len([x for x in y_ if x == 1])}")
            print(f"Other Len - {len([x for x in y_ if x == 0])}")

            logging.info(f"Concept - {concept}, Len - {len([x for x in y_ if x == 1])}")
            logging.info(f"Other Len - {len([x for x in y_ if x == 0])}")
            
            cav_key = get_cav_key(run, concept, layerno)
            cav_keys.append(cav_key)
            model_types.append(model_type)
            ops_type.append("controlled")
            X.append(X_)
            y.append(y_)

        pool = ProcessingPool(num_workers)
        to_return = pool.map(parallelized_nrof_runs, X, y, cav_keys, model_types, ops_type)

        for run in range(no_of_runs):
            concept_cavs[concept].append(to_return[run][0])

        concept_accuracies[concept] = to_return[0][1]

    return concept_cavs, concept_accuracies

def run(concept_bottleneck, tokens_bottleneck, 
        concepts2class, model_type, 
        num_workers, no_of_runs, 
        if_controlled, layers):
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
    controlled_cavs_per_layer = {}

    layer_accuracies = {}
    controlled_layer_accuracies = {}

    for layer, bottleneck_acts_per_layer in concept_bottleneck.items():
        if layer in layers:
            logging.info(f"[INFO] For layer - {layer}")
            print(f"[INFO] For layer - {layer}")
            token_bottleneck = tokens_bottleneck[layer]

            cavs_for_layer, concept_accuracies = run_for_each_layer(layer, bottleneck_acts_per_layer,
                                                token_bottleneck, concepts2class, model_type,
                                                num_workers, no_of_runs)
            layer_wise_cavs[layer] = cavs_for_layer
            layer_accuracies[layer] = concept_accuracies

            if if_controlled:
                logging.info(f"[INFO] Training for Controlled Experiment!")
                print(f"[INFO] Training for Controlled Experiment!")
                controlled_cavs_for_layer, cont_concept_accuracies = run_controlled_experiment_for_each_layer(layer, bottleneck_acts_per_layer,
                                                        token_bottleneck, concepts2class, model_type,
                                                        num_workers, no_of_runs)
                controlled_cavs_per_layer[layer] = controlled_cavs_for_layer
                controlled_layer_accuracies[layer] = cont_concept_accuracies

            print("="*50)
            print()

    plot_specificity(layer_accuracies, controlled_layer_accuracies, concepts2class)

    return layer_wise_cavs, controlled_cavs_per_layer

def plot_specificity(accuracy, controlled_accuracy, concepts2class):
    """
    Plot the specificity of the experiments (controlled and otherwise) for each layer.
    """
    combined_accuracies = {"normal": accuracy, "controlled": controlled_accuracy}
    total_layers = set()
    concepts = list(concepts2class.values())
    concept_wise_results = {"normal": {x: [] for x in concepts}, "controlled": {x: [] for x in concepts}}

    for exp_type, acc_arr in combined_accuracies.items():
        for layer, concept_dict in acc_arr.items():
            for concept, score in concept_dict.items():
                concept_wise_results[exp_type][concept].append(float(mean(list(score.values()))))
                total_layers.add(layer)

    total_layers_sorted = list(total_layers)
    total_layers_sorted.sort(key=int)
    # latex typesetting for plots.
    sns.set(rc={'text.usetex': True})
    for cpt in concepts:
        normal_accuracies = concept_wise_results["normal"][cpt]
        controlled_accuracies = concept_wise_results["controlled"][cpt]
        plt.plot(total_layers_sorted, normal_accuracies, label="Normal", marker = 'o')
        plt.plot(total_layers_sorted, controlled_accuracies, label="Controlled", marker = '^')
        plt.legend()
        plt.title(cpt)
        plt.xlabel("Layer")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        plt.savefig(f"/Users/Janjua/Desktop/specificity-{cpt}.jpg")
        plt.show()


def write_the_cavs(output_path, layer_wise_cavs, 
                   controlled_layer_wise_cavs, if_controlled):
    """
    Write the computed CAVs (for both random and concept) in pickle files for further use.

    Arguments
        output_path (str): the output folder name.
        layer_wise_cavs (dict): the layerwise computed CAVs.
    """
    layer_wise_cavs_path = output_path + "/layer_wise_cavs.pickle"
    controlled_layer_wise_cavs_path = output_path + "/controlled_layer_wise_cavs.pickle"

    with open(layer_wise_cavs_path, "wb") as writer:
        pickle.dump(layer_wise_cavs, writer)

    if if_controlled:
        with open(controlled_layer_wise_cavs_path, "wb") as writer:
            pickle.dump(controlled_layer_wise_cavs, writer)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--name",
        help="The name of the experiment.")
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
    parser.add_argument("-lm", "--linear_model_type", default="LR",
        help="The type of linear model to train.")
    parser.add_argument("-w", "--workers",
        help="The number of workers to parallelize.")
    parser.add_argument("-rs", "--runs",
        help="The number of runs.")
    parser.add_argument("-ce", "--if_controlled", default="0",
        help="Conduct controlled experiments or not.")
    parser.add_argument("-tl", "--total_layers",
        help="Total layers to run the experiment for.")
    
    args = parser.parse_args()
    num_neurons = 768
    max_sentence_l = 512
    
    name = args.name
    concept_sents = args.input_corpus
    concept_labels = args.label_corpus
    concepts = args.concepts
    layers = args.total_layers
    output_folder = args.output_folder
    model_type = args.linear_model_type
    num_workers = int(args.workers)
    no_of_runs = int(args.runs)
    if_controlled = int(args.if_controlled)
    
    # add the log file.
    logging.basicConfig(filename=f"{name}.log", level=logging.DEBUG)
    logging.info(f"Preparing Concepts for experiment {name}.")

    start = perf_counter()
    concepts_labels_data = read_file(concept_labels)
    concept_acts, num_layers = load_activations(args.embeddings, num_neurons)
    
    concepts_dict = get_concepts_dict(concepts_labels_data)
    concepts2class = assign_labels_to_concepts(concepts_dict)
    concepts2class = {i[0]: i[1] for i in enumerate(concepts.split(' '))}

    layers = list(layers.split(' '))

    print("[INFO] Concepts: ", list(concepts2class.values()))
    print("[INFO] Total Layers: ", layers)

    logging.info(f"[INFO] Concepts {list(concepts2class.values())}")
    logging.info(f"[INFO] Total Layers {layers}")

    concept_bottleneck, tokens_bottleneck = get_bottlenecks(concept_acts,
                                                            concept_sents,
                                                            concept_labels,
                                                            num_layers, max_sentence_l)
    
    layer_wise_cavs, controlled_cavs_per_layer = run(concept_bottleneck, tokens_bottleneck, concepts2class, model_type, num_workers, no_of_runs, if_controlled, layers)
    write_the_cavs(output_folder, layer_wise_cavs, controlled_cavs_per_layer, if_controlled)
    end = perf_counter()

    logging.info(f"Completed the process in {end-start}s")

if __name__ == '__main__':
    main()