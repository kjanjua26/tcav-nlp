"""
    Compute the TCAVs for the given CAVs against each concept for each sentence in the base_data.txt.
    Use base_data_labels.txt for word analysis.
    Computes both ways: sentence TCAV and word TCAV.
"""

import argparse, sys, os
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
                # acts => (no_words, 768)
                # cav => (768, )
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

def compute_word_tcav(concept_cavs, bottleneck_base, sentences, num_layers, word):
    """
    TODO: This is wrong.

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
        layer_cavs = concept_cavs[str(ix)]
        for concept, cav in layer_cavs.items():
            word_tcav[str(ix)][concept] = []
            count = 0
            for jx, sent in enumerate(sentences):
                act_per_layer_per_sent = bottleneck_base[str(ix)][jx]
                words = list(sent.split(' '))
                if word in words:
                    selected_word_index = words.index(word)
                    selected_word_acts = act_per_layer_per_sent[selected_word_index]
                    dydx = directional_derivative(selected_word_acts, cav)
                    if dydx: count += 1

            tcav = float(count)/float(len(sentences))
            word_tcav[str(ix)][concept].append((word, tcav))

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

def run_for_chosen_word_write_to_pickle(sentences, concept_cavs, bottleneck_base, num_layers, word, output_directory):
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
        word_layer_wise_tcavs = compute_word_tcav(concept_cavs, bottleneck_base, sents, num_layers, word)
        concept_masked_tcav_dict[concept_masked] = word_layer_wise_tcavs
    
    # write to pickle file.
    with open(write_file_path, "wb") as writer:
        pickle.dump(concept_masked_tcav_dict, writer)

    # pretty-write the table to an HTML file.
    #re_format_the_csv_and_write_html(write_file_path, output_directory)

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

def word_layer_wise_plots(word_weightage, output_folder, word, concept_masked="default"):
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

    plt.figure(figsize=(12,10))
    for i, row in enumerate(data):
        X = np.arange(len(row))
        rect = plt.bar(X + i * gap, row, width = gap, color = lst_of_colors[i])
        plt.xticks(X, [ix for ix in concepts], rotation=70)
        for j in range(len(row)):
            height = rect[j].get_height()
            plt.text(rect[j].get_x() + rect[j].get_width()/2.0, height, f"{i+1}", ha='center', va='bottom')
    
    plt.ylabel("TCAV Scores")
    plt.title(f"TCAV Scores for each layer for word {word}")
    plt.savefig(output_folder + f"/{concept_masked}_{word}_all_layers.png")

def re_format_the_csv_and_write_html(csv_file, output_directory):
    """
    Prepares a multi-column csv file and writes to HTML files for better viewing.

    Arguments
        csv_file (str): the path to the written csv file.
        output_directory (str): the path to write the HTML file to.
    """
    df_old = pd.read_csv(csv_file)
    
    concepts = list(set(df_old["Concept_Masked"].tolist()))
    layers = ["0"] + [str(x) for x in set(df_old["Layer"].tolist())]
    tcav = df_old["TCAV"]

    def reformat_df():
        df = pd.DataFrame(columns=layers)
        df[0] = concepts
        df = df.set_index(0)
        df = df.drop(columns=['0'])
        return df

    def re_index_df(df, df_old, num_layers):    
        for jx in range(len(concepts)):
            for ix in range(num_layers):
                layer = layers[ix+1]
                concept = concepts[jx]
                cell = df_old.loc[df_old["Concept_Masked"] == concept, "TCAV"]
                cell = cell.values[ix]
                st = ""
                for c in cell.split('-'):
                    co, v = c.split(':')
                    out_co = co.replace("'", "").replace("{", "") + " =" + v.replace("}", "")
                    st += " " + out_co
                df.loc[concept, layer] = st

        return df
    
    df = reformat_df()
    df = re_index_df(df, df_old, 13)

    unique_concepts = set(df_old["Concept_Masked"].tolist())
    layers = [str(x) for x in set(df_old["Layer"].tolist())]

    def format_for_multi_columns():
        main_lst = defaultdict(dict)
        for ix, t in enumerate(tcav):
            layer = (ix%13)+1
            cell = t.split("-")
            d = []
            concepts_tested = set()
            for c in cell:
                co, v = c.split(':')
                out_co = co.replace("'", "").replace("{", "")
                concepts_tested.add(out_co)
                out_v = float(v.replace("}", "").strip())
                d.append(out_v)
            if layer not in main_lst.keys():
                main_lst[layer] = []
                main_lst[layer].append(d)
            else:
                main_lst[layer].append(d)

        full = []
        for ix in main_lst.keys():
            full.append(main_lst[ix])
            
        full_np = np.array(full)
        full_re_np = full_np.transpose(1, 0, 2) # it is (13, 33, 4) -> (33, 13, 4)
        full_re_np = full_re_np.reshape(len(unique_concepts), len(concepts_tested) * len(layers))
        return (full_re_np, list(concepts_tested))

    def multi_column_df(concepts_tested):
        midx = pd.MultiIndex.from_product([layers, concepts_tested])
        df = pd.DataFrame(full_re_np, index=unique_concepts, columns=midx)
        return df

    full_re_np, concepts_tested = format_for_multi_columns()
    df = multi_column_df(concepts_tested)
    df_to_html = HTML(df.to_html())
    html = df_to_html.data

    with open(output_directory + '/html_file.html', 'w') as f:
        f.write(html)

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
    parser.add_argument("-bl", "--base_labels",
        help="Path to base labels.")
    parser.add_argument("-c", "--concepts_cavs",
        help="The concept CAV pickle file computed.")
    parser.add_argument("-r", "--randoms_cavs",
        help="The random CAV pickle file computed for the t-test.")
    parser.add_argument("-o", "--output_directory",
        help="The output directory to store the results in.")
    parser.add_argument("-m", "--compute_mode", default="s",
        help="The compute mode to compute the TCAV in (w for word, s for sentence, wm for testing for [MASK] only.).")
    parser.add_argument("-w", "--word", default="[MASK]",
        help="The word to compute the TCAVs for in w mode.")

    args = parser.parse_args()
    num_neurons = 768

    base_acts_to_compute = args.base_acts_to_compute
    base_sentences = args.base_sentences
    base_labels = args.base_labels
    output_directory = args.output_directory
    compute_mode = args.compute_mode
    word = args.word

    start = perf_counter()
    
    concept_cavs = load_pickle_files(args.concepts_cavs)
    random_cavs = load_pickle_files(args.randoms_cavs)
    
    sents = read_sentences(base_sentences)
    base_labels = read_sentences(base_labels)

    concepts_dict = get_concepts_dict(base_labels)
    concepts2class = assign_labels_to_concepts(concepts_dict)

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

    elif compute_mode == "wm":
        print("Computing Word Level Results.")
        masked_sents = mask_out_each_concept_in_base_sentences(sents, base_labels, list(concepts2class.values()))
        run_for_chosen_word_write_to_pickle(masked_sents, concept_cavs, bottleneck_base, base_num_layers, word, output_directory)

    elif compute_mode == "w":
        word_layer_wise_tcavs = compute_word_tcav(concept_cavs, bottleneck_base, sents, base_num_layers, word)
        write_word_tcavs(output_directory, word_layer_wise_tcavs)
        weight_dict = get_specific_word_weightage(word_layer_wise_tcavs, word)
        word_layer_wise_plots(weight_dict, output_directory, word)


if __name__ == '__main__':
    main()