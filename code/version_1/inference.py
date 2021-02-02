"""
Runs the inference on computed scores against the sentences and also in an online fashion comparing the model output.
"""
import nltk
from transformers import pipeline
import pandas as pd
import numpy as np
import argparse
import json
from tqdm import tqdm

from prepare_concepts import get_concepts_dict
from compute_tcavs import read_sentences, mask_out_each_concept_in_base_sentences

def build_the_unmasker(model):
    """
    Builds the unmasker model to use for predictions.

    Arguments
        model (str): the model type.
    
    Returns
        unmasker (transformers model): the unmasker with fill-mask model type.
    """
    unmasker = pipeline('fill-mask', model=model)
    return unmasker

def read_the_csv(csvfile):
    """
    Reads the csv to parse.

    Arguments
        csvfile (str): the path to the csv file to parse

    Returns
        df (DataFrame): the loaded csv file.
    """
    df = pd.read_csv(csvfile)
    return df

def df_of_a_layer(df, layer_no):
    """
    Returns the df of a specific layer since the csv has all the layers.

    Arguments
        df (DataFrame): the df to get the sub-df from.
        layer_no (int): the layer no to get the df of.
    
    Returns
        df (DataFrame): the extracted sub-df.
    """
    df = df.loc[df["Layer"] == layer_no]
    df.reset_index(drop=True, inplace=True)
    return df

def get_the_dict_of_tcavs_against_each_tested_concept(df):
    """
    Returns the TCAVs dict for all concepts from the.

    Arguements
        df (DataFrame): the df to get the TCAVs from.
    
    Returns
        dict_of_tcavs (dict): the tcavs of all concepts in a dict.
        tested_concepts (list): the list of concepts that TCAVs were computed for.
    """
    dict_of_tcavs = {}
    tested_concepts = set()
    for ix, val in df.iterrows():
        concept, _, _, tcav = val
        tcav = tcav.split('-')
        row_wise_dict = {}
        for c in tcav:
            co, v = c.split(':')
            out_co = co.replace("'", "").replace("{", "").strip()
            v = float(v.replace("}", ""))
            row_wise_dict[out_co] = v
            tested_concepts.add(out_co)
        # sort the dictionary based on values, descending order.
        row_wise_dict = dict(sorted(row_wise_dict.items(), key=lambda item: item[1], reverse=True))
        dict_of_tcavs[concept] = row_wise_dict
    
    return dict_of_tcavs, list(tested_concepts)

def score_tcavs_from_ground_truth(dict_of_tcavs, masked_out_sentences,
                                concepts_tested, top_k=1):
    """
    Count of the masked concept and the highest TCAV concept, called as the tcav accuracy.

    Arguments
        dict_of_tcavs (dict): the dict of TCAVs.
        masked_out_sentences (dict): the dictionary of sentences masked out for a concept.
        concepts_tested (list): the list of tested concepts.
        top_k (int): the top_k results to retrieve the scores for.
    Returns
        scores_dict (dict): the scores dictionary.
    """
    scores_dict = {}
    for concept_tested in concepts_tested:
        count = 0
        sent_len = 0
        masked_sents = masked_out_sentences[concept_tested]
        for ix, sent in enumerate(masked_sents):
            if "[MASK]" in sent.split(' '):
                sent_len += 1
                tcavs_of_tested_concepts = dict_of_tcavs[concept_tested]
                max_tcav = {key: value for key, value in tcavs_of_tested_concepts.items() if value in sorted(set(tcavs_of_tested_concepts.values()), reverse=True)[:top_k]}
                max_tcav = list(max_tcav.keys())
                if concept_tested in max_tcav: # could be with similar concepts.
                    count += 1

        scores_dict[concept_tested] = float(count)/float(sent_len)
    return scores_dict

def check_how_accurate_BERT_predictions_are_tagged(unmasker, masked_out_sentences,
                                                dict_of_tcavs, concepts_tested, conf_threshold=0.5):
    """
    Check how accurate BERT's MASK predictions are tagged compared to the ground truth (the masked sentences).
    Compares the tags by NLTK to the actual tag in the gold label.

    Arguments
        unmasker (BERT model): the unmasker.
        masked_out_sentences (dict): the dictionary of sentences masked out for a concept.
        dict_of_tcavs (dict): the dict of TCAVs.
        concepts_tested (list): the list of tested concepts.
        conf_threshold (float): the confidence threshold for BERT's prediction.
    Returns
        scores_dict (dict): the scores dictionary.
    """
    scores_dict = {}
    
    for concept_tested in concepts_tested:
        masked_sents = masked_out_sentences[concept_tested]
        count = 0
        sent_len = 0
        for ix, sent in enumerate(masked_sents):
            if "[MASK]" in sent.split(' '):
                
                index_of_word_mask = sent.split(' ').index("[MASK]")
                predictions = unmasker(sent)
                for prediction in predictions:
                    pred_sent = prediction['sequence']
                    pred_sent_conf = prediction['score']
                    # if the conf is higher than a threshold, otherwise the sentence is ambiguous already.
                    # still noise exists because nltk could tag wrongly, which is does since not many tags match the gt.
                    if pred_sent_conf > conf_threshold:
                        sent_len += 1
                        pred_sent = pred_sent.replace('[CLS]', '').replace('[SEP]', '').strip()
                        tagged_pred = nltk.pos_tag(pred_sent)
                        tagged_pred = [x[1] for x in tagged_pred]
                        tagged_word = tagged_pred[index_of_word_mask]
                        try:
                            tcavs_of_tag = dict_of_tcavs[tagged_word]
                        except:
                            tcavs_of_tag = {}
                        if tcavs_of_tag != {}:
                            if tagged_word == concept_tested:
                                count += 1

        scores_dict[concept_tested] = float(count)/float(sent_len)
        
    return scores_dict

def check_how_accurate_BERT_prediction_tag_is(unmasker, masked_out_sentences,
                                            dict_of_tcavs, concepts_tested, top_k=1, conf_threshold=0.05):
    """
    Regardless of the ground truth, check if what BERT predicted (tagged using NLTK) matches the highest scored results.

    Arguments
        unmasker (BERT model): the unmasker.
        masked_out_sentences (dict): the dictionary of sentences masked out for a concept.
        dict_of_tcavs (dict): the dict of TCAVs.
        concepts_tested (list): the list of tested concepts.
        top_k (int): the top_k results to retrieve the scores for.
        conf_threshold (float): the confidence threshold for BERT's prediction.
    Returns
        scores_dict (dict): the scores dictionary.
    """
    scores_dict = {}
    
    for concept_tested in concepts_tested:
        masked_sents = masked_out_sentences[concept_tested]
        sent_len = 0
        count = 0
        for ix, sent in enumerate(masked_sents):
            if "[MASK]" in sent.split(' '):
                index_of_word_mask = sent.split(' ').index("[MASK]")
                predictions = unmasker(sent)

                for prediction in predictions:
                    pred_sent = prediction['sequence']
                    pred_sent_conf = prediction['score']
                    # if the conf is higher than a threshold, otherwise the sentence is ambiguous already.
                    # still noise exists because nltk could tag wrongly, which is does since not many tags match the gt.
                    if pred_sent_conf > conf_threshold:
                        sent_len += 1
                        pred_sent = pred_sent.replace('[CLS]', '').replace('[SEP]', '').strip()
                        tagged_pred = nltk.pos_tag(pred_sent)
                        tagged_pred = [x[1] for x in tagged_pred]
                        tagged_word = tagged_pred[index_of_word_mask]
                        try:
                            tcavs_of_tag = dict_of_tcavs[tagged_word]
                        except:
                            tcavs_of_tag = {}
                        if tcavs_of_tag != {}:
                            max_tcav = {key: value for key, value in tcavs_of_tag.items() if value in sorted(set(tcavs_of_tag.values()), reverse=True)[:top_k]}
                            max_tcav_conps = list(max_tcav.keys())
                            max_tcav_conf = list(max_tcav.values())
                            if tagged_word in max_tcav_conps:
                                #print("Concept Tested: ", concept_tested, "\nTagged Word: ", tagged_word, "\nConf: ", pred_sent_conf, "\nMax TCAV: ", max_tcav_conps, "\nSent: ", pred_sent, "\nTCAVs: ", max_tcav_conf, "\nMasked: ", sent)
                                count += 1
                                
        scores_dict[concept_tested] = float(count)/(float(sent_len))
    return scores_dict

def other_concepts_used_in_the_prediction_write_to_csv(unmasker, masked_out_sentences, dict_of_tcavs,
                                    concepts_tested, top_k, output_directory, conf_threshold=0.05):
    """
    Check what other concepts are used in the prediction of the MASKED word.
    top_k arguments tells what the top k concepts were used in the prediction.

    Arguments
        unmasker (BERT model): the unmasker.
        masked_out_sentences (dict): the dictionary of sentences masked out for a concept.
        dict_of_tcavs (dict): the dict of TCAVs.
        concepts_tested (list): the list of tested concepts.
        top_k (int): the top_k results to retrieve the scores for.
        conf_threshold (float): the confidence threshold for BERT's prediction.
    """

    fp = open(output_directory, "w")
    fp.write("Concept_Tested" + "," + "Predicted_Sent" + "," + "Tagged_Word" + "," + "Concepts_Used" + "," + "TCAVs" + "," + "Masked_Sentence" + "," + "BERT_Confidence" + "\n")

    for concept_tested in concepts_tested:
        masked_sents = masked_out_sentences[concept_tested]
        sent_len = 0
        count = 0
        
        for ix, sent in tqdm(enumerate(masked_sents)):
            if "[MASK]" in sent.split(' '):
                index_of_word_mask = sent.split(' ').index("[MASK]")
                predictions = unmasker(sent)

                for prediction in predictions:
                    pred_sent = prediction['sequence']
                    pred_sent_conf = prediction['score']
                    if pred_sent_conf > conf_threshold:
                        sent_len += 1
                        pred_sent = pred_sent.replace('[CLS]', '').replace('[SEP]', '').strip()
                        tagged_pred = nltk.pos_tag(pred_sent)
                        tagged_pred = [x[1] for x in tagged_pred]
                        tagged_word = tagged_pred[index_of_word_mask]

                        try:
                            tcavs_of_tag = dict_of_tcavs[tagged_word]
                        except:
                            tcavs_of_tag = {}
                            
                        if tcavs_of_tag != {}:
                            max_tcav = {key: value for key, value in tcavs_of_tag.items() if value in sorted(set(tcavs_of_tag.values()), reverse=True)[:top_k]}
                            max_tcav_conps = ' '.join(list(max_tcav.keys()))
                            max_tcav_conf = ' '.join(str(x) for x in list(max_tcav.values()))
                            fp.write(concept_tested + "," + pred_sent.replace(',', '') + "," + tagged_word + "," + max_tcav_conps + "," + max_tcav_conf + "," + sent.replace(',', '') + "," + str(conf_threshold) + "\n")
    
    fp.close()

def write_to_json(results, output_file_path):
    """
    Write the results to JSON.

    Arguments
        results (dict): the dictionary to dump to json.
        output_file_path (str): the output file path.
    """
    with open(output_file_path, "w") as fp:
        json.dump(results, fp)

def run(model, base_corpus, base_labels, scores, class2labels, output_directory):
    """
    The runner code.

    Arguments
        model (str): the model.
        base_corpus (lst): the list of sentences.
        base_labels (lst): the list of labels.
        scores (df): the Dataframe.
        class2labels (lst): the list of all the concepts.
        output_directory (str): the output directory path.
    """
    results = {}
    top_ks = [1, 2, 3] # the top-k where k \in [1,2,3]
    
    unmasker = build_the_unmasker(model)
    masked_out_sentences = mask_out_each_concept_in_base_sentences(base_corpus, base_labels, list(class2labels.keys()))
    layer_13_df = df_of_a_layer(scores, 13)
    dict_of_tcavs, concepts_tested = get_the_dict_of_tcavs_against_each_tested_concept(layer_13_df)
    
    for top_k in tqdm(top_ks):
        scores_dict_gt = score_tcavs_from_ground_truth(dict_of_tcavs, masked_out_sentences,
                                concepts_tested, top_k)
                                
        scores_dict_bert = check_how_accurate_BERT_prediction_tag_is(unmasker, masked_out_sentences,
                                dict_of_tcavs, concepts_tested, top_k=top_k)
        
        other_concepts_used_in_the_prediction_write_to_csv(unmasker, masked_out_sentences, dict_of_tcavs,
                                    concepts_tested, top_k, output_directory + f"/{top_k}_other_concepts_used.csv")

        results[f"experiment_gt_{top_k}"] = scores_dict_gt
        results[f"experiment_bert_{top_k}"] = scores_dict_bert

    write_to_json(results, output_directory + "/results.json")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", 
        help="The model to load.", default="bert-base-uncased")
    parser.add_argument("-b", "--base_corpus",
        help="The sentences to test by masking each concept.")
    parser.add_argument("-l", "--base_labels",
        help="Labelling the sentences and checking the results.")
    parser.add_argument("-s", "--scores_csv",
        help="The computed scores csv file.")
    parser.add_argument("-o", "--output_directory",
        help="The output directory to write the results to.")
    
    args = parser.parse_args()

    model = args.model
    base_corpus = args.base_corpus
    base_labels = args.base_labels
    scores = args.scores_csv
    output_directory = args.output_directory

    base_corpus = read_sentences(base_corpus)
    base_labels = read_sentences(base_labels)
    scores = read_the_csv(scores)
    class2labels = get_concepts_dict(base_labels)

    run(model, base_corpus, base_labels, scores, class2labels, output_directory)

if __name__ == '__main__':
    main()