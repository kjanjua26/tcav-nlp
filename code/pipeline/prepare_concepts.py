"""
Prepare the data for concepts.
This version uses: IMDB Movie Reviews Dataset for gendered concepts.
The dataset is taken from: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
Given a corpus of sentences, this file prepares the data for concepts with appropriate labels.
    By searching for the appropriate sentence related to the concept in the input corpus.
"""

import pandas as pd
import argparse
import json, re
from tqdm import tqdm

def load_the_data(datapath):
    """Load the .csv data and get all the sentences."""
    if datapath.endswith(".csv"):
        data = pd.read_csv(datapath)
    else:
        data = open(datapath, 'r')
    
    paras = data["review"].tolist()
    return paras

def clean(sent):
    """Clean the corpus by removing all the un-necessary tags."""
    regex = re.compile("<.*?>")
    cleaned_sent = re.sub(regex, '', sent)
    cleaned_sent = cleaned_sent.replace(',', '')
    return cleaned_sent

def get_each_para(datapath):
    """Returns each paragraph as it is called."""
    paras = load_the_data(datapath)
    for para in paras:
        yield para
    return

def get_each_sentence(para):
    """Returns each sentence as it is called."""
    for sent in para:
        if len(sent) > 0: # getting rid of spaces.
            yield clean(sent.lower())
    return

def parse_concepts(concept_files):
    """Parses the concepts and stores in memory for lookup."""
    concepts = {}
    for ix, file in enumerate(concept_files):
        with open(file, 'r') as fp:
            lines = fp.readlines()
            concepts[ix] = []
            for line in lines:
                line = line.strip()
                concepts[ix].append(line)
    return concepts

def check_for_gendered_concept(sent, concepts):
    """Checks if a gendered concept exists."""
    for k, v in concepts.items():
        words = sent.split(' ')
        for word in words:
            if word in v:
                return (True, k)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i", "--input_corpus", 
        help="File path with one sentence per line to prepare concepts from.")
    parser.add_argument("-c", "--concept_type", default="gender",
        help="The type of concept to prepare data for.")
    parser.add_argument("-o", "--output_file",
        help="The output file to store the concepts data in.")
    parser.add_argument("-f", "--concept_files", nargs="+",
        default=["gender_f.txt", "gender_m.txt"],
        help="The concept files.")
    args = parser.parse_args()

    print("Reading the concepts.")
    concepts = parse_concepts(args.concept_files)

    with open(args.output_file, 'w') as output:
        output.write("sentence" + "," + "label" + "\n")
        for ix, para in tqdm(enumerate(get_each_para(args.input_corpus))):
            sents = para.split('.')
            for jx, sent in enumerate(get_each_sentence(sents)):
                sent = sent.strip()
                contains_concept = check_for_gendered_concept(sent, concepts)
                if contains_concept is not None:
                    val, label = contains_concept
                    #print("({},{})".format(ix, jx), label, sent)
                    output.write(sent)
                    output.write(",")
                    output.write(str(label))
                    output.write("\n")
    output.close()

if __name__ == '__main__':
    main()
