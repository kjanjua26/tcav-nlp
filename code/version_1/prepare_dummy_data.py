"""
    Prepare dummy data for the task of running code.
    This prepares the dummy data.
    The data is of the following form:
        
        base_data.txt, base_data_labels.txt
        base_data.txt: contains a sentence per line where each word is separated by a space
        base_data_labels.txt: contains labels per word in a single line (for a sentence) separated by space - corresponding to base_data.txt

        concept_data.txt, concept_data_labels.txt
        concept_data.txt: contains a sentence per line where each word is separated by a space
        concept_data_labels.txt: contains labels per word in a single line (for a sentence) separated by space - corresponding to concept_data.txt
    
    The dummy data is prepared from the Blog Authorship Corpus dataset which can be downloaded from this website:
        https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm

    The dataset is prepared in the following style.

    base_data.txt
        I went to the school in the morning
    concept.txt
        I am going to school
    concept_labels.txt
        LABEL1 LABEL2 LABEL3 LABEL2 LABEL1
"""

import os
from tqdm import tqdm
from glob import glob
import numpy as np
import random
import nltk 

base_path = "/Users/Janjua/Desktop/QCRI/Work/data/"
base_data_path = os.path.join(base_path, "20news-bydate/20news-bydate-test")
concept_data_path = os.path.join(base_path, "20news-bydate/20news-bydate-train")

def get_data_files(fp):
    """
    Since the data from 20news is in folders, we need to get all the documents in the set.

    Arguments
        fp (str): folder path which has the documents.
    Returns
        docs_list (list): the list of all the documents.
    """
    list_of_docs = []
    for ix, doc in enumerate(glob(fp + "/*/*")):
        list_of_docs.append(doc)
    
    return list_of_docs

def get_500_documents_only(docs_list):
    """
    Since this is a dummy data, only the 500 documents are fetched.
    The documents are bit old and not well formatted, so a certain condition has to be met to get rid of irrelevant lines.
    
    Arguments
        docs_list (list): the list of documents to choose the top 500 from.
    
    Returns
        top_500 (list): the list containing the top 500 documents.
    """
    top_500 = []
    count = 0
    random.shuffle(docs_list)

    for ix, doc in enumerate(docs_list):
        if count < 1000: # from approximately first 1000 files, 500 files meet the criteria.
            try:
                with open(doc, 'r') as fp:
                    lines = fp.readlines()
                    condition = ['True' if 'writes' in line else 'False' for line in lines]
                    if 'True' in condition:
                        top_500.append(doc)
            except:
                print(f"Invalid File # - {ix}, skipping.")
            count += 1

    return top_500

def clean_line(line):
    """
    Cleans the line by removing any un-necessary numbers, symbols, etc.

    Argument
        line (list containing the words): the line to be cleaned.
    
    Returns
        word_lst (list): the cleaned list of words of the line.
    """
    word_lst = []
    for word in line:
        if word.isalpha():
            word_lst.append(word)
    return word_lst


def prepare_data(docs_list, mode):
    """
    Prepares the base data (used in main.py).
    
    Arguments
        docs_list - top_500 (list): the list of paths of all the documents - the top 500 ones after having met a condition.
        mode (str): the type to prepare, if it is concept, prepare labels too, else just the data.
    """

    outputfile = os.path.join(base_path, f"d{mode}.txt")
    labelspath = os.path.join(base_path, "dconcept_labels.txt")
    
    outtxt = open(outputfile, 'w')
    labelstxt = open(labelspath, 'w')

    for ix, doc in tqdm(enumerate(docs_list)):
        docstring = ""
        with open(doc, 'r') as fp:
            lines = fp.readlines()
            write_word_index = [line for line in enumerate(lines) if 'writes' in line[1]][0][0]
            lines = lines[write_word_index+1:]
        
            for line in lines:
                line = line.strip()
                docstring += ' ' + line

            docstring = docstring.replace('>', '')
            docstring = docstring.split('.')
            
            for line in docstring:
                line = clean_line(line.split(' '))
                if len(line) > 20:
                    if mode == "base":
                        line = ' '.join(line)
                        outtxt.write(line + '\n')

                    if mode == "concept": # mode is concept.
                        tagged = nltk.pos_tag(line)
                        labels = [x[1] for x in tagged]
            
                        assert len(line) == len(labels)
            
                        labels = ' '.join(labels)
                        line = ' '.join(line)
                        outtxt.write(line + '\n')
                        labelstxt.write(labels + '\n')
        
    outtxt.close()
    labelstxt.close()

def main():
    """
    mode (str): base or concept
    output_file_path (str): the path to store the output file.
    """

    base_docs_list = get_data_files(base_data_path)
    concept_docs_list = get_data_files(concept_data_path)

    top_500_base = get_500_documents_only(base_docs_list)
    top_500_concept = get_500_documents_only(concept_docs_list)

    modes = {"base": top_500_base, "concept": top_500_concept}

    for mode, lst in modes.items():
        prepare_data(lst, mode)

main()