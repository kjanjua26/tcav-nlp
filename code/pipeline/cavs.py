"""
CAV: Concept Activation Vector
Read the activation files, train a linear classifier and compute the CAVs.
"""

import json
import argparse
import re
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def build_model(model_name):
    """Return the model as per the name.""""
    if model_name.startswith("logreg"):
        lm = LogisticRegression()
    elif model_name.startswith("elastic"):
        lm = ElasticNet()
    else:
        lm = LinearSVC(gamma='auto')
    
    return lm

def fit_model(model, X, y):
    """Fit the model on the X,y data."""
    model.fit(X, y)

def get_cavs(model):
    """Get the CAVs"""
    pass

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--base_acts", 
        help="File path to the base activations json file.")
    parser.add_argument("-c", "--concept_acts", 
        help="File path to the concept activations json file.")
    parser.add_argument("-f", "--concept_files", nargs="+",
        default=["gender_f.txt", "gender_m.txt"],
        help="The concept files.")
    parser.add_argument("-o", "--output_file",
        help="To store the CAVs to.")
    parser.add_argument("-lm", "--linear_model",
        help="The name of the linear model to train.")
    
    args = parser.parse_args()

if __name__ == "__main__":
    main()