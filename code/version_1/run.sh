#!/bin/bash

concept_path="../../../data/dconcept.in"
concept_labels="../../../data/dconcept_labels.in"
concept_activations="../../../data/dconcept_acts.json"

base_path="../../../data/dbase.txt"
base_labels="../../../data/dbase_labels"
base_activations="../../../data/dbase_activations.json"

output_directory="../../../data/results"

layer_wise_cav_pickle_path="../../../data/results/layer_wise_cavs.pickle"
layer_wise_cav_random_path="../../../data/results/layer_wise_random_cavs.pickle"
mode_tcav="w"
word="[MASK]"

echo "Extract Activations!"
python extraction.py -m "bert-base-uncased" -i $concept_path -o $concept_activations -t "json"
python extraction.py -m "bert-base-uncased" -i $base_path -o $base_activations -t "json"

echo "Prepare Concepts and Training CAVs!"
python prepare_concepts.py -i $concept_labels -l $concept_labels -e $concept_activations -c "DT NNS CC JJ" -o $output_directory

echo "Making Directories to Store Results!"
mkdir ../../../data/results/layer_wise
mkdir ../../../data/results/concept_wise

echo "Computing TCAVs!"
python compute_tcavs.py -b $base_activations -c $layer_wise_cav_pickle_path -r $layer_wise_cav_random_path -o $output_directory -bs $base_path -m $mode_tcav -w $word

# Other Mode Option:
# python compute_tcavs.py -b "../../../data/dbase_activations.json" -c "../../../data/results/layer_wise_cavs.pickle" -r "../../../data/results/layer_wise_random_cavs.pickle" -o "../../../data/results" -bs "../../../data/dbase.txt" -bl "../../../data/dbase_labels.txt" -m "wm" -w "[MASK]"
#python3.7 -m streamlit

