#!/bin/bash

base_folder="paper_test"
concept_path="../../../${base_folder}/dconcept.in"
concept_labels="../../../${base_folder}/dconcept_labels.in"
concept_activations="../../../${base_folder}/dconcept_acts.json"

base_path="../../../${base_folder}/dbase.txt"
base_labels="../../../${base_folder}/dbase_labels.txt"
base_activations="../../../${base_folder}/dbase_activations.json"

output_directory="../../../${base_folder}/results"

layer_wise_cav_pickle_path="../../../${base_folder}/results/layer_wise_cavs.pickle"
layer_wise_cav_random_path="../../../${base_folder}/results/layer_wise_random_cavs.pickle"
mode_tcav="wm"
word="[MASK]"

echo "Extract Activations!"
python extraction.py -m "bert-base-uncased" -i $concept_path -o $concept_activations -t "json"
python extraction.py -m "bert-base-uncased" -i $base_path -o $base_activations -t "json"

echo "Prepare Concepts and Training CAVs!"
python prepare_concepts.py -i $concept_labels -l $concept_labels -e $concept_activations -c "DT NNS CC JJ" -o $output_directory

echo "Making Directories to Store Results!"
mkdir ../../../${base_folder}/results/layer_wise
mkdir ../../../${base_folder}/results/concept_wise

echo "Computing TCAVs!"
python compute_tcavs.py -b $base_activations -c $layer_wise_cav_pickle_path -r $layer_wise_cav_random_path -o $output_directory -bs $base_path -bl $base_labels -m $mode_tcav -w $word


# Other Mode Option:
#python compute_tcavs.py -b $base_activations -c $layer_wise_cav_pickle_path -r $layer_wise_cav_random_path -o $output_directory -bs $base_path -m $mode_tcav -w $word
#python3.7 -m streamlit