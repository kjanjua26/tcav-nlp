#!/bin/bash

base_folder="p_10_concepts_test"
concept_path="../../../${base_folder}/wsj.19.dev.conllx.word"
concept_labels="../../../${base_folder}/wsj.19.dev.conllx.label"
concept_activations="../../../${base_folder}/wsj.19.dev.conllx.json"

base_path="../../../${base_folder}/wsj.20.test.conllx.word"
base_labels="../../../${base_folder}/wsj.20.test.conllx.label"
base_activations="../../../${base_folder}/wsj.20.test.conllx.json"

output_directory="../../../${base_folder}/results"

layer_wise_cav_pickle_path="../../../${base_folder}/results/layer_wise_cavs.pickle"
random_layer_wise_cav_random_path="../../../${base_folder}/results/random_layer_wise_cavs.pickle"

mode_tcav="wm"
word="[MASK]"
model="bert-base-uncased"
model_type="LR"

workers=4
runs=10
if_rand=0
inference_results="../../../${base_folder}/results/inference_word_mode.pickle"

echo "Extract Activations!"
python extraction.py -m $model -i $concept_path -o $concept_activations -t "json"
python extraction.py -m $model -i $base_path -o $base_activations -t "json"

echo "Prepare Concepts and Training CAVs!"
python -u prepare_concepts.py -i $concept_path -l $concept_labels -e $concept_activations -c "NN JJ NNS JJR JJS DT CC CD VB VBP" -o $output_directory -lm $model_type -w $workers -rs $runs -ir $if_rand

echo "Computing TCAVs!"
python -u compute_tcavs.py -b $base_activations -c $layer_wise_cav_pickle_path -rc $random_layer_wise_cav_random_path -o $output_directory -bs $base_path -bl $base_labels -w $word -rs $runs -m $model -ir $if_rand -w $workers