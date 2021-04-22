#!/bin/bash

model="bert-base-cased"
base_folder="controlled_test"
DIR="/alt/mt/work/durrani/Causation_Analysis/ProbingClassifiers/POS/Representations/${model}"
data_folder="data_folder"

concept_path="../../../${data_folder}/wsj.19.dev.conllx.word"
concept_labels="../../../${data_folder}/wsj.19.dev.conllx.label"
concept_activations="../../../${data_folder}/wsj.19.dev.conllx.json"

base_path="../../../${data_folder}/wsj.20.test.conllx.word"
base_labels="../../../${data_folder}/wsj.20.test.conllx.label"
base_activations="../../../${data_folder}/wsj.20.test.conllx.json"

output_directory="../../../${base_folder}/results"

layer_wise_cav_pickle_path="../../../${base_folder}/results/layer_wise_cavs.pickle"
controlled_layer_wise_cav_path="../../../${base_folder}/results/controlled_layer_wise_cavs.pickle"

mode_tcav="wm"
word="[MASK]"
model_type="LR"
process_mode="1" # 0 is for non-MASK, 1 is for MASK.
use_grad="0" # 0 for acts, 1 for grad.
if_controlled="1" # 0 for no controlled experiment, 1 for controlled experiment as well.
workers=4
runs=3
if_rand=1
name="controlled_experiment"

#echo "Extract Activations for Model ${model}!"
#python -u extraction.py -m $model -i $concept_path -o $concept_activations -t "json"
#python -u extraction.py -m $model -i $base_path -o $base_activations -t "json"

# 10 concepts => NN JJ NNS JJR JJS DT CC CD VB VBP

#echo "Prepare Concepts and Training CAVs!"
#python -u prepare_concepts.py -n $name -i $concept_path -l $concept_labels -e $concept_activations -c "NN JJ" -o $output_directory -lm $model_type -w $workers -rs $runs -ce $if_controlled -tl "13"

echo "Computing TCAVs!"
python -u compute_tcavs.py -n $name -b $base_activations -c $layer_wise_cav_pickle_path -rc $controlled_layer_wise_cav_path -o $output_directory -bs $base_path -bl $base_labels -w $word -rs $runs -m $model -ir $if_rand -pm $process_mode -g $use_grad