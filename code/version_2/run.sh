#!/bin/bash


base_folder="v2_test"
concept_path="../../../${base_folder}/dconcept.in"
concept_labels="../../../${base_folder}/dconcept_labels.in"
concept_activations="../../../${base_folder}/dconcept_acts.json"

base_path="../../../${base_folder}/dbase.txt"
base_labels="../../../${base_folder}/dbase_labels.txt"
base_activations="../../../${base_folder}/dbase_activations.json"

output_directory="../../../${base_folder}/results"

layer_wise_cav_pickle_path="../../../${base_folder}/results/layer_wise_cavs.pickle"

mode_tcav="wm"
word="[MASK]"
model="bert-base-uncased"
model_type="SGDC"

workers=4
runs=10

inference_results="../../../${base_folder}/results/inference_word_mode.pickle"

#echo "Extract Activations!"
#python extraction.py -m $model -i $concept_path -o $concept_activations -t "json"
#python extraction.py -m $model -i $base_path -o $base_activations -t "json"

echo "Prepare Concepts and Training CAVs!"
python prepare_concepts.py -i $concept_path -l $concept_labels -e $concept_activations -c "DT NNS CC JJ" -o $output_directory -lm $model_type -w $workers -rs $runs

#echo "Making Directories to Store Results!"
#mkdir ../../../${base_folder}/results/layer_wise
#mkdir ../../../${base_folder}/results/concept_wise

#echo "Computing TCAVs!"
#python compute_tcavs.py -b $base_activations -c $layer_wise_cav_pickle_path -o $output_directory -bs $base_path -bl $base_labels -m $mode_tcav -w $word

#echo "Scoring the results now."
#python inference.py -m $model -b $base_path -l $base_labels -s $inference_results -o $output_directory

# Other Mode Option:
#python compute_tcavs.py -b $base_activations -c $layer_wise_cav_pickle_path -r $layer_wise_cav_random_path -o $output_directory -bs $base_path -m $mode_tcav -w $word
#python3.7 -m streamlit