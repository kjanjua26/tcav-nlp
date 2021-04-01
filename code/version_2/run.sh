#!/bin/bash

#SBATCH -J mask_test_token_loss_grads_last_layer_10C # name of the job
#SBATCH -o mask_test_token_loss_grads_last_layer.txt # the output file name.
#SBATCH -p gpu-all
#SBATCH --gres gpu:1
#SBATCH -c 4
#SBATCH --mem 150000MB

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=kamranejaz98@gmail.com

module load slurm

model="bert-base-cased"
base_folder="mask_test_token_loss_grads_last_layer_10C"
DIR="/alt/mt/work/durrani/Causation_Analysis/ProbingClassifiers/POS/Representations/${model}"
data_folder="/alt/mt/tcav/data"

concept_path="${DIR}/wsj.train.conllx.word"
concept_labels="${DIR}/wsj.train.conllx.label"
concept_activations="${data_folder}/${model}/wsj.train.conllx.json"

base_path="${DIR}/wsj.20.test.conllx.word"
base_labels="${DIR}/wsj.20.test.conllx.label"
base_activations="${data_folder}/${model}/wsj.20.test.conllx.json"

output_directory="/alt/mt/tcav/${model}/${base_folder}"

layer_wise_cav_pickle_path="/alt/mt/tcav/${model}/${base_folder}/layer_wise_cavs.pickle"
random_layer_wise_cav_random_path="/alt/mt/tcav/${model}/${base_folder}/layer_wise_random_cavs.pickle"

mode_tcav="wm"
word="[MASK]"
model_type="LR"
process_mode="1" # 0 is for non-MASK, 1 is for MASK.
use_grad="1" # 0 for acts, 1 for grad.

workers=4
runs=1
if_rand=0

#echo "Extract Activations for Model ${model}!"
#python -u extraction.py -m $model -i $concept_path -o $concept_activations -t "json"
#python -u extraction.py -m $model -i $base_path -o $base_activations -t "json"

#echo "Prepare Concepts and Training CAVs!"
#python -u prepare_concepts.py -i $concept_path -l $concept_labels -e $concept_activations -c "NN JJ NNS JJR JJS DT CC CD VB VBP" -o $output_directory -lm $model_type -w $workers -rs $runs -ir $if_rand

echo "Computing TCAVs!"
python -u compute_tcavs.py -b $base_activations -c $layer_wise_cav_pickle_path -rc $random_layer_wise_cav_random_path -o $output_directory -bs $base_path -bl $base_labels -w $word -rs $runs -m $model -ir $if_rand -pm $process_mode -g $use_grad
