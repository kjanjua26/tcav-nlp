#!/bin/bash

#SBATCH -J ENet_l1_0.0001_l2_0.0001_Acts_10C # name of the job
#SBATCH -o ENet_l1_0.0001_l2_0.0001_Acts_10C.txt # the output file name.
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
base_folder="ENet_l1_0.0001_l2_0.0001"
aggregation_folder="/export/work/hsajjad/neuron_classifier/${model}/pos/l1_0.0001_l2_0.0001/models/"
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
use_grad="0" # 0 for acts, 1 for grad.

workers=4
runs=1
if_rand=0

#echo "Extract Activations for Model ${model}!"
#python -u extraction.py -m $model -i $concept_path -o $concept_activations -t "json"
#python -u extraction.py -m $model -i $base_path -o $base_activations -t "json"

#echo "Aggregation"
#python -u aggregate.py -a $aggregation_folder -lc "NN JJ NNS JJR JJS DT CC CD VB VBP" -o $output_directory

echo "Computing TCAVs!"
python -u compute_tcavs.py -b $base_activations -c $layer_wise_cav_pickle_path -rc $random_layer_wise_cav_random_path -o $output_directory -bs $base_path -bl $base_labels -w $word -rs $runs -m $model -ir $if_rand -pm $process_mode -g $use_grad
