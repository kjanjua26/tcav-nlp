## Running the Pipeline

The pipeline is divided into several steps. However, first we compute the base representations of all the sentences. 
Then we compute representations of sentences containing genders (he/she/woman/man, etc) and label those representations {0: man, 1: female}.
Once we have the representations of both the base and the concepts, we train a linear classifier on the concepts to compute the CAVs.
Finally we take the CAVs and take directional derivative to compute TCAVs to score if the concept was used in the base or not.


### Step # 01: Get the base activations

1. For this task, we compute the base activations of the sentences given in this file <a><href="https://github.com/DCSaunders/gender-debias/blob/master/data/handcrafted/handcrafted.ende"/>BASE REPRESENTATIONS</a>. However, this data contains sentences for both male and female pronouns. We take unique sentences and change his/her with [MASK] for the BERT baseline model to fill up with. Run ```pipeline/file_parser.py``` to get ```mask.txt``` which is then used to extract the base representations.

2. We extract the base representations from ```pipeline/extraction.py``` which can be run by typing the following command: ```python3 pipeline/extraction.py -model_name bert-base-uncased -input_corpus mask.txt -output_file extractions-professions-mask```. The activations for each layer (total 13 layers) will be stored in the output file specified.
