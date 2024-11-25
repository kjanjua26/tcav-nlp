# tcav-nlp
Extension of Testing with Concept Activation Vectors (TCAV) for NLP.

## Running the Pipeline
#### Location: code/pipeline

The pipeline is divided into several steps. However, first we compute the base representations of all the sentences. 
Then we compute representations of sentences containing genders (he/she/woman/man, etc) and label those representations {0: man, 1: female}.
Once we have the representations of both the base and the concepts, we train a linear classifier on the concepts to compute the CAVs.
Finally we take the CAVs and take directional derivative to compute TCAVs to score if the concept was used in the base or not.

### Step # 01: Get the base activations

1. For this task, we compute the base activations of the sentences given in this file <a href="https://github.com/DCSaunders/gender-debias/blob/master/data/handcrafted/handcrafted.ende">BASE REPRESENTATIONS</a>. However, this data contains sentences for both male and female pronouns. We take unique sentences and change his/her with [MASK] for the BERT baseline model to fill up with. Run ```code/pipeline/file_parser.py``` to get ```mask.txt``` which is then used to extract the base representations.

2. We extract the base representations from ```code/pipeline/extraction.py``` which can be run by typing the following command: ```python3 code/pipeline/extraction.py -m bert-base-uncased -i mask.txt -o extractions-professions-mask.json```. The activations for each layer (total 13 layers) will be stored in the output file specified.

### Step # 02: Get the concept activations

1. For this task, we take the concepts from IMDB Movie Reviews dataset which can be downloaded from Kaggle. Once that data is downloaded, type the following command to prepare the IMDB dataset in the form of concepts (one sentence per line) which can then be used to get the activations from BERT: ```python3 code/pipeline/prepare_concepts.py -i IMDB\ Dataset.csv -o gendered_concepts.txt -c "gender" -f gender_f.txt gender_m.txt```. These gender files are taken from the wordlist link: <a href="http://modelai.gettysburg.edu/2020/weat/student_materials/wordlists/">WORDLISTS</a>. 

2. The output of the first step will be a .txt file which will be used to pass to the ```code/pipeline/extraction.py``` to get the activations (acts) from. To get those acts for the concepts, run the following command: ```python3 code/pipeline/extraction.py -m bert-base-uncased -i gendered_concepts.txt -o gendered-concept-acts.json```. This will run, takes some time, and make the .json file with the layer-wise-token-wise activations.
