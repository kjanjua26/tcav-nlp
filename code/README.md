## Pipeline

The pipeline is divided into several steps. However, first we compute the base representations of all the sentences. 
Then we compute representations of sentences containing genders (he/she/woman/man, etc) and label those representations {0: man, 1: female}.
Once we have the representations of both the base and the concepts, we train a linear classifier on the concepts to compute the CAVs.
Finally we take the CAVs and take directional derivative to compute TCAVs to score if the concept was used in the base or not.
