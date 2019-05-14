Code for the paper "Online abuse detection: the value of preprocessing and neural attention models", in NAACL workshop on Computational Approaches to Subjectivity, Sentiment & Social Media Analysis (WASSA), Jun 2019

UPDATE:
Results when using Glove+Elmo embeddings with the BiRNN model. Just glove performed worse than this. Did not tune the embeddings. All other parameters were kept as same.

W-Tox:

minority class

p, r, f

83.55 81.31 82.41

overall f1 score

90.26

W-At

minority class

p, r, f1

83.13 77.75 80.35

overall f1 score

88.88

W-Ag

minority class

p, r, f1

82.7 76.07 79.25

overall f1 score

88.12


Put the tsv dataset files inside Data/Wikipedia/
Also create a folder structure Embeddings/Glove/ outside this folder to have the embeddings file and 

The code can be run with the following command
python main.py -embedding glove -dataset toxicity -emb_dim=300 -hidden_dim=150 -model CoAttn -lr=0.001 -epochs=2 -lstm_size=1 -batch_size=256 -sequence 175 -dropout_prob=0.05 -file_extension demo


The dataset parameter can have values 'toxicity', 'attack' and 'aggression' representing different datasets.
The model parameter can have values 'BiRNN', 'Attn', 'CoAttn' representing different models.

TODO:
1) Still using Keras functions for padding. Replace them with Pytorch
