Code for the paper "Online abuse detection: the value of preprocessing and neural attention models", in NAACL workshop on Computational Approaches to Subjectivity, Sentiment & Social Media Analysis (WASSA), Jun 2019

UPDATE:

Results when using Glove+Elmo embeddings with the baseline BiRNN model. As expected, these improve the performance when compared to using just Glove embeddings and give higher scores than what are reported in the paper. Using only Elmo embeddings performed worse. Embeddings were not tuned. All other parameters were kept as same. Thus, suggest using these embeddings for the future.

Dataset | Embedding | Minority P | Minority R | Minority F1 | Overall F1 
--- | --- | --- | --- |--- |--- 
W-Tox | Glove+Elmo(1024) | 83.55 | 81.31 | 82.41 | 90.29
W-Tox | Glove+Elmo(256) | 83.84 | 79.73 | 81.73 | 89.94
W-At | Glove+Elmo(1024) | 83.13 | 77.75 | 80.35 | 88.93
W-At | Glove+Elmo(256) | 84.02 | 75.84 | 79.72 | 88.64
W-Ag | Glove+Elmo(1024) | 82.7 | 76.07 | 79.25 | 88.21
W-Ag | Glove+Elmo(256) | 82.82 | 74.92 | 78.67 | 87.91

Put the tsv dataset files inside Data/Wikipedia/toxicity (attack, aggression)

Also create a folder structure Embeddings/Glove/ outside this folder to have the embeddings file. 

The code can be run with the following commands when using only the Glove embeddings. Replace main.py with main_elmo.py when using Glove+Elmo embeddings.

```
python main.py -embedding glove -dataset toxicity -emb_dim=300 -hidden_dim=150 -model CoAttn -lr=0.001 -epochs=3 -lstm_size=1 -batch_size=256 -sequence 175 -dropout_prob=0.1 -file_extension demo

python main.py -embedding glove -dataset attack -emb_dim=300 -hidden_dim=150 -model CoAttn -lr=0.001 -epochs=3 -lstm_size=1 -batch_size=200 -sequence 175 -dropout_prob=0.1 -file_extension demo

python main.py -embedding glove -dataset aggression -emb_dim=300 -hidden_dim=150 -model Attn -lr=0.001 -epochs=3 -lstm_size=1 -batch_size=200 -sequence 175 -dropout_prob=0.1 -file_extension demo
```

Requirements:

> Pytorch 1.0

> Keras (Tensorflow 1.10, Only using it for padding, will remove this dependency)

> Ekphrasis

> mosestokenizer

> sklearn

> nltk

TODO:
1) Still using Keras functions for padding. Replace them with Pytorch function.
2) Possibly try BERT (should perform even better).
