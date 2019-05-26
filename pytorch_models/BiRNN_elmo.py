import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from torch.nn import functional as f
from allennlp.modules.elmo import Elmo

class BiRNN(nn.Module):
	def __init__(self, vocab_size, embedding_dim_glove, hidden_dim, output_dim, n_layers, dropout_prob, embedding_weights):
		super().__init__()
		embedding_dim_elmo=1024
		embedding_dim = embedding_dim_glove + embedding_dim_elmo
		#self.options_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json')
		#self.weight_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')
		#self.options_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json')
		#self.weight_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5')
		self.options_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json')
		self.weight_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5')
		self.elmo = Elmo(self.options_file, self.weight_file, 1,requires_grad=False, dropout=0.5)
		self.embedding = nn.Embedding(vocab_size, embedding_dim_glove)
		self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional = True, dropout=dropout_prob)
		self.fc = nn.Linear(hidden_dim*2, output_dim)
		self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_weights).float(), requires_grad=False)
		self.hidden_dim = hidden_dim
		self.dropout = nn.Dropout(dropout_prob)
		self.avg_pool = False
	def forward(self, inp, inp_glove):
		
		embeddings_elmo = self.elmo(inp)
		embeddings_elmo = embeddings_elmo['elmo_representations'][0]
		#output: bts, max_sequence_length, embedding_length
		embeddings_glove = self.embedding(inp_glove)
		#batch_size, max_sequence_length, embedding_length
		embeddings = torch.cat((embeddings_glove, embeddings_elmo), dim=2)
		embeddings = embeddings.permute(1, 0, 2)
		output, hidden = self.rnn(embeddings)
		x = output.permute(1,2,0)
		#batch_size, 2*hidden_dim, max_sequence_length
		if self.avg_pool == True:
			x = f.relu(f.avg_pool1d(x, x.size(2)))
		else:
			x = f.relu(f.max_pool1d(x, x.size(2)))
		#(batch_size, 2*hidden_dim, 1)
		x = x.squeeze(2)
		x = self.dropout(x)
		#output = torch.cat((output[-1, :, :self.hidden_dim], output[0, :, self.hidden_dim:]), dim=1)
		linear=self.fc(x)
		return linear
