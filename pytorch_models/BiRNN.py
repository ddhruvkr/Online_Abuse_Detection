import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from torch.nn import functional as f

class BiRNN(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout_prob, embedding_weights):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional = True, dropout=dropout_prob)
		self.fc = nn.Linear(hidden_dim*2, output_dim)
		self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, output_dim)
		self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_weights).float(), requires_grad=True)
		self.hidden_dim = hidden_dim
		self.dropout = nn.Dropout(dropout_prob)
		self.avg_pool = False
	def forward(self, inp):
		embeddings = self.embedding(inp)
		#batch_size, max_sequence_length, embedding_length
		embeddings = embeddings.permute(1, 0, 2)
		output, hidden = self.rnn(embeddings)
		#output = [max_sequence_length, batch_size, hidden_dim * num_directions(bi-lstm)]
		x = output.permute(1,2,0)
		#batch_size, 2*hidden_dim, max_sequence_length
		if self.avg_pool == True:
			x = f.relu(f.avg_pool1d(x, x.size(2)))
		else:
			x = f.relu(f.max_pool1d(x, x.size(2)))
		#(batch_size, 2*hidden_dim, 1)
		x = x.squeeze(2)
		x = self.dropout(x)
		linear=self.fc(x)
		return linear
