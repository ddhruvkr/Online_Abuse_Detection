import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from torch.nn import functional as f

class IntraAttention(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout_prob, embedding_weights, is_cuda):
		super().__init__()
		self.par = 4
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.bilstms = nn.GRU(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional = True, dropout=dropout_prob)
		self.Wq = nn.Linear(2*embedding_dim, self.par)
		self.Wp = nn.Linear(self.par, 1)
		self.Wmy = nn.Linear(128, 1, bias=False)
		self.w = nn.Linear(2*embedding_dim, 1)
		self.sq = nn.Linear(embedding_dim, embedding_dim, bias=False)
		self.wCo = nn.Linear(3*embedding_dim, 1)
		self.fc = nn.Linear(2*hidden_dim+embedding_dim, output_dim)
		self.fc1 = nn.Linear(2*hidden_dim, output_dim)
		self.logits = nn.Linear(hidden_dim, output_dim)
		self.dropout = nn.Dropout(dropout_prob)
		self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_weights).float(), requires_grad=True)
		self.hidden_dim = hidden_dim
		self.is_cuda = is_cuda

	def forward(self, input_sequences):
		#batch_size, max_sequence_length
		embeddings = self.embedding(input_sequences)
		#batch_size, max_sequence_length, embedding_length
		s = self.bilinear_layer(embeddings)
		#s = self.dot_product(embeddings)
		#s = self.singular_intra_attention_layer(embeddings)
		#s = self.multi_dimensional_intra_attention_layer(embeddings)
		embeddings = embeddings.permute(1, 0, 2)
		#max_sequence_length, batch_size, embedding_length
		output, hidden = self.bilstms(embeddings)
		#output = [max_sequence_length, batch_size, hidden_dim * num_directions(bi-lstm)]
		x = torch.cat((output[-1, :, :self.hidden_dim], output[0, :, self.hidden_dim:]), dim=1)
		r = torch.cat((s,x), dim=1)
		fc = self.fc(r)
		return fc

	def bilinear_layer(self, embeddings):
		max_sequence_length = embeddings.shape[1]
		mask = torch.ones(max_sequence_length,max_sequence_length)
		if self.is_cuda:
			mask = mask.cuda()
		mask = mask - torch.diag(torch.diag(mask))
		#s = torch.bmm(embeddings, embeddings.permute(0,2,1))
		s = self.sq(embeddings)
		s = torch.bmm(s, embeddings.permute(0,2,1))
		#(bts, max_sequence_length, max_sequence_length)
		s = s*mask
		#s1 = s.cpu().numpy()
		#np.savetxt('foo.csv', s1[0], delimiter=",", fmt="%.1e")
		#np.savetxt('foo1.csv', s1[1], delimiter=",", fmt="%.1e")
		#doing masking to make values where word pairs are same(i == j), zero
		s = f.avg_pool1d(s, s.size()[2]).squeeze(2)
		#(bts, max_sequence_length)
		s = f.softmax(s, dim=1)
		s = s.unsqueeze(dim=1)
		#(bts, 1, max_sequence_length)
		s = torch.bmm(s, embeddings)
		#(bts, 1, max_sequence_length)(bts, max_sequence_length, embedding_length) = (bts, 1, embedding_length)
		s = s.squeeze(1)
		return s

	def simplified_intra_attention_layer(self, embeddings):
		max_sequence_length = embeddings.shape[1]
		mask = torch.ones(max_sequence_length,max_sequence_length)
		if self.is_cuda:
			mask = mask.cuda()
		mask = mask - torch.diag(torch.diag(mask))
		e = embeddings.permute(0,2,1)
		s = torch.bmm(embeddings, e)
		#(bts, max_sequence_length, max_sequence_length)
		s = s*mask
		#doing masking to make values where word pairs are same(i == j), zero
		s = f.avg_pool1d(s, s.size()[2]).squeeze(2)
		#(bts, max_sequence_length)
		s = f.softmax(s, dim=1)
		s = s.unsqueeze(dim=1)
		#(bts, 1, max_sequence_length)
		s = torch.bmm(s, embeddings)
		#(bts, 1, max_sequence_length)(bts, max_sequence_length, embedding_length) = (bts, 1, embedding_length)
		s = s.squeeze(1)
		return s

	def dot_product(self, embeddings):
		max_sequence_length = embeddings.shape[1]
		mask = torch.ones(max_sequence_length,max_sequence_length)
		if self.is_cuda:
			mask = mask.cuda()
		mask = mask - torch.diag(torch.diag(mask))
		s = torch.bmm(embeddings, embeddings.permute(0,2,1))
		#(bts, max_sequence_length, max_sequence_length)
		s = s*mask
		#doing masking to make values where word pairs are same(i == j), zero
		# do dot product instead of pooling
		s = self.Wmy(s)
		s = s.squeeze(2)
		#(bts, max_sequence_length)
		s = f.softmax(s, dim=1)
		s = s.unsqueeze(dim=1)
		#(bts, 1, max_sequence_length)
		s = torch.bmm(s, embeddings)
		#(bts, 1, max_sequence_length)(bts, max_sequence_length, embedding_length) = (bts, 1, embedding_length)
		s = s.squeeze(1)
		return s

	def singular_intra_attention_layer(self, embeddings):
		max_sequence_length = embeddings.shape[1]
		batch_size = embeddings.shape[0]
		embedding_dim = embeddings.shape[2]

		#(bts, msl, dim)
		b = embeddings
		d = embeddings
		b = b.repeat(1,1,max_sequence_length)
		#(bts, max_sequence_length, max_sequence_length*embedding_dim)
		b = b.view(batch_size, max_sequence_length, max_sequence_length, embedding_dim)
		d = d.unsqueeze(1)
		#(bts, 1, max_sequence_length, embedding_dim)
		d = d.repeat(1,max_sequence_length,1,1)
		#(batch_size, max_sequence_length, max_sequence_length, embedding_dim)
		concat = torch.cat((b,d), dim=3)
		#batch_size, max_sequence_length, max_sequence_length, 2*embedding_dim

		'''Lets assume that there are 2 words and each word embedding has 3 dimension
		So embedding matrix would look like this
		[[----1----]
		[----2----]].  (2*3)

		b =
		----1----,----1----
		----2----,----2----


		d=
		----1----,----2----
		----1----,----2----

		Now if you concatenate both, we get all the combinations of word pairs'''


		s = self.dropout(self.w(concat))
		#batch_size, max_sequence_length, max_sequence_length, 1
		s = s.squeeze(3)
		#batch_size, max_sequence_length, max_sequence_length
		mask = torch.ones(max_sequence_length,max_sequence_length)
		if self.is_cuda:
			mask = mask.cuda()
		mask = mask - torch.diag(torch.diag(mask))
		#(bts, max_sequence_length, max_sequence_length)
		s = s*mask
		s = f.max_pool1d(s, s.size()[2]).squeeze(2)
		#(bts, max_sequence_length)
		s = f.softmax(s, dim=1)
		s = s.unsqueeze(dim=1)
		#(bts, 1, max_sequence_length)
		s = self.dropout(torch.bmm(s, embeddings))
		#(bts, 1, max_sequence_length)(bts, max_sequence_length, embedding_length) = (bts, 1, embedding_length)
		s = s.squeeze(1)
		return s

	def co_attention_layer(self, embeddings):
		max_sequence_length = embeddings.shape[1]
		batch_size = embeddings.shape[0]
		embedding_dim = embeddings.shape[2]

		#(bts, msl, dim)
		b = embeddings
		d = embeddings
		b = b.repeat(1,1,max_sequence_length)
		#(bts, max_sequence_length, max_sequence_length*embedding_dim)
		b = b.view(batch_size, max_sequence_length, max_sequence_length, embedding_dim)
		d = d.unsqueeze(1)
		#(bts, 1, max_sequence_length, embedding_dim)
		d = d.repeat(1,max_sequence_length,1,1)
		#(batch_size, max_sequence_length, max_sequence_length, embedding_dim)
		bd = b*d
		concat = torch.cat((b,d,bd), dim=3)
		#batch_size, max_sequence_length, max_sequence_length, 3*embedding_dim

		'''Lets assume that there are 2 words and each word embedding has 3 dimension
		So embedding matrix would look like this
		[[----1----]
		[----2----]].  (2*3)

		b =
		----1----,----1----
		----2----,----2----


		d=
		----1----,----2----
		----1----,----2----

		Now if you concatenate both, we get all the combinations of word pairs'''


		s = self.wCo(concat)
		#batch_size, max_sequence_length, max_sequence_length, 1
		s = s.squeeze(3)
		#batch_size, max_sequence_length, max_sequence_length
		mask = torch.ones(max_sequence_length,max_sequence_length)
		if self.is_cuda:
			mask = mask.cuda()
		mask = mask - torch.diag(torch.diag(mask))
		#(bts, max_sequence_length, max_sequence_length)
		s = s*mask
		#s = f.max_pool1d(s, s.size()[2]).squeeze(2)
		s = self.Wmy(s)
		s = s.squeeze(2)
		#(bts, max_sequence_length)
		s = f.softmax(s, dim=1)
		s = s.unsqueeze(dim=1)
		#(bts, 1, max_sequence_length)
		s = torch.bmm(s, embeddings)
		#(bts, 1, max_sequence_length)(bts, max_sequence_length, embedding_length) = (bts, 1, embedding_length)
		s = s.squeeze(1)
		return s

	def multi_dimensional_intra_attention_layer(self, embeddings):
		max_sequence_length = embeddings.shape[1]
		batch_size = embeddings.shape[0]
		embedding_dim = embeddings.shape[2]

		#(bts, msl, dim)
		b = embeddings
		d = embeddings
		b = b.repeat(1,1,max_sequence_length)
		#(bts, max_sequence_length, max_sequence_length*embedding_dim)
		b = b.view(batch_size, max_sequence_length, max_sequence_length, embedding_dim)
		d = d.unsqueeze(1)
		#(bts, 1, max_sequence_length, embedding_dim)
		d = d.repeat(1,max_sequence_length,1,1)
		#(batch_size, max_sequence_length, max_sequence_length, embedding_dim)
		concat = torch.cat((b,d), dim=3)
		#batch_size, max_sequence_length, max_sequence_length, 2*embedding_dim

		'''Lets assume that there are 2 words and each word embedding has 3 dimension
		So embedding matrix would look like this
		[[----1----]
		[----2----]].  (2*3)

		b =
		----1----,----1----
		----2----,----2----


		d=
		----1----,----2----
		----1----,----2----

		Now if you concatenate both, we get all the combinations of word pairs

		----1--------1----,----1--------2----
		----2--------1----,----2--------2----

		'''

		s = self.Wq(concat)
		#batch_size, max_sequence_length, max_sequence_length, par
		s = f.relu(s)
		s = self.Wp(s)
		#batch_size, max_sequence_length, max_sequence_length, 1
		s = s.squeeze(3)
		#batch_size, max_sequence_length, max_sequence_length
		mask = torch.ones(max_sequence_length,max_sequence_length)
		if self.is_cuda:
			mask = mask.cuda()
		mask = mask - torch.diag(torch.diag(mask))
		#s = torch.bmm(embeddings, embeddings.permute(0,2,1))
		#(bts, max_sequence_length, max_sequence_length)
		s = s*mask
		s = f.max_pool1d(s, s.size()[2]).squeeze(2)
		#(bts, max_sequence_length)
		s = f.softmax(s, dim=1)
		s = s.unsqueeze(dim=1)
		#(bts, 1, max_sequence_length)
		s = torch.bmm(s, embeddings)
		#(bts, 1, max_sequence_length)(bts, max_sequence_length, embedding_length) = (bts, 1, embedding_length)
		s = s.squeeze(1)
		return s

'''
a
tensor([[0.1000, 0.2000, 0.3000],
        [0.4000, 0.5000, 0.6000]], dtype=torch.float64)

b = a
>>> b.shape
torch.Size([2, 3])
>>> b = b.repeat(1,2)
>>> b
tensor([[0.1000, 0.2000, 0.3000, 0.1000, 0.2000, 0.3000],
        [0.4000, 0.5000, 0.6000, 0.4000, 0.5000, 0.6000]], dtype=torch.float64)
>>> b = b.view(2,2,3)
>>> b
tensor([[[0.1000, 0.2000, 0.3000],
         [0.1000, 0.2000, 0.3000]],

        [[0.4000, 0.5000, 0.6000],
         [0.4000, 0.5000, 0.6000]]], dtype=torch.float64)



d.shape
torch.Size([2, 3])
>>> d = d.unsqueeze(0)
tensor([[[0.1000, 0.2000, 0.3000],
         [0.4000, 0.5000, 0.6000]]], dtype=torch.float64)
>>> d.shape
torch.Size([1, 2, 3])
>>> d=d.repeat(2,1,1)
>>> d
tensor([[[0.1000, 0.2000, 0.3000],
         [0.4000, 0.5000, 0.6000]],

        [[0.1000, 0.2000, 0.3000],
         [0.4000, 0.5000, 0.6000]]], dtype=torch.float64)
>>> d.shape
torch.Size([2, 2, 3])


>>> torch.cat((b,d),dim=2)
tensor([[[0.1000, 0.2000, 0.3000, 0.1000, 0.2000, 0.3000],
         [0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000]],

        [[0.4000, 0.5000, 0.6000, 0.1000, 0.2000, 0.3000],
         [0.4000, 0.5000, 0.6000, 0.4000, 0.5000, 0.6000]]],
       dtype=torch.float64)
>>> concat = torch.cat((b,d),dim=2)
>>> concat.shape
torch.Size([2, 2, 6])
'''
