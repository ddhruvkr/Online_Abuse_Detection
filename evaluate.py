import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_type', '-embedding', help="select the pretrained embedding", type= str)
parser.add_argument('--sequence_length', '-sequence', help="select the word sequence length to the model", type= int)
parser.add_argument('--lr_rate', '-lr', help="-learning rate", type= float)
parser.add_argument('--optimizer', '-optimizer', help="select the optimzer", type= str)
parser.add_argument('--file_extension', '-file_extension', help="select the npy file_extension", type= str)
parser.add_argument('--dataset', '-dataset', help="select the Wiki dataset", type= str)
parser.add_argument('--emb_dim', '-emb_dim', help="embedding dim", type=int)
parser.add_argument('--hidden_dim', '-hidden_dim', help="hidden dimension", type=int)
parser.add_argument('--model', '-model', help="model name", type=str)
parser.add_argument('--epochs', '-epochs', help="epochs", type=int)
parser.add_argument('--lstm_size', '-lstm_size', help="lstm size", type=int)
parser.add_argument('--batch_size', '-batch_size', help="batch size", type=int)
parser.add_argument('--dropout_prob', '-dropout_prob', help="dropout prob", type=float)
#print(parser.format_help())
# usage: test_args_4.py [-h] [--foo FOO] [--bar BAR]
# 
# optional arguments:
#   -h, --help         show this help message and exit
#   --foo FOO, -f FOO  a random options
#   --bar BAR, -b BAR  a more random option

args = parser.parse_args()
#print(args)  # Namespace(bar=0, foo='pouet')
embedding_type = args.embedding_type
sequence_length = args.sequence_length
lr_rate = args.lr_rate
optimizer = args.optimizer
file_extension = args.file_extension
dataset =args.dataset
emb_dim = args.emb_dim
hidden_dim= args.hidden_dim
model = args.model
epochs =args.epochs
lstm_sizes = args.lstm_size
batch_size= args.batch_size
dropout_prob = args.dropout_prob
print(sequence_length)
#embeddings_index = load_word_embeddings1(embedding_type, emb_dim)
import numpy as np
import os.path
from models_pytorch import *
from data import *
from preprocess import *

#embeddings_index = load_word_embeddings1(embedding_type, emb_dim)
vocab = np.load('Data/Wikipedia/' + dataset +'/vocab' + file_extension + '.npy').item()
reverse_vocab = np.load('Data/Wikipedia/' + dataset +'/reverse_vocab' + file_extension + '.npy').item()


embedding_weights = np.load('Data/Wikipedia/' + dataset +'/embedding_weights' + file_extension + '_' + embedding_type + '_nz.npy')


x = ['NEWLINE_TOKEN== THIS SUCKS!   NEWLINE_TOKENNEWLINE_TOKEN== NEWLINE_TOKENNEWLINE_TOKENI JUST GOT BLOCKED INDEFINITLY']
print(x)
incorrect_to_correct = {}
#x, incorrect_to_correct = preprocess_text(x, embeddings_index, incorrect_to_correct)
print('x_train preprocessed')
print(x)
x = [['stop', 'deleting', 'my', 'comments', 'you', 'coward'], ['shut', 'the', 'fuck', 'up', 'you', 'stupid', 'indian'], ['i', 'like', 'you', 'but', 'you', 'are', 'stupid']]
print(x)
x = np.array(generate_sequences(x, vocab))
print(x)
x = generate_pad_sequences(x, sequence_length)
#print(x)
y = [0,1, 0]
y = np.array(y)
from keras.utils import to_categorical
y = to_categorical(y)
print(y)
z = ['123', '13', '9']
p,r,f1,m_p,m_r,m_f1 = test(x,y,z,
		vocab, embedding_weights, sequence_length, emb_dim, hidden_dim, lr_rate, model, 
		epochs, lstm_sizes, batch_size, dropout_prob)
