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

args = parser.parse_args()
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

import numpy as np
import os.path
from models_pytorch import *
from data import *
from preprocess import *
from utils import *

embeddings_index = load_word_embeddings(embedding_type, emb_dim)
vocab = {}
reverse_vocab = {}
vocab['PAD'] = 0
reverse_vocab[0] = 'PAD'
vocab['UNK'] = 1
reverse_vocab[1] = 'UNK'

if os.path.isfile('Data/Wikipedia/' + dataset + '/x_train' + file_extension + '.npy') and os.path.isfile('Data/Wikipedia/' + dataset +'/x_test' + file_extension + '.npy'):
	
	x_train = np.load('Data/Wikipedia/' + dataset +'/x_train' + file_extension + '.npy')
	x_test = np.load('Data/Wikipedia/' + dataset +'/x_test' + file_extension + '.npy')
	x_validate = np.load('Data/Wikipedia/' + dataset +'/x_validate' + file_extension + '.npy')
	y_train = np.load('Data/Wikipedia/' + dataset +'/y_train' + file_extension + '.npy')
	print(y_train.dtype)
	y_test = np.load('Data/Wikipedia/' + dataset +'/y_test' + file_extension + '.npy')
	y_validate = np.load('Data/Wikipedia/' + dataset +'/y_validate' + file_extension + '.npy')
	vocab = np.load('Data/Wikipedia/' + dataset +'/vocab' + file_extension + '.npy').item()
	reverse_vocab = np.load('Data/Wikipedia/' + dataset +'/reverse_vocab' + file_extension + '.npy').item()
	train_data_rev_ids = np.load('Data/Wikipedia/' + dataset +'/train_data_rev_ids' + file_extension + '.npy')
	test_data_rev_ids = np.load('Data/Wikipedia/' + dataset +'/test_data_rev_ids' + file_extension + '.npy')
	validation_data_rev_ids = np.load('Data/Wikipedia/' + dataset +'/validation_data_rev_ids' + file_extension + '.npy')
	print("loaded train features")
else:
	test_data_comments, test_data_rev_ids = get_data('test', dataset)
	#train_data_comments, train_data_rev_ids = get_data('train', dataset)
	#validation_data_comments, validation_data_rev_ids, test_data_comments, test_data_rev_ids = get_data_test_validation('test', dataset)
	train_data_comments, train_data_rev_ids, validation_data_comments, validation_data_rev_ids = get_train_validation_data(dataset)
	#train_data_comments, train_data_rev_ids = get_data('train', dataset)
	#validation_data_comments, validation_data_rev_ids = get_data('dev', dataset)
	np.save('Data/Wikipedia/' + dataset +'/train_data_rev_ids' + file_extension + '.npy', train_data_rev_ids)
	np.save('Data/Wikipedia/' + dataset +'/test_data_rev_ids' + file_extension + '.npy', test_data_rev_ids)
	np.save('Data/Wikipedia/' + dataset +'/validation_data_rev_ids' + file_extension + '.npy', validation_data_rev_ids)
	train_data_rev_ids = np.load('Data/Wikipedia/' + dataset +'/train_data_rev_ids' + file_extension + '.npy')
	test_data_rev_ids = np.load('Data/Wikipedia/' + dataset +'/test_data_rev_ids' + file_extension + '.npy')
	validation_data_rev_ids = np.load('Data/Wikipedia/' + dataset +'/validation_data_rev_ids' + file_extension + '.npy')
	#train_data_comments, train_data_rev_ids, validation_data_comments, validation_data_rev_ids = get_train_validation_data(dataset)
	rev_id_map = get_annotations(dataset)
	print("loaded data")

	y_train = np.array(generate_ylabels(train_data_rev_ids, rev_id_map))
	y_test = np.array(generate_ylabels(test_data_rev_ids, rev_id_map))
	y_validate = np.array(generate_ylabels(validation_data_rev_ids, rev_id_map))
	print("y labels generated")
	#np.save('Data/Wikipedia/' + dataset +'/train_data_rev_ids' + file_extension + '.npy', train_data_rev_ids)
        #np.save('Data/Wikipedia/' + dataset +'/test_data_rev_ids' + file_extension + '.npy', test_data_rev_ids)
        #np.save('Data/Wikipedia/' + dataset +'/validation_data_rev_ids' + file_extension + '.npy', validation_data_rev_ids)

        #train_data_rev_ids = np.load('Data/Wikipedia/' + dataset +'/train_data_rev_ids' + file_extension + '.npy')
        #test_data_rev_ids = np.load('Data/Wikipedia/' + dataset +'/test_data_rev_ids' + file_extension + '.npy')
        #validation_data_rev_ids = np.load('Data/Wikipedia/' + dataset +'/validation_data_rev_ids' + file_extension + '.npy')
	'''train_data_comments = train_data_comments[:1000]
	test_data_comments = test_data_comments[:500]
	validation_data_comments = validation_data_comments[:500]

	y_train = y_train[:1000]
	y_test = y_test[:500]
	y_validate = y_validate[:500]'''

	incorrect_to_correct = {}
	x_train, incorrect_to_correct = preprocess_text(train_data_comments, embeddings_index, incorrect_to_correct)
	print('x_train preprocessed')
	x_validate, incorrect_to_correct = preprocess_text(validation_data_comments, embeddings_index, incorrect_to_correct)
	print('x_validate preprocessed')
	x_test, incorrect_to_correct = preprocess_text(test_data_comments, embeddings_index, incorrect_to_correct)
	print('x_test preprocessed')
	print("comments preprocessed")
	print(len(x_test))
	print(len(x_train))
	print(len(x_validate))

	train_text = []
	for text in x_train:
	    train_text.append(text)
	rest_text = []
	for text in x_validate:
		rest_text.append(text)
	for text in x_test:
		rest_text.append(text)

	vocab, reverse_vocab = extend_vocab_dataset1(train_text, embeddings_index, vocab, reverse_vocab, True)
	vocab, reverse_vocab = extend_vocab_dataset1(rest_text, embeddings_index, vocab, reverse_vocab, False)
	print("vocab created")
	print('len vocab after extension')
	print(len(vocab))

	np.save('Data/Wikipedia/' + dataset +'/vocab' + file_extension + '.npy', vocab)
	np.save('Data/Wikipedia/' + dataset + '/reverse_vocab' + file_extension + '.npy', reverse_vocab)
	# Load
	vocab = np.load('Data/Wikipedia/' + dataset +'/vocab' + file_extension + '.npy').item()
	reverse_vocab = np.load('Data/Wikipedia/' + dataset + '/reverse_vocab' + file_extension + '.npy').item()

	print(len(vocab.keys()))

	x_train = np.array(generate_sequences(x_train, vocab))
	x_test = np.array(generate_sequences(x_test, vocab))
	x_validate = np.array(generate_sequences(x_validate, vocab))
	np.save('Data/Wikipedia/' + dataset +'/x_train' + file_extension + '.npy', x_train)
	np.save('Data/Wikipedia/' + dataset +'/x_test' + file_extension + '.npy', x_test)
	np.save('Data/Wikipedia/' + dataset +'/x_validate' + file_extension + '.npy', x_validate)
	print("saved comment sequences")
	x_train = np.load('Data/Wikipedia/' + dataset +'/x_train' + file_extension + '.npy')
	x_test = np.load('Data/Wikipedia/' + dataset +'/x_test' + file_extension + '.npy')
	x_validate = np.load('Data/Wikipedia/' + dataset +'/x_validate' + file_extension + '.npy')

	np.save('Data/Wikipedia/' + dataset +'/y_train' + file_extension + '.npy', y_train)
	np.save('Data/Wikipedia/' + dataset +'/y_test' + file_extension + '.npy', y_test)
	np.save('Data/Wikipedia/' + dataset +'/y_validate' + file_extension + '.npy', y_validate)

	y_train = np.load('Data/Wikipedia/' + dataset +'/y_train' + file_extension + '.npy')
	y_test = np.load('Data/Wikipedia/' + dataset +'/y_test' + file_extension + '.npy')
	y_validate = np.load('Data/Wikipedia/' + dataset +'/y_validate' + file_extension + '.npy')

	#np.save('Data/Wikipedia/' + dataset +'/train_data_rev_ids' + file_extension + '.npy', train_data_rev_ids)
        #np.save('Data/Wikipedia/' + dataset +'/test_data_rev_ids' + file_extension + '.npy', test_data_rev_ids)
        #np.save('Data/Wikipedia/' + dataset +'/validation_data_rev_ids' + file_extension + '.npy', validation_data_rev_ids)

        #train_data_rev_ids = np.load('Data/Wikipedia/' + dataset +'/train_data_rev_ids' + file_extension + '.npy')
        #test_data_rev_ids = np.load('Data/Wikipedia/' + dataset +'/test_data_rev_ids' + file_extension + '.npy')
        #validation_data_rev_ids = np.load('Data/Wikipedia/' + dataset +'/validation_data_rev_ids' + file_extension + '.npy')

#use pytorch's methods to do this
#https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch


x_train = generate_pad_sequences(x_train, sequence_length, 'pre')
x_test = generate_pad_sequences(x_test, sequence_length, 'pre')
x_validate = generate_pad_sequences(x_validate, sequence_length, 'pre')
print("padding done")

# TODO: do this in Pytorch; remove any Tensorflow dependency
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_validate = to_categorical(y_validate)
y_test = to_categorical(y_test)
embedding_weights = []

#print("got embedding weights")
if os.path.isfile('Data/Wikipedia/' + dataset +'/embedding_weights' + file_extension + '_' + embedding_type + '_nz.npy'):
	embedding_weights = np.load('Data/Wikipedia/' + dataset +'/embedding_weights' + file_extension + '_' + embedding_type + '_nz.npy')
else:
	embedding_weights = get_embedding_matrix(embedding_type, embeddings_index, vocab, file_extension, emb_dim)
	print("got embedding weights")
	print(embedding_weights.shape)
	np.save('Data/Wikipedia/' + dataset +'/embedding_weights' + file_extension + '_' + embedding_type + '_nz.npy', embedding_weights)
	embedding_weights = np.load('Data/Wikipedia/' + dataset +'/embedding_weights' + file_extension + '_' + embedding_type + '_nz.npy')
print('vocab size')
print(len(vocab))
print(len(embedding_weights))

def get_f1_score(p, r):
	return ((2.0*p*r)/(p+r))

micro_f1 = []
micro_pr = []
micro_r = []
macro_pr = []
macro_r = []
macro_f1 = []
for i in range(5):
	print('Iteration: ', i)
	p,r,f1,m_p,m_r,m_f1 = build_and_train_network(x_train, y_train, train_data_rev_ids, x_validate, y_validate, 
		validation_data_rev_ids, x_test, y_test, test_data_rev_ids, 
		vocab, embedding_weights, sequence_length, emb_dim, hidden_dim, lr_rate, model, 
		epochs, lstm_sizes, batch_size, dropout_prob)