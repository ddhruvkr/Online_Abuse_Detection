import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.utils import data
import numpy as np
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score,precision_score, classification_report, precision_recall_fscore_support
from torch.nn import functional as f
from pytorch_models.BiRNN import BiRNN
from pytorch_models.Attn import WordAttention
from pytorch_models.CoAttn import IntraAttention

class Dataset(data.Dataset):
	def __init__(self, x_train, y_train, x_train_rev_ids, is_cuda):
		if is_cuda:
			self.x_train = torch.from_numpy(x_train).cuda()
			self.y_train = (torch.from_numpy(y_train)).double().cuda()
			self.x_train_rev_ids = x_train_rev_ids
		else:
			self.x_train = torch.from_numpy(x_train)
			self.y_train = (torch.from_numpy(y_train)).double()
			self.x_train_rev_ids = x_train_rev_ids

	def __len__(self):
		return len(self.x_train)

	def __getitem__(self, index):
		x = self.x_train[index]
		y = self.y_train[index]
		z = self.x_train_rev_ids[index]

		return x, y, z

def load_data(dataset, batch_size):
	dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
	return dataloader

def binary_accuracy(preds, y):
	#Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
	rounded_preds = torch.round(preds)
	rounded_preds = torch.narrow(rounded_preds, 1, 0, 1).squeeze(1)
	y = torch.narrow(y, 1, 0, 1).squeeze(1)
	correct = torch.sum (rounded_preds == y).float()
	acc = correct.item()/len(y)
	return acc

def calculate_precision_recall_f1(preds, y, z, model_name):
	rounded_preds = torch.round(preds)
	rounded_preds = torch.narrow(rounded_preds, 1, 0, 1).squeeze(1)
	y = torch.narrow(y, 1, 0, 1).squeeze(1)

	rounded_preds = rounded_preds.detach().numpy()
	y = y.numpy()
	save_rev_ids_for_predictions(rounded_preds, y, z, model_name)
	#print('rev ids stored')
	r = get_recall_for_minority(rounded_preds, y)*100.0
	p = get_precision_for_minority(rounded_preds, y)*100.0
	f = get_f1(p,r)
	print('minority scores', end=" ")
	print(round(p,2), end=" ")
	print(round(r,2), end=" ")
	print(round(f,2))
	#print("classification_report")
	#print(classification_report(y, rounded_preds, digits=5))
	#print('precision_recall_fscore_support')
	#print(precision_recall_fscore_support(y, rounded_preds))
	p1 = precision_score(y, rounded_preds, average='macro')*100.0
	#print("macro precision_score")
	r1 = recall_score(y, rounded_preds, average='macro')*100.0
	f1 = f1_score(y, rounded_preds, average='macro')*100.0
	print("macro scores", end=" ")
	print(round(p1,2), end=" ")
	print(round(r1,2), end=" ")
	print(round(f1,2))
	return r,p,f,r1,p1,f1

def save_rev_ids_for_predictions(y_pred, y_test, z, model_name):
	correct_minority = []
	incorrect_minority = []
	total = 0
	correct = 0
	incorrect = 0
	for i in range(0, len(y_test)):
		# if minority class in reality
		if y_test[i] == 0:
			total += 1
			# if correctly predicted minority class
			if y_pred[i] == y_test[i]:
				correct += 1
				correct_minority.append(z[i])
			else:
				incorrect += 1
				incorrect_minority.append(z[i])
	# save it in a file
	with open('correct_' + model_name +'.txt', 'w') as f:
		for item in correct_minority:
			item = item.encode('utf8')
			f.write("%s\n" % item)
	f.close()
	with open('incorrect_' + model_name +'.txt', 'w') as f:
		for item in incorrect_minority:
			item = item.encode('utf8')
			f.write("%s\n" % item)
	f.close()
	return

def get_f1(p, r):
	return ((2*p*r)/(p+r))
def get_recall_for_minority(y_pred, y_test):
	correct = 0
	total = 0
	for i in range(0, len(y_test)):
		if y_test[i] == 0:
			total += 1
			if y_pred[i] == y_test[i]:
				correct += 1
	return (correct/total)

def get_precision_for_minority(y_pred, y_test):
	correct = 0
	total = 0
	for i in range(0, len(y_test)):
		if y_pred[i] == 0:
			total += 1
			if y_pred[i] == y_test[i]:
				correct += 1
	return (correct/total)

def train(model, iterator, optimizer, criterion, epoch_no):
	epoch_loss = 0
	epoch_acc = 0
	model.train()
	i = 0
	max_val = 200
	total = len(iterator)
	for x,y,z in iterator:
		i += 1
		optimizer.zero_grad()
		predictions= model(x).squeeze(1)
		predictions = predictions.double()
		loss = criterion(predictions, y)
		predictions = torch.sigmoid(predictions)
		acc = binary_accuracy(predictions.cpu(), y.cpu())
		#p, r, f1 = calculate_precision_recall_f1(predictions, y)
		loss.backward()
		#clip_gradient(model, 5e-1)
		optimizer.step()
		epoch_loss += loss.item()
		epoch_acc += acc
		if i % max_val == 0:
			print(f'| Epoch: {epoch_no+1} | Iter: {i} of {total} Train Loss: {(epoch_loss/i):.3f} | Train Acc: {epoch_acc/i*100:.2f}%')

	print('train done')
	#calculate_precision_recall_f1(p_concat, y_concat)
	return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, model_name, from_test):
    
	epoch_loss = 0
	epoch_acc = 0
	model.eval()
	with torch.no_grad():
		i = 0
		for x,y,z in iterator:
			i += 1
			predictions = model(x).squeeze(1)
			predictions = predictions.double()
			if from_test:
				predictions = torch.sigmoid(predictions)
				print(predictions)
				return 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
			loss = criterion(predictions, y)
			predictions = torch.sigmoid(predictions)
			if i > 1:
				p_concat = torch.cat((p_concat, predictions), 0)
				y_concat = torch.cat((y_concat, y), 0)
				z_concat = z_concat + z
			else:
				p_concat = predictions
				y_concat = y
				z_concat = z
			acc = binary_accuracy(predictions.cpu(), y.cpu())
			epoch_loss += loss.item()
			epoch_acc += acc
	r,p,f,r1,p1,f1 = calculate_precision_recall_f1(p_concat.cpu(), y_concat.cpu(), z_concat, model_name)
	return epoch_loss / len(iterator), epoch_acc / len(iterator), r,p,f,r1,p1,f1

def clip_gradient(model, clip_value):
	params = list(filter(lambda p: p.grad is not None, model.parameters()))
	for p in params:
		p.grad.data.clamp_(-clip_value, clip_value)

def build_and_train_network(x_train, y_train, x_train_rev_ids, x_validation, y_validation, 
	x_validate_rev_ids, x_test, y_test, x_test_rev_ids, vocab, embedding_weights, 
	word_sequence_length, emb_dim, hidden_dim, lr,  model, epochs, lstm_sizes, batch_size, 
	dropout_prob):
	model_name = model
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(device)
	is_cuda = False
	if device.type == 'cuda':
		is_cuda = True
	if model == 'CoAttn':
		model = IntraAttention(len(vocab), emb_dim, hidden_dim, 2, lstm_sizes, dropout_prob, embedding_weights, is_cuda)
	if model == 'BiRNN':
		model = BiRNN(len(vocab), emb_dim, hidden_dim, 2, lstm_sizes, dropout_prob, embedding_weights)
	if model == 'RNN':
		model = RNN(len(vocab), emb_dim, hidden_dim, 2, lstm_sizes, dropout_prob, embedding_weights)
	if model == 'Attn':
		model = WordAttention(len(vocab), emb_dim, hidden_dim, 2, lstm_sizes, dropout_prob, embedding_weights)

	training_set = Dataset(x_train.astype(np.int64), y_train, x_train_rev_ids, is_cuda)
	train_iterator = load_data(training_set, batch_size)
	validation_set = Dataset(x_validation.astype(np.int64), y_validation, x_validate_rev_ids, is_cuda)
	validation_iterator = load_data(validation_set, batch_size)
	test_set = Dataset(x_test.astype(np.int64), y_test, x_test_rev_ids, is_cuda)
	test_iterator = load_data(test_set, batch_size)
	optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.999), eps=1e-08, weight_decay=1e-6)
	'''weights = [3.0, 1.0]
	weights = torch.DoubleTensor(weights)
	if is_cuda:
		weights = weights.cuda()'''
	criterion = nn.BCEWithLogitsLoss()
	try:
		model = model.to(device)
	except:
		print('trying 2nd time')
		model = model.to(device)
	criterion = criterion.to(device)
	validloss = 1000.0
	validf1 = 0.0
	validmf1 = 0.0
	for epoch in range(epochs):
		train_loss, train_acc = train(model, train_iterator, optimizer, criterion, epoch)
		print("validation accuracy")
		valid_loss, valid_acc, val_r, val_p, val_f1, val_macro_r, val_macro_p, val_macro_f1 = evaluate(model, validation_iterator, criterion, model_name, False)
		#new_avg_f1 = (train_f1 + test_f1)/2.0
		#since the validation set is very small compared to the test set just using the validation loss is unstable
		if validloss >= valid_loss or (validf1 <= val_f1 and validmf1 <= val_macro_f1):
			validloss = valid_loss
			validf1 = val_f1
			validmf1 = val_macro_f1
			torch.save(model.state_dict(), 'model.pt')
		print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')
		#print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% | Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')
	model.load_state_dict(torch.load('model.pt'))
	print("test accuracy")
	test_loss, test_acc, test_r, test_p, test_f1, test_macro_r, test_macro_p, test_macro_f1 = evaluate(model, test_iterator, criterion, model_name, False)
	return test_p, test_r, test_f1, test_macro_p, test_macro_r, test_macro_f1

def test(x_test, y_test, z, vocab, embedding_weights, 
	word_sequence_length, emb_dim, hidden_dim, lr,  model, epochs, lstm_sizes, batch_size, 
	dropout_prob):
	model_name = model
	if model == 'CoAttn':
		model = IntraAttention(len(vocab), emb_dim, hidden_dim, 2, lstm_sizes, dropout_prob, embedding_weights)
		model.load_state_dict(torch.load('IA.pt'))
	if model == 'BiRNN':
		model = BiRNN(len(vocab), emb_dim, hidden_dim, 2, lstm_sizes, dropout_prob, embedding_weights)
		model.load_state_dict(torch.load('BiRNN.pt'))
	if model == 'Attn':
		model = WordAttention(len(vocab), emb_dim, hidden_dim, 2, lstm_sizes, dropout_prob, embedding_weights)
		model.load_state_dict(torch.load('HAN.pt'))
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(device)
	is_cuda = False
	if device.type == 'cuda':
		is_cuda = True
	test_set = Dataset(x_test.astype(np.int64), y_test, z, is_cuda)
	test_iterator = load_data(test_set, batch_size)
	optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.999), eps=1e-08, weight_decay=1e-6)
	'''weights = [3.0, 1.0]
	weights = torch.DoubleTensor(weights).cuda()'''
	criterion = nn.BCEWithLogitsLoss()
	try:
		model = model.to(device)
	except:
		print('trying 2nd time')
		model = model.to(device)
	criterion = criterion.to(device)
	test_loss, test_acc, test_r, test_p, test_f1, test_macro_r, test_macro_p, test_macro_f1 = evaluate(model, test_iterator, criterion, model_name, True)
	print(test_acc)
