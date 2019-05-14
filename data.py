import csv
import re
from utils import majority_voting
import numpy as np
import random
#from gensim.models import KeyedVectors

def get_data(action, dataset):
    with open('Data/Wikipedia/' + dataset +'/' + dataset +'_annotated_comments.tsv', encoding='utf-8') as tsvfile:
        i = 0
        data_comments = []
        data_rev_ids = []
        reader = csv.reader(tsvfile, delimiter='\t')
        '''for row in reader:
            if i > 0:
                if action is 'train':
                    if row[6] == 'train':
                        data_comments.append(row[1])
                        data_rev_ids.append(row[0])
                elif action is 'dev':
                    if row[6] == 'dev':
                        data_comments.append(row[1])
                        data_rev_ids.append(row[0])
                elif action is 'test':
                    if row[6] == 'test':
                        data_comments.append(row[1])
                        data_rev_ids.append(row[0])
                else:
                    print("action not found!")
            i += 1'''
        for row in reader:
            if i > 0:
                if action is 'train':
                    if row[6] == 'train':
                        data_comments.append(row[1])
                        data_rev_ids.append(row[0])
                elif action is 'test':
                    if row[6] == 'dev' or row[6] == 'test':
                        data_comments.append(row[1])
                        data_rev_ids.append(row[0])
                else:
                    print("action not found!")
            i += 1
    return data_comments, data_rev_ids

'''def get_data_test_validation(action, dataset):
    with open('Data/Wikipedia/' + dataset +'/' + dataset +'_annotated_comments.tsv', encoding='utf-8') as tsvfile:
        i = 0
        data_comments = []
        data_rev_ids = []
        validate_data_comments = []
        validate_data_rev_ids = []
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            if i > 0:
                if action is 'train':
                    if row[6] == 'train':
                        data_comments.append(row[1])
                        data_rev_ids.append(row[0])
                elif action is 'dev' or action is 'test':
                    if row[6] == 'dev':
                        r = random.randint(1,101)
                        if r <= 25:
                            validate_data_comments.append(row[1])
                            validate_data_rev_ids.append(row[0])
                        else:
                            data_comments.append(row[1])
                            data_rev_ids.append(row[0])
                    elif row[6] == 'test':
                        data_comments.append(row[1])
                        data_rev_ids.append(row[0])
                else:
                    print("action not found!")
            i += 1
            #print (row[1])
    return validate_data_comments, validate_data_rev_ids, data_comments, data_rev_ids'''

def get_train_validation_data(dataset):
    with open('Data/Wikipedia/' + dataset +'/' + dataset +'_annotated_comments.tsv', encoding='utf-8') as tsvfile:
        i = 0
        train_data_comments = []
        train_data_rev_ids = []
        validate_data_comments = []
        validate_data_rev_ids = []
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            if i > 0:
                if row[6] == 'train':
                    r = random.randint(1,101)
                    if r > 5: # for attack and aggression it is 10
                        train_data_comments.append(row[1])
                        train_data_rev_ids.append(row[0])
                    else:
                        validate_data_comments.append(row[1])
                        validate_data_rev_ids.append(row[0])
            i += 1
    return train_data_comments, train_data_rev_ids, validate_data_comments, validate_data_rev_ids



def get_annotations(dataset):
    rev_id_map = {}
    with open('Data/Wikipedia/' + dataset +'/' + dataset +'_annotations.tsv', encoding='utf-8') as tsvfile:
        i = 0
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            if i > 0:
                if rev_id_map.get(row[0]) is None:
                    val = []
                else:
                    val = rev_id_map.get(row[0])
                if dataset == 'toxicity':
                    val.append(row[2])
                elif dataset == 'attack':
                    val.append(row[6])
                elif dataset == 'aggression':
                    val.append(row[2])
                rev_id_map[row[0]] = val
            i += 1
    return rev_id_map

def generate_ylabels(rev_ids, rev_id_map):
    count = 0
    y_label = []
    #print(len(rev_ids))
    for ids in rev_ids:
        #for key, value in rev_id_map.items():
        value = rev_id_map[ids]
        #print(value)
        is_toxic = majority_voting(value)
        if is_toxic == 1:
            count += 1
        y_label.append(is_toxic)
    print(count)
    return y_label