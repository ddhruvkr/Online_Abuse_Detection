import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#from nltk.tokenize.moses import MosesDetokenizer
from mosestokenizer import MosesDetokenizer
detokenizer = MosesDetokenizer()
from ekphrasis.classes.spellcorrect import SpellCorrector
from ekphrasis.classes.segmenter import Segmenter
#from autocorrect import spell
sp = SpellCorrector(corpus="english")
#seg_tw = Segmenter(corpus="twitter")
seg_eng = Segmenter(corpus="english")
#should have majority vote from annotators to be considered toxic, since non experienced
#annotators tend to label more comments as toxic
def majority_voting(toxicity_list):
    toxic = 0
    non_toxic = 0
    l = len(toxicity_list)
    vote_greater_than = l/2
    for t in toxicity_list:
        if int(float(t)) == 1:
            toxic += 1
    if (toxic > vote_greater_than):
        return 1
    else:
        return 0

def load_word_embeddings(embedding_type, embedding_dim):
    #this would load embeddings and create the vocab for all the pretrained embedding words
    #the starting index would be 1, 0 is for the paddings
    vocab = {}
    reverse_vocab = {}
    if embedding_type == 'glove':
        embeddings_index = dict()
        embeddings_index['PAD'] = np.zeros((1, embedding_dim), dtype='float32')
        embeddings_index['UNK'] = np.zeros((1, embedding_dim), dtype='float32')
        f = open('../../Embeddings/Glove/glove.42B.' + str(embedding_dim) +'d.txt', encoding='utf-8')
        vocab_index = 1
        for line in f:
            values = line.split()
            word = values[0]
            #vocab[word] = vocab_index
            #reverse_vocab[vocab_index] = word
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            vocab_index += 1
        f.close()
        return embeddings_index
    else:
        return None

def extend_vocab_dataset1(texts, embedding_index, vocab, reverse_vocab, add_oov_to_vocab):
    #here 0 is for all paddings words, 1 to n for glove words
    #then for oov words in dataset, last for UNK token
    #vocab_index = 1
    vocab_index = len(vocab.keys())
    print('vocab_index')
    print(vocab_index)
    for text in texts:
        #print(text)
        for word in text:
            #simplified_word_list = word_simplified(raw_word, embedding_index)
            #for word in simplified_word_list:
            if word not in vocab:
                if word in embedding_index:
                    vocab[word] = vocab_index
                    reverse_vocab[vocab_index] = word
                    vocab_index += 1
                if add_oov_to_vocab:
                    if word not in embedding_index:
                        vocab[word] = vocab_index
                        reverse_vocab[vocab_index] = word       # generate reverse vocab as well
                        vocab_index += 1
    return vocab, reverse_vocab

# given a text, returns a 2d matrix with its word embeddings.
# each column represents the embedding for each word
def get_embedding_matrix(embedding_type, embeddings_index, vocab, file_extension, embedding_dim):
    if embedding_type == 'glove':
        oov = []
        if embeddings_index is not None:
            print(len(vocab))
            embedding_matrix = np.zeros((len(vocab), embedding_dim))
            l = 0
            for key in vocab:
                embedding_vector = get_value_for_index(embedding_type, embeddings_index, key)
                if embedding_vector is not None:
                    embedding_matrix[l] = embedding_vector
                else:
                    oov.append(key)
                    #embedding_matrix[l] = np.zeros((1, embedding_dim))
                    #initializing it with zeros seems to work better
                    embedding_matrix[l] = np.random.uniform(low=-0.25, high=0.25, size=(1,embedding_dim))
                l += 1
            print("starting saving in oov file")
            with open('oov' + file_extension + '_glove.txt', 'w') as f:
                for item in oov:
                    item = item.encode('utf8')
                    f.write("%s\n" % item)
            f.close()
            print('len embedding matrix should be same as vocab')
            print(len(embedding_matrix))
            return embedding_matrix
        else:
            return None

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def simplify_text(text, embedding_index):
    text_simplified = []
    for word in text:
        if embedding_index.get(word) is None:
            simplified_word_list = word_simplified(word, embedding_index)
            for w in simplified_word_list:
                text_simplified.append(w)
        else:
            text_simplified.append(word)
    return text_simplified

#will segment words only comprising of alphabets and who generated segments are present in the embeddings
#so if multiple words with wrong spelling are added together then
def word_simplified(word, embedding_index):
    final_word_list = []
    #if the word is already in the embedding list we have we don't need to go through this
    if embedding_index.get(word) is None and word.isalpha():
        mid_word_list = []
        try:
            #print(word)
            mid_word_list = seg_eng.segment(word).split(' ')
            #print(seg_tw)
        except:
            print(word)
            final_word_list.append(word)
            return final_word_list
        #if all the words that are segmented are correct, carry out operation only then
        all_words_segemented_correct = True
        for mw in mid_word_list:
            if embedding_index.get(mw) is None:
                all_words_segemented_correct = False
        if all_words_segemented_correct:
            for mw in mid_word_list:
                final_word_list.append(mw)
            with open("corrected_words_seg.txt", "a+") as f:
                f.write("%s\n" % word.encode('utf8'))

                for w in final_word_list:
                    f.write("%s\n" % w.encode('utf8'))
                f.write("\n")

            f.close()
            return final_word_list
    final_word_list.append(word)
    return final_word_list

def get_value_for_index(embedding_type, embeddings_index, key):
    if embedding_type == 'glove':
        return embeddings_index.get(key)

#will autocorrect only words comprising of alphabets
def spell_correct(text, embeddings_index, incorrect_to_correct):
    spell_corrected_text = []

    for w in text:
        embedding_vector = embeddings_index.get(w)
        if embedding_vector is None:
            if w not in incorrect_to_correct:
                can_it_be_corrected, corrected_word = can_spell_correct(w, embeddings_index)
                incorrect_to_correct[w] = corrected_word
            spell_corrected_text.append(incorrect_to_correct[w])
        else:
            spell_corrected_text.append(w)
    return spell_corrected_text, incorrect_to_correct


def can_spell_correct(word, embeddings_index):
    if len(word) > 25 or not word.isalpha():
        return False, word
    corrected_word = sp.correct(word)
    with open("corrected_words.txt", "a+") as f:
        f.write("%s\n" % word.encode('utf8'))
        f.write("%s\n" % corrected_word.encode('utf8'))
    f.close()
    #if the corrected word not in the embedding, don't autocorrect
    if embeddings_index.get(corrected_word) is not None:
        return True, corrected_word
    else:
        return False, word
