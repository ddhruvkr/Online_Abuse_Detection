import nltk
import re
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from clean_text import *
from utils import *
import string

def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"').replace("`", "'") for token in nltk.word_tokenize(tokens)]
    #return [token for token in nltk.word_tokenize(tokens)]

def preprocess_text(texts, embeddings_index, incorrect_to_correct):
    processed_texts = []
    max_len = -1
    l = 0
    print(len(texts))
    for text in texts:
        l += 1
        if l % 5000 == 0:
            print(l)
        text = re.sub(r'NEWLINE_TOKEN', ' ', text)
        text = text_cleaner(text)
        text = clean_string(text)
        text = text.lower()
        text = get_separated_words(text)
        #text = word_tokenize(text)
        text = text_processor.pre_process_doc(text)
        text = tokens_cleaner(text)
        # text stores one comment at a time, i.e. list of words
        text, incorrect_to_correct = spell_correct(text, embeddings_index, incorrect_to_correct)
        text = simplify_text(text, embeddings_index)
        #stop_words = set(stopwords.words('english'))
        #text = [w for w in text if not w in stop_words]
        processed_texts.append(text)
    processed_texts = remove_words_occuring_once(processed_texts)
    return processed_texts, incorrect_to_correct

def remove_words_occuring_once(texts):
    vocab = {}
    for text in texts:
        for t in text:
            if t not in vocab:
                vocab[t] = 1
            else:
                vocab[t] = vocab[t] + 1
    processed_texts = []
    tokens = []
    for text in texts:
        for t in text:
            if t != '' and vocab[t] > 1:
                tokens.append(t)
        #print(tokens)
        processed_texts.append(tokens)
        tokens = []
    return processed_texts


def get_separated_words(text):
    #print(text)
    text = text.split(' ')
    words = ''
    #print(words)
    #words.append(text)
    for t in text:
        ws = t.split('_')
        for w in ws:
            words += ' ' + w
    return words[1:]

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=True,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

def generate_sequences(texts, vocab):
    x = []
    for tokenized_text in texts:
        seq = []
        for word in tokenized_text:
            seq.append(vocab.get(word, vocab['UNK']))
        x.append(seq)
    return x

def truncate_sequences(text, max_len):
    truncated_text = []
    for x in text:
        if len(x) > max_len:
            truncated_text.append(x[:max_len])
        else:
            truncated_text.append(x)
    return truncated_text

def generate_pad_sequences(x, max_len, p):
    x = pad_sequences(x, maxlen=max_len, padding=p)
    return x

def generate_pad_sequences_manual(x, max_len):
    padded_tokenized_list = []
    for tokenized_text in x:
        curr_len = len(tokenized_text)
        diff = abs(curr_len - max_len)
        if curr_len > max_len:
            tokenized_text = tokenized_text[:-diff or 0]
        elif curr_len < max_len:
            for i in range(diff):
                tokenized_text = np.insert(tokenized_text, curr_len, 0)
        padded_tokenized_list.append(np.array(tokenized_text))
    return padded_tokenized_list

def tokens_cleaner(tokens):
	text_simplified = []
	punct = string.punctuation
	punct_minus = punct.replace('*', '')
	punct_minus = punct_minus.replace('@', '')
	for text in tokens:
		for p in punct_minus:
			text = text.replace(p, "")
		if len(text) > 50:
			text = text[:50]
		text_simplified.append(text)
	return text_simplified

def text_cleaner(text):
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
        text = text.strip()
    return text.lower()
