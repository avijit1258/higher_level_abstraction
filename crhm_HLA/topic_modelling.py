import spacy

#spacy.load('en')
from spacy.lang.en import English

import nltk

from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

import random
text_data = []

from gensim import corpora

import pickle

import gensim

nltk.download('wordnet')

parser = English()


def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

def prepare_text_for_lda(text):
    tokens = tokenize(text)
    #print(tokens)
    tokens = [token for token in tokens if len(token) >= 2]
    #print(tokens)
    tokens = [token for token in tokens if token not in en_stop]
    #print(tokens)
    tokens = [get_lemma(token) for token in tokens]
    #print(tokens)
    return tokens


with open('people.csv') as f:
    for line in f:
        #print(line)
        tokens = prepare_text_for_lda(line)
        #if random.random() > .99:
            #print(tokens)
        text_data.append(tokens)

print(text_data)
dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]

pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')

NUM_TOPICS = 5
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=5)
for topic in topics:
    print(topic)
