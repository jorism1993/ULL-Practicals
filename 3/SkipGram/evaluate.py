# -*- coding: utf-8 -*-
"""
Created on Tue May 22 10:48:05 2018

@author: Joris
"""

import torch.nn as nn
from collections import defaultdict
import pickle
import operator
import scipy.spatial.distance as sd

word2embs_W1 = pickle.load(open('SG/word2embed_W1.pickle','rb'))
word2embs_W2 = pickle.load(open('SG/word2embed_W2.pickle','rb'))
word2idx = pickle.load(open('SG/word2idx.pickle','rb'))
word2idx = defaultdict(lambda: word2idx["<unk>"], word2idx)
word2embs_W1 = defaultdict(lambda: word2embs_W1["<unk>"], word2embs_W1)
word2embs_W2 = defaultdict(lambda: word2embs_W2["<unk>"], word2embs_W2)

idx2word = pickle.load(open('SG/idx2word.pickle','rb'))

def find_closest_word(word,word2embs,n=10):
    # Find the closest word of a given word
    # INPUT
    # - word, a string, like 'money'
    # - word2embs, dictionary mapping words (string) to numpy arrays
    # OUTPUT
    # - closest_words, a sorted list of tuples of (word,score). The first elements in the list have the highest scores

    # Check if word actually exists
    assert word in list(word2embs.keys())

    # Retrieve the word embedding of the target word
    word_embedding = word2embs[word]
    closest_words = []

    for w, e in word2embs.items():
        distance =  1 - sd.cosine(word_embedding,e)

        closest_words.append((w,distance))

    # Sort the list and reverse for high to low
    return sorted(closest_words,key=operator.itemgetter(1),reverse=True)[0:n]

word = find_closest_word('computer',word2embs_W1)
print (word)