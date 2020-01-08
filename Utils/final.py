# importing Libraries

import numpy as np
from emo_utils import *
import emoji
import matplotlib.pyplot as plt

import numpy as np
np.random.seed(0)

from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
np.random.seed(1)

import speech_recognition as sr

# Loading dataset - test/train
X_train, Y_train = read_csv('../Dataset/Train/train_emoji.csv')
X_test, Y_test = read_csv('../Dataset/Test/tesss.csv')


# set max no.of "words" in a sentence
maxLen = len(max(X_train, key=len).split())

# converting Y into one-hot reprsentation for training the softmax and converting from (m,1) --> (m,5) vector

# y_oh ---> represents the converted one-hot reprsentation

Y_oh_train = convert_to_one_hot(Y_train, C = 5)
Y_oh_test = convert_to_one_hot(Y_test, C = 5)

# Dictionary contains 400,001 words along with the corresponding indices for each word

## word_to_index ---->  contains the mapping of words to indices
## index_to_word ---->  contains the indices to the word
## word_to_vec_map ---> dictionary mapping words to their "GLOVE VECTOR" ??????????????

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('../glove.6B.50d.txt')


def sentences_to_indices(X, word_to_index, max_len):
    """
        i/p : sentences
        o/p : indices (list) of each word in i/p
        
        max_len : max words in a sentence
    """
    
    m = X.shape[0]                                   # number of training examples
    
    # Initialize X_indices as a numpy matrix of zeros
    X_indices = np.zeros((m,max_len))
    
    for i in range(m):
        # Convert the ith training sentence in lower case and split is into words.
        # We obtain a list of words.
        
        sentence_words = (X[i].lower()).split()
        j = 0
        
        # Loop over the words of sentence_words
        for w in sentence_words:
            X_indices[i, j] = word_to_index[w]
            j = j+1

    return X_indices

def pretrained_embedding_layer(word_to_vec_map, word_to_index):

    vocab_len = len(word_to_index) + 1
    
    emb_dim = word_to_vec_map["cucumber"].shape[0]
    
    emb_matrix = np.zeros((vocab_len, emb_dim))

    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)

    embedding_layer.build((None,))
    
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer
