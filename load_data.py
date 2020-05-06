import numpy as np
import pandas as pd

#load data
pd.options.display.max_colwidth=1000
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

#load word embedding
words_list = np.load('data/words_list.npy')
words_list = words_list.tolist()
word_vectors = np.load('data/word_vectors.npy')
word_vectors = np.float32(word_vectors)

word2idx = {w:i for i,w in enumerate(words_list)}