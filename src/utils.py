import os
import pandas as pd
from collections import defaultdict
from keras.preprocessing.text import Tokenizer, base_filter
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
import math
import zipfile
import numpy as np
import pandas as pd
import shelve
import nltk


models_db = "../models/shelve_db"

def load_train_test_df():
    """
    Load train and test files, creating them if necessary.
    NOTE: doesn't update them if data.p changes.
    """
    t_df = "../data/df_train.p"
    if not os.path.exists(t_df):
        df = pickle.load(open("../data/data.p", 'rb'))
        df_train, df_test = train_test_split(df, test_size=.1)
        pickle.dump(df_train, open("../data/df_train.p", 'wb'))
        pickle.dump(df_test, open("../data/df_test.p", 'wb'))
    return pickle.load(open(t_df, 'rb')), pickle.load(open("../data/df_test.p", 'rb'))


def test_model_accuracy(model, X_test, y_test):
    y_pred = predict(model, X_test.T)
    from sklearn.metrics import accuracy_score
    y_t = (np.argmax(y_test, axis=1))
    y_p = np.argmax(y_pred, axis=1)
    return accuracy_score(y_t, y_p)


def load_model(key):
    with shelve.open(models_db) as db:
        x = db[key]
    return x


def save_model(key, model):
    with shelve.open(models_db) as db:
        db[key] = model


def is_gpu_server():
    """
    Check wheter the current script is running in the GPU server. 
    """
    return os.cpu_count() > 20


def missing_embeddings(data):
    """
    Return the list of missing words from the pretrained embeddings matrix
    Args:
        data: list of pairs <question, answer>
    returns:
        list of missing words
    """
    tokenizer = Tokenizer()
    # to have all the word
    tokenizer.fit_on_texts(np.concatenate(data))
    # Lowest index from the tokenizer is 1 - we need to include 0 in our vocab
    # count
    VOCAB = len(tokenizer.word_counts) + 1

    MAX_LEN = 231  # max text size in training set.
    to_seq = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=MAX_LEN)
    prepare_data = lambda data: [to_seq(data[0]), to_seq(data[1])]
    X = prepare_data(data)

    print('Vocab size =', VOCAB)

    GLOVE_DATA = '../data/glove.zip'
    if is_gpu_server():
        GLOVE_DATA = '/home/attardi/Collection/embeddings/glove.6B.zip'

    zipf = zipfile.ZipFile(GLOVE_DATA)
    f_idx = zipf.namelist()[0]

    if not os.path.exists("../data/glove/" + f_idx):
        zipf.extract(f_idx, path="../data/glove")

    emb_words = set()
    with open("../data/glove/" + f_idx, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            emb_words.add(word)

    missing_words = []
    for word, i in tokenizer.word_index.items():
        if word not in emb_words:
            missing_words.append(word)

    print('Total number of null word embeddings: ' + str(len(missing_words)))
    return missing_words, tokenizer


if __name__=='__main__':
    X = pickle.load(open("../data/data.p", "rb"))[["Q_Body", "C_Body"]]
    X = np.array(X).T
    l, tok = missing_embeddings(X)

    for i in range(0, len(l) - 5, 5):
        print("{:15} {:15} {:15} {:15} {:15}".format(l[i], l[i+1], l[i+2], l[i+3], l[i+4]))
    print("Missing: " + str(len(l)))

