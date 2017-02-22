import numpy as np
from gensim.models import Word2Vec
import gensim
#from corpus import *
import re
import os
import zipfile
import codecs
from utils import is_gpu_server
from keras.layers import Embedding

dev_file = "../data/semeval2016-task3-cqa-ql-traindev-v3.2/v3.2/dev/SemEval2016-Task3-CQA-QL-dev-subtaskA.xml"
loaded_glove = [None, None, None, None]  # keeps the GloVe matrix in memory


def load_taskA_for_keras(embed_size, VOCAB, TRAIN_EMBED, tokenizer):
    if (embed_size==200):
        data = '../data/qatarliving_qc_size200_win5_mincnt1_rpl_skip3_phrFalse_2016_02_25.word2vec.bin'
    elif(embed_size ==100):
        data = '../data/qatarliving_qc_size100_win5_mincnt1_skip1_with_sent_repl_iter1.word2vec.bin'
    elif(embed_size == 101):
        data = '../data/qatarliving_qc_size100_win10_mincnt5_rpl_skip1_phrFalse_2016_02_23.word2vec.bin'
        embed_size = 100
    else:
        print('Select the right emb size')
        

    model =  gensim.models.Word2Vec.load(data)
    # prepare embedding matrix
    embedding_matrix = np.zeros((VOCAB, embed_size))
    for word, i in tokenizer.word_index.items():
        if word in model:
            embedding_vector = model[word]
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        #else:
        #    print('Missing from QAEMB: {}'.format(word))

    print('Total number of null word embeddings:')
    print(np.sum(np.sum(embedding_matrix, axis=1) == 0))

    return Embedding(VOCAB, embed_size, weights=[embedding_matrix], trainable=TRAIN_EMBED)

def load_glove_for_keras(embed_size, VOCAB, TRAIN_EMBED, tokenizer):
    global loaded_glove

    GLOVE_STORE = '../models/'
    GLOVE_DATA = '../data/glove.zip'
    if is_gpu_server():
        GLOVE_DATA = '/home/attardi/Collection/embeddings/glove.6B.zip'

    sizes = {
        0: 50,
        1: 100,
        2: 200,
        3: 300
    }

    # the file to retrieve in the zip archive
    idx = min(embed_size // 100, 3)
    embed_size = sizes[idx]
    zipf = zipfile.ZipFile(GLOVE_DATA)
    f_idx = zipf.namelist()[idx]
    GLOVE_STORE += f_idx
    if not os.path.exists("../data/glove/" + f_idx):
        # unzip file
        zipf.extract(f_idx, path="../data/glove")

    print('Computing GloVe')
    if loaded_glove[idx] is not None:
        embeddings_index = loaded_glove[idx]
    else:
        embeddings_index = {}

        f = codecs.open("../data/glove/" + f_idx, "r", "utf-8")
        #f = open("../data/glove/" + f_idx)

        # load matrix
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:])
            embeddings_index[word] = coefs
        f.close()
        loaded_glove[idx] = embeddings_index

    # prepare embedding matrix
    embedding_matrix = np.zeros((VOCAB, embed_size))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            # else:
            #     print('Missing from GloVe: {}'.format(word))

    # np.save(GLOVE_STORE, embedding_matrix)

    # print('Loading GloVe')
    # embedding_matrix = np.load(GLOVE_STORE + ".npy")

    print('Total number of null word embeddings:')
    print(np.sum(np.sum(embedding_matrix, axis=1) == 0))

    return Embedding(VOCAB, embed_size, weights=[embedding_matrix], trainable=TRAIN_EMBED)


def QL_embeddings():
    """
        Builds embeddings from Qatar Living corpus.
        The corpus is really small, so the resulting embeddings perform badly.
    """
    ql = QatarLivingCorpus(dev_file)
    sent = []
    for th in ql:
        for s in ThreadParser(th).get_sentences():
            sent.append(s)
    # CBOW, size=100, window=5
    w2v = Word2Vec(sent)

    # useful after training to trim unneeded model memory = use (much) less RAM
    w2v.init_sims(replace=True)
    w2v.save_word2vec_format('../models/ql_w2v_cbow_s100_w5', binary=True)

    # test for embeddings
    test_embeddings(w2v)


def test_embeddings(w2v):
    # downloaded from https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip
    test_file = r'../data/embedding_tests/questions-words.txt'
    report = w2v.accuracy(test_file)

    for section in report:
        n = len(section['correct']) + len(section['incorrect'])
        acc = (len(section['correct']) / n) if n > 0 else 0
        print(section['section'] + ": " + str(acc))


