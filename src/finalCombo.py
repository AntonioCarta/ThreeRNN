"""
accuracy with 100D, sum DP=.5:
0.6923
"""
import json
import codecs
import pickle
import math as math
import numpy as np
import keras
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.layers import merge, recurrent, Dense, Input, Dropout, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from word_embeddings import load_glove_for_keras
from word_embeddings import load_taskA_for_keras
import random
import shelve
import os
db_file = "../models/models_db"
model_key = "combo_param_list"

tags = ["ADV","AMOD","EXT","MNR","PRD","TMP","PRN","APPO","HMOD","NAME","NMOD","POSTHON",
"TITLE","DIR","DTV","EXTR","LGS","LOC","PRP","PRT","PUT","CONJ","COORD","DEP","PMOD",
"IM","SUB","VC","SBJ","OBJ","OPRD","SUFFIX","HYPH","P","ROOT","_"]


DATASET_DEV = "../data/dataForEmbeddings_DEV.p"
DATASET_TRAINING = "../data/dataForEmbeddings_TRAINING.p"
DATASET_TEST=""


def prepare_input(data):
    """
    Take data from pandas, and prepare it for training
    :param data: a pandas data frame
    :return: arrays of arrays: [wq0, tagq, wq1, wc0, tagc, wc1, data[features].values, lab]
    """

    features = ['Comment_number', 'Has_Question',
       'Has_URL', 'Has_User_In_Body', 'Jaccard_abs', 'Jaccard_norm', 'LCS',
       'Len_Comments_abs', 'Len_Comments_norm', 'Same_User']

    binary = {'Good': [1.,0.], 'Bad': [0.,1.], 'PotentiallyUseful': [0.,1.],'?':[0,0]}
    lab=[binary[str(e)]for e in data['Gold_Label']]


    wq0 = []
    tagq = []
    wq1 = []
    for el in data.Q_triples:
        l1 = []
        l2 = []
        l3 = []
        for t in el:
            w0, tag, w1 = t
            l1.append(w0)
            l2.append(tag)
            l3.append(w1)
        wq0.append(l1)
        tagq.append(l2)
        wq1.append(l3)

    wc0 = []
    tagc = []
    wc1 = []
    for el in data.Q_triples:
        l1 = []
        l2 = []
        l3 = []
        for t in el:
            w0, tag, w1 = t
            l1.append(w0)
            l2.append(tag)
            l3.append(w1)
        wc0.append(l1)
        tagc.append(l2)
        wc1.append(l3)

    #print(tagq[:20])

    T1=[]
    for t in tagq:
        t = np.array([tags.index(el) for el in t])
        a = np.zeros((len(t), len(tags)))
        a[np.arange(len(t)), t] = 1
        T1.append(a)
    tagq = T1


    T1=[]
    for t in tagc:
        t = np.array([tags.index(el) for el in t])
        a = np.zeros((len(t), len(tags)))
        a[np.arange(len(t)), t] = 1
        T1.append(a)
    tagc = T1
    return [wq0, tagq, wq1, wc0, tagc, wc1, data[features].values, lab]


def to_seq(X, tokenizer, MAX_LEN):
    """
    Take a colum X of the data (i.e, wq0) and make a padding in order to have
    all the sequence with the same lenght
    :param X: array of list of word
    :param tokenizer: tokenized word
    :param MAX_LEN: maximum lenght of the sequence
    :return: padded sequence
    """
    X = [[tokenizer.word_index[w.lower()] if w.lower() in tokenizer.word_index else 0 for w in el] for el in X]
    return pad_sequences(X, maxlen=MAX_LEN, truncating= 'post')


def pad_tag(X, MAX_LEN):
    """
    Same padding as before but for the Modifier onehot
    :param X: array of list of encoded Modifier
    :param MAX_LEN: maximum lenght of the sequence
    :return: padded sequence
    """
    ## len(tags) dimension of onehot encodings of tags
    padX = np.zeros((len(X), MAX_LEN, len(tags)))
    for ix in range(len(X)):
        for iy in range(len(X[ix][:MAX_LEN])):
            for iz in range(len(tags)):
                offset = 0
                if len(X[ix]) < MAX_LEN:
                    offset = MAX_LEN - len(X[ix])
                padX[ix, offset + iy, iz] = X[ix][iy][iz]

    return padX


def load_and_prepare_data():
    """
    Load data, and split training and validation set
    :return: train and dev (validation) list
    """
    data_training = pickle.load( open(DATASET_TRAINING, "rb" ) )
    data_training = prepare_input(data_training)
    data_dev= pickle.load( open(DATASET_DEV, "rb" ) )
    data_dev= prepare_input(data_dev)
    return data_training,data_dev

'''
def load_and_prepare_data():
    """
    Load data, and split training and validation set
    :return: train and test (validation) list
    """
    data = pickle.load( open(DATASET_TRAINING, "rb" ) )

    data_labels = data["Thread_ID"]
    index = math.ceil(len(data_labels) * 0.9)
    while (data_labels[index]==data_labels[index-1]):
        index-=1

    data = prepare_input(data)

    training = [data[0][:index], data[1][:index], data[2][:index], data[3][:index],
                data[4][:index], data[5][:index], data[6][:index], data[7][:index]]
    test = [data[0][index:], data[1][index:], data[2][index:], data[3][index:],
            data[4][index:], data[5][index:], data[6][index:], data[7][index:]]
    return training, test
'''

def get_tokenizer():
    """
    Build a tokenizer from "../data/toks.final" which stores all the keys
    :return: tokenizer
    """
    # generate tokenizer
    f = codecs.open("../data/toks.final", "r", "utf-8")
    toks = [el[:-1].lower() for el in f.readlines()]
    tokenizer = Tokenizer()
    # to have all the word
    tokenizer.fit_on_texts(toks)
    return tokenizer


def create_model(VOCAB, USE_GLOVE, EMBED_HIDDEN_SIZE, TRAIN_EMBED, SENT_HIDDEN_SIZE,
                 ACTIVATION, MAX_LEN, DP, RNN, RNN_LAYERS, HIDDEN_LAYERS, OPTIMIZER, L2,tokenizer):
    """
    Initialize the nn model with the hyper-parameters passed as args and compiles it.
    :return: model ready to be trained
    """
    if USE_GLOVE:
        embed = load_glove_for_keras(EMBED_HIDDEN_SIZE, VOCAB, TRAIN_EMBED, tokenizer)
    else:
        embed = load_taskA_for_keras(EMBED_HIDDEN_SIZE, VOCAB, TRAIN_EMBED, tokenizer)

    rnn_dim = len(tags) + 2*SENT_HIDDEN_SIZE
    rnn_kwargs = dict(output_dim=rnn_dim, dropout_W=DP, dropout_U=DP)
    SumEmbeddings = keras.layers.core.Lambda(lambda x: K.sum(x, axis=1), output_shape=(rnn_dim, ))

    translate = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))

    premise_wq0 = Input(shape=(MAX_LEN,), dtype='int32')
    hypothesis_wc0 = Input(shape=(MAX_LEN,), dtype='int32')
    premise_wq1 = Input(shape=(MAX_LEN,), dtype='int32')
    hypothesis_wc1 = Input(shape=(MAX_LEN,), dtype='int32')
    premise_tagq = Input(shape=(MAX_LEN,len(tags),), dtype='float32')
    hypothesis_tagc = Input(shape=(MAX_LEN,len(tags),), dtype='float32')

    prem_wq0 = embed(premise_wq0)
    hypo_wc0 = embed(hypothesis_wc0)
    prem_wq1 = embed(premise_wq1)
    hypo_wc1 = embed(hypothesis_wc1)

    prem_wq0 = translate(prem_wq0)
    hypo_wc0 = translate(hypo_wc0)
    prem_wq1 = translate(prem_wq1)
    hypo_wc1 = translate(hypo_wc1)

    prem = merge([prem_wq0, prem_wq1, premise_tagq], mode='concat', name='merge_wq0_wq1_tagq')
    hypo = merge([hypo_wc0, hypo_wc1, hypothesis_tagc], mode='concat', name='merge_wc0_wc1_tagc')

    if RNN == "LSTM":
        RNN = recurrent.LSTM
    elif RNN == "GRU":
        RNN = recurrent.GRU

    if RNN and RNN_LAYERS > 1:
      for l in range(RNN_LAYERS - 1):
        rnn = RNN(return_sequences=True, **rnn_kwargs)
        prem = BatchNormalization()(rnn(prem))
        hypo = BatchNormalization()(rnn(hypo))

    rnn = SumEmbeddings if not RNN else RNN(return_sequences=False, **rnn_kwargs)
    prem = rnn(prem)
    hypo = rnn(hypo)
    prem = BatchNormalization()(prem)
    hypo = BatchNormalization()(hypo)

    # input vectors with features for question and comment
    feat = Input(shape=(10,), dtype='float32')

    joint = merge([prem, hypo, feat], mode='concat')
    joint = Dropout(DP)(joint)
    for i in range(HIDDEN_LAYERS):
      joint = Dense(rnn_dim, activation=ACTIVATION, W_regularizer=l2(L2) if L2 else None)(joint)
      joint = Dropout(DP)(joint)
      joint = BatchNormalization()(joint)

    pred = Dense(2, activation='softmax')(joint)

    model = Model(input=[premise_wq0, premise_tagq, premise_wq1,
                        hypothesis_wc0, hypothesis_tagc, hypothesis_wc1, feat], output=pred)
    model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_random_parameters():
    """
    create a random configuration from using dists to select random parameters
    :return: neural network parameters for create_model
    """

    # change this dictionary to change the random distribution
    dists = dict(
        RNN = [recurrent.LSTM, recurrent.GRU, None, None],
        RNN_LAYERS = [2,1],
        USE_GLOVE = [True, False],
        TRAIN_EMBED = [False],
        HIDDEN_LAYERS = [3],
        EMBED_HIDDEN_SIZE = [100,200],
        SENT_HIDDEN_SIZE = [50,100,200],
        MAX_LEN = [5,10,25,50,75,100,150],
        DP = [0.,0.1,0.2,0.3,0.4],
        L2 = [0, .01, .001, .0001,0.00001,0.000001,0.0000001],
        ACTIVATION = ['relu', 'tanh', 'sigmoid'],
        OPTIMIZER = ['adam','rmsprop'] #[RMSprop(lr=.01), RMSprop(lr=.001)] can't pickle RMSProp
    )

    d = {}
    for k in dists:
        if hasattr(dists[k], '__len__'):
            idx = random.randint(0, len(dists[k]) - 1)
            d[k] = dists[k][idx]
        else:
            d[k] = dists[k].rvs()
    return d


'''
def eval_model(model,test):
    """
    Return the MAP value, calling Scoring and reading the genereted files
    :param model: a trained model
    :param model: test which we would like to have the MAP value (could be also for validation)
    :return: MAP (Mean Average Precision) value
    """
    predicted_val=model.predict([test[0], test[1], test[2],test[3], test[4], test[5], test[6]])

    label = {"0":'Good', "1":'Bad'}
    arr= [label[str(np.argmax(e))] for e in predicted_val]


    ## generate data for MAP
    df_data = pickle.load( open(DATASET_TRAINING, "rb" ) )
    df_data['dist_prediction']=df_data['Prediction']

    i=0
    for index, row in df_data.tail(len(arr)).iterrows():
        df_data.set_value(index, 'Prediction', arr[i])
        #### predicted_val[i][0] qui e' zero perche il primo valore rappresenta
        #### good, quindi rappresenta la prob di good
        df_data.set_value(index, 'dist_prediction', predicted_val[i][0])
        i+=1
    df_data.to_pickle("../predictions_dump/prediction_ver001.p")
    os.system('python Scorer.py 001')
    with open("../scorings/001.scores") as myfile:
        firstlines=myfile.readlines()[0:5] #put here the interval you want
    return float(firstlines[1].split(':')[1])
'''

def eval_model(model,dev):
    """
    Return the MAP value, calling Scoring and reading the genereted files
    :param model: a trained model
    :param model: dev which we would like to have the MAP value (could be also for validation)
    :return: MAP (Mean Average Precision) value
    """
    predicted_val=model.predict([dev[0], dev[1], dev[2],dev[3], dev[4], dev[5], dev[6]])
    label = {"0":'Good', "1":'Bad'}
    arr= [label[str(np.argmax(e))] for e in predicted_val]


    ## generate data for MAP
    df_data = pickle.load( open(DATASET_DEV, "rb" ) )
    df_data['dist_prediction']=df_data['Prediction']

    i=0
    for index, row in df_data.tail(len(arr)).iterrows():
    	df_data.set_value(index, 'Prediction', arr[i])
        #### predicted_val[i][0] qui e' zero perche il primo valore rappresenta
        #### good, quindi rappresenta la prob di good
    	df_data.set_value(index, 'dist_prediction', predicted_val[i][0])
    	i+=1
    df_data.to_pickle("../predictions_dump/prediction_ver001.p")
    os.system('python Scorer.py 001')
    with open("../scorings/001.scores") as myfile:
        firstlines=myfile.readlines()[0:5] #put here the interval you want
    return float(firstlines[1].split(':')[1])

if __name__=='__main__':
    training_data, test_data = load_and_prepare_data()
    tokenizer = get_tokenizer()

    # Lowest index from the tokenizer is 1 - we need to include 0 in our vocab
    VOCAB = len(tokenizer.word_counts) + 1
    PATIENCE = 8
    BATCH_SIZE = 128
    MAX_EPOCHS = 50

    while True:
        par = get_random_parameters()
        par["tokenizer"]=tokenizer
        print('RNN / Embed / Sent = {}, {}, {}'.format(par["RNN"], par["EMBED_HIDDEN_SIZE"], par["SENT_HIDDEN_SIZE"]))
        print('GloVe / Trainable Word Embeddings = {}, {}'.format(par["USE_GLOVE"], par["TRAIN_EMBED"]))

        prepare_data = lambda data, t, ml: (to_seq(data[0], t, ml), pad_tag(data[1], ml), to_seq(data[2], t, ml),
                                 to_seq(data[3], t, ml), pad_tag(data[4], ml), to_seq(data[5], t, ml),
                                 data[6], data[7])

        training = prepare_data(training_data, tokenizer, par["MAX_LEN"])
        test = prepare_data(test_data, tokenizer, par["MAX_LEN"])

        print('Build model...')
        print('Vocab size =', VOCAB)
        model = create_model(VOCAB, **par)

        #model.summary()
        print(par)

        print('Training')
        es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
        model.fit([training[0], training[1], training[2],training[3], training[4], training[5], training[6]], training[7],
                  batch_size=BATCH_SIZE, nb_epoch=MAX_EPOCHS, callbacks=[es], validation_split=0.1)

        loss, acc = model.evaluate([test[0], test[1], test[2],test[3], test[4], test[5], test[6]], test[7], batch_size=BATCH_SIZE)
        MAP = eval_model(model,test)
        print('Test loss / test accuracy / MAP = {:.4f} / {:.4f} / {:.4f}'.format(loss, acc, MAP))
        par["tokenizer"]=''
        with shelve.open(db_file) as db:
            if model_key in db:
                l = db[model_key]
                l.append((acc, MAP, par))
                l.sort(key=lambda x: x[1], reverse=True)
                best_map = l[0][1]
                db[model_key] = l
            else:
                db[model_key] = [(acc, MAP, par)]
                best_map = 0

        if best_map < MAP:
            model.save_weights("../models/final_combo.h5", overwrite=True)
            with open("../models/final_combo.json", "w") as outfile:
                json.dump(model.to_json(), outfile)
