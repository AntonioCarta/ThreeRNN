from __future__ import print_function
from functools import reduce
import json
import os
import re
import tarfile
import tempfile
#import Scorer
#import feature_extraction
#import corpus
import pickle
import numpy as np
import math as math
import codecs
import keras
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.layers import merge, recurrent, Dense, Input, Dropout, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from keras.models import load_model
from keras.models import model_from_json
from word_embeddings import load_glove_for_keras
from word_embeddings import load_taskA_for_keras
import random
import shelve
from finalTaskB import create_model,get_tokenizer,to_seq,pad_tag,prepare_input


db_file = "../models/models_db_TASKB"
model_key = "combo_param_list_TASKB"


DATASET_TEST = "../data/dataForEmbeddingsTASKB_TEST.p"
DATASET_TRAINING = "../data/dataForEmbeddingsTASKB_TRAINING.p"
DATASET_DEV = "../data/dataForEmbeddingsTASKB_DEV.p"


def load_and_prepare_data():
    """
    Load data, and split training and validation set
    :return: train and dev (validation) list
    """
    data_training = pickle.load( open(DATASET_TRAINING, "rb" ) )
    data_training = prepare_input(data_training)
    data_test= pickle.load( open(DATASET_TEST, "rb" ) )
    data_test= prepare_input(data_test)
    data_dev= pickle.load( open(DATASET_DEV, "rb" ) )
    data_dev= prepare_input(data_dev)
    return data_training,data_dev,data_test

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
    df_data.to_pickle("../predictions_dump/prediction_verB_DEV.p")
    os.system('python Scorer.py B_DEV')
    with open("../scorings/B_DEV.scores") as myfile:
        firstlines=myfile.readlines()[0:5] #put here the interval you want
    return float(firstlines[1].split(':')[1])


def eval_modelTEST(model,dev):
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
    df_data = pickle.load( open(DATASET_TEST, "rb" ) )
    df_data['dist_prediction']=df_data['Prediction']

    i=0
    for index, row in df_data.tail(len(arr)).iterrows():
    	df_data.set_value(index, 'Prediction', arr[i])
        #### predicted_val[i][0] qui e' zero perche il primo valore rappresenta
        #### good, quindi rappresenta la prob di good
    	df_data.set_value(index, 'dist_prediction', predicted_val[i][0])
    	i+=1
    df_data.to_pickle("../predictions_dump/prediction_verB_TEST.p")
    os.system('python Scorer.py B_TEST')
    with open("../scorings/B_TEST.scores") as myfile:
        firstlines=myfile.readlines()[0:5] #put here the interval you want
    return float(firstlines[1].split(':')[1])


if __name__=='__main__':
    training, dev,test = load_and_prepare_data()
    tokenizer = get_tokenizer()
    VOCAB = len(tokenizer.word_counts) + 1
    PATIENCE = 8
    BATCH_SIZE = 128
    MAX_EPOCHS = 14
    LOADMODEL = False

    with shelve.open(db_file) as db:
        l = db[model_key]
    trip = [ elem for elem in l if (len(elem)==3)]
    trip.sort(key=lambda x: x[1], reverse = True)
    bestmodel=trip[2]
    print(bestmodel)
    bestmodel[2]["tokenizer"]=tokenizer
    prepare_data = lambda data, t, ml: (to_seq(data[0], t, ml), pad_tag(data[1], ml), to_seq(data[2], t, ml),
                             to_seq(data[3], t, ml), pad_tag(data[4], ml), to_seq(data[5], t, ml),
                             data[6], data[7])

    training = prepare_data(training, tokenizer, bestmodel[2]["MAX_LEN"])
    dev = prepare_data(dev, tokenizer, bestmodel[2]["MAX_LEN"])
    test = prepare_data(test, tokenizer, bestmodel[2]["MAX_LEN"])


    if LOADMODEL:
        print("Loading best model")
        file_model="../models/final_comboTASKB.json"
        file_weight="../models/final_comboTASKB.h5"
        with open(file_model, "r") as jfile:
            model = model_from_json(json.load(jfile))
        model.load_weights(file_weight)

        model.compile(optimizer=bestmodel[2]["OPTIMIZER"], loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        print('Build model...')
        print('Vocab size =', VOCAB)
        model = create_model(VOCAB, **bestmodel[2])
        print('Training')
        es = EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=0, mode='auto')
        model.fit([training[0], training[1], training[2],training[3], training[4], training[5], training[6]], training[7],
                batch_size=BATCH_SIZE, nb_epoch=MAX_EPOCHS)#, callbacks=[es])





    loss, acc = model.evaluate([dev[0], dev[1], dev[2],dev[3], dev[4], dev[5], dev[6]], dev[7])
    print('Test loss / dev accuracy = {:.4f} / {:.4f}'.format(loss, acc))


    print('MAP DEV = {:.4f}'.format(eval_model(model,dev)))









    eval_modelTEST(model,test)
