import pandas
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, merge, Input
from keras.regularizers import l2
from treeAutoencoders import SentenceFilter
import pickle
import os
import shelve
from random import randint
import random
db_file = "../models/models_db_AUTO"
model_key = "AUTOBEST"

def numberofzeros(Q_enc_final,embed_size):
    counter=0
    for e in Q_enc_final:
        if (np.all(e==0)):
            counter= counter+1
    print(counter)

def get_embLIST(data,embed_size):
    binary = {'Good': [1.,0.], 'Bad': [0.,1.], 'PotentiallyUseful': [0.,1.]}
    lab=[binary[str(e)]for e in data['Gold_Label']]

    Q_enc_final = []
    for encodings_list in data["Q_encoded"]:
        temp = [np.zeros(embed_size)]
        if(encodings_list is not None):
            for e in encodings_list:
                if (e is not None):
                    temp = temp + e[0]
        Q_enc_final.append(temp)

    C_enc_final = []
    for encodings_list in data["C_encoded"]:
        temp = [np.zeros(embed_size)]
        if(encodings_list is not None):
            for e in encodings_list:
                if (e is not None):
                    temp = temp + e[0]
        C_enc_final.append(temp)
    return [Q_enc_final, C_enc_final, lab]

def create_model(embed_size,EMBED_HIDDEN_SIZE, ACTIVATION,L2, DP,OPTIMIZER,N_LAYER):
    """
    Initialize the nn model with the hyper-parameters passed as args and compiles it.
    :return: model ready to be trained
    """

    in_q = Input(shape=(embed_size,))
    in_c = Input(shape=(embed_size,))
    joint = merge([in_q, in_c], mode='concat')

    for l in range(N_LAYER-1):
        joint = Dropout(DP)(joint)
        joint = Dense(EMBED_HIDDEN_SIZE, W_regularizer=l2(l=L2), activation=ACTIVATION)(joint)

    # output layer
    joint = Dropout(DP)(joint)
    joint = Dense(2, W_regularizer=l2(l=L2), activation='softmax')(joint)


    model = Model(input=[in_q, in_c], output=joint)
    model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_random_parameters():
    """
    create a random configuration from using dists to select random parameters
    :return: neural network parameters for create_model
    """

    # change this dictionary to change the random distribution
    dists = dict(
        embed_size=[100],
        EMBED_HIDDEN_SIZE=[10, 20, 50, 100, 150, 200, 300, 400, 500],
        ACTIVATION=['relu', 'tanh', 'sigmoid'],
        L2 = [0, .01, .001, .0001,0.00001,0.000001,0.0000001],
        DP = [0.,0.1,0.2,0.3,0.4],
        OPTIMIZER=['adam','rmsprop'],
        N_LAYER=[1,2,3,4])

    d = {}
    for k in dists:
        if hasattr(dists[k], '__len__'):
            idx = random.randint(0, len(dists[k]) - 1)
            d[k] = dists[k][idx]
        else:
            d[k] = dists[k].rvs()
    return d


def eval_model(model,dev):
    """
    Return the MAP value, calling Scoring and reading the genereted files
    :param model: a trained model
    :param model: dev which we would like to have the MAP value (could be also for validation)
    :return: MAP (Mean Average Precision) value
    """

    predicted_val=model.predict(dev)
    label = {"0":'Good', "1":'Bad'}
    arr= [label[str(np.argmin(e))] for e in predicted_val]


    ## generate data for MAP
    df_data = pickle.load( open(DATASET_DEV, "rb" ) )
    df_data['dist_prediction']=df_data['Prediction']

    i=0
    for index, row in df_data.tail(len(arr)).iterrows():
    	df_data.set_value(index, 'Prediction', arr[i])
        #### predicted_val[i][0] qui e' zero perche il primo valore rappresenta
        #### good, quindi rappresenta la prob di good
    	df_data.set_value(index, 'dist_prediction',predicted_val[i][0])
    	i+=1
    df_data.to_pickle("../predictions_dump/prediction_verTREE.p")
    os.system('python Scorer.py TREE')
    with open("../scorings/TREE.scores") as myfile:
        firstlines=myfile.readlines()[0:5] #put here the interval you want
    return float(firstlines[1].split(':')[1])

def prepare_in(X, y):
    X0 = np.asarray(X[0])[:, 0, :]

    l = []
    for row in X[1]:
        #a = row[0]
        for el in row:
            l.append(el)
    X1 = np.array(l).reshape(X0.shape[0], X0.shape[1])

    Y =  np.asarray(y)

    return [X0,X1],Y


if __name__=='__main__':
    ############# NEW PART FOR ENCODINGS ###################
    embed_size = 100
    DATASET_DEV = "../data/dataForEmbeddings_DEV.p"
    DATASET_TRAINING = "../data/dataForEmbeddings_TRAINING.p"
    dataTR = pandas.read_pickle(DATASET_TRAINING)
    dataTest = pandas.read_pickle(DATASET_DEV)

    training= get_embLIST(dataTR,embed_size=100)
    test= get_embLIST(dataTest,embed_size=100)

    X, y = prepare_in([training[0], training[1]], np.array(training[2]))
    X_test, y_test = prepare_in([test[0], test[1]], np.array(test[2]))
    while(True):
        par = get_random_parameters()
        print(par)
        model = create_model(**par)
        model.fit(X, y, batch_size=128, nb_epoch=randint(10,200))
        loss, acc = model.evaluate(X_test, y_test)
        MAP = eval_model(model,X_test)
        print('Test loss / test accuracy / MAP = {:.4f} / {:.4f} / {:.4f}'.format(loss, acc, MAP))

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
            model.save_weights("../models/AUTO.h5", overwrite=True)
            with open("../models/AUTO.json", "w") as outfile:
                json.dump(model.to_json(), outfile)
