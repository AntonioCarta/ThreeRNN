# coding=utf-8
import pandas as pd
import numpy as np
from treeAutoencoders import TreeAutoEncoders
import collections
import Node
import re


commandID = {"commandID":1}

def parseTree(tree, commandList):
    commandID["commandID"] = 1
    if '0' in tree:
        if len(tree['0'].children) > 0:
            rel,idRoot = tree['0'].children[0] # TODO: generalize AND ALSO IN THE OTHER FILE
            generateCommand(tree, idRoot, commandList)

def generateCommand(tree, id, commandList): # TODO: restituire command list
    node = tree[id]
    children = node.children

    # Base case
    if(len(children)==0): # no child, return my encoding
        return node.lastValue

    for rel,idC in children:
        encodedVal = generateCommand(tree, idC, commandList)
        reference = "@@_" + str(commandID["commandID"]) + "_@@"
        #print ("{} {} {} {} {}".format(commandID["commandID"], node.lastValue, encodedVal, rel, reference ))
        commandList.append("{} {} {} {} {}".format(commandID["commandID"], node.lastValue, encodedVal, rel, reference))
        node.lastValue = reference
        commandID["commandID"] = commandID["commandID"] + 1

    return node.lastValue


def getEncoding(commandList, tae, tokenizer, embedModel, embed_size):
    temp = []
    encoding = None
    for line in commandList:
        val = line.split()
        w1 = val[1]
        w2 = val[2]
        tag = val[3]
        e1 = get_embedding(w1,temp,tokenizer,embedModel, embed_size)
        e2 = get_embedding(w2,temp,tokenizer,embedModel, embed_size)

        encoding = tae.predict(e1, e2, tag)
        ## be used in the later stage
        temp.append(encoding)

    return encoding # the last one produced


def get_embedding(w,temp,tokenizer,embed,embed_size):
    """
    This function is based on the command list produced by inspecting the POS tree.
    Given a word it returns its embedding vector, and, as a special case, if the w1 depend by a previous encoding (we find a placeholder in w)
    then we get that one, which is the encoding of an entire subtree
    Args:
        w: string
        temp: temp array to get previous embedding
        embeddings_index: GLOVE embedding
        embed_size: size of the embedding_vector
    returns:
        embedding corresponding to w
    """

    if (isinstance(w,str) and len(w)>=5 and w[0]=="@" and w[1]=="@"):
        index = re.search('@@_(.*)_@@', w).group(1)
        e = temp[int(index)-1]
    elif(isinstance(w,str)):
        if w.lower() in tokenizer.word_index:
            e = embed.predict(np.array([[tokenizer.word_index[w.lower()]]]))[0]
        else:
            e = np.zeros(embed_size)[np.newaxis, :]
    return e


def generateEncoding(textList, tae, tokenizer, embedModel, embed_size):
    result = []
    for text in textList:
        encodingList = []

        for sentence in text:
            node = collections.defaultdict(lambda: Node.Node()) # node represents the sentence tree

            for line in sentence:
                val = line.split()

                id = val[0]
                parent = val[6]
                relation = val[7]
                tok = val[1]
                pos = val[3]
                node[id].add_attrs({"parent":parent,"emb":[],"tok":tok, "pos":pos})
                node[parent].add_child(id, relation)

            commandList = []
            parseTree(node, commandList)
            #print(commandList)

            encodingList.append(getEncoding(commandList, tae, tokenizer, embedModel, embed_size))

        result.append(encodingList)
    return result


def add_encoding_to_pandas(pickle_file):
    dF = pd.read_pickle(pickle_file)

    USE_TRAINED=True
    embed_size = 100
    tae = TreeAutoEncoders(USE_TRAINED, embed_size=embed_size)
    tokenizer,embedModel = tae.embeddings("../data/toks.final")
    # Ogni elemento  una lista di X, dove X  una frase parsata. Infatti, una domanda commento ha più di una frase
    Q_parsing = dF["Q_parsing"]
    C_parsing = dF["C_parsing"]
    Q_encoded = generateEncoding(Q_parsing, tae, tokenizer, embedModel, embed_size)
    C_encoded = generateEncoding(C_parsing, tae, tokenizer, embedModel, embed_size)
    dF["Q_encoded"] = Q_encoded
    dF["C_encoded"] = C_encoded

    dF.to_pickle(pickle_file)


########### MAIN PROGRAM STARTS NOW ##############
if __name__=='__main__':

    DATASET_DEV = "../data/dataForEmbeddings_DEV.p"
    DATASET_TRAINING = "../data/dataForEmbeddings_TRAINING.p"
    add_encoding_to_pandas(DATASET_TRAINING)
    add_encoding_to_pandas(DATASET_DEV)
