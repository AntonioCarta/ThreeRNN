import pickle as p
import pandas as pd
import sys

dF = p.load(open(sys.argv[1],"rb"))

Q_parsing = dF["Q_parsing"]
C_parsing = dF["C_parsing"]

def generate_triples(parsingList):
    result = []
    for element in parsingList:
        elemList = []
        for sentence in element:

            diz = {}
            for line in sentence:

                val = line.split()

                id = val[0]
                word = val[1]
                parent_id = val[6]
                tag = val[7]

                diz[id] = [word, tag, parent_id]

            for k,v in diz.items():
                if v[2] == "0":
                    #print("{} {} {}".format(v[0], v[1], v[0]))
                    elemList.append([v[0], v[1], v[0]])
                else:
                    #print("{} {} {}".format(v[0], v[1], diz[v[2]][0]))
                    elemList.append([v[0], v[1], diz[v[2]][0]])
                    #print(" ")
        result.append(elemList)
    return result
Q_triples = generate_triples(Q_parsing)
C_triples = generate_triples(C_parsing)

dF["Q_triples"] = Q_triples
dF["C_triples"] = C_triples

dF.to_pickle(sys.argv[1])
