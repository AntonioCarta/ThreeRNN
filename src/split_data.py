import pandas
import pickle
import random

fin = "dataForEmbeddings"

dim_train = 0.8

def find_duplicates(sortedlist):
	duplicates = [[sortedlist[0][0]]]

	first = sortedlist[0]
	i = 1
	while i<len(sortedlist):

		second = sortedlist[i]
		
		if first[1] == second[1]:
			duplicates[-1].append(second[0])		
		else:
			duplicates.append([second[0]])

		i+=1
		
		first = second


	return duplicates

dataframe = pandas.read_pickle("../data/"+fin+".p")

questions_id = dataframe["Thread_ID"]
questions_text = dataframe["Q_Body"]

comment_id = dataframe["Comment_ID"]
comment_text = dataframe["C_Body"]

questions_couples = set(list(zip(questions_id, questions_text)))
questions_sorted = sorted(questions_couples, key=lambda x: x[1])

duplicated_questions = find_duplicates(questions_sorted)

comment_couples = set(list(zip(comment_id, comment_text)))
comment_sorted = sorted(comment_couples, key=lambda x: x[1])

duplicated_comments = find_duplicates(comment_sorted)


random.shuffle(duplicated_questions)

s = sum([len(el) for el in duplicated_questions])
s *= dim_train
s = int(s)

limit = 0
i = 0
while limit <= s and i < len(duplicated_questions):
	limit+=len(duplicated_questions[i])
	i+=1
	
train = [el for el in duplicated_questions[:i]]
test = [el for el in duplicated_questions[i:]]


dF_train = pandas.DataFrame()
for l in train:
	for el in l:
		rows = dataframe.loc[dataframe['Thread_ID'] == el]
		dF_train = dF_train.append(rows)


dF_train.to_csv("../data/"+fin + "_train.csv")
dF_train.to_pickle("../data/"+fin + "_train.p")

dF_test = pandas.DataFrame()
for l in test:
	for el in l:
		rows = dataframe.loc[dataframe['Thread_ID'] == el]
		dF_test = dF_test.append(rows)

dF_test.to_csv("../data/"+fin + "_test.csv")
dF_test.to_pickle("../data/"+fin + "_test.p")
