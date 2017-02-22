import pickle
import editdistance
import zipfile
import os

freqdist = pickle.load (open("../data/dizionario.p", "rb"))
K = 2

ON_SERVER = True

def get_glove_vocab ():

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

	return emb_words

def find_similar (to_search):
	#~ to_search = "answear"

	same_length =  {k: editdistance.eval(to_search, k) for k in freqdist if len(k) == len(to_search)}
	longer = {k: editdistance.eval(to_search, k) for k in freqdist if len(k) == len(to_search)+1}
	shorter = {k: editdistance.eval(to_search, k) for k in freqdist if len(k) == len(to_search)-1}

	ret = {}
	for el, ed in same_length.items():
		if ed < K:
			ret[el] = (ed, freqdist[el])
			
			#~ print ed
			#~ print freqdist[el]
			#~ print freqdist[el]/(ed+1)
			#~ raw_input()
			
			
	for el, ed in longer.items():
		if ed < K:
			ret[el] = (ed, freqdist[el])

	for el, ed in shorter.items():
		if ed < K:
			ret[el] = (ed, freqdist[el])
			
	
	sortedlist = sorted(ret.items(), key = lambda x: x[1][1]/(x[1][0]+1), reverse = True)		
			
	
	ret = to_search
	if len(sortedlist)>0:
		ret = sortedlist[0][0]
		
	return ret
