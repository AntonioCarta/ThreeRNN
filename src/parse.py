#import cPickle as pickle
import pickle
import subprocess as sp
import sys
import codecs
import collections
import editdistance_glove
import nltk

DO_PARSING = True

_TEST_OR_TRAIN = sys.argv[1]

file_input = "../data/corpus_"+_TEST_OR_TRAIN
file_output = "../data/data_parsed_"+_TEST_OR_TRAIN+".conll"

file_toks = "../data/toks_"+_TEST_OR_TRAIN

file_input_cl = "../data/corpus_cleaned_"+_TEST_OR_TRAIN


glove_voc = editdistance_glove.get_glove_vocab()

def clean_tok(t):
	ret = t
	
	if any(c.isalpha() for c in t):
		ret = ''
		for c in t:
			if c.isalpha() or c == "'":
				ret+=c
	return ret

def parse():
	fin = open(file_input_cl)
	fparsed = open(file_output, "w", buffering=1)

	proc = sp.call(["/home/attardi/tools/Tanl/bin/parse.sh"], stdin = fin, stdout = fparsed)	

	fin.close()
	fparsed.close()

def clean():
	substitutions = collections.defaultdict(int)
	file_input_cleaned = codecs.open(file_input_cl, "w", "utf-8")
	with codecs.open(file_input, "r", "utf-8") as f:
		i=0
		for line in f:
			
			i+=1
			if i%1000 == 0:
				print ("riga "+str(i))
				
			if line[:2] == "--" or line == "\n":
				file_input_cleaned.write(line)
			else:
				
				toks = nltk.word_tokenize(line)
				
				new_toks = []
				
				for tok in toks:
					tok = clean_tok(tok)
					if not tok.lower() in glove_voc:
						s = editdistance_glove.find_similar(tok.lower())
						substitutions[(tok.lower(), s)] +=1
						tok = s				
					new_toks.append(tok)
	
				file_input_cleaned.write(" ".join(new_toks)+"\n")
	
	file_input_cleaned.close()

	file_subst = codecs.open("../data/toks_sostituiti_"+_TEST_OR_TRAIN, "w", encoding='utf-8')
	
	for t, s in substitutions:
		file_subst.write(t+"\t"+s+"\t"+str(substitutions[(t,s)])+"\n")

	file_subst.close()


if __name__ == "__main__":


	if DO_PARSING:
		clean()
		parse()

	data_structure = collections.defaultdict(list)

	element = []
	id = "-- ID: Q0_R0_C0 --"

	inid = False

	data_parsed = file_output

	parsing_for_autoencoder = codecs.open("../data/parsing_for_autoencoder_"+_TEST_OR_TRAIN+".p", "w", encoding='utf-8')

	file_toks = codecs.open(file_toks, "w", encoding='utf-8')

	with codecs.open(data_parsed, "r", "utf-8") as fin:

		for line in fin:
			if line == "\n":
				if inid:
					inid = False
				else:
					data_structure[id].append(element)
					element = []
				
				parsing_for_autoencoder.write(line+"\n")	
			
			else:
				splitline = line.split()
				if splitline[1] == "--" and not inid:
					id = splitline[1]
					inid = True
					
				else:
					if inid:
						id += line.split()[1]
					else:
						element.append(line)
						file_toks.write(splitline[1]+"\n")
						
						parsing_for_autoencoder.write(line+"\n")
			

	parsing_for_autoencoder.close()

	fout = open("../data/parsing"+_TEST_OR_TRAIN+".p", "wb")
	pickle.dump (data_structure, fout)

	fout.close()
	file_toks.close()
