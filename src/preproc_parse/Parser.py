import preprocessor_for_parsing as preprocessor
import pickle
import sys

if __name__ == "__main__":
	
	inputDir = sys.argv[1] #"../../data/fileDev/"
	outfile = sys.argv[2]
	print("Starting preprocessing...")
	p = preprocessor.Preprocessor(inputDir, outfile)
	p.preprocess()
	
	
	if sys.argv[2] == "training":
		old_d = {}
	else:
		old_d = pickle.load(open("../../data/dizionario.p", "rb"))
	
	new_d = preprocessor.freqDistDict
	
	for k, v in new_d.items():
		if k in old_d:
			old_d[k]+=new_d[k]
		else:
			old_d[k] = new_d[k]
	
	pickle.dump(old_d, open("../../data/dizionario.p", "wb"))
	
