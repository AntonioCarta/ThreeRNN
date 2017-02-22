
all:	preproc parse toksfinal build generateTriples

preproc:
	cd preproc_parse && python Parser.py "../../data/fileTraining/" "../../data/corpus_training" &&	python Parser.py "../../data/fileDev/" "../../data/corpus_dev" && python Parser.py "../../data/fileTest/" "../../data/corpus_test" && cd ..

parse:
	python parse.py "training"
	python parse.py "dev"
	python parse.py "test"

toksfinal:
	cat ../data/toks_training ../data/toks_dev ../data/toks_test > ../data/toks
	sort -u ../data/toks > ../data/toks.final 
	rm ../data/toks

build: # genera dataForEmbeddings_ARGV[2].p
	python featureExtractor.py "../data/fileTraining/" "TRAINING"
	python featureExtractor.py "../data/fileDev/" "DEV"

buildtest:
	python featureExtractor.py ../data/fileTest/ TEST

buildgoldtest:
	python featureExtractor.py ../data/fileTestGold/ TESTGOLD

generateTriples:
	python generate_new_dataset.py "../data/dataForEmbeddings_TRAINING.p"
	python generate_new_dataset.py "../data/dataForEmbeddings_DEV.p"
	python generate_new_dataset.py ../data/dataForEmbeddings_TEST.p

splitData:
	python split_data.py

generateEncodings:
	python generate_encoding_pandas.py

all2: preproc2 parse toksfinal build2 generateTriples2

build2: 
	python featureExtractorNew.py "../data/fileTraining/" "TRAINING"
	python featureExtractorNew.py "../data/fileDev/" "DEV"

build2test:
	python featureExtractorNew.py ../data/fileTestB/ TEST

build2goldtest:
	python featureExtractorNew.py ../data/fileTestGold/ TESTGOLD

generateTriples2:
	python generate_new_dataset.py "../data/dataForEmbeddingsTASKB_TRAINING.p"
	python generate_new_dataset.py "../data/dataForEmbeddingsTASKB_DEV.p"
	python generate_new_dataset.py ../data/dataForEmbeddingsTASKB_TEST.p
