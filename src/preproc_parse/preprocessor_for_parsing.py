# -*- coding: utf-8 -*-

import os
import nltk
import nltk.data
from nltk.corpus import wordnet

import time
import re
import collections
from collections import Counter

import xml.etree.ElementTree as ET

import codecs
# Spell correction (eventualmente)
# corpus del SE, con frequenze



#~ outfile = sys.argv[2]
#~ outfile_noid = codecs.open("corpus_noid", "w", "utf-8")

freqDistDict = collections.defaultdict(int)

class PreprocessModule:
	def process(self, text):
		raise Exception('Implement process function!')


class Pipeline:
	def __init__(self, modules=[PreprocessModule]):
		self.modules = modules

	
	def processText(self, node):
		result = "[NODO VUOTO]"
		if node.text != None:
			
			result = node.text
				
			for module in self.modules:
				result = module.process(result)
		else:
			print (node)
		
		
		#~ outfile.write(result+"\n")
		return result
			

class NoiseRemover(PreprocessModule):    
	def __init__(self):
		self.c_urlRE = re.compile(r"http[s]?://[www\.]?([^\s\.]*)\.[^\s]*|www.([^\s\.]*)\.[^\s]*")
		self.c_emailRE = re.compile(r"[^\s]+@[^\s]+\.[^\s]+")
		self.c_nickRE = re.compile(r"@[^\s]+")
		self.c_numbers = re.compile(r"[\d]+")

		self.c_img = re.compile(r"\[ ?img[^\]\n]+[\]$]")
		
		self.c_multiplepunctuation = re.compile(r'([!?:;_*#\-"^+])( ?\1)+')
		
		self.c_multipleletters = re.compile(r'([A-z])\1{2,}')
		
		self.c_puntinidisospensione = re.compile(r"\.{4,}")
		
		self.c_middlestop = re.compile(r"([A-Za-z]{2,}[.,:?!;\(\)]+)([A-Za-z]+)")
		
		#self.c_smiles = re.compile(r"\s[:;]-?\S\W|^[:;]-?\S\W|\s[:;]-?\S$")

		self.c_puntoevirgola = re.compile(r";")
		self.c_dueslash = re.compile(r" // ")

		self.c_slashtraparole = re.compile(r"([A-z]{2,})/([A-z]{2,})")
		
	def process(self, text):
		# Note: this sequence MUST be respected and there MUST be spaces before and after the replacement text
		t=text
		
		t = re.sub(self.c_img, "", t)
		
		subs_made = 1
		while not subs_made == 0:
			t, subs_made = re.subn(self.c_urlRE, " url ", t)
			
		subs_made = 1
		while not subs_made == 0:
			t, subs_made = re.subn(self.c_emailRE, " email ", t)
			
		subs_made = 1
		while not subs_made == 0:
			t, subs_made = re.subn(self.c_nickRE, " nickname ", t)
		
		t = re.sub(self.c_numbers, " 0000 ", t)
		
		

		subs_made = 1
		while not subs_made == 0:

			t, subs_made = re.subn(self.c_middlestop, r"\1 \2", t)
		
		t = re.sub(self.c_dueslash, "\n", t)
		
		t= re.sub(self.c_multiplepunctuation, r"\1", t)
		
		t= re.sub(self.c_multipleletters, r"\1", t)
		
		subs_made = 1
		while not subs_made == 0:
			t, subs_made = re.subn(self.c_puntinidisospensione, "...", t)

		#t = re.sub(self.c_smiles, " ", t)
		
		t = re.sub(self.c_puntoevirgola, ",", t)
		
		subs_made = 1
		while not subs_made == 0:
			t, subs_made = re.subn(self.c_slashtraparole, r" \1 or \2 ", t)
		
		
		return t

class Expander (PreprocessModule):
	def __init__(self):
		
		self.dict_abbrevs = {
			"pls": "please",
			"plz": "please",
			"u": "you",
			"r": "are",
			"ur": "your",
			"yr": "your",
			"dont": "don't",
			"w/": "with",
			"yup": "yes",
			"nope": "no",
			"some1": "someone",
			"bt": "but",
			"nt": "not",
			"dnt": "don't",
			"dont": "don't",
			"doesnt": "doesn't",
			"thx": "thanks",
			"im": "i'm",
			"abt": "about",
			"imho": "in my humble opinion",
			"you'r": "you're",
			"cos": "because",
			"coz": "because"
		}

	def process (self, text):
		
		sentlist = text.split("\n")
		
		ret = []
		for sent in sentlist:
			
			tokenslist = nltk.word_tokenize(sent)
			
			for t in tokenslist:
				lowert = t.lower()
				
				freqDistDict[lowert]+=1
				
				if lowert in self.dict_abbrevs:
					ret.append(self.dict_abbrevs[lowert])
				else:
					ret.append(t)
					
			ret.append("\n")
		
		return " ".join(ret)
		

class Preprocessor:
	
	def __init__(self, inputFolder, outfile):
		self.inputFolder = inputFolder
		self.outfile = open(outfile, "w")
		#self.fileOutput = fileOutput
	
	def setIO(self, inputFolder, output):
		self.inputFolder = inputFolder
		#self.fileOutput = output
	
	def preprocess(self):
		#print(self.inputFolder)
		for fn in os.listdir(self.inputFolder):
			print(fn)
			tree = ET.parse(self.inputFolder + fn)
			root = tree.getroot()
				
			noiseRemover = NoiseRemover()
			abbrevsExpander = Expander()
				
				## BUILD THE PIPELINE
			pipeline = Pipeline([
								noiseRemover, # returns text
								abbrevsExpander
							])

			# Check if OrgQuestion exists, else preprocess a sequence of threads
			found = False
			for orgQuestion in root.findall('OrgQuestion'):
				found = True
				
				ide = orgQuestion.get("ORGQ_ID")
				
				orgQClean = orgQuestion.find('OrgQClean')
				orgQClean = pipeline.processText(orgQClean)
				
				self.outfile.write("--- ID: "+ide+" ---\n\n")
				
				listorgQClean = orgQClean.split("\n")
				sentences = []
				for s in listorgQClean:
					sentences+=nltk.sent_tokenize(s)
				
				
				for s in sentences:
					self.outfile.write(s+"\n\n")
					#~ self.outfile_noid.write(s+"\n\n")
				
				for thread in orgQuestion.findall("Thread"):
					ide = orgQuestion.get("THREAD_SEQUENCE")
					self.__preprocessThread(thread, pipeline)

			if not found: # There is only a list of threads!
				threads = root.findall("Thread")
				for thread in threads:
					self.__preprocessThread(thread, pipeline)
		
	def __preprocessThread(self, thread, pipeline):
		for question_comment_node in list(thread):     
			
			#~ print(question_comment_node)
			#~ print(question_comment_node.tag)
			#~ input()
			
			#~ if len(question_comment_node) == 3:
			if question_comment_node.tag == "RelQuestion":
				ide = question_comment_node.get("RELQ_ID")
				
				
			#~ elif len(question_comment_node) == 2:
			elif question_comment_node.tag == "RelComment":
				ide = question_comment_node.get("RELC_ID")
			
			
			else:
				print ("C'è qualcosa che non torna con il numero di elementi")
				
			nodeToPreprocess = list(question_comment_node)[-1]			
	
			if nodeToPreprocess != None:			
				outputNode = pipeline.processText(nodeToPreprocess)
				
				self.outfile.write("--- ID: "+ide+" ---\n\n")
				
				listoutputNode = outputNode.split("\n")
				
				sentences = []
				for s in listoutputNode:
					sentences+=nltk.sent_tokenize(s)
							
				for s in sentences:
					self.outfile.write(s+"\n\n")
					#~ self.outfile_noid.write(s+"\n\n")
