import xml.etree.ElementTree as ET
import pickle
from Comment import *
import numpy as np
import pandas as pd
import sys
import os

class FeatureExtractor:

	def __init__(self, fileInput):
		self.threadList = []
		self.pandaStructure = None
		self.t_to_y = {
			"Bad": 0,
			"PotentiallyUseful": 1,
			"Good": 2,
			"PerfectMatch": 2,
			"Relevant": 1,
			"Irrelevant": 0,
			"?": -1
		}
		self.y_to_t = {
			0: "Bad",
			1: "PotentiallyUseful",
			2: "Good",
			-1: "?"

		}
		self.threadList = self.extract_features(fileInput)

	def setIO(self, input, output):
		self.fileInput = input
		self.fileOutput = output

	def buildList(self, fileInput):
		'''
		PRE: Assumes the comments for each thread are consecutive
		Returns a list of threads, each one being a list of Comment objects, where the id_quest_comm_rel field contains a list of (thread_id, questionBody, commentBody, goldLabel)
		'''
		threads = []
		tree = ET.parse(fileInput)
		nodes = tree.iter()
		for node in nodes:
						if node.tag == "OrgQuestion":
								questionBody = node.find("OrgQBody").text
								# print("\n ORGBODY: {} \n\n".format(questionBody))
								for child in node:
										if child.tag == "Thread":
												threads.append(self.__buildCommentList(child, questionBody, fileInput)) # appends a list of Comment objects
		return threads

	def __buildCommentList(self, thread, questionBody, fileInput):
		'''
		PRE: Assumes the comments for each thread are consecutive
		Returns a list of Comment objects
		'''
		questionList = [] # represents a thread question' sequence (TASK B)
		thread_id = thread.get("THREAD_SEQUENCE")
		
		RelquestionNode = thread.find("RelQuestion")
		RelquestionRelevance = RelquestionNode.attrib["RELQ_RELEVANCE2ORGQ"]
		RelquestionUsername = RelquestionNode.attrib["RELQ_USERNAME"]
		RelquestionBody = RelquestionNode.find("RelQClean").text
		RelquestionRanking = RelquestionNode.attrib["RELQ_RANKING_ORDER"]
 #       print(" \n QUESTION RELEVANCE: {} \n REL QUESTION: {} \n".format(RelquestionRelevance, RelquestionBody))        
		
		c = Comment(RelquestionNode, subtask_B = True)
		c.setAttribute("RELQ_USERNAME", RelquestionUsername)
		c.setAttribute("RELQ_RANKING_ORDER", RelquestionRanking)
		c.set_id_quest_comm_rel( (thread_id, questionBody, RelquestionBody, self.t_to_y[RelquestionRelevance]) )
		questionList.append(c)
				
		return questionList
	
	def dumpCommentList(self, outputfile):
		f = open(outputfile,'wb')
		pickle.dump(self.threadList, f)

	def dumpIDQCtoFile(self, fileOutput):
		'''
		PRE: Assumes the comments for each thread are consecutive
		Writes in a file, for each thread, Thread_ID, Question and comments, each one on a different line
		'''
		out_file = open("../data/"+fileOutput, "w")

		prev_thread_id = "None"
		for threadObj in self.threadList:
			for commObj in threadObj:
				el = commObj.id_quest_comm_rel
				curr_thread_id = el[0]

				if curr_thread_id != prev_thread_id: ##TODO with this double loop i can avoid this kind of reasoning
					# Write thread id
					out_file.write(el[0])
					out_file.write("\n")
					# Write question body
					out_file.write(el[1])
					out_file.write("\n")
					prev_thread_id = curr_thread_id
				# Write comment body
				out_file.write(el[2])
				out_file.write("\n")
				# Write the relevance --> int value according to t_to_y dictionary
				#out_file.write(str(el[3]))
				#out_file.write("\n")

		out_file.close()

	def extract_features(self, fileInput):
		threadList = self.buildList(fileInput)
		
		for thread in threadList:
			num_comment = 0
			newThread = []
			users = []
			for comment in thread:
				users.append(comment.attributes["RELQ_USERNAME"])
				que = comment.id_quest_comm_rel[1]
				com = comment.id_quest_comm_rel[2]
				
				num_comment += 1

				
				'''feats = []
				feats.extend(self.len_comments(que, com))
				feats.append(self.longest_common_subsequence(que, com))
				feats.extend(self.jaccard(que, com))
				feats.append(self.same_user(comment))
				#feats.append(num_comment)
				#feats.append(self.has_question(com))
				#feats.append(self.has_url(com))
				'''
				## NEW PART
				feats = {}
				feats["Len_Comments"] = self.len_comments(que, com)
				feats["LCS"] = self.longest_common_subsequence(que, com)
				feats["Jaccard"] = self.jaccard(que, com)
				feats["Same_User"] = self.same_user(comment)
				feats["Comment_number"] = num_comment
				feats["Has_Question"] = self.has_question(com)
				feats["Has_URL"] = self.has_url(com)
				feats["Relevance"] = self.relevance(comment)
				#~ feats["Len_Comments"] = ""
				#~ feats["LCS"] = ""
				#~ feats["Jaccard"] = ""
				#~ feats["Same_User"] = ""
				#~ feats["Comment_number"] = ""
				#~ feats["Has_Question"] = ""
				#~ feats["Has_URL"] = ""
				
				comment.set_features(feats)
				
			users.append(comment.attributes["RELQ_USERNAME"])
			
			for comment in thread:
				features = comment.get_features()
				features["Has_User_In_Body"] = self.has_user_in_body(thread, comment.id_quest_comm_rel[2])
				comment.set_features(features)
			
		return threadList


	def buildPandaStructure(self):
		index = []
		
		parsed_data = pickle.load(open("../data/parsing"+sys.argv[2].lower()+".p", "rb"))
		
		d = {"Len_Comments_abs": [],
				"Thread_ID": [],
				"Comment_ID": [],
				"Q_Body": [],
				"C_Body": [],
				"Q_parsing" : [],
				"C_parsing" : [],
				"Gold_Label": [],
				"Prediction": [],
				"Len_Comments_norm": [], 
				"LCS": [],
				"Jaccard_abs": [],
				"Jaccard_norm": [],
				"Same_User": [],
				"Comment_number": [],
				"Has_Question": [],
				"Has_URL": [],
				"Has_User_In_Body": [],
				"Relevance" : []
		}
		for thread in self.threadList:
			for comment in thread:
				index.append((comment.get("ORGQ_ID"), comment.get("RELQ_ID")))
				f = comment.get_features()
				idqcr = comment.get_id_quest_comm_rel()
				d["Thread_ID"].append(comment.get("ORGQ_ID"))               
				d["Comment_ID"].append(comment.get("RELQ_ID"))
 
				#~ print (comment.get("RELC_ID"))
				#~ print (parsed_data["--ID:"+comment.get("RELC_ID")+"--"])
				#~ input()
 
				#TODO: prendere preprocessing diverso (con spelling correction)
				d["Q_Body"].append(idqcr[1])
				d["C_Body"].append(idqcr[2])
				
				d["Q_parsing"].append(parsed_data["--ID:"+comment.get("ORGQ_ID")+"--"])
				d["C_parsing"].append(parsed_data["--ID:"+comment.get("RELQ_ID")+"--"])
				
				#~ print (idqcr[1])
				#~ print (comment.get("RELQ_ID"))
				#~ print (parsed_data["--ID:"+comment.get("RELQ_ID")+"--"])
				
				#print (idqcr[2])
				#print (comment.get("RELC_ID"))
				#print (parsed_data["--ID:"+comment.get("RELC_ID")+"--"])
				
				#~ input()
				
				d["Gold_Label"].append(comment.get_gold_label())
				d["Prediction"].append("NaN")
				d["Len_Comments_abs"].append(f["Len_Comments"][0])
				d["Len_Comments_norm"].append(f["Len_Comments"][1])
				d["LCS"].append(f["LCS"])
				d["Jaccard_abs"].append(f["Jaccard"][0])
				d["Jaccard_norm"].append(f["Jaccard"][1])
				d["Same_User"].append(f["Same_User"])
				d["Comment_number"].append(f["Comment_number"])
				d["Has_Question"].append(f["Has_Question"])
				d["Has_URL"].append(f["Has_URL"])
				d["Has_User_In_Body"].append(f["Has_User_In_Body"])
				d["Relevance"].append(f["Relevance"])
		
		dF = pd.DataFrame(d, index=index)
		
		return dF

	def len_comments(self, que, com):
		lqb = len(que)
		lcb = len(com)
		return lcb, (lcb / lqb if lqb != 0 else 10 ** 9)

	def longest_common_subsequence(self, que, com):
		lqb = que.split()
		lcb = com.split()
		n = len(lcb)
		m = len(lqb)

		max_sub = 0
		for i in range(n):
			j = 0
			while j < m:
				curr_sub = 0
				pre_i = i
				while i < n and j < m and lcb[i] == lqb[j]:
					curr_sub += 1
					i += 1
					j += 1
				max_sub = max(curr_sub, max_sub)
				if curr_sub != 0:
					i = pre_i
				else:
					j += 1
		return max_sub

	def jaccard(self, que, com):
		lqb = que.split()
		lcb = com.split()

		lqb = set(lqb)
		lcb = set(lcb)

		res = len(lqb.intersection(lcb)) / len(lqb.union(lcb)
											   ) if len(lqb.union(lcb)) != 0 else 10 ** 9
		return len(lqb.intersection(lcb)), res

	def has_question(self, com):
		return '?' in com

	def same_user(self, comNode):
		return comNode.attributes["RELQ_USERNAME"] == comNode.attributes["RELQ_USERNAME"]

	def has_url(self, com):
		return '://' in com or 'www' in com or 'URL' in com

	def has_user_in_body(self, com, users):
		return any(u in com for u in users)
		
	def relevance (self, com):
	
		ret = com.attributes["RELQ_RANKING_ORDER"]
		#~ print (com)
		#~ print ("RELQ_RANKING")
		#~ print (ret)
		#~ input()
		return ret

if __name__ == "__main__":
	
	dataFolder = sys.argv[1]
	print("Starting feature extractor...")
	
	frames = []
	for filename in os.listdir(dataFolder):
		path = os.path.join(dataFolder, filename)
		print("Processing {}".format(path))
		fE = FeatureExtractor(path)
		pandaFrame = fE.buildPandaStructure()
		frames.append(pandaFrame)
	
	result = pd.concat(frames)
	#result.to_csv("../data/"+ "dataForEmbeddings_TASKB" + "PROVA" + ".csv")
	result.to_pickle("../data/" + "dataForEmbeddingsTASKB_" + sys.argv[2] + ".p")
