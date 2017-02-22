# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import collections

class Comment:

	def __init__(self, xml_node, subtask_B = False):
		
		self.attributes= collections.defaultdict(str)

		self.parse(xml_node, subtask_B = subtask_B)

	def parse (self, xml_node, subtask_B = False):
		
		if subtask_B:
			self.attributes["RELQ_ID"] = xml_node.attrib["RELQ_ID"]
			#~ self.attributes["ORGQ_ID"] = "_".join(xml_node.attrib["RELQ_ID"].split("_")[:2])
			self.attributes["ORGQ_ID"] = xml_node.attrib["RELQ_ID"].split("_")[0]
			self.attributes["RELQ_USERNAME"] = xml_node.attrib["RELQ_USERNAME"]
			self.gold_label = xml_node.attrib["RELQ_RELEVANCE2ORGQ"]	
		else:
			self.attributes["RELC_ID"] = xml_node.attrib["RELC_ID"]
			self.attributes["RELQ_ID"] = "_".join(xml_node.attrib["RELC_ID"].split("_")[:2])
			self.attributes["RELC_USERNAME"] = xml_node.attrib["RELC_USERNAME"]
			self.gold_label = xml_node.attrib["RELC_RELEVANCE2RELQ"]
	
	def get (self, attr):
		return self.attributes[attr]
	
	def set_features (self, feats):
		self.feats = feats

	def get_features(self):
		return self.feats

	def setAttribute (self, attrName, value):
		self.attributes[attrName] = value

	def set_id_quest_comm_rel (self, l):
		'''
		Sets a list of (questionBody, commentBody, goldLabel)
		'''
		self.id_quest_comm_rel = l

	def get_id_quest_comm_rel (self):
		return self.id_quest_comm_rel
		
	def get_predicted_values (self):
		
		return (self.rank, self.score, self.predicted_label)
		
	def set_predicted_values (self, label, rank = 0, score=None):
		
		self.predicted_label = label
		#~ self.rank = int(self.attributes["RELC_ID"].split("_")[2][1:])
		self.rank = 0
		#~ self.score = 1.0/self.rank
		
		self.score = 1 if label == "Good" else 0
		
		if score is not None:
			self.score = score
	
	def set_gold_label(self, gold):
		self.gold_label = gold

	def get_gold_label(self):
		return self.gold_label
