import random
from numpy.random import choice
from scipy.sparse import csr_matrix,rand
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import math
import sys
import pickle
import os
import time

KMEANS = "kmeans"
KPP = "kpp"
TF_IDF = "tfidf"
TF = "tf"
DEV = "HW2_dev"
TEST = "HW2_test"

inverse_doc_frequency = {}
distances = {}
SET = ""
METRIC = ""

def init_variables(set_value,metric_value):
	global SET, METRIC, distances
	SET = set_value
	METRIC = metric_value
	with open(SET+".docVectors","r") as doc_vec:
		N = len(doc_vec.readlines())

	with open(SET+".df","r") as doc_frequency:
		for line in doc_frequency.readlines():
			df = line.strip("\n").split(":")
			inverse_doc_frequency[df[0]] = math.log(N/float(df[1]))

	if os.path.exists(SET+"_"+METRIC+"_distances.p"):
		distances = pickle.load(open(SET+"_"+METRIC+"_distances.p", "rb" ))


class Document:
	
	def __init__(self,doc_id,size,doc = None,metric = TF):
		self.doc_id = int(doc_id)
		self.size = size
		self.centroid = None
		self.metric = metric
		self.norm = 0.0
		self.vector = None
		if doc != None:
			self.initialize_doc(doc)
	def __str__(self):
		return str(self.vector)

	def similarity(self,centroid):
		# inner = self.vector.dot(centroid.vector).data

		# if len(inner) > 0:
		# 	similarity =  inner/(self.norm*centroid.norm)
		# else:
		# 	similarity = 0.0
		similarity = 1 - pairwise_distances(self.vector, centroid.vector.T,metric = 'cosine')
		# print similarity

		return similarity

	def update_cluster(self,centroid_list):
		s = time.time()
		max_sim = -1
		new_c = None
		for c in centroid_list:
			similarity = self.similarity(c)
			if similarity > max_sim:
				max_sim = similarity
				new_c = c
		e = time.time()
		print e-s

		if self.centroid == None or new_c.c_id != self.centroid.c_id:
			if self.centroid != None:
				self.centroid.docs.remove(self)
			new_c.docs.add(self)
			self.centroid = new_c


		return max_sim

	def get_nonzeros(self):
		return len(self.vector.data)
	def initialize_doc(self,doc_string):
		doc_list = doc_string.strip("\n").split(" ")
		
		data = []
		columns = []
		rows = []
		for d in doc_list:
			d = d.split(":")
			if len(d) > 1:
				columns.append(d[0])
				rows.append(0)
				value = int(d[1])
				if self.metric == TF_IDF:
					value *=inverse_doc_frequency[d[0]]
				data.append(value)

		self.vector = csr_matrix((data, (rows, columns)), shape=(1, self.size))

		self.norm = self.get_norm()

	def get_norm(self):

		return  np.sqrt( self.vector.dot( self.vector.T ).data )[0]

class Centroid:

	def __init__(self,c_id,size,method = KMEANS,documents = [],centroid_list =[]):
		self.size = size
		
		self.method = method
		self.doc_id = -1
		self.vector = self.initialize_centroid(documents,centroid_list)
		self.norm =  self.get_norm()
		self.c_id = c_id
		self.docs = set()

	def initialize_centroid(self,documents =[] ,centroid_list = []):
		if self.method == KMEANS or len(documents) == 0:
			return self.initialize_kmeans(documents)
		elif KPP:
			return self.initialize_kpp(documents,centroid_list)
		else:
			raise Exception("Wrong method!")

	def initialize_kmeans(self,documents):
		return rand(self.size, 1, density=0.01, format='csr')
		
	def distance_to_power_of_two_documents(self,doc_1,doc_2):
		'''Euclidean distance'''
		return doc_1.vector.dot(doc_1.vector.T) - 2 * doc_1.vector.dot(doc_2.vector.T) + doc_2.vector.dot(doc_2.vector.T)

	def initialize_kpp(self,documents,centroid_list):

		if not distances:
			print "Initializing distances"
			for i in range(len(documents)):
				if not i in distances:
					distances[i] = {}
				for j in range(i,len(documents)):
					if not j in distances:
						distances[j]= {}
					dist = self.distance_to_power_of_two_documents(documents[i],documents[j])
					distances[i][j] = dist
			pickle.dump( distances, open( SET+"_"+METRIC+"_distances.p", "wb" ) )
					


		if len(centroid_list) == 0:
			doc = documents[random.randint(0,len(documents))]
			self.doc_id = doc.doc_id
			return doc.vector.copy().T

		v = {}
		doc_dist = [0.0]*len(documents)

		for i,d in enumerate(documents):
			for c in centroid_list:
				dist = distances[min(i,c.doc_id)][max(i,c.doc_id)]

				if doc_dist[i] < dist:
					doc_dist[i] = dist

		new_c = choice(len(documents),1,p=[i / sum(doc_dist) for i in doc_dist])
		self.doc_id = documents[new_c].doc_id
		return documents[new_c].vector.copy().T

	def get_norm(self):
		return np.sqrt( self.vector.T.dot( self.vector).data)[0]


	def update_centroid(self):
		
		if len(self.docs) == 0:#handles empty cluster
			self.vector = self.initialize_centroid()

		else:
			
			value = None
			for doc in list(self.docs):
				if value == None:
					value=doc.vector.copy()
				else:
					value = value + doc.vector
			
			self.vector = (value/len(self.docs)).T


		
		self.norm = self.get_norm()
