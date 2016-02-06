import numpy as np
import sys
import time
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import vstack,csr_matrix
from numpy.random import choice
import math
import random
from eval import *

KMEANS = "kmeans"
KPP = "kpp"
TF_IDF = "tfidf"
TF = "tf"
DEV = "HW2_dev"
TEST = "HW2_test"
inverse_doc_frequency = {}




def initialize_centroids(k,size,documents,method = KMEANS,centroid_list = []):
	if method == KMEANS:
		return initialize_kmeans(k,size,documents)
	elif KPP:
		return initialize_kpp(k,size,documents)
	else:
		raise Exception("Wrong method!")

def initialize_kmeans(k,size,docs):
	'''This function selects random initial points for centroids'''

	centroids = docs[random.randint(0,docs.shape[0]-1),:].copy()
	while centroids.shape[0] < k:
		centroids = vstack([centroids,docs[random.randint(0,docs.shape[0]-1),:].copy()])

	return centroids

def initialize_kpp(k,size,docs):
	'''This function creates centroids by following kpp algorithm, where each new document might be selected as a centroid 
	with probability D(x)**2, being D(x) the distance from the document to the nearest centroid'''
	centroids = docs[random.randint(0,docs.shape[0]-1),:].copy()
	
	while centroids.shape[0] < k:
		distribution = pairwise_distances(docs,centroids ,metric = 'cosine').min(1)
		distribution[distribution < 0] = 0
		distribution = distribution**2

		d_sum = sum(distribution)
		centroids = vstack([centroids,docs[choice(docs.shape[0],1,p=[i /d_sum  for i in distribution]),:].copy()])


	return centroids


def computed_sum_of_intra_similarity(k,docs,centroids, assignments):
	'''Function to evaluate the cluster by summing the intracluster similarity'''
	sim = 0.0
	for j in range(k):

		idxs = np.where(assignments == j)[0]

		cluster_docs = docs[idxs,:]

		centroid = centroids[j,:]

		if idxs.size >0:
			sim += np.sum(1-pairwise_distances(cluster_docs, centroid,metric = 'cosine'))
	return sim


def experiment(k,docs,start,size,method,MAX_IT,it_ave,granularity,TRIES,return_iterations = True):
	'''Run one experiment of kmeans'''

	centroids = initialize_centroids(k,size,documents = docs,method = method)
	count = 0.0
	past_sizes = []
	
	for i in range(MAX_IT):
		count +=1
		
		
		similarity = 1 - pairwise_distances(docs, centroids,metric = 'cosine')	#compute similarities

		assignments = similarity.argmax(axis=1) #assign cluster to documents
		new_centroids = []
		
		centroid_sizes = []

		for j in range(k):
			idxs = np.where(assignments == j)[0] #get idxs of assignments
			if  idxs.size == 0:
				#handles empty cluster
				centroid_sizes.append(idxs.size)
				if new_centroids == []:
					new_centroids = initialize_centroids(1,size,documents = docs,method = method)
				else:
					new_centroids = vstack([new_centroids, initialize_centroids(1,size,documents = docs,method = method)])

			else:
				current = csr_matrix(docs[idxs,:].mean(0))#update centroid

				if new_centroids == []:
					new_centroids = current
				else:

					new_centroids = vstack([new_centroids, current])
			centroid_sizes.append(idxs.size)

		#update centroids matrix
		centroids = new_centroids.copy()

		if past_sizes != []:
			done = True
			#Stopping criteria: If the centroids have not changed sizes between two iterations, stop
			for w in range(len(past_sizes)):

				if past_sizes[w] != centroid_sizes[w] or centroid_sizes[w] == 0:
					past_sizes = centroid_sizes
					done = False
					break
			if done:
				if not return_iterations:
					it_ave[(k-start)/granularity] += count/TRIES
				else:
					return count, centroids
				break
		else:
			past_sizes = centroid_sizes
	if return_iterations:
		return count, centroids
	else:
		it_ave[(k-start)/granularity] += count/TRIES

	


def kmeans(num_clusters, docs,size,method = KMEANS,MAX_IT = 40,TRIES = 10,granularity = 1,working_set = DEV,metric = TF_IDF,start = 2,compute_F1 = False):

	it_ave = [0.0] * (num_clusters-start)


	for k in range(start,num_clusters,granularity):
		print "# of clusters: " + str(k)

		s = time.time()
		for j in range(TRIES):
			experiment(k,docs,start,size,method,MAX_IT,it_ave,granularity,TRIES,return_iterations=False)
		
		e = time.time()
		print "Took " + str((e-s)/TRIES) +"s for each experiment"
		print "Average # of iterations: " + str(int(it_ave[(k-start)/granularity]))



		it, centroids =  experiment(k,docs,start,size,method,MAX_IT,it_ave,granularity,TRIES,return_iterations=True)



		if compute_F1:
			number_of_exp = 10
		else:
			number_of_exp = 1
		sum_of_similarities = 0.0

		maxF1 = -1
		best_assignment = None
		f1_list = []
		s = time.time()
		for exps in range(number_of_exp):

			similarity = 1 - pairwise_distances(docs, centroids,metric = 'cosine')
			assignments = similarity.argmax(axis=1)
			sum_of_similarities +=computed_sum_of_intra_similarity(k,docs,centroids, assignments)



			if compute_F1:
				f1 =  getMacroF1(assignments)
				f1_list.append(f1)
				if f1 > maxF1:
					maxF1 = f1
					best_assignment = assignments
			else:
				best_assignment = assignments
			it, centroids =  experiment(k,docs,start,size,method,MAX_IT,it_ave,granularity,TRIES,return_iterations=True)
		e = time.time()
		print "Compute f1 took " + str(e-s)
		
		if compute_F1:
			mean = np.mean(f1_list)
			std = np.std(f1_list)
			print "Mean f1 was " + str(mean)
			print "STD was " + str(std)
			print "Max f1 was " + str(maxF1)
        
			
        
			with open("sim/"+method +"_"+working_set+"_"+metric+"_sim.txt","a+") as sim:
				sim.write(str(k) + "," + str(sum_of_similarities/number_of_exp)+","+str(mean)+","+str(std)+","+str(maxF1)+"\n")
        
		else:
			with open("sim/"+method +"_"+working_set+"_"+metric+"_sim.txt","a+") as sim:
				sim.write(str(k) + "," + str(sum_of_similarities/number_of_exp)+"\n")

		with open("output/"+method +"_"+working_set+"_"+metric+"_"+str(k)+"_document_file.txt","w") as output:
			
			for i in range(k):

				idxs = np.where(best_assignment == i)[0]
				for d in idxs:
					output.write("%d %d\n"%(d,i))
def main():

	if len(sys.argv) < 4:
		print "Expect method ('kmeans' or 'kpp') set ('HW2_dev' or 'HW2_test') and metric ('tf' or 'tfidf')"
		print sys.argv
		sys.exit()

	METHOD = sys.argv[1].lower()
	SET = sys.argv[2]
	METRIC = sys.argv[3].lower()

	

	if METHOD != KMEANS and METHOD != KPP:
		print "Invalid method."
		sys.exit()
	if SET != DEV and SET != TEST:
		print "Invalid set."
		sys.exit()
	if METRIC != TF and METRIC != TF_IDF:
		print "Invalid metric."
		sys.exit()
	print METHOD
	documents = []
	#Get vocabulary size	
	with open(SET+".dict") as term_dict:
		size = len(term_dict.readlines())

	total = 0
	documents_matrix = None

	

	#Initialize the terms matrix

	with open(SET+".docVectors","r") as doc_vec:
		lines = doc_vec.readlines()
		N = len(lines)

		#Initialize inverse document frequency
		with open(SET+".df","r") as doc_frequency:
			for line in doc_frequency.readlines():
				df = line.strip("\n").split(":")
				inverse_doc_frequency[df[0]] = math.log(N/float(df[1]))

		for doc_string in lines:
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
					if METRIC == TF_IDF:
						value *=inverse_doc_frequency[d[0]]
					data.append(value)

			vector = csr_matrix((data, (rows, columns)), shape=(1, size))
			

			total+=len(vector.data)
			if documents_matrix == None:
				documents_matrix = vector.copy()
			else:
				documents_matrix = vstack([documents_matrix, vector.copy()])
			
	

	
	

	#heuristcs to compute the K taken from https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set 
	start = 2
	if len(sys.argv) > 4:
		start = int(sys.argv[4])
	k = size*documents_matrix.shape[0]/total

	if len(sys.argv) > 5:
		k = int(sys.argv[5])
	
	

	if SET == DEV:
		f1 = True
	else:
		f1 = False
	kmeans(k,documents_matrix,size,method = METHOD,working_set = SET,metric = METRIC,compute_F1 = f1,start = start)


if __name__ == "__main__":
    main()

