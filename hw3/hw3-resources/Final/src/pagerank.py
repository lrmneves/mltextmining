import os
import sys
from sklearn.preprocessing import normalize
from scipy.sparse import csc_matrix
from scipy import sparse
import numpy as np
import time
import warnings
import math


def get_topic_dict(path):
	'''reads file on path and returns dict of dicts, with main key user, secondary key query and value = topic prob'''

	topic_dict = {}

	with open(path) as t_file:
		lines = t_file.readlines()
		for line in lines:
			current = line.split(" ")
			if not int(current[0]) in topic_dict:
				topic_dict[int(current[0])] = {}
			data = []
			for i in range(2,len(current)):
				data.append(float(current[i].split(":")[1]))
			topic_dict[int(current[0])][int(current[1])] = np.array(data)

	return  topic_dict

def pagerank(r,M,p,alpha,beta,gamma, p0):

	transposed_M = alpha*M.T
	for it in range(10):

		old_r = r

		r = transposed_M.dot(old_r) + beta*p + gamma*p0

	return r
def main():
	path = ""
	method = ""
	alpha= 0.8
	beta_factor = 0.2
	beta = (1-alpha)*beta_factor
	warnings.filterwarnings("ignore")

	pr_weight = 1000
	relevance_weight = 0.5
	if len(sys.argv) > 1:
		method = sys.argv[1]
	if len(sys.argv) > 2:
		path = sys.argv[2]
	if len(sys.argv) > 3:
		alpha = float(sys.argv[3])
		beta_factor = float(sys.argv[4])
	if len(sys.argv) > 5:
		pr_weight = float(sys.argv[5])
		relevance_weight = float(sys.argv[6])

	if path == "":
		beta_factor = 0.0

	#initialize matrix M
	with open("transition.txt","r") as transitions:

		lines = transitions.readlines()
		data = []
		columns = []
		rows = []
		size = 0
		for line in lines:
			t = line.split(" ")

			if len(t) > 1:
				if int(t[0])-1 != int(t[1])-1:
					rows.append(int(t[0])-1)
					columns.append(int(t[1])-1)

				if len(t) > 2:
					data.append(int(t[2]))
				else:
					data.append(1)
				if size < max(int(t[0]),int(t[1])):
					size = max(int(t[0]),int(t[1]))
		#adds diagonal
		rows += range(size)
		columns += range(size)
		data += np.ones(size).tolist()

		M = csc_matrix((data, (rows, columns)), shape=(size, size), dtype="float64")

		M = normalize(M, norm='l1', axis=1)

	with open("doc_topics.txt") as topics:
		lines = topics.readlines()
		topics_dict = {}

		for line in lines:
			current = line.split(" ")

			if not int(current[1]) in topics_dict:
				topics_dict[int(current[1])] = []
			topics_dict[int(current[1])].append(int(current[0])-1)

	tspr_dict = {}

	for t in topics_dict:
		rows = topics_dict[t]
		columns = np.zeros(len(rows))
		data = np.ones(len(rows))/len(rows)
		tspr_dict[t] = csc_matrix((data, (rows, columns)), shape=(size,1))


	#initialize variables

	gamma = (1-alpha)*(1-beta_factor)


	p0 = (1.0/size)*np.ones((size,1))
	r_init = (1.0/size)*np.ones((size,1))

	if path == "":
		t_dict = get_topic_dict("query-topic-distro.txt")
	else:
		t_dict = get_topic_dict(path)


	#this part assumes indri-lists folder is unzipped on this directory

	#Gets the documents for each query and their relevance scores.
	query_relevance_dict = {}
	for root, dirs, files in os.walk("indri-lists"):
		for f in files:
			if not f.endswith(".results.txt"):
				continue
			query = f.split(".")[0]
			query_relevance_dict[query] = {}
			with open(os.path.join(root, f)) as current_file:
				for line in current_file.readlines():
					doc = line.split(" ")
					query_relevance_dict[query][int(doc[2])] = float(doc[4])


	s = time.time()
	r_cache = {}
	for t in topics_dict:
		r_cache[t] = pagerank(r_init.copy(),M,tspr_dict[t],alpha,beta,gamma,p0.copy())
	e = time.time()
	base_time = e-s
	num_queries = 0
	if os.path.isfile("trec.txt"):
		os.remove("trec.txt")
	for u in t_dict:
		for q in t_dict[u]:
			s = time.time()
			query = str(u) + "-" + str(q)
			final_r = np.zeros((size,1))
			for t in topics_dict:
				term = (r_cache[t]*t_dict[u][q][t-1])
				final_r = final_r + term

			result = final_r.tolist()


			if method != "":
				for d in query_relevance_dict[query]:
					if method == "WS":
						result[d -1] = [pr_weight * result[d-1][0] + relevance_weight*query_relevance_dict[query][d]]

					elif method == "CM":
						result[d -1] = [-(pr_weight + relevance_weight)/(pr_weight/math.log(result[d-1][0]) \
																	+relevance_weight/(math.log(abs(query_relevance_dict[query][d]))))]
					else:
						break

			ranking = sorted(range(len(result)),key=lambda x:result[x],reverse = True)
			e = time.time()
			base_time +=e-s
			num_queries +=1

			exp = "mltxt"
			rank_value = 1

			with open("trec.txt","a+") as trec:
				for i in range(len(ranking)):
					if ranking[i]+1 in query_relevance_dict[query]:
						trec.write(str(query) +" Q0 " + str(ranking[i]+1) + " " + str(rank_value) + " " + \
								'{:0.8f}'.format(result[ranking[i]][0]) + " " + exp +"\n")

						rank_value+=1


	print "Computation took " +str(base_time/num_queries) + " per query"


if __name__ == "__main__":
    main()





