import json,string
from scipy.sparse import csr_matrix
import cPickle as pickle
import os
import re
os.path.isfile("test")
class Review:

	def __init__(self,id,user,business,text_count,rating = None):
		self.id = id
		self.user = user
		self.business = business
		self.rating = rating
		self.text_count = text_count
	def create_features(self,ctf_index,df_index):
		ctf_rows = []
		ctf_cols = []
		ctf_data = []

		df_rows = []
		df_cols = []
		df_data = []

		for w in self.text_count:
			if w in ctf_index:
				ctf_data.append(self.text_count[w])
				ctf_cols.append(0)
				ctf_rows.append(ctf_index[w])
			if w in df_index:
				df_data.append(self.text_count[w])
				df_cols.append(0)
				df_rows.append(df_index[w])
		self.ctf_features = csr_matrix((ctf_data, (ctf_rows, ctf_cols)), shape=(len(ctf_index), 1))
		self.df_features = csr_matrix((df_data, (df_rows, df_cols)), shape=(len(df_index), 1))



		

stopwords_set = set()

with open("stopword.list") as stopword_file:
	for w in stopword_file.readlines():
		w=w.strip("\n")
		stopwords_set.add(w)


dictionary_index = {}
document_frequency ={}
total_count = {}
rating_count = {}
translation = string.maketrans("","")
index = 0
review_map = {}
review_index = {}
with open("yelp_reviews_train.json") as train_file:
	for line in train_file.readlines():
		has_appeared_set = set()
		review_json = json.loads(line)
		text = review_json["text"]
		text.split()	
		#removes punctuation and non-ascii after converting to lower
		text = [re.sub('[^a-z0-9]+', '',re.sub(r'[^\x00-\x7f]',r'', w.lower())) for w in text]	
		review_text_count = {}
		for w in text:
			if w.isalpha(): 
				if w in stopwords_set:
					continue
				if not w in total_count:
					total_count[w] = 0
					document_frequency[w] = 0

					
				total_count[w] +=1
				if not w in review_text_count:
					review_text_count[w] = 0

				if not w in has_appeared_set:
					document_frequency[w] +=1
					has_appeared_set.add(w)
				review_text_count[w] +=1

		if not review_json["stars"] in rating_count:
			rating_count[review_json["stars"]] = 0
		rating_count[review_json["stars"]]+=1

		review_map[review_json["review_id"]] = Review(review_json["review_id"],review_json["user_id"],\
			review_json["business_id"], review_text_count,rating = review_json["stars"])

most_frequent = sorted(total_count, key=lambda k: -total_count[k])
print [ (w,total_count[w]) for w in  most_frequent[:15]]
for i in rating_count:
	print i,rating_count[i] ,rating_count[i]*1.0/len(review_map)

top = 2000

most_frequent = most_frequent[:top]
doc_freq_most_frequent = sorted(document_frequency, key=lambda k: -document_frequency[k])[:top]

most_frequent_index = {}
doc_freq_most_frequent_index = {}
for i in range(top):
	most_frequent_index[most_frequent[i]] = i
	doc_freq_most_frequent_index[doc_freq_most_frequent[i]] = i

del most_frequent
del doc_freq_most_frequent
del total_count
del rating_count


for r_id in review_map.keys():
	if not os.path.isfile(("objects/"+r_id+".p")):
		review_map[r_id].create_features(most_frequent_index,doc_freq_most_frequent_index)
		pickle.dump(review_map[r_id], open( "objects/"+r_id+".p", "wb" ))
	del review_map[r_id]








