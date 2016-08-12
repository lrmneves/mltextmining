import json,string
from scipy.sparse import csr_matrix,vstack
import cPickle as pickle
import os
import sys
from math import log
import re
from nltk.stem.porter import PorterStemmer
class Review:
	'''Class to store review values'''
	def __init__(self,id,user,business,text_count,rating = None):
		self.id = id
		self.user = user
		self.business = business
		self.rating = rating
		self.text_count = text_count
	def create_features(self,ctf_index,df_index,mi_index,ctf_rows,ctf_cols,ctf_data,df_rows,df_cols,df_data,tfidf_rows,tfidf_cols,tfidf_data,row_id,document_frequency,total_docs):
		for w in self.text_count:
			#for each feature type, initialize values based on index
			if w in ctf_index:
				ctf_data.append(self.text_count[w])
				ctf_rows.append(row_id)
				ctf_cols.append(ctf_index[w])
				
			if w in df_index:
				df_data.append(self.text_count[w])
				df_rows.append(row_id)
				df_cols.append(df_index[w])
			if w in mi_index:
                tfidf_rows.append(row_id)
                tfidf_cols.append(mi_index[w])
                tfidf_data.append(self.text_count[w] * log(total_docs*1.0/document_frequency[w]))

stopwords_set = set()
print "Creating stopword list"
with open("stopword.list") as stopword_file:
	for w in stopword_file.readlines():
		w=w.strip("\n")
		stopwords_set.add(w)

review_map = {}
total_count = {}
document_frequency ={}
rating_count = {}
index = 0
review_index = {}
word_class_count = {}
mode = sys.argv[1]
top = 2000
total_docs = 0
review_ids = []
stemmer =  PorterStemmer()


print "Loading " + mode + " json"

with open("yelp_reviews_"+mode+".json") as train_file:
	for line in train_file.readlines():
		total_docs+=1
		has_appeared_set = set()
		review_json = json.loads(line)
		text = review_json["text"]
		text = text.split()
		#removes punctuation and non-ascii after converting to lower
		text = [re.sub('[^a-z0-9]+', '',re.sub(r'[^\x00-\x7f]',r'', w.lower())) for w in text]	
		review_text_count = {}
		
		for w in text:
			if w.isalpha():
				#stemming only used for final result 
				w = stemmer.stem(w)
				if w in stopwords_set:
					continue
				#initialize word counts
				if not w in total_count:
					total_count[w] = 0
					document_frequency[w] = 0
				
				#initialize counting of word per class to compute mutual information
				if not w in word_class_count:
					word_class_count[w] = {}								
				total_count[w] +=1

				if not w in review_text_count:
					review_text_count[w] = 0
				
				if "stars" in review_json:
                	#if training phase, initialize word/class count and increment for each document of that class
                	#the word appears
                    if not review_json["stars"] in word_class_count[w]:
                            word_class_count[w][review_json["stars"]] = 0
                    if not w in has_appeared_set:
                            word_class_count[w][review_json["stars"]]+=1
				#increment document frequency
				if not w in has_appeared_set:
					document_frequency[w] +=1
					has_appeared_set.add(w)
				review_text_count[w] +=1
		if "stars" in review_json: 
			if  not review_json["stars"] in rating_count:
				rating_count[review_json["stars"]] = 0
			rating_count[review_json["stars"]]+=1
		#keeps track of the order of ids
		review_ids.append(review_json["review_id"])

		review_map[review_json["review_id"]] = Review(review_json["review_id"],review_json["user_id"],\
			review_json["business_id"], review_text_count,rating = (review_json["stars"] if "stars" in review_json else 1)   )
most_frequent = sorted(total_count, key=lambda k: -total_count[k])
print [ (w,total_count[w]) for w in  most_frequent[:15]]

if not os.path.isfile("ctf.p"):
	print "Computing frequencies and index dictionaries"
	mutual_info = {}
	samples = int(top*1.0/ len(rating_count))
	informative_set = set()
	#print rating counts and compute pairwise word/class mutual information. Set to 0 if word does not cooccurr with class
	for i in rating_count:
		print i,rating_count[i] ,rating_count[i]*1.0/len(review_map)
		mutual_info[i] = []
		for w in document_frequency:
			if not i in word_class_count[w]:
				mi = 0.0
			elif word_class_count[w][i] == 0:			
				mi = 0.0
			else:
				mi = word_class_count[w][i]*1.0/total_docs * log((word_class_count[w][i]*1.0/total_docs)/((document_frequency[w]*1.0/total_docs)*(rating_count[i]*1.0/total_docs)),2)
			mutual_info[i].append((mi,w))
		current_list = mutual_info[i]
		most_informative = sorted(current_list, key=lambda k: -k[0])
		count_info = 0
		curr_indx = 0
		#sample top informative words from each class
		while not count_info == samples and len(informative_set) < top and curr_indx < len(most_informative):
			curr= most_informative[curr_indx][1]
			if not curr in informative_set:
				informative_set.add(curr)
				count_info+=1
			curr_indx +=1
		
	print len(informative_set)		
	#create ctf and df features
	most_frequent = most_frequent[:top]
	doc_freq_most_frequent = sorted(document_frequency, key=lambda k: -document_frequency[k])[:top]

	most_frequent_index = {}
	doc_freq_most_frequent_index = {}
	most_informative_index = {}
	informative_list = list(informative_set)
	#create index for the features and persist it to disk so dev and test can use same index

	for i in range(top):
		most_frequent_index[most_frequent[i]] = i
		doc_freq_most_frequent_index[doc_freq_most_frequent[i]] = i
		most_informative_index[informative_list[i]] = i
	#clear some memory for my poor 4GB RAM computer
	del most_frequent
	del doc_freq_most_frequent
	del total_count
	del rating_count
	pickle.dump(most_frequent_index,open("ctf.p","wb"))
	pickle.dump(doc_freq_most_frequent_index,open("df.p","wb"))
	pickle.dump(most_informative_index,open("mi.p","wb"))
else:
	print "Existing Dictionary Found. Loading existing one"
	most_frequent_index = pickle.load(open( "ctf.p", "rb" ))
	doc_freq_most_frequent_index = pickle.load(open( "df.p", "rb" ))
	most_informative_index = pickle.load(open( "mi.p", "rb" ))
labels = []
features_ctf_matrix = None
features_df_matrix = None
features_tfidf_matrix = None
ctf_rows = [] 
ctf_cols = [] 
ctf_data = [] 
df_rows = [] 
df_cols = []
df_data = [] 
tfidf_rows = [] 
tfidf_cols = [] 
tfidf_data = []
total_docs = len(review_map)

print "Creating feature and label vectors"

row_id = 0
#create sparse matrices for all feature types and persist to disk
for r_id in review_ids:
	review_map[r_id].create_features(most_frequent_index,doc_freq_most_frequent_index,most_informative_index,ctf_rows,ctf_cols,ctf_data,df_rows,df_cols,df_data,tfidf_rows,tfidf_cols,tfidf_data,row_id,document_frequency,total_docs)
	row_id+=1
	if review_map[r_id].rating == None:
		labels.append(1)
	else:
		labels.append(review_map[r_id].rating)
	
	del review_map[r_id]
features_ctf_matrix = csr_matrix((ctf_data, (ctf_rows, ctf_cols)), shape=(row_id, top))
features_df_matrix = csr_matrix((df_data, (df_rows, df_cols)), shape=(row_id, top))
features_tfidf_matrix = csr_matrix((tfidf_data, (tfidf_rows, tfidf_cols)), shape=(row_id, top))


pickle.dump(features_ctf_matrix,open("data/"+mode+"_ctf_matrix.p","wb"))
pickle.dump(features_df_matrix,open("data/"+mode+"_df_matrix.p","wb"))
pickle.dump(features_tfidf_matrix,open("data/"+mode+"_tfidf_matrix.p","wb"))
pickle.dump(labels,open("data/"+mode+"_labels.p","wb"))
