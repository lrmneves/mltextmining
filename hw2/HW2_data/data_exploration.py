def print_stats(path, vec_set_string):

	with open(path,"r") as doc_vec:
		
		text= doc_vec.readlines()
		if text[-1] == "\n" or text[-1] == "":
			doc_count = len(text) - 1
		else:
			doc_count = len(text)
		#get unique words
		term_set = set()
		#count the number of words
		total_words = 0
		#get unique words per document
		unique_per_doc = 0
		#variables for the first document stats
		first_doc = True
		unique_words = set()
		two_timers = list()
		for line in text:
			doc = line.strip("\n").split(" ")				
				
			for term_freq in doc:
				term_freq = term_freq.split(":")
				if len(term_freq) > 1:
					term_set.add(term_freq[0])
					total_words += int(term_freq[1])
					unique_per_doc+=1
					
					if first_doc:
						if int(term_freq[1]) == 2:
							two_timers.append(term_freq[0])
						unique_words.add(term_freq[0])

			if first_doc:
				first_doc = False			

		print "==== %s ===="%(vec_set_string.title())
		print "There are %d documents, %d words and %d unique terms on the %s, averaging"\
		" %d unique terms per document"%(doc_count,total_words,len(term_set),vec_set_string,unique_per_doc/(1.0*doc_count))
		print "The word ids that occur twice on the first doc are " + ",".join(two_timers) + " and there are %d " \
		"unique terms on it"%(len(unique_words))

def main():
	print_stats("HW2_dev.docVectors", "development set")
	print_stats("HW2_test.docVectors", "test set")
	
 


if __name__ == "__main__":
    main()