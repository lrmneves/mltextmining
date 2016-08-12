import cPickle as pickle
import numpy as np
from scipy.sparse import csr_matrix,vstack,rand
import sys

feature_type = sys.argv[1]

instance_matrix = pickle.load(open("data/train_"+feature_type+"_matrix.p"))
labels = pickle.load(open("data/train_labels.p"))

bias = csr_matrix((1,instance_matrix.shape[1]))
X = vstack([bias, instance_matrix])
subset = np.random.choice(X.shape[0], 1000, replace=False)

Y = np.array(labels)
W = np.random.rand(X.shape[1], len(set(labels))).astype("float64")
W = W/1000.0

hip = np.exp(X * W).astype("float64")
den = np.sum(hip,axis = 0)

prob = hip/den

labels_ind = [w -1 for w in labels]
indicator = np.zeros(prob.shape)
indicator[np.arange(1,len(labels)+1), labels_ind] = 1

learning_rate = 0.01 
lambda_value = 1
likelihood = (np.sum(np.multiply(indicator,log_prob)) - lambda_value*np.sum(np.multiply(W**2))

previous_likelihood = 0

batch_size = 100.0
epoch = 0
while epoch < 1000 and abs(previous_likelihood - likelihood) < 0.001:
	previous_likelihood = likelihood

	for i in range(int(X.shape[0]/batch_size)):
		batch_start = i*batch_size
		batch_end = min((i+1)*batch_size, X.shape[0])
		currX = X[batch_start:batch_end,:]
		currY = indicator[batch_start:batch_end]
		hip = np.exp(currX * W).astype("float64")
		den = np.sum(hip,axis = 0)
		prob = hip/den

		W = W + learning_rate*(currX.T*(currY - prob)/batch_size - lambda_value*W)
	likelihood = (np.sum(np.multiply(indicator,log_prob)) - lambda_value*np.sum(W**2))
	print "Epoch %d likelihood is %.4f"%(epoch,likelihood)


