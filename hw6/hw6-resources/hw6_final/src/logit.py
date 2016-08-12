import cPickle as pickle
import numpy as np
from scipy.sparse import csr_matrix,vstack,rand
from sklearn.utils import shuffle
import sys
import traceback
learning_rate = 10**-5
lambda_value = 0.01

def predict_lr(X,Y,W,indicator,epoch,mode = "Train",show_metrics = True,return_soft = False):
	global learning_rate, lamda_value

	'''Computes metrics and predictions'''

	hip = np.exp(np.dot(X, W)).astype("float64")
	den = np.sum(hip,axis = 1)
	prob = (hip.T/den).T
	stars = np.array(range(0,2))*np.ones(prob.shape)
	
	predictions = np.argmax(prob, axis=1)
	if show_metrics:
        	log_prob = np.log(prob)
		if mode == "Train":
			#If training, print likelihood value
			likelihood = (np.sum(np.multiply(indicator,log_prob)) - (lambda_value/2)*np.sum(np.power(W,2)))
			if isinstance(epoch,int):
				print "Epoch %d likelihood is %.4f"%(epoch,likelihood)
			else:
				print epoch + " likelihood is %.4f"%(likelihood)
		print mode + " Accuracy: %.5f"%(np.sum(predictions == Y)*1.0/predictions.shape[0])
	if return_soft:
		#compute soft prediction as the summation of the probability for each rating times the rating
		stars = np.array(range(0,2))*np.ones(prob.shape)
		soft_predictions = np.sum(np.multiply(stars,prob),axis=1)
		return predictions,soft_predictions
	if mode == "Train":
		return likelihood,predictions
	return predictions

def train_lr(X,labels,W,max_epoch = 100):
	
	for i in range(len(labels)):
		if labels[i] == -1:
			labels[i] = 0
	labels_ind = [w for w in labels]
	indicator = np.zeros((X.shape[0],W.shape[1]))
	indicator[np.arange(len(labels)), labels_ind] = 1
	Y = np.array(labels)
	likelihood,_ = predict_lr(X,Y,W,indicator,"Initial",mode = "Train",show_metrics = True,return_soft = False)
	previous_likelihood = 0

	#Shuffle data to avoid bias
	X,indicator,Y = shuffle(X,indicator,Y)


	batch_size = 100.0
	epoch = 0
	try:
		while epoch < max_epoch and abs(previous_likelihood - likelihood) > 1:
			previous_likelihood = likelihood
			for i in range(int(X.shape[0]/batch_size)):
				#update weights per batch
				batch_start = int(i*batch_size)
				batch_end = int(min((i+1)*batch_size, X.shape[0]))
				currX = X[batch_start:batch_end,:]
				currY = indicator[batch_start:batch_end]
				hip = np.exp(np.dot(currX,W)).astype("float64")
				den = np.sum(hip,axis = 1)
				prob = (hip.T/den).T
				W = W + learning_rate*(np.dot(currX.T,(currY - prob)) - lambda_value*W)
			
			likelihood,_ = predict_lr(X,Y,W,indicator,epoch,mode = "Train",show_metrics = True)
			epoch +=1
		return W[:,1] - W[:,0]
	except Exception:
		traceback.print_exc()