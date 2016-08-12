import cPickle as pickle
import numpy as np
from scipy.sparse import csr_matrix,vstack,rand
from sklearn.utils import shuffle
import sys
import traceback
from sklearn.metrics import confusion_matrix
learning_rate = 10**-5
lambda_value = 0.01

def predict(X,Y,W,indicator,epoch,mode = "Train",show_metrics = True,return_soft = False):
	global learning_rate, lamda_value

	'''Computes metrics and predictions'''

	hip = np.exp(X * W).astype("float64")
    den = np.sum(hip,axis = 1)

    prob = (hip.T/den).T
	stars = np.array(range(1,6))*np.ones(prob.shape)
	
	predictions = np.argmax(prob, axis=1) + 1
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
		print mode +" RMSE: %.5f"%(np.sqrt(np.sum((predictions - Y)**2)*1.0/predictions.shape[0]))
	if return_soft:
		#compute soft prediction as the summation of the probability for each rating times the rating
		stars = np.array(range(1,6))*np.ones(prob.shape)
		soft_predictions = np.sum(np.multiply(stars,prob),axis=1)
		return predictions,soft_predictions
	if mode == "Train":
		return likelihood,predictions
	return predictions

#usage: python logit.py <feature_type: ctf,df or tfidf> <lr> <lambda> <mode: train, test or dev>
mode = "train"
feature_type = sys.argv[1]
if len(sys.argv) > 2:
	learning_rate = float(sys.argv[2])
	lambda_value = float(sys.argv[3])
if len(sys.argv) > 4:
	mode = sys.argv[4]


print "Loading matrix"
instance_matrix = pickle.load(open("data/"+mode+"_"+feature_type+"_matrix.p"))
labels = pickle.load(open("data/"+ mode + "_labels.p"))
X = instance_matrix
Y = np.array(labels)
#Saves original value to compute final accuracy
original_X = X.copy()
original_Y = Y.copy()

if len(sys.argv)<=4:
	print "Starting training phase"
	W = np.zeros((X.shape[1], len(set(labels)))).astype("float64")
	labels_ind = [w -1 for w in labels]
	indicator = np.zeros((X.shape[0],W.shape[1]))
	indicator[np.arange(len(labels)), labels_ind] = 1
	
	likelihood,_ = predict(X,Y,W,indicator,"Initial",mode = "Train",show_metrics = True,return_soft = False)
	previous_likelihood = 0

	#Shuffle data to avoid bias
	X,indicator,Y = shuffle(X,indicator,Y)
	"Splitting training and validation sets..."

	#Separate 10% of data for validation
	valid_size = X.shape[0]/10
	X_valid = X[:valid_size]
	X = X[valid_size:]

	Y_valid = Y[:valid_size]
	Y = Y[valid_size:]
	indicator = indicator[valid_size:]


	batch_size = 1000.0
	epoch = 0
	try:
		while epoch < 2000 and abs(previous_likelihood - likelihood) > 1:
			previous_likelihood = likelihood
			for i in range(int(X.shape[0]/batch_size)):
				#update weights per batch
				batch_start = i*batch_size
				batch_end = min((i+1)*batch_size, X.shape[0])
				currX = X[batch_start:batch_end,:]
				currY = indicator[batch_start:batch_end]
				hip = np.exp(currX * W).astype("float64")
				den = np.sum(hip,axis = 1)
				prob = (hip.T/den).T
				W = W + learning_rate*(currX.T*(currY - prob) - lambda_value*W)
			
			likelihood,_ = predict(X,Y,W,indicator,epoch,mode = "Train",show_metrics = True)
			
			predict(X_valid,Y_valid,W,indicator,epoch,mode = "Validation",show_metrics = True)

			epoch +=1
	except Exception:
		traceback.print_exc()
		pass
	print "Took %d epochs"%(epoch)
	pickle.dump(W,open(feature_type +"_model.p","wb"))
else:
	#If not training, load model from disk for prediction and give dummy value to training variables
	W = pickle.load(open(feature_type +"_model.p","rb"))
	indicator = None

X = original_X
Y = original_Y
#compute predictions and confusion matrix and persist results to disk.
pred, soft_pred = predict(X,Y,W,indicator,"Final",mode = mode,show_metrics = True,return_soft = True)
print confusion_matrix(Y,pred,labels=range(1,6))
with open(mode + "_" + feature_type +"_pred.txt", "a") as pred_file:
	for i in range(pred.shape[0]):
		pred_file.write(str(pred[i]) +" "+str(soft_pred[i])+"\n")


