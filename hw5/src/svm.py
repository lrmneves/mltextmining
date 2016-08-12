import cPickle as pickle
import math
from sklearn.svm import LinearSVC
print "Loading files..."

dev_labels = pickle.load(open("data/dev_labels.p","rb"))
train_labels = pickle.load(open("data/train_labels.p","rb"))
train_instances_ctf = pickle.load(open("data/train_ctf_matrix.p","rb"))
train_instances_df = pickle.load(open("data/train_df_matrix.p","rb"))
dev_instances_ctf = pickle.load(open("data/dev_ctf_matrix.p","rb"))
dev_instances_df = pickle.load(open("data/dev_df_matrix.p","rb"))

print "Done! Training models on different features!"
ctf_model = LinearSVC().fit(train_instances_ctf,train_labels)
df_model = LinearSVC().fit(train_instances_df,train_labels)

print "Predicting labels..."
train_ctf_labs = ctf_model.predict( train_instances_ctf)
train_df_labs = df_model.predict(train_instances_df)


dev_ctf_labs = ctf_model.predict(dev_instances_ctf)    
dev_df_labs = df_model.predict(dev_instances_df)

print "Persisting models to disk"
pickle.dump(ctf_model,open("svm_ctf_model.p","wb"))
pickle.dump(df_model,open("svm_df_model.p","wb")) 

print "Done!"
#printing results and storing predictions to disk.
rmse = 0.0
right_preds = 0.0
for i in range(len(train_ctf_labs)):
	rmse+= (train_ctf_labs[i] - float(train_labels[i]))**2
	if abs(float(train_ctf_labs[i]) -  float(train_labels[i])) < 0.5:
		right_preds +=1.0
print "RMSE: %.6f"%(math.sqrt(rmse/len(train_labels)))
print "Acc: %.6f"%(right_preds/len(train_labels))

rmse = 0.0
right_preds = 0.0
for i in range(len(train_df_labs)):
	rmse+= (train_df_labs[i] - float(train_labels[i]))**2
	if abs(float(train_df_labs[i]) -  float(train_labels[i])) < 0.5:
		right_preds +=1.0
print "RMSE: %.6f"%(math.sqrt(rmse/len(train_labels)))
print "Acc: %.6f"%(right_preds/len(train_labels))
	
with open("dev_ctf_pred_svm","a") as dev_ctf:
	for n in dev_ctf_labs:
		dev_ctf.write(str(int(round(n))) + " " + str(n) + "\n")
with open("dev_df_pred_svm","a") as dev_df:
        for n in dev_df_labs:
                dev_df.write(str(int(round(n))) + " " + str(n) + "\n")
