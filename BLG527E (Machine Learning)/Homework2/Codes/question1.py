import csv
import numpy as np
from numpy.linalg import inv
import math

results = []
test_results = []


def liner():
	print("------------------------------")

print("Getting training data..")
with open("optdigits.tra") as csvfile:
    reader = csv.reader(csvfile) # change contents to floats
    for row in reader: # each row is a list
        results.append(row)

liner()

print("Getting test data..")
with open("optdigits.tes") as csvfile:
    reader = csv.reader(csvfile) # change contents to floats
    for row in reader: # each row is a list
        test_results.append(row)

liner()

feature_cache = []
features =[]
labels = []

features_test =[]
labels_test = []

print("Creating matrices for the data..")
for i in range(len(np.asarray(results).T)): # getting training data
	for j in range(len(results)):
		feature_cache.append(results[j][i])
	if (i<len(np.asarray(results).T)-1):
		features.append(feature_cache)
	else:
		labels=feature_cache
	feature_cache = []

for i in range(len(np.asarray(test_results).T)): # getting test data
	for j in range(len(test_results)):
		feature_cache.append(test_results[j][i])
	if (i<len(np.asarray(test_results).T)-1):
		features_test.append(feature_cache)
	else:
		labels_test=feature_cache
	feature_cache = []

labels = list(map(int, labels))
labels_test = list(map(int, labels_test))
to_delete=[]
variances=[]

liner()

print("Searching unnecessary features..")

liner()
for i in range(64): # find and delete unnecessary features
	features[i] = list(map(int, features[i]))
	features_test[i] = list(map(int, features_test[i]))
	variance = np.var(features[i])
	if (variance==0):
		print("Feature "+str(i)+" has zero variance!")
		to_delete.append(i)
	else:
		variances.append(variance)


dcount =0

liner()

for x in to_delete:
	print("Feature "+str(x)+" is deleting..")
	features.pop(x-dcount)
	features_test.pop(x-dcount)
	dcount = dcount+1

liner()

print("Calculating base covariance, mean matrices etc..")

liner()
priors = [0,0,0,0,0,0,0,0,0,0]

m_i = np.zeros((10, 62))
m_general = np.zeros(62)
s_i = np.zeros((10, 62, 62))
s_b = np.zeros((62, 62))
s_common = np.zeros((62, 62))
s_w = np.zeros((62, 62))

for t in range(3823):
	priors[labels[t]]=priors[labels[t]]+(1/3823)


priors= np.array(priors)
variances= np.array(variances)
features= np.array(features)
labels= np.array(labels)
features_test= np.array(features_test)
labels_test= np.array(labels_test)

for cl_num in range(10): # Calculating m_i
	temp_labels = (labels == cl_num).astype(int)
	for t in range(3823):
		m_i[cl_num] = m_i[cl_num]+features[:,t]*temp_labels[t]
	m_i[cl_num]=m_i[cl_num]/np.sum(temp_labels)
	m_general = m_general + m_i[cl_num]/10

for cl_num in range(10): # Calculating S_i
	temp_labels = (labels == cl_num).astype(int)
	for t in range(3823):
		temp_array = features[:,t]-m_i[cl_num]
		s_i[cl_num] = s_i[cl_num]+np.outer(temp_array,temp_array)*temp_labels[t]
	s_i[cl_num]=s_i[cl_num]/np.sum(temp_labels)	

for cl_num in range(10): # Calculating S_b
	temp_array = m_i[cl_num]-m_general
	s_b = s_b+np.outer(temp_array,temp_array)*(labels == cl_num).sum()

for cl_num in range(10): # Calculating S_common
	s_common = s_common + priors[cl_num]*s_i[cl_num]
	s_w = s_w + s_i[cl_num]

inv_scommon = inv(s_common)


def discriminant_func(x,i): # g_i(x)
	diff = x-m_i[i]
	middleterm=(-1/2)*np.dot(np.dot((diff)[None,:],inv_scommon),(diff))
	return math.log(priors[i])+middleterm

def selector(x): # find the largest g_i(x)
	largestval=-10000000000000000
	decision_class=0
	for tryc in range(10):
		value=discriminant_func(x,tryc)
		if (value>largestval):
			largestval=value
			decision_class=tryc
	return decision_class


def calc_trainingerror():
	conf_matrix = np.zeros((10, 10))
	true_tra = 0
	false_tra = 0
	for i in range(3823):
		tahmin = selector(features[:,i])
		if (tahmin==labels[i]):
			true_tra = true_tra+1
		else:
			false_tra = false_tra+1
		conf_matrix[labels[i]][tahmin] = conf_matrix[labels[i]][tahmin]+1
	print("Conf Matrix of Training Set")
	liner()
	print(conf_matrix)
	liner()
	np.savetxt("conf_mat_tra.csv", conf_matrix, delimiter=",", fmt='%i')


	print("Accuracy: "+str(true_tra/(true_tra+false_tra))+", Error: "+str(false_tra/(true_tra+false_tra))+" // TRAINING SET")

	liner()

def calc_testerror():
	conf_matrix = np.zeros((10, 10))
	true_tes = 0
	false_tes = 0

	for i in range(1797):
		tahmin = selector(features_test[:,i])
		if (tahmin==labels_test[i]):
			true_tes = true_tes+1
		else:
			false_tes = false_tes+1
		conf_matrix[labels_test[i]][tahmin] = conf_matrix[labels_test[i]][tahmin]+1
	print("Conf Matrix of Test Set")
	liner()
	print(conf_matrix)
	liner()
	np.savetxt("conf_mat_test.csv", conf_matrix.astype(int), delimiter=",", fmt='%i')

	print("Accuracy: "+str(true_tes/(true_tes+false_tes))+", Error: "+str(false_tes/(true_tes+false_tes))+" // TEST SET")

	liner()

def get_training_data():
	return features,labels,s_w,s_b

def get_test_data():
	return features_test,labels_test


## TO RUN THIS QUESTION

if __name__ == "__main__":
	calc_trainingerror()
	calc_testerror()