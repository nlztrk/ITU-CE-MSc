import question1 as q1
import numpy as np
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
from numpy.linalg import norm
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
import csv
from numpy.linalg import inv
import math

numpy.warnings.filterwarnings('ignore')

mode = 1 # 0 for training data, 1 for test data // INPUT SELECTION
which = 1 # 0 for PCA, 1 for LDA // METHOD SELECTION

features,labels,s_w,s_b = q1.get_training_data() # Getting the necessary values from question 1 - training
features_tes,labels_tes = q1.get_test_data() # Getting the necessary values from question 1 - test
inputs = features.T # Correcting the input format
inputs_tes = features_tes.T # Correcting the input format

class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(xy=center, width=height + xdescent,
                             height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

# Defining class colors
colors = ["red","blue","yellow","orange","green","cyan","magenta","black","indigo","maroon"]

def distance(p1,p2): # Getting euclidean distance between to points
	return np.linalg.norm(p1-p2)

def find_closest(p, mukayese):
	minval=100000000000
	closest_index=0
	forlen=len(features.T)

	for i in range(forlen): 
		dist = distance(mukayese[i],p)
		if (dist!=0 and dist<minval):
			minval = dist
			closest_index = i
	#print("verilen: "+str(p[0])+" "+str(p[1])+", en yakın: "+str(instances[closest_index][0])+" "+str(instances[closest_index][1]))
	return closest_index

def error(instances,mukayese):
	error=0
	if(mode==0):
		forlen=len(features.T)
		label_mat = labels
	else:
		forlen=len(features_tes.T)
		label_mat = labels_tes

	for i in range(forlen):
		closest_i = find_closest(instances[i], mukayese)
		if (labels[closest_i] != label_mat[i]):
			error = error + 1

	return error/forlen

if (which==0):

	covariance=np.cov(inputs.T); # Calculating the covariance matrix

	[e_values, e_vectors]=np.linalg.eig(covariance); # Getting eigenvalues and eigenvectors

	e_tuples = [(np.abs(e_values[i]), e_vectors[:,i]) for i in range(len(e_values))] # Creating a tuple that has eigen values and vectors

	# Descending order sorting
	e_tuples.sort()
	e_tuples.reverse()

	projection_multiplier = np.hstack((e_tuples[0][1].reshape(62,1), e_tuples[1][1].reshape(62,1))) # Creating 'w' matrice

	print("Creating projection on PCA..")
	q1.liner()

	# Creating projection 2D
	if (mode==0):
		projected_features = inputs.dot(projection_multiplier)
	else:
		projected_features = inputs_tes.dot(projection_multiplier)

	projected_features_tra = inputs.dot(projection_multiplier)

	if (mode==0):
		print("Plotting PCA-training..")
		q1.liner()
		for i in range(len(features.T)): # Plotting the PCA on training set
			plt.scatter(projected_features[i][0],projected_features[i][1],color=colors[labels[i]],alpha=0.7,label=labels[i])
	elif(mode==1):
		print("Plotting PCA-test..")
		q1.liner()
		for i in range(len(features_tes.T)): # Plotting the PCA on test set
			plt.scatter(projected_features[i][0],projected_features[i][1],color=colors[labels_tes[i]],alpha=0.7,label=labels_tes[i])

	handles=[]

	for i in range(10):
		handles.append(mpatches.Circle((0.5, 0.5), radius = 0.25, facecolor=colors[i], label=str(i)))


	plt.legend(handles=handles, fancybox=True, scatterpoints=1, handler_map={mpatches.Circle: HandlerEllipse()})

	if (mode==0):
		plt.title('PCA on Training Set')
	else:
		plt.title('PCA on Test Set')	
	plt.xlabel('PC1')
	plt.ylabel('PC2')

	plt.show()
	print("Calculating error..")
	q1.liner()
	print(error(projected_features,projected_features_tra))
	q1.liner()

### LDA PART

elif(which==1):
	ldacov = np.dot(inv(s_w),s_b)

	[e_values, e_vectors]=np.linalg.eig(ldacov); # Getting eigenvalues and eigenvectors

	e_tuples = [(np.abs(e_values[i]), e_vectors[:,i]) for i in range(len(e_values))] # Creating a tuple that has eigen values and vectors


	projection_multiplier = np.hstack((e_tuples[0][1].reshape(62,1), e_tuples[1][1].reshape(62,1))) # Creating 'w' matrice

	print("Creating projection on LDA..")
	q1.liner()
	# Creating projection 2D
	if (mode==0):
		projected_features = inputs.dot(projection_multiplier)
	else:
		projected_features = inputs_tes.dot(projection_multiplier)

	projected_features_tra = inputs.dot(projection_multiplier)

	if (mode==0):
		print("Plotting LDA-training..")
		q1.liner()
		for i in range(len(features.T)): # Plotting the PCA on training set
			plt.scatter(projected_features[i][0],projected_features[i][1],color=colors[labels[i]],alpha=0.7,label=labels[i])
	elif(mode==1):
		print("Plotting LDA-test..")
		q1.liner()
		for i in range(len(features_tes.T)): # Plotting the PCA on training set
			plt.scatter(projected_features[i][0],projected_features[i][1],color=colors[labels_tes[i]],alpha=0.7,label=labels_tes[i])

	handles=[]

	for i in range(10):
		handles.append(mpatches.Circle((0.5, 0.5), radius = 0.25, facecolor=colors[i], label=str(i)))

	plt.legend(handles=handles, fancybox=True, scatterpoints=1, handler_map={mpatches.Circle: HandlerEllipse()})

	if (mode==0):
		plt.title('LDA on Training Set')
	else:
		plt.title('LDA on Test Set')	
	plt.xlabel('PC1')
	plt.ylabel('PC2')

	plt.show()
	print("Calculating error..")
	q1.liner()
	print(error(projected_features,projected_features_tra))
	q1.liner()