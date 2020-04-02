import question1 as q1
import random
import numpy as np
from scipy import stats

features,labels,s_w,s_b = q1.get_training_data() # Getting the necessary values from question 1 - training

features = features.T

def create_cluster(): ## Creating a cluster with a center with random input values
	cluster_positions=[]
	for pos_num in range(62):
		cluster_positions.append(random.randint(0,16))
	return cluster_positions

def create_n_clusters(n): ## Creating n clusters with creating function
	clusters=[]
	for i in range(n):
		clusters.append(create_cluster())
	return np.asarray(clusters)

def po_cc_dist(point,cluster): ## Measuring the distance between a point and a cluster center
	return np.linalg.norm((point-cluster),ord=1) ## L1 norm

def nearest_cluster_to_point(point,clusters): ## Finding nearest cluster center to a point
	cluster_id=84358
	min_dist=100000000
	for i in range(len(clusters)):
		dist = po_cc_dist(point,clusters[i])
		if(dist<min_dist):
			min_dist = dist
			cluster_id=i
	return cluster_id

def set_point_labels(instances,clusters,cids): ## Setting cluster of a point with nearest-cluster daha
	for i in range(len(instances)):
		point = instances[i]
		cids[i] = nearest_cluster_to_point(point,clusters)

	return cids

def calculate_cluster_centers(cids,instances,clusters): ## Calculate new cluster centers after new cluster element formation

	for c in range(len(clusters)): # for each cluster
		element_ids = np.where(cids == c)[0]

		elements=[]

		for eid in element_ids: # adding ex-cluster elements
			elements.append(instances[eid])

		elements = np.asarray(elements)

		if(elements.size!=0):
			new_center = elements.mean(0)
		else:
			new_center = clusters[c]

		clusters[c] = new_center
	
	return clusters

def calculate_error(cids,labels,clusters): ## Calculate wrongly placed cluster members

	error=0

	for c in range(len(clusters)):

		from_c = np.where(cids == c)[0] ## finding all instance indexes of a cluster

		c_labels=[]

		for i in from_c: # getting labels for cluster
			c_labels.append(labels[i])

		check = np.asarray(c_labels)

		if(check.size!=0):

			label_of_cluster = int(stats.mode(c_labels)[0])
			c_labels = np.asarray(c_labels)
			error = error + (c_labels != label_of_cluster).sum()

	return error/len(features)

## STARTING THE ROUTINE

kumeler = create_n_clusters(20)

print("Created "+str(len(kumeler))+" clusters!")
q1.liner()

kumenolar=np.zeros(3823)

print("Executing initial k-means iteration..")
q1.liner()

kumenolar=set_point_labels(features,kumeler,kumenolar)
kumeler=calculate_cluster_centers(kumenolar,features,kumeler)

ex_kumeler=np.zeros((len(kumeler),62))

iteration_count=0;

## ITERATION LOOP
while(1):

	kumenolar=set_point_labels(features,kumeler,kumenolar)
	kumeler=calculate_cluster_centers(kumenolar,features,kumeler)
	iteration_count = iteration_count+1
	if (np.array_equal(ex_kumeler,kumeler)): ## Find the iteration
		q1.liner()
		print("Finished iterating!")
		q1.liner()
		break

	print("Iteration: "+str(iteration_count))
	ex_kumeler[:]=kumeler[:]

## PRINTING THE ERROR
print("L1 Error: "+str(calculate_error(kumenolar,labels,kumeler)))
q1.liner()
