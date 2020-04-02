import numpy as np
import pandas as pd


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

opt = 0

print ("Question 2 has been started! (Decision Tree)")

## Reading the data

# Since the .csv file doesn't include headers
test_df = pd.read_csv('optdigits.tes', header=None)
tra_df = pd.read_csv('optdigits.tra', header=None)

# Splitting the features and labels
X_tes = test_df.drop(64,axis=1).values
Y_tes = test_df[64].values

# Splitting the features and label
X_tra = tra_df.drop(64,axis=1).values
Y_tra = tra_df[64].values

## Importing DT Classifier

from sklearn import tree

print() ## Blank line

## Optimizing Hyper-Parameters

from sklearn.model_selection import ParameterGrid

if (opt):
	param_grid = {'criterion': ["gini","entropy"], 'max_depth' : [7, 8, 9, 10, 11, 12, 13], 'random_state':[1337], 'min_samples_split':[3,4,5,6,7]}

	grid = ParameterGrid(param_grid)

	best_tes_score = 0
	best_param = {"null":"null"}

	for params in grid:

		tclf = tree.DecisionTreeClassifier(**params)
		tclf.fit(X_tra,Y_tra)
		for_score = tclf.score(X_tes, Y_tes)

		if (for_score>best_tes_score):
			best_tes_score = for_score
			best_param = params

	print(best_param)

## Defining the DT Classifier

tclf = tree.DecisionTreeClassifier(criterion= 'entropy', max_depth= 10, random_state= 1337, min_samples_split= 5) 
tclf.fit(X_tra, Y_tra)

print("Training Accuracy: " + str(tclf.score(X_tra, Y_tra)) + ", Test Accuracy: " + str(tclf.score(X_tes, Y_tes)))

print() ## Blank line

## Getting the confusion matrix

from sklearn.metrics import confusion_matrix 

y_tra_pred = tclf.predict(X_tra)
y_tes_pred = tclf.predict(X_tes)

conf_tra = confusion_matrix(Y_tra,y_tra_pred)
conf_tes = confusion_matrix(Y_tes,y_tes_pred)

print("Training Confusion Matrix: ")
print(conf_tra)

print() ## Blank line

print("Test Confusion Matrix: ")
print(conf_tes)

print() ## Blank line

## Getting the precision and recall values for both sets

from sklearn.metrics import precision_score



precision_tra = precision_score(Y_tra, y_tra_pred, average='macro')
precision_tes = precision_score(Y_tes, y_tes_pred, average='macro')

precisions_tes, precisions_tra = [], []

for label in range(0,10):
	precisions_tes.append(precision_score(Y_tes, y_tes_pred, labels=[label], average='micro', sample_weight=None))
	precisions_tra.append(precision_score(Y_tra, y_tra_pred, labels=[label], average='micro', sample_weight=None))

print ("Training Precision: " + str(precision_tra))
print ("Test Precision: " + str(precision_tes))

print() ## Blank line

print("Precisions of training respect to labels (0,1,...,9)")
print(precisions_tra)

print() ## Blank line

print("Precisions of test respect to labels (0,1,...,9)")
print(precisions_tes)

print() ## Blank line
