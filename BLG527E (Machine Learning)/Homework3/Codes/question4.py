import numpy as np
import pandas as pd

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

opt = 0

print ("Question 4 has been started! (Support Vector Machine)")

## Reading the data

# Since the .csv file doesn't include headers
test_df = pd.read_csv('optdigits.tes', header=None)
tra_df = pd.read_csv('optdigits.tra', header=None)

# Splitting the features and labels
X_tes = test_df.drop(64,axis=1).values
Y_tes = test_df[64].values

# Splitting the features and labels
X_tra = tra_df.drop(64,axis=1).values
Y_tra = tra_df[64].values

## Importing MP Classifier

from sklearn import svm

print() ## Blank line

## Optimizing Hyper-Parameters

from sklearn.model_selection import ParameterGrid

if (opt):
	
	param_grid = {'kernel': ["rbf","linear","poly","sigmoid"], 'shrinking':[True,False], 'probability':[True,False], 'decision_function_shape':["ovo","ovr"], "random_state":[1337] }

	grid = ParameterGrid(param_grid)

	best_tes_score = 0
	best_param = {"null":"null"}

	for params in grid:

		svmclf = svm.SVC(**params)
		svmclf.fit(X_tra,Y_tra)
		for_score = svmclf.score(X_tes, Y_tes)

		if (for_score>best_tes_score):
			best_tes_score = for_score
			best_param = params

	print(best_param)

## Defining the MP Classifier

svmclf = svm.SVC(kernel='poly', decision_function_shape="ovo", probability=True, random_state=1337, shrinking=True)
svmclf.fit(X_tra, Y_tra)

print("Training Accuracy: " + str(svmclf.score(X_tra, Y_tra)) + ", Test Accuracy: " + str(svmclf.score(X_tes, Y_tes)))

print() ## Blank line

## Getting the confusion matrix

from sklearn.metrics import confusion_matrix 

y_tra_pred = svmclf.predict(X_tra)
y_tes_pred = svmclf.predict(X_tes)

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
