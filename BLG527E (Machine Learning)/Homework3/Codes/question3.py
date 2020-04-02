import numpy as np
import pandas as pd

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

opt = 0

print ("Question 3 has been started! (Multilayer Perceptrons)")

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

from sklearn.neural_network import MLPClassifier

print() ## Blank line

## Optimizing Hyper-Parameters

from sklearn.model_selection import ParameterGrid

if (opt):

	param_grid = {'hidden_layer_sizes': [90,100,110], 'activation' : ["relu","identity","logistic","tanh"], 'random_state':[1337], 'solver':["adam"]}

	grid = ParameterGrid(param_grid)

	best_tes_score = 0
	best_param = {"null":"null"}

	for params in grid:

		mpclf = MLPClassifier(**params)
		mpclf.fit(X_tra,Y_tra)
		for_score = mpclf.score(X_tes, Y_tes)

		if (for_score>best_tes_score):
			best_tes_score = for_score
			best_param = params

	print(best_param)

## Defining the MP Classifier

mpclf = MLPClassifier(activation="relu", hidden_layer_sizes=100, random_state=1337, solver="adam")
mpclf.fit(X_tra, Y_tra)

print("Training Accuracy: " + str(mpclf.score(X_tra, Y_tra)) + ", Test Accuracy: " + str(mpclf.score(X_tes, Y_tes)))

print() ## Blank line

## Getting the confusion matrix

from sklearn.metrics import confusion_matrix 

y_tra_pred = mpclf.predict(X_tra)
y_tes_pred = mpclf.predict(X_tes)

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
