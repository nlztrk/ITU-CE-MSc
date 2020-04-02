import numpy as np
import pandas as pd


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

print ("Question 1 has been started! (k-NN)")

do_opt = 0 ## Do the k-optimization, 0 if use the already-optimised value

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

## Importing k-NN Classifier

from sklearn.neighbors import KNeighborsClassifier

#We need to get best results, so we need to try different k's

k_val = np.arange(1,12)
tra_acc =np.empty(len(k_val))
tes_acc = np.empty(len(k_val))
best_acc = 0
best_acc_k = -1

print() ## Blank line

for k in range(1,12):
    if (do_opt==0):
        best_acc_k = 0
        break
    # We set-up a k-NNC with given neighbor value
    knn = KNeighborsClassifier(n_neighbors=k_val[k-1])
    
    # We fit the model
    knn.fit(X_tra, Y_tra)

    # We compute the accuracy on training set
    tra_acc[k-1] = accuracy_score(X_tra, Y_tra)
    
    # We compute the accuracy on test set
    tes_acc[k-1] = knn.score(X_tes, Y_tes) 

    print("Testing k: " + str(k_val[k-1]) + ", Training Accuracy: " + str(tra_acc[k-1]) + ", Test Accuracy: " + str(tes_acc[k-1]))
    
    # We will do the (best) comparison depending on the test accuracy value
    if (tes_acc[k-1]>=best_acc):
        best_acc = tes_acc[k-1]
        best_acc_k = k-1

print() ## Blank line

best_acc_k = best_acc_k + 1

## Fitting with optimal k value

knn = KNeighborsClassifier(n_neighbors=best_acc_k) # Optimal k=1
knn.fit(X_tra, Y_tra)

print("Best k: " + str(best_acc_k) + ", Training Accuracy: " + str(knn.score(X_tra, Y_tra)) + ", Test Accuracy: " + str(knn.score(X_tes, Y_tes)))

print() ## Blank line

## Getting the confusion matrix

from sklearn.metrics import confusion_matrix 

y_tra_pred = knn.predict(X_tra)
y_tes_pred = knn.predict(X_tes)

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
