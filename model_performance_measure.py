# Confusion metrix to measure the performance of the model
# +------------+------------+-----------+
# |            | Predicted  | Predicted |
# |            | Legitimate | Fraud     |
# +------------+------------+-----------+
# | Actual     | True       | False     |
# | Legitimate | Negative   | Positive  |
# +------------+------------+-----------+
# | Actual     | False      | True      |
# | Fraud      | Negative   | Positive  |
# +------------+------------+-----------+
#
# Some of the important parameters whichh can be derived from the confusion metrix
# Accuracy = (True positive + True negetive)/(True Positive + True Negative + False Positive + False Positive)
# Precision = (True positive)/(True Positive + False Positive)
# Recall/Sensitivity = (True positive)/(True Positive + False Negative)
# F1 score = (Precision * Recall)/(Precision + Recall)
#########################################################################################################

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("train_telecom_data.csv")
print (df.head())
columns = df.columns
print ("#################")
print ("Columns")
print ("#################")
print (columns)
X = df[["total_day_charge", "total_eve_charge"]].values
y = df["churn"].values

#############################################################################################
# We choose the n_neighbors. This is one hyperparameter which might help to tune the model.
# Choose correct hyperparameter after trying multiple values. This is called Hyper-parameter tuning
# While doing hyper-parameter tuning, we do cross-validation.
#############################################################################################
knn = KNeighborsClassifier(n_neighbors=7)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=21, stratify=y)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

print (confusion_matrix(y_test, y_pred))

print ("############## Classification report ##############")
print (classification_report(y_test, y_pred))

#########################################################################
# One of the hyper-parameter tuning method : Grid-search cross validation 
#########################################################################
