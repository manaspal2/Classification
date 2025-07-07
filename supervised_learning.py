#This is a general structure of a Scikit learn model usage
#from sklearn.module import Model 
#model = Model ()
#model.fit(X,y)
#prediction = model.predict(X_New)
#print(prediction)  
#################################################
# Classification problem - knn classification
#################################################
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
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

print (X.shape, y.shape)
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X,y)

X_new = np.array([[56.8,17.5],
        [24.4,24.1],
        [50.1,10.9]])
print (X_new.shape)
prediction = knn.predict(X_new)
print (prediction)

############################
# Model accuracy
############################
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=21, stratify=y)
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train,y_train)
print (knn.score(X_test, y_test))

##############################
# Model complexity curve
##############################
train_accuracy = {}
test_accuracy = {}
neighbors = np.arange(1,26)

for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train,y_train)
    train_accuracy[neighbor]= knn.score(X_train, y_train)
    test_accuracy[neighbor]= knn.score(X_test, y_test)

plt.plot (neighbors, train_accuracy.values(), label = "Training accuracy")
plt.plot (neighbors, test_accuracy.values(), label = "Testing accuracy")
plt.xlabel("Number of neighbors")
plt.ylabel("Accuracy")
plt.show()