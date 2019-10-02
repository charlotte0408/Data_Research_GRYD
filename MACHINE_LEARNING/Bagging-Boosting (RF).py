import numpy as np
import pandas as pd
import time
from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
import sys
import tensorflow as tf
from ELM.elm import ELM
import os
import argparse


# Check if the data is eligible
def If_Elig(data):
    return (data>=5).astype(int)

# Find the position of the data
def find_pos(data,value, order):
    max_num = np.count_nonzero(data[:,0]==value)
    if order > max_num:
        return -1
    else:
        return np.nonzero(data[:,0]==value)[0][order-1]

# Model Prediction
def model_predict(model,x):
    return model.predict(x)

# Hand Written Soft Voting
def Vote(Classifier,Prediction,Score):
    leng = len(Classifier)
    sum_score = sum(Score)
    Weighted = 0
    for i in range(leng):
        Weighted += Prediction[i]*Score[i]/sum_score
    return Weighted


# The following def are packages training, which can be searched on sklearn

def KNN(n,x,y):
    knn = KNeighborsClassifier(n_neighbors=n, n_jobs=-1)
    knn.fit(x,y)
    return knn


def SVM(x,y):
    svc = SVC()
    svc.fit(x,y)
    return svc


def NuSVM(x,y):
    nus = [_ / 10 for _ in range(1, 11, 1)]
    for nu in nus:
        nusvc = NuSVC(nu=nu)
        try:
            nusvc.fit(x,y)
            return nusvc
        except ValueError as e:
            print("nu {} not feasible".format(nu))

def Bayesian(x,y):
    bayesian = GaussianNB()
    bayesian.fit(x,y)
    return bayesian


def DecisionTree(x,y,criterion='gini',max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None):
    Tree = DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,min_samples_split= \
                                               min_samples_split,min_samples_leaf=min_samples_leaf,
                                               max_features=max_features)
    Tree.fit(x,y)
    return Tree


def RandomForest(x,y,n_estimator=10,criterion='gini'):
    Forest = RandomForestClassifier(n_estimators=n_estimator,criterion=criterion)
    Forest.fit(x,y)
    return Forest


def LinearC(x,y):
    linear = SGDClassifier()
    linear.fit(x,y)
    return linear


def MLP(x,y,hidden_layer_size=(100,),activation='relu',solver='adam',learning_rate_init=0.01, learning_rate='adaptive'):
    if solver=='adam':
        MLP = MLPClassifier(hidden_layer_sizes=hidden_layer_size, activation=activation, solver= \
                                               solver, learning_rate_init=learning_rate_init, learning_rate=learning_rate)
    elif solver=='sgd':
        MLP = MLPClassifier(hidden_layer_sizes=hidden_layer_size, activation=activation, solver= \
            solver, learning_rate_init=learning_rate_init)
    else:
        MLP = MLPClassifier(hidden_layer_sizes=hidden_layer_size, activation=activation, solver= \
            solver)
    MLP.fit(x,y)
    return MLP


# Read Data
np.set_printoptions(threshold=sys.maxsize)
df = pd.read_csv('New_Cleaned_Data.csv')
dfr = df.loc[:, 'UniqueID':'ij56_ever_combo']
dft = df.loc[:,"ReconstructedFactor"]
d = dfr.convert_objects(convert_numeric=True)
dt = dft.convert_objects(convert_numeric=True)
xr = d.to_numpy()
y = dt.to_numpy()
x = xr[~np.isnan(xr).any(axis=1)]
y = y[~np.isnan(xr).any(axis=1)]
x_revise = np.empty((0,40))
y_revise = np.empty((0,))

for i in range(x.shape[0]):
    if i == 0 and x[i,0] == x[i+1,0]:
        x_revise = np.vstack([x_revise,[x[i,0:40]]])
        y_revise = np.append(y_revise,y[i])
    elif i == x.shape[0]-1 and x[i,0] == x[i-1,0]:
        x_revise = np.vstack([x_revise,[x[i,0:40]]])
        y_revise = np.append(y_revise,y[i])
    elif i == x.shape[0]-1:
        continue
    elif x[i,0]==x[i+1,0] or x[i,0]==x[i-1,0]:
        x_revise = np.vstack([x_revise,[x[i,0:40]]])
        y_revise = np.append(y_revise,y[i])
new_x = np.zeros((x_revise.shape[0],9))
for i in range(x_revise.shape[0]):
    new_x[i,0] = x_revise[i,0]
    new_x[i,1] = np.sum(x_revise[i,1:7])
    new_x[i,2] = np.sum(x_revise[i,7:10])
    new_x[i,3] = np.sum(x_revise[i,10:17])
    new_x[i,4] = np.sum(x_revise[i,17:21])
    new_x[i,5] = np.sum(x_revise[i,21:27])
    new_x[i,6] = np.sum(x_revise[i,27:32])
    new_x[i,7] = np.sum(x_revise[i,32:38])
    new_x[i,8] = np.sum(x_revise[i,38:40])

# Without scale score
'''
real_x = np.empty((0,x_revise.shape[1]))
real_y = np.empty((0,))
for j in range(1,4):
    for i in range(1000,int(max(x_revise[:,0]))):
        pos1 = find_pos(x_revise,i,j)
        pos2 = find_pos(x_revise,i,j+1)
        if pos1 != -1 and pos2 != -1:
            real_x = np.vstack([real_x,[x_revise[pos1,:]]])
            real_y = np.append(real_y,y_revise[pos2])
'''

# With scale score

real_x = np.empty((0,new_x.shape[1]-1))
real_y = np.empty((0,))
for j in range(1,4):
    for i in range(1000,int(max(new_x[:,0]))):
        pos1 = find_pos(new_x,i,j)
        pos2 = find_pos(new_x,i,j+1)
        if pos1 != -1 and pos2 != -1:
            real_x = np.vstack([real_x,[new_x[pos1,1:]]])
            real_y = np.append(real_y,y_revise[pos2])

# Balance the dataset
for i in range(10):
    print("Number of Instance %d is %d" % (i,np.count_nonzero(real_y==i)))
zero = 0
for i in range(real_y.shape[0]):
    if real_y[i] == 0 and zero<1000:
        real_x = np.vstack([real_x,[real_x[i,:]]])
        real_y = np.append(real_y,real_y[i])
        zero += 1
    if real_y[i] == 1:
        real_x = np.vstack([real_x,[real_x[i,:]]])
        real_y = np.append(real_y,real_y[i])
    elif real_y[i] == 2:
        for _ in range(2):
            real_x = np.vstack([real_x,[real_x[i,:]]])
            real_y = np.append(real_y,real_y[i])
    elif real_y[i] == 3:
        for _ in range(2):
            real_x = np.vstack([real_x,[real_x[i,:]]])
            real_y = np.append(real_y,real_y[i])
    elif real_y[i] == 4 or real_y[i] == 5:
        for _ in range(3):
            real_x = np.vstack([real_x,[real_x[i,:]]])
            real_y = np.append(real_y,real_y[i])
    elif real_y[i] == 6:
        for _ in range(4):
            real_x = np.vstack([real_x,[real_x[i,:]]])
            real_y = np.append(real_y,real_y[i])
    elif real_y[i] == 7:
        for _ in range(7):
            real_x = np.vstack([real_x,[real_x[i,:]]])
            real_y = np.append(real_y,real_y[i])
    elif real_y[i] == 8:
        for _ in range(16):
            real_x = np.vstack([real_x,[real_x[i,:]]])
            real_y = np.append(real_y,real_y[i])
for i in range(10):
    print("Number of Instance %d is %d" % (i,np.count_nonzero(real_y==i)))
# After pressing Enter, the training starts
wait = input('Enter')
start_time = time.time()

# Train Test Split
x_train,x_test,y_train,y_test = train_test_split(real_x,real_y,test_size=0.2)
model1 = RandomForestClassifier()
model1.fit(x_train,y_train)
model2 = RandomForestClassifier(criterion='entropy')
model2.fit(x_train,y_train)
model3 = RandomForestClassifier(bootstrap=False)
model3.fit(x_train,y_train)
bag = VotingClassifier(estimators=[('RF1',model1),('RF2',model2),('RF3',model3)],voting='soft')
boost = AdaBoostClassifier(base_estimator=model1,n_estimators=5)
bag.fit(x_train,y_train)
boost.fit(x_train,y_train)
y_pred1 = model1.predict(x_test)
y_pred2 = bag.predict(x_test)
y_pred3 = boost.predict(x_test)
# Prediction
print(classification_report(y_pred1,y_test))
print(classification_report(y_pred2,y_test))
print(classification_report(y_pred3,y_test))
# Demo for the first 50 data
print(y_pred3[0:50])
print(y_test[0:50])
print("Total Time Used: %.4f" % float(time.time()-start_time))

