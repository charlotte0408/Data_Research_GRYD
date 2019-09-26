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
import random


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
d = dfr.convert_objects(convert_numeric=True)
xr = d.to_numpy()
x = xr[~np.isnan(xr).any(axis=1)]
x_revise = np.empty((0,40))
y_revise = np.empty((0,17))

for i in range(x.shape[0]):
    if i == 0 and x[i,0] == x[i+1,0]:
        x_revise = np.vstack([x_revise,[x[i,0:40]]])
        y_revise = np.vstack([y_revise,[x[i,40:57]]])
    elif i == x.shape[0]-1 and x[i,0] == x[i-1,0]:
        x_revise = np.vstack([x_revise,[x[i,0:40]]])
        y_revise = np.vstack([y_revise,[x[i,40:57]]])
    elif i == x.shape[0]-1:
        continue
    elif x[i,0]==x[i+1,0] or x[i,0]==x[i-1,0]:
        x_revise = np.vstack([x_revise,[x[i,0:40]]])
        y_revise = np.vstack([y_revise,[x[i,40:57]]])
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
real_y = np.empty((0,17))
for j in range(1,4):
    for i in range(1000,int(max(new_x[:,0]))):
        pos1 = find_pos(new_x,i,j)
        pos2 = find_pos(new_x,i,j+1)
        if pos1 != -1 and pos2 != -1:
            real_x = np.vstack([real_x,[new_x[pos1,1:]]])
            real_y = np.vstack([real_y,[y_revise[pos2,:]]])

# After pressing Enter, the training starts
wait = input('Enter')
START = time.time()
BagHard = []
BagSoft = []
Boosting = []
for i in range(17):
    start_time = time.time()
    y_current = real_y[:,i]
    x_current = real_x
    num0 = np.count_nonzero(y_current==0)
    num1 = np.count_nonzero(y_current==1)
    ratio = num0/num1
    print("The Ratio of 0 to 1 is: %.2f" % ratio)
    # Balance the Ratio of 0 and 1 s
    if ratio>=2:
        for j in range(y_current.shape[0]):
            if y_current[j]==1:
                for _ in range(round(ratio)):
                    x_current = np.vstack([x_current,[x_current[j,:]]])
                    y_current = np.append(y_current,y_current[j])
    elif ratio <= 0.5:
        for j in range(y_current.shape[0]):
            if y_current[j]==0:
                for _ in range(round(1./ratio)):
                    x_current = np.vstack([x_current,[x_current[j,:]]])
                    y_current = np.append(y_current,y_current[j])
    x_train,x_test,y_train,y_test = train_test_split(x_current,y_current,test_size=0.2)
    print('Time of Dealing with Data for Question %d is %.2f' %(i+40,time.time()-start_time))
    print()

    Classifiers = []
    clf1 = GaussianNB()
    Classifiers.append(('GaussianNB', clf1))
    clf2 = DecisionTreeClassifier()
    Classifiers.append(('DecisionTreeClassifier', clf2))
    clf3 = RandomForestClassifier(n_estimators=2, criterion='gini')
    Classifiers.append(('RandomForestClassifier1', clf3))
    clf4 = RandomForestClassifier(n_estimators=2, criterion='entropy')
    Classifiers.append(('RandomForestClassifier2', clf4))
    # Train the Bagging
    eclf1 = VotingClassifier(estimators=Classifiers, voting='hard')
    eclf2 = VotingClassifier(estimators=Classifiers, voting='soft')
    eclf1.fit(x_train, y_train)
    eclf2.fit(x_train, y_train)
    BagHard.append(eclf1)
    BagSoft.append(eclf2)
    y_pred1 = eclf1.predict(x_test)
    y_pred2 = eclf2.predict(x_test)
    print("Classification Report of Bagging 1 for Question %d : " % (i+40))
    print(classification_report(y_test, y_pred1))
    print("Classification Report of Bagging 2 for Question %d : " % (i+40))
    print(classification_report(y_test, y_pred2))
    print()
    print('Time Cost of Bagging for Question %d is %.2f : ' % (i+40,float(time.time() - start_time)))
    print()

    start_time = time.time()
    # Train the Boosting
    Boost = AdaBoostClassifier(base_estimator=eclf2, n_estimators=10)
    Boost.fit(x_train, y_train)
    Boosting.append(Boost)
    pred = Boost.predict(x_test)
    print("Classification Report of Boosting for Question %d : " % (i+40))
    print(classification_report(y_test, pred))
    print()
    print('Time Cost of Boosting for Question %d is %.2f' % (i+40,float(time.time() - start_time)))
    print()

print("Total Training Time: %.2f" % float(time.time()-START))
wait = input('Enter')
# Test on the Original Dataset
x_train,x_test,y_train,y_test = train_test_split(real_x,real_y,test_size=0.2)
num1 = 0
num2 = 0
num3 = 0
# Record the Result and Test the Accuracy
for i in range(x_test.shape[0]):
    Prediction1 = np.empty(0,)
    Prediction2 = np.empty(0,)
    Prediction3 = np.empty(0,)
    for j in range(len(BagHard)):
        Prediction1 = np.append(Prediction1,BagHard[j].predict([x_test[i]]))
    for j in range(len(BagSoft)):
        Prediction2 = np.append(Prediction2,BagSoft[j].predict([x_test[i]]))
    for j in range(len(Boosting)):
        Prediction3 = np.append(Prediction3,Boosting[j].predict([x_test[i]]))
    '''
    Tolerance Level: 0 mistake
    if np.all(Prediction1 == y_test[i]):
        num1 += 1
    if np.all(Prediction2 == y_test[i]):
        num2 += 1
    if np.all(Prediction3 == y_test[i]):
        num3 += 1    
    '''


    #Tolerance Level: 1 mistake
    if np.count_nonzero(Prediction1 != y_test[i]) <= 1:
        num1 += 1
    if np.count_nonzero(Prediction2 != y_test[i]) <= 1:
        num2 += 1
    if np.count_nonzero(Prediction3 != y_test[i]) <= 1:
        num3 += 1

# Print the Result
print()
print("The Performance of Hard Bagging on Testing Set is %.4f" % float(num1/x_test.shape[0]))
print()
print("The Performance of Soft Bagging on Testing Set is %.4f" % float(num2/x_test.shape[0]))
print()
print("The Performance of Boosting on Testing Set is %.4f" % float(num3/x_test.shape[0]))

