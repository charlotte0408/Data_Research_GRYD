#https://quantdare.com/what-is-the-difference-between-bagging-and-boosting/

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
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
import sys
from sklearn.metrics import classification_report

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
x_revise = np.empty((0,x.shape[1]))
y_revise = np.empty((0,))

for i in range(x.shape[0]):
    if i == 0 and x[i,0] == x[i+1,0]:
        x_revise = np.vstack([x_revise,[x[i,:]]])
        y_revise = np.append(y_revise,y[i])
        print(x_revise[0,:])
    elif i == x.shape[0]-1 and x[i,0] == x[i-1,0]:
        x_revise = np.vstack([x_revise,[x[i,:]]])
        y_revise = np.append(y_revise,y[i])
    elif i == x.shape[0]-1:
        continue
    elif x[i,0]==x[i+1,0] or x[i,0]==x[i-1,0]:
        x_revise = np.vstack([x_revise,[x[i,:]]])
        y_revise = np.append(y_revise,y[i])

new_x = np.zeros((x_revise.shape[0],11))
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
    new_x[i,9] = np.sum(x_revise[i,40:57])
    new_x[i,10] = np.sum(x_revise[i,57:75])
real_x = np.empty((0,new_x.shape[1]-1))
real_y = np.empty((0,))
for j in range(1,4):
    for i in range(1000,int(max(new_x[:,0]))):
        pos1 = find_pos(new_x,i,j)
        pos2 = find_pos(new_x,i,j+1)
        if pos1 != -1 and pos2 != -1:
            real_x = np.vstack([real_x,[new_x[pos1,1:]]])
            real_y = np.append(real_y,y_revise[pos2])

# Convert the Y to 0 and 1 (0 means eligible, 1 means not)
real_y = If_Elig(real_y)
for i in range(real_y.shape[0]):
    if real_y[i]==1:
        for _ in range(3):
            real_x = np.vstack([real_x,[real_x[i,:]]])
            real_y = np.append(real_y,real_y[i])

# After pressing Enter, the training starts
wait = input('Enter')
start_time = time.time()

# Train Test Split
x_train,x_test,y_train,y_test = train_test_split(real_x,real_y,test_size=0.2)
Classifiers = []
# If I comment the append instruction, the algorithm is not included into bagging or boosting
clf1 = KNeighborsClassifier(n_neighbors=2)
#Classifiers.append(('KNeighborsClassifier',clf1))
clf2 = SVC(probability=True)
#Classifiers.append(('SVC',clf2))
clf3 = NuSVC(nu=0.1,probability=True)
#Classifiers.append(('NuSVC',clf3))
clf4 = GaussianNB()
Classifiers.append(('GaussianNB',clf4))
clf5 = DecisionTreeClassifier()
Classifiers.append(('DecisionTreeClassifier',clf5))
clf6 = RandomForestClassifier(n_estimators=2,criterion='gini')
Classifiers.append(('RandomForestClassifier1',clf6))
clf7 = RandomForestClassifier(n_estimators=2,criterion='entropy')
Classifiers.append(('RandomForestClassifier2',clf7))
clf8 = SGDClassifier(loss='modified_huber')
#Classifiers.append(('SGDClassifier',clf8))
clf9 = MLPClassifier()
#Classifiers.append(('MLPClassifier',clf9))

# Create two bagging classifiers
eclf1 = VotingClassifier(estimators=Classifiers,voting='hard')
eclf2 = VotingClassifier(estimators=Classifiers,voting='soft')
'''
eclf1.fit(x_train,y_train)
eclf2.fit(x_train,y_train)
y_pred1 = eclf1.predict(x_test)
y_pred2 = eclf2.predict(x_test)
Score1 = eclf1.score(x_test,y_test)
print('Score of Bagging 1: %.2f' % Score1)
Score2 = eclf2.score(x_test,y_test)
print('Score of Bagging 2: %.2f' % Score2)
print("Classification Report of Bagging 1" )
print(classification_report(y_test,y_pred1))
print("Classification Report of Bagging 2" )
print(classification_report(y_test,y_pred2))
print('Time Cost for the Program: %s' % (time.time()-start_time))
'''

# Create the boosting classifier based on soft bagging
Boost = AdaBoostClassifier(base_estimator=eclf2,n_estimators=10)
Boost.fit(x_train,y_train)
score = Boost.score(x_test,y_test)
pred = Boost.predict(x_test)
print("Classification Report of Boosting: ")
print(classification_report(y_test,pred))
print('Time Cost for the Program: %s' % (time.time()-start_time))


# Hand Written Bagging
Classifier = []
Score = []
Prediction = []
Classifier.append(KNN(2,x_train,y_train))
print('KNN Train Finished')
Classifier.append(SVM(x_train,y_train))
print('SVM Train Finished')
Classifier.append(NuSVM(x_train,y_train))
print('NuSVM Train Finished')
Classifier.append(Bayesian(x_train,y_train))
print('Bayesian Train Finished')
Classifier.append(DecisionTree(x_train,y_train))
print('Decision Tree Train Finished')
Classifier.append(RandomForest(x_train,y_train,n_estimator=2))
print('Random Forest Gini Train Finished')
Classifier.append(RandomForest(x_train,y_train,n_estimator=2,criterion='entropy'))
print('Random Forest Entropy Train Finished')
Classifier.append(LinearC(x_train,y_train))
print('Linear Model Train Finished')
Classifier.append(MLP(x_train,y_train))
print('MLP Train Finished')

# Test the Accuracy of Each Classifier
for i in range(len(Classifier)):
    num = 0
    y_predict = Classifier[i].predict(x_test)
    print("Classification Report: %d" % i)
    print(classification_report(y_test,y_predict))
    for j in range(x_test.shape[0]):
        if y_predict[j]==y_test[j]:
            num+=1
    Score.append(num/x_test.shape[0])
Overall_Acc = 0

# Calculate the Estimated Accuracy
for k in range(0,5):
    for i in list(combinations([0,1,2,3,4,5,6,7,8],k)):
        temp = 1
        for j in range(9):
            if j in i:
                temp = temp * (1-Score[j])
            else:
                temp = temp * Score[j]
        Overall_Acc = Overall_Acc + temp
print(Overall_Acc)


'''
# Normal Voter

for i in range(len(Classifier)):
    Prediction.append(Classifier[i].predict(real_x[0]))
Voted_Prediction = max(Prediction,key=Prediction.count)
print("The Normal Voted Prediction is %s" % Voted_Prediction)
Weighted_Prediction = Vote(Classifier,Prediction,Score,real_x[0])
print("The Weighted Voted Prediction is %s" % Weighted_Prediction)
print("The Rounded Voted Prediction is %s" % round(Weighted_Prediction))
'''


# Hard Voting Accuracy of the Whole Dataset
num = 0
Prediction = []
for j in range(len(Classifier)):
    Prediction.append(Classifier[j].predict(real_x))
Prediction = np.array(Prediction)
for i in range(real_x.shape[0]):
    Voted_Prediction = np.sum(Prediction[:,i])/9
    if Voted_Prediction>=0.5:
        Voted_Prediction = 1
    else:
        Voted_Prediction = 0
    if Voted_Prediction == real_y[i]:
        num += 1
print("The Accuracy of Normal Voting on Overall Dataset: %.2f %%" % float(100*num/real_x.shape[0]))

# Soft Voting Result of the Whole Dataset
num = 0
Prediction = []
for j in range(len(Classifier)):
    Prediction.append(Classifier[j].predict(real_x))
Prediction = np.array(Prediction)
for i in range(real_x.shape[0]):
    Weighted_Prediction = Vote(Classifier,Prediction[:,i],Score)
    if Weighted_Prediction>=0.5:
        Weighted_Prediction = 1
    else:
        Weighted_Prediction = 0
    if Weighted_Prediction == real_y[i]:
        num += 1
print("The Accuracy of Weighted Voting on Overall Dataset: %.2f %%" % float(100*num/real_x.shape[0]))

