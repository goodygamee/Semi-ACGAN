# -*- coding: utf-8 -*-
"""
Created on Sat May 11 10:36:06 2019

@author: 玉强
"""
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

'''
def dropna_str(data):
    data = data.drop([136488,136496],axis=0)
    cnt = 0
    for item in data[2]:
        if type(item) != int:
            data = data.drop([cnt],axis=0)
        cnt+=1
    return data
'''
            

kdd_new = pd.read_csv('D:/NNlearning/code/kddcup.data_new4',names=range(42))
kdd_test = pd.read_csv('D:/NNlearning/code/kddcup.testdata_new4',names=range(42))
kdd_test =kdd_test.drop([136488,136496],axis=0)

#kdd_test = dropna_str(kdd_test)

target = kdd_new[41]
del kdd_new[41]
#label_list=['normal.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.', 'smurf.', \
#     'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.', 'ipsweep.', 'land.', 'ftp_write.', \
#     'back.', 'imap.', 'satan.', 'phf.', 'nmap.', 'multihop.', 'warezmaster.', 'warezclient.', \
#     'spy.', 'rootkit.']



X_train = kdd_new
y_train = target

y_test = kdd_test[41]
del kdd_test[41]
X_test = kdd_test

hot_enc = OneHotEncoder(categorical_features = [1,2,3,6,11,13,14,21],sparse=False)
hot_enc.fit(X_train)

X_train = hot_enc.transform(X_train)
X_test = hot_enc.transform(X_test)

def get_random_batch(X,Y,index,batch_size=1000):
    np.random.seed(1234)
    ind = np.random.choice(index,(batch_size))
    X_batch = X[ind]
    Y_batch = Y[ind]    
    return X_batch, Y_batch

X_train, y_train = get_random_batch(X_train,y_train,kdd_new.index,100)



#scaler = StandardScaler()
#X_train = scaler.transform(X_train)

'''
grid = GridSearchCV(LinearSVC(), param_grid={"C":[0.1,0.6,5]}, cv=4)   
grid.fit(X_train,y_train)

print("The best parameters are %s with a score of %0.2f"% (grid.best_params_, grid.best_score_))
'''
clf = LinearSVC(dual=False,C=10,max_iter=5000)
#clf = SVC(C=0.5,gamma=0.6)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print ("model:SVM \n", classification_report(y_test,y_pred))
print ("Accuracy score:",accuracy_score(y_test, y_pred))
